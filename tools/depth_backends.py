"""Unified adapters for the monocular depth models used in the stage-1 error study.

Every backend exposes the same contract, so the benchmark never needs to know which
model it is talking to:

    backend = get_backend("unidepth_v2")
    backend.load(device="cuda")
    out = backend.infer(bgr)          # bgr = HxWx3 uint8, OpenCV order

    out["depth_m"]     HxW float32.  Z-depth along the optical axis, in metres.
                                     None for backends that are scale-free.
    out["relative"]    HxW float32 or None.  Affine-invariant output (disparity for
                                     Depth Anything V2) for backends that have no metric scale.
    out["intrinsics"]  3x3 float32 or None.  Whatever the model predicted for itself.

Imports are deliberately deferred into `load()`.  The repositories have mutually
incompatible dependency pins, so importing them all up front would mean one broken
model takes down the whole benchmark.

Repositories are expected under `depth_models/` (see README.md, "Depth models").
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "depth_models"


def _add_path(sub: str) -> Path:
    p = MODELS_DIR / sub
    if not p.exists():
        raise FileNotFoundError(
            f"missing vendored repository: {p}\n"
            f"clone it first -- see README.md"
        )
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def _shim_moviepy_editor() -> None:
    """Depth Anything 3 imports `moviepy.editor` from its *export* module, which the
    package pulls in at import time.  moviepy 2.x dropped that submodule.  Inference
    never touches it, so a stub is enough and keeps the vendored tree unpatched."""
    if "moviepy.editor" in sys.modules:
        return
    try:
        import moviepy
    except ImportError:
        return
    shim = types.ModuleType("moviepy.editor")
    for name in ("ImageSequenceClip", "VideoFileClip", "concatenate_videoclips"):
        shim.__dict__[name] = getattr(moviepy, name, None)
    sys.modules["moviepy.editor"] = shim


class Backend:
    name = "base"
    kind = "metric"        # "metric" = metres out of the box; "relative" = needs anchoring
    paper = ""

    def load(self, device: str = "cuda", **kw):
        raise NotImplementedError

    def infer(self, bgr: np.ndarray) -> dict:
        raise NotImplementedError

    def infer_full(self, bgr: np.ndarray, fx: float | None = None) -> dict:
        """`infer` at the *original* image resolution.

        Several backends predict at their own internal size and return it as-is --
        Depth Anything 3, for instance, runs a 2048x1362 frame at 336x504.  Sampling
        a target pixel in that output would read a completely different place in the
        scene, so every consumer must go through this wrapper rather than `infer`.
        """
        import cv2
        try:
            out = self.infer(bgr, fx=fx) if fx else self.infer(bgr)
        except TypeError:                      # backends that take no intrinsics
            out = self.infer(bgr)
        h, w = bgr.shape[:2]
        for key in ("depth_m", "relative"):
            arr = out.get(key)
            if arr is not None and arr.shape[:2] != (h, w):
                out[f"{key}_native_hw"] = tuple(arr.shape[:2])
                out[key] = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        return out

    def unload(self):
        self.model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


class UniDepthV2(Backend):
    name = "unidepth_v2"
    kind = "metric"
    paper = "Piccinelli et al., UniDepthV2, TPAMI 2025"

    def __init__(self, backbone: str = "vitl14"):
        self.backbone = backbone
        self.model = None

    def load(self, device: str = "cuda", **kw):
        _add_path("UniDepth")
        import torch
        from unidepth.models import UniDepthV2 as _M
        self.device = device
        self.model = _M.from_pretrained(f"lpiccinelli/unidepth-v2-{self.backbone}")
        self.model = self.model.to(device).eval()
        self._torch = torch
        return self

    def infer(self, bgr):
        torch = self._torch
        rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1)
        with torch.no_grad():
            pred = self.model.infer(rgb.to(self.device))
        depth = pred["depth"].squeeze().float().cpu().numpy()
        K = pred.get("intrinsics")
        return {
            "depth_m": depth.astype(np.float32),
            "relative": None,
            "intrinsics": None if K is None else K.squeeze().float().cpu().numpy(),
        }


class DepthPro(Backend):
    name = "depth_pro"
    kind = "metric"
    paper = "Bochkovskii et al., Depth Pro, ICLR 2025 (the encoder SHARP is built on)"

    def __init__(self):
        self.model = None

    def load(self, device: str = "cuda", **kw):
        _add_path("ml-depth-pro")
        import torch
        import depth_pro
        ckpt = MODELS_DIR / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing weights: {ckpt}")
        import dataclasses
        cfg = dataclasses.replace(
            depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(ckpt)
        )
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=cfg, device=torch.device(device), precision=torch.half
        )
        self.model.eval()
        self.device, self._torch = device, torch
        return self

    def infer(self, bgr):
        torch = self._torch
        rgb = bgr[:, :, ::-1].copy()
        with torch.no_grad():
            pred = self.model.infer(self.transform(rgb), f_px=None)
        depth = pred["depth"].squeeze().float().cpu().numpy()
        f_px = float(pred["focallength_px"])
        h, w = depth.shape
        K = np.array([[f_px, 0, w / 2.0], [0, f_px, h / 2.0], [0, 0, 1]], np.float32)
        return {"depth_m": depth.astype(np.float32), "relative": None, "intrinsics": K}


class DA3Metric(Backend):
    name = "da3_metric"
    kind = "metric"
    paper = "Depth Anything 3, ICLR 2026 (DA3METRIC-LARGE; Apache 2.0, commercial OK)"

    def __init__(self, repo_id: str = "depth-anything/DA3METRIC-LARGE"):
        self.repo_id = repo_id
        self.model = None

    def load(self, device: str = "cuda", **kw):
        _shim_moviepy_editor()
        import torch
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(self.repo_id).to(device=device)
        self.device, self._torch = device, torch
        return self

    def infer(self, bgr, fx=None):
        """DA3METRIC-LARGE outputs *canonical* depth, not metres.

        The model card says "multiplying by focal length gives metric depth", and
        the repo's own `utils/alignment.apply_metric_scaling` spells it out:

            metric = canonical * focal / 300        focal = (fx + fy) / 2

        `DepthAnything3.inference()` does NOT apply this -- it returns the canonical
        tensor and leaves `intrinsics` as None -- so the caller must.  Reading
        `pred.depth` as metres (as this adapter first did) is simply wrong.

        The focal must be expressed in the *processed* image's pixels, because that
        is the resolution the depth tensor is at.  With no intrinsics supplied we
        fall back to fx = 0.7955 * width, the same fallback `metric3d_v2` uses, so
        the two are directly comparable.  (0.7955 is what SHARP's 30 mm-equivalent
        rule gives a 16:9 frame; on a 4:3 frame such as comma2k19's 1164x874 SHARP
        itself would use 0.867 * width.)

        ⚠ So DA3METRIC is *not* camera-agnostic: like Metric3D it needs a focal
        length, and its metres inherit whatever error that focal carries.
        """
        rgb = bgr[:, :, ::-1].copy()
        h0, w0 = bgr.shape[:2]
        if fx is None:
            fx = 0.7955 * w0
        pred = self.model.inference([rgb])
        canonical = np.asarray(pred.depth)[0].astype(np.float32)
        proc_h, proc_w = canonical.shape[:2]
        fx_proc = fx * (proc_w / float(w0))
        depth = canonical * (fx_proc / 300.0)
        K = np.array([[fx, 0, w0 / 2.0], [0, fx, h0 / 2.0], [0, 0, 1]], np.float32)
        return {"depth_m": depth, "relative": None, "intrinsics": K,
                "canonical_median": float(np.median(canonical))}


class DepthAnythingV2(Backend):
    name = "depth_anything_v2"
    kind = "relative"
    paper = "Yang et al., Depth Anything V2, NeurIPS 2024 (affine-invariant disparity)"

    CFG = {
        "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
        "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
        "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
    }

    def __init__(self, encoder: str = "vitl"):
        self.encoder = encoder
        self.model = None

    def load(self, device: str = "cuda", **kw):
        root = _add_path("Depth-Anything-V2")
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2 as _M
        ckpt = root / "checkpoints" / f"depth_anything_v2_{self.encoder}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing weights: {ckpt}")
        self.model = _M(**self.CFG[self.encoder])
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = self.model.to(device).eval()
        self.device, self._torch = device, torch
        return self

    def infer(self, bgr):
        with self._torch.no_grad():
            disp = self.model.infer_image(bgr)      # affine-invariant disparity, larger = nearer
        return {"depth_m": None, "relative": disp.astype(np.float32), "intrinsics": None}


class Metric3DV2(Backend):
    name = "metric3d_v2"
    kind = "metric"
    paper = "Hu et al., Metric3D v2, TPAMI 2024 (requires camera intrinsics)"

    def __init__(self, variant: str = "metric3d_vit_large"):
        self.variant = variant
        self.model = None

    def load(self, device: str = "cuda", **kw):
        import torch
        # trust_repo: torch.hub otherwise blocks on an interactive y/N prompt
        self.model = torch.hub.load("yvanyin/metric3d", self.variant,
                                    pretrain=True, trust_repo=True)
        self.model = self.model.to(device).eval()
        self.device, self._torch = device, torch
        return self

    INPUT_SIZE = (616, 1064)   # the ViT variants' canonical input; ConvNeXt uses (544, 1216)

    def infer(self, bgr, fx=None):
        """Metric3D needs intrinsics.  With none supplied we fall back to
        fx = 0.7955 * width -- SHARP's 30 mm-equivalent focal on a 16:9 frame.  SHARP
        scales that rule with the image diagonal, so on a 4:3 frame (comma2k19,
        1164x874) it would use 0.867 * width instead; the fallback is kept fixed so
        existing results reproduce, and `run --fx` measures what the choice costs.

        The resize-and-pad to the canonical input size is not optional: the model
        expects it, and feeding a full-resolution frame straight in both breaks the
        de-canonicalisation and blows up VRAM (2048x1362 asks for 12 GB).
        """
        import cv2
        torch = self._torch
        h0, w0 = bgr.shape[:2]
        if fx is None:
            fx = 0.7955 * w0
        rgb = bgr[:, :, ::-1].copy()

        # keep-ratio resize, then pad to the canonical size
        scale = min(self.INPUT_SIZE[0] / h0, self.INPUT_SIZE[1] / w0)
        rgb = cv2.resize(rgb, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
        pad_h, pad_w = self.INPUT_SIZE[0] - h, self.INPUT_SIZE[1] - w
        t, l = pad_h // 2, pad_w // 2
        b, r = pad_h - t, pad_w - l
        rgb = cv2.copyMakeBorder(rgb, t, b, l, r, cv2.BORDER_CONSTANT,
                                 value=[123.675, 116.28, 103.53])

        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        inp = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        inp = ((inp - mean) / std)[None].to(self.device)

        with torch.no_grad():
            pred, _conf, _out = self.model.inference({"input": inp})

        depth = pred.squeeze()
        depth = depth[t:depth.shape[0] - b, l:depth.shape[1] - r]              # un-pad
        depth = torch.nn.functional.interpolate(
            depth[None, None], (h0, w0), mode="bilinear").squeeze()            # back to original
        # de-canonical: the canonical camera has f = 1000 px, and the intrinsics must
        # be scaled by the same factor the image was
        depth = depth * (fx * scale / 1000.0)
        depth = torch.clamp(depth, 0, 300).float().cpu().numpy()

        K = np.array([[fx, 0, w0 / 2.0], [0, fx, h0 / 2.0], [0, 0, 1]], np.float32)
        return {"depth_m": depth.astype(np.float32), "relative": None, "intrinsics": K}


def _da2(encoder: str):
    """Register each Depth Anything V2 size as its own backend name.

    The sizes are not interchangeable for this project: only **vits** (Small) is
    Apache 2.0.  vitb / vitl / vitg are CC-BY-NC-4.0, so they cannot be deployed
    in anything commercial.
    """
    def make(**kw):
        b = DepthAnythingV2(encoder=encoder, **kw)
        b.name = f"depth_anything_v2_{encoder}"
        if encoder == "vits":
            b.paper = DepthAnythingV2.paper + " -- Small, Apache 2.0, commercial OK"
        else:
            b.paper = DepthAnythingV2.paper + f" -- {encoder}, CC-BY-NC-4.0, NON-commercial"
        return b
    return make


_REGISTRY = {
    "unidepth_v2": UniDepthV2,
    "depth_pro": DepthPro,
    "da3_metric": DA3Metric,
    "depth_anything_v2": DepthAnythingV2,          # defaults to vitl (CC-BY-NC)
    "depth_anything_v2_vits": _da2("vits"),        # Apache 2.0
    "depth_anything_v2_vitb": _da2("vitb"),        # CC-BY-NC
    "depth_anything_v2_vitl": _da2("vitl"),        # CC-BY-NC
    "metric3d_v2": Metric3DV2,
}


def available() -> list[str]:
    return list(_REGISTRY)


def get_backend(name: str, **kw) -> Backend:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; available: {', '.join(_REGISTRY)}")
    return _REGISTRY[name](**kw)
