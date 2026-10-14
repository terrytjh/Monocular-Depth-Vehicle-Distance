"""MoGe-2 depth backend (Microsoft MoGe, MIT licence; weights Ruicheng/moge-2-vitl-normal, MIT), used by the eighth version
(docs/DEPTH_DASH_V8_PREREG.md, step S6). Same interface as tools/depth_backends.py: depth = the z of MoGe's metric point
map for one image. The vendored repository is depth_models/MoGe (github.com/microsoft/MoGe); its two extra packages
(utils3d, flexgemm) are installed outside the main environment, in ~/venvs/depth_extra.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_backends import Backend, _add_path  # noqa: E402

EXTRA = (str(Path.home() / "venvs/depth_extra"), str(Path.home() / "venvs/depth_extra/flexgemm"))


class MoGe(Backend):
    kind = "metric"
    paper = "Wang et al., MoGe-2 (NeurIPS 2025)"

    def __init__(self, version: str = "v2", repo: str = "Ruicheng/moge-2-vitl-normal", **kw):
        self.version, self.repo, self.model = version, repo, None
        self.name = f"moge_{version}"

    def load(self, device: str = "cuda", **kw):
        import importlib
        import torch
        for x in EXTRA:
            if x not in sys.path:
                sys.path.insert(0, x)
        _add_path("MoGe")
        mod = importlib.import_module(f"moge.model.{self.version}")
        self.model = mod.MoGeModel.from_pretrained(self.repo).to(device).eval()
        self.device, self._torch = device, torch
        return self

    def infer(self, bgr):
        torch = self._torch
        rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.no_grad():
            out = self.model.infer(rgb, apply_mask=False)
        return {"depth_m": out["depth"].float().cpu().numpy().astype(np.float32), "relative": None,
                "intrinsics": out["intrinsics"].float().cpu().numpy() if "intrinsics" in out else None}
