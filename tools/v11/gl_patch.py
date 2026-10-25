"""(a) GL, ground calibration per lane (CRITERIA.txt): text patches applied to a work copy of the repository's
tools/depth_dash_multicar.py (arm gl only). Every anchor must occur exactly once, else the patch stops.

The own-lane rows, the own fit, the acceptance of a frame and every other step are untouched; two more row sets (left
lane [x_l - W, x_l], right lane [x_r, x_r + W]) are fitted the same way, and a car whose lane is -1 / +1 reads with that
side's fit when it is accepted at the sample and |a_side / a_own - 1| <= GL_GUARD."""
from pathlib import Path

NEW_FUNCS = '''

# v11extra (a) GL, 2026-10-25 (development arm): the road rows of the two neighbouring lanes, fitted like the own lane.
GL_GUARD = 0.15      # a side fit is used only within this relative distance of the own fit's slope (CRITERIA.txt, fixed before)
GL_INSIDE = 0.70     # a side region needs this share of its width inside the image


def ground_rows_sides(depth, wd, A, boxes, h, w, bonnet, p=GROUND_RULER):
    """(left, right): (D_g, z_row, y) per half-resolution row of the left lane [x_l - W, x_l] and the right lane
    [x_r, x_r + W], W = x_r - x_l at that row; same rows, middle share, box mask and pixel count as ground_rows."""
    dh, dw = depth.shape
    vp, L, R = wd["vp_row"], wd["left"], wd["right"]
    mask = np.zeros((dh, dw), bool)
    for b in boxes:
        x0, y0, x1, y1, conf = b[:5]
        if conf < p["box_conf"]:
            continue
        bw, bh = x1 - x0, y1 - y0
        X0, X1 = (x0 - p["box_wpad"] * bw) * dw / w, (x1 + p["box_wpad"] * bw) * dw / w
        Y0, Y1 = y0 * dh / h, (y1 + p["box_hpad"] * bh) * dh / h
        mask[max(0, int(np.floor(Y0))): max(0, int(np.ceil(Y1)) + 1), max(0, int(np.floor(X0))): max(0, int(np.ceil(X1)) + 1)] = True
    out = ([], [])
    for hy in range(dh):
        y = (hy + 0.5) * h / dh
        if y <= vp + 1 or y >= bonnet - PARAMS["bonnet_px"]:
            continue
        dg = A / (y - vp)
        if not (p["d_min"] <= dg <= p["d_max"]):
            continue
        xl, xr = L["a"] * y + L["b"], R["a"] * y + R["b"]
        W = xr - xl
        if W < 10:
            continue
        pad = (1 - p["lane_keep"]) / 2 * W
        for k, (u0, u1) in enumerate(((xl - W + pad, xl - pad), (xr + pad, xr + W - pad))):
            c0f, c1f = int(np.ceil(u0 * dw / w)), int(np.floor(u1 * dw / w))
            full = c1f - c0f + 1
            c0, c1 = max(0, c0f), min(dw - 1, c1f)
            if full <= 0 or c1 < c0 or (c1 - c0 + 1) < GL_INSIDE * full:
                continue
            seg = depth[hy, c0: c1 + 1][~mask[hy, c0: c1 + 1]]
            seg = seg[np.isfinite(seg) & (seg > 0)]
            if len(seg) >= p["min_px"]:
                out[k].append((dg, float(np.median(seg)), y))
    return tuple(np.array(o, float).reshape(-1, 3) for o in out)


def gl_side_fit(fs, f):
    """The side fit with ok = accepted, the own fit accepted, and |a_side / a_own - 1| <= GL_GUARD."""
    if not fs:
        return None
    good = bool(fs["ok"] and f and f["ok"] and f["a"] and abs(fs["a"] / f["a"] - 1) <= GL_GUARD)
    return dict(fs, accepted=bool(fs["ok"]), ok=good)
'''

A_ANCHOR = '''def ground_fit(D_g, z, p=GROUND_RULER):'''

B_OLD = '''            rows.append(ground_rows(depth, wm, rul[int(np.argmin(np.abs(rf - fi)))]["A_local"], tr.get(stem, []), h, w, bonnet)
                        if wm is not None else None)'''
B_NEW = B_OLD + '''
            srows.append(ground_rows_sides(depth, wm, rul[int(np.argmin(np.abs(rf - fi)))]["A_local"], tr.get(stem, []), h, w,
                                           bonnet) if wm is not None else None)          # v11extra GL'''

C_OLD = '''        pend, rows = [], []                        # per sample: (object, z) waiting for the window fit; the road rows'''
C_NEW = C_OLD + '''
        srows = []                                 # v11extra GL: per sample (left rows, right rows) or None'''

D_OLD = '''        fits = ground_window_fits([s["t"] for s in samples], rows)
        for s, f, po in zip(samples, fits, pend):
            s["ground_fit"] = f
            if not (f and f["ok"]):
                continue                           # no accepted fit: no car readings on this frame
            for o, z in po:
                d = f["a"] * z + f["b"]
                if np.isfinite(d) and d > 0:
                    s["objects"].append(dict(o, dist=float(d), far=bool(d > reach)))'''
D_NEW = '''        fits = ground_window_fits([s["t"] for s in samples], rows)
        tt = [s["t"] for s in samples]                                   # v11extra GL: the two side fits
        fitsL = ground_window_fits(tt, [x[0] if x is not None else None for x in srows])
        fitsR = ground_window_fits(tt, [x[1] if x is not None else None for x in srows])
        for s, f, po, fl, fr in zip(samples, fits, pend, fitsL, fitsR):
            s["ground_fit"] = f
            s["ground_fit_left"], s["ground_fit_right"] = gl_side_fit(fl, f), gl_side_fit(fr, f)
            if not (f and f["ok"]):
                continue                           # no accepted fit: no car readings on this frame
            for o, z in po:
                side = {-1: "left", 1: "right"}.get(o.get("lane"))
                fs = s.get(f"ground_fit_{side}") if side else None
                use = fs if (fs and fs.get("ok")) else f
                d = use["a"] * z + use["b"]
                if np.isfinite(d) and d > 0:
                    s["objects"].append(dict(o, dist=float(d), far=bool(d > reach), gfit=side if use is fs else "own"))'''


def apply(path):
    p = Path(path)
    t = p.read_text()
    for old in (A_ANCHOR, B_OLD, C_OLD, D_OLD):
        assert t.count(old) == 1, old[:60]
    t = t.replace(A_ANCHOR, NEW_FUNCS.lstrip("\n") + "\n\n" + A_ANCHOR)
    t = t.replace(B_OLD, B_NEW).replace(C_OLD, C_NEW).replace(D_OLD, D_NEW)
    p.write_text(t)


if __name__ == "__main__":
    import sys
    apply(sys.argv[1])
