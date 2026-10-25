"""(c) EG-T, truncation-free speed ruler (CRITERIA.txt addendum 5): text patches applied to a work copy of the
repository's tools/marking_geometry.py (arm eg only). Every anchor must occur exactly once, else the patch stops.

The brightness profile of a marking run ends near 30 m. A cycle that starts far can only be found when it is measured
short enough to end inside the profile, so the median cycle -- the ruler -- is pulled short and A long (truth-free:
cycles starting <= 9 m are about 1 % longer than the median of all cycles). EG-T: the per-line rulers of the SPEED step
(A_side, local and clip-wide) use only cycles that a cycle of 1.25 legal lengths plus one dash would still fit after
(z_near + (1.25 + duty) L <= the far end of that sample's profile). The pooled ruler A_local (distances, ground ruler),
A, the gates and everything else are unchanged."""
from pathlib import Path

EG_FIT = 1.25

A_OLD = '''            for cyc, z0 in D.dash_cycles(pr[0], pr[1], L, duty, dp):
                out.append(dict(n=s["n"], t=s["t"], side=side, period=cyc, z_near=z0, slope=g[side]["a"]))'''
A_NEW = '''            for cyc, z0 in D.dash_cycles(pr[0], pr[1], L, duty, dp):
                out.append(dict(n=s["n"], t=s["t"], side=side, period=cyc, z_near=z0, slope=g[side]["a"],
                                z_end=float(pr[0][-1])))       # v11extra EG-T: the far end of this profile'''

B_OLD = '''    A_side = {sd: sc["sides"][sd]["A"] for sd in acc}
    local_rulers(samples, cycles, acc, L, A0, A, A_side, p["local_s"], p["local_min"])'''
B_NEW = '''    A_side = {sd: sc["sides"][sd]["A"] for sd in acc}
    local_rulers(samples, cycles, acc, L, A0, A, A_side, p["local_s"], p["local_min"])
    # v11extra EG-T: the speed rulers (A_side) from the cycles that could not have been cut by the profile's far end
    fit = [c for c in cycles if c["z_near"] + (%r + duty) * L <= c["z_end"]]
    eg_side = {sd: (L * A0 / float(np.median([c["period"] for c in fit if c["side"] == sd]))
                    if any(c["side"] == sd for c in fit) else A_side[sd]) for sd in acc}
    keep_ = [(s["A_local"], s["n_local"], s["ruler"], s["A_side"]) for s in samples]
    local_rulers(samples, fit, acc, L, A0, A, eg_side, p["local_s"], p["local_min"])
    for s, (al, nl, ru, _as) in zip(samples, keep_):
        s["A_side_all"] = _as
        s["A_local"], s["n_local"], s["ruler"] = al, nl, ru
    res["eg_t"] = dict(fit=%r, cycles=len(cycles), cycles_fit=len(fit),
                       A_side_all={sd: _f(v, 2) for sd, v in A_side.items()}, A_side_fit={sd: _f(v, 2) for sd, v in eg_side.items()})''' % (EG_FIT, EG_FIT)


def apply(path):
    p = Path(path)
    t = p.read_text()
    for old in (A_OLD, B_OLD):
        assert t.count(old) == 1, old[:60]
    p.write_text(t.replace(A_OLD, A_NEW).replace(B_OLD, B_NEW))


if __name__ == "__main__":
    import sys
    apply(sys.argv[1])
