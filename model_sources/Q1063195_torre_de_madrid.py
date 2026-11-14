"""
Torre de Madrid (Q1063195), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Julián Otamendi Machimbarrena (1889–1966, architect) with his brother José María Otamendi
(1885–1959, structural engineer), 1954–1957 (first phase finished 15 October 1957, second phase 1960), for the
Compañía Inmobiliaria Metropolitana; once the tallest concrete building in the world. Façades and interiors
rehabilitated 2010–2016 (flats on floors 10–34, Barceló hotel on floors 1–9). Rights: protected.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 4085065 (the whole block, height 142, 37 levels) and its building:part pieces, in
  model x/z for heading 143.7° — the direction the tower's long south-east face looks, flush with the
  Plaza de España frontage of the block:
  · the tower shaft (relation 4085066, 33 levels): an elongated hexagon, about 35 × 34 m, with two chamfered
    corners (front-right and back-left);
  · floor 34 set back about 1.4 m (way 307130335) and the crown, floors 35–37 (way 307130336);
  · the round bay on the back-left chamfer (way 307130337, 37 levels);
  · two 16-level wings against the tower (ways 307130344, 307130345) and the 8-level podium (way 307130340).
- Heights: 142 m to the top of the crown (OSM, en.wikipedia); 128 m to the roof of floor 34 («altura de la
  azotea», es.wikipedia). Floors on the grid: ground floor 5 m, then 3.73 m (34 floors to 128.1 m); the
  podium and the wings 3.5 m.
- From the photos (Wikimedia Commons, «Torre de Madrid 2025.jpg» and others, after the 2010–2016
  rehabilitation): cream-white cladding with thin vertical pilaster strips between the window columns, stacks of
  rounded balconies on the corners of the hexagon, a deep projecting cornice over the shaft, the crown with its
  stepped parapet rising at the corners, and a short mast (the tall antenna that once took it to 162 m is not
  on the current photos and is not modelled).

Axes (docs/MODEL_SPEC.md): +Y up, the south-east face (towards the Plaza de España) faces +Z, +X points to the
north-east, origin = centroid of the OSM outline of the block,
y = 0 the ground there.

    python model_sources/Q1063195_torre_de_madrid.py   -> models/Q1063195/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import (GLASS, Mesh, box, cap, cyl, lathe, merge, offset, ring_area,  # noqa: E402
                     walls, write_glb)

# ── OpenStreetMap relation 4085065 and its building:part pieces (© OpenStreetMap contributors, ODbL), in model
#    metres for heading 143.7 and origin 40.4244747, -3.712379 (tools/osm.py) ────────────────────────────────
OUTLINE = [(-31.16, 2.29), (-31.17, 27.68), (-15.52, 27.68), (-9.13, 27.69), (8.44, 27.68), (11.42, 27.69), (13.85, 27.68), (15.61, 26.12), (18.39, 23.65), (23.17, 19.41), (23.55, 16.26), (24.09, 11.77), (24.68, 6.95), (25.53, -0.14), (25.9, -3.2), (27.19, -13.39), (29.22, -29.36), (20.46, -30.61), (19.87, -25.77), (-15.65, -33.24), (-20.29, 2.74)]
PARTS = [   # (OSM id, height tag, levels tag, min level, outline x/z)
    ('way/307130335', None, 34, None, [(24.04, -2.2), (3.53, -5.05), (2.68, -4.29), (-0.85, -1.11), (-7.21, 4.59), (-7.75, 5.08), (-7.76, 26.22), (13.19, 25.97), (21.54, 18.56)]),
    ('way/307130336', 147.0, 37, None, [(-0.85, -1.11), (2.68, -4.29), (7.43, -3.67), (7.24, -2.2), (21.88, -0.28), (19.58, 17.9), (12.85, 24.15), (9.29, 24.15), (-5.64, 24.18), (-5.73, 9.23), (-7.16, 9.23), (-7.21, 4.59)]),
    ('way/307130337', None, 37, None, [(-1.38, -1.71), (-0.85, -1.11), (0.48, 0.36), (-2.17, 2.71), (-4.02, 0.64), (-4.83, -0.27), (-4.79, -0.55), (-4.64, -0.95), (-4.36, -1.36), (-4.09, -1.73), (-3.79, -1.99), (-3.38, -2.23), (-2.85, -2.49), (-2.38, -2.58), (-2.2, -2.63)]),
    ('way/307130340', None, 8, None, [(19.87, -25.77), (20.46, -30.61), (29.22, -29.36), (27.19, -13.39), (25.9, -3.2), (23.25, -3.57), (25.31, -18.81), (7.32, -21.2), (5.34, -6.02), (3.72, -6.24), (-1.38, -1.71), (-4.02, 0.64), (-9.07, 5.13), (-9.07, 6.79), (-24.37, 6.79), (-24.41, 25.12), (-9.12, 25.12), (-9.13, 27.69), (-15.52, 27.68), (-31.17, 27.68), (-31.16, 2.29), (-20.29, 2.74), (-15.65, -33.24)]),
    ('way/307130344', None, 16, None, [(23.25, -3.57), (14.65, -4.75), (5.34, -6.02), (7.32, -21.2), (25.31, -18.81)]),
    ('way/307130345', None, 16, None, [(-9.07, 6.79), (-9.1, 17.01), (-9.12, 25.12), (-24.41, 25.12), (-24.37, 6.79)]),
    ('relation/4085066', None, 33, None, [(25.9, -3.2), (23.25, -3.57), (14.65, -4.75), (5.34, -6.02), (3.72, -6.24), (-1.38, -1.71), (-4.02, 0.64), (-9.07, 5.13), (-9.07, 6.79), (-9.1, 17.01), (-9.12, 25.12), (-9.13, 27.69), (8.44, 27.68), (11.42, 27.69), (13.85, 27.68), (15.61, 26.12), (18.39, 23.65), (23.17, 19.41), (23.55, 16.26), (24.09, 11.77), (24.68, 6.95), (25.53, -0.14)]),
]


# ── Colours and constants ────────────────────────────────────────────────────
STONE = np.array((0.905, 0.88, 0.815))         # cream-white cladding (as rehabilitated)
STONE_L = np.array((0.95, 0.935, 0.89))        # pilaster strips, balconies, cornices
STONE_D = np.array((0.80, 0.78, 0.72))         # underside of the cornices, podium base
ROOF = (0.64, 0.635, 0.615)
GLASS_C = (0.21, 0.26, 0.31)
SHOP = (0.17, 0.20, 0.23)
METAL_C = (0.50, 0.52, 0.55)
FOUND = -5.0

G0 = 5.0                                       # ground floor
PT = 3.73                                      # tower floors
PL = 3.5                                       # podium and wings
SHAFT = G0 + 32 * PT                           # 124.4: top of floor 33
F34 = G0 + 33 * PT                             # 128.1: roof of floor 34 («altura de la azotea» 128 m)
CROWN = G0 + 36 * PT                           # 139.3: roof of floor 37
TOP = 142.0                                    # top of the stepped parapet (OSM height, en.wikipedia)

HEIGHTS = {"relation/4085066": SHAFT, "way/307130335": F34, "way/307130336": CROWN, "way/307130337": CROWN,
           "way/307130344": G0 + 15 * PL, "way/307130345": G0 + 15 * PL, "way/307130340": G0 + 7 * PL}
rng = np.random.default_rng(1957)


# ── Geometry helpers ─────────────────────────────────────────────────────────
def ccw(P):
    P = [tuple(p) for p in P]
    return P if ring_area(P) > 0 else P[::-1]


def pip(pts, poly):
    """Even-odd point in polygon for many points (N,2)."""
    P = np.asarray(poly, float)
    x, z = pts[:, 0][:, None], pts[:, 1][:, None]
    xi, zi = P[:, 0][None, :], P[:, 1][None, :]
    xj, zj = np.roll(P[:, 0], 1)[None, :], np.roll(P[:, 1], 1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        hit = ((zi > z) != (zj > z)) & (x < (xj - xi) * (z - zi) / (zj - zi) + xi)
    return (np.sum(hit, 1) % 2) == 1


class Solids:
    """The prisms of the massing, to know which bits of wall are open to the air."""

    def __init__(self):
        self.items = []                              # (poly, top)

    def add(self, poly, top):
        self.items.append((np.asarray(poly, float), float(top)))

    def hidden(self, pts, ys, skip=None):
        pts = np.asarray(pts, float).reshape(-1, 2); ys = np.asarray(ys, float).ravel()
        occ = np.zeros(len(pts), bool)
        for k, (poly, top) in enumerate(self.items):
            if k == skip:
                continue
            m = (ys < top) & ~occ
            if m.any():
                idx = np.where(m)[0]
                occ[idx[pip(pts[idx], poly)]] = True
        return occ


def edges(P):
    """(a, b, unit direction, outward normal, length) of a counter-clockwise ring, merging straight runs."""
    P = [np.array(p, float) for p in P]
    keep = []
    for i in range(len(P)):
        d0 = P[i] - P[i - 1]; d1 = P[(i + 1) % len(P)] - P[i]
        l0, l1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if l0 < 1e-6 or l1 < 1e-6:
            continue
        if abs(d0[0] * d1[1] - d0[1] * d1[0]) / (l0 * l1) > 0.012 or np.dot(d0, d1) < 0:
            keep.append(P[i])
    out = []
    for i in range(len(keep)):
        a, b = keep[i], keep[(i + 1) % len(keep)]
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        e = (b - a) / L
        out.append((a, b, e, np.array([e[1], -e[0]]), L))
    return out


def wall_quad(a, e, n, s0, s1, y0, y1, off, c):
    """A flat rectangle on a wall, from s0 to s1 along the edge, off metres in front of it, facing out."""
    p0 = a + e * s0 + n * off; p1 = a + e * s1 + n * off
    V = [(p0[0], y0, p0[1]), (p1[0], y0, p1[1]), (p1[0], y1, p1[1]), (p0[0], y1, p0[1])]
    return Mesh(V, [(0, 2, 1), (0, 3, 2)], c)


def edge_box(a, e, n, s0, s1, y0, y1, o0, o1, c):
    """A box lying along a wall: from s0 to s1 along it, o0..o1 metres out of it (negative = into the wall)."""
    w, d = s1 - s0, o1 - o0
    m = box(w, y1 - y0, d, c)
    m.R(0, math.atan2(-e[1], e[0]), 0)
    cx = a + e * (s0 + s1) / 2 + n * (o0 + o1) / 2
    return m.T(cx[0], y0, cx[1])


def runs(flags):
    """[(i0, i1)] runs of True in a boolean list (i1 exclusive)."""
    out, i0 = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and i0 is None:
            i0 = i
        elif not f and i0 is not None:
            out.append((i0, i)); i0 = None
    return out


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sphere(r, c, seg=8, rings=5):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)




def columns(L, pitch, m0, m1):
    """Evenly spaced window columns along a wall of length L, keeping m0/m1 clear at its ends."""
    span = L - m0 - m1
    if span < 0:
        return np.zeros(0)
    n = int(span // pitch) + 1
    return m0 + (span - (n - 1) * pitch) / 2 + np.arange(n) * pitch


def tower_part(pid):
    return pid in ("relation/4085066", "way/307130335", "way/307130336", "way/307130337")


def rows_for(pid, top):
    """(y0, y1, width, kind) of the window rows of a piece."""
    if tower_part(pid):
        rows = [(0.4, 4.4, 2.2, "shop")]
        rows += [(G0 + k * PT + 0.75, G0 + k * PT + 2.65, 1.25, "std") for k in range(36)]
    else:
        rows = [(0.4, 4.4, 2.4, "shop")]
        rows += [(G0 + k * PL + 0.75, G0 + k * PL + 2.55, 1.35, "std") for k in range(15)]
    return [r for r in rows if r[1] < top - 0.7]


def build():
    stone, glass = [], []
    parts = [(pid, HEIGHTS[pid], ccw(P)) for pid, h, lv, ml, P in PARTS]
    parts.append(("outline", G0 + 7 * PL - 1.0, ccw(offset(ccw(OUTLINE), -0.4))))
    tops = sorted({round(t, 2) for _, t, _ in parts})
    solids = Solids()
    geo = []
    for pid, top, P in parts:
        r = tops.index(round(top, 2))
        G = offset(P, 0.02 * r) if r else P
        geo.append((pid, top + (0.004 if pid == "way/307130337" else 0.0), ccw(G)))
        solids.add(P, top)
    for i, (pid, top, G) in enumerate(geo):
        stone.append(walls(G, FOUND, top, STONE))
        stone.append(cap(G, top, ROOF))

    for i, (pid, top, G) in enumerate(geo):
        if pid == "outline":
            continue
        tower = tower_part(pid)
        round_bay = pid == "way/307130337"
        EG = edges(G)
        for ei, (a, b, e, n, L) in enumerate(EG):
            kk = 2 if (ei == len(EG) - 1 and len(EG) % 2) else ei % 2
            if L < 0.9:
                continue
            # cornices: a deep one over the shaft, lighter ones elsewhere
            ns = max(1, int(round(L / 0.8)))
            ss = (np.arange(ns) + 0.5) * L / ns
            pts = a + np.outer(ss, e) + n * 0.35
            open_top = ~solids.hidden(pts, np.full(ns, top - 0.35))
            cont = [bool(solids.hidden((end + e * sg * 0.6 - n * 0.3)[None], np.array([top - 0.35]))[0])
                    for end, sg in ((a, -1), (b, 1))]           # the roof edge goes on past this end
            for i0, i1 in runs(open_top):
                s0, s1 = i0 * L / ns, i1 * L / ns
                ext = 1.7 if pid == "relation/4085066" else 0.5
                s0 -= ext if (i0 == 0 and not cont[0]) else 0.0; s1 += ext if (i1 == ns and not cont[1]) else 0.0
                lift = 0.013 * kk + 0.004 * (i % 2)
                if pid == "relation/4085066":                       # the deep cornice over the shaft
                    stone.append(edge_box(a, e, n, s0, s1, top - 0.75 - lift, top + 0.3 + lift, -0.3, 1.7, STONE_L))
                elif pid == "way/307130336":                        # crown: parapet (the stepped corners apart)
                    stone.append(edge_box(a, e, n, s0, s1, top - 0.4 - lift, top + 1.4 + lift, -0.3, 0.25, STONE_L))
                    stone.append(edge_box(a, e, n, s0 + 0.2, s1 - 0.2, top + 1.4 + lift, top + 1.5 + lift, -0.25,
                                          0.05, METAL_C))
                elif round_bay:
                    stone.append(edge_box(a, e, n, s0, s1, top - 0.4 - lift, top + 0.6 + lift, -0.2, 0.3, STONE_L))
                else:
                    stone.append(edge_box(a, e, n, s0, s1, top - 0.6 - lift, top + 0.25 + lift, -0.25, 0.55, STONE_L))
                    stone.append(edge_box(a, e, n, s0, s1, top + 0.25 + lift, top + 1.1 + lift, -0.05, 0.2, STONE))
            # windows
            m = []
            for end, sg in ((a, -1), (b, 1)):
                q = end + e * sg * 0.6 - n * 0.3
                m.append(0.0 if solids.hidden(q[None], np.array([top - 2.0]))[0] else 0.8)
            pitch = 2.4 if tower else 3.0
            s_w = columns(L, pitch, *m) if L > 1.6 else np.zeros(0)
            s_w = s_w[(s_w > 0.05) & (s_w < L - 0.05)]
            cand = [(s, r) for r in rows_for(pid, top) for s in s_w]
            if cand:
                cp = np.array([a + e * s + n * 0.4 for s, r in cand])
                cy = np.array([(r[0] + r[1]) / 2 for s, r in cand])
                vis = ~solids.hidden(cp, cy)
                for (s, r), v in zip(cand, vis):
                    if not v:
                        continue
                    y0, y1, w, kind = r
                    w = min(w, L * 0.6)
                    c = SHOP if kind == "shop" else tuple(np.clip(np.array(GLASS_C) * (1 + rng.normal(0, 0.08)), 0, 1))
                    glass.append(wall_quad(a, e, n, s - w / 2, s + w / 2, y0, y1, 0.05, c))
            # pilaster strips between the window columns of the tower
            if tower and not round_bay and len(s_w) > 1 and L > 4:
                for s in (s_w[:-1] + s_w[1:]) / 2:
                    lo = None
                    for y in np.arange(G0 + 1.0, top - 1.0, 4.0):
                        q = a + e * s + n * 0.3
                        if not solids.hidden(q[None], np.array([y]))[0]:
                            lo = y - 1.0
                            break
                    if lo is not None:
                        stone.append(edge_box(a, e, n, s - 0.22, s + 0.22, max(lo, G0 - 0.5), top - 0.6, -0.05, 0.2,
                                              STONE_L))
            # the base of the podium and the wings: a light band with the shop windows
            okb = ~solids.hidden(pts, np.full(ns, 2.0))
            for i0, i1 in runs(okb):
                lift2 = 0.013 * kk
                stone.append(edge_box(a, e, n, i0 * L / ns - 0.45 * (i0 == 0), i1 * L / ns + 0.45 * (i1 == ns), G0 - 0.25 - lift2, G0 + 0.35 + lift2, -0.1, 0.4, STONE_D))
    return stone, glass, solids, {pid: G for pid, _, G in geo}


def corners(P, min_turn=25):
    """Convex corners of a ring (counter-clockwise): (point, outward bisector)."""
    out = []
    E = edges(P)
    for k in range(len(E)):
        e0, n0 = E[k - 1][2], E[k - 1][3]
        e1, n1 = E[k][2], E[k][3]
        cr = e0[0] * e1[1] - e0[1] * e1[0]
        turn = math.degrees(math.asin(max(-1.0, min(1.0, cr))))
        if cr > math.sin(math.radians(min_turn)) or (np.dot(e0, e1) < 0 and cr > 0):
            bis = n0 + n1
            out.append((E[k][0], bis / np.linalg.norm(bis), turn))
    return out


def balconies(solids, geo):
    """Stacks of rounded balconies on the corners of the tower shaft (from the photos)."""
    out = []
    for p, bis, turn in corners(geo["relation/4085066"]):
        for k in range(1, 32):
            y = G0 + k * PT
            q = p + bis * 1.2
            if solids.hidden(q[None], np.array([y + 0.5]))[0]:
                continue
            c = p + bis * 0.15
            out.append(cyl(1.45, 1.45, 0.2, STONE_L, 14).T(c[0], y - 0.2, c[1]))
            out.append(cyl(1.42, 1.42, 0.85, STONE_L, 14, caps=False).T(c[0], y, c[1]))
    return out


def crown(geo):
    """The stepped parapet rising at the corners of the crown, the cornice of floor 34 and the mast."""
    out = []
    P = geo["way/307130336"]
    used = []
    for p, bis, turn in corners(P, 30):
        if any(np.linalg.norm(p - q) < 6.0 for q in used):
            continue
        used.append(p)
        ang = math.atan2(bis[1], bis[0])
        for size, y1 in ((4.2, CROWN + 1.9), (2.8, CROWN + 2.4), (1.6, TOP)):
            c = p - bis * (size * 0.707)
            out.append(box(size, y1 - CROWN, size, STONE_L).R(0, -ang + math.pi / 4, 0).T(c[0], CROWN, c[1]))
    # floor 34: a lighter cornice
    cx = np.mean([q[0] for q in P]); cz = np.mean([q[1] for q in P])
    out.append(cyl(0.18, 0.08, 6.0, METAL_C, 6).T(cx, CROWN, cz))
    out.append(box(3.0, 2.6, 3.0, STONE).T(cx, CROWN, cz))          # lift machinery under the mast
    return out


def main():
    out = ROOT / "models" / "Q1063195"
    out.mkdir(parents=True, exist_ok=True)
    stone, glass, solids, geo = build()
    m_stone = merge(stone + balconies(solids, geo) + crown(geo))
    m_glass = merge(glass)
    tris, size = write_glb(out / "model.glb", [("torre_de_madrid", m_stone), ("glass", m_glass, {"material": "glass"})],
                           roughness=0.88, materials={"glass": GLASS})
    allv = np.vstack([m_stone.v, m_glass.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
