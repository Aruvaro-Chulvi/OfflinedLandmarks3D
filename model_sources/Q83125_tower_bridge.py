"""
Tower Bridge (Q83125), London — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The bridge: architect Sir Horace Jones (died 1887), engineer Sir John Wolfe Barry (died 1918), with Henry Marc
Brunel; after Jones's death George D. Stevenson gave the towers their Victorian Gothic stone dress. Built
1886–1894, opened 30 June 1894. Public domain (see meta.json "rights").

Measurements (see meta.json for the sources):
- OpenStreetMap relation 5641637 (Tower Bridge, Q83125), its building:parts, and the bridge outline way
  378541210. Heading from the centres of the two towers (ways 367652791 and 367652787, 80.5 m apart on a
  bearing of 22.9°; the long axis of the bridge outline is 22.6–23.3°). In model axes: the deck 19–20 m wide
  (x −9.8…9.3), the inner faces of the main towers and the ends of the walkways at z ±36.2, the tower bodies
  10.5 m along the bridge (z 36.2…46.7) and 20 m across, with small round corner turrets reaching z 35.0 and
  47.9; the walkways x −9.8…−5.5 and 5.0…9.2 at 40–45 m; the piers with rounded cutwaters 45 m across
  (x ±23) and ~20 m along the bridge (z 31…51); the tower roofs: a pyramidal roof 47–65 m, a gable across the
  river faces 39–54 m and one across the road faces 47–57 m; the abutment towers at z −142.6…−135.4 and
  134.7…141.7, 22.5 m across, 22 m of stone and a hipped roof to 30 m; the deck parts 7–9 m; the ends of the
  bridge outline at z −157.5 and +163.5.
- Wikipedia (en) «Tower Bridge»: towers 213 ft (65 m) high; central span 200 ft (61 m) between the piers;
  side suspension spans 270 ft (82 m); total length 940 ft (290 m); piers 185 × 70 ft (56 × 21 m, the size
  set by the Act, foundations included); walkways 143 ft (44 m) above high water, each cantilevered 55 ft
  (16.8 m) from its tower with girders spanning 120 ft (36.6 m) between the cantilever ends — so the tower
  faces are 70.1 m apart (OSM 72.4 m); clearance 29 ft (8.8 m) above Trinity high water with the bascules
  closed; deck 60 ft (18.3 m) between parapets, 49 ft (14.9 m) across the opening span; Cornish granite and
  Portland stone over the steel frame; the blue and white paint (restored in 2008–2012; red, white and blue
  only from 1977).
- EU-DEM 25 m via OpenTopoData along the axis: the river at 2.5 m, the north bank (Tower Hill side) 8–11 m
  above the water at z 120…160, the south bank (Shad Thames side) 4–5.5 m above the water at z −140…−160.
- Estimated (±): the roadway at 10.0 m over the river spans (underside of the bascules 8.9 m at mid-span,
  against 8.8 m in the Act) and the grades to the ends (7.4 m at the south end, 11.0 m at the north end,
  ±1 m); the walkway floor at 40.4 m and their roof at 45 m (OSM 40–45; Wikipedia's 44 m is taken as their
  top, ±1.5 m); the Gothic road arches (spring 17 m, apex 23.5 m in the main towers, ±2 m); the corner turrets
  (octagons 3.4 m across, pointed roofs to ~59 m), the gables and the flèche (±15 %); the curve of the
  suspension chains (upper chord 39 m at the main tower, low point 14.5 m about 30 m from the abutment tower,
  21 m at the abutment tower, ±2 m), their depth (2 m) and the hanger spacing (5.5 m); the bascule girders
  (four per leaf, 1.1 m deep at the tip, 4.2 m at the pier); the pier top at 4 m and the granite plinth to
  6 m; the windows (positions by eye from photographs, not measured).

Axes (docs/MODEL_SPEC.md): +Y up; +Z points along the bridge towards the north bank (the City, the Tower of
London side; heading 22.9°), +X points to 292.9° (west-north-west, upstream). A bridge has no main façade:
the axis was chosen so that the model runs along +Z. Origin at the middle of the opening span, on the water,
half-way between the two tower centres: y = 0 is the Thames at high water (the DEM water level, 2.5 m, is
taken as y = 0; mean high water springs at Tower Pier is about 1 m higher — within the ±1 m of the road
levels). meta.json gives "ground": "keep", so the banks at the two ends (4–11 m above the water) stay as
they are, and the abutments and approaches go down to y = −4 into them; the piers go down to y = −6.

    python model_sources/Q83125_tower_bridge.py   -> models/Q83125/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, extrude, quad, write_glb  # noqa: E402

STONE = (0.85, 0.81, 0.72)        # Portland stone
STONE_D = (0.74, 0.70, 0.62)
GRANITE = (0.62, 0.61, 0.59)      # Cornish granite
GRANITE_D = (0.50, 0.49, 0.48)
SLATE = (0.34, 0.37, 0.42)
BLUE = (0.43, 0.66, 0.85)         # the pale blue of the steelwork
BLUE_D = (0.31, 0.50, 0.67)
WHITE = (0.93, 0.94, 0.93)
ASPH = (0.27, 0.27, 0.29)
WALK = (0.62, 0.61, 0.58)
LINE = (0.90, 0.90, 0.86)
WIN = (0.15, 0.17, 0.21)
GLAZE = (0.50, 0.60, 0.68)
GOLD = (0.80, 0.66, 0.33)

# ── Main figures ─────────────────────────────────────────────────────────────
Z_PIER = 30.5               # faces of the piers (61 m opening span)
Z_T0, Z_T1 = 35.0, 47.0     # main tower body along the bridge (walkway girders 70.1 m between faces)
Z_PB = 51.5                 # back of the piers (pier ~21 m along the bridge)
Z_A0, Z_A1 = 135.0, 142.2   # abutment towers
Z_S, Z_N = -160.0, 163.5    # ends of the bridge (OSM outline)
TX = 10.0                   # main tower half-width across the bridge
T_TOP = 47.0                # top of the stone body (OSM)
W_MAIN = 7.45               # half-width between parapets over the opening span (49 ft)
W_SIDE = 9.15               # half-width between parapets on the side spans (60 ft)
WK_X, WK_HW = 7.4, 2.1      # walkways: centre x and half-width
WK_Y0, WK_Y1 = 40.0, 45.0
CH_X = 9.9                  # planes of the suspension chains
ROAD = [(Z_S, 7.4), (-Z_A1, 8.4), (-Z_PB, 10.0), (Z_PB, 10.0), (Z_A1, 10.6), (Z_N, 11.0)]
PIER_TOP = 4.0
PLINTH_TOP = 6.0
DEEP = -6.0                 # pier foundations shown down to here
BANK_LOW = -4.0             # abutments and approaches go down to here


def road_y(z):
    return np.interp(z, [p[0] for p in ROAD], [p[1] for p in ROAD])


# ── Helpers ──────────────────────────────────────────────────────────────────
def prism_z(profile, z0, z1, c, holes=()):
    """A prism whose cross-section (x, y) is extruded along z from z0 to z1."""
    m = extrude(profile, -z1, -z0, c, holes=holes, bottom=True)
    m.v = np.stack([m.v[:, 0], m.v[:, 2], -m.v[:, 1]], 1)
    return m


def prism_x(profile, x0, x1, c):
    """A prism whose cross-section (z, y) is extruded along x from x0 to x1."""
    m = extrude(profile, x0, x1, c, bottom=True)
    m.v = m.v[:, [1, 2, 0]]
    return m


def beam(p0, p1, w, h, c):
    """A straight bar of section w × h from p0 to p1 (w horizontal when possible)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    t = d / L
    a = np.cross([0.0, 1.0, 0.0], t)
    if np.linalg.norm(a) < 1e-6:
        a = np.array([1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(t, a)
    m = box(w, h, L, c, bottom=False)
    v = m.v
    m.v = v[:, :1] * a + v[:, 1:2] * b + v[:, 2:3] * t + (p0 + p1) / 2
    return m


def sweep(points, w, h, c):
    """A bar of rectangular section along a polyline lying in a plane x = const (w along x)."""
    P = np.asarray(points, float)
    n = len(P)
    a = np.array([1.0, 0.0, 0.0])
    rings = []
    for i in range(n):
        t = P[min(i + 1, n - 1)] - P[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        b = np.cross(t, a)
        b /= np.linalg.norm(b)
        rings.append([P[i] + sa * w / 2 * a + sb * h / 2 * b for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    rings = np.array(rings)
    ms = []
    for k in range(4):
        j = (k + 1) % 4
        v, f = [], []
        for i in range(n):
            v += [rings[i, k], rings[i, j]]
        for i in range(n - 1):
            o = 2 * i
            f += [(o, o + 1, o + 3), (o, o + 3, o + 2)]
        ms.append(Mesh(v, f, c))
    for i, s in ((0, 1), (n - 1, -1)):
        r = rings[i]
        ms.append(Mesh(r, [(0, 1, 2), (0, 2, 3)] if s < 0 else [(0, 2, 1), (0, 3, 2)], c))
    return merge(ms)


def solid(faces, centre, colours):
    """A convex solid from its faces (lists of points); each face wound outwards from centre."""
    ms = []
    centre = np.asarray(centre, float)
    for pts, c in zip(faces, colours):
        P = np.asarray(pts, float)
        n = np.zeros(3)
        for i in range(len(P)):
            p, q = P[i], P[(i + 1) % len(P)]
            n += np.cross(p, q)
        if np.dot(n, P.mean(0) - centre) < 0:
            P = P[::-1]
        f = [(0, i, i + 1) for i in range(1, len(P) - 1)]
        ms.append(Mesh(P, f, c))
    return merge(ms)


def hip_roof(x0, x1, z0, z1, y0, y1, c, ridge_frac=None):
    """A hipped roof over a rectangle; the ridge runs along the longer side (a point if ridge_frac = 0)."""
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    wx, wz = x1 - x0, z1 - z0
    if ridge_frac is None:
        r = abs(wx - wz) / 2
    else:
        r = ridge_frac * max(wx, wz) / 2
    if wx >= wz:
        A, B = (cx - r, y1, cz), (cx + r, y1, cz)
    else:
        A, B = (cx, y1, cz - r), (cx, y1, cz + r)
    p = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
    if wx >= wz:
        faces = [[p[0], p[1], B, A], [p[2], p[3], A, B], [p[3], p[0], A], [p[1], p[2], B]]
    else:
        faces = [[p[1], p[2], B, A], [p[3], p[0], A, B], [p[0], p[1], A], [p[2], p[3], B]]
    faces = [f for f in faces if len({tuple(np.round(q, 6)) for q in f}) >= 3]
    faces = [[q for i, q in enumerate(f) if all(np.linalg.norm(np.subtract(q, f[j])) > 1e-6 for j in range(i))]
             for f in faces]
    return solid(faces, (cx, y0 + 0.01, cz), [c] * len(faces))


def arch_ring(w, yb, ys, ya, n=8):
    """Opening of a pointed (Gothic) arch, width w, from yb up, springing at ys, apex ya — as an (x, y) ring."""
    hA = ya - ys
    c = max((hA * hA - w * w / 4) / w, 0.0)
    R = w / 2 + c
    pts = [(-w / 2, yb), (w / 2, yb), (w / 2, ys)]
    a0 = 0.0
    a1 = math.acos(c / R)
    for i in range(1, n + 1):                       # right arc, centre (−c, ys)
        a = a0 + (a1 - a0) * i / n
        pts.append((-c + R * math.cos(a), ys + R * math.sin(a)))
    for i in range(n - 1, -1, -1):                  # left arc, centre (c, ys)
        a = a0 + (a1 - a0) * i / n
        pts.append((c - R * math.cos(a), ys + R * math.sin(a)))
    pts[-1] = (-w / 2, ys)
    out = []
    for p in pts:
        if not out or math.dist(out[-1], p) > 1e-4:
            out.append(p)
    return out


def octagon(cx, cz, r, y0, y1, c):
    return cyl(r, r, y1 - y0, c, seg=8).R(0, math.pi / 8, 0).T(cx, y0, cz)


def spire(cx, cz, r, y0, y1, c, seg=8):
    return cyl(r, 0.0, y1 - y0, c, seg=seg, caps=False).R(0, math.pi / 8, 0).T(cx, y0, cz)


def ring_band(x0, x1, z0, z1, y0, y1, c, out=0.25):
    return box(x1 - x0 + 2 * out, y1 - y0, z1 - z0 + 2 * out, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def gable(face, u0, u1, y0, ya, depth, c_wall, c_roof, at, thick=None):
    """A gabled dormer: wall from y0 with its peak at ya, width u0..u1 on a face, running depth into the roof.
    face '+x'/'-x': u is z, the wall stands at x = at; '+z'/'-z': u is x, the wall at z = at."""
    ys = ya - (u1 - u0) / 2 * 1.1                     # eaves of the gable (steep, ~48°)
    s = 1 if face[0] == "+" else -1
    back = at - s * depth
    um = (u0 + u1) / 2

    def P(u, y, w):
        return (w, y, u) if face[1] == "x" else (u, y, w)
    front = [P(u0, y0, at), P(u1, y0, at), P(u1, ys, at), P(um, ya, at), P(u0, ys, at)]
    rear = [P(u0, y0, back), P(u1, y0, back), P(u1, ys, back), P(um, ya, back), P(u0, ys, back)]
    faces = [front, rear[::-1],
             [front[0], front[1], rear[1], rear[0]],
             [front[1], front[2], rear[2], rear[1]],
             [front[2], front[3], rear[3], rear[2]],
             [front[3], front[4], rear[4], rear[3]],
             [front[4], front[0], rear[0], rear[4]]]
    cols = [c_wall, c_wall, c_wall, c_wall, c_roof, c_roof, c_wall]
    cen = np.mean(front + rear, axis=0)
    return solid(faces, cen, cols)


# ── Piers ────────────────────────────────────────────────────────────────────
def stadium(hx, z0, z1, n=10):
    """A plan with semicircular cutwaters at x = ±hx (pier of a river bridge)."""
    r = (z1 - z0) / 2
    zc = (z0 + z1) / 2
    pts = []
    for s in (1, -1):
        xc = s * (hx - r)
        for i in range(n + 1):
            a = -math.pi / 2 + math.pi * i / n
            pts.append((xc + s * r * math.cos(a), zc + s * r * math.sin(a)))
    return pts


def pier(sgn):
    z0, z1 = sorted((sgn * Z_PIER, sgn * Z_PB))
    ms = [extrude(stadium(23.0, z0, z1), DEEP, PIER_TOP - 0.6, GRANITE_D),
          extrude(stadium(22.6, z0 + 0.4, z1 - 0.4), PIER_TOP - 0.6, PIER_TOP, GRANITE),
          box(24.0, PLINTH_TOP - PIER_TOP, z1 - z0 - 0.6, GRANITE).T(0, PIER_TOP, (z0 + z1) / 2)]
    # the pier's back, under the deck between the tower and the side span
    zb0, zb1 = sorted((sgn * (Z_T1 - 0.2), sgn * Z_PB))
    ms.append(box(2 * W_SIDE, 9.6 - PLINTH_TOP + 0.1, zb1 - zb0, GRANITE).T(0, PLINTH_TOP - 0.1, (zb0 + zb1) / 2))
    return merge(ms)


# ── Main towers ──────────────────────────────────────────────────────────────
def main_tower(sgn):
    z0, z1 = sorted((sgn * Z_T0, sgn * Z_T1))
    zc = (z0 + z1) / 2
    zi = sgn * Z_T0                     # inner face (towards the opening span)
    zo = sgn * Z_T1                     # outer face (towards the side span)
    stone, dark, roof, gold = [], [], [], []
    # body with the road arch through it (granite to the road, Portland stone above)
    body = [(-TX, PLINTH_TOP - 0.1), (TX, PLINTH_TOP - 0.1), (TX, T_TOP), (-TX, T_TOP)]
    hole = arch_ring(15.6, 9.8, 17.0, 23.5)
    stone.append(prism_z(body, z0, z1, STONE, holes=[hole]))
    stone.append(ring_band(-TX, TX, z0, z1, PLINTH_TOP - 0.1, 10.4, GRANITE, out=0.12))
    # string courses and the parapet
    for y in (24.6, 37.6):
        stone.append(ring_band(-TX, TX, z0, z1, y, y + 0.5, STONE_D, out=0.2))
    stone.append(ring_band(-TX, TX, z0, z1, T_TOP - 0.6, T_TOP + 1.3, STONE_D, out=0.3))
    # arch surround on both road faces: a moulding just outside the opening
    for zf, sd in ((z0, -1), (z1, 1)):
        O = arch_ring(17.2, 9.8, 17.0, 24.6)
        In = arch_ring(15.6, 9.8, 17.0, 23.5)
        U = O[1:] + [O[0]] + (In[1:] + [In[0]])[::-1]          # a U-shaped band round the opening
        stone.append(prism_z(U, zf - 0.25 if sd < 0 else zf, zf if sd < 0 else zf + 0.25, STONE_D))
    # corner turrets (octagons) with pointed roofs and gilded finials
    for x in (-TX, TX):
        for z in (z0, z1):
            stone.append(octagon(x, z, 1.75, PLINTH_TOP - 0.1, 50.5, STONE))
            stone.append(octagon(x, z, 1.95, 50.0, 50.6, STONE_D))
            stone.append(octagon(x, z, 1.95, 37.6, 38.1, STONE_D))
            roof.append(spire(x, z, 1.9, 50.6, 58.5, SLATE))
            gold.append(cyl(0.12, 0.05, 1.6, GOLD, seg=6).T(x, 58.4, z))
            for k in range(4):                       # slit windows in the turrets
                a = math.pi / 4 + k * math.pi / 2
                dx, dz = math.cos(a), math.sin(a)
                if dx * x > 0 or dz * (z - zc) > 0:
                    for y in (16.0, 28.0, 42.0):
                        dark.append(beam((x + 1.72 * dx, y, z + 1.72 * dz), (x + 1.72 * dx, y + 2.2, z + 1.72 * dz),
                                         0.45, 0.12, WIN))
    # battlements on the parapet (merlons), between the corner turrets
    for zf in (z0, z1):
        for x in np.arange(-TX + 2.4, TX - 2.3, 1.3):
            if abs(x) > 3.4:
                stone.append(box(0.6, 0.7, 0.5, STONE_D).T(x, T_TOP + 1.3, zf))
    for xf in (-TX, TX):
        for z in np.arange(z0 + 2.4, z1 - 2.3, 1.3):
            if abs(z - zc) > 2.7:
                stone.append(box(0.5, 0.7, 0.6, STONE_D).T(xf, T_TOP + 1.3, z))
    # the high central roof with a flèche
    roof.append(hip_roof(-TX + 0.6, TX - 0.6, z0 + 0.6, z1 - 0.6, T_TOP + 1.3, 61.5, SLATE, ridge_frac=0.08))
    stone.append(octagon(0, zc, 0.9, 60.8, 62.6, STONE_D))
    roof.append(spire(0, zc, 1.0, 62.6, 64.4, SLATE))
    gold.append(cyl(0.14, 0.04, 1.0, GOLD, seg=6).T(0, 64.2, zc))
    # gables: across the river faces 39–54 m (OSM), across the road faces 47–57 m (OSM)
    for xf in (-TX, TX):
        face = "+x" if xf > 0 else "-x"
        stone.append(gable(face, zc - 2.3, zc + 2.3, 36.0, 54.0, 5.0, STONE, SLATE, xf + (0.3 if xf > 0 else -0.3)))
        roof.append(spire(xf + (0.3 if xf > 0 else -0.3), zc, 0.35, 53.9, 55.8, STONE_D, seg=6))
        # windows of the river faces: three bays of lancets, a tall window in the gable
        s = 1 if xf > 0 else -1
        for zz in (zc - 3.6, zc + 3.6):
            for y in (12.5, 18.5, 30.5):
                dark.append(quad(xf, y + 1.7, zz, 1.1, 3.4, face, WIN, eps=0.05))
        for y in (12.0, 18.0, 26.0):
            dark.append(quad(xf, y + 2.2, zc, 1.6, 4.4, face, WIN, eps=0.05))
        dark.append(quad(xf + 0.3 * s, 44.5, zc, 2.0, 6.5, face, WIN, eps=0.06))
    for zf in (z0, z1):
        face = "+z" if zf > zc else "-z"
        s = 1 if zf > zc else -1
        stone.append(gable(face, -3.0, 3.0, 45.5, 57.0, 6.0, STONE, SLATE, zf + 0.3 * s))
        roof.append(spire(0, zf + 0.3 * s, 0.35, 56.9, 58.8, STONE_D, seg=6))
        dark.append(quad(0, 51.0, zf + 0.3 * s, 2.4, 6.0, face, WIN, eps=0.06))
        for x in (-4.6, 4.6):
            dark.append(quad(x, 29.0, zf, 1.2, 3.6, face, WIN, eps=0.05))
            dark.append(quad(x, 34.4, zf, 1.2, 3.6, face, WIN, eps=0.05))
        for x in (-1.6, 1.6):
            dark.append(quad(x, 31.5, zf, 1.0, 5.0, face, WIN, eps=0.05))
        # small pinnacles on the parapet corners above the road faces
        for x in (-5.5, 5.5):
            stone.append(octagon(x, zf, 0.35, T_TOP + 1.2, T_TOP + 2.2, STONE_D))
            roof.append(spire(x, zf, 0.4, T_TOP + 2.2, T_TOP + 3.6, STONE_D, seg=6))
    return merge(stone), merge(dark), merge(roof), merge(gold)


# ── Walkways ─────────────────────────────────────────────────────────────────
def walkways():
    steel, white, glass = [], [], []
    zA, zB = -Z_T0, Z_T0
    for xc in (-WK_X, WK_X):
        steel.append(box(2 * WK_HW, 0.45, zB - zA, BLUE_D).T(xc, WK_Y0, 0))
        steel.append(box(2 * WK_HW + 0.3, 0.3, zB - zA, WHITE).T(xc, WK_Y1 - 0.3, 0))
        steel.append(box(2 * WK_HW - 0.6, 0.25, zB - zA, GLAZE).T(xc, WK_Y1 - 0.05, 0))
        glass.append(box(2 * WK_HW - 0.6, WK_Y1 - WK_Y0 - 0.7, zB - zA, GLAZE).T(xc, WK_Y0 + 0.4, 0))
        for side in (-1, 1):
            x = xc + side * (WK_HW - 0.15)
            white.append(beam((x, WK_Y0 + 0.3, zA), (x, WK_Y0 + 0.3, zB), 0.32, 0.6, WHITE))
            white.append(beam((x, WK_Y1 - 0.3, zA), (x, WK_Y1 - 0.3, zB), 0.32, 0.5, WHITE))
            n = 20
            zs = np.linspace(zA, zB, n + 1)
            for i in range(n + 1):
                steel.append(beam((x, WK_Y0 + 0.55, zs[i]), (x, WK_Y1 - 0.5, zs[i]), 0.26, 0.3, BLUE))
            for i in range(n):
                y0, y1 = WK_Y0 + 0.6, WK_Y1 - 0.55
                steel.append(beam((x, y0, zs[i]), (x, y1, zs[i + 1]), 0.18, 0.18, BLUE))
                steel.append(beam((x, y1, zs[i]), (x, y0, zs[i + 1]), 0.18, 0.18, BLUE))
            # cantilever brackets under the ends of the walkways
            for s in (-1, 1):
                zt = s * Z_T0
                steel.append(beam((x, WK_Y0 - 4.5, zt), (x, WK_Y0 + 0.1, zt - s * 7.0), 0.4, 0.5, BLUE))
                steel.append(beam((x, WK_Y0 - 2.0, zt), (x, WK_Y0 + 0.1, zt - s * 3.5), 0.3, 0.4, BLUE))
    # cross bracing between the two walkways at the tower faces and at the cantilever ends
    for z in (-Z_T0 + 0.5, -18.3, 18.3, Z_T0 - 0.5):
        steel.append(beam((-WK_X + WK_HW, WK_Y1 - 0.4, z), (WK_X - WK_HW, WK_Y1 - 0.4, z), 0.35, 0.45, BLUE))
        steel.append(beam((-WK_X + WK_HW, WK_Y0 + 0.3, z), (WK_X - WK_HW, WK_Y0 + 0.3, z), 0.35, 0.45, BLUE))
    return merge(steel + white), merge(glass)


# ── Deck: bascules, side spans, roads through the towers, approaches ─────────
def strip(x0, x1, zs, y_off, thick, c, top_only=False):
    """A slab following the road profile between x0 and x1 over the stations zs; top at road_y + y_off."""
    zs = np.asarray(zs, float)
    ys = road_y(zs) + y_off
    v, f = [], []
    n = len(zs)
    for i in range(n):
        v += [(x0, ys[i], zs[i]), (x1, ys[i], zs[i]), (x0, ys[i] - thick, zs[i]), (x1, ys[i] - thick, zs[i])]
    for i in range(n - 1):
        a, b = 4 * i, 4 * (i + 1)
        f += [(a, b, b + 1), (a, b + 1, a + 1)]                      # top
        if not top_only:
            f += [(a + 2, a + 3, b + 3), (a + 2, b + 3, b + 2)]          # bottom
            f += [(a, a + 2, b + 2), (a, b + 2, b)]                      # side x0
            f += [(a + 1, b + 1, b + 3), (a + 1, b + 3, a + 3)]          # side x1
    if not top_only:
        for a, s in ((0, -1), (4 * (n - 1), 1)):
            f += [(a, a + 1, a + 3), (a, a + 3, a + 2)] if s < 0 else [(a, a + 3, a + 1), (a, a + 2, a + 3)]
    m = Mesh(v, f, c)
    return m


def stations(z0, z1, step=3.0):
    knots = [p[0] for p in ROAD if z0 < p[0] < z1]
    zs = sorted(set([z0, z1] + knots + list(np.arange(z0, z1, step))))
    return zs


def deck():
    road, steel, white, dark = [], [], [], []
    segs = [(Z_S, -Z_PIER, W_SIDE), (-Z_PIER, Z_PIER, W_MAIN), (Z_PIER, Z_N, W_SIDE)]
    for z0, z1, hw in segs:
        zs = stations(z0, z1)
        walk = 2.4 if hw > 8 else 2.0
        road.append(strip(-hw + walk, hw - walk, zs, 0.0, 0.45, ASPH))
        road.append(strip(-hw, -hw + walk, zs, 0.18, 0.6, WALK))
        road.append(strip(hw - walk, hw, zs, 0.18, 0.6, WALK))
        # centre line (dashes)
        for z in np.arange(z0 + 1.5, z1 - 1.5, 6.0):
            zz = [z, z + 3.0]
            road.append(strip(-0.08, 0.08, zz, 0.02, 0.01, LINE, top_only=True))
    # the joint between the two bascules
    road.append(box(2 * W_MAIN, 0.03, 0.12, (0.15, 0.15, 0.16)).T(0, road_y(0) + 0.0, 0))
    # bascule girders: four per leaf, 1.1 m deep at the tip, 4.2 m at the pier face
    for s in (-1, 1):
        for x in (-6.6, -2.2, 2.2, 6.6):
            prof = []
            n = 12
            for i in range(n + 1):
                z = s * Z_PIER * i / n
                d = 1.1 + 3.1 * (i / n) ** 1.6
                prof.append((z, road_y(z) - 0.45 - d))
            top = [(s * Z_PIER, road_y(0) - 0.45), (0.0, road_y(0) - 0.45)]
            ring = prof + top
            steel.append(prism_x(ring, x - 0.35, x + 0.35, BLUE))
        # cross girders and the underside plate
        for i in range(1, 6):
            z = s * Z_PIER * i / 6
            d = 1.1 + 3.1 * (i / 6) ** 1.6
            steel.append(box(13.6, 0.6, 0.4, BLUE_D).T(0, road_y(z) - 0.45 - d * 0.7, z))
        steel.append(box(2 * W_MAIN, 0.25, Z_PIER, BLUE_D).T(0, road_y(0) - 0.7, s * Z_PIER / 2))
    # side-span stiffening girders (plate girders at the deck edges) and floor beams
    for s in (-1, 1):
        za, zb = s * Z_PB, s * Z_A0
        zs = stations(min(za, zb), max(za, zb), 2.0)
        for x in (-W_SIDE - 0.2, W_SIDE + 0.2):
            steel.append(strip(x - 0.3, x + 0.3, zs, 1.25, 2.9, BLUE))
            white.append(strip(x - 0.36, x + 0.36, zs, 1.35, 0.12, WHITE))
        steel.append(strip(-W_SIDE, W_SIDE, zs, -0.45, 0.25, BLUE_D))
        for z in np.arange(min(za, zb) + 2.75, max(za, zb), 5.5):
            steel.append(box(2 * W_SIDE, 0.9, 0.35, BLUE_D).T(0, road_y(z) - 1.6, z))
    # railings over the opening span: posts and a white top rail
    for s in (-1, 1):
        for x in (-W_MAIN - 0.1, W_MAIN + 0.1):
            zz = np.array([0.15 * s, s * Z_T0])
            y = road_y(0) + 0.18
            white.append(beam((x, y + 1.15, zz[0]), (x, y + 1.15, zz[1]), 0.18, 0.14, WHITE))
            steel.append(beam((x, y + 0.6, zz[0]), (x, y + 0.6, zz[1]), 0.08, 0.08, BLUE))
            for z in np.arange(min(zz), max(zz) + 0.01, 1.8):
                steel.append(beam((x, y, z), (x, y + 1.1, z), 0.1, 0.1, BLUE))
    return merge(road), merge(steel), merge(white), merge(dark)


# ── Suspension chains of the side spans ──────────────────────────────────────
def chain_y(u):
    """Upper chord over the side span, u = distance from the main tower's outer face (0) to the abutment (88)."""
    zv, yv, a = 58.1, 14.5, 0.00727
    return yv + a * (u - zv) ** 2


def chains():
    blue, white = [], []
    D = 2.0
    span = Z_A0 - Z_T1
    for s in (-1, 1):
        us = np.linspace(0.0, span, 60)
        for x in (-CH_X, CH_X):
            up = [(x, chain_y(u), s * (Z_T1 + u)) for u in us]
            lo = [(x, chain_y(u) - D, s * (Z_T1 + u)) for u in us]
            white.append(sweep(up, 0.55, 0.55, WHITE))
            white.append(sweep(lo, 0.55, 0.55, WHITE))
            # web: verticals every 5.5 m and diagonals
            uw = np.arange(0.0, span + 0.01, span / 16)
            for i, u in enumerate(uw):
                z = s * (Z_T1 + u)
                blue.append(beam((x, chain_y(u) - D + 0.25, z), (x, chain_y(u) - 0.25, z), 0.32, 0.32, BLUE))
                if i + 1 < len(uw):
                    u2 = uw[i + 1]
                    z2 = s * (Z_T1 + u2)
                    if i % 2 == 0:
                        blue.append(beam((x, chain_y(u) - D + 0.25, z), (x, chain_y(u2) - 0.25, z2), 0.22, 0.22, BLUE))
                    else:
                        blue.append(beam((x, chain_y(u) - 0.25, z), (x, chain_y(u2) - D + 0.25, z2), 0.22, 0.22, BLUE))
                # hangers down to the deck edge
                if 0 < i < len(uw) - 1:
                    yd = float(road_y(z)) + 1.2
                    blue.append(beam((x, chain_y(u) - D, z), (x, yd, z), 0.22, 0.22, BLUE))
            # backstay: over the abutment tower and down into the approach
            zb0, zb1 = s * Z_A1, s * (Z_A1 + 13.5)
            yr = float(road_y(zb1))
            white.append(sweep([(x, 21.0, zb0), (x, yr + 1.0, zb1)], 0.55, 0.55, WHITE))
            white.append(sweep([(x, 19.0, zb0), (x, yr + 0.4, s * (Z_A1 + 11.0))], 0.55, 0.55, WHITE))
            for k in range(1, 4):
                t = k / 4
                p = (x, 21.0 + (yr + 1.0 - 21.0) * t, zb0 + (zb1 - zb0) * t)
                q = (x, 19.0 + (yr + 0.4 - 19.0) * t, zb0 + (s * (Z_A1 + 11.0) - zb0) * t)
                blue.append(beam(p, q, 0.25, 0.25, BLUE))
    return merge(blue), merge(white)


# ── Abutment towers and approaches ───────────────────────────────────────────
def abutment(sgn):
    z0, z1 = sorted((sgn * Z_A0, sgn * Z_A1))
    zc = (z0 + z1) / 2
    yr = float(road_y(zc))
    stone, dark, roof = [], [], []
    AX = 11.0
    body = [(-AX, BANK_LOW), (AX, BANK_LOW), (AX, 22.0), (-AX, 22.0)]
    hole = arch_ring(12.4, yr - 0.2, yr + 4.6, yr + 8.6)
    stone.append(prism_z(body, z0, z1, STONE, holes=[hole]))
    stone.append(ring_band(-AX, AX, z0, z1, BANK_LOW, yr + 0.3, GRANITE, out=0.15))
    stone.append(ring_band(-AX, AX, z0, z1, 21.4, 22.6, STONE_D, out=0.25))
    for x in (-AX, AX):
        for z in (z0, z1):
            stone.append(octagon(x, z, 1.25, BANK_LOW, 24.5, STONE))
            roof.append(spire(x, z, 1.4, 24.5, 29.0, SLATE))
    roof.append(hip_roof(-AX + 0.5, AX - 0.5, z0 + 0.4, z1 - 0.4, 22.6, 30.0, SLATE))
    for zf in (z0, z1):
        face = "+z" if zf > zc else "-z"
        s = 1 if zf > zc else -1
        stone.append(gable(face, -2.0, 2.0, 21.0, 26.5, 3.0, STONE, SLATE, zf + 0.25 * s))
        dark.append(quad(0, 23.4, zf + 0.25 * s, 1.4, 3.2, face, WIN, eps=0.06))
        for x in (-6.0, 6.0):
            dark.append(quad(x, yr + 11.0, zf, 1.1, 3.2, face, WIN, eps=0.05))
        for x in (-2.5, 2.5):
            dark.append(quad(x, yr + 12.5, zf, 0.9, 2.4, face, WIN, eps=0.05))
    for xf in (-AX, AX):
        face = "+x" if xf > 0 else "-x"
        for y in (yr + 4.0, yr + 10.0):
            dark.append(quad(xf, y, zc, 1.1, 3.0, face, WIN, eps=0.05))
    # the approach to the end of the bridge: masonry with parapets
    za, zb = sorted((sgn * Z_A1, sgn * (Z_N if sgn > 0 else -Z_S)))
    zs = stations(za, zb, 2.0)
    yr0 = road_y(np.asarray(zs))
    for x0, x1 in ((-W_SIDE - 0.6, -W_SIDE + 0.2), (W_SIDE - 0.2, W_SIDE + 0.6)):
        stone.append(strip(x0, x1, zs, 1.1, 1.1 - BANK_LOW + float(yr0.max()) - float(yr0.min()) + 0.5, STONE_D))
    stone.append(strip(-W_SIDE + 0.2, W_SIDE - 0.2, zs, -0.4, float(yr0.max()) - BANK_LOW, GRANITE_D))
    # the side span's end: a granite bank seat under the deck in front of the abutment tower
    zf = sgn * Z_A0
    stone.append(box(2 * AX, float(road_y(zf)) - 1.6 - BANK_LOW, 3.0, GRANITE).T(0, BANK_LOW, zf - sgn * 1.5))
    return merge(stone), merge(dark), merge(roof)


def side_span_parapets():
    """Parapet railings along the side spans (between the main towers and the abutment towers)."""
    steel, white = [], []
    for s in (-1, 1):
        za, zb = s * Z_T1, s * Z_A0
        for x in (-W_SIDE - 0.05, W_SIDE + 0.05):
            n = 24
            zs = np.linspace(za, zb, n + 1)
            pts = [(x, float(road_y(z)) + 1.35, z) for z in zs]
            white.append(sweep(pts, 0.16, 0.14, WHITE))
            for z in np.arange(min(za, zb), max(za, zb), 2.0):
                y = float(road_y(z)) + 0.18
                steel.append(beam((x, y, z), (x, y + 1.15, z), 0.1, 0.1, BLUE))
    return merge(steel), merge(white)


def lamps():
    """Lamp standards along the parapets of the side spans and the approaches (every ~11 m, estimated)."""
    posts, heads = [], []
    for s in (-1, 1):
        for z in np.arange(Z_T1 + 8.0, Z_A0 - 2.0, 11.0):
            for x in (-W_SIDE + 0.3, W_SIDE - 0.3):
                y = float(road_y(s * z)) + 0.18
                posts.append(cyl(0.12, 0.09, 4.2, (0.20, 0.30, 0.36), seg=6).T(x, y, s * z))
                heads.append(cyl(0.28, 0.18, 0.7, (0.95, 0.88, 0.62), seg=6).T(x, y + 4.2, s * z))
                heads.append(cyl(0.32, 0.0, 0.35, (0.20, 0.30, 0.36), seg=6).T(x, y + 4.9, s * z))
    return merge(posts + heads)


def footprint():
    """A strip along the bridge, wider round the piers; the approaches to the two ends."""
    hw, pw = 13.0, 24.0
    right = [(hw, Z_S - 0.5), (hw, -Z_PB - 1.0), (pw, -Z_PB - 1.0), (pw, -Z_PIER + 0.5), (hw, -Z_PIER + 0.5),
             (hw, Z_PIER - 0.5), (pw, Z_PIER - 0.5), (pw, Z_PB + 1.0), (hw, Z_PB + 1.0), (hw, Z_N + 0.5)]
    left = [(-x, z) for x, z in reversed(right)]
    return [[round(x, 1), round(z, 1)] for x, z in right + left]


def main():
    out = ROOT / "models" / "Q83125"
    out.mkdir(parents=True, exist_ok=True)
    stone, dark, roof, gold = [], [], [], []
    for s in (-1, 1):
        stone.append(pier(s))
        a, b, c, d = main_tower(s)
        stone.append(a); dark.append(b); roof.append(c); gold.append(d)
        a, b, c = abutment(s)
        stone.append(a); dark.append(b); roof.append(c)
    road, d_steel, d_white, d_dark = deck()
    w_steel, w_glass = walkways()
    c_blue, c_white = chains()
    p_steel, p_white = side_span_parapets()
    steel = merge([d_steel, d_white, w_steel, c_blue, c_white, p_steel, p_white, lamps()])
    masonry = merge(stone + roof + [road])
    glass = merge(dark + [d_dark, w_glass])
    gilt = merge(gold)
    parts = [("steelwork", steel, {"material": "steel"}), ("stone_and_deck", masonry),
             ("windows", glass, {"material": "glass"}), ("finials", gilt, {"material": "gilt"})]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.85,
                           materials={"steel": {"metallic": 0.25, "roughness": 0.55},
                                      "glass": {"metallic": 0.35, "roughness": 0.18},
                                      "gilt": {"metallic": 0.7, "roughness": 0.4}})
    allv = np.vstack([p[1].v for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.1f}..{hi[0]:.1f}, "
          f"y {lo[1]:.1f}..{hi[1]:.1f}, z {lo[2]:.1f}..{hi[2]:.1f}")
    fp = footprint()
    meta = out / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        m["footprint"] = fp
        m["triangles"] = tris
        meta.write_bytes((json.dumps(m, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print("meta.json: footprint", len(fp), "points, triangles", tris)


if __name__ == "__main__":
    main()
