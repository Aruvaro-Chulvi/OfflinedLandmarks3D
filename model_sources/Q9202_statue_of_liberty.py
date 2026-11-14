"""
Statue of Liberty (Q9202), Liberty Island, New York — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: "Liberty Enlightening the World", copper statue by Frédéric Auguste Bartholdi (1834–1904) on an
iron frame by Gustave Eiffel (1832–1923) after Eugène Viollet-le-Duc (1814–1879); granite pedestal by Richard
Morris Hunt (1827–1895) on a concrete foundation by Charles P. Stone, inside the 11-pointed star of Fort Wood
(1808–1811). Dedicated 28 October 1886. Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 32965412 (the monument, Q9202): the star of Fort Wood (33 points, 96 × 94 m) and its
  building:parts — squares of the lower terrace (39.7 × 40.1 m), the upper terrace (25.2 × 25.5 m), the
  pedestal base (17.9 × 18.9 m), shaft (15.0 × 15.8 m) and balcony (16.4 × 17.3 m). All the squares share one
  centre and one orientation: their faces look 146.9° / 56.9° / 326.9° / 236.9°. The statue faces the
  harbour entrance (south-east): heading 146.9°. The ferry pier and the fort's entrance are behind her (NW).
- HAER NY-138 (Historic American Engineering Record, 2011), sheet 15 (the four elevations at 1:384) and
  sheets 20–21 (south fort and pedestal elevations at 1:128). Levels above the entrance ground level: top of
  the fort 8.69 m, terrace walls 13.17 and 16.21 m, top of the pedestal 41.83 m, bottom of the statue 43.49 m,
  top of the head 77.31 m, bottom of the torch 79.34 m, top of the flame 88.42 m. Every step of the pedestal
  (cornices, disc frieze, loggia, balcony, attic, plinth) measured on sheet 21; the four loggia columns, ten
  discs, eight little gables and the corner posts of the balcony counted there.
- The statue: horizontal extents of the figure measured every 0.6 m on the front (south) and side elevations
  of sheet 15 (silhouettes traced at 10.95 px/m): they give the body sections below (x across, z front/back,
  both from the centre of the pedestal). Arm, head, crown, tablet and torch placed on the same drawings, with
  the NPS/Wikipedia figures: head 5.26 m chin to cranium, 3.05 m ear to ear, eye 0.76 m, nose 1.48 m, mouth
  0.91 m; right arm 12.8 m, 3.66 m thick; tablet 7.19 × 4.14 × 0.61 m; crown with seven rays (~2.7 m) and 25
  windows. Colours: verdigris copper, gilded flame, Stony Creek granite pedestal, grey concrete terraces and
  granite fort walls, from photos (Wikimedia Commons).
The statue is stylised: a robe lofted from the measured sections, with the folds of the chiton below the
diagonal hem of the mantle and the diagonal folds of the mantle as relief; the face reduced to brow, eyes,
nose, mouth and chin.

Axes (docs/MODEL_SPEC.md): +Y up, the statue faces +Z (heading 146.9°), +X is her left. Origin on the ground
(the entrance level of the fort) at the centre of the pedestal.

    python model_sources/Q9202_statue_of_liberty.py   -> models/Q9202/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, extrude, offset, write_glb, fbm, sstep  # noqa: E402

COPPER = np.array((0.42, 0.64, 0.56))      # verdigris
COPPER_D = np.array((0.30, 0.48, 0.43))
GOLD = (0.86, 0.66, 0.22)                 # gilded flame
PLINTH = (0.52, 0.62, 0.56)               # the copper-stained top of the pedestal
GRANITE = (0.70, 0.64, 0.56)              # Stony Creek granite, pinkish
GRANITE_L = (0.75, 0.69, 0.61)
GRANITE_D = (0.58, 0.53, 0.47)
CONCRETE = (0.76, 0.73, 0.67)             # the terraces
PAVE = (0.70, 0.68, 0.64)
FORT = (0.64, 0.62, 0.58)                 # Fort Wood granite walls
FORT_L = (0.74, 0.72, 0.67)
DARK = (0.17, 0.16, 0.15)
FOUND = -3.0

# Levels (HAER NY-138, metres above the entrance ground level)
Y_FORT, Y_TER1, Y_TER2 = 8.69, 13.17, 16.21
Y_ST = 43.49                              # bottom of the statue (top of the plinth)
H_TER1, H_TER2 = 20.2, 13.0               # half-widths of the two terraces

# Fort Wood, OSM way 32965412 in model x/z (heading 146.9, origin 40.689253, -74.0445321)
STAR = [(-35.33, -18.85), (-35.09, -35.14), (-18.34, -34.23), (-13.99, -44.41), (-13.23, -46.09), (-12.83, -46.98),
        (-12.58, -47.53), (-4.33, -39.27), (-1.19, -39.38), (1.5, -39.41), (9.36, -48.05), (9.78, -47.06),
        (16.18, -34.64), (32.93, -34.35), (32.73, -29.59), (32.63, -21.84), (32.57, -18.07), (45.32, -12.51),
        (34.67, -0.98), (46.62, 9.96), (32.95, 17.25), (33.25, 36.97), (19.38, 34.08), (18.16, 33.79), (17.52, 33.64),
        (16.0, 35.18), (-0.6, 52.2), (-19.15, 33.3), (-35.49, 36.27), (-35.51, 16.94), (-49.18, 10.19),
        (-37.78, -0.65), (-49.21, -12.77)]

# Body sections of the statue (HAER sheet 15): y above the bottom of the statue, x left/right (−x is her
# right), z back/front, metres from the centre of the pedestal. Where the tablet and the raised sleeve hide
# the body, the edge is interpolated.
BODY = [
    (-0.2, -5.70, 5.70, -4.70, 5.00), (0.8, -5.71, 5.71, -4.70, 4.98), (1.4, -5.25, 5.71, -4.75, 4.88),
    (2.0, -4.88, 5.61, -4.75, 4.79), (2.6, -4.61, 5.61, -4.75, 4.79), (3.2, -4.52, 5.52, -4.70, 4.79),
    (3.8, -4.52, 5.52, -4.60, 4.79), (4.4, -4.43, 5.52, -4.45, 4.79), (5.0, -4.34, 5.89, -4.06, 5.16),
    (5.6, -4.34, 5.89, -3.97, 5.25), (6.2, -4.52, 5.80, -3.79, 5.25), (6.8, -4.52, 5.80, -3.61, 5.34),
    (7.4, -4.43, 5.80, -3.42, 5.52), (8.0, -4.25, 5.89, -3.24, 6.07), (8.6, -4.06, 6.07, -3.15, 6.07),
    (9.2, -4.15, 5.98, -3.06, 6.07), (9.8, -4.06, 6.30, -3.33, 6.07), (10.4, -3.79, 6.25, -3.24, 6.07),
    (11.0, -3.70, 6.16, -3.15, 6.07), (11.6, -3.88, 6.16, -3.15, 6.07), (12.2, -3.97, 6.16, -3.79, 5.98),
    (12.8, -3.97, 6.07, -3.79, 5.98), (13.4, -4.06, 6.07, -3.70, 5.98), (14.0, -4.25, 5.98, -3.70, 5.98),
    (14.6, -4.34, 5.89, -3.61, 5.98), (15.2, -4.52, 5.80, -3.61, 5.98), (15.8, -4.34, 5.75, -3.51, 5.98),
    (16.4, -4.06, 5.70, -3.51, 5.98), (17.0, -3.79, 5.65, -3.51, 5.98), (17.6, -3.79, 5.60, -3.61, 5.98),
    (18.2, -4.25, 5.60, -3.70, 5.98), (18.8, -4.88, 5.60, -3.79, 5.89), (19.4, -4.88, 5.55, -3.97, 5.80),
    (20.0, -4.70, 5.50, -3.97, 5.71), (20.6, -4.43, 5.45, -4.06, 5.71), (21.2, -4.06, 5.40, -4.15, 5.61),
    (21.8, -3.80, 5.35, -4.15, 5.61), (22.4, -3.85, 5.30, -4.15, 5.61), (23.0, -3.97, 5.20, -3.97, 5.52),
    (23.6, -4.34, 5.10, -3.61, 5.34), (24.2, -4.70, 5.00, -3.33, 5.16), (24.8, -4.61, 4.79, -3.15, 4.98),
    (25.4, -4.65, 4.70, -3.06, 4.79), (26.0, -4.52, 4.52, -2.97, 4.55), (26.6, -4.45, 4.25, -2.60, 4.10),
    (27.2, -4.05, 3.88, -2.33, 3.85), (27.8, -3.30, 3.15, -1.87, 3.50), (28.3, -2.20, 2.40, -1.10, 2.90),
    (28.7, -1.00, 1.70, -0.40, 2.30),
]

# Head (HAER sheet 15): chin 28.6, top of the head 33.8 above the bottom of the statue
HX, HY, HZ = 0.45, 31.2, 1.2
# Right arm: shoulder, elbow, wrist (front view x, side view z)
ARM = [(-3.1, 26.6, 1.5, 1.30), (-3.7, 28.8, 2.6, 1.22), (-4.0, 31.3, 3.65, 1.08), (-4.15, 33.6, 4.35, 1.0),
       (-4.3, 35.4, 4.85, 0.9)]
TX, TZ = -4.75, 6.42                      # axis of the torch


def jit(c, s=0.015, rng=np.random.default_rng(1886)):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, s)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sq(h, y0, y1, c):
    """A square block, half-width h, centred on the axis."""
    return box(2 * h, y1 - y0, 2 * h, c).T(0, y0, 0)


def frustum(h0, h1, y0, y1, c):
    """A square frustum (sloped cornice), each side its own quad so the edges stay sharp."""
    v, f = [], []
    for k in range(4):
        a = k * math.pi / 2
        ca, sa = math.cos(a), math.sin(a)

        def P(h, y, s):                       # point on side k: outward (sa, ca) rotated, s along the side
            return (h * sa + s * h * ca, y, h * ca - s * h * sa)
        o = len(v)
        v += [P(h0, y0, -1), P(h0, y0, 1), P(h1, y1, 1), P(h1, y1, -1)]
        f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    m = Mesh(v, f, c)
    m.f = _outward(m, np.array([0.0, (y0 + y1) / 2, 0.0]))
    return merge([m, sq(h1, y1 - 0.001, y1, c)]) if h1 > 0 else m


def _outward(m, centre):
    """Flip triangles whose normals point towards centre (for hand-built pieces)."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    n = np.cross(b - a, c - a)
    d = ((a + b + c) / 3 - centre)
    bad = np.einsum("ij,ij->i", n, d) < 0
    f = m.f.copy()
    f[bad] = f[bad][:, ::-1]
    return f


def rot_y(m, k):
    return m.copy().R(0, k * math.pi / 2, 0)


def four(ms):
    """The same feature on the four faces of the pedestal (built on the +Z face)."""
    m = merge(ms)
    return merge([rot_y(m, k) for k in range(4)])


def sphere(r, c, seg=16, rings=10):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def ell(rx, ry, rz, c, seg=16, rings=10):
    return sphere(1.0, c, seg, rings).S(rx, ry, rz)


def _align(m, p0, p1):
    """Rotate a piece built along +y (from 0 to |p1 − p0|) onto the segment p0 → p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    y = np.array([0.0, 1.0, 0.0]); t = d / L
    v = np.cross(y, t); s = np.linalg.norm(v); cth = float(np.dot(y, t))
    if s > 1e-9:
        k = v / s; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rm = np.eye(3) + math.sin(math.atan2(s, cth)) * K + (1 - cth) * K @ K
        m.v = m.v @ Rm.T
    elif cth < 0:
        m.v[:, 1] *= -1; m.f = m.f[:, ::-1]
    return m.T(*p0)


def limb(p0, p1, r0, r1, c, seg=10):
    L = float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float)))
    return _align(cyl(r0, r1, L, c, seg), p0, p1)


def sweep(pts, radii, c, seg=16, caps=True, folds=(0, 0.0)):
    """A tube of varying radius along a polyline (arms, locks, drapes), normals facing out; folds = (n, amp)
    ripples the section into n folds."""
    P = np.asarray(pts, float)
    n = len(P)
    T = np.array([P[min(i + 1, n - 1)] - P[max(i - 1, 0)] for i in range(n)])
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    a = np.cross(T[0], [0, 0, 1] if abs(T[0][2]) < 0.9 else [1, 0, 0]); a /= np.linalg.norm(a)
    v = []
    for i in range(n):
        if i:
            a = a - T[i] * np.dot(a, T[i]); a /= np.linalg.norm(a)          # parallel transport
        b = np.cross(T[i], a)
        for k in range(seg):
            th = 2 * math.pi * k / seg
            r = radii[i] * (1 + folds[1] * math.sin(folds[0] * th + 0.7 * i))
            v.append(P[i] + r * (math.cos(th) * a + math.sin(th) * b))
    f = []
    for i in range(n - 1):
        for k in range(seg):
            j = (k + 1) % seg
            p, q = i * seg + k, i * seg + j
            f += [(p, q, q + seg), (p, q + seg, p + seg)]
    V = np.array(v)
    if caps:
        for i, end in ((0, False), (n - 1, True)):
            o = len(V); V = np.vstack([V, P[i]])
            for k in range(seg):
                j = (k + 1) % seg
                p, q = i * seg + k, i * seg + j
                f.append((o, q, p) if end else (o, p, q))
    m = Mesh(V, f, c)
    # make every face look away from the polyline
    a_, b_, c_ = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    nrm = np.cross(b_ - a_, c_ - a_)
    cen = (a_ + b_ + c_) / 3
    near = np.array([P[np.argmin(np.linalg.norm(P - q, axis=1))] for q in cen])
    bad = np.einsum("ij,ij->i", nrm, cen - near) < 0
    if bad.mean() > 0.5:
        m.f = m.f[:, ::-1]
    return m


def prism(pts2, depth, c):
    """A flat piece from an outline in the x/y plane (gables, pediments), from z = 0 to z = depth."""
    P = np.asarray(pts2, float)
    n = len(P)
    v, f = [], []
    for z in (0.0, depth):
        o = len(v); v += [(x, y, z) for x, y in P]
        for i in range(1, n - 1):
            f.append((o, o + i, o + i + 1))
    for i in range(n):
        j = (i + 1) % n
        o = len(v)
        v += [(P[i][0], P[i][1], 0), (P[j][0], P[j][1], 0), (P[j][0], P[j][1], depth), (P[i][0], P[i][1], depth)]
        f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    m = Mesh(v, f, c)
    cen = np.array([P[:, 0].mean(), P[:, 1].mean(), depth / 2])
    m.f = _outward(m, cen)
    return m


# ── Fort Wood and the terraces ──────────────────────────────────────────────
def fort():
    out = [extrude(STAR, FOUND, Y_FORT - 0.38, jit(FORT), top=False)]
    out.append(extrude(offset(STAR, 0.14), Y_FORT - 0.38, Y_FORT, jit(FORT_L), top_c=PAVE, bottom=True))      # coping + paving
    # lower terrace with its vent grilles, upper terrace
    out.append(sq(H_TER1, Y_FORT - 0.2, Y_TER1 - 0.23, jit(CONCRETE)))
    out.append(sq(H_TER1 + 0.12, Y_TER1 - 0.25, Y_TER1, jit(CONCRETE, 0.01)))
    g = []
    for s in (-1, 1):
        for x in (14.6, 16.2):
            g.append(B(s * x - 0.55, s * x + 0.55, 9.05, 10.65, H_TER1 - 0.05, H_TER1 + 0.05, DARK))
    out.append(four(g))
    out.append(sq(H_TER2, Y_TER1 - 0.2, Y_TER2 - 0.23, jit(CONCRETE)))
    out.append(sq(H_TER2 + 0.1, Y_TER2 - 0.25, Y_TER2, jit(CONCRETE, 0.01)))
    out.append(B(-2.3, -0.1, 0.0, 4.2, -39.5, -39.3, DARK))           # the sally port, in the cut-off back point
    return out


# ── The pedestal (Richard Morris Hunt) ──────────────────────────────────────
def pedestal():
    out = []
    G, GL, GD = GRANITE, GRANITE_L, GRANITE_D
    # the stack of square blocks (half-width, bottom, top), HAER sheet 21
    for h, y0, y1, c in ((9.75, Y_TER2 - 0.2, 19.30, G), (9.00, 19.30, 19.84, GL), (8.70, 19.84, 21.77, G),
                         (8.95, 21.77, 22.30, GL), (8.35, 22.85, 23.92, G), (8.10, 23.92, 24.78, GL),
                         (7.85, 24.78, 25.96, G), (7.50, 25.96, 29.08, G), (7.72, 29.08, 29.40, GL),
                         (7.50, 29.40, 30.50, G), (6.40, 30.40, 34.44, DARK), (7.60, 34.40, 35.20, G),
                         (7.92, 35.20, 35.96, GL), (8.22, 35.96, 36.93, G), (8.17, 36.93, 37.20, G),
                         (6.75, 37.15, 39.73, G), (6.40, 39.73, 41.12, G), (6.62, 41.12, 41.60, GL)):
        out.append(sq(h, y0 - (0.03 if y0 > 17 else 0), y1, jit(c, 0.01)))
    out.append(frustum(8.95, 8.40, 22.30, 22.85, jit(GL)))
    out.append(frustum(6.62, 6.20, 41.60, 42.09, jit(GL)))
    out.append(sq(6.08, 42.05, Y_ST, PLINTH))                               # the plinth under her feet
    # rusticated corner piers of the shaft (25.96–34.45): courses with sunk joints
    piers = []
    for sx in (-1, 1):
        for sz in (-1, 1):
            piers.append(B(*sorted((sx * 4.88, sx * 7.55)), 25.91, 34.46, *sorted((sz * 4.88, sz * 7.55)), G))
            y = 26.0
            while y < 34.3:
                y1 = min(y + 0.56, 34.42)
                piers.append(B(*sorted((sx * 4.80, sx * 7.63)), y, y1, *sorted((sz * 4.80, sz * 7.63)), jit(G, 0.03)))
                y = y1 + 0.1
    out += piers
    # loggia: four columns on each face, with bases and capitals
    cols = []
    for x in (-4.34, -1.55, 1.40, 4.25):
        cols.append(cyl(0.5, 0.45, 3.5, jit(GL), 14).T(x, 30.5, 6.95))
        cols.append(B(x - 0.65, x + 0.65, 30.45, 30.75, 6.35, 7.53, GL))
        cols.append(B(x - 0.68, x + 0.68, 33.95, 34.41, 6.32, 7.58, GL))
    # panel under the loggia, the eight little gables, the ten discs, the entrance, the attic window
    cols.append(B(-3.65, 3.65, 26.55, 28.45, 7.5, 7.56, GD))
    for x in (-7.2, -5.15, -3.1, -1.05, 1.05, 3.1, 5.15, 7.2):
        cols.append(prism([(x - 0.55, 22.95), (x + 0.55, 22.95), (x, 23.55)], 0.32, jit(GL)).T(0, 0, 8.3))
    for i in range(10):
        x = (i - 4.5) * 1.72
        cols.append(cyl(0.66, 0.62, 0.16, jit(GL), 18).R(math.pi / 2, 0, 0).T(x, 20.8, 8.68))
    cols.append(B(-2.15, 2.15, Y_TER2 - 0.1, 18.55, 9.6, 10.05, GL))                    # entrance aedicule
    cols.append(prism([(-2.35, 18.55), (2.35, 18.55), (0, 19.45)], 0.5, GL).T(0, 0, 9.6))
    cols.append(B(-0.6, 0.6, Y_TER2, 18.0, 10.03, 10.09, DARK))
    for x in (-1.35, 1.35):
        cols.append(cyl(0.38, 0.38, 0.04, DARK, 12).R(math.pi / 2, 0, 0).T(x, 17.4, 10.09))
    cols.append(B(-1.2, 1.2, 37.15, 39.9, 6.6, 6.95, GL))                                # attic window
    cols.append(prism([(-1.35, 39.9), (1.35, 39.9), (0, 40.5)], 0.42, GL).T(0, 0, 6.55))
    cols.append(B(-0.55, 0.55, 37.25, 39.4, 6.93, 6.99, DARK))
    out.append(four(cols))
    # the balcony: parapet with round-topped posts, its floor
    par = []
    par.append(B(-7.77, 8.17, 36.90, 38.22, 7.77, 8.17, jit(G)))          # pinwheel: the four sides do not overlap
    for x in (-7.92, -4.72, -1.6, 1.55, 4.69):
        par.append(B(x - 0.3, x + 0.3, 36.93, 38.3, 7.74, 8.29, GL))
        par.append(cyl(0.3, 0.3, 0.55, GL, 10).R(math.pi / 2, 0, 0).T(x, 38.3, 7.74))
    out.append(four(par))
    return out


# ── The statue ───────────────────────────────────────────────────────────────
def _hem(x, front):
    """Height of the lower edge of the mantle: a diagonal from her right hip down to a point near her left
    knee in front (HAER sheet 15), long at the back."""
    hf = np.maximum(15.0 - (x + 4.3) * 1.686, 3.2 + (x - 2.7) * 6.0)
    w = sstep(-0.35, 0.35, front)
    return w * np.minimum(hf, 30.0) + (1 - w) * 4.6


def body(N=112, dy=0.3, p=2.25):
    T = np.array(BODY)
    ys = np.arange(T[0, 0], T[-1, 0] + 1e-6, dy)
    cols = [np.interp(ys, T[:, 0], T[:, k]) for k in range(1, 5)]
    k3 = np.array([0.25, 0.5, 0.25])
    cols = [np.concatenate([c[:1], np.convolve(c, k3, "valid"), c[-1:]]) for c in cols]
    xl, xr, zb, zf = cols
    t = np.linspace(0, 2 * math.pi, N, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    S = lambda u: np.sign(u) * np.abs(u) ** (2 / p)
    rings, cols_v = [], []
    for i, y in enumerate(ys):
        cx, ax = (xl[i] + xr[i]) / 2, (xr[i] - xl[i]) / 2
        cz, az = (zb[i] + zf[i]) / 2, (zf[i] - zb[i]) / 2
        X = cx + ax * S(ct); Z = cz + az * S(st)
        # outward normal of the ring
        tx = np.roll(X, -1) - np.roll(X, 1); tz = np.roll(Z, -1) - np.roll(Z, 1)
        nx, nz = tz, -tx
        ln = np.hypot(nx, nz) + 1e-9; nx /= ln; nz /= ln
        flip = (nx * (X - cx) + nz * (Z - cz)) < 0
        nx[flip] *= -1; nz[flip] *= -1
        front = st                                    # +1 in front, −1 at the back
        h = _hem(X, front)
        mant = sstep(h - 0.05, h + 0.25, y)           # 1 on the mantle, 0 on the chiton below it
        taper = sstep(28.4, 26.0, y)                  # no folds on the neck opening
        # chiton: deep vertical folds, a little wavy
        nz_ = fbm(np.stack([X, np.full(N, y), Z], 1), 0.35, 11, 2)
        ch = (np.abs(np.sin(t * 11.5 + 0.6 * np.sin(y * 0.55 + t * 3) + 2.5 * nz_)) - 0.64) * 2             * (0.09 + 0.09 * sstep(14, 2, y)) - 0.1
        # mantle: diagonal folds in front (from her left shoulder down to her right hip), vertical at the back
        ph = 2 * math.pi * (y - 0.95 * (X - cx)) / 7.0 + 2.0 * nz_
        md = (np.abs(np.sin(ph)) - 0.64) * 2 * 0.13 * sstep(-0.6, 0.2, front)
        mb = (np.abs(np.sin(t * 7.5 + y * 0.12 + 2 * nz_)) - 0.64) * 2 * 0.09 * sstep(0.2, -0.6, front)
        fold = (1 - mant) * ch + mant * (md + mb)
        fold = fold * taper
        X = X + nx * fold; Z = Z + nz * fold
        rings.append(np.stack([X, np.full(N, y), Z], 1))
        # colour: darker in the valleys and along the hem
        val = np.clip(0.5 + (fold + 0.1 * (1 - mant)) * 2.6, 0, 1)
        lip = sstep(h - 0.45, h - 0.05, y) * (1 - sstep(h - 0.05, h + 0.05, y))
        k = (0.80 + 0.30 * val) * (1 - 0.25 * lip)
        cols_v.append(np.clip(np.outer(k, COPPER), 0, 1))
    V = np.vstack(rings)
    C = np.vstack(cols_v)
    f = []
    for i in range(len(rings) - 1):
        for j in range(N):
            k = (j + 1) % N
            a, b = i * N + j, i * N + k
            f += [(a, a + N, b + N), (a, b + N, b)]
    # caps: bottom (inside the plinth) and the neck opening at the top
    for i, up in ((0, False), (len(rings) - 1, True)):
        o = len(V); V = np.vstack([V, rings[i].mean(0)]); C = np.vstack([C, COPPER])
        for j in range(N):
            k = (j + 1) % N
            f.append((o, i * N + k, i * N + j) if up else (o, i * N + j, i * N + k))
    m = Mesh(V, f, C)
    m.f = _orient_rings(m)
    # patina streaks
    s = fbm(m.v, (0.35, 0.04, 0.35), 5, 3)
    m.c = np.clip(m.c * (0.92 + 0.16 * s[:, None]), 0, 1)
    return m


def _orient_rings(m):
    """Make the loft's faces look away from its vertical axis (one check for the whole mesh)."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    n = np.cross(b - a, c - a)
    cen = (a + b + c) / 3
    d = cen - np.stack([np.full(len(cen), 0.7), cen[:, 1], np.full(len(cen), 1.1)], 1)
    side = np.abs(n[:, 1]) < 0.7 * np.linalg.norm(n, axis=1)
    bad = (np.einsum("ij,ij->i", n, d) < 0) & side
    return m.f[:, ::-1] if bad.mean() > 0.5 else m.f


def head():
    out = []
    c, cd = jit(COPPER), tuple(COPPER_D)
    sk = ell(1.5, 2.6, 1.85, c, 22, 14).T(HX, HY, HZ)

    def jaw(v):
        t = np.clip((HY - v[:, 1]) / 2.6, 0, 1)
        v[:, 0] = HX + (v[:, 0] - HX) * (1 - 0.30 * t ** 2)
        v[:, 2] = HZ + (v[:, 2] - HZ) * np.where(v[:, 2] > HZ, 0.93, 1.0)
        return v
    out.append(sk.warp(jaw))
    out.append(ell(1.2, 0.2, 0.3, c).T(HX, 31.72, 2.62))                                  # brow
    for s in (-1, 1):
        out.append(ell(0.36, 0.19, 0.14, tuple(COPPER * 0.78), 12, 6).T(HX + s * 0.6, 31.38, 2.73))   # eyes
    out.append(limb((HX, 31.6, 2.82), (HX, 30.32, 3.38), 0.2, 0.3, c, 6))                  # nose
    out.append(ell(0.32, 0.16, 0.22, c, 10, 6).T(HX, 30.3, 3.3))
    out.append(ell(0.46, 0.12, 0.1, tuple(COPPER * 0.82), 12, 6).T(HX, 29.75, 2.72))       # mouth
    out.append(ell(0.5, 0.4, 0.4, c, 12, 8).T(HX, 29.1, 2.12))                             # chin
    hair = jit(COPPER * 0.93)
    out.append(ell(1.95, 2.1, 1.85, hair, 22, 12).T(HX, 31.6, 1.05))                       # hair, full at the temples
    out.append(ell(0.95, 1.05, 0.8, hair, 14, 8).T(HX, 30.15, -0.45))                      # bun
    for s in (-1, 1):                                                                      # locks on the shoulders
        out.append(sweep([(HX + s * 1.45, 31.2, 0.75), (HX + s * 1.7, 29.6, 0.45), (HX + s * 1.95, 28.1, 0.15),
                          (HX + s * 2.1, 27.3, 0.0)], [0.5, 0.48, 0.42, 0.3], hair, 10))
    out.append(limb((HX, 27.4, 0.95), (HX, 29.4, 1.25), 1.05, 0.92, c, 14))                # neck
    return out


def crown():
    out = []
    c = jit(COPPER * 1.04)
    cx, cz, rx, rz = HX, 1.05, 1.86, 1.92
    band = lathe([(1.0, 32.35), (1.07, 32.45), (1.07, 33.2), (1.03, 33.32), (0.88, 33.36)], c, 32).S(rx, 1, rz).T(cx, 0, cz)
    out.append(band)
    for a in np.radians(np.linspace(-72, 72, 25)):                                         # the 25 windows
        x, z = cx + rx * 1.07 * math.sin(a), cz + rz * 1.07 * math.cos(a)
        w = box(0.16, 0.36, 0.06, DARK, bottom=False).R(0, a, 0)
        out.append(w.T(x, 32.85, z))
    for phi, th in ((0, 55), (45, 22), (-45, 22), (85, 5), (-85, 5), (120, -10), (-120, -10)):  # the seven rays
        p, t = math.radians(phi), math.radians(th)
        b = np.array([cx + rx * 0.98 * math.sin(p), 33.05, cz + rz * 0.98 * math.cos(p)])
        d = np.array([math.sin(p) * math.cos(t), math.sin(t), math.cos(p) * math.cos(t)])
        out.append(limb(b, b + 2.85 * d, 0.27, 0.0, c, 4))
    return out


def right_arm():
    out = []
    c = jit(COPPER)
    P = [a[:3] for a in ARM]
    out.append(sweep(P + [(-4.5, 36.4, 5.3)], [a[3] for a in ARM] + [0.85], c, 16))
    # the loose sleeve hanging from the raised upper arm
    sl = sweep([(-3.0, 26.2, 1.1), (-3.55, 27.6, 1.9), (-3.9, 29.0, 2.7), (-4.0, 30.2, 3.2), (-4.05, 31.2, 3.55),
                (-4.08, 32.0, 3.85)], [1.2, 1.75, 1.85, 1.55, 1.15, 0.92], jit(COPPER * 0.97), 24, folds=(7, 0.07))
    k = 0.9 + 0.1 * np.sin(np.arctan2(sl.v[:, 2] - 2, sl.v[:, 0] + 4) * 7)
    sl.c = np.clip(sl.c * k[:, None], 0, 1)
    out.append(sl)
    # the fist round the torch handle, with the thumb over it
    out.append(ell(1.05, 1.4, 1.2, c, 16, 10).T(-4.68, 36.85, 5.85))
    out.append(limb((-4.2, 37.6, 6.6), (-4.45, 38.1, 7.2), 0.32, 0.26, c, 8))
    return out


def torch():
    copper, gold = [], []
    c = jit(COPPER * 1.05)
    prof = [(0.0, 35.85), (0.5, 35.86), (0.78, 36.05), (0.86, 36.4), (0.7, 36.75), (0.68, 37.0), (0.8, 38.4),
            (0.88, 39.9), (0.98, 40.2), (0.98, 40.6), (1.2, 40.95), (1.5, 41.12), (1.7, 41.25), (1.7, 41.42),
            (1.2, 41.45), (0.0, 41.46)]
    copper.append(lathe(prof, c, 20).T(TX, 0, TZ))
    # balcony railing: posts and a top rail
    for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        x, z = TX + 1.62 * math.cos(a), TZ + 1.62 * math.sin(a)
        copper.append(cyl(0.07, 0.07, 0.62, c, 5, caps=False).T(x, 41.42, z))
    copper.append(lathe([(1.56, 42.0), (1.7, 42.0), (1.7, 42.12), (1.56, 42.12), (1.56, 42.0)], c, 24).T(TX, 0, TZ))
    copper.append(lathe([(0.0, 41.4), (0.85, 41.4), (1.0, 42.1), (1.08, 42.4), (0.0, 42.41)], c, 18).T(TX, 0, TZ))
    # the flame, gilded, licking back and to her left
    fp = [(0.0, 42.3), (0.98, 42.32), (1.15, 42.9), (1.12, 43.6), (0.95, 44.1), (0.6, 44.5), (0.25, 44.78), (0.0, 44.93)]
    fl = lathe(fp, GOLD, 24)

    def lick(v):
        t = np.clip((v[:, 1] - 42.3) / 2.63, 0, 1)
        a = np.arctan2(v[:, 2], v[:, 0])
        k = 1 + 0.14 * np.sin(5 * a + 6 * t) * t
        v[:, 0] = v[:, 0] * k + 0.95 * t ** 3
        v[:, 2] = v[:, 2] * k - 1.5 * t ** 3
        return v
    fl.warp(lick).T(TX, 0, TZ)
    sh = 0.82 + 0.25 * (fl.v[:, 1] - 42.3) / 2.6
    fl.c = np.clip(np.outer(sh, GOLD), 0, 1)
    gold.append(fl)
    return copper, gold


def tablet_and_left_arm():
    out = []
    c = jit(COPPER)
    # tablet 7.19 × 4.14 × 0.61 m; its face looks 50° from the front towards her left; leaning outwards
    tb = merge([box(4.14, 7.19, 0.61, jit(COPPER * 1.02), bottom=False),
                box(3.7, 0.22, 0.06, COPPER_D, bottom=False).T(0, 1.6, 0.33),               # JULY
                box(1.4, 0.22, 0.06, COPPER_D, bottom=False).T(0, 0.4, 0.33),               # IV
                box(3.4, 0.22, 0.06, COPPER_D, bottom=False).T(0, -0.8, 0.33)])             # MDCCLXXVI
    rot, pos = (0, math.radians(50), -0.36), (7.1, 20.8, 0.95)

    def on_tablet(p):
        return Mesh([p], [], c).R(*rot).T(*pos).v[0]
    tb.R(*rot).T(*pos)
    out.append(tb)
    # left forearm coming up to the tablet, the hand over the front of its outer edge
    hand = on_tablet((1.45, -0.9, 0.55))
    out.append(sweep([(5.0, 18.6, 0.3), (6.0, 19.2, 1.2), hand - np.array([0.5, 0.5, 0.35])], [0.95, 0.85, 0.7], c, 12))
    out.append(ell(0.8, 1.25, 0.5, c, 12, 8).R(*rot).T(*hand))
    # the end of the mantle hanging from the left forearm
    dr = sweep([(5.6, 21.6, 0.7), (5.85, 19.6, 0.75), (5.95, 17.8, 0.6), (5.75, 16.6, 0.45), (5.5, 16.1, 0.4)],
               [1.05, 1.1, 0.9, 0.55, 0.25], jit(COPPER * 0.97), 14)
    k = 0.88 + 0.12 * np.sin(np.arctan2(dr.v[:, 2] - 0.6, dr.v[:, 0] - 6.0) * 6)
    dr.c = np.clip(dr.c * k[:, None], 0, 1)
    out.append(dr)
    # her right foot, heel raised behind
    out.append(ell(0.85, 0.65, 1.5, c, 12, 8).R(-0.45, 0, 0).T(-1.5, 2.6, -4.3))
    return out


def statue():
    copper = [body()] + head() + crown() + right_arm() + tablet_and_left_arm()
    tc, gold = torch()
    copper += tc
    m = merge(copper).T(0, Y_ST, 0)
    g = merge(gold).T(0, Y_ST, 0)
    return m, g


def main():
    out = ROOT / "models" / "Q9202"
    out.mkdir(parents=True, exist_ok=True)
    stone = merge(fort() + pedestal())
    bronze, gold = statue()
    parts = [("fort_and_pedestal", stone), ("statue", bronze, {"material": "patina"}),
             ("flame", gold, {"material": "gilt"})]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.9,
                           materials={"patina": {"metallic": 0.15, "roughness": 0.65},
                                      "gilt": {"metallic": 0.8, "roughness": 0.32}})
    allv = np.vstack([stone.v, bronze.v, gold.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    fp = [[round(x, 2), round(z, 2)] for x, z in offset(STAR, 1.0)]
    print("footprint:", json.dumps(fp))


if __name__ == "__main__":
    main()
