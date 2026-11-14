"""
Torre de Cristal (Q953610), Cuatro Torres Business Area, Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: César Pelli (1926–2019, Pelli Clarke Pelli) with Íñigo Ortiz and Enrique León, 2004–2008 — protected.

Measurements (see meta.json for the sources):
- OpenStreetMap way 979555466 and its 13 building:part ways (glass, skillion roofs with heights):
  · the tower stands on a rectangle of 32.5 × 49.5 m; from 12 m up, each of its four corners is cut by a sloping
    triangular facet that grows to the top, so the east and west faces become tall vertical triangles, the north
    and south faces tall trapezoids, and the top a pointed hexagon (the «cut crystal», «diamond-shaped» tower);
  · the top is one sloping plane: 249 m over the east point, 226.4 m over the west corners of the hexagon,
    220 m over the west point (OSM roof:direction 281.85°, roof:height 29 m);
  · round the foot, a glazed hall whose glass roofs slope from 22 m against the tower (north, south, west; 12 m on
    the east) down to 4 m at the edge of the 37.9 × 53.5 m outline, and a glass entrance pavilion 18.6 m wide and
    12 m high in the middle of the east side.
- Wikipedia / Arquitectura Viva: 249 m, 52 floors (46 of offices over the entrance level and two plant floors),
  top floor at 210 m; above floor 46 a greenhouse with a garden (Balmori Associates; plant wall by Patrick Blanc)
  behind the glass of the crown. Clear low-e glass that takes the colour of the sky: here a light blue-green, with
  thin floor lines, and a lighter glass on the crown.

Heading: the four faces are alike; the main façade (+Z) is the east one, where OSM maps the glazed entrance
pavilion (the tower stands on the south-west corner of its plot, its garden and access to the north and east; the
Paseo de la Castellana side, west, has the 22 m hall roof but no entrance): heading 101.9°.

Axes (docs/MODEL_SPEC.md): +Y up, the east façade faces +Z, +X points north-north-east (11.9°);
origin = centroid of the OSM outline.

    python model_sources/Q953610_torre_de_cristal.py   -> models/Q953610/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, write_glb, GLASS  # noqa: E402

FOUND = -3.0
GLASS_C = (0.44, 0.60, 0.65)        # clear low-e glass reflecting the sky: light blue-green
BAND_C = (0.56, 0.68, 0.72)         # floor lines
CROWN_C = (0.74, 0.84, 0.83)        # the greenhouse crown, lighter
CROWN_LINE = (0.60, 0.70, 0.69)
ROOF_C = (0.34, 0.37, 0.40)         # sloping roof (metal, photovoltaic glass)
HALL_C = (0.56, 0.68, 0.72)         # the glass hall round the foot
FRAME_C = (0.70, 0.73, 0.75)
MULLION_C = (0.60, 0.70, 0.73)       # vertical mullions

# Plan (OSM building:parts in model x/z for heading 101.9, made symmetric within 8 cm)
XW = 16.27                          # half width of the tower (north/south faces at x = ±XW)
ZF, ZB = 24.14, -25.35              # east (front) and west (back) faces
ZA, ZD = 15.44, -14.49              # where the sloping corner facets meet the north/south faces at the top
Y_FACET = 12.0                      # the corner facets start here
H_FRONT, H_BACK = 249.0, 220.0      # the sloping top plane over the east and west points
Y_CROWN = 208.0                     # the greenhouse crown above the last office floor
OUT_X, OUT_ZF, OUT_ZB = 18.95, 26.63, -26.89    # the outline of the glass hall
PAV_X0, PAV_X1, PAV_Z = -9.21, 9.41, 27.15       # the east entrance pavilion


def roof_y(z):
    return H_BACK + (H_FRONT - H_BACK) * (z - ZB) / (ZF - ZB)


def clip_y(poly, y, above):
    out, n = [], len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ia = a[1] >= y if above else a[1] <= y
        ib = b[1] >= y if above else b[1] <= y
        if ia:
            out.append(a)
        if ia != ib:
            t = (y - a[1]) / (b[1] - a[1])
            out.append(tuple(a[k] + (b[k] - a[k]) * t for k in range(3)))
    return out


def fan(poly, c, inside=None):
    """A planar convex polygon; if inside is given, the winding is made to face away from it."""
    if len(poly) < 3:
        return None
    P = np.array(poly, float)
    if inside is not None:
        n = np.cross(P[1] - P[0], P[2] - P[0])
        if np.linalg.norm(n) < 1e-9 and len(P) > 3:
            n = np.cross(P[2] - P[0], P[3] - P[0])
        if np.dot(n, P.mean(0) - np.asarray(inside, float)) < 0:
            P = P[::-1]
    return Mesh(P, [(0, i, i + 1) for i in range(1, len(P) - 1)], c)


LEVELS = [22.0 + k * (Y_CROWN - 22.0) / 46 for k in range(47)]       # 46 office floors + the crown floor line
CROWN_LEVELS = [Y_CROWN + 9.0 * k for k in range(1, 6)]              # glazing rails of the greenhouse


def colour_at(y0, y1):
    m = (y0 + y1) / 2
    if m >= Y_CROWN + 0.6:
        return CROWN_LINE if any(L <= m < L + 0.35 for L in CROWN_LEVELS) else CROWN_C
    return BAND_C if any(L <= m < L + 0.45 for L in LEVELS) else GLASS_C


def banded_face(poly, inside):
    ys = [p[1] for p in poly]
    lo, hi = min(ys), max(ys)
    cuts = sorted({lo, hi} | {y for L in LEVELS for y in (L, L + 0.45) if lo < y < hi}
                  | {y for L in CROWN_LEVELS for y in (L, L + 0.35) if lo < y < hi}
                  | ({Y_CROWN + 0.6} if lo < Y_CROWN + 0.6 < hi else set()))
    ms = []
    for a, b in zip(cuts, cuts[1:]):
        q = clip_y(clip_y(poly, a, True), b, False)
        ms.append(fan(q, colour_at(a, b), inside))
    return merge(ms)


def tower():
    C1, C2, C3, C4 = (XW, ZF), (-XW, ZF), (-XW, ZB), (XW, ZB)         # base corners (x, z)
    EF, A, B, WB, D, G = (0.0, ZF), (XW, ZA), (XW, ZD), (0.0, ZB), (-XW, ZD), (-XW, ZA)   # top hexagon

    def at(p, y):
        return (p[0], y, p[1])

    def top(p):
        return (p[0], roof_y(p[1]), p[1])

    axis = (0.0, 120.0, (ZF + ZB) / 2)
    faces = [
        [at(C2, FOUND), at(C1, FOUND), at(C1, Y_FACET), top(EF), at(C2, Y_FACET)],                     # east
        [at(C1, FOUND), at(C4, FOUND), at(C4, Y_FACET), top(B), top(A), at(C1, Y_FACET)],              # north (+x)
        [at(C4, FOUND), at(C3, FOUND), at(C3, Y_FACET), top(WB), at(C4, Y_FACET)],                     # west
        [at(C3, FOUND), at(C2, FOUND), at(C2, Y_FACET), top(G), top(D), at(C3, Y_FACET)],              # south (-x)
        [at(C1, Y_FACET), top(EF), top(A)], [at(C2, Y_FACET), top(G), top(EF)],                        # corner facets
        [at(C3, Y_FACET), top(WB), top(D)], [at(C4, Y_FACET), top(B), top(WB)],
    ]
    glass = [banded_face(f, axis) for f in faces]
    # vertical mullions on the four vertical faces (every 2.7 m, 4 cm proud of the glass)
    mull = []
    for f, ax, out in ((faces[0], 0, (0, 0, 1)), (faces[1], 2, (1, 0, 0)), (faces[2], 0, (0, 0, -1)),
                       (faces[3], 2, (-1, 0, 0))):
        mull += mullions(f, ax, np.array(out, float))
    # the sloping roof over the greenhouse, and a thin metal edge round it
    hexa = [top(p) for p in (EF, A, B, WB, D, G)]
    roof = fan(hexa, ROOF_C, (0.0, 0.0, (ZF + ZB) / 2))
    edge = []
    for i in range(6):
        p, q = np.array(hexa[i]), np.array(hexa[(i + 1) % 6])
        lip = [tuple(p), tuple(q), tuple(q + (0, 0.6, 0)), tuple(p + (0, 0.6, 0))]
        edge.append(fan(lip, FRAME_C, (0.0, 230.0, (ZF + ZB) / 2)))
    return merge(glass), merge([roof] + edge + mull)


def mullions(face, ax, out, step=2.7, w=0.09):
    """Thin vertical strips on a vertical planar face, at fixed positions along its horizontal axis ax (0 = x, 2 = z),
    from the bottom of the glass (0 m) to where the strip leaves the face (the facets, the sloping top)."""
    P = np.array(face, float)
    lo, hi = P[:, ax].min(), P[:, ax].max()
    ms = []
    for k in range(math.ceil(lo / step), math.floor(hi / step) + 1):
        g = k * step
        if not lo + 0.5 < g < hi - 0.5:
            continue
        ys = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            if (a[ax] - g) * (b[ax] - g) <= 0 and a[ax] != b[ax]:
                ys.append(a[1] + (b[1] - a[1]) * (g - a[ax]) / (b[ax] - a[ax]))
        if len(ys) < 2:
            continue
        y0, y1 = max(min(ys), 0.0), max(ys) - 0.3
        if y1 - y0 < 1.0:
            continue
        base = P[0].copy(); base[ax] = g
        side = np.zeros(3); side[ax] = w
        o = out * 0.04
        q = [base + o - side, base + o + side]
        quad = [(q[0][0], y0, q[0][2]), (q[1][0], y0, q[1][2]), (q[1][0], y1, q[1][2]), (q[0][0], y1, q[0][2])]
        ms.append(fan(quad, MULLION_C, tuple(base - out * 5 + (0, (y0 + y1) / 2 - base[1], 0))))
    return ms


def hall():
    """The glazed hall round the foot of the tower: glass roofs sloping from the tower down to 4 m at the edge
    of the outline (from 22 m on the north, south and west sides, from 12 m on the east), and the east pavilion."""
    ms = []
    inner = {"e": ZF, "w": ZB}
    y_out = 4.0
    # outline corners (outer) and tower corners (inner), with the height of the hall roof against the tower
    o = {"ne": (OUT_X, OUT_ZF), "nw": (OUT_X, OUT_ZB), "sw": (-OUT_X, OUT_ZB), "se": (-OUT_X, OUT_ZF)}
    t = {"ne": (XW, inner["e"]), "nw": (XW, inner["w"]), "sw": (-XW, inner["w"]), "se": (-XW, inner["e"])}
    centre = (0.0, 0.0, (ZF + ZB) / 2)

    def P(xz, y):
        return (xz[0], y, xz[1])
    # north, west, south: 22 m against the tower
    for a, b in (("ne", "nw"), ("nw", "sw"), ("sw", "se")):
        ms.append(fan([P(o[a], y_out), P(o[b], y_out), P(t[b], 22.0), P(t[a], 22.0)], HALL_C, centre))
        ms.append(fan([P(o[a], FOUND), P(o[b], FOUND), P(o[b], y_out), P(o[a], y_out)], HALL_C, centre))
    # east: 12 m against the tower, either side of the pavilion
    for xa, xb in ((OUT_X, PAV_X1), (PAV_X0, -OUT_X)):
        ta, tb = (min(max(xa, -XW), XW), ZF), (min(max(xb, -XW), XW), ZF)
        ms.append(fan([P((xa, OUT_ZF), y_out), P((xb, OUT_ZF), y_out), P(tb, 12.0), P(ta, 12.0)], HALL_C, centre))
        ms.append(fan([P((xa, OUT_ZF), FOUND), P((xb, OUT_ZF), FOUND), P((xb, OUT_ZF), y_out), P((xa, OUT_ZF), y_out)],
                      HALL_C, centre))
    # the north-east and south-east corners, where the 22 m and 12 m roofs meet: a closing triangle
    for k in ("ne", "se"):
        ms.append(fan([P(t[k], 12.0), P(t[k], 22.0), P(o[k], y_out)], HALL_C, (0.0, 10.0, 0.0)))
    # the entrance pavilion: 12 m against the tower, 10 m at its front (no back wall: it stands against the tower)
    x0, x1 = PAV_X0, PAV_X1
    ms.append(fan([(x0, 12.0, ZF), (x1, 12.0, ZF), (x1, 10.0, PAV_Z), (x0, 10.0, PAV_Z)], HALL_C, (0, 0, ZF)))
    ms.append(fan([(x0, FOUND, PAV_Z), (x1, FOUND, PAV_Z), (x1, 10.0, PAV_Z), (x0, 10.0, PAV_Z)], HALL_C, (0, 5, ZF)))
    for x, s in ((x0, -1), (x1, 1)):
        ms.append(fan([(x, FOUND, ZF), (x, FOUND, PAV_Z), (x, 10.0, PAV_Z), (x, 12.0, ZF)], HALL_C, (x - s, 5, ZF + 1)))
    # mullion lines on the pavilion front
    for k in range(1, 8):
        x = x0 + (x1 - x0) * k / 8
        ms.append(fan([(x - 0.08, 0.0, PAV_Z + 0.03), (x + 0.08, 0.0, PAV_Z + 0.03), (x + 0.08, 10.0, PAV_Z + 0.03),
                       (x - 0.08, 10.0, PAV_Z + 0.03)], FRAME_C, (0, 5, ZF)))
    return merge(ms)


def main():
    out = ROOT / "models" / "Q953610"
    out.mkdir(parents=True, exist_ok=True)
    g, r = tower()
    h = hall()
    tris, size = write_glb(out / "model.glb", [("tower_glass", g, {"material": "glass"}),
                                                ("hall_glass", h, {"material": "glass"}),
                                                ("roof", r)],
                           materials={"glass": GLASS}, roughness=0.6, metallic=0.3)
    m = merge([g, h, r])
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
