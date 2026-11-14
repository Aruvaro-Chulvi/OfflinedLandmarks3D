"""
Faro de Moncloa (Q386413), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The tower: Salvador Pérez Arroyo (born 1945), designed 1990, built 1992 (lighting and communications tower of
the City of Madrid, public viewpoint); refurbished and reopened after 2009. Protected.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 4101237 (height 110 m) and its building:parts: the shaft (way 307143827) is a
  cylinder 6.0 m across; the viewing deck (way 307143828) is a "D": a circle 27.8 m across whose centre lies
  3.9 m south of the shaft, cut by a straight edge 1.1 m south of the shaft axis, with a glazed block jutting
  2.4 m north of it around the shaft; the rest of the circle is closed by a ring walkway (way 307143829) 1.25 m
  wide; a tiny entrance porch on the south of the shaft (way 307143831).
- Heights: 110 m to the top of the mast (es.wikipedia, City of Madrid); viewing deck at 92 m (es.wikipedia,
  City of Madrid «a 92 metros»). The rest measured on photos (Wikimedia Commons «Faro de la Moncloa.jpg»,
  «Faro de la Moncloa - 01.jpg», «Faro de Moncloa (Madrid) 01/03/05.jpg»), ±10 %: underside from 83.5 m, deck
  roof 97.4 m, ring walkway at roof level, shaft cap 101 m, lattice mast to 110 m.
- The shaft is clad in stainless-steel plates (es.wikipedia), with a dark vertical slot of small stair windows
  facing south and the guide frames of the two exterior glass lifts on the east and west sides. The deck: an
  inverted cone of light metal panels, a dark band with round lamps near the rim, a sloping glazed band, and the
  glazed gallery above.

Axes (docs/MODEL_SPEC.md): +Y up, +Z faces 175° (south: the side the D-shaped deck opens to, roof:direction
175.2 in OSM, and the side of the entrance porch), +X points 85° (east). Origin on the ground at the axis of the
shaft (not the centroid of the deck outline, which lies 3.9 m to the south in mid-air).

    python model_sources/Q386413_faro_de_moncloa.py   -> models/Q386413/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, tube, write_glb, GLASS, METAL  # noqa: E402

STEEL = (0.80, 0.81, 0.83)            # stainless cladding
STEEL_D = (0.62, 0.63, 0.65)          # plate joints
PANEL = (0.74, 0.71, 0.64)            # the warm light panels under the deck
DARK = (0.10, 0.11, 0.12)
IRON = (0.20, 0.21, 0.23)             # lift guide frames
GLASS_C = (0.42, 0.52, 0.58)
LAMP = (0.93, 0.92, 0.86)
MAST = (0.78, 0.78, 0.76)
RED = (0.80, 0.12, 0.10)
FOUND = -3.0

R_SHAFT = 3.0
CX, CZ, R = 0.45, 3.89, 13.9          # circle of the deck (model x/z), from the OSM outline
ZC = 1.1                              # the straight edge of the D
Y_COLLAR, Y_MID, Y_RIM, Y_FLOOR, Y_GAL, Y_ROOF = 83.5, 88.6, 90.3, 92.0, 92.7, 97.4
Y_CAP, Y_MAST = 101.0, 110.0


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def arc_pts(r, n=72):
    """Points of the deck circle (radius r about the deck centre) south of the straight edge, east to west."""
    dz = ZC - CZ
    t0 = math.atan2(dz, math.sqrt(max(r * r - dz * dz, 0)))               # east end
    t1 = math.pi - t0                                                      # west end
    return [(CX + r * math.cos(t), CZ + r * math.sin(t)) for t in np.linspace(t0, t1, n + 1)]


def ring_strip(A, ya, Bq, yb, c, flip=False):
    """Quads between two polylines of the same length (open, not closed)."""
    v, f = [], []
    for i in range(len(A) - 1):
        o = len(v)
        v += [(A[i][0], ya[i] if np.ndim(ya) else ya, A[i][1]), (A[i + 1][0], ya[i + 1] if np.ndim(ya) else ya, A[i + 1][1]),
              (Bq[i + 1][0], yb[i + 1] if np.ndim(yb) else yb, Bq[i + 1][1]), (Bq[i][0], yb[i] if np.ndim(yb) else yb, Bq[i][1])]
        f += [(o, o + 1, o + 2), (o, o + 2, o + 3)] if not flip else [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    return Mesh(v, f, c)


def poly_y(P, y, c, up=True):
    """Flat convex-ish polygon at height y, fan from its centroid."""
    cx = sum(p[0] for p in P) / len(P); cz = sum(p[1] for p in P) / len(P)
    V = [(cx, y, cz)] + [(x, y, z) for x, z in P]
    F = [(0, i + 1, (i + 1) % len(P) + 1) for i in range(len(P))]
    m = Mesh(V, F, c)
    if up:
        m.f = m.f[:, ::-1]
    return m


def poly_wall(P2, z, c):
    """Planar polygon (x, y) in the plane z (fan from the first vertex: convex polygons only)."""
    V = [(x, y, z) for x, y in P2]
    F = [(0, i, i + 1) for i in range(1, len(P2) - 1)]
    return Mesh(V, F, c)


# ── Shaft ────────────────────────────────────────────────────────────────────
def shaft():
    metal, dark, glass = [], [], []
    metal.append(cyl(R_SHAFT, R_SHAFT, Y_CAP - FOUND, STEEL, 40).T(0, FOUND, 0))
    for y in np.arange(2.6, Y_COLLAR, 2.6):                                  # plate joints
        metal.append(cyl(R_SHAFT + 0.025, R_SHAFT + 0.025, 0.07, STEEL_D, 40, caps=False).T(0, y, 0))
    # dark slot of stair windows facing south
    dark.append(B(-0.55, 0.55, 0.5, Y_COLLAR + 0.5, R_SHAFT - 0.12, R_SHAFT + 0.1, DARK))
    for y in np.arange(4.0, Y_COLLAR - 2, 3.3):
        glass.append(B(-0.32, 0.32, y, y + 1.1, R_SHAFT + 0.1, R_SHAFT + 0.13, GLASS_C))
    # cap above the deck roof
    metal.append(cyl(R_SHAFT + 0.15, R_SHAFT + 0.15, 0.5, STEEL_D, 40).T(0, Y_CAP, 0))
    # entrance porch (OSM way 307143831)
    glass.append(B(-1.5, 0.85, 0.0, 2.9, 2.55, 4.05, GLASS_C))
    metal.append(B(-1.7, 1.05, 2.9, 3.25, 2.4, 4.25, STEEL))
    metal.append(B(-1.7, 1.05, FOUND, 0.0, 2.4, 4.25, STEEL_D))
    return merge(metal), merge(dark), merge(glass)


def lifts():
    out, glass = [], []
    for sx in (1, -1):
        x_r = sx * (R_SHAFT + 1.7)
        for z in (-0.75, 0.75):
            out.append(B(min(x_r, x_r + sx * 0.25), max(x_r, x_r + sx * 0.25), 0.0, 87.0, z - 0.12, z + 0.12, IRON))
        for y in np.arange(3.0, 86.0, 3.5):                                  # brackets to the shaft
            out.append(B(min(sx * 2.7, x_r), max(sx * 2.7, x_r), y, y + 0.25, -0.85, 0.85, IRON))
            p0, p1 = (x_r, y + 0.25, 0.0), (sx * 2.85, y + 1.6, 0.0)
            out.append(tube([p0, p1], 0.08, IRON, 5))
        yc = 46.0 if sx > 0 else 0.3                                         # the glass cabins
        glass.append(B(min(sx * 3.25, sx * 4.6), max(sx * 3.25, sx * 4.6), yc, yc + 2.6, -0.62, 0.62, GLASS_C))
        out.append(B(min(sx * 3.25, sx * 4.6), max(sx * 3.25, sx * 4.6), yc + 2.6, yc + 2.8, -0.65, 0.65, STEEL))
    return merge(out), merge(glass)


# ── The deck ─────────────────────────────────────────────────────────────────
def deck():
    metal, dark, glass, lamps = [], [], [], []
    n = 72
    P = arc_pts(R, n)                                                         # outer edge
    P_in = arc_pts(R - 0.45, n)
    P_gal = arc_pts(R - 0.3, n)

    def collar(r):
        out = []
        for i, (x, z) in enumerate(P):
            a = math.atan2(z, x)
            if i in (0, n):                                                   # the ends lie on the straight edge
                xx = math.copysign(math.sqrt(max(r * r - ZC * ZC, 0)), x)
                out.append((xx, ZC))
            else:
                out.append((r * math.cos(a), r * math.sin(a)))
        return out
    Q = collar(R_SHAFT + 0.2)
    M = [(q[0] + 0.72 * (p[0] - q[0]), q[1] + 0.72 * (p[1] - q[1])) for q, p in zip(Q, P_in)]
    metal.append(ring_strip(Q, Y_COLLAR, M, Y_MID, PANEL))                    # underside: light panels
    dark.append(ring_strip(M, Y_MID, P_in, Y_RIM, DARK))                      # dark band with the lamps
    for k in range(4, n - 3, 3):
        q, p = np.array(M[k]), np.array(P_in[k])
        c = q + 0.55 * (p - q)
        y = Y_MID + 0.55 * (Y_RIM - Y_MID) - 0.05
        lamps.append(cyl(0.32, 0.32, 0.06, LAMP, 10).T(c[0], y - 0.03, c[1]))
    glass.append(ring_strip(P_in, Y_RIM, P, Y_FLOOR, GLASS_C))               # sloping glazed band
    P_out = arc_pts(R + 0.1, n)
    metal.append(ring_strip(P_out, Y_FLOOR, P_out, Y_GAL, STEEL))             # slab edge
    glass.append(ring_strip(P_gal, Y_GAL, P_gal, Y_ROOF - 0.6, GLASS_C))      # the gallery
    for i in range(0, n + 1, 3):                                               # its mullions
        x, z = P_gal[i]
        metal.append(B(x - 0.08, x + 0.08, Y_GAL, Y_ROOF - 0.6, z - 0.08, z + 0.08, STEEL))
    metal.append(ring_strip(P_out, Y_ROOF - 0.6, P_out, Y_ROOF, STEEL))      # roof edge
    metal.append(ring_strip(P_out, Y_FLOOR, P, Y_FLOOR, STEEL))              # little ledges
    metal.append(ring_strip(P_out, Y_ROOF, P_gal, Y_ROOF, STEEL))
    roof = [(x, z) for x, z in P_gal]
    metal.append(poly_y(roof, Y_ROOF, STEEL_D))
    metal.append(ring_strip(P_out, Y_GAL, P_gal, Y_GAL, STEEL, flip=True))
    # the straight edge (north side of the D): closing walls each side of the shaft
    xe, xw = P[0][0], P[-1][0]
    xs = math.sqrt(R_SHAFT ** 2 - ZC ** 2)
    for x_end, x_sh in ((xe, xs), (xw, -xs)):
        metal.append(poly_wall([(x_sh, Y_COLLAR), (x_end, Y_RIM), (x_end, Y_GAL), (x_sh, Y_GAL)], ZC, PANEL))
        glass.append(poly_wall([(x_sh, Y_GAL), (x_end, Y_GAL), (x_end, Y_ROOF - 0.6), (x_sh, Y_ROOF - 0.6)], ZC - 0.02, GLASS_C))
        metal.append(poly_wall([(x_sh, Y_ROOF - 0.6), (x_end, Y_ROOF - 0.6), (x_end, Y_ROOF), (x_sh, Y_ROOF)], ZC - 0.04, STEEL))
        for x in np.linspace(x_sh, x_end, 6)[1:-1]:
            metal.append(B(x - 0.08, x + 0.08, Y_GAL, Y_ROOF - 0.6, ZC - 0.2, ZC, STEEL))
    # the glazed block jutting north around the shaft (OSM: x −5.4 … 7.8, to 1.3 m north of the edge)
    bx0, bx1, bz0 = -5.4, 7.8, -1.3
    metal.append(B(bx0, bx1, 86.0, 86.4, bz0, ZC + 0.2, PANEL))
    glass.append(B(bx0 + 0.1, bx1 - 0.1, 86.4, Y_ROOF - 0.6, bz0 + 0.1, ZC + 0.2, GLASS_C))
    metal.append(B(bx0, bx1, Y_ROOF - 0.6, Y_ROOF + 0.05, bz0, ZC + 0.2, STEEL))
    for y in (89.2, Y_FLOOR, 94.9):
        metal.append(B(bx0, bx1, y, y + 0.3, bz0 - 0.02, bz0 + 0.12, STEEL))
    for x in np.linspace(bx0, bx1, 9):
        metal.append(B(x - 0.09, x + 0.09, 86.4, Y_ROOF - 0.6, bz0 - 0.04, bz0 + 0.12, STEEL))
    return merge(metal), merge(dark), merge(glass), merge(lamps)


def ring():
    """The ring walkway that closes the circle on the north, with its railing and the struts to the shaft."""
    out = []
    dz = ZC - CZ
    t0 = math.atan2(dz, math.sqrt(R * R - dz * dz))
    ts = np.linspace(math.pi - t0, 2 * math.pi + t0, 49)                      # west end, round the north, east end
    o = [(CX + R * math.cos(t), CZ + R * math.sin(t)) for t in ts]
    i_ = [(CX + (R - 1.25) * math.cos(t), CZ + (R - 1.25) * math.sin(t)) for t in ts]
    y0, y1 = Y_ROOF - 1.0, Y_ROOF - 0.1
    out.append(ring_strip(o, y0, o, y1, STEEL, flip=True))
    out.append(ring_strip(i_, y0, i_, y1, STEEL))
    out.append(ring_strip(o, y1, i_, y1, STEEL_D))
    out.append(ring_strip(o, y0, i_, y0, STEEL, flip=True))
    rail = [(CX + (R - 0.1) * math.cos(t), CZ + (R - 0.1) * math.sin(t)) for t in ts]
    out.append(ring_strip(rail, y1, rail, y1 + 1.1, (0.55, 0.56, 0.58), flip=True))
    for t in ts[::4]:
        out.append(cyl(0.05, 0.05, 1.15, STEEL, 5).T(CX + (R - 0.6) * math.cos(t), y1, CZ + (R - 0.6) * math.sin(t)))
    for t in (math.radians(-90 - 42), math.radians(-90 + 42)):                 # V struts from the shaft
        tip = (CX + (R - 0.6) * math.cos(t), y0 + 0.1, CZ + (R - 0.6) * math.sin(t))
        a = math.atan2(tip[2], tip[0])
        for ys in (89.0, 94.5):
            base = (R_SHAFT * math.cos(a), ys, R_SHAFT * math.sin(a))
            out.append(tube([base, tip], 0.22, STEEL, 8))
    return merge(out)


def mast():
    out = []
    h = Y_MAST - Y_CAP - 0.5
    y0 = Y_CAP + 0.5
    legs = []
    for k in range(3):
        a = 2 * math.pi * k / 3 + 0.3
        legs.append(((0.9 * math.cos(a), 0.9 * math.sin(a)), (0.3 * math.cos(a), 0.3 * math.sin(a))))
    for (b, t) in legs:
        out.append(tube([(b[0], y0, b[1]), (t[0], y0 + h, t[1])], 0.06, MAST, 5))
    nseg = 7
    for s in range(nseg):                                                      # zig-zag bracing
        f0, f1 = s / nseg, (s + 1) / nseg
        for k in range(3):
            (b0, t0), (b1, t1) = legs[k], legs[(k + 1) % 3]
            p = lambda bb, tt, f: (bb[0] + (tt[0] - bb[0]) * f, y0 + h * f, bb[1] + (tt[1] - bb[1]) * f)  # noqa: E731
            out.append(tube([p(b0, t0, f0), p(b1, t1, f1)], 0.03, MAST, 4))
    for yy, w in ((y0 + 3.5, 2.6), (y0 + 6.5, 2.0)):                           # cross-arms with antennas
        out.append(B(-w / 2, w / 2, yy, yy + 0.1, -0.05, 0.05, MAST))
        for sx in (-1, 1):
            out.append(cyl(0.06, 0.06, 1.2, MAST, 5).T(sx * w / 2, yy - 0.6, 0))
    out.append(cyl(0.12, 0.12, 0.25, RED, 8).T(0, y0 + h, 0))
    return merge(out)


def main():
    out = ROOT / "models" / "Q386413"
    out.mkdir(parents=True, exist_ok=True)
    s_metal, s_dark, s_glass = shaft()
    l_iron, l_glass = lifts()
    d_metal, d_dark, d_glass, d_lamps = deck()
    metal = merge([s_metal, d_metal, ring(), mast()])
    glass = merge([s_glass, l_glass, d_glass])
    other = merge([s_dark, l_iron, d_dark, d_lamps])
    parts = [("steel", metal, {"material": "steel"}), ("glass", glass, {"material": "glass"}), ("dark", other)]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.7, materials={"glass": GLASS, "steel": METAL})
    allv = np.vstack([metal.v, glass.v, other.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
