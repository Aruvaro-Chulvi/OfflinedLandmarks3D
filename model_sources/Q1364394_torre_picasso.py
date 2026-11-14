"""
Torre Picasso (Q1364394), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Minoru Yamasaki (1912–1986), design 1974, built 1980–1988 (opened January 1989); final design and
site work by Genaro Alas and Pedro Casariego. Protected (Spain: 80 years after the architect's death).

Measurements (see meta.json for the sources):
- OpenStreetMap way 11089668 (height 157 m, 47 levels): main body 37.5 × 49.6 m (es.wikipedia: 38 × 50 m), long
  axis at 4.0°, with notched corners (~1.45 m) and, on the west face, the narrow addition with the extra stairs
  and the ventilation shaft (4.0 × 21.5 m, es.wikipedia). The main façade faces the Plaza de Pablo Ruiz Picasso
  (south, 184.0°): it has the great entrance arch "that carries the whole façade" (es.wikipedia).
- Heights: roof of the crown 157 m (OSM, es.wikipedia). Measured on photos (Wikimedia Commons «Madrid Torre
  Picasso.jpg», «Madrid - AZCA. Torre Picasso 1/3/6.jpg», «Plaza de Pablo Ruiz Picasso, Madrid
  (27994224413).jpg»), ±5 %: stone podium 10.5 m, white crown band 147.5–157 m with a small outward flare at the
  top, west addition up to 146 m.
- Façade: white aluminium piers with narrow recessed window strips (two narrow lights and a white mullion each,
  «Detalle de Torre Picasso.jpg»); 13 strips on the short faces, 17 on the long ones, dark glass in the corner
  notches. Podium in light stone tiles: the big elliptical arch on the south, a row of small round-headed
  windows on the other sides. On the roof, the top slab of the lift block is a heliport (es.wikipedia; OSM
  way 393341921).

Axes (docs/MODEL_SPEC.md): +Y up, the south (entrance) façade faces +Z, +X points east (94°), origin on the
ground at the centroid of the OSM outline.

    python model_sources/Q1364394_torre_picasso.py   -> models/Q1364394/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, quad, tube, extrude, cap, walls, offset, write_glb, GLASS  # noqa: E402

WHITE = (0.93, 0.93, 0.91)            # white lacquered aluminium
WHITE_D = (0.84, 0.845, 0.84)         # mullions, grooves in the crown
STONE = (0.83, 0.82, 0.79)            # podium tiles
GLASS_C = (0.17, 0.19, 0.25)          # dark tinted glass
INK = (0.11, 0.12, 0.14)
ROOF = (0.50, 0.50, 0.50)
PAD = (0.32, 0.33, 0.34)
FOUND = -3.0

# Main body in model x/z (from osm.py way 11089668 --heading 184)
X0, X1, Z0, Z1 = -17.84, 19.67, -24.77, 24.79
NOTCH = 1.45
INSET = 0.35                          # depth of the window strips (glass plane behind the piers)
Y_POD, Y_CROWN, Y_TOP = 10.5, 147.5, 157.0
ADD_X0, ADD_Z0, ADD_Z1, ADD_TOP = -21.82, -10.65, 10.85, 146.0
WW = 1.05                             # window strip width (two narrow lights and a mullion)


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def notched(x0, x1, z0, z1, n):
    """Rectangle with square re-entrant notches at the corners, counter-clockwise in (x, z)."""
    return [(x0 + n, z0), (x1 - n, z0), (x1 - n, z0 + n), (x1, z0 + n), (x1, z1 - n), (x1 - n, z1 - n),
            (x1 - n, z1), (x0 + n, z1), (x0 + n, z1 - n), (x0, z1 - n), (x0, z0 + n), (x0 + n, z0 + n)]


def loft(rings, c):
    """Side walls between rings of the same length [(ring, y), …] (cornices, flares)."""
    out = []
    for (A, ya), (Bq, yb) in zip(rings, rings[1:]):
        v, f = [], []
        n = len(A)
        for i in range(n):
            j = (i + 1) % n
            o = len(v)
            v += [(A[i][0], ya, A[i][1]), (A[j][0], ya, A[j][1]), (Bq[j][0], yb, Bq[j][1]), (Bq[i][0], yb, Bq[i][1])]
            f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
        out.append(Mesh(v, f, c))
    return merge(out)


def strips(L, n, ww):
    """Centres of n window strips spread over a face of length L with equal piers at both ends."""
    pw = (L - n * ww) / (n + 1)
    return [pw + ww / 2 + i * (pw + ww) for i in range(n)], pw


# ── The shaft ────────────────────────────────────────────────────────────────
def shaft():
    glass, white = [], []
    core = notched(X0 + INSET, X1 - INSET, Z0 + INSET, Z1 - INSET, NOTCH - INSET)
    glass.append(extrude(core, Y_POD, Y_CROWN, GLASS_C, top=False))
    # Piers on the four faces: face = (axis, fixed coordinate, outward sign, a0, a1, number of strips)
    faces = [("z", Z1, +1, X0 + NOTCH, X1 - NOTCH, 13), ("z", Z0, -1, X0 + NOTCH, X1 - NOTCH, 13),
             ("x", X1, +1, Z0 + NOTCH, Z1 - NOTCH, 17), ("x", X0, -1, Z0 + NOTCH, Z1 - NOTCH, 17)]
    for ax, c0, s, a0, a1, n in faces:
        L = a1 - a0
        cs, pw = strips(L, n, WW)
        edges = [a0] + [a0 + c + k for c in cs for k in (-WW / 2, WW / 2)] + [a1]
        d0, d1 = (c0 - s * INSET, c0) if s > 0 else (c0, c0 - s * INSET)
        for k in range(0, len(edges), 2):
            e0, e1 = edges[k], edges[k + 1]
            if ax == "z":
                white.append(B(e0, e1, Y_POD, Y_CROWN, d0, d1, WHITE))
            else:
                white.append(B(d0, d1, Y_POD, Y_CROWN, e0, e1, WHITE))
        for cc in cs:                                     # the mullion between the two narrow lights
            m0, m1 = c0 - s * INSET, c0 - s * 0.2
            m0, m1 = min(m0, m1), max(m0, m1)
            if ax == "z":
                white.append(B(a0 + cc - 0.06, a0 + cc + 0.06, Y_POD, Y_CROWN, m0, m1, WHITE_D))
            else:
                white.append(B(m0, m1, Y_POD, Y_CROWN, a0 + cc - 0.06, a0 + cc + 0.06, WHITE_D))
    # The notches: two narrow white mullions in each, up the corner glass
    for sx, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        xc = (X1 if sx > 0 else X0) - sx * NOTCH
        zc = (Z1 if sz > 0 else Z0) - sz * NOTCH
        white.append(B(min(xc, xc + sx * 0.12), max(xc, xc + sx * 0.12), Y_POD, Y_CROWN,
                       min(zc, zc + sz * NOTCH), max(zc, zc + sz * NOTCH), WHITE))
        white.append(B(min(xc, xc + sx * NOTCH), max(xc, xc + sx * NOTCH), Y_POD, Y_CROWN,
                       min(zc, zc + sz * 0.12), max(zc, zc + sz * 0.12), WHITE))
    return merge(glass), merge(white)


def crown():
    out = []
    base = offset(notched(X0, X1, Z0, Z1, NOTCH), 0.0)
    body = extrude(base, Y_CROWN, 155.6, WHITE, top=False)
    out.append(body)
    flare = [(offset(base, d), y) for d, y in ((0.0, 155.6), (0.12, 156.15), (0.32, 156.55), (0.55, 156.8), (0.7, 156.92))]
    out.append(loft(flare, WHITE))
    outer = flare[-1][0]
    inner = offset(base, -0.6)
    out.append(walls(outer, 156.92, Y_TOP, WHITE))
    out.append(cap(outer, Y_TOP, WHITE, holes=[inner]))
    out.append(walls(inner, 154.5, Y_TOP, WHITE_D, hole=True))
    out.append(cap(inner, 154.5, ROOF))
    # light grey grooves continuing the window strips up the crown
    grooves = []
    for ax, c0, s, a0, a1, n in [("z", Z1, +1, X0 + NOTCH, X1 - NOTCH, 13), ("z", Z0, -1, X0 + NOTCH, X1 - NOTCH, 13),
                                 ("x", X1, +1, Z0 + NOTCH, Z1 - NOTCH, 17), ("x", X0, -1, Z0 + NOTCH, Z1 - NOTCH, 17)]:
        cs, _ = strips(a1 - a0, n, WW)
        for cc in cs:
            if ax == "z":
                grooves.append(quad(a0 + cc, (Y_CROWN + 155.4) / 2, c0, 0.5, 155.4 - Y_CROWN - 0.3, "+z" if s > 0 else "-z", WHITE_D, eps=0.03))
            else:
                grooves.append(quad(c0, (Y_CROWN + 155.4) / 2, a0 + cc, 0.5, 155.4 - Y_CROWN - 0.3, "+x" if s > 0 else "-x", WHITE_D, eps=0.03))
    # a dark band where the notches stop under the crown (the corner glass ends there)
    return merge(out), merge(grooves)


def heliport():
    out = [B(-7.6, 9.4, 154.5, 156.4, -12.5, 12.5, ROOF)]
    xc, zc, y = 0.9, 0.0, 156.4
    a = np.linspace(0, 2 * math.pi, 41)
    ring = [(xc + 7.0 * math.cos(t), y + 0.03, zc + 7.0 * math.sin(t)) for t in a]
    out.append(tube(ring, 0.18, WHITE, 4).S(1, 1, 1))
    for dx in (-1.6, 1.6):
        out.append(quad(xc + dx, y, zc, 0.7, 5.0, "+y", WHITE, eps=0.03))
    out.append(quad(xc, y, zc, 2.6, 0.7, "+y", WHITE, eps=0.04))
    pad = quad(xc, y, zc, 15.5, 22.0, "+y", PAD, eps=0.015)
    out.append(pad)
    out.append(cyl(0.08, 0.05, 6.0, WHITE_D, 6).T(xc - 6.5, Y_TOP, zc - 11.0))      # small mast
    return merge(out)


# ── The west addition (stairs and ventilation shaft) ─────────────────────────
def addition():
    out = [B(ADD_X0, X0 + 0.1, FOUND, ADD_TOP, ADD_Z0, ADD_Z1, WHITE),
           B(ADD_X0 - 0.15, X0 + 0.1, ADD_TOP, ADD_TOP + 0.6, ADD_Z0 - 0.15, ADD_Z1 + 0.15, WHITE_D)]
    for z in np.linspace(ADD_Z0 + 3.0, ADD_Z1 - 3.0, 4):
        out.append(quad(ADD_X0, (Y_POD + ADD_TOP - 1) / 2, z, 0.6, ADD_TOP - Y_POD - 3, "-x", INK, eps=0.03))
    return merge(out)


# ── The podium ───────────────────────────────────────────────────────────────
ARCH_W, ARCH_H, ARCH_X = 22.0, 9.2, 0.915
POD_IN = 0.3


def arch_y(dx):
    return ARCH_H * math.sqrt(max(0.0, 1 - (dx / (ARCH_W / 2)) ** 2))


def podium():
    YP = Y_POD - 0.2                                      # tucked under the ledge
    out, dark, glass = [], [], []
    px0, px1, pz0, pz1 = X0 + POD_IN, X1 - POD_IN, Z0 + POD_IN, Z1 - POD_IN
    DEPTH = 3.2                                           # the arch is cut this deep into the podium
    out.append(B(px0, px1, FOUND, YP, pz0, pz1 - DEPTH, STONE))
    # front slab with the elliptical arch: left and right blocks, then the spandrel over the arch
    a0, a1 = ARCH_X - ARCH_W / 2, ARCH_X + ARCH_W / 2
    out.append(B(px0, a0, FOUND, YP, pz1 - DEPTH, pz1, STONE))
    out.append(B(a1, px1, FOUND, YP, pz1 - DEPTH, pz1, STONE))
    seg = 32
    xs = np.linspace(a0, a1, seg + 1)
    for z in (pz1, pz1 - DEPTH):                          # front and back faces of the spandrel
        V = [(x, arch_y(x - ARCH_X), z) for x in xs] + [(a0, YP, z), (a1, YP, z)]
        TL, TR = seg + 1, seg + 2
        F = [(TL, i + 1, i) for i in range(seg // 2)] + [(TL, TR, seg // 2)] + [(TR, i + 1, i) for i in range(seg // 2, seg)]
        m = Mesh(V, F, STONE)
        if z < pz1:
            m.f = m.f[:, ::-1]
        out.append(m)
    V, F = [], []                                         # the soffit
    for x in xs:
        V += [(x, arch_y(x - ARCH_X), pz1), (x, arch_y(x - ARCH_X), pz1 - DEPTH)]
    for i in range(seg):
        a = 2 * i; F += [(a, a + 3, a + 1), (a, a + 2, a + 3)]
    out.append(Mesh(V, F, STONE))
    # glazed screen at the back of the arch, with its mullions
    V = [(ARCH_X, 0.0, pz1 - DEPTH + 0.03)] + [(x, arch_y(x - ARCH_X), pz1 - DEPTH + 0.03) for x in xs]
    F = [(0, i + 1, i + 2) for i in range(seg)]
    gm = Mesh(V, F, GLASS_C); gm.f = gm.f[:, ::-1]
    glass.append(gm)
    for x in np.linspace(a0 + 2.0, a1 - 2.0, 9):
        h = arch_y(x - ARCH_X)
        out.append(B(x - 0.07, x + 0.07, 0, h, pz1 - DEPTH, pz1 - DEPTH + 0.15, WHITE_D))
    out.append(B(a0 + 0.5, a1 - 0.5, 3.4, 3.55, pz1 - DEPTH, pz1 - DEPTH + 0.11, WHITE_D))
    # the raised white band round the arch
    ra, rb = ARCH_W / 2 + 0.3, ARCH_H + 0.3
    pts = [(ARCH_X - ra * math.cos(t), rb * math.sin(t), pz1 + 0.1) for t in np.linspace(0, math.pi, 41)]
    out.append(tube(pts, 0.3, WHITE, 8))
    # ledge between podium and shaft
    out.append(extrude(offset([(X0, Z0), (X1, Z0), (X1, Z1), (X0, Z1)], 0.05), Y_POD - 0.45, Y_POD, WHITE, bottom=True))
    # round-headed windows on the east, north and west sides
    def arched_window(u, face):
        w, h0, h1 = 1.3, 1.0, 3.6
        n = 10
        pts2 = [(u - w / 2, h0), (u + w / 2, h0), (u + w / 2, h1)] + \
               [(u + w / 2 * math.cos(t), h1 + w / 2 * math.sin(t)) for t in np.linspace(0, math.pi, n + 1)[1:-1]] + [(u - w / 2, h1)]
        cx_, cy_ = u, (h0 + h1) / 2
        V = [(cx_, cy_)] + pts2
        F = [(0, i + 1, (i + 1) % len(pts2) + 1) for i in range(len(pts2))]
        if face == "+x":
            V3 = [(px1 + 0.04, y, z) for z, y in V]
        elif face == "-x":
            V3 = [(px0 - 0.04, y, z) for z, y in V]
        else:
            V3 = [(x, y, pz0 - 0.04) for x, y in V]
        m = Mesh(V3, F, INK)
        if face in ("+x",):
            m.f = m.f[:, ::-1]
        return m
    for z in np.arange(pz0 + 2.6, pz1 - 2.0, 2.7):
        dark.append(arched_window(z, "+x"))
        if not (ADD_Z0 - 1.0 < z < ADD_Z1 + 1.0):
            dark.append(arched_window(z, "-x"))
    for x in np.arange(px0 + 2.4, px1 - 2.0, 2.7):
        dark.append(arched_window(x, "-z"))
    return merge(out), merge(dark), merge(glass)


def main():
    out = ROOT / "models" / "Q1364394"
    out.mkdir(parents=True, exist_ok=True)
    g_shaft, w_shaft = shaft()
    c_body, c_groove = crown()
    p_stone, p_dark, p_glass = podium()
    white = merge([w_shaft, c_body, c_groove, addition(), heliport(), p_stone, p_dark])
    glass = merge([g_shaft, p_glass])
    parts = [("torre_picasso", white), ("glass", glass, {"material": "glass"})]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.6, materials={"glass": GLASS})
    allv = np.vstack([white.v, glass.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
