"""
Sinagoga del Tránsito, or Synagogue of Samuel ha-Leví (Q1811506), Toledo — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: 1355–1357, for Samuel ha-Leví, treasurer of Pedro I of Castile; builder unknown; public domain.
Church of the Order of Calatrava after 1492, home of the Museo Sefardí since 1964.

Measurements (see meta.json for the sources):
- OpenStreetMap way 113942748 with its four building:parts: the prayer hall (4 levels) 25.6 × 12.8 m, long axis
  at 48.6° (Wikipedia gives 23 × 9.5 m inside), a strip of 3 levels along its south-east side (the old women's
  gallery and the entrance block under the bell tower), a strip of 2 levels along the north-west side and a small
  one-level room at the east corner. The street front of the south-east strip curves a little.
- The entrance façade, with the bell tower and the museum door, faces the small square of Calle Samuel Leví to
  the south-east (photos on Wikimedia Commons; PNOA orthophoto): heading 138.6°.
- Heights measured on a near-frontal photo of that façade (Wikimedia Commons, «España - Toledo - Sinagoga del
  Tránsito - Fachada.JPG», scaled by the 26 m of the front, ±5 %): masonry of the front up to the tiled ledge at
  12 m, the brick upper band of the hall with its row of small arched windows, cornice at 15.2 m, hipped tiled roof
  to about 18 m; the entrance block 9.3 m wide up to 14.9 m, with a little open loggia at its west corner; the
  bell tower 4.5 m wide with two bell arches up to its cornice at 18.4 m, a small pediment to 20.3 m and an iron
  cross to 21.3 m.
Toledo masonry (rubble between courses of brick) below, brick above and at the corners, Arab tiles on the roofs.

Axes (docs/MODEL_SPEC.md): +Y up, the entrance façade faces +Z, +X to the north-east (heading − 90°), origin at
the centroid of the OSM outline; y = 0 is the ground at the museum door (ground_at), the ground falls about 1.5 m
to the south-west corner and rises about 2.5 m to the east one ("ground": "keep").

    python model_sources/Q1811506_sinagoga_del_transito.py   -> models/Q1811506/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, quad, extrude, offset, cap, write_glb  # noqa: E402

rng = np.random.default_rng(1357)

MASON = np.array((0.66, 0.59, 0.49))     # Toledo masonry: grey and ochre rubble
BRICK = np.array((0.80, 0.62, 0.45))
ROOF = (0.56, 0.46, 0.39)                # old Arab tiles, greyish brown
DARK = (0.12, 0.10, 0.09)
WOOD = (0.36, 0.22, 0.13)
IRON = (0.10, 0.10, 0.11)
BRONZE = (0.42, 0.34, 0.18)
FOUND = -5.0

# OSM building:parts in model metres (osm.py way 113942748 --heading 138.6)
HALL = [(12.1, -6.3), (-13.5, -6.3), (-13.4, -5.4), (-12.6, 6.5), (-5.0, 6.4), (12.1, 6.3)]
FRONT = [(12.1, 5.1), (12.1, 6.3), (-5.0, 6.4), (-12.1, 6.5), (-12.1, 10.5), (-10.0, 10.6), (-9.9, 10.8), (-8.2, 10.9),
         (-6.0, 11.1), (-4.2, 11.4), (-4.1, 11.5), (7.4, 10.3), (9.7, 9.7), (10.9, 9.5), (12.0, 8.9), (12.8, 8.7),
         (13.5, 8.8), (14.4, 8.6), (14.4, 8.0), (13.8, 5.2), (13.8, 5.0)]
BACK = [(-13.5, -6.3), (12.1, -6.3), (13.0, -6.3), (13.2, -9.5), (13.3, -10.8), (11.5, -10.7), (9.8, -10.8),
        (-12.4, -10.7), (-13.2, -10.7)]
SMALL = [(13.8, 5.2), (14.4, 8.0), (14.4, 8.6), (16.8, 7.8), (16.7, 7.3), (16.0, 4.8)]
# the street front of the south-east strip, west to east
FRONT_EDGE = [(-12.1, 10.5), (-10.0, 10.6), (-9.9, 10.8), (-8.2, 10.9), (-6.0, 11.1), (-4.2, 11.4), (-4.1, 11.5),
              (7.4, 10.3), (9.7, 9.7), (10.9, 9.5), (12.0, 8.9), (12.8, 8.7), (13.5, 8.8), (14.4, 8.6)]
BACK_EDGE = [(-13.2, -10.7), (-12.4, -10.7), (9.8, -10.8), (11.5, -10.7), (13.3, -10.8)]

HALL_EAVE = 15.2
FRONT_TOP = 12.0
BACK_TOP = 7.4
SMALL_TOP = 4.6
TB_X0, TB_X1, TB_TOP = -12.1, -2.8, 14.9          # the entrance block under the bell tower
BT_X0, BT_X1, BT_Z0, BT_Z1, BT_TOP = -7.4, -2.9, 7.0, 10.6, 18.4   # the bell tower


def jit(c, k=1.0, s=0.03):
    return tuple(np.clip(np.asarray(c) * k * (1 + rng.normal(0, s)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def frustum(ax, az, y0, bx, bz, y1, c, top=True):
    """A roof from a rectangle ±ax × ±az at y0 up to ±bx × ±bz at y1 (bz = 0: a ridge along x)."""
    lo = [(-ax, y0, -az), (ax, y0, -az), (ax, y0, az), (-ax, y0, az)]
    hi = [(-bx, y1, -bz), (bx, y1, -bz), (bx, y1, bz), (-bx, y1, bz)]
    F = []
    for i in range(4):
        j = (i + 1) % 4
        F += [(i, 4 + j, j), (i, 4 + i, 4 + j)]
    if top and bx > 0 and bz > 0:
        F += [(4, 6, 5), (4, 7, 6)]
    return Mesh(lo + hi, F, c)


def zf(x, edge=FRONT_EDGE):
    xs = [p[0] for p in edge]; zs = [p[1] for p in edge]
    return float(np.interp(x, xs, zs))


def slope_at(x, edge=FRONT_EDGE):
    for (xa, za), (xb, zb) in zip(edge, edge[1:]):
        if xa <= x <= xb and xb - xa > 0.3:
            return (zb - za) / (xb - xa)
    return 0.0


def on_front(ms, x, edge=FRONT_EDGE):
    """Place features built facing +z at x = 0, z = 0 on the curved street front, turned with it."""
    a = -math.atan(slope_at(x, edge))
    z = zf(x, edge)
    out = []
    for m in ms:
        out.append(m.copy().R(0, a, 0).T(x, 0, z))
    return out


def dist_to(edge, P):
    """Distance from points (N,2) in the (x, z) plane to a polyline."""
    P = np.asarray(P, float)
    d = np.full(len(P), np.inf)
    for (xa, za), (xb, zb) in zip(edge, edge[1:]):
        a = np.array([xa, za]); b = np.array([xb, zb]); ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        d = np.minimum(d, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    return d


def shed_roof(poly, edge, y0, rise_per_m, c, over=0.35, thick=0.18):
    """A tiled lean-to roof over a strip: level y0 at the street edge, rising towards the wall it leans on."""
    ring = offset(poly, over)
    m = extrude(ring, y0 - thick, y0, c)
    d = dist_to(edge, m.v[:, [0, 2]])
    m.v[:, 1] += np.maximum(d - over, 0) * rise_per_m
    return m


def bands(poly, y0, y1, step=1.25, h=0.2, c=BRICK):
    """Courses of brick between the rubble of Toledo masonry (a thin proud band around the walls)."""
    out = []
    y = y0
    ring = offset(poly, 0.03)
    while y < y1 - 0.3:
        out.append(extrude(ring, y, y + h, jit(c, 1.0, 0.04), top=False))
        y += step
    return out


def arch_face(w, h, c, seg=10):
    """A flat round-headed opening facing +z, sill at y = 0, centred on x = 0."""
    r = w / 2
    V = [(-r, 0, 0), (r, 0, 0), (r, h - r, 0), (-r, h - r, 0)]
    F = [(0, 1, 2), (0, 2, 3)]
    o = len(V)
    V.append((0, h - r, 0))
    for k, a in enumerate(np.linspace(0, math.pi, seg + 1)):
        V.append((r * math.cos(a), h - r + r * math.sin(a), 0))
    for k in range(seg):
        F.append((o, o + 1 + k, o + 2 + k))
    return Mesh(V, F, c)


def horseshoe_face(w, h, c, seg=12):
    """A small horseshoe (more than half a circle) opening facing +z, sill at y = 0."""
    r = w / 2 * 1.12
    yc = h - r
    V = [(-w / 2, 0, 0), (w / 2, 0, 0)]
    # arc from the right springing (below the centre) round over the top to the left springing
    t0 = -math.acos(min(1.0, (w / 2) / r))
    arc = np.linspace(t0, math.pi - t0, seg + 1)
    pts = [(r * math.cos(t), yc + r * math.sin(t), 0) for t in arc]
    V += pts
    F = []
    n = len(V)
    # fan from the middle of the sill
    V.append((0, 0, 0))
    ctr = n
    ring = [1] + list(range(2, n)) + [0]
    for i in range(len(ring) - 1):
        F.append((ctr, ring[i], ring[i + 1]))
    return Mesh(V, F, c)


def window_grille(w, h, c=DARK):
    out = [quad(0, h / 2, 0, w, h, "+z", c, eps=0.03)]
    out.append(B(-w / 2 - 0.15, w / 2 + 0.15, -0.12, 0.0, 0.0, 0.15, jit(BRICK, 1.05)))
    for k in range(1, 5):
        x = -w / 2 + w * k / 5
        out.append(B(x - 0.02, x + 0.02, 0, h, 0.04, 0.08, IRON))
    for k in range(1, 6):
        y = h * k / 6
        out.append(B(-w / 2, w / 2, y - 0.02, y + 0.02, 0.04, 0.08, IRON))
    return out


def lattice(w, h):
    """The wooden lattice (celosía) shutters of the window over the door."""
    out = [quad(0, h / 2, 0, w + 0.3, h + 0.3, "+z", jit(BRICK, 1.08), eps=0.02),
           quad(0, h / 2, 0, w, h, "+z", WOOD, eps=0.05)]
    for k in range(1, 4):
        x = -w / 2 + w * k / 4
        out.append(B(x - 0.04, x + 0.04, 0, h, 0.05, 0.1, (0.28, 0.17, 0.10)))
    for k in range(1, 8):
        y = h * k / 8
        out.append(B(-w / 2, w / 2, y - 0.015, y + 0.015, 0.05, 0.08, (0.28, 0.17, 0.10)))
    return out


def main_door(w=3.0, h=4.6):
    """The museum door: a deep wooden door in a brick frame, with a step and a ramp."""
    out = [quad(0, h / 2, 0, w + 0.6, h + 0.3, "+z", jit(BRICK, 1.06), eps=0.03),
           quad(0, h / 2 - 0.15, 0, w, h - 0.3, "+z", WOOD, eps=0.06),
           quad(0, h / 2 + 0.8, 0, 0.06, h - 1.6, "+z", (0.22, 0.13, 0.08), eps=0.08),
           B(-w / 2 - 0.6, w / 2 + 2.6, FOUND, 0.12, 0.0, 1.1, jit(MASON, 1.08))]
    return out


def hall():
    parts = []
    # walls: masonry with courses of brick, and the brick upper band (from the front's ledge to the cornice)
    parts.append(extrude(HALL, FOUND, HALL_EAVE - 0.35, jit(MASON)))
    parts += bands(HALL, 0.6, 11.8)
    parts.append(extrude(offset(HALL, 0.05), 11.9, HALL_EAVE - 0.35, jit(BRICK, 0.98), top=False))
    parts.append(extrude(offset(HALL, 0.18), HALL_EAVE - 0.35, HALL_EAVE - 0.15, jit(BRICK, 1.05)))   # dentil cornice
    parts.append(extrude(offset(HALL, 0.32), HALL_EAVE - 0.15, HALL_EAVE, jit(BRICK, 1.08)))
    for x in np.linspace(-13.0, 11.7, 64):                                                     # the dentils
        for z in (6.45 + 0.12, -6.3 - 0.12):
            parts.append(B(x - 0.07, x + 0.07, HALL_EAVE - 0.55, HALL_EAVE - 0.35, z - 0.07, z + 0.07, jit(BRICK, 0.9)))
    # the row of small horseshoe windows and blind brick panels under the eaves, both long sides
    for side, z0, xs in ((1, 6.42, np.arange(-1.4, 11.5, 2.45)), (-1, -6.32, np.arange(-12.2, 11.5, 2.45))):
        for x in xs:
            w = horseshoe_face(0.62, 1.15, DARK)
            pan = quad(0, 0, 0, 1.4, 1.1, "+z", jit(BRICK, 1.07), eps=0.03)
            if side < 0:
                w.R(0, math.pi, 0); pan.R(0, math.pi, 0)
            parts += [w.T(x, 12.75, z0 + 0.06 * side), pan.T(x + 1.22 * side, 13.45, z0)]
    # NE end: two small high windows
    for z in (-2.5, 2.5):
        w = horseshoe_face(0.62, 1.15, DARK).R(0, math.pi / 2, 0).T(12.16, 12.75, z)
        parts.append(w)
    # hipped roof of Arab tiles
    cx, cz = (-13.5 + 12.1) / 2, (-6.3 + 6.45) / 2
    ax, az = (12.1 + 13.5) / 2 + 0.45, (6.45 + 6.3) / 2 + 0.45
    parts.append(frustum(ax, az, HALL_EAVE, ax - az + 0.2, 0.0, HALL_EAVE + 3.0, ROOF).T(cx, 0, cz))
    parts.append(tube([(cx - (ax - az + 0.2), HALL_EAVE + 3.05, cz), (cx + (ax - az + 0.2), HALL_EAVE + 3.05, cz)], 0.14,
                      (0.62, 0.52, 0.44), 6))
    return parts


def front():
    parts = []
    parts.append(extrude(FRONT, FOUND, FRONT_TOP, jit(MASON, 1.02)))
    parts += bands(FRONT, 0.6, FRONT_TOP)
    parts.append(extrude(offset(FRONT, 0.06), FOUND, 0.5, jit(MASON, 0.9)))                    # plinth
    parts.append(shed_roof(FRONT, FRONT_EDGE, FRONT_TOP + 0.25, 0.2, ROOF))
    # the tiled ledge at the street front
    parts.append(tube([(x, FRONT_TOP + 0.05, zf(x) + 0.32) for x in np.linspace(-2.8, 14.4, 30)], 0.1, jit(ROOF, 0.9), 5))
    # the entrance block: brick corners and upper part, masonry below
    tb = [(TB_X0, 6.45), (TB_X1, 6.4), (TB_X1, zf(TB_X1))] + [p for p in FRONT_EDGE[::-1] if TB_X0 <= p[0] < TB_X1]
    parts.append(extrude(tb, FOUND, TB_TOP - 0.3, jit(MASON, 1.0)))
    parts.append(extrude(offset(tb, 0.04), 10.3, TB_TOP - 0.3, jit(BRICK, 1.0), top=False))
    parts.append(extrude(offset(tb, 0.2), TB_TOP - 0.3, TB_TOP, jit(BRICK, 1.06), top_c=ROOF))
    parts += bands(tb, 0.6, 10.2)
    for x in (TB_X0 + 0.35, -8.4, TB_X1 - 0.35):                                            # brick pilasters
        parts += on_front([B(-0.4, 0.4, 0.4, TB_TOP - 0.3, -0.3, 0.08, jit(BRICK))], x)
    parts.append(B(TB_X0 - 0.08, TB_X0 + 0.5, 0.4, TB_TOP - 0.3, 6.5, 10.5, jit(BRICK)))
    # the loggia at the west corner: an arch to the front and one to the side
    parts += on_front([arch_face(1.2, 2.1, DARK).T(0, 12.2, 0.06)], -10.3)
    parts.append(arch_face(1.2, 2.1, DARK).R(0, -math.pi / 2, 0).T(TB_X0 - 0.06, 12.2, 8.4))
    parts += on_front([B(-2.0, 2.0, 10.2, 10.45, 0, 0.28, jit(BRICK, 1.1))], -6.3)           # string course
    # the door, the lattice window over it, a barred window to its right and a lantern
    parts += on_front(main_door(), -6.3)
    parts += on_front([m.T(0, 8.0, 0) for m in lattice(2.8, 2.9)], -6.3)
    parts += on_front([m.T(0, 1.4, 0) for m in window_grille(1.4, 3.1)], -0.3)
    parts += on_front([B(-0.12, 0.12, 6.1, 6.6, 0, 0.35, IRON)], -3.6)
    # windows of the women's gallery further along the front
    for x in (4.0, 8.2):
        parts += on_front([m.T(0, 7.6, 0) for m in window_grille(1.0, 1.5)], x)
    # the bell tower
    parts.append(B(BT_X0, BT_X1, TB_TOP - 0.3, BT_TOP, BT_Z0, BT_Z1, jit(BRICK, 1.0)))
    parts.append(B(BT_X0 - 0.2, BT_X1 + 0.2, BT_TOP, BT_TOP + 0.35, BT_Z0 - 0.2, BT_Z1 + 0.2, jit(BRICK, 1.08)))
    for xb in (-6.25, -4.05):
        for side in (1, -1):
            z = BT_Z1 + 0.05 if side > 0 else BT_Z0 - 0.05
            a = arch_face(0.85, 1.7, DARK).T(xb, 16.2, 0)
            if side < 0:
                a.R(0, math.pi, 0).T(2 * xb, 0, 0)
            parts.append(a.T(0, 0, z))
            fr = arch_face(1.15, 1.95, jit(BRICK, 1.12)).T(xb, 16.05, 0)
            if side < 0:
                fr.R(0, math.pi, 0).T(2 * xb, 0, 0)
            parts.append(fr.T(0, 0, z - 0.02 * side))
        bell = lathe([(0.0, 0.0), (0.33, 0.0), (0.3, 0.12), (0.2, 0.45), (0.17, 0.62), (0.0, 0.66)], BRONZE, 12)
        parts.append(bell.T(xb, 16.35, BT_Z1 - 0.35))
    # the small pediment: a gable roof across the tower with its triangular brick front
    yb, yt = BT_TOP + 0.35, BT_TOP + 1.9
    hw = (BT_X1 - BT_X0) / 2 + 0.3
    xc = (BT_X0 + BT_X1) / 2
    for s in (-1, 1):
        V = [(xc - hw, yb, BT_Z1 + 0.25), (xc + hw, yb, BT_Z1 + 0.25), (xc, yt, BT_Z1 + 0.25)]
        tri = Mesh(V, [(0, 1, 2)] if s > 0 else [(0, 2, 1)], jit(BRICK, 1.06))
        if s < 0:
            tri.v[:, 2] = BT_Z0 - 0.25
        parts.append(tri)
    for s in (-1, 1):
        V = [(xc + s * hw, yb, BT_Z0 - 0.3), (xc + s * hw, yb, BT_Z1 + 0.3), (xc, yt + 0.05, BT_Z1 + 0.3), (xc, yt + 0.05, BT_Z0 - 0.3)]
        F = [(0, 2, 1), (0, 3, 2)] if s > 0 else [(0, 1, 2), (0, 2, 3)]
        parts.append(Mesh(V, F, ROOF))
    parts.append(B(xc - 0.03, xc + 0.03, yt, yt + 1.0, BT_Z1 + 0.15, BT_Z1 + 0.21, IRON))       # iron cross
    parts.append(B(xc - 0.28, xc + 0.28, yt + 0.62, yt + 0.68, BT_Z1 + 0.15, BT_Z1 + 0.21, IRON))
    return parts


def back():
    parts = [extrude(BACK, FOUND, BACK_TOP, jit(MASON, 0.98))]
    parts += bands(BACK, 0.6, BACK_TOP)
    parts.append(shed_roof(BACK, BACK_EDGE, BACK_TOP + 0.2, 0.22, ROOF))
    for x in np.arange(-10.5, 11.0, 4.2):
        w = arch_face(0.9, 1.5, DARK).R(0, math.pi, 0).T(x, 4.2, -10.83)
        parts.append(w)
        parts.append(quad(x, 1.9, -10.75, 0.9, 1.6, "-z", DARK, eps=0.05))
    return parts


def small():
    edge = [(14.4, 8.6), (16.8, 7.8)]
    parts = [extrude(SMALL, FOUND, SMALL_TOP, jit(MASON, 1.0))]
    parts += bands(SMALL, 0.6, SMALL_TOP)
    parts.append(shed_roof(SMALL, edge, SMALL_TOP + 0.2, 0.15, ROOF))
    return parts


def main():
    out = ROOT / "models" / "Q1811506"
    out.mkdir(parents=True, exist_ok=True)
    m = merge(hall() + front() + back() + small())
    tris, size = write_glb(out / "model.glb", [("sinagoga_del_transito", m)], roughness=0.9)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
