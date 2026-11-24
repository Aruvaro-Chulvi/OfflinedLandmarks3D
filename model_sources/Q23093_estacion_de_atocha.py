"""
Estación de Atocha (Q23093), Madrid — the historic station of 1892 with its tropical garden, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Alberto de Palacio (1856–1939), with the engineer Henri Saint-James, 1888–1892: an iron and glass
shed 152 m long, 48 m in span and 27 m high, between brick buildings, the north front on the Glorieta del
Emperador Carlos V. Converted by Rafael Moneo (living) in 1985–1992, when the platforms were replaced by a tropical
garden and the clock tower was built at the south-east corner; the new terminal and platforms (also Moneo) are
separate OpenStreetMap buildings and are not included. Because of Moneo's parts the model is «protected»; the
1892 building alone is in the public domain since 2020.

Measurements (see meta.json for the sources):
- OpenStreetMap way 255246387 «Atocha (Histórica)»: outline 164.2 × 89.2 m, long axis at 150.7°. The north
  front, on the Glorieta, faces 330.7°: that is +Z. Origin: centroid of the outline. Its building:part ways:
  the shed (way 307557629: 152.7 × 50.1 m, height 26, roof:height 13, roof:shape round), the glazed ridge
  lantern (way 307557632: 14 m wide, 25–27 m), the side buildings (12 m + 2 m tiled hipped roofs), the north
  pavilions (17 m + 3–4 m hipped metal roofs), the low north porch (8 m), the south corner blocks (10 m) and
  Moneo's clock tower (way 307557636, 35 m).
- es.wikipedia: «152 m de largo, 27 metros de altura y 48 m de luz»; red pressed brick, white brick ornament.
- PNOA orthophoto (IGN): dark grey-green metal roof with the glazed ridge strip, terracotta tile roofs on the
  side buildings, pyramidal pavilion roofs.

Axes (docs/MODEL_SPEC.md): +Y up, the north front faces +Z, +X points to 240.7° (west-south-west, towards the
Calle de Méndez Álvaro side), origin on the ground at the centroid of the outline.

    python model_sources/Q23093_estacion_de_atocha.py   -> models/Q23093/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import (GLASS, Mesh, box, cap, cyl, extrude, fbm, grid, lathe, merge, offset, ring_area,  # noqa: E402
                     triangulate, tube, walls, write_glb)

rng = np.random.default_rng(1892)

BRICK = np.array((0.66, 0.33, 0.24))      # red pressed brick
BRICK_W = np.array((0.88, 0.84, 0.74))    # white brick and stone bands
MONEO = np.array((0.62, 0.36, 0.27))      # the brick of the 1992 clock tower
SHED = (0.36, 0.42, 0.41)                 # the metal shed roof (dark grey-green)
TILE = (0.70, 0.40, 0.28)                 # terracotta tiles of the side buildings
ZINC = (0.50, 0.53, 0.54)
GLASS_C = (0.55, 0.66, 0.70)
IRON = (0.17, 0.20, 0.21)
WIN = (0.12, 0.14, 0.15)
CLOCK = (0.95, 0.94, 0.90)
FLAT = (0.60, 0.59, 0.56)
FOUND = -3.0

# ── OpenStreetMap way 255246387 and its building:parts (© OpenStreetMap contributors, ODbL), heading 330.7 ─────
OUTLINE = [(-40.03, -82.67), (-40.03, -78.85), (-40.03, -75.07), (-40.05, -73.07), (-39.97, -69.27), (-40.0, 5.16),
           (-39.97, 60.73), (-42.42, 60.76), (-44.52, 60.78), (-44.6, 80.91), (-25.33, 80.91), (-22.08, 80.92),
           (-18.99, 80.92), (18.51, 80.91), (24.79, 80.91), (44.56, 80.94), (44.52, 61.67), (40.21, 61.66),
           (39.89, -46.39), (39.82, -69.53), (39.8, -74.16), (39.76, -82.96), (24.64, -82.85), (15.25, -83.24),
           (8.21, -83.22), (-9.4, -83.02), (-24.53, -82.91), (-32.42, -82.63)]
SIDES = [[(-25.28, 60.72), (-39.97, 60.73), (-40.0, 5.16), (-39.97, -69.27), (-24.69, -69.4)],      # 12 m + tiles
         [(24.35, -69.46), (24.74, 60.59), (40.21, 61.66), (39.89, -46.39), (39.82, -69.53)]]
PAVS = [[(44.52, 61.67), (44.56, 80.94), (24.79, 80.91), (24.74, 60.59), (40.21, 61.66)],          # 17 m + roof
        [(-44.6, 80.91), (-44.52, 60.78), (-39.97, 60.73), (-25.28, 60.72), (-25.33, 80.91)]]
PORCH = [(24.79, 80.91), (24.76, 69.48), (-25.3, 69.21), (-25.33, 80.91)]                          # 8 m
SOUTH = [[(-32.42, -82.63), (-24.53, -82.91), (-24.69, -69.4), (-39.97, -69.27), (-40.03, -82.67)],  # 10 m
         [(39.82, -69.53), (24.35, -69.46), (24.64, -82.85), (39.76, -82.96)]]
TOWER = [(-40.03, -82.67), (-40.03, -75.07), (-32.47, -75.15), (-32.42, -82.63)]                   # 35 m (Moneo)

HW, Z0, Z1 = 24.6, -83.1, 69.3            # the shed: half span, south and north ends
SPRING, CROWN = 13.0, 27.0                # walls, crown of the vault
LANT = 7.0                                # half width of the glazed ridge lantern
R = (HW ** 2 + (CROWN - SPRING) ** 2) / (2 * (CROWN - SPRING))     # radius of the circular arch
YC = CROWN - R


def jit(c, s=0.012):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, s)), 0, 1))


def ccw(P):
    P = [tuple(map(float, p)) for p in P]
    return P if ring_area(P) > 0 else P[::-1]


def edges(P):
    out = []
    for i in range(len(P)):
        a, b = np.array(P[i]), np.array(P[(i + 1) % len(P)])
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        u = (b - a) / L
        out.append((a, b, u, np.array([u[1], -u[0]]), L))
    return out


def on_wall(poly_sy, p0, u, n, c, eps=0.05):
    pts, tris = triangulate(poly_sy)
    v = np.array([(p0[0] + u[0] * s + n[0] * eps, y, p0[1] + u[1] * s + n[1] * eps) for s, y in pts])
    m = Mesh(v, tris, c)
    a, b, cc = v[tris[0, 0]], v[tris[0, 1]], v[tris[0, 2]]
    if np.dot(np.cross(b - a, cc - a), (n[0], 0, n[1])) < 0:
        m.f = m.f[:, ::-1]
    return m


def rect(s0, s1, y0, y1):
    return [(s0, y0), (s1, y0), (s1, y1), (s0, y1)]


def arch(s0, s1, y0, ys, seg=8):
    r, sc = (s1 - s0) / 2, (s0 + s1) / 2
    return [(s0, y0), (s1, y0)] + [(sc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)][:-1] + [(s0, ys)]


def slab(p0, u, n, s0, s1, y0, y1, d_out, d_in, c):
    q = [np.array(p0) + u * s0 + n * d_out, np.array(p0) + u * s1 + n * d_out,
         np.array(p0) + u * s1 - n * d_in, np.array(p0) + u * s0 - n * d_in]
    return extrude([tuple(x) for x in q], y0, y1, c, bottom=True)


def band(lo, hi, ylo, yhi, c):
    v, f = [], []
    n = len(lo)
    for i in range(n):
        j = (i + 1) % n
        o = len(v)
        v += [(lo[i][0], ylo, lo[i][1]), (lo[j][0], ylo, lo[j][1]), (hi[j][0], yhi, hi[j][1]), (hi[i][0], yhi, hi[i][1])]
        f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    return Mesh(v, f, c)


def bbox(P):
    xs, zs = [p[0] for p in P], [p[1] for p in P]
    return min(xs), max(xs), min(zs), max(zs)


def hip(P, H, rise, c, ridge=True):
    """A hipped roof on the bounding rectangle of P (a ridge along the long side, or a point)."""
    x0, x1, z0, z1 = bbox(P)
    w, d = x1 - x0, z1 - z0
    k = min(w, d) / 2
    lo = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
    if ridge and d > w:
        hi = [(x0 + k, z0 + k), (x1 - k, z0 + k), (x1 - k, z1 - k), (x0 + k, z1 - k)]
    elif ridge:
        hi = [(x0 + k, z0 + k), (x1 - k, z0 + k), (x1 - k, z1 - k), (x0 + k, z1 - k)]
    else:
        hi = [(x0 + k * 0.9, z0 + k * 0.9), (x1 - k * 0.9, z0 + k * 0.9), (x1 - k * 0.9, z1 - k * 0.9), (x0 + k * 0.9, z1 - k * 0.9)]
    return [band(lo, hi, H, H + rise, jit(c)), cap(hi, H + rise, jit(c))]


def arc_x(t):
    """The arch of the shed at angle t (0 = right springing, 1 = left), → (x, y)."""
    a0 = math.asin(HW / R)
    a = -a0 + 2 * a0 * t
    return R * math.sin(a), YC + R * math.cos(a)


def shed():
    roof, glass, iron = [], [], []
    seg = 24
    ts = np.linspace(0, 1, seg + 1)
    P = [arc_x(t) for t in ts]
    # the metal vault, leaving the lantern open in the middle
    for i in range(seg):
        (xa, ya), (xb, yb) = P[i], P[i + 1]
        if abs((xa + xb) / 2) < LANT:
            continue
        v = [(xa, ya, Z0), (xb, yb, Z0), (xb, yb, Z1), (xa, ya, Z1)]
        m = Mesh(v, [(0, 2, 1), (0, 3, 2)], jit(SHED, 0.02))
        if np.cross(np.subtract(v[1], v[0]), np.subtract(v[2], v[0]))[1] < 0:
            m.f = m.f[:, ::-1]
        roof.append(m)
    # the ridge lantern (way 307557632): low glazed gable from 25 to 27 m, glazed in slats so the garden shows
    yl = YC + math.sqrt(R ** 2 - LANT ** 2)
    zl0, zl1 = -63.3, 53.5
    for x in (-LANT, LANT):
        roof.append(box(0.4, 25.0 - yl + 0.3, zl1 - zl0, jit(SHED)).T(x, yl - 0.3, (zl0 + zl1) / 2))
    for side in (-1, 1):                                       # metal strips of the vault where the lantern is not
        for za, zb in ((Z0, zl0), (zl1, Z1)):
            v = [(side * LANT, yl, za), (0, CROWN - 0.1, za), (0, CROWN - 0.1, zb), (side * LANT, yl, zb)]
            m = Mesh(v, [(0, 1, 2), (0, 2, 3)], jit(SHED))
            if np.cross(np.subtract(v[1], v[0]), np.subtract(v[2], v[0]))[1] < 0:
                m.f = m.f[:, ::-1]
            roof.append(m)
    z = zl0
    while z < zl1 - 0.5:
        for side in (-1, 1):
            v = [(side * LANT, 25.0, z), (0, 27.0, z), (0, 27.0, z + 1.05), (side * LANT, 25.0, z + 1.05)]
            m = Mesh(v, [(0, 1, 2), (0, 2, 3)], GLASS_C)
            if np.cross(np.subtract(v[1], v[0]), np.subtract(v[2], v[0]))[1] < 0:
                m.f = m.f[:, ::-1]
            glass.append(m)
        z += 1.5
    iron.append(box(0.35, 0.35, zl1 - zl0, IRON).T(0, 27.0, (zl0 + zl1) / 2))
    # iron arches (the De Dion trusses) seen at the ends and every 12 m over the roof
    for zz in list(np.arange(Z0 + 0.4, Z1, 12.0)) + [Z1 - 0.4]:
        pts = [(x * 1.004, y + 0.25, zz) for x, y in P]
        iron.append(tube(pts, 0.22, IRON, 4))
    # the glazed end walls (north on the Glorieta, south towards the new terminal): iron frame and glass
    prof = [(HW, 0.0), (HW, SPRING)] + [arc_x(t) for t in ts[::-1]][1:-1] + [(-HW, SPRING), (-HW, 0.0)]
    for zz, nz in ((Z1, 1), (Z0, -1)):
        poly = [(x, y) for x, y in prof]
        pts, tris = triangulate(poly)
        m = Mesh(np.array([(x, y, zz) for x, y in pts]), tris if nz > 0 else tris[:, ::-1], GLASS_C)
        glass.append(m)
        for xm in np.linspace(-HW + 3, HW - 3, 9):           # mullions
            ytop = YC + math.sqrt(max(R ** 2 - xm ** 2, 0)) - 0.2
            iron.append(box(0.3, ytop, 0.3, IRON).T(xm, 0.0, zz + nz * 0.2))
        for yy in (SPRING, 18.0, 22.5):                       # transoms
            half = math.sqrt(max(R ** 2 - (yy - YC) ** 2, 0)) if yy > SPRING else HW
            iron.append(box(2 * min(half, HW) - 0.4, 0.3, 0.3, IRON).T(0, yy, zz + nz * 0.2))
        iron.append(tube([(x, y, zz + nz * 0.25) for x, y in P], 0.35, IRON, 4))
    return roof, glass, iron


def garden():
    """The tropical garden (1992) on the floor of the shed: a dense canopy and taller palms, all generic."""
    green = []
    x0, x1, z0, z1 = -21.0, 21.0, -64.0, 58.0

    def h(X, Z):
        p = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
        n = fbm(p, 0.09, 7, 3).reshape(X.shape)
        edge = np.minimum.reduce([X - x0, x1 - X, Z - z0, z1 - Z])
        return (4.0 + 11.0 * n) * np.clip(edge / 4.0, 0.2, 1.0)
    g = grid(x0, x1, z0, z1, 21, 61, h, (0.2, 0.42, 0.2))
    g.f = g.f[:, ::-1]
    nrm = fbm(g.v, 0.3, 11, 2)
    g.c = np.stack([0.15 + 0.12 * nrm, 0.33 + 0.18 * nrm, 0.14 + 0.08 * nrm], 1)
    green.append(g)
    for k in range(46):                                       # palms above the canopy
        x, z = rng.uniform(x0 + 2, x1 - 2), rng.uniform(z0 + 2, z1 - 2)
        ht = min(rng.uniform(14.0, 22.0), YC + math.sqrt(R ** 2 - (abs(x) + 3.0) ** 2) - 2.5)   # under the vault
        green.append(cyl(0.3, 0.22, ht, (0.45, 0.38, 0.28), 5, caps=False).T(x, 0, z))
        crown = lathe([(0.0, -0.4), (3.0, 0.0), (2.6, 0.9), (0.0, 1.3)], jit((0.22, 0.45, 0.2), 0.08), 7)
        green.append(crown.T(x, ht - 0.6, z))
    return green


def building():
    brick, white, roofs, dark = [], [], [], []
    O = ccw(OUTLINE)
    white.append(extrude(offset(O, 0.15), FOUND, 0.6, jit(BRICK_W), top=False))
    # the shed's side walls (above the side buildings) and its gable footings
    for sx in (-1, 1):
        brick.append(box(1.2, SPRING - FOUND, Z1 - Z0, jit(BRICK)).T(sx * (HW + 0.6), FOUND, (Z0 + Z1) / 2))
    # side buildings: brick with white bands, arched windows, terracotta hipped roofs
    for P in SIDES:
        P = ccw(P)
        brick.append(extrude(P, 0.6, 12.0, jit(BRICK), top_c=jit(FLAT)))
        roofs += hip(P, 12.0, 2.0, TILE)
        for a, b, u, n, L in edges(P):
            if L < 8:
                continue
            mid = a + u * L / 2
            if abs(mid[0]) < HW + 1.5:                         # the side against the shed
                continue
            white.append(slab(a, u, n, 0, L, 11.2, 12.0, 0.4, 0.0, jit(BRICK_W)))
            white.append(slab(a, u, n, 0, L, 5.4, 5.8, 0.15, 0.0, jit(BRICK_W)))
            nb = int(L / 4.6)
            bay = L / nb
            for i in range(nb):
                s = (i + 0.5) * bay
                dark.append(on_wall(arch(s - 0.9, s + 0.9, 1.0, 4.4), a, u, n, WIN))
                dark.append(on_wall(arch(s - 0.8, s + 0.8, 6.6, 9.6), a, u, n, WIN))
                white.append(slab(a, u, n, i * bay - 0.4, i * bay + 0.4, 0.6, 11.2, 0.2, 0.0, jit(BRICK_W)))
    # north pavilions (17 m) with hipped metal roofs and lanterns
    for P in PAVS:
        P = ccw(P)
        brick.append(extrude(P, 0.6, 17.0, jit(BRICK), top_c=jit(FLAT)))
        roofs += hip(P, 17.0, 4.0, ZINC, ridge=False)
        x0, x1, z0, z1 = bbox(P)
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        white.append(box(2.6, 1.6, 2.6, jit(BRICK_W)).T(cx, 21.0, cz))
        roofs.append(cyl(1.9, 0.0, 1.8, jit(ZINC), 4).R(0, math.pi / 4, 0).T(cx, 22.6, cz))
        for a, b, u, n, L in edges(P):
            if L < 8:
                continue
            white.append(slab(a, u, n, 0, L, 16.0, 17.0, 0.5, 0.0, jit(BRICK_W)))
            white.append(slab(a, u, n, 0, L, 8.0, 8.5, 0.2, 0.0, jit(BRICK_W)))
            for s in np.linspace(L / 6, 5 * L / 6, 3):
                dark.append(on_wall(arch(s - 1.0, s + 1.0, 1.0, 6.2), a, u, n, WIN))
                dark.append(on_wall(arch(s - 0.9, s + 0.9, 9.6, 14.0), a, u, n, WIN))
            white.append(slab(a, u, n, -0.5, 0.5, 0.6, 17.0, 0.5, 0.0, jit(BRICK_W)))
            white.append(slab(a, u, n, L - 0.5, L + 0.5, 0.6, 17.0, 0.5, 0.0, jit(BRICK_W)))
    # the low north porch (8 m) under the great glazed arch: arcade towards the Glorieta
    P = ccw(PORCH)
    brick.append(extrude(P, 0.6, 8.0, jit(BRICK), top_c=jit(FLAT)))
    for a, b, u, n, L in edges(P):
        if n[1] > 0.9:
            white.append(slab(a, u, n, 0, L, 7.2, 8.0, 0.4, 0.0, jit(BRICK_W)))
            white.append(slab(a, u, n, 0, L, 8.0, 8.8, 0.0, 0.3, jit(BRICK_W)))
            for s in np.linspace(3.0, L - 3.0, 9):
                dark.append(on_wall(arch(s - 1.4, s + 1.4, 0.6, 5.4), a, u, n, WIN))
    # south corner blocks (10 m)
    for P in SOUTH:
        P = ccw(P)
        brick.append(extrude(P, 0.6, 10.0, jit(BRICK), top_c=jit(FLAT)))
        for a, b, u, n, L in edges(P):
            white.append(slab(a, u, n, 0, L, 9.3, 10.0, 0.3, 0.0, jit(BRICK_W)))
    # the clock on the north glazed gable, with the two griffins on the springers of the arch
    y_clock = 21.0
    white.append(cyl(2.4, 2.4, 0.4, CLOCK, 20).R(math.pi / 2, 0, 0).T(0, y_clock, Z1 + 0.9))
    dark.append(box(0.18, 1.6, 0.1, IRON).T(0, y_clock, Z1 + 1.15))
    dark.append(box(1.2, 0.16, 0.1, IRON).T(0.6, y_clock, Z1 + 1.15))
    for sx in (-1, 1):
        white.append(box(2.0, 1.6, 2.4, jit(BRICK_W)).T(sx * (HW - 0.8), SPRING, Z1 + 0.6))
        white.append(box(1.2, 2.0, 1.8, jit((0.33, 0.36, 0.36))).T(sx * (HW - 0.8), SPRING + 1.6, Z1 + 0.6))
    # Moneo's clock tower (1992): a plain brick prism with clocks near the top
    T = ccw(TOWER)
    tower = [extrude(T, FOUND, 35.0, jit(MONEO), top_c=jit(FLAT))]
    x0, x1, z0, z1 = bbox(T)
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    tower.append(extrude(offset(T, 0.25), 33.6, 35.2, jit(MONEO * 0.92), bottom=True))
    for a, b, u, n, L in edges(T):
        tower.append(on_wall([(L / 2 + 1.7 * math.cos(t), 30.0 + 1.7 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 17)[:-1]],
                             a, u, n, CLOCK, 0.06))
        dark.append(on_wall(rect(L / 2 - 0.08, L / 2 + 0.08, 30.0, 31.3), a, u, n, IRON, 0.09))
        dark.append(on_wall(rect(0.8, L - 0.8, 1.0, 4.0), a, u, n, WIN))
    return brick, white, roofs, dark, tower


def main():
    out = ROOT / "models" / "Q23093"
    out.mkdir(parents=True, exist_ok=True)
    roof, glass, iron = shed()
    brick, white, roofs, dark, tower = building()
    green = garden()
    parts = [("brick", merge(brick)), ("stone", merge(white)), ("tiles", merge(roofs)),
             ("shed_roof", merge(roof), {"material": "metal"}), ("glass", merge(glass), {"material": "glass"}),
             ("iron", merge(iron + dark)), ("clock_tower_1992", merge(tower)), ("tropical_garden", merge(green))]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.9,
                           materials={"glass": GLASS, "metal": {"metallic": 0.4, "roughness": 0.5}})
    allv = np.vstack([p[1].v for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
