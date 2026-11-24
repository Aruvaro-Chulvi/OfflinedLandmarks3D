"""
Palacio de Biblioteca y Museos Nacionales (Q6057871), Paseo de Recoletos / Calle de Serrano, Madrid — built by code.
The Biblioteca Nacional de España (west, on Recoletos) and the Museo Arqueológico Nacional (east, on Serrano).

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Francisco Jareño y Alarcón (1818–1892), from 1866; finished by Antonio Ruiz de Salces (1820–1899),
1884–1892. Sculpture of the Recoletos front: the pediment by Agustín Querol (died 1909), the seated San Isidoro
(José Alcoverro, died 1908) and Alfonso X (José Alcoverro), the standing Vives, Nebrija, Cervantes and Lope de Vega
(Manuel Fuxá, Anselmo Nogués, Juan Vancell, Pedro Estany, all died before 1945). Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 4099126: outline (14 640 m², 133.3 × 125.6 m), its three courtyards and its 19
  building:part pieces with levels and roof shapes (domes on the four corner pavilions, glass hipped roofs on the
  two rear courts, the pyramidal roof of the central hall, the gabled portico). The long sides face 109.2° /
  289.2°; the main front, on the Paseo de Recoletos, faces 289.2° (west-north-west): +Z. Origin: centroid.
- PNOA orthophoto (IGN): ranges about 11 m deep (16 m at the back), grey metal roofs, slate pavilion roofs, the
  glazed apex of the central pyramid, the open front courtyards.
- Heights measured on frontal photos of the Recoletos portico (Wikimedia Commons «National Library of
  Spain.jpg», «Biblioteca Nacional de Madrid.jpg») scaled by a 1.7 m visitor and the 21 m wide portico (±8 %):
  the doors sit on a podium 3 m above the street; ground storey to 10.6 m, eight Corinthian columns from
  11.4 to 20.4 m, entablature to 22.3 m, pediment apex 28.3 m, the crowning statue to about 31.5 m. OSM
  levels mapped to heights: 3 → 16 m, 4 → 22 m (+ 3 m roofs), 5 → 25 m (pavilions, + 6 m slate domes).

Axes (docs/MODEL_SPEC.md): +Y up, the Recoletos front faces +Z, +X points to 199.2° (south-south-west, towards
Calle de Jorge Juan), origin on the ground at the centroid of the outline (the street level on Recoletos).

    python model_sources/Q6057871_biblioteca_y_museos_nacionales.py   -> models/Q6057871/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import (GLASS, Mesh, box, cap, cyl, extrude, lathe, merge, offset, ring_area,  # noqa: E402
                     triangulate, walls, write_glb)

rng = np.random.default_rng(1892)

STONE = np.array((0.88, 0.85, 0.78))      # Colmenar / Novelda limestone
STONE_D = np.array((0.80, 0.77, 0.70))
BRICK = np.array((0.80, 0.68, 0.52))      # the buff brick of the wall panels
ORN = (0.92, 0.91, 0.87)                  # marble sculpture
ZINC = (0.52, 0.55, 0.57)
SLATE = (0.30, 0.33, 0.38)
FLAT = (0.62, 0.62, 0.60)
GLASS_C = (0.52, 0.63, 0.68)
WIN = (0.12, 0.14, 0.17)
IRON = (0.10, 0.10, 0.11)
GRASS = (0.40, 0.52, 0.30)
FOUND = -4.0
PODIUM = 3.0                              # floor of the Recoletos doors

# ── OpenStreetMap relation 4099126 (© OpenStreetMap contributors, ODbL), model x/z for heading 289.2 ────────
OUTLINE = [(66.5, 60.6), (54.5, 60.6), (54.5, 60.1), (14.4, 59.9), (10.1, 59.9), (10.1, 64.3), (-10.7, 64.3),
           (-10.7, 59.9), (-54.8, 59.8), (-54.8, 60.7), (-66.8, 60.7), (-66.7, 48.2), (-65.5, 48.2), (-65.4, -44.5),
           (-66.7, -44.5), (-66.7, -61.1), (-54.4, -61.0), (-54.4, -60.6), (-16.0, -60.5), (-16.0, -61.3),
           (-0.6, -61.2), (15.6, -61.2), (15.6, -60.4), (48.3, -60.3), (54.8, -60.3), (54.8, -60.6), (60.1, -60.6),
           (66.5, -60.6), (66.5, -44.8), (65.7, -44.8), (65.5, 48.6), (66.5, 48.6)]
COURTS = [[(-17.8, 14.2), (-15.7, 14.2), (15.0, 14.2), (17.6, 14.2), (17.6, 20.4), (-17.8, 20.3)],
          [(-54.7, 48.3), (-54.7, 28.3), (-26.7, 28.5), (-26.8, 48.4)],
          [(54.5, 48.6), (54.5, 28.6), (26.5, 28.4), (26.5, 48.5)]]
# building:parts (id, levels, roof shape, roof material, outline)
PARTS = [
    (306992372, 4, "", "metal", [(17.6, 20.4), (26.6, 20.4), (54.6, 20.7), (54.7, -26.8), (26.5, -26.9), (26.6, -44.7),
        (-23.3, -44.6), (-23.3, -26.2), (-54.5, -26.4), (-54.6, 20.4), (-26.8, 20.3), (-17.8, 20.3), (-17.8, 14.2),
        (-15.7, 14.2), (-15.7, -16.6), (15.0, -16.6), (15.0, 14.2), (17.6, 14.2)]),
    (306992375, 3, "", "", [(26.5, 28.4), (26.6, 20.4), (54.6, 20.7), (54.5, 28.6)]),
    (306992376, 4, "hipped", "glass", [(54.8, -44.8), (54.7, -26.8), (26.5, -26.9), (26.6, -44.7)]),
    (306992377, 3, "", "", [(-26.7, 28.5), (-26.8, 20.3), (-54.6, 20.4), (-54.7, 28.3)]),
    (306992378, 4, "hipped", "glass", [(-54.4, -44.5), (-54.5, -26.4), (-23.3, -26.2), (-23.3, -44.6)]),
    (306992380, 4, "hipped", "metal", [(-17.8, 48.4), (-17.8, 32.5), (-10.6, 32.5), (-10.7, 48.4)]),
    (306992381, 4, "hipped", "metal", [(10.1, 48.5), (10.1, 32.5), (17.5, 32.6), (17.4, 48.5)]),
    (306992384, 4, "pyramidal", "metal", [(-15.7, 14.2), (15.0, 14.2), (15.0, -16.6), (-15.7, -16.6)]),
    (306992385, 4, "hipped", "metal", [(10.1, 32.5), (10.1, 59.9), (-10.7, 59.9), (-10.6, 32.5)]),
    (306992386, 4, "hipped", "metal", [(26.5, 48.5), (26.5, 28.4), (26.6, 20.4), (17.6, 20.4), (17.5, 32.6), (17.4, 48.5)]),
    (306992387, 4, "gabled", "metal", [(-10.7, 64.3), (10.1, 64.3), (10.1, 59.9), (-10.7, 59.9)]),
    (306992388, 3, "skillion", "metal", [(-10.6, 32.5), (17.5, 32.6), (17.6, 20.4), (-17.8, 20.3), (-17.8, 32.5)]),
    (306992390, 4, "hipped", "metal", [(-26.7, 28.5), (-26.8, 48.4), (-17.8, 48.4), (-17.8, 20.3), (-26.8, 20.3)]),
    (306992391, 5, "dome", "slate", [(54.5, 60.6), (54.5, 48.6), (66.5, 48.6), (66.5, 60.6)]),
    (306992392, 5, "dome", "slate", [(-66.7, -44.5), (-66.7, -61.1), (-54.4, -61.0), (-54.4, -44.5)]),
    (306992393, 5, "dome", "slate", [(54.8, -60.6), (66.5, -60.6), (66.5, -44.8), (54.8, -44.8)]),
    (306992394, 5, "dome", "slate", [(-54.8, 60.7), (-66.8, 60.7), (-66.7, 48.2), (-54.7, 48.3)]),
]
# the perimeter ranges (part relation 4099127): the outline minus the inner block; inner edge from the parts
INNER = [(-54.6, -44.6), (54.7, -44.8), (54.6, 48.5), (-54.7, 48.4)]
OUTER = [(-65.4, -60.6), (65.7, -60.3), (65.5, 60.0), (-65.5, 59.8)]      # the outline without its jogs
HT = {3: 16.0, 4: 22.0, 5: 25.0}
TOP = 22.0
PORT = (-10.7, 10.1, 64.3)                # the Recoletos portico: x span and front face


def jit(c, s=0.012):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, s)), 0, 1))


def ccw(P):
    P = [tuple(map(float, p)) for p in P]
    return P if ring_area(P) > 0 else P[::-1]


def in_poly(x, z, P):
    ins = False
    for i in range(len(P)):
        (xi, zi), (xj, zj) = P[i], P[i - 1]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            ins = not ins
    return ins


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


def shrink_ok(P, d):
    Q = offset(P, -d)
    if ring_area(Q) <= 0.05 * ring_area(ccw(P)):
        return None
    if not all(in_poly(x, z, P) for x, z in Q):
        return None
    for (a, b, u, n, L), (a2, b2, u2, n2, L2) in zip(edges(ccw(P)), edges(Q)):
        if np.dot(u, u2) < 0:
            return None
    return Q


def band(lo, hi, ylo, yhi, c):
    v, f = [], []
    n = len(lo)
    for i in range(n):
        j = (i + 1) % n
        o = len(v)
        v += [(lo[i][0], ylo, lo[i][1]), (lo[j][0], ylo, lo[j][1]), (hi[j][0], yhi, hi[j][1]), (hi[i][0], yhi, hi[i][1])]
        f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    return Mesh(v, f, c)


def hipped(P, H, c, rise=3.0, inset=0.3, depth=None):
    """A low hipped roof: a sloping band from the eaves to a flat ridge area (rise metres up)."""
    lo = shrink_ok(P, inset) or ccw(P)
    w = min(max(p[0] for p in P) - min(p[0] for p in P), max(p[1] for p in P) - min(p[1] for p in P))
    for d in ([depth] if depth else []) + [w * 0.42, w * 0.3, w * 0.2, 3.0, 1.5, 0.8]:
        hi = shrink_ok(P, inset + d)
        if hi is not None and len(hi) == len(lo):
            return [band(lo, hi, H, H + rise, jit(c)), cap(hi, H + rise, jit(c))]
    return [cap(P, H + 0.3, jit(c))]


def rect_dome(P, H, c, rise, lantern=True):
    """A slate pavilion roof: a curved square dome (like a cloister vault) with a small lantern."""
    xs, zs = [p[0] for p in P], [p[1] for p in P]
    cx, cz, w, d = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2, max(xs) - min(xs), max(zs) - min(zs)
    prof = [(1.0, 0.0), (0.92, 0.32), (0.75, 0.62), (0.5, 0.86), (0.22, 1.0)]
    ring = lambda s: [(cx - (w / 2 - 0.4) * s, cz - (d / 2 - 0.4) * s), (cx + (w / 2 - 0.4) * s, cz - (d / 2 - 0.4) * s),
                      (cx + (w / 2 - 0.4) * s, cz + (d / 2 - 0.4) * s), (cx - (w / 2 - 0.4) * s, cz + (d / 2 - 0.4) * s)]
    out = []
    for (s0, h0), (s1, h1) in zip(prof[:-1], prof[1:]):
        out.append(band(ring(s0), ring(s1), H + h0 * rise, H + h1 * rise, jit(c)))
    out.append(cap(ring(0.22), H + rise, jit(c)))
    if lantern:
        out += [cyl(1.0, 1.0, 1.6, jit(STONE), 8).T(cx, H + rise, cz), cyl(1.2, 0.0, 1.5, jit(c), 8).T(cx, H + rise + 1.6, cz),
                cyl(0.08, 0.08, 1.2, IRON, 4).T(cx, H + rise + 3.0, cz)]
    return out


def statue(x, z, y0, h, c, seated=False):
    if seated:
        out = [box(0.55 * h, 0.45 * h, 0.6 * h, c).T(x, y0, z),
               lathe([(0.2 * h, 0), (0.16 * h, 0.35 * h), (0.07 * h, 0.45 * h), (0, 0.46 * h)], c, 8).T(x, y0 + 0.45 * h, z - 0.08 * h)]
        out.append(lathe([(0.0, -0.06 * h), (0.07 * h, 0.0), (0.0, 0.07 * h)], c, 8).T(x, y0 + 0.97 * h, z - 0.08 * h))
        return out
    prof = [(0.17 * h, 0), (0.15 * h, 0.35 * h), (0.11 * h, 0.6 * h), (0.12 * h, 0.75 * h), (0.05 * h, 0.82 * h), (0, 0.83 * h)]
    return [lathe(prof, c, 8).T(x, y0, z), lathe([(0.0, -0.07 * h), (0.07 * h, 0.0), (0.0, 0.07 * h)], c, 8).T(x, y0 + 0.9 * h, z)]


# ── The building ─────────────────────────────────────────────────────────────
def palace():
    stone, brick, roofs, glass, dark, orn = [], [], [], [], [], []
    O = ccw(OUTLINE)
    holes = [ccw(h) for h in COURTS]

    # the perimeter ranges: stone plinth, brick walls framed by stone, cornice, balustrade, metal hipped roof
    stone.append(extrude(offset(O, 0.25), FOUND, PODIUM + 1.0, jit(STONE_D), top=False))
    stone.append(walls(O, PODIUM + 1.0, 10.6, jit(STONE)))
    brick.append(walls(O, 10.6, TOP - 1.2, jit(BRICK)))
    stone.append(walls(O, TOP - 1.2, TOP, jit(STONE)))
    for h in holes:
        stone.append(walls(h, FOUND, 10.6, jit(STONE), hole=True))
        brick.append(walls(h, 10.6, TOP, jit(BRICK), hole=True))
    roofs.append(cap(O, TOP, jit(FLAT), holes=holes))
    # the perimeter ranges (relation 4099127): a hipped metal roof between the outer wall and the inner block
    outer = ccw(OUTER)
    inner = ccw(offset(INNER, 0.4))
    ridge = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(outer, inner)]
    roofs.append(band(offset(outer, -0.6), ridge, TOP + 1.0, TOP + 4.2, jit(ZINC)))
    roofs.append(band(ridge, inner, TOP + 4.2, TOP + 1.0, jit(ZINC)))
    for a, b, u, n, L in edges(O):
        stone.append(slab(a, u, n, 0, L, TOP - 0.7, TOP, 0.6, 0.0, jit(STONE)))
        stone.append(slab(a, u, n, 0, L, TOP, TOP + 1.0, 0.0, 0.35, jit(STONE)))      # parapet / balustrade
        stone.append(slab(a, u, n, 0, L, 10.3, 10.9, 0.3, 0.0, jit(STONE)))           # string course
    # courtyards: a paved floor at the podium level
    for h in holes:
        stone.append(cap(h, PODIUM, jit(FLAT)))

    # inner parts
    for pid, lv, shape, mat, P in PARTS:
        P = ccw(P)
        H = HT[lv]
        if shape == "dome":                                   # corner pavilions: stone, slate domes
            stone.append(walls(offset(P, 0.3), PODIUM, H, jit(STONE)))
            stone.append(extrude(offset(P, 0.9), H - 0.8, H, jit(STONE), bottom=True))
            stone.append(extrude(offset(P, 0.4), H, H + 1.0, jit(STONE), holes=[offset(P, 0.0)]))
            roofs += rect_dome(P, H, SLATE, 6.0)
            continue
        if pid == 306992387:                                  # the portico: pediment roof (ridge along z)
            continue
        if lv == 3:
            roofs.append(cap(P, H, jit(FLAT)))
            continue
        if mat == "glass":
            glass += hipped(P, H, GLASS_C, 3.0)
            stone.append(extrude(offset(P, 0.2), H, H + 0.4, jit(STONE), holes=[offset(P, -0.3)]))
        elif shape == "pyramidal":
            xs, zs = [p[0] for p in P], [p[1] for p in P]
            cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
            stone.append(walls(P, TOP, TOP + 3.0, jit(BRICK)))
            roofs.append(band(P, [(cx + (x - cx) * 0.33, cz + (z - cz) * 0.33) for x, z in P], TOP + 3.0, TOP + 9.0, jit(ZINC)))
            glass.append(band([(cx + (x - cx) * 0.33, cz + (z - cz) * 0.33) for x, z in P],
                              [(cx + (x - cx) * 0.02, cz + (z - cz) * 0.02) for x, z in P], TOP + 9.0, TOP + 11.6, GLASS_C))
        else:
            roofs += hipped(P, H, ZINC, 3.0)

    # windows: two storeys of tall windows with pediments, small attic windows, between stone pilasters
    def facade(a, u, n, L, front):
        nb = max(1, int(L / 4.3))
        bay = L / nb
        for i in range(nb):
            s = (i + 0.5) * bay
            p = a + u * s
            if front and PORT[0] - 4.5 < p[0] < PORT[1] + 4.5 and p[1] > 58:
                continue
            dark.append(on_wall(rect(s - 0.75, s + 0.75, PODIUM + 2.2, 9.2), a, u, n, WIN))
            stone.append(slab(a, u, n, s - 1.15, s + 1.15, 9.2, 9.6, 0.25, 0.0, jit(STONE)))
            dark.append(on_wall(rect(s - 0.8, s + 0.8, 12.0, 17.6), a, u, n, WIN))
            stone.append(on_wall([(s - 1.3, 17.7), (s + 1.3, 17.7), (s, 18.6)], a, u, n, STONE * 1.03, 0.2))
            stone.append(slab(a, u, n, s - 1.2, s + 1.2, 11.6, 12.0, 0.3, 0.0, jit(STONE)))
            dark.append(on_wall(rect(s - 0.5, s + 0.5, 19.2, 20.4), a, u, n, WIN))
            stone.append(slab(a, u, n, i * bay - 0.35, i * bay + 0.35, 10.6, TOP - 1.2, 0.15, 0.0, jit(STONE)))

    for a, b, u, n, L in edges(O):
        if L < 3:
            continue
        facade(a, u, n, L, n[1] > 0.9)
    for h in holes:
        for a, b, u, n, L in edges(ccw(h)[::-1]):
            if L > 4:                                        # courtyard walls (the ring runs clockwise: n points in)
                facade(a, u, n, L, False)

    # pavilion façades: one window per storey in the middle, stone quoins
    for pid, lv, shape, mat, P in PARTS:
        if shape != "dome":
            continue
        P = offset(ccw(P), 0.3)
        for a, b, u, n, L in edges(P):
            s = L / 2
            dark.append(on_wall(rect(s - 0.9, s + 0.9, PODIUM + 2.2, 9.2), a, u, n, WIN))
            dark.append(on_wall(arch(s - 0.95, s + 0.95, 12.0, 17.0), a, u, n, WIN))
            dark.append(on_wall(rect(s - 0.6, s + 0.6, 19.8, 22.2), a, u, n, WIN))

    # the Recoletos portico: three arched doors, statues, eight Corinthian columns, entablature, pediment
    x0, x1, zf = PORT
    p0, u, n = np.array((x0, zf)), np.array((1.0, 0.0)), np.array((0.0, 1.0))
    L = x1 - x0
    stone.append(extrude([(x0, 59.9), (x1, 59.9), (x1, zf), (x0, zf)], PODIUM, 22.3, jit(STONE)))
    for k in (-1, 0, 1):
        sc = L / 2 + k * 5.6
        dark.append(on_wall(arch(sc - 1.7, sc + 1.7, PODIUM, 8.0), p0, u, n, WIN))
        stone.append(on_wall(arch(sc - 2.2, sc + 2.2, PODIUM, 8.0), p0, u, n, STONE * 1.03, 0.03))
    for k in range(4):                                       # the four standing statues between the doors
        s = L / 2 + (k - 1.5) * 5.6
        stone.append(box(1.1, 1.4, 1.0, jit(STONE)).T(x0 + s, PODIUM, zf + 0.6))
        orn += statue(x0 + s, zf + 0.6, PODIUM + 1.4, 2.6, ORN)
    stone.append(slab(p0, u, n, -0.4, L + 0.4, 10.6, 11.4, 1.8, 0.0, jit(STONE)))          # the balcony
    for k in range(8):
        cx = x0 + 1.0 + k * (L - 2.0) / 7
        cz = zf + 0.9
        stone.append(cyl(0.48, 0.42, 7.9, jit(STONE), 12).T(cx, 11.4, cz))
        stone.append(cyl(0.42, 0.62, 1.1, jit(ORN), 8).T(cx, 19.3, cz))
        stone.append(box(1.3, 0.5, 1.3, jit(STONE)).T(cx, 11.4, cz))
        if k < 7:
            dark.append(on_wall(rect(1.0 + k * (L - 2.0) / 7 + 0.7, 1.0 + (k + 1) * (L - 2.0) / 7 - 0.7, 12.4, 17.0), p0, u, n, WIN))
    stone.append(extrude([(x0 - 0.3, zf - 0.5), (x1 + 0.3, zf - 0.5), (x1 + 0.3, zf + 1.7), (x0 - 0.3, zf + 1.7)],
                         20.4, 22.3, jit(STONE), bottom=True))
    # pediment: a gable over the portico, ridge along z
    zb, ze, yb, ya = 51.0, zf + 1.7, 22.3, 28.3
    xm = (x0 + x1) / 2
    V = [(x0 - 0.3, yb, ze), (x1 + 0.3, yb, ze), (xm, ya, ze), (x0 - 0.3, yb, zb), (x1 + 0.3, yb, zb), (xm, ya, zb)]
    F = [(0, 1, 2), (3, 5, 4), (0, 2, 5), (0, 5, 3), (1, 4, 5), (1, 5, 2)]
    roofs.append(Mesh(V, F, jit(ZINC)))
    stone.append(Mesh([(x0 - 0.2, yb, ze + 0.02), (x1 + 0.2, yb, ze + 0.02), (xm, ya - 0.1, ze + 0.02)], [(0, 1, 2)], jit(STONE)))
    for xs in np.linspace(x0 + 2.5, x1 - 2.5, 9):           # Querol's tympanum group, stylised
        hgt = (ya - yb - 1.0) * (1 - abs(xs - xm) / (xm - x0)) * 0.8
        if hgt > 0.6:
            orn.append(box(1.6, hgt, 0.5, jit(ORN)).T(xs, yb + 0.4, ze - 0.1))
    orn += statue(xm, ze - 0.8, ya, 3.4, ORN, seated=True)
    for xs in (x0 + 0.3, x1 - 0.3):
        orn += statue(xs, ze - 0.8, yb, 2.2, ORN, seated=True)

    # the stair down to the garden, the seated San Isidoro and Alfonso X on their pedestals
    for k in range(10):
        stone.append(box(32.0 - 0.0 * k, PODIUM - 0.3 * k - FOUND, 0.75, jit((0.82, 0.81, 0.78), 0.01))
                     .T(xm, FOUND, zf + 0.375 + 0.75 * k))
    for sx in (-1, 1):
        stone.append(box(2.0, 2.5, 2.8, jit(STONE)).T(xm + sx * 5.0, 0.0, zf + 7.8))
        orn += statue(xm + sx * 5.0, zf + 7.8, 2.5, 3.4, ORN, seated=True)
        stone.append(box(1.2, PODIUM + 0.9 - FOUND, 8.0, jit(STONE)).T(xm + sx * 16.6, FOUND, zf + 3.6))

    # the Serrano front: a portal with columns in the middle
    for k in range(4):
        cx = -9.0 + k * 6.0
        stone.append(cyl(0.5, 0.45, 7.0, jit(STONE), 10).T(cx, PODIUM, -62.3))
    stone.append(box(22.0, 1.2, 2.4, jit(STONE)).T(0.0, PODIUM + 7.0, -62.0))
    stone.append(box(24.0, PODIUM - FOUND, 3.0, jit(STONE_D)).T(0.0, FOUND, -62.6))
    return stone, brick, roofs, glass, dark, orn


def main():
    out = ROOT / "models" / "Q6057871"
    out.mkdir(parents=True, exist_ok=True)
    parts = palace()
    S, B, R, G, D, O = (merge(x) for x in parts)
    tris, size = write_glb(out / "model.glb", [("stone", S), ("brick", B), ("roofs", R, {"material": "zinc"}),
                                               ("glass", G, {"material": "glass"}), ("windows", D), ("sculpture", O)],
                           roughness=0.9, materials={"glass": GLASS, "zinc": {"metallic": 0.3, "roughness": 0.55}})
    allv = np.vstack([m.v for m in (S, B, R, G, D, O)])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
