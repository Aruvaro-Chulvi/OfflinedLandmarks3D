"""
Museo del Prado (Q160112), Madrid — the Villanueva building, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Juan de Villanueva (1739–1811), designed 1785, finished after his death by Antonio López Aguado
(1764–1831), opened 1819; later works by Narciso Pascual y Colomer (apse, 1853), Francisco Jareño (north
front, 1880s), Fernando Arbós (east ranges, 1911–1921), Pedro Muguruza (Goya stair, 1943–1946), Chueca Goitia
and Lorente (east ranges, 1954–1956). Rafael Moneo's 2007 extension (the Jerónimos cube) is a separate building
in OpenStreetMap and is NOT modelled.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 13046997 «Edificio Villanueva» (outline traced from aerial imagery, ±0.5 m): 210.6 ×
  61.5 m, long axis at 170.5°, so the west façade (Paseo del Prado, Puerta de Velázquez) faces 260.5°. One
  courtyard (south pavilion). Its building:parts give the central portico (20.35 m wide, 9.3 m deep), the
  apse, the Goya podium and stair landings, the rotunda (centre −75.9, 6.9, radius ≈ 10.4 m) and the taller
  blocks of the Goya front (z −5.5…19).
- PNOA orthophoto (IGN, CC BY 4.0): roofs — long glazed lanterns over the west galleries, a ring of glazing
  round each end pavilion (around the rotunda in the north, around the courtyard in the south), the glazed
  apse, saw-tooth skylight roofs on the eastern ranges.
- Heights measured on near-orthographic photos (Wikimedia Commons «Museo del Prado 2016 (25185969599).jpg»,
  «Museo del Prado (Madrid) 04.jpg», «Museo del Prado 00 Fachada 1a/1b.jpg», «Puerta de Goya del Museo del
  Prado.jpg»), ±5 %: ground storey 6.4 m, Ionic loggia columns 7.1–11.9, gallery cornice 14.8; Doric portico
  columns 10.0 m, its entablature 13.4, attic side blocks 16.6, relief block 18.1 (+ slab 18.4).
- West front: Doric portico of six granite columns with Ramón Barba's relief on the attic; long galleries with
  arched openings alternating with niches (statues, vases) and portrait medallions on brick, and above them
  an Ionic colonnade in front of the recessed upper wall, a frieze with small windows and brick panels, and a
  modillion cornice. North front (Goya): podium with two stairs and an Ionic portico in antis on the upper
  floor between two taller blocks. South front (Murillo): lintelled door and a loggia of six Corinthian
  columns (sources say Corinthian, not Ionic). Ends and east side: red brick with granite.
Simplified: the plan inside the outline, the roof shapes (one cornice height, low hipped roofs), the stairs,
the sculpture (stylised volumes). Foundations go 5 m down: the ground rises towards the east.

Axes (docs/MODEL_SPEC.md): +Y up, the west façade faces +Z, +X points south (towards the Murillo end), origin
on the ground at the centroid of the OSM outline; y = 0 is the level of the Paseo del Prado front.

    python model_sources/Q160112_museo_del_prado.py   -> models/Q160112/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import (GLASS, Mesh, box, cap, cyl, lathe, merge, offset, ring_area, tube, walls,  # noqa: E402
                     write_glb)

rng = np.random.default_rng(1819)

STONE_W = np.array((0.80, 0.79, 0.755))     # the light granite + Colmenar stone of the west front
GRANITE = np.array((0.66, 0.65, 0.62))      # granite bands, plinths, pilasters of the brick fronts
STONE = (0.87, 0.855, 0.81)                 # columns, cornices, capitals
BRICK = np.array((0.66, 0.37, 0.28))
MARBLE = (0.94, 0.93, 0.90)
ROOF = (0.50, 0.52, 0.55)                   # lead and zinc
ROOF_FLAT = (0.44, 0.45, 0.47)
GLASS_C = (0.47, 0.57, 0.63)
DARK = (0.12, 0.13, 0.15)
NICHE = (0.52, 0.51, 0.48)
IRON = (0.17, 0.18, 0.19)
FOUND = -5.0

# OSM relation 13046997 in model x/z (heading 260.5°, origin = centroid 40.4138147, −3.6922155)
OUTLINE = [(-6.50, -21.13), (-11.85, -21.13), (-11.83, -22.34), (-12.75, -22.43), (-12.74, -25.91), (-14.20, -25.88),
           (-14.29, -31.87), (-46.35, -31.75), (-55.36, -31.78), (-55.41, -23.05), (-55.27, -21.04), (-55.26, -15.57),
           (-97.94, -15.43), (-97.95, -12.23), (-107.26, -12.22), (-107.27, -9.24), (-107.26, -6.79), (-103.56, -6.80),
           (-103.57, -2.15), (-103.57, 16.09), (-103.57, 19.93), (-107.30, 19.93), (-107.30, 22.25), (-107.30, 26.07),
           (-98.21, 26.06), (-98.22, 29.32), (-53.31, 29.27), (-53.28, 17.98), (-50.47, 17.98), (-7.09, 17.97),
           (-7.09, 21.64), (-7.09, 27.26), (2.96, 27.25), (13.26, 27.24), (13.26, 21.63), (13.26, 17.71), (58.76, 17.53),
           (58.69, 29.41), (103.31, 29.36), (103.30, 17.56), (103.31, 10.10), (103.32, 6.64), (103.31, 1.94),
           (103.30, -4.04), (103.31, -15.82), (60.06, -15.77), (60.19, -23.31), (60.20, -29.77), (60.20, -32.10),
           (53.83, -32.08), (51.02, -32.07), (19.97, -31.90), (19.92, -25.89), (18.81, -25.94), (18.81, -22.36),
           (17.25, -22.44), (17.22, -21.09), (12.50, -21.11), (11.01, -24.16), (9.58, -25.84), (7.81, -27.15),
           (5.76, -28.04), (3.56, -28.44), (1.56, -28.36), (-0.38, -27.87), (-3.02, -26.42), (-4.30, -25.23),
           (-5.34, -23.84), (-6.04, -22.49)]
COURT = [(75.56, 13.91), (86.78, 14.00), (86.79, 10.11), (86.73, 1.93), (86.70, -0.57), (75.52, -0.56), (75.50, 0.33),
         (75.56, 12.30)]

# The body without the Goya podium (points 14–23) and without the open portico (31–33)
MAIN = OUTLINE[0:14] + OUTLINE[24:31] + OUTLINE[34:]
LOGGIA = 2.2                                  # the upper wall of the west galleries stands this far back
GAL_N = (-53.28, -7.09)                       # x range of the north gallery, front z 17.97
GAL_S = (13.26, 58.76)                        # x range of the south gallery, front z 17.71 -> 17.53
PORT = (-7.09, 13.26)                         # the portico / central body
XC = (PORT[0] + PORT[1]) / 2                  # its axis
G1 = 6.45                                     # top of the ground storey
TOP = 14.0                                    # top of the walls (frieze), cornice up to 14.8
ROT_C, ROT_R = (-75.9, 6.9), 10.4             # the Goya rotunda


def zf(x):
    """Front line (z) of the west galleries at x."""
    if x < 0:
        return 17.975
    return 17.71 + (x - 13.26) / (58.76 - 13.26) * (17.53 - 17.71)


def upper_ring():
    out = []
    for x, z in MAIN:
        if 15 < z < 19 and (GAL_N[0] - 0.1 <= x <= GAL_N[1] + 0.1 or GAL_S[0] - 0.1 <= x <= GAL_S[1] + 0.1):
            z -= LOGGIA
        out.append((x, z))
    return out


def jit(c, k=0.02):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, k)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def west_part(x, z):
    return z > 15 and GAL_N[0] - 0.5 < x < GAL_S[1] + 0.5


def walls_c(ring, y0, y1, colfn, hole=False):
    """Walls along a ring, each side its own colour from colfn(mid x, mid z)."""
    P = ring if (ring_area(ring) > 0) != hole else ring[::-1]
    ms = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        ms.append(_quad_wall(a, b, y0, y1, colfn((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)))
    return merge(ms)


def _quad_wall(a, b, y0, y1, c):
    (x0, z0), (x1, z1) = a, b
    return Mesh([(x0, y0, z0), (x1, y0, z1), (x1, y1, z1), (x0, y1, z0)], [(0, 2, 1), (0, 3, 2)], c)


def extrude_c(ring, y0, y1, colfn, holes=(), top_c=None):
    ms = [walls_c(ring, y0, y1, colfn)] + [walls_c(h, y0, y1, colfn, hole=True) for h in holes]
    if top_c is not None:
        ms.append(cap(ring, y1, top_c, holes))
    return merge(ms)


def band(ring, d, y0, y1, c, holes=(), top=True):
    """A cornice or string course: the ring grown by d, from y0 to y1."""
    hs = [offset(h, -d) for h in holes]
    ms = [walls(offset(ring, d), y0, y1, c)] + [walls(h, y0, y1, c, hole=True) for h in hs]
    if top:
        ms.append(cap(offset(ring, d), y1, c, hs))
    ms.append(cap(offset(ring, d), y0, c, hs, up=False))
    return merge(ms)


# ── Oriented pieces on any wall ─────────────────────────────────────────────
def orient(m, o, u, n):
    """Local x along u, y up, local z along the outward normal n (all in the x/z plane)."""
    u3, n3 = np.array([u[0], 0, u[1]]), np.array([n[0], 0, n[1]])
    V = m.v
    W = np.array([o[0], 0, o[1]]) + V[:, :1] * u3 + V[:, 1:2] * np.array([0, 1, 0]) + V[:, 2:3] * n3
    f = m.f if (u[0] * n[1] - u[1] * n[0]) > 0 else m.f[:, ::-1]
    return Mesh(W, f, m.c)


def edges(ring, hole=False):
    """[(a, u, n, length)] of a ring, n pointing out of the solid."""
    P = ring if (ring_area(ring) > 0) != hole else ring[::-1]
    out = []
    for i in range(len(P)):
        a, b = np.array(P[i]), np.array(P[(i + 1) % len(P)])
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        u = (b - a) / L
        out.append((a, u, np.array([u[1], -u[0]]), L))
    return out


def LB(t0, t1, y0, y1, d0, d1, c):
    """A box in edge-local coordinates (t along the wall, d out of it)."""
    return B(t0, t1, y0, y1, d0, d1, c)


def figure(h, c):
    prof = [(0.17 * h, 0), (0.15 * h, 0.3 * h), (0.11 * h, 0.55 * h), (0.1 * h, 0.72 * h), (0.055 * h, 0.79 * h), (0, 0.8 * h)]
    head = lathe([(0, -0.08 * h), (0.07 * h, -0.05 * h), (0.075 * h, 0), (0.07 * h, 0.05 * h), (0, 0.08 * h)], c, 8).T(0, 0.88 * h, 0)
    return merge([lathe(prof, c, 10), head])


def vase(h, c):
    return lathe([(0.12 * h, 0), (0.2 * h, 0.1 * h), (0.12 * h, 0.25 * h), (0.32 * h, 0.55 * h), (0.25 * h, 0.85 * h),
                  (0.3 * h, 0.95 * h), (0, 1.0 * h)], c, 10)


def arch_face(w, ys, c, seg=12):
    """Flat round-headed opening in local x/y (normal +z), centred on x = 0, from y = 0 up to ys + w/2."""
    r = w / 2
    V = [(-r, 0, 0), (r, 0, 0)] + [(r * math.cos(a), ys + r * math.sin(a), 0) for a in np.linspace(0, math.pi, seg + 1)]
    F = [(0, 1, 2)] + [(0, 2 + i, 3 + i) for i in range(seg)]
    return Mesh(V, F, c)


# ── Columns ──────────────────────────────────────────────────────────────────
def doric(x, z, y0, y1, r):
    """Unfluted Doric (Tuscan) column, base at y0, top of the abacus at y1."""
    return [cyl(r * 1.22, r * 1.15, 0.38, STONE, 18).T(x, y0, z),
            cyl(r, r * 0.86, y1 - y0 - 1.05, jit(STONE, 0.01), 18, caps=False).T(x, y0 + 0.38, z),
            cyl(r * 0.88, r * 1.2, 0.32, STONE, 18).T(x, y1 - 0.67, z),
            B(x - r * 1.3, x + r * 1.3, y1 - 0.35, y1, z - r * 1.3, z + r * 1.3, STONE)]


def ionic(x, z, y0, y1, r, corinthian=False, axis="z"):
    """Ionic (or Corinthian: taller bell capital) column, volutes facing ±z (axis 'z') or ±x."""
    hc = 0.8 if corinthian else 0.45
    out = [cyl(r * 1.3, r * 1.18, 0.28, STONE, 14).T(x, y0, z),
           cyl(r, r * 0.88, y1 - y0 - 0.28 - hc, jit(STONE, 0.01), 14, caps=False).T(x, y0 + 0.28, z)]
    if corinthian:
        out.append(cyl(r * 0.9, r * 1.35, hc - 0.15, STONE, 14).T(x, y1 - hc, z))
    else:
        out.append(cyl(r * 0.95, r * 1.12, hc - 0.15, STONE, 14).T(x, y1 - hc, z))
        for s in (-1, 1):
            v = cyl(r * 0.33, r * 0.33, r * 2.6, STONE, 8)
            if axis == "z":
                out.append(v.R(math.pi / 2, 0, 0).T(x + s * r * 1.05, y1 - 0.38, z - r * 1.3))
            else:
                out.append(v.R(0, 0, math.pi / 2).T(x + r * 1.3, y1 - 0.38, z + s * r * 1.05))
    out.append(B(x - r * 1.45, x + r * 1.45, y1 - 0.15, y1, z - r * 1.45, z + r * 1.45, STONE))
    return out


# ── Roofs ────────────────────────────────────────────────────────────────────
def hip(x0, x1, z0, z1, y0, y1, c, along="x"):
    """Hipped roof, ridge along x (or z)."""
    if along == "x":
        d, zc = (z1 - z0) / 2, (z0 + z1) / 2
        r0, r1 = x0 + d, x1 - d
        V = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (r0, y1, zc), (r1, y1, zc)]
    else:
        d, xc = (x1 - x0) / 2, (x0 + x1) / 2
        r0, r1 = z0 + d, z1 - d
        V = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (xc, y1, r0), (xc, y1, r1)]
    if along == "x":
        F = [(0, 1, 5), (0, 5, 4), (3, 4, 5), (3, 5, 2), (0, 4, 3), (1, 2, 5)]
    else:
        F = [(1, 2, 5), (1, 5, 4), (0, 4, 5), (0, 5, 3), (0, 1, 4), (3, 5, 2)]
    m = Mesh(V, F, c)
    return _fix_up(m)


def _fix_up(m):
    """Turn faces so their normals point up/out (for hand-made roofs)."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    n = np.cross(b - a, c - a)
    ctr = m.v.mean(0)
    out = ((a + b + c) / 3 - ctr)
    out[:, 1] = np.maximum(out[:, 1], 0) + 0.01
    flip = np.einsum("ij,ij->i", n, out) < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def lantern(x0, x1, z0, z1, y0, y1, along="x"):
    """A glazed lantern: low glass walls and a glass gable on top."""
    if along == "x":
        body = B(x0, x1, y0, y0 + (y1 - y0) * 0.45, z0, z1, GLASS_C)
        roof = hip(x0, x1, z0, z1, y0 + (y1 - y0) * 0.45, y1, GLASS_C)
    else:
        body = B(x0, x1, y0, y0 + (y1 - y0) * 0.45, z0, z1, GLASS_C)
        roof = hip(x0, x1, z0, z1, y0 + (y1 - y0) * 0.45, y1, GLASS_C, along="z")
    return [body, roof]


def ring_roof(o, i, y0, y1, y2, glass_sides=(), lead_c=ROOF):
    """Hipped ring roof round a rectangular well: outer rect o (x0, x1, z0, z1) at y0, ridge halfway at y1,
    inner rect i at y2. glass_sides: which inner slopes are glazed ('x0', 'x1', 'z0', 'z1')."""
    r = ((o[0] + i[0]) / 2, (o[1] + i[1]) / 2, (o[2] + i[2]) / 2, (o[3] + i[3]) / 2)
    def corners(q, y):
        return [(q[0], y, q[2]), (q[1], y, q[2]), (q[1], y, q[3]), (q[0], y, q[3])]
    O, R, I = corners(o, y0), corners(r, y1), corners(i, y2)
    names = ("z0", "x1", "z1", "x0")
    lead, glass = [], []
    for k in range(4):
        j = (k + 1) % 4
        lead.append(_fix_up(Mesh([O[k], O[j], R[j], R[k]], [(0, 1, 2), (0, 2, 3)], lead_c)))
        m = Mesh([R[k], R[j], I[j], I[k]], [(0, 1, 2), (0, 2, 3)], GLASS_C if names[k] in glass_sides else lead_c)
        a, b, c = m.v[0], m.v[1], m.v[2]
        if np.cross(b - a, c - a)[1] < 0:
            m.f = m.f[:, ::-1]
        (glass if names[k] in glass_sides else lead).append(m)
    return merge(lead), merge(glass)


def sawtooth(x0, x1, z0, z1, y0, y1, n):
    """n parallel gables running along z between x0 and x1; the south (+x) slopes glazed."""
    lead, glass = [], []
    w = (x1 - x0) / n
    for k in range(n):
        a, b = x0 + k * w, x0 + (k + 1) * w
        m = (a + b) / 2
        lead.append(Mesh([(a, y0, z0), (a, y0, z1), (m, y1, z1), (m, y1, z0)], [(0, 1, 2), (0, 2, 3)], ROOF))
        glass.append(Mesh([(m, y1, z0), (m, y1, z1), (b, y0, z1), (b, y0, z0)], [(0, 1, 2), (0, 2, 3)], GLASS_C))
        for z, s in ((z0, -1), (z1, 1)):
            t = Mesh([(a, y0, z), (b, y0, z), (m, y1, z)], [(0, 1, 2)], BRICK * 0.95)
            if s < 0:
                t.f = t.f[:, ::-1]
            lead.append(t)
    return [_fix_up(m) for m in lead], [_fix_up(m) for m in glass]


# ── The building ─────────────────────────────────────────────────────────────
def body():
    parts, glass = [], []
    U = upper_ring()
    col_low = lambda x, z: jit(STONE_W, 0.01) if west_part(x, z) else jit(GRANITE, 0.015)        # noqa: E731
    col_up = lambda x, z: jit(STONE_W, 0.01) if west_part(x, z) else jit(BRICK, 0.025)           # noqa: E731
    parts.append(extrude_c(MAIN, FOUND, G1, col_low, holes=[COURT], top_c=STONE_W * 0.97))
    parts.append(extrude_c(U, G1, 11.95, col_up, holes=[COURT]))
    parts.append(extrude_c(MAIN, 11.95, TOP, col_up, holes=[COURT]))
    parts.append(cap(MAIN, 11.95, STONE, [COURT], up=False))                                      # loggia ceilings
    # string course, two-step cornice, blocking course
    parts.append(band(MAIN, 0.14, G1 - 0.3, G1 + 0.05, jit(GRANITE), holes=[COURT]))
    parts.append(band(MAIN, 0.3, 13.85, 14.15, jit(STONE), holes=[COURT], top=False))
    parts.append(band(MAIN, 0.75, 14.15, 14.8, STONE, holes=[COURT], top=False))
    parts.append(cap(offset(MAIN, 0.75), 14.8, ROOF_FLAT, [offset(COURT, -0.75)]))
    o, i = offset(MAIN, 0.7), offset(MAIN, 0.35)
    parts += [walls(o, 14.75, 15.25, STONE), walls(i, 14.75, 15.25, STONE, hole=True), cap(o, 15.25, STONE, [i])]
    return parts, glass


def west_galleries():
    out = []
    for x0, x1 in (GAL_N, GAL_S):
        n = 17
        p = (x1 - x0) / n
        z_a, z_b = zf(x0 + 0.1), zf(x1 - 0.1)
        for k in range(n):
            xc = x0 + (k + 0.5) * p
            z = zf(xc)
            if k % 2 == 1:                                                          # arched opening with its grille
                out.append(arch_face(2.2, 4.2, DARK).T(xc, 0.25, z + 0.04))
                arc = [(xc + 1.27 * math.cos(a), 4.45 + 1.27 * math.sin(a), z + 0.06) for a in np.linspace(0, math.pi, 13)]
                out.append(tube(arc, 0.1, STONE, 6))
                for s in (-1, 1):
                    out.append(B(xc + s * 1.1 - 0.12, xc + s * 1.1 + 0.12, 0.2, 4.45, z, z + 0.12, STONE))
                for gx in np.linspace(xc - 0.85, xc + 0.85, 6):
                    out.append(B(gx - 0.03, gx + 0.03, 0.3, 4.3, z + 0.09, z + 0.13, IRON))
            else:                                                                   # niche, statue or vase, medallion
                out.append(B(xc - 0.82, xc + 0.82, 0.8, 3.65, z, z + 0.04, NICHE))
                out.append(B(xc - 0.55, xc + 0.55, 0.8, 1.15, z, z + 0.45, STONE))
                if k % 4 == 2:
                    out.append(vase(1.4, MARBLE).T(xc, 1.15, z + 0.3))
                else:
                    out.append(figure(2.2, MARBLE).S(1, 1, 0.75).T(xc, 1.15, z + 0.3))
                out.append(B(xc - 0.8, xc + 0.8, 4.1, 5.7, z, z + 0.05, BRICK))
                out.append(cyl(0.62, 0.62, 0.12, MARBLE, 16).R(math.pi / 2, 0, 0).T(xc, 4.9, z + 0.17))
            # the upper wall: window with its frame and railing
            zu = z - LOGGIA
            out.append(B(xc - 0.95, xc + 0.95, 7.0, 11.4, zu, zu + 0.05, STONE))
            out.append(B(xc - 0.72, xc + 0.72, 7.1, 11.0, zu + 0.01, zu + 0.09, DARK))
            out.append(B(xc - 0.85, xc + 0.85, 6.85, 7.75, zu + 0.3, zu + 0.36, IRON))
            # the frieze: small windows and brick panels
            if k % 2:
                out.append(B(xc - 0.55, xc + 0.55, 12.75, 13.35, z, z + 0.05, DARK))
            else:
                out.append(B(xc - 0.75, xc + 0.75, 12.75, 13.35, z, z + 0.04, BRICK))
        # stylobate of the colonnade and the columns over the piers
        out.append(B(x0, x1, G1 - 0.02, 6.85, min(z_a, z_b) - 1.4, max(z_a, z_b) + 0.02, jit(STONE)))
        for k in range(1, n):
            x = x0 + k * p
            out += ionic(x, zf(x) - 0.7, 6.85, 11.95, 0.33)
        for x in (x0 + 0.35, x1 - 0.35):                                            # end pilasters
            out.append(B(x - 0.35, x + 0.35, 6.85, 11.95, zf(x) - 1.0, zf(x) - 0.4, STONE))
        out.append(B(x0, x1, 11.9, 12.6, min(z_a, z_b) - 1.25, max(z_a, z_b) + 0.06, jit(STONE)))    # architrave
        # modillions under the cornice
        for x in np.arange(x0 + 0.45, x1 - 0.3, 0.9):
            out.append(B(x - 0.1, x + 0.1, 14.0, 14.16, zf(x) + 0.3, zf(x) + 0.7, STONE))
    return out


def portico():
    out = []
    x0, x1 = PORT
    zb, zfr = 21.64, 27.25
    out.append(B(x0 - 0.2, x1 + 0.2, FOUND, 0.18, zb - 0.2, zfr + 0.5, jit(GRANITE)))           # steps
    out.append(B(x0 - 0.1, x1 + 0.1, 0.18, 0.36, zb - 0.2, zfr + 0.15, jit(GRANITE, 0.01)))
    for dx in (-8.3, -5.1, -1.9, 1.9, 5.1, 8.3):
        out += doric(XC + dx, 26.2, 0.36, 10.0, 0.7)
        out.append(B(XC + dx - 0.6, XC + dx + 0.6, 0.36, 9.7, zb, zb + 0.3, STONE_W))          # pilasters on the back wall
    for x in (x0 + 0.65, x1 - 0.65):                                                           # antae at the back corners
        out.append(B(x - 0.65, x + 0.65, 0.36, 10.0, zb - 0.05, zb + 1.3, jit(STONE_W)))
    out.append(B(x0, x1, 10.0, 11.4, zb - 0.1, 26.95, jit(STONE)))                              # architrave
    out.append(B(x0, x1, 11.4, 12.6, zb - 0.1, 26.9, jit(STONE_W)))                             # frieze
    for x in np.arange(x0 + 0.6, x1, 1.25):                                                     # triglyphs
        out.append(B(x - 0.3, x + 0.3, 11.45, 12.55, 26.9, 27.0, STONE))
    out.append(B(x0 - 0.3, x1 + 0.3, 12.6, 13.4, zb - 0.1, 27.5, jit(STONE)))                   # cornice
    # the back wall: door, brick panels, windows
    out.append(B(XC - 1.6, XC + 1.6, 0.36, 5.9, zb, zb + 0.07, DARK))
    out.append(B(XC - 2.0, XC + 2.0, 5.9, 6.4, zb, zb + 0.12, STONE))
    for dx in (-6.7, 6.7):
        out.append(B(XC + dx - 0.75, XC + dx + 0.75, 1.0, 4.6, zb, zb + 0.05, NICHE))
        out.append(B(XC + dx - 0.75, XC + dx + 0.75, 5.6, 7.9, zb, zb + 0.05, BRICK))
        out.append(B(XC + dx - 0.5, XC + dx + 0.5, 5.85, 7.65, zb + 0.01, zb + 0.08, STONE_W))
    # the attic: side blocks, the relief block, its top slab
    out.append(B(x0 + 0.1, x1 - 0.1, 13.4, 13.8, 14.6, 27.0, jit(STONE)))
    for a, b in ((x0 + 0.4, XC - 5.6), (XC + 5.6, x1 - 0.4)):
        out.append(B(a, b, 13.8, 16.2, 14.6, 26.6, jit(STONE_W)))
        out.append(B(a - 0.25, b + 0.25, 16.2, 16.6, 14.4, 26.85, STONE))
        out.append(B((a + b) / 2 - 0.5, (a + b) / 2 + 0.5, 14.3, 15.4, 26.6, 26.66, BRICK))
    out.append(B(XC - 5.6, XC + 5.6, 13.8, 17.7, 14.6, 26.8, jit(STONE_W)))
    out.append(B(XC - 5.9, XC + 5.9, 17.7, 18.1, 14.3, 27.1, STONE))
    out.append(B(XC - 5.35, XC + 5.35, 18.1, 18.4, 15.0, 26.6, jit(STONE)))
    out.append(B(XC - 4.85, XC + 4.85, 14.2, 16.95, 26.8, 26.9, MARBLE))                        # Barba's relief
    for k, dx in enumerate(np.linspace(-4.3, 4.3, 15)):
        h = 2.25 if k % 4 != 1 else 1.5
        out.append(figure(h, MARBLE).S(1, 1, 0.4).T(XC + dx, 14.4, 26.95))
    # the flag on its pole
    out.append(cyl(0.06, 0.05, 6.0, IRON, 8).T(XC, 18.4, 21.0))
    for k, c in enumerate(((0.75, 0.1, 0.1), (0.95, 0.75, 0.1), (0.95, 0.75, 0.1), (0.75, 0.1, 0.1))):
        out.append(B(XC + 0.06, XC + 2.0, 23.2 - k * 0.32, 23.52 - k * 0.32, 20.97, 21.03, c))
    return out


def window_wall(a, u, n, L, rows, pitch, skip=(), frame=GRANITE, brick=False):
    """Windows along one side: rows = [(y0, y1, w)], spaced by pitch, none inside the skip t-ranges."""
    out = []
    k = max(1, int(L // pitch))
    off = (L - k * pitch) / 2 + pitch / 2
    for j in range(k):
        t = off + j * pitch
        if any(s0 - 1.0 < t < s1 + 1.0 for s0, s1 in skip):
            continue
        for y0, y1, w in rows:
            if frame is not None:
                out.append(orient(LB(t - w / 2 - 0.22, t + w / 2 + 0.22, y0 - 0.25, y1 + 0.3, -0.05, 0.05, frame), a, u, n))
            out.append(orient(LB(t - w / 2, t + w / 2, y0, y1, -0.04, 0.09, DARK), a, u, n))
    return out


def pilasters(a, u, n, L, every, y0, y1, c, w=0.9):
    out = []
    ts = [0.45, L - 0.45]
    if every:
        k = max(1, int(round(L / every)))
        ts += [L * j / k for j in range(1, k)]
    for t in ts:
        out.append(orient(LB(t - w / 2, t + w / 2, y0, y1, -0.06, 0.12, jit(c)), a, u, n))
    return out


def brick_fronts():
    """Pavilions and the east side: granite ground storey, brick above with granite pilasters, windows."""
    out = []
    for a, u, n, L in edges(MAIN):
        mid = a + u * L / 2
        if west_part(mid[0], mid[1]) or L < 3.0:
            continue
        skip = []
        if abs(mid[0] + 98.1) < 0.6 and abs(n[0] + 1) < 0.05:          # Goya front: the portico and the podium
            ta, tb = sorted(((-5.5 - a[1]) / u[1], (19.3 - a[1]) / u[1]))
            skip = [(ta, tb)]                                          # the Goya blocks carry their own windows
        if abs(mid[0] - 103.3) < 0.5:                                  # Murillo front: door and loggia
            t_c = (6.9 - a[1]) / u[1]
            skip = [(t_c - 8.5, t_c + 8.5)]
        rows = [(1.8, 4.4, 1.3), (7.6, 11.6, 1.5)]
        if abs(mid[0] + 98.1) < 0.6:
            rows = [(7.6, 11.6, 1.5)]                                  # the podium hides the ground storey
        out += window_wall(a, u, n, L, rows, 4.2, skip)
        out += pilasters(a, u, n, L, 12.0 if L > 20 else 0, G1, TOP, GRANITE)
    for a, u, n, L in edges(COURT, hole=True):
        if L > 3:
            out += window_wall(a, u, n, L, [(1.8, 4.4, 1.2), (7.6, 11.4, 1.4)], 3.6)
    return out


def goya():
    """North front: the podium with its two stairs, the Ionic portico in antis, the two taller blocks."""
    out = []
    zc = ROT_C[1]
    xf = -98.05
    pod = [(-97.6, -12.23), (-103.56, -12.23), (-103.57, 26.06), (-97.6, 26.06)]
    out.append(merge([walls(pod, FOUND, 5.5, jit(GRANITE)), cap(pod, 5.5, STONE_W * 0.95)]))
    for z in np.linspace(-11.5, 25.3, 5):                                      # lower doors / windows
        out.append(B(-103.63, -103.5, 0.3, 3.6, z - 1.0, z + 1.0, DARK))
    # balustrade round the terrace
    for (xa, za), (xb, zb) in (((-103.4, -12.0), (-103.4, 25.85)), ((-103.4, -12.0), (-98.2, -12.0)), ((-103.4, 25.85), (-98.2, 25.85))):
        out.append(tube([(xa, 6.45, za), (xb, 6.45, zb)], 0.09, STONE, 6))
        L = math.hypot(xb - xa, zb - za)
        for t in np.arange(0.2, L, 0.45):
            px, pz = xa + (xb - xa) * t / L, za + (zb - za) * t / L
            out.append(cyl(0.07, 0.07, 0.95, STONE, 6).T(px, 5.5, pz))
    # the two stairs (landings of the OSM outline): steps rising towards the façade's middle
    for z_out, z_in in ((-12.22, -6.79), (26.07, 19.93)):
        nst = 11
        for k in range(nst):
            za, zb = z_out + (z_in - z_out) * k / nst, z_out + (z_in - z_out) * (k + 1) / nst
            out.append(B(-107.28, -103.5, FOUND, (k + 1) * 5.5 / nst, min(za, zb), max(za, zb), jit(GRANITE, 0.01)))
    # portico in antis: two Ionic columns, antae, entablature, the dark loggia behind
    out.append(B(xf - 0.04, xf + 0.05, 5.5, 12.6, zc - 2.7, zc + 2.7, DARK))
    for dz in (-1.15, 1.15):
        out += ionic(xf - 1.0, zc + dz, 5.5, 12.6, 0.42, axis="x")
    for dz in (-3.15, 3.15):
        out.append(B(xf - 1.6, xf + 0.1, 5.5, 12.6, zc + dz - 0.45, zc + dz + 0.45, jit(STONE_W)))
    out.append(B(xf - 1.8, xf + 0.1, 12.6, 13.95, zc - 3.75, zc + 3.75, jit(STONE)))
    # the two taller blocks of the Goya front
    for z0, z1 in ((-5.5, zc - 3.6), (zc + 3.6, 19.3)):
        ring = [(xf, z0), (xf + 8.0, z0), (xf + 8.0, z1), (xf, z1)]
        out.append(walls(ring, 14.7, 18.3, jit(BRICK)))
        out.append(B(xf - 0.25, xf + 8.25, 18.3, 18.75, z0 - 0.25, z1 + 0.25, STONE))
        out.append(hip(xf + 0.1, xf + 7.9, z0 + 0.1, z1 - 0.1, 18.7, 20.4, ROOF, along="z"))
        for z in (z0 + 0.4, z1 - 0.4):
            out.append(B(xf - 0.1, xf + 0.6, 14.7, 18.3, z - 0.45, z + 0.45, jit(GRANITE)))
        zm = (z0 + z1) / 2
        out.append(B(xf - 0.08, xf + 0.1, 15.4, 17.6, zm - 0.8, zm + 0.8, DARK))
        out.append(B(xf - 0.1, xf + 0.1, 8.8, 12.9, zm - 3.2, zm + 3.2, jit(BRICK)))           # brick panel with window
        out.append(B(xf - 0.13, xf + 0.08, 8.0, 12.2, zm - 0.85, zm + 0.85, DARK))
    return out


def murillo():
    """South front: lintelled door and the loggia of six Corinthian columns."""
    out = []
    zc, xf = 6.9, 103.31
    out.append(B(xf - 0.05, xf + 0.08, 0.0, 4.6, zc - 1.6, zc + 1.6, DARK))
    out.append(B(xf - 0.05, xf + 0.2, 4.6, 5.2, zc - 2.1, zc + 2.1, STONE))
    for dz in (-1.85, 1.85):
        out.append(B(xf - 0.05, xf + 0.2, 0.0, 4.6, zc + dz - 0.25, zc + dz + 0.25, STONE))
    out.append(B(xf - 0.05, xf + 0.05, 6.9, 12.7, zc - 7.6, zc + 7.6, DARK))
    out.append(B(xf - 0.1, xf + 1.0, G1 - 0.1, 6.95, zc - 8.2, zc + 8.2, jit(STONE)))
    for k in range(6):
        out += ionic(xf + 0.5, zc + (k - 2.5) * 2.95, 6.95, 12.7, 0.36, corinthian=True, axis="x")
    for dz in (-7.95, 7.95):
        out.append(B(xf - 0.1, xf + 0.95, 6.95, 12.7, zc + dz - 0.5, zc + dz + 0.5, jit(STONE_W)))
    out.append(B(xf - 0.1, xf + 1.1, 12.7, 13.9, zc - 8.4, zc + 8.4, jit(STONE)))
    return out


def roofs():
    lead, glass = [], []
    # the long west galleries: low hipped roofs with a glazed lantern along the ridge
    for x0, x1 in ((-53.6, -7.4), (13.6, 58.4)):
        lead.append(hip(x0, x1, 1.8, 16.3, 14.75, 17.3, ROOF))
        glass += lantern(x0 + 1.5, x1 - 1.5, 7.6, 10.6, 16.7, 18.2)
    # the middle ranges behind them
    for x0, x1 in ((-55.0, -14.4), (20.2, 59.8)):
        lead.append(hip(x0, x1, -15.3, 1.6, 14.75, 17.0, ROOF))
        glass += lantern(x0 + 3.0, x1 - 3.0, -8.0, -5.4, 16.4, 17.6)
    # the eastern ranges (Arbós, Chueca): saw-tooth skylight roofs
    for x0, x1 in ((-55.0, -14.6), (20.3, 59.8)):
        lt, gl = sawtooth(x0, x1, -31.5, -15.6, 14.75, 16.7, 9)
        lead += lt; glass += gl
    # the basilica (central hall) and the apse
    hall = [(-5.6, -21.0), (11.6, -21.0), (11.6, 14.4), (-5.6, 14.4)]
    lead.append(merge([walls_c(hall, 14.7, 17.4, lambda x, z: jit(BRICK)), cap(hall, 17.4, ROOF_FLAT)]))
    lead.append(band(hall, 0.4, 17.4, 17.8, STONE))
    lead.append(hip(-5.4, 11.4, -20.8, 14.2, 17.75, 20.1, ROOF, along="z"))
    glass += lantern(1.5, 4.5, -18.0, 11.5, 19.4, 20.8, along="z")
    arc = [(12.5, -21.11)] + OUTLINE[57:69] + [(-6.5, -21.13)]
    apex = np.array([3.0, 17.2, -21.1])
    V = [apex] + [(x, 14.78, z) for x, z in offset_arc(arc, 0.7)]
    F = [(0, i + 1, i + 2) for i in range(len(V) - 2)]
    lead.append(_fix_up(Mesh(V, F, ROOF)))
    gl = [apex + np.array([0, 0.05, 0])] + [tuple(apex + 0.45 * (np.array([x, 14.78, z]) - apex) + np.array([0, 0.05, 0])) for x, z in offset_arc(arc, 0.7)]
    glass.append(_fix_up(Mesh(gl, F, GLASS_C)))
    # end pavilions: hipped ring roofs, glazed on the inside
    lt, gl = ring_roof((-97.6, -53.7, -15.0, 28.85), (ROT_C[0] - 10.8, ROT_C[0] + 10.8, ROT_C[1] - 10.8, ROT_C[1] + 10.8),
                       14.75, 17.4, 16.2, glass_sides=("z0", "z1"))
    lead.append(lt); glass.append(gl)
    sq = (ROT_C[0] - 10.8, ROT_C[0] + 10.8, ROT_C[1] - 10.8, ROT_C[1] + 10.8)
    lead.append(cap([(sq[0], sq[2]), (sq[1], sq[2]), (sq[1], sq[3]), (sq[0], sq[3])], 16.2, ROOF_FLAT))
    cx, cz = ROT_C
    lead.append(cyl(ROT_R, ROT_R, 1.9, jit(BRICK), 40, caps=False).T(cx, 16.0, cz))
    lead.append(cyl(ROT_R + 0.35, ROT_R + 0.35, 0.4, STONE, 40, caps=False).T(cx, 17.9, cz))
    lead.append(lathe([(ROT_R + 0.35, 18.3), (ROT_R + 0.35, 17.9), (ROT_R, 18.3), (ROT_R * 0.8, 19.3), (ROT_R * 0.5, 20.0),
                       (3.2, 20.25)][1:], ROOF, 40).T(cx, 0, cz))
    glass.append(cyl(3.2, 2.2, 0.9, GLASS_C, 24).T(cx, 20.2, cz))
    lt, gl = ring_roof((59.1, 102.95, -15.4, 28.95), (76.2, 86.1, 0.1, 13.35), 14.75, 17.3, 14.78,
                       glass_sides=("x0", "x1", "z0", "z1"))
    lead.append(lt); glass.append(gl)
    return lead, glass


def offset_arc(arc, d):
    """Push the open polyline of the apse out by d, away from its centre."""
    c = np.array([3.0, -21.1])
    out = []
    for p in arc:
        v = np.array(p) - c
        ln = np.linalg.norm(v) or 1
        out.append(tuple(np.array(p) + v / ln * d))
    return out


def main():
    out = ROOT / "models" / "Q160112"
    out.mkdir(parents=True, exist_ok=True)
    stone, glass = body()
    stone += west_galleries() + portico() + brick_fronts() + goya() + murillo()
    lead, gl2 = roofs()
    m_stone = merge(stone)
    m_roof = merge(lead)
    m_glass = merge(glass + gl2)
    tris, size = write_glb(out / "model.glb", [("museo_del_prado", m_stone), ("roofs", m_roof, {"material": "zinc"}),
                                               ("skylights", m_glass, {"material": "glass"})],
                           roughness=0.9, materials={"glass": GLASS, "zinc": {"metallic": 0.3, "roughness": 0.6}})
    allv = np.vstack([m_stone.v, m_roof.v, m_glass.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
