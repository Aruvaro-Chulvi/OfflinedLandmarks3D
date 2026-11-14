"""
Palacio de Cibeles (Q1849031), the former Palacio de Comunicaciones, Plaza de Cibeles, Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Antonio Palacios (1874–1945) and Joaquín Otamendi (1874–1960), 1904–1919; sculpture by Ángel
García Díaz. Glass roof over the old courtyard by Schlaich Bergermann und Partner (2008–2009) during the
conversion into the city hall and CentroCentro (Arquimática, 2003–2011). Still protected (Otamendi died in 1960).

Measurements (see meta.json for the sources):
- OpenStreetMap way 16322055: outline (11 789 m², 133 × 119 m), and its building:part ways with levels; the
  main façade is the cut corner that faces the Plaza de Cibeles, 50 m long between the two corner towers,
  facing 308.5° (from the outline; it looks straight at the fountain). Origin: centroid of the outline.
- Heights, measured on frontal photos (Wikimedia Commons «Fachada desde Plaza-Palacio de Comunicaciones.JPG»,
  «Palacio de Cibeles, Madrid, España.jpg») scaled by the 55 m between the corner towers (±5 %): main cornice
  27 m, attic around the tower 33.5 m, corner towers 36 m with their pinnacle crowns at about 45 m, central
  tower base 38 m, its shaft to 55 m, crown and finial to about 70 m (es.wikipedia: «torreón central de
  setenta metros»). OSM levels map to heights: 5 → 24 m, 6 → 27, 7 → 30.5, 8 → 33.5, 9 → 38.
- Roofs from the PNOA orthophoto (IGN) and photos from the mirador: grey slate/zinc mansards behind white
  balustrades, the big glass vault over the old courtyard (OSM part 331693280, about 3 000 m²), three glass
  lanterns round the tower and a glass gable; the rear range along the old Pasaje de Alarcón (no OSM part)
  is a 24 m block 19.6 m deep.
White Novelda stone (biocalcarenite), dark slate, glass.

Axes (docs/MODEL_SPEC.md): +Y up, the main façade (to the Plaza de Cibeles) faces +Z, origin on the ground at
the centroid of the outline.

    python model_sources/Q1849031_palacio_de_cibeles.py   -> models/Q1849031/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import (GLASS, METAL, Mesh, merge, box, cyl, lathe, walls, cap, extrude, offset,  # noqa: E402
                     triangulate, ring_area, write_glb)

rng = np.random.default_rng(1919)

STONE = np.array((0.92, 0.90, 0.85))      # Novelda stone
STONE_D = np.array((0.84, 0.82, 0.77))
SLATE = (0.40, 0.43, 0.48)
ROOF_FLAT = (0.42, 0.43, 0.45)
GLASS_C = (0.50, 0.62, 0.70)
WIN = (0.19, 0.21, 0.25)
CLOCK = (0.95, 0.94, 0.90)
FOUND = -3.0
BASE = 18.0                               # everything rises to this; glass courts sit on it

# ── OpenStreetMap data (© OpenStreetMap contributors, ODbL), in model x/z for heading 308.5 ──────────
# way 16322055: the outline
OUTLINE = [(22.67, 41.94), (7.00, 42.19), (3.02, 42.25), (2.94, 43.03), (-7.65, 43.43), (-7.66, 42.41),
    (-11.72, 42.31), (-27.45, 41.92), (-25.92, 45.65), (-27.84, 48.60), (-31.59, 48.70), (-32.09, 49.94),
    (-39.34, 54.08), (-45.59, 48.51), (-44.96, 47.00), (-60.96, 33.08), (-61.79, 32.35), (-67.64, 27.27),
    (-73.48, 22.19), (-78.01, 18.24), (-79.02, 19.10), (-85.41, 13.22), (-84.51, 12.51), (-88.53, 8.99),
    (-82.94, 4.39), (-80.22, 2.14), (-73.07, -3.76), (-69.81, 0.32), (-61.95, -5.97), (-65.36, -10.12),
    (-48.87, -23.71), (-37.27, -33.27), (-34.41, -35.63), (-23.55, -44.59), (-17.37, -49.69), (-11.94, -54.16),
    (-9.76, -55.96), (-8.82, -56.73), (-5.51, -59.46), (0.88, -64.72), (4.37, -67.60), (5.30, -68.37),
    (13.93, -75.48), (25.19, -59.31), (29.66, -52.89), (34.25, -46.29), (49.12, -24.94), (66.73, 0.33),
    (70.75, 6.11), (75.52, 12.97), (80.19, 19.66), (77.46, 26.94), (70.71, 31.14), (69.64, 29.47), (40.76, 47.47),
    (41.66, 49.51), (34.70, 53.84), (27.58, 50.10), (26.90, 48.42), (23.07, 48.51), (20.97, 45.25)]
# its building:part ways: (id, levels, roof, outline). Parts of fewer than 5 levels that are not glass
# roofs stay inside the 18 m base and are left out.
PARTS = [
    (331666154, 7, "", [(6.18, 31.59), (7.83, 31.59), (7.83, 32.57), (7.84, 36.26), (-12.60, 36.34),
        (-12.57, 32.16), (-12.58, 31.69), (-10.60, 31.65), (-10.54, 32.22), (-8.43, 34.46), (3.81, 34.42),
        (6.19, 31.93)]),
    (331666156, 8, "", [(0.37, 31.61), (6.18, 31.59), (6.21, 19.90), (4.05, 17.65), (-8.67, 17.78),
        (-10.74, 19.82), (-10.60, 31.65), (-5.14, 31.62), (-8.15, 28.61), (-8.12, 23.20), (-5.08, 20.25),
        (0.73, 20.16), (3.60, 23.23), (3.49, 28.78)]),
    (331666158, 5, "", [(40.76, 47.47), (38.33, 41.42), (37.95, 41.33), (37.73, 37.52), (42.38, 34.57),
        (43.40, 36.16), (50.38, 31.80), (49.33, 30.23), (53.92, 27.31), (54.94, 28.73), (60.59, 25.18),
        (59.67, 23.75), (64.06, 21.01), (69.64, 29.47)]),
    (331666159, 6, "", [(17.00, 32.21), (16.97, 39.13), (7.00, 39.35), (7.00, 42.19), (3.02, 42.26), (2.94, 43.03),
        (-7.65, 43.43), (-7.66, 42.41), (-11.72, 42.31), (-11.66, 39.68), (-21.74, 39.40), (-21.74, 32.10),
        (-12.57, 32.16), (-12.60, 36.34), (7.84, 36.26), (7.83, 32.57)]),
    (331666163, 6, "", [(59.00, 13.26), (55.53, 7.96), (66.73, 0.33), (70.74, 6.12), (65.27, 9.62), (70.08, 16.45),
        (75.52, 12.98), (80.19, 19.66), (77.46, 26.94), (70.71, 31.14), (69.64, 29.47), (64.06, 21.01),
        (59.67, 23.75), (53.92, 27.31), (49.33, 30.23), (42.38, 34.57), (37.73, 37.52), (37.95, 41.33),
        (38.33, 41.42), (40.76, 47.47), (41.65, 49.51), (34.70, 53.85), (27.58, 50.10), (26.90, 48.42),
        (28.81, 44.87), (27.02, 41.97), (29.66, 41.80), (29.82, 31.55), (35.31, 28.11), (53.54, 16.67)]),
    (331666170, 6, "", [(75.52, 12.98), (70.08, 16.45), (65.27, 9.62), (70.74, 6.12)]),
    (331666172, 4, "glass_gabled", [(35.31, 28.11), (25.20, 11.76), (42.49, 0.03), (53.54, 16.67)]),
    (331666174, 7, "", [(27.02, 41.97), (22.66, 41.94), (20.97, 45.25), (23.06, 48.51), (26.90, 48.42),
        (28.81, 44.87)]),
    (331666175, 4, "glass_mansard", [(4.37, 15.66), (4.24, 10.49), (4.18, 7.93), (0.36, 4.42), (-0.50, 4.33),
        (-0.52, 3.80), (-4.96, 3.86), (-9.45, 7.82), (-9.47, 10.52), (-9.52, 15.72)]),
    (331666177, 4, "glass_mansard", [(25.35, 28.46), (25.26, 23.52), (21.69, 20.40), (21.29, 20.04), (7.88, 19.98),
        (7.83, 31.59), (7.83, 32.57), (17.00, 32.21), (21.59, 32.03)]),
    (331666179, 4, "glass_mansard", [(-12.57, 32.16), (-12.58, 31.69), (-12.56, 19.84), (-21.43, 19.83),
        (-26.43, 19.82), (-30.19, 23.10), (-30.21, 28.67), (-26.75, 32.07), (-21.74, 32.10)]),
    (331666181, 8, "", [(6.18, 31.59), (0.37, 31.61), (-5.14, 31.62), (-10.60, 31.65), (-10.54, 32.22),
        (-8.43, 34.46), (3.81, 34.42), (6.19, 31.93)]),
    (331666184, 9, "", [(0.37, 31.61), (3.49, 28.78), (3.60, 23.23), (0.73, 20.16), (-5.08, 20.25), (-8.12, 23.20),
        (-8.15, 28.61), (-5.14, 31.62)]),
    (331692042, 5, "", [(25.00, 11.52), (20.64, 4.46), (20.67, 10.53), (13.32, 10.53), (12.31, 10.53),
        (12.41, 15.14), (7.83, 15.18), (7.83, 15.64), (7.88, 19.98), (21.29, 20.04), (21.69, 20.40),
        (25.15, 14.61)]),
    (331692045, 5, "", [(25.20, 11.76), (25.00, 11.52), (25.15, 14.61), (21.69, 20.40), (25.26, 23.52),
        (25.35, 28.46), (21.59, 32.03), (17.00, 32.21), (16.97, 39.13), (7.00, 39.35), (7.00, 42.19),
        (22.66, 41.94), (27.02, 41.97), (29.66, 41.80), (29.82, 31.55), (35.31, 28.11)]),
    (331693266, 5, "", [(-26.75, 32.07), (-26.81, 33.07), (-24.92, 33.15), (-24.98, 38.86), (-30.93, 38.85),
        (-30.97, 41.79), (-27.45, 41.93), (-11.72, 42.31), (-11.66, 39.68), (-21.74, 39.40), (-21.74, 32.10)]),
    (331693267, 8, "", [(43.66, -21.17), (40.06, -18.71), (40.35, -18.29), (44.05, -13.12), (20.54, 3.10),
        (20.64, 4.46), (25.00, 11.52), (25.20, 11.76), (42.49, 0.03), (47.84, -3.52), (43.81, -9.58),
        (51.60, -14.96), (49.87, -17.48), (47.47, -15.79)]),
    (331693268, 5, "", [(-4.96, 3.86), (-9.45, 7.82), (-9.47, 10.52), (-18.36, 10.45), (-18.38, 5.27),
        (-7.12, -4.11), (4.02, -3.94), (13.30, 5.68), (13.32, 10.53), (12.31, 10.53), (4.24, 10.49), (4.18, 7.93),
        (0.36, 4.42), (-0.50, 4.33), (-0.52, 3.80)]),
    (331693269, 5, "", [(-21.24, 10.45), (-21.38, 15.28), (-21.43, 19.83), (-26.43, 19.82), (-29.55, 14.90),
        (-35.13, 19.37), (-35.14, 21.74), (-37.90, 22.10), (-45.14, 28.24), (-38.21, 34.33), (-35.23, 34.37),
        (-35.30, 41.76), (-42.48, 41.56), (-43.40, 43.53), (-58.33, 29.40), (-51.04, 23.47), (-25.03, 2.14),
        (-24.88, 10.46)]),
    (331693270, 6, "", [(-30.97, 41.79), (-31.64, 41.81), (-33.75, 45.29), (-31.59, 48.71), (-32.10, 49.94),
        (-39.35, 54.08), (-45.59, 48.51), (-44.96, 47.00), (-43.40, 43.53), (-42.48, 41.56), (-35.30, 41.76),
        (-35.23, 34.37), (-35.14, 21.74), (-35.13, 19.37), (-29.55, 14.90), (-26.43, 19.82), (-30.19, 23.10),
        (-30.21, 28.67), (-26.75, 32.07), (-26.81, 33.07), (-24.92, 33.15), (-24.98, 38.86), (-30.93, 38.85)]),
    (331693271, 5, "", [(-58.33, 29.40), (-61.79, 32.36), (-44.96, 47.00), (-43.40, 43.53)]),
    (331693272, 7, "", [(25.18, -39.87), (20.97, -45.86), (16.53, -52.20), (25.18, -59.31), (29.65, -52.89),
        (34.25, -46.29)]),
    (331693273, 8, "", [(50.74, -18.09), (49.87, -17.48), (47.47, -15.79), (43.66, -21.17), (46.94, -23.48)]),
    (331693274, 5, "", [(-12.50, 15.73), (-12.49, 15.21), (-21.38, 15.28), (-21.43, 19.83), (-12.56, 19.84)]),
    (331693276, 5, "", [(53.54, 16.67), (42.49, 0.03), (47.84, -3.52), (55.53, 7.96), (59.00, 13.26)]),
    (331693277, 7, "", [(40.06, -18.71), (43.66, -21.17), (46.94, -23.48), (49.12, -24.94), (34.25, -46.29),
        (25.18, -39.87)]),
    (331693278, 7, "", [(-27.45, 41.93), (-25.92, 45.65), (-27.84, 48.61), (-31.59, 48.71), (-33.75, 45.29),
        (-31.64, 41.81), (-30.97, 41.79)]),
    (331693279, 7, "", [(-21.38, 15.28), (-21.24, 10.45), (-18.36, 10.45), (-9.47, 10.52), (-9.52, 15.72),
        (-12.50, 15.73), (-12.49, 15.21)]),
    (331693280, 4, "glass_round", [(13.32, 10.53), (13.30, 5.68), (4.02, -3.94), (-7.12, -4.11), (-18.38, 5.27),
        (-18.36, 10.45), (-21.24, 10.45), (-24.88, 10.46), (-25.03, 2.14), (-51.04, 23.47), (-55.87, 17.63),
        (-60.81, 11.66), (-52.84, 5.08), (16.53, -52.20), (20.97, -45.86), (25.18, -39.87), (40.06, -18.71),
        (40.35, -18.29), (44.05, -13.12), (20.54, 3.10), (20.64, 4.46), (20.67, 10.53)]),
    (331693281, 11, "", [(-0.59, 22.06), (-3.83, 21.99), (-6.17, 24.24), (-6.20, 25.87), (-6.21, 27.49),
        (-3.96, 29.84), (-2.36, 29.87), (-0.71, 29.90), (1.62, 27.65), (1.64, 26.11), (1.67, 24.40)]),
    (331693282, 7, "", [(55.53, 7.96), (47.84, -3.52), (43.81, -9.58), (51.60, -14.96), (49.87, -17.48),
        (50.74, -18.09), (46.94, -23.48), (49.12, -24.94), (66.73, 0.33)]),
    (464810792, 6, "", [(-72.55, 13.65), (-72.97, 14.02), (-78.01, 18.24), (-79.02, 19.10), (-85.41, 13.23),
        (-84.51, 12.52), (-79.58, 8.36), (-77.88, 6.94)]),
    (464810793, 5, "", [(-69.48, 18.77), (-72.34, 16.28), (-71.52, 15.47), (-72.97, 14.02), (-78.01, 18.24),
        (-73.48, 22.20)]),
    (464810794, 5, "", [(-60.81, 11.66), (-69.48, 18.77), (-73.48, 22.20), (-67.64, 27.28), (-61.79, 32.36),
        (-58.33, 29.40), (-51.04, 23.47), (-55.87, 17.63)]),
    (464810795, 5, "", [(-61.96, -5.97), (-69.81, 0.32), (-73.07, -3.76), (-80.22, 2.15), (-82.94, 4.39),
        (-79.58, 8.36), (-77.88, 6.94), (-72.55, 13.65), (-72.97, 14.02), (-71.52, 15.47), (-72.34, 16.28),
        (-69.48, 18.77), (-60.81, 11.66), (-52.84, 5.08)]),
    (464810797, 7, "", [(-34.34, -28.03), (-37.21, -25.75), (-32.91, -20.34), (-30.11, -22.71)]),
    (464810798, 8, "", [(-22.90, -37.45), (-23.71, -36.78), (-27.53, -33.63), (-23.59, -28.57), (-18.88, -32.35)]),
    (965320611, 7, "glass_hipped", [(-41.70, -22.20), (-56.82, -9.52), (-51.71, -3.38), (-36.45, -15.87)]),
]
# pinnacles mapped as tiny building:part ways: (id, levels, x, z)
PINNACLES = [
    (444486820, 12, -3.69, 22.75), (444486821, 8, 23.03, 42.56), (444486822, 7, 37.83, 42.05), (444486823, 8, 21.72, 45.23),
    (444486824, 7, -39.26, 53.33), (444486825, 8, 28.08, 44.88), (444486826, 8, 23.42, 47.88), (444486827, 12, 0.87, 27.49),
    (444486829, 8, 26.52, 47.85), (444486830, 7, 29.36, 42.54), (444486831, 7, -44.89, 48.34), (444486832, 7, 34.73, 53.02),
    (444486833, 12, -5.52, 24.64), (444486834, 7, -34.74, 42.34), (444486835, 12, -0.75, 22.81), (444486837, 7, 40.65, 49.22),
    (444486839, 12, -3.58, 29.27), (444486840, 12, 1.04, 24.8), (444486841, 12, -1.25, 29.38), (444486842, 8, 26.68, 42.61),
    (444486843, 7, -42.14, 42.16), (444486844, 12, -5.61, 27.26), (444486845, 7, -32.73, 49.44), (444486846, 7, 28.17, 49.58),
    (965165353, 8, -28.03, 42.68), (965165354, 8, -31.2, 42.53), (965165355, 8, -32.9, 45.25), (965165356, 8, -31.2, 47.98),
    (965165357, 8, -28.21, 47.86), (965165358, 8, -26.76, 45.54),
]

HT = {5: 24.0, 6: 27.0, 7: 30.5, 8: 33.5, 9: 38.0}
TOWERS = {331693278: 36.0, 331666174: 36.0}                     # the two corner towers (their crowns in OSM)
TBASE, SHAFT, LANTERN = 331666156, 331666184, 331693281          # central tower: base block, shaft, lantern
HOVR = {**TOWERS, TBASE: 41.0}
FLAT = {0, TBASE, 331666181, 331666154} | set(TOWERS)           # the attic round the central tower, its base
# the rear range on the old Pasaje de Alarcón: no building:part in OSM; it fills the 19.6 m between the straight
# back edge and the glass vault
ALARCON = [(-65.36, -10.12), (13.93, -75.48), (25.17, -59.34), (-52.88, 5.00)]
PORTICO = (-8.0, 3.3)                                           # x span of the central portico on the front


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


def seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def edges(P):
    """(a, b, unit direction, outward normal, length) for a counter-clockwise ring."""
    out = []
    for i in range(len(P)):
        a, b = np.array(P[i]), np.array(P[(i + 1) % len(P)])
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        u = (b - a) / L
        out.append((a, b, u, np.array([u[1], -u[0]]), L))
    return out


# ── Things on a wall ─────────────────────────────────────────────────────────
def on_wall(poly_sy, p0, u, n, c, eps=0.05):
    """A flat shape drawn on a wall: poly_sy = [(s, y)] along the wall direction u from p0, pushed eps along n."""
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
    """A box along a wall: from s0 to s1, d_out in front of it and d_in behind (cornices, parapets, bands)."""
    q = [np.array(p0) + u * s0 + n * d_out, np.array(p0) + u * s1 + n * d_out,
         np.array(p0) + u * s1 - n * d_in, np.array(p0) + u * s0 - n * d_in]
    return extrude([tuple(x) for x in q], y0, y1, c, bottom=True)


ROWS = [(1.6, 6.2, "arch"), (7.8, 11.0, ""), (12.4, 15.2, ""), (16.6, 19.2, ""), (20.6, 22.8, ""),
        (24.2, 25.8, ""), (27.6, 29.0, ""), (30.6, 32.0, ""), (33.4, 34.9, ""), (36.4, 38.4, "")]


def pinnacle(x, z, y0, h, c, w=0.75):
    """A pinnacle: a square shaft, a little capital and a needle with a ball (the original telegraph posts)."""
    out = [box(w, h * 0.62, w, c).T(x, y0, z),
           box(w * 1.35, h * 0.07, w * 1.35, c).T(x, y0 + h * 0.62, z),
           cyl(w * 0.55, 0.0, h * 0.31, c, 6).T(x, y0 + h * 0.69, z)]
    return out


def statue(x, z, y0, h, c):
    prof = [(0.17 * h, 0), (0.15 * h, 0.35 * h), (0.11 * h, 0.6 * h), (0.12 * h, 0.75 * h), (0.05 * h, 0.82 * h), (0, 0.83 * h)]
    head = lathe([(0.0, -0.07 * h), (0.07 * h, 0.0), (0.0, 0.07 * h)], c, 8)
    return [lathe(prof, c, 10).T(x, y0, z), head.T(x, y0 + 0.9 * h, z)]


# ── Roofs ────────────────────────────────────────────────────────────────────
def shrink_ok(P, d):
    Q = offset(P, -d)
    if ring_area(Q) <= 0.15 * ring_area(ccw(P)):
        return None
    if not all(in_poly(x, z, P) for x, z in Q):
        return None
    for (a, b, u, n, L), (a2, b2, u2, n2, L2) in zip(edges(ccw(P)), edges(Q)):
        if np.dot(u, u2) < 0:                                   # an edge turned round: the offset folded
            return None
    return Q


def band(lo, hi, ylo, yhi, c):
    """Sloped band between two rings with the same number of points (a mansard, a frustum)."""
    v, f = [], []
    n = len(lo)
    for i in range(n):
        j = (i + 1) % n
        o = len(v)
        v += [(lo[i][0], ylo, lo[i][1]), (lo[j][0], ylo, lo[j][1]), (hi[j][0], yhi, hi[j][1]), (hi[i][0], yhi, hi[i][1])]
        f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    return Mesh(v, f, c)


def mansard(P, H, c=SLATE, inset=0.45, rise=1.0, dmax=3.4):
    lo = shrink_ok(P, inset)
    if lo is None:
        return []
    for d in (dmax, 2.4, 1.6, 1.0):
        hi = shrink_ok(P, inset + d)
        if hi is not None:
            return [band(lo, hi, H, H + d * rise, jit(c)), cap(hi, H + d * rise, jit(c))]
    return []


def subdivide(v, f, k):
    for _ in range(k):
        V = list(map(tuple, v)); idx = {p: i for i, p in enumerate(V)}; F = []

        def mid(a, b):
            p = tuple((np.array(V[a]) + np.array(V[b])) / 2)
            if p not in idx:
                idx[p] = len(V); V.append(p)
            return idx[p]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        v, f = np.array(V), np.array(F)
    return v, f


def vault(P, y0, rise, reach=4.5, k=3):
    """The glass vault over the old courtyard: a cushion rising with the distance from its edge."""
    P = ccw(P)
    pts, tris = triangulate(P)
    v, f = subdivide(pts, tris, k)
    E = edges(P)
    d = np.array([min(seg_dist(p, a, b) for a, b, *_ in E) for p in v])
    y = y0 + rise * (1 - np.exp(-d / reach))
    m = Mesh(np.stack([v[:, 0], y, v[:, 1]], 1), f[:, ::-1], GLASS_C)
    return m, [beam(P, y0)]


def beam(P, y0):
    """The stone ring beam round a glass roof."""
    return extrude(offset(P, 0.15), y0, y0 + 0.45, jit(STONE_D), holes=[offset(P, -0.25)], bottom=True)


def glass_roof(pid, shape, P, y0):
    P = ccw(P)
    if shape == "glass_round":
        return vault(P, y0, 5.0)
    if shape in ("glass_mansard", "glass_hipped"):
        lo = shrink_ok(P, 0.3) or P
        for d in (3.0, 2.2, 1.5, 1.0, 0.6):
            hi = shrink_ok(P, 0.3 + d)
            if hi is not None:
                break
        rise = 3.0 if shape == "glass_mansard" else 2.4
        if hi is None or len(hi) != len(lo):
            return cap(P, y0 + 0.4, GLASS_C), [beam(P, y0)]
        return merge([band(lo, hi, y0, y0 + rise, GLASS_C), cap(hi, y0 + rise, GLASS_C)]), \
            [beam(P, y0)]
    if shape == "glass_gabled":                                # a ridge along the long sides
        E = edges(P)
        if E[0][4] + E[2][4] < E[1][4] + E[3][4]:
            P = P[1:] + P[:1]
        a, b, c, dd = [np.array(p) for p in P]
        m1, m2 = (b + c) / 2, (dd + a) / 2
        r1, r2 = m1 + (m2 - m1) * 0.0, m2
        Y = y0 + 3.0
        V = [(a[0], y0, a[1]), (b[0], y0, b[1]), (r1[0], Y, r1[1]), (r2[0], Y, r2[1]),
             (c[0], y0, c[1]), (dd[0], y0, dd[1])]
        F = [(0, 2, 1), (0, 3, 2), (4, 3, 5), (4, 2, 3), (1, 2, 4), (5, 3, 0)]
        return Mesh(V, F, GLASS_C), [beam(P, y0)]
    return None, []


# ── The building ─────────────────────────────────────────────────────────────
def palace():
    stone, roofs, glass, dark = [], [], [], []
    O = ccw(OUTLINE)
    parts = [(pid, lv, roof, ccw(P)) for pid, lv, roof, P in PARTS] + [(0, 5, "", ccw(ALARCON))]
    solid = []                                               # (id, height, ring) of the parts that rise above BASE
    for pid, lv, roof, P in parts:
        if roof.startswith("glass") or pid in (SHAFT, LANTERN):
            continue
        H = HOVR.get(pid, HT.get(lv))
        if H is None or H <= BASE:
            continue
        solid.append((pid, H, offset(P, 1.0) if pid in TOWERS else P))     # the towers stand out of the walls

    # the base and its plinth
    stone.append(extrude(O, FOUND, BASE, jit(STONE), top_c=ROOF_FLAT))
    stone.append(walls(offset(O, 0.15), FOUND, 1.1, jit(STONE_D)))

    # parts: walls, roofs
    for pid, H, P in solid:
        stone.append(walls(P, FOUND if pid in TOWERS else BASE, H, jit(STONE)))
        if pid in FLAT:
            roofs.append(cap(P, H, jit(ROOF_FLAT)))
        else:
            roofs.append(cap(P, H, jit(SLATE)))
            roofs += mansard(P, H)
        stone.append(extrude(offset(P, 0.35), H - 0.55, H - 0.02, jit(STONE), bottom=True))     # cornice
        stone.append(extrude(P, H, H + 1.0, jit(STONE), holes=[offset(P, -0.32)]))               # parapet

    # glass roofs
    for k, (pid, lv, roof, P) in enumerate(p for p in parts if p[2].startswith("glass")):
        if roof.startswith("glass"):
            y0 = 24.0 if pid == 965320611 else BASE
            g, ribs = glass_roof(pid, roof, P, y0 + 0.02)
            for r in ribs:                                     # neighbouring courts share an edge: stagger the beams
                r.v[:, 1] = np.where(r.v[:, 1] > y0 + 0.3, r.v[:, 1] + 0.04 * k, r.v[:, 1])
            if g is not None:
                glass.append(g); stone += ribs

    # façades: windows, bands and pinnacles along the outline, at the height of the part behind each bay
    def local_h(p):
        best, bh = 1.2, BASE
        for pid, H, P in solid:
            for a, b, u, n, L in edges(P):
                d = seg_dist(p, a, b)
                if d < best:
                    best, bh = d, H
        return bh

    for ei, (a, b, u, n, L) in enumerate(edges(O)):
        if L < 2.6:
            continue
        nb = max(1, int(L / 3.7))
        bay = L / nb
        front = n[1] > 0.6
        for i in range(nb):
            s = (i + 0.5) * bay
            p = a + u * s
            if front and PORTICO[0] < p[0] < PORTICO[1] and p[1] > 40:
                continue
            H = local_h(p)
            for y0, y1, kind in ROWS:
                if y1 > H - 1.0:
                    break
                w = 1.75 if kind else 1.45
                shp = arch(s - w / 2, s + w / 2, y0, y1 - w / 2) if kind else rect(s - w / 2, s + w / 2, y0, y1)
                dark.append(on_wall(shp, a, u, n, WIN))
                stone.append(slab(a, u, n, s - w / 2 - 0.15, s + w / 2 + 0.15, y0 - 0.25, y0, 0.18, 0.0, jit(STONE)))
            # string course over the ground floor, cornice and parapet when no part stands on this edge
            stone.append(slab(a, u, n, i * bay, (i + 1) * bay, 6.9, 7.3 + 0.03 * (ei % 2), 0.22, 0.0, jit(STONE)))
            if H <= BASE:
                stone.append(slab(a, u, n, i * bay, (i + 1) * bay, BASE - 0.5, BASE, 0.3, 0.0, jit(STONE)))
                stone.append(slab(a, u, n, i * bay, (i + 1) * bay, BASE, BASE + 0.9, 0.0, 0.3, jit(STONE)))
            if front and i % 2 == 1 and H > BASE:
                stone += pinnacle(*(p - n * 0.15), H + 0.9, 3.6, jit(STONE), 0.6)
    # windows on the walls that rise above their neighbours (courtyards, the attic, the back of the towers)
    P_shaft = next(ccw(Q) for i, _, _, Q in PARTS if i == SHAFT)

    def h_at(q):
        if in_poly(q[0], q[1], P_shaft):
            return 58.0
        hs = [H for pid, H, P in solid if in_poly(q[0], q[1], P)]
        return max(hs) if hs else (BASE if in_poly(q[0], q[1], O) else 0.0)

    OE = edges(O)
    for pid, H, P in solid:
        for a, b, u, n, L in edges(P):
            if L < 2.6 or (pid == TBASE and n[1] > 0.9):
                continue
            if pid not in TOWERS and min(seg_dist(a + u * L / 2, a2, b2) for a2, b2, *_ in OE) < 1.2:
                continue
            nb = max(1, int(L / 3.7)); bay = L / nb
            for i in range(nb):
                s = (i + 0.5) * bay
                hq = h_at(a + u * s + n * 1.5)
                for y0, y1, kind in ROWS:
                    if y0 >= hq + 0.6 and y1 <= H - 1.0:
                        w = 1.75 if kind else 1.4
                        shp = arch(s - w / 2, s + w / 2, y0, y1 - w / 2) if kind else rect(s - w / 2, s + w / 2, y0, y1)
                        dark.append(on_wall(shp, a, u, n, WIN))

    # pinnacles mapped in OSM: tower crowns (8 levels), pavilion corners (7), round the central tower (12)
    for pid, lv, x, z in PINNACLES:
        if lv == 12:
            stone += pinnacle(x, z, 58.0, 8.0, jit(STONE), 0.85)
        elif lv == 8:
            stone += pinnacle(x, z, 36.9, 9.0, jit(STONE), 1.0)
        else:
            stone += pinnacle(x, z, 28.0, 6.0, jit(STONE), 0.85)

    # corner towers: windows on their exposed sides and crowns
    for pid in TOWERS:
        P = next(ccw(Q) for i, _, _, Q in PARTS if i == pid)
        for x, z in offset(P, 0.7):                           # pinnacles on the corners of the tower parapet
            stone += pinnacle(x, z, 36.9, 4.6, jit(STONE), 0.7)
        Q = offset(P, -1.6)
        stone.append(extrude(Q, 37.0, 41.0, jit(STONE), top_c=SLATE))
        for a, b, u, n, L in edges(Q):
            if L > 1.6:
                dark.append(on_wall(arch(L / 2 - 0.55, L / 2 + 0.55, 37.8, 40.0 - 0.55), a, u, n, WIN))
            stone.append(slab(a, u, n, 0, L, 41.0, 41.5, 0.25, 0.0, jit(STONE)))
        cx, cz = np.mean(np.array(P), 0)
        stone.append(cyl(1.6, 0.0, 3.2, jit(SLATE), 8).T(cx, 41.5, cz))
        stone += pinnacle(cx, cz, 44.0, 3.2, jit(STONE), 0.4)

    # central tower: base block (big arch and statues), octagonal shaft with windows and clocks, lantern, finial
    Ps = next(ccw(Q) for i, _, _, Q in PARTS if i == SHAFT)
    cx, cz = np.mean(np.array(Ps), 0)
    stone.append(extrude(Ps, 33.4, 58.0, jit(STONE)))          # the octagon rises from the attic, in the base block
    ang = np.linspace(0, 2 * math.pi, 21)[:-1]
    for a, b, u, n, L in edges(Ps):
        stone.append(slab(a, u, n, 0, L, 57.4, 58.0, 0.45, 0.0, jit(STONE)))
        stone.append(slab(a, u, n, 0, L, 41.0, 42.0, 0.3, 0.0, jit(STONE)))
        if n[1] > 0.9:              # the front of the base: a great arched window between two statues
            dark.append(on_wall(arch(L / 2 - 1.8, L / 2 + 1.8, 34.6, 38.6), a, u, n, WIN))
            stone.append(on_wall(arch(L / 2 - 2.4, L / 2 + 2.4, 34.5, 38.6), a, u, n, STONE * 1.02, 0.03))
            for s_ in (-1.6, L + 1.6):
                q = a + u * s_ + n * 0.8
                stone += statue(q[0], q[1], 34.5, 4.2, jit(STONE))
        if L > 3.5:                                                     # the four main faces
            for s in (L / 2 - 1.05, L / 2 + 1.05):
                dark.append(on_wall(arch(s - 0.7, s + 0.7, 45.5, 51.8), a, u, n, WIN))
            # the clock (3 m across) above the windows, with its hands
            stone.append(on_wall([(L / 2 + 1.5 * math.cos(t), 55.2 + 1.5 * math.sin(t)) for t in ang], a, u, n, CLOCK, 0.06))
            dark.append(on_wall(rect(L / 2 - 0.07, L / 2 + 0.07, 55.2, 56.3), a, u, n, WIN, 0.09))
            dark.append(on_wall(rect(L / 2 - 0.05, L / 2 + 0.8, 55.13, 55.27), a, u, n, WIN, 0.09))
    Pl = next(ccw(Q) for i, _, _, Q in PARTS if i == LANTERN)
    stone.append(extrude(Pl, 58.0, 62.5, jit(STONE), top_c=SLATE))
    for a, b, u, n, L in edges(Pl):
        if L > 1.6:
            dark.append(on_wall(arch(L / 2 - 0.45, L / 2 + 0.45, 58.8, 61.6), a, u, n, WIN))
        stone.append(slab(a, u, n, 0, L, 62.5, 63.0, 0.3, 0.0, jit(STONE)))
    stone.append(cyl(3.0, 1.0, 3.0, jit(STONE), 8).T(cx, 63.0, cz))
    stone.append(cyl(1.0, 1.0, 1.4, jit(STONE), 8).T(cx, 66.0, cz))
    stone.append(cyl(1.05, 0.0, 2.6, jit(STONE), 8).T(cx, 67.4, cz))

    # the portico: three doors under a great arched window, in the middle of the main façade
    for a, b, u, n, L in edges(O):
        if n[1] > 0.9 and min(a[0], b[0]) < PORTICO[0] + 1 and max(a[0], b[0]) > PORTICO[1] - 1 and a[1] > 42.5:
            sm = L / 2
            for k in (-1, 0, 1):
                dark.append(on_wall(arch(sm + k * 3.3 - 1.2, sm + k * 3.3 + 1.2, 0.0, 4.6), a, u, n, WIN))
            stone.append(slab(a, u, n, 0.3, L - 0.3, 7.0, 7.8, 0.6, 0.0, jit(STONE)))
            dark.append(on_wall(arch(sm - 3.0, sm + 3.0, 9.0, 17.0), a, u, n, WIN))
            stone.append(on_wall(arch(sm - 3.7, sm + 3.7, 8.6, 17.0), a, u, n, STONE * 1.02, 0.03))
            for k in (-1, 1):                                 # mullions of the big window
                stone.append(slab(a, u, n, sm + k * 1.0 - 0.12, sm + k * 1.0 + 0.12, 9.0, 18.6, 0.09, 0.0, jit(STONE)))
            stone.append(slab(a, u, n, sm - 3.0, sm + 3.0, 13.4, 13.7, 0.09, 0.0, jit(STONE)))
    return stone, roofs, glass, dark


def main():
    out = ROOT / "models" / "Q1849031"
    out.mkdir(parents=True, exist_ok=True)
    stone, roofs, glass, dark = palace()
    S, R, G, D = merge(stone), merge(roofs), merge(glass), merge(dark)
    tris, size = write_glb(out / "model.glb", [("palacio", S), ("roofs", R, {"material": "zinc"}),
                                               ("glass", G, {"material": "glass"}), ("windows", D)],
                           roughness=0.9, materials={"glass": GLASS, "zinc": {"metallic": 0.3, "roughness": 0.55}})
    allv = np.vstack([S.v, R.v, G.v, D.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
