"""
Catedral Primada Santa María de Toledo (Q1123180), Toledo — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: begun 1226 under archbishop Rodrigo Ximénez de Rada and King Ferdinand III (masters Martín and Petrus
Petri, the chevet and double ambulatory), closed 1493; cloister 1389–1425 (Rodrigo Alfonso, Alvar Martínez); west
front from 1418 and the tower from 1425 by Alvar Martínez, its octagon and spire with the three crowns by Hanequin of
Brussels; Mozarabic Chapel for Cardinal Cisneros (c. 1500–1504, Enrique Egas) with the dome of Jorge Manuel
Theotocópuli (1626–1631); sacristy, Sagrario and Ochavo (Nicolás de Vergara the Younger, Juan Bautista Monegro,
the Ochavo dome by J. M. Theotocópuli, late 16th–17th c.); the Transparente by Narciso Tomé (1729–1732); the west
front restored by Eugenio Durango (1775–1780). Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 1605452 (the cathedral, 13 086 m², height 43 m) and its 27 building:parts, in model x/z for
  heading 268.5° and the origin at the crossing (python tools/osm.py relation 1605452 --heading 268.5 --origin
  39.8570768,-4.0235672): the high body (way 1552164151, 37 m: nave 15 m wide from the crossing to the west front,
  transept 55 m long and 15 m wide, the choir with its seven-sided apse), the double ambulatory (way 1552164148,
  18 m, with its buttress piers), the low body (relation 21278404: aisles, west front, north dependencies), the
  cloister (relation 21278403, with its courtyard), the tower (way 660075807, 90 m), the dome of the Mozarabic
  Chapel (way 1552164154, 39 m), the octagonal chapels of San Ildefonso and Santiago (ways 1552164146 and
  1552164147, 23 m), the Ochavo (way 1552164144, 42 m), the Sagrario (way 1552164142, 37 m), the sacristy (way
  1552164138) and the other dependencies (levels -> about 3.5–4 m each).
  The outer walls of the aisles and of the west front give 268.5° (±0.7°) for the axis (the nave walls drawn on the
  orthophoto lean with the parallax of the high roofs).
- Wikipedia (es/en): 120 m long, 59 m wide, five naves, transept and double ambulatory, tower 92 m (a square shaft
  of four bodies and the belfry, an octagonal body with pinnacles and flying buttresses and a spire with three crowns
  of spikes, «a tiara»), the west front with three portals (Infierno, Perdón, Juicio Final) between the tower and the
  unfinished south tower with the Mozarabic Chapel, the Puerta de los Leones (south transept) and the Puerta del Reloj
  (north transept), the cloister to the north.
- Heights from photographs (Wikimedia Commons, see meta.json) scaled by the OSM widths (±10 %): the square shaft of
  the tower to 55 m, its octagon to 72 m, the spire to 87 m and the cross to 92 m; clerestory eaves 32.5 m and ridge
  37 m (OSM); inner aisles 22.5 m, outer aisles and chapels 15.5 m, ambulatory 15.5–21 m; west front 30–35 m with the
  Perdón portal 18 m high; Mozarabic base 21 m, drum to 31 m, dome and lantern to 41.5 m.
- PNOA orthophoto (IGN): the seven bays of the nave with their double flying buttresses, the radial flyers of the
  ambulatory, the slate pyramid over the Transparente's skylight behind the apse, the cloister garden in 4 × 4 beds.
- Terrain: IGN MDT05. The old town falls ~12 m from the north-west corner of the cloister to the south-east end of the
  chevet; y = 0 is the floor of the cathedral and of the cloister, level with the Plaza del Ayuntamiento in front of
  the Puerta del Perdón (ground_at); every wall goes down to 1.5 m below the real ground under it.

Axes (docs/MODEL_SPEC.md): +Y up, the west front (Puerta del Perdón) faces +Z, +X points to 178.5° (south), origin on
the ground at the crossing.

    python model_sources/Q1123180_catedral_de_toledo.py   -> models/Q1123180/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, extrude, cap, walls, triangulate, ring_area, write_glb, METAL  # noqa: E402

rng = np.random.default_rng(1226)

# ── Colours ──────────────────────────────────────────────────────────────────
STONE = np.array((0.80, 0.74, 0.62))       # white limestone of Olihuelas, warm and weathered
STONE_L = (0.89, 0.85, 0.76)               # mouldings, tracery, pinnacles, statues
STONE_D = (0.58, 0.53, 0.45)               # recesses, porches
TOWER_C = np.array((0.70, 0.64, 0.55))     # the tower: darker, more weathered stone
NICHE = (0.38, 0.35, 0.31)
TILE = (0.58, 0.46, 0.38)                  # old Arab tiles of the low-pitched roofs
TILE_D = (0.50, 0.40, 0.34)
SLATE = (0.33, 0.34, 0.37)
LEAD = (0.52, 0.53, 0.55)
GLASS = (0.15, 0.17, 0.25)
ROSE_G = (0.24, 0.18, 0.34)
DARK = (0.16, 0.15, 0.14)
DOOR = (0.30, 0.20, 0.13)
WOOD = (0.36, 0.25, 0.17)
RENDER = (0.90, 0.85, 0.72)                # rendered upper floor of the cloister
IRON = (0.12, 0.12, 0.13)
GOLD = (0.86, 0.69, 0.30)
GRASS = (0.38, 0.46, 0.27)
HEDGE = (0.20, 0.32, 0.17)
GRAVEL = (0.78, 0.72, 0.60)
PAVE = (0.66, 0.62, 0.56)

# ── Levels (m above the floor of the cathedral = the plaza in front of the west front) ──
FOUND = -3.0             # foundations (pushed down to 1.5 m below the real ground under each wall)
OUT_H = 15.5             # outer aisles, side chapels, outer ambulatory
IN_H = 22.5              # walls of the inner aisles
IN_TOP = 26.0            # their lean-to roofs rise to here against the clerestory
EAVE = 32.5              # clerestory eaves
RIDGE = 37.0             # ridge of nave, transept and choir (OSM 37)
NORTH_H = 12.0           # dependencies on the north side
TZ_HW = 7.5              # half width of the transept (z)
NAVE_N = lambda z: -8.4 + (z - 7.5) * (0.8 / 48.2)    # north and south walls of the nave (OSM)
NAVE_S = lambda z: 6.8 + (z - 7.5) * (0.7 / 48.4)
AISLE_N, AISLE_S = -19.7, 18.9                          # wall between the inner and the outer aisle
OUT_N, OUT_S = -31.0, 30.0                              # outer walls of the aisles
BAYS = [7.5 + 7.9 * k for k in range(8)]                # pier lines of the nave (orthophoto: seven bays)
AC = (-1.5, -19.0)                                      # centre of the apse
CHOIR_N, CHOIR_S = -8.6, 6.0
R_MID, R_OUT = 16.5, 25.6                               # ambulatory: inner/outer division, outer wall
TOWER = (-32.0, -18.0, 61.2, 75.2)                      # square shaft of the tower (x0, x1, z0, z1)
MOZ_B = (15.5, 29.8, 59.6, 75.0)                        # base of the unfinished south tower (Mozarabic Chapel)
GROUND_AT = (0.0, 82.0)

# ── OpenStreetMap plans (model x/z, heading 268.5°, origin 39.8570768, -4.0235672; © OpenStreetMap contributors, ODbL) ──
OUTLINE = [  # relation/1605452: relation 1605452 (outer)
    (-59.01, 4.60), (-75.67, 4.42), (-75.69, 5.19), (-79.84, 6.97), (-80.17, 6.96), (-81.29, 7.01),
    (-84.29, 7.19), (-84.63, 7.27), (-84.21, 8.22), (-87.54, 8.39), (-88.10, 8.38), (-89.09, 8.26),
    (-90.20, 8.23), (-90.14, 9.86), (-90.05, 10.98), (-89.72, 15.27), (-92.28, 15.46), (-92.24, 14.00),
    (-95.38, 14.95), (-95.29, 15.72), (-96.88, 16.96), (-96.66, 21.51), (-96.60, 23.48), (-96.31, 29.13),
    (-96.20, 29.31), (-94.53, 29.35), (-91.62, 36.79), (-91.51, 36.88), (-88.72, 40.80), (-86.77, 46.85),
    (-85.71, 48.67), (-85.71, 48.76), (-85.72, 49.01), (-85.97, 54.57), (-85.76, 71.70), (-85.80, 73.24),
    (-83.40, 74.92), (-34.17, 73.99), (-33.50, 74.01), (-31.95, 73.96), (-29.95, 74.01), (-29.50, 73.94),
    (-28.06, 73.89), (-27.99, 75.52), (-26.21, 75.48), (-26.30, 74.54), (-18.63, 74.22), (-18.54, 75.17),
    (-16.98, 75.12), (-16.97, 74.44), (-17.05, 73.58), (-15.50, 73.54), (-15.56, 71.65), (-16.45, 71.71),
    (-16.64, 66.32), (-15.63, 66.30), (-7.75, 66.21), (-7.81, 68.52), (-5.59, 68.57), (-5.55, 66.78),
    (5.89, 66.91), (5.85, 68.53), (7.96, 68.50), (8.01, 66.62), (15.34, 66.64), (15.50, 73.15), (14.05, 73.20),
    (14.02, 74.48), (15.80, 74.44), (15.78, 75.38), (15.77, 75.72), (15.86, 76.24), (17.42, 76.28),
    (17.46, 74.83), (24.56, 74.84), (26.56, 74.89), (26.52, 76.43), (27.96, 76.47), (28.00, 75.02),
    (29.78, 74.98), (29.82, 73.53), (29.94, 56.24), (30.04, 39.46), (31.93, 39.51), (32.05, 35.03),
    (32.17, 30.53), (30.39, 30.48), (30.12, 23.71), (31.90, 23.67), (32.11, 7.16), (32.07, 4.67), (31.97, -0.04),
    (31.90, -1.66), (31.80, -6.55), (31.74, -8.26), (31.66, -9.63), (31.82, -11.34), (26.37, -11.14),
    (26.28, -16.28), (27.95, -16.32), (27.88, -18.03), (31.66, -18.19), (32.09, -25.80), (32.15, -28.19),
    (33.26, -28.25), (26.85, -33.81), (26.64, -34.07), (25.77, -35.04), (25.24, -35.82), (25.24, -35.91),
    (25.49, -36.93), (21.26, -45.00), (21.27, -45.34), (21.17, -45.86), (20.41, -67.79), (19.90, -69.35),
    (19.58, -70.04), (10.07, -67.29), (9.85, -67.21), (6.83, -66.44), (2.02, -65.11), (1.68, -65.11),
    (-10.10, -64.65), (-22.00, -64.02), (-30.35, -63.47), (-31.46, -63.41), (-34.47, -63.15), (-35.60, -62.07),
    (-37.09, -60.65), (-37.31, -60.74), (-38.96, -61.30), (-40.40, -61.42), (-40.84, -61.44), (-42.06, -61.47),
    (-42.51, -61.56), (-53.70, -62.29), (-53.81, -62.29), (-62.35, -63.03), (-63.12, -63.13), (-65.17, -35.88),
    (-65.66, -29.90), (-65.67, -29.47), (-66.38, -19.48), (-66.39, -19.05), (-66.40, -18.62), (-67.26, -6.91),
    (-67.42, -5.21), (-63.20, -5.01), (-57.10, -4.68), (-50.00, -4.41), (-49.89, -4.41), (-49.79, -0.09),
    (-49.68, 4.50)]
HIGH = [  # way/1552164151: nave, transept and choir with the apse, 37 m
    (-28.26, -7.52), (-28.28, -6.66), (-28.24, 0.18), (-28.21, 7.63), (-10.10, 7.51), (-8.44, 7.47),
    (-7.59, 55.68), (-7.42, 62.11), (-7.44, 62.88), (7.67, 62.42), (7.69, 61.73), (7.51, 55.91), (6.78, 7.44),
    (26.23, 7.00), (26.14, 6.32), (26.72, 5.39), (26.94, 5.31), (26.75, -0.00), (26.48, -6.86), (25.93, -7.21),
    (25.94, -7.47), (6.17, -7.39), (5.95, -7.39), (5.98, -17.15), (5.69, -18.70), (5.03, -23.25), (1.00, -26.44),
    (-4.11, -26.40), (-8.07, -23.34), (-8.64, -18.65), (-8.80, -16.94), (-8.49, -7.60)]
GIROLA = [  # way/1552164148: double ambulatory and its buttress piers, 18 m
    (26.28, -16.28), (25.62, -16.30), (25.57, -18.69), (26.01, -18.68), (24.21, -26.09), (24.43, -26.09),
    (25.21, -26.24), (27.89, -26.77), (27.80, -27.71), (23.90, -26.95), (22.62, -29.13), (26.22, -30.75),
    (25.90, -31.44), (22.31, -29.91), (19.86, -33.91), (22.69, -35.98), (22.04, -36.76), (19.33, -34.78),
    (17.82, -36.62), (20.11, -39.30), (19.46, -39.83), (17.06, -37.15), (13.25, -40.33), (15.43, -43.10),
    (14.67, -43.63), (14.44, -43.38), (12.60, -40.95), (10.30, -42.12), (11.51, -45.85), (10.74, -46.05),
    (9.64, -42.48), (4.58, -44.24), (5.79, -48.23), (5.02, -48.42), (3.80, -44.26), (2.38, -45.15),
    (2.58, -44.20), (2.00, -43.28), (1.44, -43.21), (0.88, -43.05), (0.77, -43.14), (-0.10, -43.76),
    (-0.29, -44.88), (0.29, -45.81), (-4.83, -45.51), (-8.95, -49.30), (-9.04, -50.16), (-9.15, -49.99),
    (-9.49, -49.83), (-9.82, -49.84), (-10.04, -49.93), (-11.60, -49.80), (-11.72, -49.72), (-13.34, -43.00),
    (-13.34, -42.91), (-13.23, -42.91), (-13.12, -42.82), (-13.12, -42.73), (-13.01, -42.65), (-13.01, -42.56),
    (-12.91, -42.38), (-12.91, -42.30), (-12.91, -42.21), (-12.91, -42.13), (-12.92, -42.04), (-12.92, -41.87),
    (-12.92, -41.79), (-12.93, -41.70), (-12.93, -41.62), (-13.04, -41.62), (-13.04, -41.53), (-13.15, -41.45),
    (-13.16, -41.36), (-13.27, -41.28), (-13.38, -41.28), (-13.49, -41.20), (-13.61, -41.12), (-13.72, -41.12),
    (-13.94, -41.13), (-14.49, -41.31), (-16.52, -40.17), (-16.53, -39.74), (-16.88, -39.24), (-20.63, -36.00),
    (-23.57, -38.21), (-24.03, -37.54), (-21.42, -35.50), (-23.03, -33.41), (-25.99, -34.68), (-26.23, -33.92),
    (-23.49, -32.73), (-25.50, -28.08), (-28.68, -29.70), (-29.03, -28.94), (-25.96, -27.40), (-27.02, -25.04),
    (-30.02, -25.03), (-30.04, -24.26), (-27.49, -24.28), (-28.76, -18.15), (-27.87, -18.12), (-27.80, -16.41),
    (-27.83, -15.30), (-28.28, -15.31), (-28.26, -7.52), (-8.49, -7.60), (-8.80, -16.94), (-8.64, -18.65),
    (-8.07, -23.34), (-4.11, -26.40), (1.00, -26.44), (5.03, -23.25), (5.69, -18.70), (5.98, -17.15),
    (5.95, -7.39), (6.17, -7.39), (25.94, -7.47), (25.74, -8.33), (26.32, -9.26), (26.43, -9.34), (27.44, -9.48),
    (27.77, -9.30), (31.66, -9.63), (31.82, -11.34), (26.37, -11.14)]
LOW = [  # relation/21278404: aisles, west front and the north dependencies
    (-49.68, 4.50), (-34.13, 4.48), (-33.90, 21.35), (-33.68, 55.34), (-33.65, 64.66), (-33.63, 66.21),
    (-33.50, 74.01), (-31.95, 73.96), (-29.95, 74.01), (-29.50, 73.94), (-28.06, 73.89), (-27.99, 75.52),
    (-26.21, 75.48), (-26.30, 74.54), (-18.63, 74.22), (-18.54, 75.17), (-16.98, 75.12), (-16.97, 74.44),
    (-17.05, 73.58), (-15.50, 73.54), (-15.56, 71.65), (-16.45, 71.71), (-16.64, 66.32), (-15.63, 66.30),
    (-7.75, 66.21), (-7.81, 68.52), (-5.59, 68.57), (-5.55, 66.78), (5.89, 66.91), (5.85, 68.53), (7.96, 68.50),
    (8.01, 66.62), (15.34, 66.64), (15.50, 73.15), (14.05, 73.20), (14.02, 74.48), (15.80, 74.44),
    (15.78, 75.38), (15.77, 75.72), (15.86, 76.24), (17.42, 76.28), (17.46, 74.83), (24.56, 74.84),
    (26.56, 74.89), (26.52, 76.43), (27.96, 76.47), (28.00, 75.02), (29.78, 74.98), (29.82, 73.53),
    (29.94, 56.24), (30.04, 39.46), (31.93, 39.51), (32.05, 35.03), (32.17, 30.53), (30.39, 30.48),
    (30.12, 23.71), (31.90, 23.67), (32.11, 7.16), (32.07, 4.67), (31.97, -0.04), (31.90, -1.66), (31.80, -6.55),
    (31.74, -8.26), (31.66, -9.63), (27.77, -9.30), (28.42, -8.86), (28.61, -7.74), (28.04, -6.82),
    (26.92, -6.59), (26.48, -6.86), (26.75, -0.00), (26.94, 5.31), (27.83, 5.16), (28.82, 5.79), (29.01, 6.91),
    (28.78, 7.24), (28.43, 7.75), (28.43, 7.83), (27.31, 8.06), (26.55, 7.53), (26.33, 7.43), (26.23, 7.00),
    (6.78, 7.44), (7.51, 55.91), (7.69, 61.73), (7.67, 62.42), (-7.44, 62.88), (-7.42, 62.11), (-7.59, 55.68),
    (-8.44, 7.47), (-10.10, 7.51), (-28.21, 7.63), (-28.24, 0.18), (-28.28, -6.66), (-28.26, -7.52),
    (-28.28, -15.31), (-28.95, -15.24), (-29.61, -15.26), (-30.39, -15.19), (-30.84, -15.21), (-30.98, -18.03),
    (-29.54, -18.08), (-28.76, -18.15), (-27.49, -24.28), (-30.04, -24.26), (-30.02, -25.03), (-27.02, -25.04),
    (-25.96, -27.40), (-29.03, -28.94), (-28.68, -29.70), (-25.50, -28.08), (-23.49, -32.73), (-26.23, -33.92),
    (-25.99, -34.68), (-23.03, -33.41), (-21.42, -35.50), (-24.03, -37.54), (-23.57, -38.21), (-20.63, -36.00),
    (-16.88, -39.24), (-16.53, -39.74), (-16.52, -40.17), (-14.49, -41.31), (-13.94, -41.13), (-13.72, -41.12),
    (-13.61, -41.12), (-13.49, -41.20), (-13.38, -41.28), (-13.27, -41.28), (-13.16, -41.36), (-13.15, -41.45),
    (-13.04, -41.53), (-13.04, -41.62), (-12.93, -41.62), (-12.93, -41.70), (-12.92, -41.79), (-12.92, -41.87),
    (-12.92, -42.04), (-12.91, -42.13), (-12.91, -42.21), (-12.91, -42.30), (-12.91, -42.38), (-13.01, -42.56),
    (-13.01, -42.65), (-13.12, -42.73), (-13.12, -42.82), (-13.23, -42.91), (-13.34, -42.91), (-13.34, -43.00),
    (-11.72, -49.72), (-11.60, -49.80), (-10.04, -49.93), (-9.82, -49.84), (-9.49, -49.83), (-9.15, -49.99),
    (-9.04, -50.16), (-9.04, -50.24), (-8.92, -50.50), (-8.91, -50.76), (-9.01, -51.10), (-9.25, -54.96),
    (-5.47, -59.14), (0.20, -59.33), (4.32, -55.63), (4.62, -49.97), (0.96, -45.87), (1.41, -46.03),
    (2.39, -45.32), (2.38, -45.15), (3.80, -44.26), (5.02, -48.42), (5.79, -48.23), (4.58, -44.24),
    (9.64, -42.48), (10.74, -46.05), (10.28, -58.30), (7.27, -57.86), (6.83, -66.44), (2.02, -65.11),
    (1.68, -65.11), (-10.10, -64.65), (-22.00, -64.02), (-30.35, -63.47), (-31.46, -63.41), (-34.47, -63.15),
    (-35.60, -62.07), (-37.09, -60.65), (-37.90, -59.30), (-38.37, -53.92), (-38.38, -53.50), (-38.52, -52.39),
    (-38.87, -47.52), (-39.98, -47.63), (-40.79, -37.89), (-44.57, -37.91), (-55.68, -37.77), (-55.84, -35.80),
    (-56.84, -35.83), (-62.61, -35.90), (-65.17, -35.88), (-65.66, -29.90), (-65.67, -29.47), (-66.38, -19.48),
    (-66.39, -19.05), (-66.40, -18.62), (-67.26, -6.91), (-67.42, -5.21), (-63.20, -5.01), (-57.10, -4.68),
    (-50.00, -4.41), (-49.89, -4.41), (-49.79, -0.09)]
CLOISTER = [  # relation/21278403: cloister ranges
    (-59.01, 4.60), (-58.87, 11.79), (-58.89, 12.65), (-58.92, 13.76), (-58.79, 17.28), (-83.02, 18.10),
    (-83.89, 17.39), (-83.16, 35.98), (-83.09, 46.26), (-84.09, 46.40), (-86.77, 46.85), (-85.71, 48.67),
    (-85.71, 48.76), (-85.72, 49.01), (-85.97, 54.57), (-85.76, 71.70), (-85.80, 73.24), (-83.40, 74.92),
    (-34.17, 73.99), (-33.50, 74.01), (-33.63, 66.21), (-33.65, 64.66), (-33.68, 55.34), (-33.90, 21.35),
    (-34.13, 4.48), (-49.68, 4.50)]
TOWER_OSM = [  # way/660075807: tower, 90 m
    (-15.63, 66.30), (-15.69, 64.20), (-17.18, 64.25), (-17.20, 62.93), (-18.63, 62.95), (-18.63, 63.78),
    (-26.52, 64.00), (-26.52, 63.10), (-28.12, 63.13), (-28.10, 64.53), (-33.65, 64.66), (-33.63, 66.21),
    (-33.50, 74.01), (-31.95, 73.96), (-29.95, 74.01), (-29.50, 73.94), (-28.06, 73.89), (-27.99, 75.52),
    (-26.21, 75.48), (-26.30, 74.54), (-18.63, 74.22), (-18.54, 75.17), (-16.98, 75.12), (-16.97, 74.44),
    (-17.05, 73.58), (-15.50, 73.54), (-15.56, 71.65), (-16.45, 71.71), (-16.64, 66.32)]
MOZ = [  # way/1552164154: dome of the Mozarabic Chapel, 39 m
    (20.04, 73.70), (24.26, 73.72), (27.22, 70.80), (27.22, 66.69), (24.30, 63.71), (20.19, 63.69),
    (19.74, 64.10), (17.23, 66.60), (17.23, 70.71)]
ILDEFONSO = [  # way/1552164146: Chapel of San Ildefonso, 23 m
    (-9.04, -50.16), (-8.95, -49.30), (-4.83, -45.51), (0.29, -45.81), (0.73, -45.88), (0.96, -45.87),
    (4.62, -49.97), (4.32, -55.63), (0.20, -59.33), (-5.47, -59.14), (-9.25, -54.96), (-9.01, -51.10),
    (-8.91, -50.76), (-8.92, -50.50), (-9.04, -50.24)]
SANTIAGO = [  # way/1552164147: Chapel of Santiago (Alvaro de Luna), 23 m
    (-17.96, -40.38), (-14.35, -42.59), (-12.36, -50.50), (-15.89, -55.82), (-22.18, -57.27), (-27.49, -53.81),
    (-28.88, -47.51), (-25.46, -42.12)]
OCHAVO = [  # way/1552164144: Ochavo (Relicario), 42 m
    (-57.53, -5.20), (-53.44, -8.69), (-52.97, -14.07), (-56.52, -18.19), (-61.85, -18.59), (-65.94, -15.01),
    (-66.41, -9.72), (-62.85, -5.60)]
SAGRARIO = [  # way/1552164142: Chapel of the Sagrario, 37 m
    (-49.89, -4.41), (-35.90, -4.21), (-35.97, -5.92), (-36.01, -17.05), (-36.06, -19.37), (-36.50, -19.38),
    (-37.50, -19.32), (-50.06, -19.13), (-49.96, -18.70), (-49.97, -18.45), (-50.00, -4.41)]
SACRISTY = [  # way/1552164138: Sacristy
    (-49.97, -18.45), (-49.96, -18.70), (-50.06, -19.13), (-37.50, -19.32), (-37.48, -24.28), (-37.45, -29.93),
    (-40.45, -29.92), (-41.33, -29.95), (-56.89, -29.84), (-62.66, -29.82), (-64.11, -29.86), (-65.22, -29.89),
    (-65.40, -18.68), (-62.62, -18.78), (-62.05, -19.19), (-56.18, -18.78), (-55.85, -18.34)]
B130 = [  # way/1552164130: 4 levels
    (-89.09, 8.26), (-90.20, 8.23), (-90.14, 9.86), (-90.05, 10.98), (-89.72, 15.27), (-87.05, 15.08),
    (-84.16, 14.81), (-83.89, 17.39), (-83.02, 18.10), (-58.79, 17.28), (-58.92, 13.76), (-58.89, 12.65),
    (-58.87, 11.79), (-59.01, 4.60), (-75.67, 4.42), (-75.69, 5.19), (-79.84, 6.97), (-80.17, 6.96),
    (-81.29, 7.01), (-84.29, 7.19), (-84.63, 7.27), (-84.21, 8.22), (-87.54, 8.39), (-88.10, 8.38)]
B131 = [  # way/1552164131: 3 levels
    (9.85, -67.21), (10.28, -58.30), (10.74, -46.05), (11.51, -45.85), (15.28, -45.84), (21.17, -45.86),
    (20.41, -67.79), (19.90, -69.35), (19.58, -70.04), (10.07, -67.29)]
B132 = [  # way/1552164132: 3 levels
    (-65.17, -35.88), (-62.61, -35.90), (-56.84, -35.83), (-55.84, -35.80), (-55.68, -37.77), (-55.03, -49.65),
    (-54.74, -52.55), (-54.61, -53.24), (-53.83, -61.69), (-53.81, -62.29), (-62.35, -63.03), (-63.12, -63.13)]
B133 = [  # way/1552164133: 2 levels
    (-53.81, -62.29), (-53.83, -61.69), (-54.61, -53.24), (-54.74, -52.55), (-40.30, -52.43), (-39.74, -52.42),
    (-38.52, -52.39), (-38.38, -53.50), (-38.37, -53.92), (-37.90, -59.30), (-37.09, -60.65), (-37.31, -60.74),
    (-38.96, -61.30), (-40.40, -61.42), (-40.84, -61.44), (-42.06, -61.47), (-42.51, -61.56), (-53.70, -62.29)]
B136 = [  # way/1552164136: 1 level
    (21.17, -45.86), (15.28, -45.84), (11.51, -45.85), (10.30, -42.12), (12.60, -40.95), (14.44, -43.38),
    (14.67, -43.63), (15.43, -43.10), (13.25, -40.33), (17.06, -37.15), (19.46, -39.83), (19.50, -41.62),
    (18.97, -42.49), (18.21, -43.11), (18.54, -43.10), (21.21, -43.20), (21.26, -45.00), (21.27, -45.34)]
B137 = [  # way/1552164137: 1 level
    (21.26, -45.00), (21.21, -43.20), (18.54, -43.10), (18.21, -43.11), (18.97, -42.49), (19.50, -41.62),
    (19.46, -39.83), (20.11, -39.30), (17.82, -36.62), (19.33, -34.78), (22.04, -36.76), (22.69, -35.98),
    (19.86, -33.91), (22.31, -29.91), (25.90, -31.44), (26.22, -30.75), (22.62, -29.13), (23.90, -26.95),
    (27.80, -27.71), (27.89, -26.77), (25.21, -26.24), (32.09, -25.80), (32.15, -28.19), (33.26, -28.25),
    (26.85, -33.81), (26.64, -34.07), (25.77, -35.04), (25.24, -35.82), (25.24, -35.91), (25.49, -36.93)]
B139 = [  # way/1552164139: 1 level
    (32.09, -25.80), (25.21, -26.24), (24.43, -26.09), (24.21, -26.09), (26.01, -18.68), (26.23, -18.68),
    (27.79, -18.72), (27.88, -18.03), (31.66, -18.19)]
B150 = [  # way/1552164150: 1 level
    (-91.62, 36.79), (-83.94, 36.05), (-83.16, 35.98), (-83.89, 17.39), (-84.16, 14.81), (-87.05, 15.08),
    (-89.72, 15.27), (-92.28, 15.46), (-92.24, 14.00), (-95.38, 14.95), (-95.29, 15.72), (-96.88, 16.96),
    (-96.66, 21.51), (-96.60, 23.48), (-96.31, 29.13), (-96.20, 29.31), (-94.53, 29.35)]
B405 = [  # relation/21278405: 1 level
    (-86.77, 46.85), (-84.09, 46.40), (-83.09, 46.26), (-83.16, 35.98), (-83.94, 36.05), (-91.62, 36.79),
    (-91.51, 36.88), (-88.72, 40.80)]
T134 = [  # way/1552164134: 
    (6.83, -66.44), (7.27, -57.86), (10.28, -58.30), (9.85, -67.21)]
T135 = [  # way/1552164135: 
    (0.96, -45.87), (0.73, -45.88), (0.29, -45.81), (-0.29, -44.88), (-0.10, -43.76), (0.77, -43.14),
    (0.88, -43.05), (1.44, -43.21), (2.00, -43.28), (2.58, -44.20), (2.38, -45.15), (2.39, -45.32),
    (1.41, -46.03)]
T140 = [  # way/1552164140: 
    (27.88, -18.03), (27.79, -18.72), (26.23, -18.68), (26.01, -18.68), (25.57, -18.69), (25.62, -16.30),
    (26.28, -16.28), (27.95, -16.32)]
T141 = [  # way/1552164141: 
    (-28.28, -15.31), (-27.83, -15.30), (-27.80, -16.41), (-27.87, -18.12), (-28.76, -18.15), (-29.54, -18.08),
    (-30.98, -18.03), (-30.84, -15.21), (-30.39, -15.19), (-29.61, -15.26), (-28.95, -15.24)]
T143 = [  # way/1552164143: 
    (26.48, -6.86), (26.92, -6.59), (28.04, -6.82), (28.61, -7.74), (28.42, -8.86), (27.77, -9.30),
    (27.44, -9.48), (26.43, -9.34), (26.32, -9.26), (25.74, -8.33), (25.94, -7.47), (25.93, -7.21)]
T145 = [  # way/1552164145: 
    (26.23, 7.00), (26.33, 7.43), (26.55, 7.53), (27.31, 8.06), (28.43, 7.83), (28.43, 7.75), (28.78, 7.24),
    (29.01, 6.91), (28.82, 5.79), (27.83, 5.16), (26.94, 5.31), (26.72, 5.39), (26.14, 6.32)]
T152 = [  # way/1552164152: 
    (-86.72, 40.51), (-86.71, 40.34), (-86.82, 40.17), (-86.93, 40.08), (-87.15, 40.16), (-87.27, 40.33),
    (-87.27, 40.50), (-87.17, 40.67), (-86.94, 40.68)]
COURT_OSM = [  # courtyard (inner ring) of relation 1605452
    (-74.21, 54.71), (-74.21, 53.97), (-74.24, 47.43), (-74.28, 44.97), (-74.38, 39.98), (-74.41, 35.81),
    (-74.47, 26.80), (-68.36, 26.70), (-65.81, 26.64), (-60.81, 26.55), (-59.47, 26.50), (-56.33, 26.44),
    (-51.92, 26.36), (-47.21, 26.33), (-44.03, 26.31), (-38.92, 26.18), (-38.81, 35.16), (-38.70, 44.33),
    (-38.57, 55.30), (-38.40, 61.64), (-46.57, 61.91), (-55.69, 62.22), (-65.15, 62.54), (-74.08, 62.84),
    (-74.22, 55.13)]


# ── Terrain (IGN MDT05 in models/Q1123180/terrain.json): foundations follow the real ground ──
HEADING = 268.5


def _terrain():
    p = ROOT / "models" / "Q1123180" / "terrain.json"
    if not p.exists():
        return None
    import base64
    t = json.loads(p.read_text("utf-8"))
    n = t["n"]
    q = np.frombuffer(base64.b64decode(t["h"]), "<i2").reshape(n, n).astype(np.float64)
    h = np.where(q == -32768, np.nan, t["off"] + q / 10)
    a = math.radians(HEADING)
    ez = np.array([math.sin(a), math.cos(a)])
    ex = np.array([math.sin(a - math.pi / 2), math.cos(a - math.pi / 2)])

    def at(x, z):
        e = np.asarray(x) * ex[0] + np.asarray(z) * ez[0]
        nn = np.asarray(x) * ex[1] + np.asarray(z) * ez[1]
        fi = (e - t["x0"]) / t["c"]; fj = (t["y0"] - nn) / t["c"]
        i0 = np.clip(np.floor(fi).astype(int), 0, n - 2); j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
        tx = np.clip(fi - i0, 0, 1); tz = np.clip(fj - j0, 0, 1)
        return (h[j0, i0] * (1 - tx) * (1 - tz) + h[j0, i0 + 1] * tx * (1 - tz)
                + h[j0 + 1, i0] * (1 - tx) * tz + h[j0 + 1, i0 + 1] * tx * tz)
    g0 = float(at(np.array([GROUND_AT[0]]), np.array([GROUND_AT[1]]))[0])
    return lambda x, z: at(x, z) - g0


TERRAIN = _terrain()


def sink(m):
    """Foundations: every vertex at FOUND goes down to 1.5 m below the real ground under it (and never above FOUND)."""
    if TERRAIN is None or not len(m.v):
        return m
    k = m.v[:, 1] <= FOUND + 1e-6
    if k.any():
        g = TERRAIN(m.v[k, 0], m.v[k, 2])
        g = np.where(np.isnan(g), 0.0, g)
        m.v[k, 1] = np.minimum(FOUND, np.floor((g - 1.5) * 2) / 2)
    return m


# ── Small helpers (as in Q2981_notre_dame_de_paris.py) ───────────────────────
def stone(k=1.0, base=STONE):
    return tuple(np.clip(np.asarray(base) * k * (1 + rng.normal(0, 0.015)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sphere(r, c, seg=8, rings=5):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def obox(p, q, y0, y1, w, c):
    """A box along the segment p→q (plan), w thick, from y0 to y1."""
    (x0, z0), (x1, z1) = p, q
    L = math.hypot(x1 - x0, z1 - z0)
    a = math.atan2(-(z1 - z0), x1 - x0)
    return box(L, y1 - y0, w, c).R(0, a, 0).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def ccw(ring):
    return list(ring) if ring_area(ring) > 0 else list(ring)[::-1]


def centroid(ring):
    P = np.asarray(ring, float)
    return float(P[:, 0].mean()), float(P[:, 1].mean())


def ngon(cx, cz, r, n=8, rot=0.0):
    return [(cx + r * math.cos(rot + 2 * math.pi * k / n), cz + r * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def rect(x0, x1, z0, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def fr(p, n):
    """A wall frame at plan point p with outward normal n: (origin, tangent, normal); u runs along the tangent."""
    nn = np.array([n[0], 0.0, n[1]], float)
    nn /= np.linalg.norm(nn)
    return np.array([p[0], 0.0, p[1]], float), np.array([nn[2], 0.0, -nn[0]]), nn


def plane_poly(pts_uv, origin, t, n, c):
    pts, tris = triangulate(pts_uv)
    o = np.asarray(origin, float)
    V = np.array([o + u * np.asarray(t) + np.array([0, v, 0]) for u, v in pts])
    m = Mesh(V, tris, c)
    if np.dot(np.cross(t, [0.0, 1.0, 0.0]), n) < 0:
        m.f = m.f[:, ::-1]
    return m


def poly_on(F, pts, eps, c):
    o, t, n = F
    return plane_poly(pts, o + n * eps, t, n, c)


def rect_on(F, s, y0, w, h, eps, c):
    return poly_on(F, [(s - w / 2, y0), (s + w / 2, y0), (s + w / 2, y0 + h), (s - w / 2, y0 + h)], eps, c)


def lancet_pts(s, y0, w, h, sharp=0.9, seg=5):
    """A pointed (Gothic) arch: two arcs of radius sharp·w meeting at the apex, total height h."""
    R = max(sharp, 0.5) * w
    rise = math.sqrt(max(R * R - (R - w / 2) ** 2, 0.0))
    ys = y0 + h - rise
    aa = math.acos(max(-1.0, min(1.0, (w / 2 - R) / R)))
    pts = [(s - w / 2, y0), (s + w / 2, y0)]
    cr = s + w / 2 - R
    pts += [(cr + R * math.cos(a), ys + R * math.sin(a)) for a in np.linspace(0, math.pi - aa, seg + 1)]
    cl = s - w / 2 + R
    pts += [(cl + R * math.cos(a), ys + R * math.sin(a)) for a in np.linspace(aa, math.pi, seg + 1)[1:]]
    out = []
    for p in pts:
        if not out or math.dist(out[-1], p) > 1e-4:
            out.append(p)
    if math.dist(out[0], out[-1]) < 1e-4:
        out.pop()
    return out


def round_pts(s, y0, w, h, seg=8):
    """A round-headed (semicircular) opening, total height h."""
    r = w / 2
    ys = y0 + h - r
    pts = [(s - r, y0), (s + r, y0)] + [(s + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)]
    return pts


def lancet_on(F, s, y0, w, h, eps, c, sharp=0.9):
    return poly_on(F, lancet_pts(s, y0, w, h, sharp), eps, c)


def disc_on(F, s, y, r, eps, c, seg=20):
    return poly_on(F, [(s + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, seg, endpoint=False)], eps, c)


def ring_on(F, s, y, r0, r1, eps, c, seg=24):
    o, t, n = F
    a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    V, f = [], []
    for r in (r0, r1):
        for ang in a:
            V.append(o + n * eps + t * (s + r * math.cos(ang)) + np.array([0, y + r * math.sin(ang), 0]))
    for i in range(seg):
        j = (i + 1) % seg
        f += [(i, seg + i, seg + j), (i, seg + j, j)]
    m = Mesh(V, f, c)
    tri = m.v[m.f[0]]
    if np.dot(np.cross(tri[1] - tri[0], tri[2] - tri[0]), n) < 0:
        m.f = m.f[:, ::-1]
    return m


def spoke_on(F, s, y, r0, r1, ang, w, eps, c):
    ca, sa = math.cos(ang), math.sin(ang)
    px, py = -sa * w / 2, ca * w / 2
    pts = [(s + ca * r0 + px, y + sa * r0 + py), (s + ca * r0 - px, y + sa * r0 - py),
           (s + ca * r1 - px, y + sa * r1 - py), (s + ca * r1 + px, y + sa * r1 + py)]
    return poly_on(F, pts, eps, c)


def rose(F, s, y, r, eps, spokes=16):
    """A rose window: glass, stone rim, inner ring, radiating mullions."""
    ms = [disc_on(F, s, y, r, eps, ROSE_G, seg=24), ring_on(F, s, y, r, r * 1.14, eps + 0.03, STONE_L, seg=24),
          ring_on(F, s, y, r * 0.24, r * 0.32, eps + 0.04, STONE_L, seg=12),
          ring_on(F, s, y, r * 0.64, r * 0.7, eps + 0.04, STONE_L, seg=20)]
    for k in range(spokes):
        a = 2 * math.pi * k / spokes
        ms.append(spoke_on(F, s, y, r * 0.32, r, a, max(0.1, r * 0.04), eps + 0.05, STONE_L))
    return merge(ms)


def slab(F, pts_uv, thick, c, off=0.0):
    """A solid plate: a polygon drawn in a wall plane (u, height), thick along the frame's normal."""
    o, t, n = F
    pts, tris = triangulate(pts_uv)
    k = len(pts)

    def P(u, v, w):
        return o + t * u + np.array([0, v, 0]) + n * w

    V = [P(u, v, off + thick / 2) for u, v in pts] + [P(u, v, off - thick / 2) for u, v in pts]
    f = [tuple(tr) for tr in tris] + [(a + k, c_ + k, b + k) for a, b, c_ in tris]
    m = Mesh(V, f, c)
    if np.dot(np.cross(m.v[f[0][1]] - m.v[f[0][0]], m.v[f[0][2]] - m.v[f[0][0]]), n) < 0:
        m.f = m.f[:, ::-1]
    R = ccw(pts.tolist())
    rim = []
    for i in range(len(R)):
        (u0, v0), (u1, v1) = R[i], R[(i + 1) % len(R)]
        a, b = P(u0, v0, off + thick / 2), P(u1, v1, off + thick / 2)
        cc, d = P(u1, v1, off - thick / 2), P(u0, v0, off - thick / 2)
        rim.append(Mesh([a, b, cc, d], [(0, 2, 1), (0, 3, 2)], c))
    return merge([m] + rim)


def along(p, q):
    """Frame for a slab drawn along the plan segment p→q (u from p towards q)."""
    d = np.array(q, float) - np.array(p, float)
    L = float(np.linalg.norm(d))
    F = fr(p, (-d[1], d[0]))
    if np.dot(F[1], [d[0], 0, d[1]]) < 0:
        F = fr(p, (d[1], -d[0]))
    return F, L


def hip(x0, x1, z0, z1, y0, y1, c):
    """Hipped roof over a rectangle."""
    dx, dz = x1 - x0, z1 - z0
    if dx >= dz:
        h = dz / 2
        a, b = (x0 + h, y1, (z0 + z1) / 2), (x1 - h, y1, (z0 + z1) / 2)
        V = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), a, b]
        f = [(0, 4, 5), (0, 5, 1), (1, 5, 2), (2, 5, 4), (2, 4, 3), (3, 4, 0)]
    else:
        h = dx / 2
        a, b = ((x0 + x1) / 2, y1, z0 + h), ((x0 + x1) / 2, y1, z1 - h)
        V = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), a, b]
        f = [(0, 4, 1), (1, 4, 5), (1, 5, 2), (2, 5, 3), (3, 5, 4), (3, 4, 0)]
    return Mesh(V, f, c)


def hip_ring(ring, y0, rise, c, over=0.4):
    """Hipped roof over a nearly rectangular ring (its oriented bounding box), slope ~ rise."""
    A = np.asarray(ring, float)
    cxz = A.mean(0)
    _, _, vt = np.linalg.svd(A - cxz)
    u = vt[0]
    a = math.atan2(u[1], u[0])
    L = (A - cxz) @ u
    W = (A - cxz) @ np.array([-u[1], u[0]])
    m = hip(L.min() - over, L.max() + over, W.min() - over, W.max() + over, y0, y0 + rise, c)
    return m.R(0, -a, 0).T(cxz[0], 0, cxz[1])


def pyramid(ring, y0, y1, c, apex=None):
    cx, cz = apex or centroid(ring)
    R = ccw(ring)
    V = [(x, y0, z) for x, z in R] + [(cx, y1, cz)]
    k = len(R)
    return Mesh(V, [(i, k, (i + 1) % k) for i in range(k)], c)


def ray_hit(ring, p, d):
    """Farthest crossing of the ray p + t·d with a ring: (point, outward normal)."""
    R = ccw(ring)
    best = None
    for i in range(len(R)):
        a, b = np.array(R[i]), np.array(R[(i + 1) % len(R)])
        e = b - a
        M = np.array([[d[0], -e[0]], [d[1], -e[1]]])
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        t, s = np.linalg.solve(M, a - np.asarray(p))
        if t > 0 and -1e-6 <= s <= 1 + 1e-6 and (best is None or t > best[0]):
            L = np.linalg.norm(e)
            best = (t, np.asarray(p) + t * np.asarray(d), np.array([e[1], -e[0]]) / L)
    return None if best is None else (best[1], best[2])


def parapet(ring, y, h, w, c, closed=True, min_len=0.3):
    """A balustrade wall along every edge of a ring, set inside by w/2."""
    R = ccw(ring)
    ms = []
    n = len(R) if closed else len(R) - 1
    for i in range(n):
        p, q = np.array(R[i]), np.array(R[(i + 1) % len(R)])
        L = np.linalg.norm(q - p)
        if L < min_len:
            continue
        e = (q - p) / L
        nrm = np.array([e[1], -e[0]])
        ms.append(obox(p - nrm * w / 2, q - nrm * w / 2, y, y + h, w, c))
    return merge(ms)


def crenels(ring, y, h, w, c, pitch=1.6, merlon=0.9):
    """Merlons along every edge of a ring (the battlements of the Chapel of Santiago)."""
    R = ccw(ring)
    ms = []
    for i in range(len(R)):
        p, q = np.array(R[i]), np.array(R[(i + 1) % len(R)])
        L = float(np.linalg.norm(q - p))
        e = (q - p) / L
        nrm = np.array([e[1], -e[0]])
        n = max(1, int(L / pitch))
        for k in range(n):
            s0 = (k + 0.5) * L / n - merlon / 2
            a, b = p + e * s0 - nrm * w / 2, p + e * (s0 + merlon) - nrm * w / 2
            ms.append(obox(a, b, y, y + h, w, c))
    return merge(ms)


def figure(x, y, z, h, c, r=None, face=0.0):
    """A standing statue: a tapering body on a little base, a head."""
    r = r or h * 0.15
    body = lathe([(r * 1.1, 0), (r * 1.1, h * 0.06), (r, h * 0.08), (r * 0.95, h * 0.45), (r * 0.8, h * 0.78),
                  (r * 0.55, h * 0.82), (0.0, h * 0.82)], c, 6)
    head = sphere(r * 0.62, c, 6, 4).T(0, h * 0.82 + r * 0.5, 0)
    return merge([body, head]).R(0, face, 0).T(x, y, z)


def pinnacle(x, z, y0, h, r, c=STONE_L, seg=4, crockets=True, rot=math.pi / 4):
    """A Gothic pinnacle: a little shaft and a spirelet with crockets."""
    ms = [cyl(r, r, h * 0.35, c, seg=seg).R(0, rot, 0).T(x, y0, z),
          cyl(r * 1.05, 0.03, h * 0.65, c, seg=seg).R(0, rot, 0).T(x, y0 + h * 0.35, z)]
    if crockets:
        for k in range(seg):
            a = rot + 2 * math.pi * k / seg
            ms.append(cyl(0.09 * r / 0.3, 0.01, h * 0.22, c, seg=3).T(x + math.cos(a) * r * 0.75, y0 + h * 0.48, z + math.sin(a) * r * 0.75))
    return merge(ms)


def strip(A, Bp, c):
    """Quad strip between two polylines of 3D points (a roof slope, a lean-to)."""
    A, Bp = np.asarray(A, float), np.asarray(Bp, float)
    n = len(A)
    V = np.vstack([A, Bp])
    f = []
    for i in range(n - 1):
        f += [(i, i + 1, n + i + 1), (i, n + i + 1, n + i)]
    return Mesh(V, f, c)


def clip_x(P, a, keep_low):
    """Sutherland–Hodgman: the part of a polygon with x <= a (keep_low) or x >= a."""
    def inside(p):
        return p[0] <= a if keep_low else p[0] >= a
    out = []
    n = len(P)
    for i in range(n):
        p, q = P[i], P[(i + 1) % n]
        ip, iq = inside(p), inside(q)
        if ip:
            out.append(p)
        if ip != iq:
            t = (a - p[0]) / (q[0] - p[0])
            out.append((a, p[1] + t * (q[1] - p[1])))
    clean = []
    for p in out:
        if not clean or math.dist(clean[-1], p) > 0.05:
            clean.append(p)
    return clean


def flyer(p_out, p_in, y_out, y_in, c, thick=0.8, depth=1.4):
    """A flying buttress in the vertical plane through p_out→p_in: from the top of a pier (y_out, its top edge) to a
    wall (y_in): a straight coping on a quarter-arch."""
    F, L = along(p_out, p_in)
    top0, top1 = y_out, y_in
    bot = [(L * k / 8, (y_out - depth * 1.6) + ((y_in - depth) - (y_out - depth * 1.6)) * math.sin(math.pi / 2 * k / 8) ** 0.7)
           for k in range(9)]
    bot[0] = (0.0, y_out - depth * 2.6)
    pts = bot + [(L + 0.2, top1), (0.0, top0)]
    ms = [slab(F, pts, thick, c)]
    ms.append(slab(F, [(0, top0 - 0.05), (L + 0.2, top1 - 0.05), (L + 0.2, top1 + 0.3), (0, top0 + 0.3)], thick * 1.3, STONE_L))
    return merge(ms)


# ── Plans ────────────────────────────────────────────────────────────────────
CATH_LOW = clip_x(ccw(LOW), -34.0, keep_low=False)
NORTH_LOW = clip_x(ccw(LOW), -34.0, keep_low=True)
COURT = [(-38.45, 61.65), (-74.15, 62.80), (-74.45, 26.75), (-38.90, 26.20)]       # the four corners of the courtyard


def apse_r(a):
    """Radius of the apse wall (HIGH) from the centre of the apse, towards angle a."""
    h = ray_hit(HIGH, AC, (math.cos(a), math.sin(a)))
    return float(np.linalg.norm(h[0] - np.asarray(AC))) if h else 7.5


GIR_PTS = []
for x, z in GIROLA:
    r = math.hypot(x - AC[0], z - AC[1])
    a = math.atan2(z - AC[1], x - AC[0])
    if z < AC[1] + 0.5 and 22.0 < r < 27.6:
        GIR_PTS.append((a, r))
GIR_PTS.sort()


def girola_r(a):
    """Radius of the outer wall of the ambulatory (without the buttress piers) towards angle a."""
    A = np.array([p[0] for p in GIR_PTS]); R = np.array([p[1] for p in GIR_PTS])
    return float(np.interp(a, A, R))


def teeth():
    """The buttress piers drawn round the ambulatory: (angle, outer radius) of each one."""
    out = []
    pts = [(math.atan2(z - AC[1], x - AC[0]), math.hypot(x - AC[0], z - AC[1])) for x, z in GIROLA if z < AC[1] - 1]
    far = [(a, r) for a, r in pts if 28.6 < r < 31.0]
    far.sort()
    group = []
    for a, r in far:
        if group and a - group[-1][0] > 0.06:
            out.append((float(np.mean([g[0] for g in group])), float(max(g[1] for g in group)))); group = []
        group.append((a, r))
    if group:
        out.append((float(np.mean([g[0] for g in group])), float(max(g[1] for g in group))))
    return out


# ── Parts of the build ───────────────────────────────────────────────────────
P = {k: [] for k in ("stone", "tower", "decor", "glass", "roofs", "slate", "metal", "statues", "garden")}
S, TW, D, G, RF, SL, ME, ST, GD = (P[k] for k in ("stone", "tower", "decor", "glass", "roofs", "slate", "metal", "statues", "garden"))


def body():
    """Low bodies, the high cross, the ambulatory: walls, roofs, windows, flying buttresses."""
    # low bodies
    S.append(walls(CATH_LOW, FOUND, OUT_H, stone()))
    RF.append(cap(CATH_LOW, OUT_H, TILE_D))
    S.append(walls(NORTH_LOW, FOUND, NORTH_H, stone(0.97)))
    RF.append(cap(NORTH_LOW, NORTH_H, TILE))
    S.append(walls(GIROLA, FOUND, OUT_H, stone()))
    RF.append(cap(GIROLA, OUT_H - 0.05, TILE_D))
    S.append(parapet(CATH_LOW, OUT_H, 0.45, 0.4, STONE_L, min_len=1.0))
    # high cross: nave, transept, choir and apse
    S.append(walls(HIGH, FOUND, EAVE, stone(1.02)))
    S.append(parapet(HIGH, EAVE, 0.5, 0.5, STONE_L, min_len=1.0))
    zs = [-17.0, 7.5, 62.9]
    xn = [CHOIR_N - 0.35, NAVE_N(7.5) - 0.35, NAVE_N(62.9) - 0.35]
    xs = [CHOIR_S + 0.35, NAVE_S(7.5) + 0.35, NAVE_S(62.9) + 0.35]
    xm = [(a + b) / 2 for a, b in zip(xn, xs)]
    RF.append(strip([(x, EAVE, z) for x, z in zip(xn, zs)], [(x, RIDGE, z) for x, z in zip(xm, zs)], TILE))
    RF.append(strip([(x, RIDGE, z) for x, z in zip(xm, zs)], [(x, EAVE, z) for x, z in zip(xs, zs)], TILE))
    RF.append(strip([(-28.3, EAVE, -TZ_HW - 0.35), (26.9, EAVE, -TZ_HW - 0.35)], [(-28.3, RIDGE, 0.0), (26.9, RIDGE, 0.0)], TILE))
    RF.append(strip([(-28.3, RIDGE, 0.0), (26.9, RIDGE, 0.0)], [(-28.3, EAVE, TZ_HW + 0.35), (26.9, EAVE, TZ_HW + 0.35)], TILE))
    apex = (xm[0], RIDGE, -17.0)
    ap = [(x, z) for x, z in ccw(HIGH) if z < -16.85 and -9.0 < x < 6.5]
    ap.sort(key=lambda p: (math.atan2(p[1] - AC[1], p[0] - AC[0]) - math.pi / 2) % (2 * math.pi))
    V = [apex] + [(AC[0] + (x - AC[0]) * 1.04, EAVE, AC[1] + (z - AC[1]) * 1.04) for x, z in ap]
    RF.append(Mesh(V, [(0, i, i + 1) for i in range(1, len(V) - 1)], TILE))
    ME.append(B(-0.25, 0.25, RIDGE, RIDGE + 0.45, -17.0, 62.9, TILE_D).warp(lambda v: v + np.c_[np.interp(v[:, 2], zs, xm), 0 * v[:, 1], 0 * v[:, 1]]))
    ME.append(B(-28.3, 26.9, RIDGE, RIDGE + 0.45, -0.25, 0.25, TILE_D))
    # gable of the nave against the west front
    F, L = along((xn[2] + 0.35, 62.9), (xs[2] - 0.35, 62.9))
    S.append(slab(F, [(0, EAVE), (L, EAVE), (L / 2, RIDGE + 0.6)], 0.8, stone(1.02)))

    # clerestory windows of the nave and transept, the oculi of the inner aisles, the aisle windows
    mids = [(BAYS[i] + BAYS[i + 1]) / 2 for i in range(len(BAYS) - 1)][:-1]
    for z in mids:
        for side, xw in ((-1, NAVE_N(z)), (1, NAVE_S(z))):
            F = fr((xw, z), (side, 0))
            D.append(lancet_on(F, 0, IN_TOP + 0.5, 3.4, 5.6, 0.06, STONE_D))
            G.append(lancet_on(F, 0, IN_TOP + 0.7, 2.8, 5.2, 0.1, GLASS))
            D.append(rect_on(F, 0, IN_TOP + 0.7, 0.16, 3.6, 0.14, STONE_L))
            xa = AISLE_N if side < 0 else AISLE_S
            F = fr((xa + side * 0.5, z), (side, 0))
            D.append(rose(F, 0, 19.6, 1.35, 0.05, spokes=8))
        p = ray_hit(CATH_LOW, (0.0, z), (1.0, 0.0))
        if p is not None:
            F = fr(p[0], (1, 0))
            D.append(lancet_on(F, 0, 4.0, 3.0, 9.5, 0.06, STONE_D))
            G.append(lancet_on(F, 0, 4.3, 2.4, 8.9, 0.1, GLASS))
            D.append(rect_on(F, 0, 4.3, 0.14, 6.5, 0.14, STONE_L))
    for x in (-24.0, -18.0, -12.5, 10.5, 16.0, 22.0):                    # transept, east and west faces
        for side in (-1, 1):
            F = fr((x, side * (TZ_HW + 0.02)), (0, side))
            D.append(lancet_on(F, 0, IN_TOP + 0.5, 3.0, 5.6, 0.06, STONE_D))
            G.append(lancet_on(F, 0, IN_TOP + 0.7, 2.5, 5.2, 0.1, GLASS))
    for side, xw in ((-1, CHOIR_N), (1, CHOIR_S)):                        # straight bay of the choir
        F = fr((xw, -12.3), (side, 0))
        D.append(lancet_on(F, 0, IN_TOP + 0.5, 3.2, 5.6, 0.06, STONE_D))
        G.append(lancet_on(F, 0, IN_TOP + 0.7, 2.6, 5.2, 0.1, GLASS))
    for i in range(len(ap) - 1):                                          # the apse
        (x0, z0), (x1, z1) = ap[i], ap[i + 1]
        L = math.hypot(x1 - x0, z1 - z0)
        if L < 2.5:
            continue
        mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
        nrm = np.array([z1 - z0, -(x1 - x0)])
        if np.dot(nrm, [mx - AC[0], mz - AC[1]]) < 0:
            nrm = -nrm
        F = fr((mx, mz), nrm)
        D.append(lancet_on(F, 0, IN_TOP + 0.5, min(3.0, L - 1.0), 5.6, 0.06, STONE_D))
        G.append(lancet_on(F, 0, IN_TOP + 0.7, min(2.5, L - 1.5), 5.2, 0.1, GLASS))

    # aisles of the nave: inner aisle wall, lean-to roofs, double flying buttresses
    z0, z1 = 7.5, 62.0
    for side, xo, xa in ((-1, OUT_N, AISLE_N), (1, OUT_S, AISLE_S)):
        nw = NAVE_N if side < 0 else NAVE_S
        S.append(B(min(xa, xa - side * 1.0), max(xa, xa - side * 1.0), OUT_H - 0.5, IN_H, z0, z1, stone(0.99)))
        S.append(B(min(xa, xa - side * 1.0) - 0.15, max(xa, xa - side * 1.0) + 0.15, IN_H, IN_H + 0.35, z0, z1, STONE_L))
        RF.append(strip([(nw(z0), IN_TOP, z0), (nw(z1), IN_TOP, z1)], [(xa + side * 0.2, IN_H + 0.3, z0), (xa + side * 0.2, IN_H + 0.3, z1)], TILE))
        RF.append(strip([(xa - side * 1.0, 19.2, z0), (xa - side * 1.0, 19.2, z1)], [(xo + side * 0.3, OUT_H + 0.2, z0), (xo + side * 0.3, OUT_H + 0.2, z1)], TILE))
        for z in BAYS[1:-1]:
            S.append(B(min(xo - side * 0.6, xo + side * 1.6), max(xo - side * 0.6, xo + side * 1.6), FOUND, 21.8, z - 0.9, z + 0.9, stone(0.97)))
            S.append(pinnacle(xo + side * 0.5, z, 21.8, 4.8, 0.55))
            S.append(B(xa - 0.75, xa + 0.75, OUT_H, 27.6, z - 0.8, z + 0.8, stone(0.98)))
            S.append(pinnacle(xa, z, 27.6, 4.2, 0.5))
            S.append(flyer((xo + side * 0.4, z), (xa + side * 0.75, z), 21.6, 25.4, stone(1.03), thick=0.7, depth=1.1))
            S.append(flyer((xa - side * 0.75, z), (nw(z), z), 27.3, 30.8, stone(1.03), thick=0.7, depth=1.2))

    # choir straight bay: the inner and outer ambulatory along it
    for side, xc in ((-1, CHOIR_N), (1, CHOIR_S)):
        xa, xo = AC[0] + side * R_MID, AC[0] + side * (R_OUT - 0.2)
        zz = (-19.0, -7.5)
        RF.append(strip([(xc, IN_TOP, zz[0]), (xc, IN_TOP, zz[1])], [(xa, IN_H + 0.3, zz[0]), (xa, IN_H + 0.3, zz[1])], TILE))
        RF.append(strip([(xa, 19.2, zz[0]), (xa, 19.2, zz[1])], [(xo, OUT_H + 0.2, zz[0]), (xo, OUT_H + 0.2, zz[1])], TILE))
        S.append(B(min(xa, xa - side), max(xa, xa - side), OUT_H - 0.5, IN_H, zz[0], zz[1], stone(0.99)))

    # the ambulatory round the apse: radial lean-to roofs, curved wall between the two ambulatories
    angs = np.linspace(-math.pi, 0.0, 41)
    inn = [(AC[0] + (apse_r(a) + 0.05) * math.cos(a), IN_TOP, AC[1] + (apse_r(a) + 0.05) * math.sin(a)) for a in angs]
    mid = [(AC[0] + R_MID * math.cos(a), IN_H + 0.3, AC[1] + R_MID * math.sin(a)) for a in angs]
    mid2 = [(AC[0] + (R_MID - 1.0) * math.cos(a), 19.2, AC[1] + (R_MID - 1.0) * math.sin(a)) for a in angs]
    out = [(AC[0] + (girola_r(a) - 0.2) * math.cos(a), OUT_H + 0.2, AC[1] + (girola_r(a) - 0.2) * math.sin(a)) for a in angs]
    RF.append(strip(inn, mid, TILE))
    RF.append(strip(mid2, out, TILE))
    lo = [(x, OUT_H - 0.5, z) for x, _, z in mid]
    hi = [(x, IN_H, z) for x, _, z in mid]
    S.append(strip(lo, hi, stone(0.99)))
    S.append(strip([(x, OUT_H - 0.5, z) for x, _, z in mid2], [(x, IN_H, z) for x, _, z in mid2], stone(0.99)))
    S.append(strip(hi, [(x, IN_H, z) for x, _, z in mid2], STONE_L))
    for k in range(1, 10):                                              # oculi in the wall between the ambulatories
        a = -math.pi + k * math.pi / 10
        F = fr((AC[0] + R_MID * math.cos(a), AC[1] + R_MID * math.sin(a)), (math.cos(a), math.sin(a)))
        D.append(rose(F, 0, 19.7, 1.1, 0.05, spokes=6))
    for a, rt in teeth():                                               # radial piers and double flyers
        c_, s_ = math.cos(a), math.sin(a)
        po = (AC[0] + (girola_r(a) - 0.6) * c_, AC[1] + (girola_r(a) - 0.6) * s_)
        pt = (AC[0] + rt * c_, AC[1] + rt * s_)
        S.append(obox(po, pt, FOUND, 20.6, 1.25, stone(0.97)))
        S.append(pinnacle(AC[0] + (rt - 0.6) * c_, AC[1] + (rt - 0.6) * s_, 20.6, 4.4, 0.5))
        pm = (AC[0] + R_MID * c_, AC[1] + R_MID * s_)
        S.append(cyl(0.95, 0.95, 27.0 - OUT_H, stone(0.98), seg=4).R(0, -a + math.pi / 4, 0).T(pm[0], OUT_H, pm[1]))
        S.append(pinnacle(pm[0], pm[1], 27.0, 4.0, 0.5, rot=-a + math.pi / 4))
        pin = (AC[0] + (apse_r(a) + 0.1) * c_, AC[1] + (apse_r(a) + 0.1) * s_)
        S.append(flyer((AC[0] + (rt - 1.0) * c_, AC[1] + (rt - 1.0) * s_), (AC[0] + (R_MID + 0.7) * c_, AC[1] + (R_MID + 0.7) * s_), 20.4, 24.6, stone(1.03), thick=0.6, depth=1.0))
        S.append(flyer((AC[0] + (R_MID - 0.7) * c_, AC[1] + (R_MID - 0.7) * s_), pin, 26.7, 30.6, stone(1.03), thick=0.6, depth=1.1))
    # outer windows of the ambulatory chapels
    for k in range(1, 12):
        a = -math.pi + k * math.pi / 12
        if -2.05 < a < -1.0:                                            # San Ildefonso and Santiago stand here
            continue
        r = girola_r(a)
        F = fr((AC[0] + r * math.cos(a), AC[1] + r * math.sin(a)), (math.cos(a), math.sin(a)))
        D.append(lancet_on(F, 0, 5.0, 2.4, 7.5, 0.08, STONE_D))
        G.append(lancet_on(F, 0, 5.3, 1.9, 7.0, 0.12, GLASS))

    # the Transparente: Narciso Tomé's skylight opened in the vault behind the high altar
    tx, tz = AC[0], AC[1] - 11.6
    S.append(B(tx - 2.6, tx + 2.6, 18.0, 25.2, tz - 2.6, tz + 2.6, stone(1.02)))
    for nrm in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        F = fr((tx + nrm[0] * 2.6, tz + nrm[1] * 2.6), nrm)
        G.append(poly_on(F, round_pts(0, 21.8, 2.6, 3.0), 0.06, GLASS))
    SL.append(pyramid(rect(tx - 3.0, tx + 3.0, tz - 3.0, tz + 3.0), 25.2, 29.2, SLATE))
    ME.append(cyl(0.08, 0.08, 1.6, IRON, seg=4).T(tx, 29.1, tz))
    ME.append(B(tx - 0.45, tx + 0.45, 30.1, 30.22, tz - 0.05, tz + 0.05, IRON))


def frontispiece(face_x, z0, z1, nrm, top, gable, rose_y, rose_r, portal_w, portal_h, name):
    """A transept façade (Puerta de los Leones to the south, Puerta del Reloj to the north)."""
    back = face_x - nrm * 5.2
    S.append(B(min(face_x, back), max(face_x, back), FOUND, top, z0, z1, stone(1.03)))
    S.append(parapet(rect(min(face_x, back), max(face_x, back), z0, z1), top, 0.9, 0.35, STONE_L))
    zc = (z0 + z1) / 2
    F = fr((face_x, zc), (nrm, 0))
    w = z1 - z0
    S.append(slab(F, [(-w / 2 + 1.2, top), (w / 2 - 1.2, top), (0, gable)], 0.9, stone(1.04), off=-0.45))
    ST.append(figure(face_x - nrm * 0.4, gable - 0.3, zc, 2.6, STONE_L, face=math.pi / 2 * nrm))
    D.append(rose(F, 0, rose_y, rose_r, 0.1, spokes=16))
    D.append(lancet_on(F, 0, rose_y - rose_r - 1.0, rose_r * 2 + 2.4, rose_r * 2 + 3.4, 0.05, STONE_D, sharp=1.0))
    D.append(lancet_on(F, 0, 0.0, portal_w + 2.0, portal_h + 1.4, 0.06, STONE_L, sharp=0.8))
    D.append(lancet_on(F, 0, 0.0, portal_w, portal_h, 0.1, STONE_D, sharp=0.8))
    for k in range(1, 4):
        D.append(lancet_on(F, 0, 0.0, portal_w - 0.55 * k, portal_h - 0.5 * k, 0.1 + 0.03 * k, STONE_L if k % 2 else STONE_D, sharp=0.8))
    D.append(rect_on(F, 0, 0.0, portal_w - 2.2, 6.0, 0.24, DOOR))
    D.append(rect_on(F, 0, 6.0, portal_w - 2.2, 0.5, 0.25, STONE_L))
    D.append(rect_on(F, 0, 0.0, 0.5, 6.0, 0.27, STONE_L))
    S.append(slab(F, [(-portal_w / 2 - 0.6, portal_h - 2.5), (portal_w / 2 + 0.6, portal_h - 2.5), (0, portal_h + 3.2)], 0.4, STONE_L, off=0.3))
    for k in (-1, 1):                                                  # statues in niches beside the portal
        for j in range(3):
            s = k * (portal_w / 2 + 0.9 + j * 1.15)
            D.append(lancet_on(F, s, 6.5, 0.95, 3.4, 0.08, NICHE))
            p = F[0] + F[1] * s + F[2] * 0.35
            ST.append(figure(p[0], 6.6, p[2], 2.3, STONE_L, face=math.pi / 2 * nrm))
    D.append(rect_on(F, 0, portal_h + 4.0, w - 1.0, 0.5, 0.08, STONE_L))
    for k in range(-4, 5):                                             # gallery of little arches over the portal
        D.append(lancet_on(F, k * (w - 2.0) / 9, portal_h + 4.6, 1.0, 2.4, 0.08, NICHE))


def west_front():
    """The west front between the tower and the Mozarabic Chapel, with the compás and its railing."""
    bays = [(-16.6, -7.8, 66.27, 29.5), (-5.55, 5.88, 66.85, 31.0), (8.0, 15.4, 66.63, 29.5)]
    for x0, x1, zf, h in bays:
        S.append(B(x0, x1, FOUND, h, 56.0, zf, stone(1.04)))
        S.append(parapet(rect(x0, x1, 56.0, zf), h, 1.0, 0.35, STONE_L))
    for x0, x1 in ((-7.8, -5.55), (5.88, 8.0)):                          # the two buttresses between the portals
        S.append(B(x0, x1, FOUND, 32.5, 56.0, 68.5, stone(1.05)))
        S.append(pinnacle((x0 + x1) / 2, 67.4, 32.5, 4.6, 0.7))
        S.append(pinnacle((x0 + x1) / 2, 64.0, 32.5, 3.6, 0.6))
        for y in (7.6, 15.0, 22.5):
            F = fr(((x0 + x1) / 2, 68.5), (0, 1))
            D.append(lancet_on(F, 0, y, 1.3, 3.6, 0.06, NICHE))
            ST.append(figure((x0 + x1) / 2, y + 0.1, 68.3, 2.4, STONE_L))
    # the central bay: Puerta del Perdón, the Last Supper relief, two great arches and the pediment
    F = fr((0.16, 66.85), (0, 1))
    D.append(lancet_on(F, 0, 0.0, 10.4, 19.2, 0.05, STONE_L, sharp=0.8))
    D.append(lancet_on(F, 0, 0.0, 8.8, 17.6, 0.08, STONE_D, sharp=0.8))
    for k in range(1, 5):
        D.append(lancet_on(F, 0, 0.0, 8.8 - 0.6 * k, 17.6 - 0.55 * k, 0.08 + 0.03 * k, STONE_L if k % 2 else STONE_D, sharp=0.8))
    D.append(rect_on(F, 0, 0.0, 4.6, 7.4, 0.25, DOOR))
    D.append(rect_on(F, 0, 7.4, 4.6, 3.0, 0.24, STONE_L))                # tympanum with the Imposition of the chasuble
    D.append(rect_on(F, 0, 0.0, 0.45, 7.4, 0.27, STONE_L))
    for k in (-1, 1):
        for j in range(4):
            ST.append(figure(k * (2.9 + j * 0.42), 1.6, 66.4 + j * 0.15, 2.2, STONE_L))
    S.append(slab(F, [(-5.6, 16.0), (5.6, 16.0), (0, 21.5)], 0.45, STONE_L, off=0.35))
    D.append(rect_on(F, 0, 21.8, 11.2, 1.6, 0.08, STONE_L))                # Last Supper frieze
    for k in range(13):
        ST.append(figure(-4.8 + k * 0.8, 22.0, 67.0, 1.2, STONE_L))
    for s in (-2.6, 2.6):
        D.append(poly_on(F, round_pts(s, 24.2, 4.0, 5.8), 0.06, STONE_L))
        D.append(poly_on(F, round_pts(s, 24.4, 3.4, 5.4), 0.1, DARK))
    S.append(slab(F, [(-5.7, 31.0), (5.7, 31.0), (0, 35.0)], 0.8, stone(1.05), off=-0.3))
    D.append(disc_on(F, 0, 32.6, 0.8, 0.18, STONE_L, seg=12))
    ST.append(figure(0.16, 34.8, 66.6, 1.8, STONE_L))
    # side bays: Puerta del Infierno (north) and del Juicio Final (south), niches, gallery, balustrade
    for xc, zf in ((-12.2, 66.27), (11.7, 66.63)):
        F = fr((xc, zf), (0, 1))
        D.append(lancet_on(F, 0, 0.0, 7.0, 13.8, 0.05, STONE_L, sharp=0.75))
        D.append(lancet_on(F, 0, 0.0, 5.6, 12.6, 0.08, STONE_D, sharp=0.75))
        for k in range(1, 4):
            D.append(lancet_on(F, 0, 0.0, 5.6 - 0.55 * k, 12.6 - 0.5 * k, 0.08 + 0.03 * k, STONE_L if k % 2 else STONE_D, sharp=0.75))
        D.append(rect_on(F, 0, 0.0, 3.0, 6.2, 0.22, DOOR))
        for row, y in enumerate((15.0, 19.5)):
            for k in range(-2, 3):
                D.append(lancet_on(F, k * 1.45, y, 1.0, 3.3, 0.06, NICHE))
                ST.append(figure(xc + k * 1.45, y + 0.1, zf + 0.25, 2.3, STONE_L))
        D.append(rect_on(F, 0, 23.0, 7.6, 0.4, 0.08, STONE_L))
        for k in range(-3, 4):
            D.append(poly_on(F, round_pts(k * 1.1, 23.8, 0.8, 3.0), 0.07, DARK))
    # the compás in front of the portals: pavement, stone posts with urns and the iron railing
    S.append(B(-16.6, 15.4, FOUND, 0.3, 66.3, 73.6, PAVE))
    xs = np.linspace(-16.3, 15.1, 11)
    for x in xs:
        S.append(B(x - 0.35, x + 0.35, FOUND, 2.4, 73.0, 73.7, stone(1.05)))
        ST.append(lathe([(0.0, 0), (0.25, 0.05), (0.33, 0.35), (0.18, 0.6), (0.28, 0.75), (0.0, 0.85)], STONE_L, 8).T(x, 2.4, 73.35))
    S.append(B(-16.3, 15.1, 0.0, 0.7, 73.1, 73.6, stone(1.03)))
    for x in np.arange(-16.0, 15.0, 0.32):
        if min(abs(x - xx) for xx in xs) > 0.45:
            ME.append(B(x - 0.03, x + 0.03, 0.7, 2.6, 73.3, 73.36, IRON))
    ME.append(B(-16.3, 15.1, 2.4, 2.5, 73.28, 73.38, IRON))


def tower():
    """Alvar Martínez's square shaft (1425–1440) and Hanequin of Brussels' octagon and spire with three crowns."""
    x0, x1, z0, z1 = TOWER
    cx, cz, hw = (x0 + x1) / 2, (z0 + z1) / 2, (x1 - x0) / 2
    TOP, OCT_TOP = 55.0, 71.5
    TW.append(extrude(rect(x0, x1, z0, z1), FOUND, TOP, stone(1.0, TOWER_C), top_c=TILE_D))
    for sx in (-1, 1):                                                   # corner buttresses
        for sz in (-1, 1):
            px, pz = cx + sx * hw, cz + sz * hw
            TW.append(B(px - 0.75, px + 0.75, FOUND, TOP, pz - 0.75, pz + 0.75, stone(1.03, TOWER_C)))
    for y, h in ((14.6, 0.7), (24.0, 0.6), (33.0, 0.6), (41.4, 0.6), (47.0, 0.7)):
        TW.append(B(x0 - 0.95, x1 + 0.95, y, y + h, z0 - 0.95, z1 + 0.95, STONE_L))
    TW.append(B(x0 - 1.1, x1 + 1.1, TOP - 0.2, TOP + 0.8, z0 - 1.1, z1 + 1.1, STONE_L))
    for nrm in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        F = fr((cx + nrm[0] * (hw + 0.75), cz + nrm[1] * (hw + 0.75)), nrm)
        w = 2 * hw - 1.6
        D.append(rect_on(F, 0, 21.6, w, 1.3, 0.05, DARK))               # the band of coats of arms
        for k in range(-4, 5):
            D.append(rect_on(F, k * w / 9, 21.8, 0.75, 0.9, 0.08, STONE_L))
        for y0, y1, n in ((1.5, 14.0, 5), (15.6, 21.2, 7), (24.9, 32.6, 7), (33.9, 41.0, 7), (42.2, 46.6, 9)):
            for k in range(n):
                s = -w / 2 + (k + 0.5) * w / n
                D.append(lancet_on(F, s, y0, w / n * 0.72, y1 - y0, 0.05, stone(0.86, TOWER_C), sharp=0.8))
        for s in (-2.0, 2.0):                                            # the belfry
            D.append(poly_on(F, round_pts(s, 47.9, 3.3, 6.6), 0.05, STONE_L))
            D.append(poly_on(F, round_pts(s, 48.1, 2.7, 6.2), 0.09, DARK))
        for s in (-5.6, 5.6):
            D.append(poly_on(F, round_pts(s, 48.4, 1.6, 5.8), 0.06, stone(0.86, TOWER_C)))
        p = F[0] + F[2] * 0.4
        ST.append(figure(p[0], 49.4, p[2], 2.6, STONE_L))
    TW.append(parapet(rect(x0 - 1.0, x1 + 1.0, z0 - 1.0, z1 + 1.0), TOP + 0.8, 1.2, 0.35, STONE_L))
    # the octagon (flats facing the axes)
    ap = 4.7
    R = ap / math.cos(math.pi / 8)
    oc = ngon(cx, cz, R, 8, math.pi / 8)
    TW.append(extrude(oc, TOP, OCT_TOP, stone(1.02, TOWER_C), top_c=TILE_D))
    for k in range(8):
        a = k * math.pi / 4
        F = fr((cx + ap * math.cos(a), cz + ap * math.sin(a)), (math.cos(a), math.sin(a)))
        D.append(lancet_on(F, 0, 59.0, 2.7, 10.0, 0.05, STONE_L))
        D.append(lancet_on(F, 0, 59.3, 2.2, 9.4, 0.09, DARK))
        D.append(rect_on(F, 0, 59.3, 0.18, 6.8, 0.13, STONE_L))
        TW.append(slab(F, [(-1.6, 67.2), (1.6, 67.2), (0, 71.0)], 0.25, STONE_L, off=0.2))
        TW.append(cyl(0.12, 0.02, 1.2, STONE_L, seg=4).T(*(F[0] + F[2] * 0.25 + np.array([0, 70.8, 0]))))
    for vx, vz in oc:
        TW.append(pinnacle(cx + (vx - cx) * 1.06, cz + (vz - cz) * 1.06, 58.0, 15.6, 0.32))
    TW.append(parapet(oc, OCT_TOP, 0.9, 0.3, STONE_L))
    for sx in (-1, 1):                                                  # corner pinnacles with their flyers
        for sz in (-1, 1):
            px, pz = cx + sx * (hw - 0.4), cz + sz * (hw - 0.4)
            TW.append(cyl(0.95, 0.95, 6.0, stone(1.0, TOWER_C), seg=4).R(0, math.pi / 4, 0).T(px, TOP + 0.8, pz))
            TW.append(pinnacle(px, pz, TOP + 6.8, 9.0, 0.85))
            for k in (-1, 1):
                TW.append(pinnacle(px - sx * 0.9 * (k > 0), pz - sz * 0.9 * (k < 0), TOP + 2.0, 6.5, 0.32))
            a = math.atan2(sz, sx)
            vin = (cx + R * math.cos(a) * 1.0, cz + R * math.sin(a) * 1.0)
            TW.append(flyer((px, pz), vin, 63.6, 67.4, stone(1.05, TOWER_C), thick=0.45, depth=0.8))
    for k in range(4):                                                  # little slate spirelets before the faces
        a = k * math.pi / 2
        px, pz = cx + (ap + 1.9) * math.cos(a), cz + (ap + 1.9) * math.sin(a)
        TW.append(cyl(0.95, 0.95, 3.2, stone(1.0, TOWER_C), seg=8).T(px, TOP + 0.8, pz))
        SL.append(cyl(1.05, 0.05, 5.0, SLATE, seg=8).T(px, TOP + 4.0, pz))
    # the spire: slate, three crowns of spikes, lead finial, cross and weather vane
    prof = [(3.55, 71.5), (3.55, 72.6), (3.25, 72.8), (2.65, 75.6), (2.35, 75.9), (1.85, 79.2), (1.6, 79.5),
            (1.12, 82.6), (0.9, 82.9), (0.1, 87.6)]
    SL.append(lathe(prof, SLATE, 8).R(0, math.pi / 8, 0).T(cx, 0, cz))
    for y, r0, n, L in ((75.7, 2.5, 24, 2.3), (79.3, 1.75, 20, 1.9), (82.75, 1.05, 16, 1.5)):
        ME.append(cyl(r0 + 0.35, r0 + 0.35, 0.4, LEAD, seg=16).T(cx, y - 0.25, cz))
        for k in range(n):                                              # the crown: spikes rising outwards
            a = 2 * math.pi * k / n
            for elev, LL in ((0.55, L), (1.0, L * 0.6)):
                sp = cyl(0.15, 0.01, LL, LEAD, seg=4, caps=False).R(0, 0, -(math.pi / 2 - elev)).R(0, -a, 0)
                ME.append(sp.T(cx + (r0 + 0.2) * math.cos(a), y, cz + (r0 + 0.2) * math.sin(a)))
    ME.append(cyl(0.11, 0.08, 4.8, LEAD, seg=6).T(cx, 87.3, cz))
    ME.append(sphere(0.38, LEAD, 8, 5).T(cx, 88.4, cz))
    ME.append(sphere(0.28, LEAD, 8, 5).T(cx, 89.3, cz))
    ME.append(B(cx - 0.06, cx + 0.06, 89.6, 92.0, cz - 0.06, cz + 0.06, IRON))
    ME.append(B(cx - 0.55, cx + 0.55, 91.05, 91.2, cz - 0.05, cz + 0.05, IRON))
    ME.append(B(cx - 0.04, cx + 0.04, 90.1, 90.6, cz - 0.7, cz + 0.7, IRON))
    ME.append(B(cx - 0.03, cx + 0.03, 90.0, 90.7, cz + 0.55, cz + 0.95, IRON))


def dome(cx, cz, r0, y0, h, c, seg=16, ribs=8, rib_c=None, rot=0.0):
    """A dome of revolution (seg sides) with ribs."""
    prof = [(r0 * math.cos(t), y0 + h * math.sin(t)) for t in np.linspace(0, math.pi / 2 * 0.93, 7)]
    ms = [lathe(prof, c, seg).R(0, rot, 0).T(cx, 0, cz)]
    for k in range(ribs):
        a = rot + 2 * math.pi * k / ribs + (math.pi / seg if seg == ribs else 0)
        pts = [(cx + (r + 0.12) * math.cos(a), y + 0.05, cz + (r + 0.12) * math.sin(a)) for r, y in prof]
        ms.append(tube(pts, 0.16, rib_c or c, seg=4))
    return merge(ms)


def lantern(cx, cz, y0, r, h, c=STONE_L, top=SLATE, cross=True):
    ms = [cyl(r + 0.15, r + 0.15, 0.3, c, seg=8).T(cx, y0, cz), cyl(r * 0.8, r * 0.8, h, DARK, seg=8).T(cx, y0 + 0.3, cz)]
    for k in range(8):
        a = 2 * math.pi * k / 8
        ms.append(B(-0.14, 0.14, 0, h, -0.14, 0.14, c).T(cx + r * math.cos(a), y0 + 0.3, cz + r * math.sin(a)))
    ms.append(cyl(r + 0.2, r + 0.2, 0.3, c, seg=8).T(cx, y0 + 0.3 + h, cz))
    ms.append(lathe([(r, 0), (r * 0.8, r * 0.45), (r * 0.4, r * 0.8), (0.0, r * 0.9)], top, 8).T(cx, y0 + 0.6 + h, cz))
    yt = y0 + 0.6 + h + r * 0.9
    if cross:
        ms.append(sphere(0.25, GOLD, 8, 5).T(cx, yt + 0.25, cz))
        ms.append(B(cx - 0.05, cx + 0.05, yt + 0.4, yt + 1.7, cz - 0.05, cz + 0.05, IRON))
        ms.append(B(cx - 0.35, cx + 0.35, yt + 1.25, yt + 1.35, cz - 0.04, cz + 0.04, IRON))
    return merge(ms)


def mozarabic():
    """The base of the unfinished south tower (Mozarabic Chapel, Cisneros) and J. M. Theotocópuli's dome."""
    x0, x1, z0, z1 = MOZ_B
    S.append(extrude(rect(x0, x1, z0, z1), FOUND, 21.0, stone(0.97, TOWER_C), top_c=TILE_D))
    for px in (x0, x1):
        for pz in (z1,):
            S.append(B(px - 0.8, px + 0.8, FOUND, 21.6, pz - 1.4, pz + 1.0, stone(1.0, TOWER_C)))
    S.append(B(x0 - 0.3, x1 + 0.3, 20.6, 21.3, z0 - 0.3, z1 + 0.3, STONE_L))
    S.append(parapet(rect(x0 - 0.3, x1 + 0.3, z0 - 0.3, z1 + 0.3), 21.3, 1.1, 0.3, STONE_L))
    for px, pz in ((x0, z0), (x0, z1), (x1, z0), (x1, z1)):
        S.append(pinnacle(px, pz, 21.3, 3.6, 0.4))
    for nrm, p in (((0, 1), ((x0 + x1) / 2, z1)), ((1, 0), (x1, (z0 + z1) / 2))):
        F = fr(p, nrm)
        D.append(lancet_on(F, 0, 6.0, 2.4, 6.5, 0.06, STONE_D))
        G.append(lancet_on(F, 0, 6.3, 1.8, 6.0, 0.1, GLASS))
    cx, cz = centroid(MOZ)
    ap = 4.8
    R = ap / math.cos(math.pi / 8)
    oc = ngon(cx, cz, R, 8, math.pi / 8)
    S.append(extrude(oc, 21.0, 30.0, stone(1.04), top_c=TILE_D))
    for k in range(8):
        a = k * math.pi / 4
        F = fr((cx + ap * math.cos(a), cz + ap * math.sin(a)), (math.cos(a), math.sin(a)))
        D.append(lancet_on(F, 0, 23.4, 2.4, 4.8, 0.05, STONE_D))
        G.append(lancet_on(F, -0.55, 23.6, 0.95, 4.0, 0.09, GLASS))
        G.append(lancet_on(F, 0.55, 23.6, 0.95, 4.0, 0.09, GLASS))
    for vx, vz in oc:
        S.append(B(vx - 0.4, vx + 0.4, 21.0, 30.0, vz - 0.4, vz + 0.4, STONE_L))
        S.append(pinnacle(vx, vz, 30.0, 3.2, 0.32))
    S.append(cyl(R + 0.35, R + 0.35, 0.5, STONE_L, seg=8).R(0, math.pi / 8, 0).T(cx, 29.7, cz))
    S.append(parapet(ngon(cx, cz, R + 0.2, 8, math.pi / 8), 30.2, 1.0, 0.25, STONE_L))
    S.append(cyl(R - 0.3, R - 0.3, 1.6, stone(1.02), seg=8).R(0, math.pi / 8, 0).T(cx, 30.0, cz))
    SL.append(dome(cx, cz, R - 0.4, 31.6, 5.7, (0.47, 0.47, 0.46), seg=16, ribs=8, rib_c=(0.55, 0.55, 0.53)))
    for k in range(4):                                                  # dormers
        a = k * math.pi / 2 + math.pi / 8
        px, pz = cx + (R - 1.6) * math.cos(a), cz + (R - 1.6) * math.sin(a)
        d = merge([B(-0.6, 0.6, 0, 1.4, -0.5, 0.5, STONE_L), B(-0.42, 0.42, 0.15, 1.15, 0.45, 0.52, DARK)])
        SL.append(d.R(0, -a + math.pi / 2, 0).T(px, 32.9, pz))
    SL.append(lantern(cx, cz, 36.9, 1.05, 1.9, STONE_L, (0.47, 0.47, 0.46)))


def north_side():
    """Sagrario, Ochavo (Relicario), sacristy and the other dependencies to the north and the east."""
    # Sagrario: square chapel, slate pyramid and lantern
    S.append(extrude(SAGRARIO, FOUND, 24.0, stone(1.02), top_c=TILE_D))
    S.append(parapet(SAGRARIO, 24.0, 0.6, 0.4, STONE_L))
    cx, cz = centroid(SAGRARIO)
    SL.append(pyramid(rect(cx - 7.4, cx + 7.4, cz - 7.8, cz + 7.8), 24.2, 31.8, SLATE))
    S.append(cyl(1.5, 1.5, 1.0, STONE_L, seg=8).T(cx, 31.0, cz))
    SL.append(lantern(cx, cz, 32.0, 1.25, 2.4))
    for nrm, off in (((-1, 0), 7.1), ((0, -1), 7.6), ((0, 1), 7.6)):
        F = fr((cx + nrm[0] * off, cz + nrm[1] * off), nrm)
        for s in (-3.5, 3.5):
            G.append(poly_on(F, round_pts(s, 15.5, 1.6, 3.6), 0.1, GLASS))
    # Ochavo: octagon, slate dome, lantern
    cx, cz = centroid(OCHAVO)
    R = 6.9
    oc = ngon(cx, cz, R, 8, math.pi / 8)
    S.append(extrude(oc, FOUND, 26.0, stone(1.03), top_c=TILE_D))
    S.append(cyl(R + 0.4, R + 0.4, 0.7, STONE_L, seg=8).R(0, math.pi / 8, 0).T(cx, 25.6, cz))
    S.append(cyl(R - 0.6, R - 0.6, 2.0, stone(1.0), seg=8).R(0, math.pi / 8, 0).T(cx, 26.0, cz))
    SL.append(dome(cx, cz, R - 0.7, 28.0, 7.6, SLATE, seg=8, ribs=8, rib_c=LEAD, rot=math.pi / 8))
    SL.append(lantern(cx, cz, 35.2, 1.25, 2.8))
    for k in range(8):
        a = k * math.pi / 4
        ap = (R - 0.6) * math.cos(math.pi / 8)
        F = fr((cx + ap * math.cos(a), cz + ap * math.sin(a)), (math.cos(a), math.sin(a)))
        G.append(poly_on(F, round_pts(0, 26.3, 1.3, 1.6), 0.08, GLASS))
    # sacristy: long hall with Luca Giordano's vault, hipped slate roof
    S.append(extrude(SACRISTY, FOUND, 18.5, stone(0.99), top_c=TILE_D))
    SL.append(hip_ring(SACRISTY, 18.5, 4.8, SLATE))
    for x in (-60.0, -53.0, -46.0):
        F = fr((x, -29.9), (0, -1))
        G.append(poly_on(F, round_pts(0, 11.0, 1.6, 3.8), 0.1, GLASS))
    # the patio of the treasurer's house, between the sacristy and the dependencies: a paved floor
    S.append(B(-53.8, -38.7, FOUND, 0.15, -52.4, -37.9, PAVE))
    # dependencies (levels -> heights), tiled hipped roofs
    for ring, h, rise in ((B130, 16.5, 3.2), (B132, 13.5, 3.0), (B133, 10.0, 2.6), (B131, 14.0, 3.0),
                          (B137, 14.0, 3.2), (B136, 12.0, 2.6), (B139, 12.0, 2.4), (B150, 9.0, 2.4), (B405, 9.0, 2.2)):
        S.append(extrude(ring, FOUND, h, stone(0.96), top_c=TILE_D))
        RF.append(hip_ring(ring, h, rise, TILE, over=0.3))
        A = np.asarray(ring)
        for i in range(len(ring)):                                       # windows on the long walls
            p, q = A[i], A[(i + 1) % len(A)]
            L = float(np.linalg.norm(q - p))
            if L < 7.0:
                continue
            e = (q - p) / L
            nrm = (e[1], -e[0]) if ring_area(ring) > 0 else (-e[1], e[0])
            n = int(L / 3.6)
            for k in range(n):
                s = (k + 0.5) * L / n
                pt = p + e * s
                F = fr(pt, nrm)
                for y in np.arange(3.0, h - 2.0, 3.8):
                    D.append(rect_on(F, 0, y, 0.9, 1.5, 0.06, DARK))
    for ring, h in ((T134, 13.0), (T135, 17.0), (T140, 19.0), (T141, 19.0), (T152, 8.0)):
        S.append(extrude(ring, FOUND, h, stone(1.0), top_c=TILE_D))
        cx, cz = centroid(ring)
        SL.append(pyramid(ring, h, h + 2.0, SLATE))
    # the two stair turrets flanking the Puerta de los Leones
    for ring in (T143, T145):
        S.append(extrude(ring, FOUND, 33.0, stone(1.04), top_c=STONE_L))
        cx, cz = centroid(ring)
        S.append(pyramid(ring, 33.0, 37.5, STONE_L))
        for k in range(4):
            a = math.pi / 4 + k * math.pi / 2
            S.append(cyl(0.1, 0.01, 1.2, STONE_L, seg=3).T(cx + math.cos(a) * 1.0, 34.0, cz + math.sin(a) * 1.0))


def octagon_chapels():
    """San Ildefonso (Cardinal Albornoz, late 14th c.) and Santiago (Álvaro de Luna, 1435) behind the ambulatory."""
    for ring, h, crenel in ((ILDEFONSO, 21.5, False), (SANTIAGO, 21.0, True)):
        cx, cz = centroid(ring)
        S.append(extrude(ring, FOUND, h, stone(1.02), top_c=TILE_D))
        RF.append(pyramid(ring, h - 0.2, h + 2.2, TILE))
        R = ccw(ring)
        for i in range(len(R)):
            p, q = np.array(R[i]), np.array(R[(i + 1) % len(R)])
            m = (p + q) / 2
            if np.linalg.norm(q - p) < 3.0:
                continue
            e = (q - p) / np.linalg.norm(q - p)
            nrm = (e[1], -e[0])
            out_dir = np.array(m) - np.array(AC)
            if np.dot(nrm, out_dir) < 0.3 * np.linalg.norm(out_dir):
                continue
            F = fr(m, nrm)
            if crenel:
                D.append(rose(F, 0, 14.3, 1.6, 0.06, spokes=10))
                D.append(lancet_on(F, 0, 4.5, 2.2, 7.0, 0.06, STONE_D))
                G.append(lancet_on(F, 0, 4.8, 1.7, 6.5, 0.1, GLASS))
            else:
                D.append(lancet_on(F, 0, 7.0, 2.8, 10.0, 0.06, STONE_D))
                G.append(lancet_on(F, 0, 7.3, 2.2, 9.4, 0.1, GLASS))
                D.append(rect_on(F, 0, 7.3, 0.15, 6.5, 0.13, STONE_L))
        if crenel:
            S.append(crenels(ring, h, 1.6, 0.55, STONE_L, pitch=1.7, merlon=0.9))
            for vx, vz in ring:
                S.append(cyl(0.95, 0.95, h + 1.0 - FOUND, stone(1.0), seg=8).T(vx, FOUND, vz))
                S.append(crenels(ngon(vx, vz, 1.1, 8), h + 1.0, 1.0, 0.3, STONE_L, pitch=0.8, merlon=0.45))
        else:
            S.append(parapet(ring, h, 0.9, 0.3, STONE_L))
            for vx, vz in ring[::2]:
                S.append(pinnacle(vx, vz, h, 3.4, 0.35))


def cloister():
    """The cloister (1389–1425): stone arcades below, the wooden upper gallery, rendered top floor, tiled roofs,
    and the garden with its sixteen beds."""
    RW = 3.6                                                             # depth of the roof slope over the galleries
    TOPY, EAVY = 15.2, 13.1
    outer = []
    for i in range(4):
        p, q, r = np.array(COURT[i - 1]), np.array(COURT[i]), np.array(COURT[(i + 1) % 4])
        c0 = np.mean(COURT, axis=0)
        outer.append(tuple(q + (q - c0) / np.linalg.norm(q - c0) * RW * 1.41))
    S.append(walls(CLOISTER, FOUND, TOPY, stone(0.98)))
    RF.append(cap(CLOISTER, TOPY, TILE, holes=[outer]))
    c0 = np.mean(COURT, axis=0)
    inner_e = [tuple(np.array(p) + (np.array(p) - c0) / np.linalg.norm(np.array(p) - c0) * -1.1) for p in COURT]
    RF.append(strip([(x, TOPY, z) for x, z in outer + outer[:1]], [(x, EAVY, z) for x, z in inner_e + inner_e[:1]], TILE))
    S.append(walls(COURT, FOUND, EAVY + 0.2, stone(1.0), hole=True))
    for i in range(4):
        p, q = np.array(COURT[i]), np.array(COURT[(i + 1) % 4])
        L = float(np.linalg.norm(q - p))
        e = (q - p) / L
        nrm = (e[1], -e[0])
        if np.dot(nrm, c0 - (p + q) / 2) < 0:
            nrm = (-e[1], e[0])
        n = 8
        for k in range(n + 1):                                          # buttresses between the arches
            pt = p + e * (k * L / n)
            a = pt + np.array(nrm) * 0.0
            b = pt + np.array(nrm) * 1.0
            S.append(obox(a, b, FOUND, 6.9, 0.9, stone(1.03)))
        for k in range(n):
            s = (k + 0.5) * L / n
            F = fr(p + e * s, nrm)
            D.append(lancet_on(F, 0, 0.0, L / n - 1.0, 5.6, 0.06, STONE_D, sharp=0.75))
            D.append(lancet_on(F, 0, 0.0, L / n - 1.6, 5.1, 0.1, DARK, sharp=0.75))
        F = fr((p + q) / 2, nrm)
        D.append(rect_on(F, 0, 6.6, L, 0.7, 0.06, STONE_L))
        D.append(rect_on(F, 0, 7.3, L, 2.9, 0.05, DARK))                 # upper gallery
        D.append(rect_on(F, 0, 7.3, L, 0.9, 0.3, WOOD))
        D.append(rect_on(F, 0, 10.2, L, 2.9, 0.05, RENDER))
        D.append(rect_on(F, 0, 10.0, L, 0.3, 0.32, WOOD))
        for k in range(int(L / 1.6) + 1):
            pt = p + e * (k * L / int(L / 1.6))
            o = pt + np.array(nrm) * 0.32
            D.append(B(o[0] - 0.09, o[0] + 0.09, 8.2, 10.0, o[1] - 0.09, o[1] + 0.09, WOOD))
        for k in range(int(L / 3.2)):
            s = (k + 0.5) * L / int(L / 3.2)
            D.append(rect_on(fr(p + e * s, nrm), 0, 11.0, 0.8, 1.2, 0.08, DARK))
    # outer windows (west and north faces)
    R = ccw(CLOISTER)
    for i in range(len(R)):
        p, q = np.array(R[i]), np.array(R[(i + 1) % len(R)])
        L = float(np.linalg.norm(q - p))
        if L < 8:
            continue
        e = (q - p) / L
        nrm = (e[1], -e[0])
        m = (p + q) / 2
        if not ((nrm[0] < -0.7 and m[0] < -80) or (nrm[1] > 0.7 and m[1] > 72)):
            continue
        n = int(L / 4.0)
        for k in range(n):
            F = fr(p + e * (k + 0.5) * L / n, nrm)
            for y in (4.5, 8.5, 12.0):
                D.append(rect_on(F, 0, y, 0.9, 1.5, 0.06, DARK))
    # the garden: floor, gravel paths, 4 × 4 beds of shrubs, a well, two cypresses
    GD.append(cap(COURT, 0.02, GRAVEL))
    GD.append(walls(COURT, FOUND, 0.02, GRAVEL, hole=True))
    xs = np.linspace(c0[0] - 13.3, c0[0] + 13.3, 4)
    zs = np.linspace(c0[1] - 13.3, c0[1] + 13.3, 4)
    for i, x in enumerate(xs):
        for j, z in enumerate(zs):
            h = 0.9 + 0.5 * rng.random()
            GD.append(B(x - 3.4, x + 3.4, 0.0, 0.35, z - 3.4, z + 3.4, (0.55, 0.47, 0.36)))
            bed = merge([B(x - 3.2, x + 3.2, 0.3, 0.3 + h, z - 3.2, z + 3.2, HEDGE)])
            GD.append(bed)
            for k in range(3):
                bx, bz = x + rng.uniform(-2.2, 2.2), z + rng.uniform(-2.2, 2.2)
                GD.append(sphere(rng.uniform(0.9, 1.4), (0.24, 0.36, 0.2), 7, 4).S(1, 0.8, 1).T(bx, 0.6 + h, bz))
    GD.append(cyl(1.0, 1.0, 0.9, STONE_L, seg=10).T(c0[0], 0.0, c0[1]))
    GD.append(cyl(0.75, 0.75, 0.05, (0.2, 0.25, 0.3), seg=10).T(c0[0], 0.88, c0[1]))
    for x, z in ((c0[0] - 15.5, c0[1] + 15.5), (c0[0] + 15.5, c0[1] - 15.5)):
        GD.append(lathe([(0.0, 0), (0.9, 0.8), (1.25, 3.0), (1.1, 6.0), (0.7, 8.5), (0.0, 10.5)], (0.14, 0.24, 0.13), 8).T(x, 0.0, z))


def leones_compas():
    """The little forecourt of the Puerta de los Leones: posts crowned with lions and the railing."""
    x = 37.2
    zs = np.linspace(-9.0, 6.6, 7)
    S.append(B(32.0, x + 0.3, FOUND, 0.3, -9.4, 7.0, PAVE))
    for z in zs:
        S.append(B(x - 0.4, x + 0.4, FOUND, 2.9, z - 0.4, z + 0.4, stone(1.05)))
        lion = merge([B(-0.55, 0.55, 0, 0.55, -0.22, 0.22, STONE_L), sphere(0.28, STONE_L, 6, 4).T(0.45, 0.75, 0),
                      B(-0.6, -0.3, 0, 0.4, -0.18, 0.18, STONE_L)])
        ST.append(lion.T(x, 2.9, z))
    for zz in (-9.4, 7.0):
        S.append(B(32.0, x, 0.0, 0.7, zz - 0.2, zz + 0.2, stone(1.03)))
    S.append(B(x - 0.25, x + 0.25, 0.0, 0.7, -9.0, 6.6, stone(1.03)))
    for z in np.arange(-8.8, 6.5, 0.3):
        if min(abs(z - zz) for zz in zs) > 0.45:
            ME.append(B(x - 0.03, x + 0.03, 0.7, 2.7, z - 0.03, z + 0.03, IRON))
    ME.append(B(x - 0.05, x + 0.05, 2.55, 2.65, -9.0, 6.6, IRON))


def build():
    body()
    frontispiece(32.0, -9.4, 7.1, 1, 31.0, 37.5, 23.5, 3.6, 7.6, 15.0, "Leones")
    frontispiece(-33.4, -7.5, 4.5, -1, 30.0, 35.5, 23.0, 3.4, 6.8, 13.5, "Reloj")
    leones_compas()
    west_front()
    tower()
    mozarabic()
    north_side()
    octagon_chapels()
    cloister()


def main():
    build()
    mats = {"glass": {"metallic": 0.35, "roughness": 0.2}, "metal": METAL, "slate": {"metallic": 0.15, "roughness": 0.6}}
    groups = [("stone", "stone", None), ("tower", "tower", None), ("decor", "decor", None), ("statues", "statues", None),
              ("garden", "garden", None), ("roofs", "roofs", None), ("glass", "glass", "glass"), ("slate", "slate", "slate"),
              ("metal", "metal", "metal")]
    out = []
    for name, key, mat in groups:
        m = sink(merge(P[key]))
        out.append((name, m, {"material": mat}) if mat else (name, m))
    dst = ROOT / "models" / "Q1123180"
    dst.mkdir(parents=True, exist_ok=True)
    tris, size = write_glb(dst / "model.glb", out, materials=mats)
    print(f"model.glb: {tris} triangles, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
