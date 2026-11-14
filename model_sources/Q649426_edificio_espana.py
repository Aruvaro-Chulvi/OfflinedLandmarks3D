"""
Edificio España (Q649426), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Julián Otamendi Machimbarrena (1889–1966, architect) with his brother José María Otamendi
(1885–1959, structural engineer), 1948–1953, for the Compañía Inmobiliaria Metropolitana; façades restored
2005–2010 (Rubio & Álvarez-Sala); refurbished as the Riu Plaza España hotel 2017–2019 (rooftop terrace).
Rights: protected (architects died after 1945).

Measurements (see meta.json for the sources):
- OpenStreetMap way 49031978 (outline, height 117) and its 90 building:part pieces (heights and levels), in
  model x/z for heading 235.2° (the long façade on the Plaza de España faces west-south-west). The parts give
  the whole stepped massing: the 150 m-wide front (≈105 m between the curved corners), the wings with curved
  corners (38–42 m), the main slab (70 m front, 73.8 m attic), the narrow side bodies (70 m), the upper tower
  (32 × 30 m), the lantern, the five deep courtyards open to the back and the 5-level rear podium.
- Floors and vertical layout measured on the perspective-corrected photo «Edificio España (Madrid) 2014-02-10
  perspective corrected.jpg» (Wikimedia Commons) by counting floors: granite base (shop floor + mezzanine,
  0–9 m), a principal floor with balconies, 15 brick floors, a stone floor and the set-back attic of the slab
  (cornice 70 m, attic 73.8 m as OSM); upper tower = a stone floor with balconies, 4 brick floors, a heavy
  cornice, the glazed attic gallery (floor 26, today the sky bar) and a balustrade with paired pinnacles
  (≈ 98.5 m); the lantern with its round oculus to ≈ 105.5 m and the flagpole to the quoted 117 m.
  OSM levels without a height are turned into heights with the same floor grid.
- Colours from the photos of 2014 and 2025 (after the refurbishment): white artificial stone, salmon-red brick
  panels (front centre, wing fronts, all four sides of the upper tower), light granite base, the Baroque stone
  portal in the middle of the front (about 12 m wide, 25 m high).
- Rooftop of the upper tower: terrace with the two glazed pavilions of 2019 (OSM parts with 27 levels).

Axes (docs/MODEL_SPEC.md): +Y up, the Plaza de España façade faces +Z, +X points along the front towards the
south-east (Gran Vía end), origin = centroid of the OSM outline, y = 0 the ground there.

    python model_sources/Q649426_edificio_espana.py   -> models/Q649426/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import (GLASS, Mesh, box, cap, cyl, lathe, merge, offset, quad, ring_area,  # noqa: E402
                     walls, write_glb)

# ── OpenStreetMap way 49031978 and its building:part pieces (© OpenStreetMap contributors, ODbL), in model
#    metres for heading 235.2 and origin 40.4241196, -3.7109543 (tools/osm.py) ──────────────────────────────
OUTLINE = [(-47.06, -22.56), (-52.17, -17.25), (-52.17, -12.76), (-52.17, -3.83), (-52.15, -0.4), (-52.16, 10.74), (-52.15, 11.35), (-52.15, 16.51), (-51.41, 16.84), (-50.58, 17.44), (-49.14, 18.36), (-48.23, 19.24), (-47.23, 20.26), (-45.74, 22.52), (-45.38, 23.28), (-39.44, 23.28), (-27.79, 23.29), (-27.74, 22.29), (-16.43, 22.28), (-0.64, 22.27), (15.85, 22.24), (27.3, 22.24), (27.31, 23.28), (40.36, 23.23), (45.99, 23.21), (46.52, 22.19), (47.74, 20.34), (49.65, 18.4), (51.27, 17.27), (52.74, 16.44), (52.57, 11.34), (52.16, -0.75), (52.04, -3.88), (51.82, -10.39), (51.76, -12.3), (51.72, -13.32), (51.58, -17.3), (46.05, -22.64), (41.48, -22.61), (39.93, -22.6), (37.0, -22.59), (26.99, -22.61), (25.13, -22.61), (13.77, -22.63), (-4.18, -22.57), (-21.9, -22.61), (-41.94, -22.56)]
PARTS = [   # (OSM id, height tag, levels tag, min level, outline x/z)
    ('way/307031389', 42.0, 14, None, [(43.86, 20.23), (44.47, 19.04), (45.16, 17.99), (45.87, 17.12), (47.17, 15.91), (48.38, 15.09), (49.65, 14.49), (49.56, 11.33), (35.34, 11.32), (35.41, 17.49), (28.38, 17.51), (28.37, 19.21), (28.35, 20.27)]),
    ('way/307031391', 42.0, 11, None, [(-29.02, 17.53), (-29.04, 19.23), (-29.02, 20.3), (-44.22, 20.37), (-45.26, 18.79), (-46.4, 17.44), (-47.64, 16.27), (-48.94, 15.25), (-49.95, 14.57), (-49.94, 11.34), (-35.76, 11.38), (-35.73, 17.54)]),
    ('way/307031393', None, 5, None, [(-52.17, -3.83), (-52.17, -12.76), (-52.17, -17.25), (-47.06, -22.56), (-41.94, -22.56), (-21.9, -22.61), (-4.18, -22.57), (13.77, -22.63), (25.13, -22.61), (26.99, -22.61), (26.99, -17.7), (24.33, -17.75), (24.42, -15.56), (14.37, -15.55), (14.36, -13.77), (9.21, -13.73), (9.27, -11.45), (3.8, -11.43), (3.35, -10.47), (3.1, -10.43), (3.07, -9.94), (3.05, -9.6), (-2.75, -9.58), (-2.77, -7.36), (-3.61, -7.35), (-3.63, -9.98), (-3.65, -10.43), (-3.85, -10.43), (-4.34, -11.4), (-14.05, -11.44), (-14.57, -10.42), (-14.8, -10.43), (-14.81, -9.96), (-14.83, -7.34), (-21.82, -7.34), (-21.88, -10.37), (-22.14, -10.46), (-22.64, -11.4), (-32.3, -11.41), (-32.78, -10.4), (-33.08, -10.39), (-33.07, -7.36), (-33.08, -3.85), (-41.78, -3.84), (-50.49, -3.8)]),
    ('way/307031394', 70.0, 21, None, [(-29.02, 17.53), (-35.73, 17.54), (-35.76, 11.38), (-35.77, 10.54), (-33.04, 10.54), (-33.05, 11.35), (-33.06, 11.72), (-29.0, 11.75)]),
    ('way/307031395', 38.03, 10, None, [(27.3, 22.24), (27.31, 23.28), (40.36, 23.23), (40.33, 22.05), (28.37, 22.07), (28.37, 22.27)]),
    ('way/307112147', 70.0, 18, None, [(35.41, 17.49), (35.34, 11.32), (35.36, 10.49), (32.45, 10.49), (32.42, 11.73), (28.39, 11.71), (28.39, 13.06), (28.38, 17.51)]),
    ('way/307125625', 117.0, 29, None, [(1.5, 9.3), (1.52, 12.34), (-2.14, 12.34), (-2.16, 9.34), (-3.6, 9.33), (-3.61, 9.55), (-3.49, 9.61), (-3.51, 14.93), (-3.25, 14.95), (-3.27, 16.12), (-2.76, 16.62), (2.23, 16.64), (2.6, 16.11), (2.62, 14.45), (3.3, 14.43), (3.3, 9.37), (3.07, 9.36)]),
    ('way/307125627', 107.0, 25, None, [(15.83, 19.23), (15.82, 16.77), (13.85, 16.81), (13.86, 13.32), (8.79, 13.32), (8.81, 14.45), (6.47, 14.43), (6.46, 16.8), (-6.48, 16.79), (-6.48, 15.86), (-6.48, 14.97), (-9.44, 14.92), (-9.46, 13.34), (-14.42, 13.36), (-14.42, 15.76), (-15.01, 15.75), (-15.01, 16.89), (-16.4, 16.86), (-16.43, 19.22), (-16.4, 19.84), (15.79, 19.83)]),
    ('way/344007050', 73.8, 20, None, [(22.83, 13.13), (28.39, 13.06), (28.39, 11.71), (32.42, 11.73), (32.45, 10.49), (32.45, -7.33), (21.25, -7.35), (21.23, -4.3), (21.2, 7.56), (21.21, 9.35), (19.49, 9.31), (19.46, 8.72), (18.75, 8.09), (16.58, 8.1), (15.81, 8.76), (15.81, 9.32), (15.82, 16.77), (22.88, 16.79)]),
    ('way/344007051', 73.8, 22, None, [(15.79, 19.83), (-16.4, 19.84), (-16.42, 21.24), (15.82, 21.22)]),
    ('way/344007052', 70.0, 20, None, [(15.85, 22.24), (15.82, 21.22), (-16.42, 21.24), (-16.43, 22.28), (-0.64, 22.27)]),
    ('way/344007053', 70.0, 18, None, [(27.3, 22.24), (15.85, 22.24), (15.82, 21.22), (15.79, 19.83), (15.83, 19.23), (28.37, 19.21), (28.35, 20.27), (28.37, 22.07), (28.37, 22.27)]),
    ('way/344007226', 73.8, 20, None, [(-33.04, 10.54), (-33.05, 11.35), (-28.02, 11.35), (-28.02, 13.35), (-24.01, 13.36), (-23.99, 16.88), (-22.04, 16.87), (-22.1, 9.33), (-21.88, 4.28), (-21.82, -7.34), (-33.07, -7.36), (-33.08, -3.85)]),
    ('way/344007228', 70.0, 21, None, [(-22.05, 19.23), (-29.04, 19.23), (-29.02, 20.3), (-29.02, 22.09), (-29.02, 22.28), (-27.74, 22.29), (-22.02, 22.32)]),
    ('way/344008005', 38.03, 9, None, [(-39.44, 23.28), (-45.38, 23.28), (-45.74, 22.52), (-47.23, 20.26), (-48.23, 19.24), (-49.14, 18.36), (-50.58, 17.44), (-51.41, 16.84), (-52.15, 16.51), (-52.15, 11.35), (-50.59, 11.34), (-50.6, 15.82), (-49.74, 16.29), (-48.77, 16.9), (-47.79, 17.72), (-46.93, 18.5), (-45.94, 19.6), (-45.22, 20.66), (-44.44, 22.09), (-39.41, 22.1)]),
    ('way/344008006', 40.0, 10, None, [(-29.02, 20.3), (-29.02, 22.09), (-39.41, 22.1), (-44.44, 22.09), (-45.22, 20.66), (-45.94, 19.6), (-46.93, 18.5), (-47.79, 17.72), (-48.77, 16.9), (-49.74, 16.29), (-50.6, 15.82), (-50.59, 11.34), (-49.94, 11.34), (-49.95, 14.57), (-48.94, 15.25), (-47.64, 16.27), (-46.4, 17.44), (-45.26, 18.79), (-44.22, 20.37)]),
    ('way/344008007', 40.0, 10, None, [(28.37, 22.07), (40.33, 22.05), (45.26, 22.05), (45.72, 20.97), (46.32, 20.0), (46.99, 19.09), (47.68, 18.3), (48.52, 17.53), (49.42, 16.91), (50.14, 16.41), (51.23, 15.91), (51.08, 11.32), (49.56, 11.33), (49.65, 14.49), (48.38, 15.09), (47.17, 15.91), (45.87, 17.12), (45.16, 17.99), (44.47, 19.04), (43.86, 20.23), (28.35, 20.27)]),
    ('way/344008008', 38.03, 10, None, [(-39.44, 23.28), (-39.41, 22.1), (-29.02, 22.09), (-29.02, 22.28), (-27.74, 22.29), (-27.79, 23.29)]),
    ('way/344008009', 38.03, 9, None, [(40.36, 23.23), (45.99, 23.21), (46.52, 22.19), (47.74, 20.34), (49.65, 18.4), (51.27, 17.27), (52.74, 16.44), (52.57, 11.34), (51.08, 11.32), (51.23, 15.91), (50.14, 16.41), (49.42, 16.91), (48.52, 17.53), (47.68, 18.3), (46.99, 19.09), (46.32, 20.0), (45.72, 20.97), (45.26, 22.05), (40.33, 22.05)]),
    ('way/1341773966', None, 3, 1, [(-52.17, -12.76), (-52.64, -13.2), (-52.98, -13.54), (-53.12, -13.86), (-53.17, -14.34), (-53.1, -17.65), (-52.95, -18.04), (-51.53, -19.45), (-47.41, -23.54), (-47.09, -23.66), (-46.47, -23.7), (-43.57, -23.71), (-42.97, -23.64), (-42.71, -23.38), (-42.25, -22.95), (-41.96, -22.81), (-41.94, -22.56), (-47.06, -22.56), (-52.17, -17.25)]),
    ('way/1341773967', 38.03, 10, None, [(-52.15, 11.35), (-52.16, 10.74), (-52.15, -0.4), (-50.52, -0.4), (-50.59, 11.34)]),
    ('way/1341773968', 38.03, 11, None, [(-52.15, -0.4), (-52.17, -3.83), (-50.49, -3.8), (-50.52, -0.4)]),
    ('way/1341773969', 40.0, 11, None, [(-50.59, 11.34), (-50.52, -0.4), (-49.97, -0.4), (-49.94, 11.34)]),
    ('way/1341773970', 42.0, 1, None, [(-35.76, 11.38), (-35.77, 10.54), (-41.73, 10.55), (-41.78, -3.84), (-50.49, -3.8), (-50.52, -0.4), (-49.97, -0.4), (-49.94, 11.34)]),
    ('way/1341773971', None, 6, None, [(-33.08, -3.85), (-33.04, 10.54), (-35.77, 10.54), (-41.73, 10.55), (-41.78, -3.84)]),
    ('way/1341773972', None, 7, 1, [(-39.44, 23.28), (-39.65, 23.53), (-39.99, 23.57), (-40.08, 23.87), (-40.44, 24.21), (-40.77, 24.3), (-41.04, 24.27), (-45.21, 24.25), (-45.48, 24.19), (-45.89, 23.85), (-46.23, 23.17), (-46.88, 21.9), (-47.41, 21.06), (-47.99, 20.36), (-49.07, 19.27), (-49.93, 18.63), (-51.16, 17.85), (-52.01, 17.51), (-52.58, 17.3), (-52.94, 17.05), (-53.09, 16.8), (-53.16, 16.14), (-53.18, 11.95), (-53.0, 11.51), (-52.67, 11.21), (-52.57, 11.31), (-52.33, 11.08), (-52.16, 10.74), (-52.15, 11.35), (-52.15, 16.51), (-51.41, 16.84), (-50.58, 17.44), (-49.14, 18.36), (-48.23, 19.24), (-47.23, 20.26), (-45.74, 22.52), (-45.38, 23.28)]),
    ('way/1341773973', 73.8, 19, None, [(-33.05, 11.35), (-28.02, 11.35), (-28.02, 13.35), (-24.01, 13.36), (-23.99, 16.88), (-22.04, 16.87), (-22.05, 19.23), (-29.04, 19.23), (-29.02, 17.53), (-29.0, 11.75), (-33.06, 11.72)]),
    ('way/1341773974', 70.0, 21, None, [(-22.05, 19.23), (-16.43, 19.22), (-16.4, 19.84), (-16.42, 21.24), (-16.43, 22.28), (-22.02, 22.32)]),
    ('way/1341773975', 73.8, 19, None, [(-21.82, -7.34), (-21.88, -10.37), (-22.14, -10.46), (-22.64, -11.4), (-32.3, -11.41), (-32.78, -10.4), (-33.08, -10.39), (-33.07, -7.36)]),
    ('way/1341773976', None, 19, None, [(-22.04, 16.87), (-16.4, 16.86), (-16.43, 19.22), (-22.05, 19.23)]),
    ('way/1341773977', None, 20, None, [(-16.4, 16.86), (-16.39, 9.35), (-16.39, 8.76), (-17.23, 8.12), (-19.46, 8.13), (-20.25, 8.72), (-20.25, 9.38), (-22.1, 9.33), (-22.04, 16.87)]),
    ('way/1341773978', None, 5, None, [(40.61, -3.84), (40.04, -19.31), (39.98, -20.82), (36.99, -20.78), (37.0, -22.59), (26.99, -22.61), (26.99, -17.7), (24.33, -17.75), (24.42, -15.56), (14.37, -15.55), (14.36, -13.77), (9.21, -13.73), (9.27, -11.45), (13.56, -11.45), (14.05, -10.51), (14.31, -10.42), (14.31, -9.95), (14.31, -7.33), (21.25, -7.35), (21.22, -9.94), (21.51, -9.93), (21.97, -11.41), (31.64, -11.49), (32.18, -10.47), (32.42, -10.45), (32.45, -7.33), (33.25, -7.33), (33.25, -3.82)]),
    ('way/1341773979', None, 19, None, [(32.45, -7.33), (32.42, -10.45), (32.18, -10.47), (31.64, -11.49), (21.97, -11.41), (21.51, -9.93), (21.22, -9.94), (21.25, -7.35)]),
    ('way/1342396451', None, 19, None, [(15.82, 16.77), (15.83, 19.23), (28.37, 19.21), (28.38, 17.51), (28.39, 13.06), (22.83, 13.13), (22.88, 16.79)]),
    ('way/1342396452', None, 10, None, [(52.57, 11.34), (51.08, 11.32), (50.56, -0.68), (52.16, -0.75)]),
    ('way/1342396453', None, 10, None, [(50.56, -0.68), (50.48, -3.15), (50.45, -3.83), (52.04, -3.88), (52.16, -0.75)]),
    ('way/1342396454', None, 9, None, [(52.04, -3.88), (51.82, -10.39), (50.18, -10.29), (50.34, -6.53), (50.45, -3.83)]),
    ('way/1342396455', None, 9, None, [(50.18, -10.29), (50.09, -13.3), (51.72, -13.32), (51.76, -12.3), (51.82, -10.39)]),
    ('way/1342396456', None, 4, None, [(51.72, -13.32), (51.58, -17.3), (46.05, -22.64), (41.48, -22.61), (39.93, -22.6), (40.01, -21.31), (45.69, -21.24), (49.91, -17.11), (50.09, -13.3)]),
    ('way/1342396457', None, 4, None, [(37.0, -22.59), (39.93, -22.6), (40.01, -21.31), (39.98, -20.82), (36.99, -20.78)]),
    ('way/1342396458', None, 2, 1, [(41.48, -22.61), (41.83, -22.85), (42.13, -22.95), (42.36, -23.18), (42.24, -23.29), (42.55, -23.52), (42.97, -23.59), (46.76, -23.56), (52.83, -17.91), (53.01, -13.87), (52.57, -13.16), (52.44, -13.28), (52.13, -12.91), (52.09, -12.57), (52.09, -12.34), (51.76, -12.3), (51.72, -13.32), (51.58, -17.3), (46.05, -22.64)]),
    ('way/1342396459', None, 11, None, [(49.56, 11.33), (49.14, -0.66), (50.56, -0.68), (51.08, 11.32)]),
    ('way/1342396460', None, 11, None, [(49.14, -0.66), (49.06, -3.1), (50.48, -3.15), (50.56, -0.68)]),
    ('way/1342396461', None, 12, None, [(49.14, -0.66), (46.62, -0.57), (46.52, -3.85), (50.45, -3.83), (50.48, -3.15), (49.06, -3.1)]),
    ('way/1342396462', None, 11, None, [(46.52, -3.85), (46.29, -10.12), (48.8, -10.27), (48.94, -6.47), (50.34, -6.53), (50.45, -3.83)]),
    ('way/1342396463', None, 10, None, [(50.34, -6.53), (50.18, -10.29), (48.8, -10.27), (48.94, -6.47)]),
    ('way/1342396464', None, 10, None, [(50.18, -10.29), (50.09, -13.3), (48.71, -13.33), (48.8, -10.27)]),
    ('way/1342396465', None, 5, None, [(50.09, -13.3), (49.91, -17.11), (45.69, -21.24), (40.01, -21.31), (39.98, -20.82), (40.04, -19.31), (40.78, -19.31), (44.86, -19.26), (48.66, -15.55), (48.71, -13.33)]),
    ('way/1342396466', None, 6, None, [(48.71, -13.33), (48.1, -13.33), (48.08, -15.63), (45.48, -18.35), (40.84, -18.35), (40.78, -19.31), (44.86, -19.26), (48.66, -15.55)]),
    ('way/1342396467', None, 11, None, [(46.52, -3.85), (40.61, -3.84), (40.04, -19.31), (40.78, -19.31), (40.84, -18.35), (45.48, -18.35), (48.08, -15.63), (48.1, -13.33), (48.71, -13.33), (48.8, -10.27), (46.29, -10.12)]),
    ('way/1342396468', None, 12, None, [(40.61, -3.84), (41.13, 10.44), (35.36, 10.49), (35.34, 11.32), (49.56, 11.33), (49.14, -0.66), (46.62, -0.57), (46.52, -3.85)]),
    ('way/1342396469', None, None, None, [(32.45, -7.33), (32.45, 10.49), (35.36, 10.49), (41.13, 10.44), (40.61, -3.84), (33.25, -3.82), (33.25, -7.33)]),
    ('way/1342396470', None, 26, None, [(-16.4, 16.86), (-16.39, 9.35), (-14.82, 9.38), (-14.81, 4.33), (-14.35, 4.33), (-14.34, 7.5), (-6.66, 7.53), (-6.65, 9.36), (-6.65, 10.54), (-9.47, 10.57), (-9.46, 13.34), (-14.42, 13.36), (-14.42, 15.76), (-15.01, 15.75), (-15.01, 16.89)]),
    ('way/1342396471', None, 27, None, [(-6.65, 10.54), (-6.5, 10.56), (-6.5, 13.34), (-9.46, 13.34), (-9.47, 10.57)]),
    ('way/1342396472', None, 26, None, [(-9.46, 13.34), (-9.44, 14.92), (-6.48, 14.97), (-6.5, 13.34)]),
    ('way/1342396474', None, 27, None, [(-3.61, 0.6), (-3.62, -3.4), (-5.8, -3.42), (-5.82, 0.59)]),
    ('way/1342396475', None, 25, None, [(-13.96, -7.32), (-13.97, -9.96), (-14.81, -9.96), (-14.83, -7.34)]),
    ('way/1342396476', None, 20, None, [(-14.81, -9.96), (-14.8, -10.43), (-14.57, -10.42), (-14.05, -11.44), (-4.34, -11.4), (-3.85, -10.43), (-3.65, -10.43), (-3.63, -9.98), (-9.55, -9.96), (-9.57, -11.06), (-13.94, -11.07), (-13.97, -9.96)]),
    ('way/1342396477', None, 20, None, [(-13.97, -9.96), (-9.55, -9.96), (-9.57, -11.06), (-13.94, -11.07)]),
    ('way/1342396478', None, 25, None, [(-13.97, -9.96), (-13.96, -7.32), (-9.54, -7.35), (-9.55, -9.96)]),
    ('way/1342396479', None, 25, None, [(-9.55, -9.96), (-3.63, -9.98), (-3.61, -7.35), (-9.54, -7.35)]),
    ('way/1342396480', None, 26, None, [(-13.99, 0.29), (-13.96, -7.32), (-9.54, -7.35), (-9.56, 0.31)]),
    ('way/1342396481', None, 28, None, [(-6.66, 7.53), (-3.6, 7.56), (-3.6, 9.33), (-6.65, 9.36)]),
    ('way/1342396482', None, 6, None, [(-21.82, -7.34), (-21.88, 4.28), (-14.81, 4.33), (-14.83, -7.34)]),
    ('way/1342396483', None, 6, None, [(-21.88, 4.28), (-22.1, 9.33), (-20.25, 9.38), (-20.25, 8.72), (-19.46, 8.13), (-17.23, 8.12), (-16.39, 8.76), (-16.39, 9.35), (-14.82, 9.38), (-14.81, 4.33)]),
    ('way/1342396484', None, 29, None, [(-6.65, 9.36), (-3.6, 9.33), (-3.61, 9.55), (-3.49, 9.61), (-3.51, 14.93), (-3.5, 15.88), (-6.48, 15.86), (-6.48, 14.97), (-6.5, 13.34), (-6.5, 10.56), (-6.65, 10.54)]),
    ('way/1342396485', None, 26, None, [(-6.48, 15.86), (-6.48, 16.79), (6.46, 16.8), (6.47, 14.43), (6.29, 14.46), (6.31, 15.87), (3.31, 15.87), (3.3, 14.43), (2.62, 14.45), (2.6, 16.11), (2.23, 16.64), (-2.76, 16.62), (-3.27, 16.12), (-3.25, 14.95), (-3.51, 14.93), (-3.5, 15.88)]),
    ('way/1342396486', None, 28, None, [(-2.14, 12.34), (-2.16, 9.34), (-2.16, 8.76), (-1.43, 8.11), (0.71, 8.11), (1.5, 8.72), (1.5, 9.3), (1.52, 12.34)]),
    ('way/1342396487', None, 29, None, [(6.31, 15.87), (3.31, 15.87), (3.3, 14.43), (3.3, 9.37), (5.97, 9.31), (5.98, 10.64), (6.0, 14.45), (6.29, 14.46)]),
    ('way/1342396488', None, 27, None, [(6.0, 14.45), (6.29, 14.46), (6.47, 14.43), (8.81, 14.45), (8.79, 13.32), (8.81, 10.68), (5.98, 10.64)]),
    ('way/1342396489', None, 28, None, [(5.97, 9.31), (6.0, 7.52), (3.08, 7.5), (3.07, 9.36), (3.3, 9.37)]),
    ('way/1342396490', None, 6, None, [(3.07, 9.36), (3.08, 7.5), (-3.6, 7.56), (-3.6, 9.33), (-2.16, 9.34), (-2.16, 8.76), (-1.43, 8.11), (0.71, 8.11), (1.5, 8.72), (1.5, 9.3)]),
    ('way/1342396491', None, 6, None, [(-3.6, 7.56), (-3.59, 3.43), (-2.42, 3.37), (-2.4, 0.87), (1.92, 0.86), (1.92, 3.39), (3.05, 3.4), (3.08, 7.5)]),
    ('way/1342396492', None, 6, None, [(-2.78, 0.87), (-2.4, 0.87), (1.92, 0.86), (3.24, 0.86), (3.25, -2.81), (3.27, -7.34), (-2.77, -7.36)]),
    ('way/1342396493', None, 5, None, [(-2.77, -7.36), (-2.75, -9.58), (3.05, -9.6), (3.24, -9.6), (3.25, -8.47), (3.27, -7.34)]),
    ('way/1342396494', None, 26, None, [(15.82, 16.77), (13.85, 16.81), (13.86, 13.32), (8.79, 13.32), (8.81, 10.68), (5.98, 10.64), (5.97, 9.31), (6.0, 7.52), (14.29, 7.51), (14.29, 9.35), (15.81, 9.32)]),
    ('way/1342396495', None, 6, None, [(21.21, 9.35), (21.2, 7.56), (14.29, 7.51), (14.29, 9.35), (15.81, 9.32), (15.81, 8.76), (16.58, 8.1), (18.75, 8.09), (19.46, 8.72), (19.49, 9.31)]),
    ('way/1342396496', None, 6, None, [(21.2, 7.56), (21.23, -4.3), (14.27, -4.31), (14.29, 7.51)]),
    ('way/1342396497', None, 6, None, [(21.23, -4.3), (21.25, -7.35), (14.31, -7.33), (14.27, -4.31)]),
    ('way/1342396498', None, 20, None, [(14.31, -9.95), (9.21, -10.0), (9.27, -11.45), (13.56, -11.45), (14.05, -10.51), (14.31, -10.42)]),
    ('way/1342396499', None, 20, None, [(9.21, -10.0), (3.07, -9.94), (3.1, -10.43), (3.35, -10.47), (3.8, -11.43), (9.27, -11.45)]),
    ('way/1342396500', None, 26, None, [(14.29, 7.51), (14.27, -4.31), (13.82, -4.3), (13.8, 0.43), (9.27, 0.41), (9.26, -0.94), (7.38, -1.02), (7.37, 0.0), (5.63, -0.01), (5.6, -2.78), (3.25, -2.81), (3.24, 0.86), (1.92, 0.86), (1.92, 3.39), (3.05, 3.4), (3.08, 7.5), (6.0, 7.52)]),
    ('way/1342396501', None, 27, None, [(9.26, -0.94), (9.22, -7.35), (3.27, -7.34), (3.25, -2.81), (5.6, -2.78), (5.63, -0.01), (7.37, 0.0), (7.38, -1.02)]),
    ('way/1342396502', None, 26, None, [(9.26, -0.94), (12.7, -0.97), (12.72, -7.38), (14.31, -7.33), (14.27, -4.31), (13.82, -4.3), (13.8, 0.43), (9.27, 0.41)]),
    ('way/1342396503', None, 27, None, [(9.26, -0.94), (9.22, -7.35), (12.72, -7.38), (12.7, -0.97)]),
    ('way/1342396504', None, 26, None, [(12.72, -7.38), (12.66, -8.43), (9.24, -8.47), (9.22, -7.35)]),
    ('way/1342396505', None, 26, None, [(9.22, -7.35), (3.27, -7.34), (3.25, -8.47), (9.24, -8.47)]),
    ('way/1342396506', None, 25, None, [(3.25, -8.47), (3.24, -9.6), (3.05, -9.6), (3.07, -9.94), (9.21, -10.0), (9.24, -8.47)]),
    ('way/1342396507', None, 25, None, [(9.21, -10.0), (14.31, -9.95), (14.31, -7.33), (12.72, -7.38), (12.66, -8.43), (9.24, -8.47)]),
    ('relation/18417426', None, 26, None, [(-14.34, 7.5), (-6.66, 7.53), (-3.6, 7.56), (-3.59, 3.43), (-2.42, 3.37), (-2.4, 0.87), (-2.78, 0.87), (-2.77, -7.36), (-3.61, -7.35), (-9.54, -7.35), (-9.56, 0.31), (-13.99, 0.29), (-13.96, -7.32), (-14.83, -7.34), (-14.81, 4.33), (-14.35, 4.33)]),
]


# ── Colours and constants ────────────────────────────────────────────────────
STONE = np.array((0.915, 0.895, 0.835))        # white artificial stone (as restored)
STONE_D = np.array((0.84, 0.82, 0.76))         # cornices in shade, portal relief
BRICK = np.array((0.78, 0.47, 0.37))           # salmon-red brick panels
GRANITE = np.array((0.75, 0.735, 0.70))        # the base
ROOF = (0.66, 0.655, 0.63)                     # flat roofs, terraces
GLASS_C = (0.20, 0.25, 0.30)                   # windows
GLASS_P = (0.36, 0.45, 0.52)                   # 2019 rooftop pavilions
DOOR = (0.16, 0.15, 0.14)
METAL_C = (0.45, 0.47, 0.50)
FOUND = -6.0                                   # foundations (the plaza is lower than the back streets)

# vertical grid (m): base (shop floor + mezzanine), then the slab floors, then the upper tower
BASE_TOP = 9.0
P1 = 3.55                                      # floors 3–20
F20 = BASE_TOP + 18 * P1                       # 72.9: top of floor 20
P2 = 3.7                                       # floors 21–25 of the upper tower
F25 = F20 + 5 * P2                             # 91.4
UT_ROOF = 97.4                                 # upper tower: roof over the attic gallery (floor 26)
LANT_TOP = 105.5
MAST_TOP = 117.0
UT_X0, UT_X1, UT_Z0, UT_Z1 = -16.43, 15.85, -11.45, 19.8   # upper tower box (OSM parts)
XC = (UT_X0 + UT_X1) / 2                       # axis of symmetry of the front
PITCH = 3.2                                    # window columns (9 on the brick panels, photo)

SKIP = {"way/1341773966", "way/1341773972", "way/1342396458"}       # overhangs with min_level (balconies)
PAVILIONS = {"way/1342396471", "way/1342396488"}                    # glazed rooftop pavilions (2019)
rng = np.random.default_rng(1953)


def level_height(lv):
    """Top of a part with lv levels and no height tag, on the floor grid measured on the photos."""
    if lv <= 20:
        return BASE_TOP + (lv - 2) * P1
    return {25: F25 + 1.2, 26: UT_ROOF, 27: UT_ROOF + 3.2, 28: UT_ROOF + 2.6, 29: UT_ROOF + 4.4}[min(lv, 29)]


def part_height(pid, h, lv):
    if pid == "way/307125625":                      # the lantern (OSM 117 = with its flagpole)
        return LANT_TOP
    if pid == "way/307125627":                      # front of the upper tower (OSM 107)
        return UT_ROOF
    if h is not None:
        return h
    if lv is None:
        return level_height(6)
    return level_height(lv)


# ── Geometry helpers ─────────────────────────────────────────────────────────
def ccw(P):
    P = [tuple(p) for p in P]
    return P if ring_area(P) > 0 else P[::-1]


def pip(pts, poly):
    """Even-odd point in polygon for many points (N,2)."""
    P = np.asarray(poly, float)
    x, z = pts[:, 0][:, None], pts[:, 1][:, None]
    xi, zi = P[:, 0][None, :], P[:, 1][None, :]
    xj, zj = np.roll(P[:, 0], 1)[None, :], np.roll(P[:, 1], 1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        hit = ((zi > z) != (zj > z)) & (x < (xj - xi) * (z - zi) / (zj - zi) + xi)
    return (np.sum(hit, 1) % 2) == 1


class Solids:
    """The prisms of the massing, to know which bits of wall are open to the air."""

    def __init__(self):
        self.items = []                              # (poly, top)

    def add(self, poly, top):
        self.items.append((np.asarray(poly, float), float(top)))

    def hidden(self, pts, ys, skip=None):
        pts = np.asarray(pts, float).reshape(-1, 2); ys = np.asarray(ys, float).ravel()
        occ = np.zeros(len(pts), bool)
        for k, (poly, top) in enumerate(self.items):
            if k == skip:
                continue
            m = (ys < top) & ~occ
            if m.any():
                idx = np.where(m)[0]
                occ[idx[pip(pts[idx], poly)]] = True
        return occ


def edges(P):
    """(a, b, unit direction, outward normal, length) of a counter-clockwise ring, merging straight runs."""
    P = [np.array(p, float) for p in P]
    keep = []
    for i in range(len(P)):
        d0 = P[i] - P[i - 1]; d1 = P[(i + 1) % len(P)] - P[i]
        l0, l1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if l0 < 1e-6 or l1 < 1e-6:
            continue
        if abs(d0[0] * d1[1] - d0[1] * d1[0]) / (l0 * l1) > 0.012 or np.dot(d0, d1) < 0:
            keep.append(P[i])
    out = []
    for i in range(len(keep)):
        a, b = keep[i], keep[(i + 1) % len(keep)]
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        e = (b - a) / L
        out.append((a, b, e, np.array([e[1], -e[0]]), L))
    return out


def wall_quad(a, e, n, s0, s1, y0, y1, off, c):
    """A flat rectangle on a wall, from s0 to s1 along the edge, off metres in front of it, facing out."""
    p0 = a + e * s0 + n * off; p1 = a + e * s1 + n * off
    V = [(p0[0], y0, p0[1]), (p1[0], y0, p1[1]), (p1[0], y1, p1[1]), (p0[0], y1, p0[1])]
    return Mesh(V, [(0, 2, 1), (0, 3, 2)], c)


def edge_box(a, e, n, s0, s1, y0, y1, o0, o1, c):
    """A box lying along a wall: from s0 to s1 along it, o0..o1 metres out of it (negative = into the wall)."""
    w, d = s1 - s0, o1 - o0
    m = box(w, y1 - y0, d, c)
    m.R(0, math.atan2(-e[1], e[0]), 0)
    cx = a + e * (s0 + s1) / 2 + n * (o0 + o1) / 2
    return m.T(cx[0], y0, cx[1])


def runs(flags):
    """[(i0, i1)] runs of True in a boolean list (i1 exclusive)."""
    out, i0 = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and i0 is None:
            i0 = i
        elif not f and i0 is not None:
            out.append((i0, i)); i0 = None
    return out


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sphere(r, c, seg=8, rings=5):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def pinnacle(x, z, y, h, c=STONE, w=0.9):
    """Baroque pinnacle: pedestal, obelisk, ball."""
    out = [B(x - w / 2, x + w / 2, y, y + 0.28 * h, z - w / 2, z + w / 2, c),
           B(x - w * 0.6, x + w * 0.6, y + 0.28 * h, y + 0.33 * h, z - w * 0.6, z + w * 0.6, STONE_D),
           cyl(w * 0.42, w * 0.08, 0.55 * h, c, 4).R(0, math.pi / 4, 0).T(x, y + 0.33 * h, z),
           sphere(w * 0.16, c, 6, 4).T(x, y + 0.92 * h, z)]
    return out


# ── The model ────────────────────────────────────────────────────────────────
def build():
    stone, glass = [], []
    parts = [(pid, part_height(pid, h, lv), ccw(P)) for pid, h, lv, ml, P in PARTS if pid not in SKIP and not ml]
    parts.append(("outline", level_height(4) - 1.0, ccw(offset(ccw(OUTLINE), -0.4))))   # filler, inside the façades

    # rank by height: taller prisms are grown a little so that no two walls share a plane, and equal roofs
    # are lifted by millimetres so that no two roofs share one
    tops = sorted({round(t, 2) for _, t, _ in parts})
    solids = Solids()
    geo = []
    seen = {}
    for pid, top, P in parts:
        r = tops.index(round(top, 2))
        k = seen.get(round(top, 2), 0); seen[round(top, 2)] = k + 1
        G = offset(P, 0.015 * r + 0.0075 * (k % 2)) if r else P
        t = top + 0.004 * k
        geo.append((pid, t, ccw(G)))
        solids.add(P, top)

    for i, (pid, top, G) in enumerate(geo):
        glazed = pid in PAVILIONS
        wall_c = GLASS_P if glazed else STONE
        stone_or_glass = glass if glazed else stone
        stone_or_glass.append(walls(G, FOUND, top, wall_c))
        stone_or_glass.append(cap(G, top, ROOF if not glazed else (0.55, 0.57, 0.58)))

    # façades: base, brick panels, windows, cornices, along every wall open to the air
    for i, (pid, top, G) in enumerate(geo):
        if pid == "outline" or pid in PAVILIONS:
            continue
        upper = top > F20 + 1.5
        lantern = top > UT_ROOF + 1.0
        EG = edges(G)
        for ei, (a, b, e, n, L) in enumerate(EG):
            kk = 2 if (ei == len(EG) - 1 and len(EG) % 2) else ei % 2
            if L < 0.9:
                continue
            # 1. cornice + parapet on the open stretches of the roof edge
            ns = max(1, int(round(L / 0.8)))
            ss = (np.arange(ns) + 0.5) * L / ns
            pts = a + np.outer(ss, e) + n * 0.35
            open_top = ~solids.hidden(pts, np.full(ns, top - 0.35))
            cont = [bool(solids.hidden((end + e * sg * 0.6 - n * 0.3)[None], np.array([top - 0.35]))[0])
                    for end, sg in ((a, -1), (b, 1))]           # the roof edge goes on past this end
            for i0, i1 in runs(open_top):
                s0, s1 = i0 * L / ns, i1 * L / ns
                s0 -= 0.6 if (i0 == 0 and not cont[0]) else 0.0; s1 += 0.6 if (i1 == ns and not cont[1]) else 0.0
                lift = 0.013 * kk + 0.004 * (i % 2)
                if lantern:
                    stone.append(edge_box(a, e, n, s0, s1, top - 0.6 - lift, top + 0.15 + lift, -0.2, 0.35, STONE_D))
                    continue
                proj = 0.85 if upper else 0.6
                stone.append(edge_box(a, e, n, s0, s1, top - 0.85 - lift, top + 0.2 + lift, -0.25, proj, STONE_D))
                if top > 30:
                    stone.append(edge_box(a, e, n, s0 + 0.3 * (i0 == 0), s1 - 0.3 * (i1 == ns), top + 0.2 + lift,
                                          top + 1.15 + lift, -0.05, 0.25, STONE))
            if lantern:
                continue
            # 2. columns of windows
            m = []
            for end, sg in ((a, -1), (b, 1)):                      # does the façade go on past this end?
                q = end + e * sg * 0.6 - n * 0.3
                m.append(0.0 if solids.hidden(q[None], np.array([top - 2.0]))[0] else 0.75)
            s_w = window_columns(a, e, L, *m)
            if not len(s_w):
                continue
            rows = []                                              # (y0, y1, width, kind)
            rows.append((0.45, 4.3, 2.6, "shop")); rows.append((5.9, 7.9, 1.3, "mezz"))
            for k in range(18):
                y0 = BASE_TOP + k * P1
                rows.append((y0 + 0.8, y0 + 2.7, 1.25, "std"))
            for k in range(5):
                y0 = F20 + k * P2
                rows.append((y0 + 0.85, y0 + 2.85, 1.3, "up"))
            rows.append((F25 + 1.9, F25 + 4.9, 2.7, "attic"))
            rows = [r for r in rows if r[1] < top - 0.7]
            cand = [(s, r) for r in rows for s in s_w]
            if not cand:
                continue
            cp = np.array([a + e * s + n * 0.4 for s, r in cand])
            cy = np.array([(r[0] + r[1]) / 2 for s, r in cand])
            vis = ~solids.hidden(cp, cy)
            for (s, r), v in zip(cand, vis):
                if not v:
                    continue
                y0, y1, w, kind = r
                if kind == "shop":
                    if abs(n[1]) > 0.9 and n[1] > 0 and abs((a + e * s)[0]) < 7.2:
                        continue                                    # the portal
                    glass.append(wall_quad(a, e, n, s - w / 2, s + w / 2, y0, y1, 0.06, DOOR))
                    continue
                c = tuple(np.clip(np.array(GLASS_C) * (1 + rng.normal(0, 0.08)), 0, 1))
                if brick_at(a + e * s, n, (y0 + y1) / 2, pid):
                    stone.append(wall_quad(a, e, n, s - w / 2 - 0.22, s + w / 2 + 0.22, y0 - 0.25, y1 + 0.2, 0.04, STONE))
                glass.append(wall_quad(a, e, n, s - w / 2, s + w / 2, y0, y1, 0.06, c))
            # 3. brick panels and the granite base
            ns = max(1, int(round(L / 0.5)))
            ss = (np.arange(ns) + 0.5) * L / ns
            pts = a + np.outer(ss, e) + n * 0.4
            for y0, y1, f in BRICK_REGIONS:
                y1 = min(y1, top - 1.0)
                if y1 <= y0:
                    continue
                ym = np.full(ns, (y0 + y1) / 2)
                ok = ~solids.hidden(pts, ym) & np.array([bool(f(a + e * s, n)) for s in ss])
                for i0, i1 in runs(ok):
                    stone.append(wall_quad(a, e, n, i0 * L / ns, i1 * L / ns, y0, y1, 0.02,
                                           tuple(BRICK * (1 + rng.normal(0, 0.02)))))
            okb = ~solids.hidden(pts, np.full(ns, 4.0))
            for i0, i1 in runs(okb):
                stone.append(wall_quad(a, e, n, i0 * L / ns, i1 * L / ns, FOUND, BASE_TOP, 0.02, tuple(GRANITE)))
                lift2 = 0.013 * kk
                stone.append(edge_box(a, e, n, i0 * L / ns - 0.45 * (i0 == 0), i1 * L / ns + 0.45 * (i1 == ns), BASE_TOP - 0.15 - lift2, BASE_TOP + 0.35 + lift2, -0.1, 0.45,
                                      STONE_D))
    return stone, glass, {pid: G for pid, _, G in geo}


def window_columns(a, e, L, m0=0.75, m1=0.75):
    """Positions of the window columns along a wall: on the façade grid (XC + k·PITCH along x, k·PITCH along
    z) for straight walls, so that the columns line up across the OSM pieces; evenly spread on curved ones."""
    for ax, c0 in ((0, XC), (1, 0.0)):
        if abs(e[ax]) > 0.995:
            u0, u1 = a[ax], a[ax] + e[ax] * L
            if e[ax] > 0:
                k0, k1 = math.ceil((u0 + m0 - c0) / PITCH - 1e-6), math.floor((u1 - m1 - c0) / PITCH + 1e-6)
            else:
                k0, k1 = math.ceil((u1 + m1 - c0) / PITCH - 1e-6), math.floor((u0 - m0 - c0) / PITCH + 1e-6)
            us = c0 + np.arange(k0, k1 + 1) * PITCH
            ss = (us - a[ax]) / e[ax]
            return ss[(ss > 0.05) & (ss < L - 0.05)] if m0 == 0 or m1 == 0 else ss
    nw = int((L - 1.1) // PITCH) + 1 if L >= 2.2 else 0
    return (L - (nw - 1) * PITCH) / 2 + np.arange(max(nw, 0)) * PITCH


def in_upper_tower(p):
    return UT_X0 - 0.5 < p[0] < UT_X1 + 0.5 and UT_Z0 - 0.5 < p[1] < UT_Z1 + 0.5


def _ut_panel(p, n):
    if abs(n[1]) > 0.7:
        return UT_X0 + 2.2 < p[0] < UT_X1 - 2.2
    return UT_Z0 + 2.2 < p[1] < UT_Z1 - 2.2


BRICK_REGIONS = [   # (y0, y1, where): the brick panels seen on the photos; the rest is white stone
    (BASE_TOP + P1 - 0.3, BASE_TOP + 16 * P1 - 0.2,                     # the central panel of the slab
     lambda p, n: n[1] > 0.9 and abs(p[0] - XC) < 14.4 and p[1] > 21.5),
    (BASE_TOP + P1 - 0.3, BASE_TOP + 7 * P1 - 0.2,                      # the fronts of the wings
     lambda p, n: n[1] > 0.9 and 29.0 < abs(p[0] - XC) < 39.8 and p[1] > 21.5),
    (F20 + P2 - 0.25, F25 - 0.15,                                       # the four sides of the upper tower
     lambda p, n: in_upper_tower(p) and _ut_panel(p, n)),
]


def brick_at(p, n, y, pid=None):
    return any(y0 < y < y1 and f(p, n) for y0, y1, f in BRICK_REGIONS)


def portal():
    """The Baroque stone portal in the middle of the front (z of the façade 22.27)."""
    zf = 22.27
    c, d = STONE, STONE_D
    out = [B(-6.2, 6.2, FOUND, 9.4, zf - 0.2, zf + 0.9, c),                     # lower body
           B(-6.6, 6.6, 8.6, 9.6, zf - 0.2, zf + 1.3, d),                         # entablature
           B(-4.6, 4.6, 9.6, 20.0, zf - 0.2, zf + 0.6, c),                        # upper body
           B(-4.9, 4.9, 19.4, 20.4, zf - 0.2, zf + 0.85, d),
           B(-3.4, 3.4, 20.4, 23.0, zf - 0.2, zf + 0.5, c),                       # crowning piece
           B(-1.2, 1.2, 23.0, 24.4, zf - 0.2, zf + 0.45, c)]
    out += pinnacle(0, zf + 0.2, 24.4, 2.2, c, 0.7)
    for sx in (-1, 1):
        out += pinnacle(sx * 4.2, zf + 0.35, 20.4, 2.4, c, 0.6)
        for x in (2.9, 5.0):                                                       # columns of the doorway
            out.append(cyl(0.42, 0.38, 7.4, c, 12).T(sx * x, 0.9, zf + 1.35))
            out.append(B(sx * x - 0.55, sx * x + 0.55, 0, 0.9, zf + 0.8, zf + 1.9, d))
            out.append(B(sx * x - 0.55, sx * x + 0.55, 8.3, 8.6, zf + 0.8, zf + 1.9, d))
        out.append(B(sx * 3.3 - 0.35, sx * 3.3 + 0.35, 9.6, 19.4, zf + 0.6, zf + 0.95, d))   # pilasters above
    # the arched doorway and the windows of the portal
    door = [quad(0, 3.4, zf + 0.9, 3.6, 6.8, "+z", DOOR, 0.03)]
    door.append(cyl(1.8, 1.8, 0.05, DOOR, 16, sx=1.0).R(math.pi / 2, 0, 0).T(0, 6.8, zf + 0.95))
    door.append(quad(0, 13.0, zf + 0.6, 2.2, 3.4, "+z", GLASS_C, 0.04))
    door.append(quad(0, 17.4, zf + 0.6, 1.8, 2.0, "+z", GLASS_C, 0.04))
    door.append(B(-1.6, 1.6, 10.6, 10.9, zf + 0.6, zf + 1.4, d))                   # balcony
    door.append(cyl(0.9, 0.9, 0.05, d, 14).R(math.pi / 2, 0, 0).T(0, 21.6, zf + 0.55))   # cartouche
    return out, door


def roof_at(x, z):
    """Height of the roof at a point (the tallest OSM piece there)."""
    t = [part_height(pid, h, lv) for pid, h, lv, ml, P in PARTS
         if pid not in SKIP and not ml and pip(np.array([[x, z]]), ccw(P))[0]]
    return max(t) if t else 0.0


def pin_on_roof(x, z, h, w=0.9):
    """A pinnacle standing on the roof beside the parapet, its pedestal rising above the parapet."""
    y = roof_at(x, z)
    return pinnacle(x, z, y, h + 1.15, STONE, w)


def tops_and_lantern(geo):
    out, gl = [], []
    # paired pinnacles on the front corners of the upper tower, single ones at the back, urns along the front
    for x in (UT_X0 + 0.6, UT_X0 + 1.8, UT_X1 - 1.8, UT_X1 - 0.6):
        out += pin_on_roof(x, UT_Z1 - 0.6, 3.6, 0.75)
    for x in (-13.9, 13.4):
        out += pin_on_roof(x, -9.4, 3.0, 0.7)
    for x in np.linspace(UT_X0 + 4.5, UT_X1 - 4.5, 7):
        if abs(x - XC) > 4:
            out += pin_on_roof(x, UT_Z1 - 0.6, 1.4, 0.5)
    # the slab: pinnacles beside the upper tower, on the front corners and on the side bodies; the wings
    for sx in (-1, 1):
        for x, z, h in ((15.0, 20.6, 3.2), (27.2, 21.8, 3.0), (35.0, 16.9, 3.0), (31.9, -6.7, 3.0),
                        (29.8, 19.8, 2.6), (47.5, 19.3, 2.6), (39.7, 22.6, 2.4)):
            out += pin_on_roof(XC + sx * x, z, h)
    # lantern: finials, oculus, flagpole
    for x in (-2.9, 2.6):
        for z in (10.0, 15.9):
            out += pinnacle(x, z, LANT_TOP + 0.15, 2.0, STONE, 0.55)
    for x in (-6.1, 5.7):
        out += pinnacle(x, 15.3, UT_ROOF + 4.4 + 0.15, 1.6, STONE, 0.5)
    zl = max(z for x, z in geo['way/307125625'])
    gl.append(cyl(0.75, 0.75, 0.05, GLASS_C, 14).R(math.pi / 2, 0, 0).T(-0.15, LANT_TOP - 2.6, zl + 0.02))
    out.append(cyl(0.16, 0.07, MAST_TOP - LANT_TOP, METAL_C, 6).T(-0.15, LANT_TOP, 13.0))
    return out, gl


def main():
    out = ROOT / "models" / "Q649426"
    out.mkdir(parents=True, exist_ok=True)
    stone, glass, geo = build()
    p, d = portal()
    t, tg = tops_and_lantern(geo)
    m_stone = merge(stone + p + t)
    m_glass = merge(glass + d + tg)
    tris, size = write_glb(out / "model.glb", [("edificio_espana", m_stone), ("glass", m_glass, {"material": "glass"})],
                           roughness=0.9, materials={"glass": GLASS})
    allv = np.vstack([m_stone.v, m_glass.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
