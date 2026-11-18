"""
Basílica de la Sagrada Família (Q48435), Barcelona — the basilica as it stands in late 2026, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Francisco de Paula del Villar (crypt, 1882), Antoni Gaudí (1883–1926, died 1926), then his
successors (Domènec Sugrañes, Francesc Quintana, Isidre Puig i Boada, Lluís Bonet i Garí, Jordi Bonet i
Armengol, Jordi Faulí i Oller and the team of the Junta Constructora). Still being built: protected (see
meta.json).

What is modelled (late 2026):
- The Latin-cross church: the five-aisled nave (5 bays of 7.5 m) with its stepped section (outer aisles,
  clerestory, steep central roof with a row of lanterns), the three-aisled transept, the crossing, the apse
  with its ring of seven chapels and their neo-Gothic pinnacles, the cloister ring round the apse, the west
  sacristy with its ribbed cupola, the cloister wing on the Passion side.
- The 14 finished towers: the Tower of Jesus Christ (172.5 m, finished on 20 February 2026, with its white
  four-armed cross), the Tower of the Virgin Mary (138 m, with its 12-pointed glass star, 2021), the four
  towers of the Evangelists (135 m, 2022, with their symbols on top), the four bell towers of the Nativity
  façade (north-east) and the four of the Passion façade (south-west), with their spiral openings and their
  coloured Venetian-glass mosaic finials (mitre, balls, cross).
- The Nativity façade (Gaudí's, the richest sculpture): three portals with sculpted reliefs, the green
  cypress with its white doves over the Portal of Charity, the two columns on turtles. The Passion façade:
  the porch on six leaning bone-like columns, the frieze of bones, the bridge with the Risen Christ between
  the central towers.
- The Glory façade (south-east, the main one, still being built): the end wall of the nave and the grey
  concrete structure of its portico, simplified; the four Glory towers and the great stair do not exist yet.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap relation 9194723 (wikidata Q48435), its outline (441 points, simplified to 258 within
  0.25 m) and building:parts: the bell towers (ways 661321474, 1207779425–1207779428, 1207779447/48/69), the
  Evangelists (relations 16329171–16329174), the Virgin Mary (way 1207779418), the Jesus Christ tower (way
  359084982 and its stages 359084980/83/90/91/93), the crossing stages (ways 359084987, 359084975, relation
  16329175), the west sacristy (relation 16329170), the cloister wing (relation 16329178), the portico columns
  (the small rings of the outline). In model x/z for heading 134.0°: the rows of four bell towers on each
  façade run at 313.3–314.7° (Passion and Nativity), the nave walls at 314.4°: the axis is 314.0°/134.0°.
- es/en.wikipedia and sagradafamilia.org: 90 m long and 60 m wide inside, nave 45 m wide (five aisles,
  7.5 m grid), central nave vault 45 m, side vaults 30 m, crossing 60 m, apse vault 75 m, main floor ~4 m
  above the street; Tower of Jesus Christ 172.5 m (final piece 20 Feb 2026, blessed 10 Jun 2026), cross 17 m
  tall and 13.5 m wide; Virgin Mary 138 m with a 7 m star; Evangelists 135 m; the bell towers 98–112 m
  (OSM: Nativity 98/107/107/98, Passion 107/112/112/107).
- Estimated from photos (±10 %): outer aisle walls 33 m, clerestory 50 m, nave and transept ridges 62 m with
  lanterns to 68 m; crossing stages 60 m (OSM), 72 m and 84 m (OSM says 80 and 100: lowered, the Evangelist
  shafts stand free above the nave roofs); apse chapels 27 m with pinnacles to 45 m, apse body 48 m; the
  Nativity portals 46 m (Charity) and 38 m, the cypress to 62 m; the Passion porch at 24–27 m with the bones to
  37 m; sacristy walls 16 m, cupola to 30 m (OSM), pinnacle to 38 m. Finials of the bell towers ~22 m.
- Ground: the Eixample is nearly flat here (~1–2 m down to the south-east): default levelling, foundations
  to −3 m.
Colours: warm ochre Montjuïc sandstone, darkened, on the Nativity façade and the apse (the oldest parts);
lighter beige-grey stone on the Passion façade and the nave; grey newer stone and white glazed ceramic on the
central towers; red, gold and white mosaic finials; green cypress; grey concrete on the Glory works.

Axes (docs/MODEL_SPEC.md): +Y up, the main (Glory, south-east, towards Carrer de Mallorca) façade faces +Z
(heading 134.0°), +X points to 44° (the Nativity façade, Carrer de la Marina), −X to the Passion façade
(Carrer de Sardenya), −Z to the apse (Carrer de Provença). Origin on the ground under the centre of the Tower
of Jesus Christ (the crossing).

    python model_sources/Q48435_sagrada_familia.py   -> models/Q48435/model.glb (--footprint prints the footprint)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import (Mesh, merge, box, cyl, lathe, extrude, triangulate, offset, fbm, ring_area,  # noqa: E402
                     write_glb, GLASS as GLASS_MAT, METAL)

rng = np.random.default_rng(1882)

# ── Colours ──────────────────────────────────────────────────────────────────
OCHRE_OLD = np.array((0.60, 0.49, 0.36))   # Montjuïc sandstone, darkened: Nativity, apse, cloister
OCHRE = np.array((0.74, 0.65, 0.52))       # nave, transept, podium
PASSION = np.array((0.78, 0.74, 0.66))     # Passion façade, lighter and austere
GREY_NEW = np.array((0.74, 0.72, 0.68))    # the newest stone of the central towers
ROOF = np.array((0.62, 0.56, 0.46))        # stone roofs
WHITE_GLAZE = (0.93, 0.93, 0.91)           # glazed ceramic on the top of the central towers
GLASS_C = (0.86, 0.90, 0.94)               # glass of the cross and the star
DARK = (0.12, 0.11, 0.12)                  # openings
STAINED = (0.16, 0.14, 0.20)               # windows (stained glass, dark from outside)
CONCRETE = (0.66, 0.66, 0.64)              # the Glory works
GREEN = (0.24, 0.44, 0.27)                 # the cypress
RED = (0.76, 0.18, 0.13)
GOLD = (0.90, 0.70, 0.22)
ORANGE = (0.90, 0.48, 0.15)
YELLOW = (0.95, 0.82, 0.30)
PURPLE = (0.45, 0.20, 0.42)
WHITE = (0.95, 0.94, 0.92)
FRUITS = [RED, ORANGE, YELLOW, GREEN, PURPLE, GOLD]

FOUND = -3.0        # foundations
PODIUM = 4.5        # the plinth of the church: the main floor is ~4 m above the street
CLOISTER_H = 14.0   # the cloister wing and the cloister by the sacristy

# ── OpenStreetMap (relation 9194723, heading 134.0°, origin 41.4035891, 2.1743496) ──
OUTLINE = [(-40.16, 19.53), (-40.09, 16.41), (-43.07, 14.28), (-42.21, 11.62), (-42.01, 7.98), (-43.19, 7.67), (-43.74,
           6.73), (-41.78, 4.83), (-41.8, 3.22), (-41.45, 2.76), (-40.43, 2.61), (-39.73, -0.13), (-40.21, -0.94), (-39.7,
           -1.19), (-40.31, -4.21), (-41.48, -4.31), (-41.79, -6.27), (-43.68, -8.13), (-43.45, -8.76), (-41.89, -9.46),
           (-42.26, -13.85), (-42.82, -15.08), (-40.36, -16.9), (-40.09, -47.44), (-39.65, -47.47), (-40, -49.89), (-40.75,
           -50.35), (-40.72, -50.78), (-39.83, -50.89), (-39.35, -51.44), (-39.81, -52.43), (-39.41, -52.86), (-38.74,
           -52.86), (-38.32, -52.35), (-37.71, -52.67), (-37.8, -53.21), (-37.39, -53.57), (-36.69, -53.01), (-33.9,
           -52.59), (-33.89, -53.38), (-19.96, -53.22), (-19.77, -44.68), (-19.08, -44.1), (-18.24, -44.85), (-16.26,
           -44.9), (-14.83, -43.43), (-14.78, -41.41), (-16.25, -39.92), (-18.31, -39.86), (-19.01, -40.69), (-19.63,
           -40.12), (-19.5, -37.79), (-20.28, -37.72), (-24.4, -33.27), (-29.15, -33), (-32.57, -33.3), (-32.69, -17.67),
           (-31.05, -16.29), (-28.79, -16.4), (-28.29, -15.68), (-22.87, -15.58), (-22.86, -16.52), (-21.72, -16.55),
           (-22.73, -17.76), (-23.8, -17.56), (-24.03, -18.47), (-22.94, -18.78), (-22.91, -20.39), (-23.89, -20.84),
           (-23.52, -21.73), (-22.63, -21.4), (-21.64, -22.73), (-22.5, -23.71), (-21.53, -24.55), (-20.71, -23.58),
           (-18.75, -24.12), (-20.18, -26.35), (-21.39, -26.23), (-21.47, -27.11), (-20.33, -27.23), (-19.93, -29.25),
           (-20.86, -30.11), (-20.21, -30.92), (-19.19, -30.05), (-17.32, -31.26), (-17.8, -31.8), (-17.46, -32.41),
           (-16.69, -32.58), (-16.81, -34.68), (-18.03, -35.01), (-17.8, -36.1), (-16.48, -35.82), (-15.33, -37.12),
           (-16.04, -38.58), (-15.14, -38.98), (-14.43, -37.68), (-12.29, -37.95), (-12.38, -38.61), (-11.8, -39.13),
           (-11.05, -38.84), (-10.32, -40.75), (-11.4, -41.78), (-10.69, -42.62), (-9.7, -41.67), (-7.96, -42.45), (-7.93,
           -43.92), (-6.89, -43.98), (-6.82, -42.38), (-4.85, -41.83), (-4.54, -42.55), (-3.95, -42.7), (-3.26, -42.12),
           (-1.71, -43.57), (-2.34, -44.83), (-1.45, -45.34), (-0.72, -44.02), (1, -43.93), (1.83, -45.23), (2.64, -44.73),
           (2.02, -43.36), (3.56, -42), (4.23, -42.8), (5, -42.56), (5.2, -41.67), (7.15, -42.35), (7.25, -43.69), (8.17,
           -43.63), (8.14, -42.17), (9.67, -41.39), (10.93, -42.26), (11.59, -41.52), (10.47, -40.48), (11.31, -38.6),
           (11.97, -38.85), (12.55, -38.32), (12.46, -37.54), (14.52, -37.37), (15.45, -38.62), (16.29, -38.11), (15.53,
           -36.64), (16.77, -35.23), (18.04, -35.59), (18.25, -34.83), (16.96, -34.4), (16.81, -32.14), (17.58, -32.08),
           (17.89, -31.47), (17.38, -30.79), (18.99, -29.65), (20.29, -30.52), (20.88, -29.65), (19.76, -28.6), (20.19,
           -26.91), (21.56, -26.71), (21.48, -25.78), (20.01, -25.93), (18.7, -23.75), (20.64, -22.86), (21.49, -23.88),
           (22.37, -23.25), (21.55, -22.3), (22.39, -20.97), (23.51, -21.27), (23.79, -20.37), (22.81, -19.9), (22.83,
           -18.31), (24.05, -17.85), (23.74, -17.05), (22.6, -17.35), (21.62, -15.98), (22.6, -16.02), (22.67, -14.7),
           (29.4, -14.64), (32.64, -18.08), (32.55, -19.26), (39.55, -19.5), (39.57, -13.24), (38.44, -13.13), (37.67,
           -12.19), (37.76, -8.45), (38.71, -7.58), (39.64, -7.6), (39.63, -6.25), (38.57, -6.23), (35.67, -3.32), (35.71,
           4.68), (38.22, 7.36), (39.29, 7.49), (39.31, 8.36), (38.2, 8.46), (37.7, 9.14), (37.53, 13.87), (38.32, 14.82),
           (39.43, 14.84), (39.4, 20.74), (32.41, 20.32), (32.25, 19.15), (29.36, 16.03), (23.24, 16.01), (23.19, 21.47),
           (24.41, 22.21), (24.44, 22.8), (23.22, 23.71), (23.24, 28.83), (24.42, 29.57), (24.44, 30.2), (23.62, 31),
           (23.15, 30.96), (23.13, 36.28), (24.34, 36.97), (24.35, 37.67), (23.42, 38.6), (23.01, 38.57), (23.32, 52.24),
           (21.29, 53.71), (19.38, 54.4), (17.79, 53.04), (15.72, 52.99), (14.12, 54.13), (9.98, 54.08), (8.25, 52.9),
           (6.39, 52.92), (2.48, 55.02), (1.74, 54.72), (-1.07, 55.06), (-2.69, 54.7), (-3.45, 54.9), (-7.62, 52.48),
           (-9.37, 52.69), (-11.13, 54.07), (-14.57, 54.05), (-16.77, 52.51), (-18.87, 52.82), (-20.63, 53.97), (-23.57,
           52.37), (-23.79, 38.12), (-25.02, 37.25), (-25.15, 36.63), (-24.22, 35.74), (-23.63, 35.74), (-23.65, 30.58),
           (-24.99, 29.82), (-25.07, 29.1), (-24.15, 28.22), (-23.61, 28.24), (-23.71, 23.23), (-24.88, 22.42), (-24.98,
           21.81), (-24.17, 20.91), (-23.52, 20.94), (-23.43, 15.64), (-28.8, 15.68), (-28.95, 16.48), (-30.03, 17.17),
           (-31.51, 15.72), (-32.85, 17.05), (-33.27, 45.34), (-40.78, 45.21)]
CROSS80 = [(14.1, 7.45), (15.84, 9.28), (15.6, 13.43), (12.99, 15.93), (8.82, 16.02), (7.07, 14.21), (1.67, 19.39), (-2.44,
           19.3), (-7.87, 13.66), (-9.7, 15.41), (-13.68, 15.28), (-16.27, 12.6), (-16.2, 8.65), (-14.36, 6.88), (-19.58,
           1.47), (-19.5, -2.77), (-14.15, -7.89), (-15.91, -9.71), (-15.64, -13.88), (-13.04, -16.37), (-8.88, -16.47),
           (-7.12, -14.65), (-4.55, -17.12), (5.25, -16.38), (7.46, -14.06), (9.29, -15.83), (13.28, -15.7), (15.82,
           -13.06), (15.78, -9.06), (13.96, -7.31), (19.69, -1.34), (19.45, 2.31)]
CROSS60 = [(15.75, 7.49), (17.04, 8.84), (16.73, 13.96), (13.48, 17.09), (8.33, 17.21), (7.04, 15.86), (2.13, 20.58),
           (-2.96, 20.46), (-7.9, 15.31), (-9.23, 16.6), (-14.2, 16.44), (-17.45, 13.06), (-17.35, 8.14), (-16.02, 6.85),
           (-20.75, 1.93), (-20.66, -3.27), (-15.8, -7.94), (-17.11, -9.27), (-16.79, -14.41), (-13.53, -17.54), (-8.38,
           -17.65), (-7.09, -16.31), (-5, -18.33), (5.77, -17.51), (7.49, -15.73), (8.83, -17.01), (13.8, -16.86), (17,
           -13.52), (16.95, -8.56), (15.61, -7.27), (20.89, -1.78), (20.58, 2.85)]
MARY_BASE = [(-8.74, -23.69), (-7.81, -19.12), (-5.4, -16.63), (-2.13, -15.1), (1.5, -15.1), (5.44, -16.33), (7.62, -19.02),
             (8.71, -23.32), (8.1, -27.09), (5.58, -30.3), (1.88, -31.88), (-1.85, -31.97), (-5.35, -30.4), (-7.7, -27.58)]
SACRISTY = [(-19.81, -46.11), (-19.96, -53.22), (-20.34, -52.99), (-24.43, -53.29), (-28.8, -53.1), (-28.81, -53.37),
            (-29.41, -53.38), (-29.41, -53.17), (-33.44, -53.15), (-33.89, -53.38), (-33.9, -52.59), (-35.64, -53.07),
            (-36.69, -53.01), (-36.91, -53.46), (-37.39, -53.57), (-37.8, -53.21), (-37.71, -52.67), (-38.32, -52.35),
            (-38.74, -52.86), (-39.41, -52.86), (-39.81, -52.43), (-39.78, -51.84), (-39.35, -51.44), (-39.83, -50.89),
            (-40.3, -51.07), (-40.72, -50.78), (-40.75, -50.35), (-40, -49.89), (-39.65, -47.47), (-40.09, -47.44), (-40.25,
            -33.58), (-29.15, -33), (-29.11, -33.23), (-24.81, -33.03), (-24.4, -33.27), (-24.46, -33.52), (-20.28, -37.72),
            (-19.5, -37.79), (-19.63, -40.12), (-19.01, -40.69), (-18.31, -39.86), (-16.25, -39.92), (-14.78, -41.41),
            (-14.83, -43.43), (-16.26, -44.9), (-18.24, -44.85), (-19.08, -44.1), (-19.77, -44.68)]
CROSS100 = [(-7.17, -12.46), (-3.98, -15.52), (4.55, -14.88), (7.42, -11.87), (9.9, -14.26), (12.6, -14.17), (14.26,
            -12.45), (14.23, -9.74), (11.76, -7.36), (18.1, -0.76), (17.93, 1.61), (11.9, 7.4), (14.25, 9.85), (14.08,
            12.73), (12.35, 14.39), (9.45, 14.46), (7.1, 12.01), (1.05, 17.83), (-1.78, 17.76), (-7.83, 11.46), (-10.31,
            13.84), (-13.01, 13.75), (-14.71, 12), (-14.66, 9.32), (-12.17, 6.94), (-18.01, 0.86), (-17.96, -2.08), (-11.96,
            -7.85), (-14.31, -10.3), (-14.14, -13.17), (-12.4, -14.84), (-9.53, -14.9)]
CLOISTER_W = [(-32.84, 19.7), (-40.16, 19.53), (-40.78, 45.21), (-33.27, 45.34)]
COLUMN_FEET = [(-50.08, -9.54), (-48.07, -17.15), (-45.22, -22.76), (-48.35, 16.01), (-50.29, 8.37), (-45.54, 21.72), (40.42, -6.76), (40.20, 7.94)]   # 6 Passion portico columns (16-point stars), 2 Nativity columns (hexagons)

JESUS = (-0.30, 0.20)
MARY = (0.58, -23.07)
EVANGELISTS = [(10.85, 10.45, "Joan"), (-10.83, 10.62, "Mateu"), (-10.90, -10.98, "Marc"), (10.50, -11.09, "Lluc")]
# bell towers: (x, z, height, radius, name) from the OSM parts
PASSION_TOWERS = [(-36.10, 13.75, 107, 4.2, "Felip"), (-36.08, 6.58, 112, 4.2, "Tomàs"),
                  (-36.49, -7.53, 112, 4.2, "Bartomeu"), (-36.46, -14.71, 107, 4.2, "Jaume el Menor")]
NATIVITY_TOWERS = [(34.26, 16.23, 98, 3.6, "Bernabeu"), (34.09, 8.90, 107, 3.6, "Simó"),
                   (34.10, -6.45, 107, 3.6, "Judes Tadeu"), (34.63, -13.99, 98, 3.6, "Maties")]
NAVE_HALF, NAVE_Z0, NAVE_Z1 = 23.2, 15.0, 52.4      # nave block: x ±23.2, from the crossing to the Glory wall
TR_HALF = 16.0                                      # transept arms: z ±16
PAS_X, NAT_X = -32.6, 30.0                          # ends of the transept arms
APSE_C, APSE_R, CHAPEL_R = (0.4, -22.5), 15.0, 22.3
PAS_CLOISTER = [(-40.15, -17.5), (-32.57, -17.9), (-32.34, -33.52), (-40.0, -33.7)]   # from the OSM outline


def apse_ring():
    """The ring of apse chapels: the OSM outline round the apse (its teeth are the chapels' pinnacles and
    buttresses), closed along z = −15 against the transept."""
    run = [(x, z) for x, z in OUTLINE if z < -15.3 and -24.5 < x < 24.5]
    return [(-23.0, -15.0)] + run + [(23.0, -15.0)]


# ── Helpers ──────────────────────────────────────────────────────────────────
def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def stone(m, base, var=0.10, freq=0.12, seed=1, soot=0.0):
    """Stone colour with soft patches (and dark soot streaks low down on the old façades)."""
    base = np.asarray(base, float)
    n = fbm(m.v, freq, seed, 3)
    k = 1 - var / 2 + var * n
    c = base[None, :] * k[:, None]
    if soot:
        s = fbm(m.v, (0.3, 0.04, 0.3), seed + 5, 3)
        c = c * (1 - soot * np.clip((s - 0.45) * 3, 0, 1))[:, None]
    m.c = np.clip(c, 0, 1)
    return m


def prism_xy(P, z0, z1, cols=None, c=(0.5, 0.5, 0.5)):
    """A prism: polygon P in the (x, y) plane, extruded along z from z0 to z1; cols = a colour per edge."""
    P = [tuple(p) for p in P]
    if ring_area(P) < 0:
        P = P[::-1]
        if cols is not None:
            cols = cols[::-1][1:] + cols[::-1][:1]
    pts, tris = triangulate(P)
    ms = []
    v1 = np.stack([pts[:, 0], pts[:, 1], np.full(len(pts), z1)], 1)
    v0 = np.stack([pts[:, 0], pts[:, 1], np.full(len(pts), z0)], 1)
    ms.append(Mesh(v1, tris, c)); ms.append(Mesh(v0, tris[:, ::-1], c))
    for i in range(len(P)):
        (xa, ya), (xb, yb) = P[i], P[(i + 1) % len(P)]
        v = [(xa, ya, z0), (xb, yb, z0), (xb, yb, z1), (xa, ya, z1)]
        ms.append(Mesh(v, [(0, 1, 2), (0, 2, 3)], c if cols is None else cols[i]))
    return merge(ms)


def rot_y90(m):
    """(x, z) -> (z, -x): a piece built along +z turns to run along +x."""
    v = m.v.copy()
    m.v = np.stack([v[:, 2], v[:, 1], -v[:, 0]], 1)
    return m


def sphere(r, c, seg=8, rings=5):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def lobed(m, cx, cz, n, amp):
    """Flutes round a body of revolution (n lobes, amp as a fraction of the radius)."""
    def f(v):
        dx, dz = v[:, 0] - cx, v[:, 2] - cz
        a = np.arctan2(dz, dx)
        k = 1 + amp * np.cos(n * a)
        v[:, 0] = cx + dx * k; v[:, 2] = cz + dz * k
        return v
    return m.warp(f)


def panel(p0, t, n, pts_uv, c, eps=0.06):
    """A flat polygon drawn on a wall: origin p0 (3D), tangent t, outward normal n, pts in (u along t, v up)."""
    pts, tris = triangulate(pts_uv)
    o = np.asarray(p0, float) + np.asarray(n, float) * eps
    t = np.asarray(t, float)
    V = np.array([o + u * t + np.array([0, v, 0]) for u, v in pts])
    m = Mesh(V, tris, c)
    if np.dot(np.cross(t, [0, 1, 0]), n) < 0:
        m.f = m.f[:, ::-1]
    return m


def arch_pts(s, y0, w, h, seg=7, kind="parabolic"):
    """Outline of an opening of width w and height h centred at s: Gaudí's parabolic arch (or round)."""
    r = w / 2
    pts = [(s - r, y0), (s + r, y0)]
    if kind == "parabolic":
        ys = y0 + h * 0.35
        for u in np.linspace(1, -1, seg + 2):
            pts.append((s + r * u, ys + (h * 0.65) * (1 - u * u)))
    else:
        ys = y0 + h - r
        pts += [(s + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)]
    return pts


def win_x(x, z, y0, w, h, side, c=STAINED, kind="parabolic"):
    """A window on a wall facing +x (side 1) or −x (side −1), centred at z."""
    n = (side, 0, 0)
    t = (0, 0, -side)                       # u runs along t, so n = t × up points outwards
    return panel((x, 0, z), t, n, arch_pts(0, y0, w, h, kind=kind), c)


def win_z(x, z, y0, w, h, side, c=STAINED, kind="parabolic"):
    n = (0, 0, side)
    t = (side, 0, 0)
    return panel((x, 0, z), t, n, arch_pts(0, y0, w, h, kind=kind), c)


def fruit_top(x, y, z, s, k):
    """The coloured fruit baskets that crown the buttresses of the nave (mosaic of Venetian glass)."""
    ms = [sphere(0.9 * s, FRUITS[k % 6], 7, 4).T(x, y + 0.9 * s, z)]
    for j in range(3):
        a = 2 * math.pi * j / 3 + k
        ms.append(sphere(0.55 * s, FRUITS[(k + j + 1) % 6], 6, 3).T(x + 0.75 * s * math.cos(a), y + 1.6 * s, z + 0.75 * s * math.sin(a)))
    ms.append(sphere(0.5 * s, WHITE, 6, 3).T(x, y + 2.3 * s, z))
    return merge(ms)


def pinnacle(x, z, y0, h, r, c, top=None, seg=6, crockets=True):
    """A tapering stone pinnacle with knobs (crockets) and an optional coloured top."""
    ms = [cyl(r, r * 0.18, h, c, seg).T(x, y0, z)]
    if crockets:
        for k in range(1, 4):
            yy = y0 + h * k / 4.5
            rr = r * (1 - 0.8 * k / 4.5) + 0.2
            ms.append(cyl(rr, rr * 0.6, 0.6, c, seg).T(x, yy, z))
    if top is not None:
        ms.append(sphere(r * 0.55, top, 6, 3).T(x, y0 + h, z))
    return merge(ms)


# ── Podium, cloister and sacristy ────────────────────────────────────────────
def podium():
    ms = [stone(extrude(OUTLINE, FOUND, PODIUM, OCHRE * 0.95, top_c=OCHRE * 0.95), OCHRE * 0.95, seed=3)]
    # the cloister wing on the Passion side and the cloister between the Passion façade and the sacristy
    ms.append(stone(extrude(CLOISTER_W, PODIUM - 0.01, 16.0, OCHRE), OCHRE, seed=4))
    ms.append(stone(extrude(PAS_CLOISTER, PODIUM - 0.01, CLOISTER_H, OCHRE), OCHRE, seed=5))
    # windows of the cloister: round-headed openings on the long straight walls
    for i in range(len(OUTLINE)):
        (xa, za), (xb, zb) = OUTLINE[i], OUTLINE[(i + 1) % len(OUTLINE)]
        L = math.hypot(xb - xa, zb - za)
        if L < 6 or not (min(xa, xb) < -32.0 and max(za, zb) < 46.0):
            continue
        t = np.array([(xb - xa) / L, 0, (zb - za) / L])
        n = np.array([t[2], 0, -t[0]])               # outward for a counter-clockwise outline
        if ring_area(OUTLINE) < 0:
            n = -n
        nw = int(L // 4.5)
        for j in range(nw):
            s = (j + 0.5) * L / nw
            ms.append(panel((xa, 0, za), t, n, arch_pts(s, 6.0, 1.8, 5.5, kind="round"), DARK))
    return ms


def sacristy():
    ms = [stone(extrude(SACRISTY, PODIUM - 0.01, 16.0, OCHRE), OCHRE, seed=7)]
    P = np.array(SACRISTY)
    x, z = P[:, 0], P[:, 1]
    k = x * np.roll(z, -1) - np.roll(x, -1) * z
    cx = float(((x + np.roll(x, -1)) * k).sum() / (3 * k.sum()))     # centroid of the area
    cz = float(((z + np.roll(z, -1)) * k).sum() / (3 * k.sum()))
    prof = [(9.0, 16.0), (8.8, 19.0), (8.0, 22.0), (6.4, 25.0), (4.2, 27.8), (2.0, 29.6), (0.6, 30.3), (0.0, 30.4)]
    d = lobed(lathe(prof, OCHRE, 24), 0, 0, 12, 0.07).T(cx, 0, cz)
    ms.append(stone(d, PASSION, seed=8))
    ms.append(pinnacle(cx, cz, 30.0, 8.0, 0.9, PASSION, seg=8))
    ms += [sphere(0.6, (RED, GOLD, WHITE)[j % 3], 6, 3).T(cx + 1.0 * math.cos(j * 1.57), 35.5, cz + 1.0 * math.sin(j * 1.57)) for j in range(4)]
    ms.append(B(cx - 0.15, cx + 0.15, 38.0, 40.2, cz - 0.15, cz + 0.15, GOLD))
    ms.append(B(cx - 0.8, cx + 0.8, 39.2, 39.5, cz - 0.15, cz + 0.15, GOLD))
    return ms


# ── Nave, transept, apse, crossing ───────────────────────────────────────────
def section(half_out):
    """The stepped section of the nave (and the transept): outer aisles, clerestory, central roof."""
    P = [(-half_out, PODIUM - 0.01), (half_out, PODIUM - 0.01), (half_out, 33.0), (12.0, 36.0), (9.0, 36.0), (9.0, 50.0),
         (0.0, 62.0), (-9.0, 50.0), (-9.0, 36.0), (-12.0, 36.0), (-half_out, 33.0)]
    W, R = OCHRE, ROOF
    cols = [W, W, R, R, W, R, R, W, R, R, W]
    return P, cols


def arm(z0, z1, half_out, end_wall=None):
    """A nave-like arm running along z from z0 to z1 (built for the nave, turned for the transept)."""
    P, cols = section(half_out)
    ms = [stone(prism_xy(P, z0, z1, cols), OCHRE, seed=11)]
    body = ms[0]                                    # roofs keep the roof colour
    yy = body.v[:, 1]
    roofish = (yy > 33.5) & ((np.abs(body.v[:, 0]) < 8.9) | (yy < 36.5))
    body.c[roofish] = body.c[roofish] * 0.0 + ROOF * (0.95 + 0.1 * fbm(body.v[roofish], 0.2, 2, 2))[:, None]
    nb = int(round((z1 - z0) / 7.5))
    bay = (z1 - z0) / nb
    for i in range(nb + 1):
        z = z0 + i * bay
        for s in (-1, 1):
            # buttresses of the outer wall and their fruit baskets
            ms.append(stone(B(s * half_out - 0.7, s * half_out + 0.7, PODIUM, 35.0, z - 0.7, z + 0.7, OCHRE), OCHRE, seed=12))
            ms.append(pinnacle(s * half_out, z, 35.0, 4.0, 0.7, OCHRE, seg=6, crockets=False))
            ms.append(fruit_top(s * half_out, 38.6, z, 0.9, i * 2 + (s > 0)))
            # clerestory buttresses with taller pinnacles and fruit
            ms.append(stone(B(s * 9.0 - 0.6, s * 9.0 + 0.6, 36.0, 51.0, z - 0.6, z + 0.6, OCHRE), OCHRE, seed=13))
            ms.append(pinnacle(s * 9.0, z, 51.0, 6.0, 0.65, OCHRE, seg=6))
            ms.append(fruit_top(s * 9.0, 56.6, z, 0.8, i * 3 + (s > 0) + 1))
    for i in range(nb):
        zc = z0 + (i + 0.5) * bay
        for s in (-1, 1):
            ms.append(win_x(s * half_out, zc, 15.5, 2.6, 14.0, s))
            ms.append(win_x(s * half_out, zc, 7.0, 2.2, 6.0, s))
            ms.append(win_x(s * 9.0, zc, 38.0, 3.6, 10.5, s))
        # lanterns along the ridge
        ms.append(stone(lobed(cyl(2.4, 0.15, 9.0, ROOF, 8).T(0, 59.0, zc), 0, zc, 4, 0.0), ROOF * 1.05, seed=14))
        ms.append(sphere(0.45, GOLD, 6, 3).T(0, 68.0, zc))
    return ms


def nave():
    ms = arm(NAVE_Z0, NAVE_Z1, NAVE_HALF)
    # the Glory end wall: big windows on the gable
    for x in (-16.0, 16.0):
        ms.append(win_z(x, NAVE_Z1, 15.0, 4.0, 15.0, 1))
    ms.append(win_z(0.0, NAVE_Z1, 36.0, 8.0, 18.0, 1))
    for x in (-4.5, 4.5):
        ms.append(win_z(x, NAVE_Z1, 24.0, 3.0, 9.0, 1))
    return ms


def glory():
    """The Glory portico, still a grey concrete structure in 2026 (simplified)."""
    ms = []
    zf = 54.8
    ms.append(stone(B(-NAVE_HALF, NAVE_HALF, PODIUM - 0.01, 22.0, NAVE_Z1 - 0.3, zf - 1.4, CONCRETE), CONCRETE, var=0.06, seed=21))
    for k, x in enumerate((-19.0, -13.0, -6.5, 0.0, 6.5, 13.0, 19.0)):             # seven doors
        ms.append(win_z(x, zf - 1.4, PODIUM, 3.6, 8.0, 1, c=DARK))
    for x in (-21.5, -9.5, 9.5, 21.5):                                               # bases of the four Glory towers
        ms.append(stone(lobed(cyl(2.6, 2.3, 30.0 - PODIUM, CONCRETE, 12).T(x, PODIUM - 0.01, zf - 2.2), x, zf - 2.2, 6, 0.08),
                        CONCRETE, var=0.06, seed=22))
    ms.append(B(-NAVE_HALF, NAVE_HALF, 22.0, 23.0, NAVE_Z1 - 0.3, zf - 0.8, (0.6, 0.6, 0.58)))
    return ms


def transept():
    ms = []
    # Passion arm (−x): built along z from −PAS_X… then turned (z -> x)
    for a, b in ((15.0, -PAS_X), (15.0, NAT_X)):
        m = arm(a, b, TR_HALF)
        if b == -PAS_X:                                  # mirror: run to −x
            for p in m:
                p.v[:, 2] *= -1; p.f = p.f[:, ::-1]
        ms += [rot_y90(p) for p in m]
    return ms


def apse():
    ms = []
    cx, cz = APSE_C
    # ring of chapels: half disc of radius CHAPEL_R plus the straight part to the transept
    ring = apse_ring()
    ms.append(stone(extrude(ring, PODIUM - 0.01, 27.0, OCHRE_OLD, top_c=ROOF * 0.95), OCHRE_OLD, seed=31, soot=0.25))
    # the many small pinnacles on the outer wall of the chapels (the teeth of the OSM outline)
    for i, (x, z) in enumerate(ring):
        if z < -15.5 and i % 3 == 0:
            ms.append(pinnacle(x, z, 27.0, 5.0, 0.5, OCHRE_OLD * 1.05, top=FRUITS[i % 6], seg=5, crockets=False))
    # apse body (ambulatory walls and clerestory)
    body = [(-APSE_R, -15.0), (APSE_R, -15.0)] + \
           [(cx + APSE_R * math.cos(a), cz + APSE_R * math.sin(a)) for a in np.linspace(0, -math.pi, 18)]
    ms.append(stone(extrude(body, 26.9, 48.0, OCHRE_OLD, top_c=ROOF), OCHRE_OLD * 1.06, seed=32, soot=0.15))
    # roof cone up to the Virgin Mary tower
    prof = [(APSE_R, 48.0), (12.5, 52.0), (10.0, 57.0), (8.9, 61.0)]
    ms.append(stone(lathe(prof, ROOF, 24).T(cx, 0, cz), ROOF, seed=33))
    # chapel windows and pinnacles
    for k in range(7):
        a = -math.pi * (k + 0.5) / 7
        x, z = cx + CHAPEL_R * math.cos(a), cz + CHAPEL_R * math.sin(a)
        n = np.array([math.cos(a), 0, math.sin(a)])
        t = np.array([n[2], 0, -n[0]])
        ms.append(panel((x, 0, z), t, n, arch_pts(0, 15.0, 2.4, 10.0), STAINED, eps=0.15))
        x2, z2 = cx + APSE_R * math.cos(a), cz + APSE_R * math.sin(a)
        ms.append(panel((x2, 0, z2), t, n, arch_pts(0, 33.0, 2.8, 11.0), STAINED, eps=0.15))
    for k in range(8):
        a = -math.pi * k / 7
        x, z = cx + (CHAPEL_R + 0.3) * math.cos(a), cz + (CHAPEL_R + 0.3) * math.sin(a)
        ms.append(stone(B(x - 0.9, x + 0.9, PODIUM, 29.0, z - 0.9, z + 0.9, OCHRE_OLD), OCHRE_OLD, seed=34, soot=0.3))
        ms.append(stone(pinnacle(x, z, 29.0, 16.0, 1.1, OCHRE_OLD, seg=8), OCHRE_OLD * 1.04, seed=35, soot=0.2))
        ms.append(sphere(0.7, GOLD, 6, 3).T(x, 45.0, z))                         # ears of wheat, gilded
        x2, z2 = cx + APSE_R * math.cos(a), cz + APSE_R * math.sin(a)
        ms.append(pinnacle(x2, z2, 48.0, 7.0, 0.7, OCHRE_OLD * 1.04, top=GOLD, seg=6))
    # small pinnacles on the chapel roofs (gables of each chapel)
    for k in range(7):
        a = -math.pi * (k + 0.5) / 7
        x, z = cx + (CHAPEL_R - 0.5) * math.cos(a), cz + (CHAPEL_R - 0.5) * math.sin(a)
        ms.append(pinnacle(x, z, 27.0, 7.0, 0.8, OCHRE_OLD, top=FRUITS[k % 6], seg=6))
    return ms


def crossing():
    c = GREY_NEW * 0.97
    ms = [stone(extrude(CROSS60, PODIUM - 0.01, 60.0, c, top_c=ROOF), c, seed=41),
          stone(extrude(CROSS80, 60.0, 72.0, c, top_c=ROOF), c, seed=42),
          stone(extrude(CROSS100, 72.0, 84.0, c, top_c=GREY_NEW), GREY_NEW, seed=43)]
    # windows round the stages
    for ring_, y0, h in ((CROSS80, 62.0, 8.0), (CROSS100, 74.0, 8.0)):
        R = ring_ if ring_area(ring_) > 0 else ring_[::-1]
        for i in range(len(R)):
            (xa, za), (xb, zb) = R[i], R[(i + 1) % len(R)]
            L = math.hypot(xb - xa, zb - za)
            if L < 3.0:
                continue
            t = np.array([(xb - xa) / L, 0, (zb - za) / L]); n = np.array([t[2], 0, -t[0]])
            ms.append(panel((xa, 0, za), t, n, arch_pts(L / 2, y0, min(2.2, L * 0.4), h), DARK))
    return ms


# ── Towers ───────────────────────────────────────────────────────────────────
def spiral_paint(m, cx, cz, y0, y1, base, dark, turns=0.6, n=4):
    """The spiral openings of the bell towers: dark bands winding up the spire."""
    v = m.v
    a = np.arctan2(v[:, 2] - cz, v[:, 0] - cx)
    ph = (a * n / (2 * math.pi) + (v[:, 1] - y0) / 7.0 * turns * n) % 1.0
    band = (ph < 0.35) & (v[:, 1] > y0) & (v[:, 1] < y1)
    c = np.tile(np.asarray(base, float), (len(v), 1))
    c[band] = dark
    m.c = c
    return m


def bell_tower(x, z, H, r0, base, seed, facade):
    ms = []
    hs = H - 22.0                     # top of the stone spire, start of the mosaic finial
    yb = 45.0
    prof = [(r0 * 1.04, PODIUM - 0.01), (r0 * 1.04, 30.0), (r0, yb)]
    for t in np.linspace(0, 1, 26)[1:]:
        prof.append((r0 * (1 - 0.66 * t ** 1.5), yb + t * (hs - yb)))
    prof.append((0.0, hs + 0.01))
    m = lathe(prof, base, 16).T(x, 0, z)
    m = spiral_paint(m, x, z, yb + 2, hs - 1, base, np.asarray(base) * 0.35)
    ms.append(m)
    # louvres on the lower shaft facing out
    sgn = 1 if facade == "nat" else -1
    for k in range(3):
        ms.append(win_x(x + sgn * (r0 * 1.04 - 0.05), z, 20.0 + k * 8.5, 1.2, 6.0, sgn, c=DARK))
    # the finial: stem, hexagonal body with white balls, mitre, cross
    rf = r0 * 0.34 + 0.2
    ms.append(cyl(rf, rf * 1.25, 5.0, GOLD, 6).T(x, hs, z))
    body = lathe([(rf * 1.25, 0), (rf * 1.6, 2.5), (rf * 1.65, 5.0), (rf * 1.3, 7.5), (rf * 0.9, 8.5)], RED, 6).T(x, hs + 5.0, z)
    body.c[body.v[:, 1] > hs + 9.0] = ORANGE if facade == "pas" else GOLD      # mosaic bands
    ms.append(body)
    for j in range(6):
        a = j * math.pi / 3 + math.pi / 6
        ms.append(sphere(rf * 0.5, WHITE, 6, 3).T(x + rf * 1.75 * math.cos(a), hs + 7.6, z + rf * 1.75 * math.sin(a)))
    ms.append(lathe([(rf * 0.9, 0), (rf * 1.2, 2.0), (rf * 0.5, 5.5), (0.0, 6.2)], GOLD, 6).T(x, hs + 13.5, z))
    ms.append(sphere(rf * 0.9, WHITE, 8, 4).T(x, hs + 13.5, z))
    ms.append(B(x - 0.18, x + 0.18, hs + 19.4, H, z - 0.18, z + 0.18, YELLOW))
    ms.append(B(x - 0.8, x + 0.8, H - 1.6, H - 1.2, z - 0.18, z + 0.18, YELLOW))
    ms.append(B(x - 0.18, x + 0.18, H - 1.6, H - 1.2, z - 0.8, z + 0.8, YELLOW))
    return ms


def jesus_tower():
    cx, cz = JESUS
    prof = [(10.4, 83.9), (10.2, 100.0), (10.0, 110.0), (9.4, 120.0), (8.7, 130.0), (7.4, 140.0), (5.8, 150.0),
            (4.3, 154.5), (2.2, 156.5), (0.0, 157.0)]
    m = lobed(lathe(prof, GREY_NEW, 36).T(cx, 0, cz), cx, cz, 12, 0.07)
    v = m.v
    a = np.arctan2(v[:, 2] - cz, v[:, 0] - cx)
    groove = np.cos(12 * a) < -0.55
    c = np.tile(GREY_NEW, (len(v), 1)) * (0.95 + 0.08 * fbm(v, 0.1, 51, 2))[:, None]
    up = v[:, 1] > 134.0
    c[up] = WHITE_GLAZE
    c[groove & (v[:, 1] > 90) & (v[:, 1] < 150)] = (0.30, 0.31, 0.34)
    m.c = c
    ms = [m]
    # the four-armed cross: 17 m tall, arms 13.5 m wide, white glass and ceramic
    y0, y1, ya = 155.5, 172.5, 166.0
    cross = merge([B(cx - 1.2, cx + 1.2, y0, y1, cz - 1.2, cz + 1.2, GLASS_C),
                   B(cx - 6.75, cx + 6.75, ya - 1.1, ya + 1.1, cz - 1.1, cz + 1.1, GLASS_C),
                   B(cx - 1.1, cx + 1.1, ya - 1.1, ya + 1.1, cz - 6.75, cz + 6.75, GLASS_C)])
    return ms, cross


def mary_tower():
    cx, cz = MARY
    prof = [(8.9, 57.0), (8.7, 80.0), (8.0, 95.0), (6.9, 108.0), (5.3, 118.0), (3.4, 125.5), (1.6, 129.5), (0.0, 130.5)]
    m = lobed(lathe(prof, GREY_NEW, 32).T(cx, 0, cz), cx, cz, 12, 0.07)
    v = m.v
    a = np.arctan2(v[:, 2] - cz, v[:, 0] - cx)
    c = np.tile(GREY_NEW, (len(v), 1)) * (0.95 + 0.08 * fbm(v, 0.1, 61, 2))[:, None]
    c[v[:, 1] > 108] = WHITE_GLAZE
    c[(np.cos(12 * a) < -0.55) & (v[:, 1] > 66) & (v[:, 1] < 124)] = (0.30, 0.31, 0.34)
    m.c = c
    # the 12-pointed star (7 m across): a point to each vertex of an icosahedron
    g = (1 + 5 ** 0.5) / 2
    dirs = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            dirs += [(0, s1, s2 * g), (s1, s2 * g, 0), (s2 * g, 0, s1)]
    star = []
    yc = 134.4
    for d in dirs:
        d = np.array(d, float); d /= np.linalg.norm(d)
        p = cyl(0.7, 0.0, 3.5, GLASS_C, 5)
        # orient +y to d
        yax = np.array([0, 1.0, 0])
        ax = np.cross(yax, d); s = np.linalg.norm(ax); cth = float(np.dot(yax, d))
        if s < 1e-9:
            Rm = np.eye(3) if cth > 0 else np.diag([1, -1, -1])
        else:
            ax /= s
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Rm = np.eye(3) + s * K + (1 - cth) * K @ K
        p.v = p.v @ Rm.T + np.array([cx, yc, cz])
        star.append(p)
    star.append(sphere(1.1, GLASS_C, 10, 6).T(cx, yc, cz))
    return [m], merge(star)


def evangelist_towers():
    ms = []
    for k, (x, z, name) in enumerate(EVANGELISTS):
        prof = [(4.4, 83.9), (4.2, 95.0), (3.8, 105.0), (3.1, 113.0), (2.2, 119.0), (1.4, 123.0)]
        m = lobed(lathe(prof, GREY_NEW, 20).T(x, 0, z), x, z, 8, 0.08)
        v = m.v
        c = np.tile(GREY_NEW, (len(v), 1)) * (0.95 + 0.08 * fbm(v, 0.1, 70 + k, 2))[:, None]
        c[v[:, 1] > 108] = WHITE_GLAZE
        a = np.arctan2(v[:, 2] - z, v[:, 0] - x)
        c[(np.cos(8 * a) < -0.6) & (v[:, 1] > 88) & (v[:, 1] < 118)] = (0.32, 0.33, 0.35)
        m.c = c
        ms.append(m)
        # the symbol on top (winged man, lion, ox, eagle), white and gold, ~10 m
        ms.append(lathe([(1.4, 0), (2.4, 2.5), (2.6, 5.0), (1.8, 7.5), (0.9, 9.0), (0.0, 9.6)], WHITE, 10).T(x, 123.0, z))
        ang = math.atan2(z, x)
        for s in (-1, 1):                                                   # wings
            w = box(0.5, 6.0, 3.0, GOLD).R(0, 0, s * 0.35).T(0, 0, 0)
            w.R(0, -ang, 0)
            ox, oz = -math.sin(ang) * s * 2.6, math.cos(ang) * s * 2.6
            ms.append(w.T(x + ox, 126.0, z + oz))
        ms.append(sphere(0.9, GOLD, 8, 4).T(x, 133.3, z))
        ms.append(B(x - 0.15, x + 0.15, 133.8, 135.0, z - 0.15, z + 0.15, GOLD))
    return ms


# ── Façades ──────────────────────────────────────────────────────────────────
def relief(x_front, z0, z1, y0, top, amp, seed, base, nz=None, ny=None, side=1, soot=0.3):
    """A sculpted relief: a grid on the plane x = x_front (facing +x), pushed out by noise, cut at top(z)."""
    nz = nz or max(4, int((z1 - z0) / 0.9))
    ymax = max(top(z) for z in np.linspace(z0, z1, 9))
    ny = ny or max(4, int((ymax - y0) / 1.0))
    V, C = [], []
    for j in range(ny + 1):
        for i in range(nz + 1):
            z = z0 + (z1 - z0) * i / nz
            yt = top(z)
            y = y0 + (yt - y0) * j / ny
            V.append((0.0, y, z))
    V = np.array(V)
    d = fbm(V * np.array([1, 1, 1]) + seed, 0.45, seed, 3)
    edge = np.minimum.reduce([np.full(len(V), 1.0), (V[:, 2] - z0) / 1.5, (z1 - V[:, 2]) / 1.5, (V[:, 1] - y0) / 1.5])
    edge = np.clip(edge, 0, 1)
    V[:, 0] = x_front + side * amp * d * edge
    F = []
    W = nz + 1
    for j in range(ny):
        for i in range(nz):
            a = j * W + i
            F += [(a, a + 1, a + W), (a + 1, a + W + 1, a + W)]
    m = Mesh(V, F, base)
    if side > 0:
        m.f = m.f[:, ::-1]
    sh = 0.72 + 0.4 * d
    c = np.asarray(base, float)[None, :] * sh[:, None]
    s = fbm(V, (0.3, 0.05, 0.3), seed + 9, 3)
    c *= (1 - soot * np.clip((s - 0.45) * 3, 0, 1))[:, None]
    m.c = np.clip(c, 0, 1)
    return m


def nativity():
    """The Nativity façade (north-east, +x): three portals with reliefs, the cypress, the columns on turtles."""
    ms = []
    base = OCHRE_OLD
    xb = NAT_X - 0.5
    # the façade body between and round the towers, to the front of the portals
    body = [(xb, -19.0), (37.5, -19.0), (37.5, 20.6), (xb, 20.6)]
    ms.append(stone(extrude(body, PODIUM - 0.01, 30.0, base, top_c=base * 0.9), base, seed=81, soot=0.35))
    ms.append(relief(37.5, -19.0, 20.6, PODIUM, lambda z: 30.0, 1.2, 80, base))
    portals = [(1.2, 4.6, 46.0, 39.6), (-10.2, 2.6, 38.0, 38.6), (12.6, 2.6, 38.0, 38.6)]   # (z, half width, top, front x)
    for k, (zc, hw, top, xf) in enumerate(portals):
        # the portal mass with a pointed-parabolic crown
        def crown(z, zc=zc, hw=hw, top=top):
            u = (z - zc) / (hw + 0.8)
            return 30.0 + (top - 30.0) * max(0.0, 1 - u * u) ** 0.7
        blk = [(xb, zc - hw - 0.8), (xf - 1.0, zc - hw - 0.8), (xf - 1.0, zc + hw + 0.8), (xb, zc + hw + 0.8)]
        ms.append(stone(extrude(blk, PODIUM - 0.01, 30.0, base), base, seed=82 + k, soot=0.35))
        # gable of the portal (sculpted relief on its front and back)
        ms.append(relief(xf - 1.0, zc - hw - 0.8, zc + hw + 0.8, PODIUM, crown, 2.2, 83 + k, base))
        if k:                                                   # pinnacles on the crowns of the side portals
            for dz in (-hw * 0.7, 0.0, hw * 0.7):
                ms.append(pinnacle(xf - 1.5, zc + dz, crown(zc + dz) - 1.0, 6.0 if dz == 0 else 4.0, 0.6, base * 1.05, top=GOLD, seg=6))
        g = relief(xb + 0.2, zc - hw - 0.8, zc + hw + 0.8, 29.0, lambda z, c=crown: c(z) - 0.5, 0.6, 90 + k, base, side=-1)
        ms.append(g)
        # the deep portal opening (dark) with a lighter frame
        op = arch_pts(0, PODIUM + 2.0, hw * 1.25, (top - 30.0) * 0.5 + 12.0)
        ms.append(panel((xf + 0.9, 0, zc), (0, 0, -1), (1, 0, 0), op, DARK, eps=0.6))
    # the cypress over the Portal of Charity: green ceramic, white doves, a red cross-shaped top
    zc = 1.2
    prof = [(0.0, 44.0), (2.0, 45.0), (2.6, 48.0), (2.4, 52.0), (1.8, 56.0), (1.0, 59.5), (0.0, 62.0)]
    cyp = lathe(prof, GREEN, 12).T(37.0, 0, zc)
    cyp.c = np.tile(GREEN, (len(cyp.v), 1)) * (0.85 + 0.3 * fbm(cyp.v, 0.8, 99, 2))[:, None]
    ms.append(lobed(cyp, 37.0, zc, 10, 0.12))
    for j in range(12):
        a = j * 2.4
        y = 45.5 + (j % 6) * 2.4
        r = 2.6 - (y - 45) * 0.1 + 0.3
        ms.append(sphere(0.35, WHITE, 6, 3).T(37.0 + r * math.cos(a), y, zc + r * math.sin(a)))
    ms.append(B(36.85, 37.15, 61.5, 63.5, zc - 0.15, zc + 0.15, RED))
    ms.append(B(36.85, 37.15, 62.6, 62.9, zc - 0.7, zc + 0.7, RED))
    # the bridge between the central towers with the figure of the Annunciation (simplified)
    ms.append(stone(B(31.0, 35.5, 52.0, 54.0, -3.0, 5.4, base), base, seed=97))
    # the columns on turtles in front of the central portal
    for (x, z) in COLUMN_FEET[6:]:
        ms.append(sphere(1.1, (0.42, 0.45, 0.40), 8, 4).S(1, 0.45, 1).T(x, 0.3, z))
        ms.append(stone(cyl(0.38, 0.32, 8.5, base, 6).T(x, 0.0, z), base * 1.05, seed=98))
    return ms


def passion():
    """The Passion façade (south-west, −x): the porch on six leaning columns, the frieze of bones, the gable."""
    ms = []
    base = PASSION
    xw = PAS_X
    # the porch roof slab, slightly tilted
    slab = B(-45.0, xw + 0.5, 24.0, 26.5, -21.0, 21.0, base)
    ms.append(stone(slab, base * 0.95, var=0.06, seed=101))
    # six leaning columns from their star-shaped feet to the slab
    for k, (x, z) in enumerate(COLUMN_FEET[:6]):
        top = (-43.5, 24.2, z * 0.92)
        bot = (x, -1.0, z)
        d = np.array(top) - np.array(bot); L = float(np.linalg.norm(d))
        c = lobed(cyl(1.1, 0.75, L, base, 10), 0, 0, 5, 0.25)
        # tilt +y onto d
        yax = np.array([0, 1.0, 0]); u = d / L
        ax = np.cross(yax, u); s = np.linalg.norm(ax); cth = float(np.dot(yax, u))
        ax /= s
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rm = np.eye(3) + s * K + (1 - cth) * K @ K
        c.v = c.v @ Rm.T + np.array(bot)
        ms.append(stone(c, base * 0.98, var=0.08, seed=102 + k))
    # the frieze of bones along the front of the porch
    for k in range(18):
        z = -19.5 + k * 39.0 / 17
        h = 9.0 + 2.0 * math.sin(k * 0.9)
        ms.append(stone(cyl(0.65, 0.45, h, base, 6).T(-44.6, 26.5, z), base * 1.02, var=0.06, seed=110 + k))
        ms.append(sphere(0.6, base, 6, 3).T(-44.6, 26.5 + h, z))
    # the wall of the façade behind the porch with its openings
    wall = [(-40.2, -12.0), (xw + 0.3, -12.0), (xw + 0.3, 12.0), (-40.2, 12.0)]
    ms.append(stone(extrude(wall, PODIUM - 0.01, 24.0, base, top_c=base * 0.9), base, seed=120))
    for z in (-6.5, 0.0, 6.5):
        ms.append(win_x(-40.2, z, PODIUM + 0.5, 3.4, 10.5, -1, c=DARK))
    # the high gable between the central towers (the Passion pediment, simplified)
    gable = prism_xy([(-10.0, 26.4), (10.0, 26.4), (6.0, 48.0), (0.0, 58.0), (-6.0, 48.0)], -40.0, -38.4, c=base)
    gable = stone(rot_y90(gable), base, seed=121)
    ms.append(gable)
    # the bridge between the central towers with the Risen Christ (gilded)
    ms.append(stone(B(-38.4, -34.0, 60.0, 62.0, -7.5, 6.6, base), base, seed=122))
    ms.append(lathe([(0.6, 0), (0.7, 1.6), (0.5, 3.0), (0.35, 3.4), (0.0, 3.6)], GOLD, 8).T(-36.3, 62.0, -0.5))
    return ms


# ── Assembly ─────────────────────────────────────────────────────────────────
def footprint(parts):
    """The convex hull of the model seen from above, grown by 1 m: the church and its own grounds (cloister,
    porches, the Passion columns) inside its Eixample block; the streets and the two squares stay clear."""
    allv = np.vstack([m.v for m in parts.values()])
    pts = sorted(set((round(float(x), 2), round(float(z), 2)) for x, z in allv[:, [0, 2]]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    hull = lo[:-1] + up[:-1]
    return [[round(x, 2), round(z, 2)] for x, z in offset(hull, 1.0)]


def build():
    jt, cross_ = jesus_tower()
    mt, star = mary_tower()
    nat_t, pas_t = [], []
    for k, (x, z, H, r, _) in enumerate(NATIVITY_TOWERS):
        nat_t += bell_tower(x, z, H, r, OCHRE_OLD * 1.02, 200 + k, "nat")
    for k, (x, z, H, r, _) in enumerate(PASSION_TOWERS):
        pas_t += bell_tower(x, z, H, r, PASSION, 210 + k, "pas")
    parts = {
        "podium_cloister": merge(podium() + sacristy()),
        "nave": merge(nave() + glory()),
        "transept": merge(transept()),
        "apse": merge(apse()),
        "crossing": merge(crossing()),
        "tower_jesus": merge(jt),
        "tower_mary": merge(mt),
        "towers_evangelists": merge(evangelist_towers()),
        "nativity": merge(nativity() + nat_t),
        "passion": merge(passion() + pas_t),
        "cross_star": merge([cross_, star]),
    }
    return parts


def main():
    out = ROOT / "models" / "Q48435"
    out.mkdir(parents=True, exist_ok=True)
    parts = build()
    mats = {"glass": GLASS_MAT}
    plist = [(k, m, {"material": "glass"}) if k == "cross_star" else (k, m) for k, m in parts.items()]
    tris, size = write_glb(out / "model.glb", plist, roughness=0.9, materials=mats)
    allv = np.vstack([m.v for m in parts.values()])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    for k, m in parts.items():
        print(f"  {k}: {m.tris:,}")
    if "--footprint" in sys.argv:
        import json
        print(json.dumps(footprint(parts)))


if __name__ == "__main__":
    main()
