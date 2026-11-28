"""
Hagia Sophia, Istanbul (Q12506) — the great church-mosque, its minarets, tombs and precinct, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Anthemius of Tralles (died 534) and Isidore of Miletus (died c. 537), for Justinian I, 532–537;
the dome rebuilt higher by Isidore the Younger in 562; Ottoman buttresses, the two western minarets and the tomb
of Selim II by Mimar Sinan (died 1588); the other minarets under Mehmed II and Bayezid II (15th c.), the tombs of
Murad III (Davud Ağa, 1594–99) and Mehmed III (Dalgıç Ahmed Ağa, 1608), the şadırvan of Mahmud I (1740).
Public domain.

What is modelled:
- The church: the main body (aisles and galleries, 75 × 77 m) with its two rows of windows, the raised nave, the
  square base of the dome with the great tympana of the north and south arches (two rows of windows under a
  relieving arch), the four buttress towers rising above the aisles on the north and south sides; the main dome
  (Ø 33 m) on its ring of 40 windows between 40 small buttresses, the lead-covered shallow cap 56 m above the
  floor and its gilded finial (alem) with the crescent; the two semi-domes east and west on their rings of
  windows, the four exedra semi-domes beside them, the apse with its polygonal wall and half-dome, the two
  little domed turrets at the west end of the nave.
- The Ottoman buttresses (sloping-topped masses on the north, south and east), the inner narthex (skillion
  roof) and the low exonarthex on the west with the six buttresses of the west front, the low annexes inside the
  OpenStreetMap outline (ramps, the sultan's lodge, offices), all in the pinkish-ochre plaster.
- The four minarets, each different: the two western ones by Sinan (white stone, fluted, two balconies, ~60 m),
  the thicker north-eastern one of Bayezid II (one balcony) and the south-eastern one in red brick (Mehmed II,
  one balcony, white stone top); lead cones and gilded finials.
- The precinct (south and west of the church): the tombs (türbe) of Selim II, Murad III and Mehmed III with
  their porches, drums, lead domes and finials, the tomb of the princes (Şehzadeler), the old baptistery (tomb
  of Mustafa I and İbrahim), the şadırvan of Mahmud I (eight columns under a wide lead roof, the marble basin
  inside), the primary school (sıbyan mektebi), an Ottoman outbuilding beside the şadırvan, the round
  skeuophylakion on the north side, the precinct walls and railing, trees.
- The ground of the precinct (since 2026-11-28; before, the game saw through it and drew its own): it follows the
  terrain the game uses (models/Q12506/terrain.json, SRTM), paved in the west courtyard and lawn in the gardens
  elsewhere, with OpenStreetMap's grass, gardens, paved areas and paths (model_sources/Q12506_ground_osm.json) on
  top, and its 11 mapped trees added to the ones placed by hand (which now stand on that ground too).
- Not modelled: the interiors, the streets and squares outside the railing, the archaeological
  remains of Theodosius' church in front of the west façade, the modern ticket and security buildings.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap way 109862851 (the outline, 83 points) and its 56 building:parts (main body 776020497,
  nave 776020492, dome base 776020498, buttress towers 776020495/6, dome 776067032 Ø 32.9 m, drum 305585921,
  semi-domes 776067030/1 Ø 32.9 m, exedrae 776067024/5, 776086648/9 Ø 12.5 m, apse 776086647, narthex
  776067034, exonarthex 776067033, west buttresses 776020484–87 and 776067028/9, minarets 776020481–91,
  776211441, 227758675, 335371711–15, Ottoman buttresses 776086635–46); the tombs: ways 709381141 (Selim II),
  109836395 (Murad III), 109836394 (Mehmed III), 132277807 (Şehzadeler) and their dome parts 776211438–40;
  the şadırvan 109738930, the sıbyan mektebi 109738926, the outbuilding 109738924, the skeuophylakion 776274478;
  the railing 109794387 and the north wall 103339336, which close the precinct with the east front.
  The edges of the parts run at 303.1° (the narthex, the nave and the buttress towers agree within 0.2°), which is
  the heading of the west front; the origin is the centre of the dome (part 776067032).
- Heights (Wikipedia en, «Hagia Sophia»): dome 55.6 m above the floor at its crown, Ø 31–33 m, 40 windows round
  its base; building 82 × 73 m, 55 m high; the two western minarets 60 m. OpenStreetMap parts: aisles 25 m,
  buttress towers 36 m, dome base 41 m, window ring 40–46 m, dome cap 40–56 m, semi-domes 23–39 m, exedrae to
  30 m, narthex 25 m, exonarthex 9 m, west buttresses 12 m, tomb domes 16–23 m, the school 12 m, the tombs 16–17 m.
  Estimated (±10–25 %): the nave roof 26 m, the window ring of the dome 41–46 m and the shell 46–56.3 m, the
  semi-dome rings 26–33 m and shells 33–40 m, the exedra rings and shells (25.9–31 m), the apse (21 m + half-dome
  to 27.5 m), the slopes of the Ottoman buttresses (12–24 m), the annexes 7 m; the minarets of Bayezid II (55 m,
  balcony at 44 m) and of Mehmed II (53 m, balcony at 42 m), the balconies of Sinan's minarets (36.5 and 47.5 m);
  the drums of the tombs, the şadırvan roof (to 9.5 m), the skeuophylakion (11 m), the walls and the trees.
- Ground: EU-DEM 25 m via OpenTopoData: the west courtyard and the şadırvan ≈ 40 m, the tombs ≈ 40.5–41.5 m,
  the north-west corner of the precinct ≈ 35.5 m, the east front ≈ 46–48 m. The site is kept as it is
  ("ground": "keep"); y = 0 is the courtyard by the şadırvan (= the floor of the church, entered level from the
  west); every wall goes down to −6 m.
Colours: the pinkish-ochre plaster of the walls (Fossati, 1847–49, and later), lead-grey domes and roofs, white
stone of the minarets and tombs, red brick of the south-east minaret, gilded finials.

Axes (docs/MODEL_SPEC.md): +Y up, the west front (exonarthex, Imperial Gate) faces +Z (heading 303.1°, from the
OSM parts), +X points to 213.1° (south-south-west): the tombs lie at +X. Origin at the centre of the dome.

    python model_sources/Q12506_hagia_sophia.py   -> models/Q12506/model.glb (+ footprint in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, extrude, triangulate, fbm, write_glb, METAL  # noqa: E402
from meshkit import _orient  # noqa: E402

rng = np.random.default_rng(537)
HEADING = 303.1

# ── Colours ──────────────────────────────────────────────────────────────────
PLASTER = (0.84, 0.62, 0.50)          # pinkish-ochre plaster
PLASTER_L = (0.88, 0.69, 0.57)        # the upper walls in the sun
PLASTER_D = (0.76, 0.54, 0.43)        # the Ottoman buttresses
TRIM = (0.93, 0.88, 0.78)             # cornices, window frames, relieving arches
LEAD = (0.58, 0.60, 0.63)             # lead domes and roofs
LEAD_D = (0.48, 0.50, 0.53)           # flat lead terraces
WIN = (0.17, 0.18, 0.21)              # windows
DOOR = (0.24, 0.17, 0.13)             # bronze-clad doors
STONE = (0.87, 0.84, 0.77)            # white stone: minarets, tombs
STONE_D = (0.74, 0.71, 0.65)          # porches in shadow
BRICK = (0.64, 0.33, 0.24)            # the red-brick minaret
BRICK_D = (0.55, 0.30, 0.23)          # the skeuophylakion
GOLD = (0.83, 0.66, 0.30)
RAIL = (0.30, 0.31, 0.32)
WALLC = (0.78, 0.72, 0.62)            # rubble walls of the precinct
TRUNK = (0.36, 0.27, 0.19)
CYPRESS = (0.15, 0.29, 0.14)
TREES = [(0.24, 0.40, 0.18), (0.28, 0.44, 0.20), (0.21, 0.36, 0.17)]
WATER = (0.29, 0.43, 0.50)
PAVE = (0.74, 0.72, 0.67)              # stone paving of the courtyard and paths
LAWN = (0.42, 0.52, 0.28)              # the lawns of the gardens

FOUND = -6.0        # walls go down to here (lowest ground around: −4.6)
TY = 0.5            # ground of the tombs above y = 0 (EU-DEM, ±1 m)

# ── Plans (OpenStreetMap, model metres, heading 303.1°, origin = centre of the dome) ──
OUTLINE = [(-37.9, -31.9), (-37.8, -19.9), (-54.8, -20.3), (-57.2, -1.3), (-47.0, -0.7), (-47.4, 4.2), (-37.8, 4.5),
           (-37.7, 8.2), (-49.3, 7.4), (-50.1, 22.2), (-37.6, 22.3), (-37.8, 28.7), (-54.1, 28.6), (-54.3, 43.6),
           (-54.3, 50.6), (-43.0, 50.9), (-43.0, 57.8), (-35.5, 57.7), (-35.5, 65.0), (-41.0, 65.0), (-41.0, 71.1),
           (-25.7, 71.1), (-25.8, 57.8), (-13.4, 57.7), (-13.3, 61.9), (-11.0, 61.9), (-10.9, 57.8), (-6.1, 57.8),
           (-6.1, 61.8), (-3.7, 61.7), (-3.8, 57.7), (1.1, 57.7), (1.1, 61.8), (3.5, 61.9), (3.8, 57.7),
           (7.8, 57.7), (7.8, 62.1), (10.8, 62.0), (10.9, 57.6), (21.6, 57.6), (21.2, 71.9), (30.6, 72.0),
           (30.7, 69.4), (43.5, 69.6), (43.6, 59.0), (54.7, 59.2), (54.9, 48.1), (52.6, 48.1), (52.8, 41.4),
           (74.0, 42.1), (74.4, 35.5), (62.3, 35.1), (62.3, 18.2), (54.8, 17.9), (54.9, 9.2), (36.5, 9.2),
           (36.4, 6.7), (48.4, 6.6), (48.2, -1.7), (36.7, -1.5), (37.3, -8.7), (47.8, -8.9), (47.5, -18.7),
           (37.3, -18.8), (37.4, -25.7), (47.4, -25.7), (47.5, -30.2), (60.9, -29.5), (61.4, -44.9), (39.0, -46.1),
           (39.0, -49.8), (-15.2, -51.7), (-14.7, -61.8), (-26.3, -62.5), (-36.1, -63.2), (-36.1, -55.4),
           (-43.4, -56.1), (-43.3, -48.5), (-41.9, -48.5), (-41.9, -31.8)]
# main body (aisles + galleries), squared: OSM 776020497 runs x −37.8 … 36.1–37.3, z −38.2 … 39.2
MX0, MX1, MZ0, MZ1 = -37.6, 36.8, -38.2, 39.2
NX = 22.6                               # nave and dome base: x ±22.6 (776020492, 776020498)
DZ0, DZ1 = -19.2, 20.8                  # dome base square (776020498)
DOME_C = (0.0, 0.5)                     # centre of the dome cap (776067032), Ø 32.9
SEMI = [(0.0, -18.9, -1), (0.0, 18.3, 1)]                    # semi-domes (776067030 east, 776067031 west)
EXEDRAE = [(13.9, 25.85, 18.3), (-14.25, 25.75, 18.3), (14.35, -25.75, -18.9), (-13.85, -25.85, -18.9)]
TOWERS = [(-17.9, -10.6), (12.0, 19.3)]                      # buttress towers (776020495/6), z ranges
NARTHEX = (-37.4, 36.5, 39.15, 50.15)                        # 776067034
EXO = (-35.6, 29.3, 50.15, 57.7)                             # 776067033
WEST_BUTTRESSES = [(-28.1, -25.7), (-13.3, -10.9), (-6.2, -3.7), (1.1, 3.7), (7.8, 11.1), (21.9, 24.8)]
BUTTRESSES = [                                               # Ottoman buttresses (x outer side)
    [(-47.4, 4.2), (-46.4, -10.0), (-37.9, -9.9), (-37.8, 4.5)],                       # 776086643
    [(-37.6, 22.3), (-50.1, 22.2), (-49.3, 7.4), (-37.7, 8.2)],                        # 776086644
    [(-37.9, -9.9), (-55.5, -10.1), (-54.8, -20.3), (-37.8, -19.9)],                   # 776086646
    [(-49.5, 40.9), (-49.6, 34.3), (-37.9, 34.1), (-37.7, 40.7)],                      # 776086635
    [(36.4, 6.7), (36.7, -1.5), (48.2, -1.7), (48.4, 6.6)],                            # 776086639
    [(50.2, 20.1), (36.4, 20.0), (36.4, 11.8), (50.2, 11.9)],                          # 776086640
    [(47.4, -25.7), (36.4, -25.7), (36.3, -38.2), (47.6, -37.9), (47.5, -30.2)],       # 776086641
    [(47.8, -8.9), (37.3, -8.7), (37.3, -18.8), (47.5, -18.7)],                        # 776086642
]
APSE = [(-2.9, -43.9), (2.8, -43.8), (6.8, -39.7), (6.7, -34.1), (2.6, -30.1), (-3.1, -30.2), (-7.0, -34.3),
        (-6.9, -40.0)]                                                                  # 776086647
EAST_BLOCK = [(-22.3, -38.0), (-22.2, -50.5), (-7.3, -50.4), (-7.4, -37.9)]            # 776086637
NE_ANNEX = [(-36.1, -63.2), (-26.3, -62.5), (-14.7, -61.8), (-15.1, -53.7), (-36.1, -55.4)]   # 776086638
MINARETS = {"NW": (-39.25, 53.95), "SW": (33.05, 53.85), "SE": (32.9, -42.95), "NE": (-39.65, -52.3)}
BAPTISTERY = [(62.0, 32.7), (58.5, 36.0), (49.2, 35.7), (46.1, 32.6), (46.0, 23.3), (49.2, 20.2), (59.0, 20.2),
              (62.1, 23.6)]                                                             # 776086636
SELIM_BODY = [(79.7, -1.5), (66.2, -2.9), (64.2, -5.6), (65.2, -20.1), (68.7, -22.6), (82.5, -21.6), (85.7, -18.2),
              (83.8, -3.6)]                                                             # 709381141
SELIM_PORCH = [(64.6, -6.3), (56.7, -6.8), (57.9, -20.8), (65.2, -20.1)]
MURAD_BODY = [(87.6, 16.3), (78.8, 10.0), (79.8, -0.8), (89.7, -5.3), (98.5, 1.0), (97.5, 11.9)]   # 109836395
MURAD_PORCH = [(78.8, 10.0), (72.3, 9.4), (73.3, -1.5), (79.8, -0.8)]
MEHMED_C = (81.65, -33.5)                                                               # 109836394, 776211438
MEHMED_PORCH = (63.9, 72.4, -44.3, -25.4)
SEHZADE = [(81.4, 18.4), (78.6, 20.3), (75.2, 19.6), (73.3, 16.7), (74.0, 13.3), (76.8, 11.4), (80.2, 12.1),
           (82.1, 15.0)]                                                                # 132277807
SADIRVAN_C = (67.5, 62.35)                                                              # 109738930, Ø 15.4
SCHOOL = [(85.4, 42.8), (85.9, 32.6), (96.2, 33.2), (95.7, 43.4)]                      # 109738926
OUTBUILDING = [(79.2, 68.1), (81.5, 55.0), (91.6, 56.7), (89.3, 69.9)]                 # 109738924
SKEUO_C, SKEUO_R = (-50.95, -29.8), 7.8                                                 # 776274478
RAILING = [(-84.5, 44.0), (-86.6, 68.0), (-89.7, 101.1), (-18.0, 99.8), (54.6, 96.0), (96.2, 85.2), (96.5, 79.6),
           (97.1, 75.3), (98.3, 50.0), (101.8, -42.8), (64.8, -44.8), (61.4, -44.9)]  # 109794387
NORTH_WALL = [(-54.3, 43.6), (-63.0, 43.7), (-84.5, 44.0), (-83.0, 26.3), (-73.4, -42.0), (-72.2, -49.9),
              (-70.5, -51.1), (-67.5, -53.1), (-63.3, -56.2), (-43.4, -56.1)]           # 103339336
GATES = [(-45.0, 100.5, 8.0), (97.6, 65.0, 6.0)]        # gaps in the railing: (x, z, width) — west gate, south gate


# ── Small helpers ────────────────────────────────────────────────────────────
def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def oquad(pts, c, out):
    """A quad whose front faces the direction out."""
    P = np.array(pts, float)
    m = Mesh(P, [(0, 1, 2), (0, 2, 3)], c)
    n = np.cross(P[1] - P[0], P[2] - P[0])
    if np.dot(n, out) < 0:
        m.f = m.f[:, ::-1]
    return m


def win_x(x, sgn, z, y0, y1, w, c=WIN):
    """A window on a wall facing ±x."""
    xx = x + sgn * 0.06
    return oquad([(xx, y0, z - w / 2), (xx, y0, z + w / 2), (xx, y1, z + w / 2), (xx, y1, z - w / 2)], c, (sgn, 0, 0))


def win_z(z, sgn, x, y0, y1, w, c=WIN):
    zz = z + sgn * 0.06
    return oquad([(x - w / 2, y0, zz), (x + w / 2, y0, zz), (x + w / 2, y1, zz), (x - w / 2, y1, zz)], c, (0, 0, sgn))


def arched_win(axis, coord, sgn, u, y0, y1, w, c=WIN, n=5):
    """A round-headed window (flat polygon) on a wall facing ±x (axis 'x') or ±z (axis 'z')."""
    r = w / 2
    pts = [(u - r, y0), (u + r, y0), (u + r, y1 - r)]
    pts += [(u + r * math.cos(t), y1 - r + r * math.sin(t)) for t in np.linspace(0, math.pi, n + 2)[1:-1]]
    pts += [(u - r, y1 - r)]
    p2, tr = triangulate(pts)
    off = coord + sgn * 0.06
    if axis == "x":
        v = np.stack([np.full(len(p2), off), p2[:, 1], p2[:, 0]], 1)
        out = (sgn, 0, 0)
    else:
        v = np.stack([p2[:, 0], p2[:, 1], np.full(len(p2), off)], 1)
        out = (0, 0, sgn)
    m = Mesh(v, tr, c)
    a, b, cc = v[tr[0, 0]], v[tr[0, 1]], v[tr[0, 2]]
    if np.dot(np.cross(b - a, cc - a), out) < 0:
        m.f = m.f[:, ::-1]
    return m


def rev(prof, c, cx=0.0, cz=0.0, a0=0.0, a1=2 * math.pi, seg=24):
    """Surface of revolution of prof [(r, y)] (bottom -> top, outside first) about the vertical axis at (cx, cz),
    over the angles a0 … a1 (a full turn by default). Same winding as meshkit.lathe: normals point out."""
    full = abs((a1 - a0) - 2 * math.pi) < 1e-9
    n = seg if full else seg + 1
    a = np.linspace(a0, a1, seg, endpoint=False) if full else np.linspace(a0, a1, n)
    v, f = [], []
    for r, y in prof:
        v += [(cx + r * math.cos(t), y, cz + r * math.sin(t)) for t in a]
    for i in range(len(prof) - 1):
        for j in range(seg):
            k = (j + 1) % n
            p, q = i * n + j, i * n + k
            f += [(p, p + n, q + n), (p, q + n, q)]
    return Mesh(v, f, c)


def ring_cap(r_out, r_in, y, c, cx=0.0, cz=0.0, a0=0.0, a1=2 * math.pi, seg=24):
    """A flat annulus (or a disc, r_in = 0) facing up."""
    return rev([(r_out, y), (r_in, y)], c, cx, cz, a0, a1, seg)


def half_wall(r, y0, y1, c, cx, cz, a0, a1, seg):
    """The flat side closing a half-dome or half-drum (a vertical slab along the chord)."""
    p0 = (cx + r * math.cos(a0), cz + r * math.sin(a0))
    p1 = (cx + r * math.cos(a1), cz + r * math.sin(a1))
    mid = (a0 + a1) / 2
    out = (-math.cos(mid), 0, -math.sin(mid))
    return oquad([(p0[0], y0, p0[1]), (p1[0], y0, p1[1]), (p1[0], y1, p1[1]), (p0[0], y1, p0[1])], c, out)


def dome_prof(R, y0, H, n=8, r_top=0.0):
    """A shallow dome: r = R sin t, y = y0 + H cos t (t from 90° to 0)."""
    ts = np.linspace(math.pi / 2, 0, n + 1)
    return [(max(R * math.sin(t), r_top), y0 + H * math.cos(t)) for t in ts]


def ring_windows(cx, cz, r, y0, y1, n, w, a0=0.0, a1=2 * math.pi, c=WIN):
    """n dark window panes round a drum (tangent quads just outside radius r), evenly over a0 … a1."""
    ms = []
    step = (a1 - a0) / n
    for i in range(n):
        a = a0 + step * (i + 0.5)
        ca, sa = math.cos(a), math.sin(a)
        tx, tz = -sa, ca
        rr = r + 0.07
        px, pz = cx + rr * ca, cz + rr * sa
        pts = [(px - tx * w / 2, y0, pz - tz * w / 2), (px + tx * w / 2, y0, pz + tz * w / 2),
               (px + tx * w / 2, y1, pz + tz * w / 2), (px - tx * w / 2, y1, pz - tz * w / 2)]
        ms.append(oquad(pts, c, (ca, 0, sa)))
    return ms


def radial_box(cx, cz, a, r0, r1, w, y0, y1, c, y1_out=None):
    """A small radial buttress from radius r0 to r1 at angle a, its top sloping from y1 to y1_out."""
    y1o = y1 if y1_out is None else y1_out
    ca, sa = math.cos(a), math.sin(a)
    tx, tz = -sa, ca
    P = []
    for r in (r0, r1):
        for s in (-1, 1):
            P.append((cx + r * ca + s * tx * w / 2, cz + r * sa + s * tz * w / 2))
    ring = [P[0], P[2], P[3], P[1]]
    tops = {P[0]: y1, P[1]: y1, P[2]: y1o, P[3]: y1o}
    return wedge(ring, y0, lambda x, z: tops.get((x, z), y1), c, c)


def wedge(ring, y0, topf, c, top_c=None):
    """A prism whose top follows topf(x, z) (a sloping buttress, a skillion roof)."""
    P = _orient(ring, True)
    v, f = [], []
    for i in range(len(P)):
        (x0, z0), (x1, z1) = P[i], P[(i + 1) % len(P)]
        o = len(v)
        v += [(x0, y0, z0), (x1, y0, z1), (x1, topf(x1, z1), z1), (x0, topf(x0, z0), z0)]
        f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    m = Mesh(v, f, c)
    pts, tris = triangulate(P)
    tv = np.stack([pts[:, 0], [topf(float(x), float(z)) for x, z in pts], pts[:, 1]], 1)
    return merge([m, Mesh(tv, tris[:, ::-1], c if top_c is None else top_c)])


def hip(x0, x1, z0, z1, y0, h, c, ridge=True):
    """A hipped (or pyramidal) lead roof over a rectangle."""
    w, d = x1 - x0, z1 - z0
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    if not ridge or abs(w - d) < 0.5:
        a = b = 0.0
    elif w > d:
        a, b = (w - d) / 2, 0.0
    else:
        a, b = 0.0, (d - w) / 2
    top = [(cx - a, y0 + h, cz - b), (cx + a, y0 + h, cz + b)]
    c4 = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
    ms = []
    centre = np.array([cx, y0 - 1.0, cz])
    if w >= d:                                          # the four slopes, each oriented out from the centre
        faces = [(c4[0], c4[1], top[1], top[0]), (c4[2], c4[3], top[0], top[1]), (c4[3], c4[0], top[0]), (c4[1], c4[2], top[1])]
    else:
        faces = [(c4[0], c4[1], top[0]), (c4[2], c4[3], top[1]), (c4[3], c4[0], top[0], top[1]), (c4[1], c4[2], top[1], top[0])]
    for fc in faces:
        P = np.array(fc, float)
        out = P.mean(0) - centre
        if len(fc) == 4:
            ms.append(oquad(fc, c, out))
        else:
            m = Mesh(P, [(0, 1, 2)], c)
            if np.dot(np.cross(P[1] - P[0], P[2] - P[0]), out) < 0:
                m.f = m.f[:, ::-1]
            ms.append(m)
    return merge(ms)


def finial(x, y, z, s=1.0):
    """The gilded alem: stacked balls and the crescent, s = scale (1 ≈ 4 m)."""
    ms = [cyl(0.06 * s, 0.06 * s, 2.6 * s, GOLD, 6, caps=False).T(x, y, z)]
    for yb, rb in ((0.55, 0.32), (1.25, 0.26), (1.85, 0.2)):
        ms.append(lathe([(1e-3, -rb), (rb * 0.87, -rb * 0.5), (rb, 0), (rb * 0.87, rb * 0.5), (1e-3, rb)],
                        GOLD, 8).S(s).T(x, y + yb * s, z))
    pts = [(x + 0.55 * s * math.sin(t), y + (2.75 + 0.55 * math.cos(t)) * s, z)
           for t in np.linspace(-2.3, 2.3, 9)]
    ms.append(tube(pts, 0.08 * s, GOLD, 5))
    return ms


def wall_run(x0, z0, x1, z1, y0, h, t, c, cap_c=None):
    L = math.hypot(x1 - x0, z1 - z0)
    a = math.atan2(x1 - x0, z1 - z0)
    ms = [box(t, h - y0, L, c).T(0, y0, 0)]
    if cap_c is not None:
        ms.append(box(t + 0.3, 0.25, L, cap_c).T(0, h, 0))
    return [m.R(0, a, 0).T((x0 + x1) / 2, 0, (z0 + z1) / 2) for m in ms]


def poly_center(P):
    P = np.array(P, float)
    return float(P[:, 0].mean()), float(P[:, 1].mean())


def regular(cx, cz, R, n, rot=0.0):
    return [(cx + R * math.cos(rot + 2 * math.pi * i / n), cz + R * math.sin(rot + 2 * math.pi * i / n)) for i in range(n)]


def tree(x, z, y0=0.0, kind=None):
    k = kind if kind is not None else int(rng.integers(0, 3))
    if k == 0:                                           # cypress
        h = rng.uniform(9, 13)
        return [cyl(0.25, 0.2, 1.5, TRUNK, 5).T(x, y0 - 1, z),
                lathe([(0.01, 0.0), (1.3, 1.5), (1.4, h * 0.45), (0.9, h * 0.8), (0.01, h)], CYPRESS, 7).T(x, y0, z)]
    h = rng.uniform(8, 13); r = rng.uniform(3.5, 5.5)
    col = TREES[int(rng.integers(0, len(TREES)))]
    crown = lathe([(0.01, -r * 0.85), (r * 0.75, -r * 0.5), (r, 0), (r * 0.8, r * 0.55), (0.01, r * 0.9)], col, 8)
    return [cyl(0.35, 0.28, h - r, TRUNK, 6).T(x, y0 - 1, z), crown.T(x, y0 + h - r * 0.5, z)]


# ── The church ───────────────────────────────────────────────────────────────
def annexes():
    """The OSM outline as a low mass (ramps, annexes, the sultan's lodge): the rest stands on it."""
    ms = [extrude(OUTLINE, FOUND, 7.0, PLASTER_D, top_c=LEAD_D)]
    ms.append(extrude(EAST_BLOCK, FOUND, 14.0, PLASTER_D, top_c=LEAD_D))
    ms.append(wedge(NE_ANNEX, FOUND, lambda x, z: 9.0 + 0.2 * (z + 63.2), PLASTER_D, LEAD))
    return ms


def main_body():
    ms = []
    rect = [(MX0, MZ0), (MX1, MZ0), (MX1, MZ1), (MX0, MZ1)]
    ms.append(extrude(rect, FOUND, 24.4, PLASTER, top_c=LEAD_D))
    ms.append(extrude([(MX0 - 0.35, MZ0 - 0.35), (MX1 + 0.35, MZ0 - 0.35), (MX1 + 0.35, MZ1 + 0.35), (MX0 - 0.35, MZ1 + 0.35)],
                      24.4, 25.0, TRIM, top_c=LEAD_D))
    # aisle roofs: low lead hips on each side of the nave
    for x0, x1 in ((MX0, -NX), (NX, MX1)):
        ms.append(hip(x0, x1, MZ0, MZ1, 25.0, 2.2, LEAD))
    # windows: ground floor and galleries on the north and south walls, the east wall
    for x, s in ((MX0, -1), (MX1, 1)):
        for z in np.arange(-35.0, 37.5, 4.2):
            ms.append(arched_win("x", x, s, float(z), 4.0, 9.0, 1.8))
            ms.append(arched_win("x", x, s, float(z), 13.5, 20.0, 2.4))
    for xw in np.arange(-34.0, 34.5, 4.25):
        if abs(xw) < 8.5:
            continue
        ms.append(arched_win("z", MZ0, -1, float(xw), 4.0, 9.0, 1.8))
        ms.append(arched_win("z", MZ0, -1, float(xw), 13.5, 20.0, 2.4))
    # the raised nave
    nave = [(-NX, MZ0), (NX, MZ0), (NX, NARTHEX[2]), (-NX, NARTHEX[2])]
    ms.append(extrude(nave, 20.0, 26.0, PLASTER_L, top_c=LEAD_D))
    for x, s in ((-NX, -1), (NX, 1)):
        for z in list(np.arange(-34.0, -21.0, 3.2)) + list(np.arange(24.0, 37.0, 3.2)):
            ms.append(arched_win("x", x, s, float(z), 22.0, 25.0, 1.4))
    return ms


def dome_base():
    ms = []
    ms.append(extrude([(-NX, DZ0), (NX, DZ0), (NX, DZ1), (-NX, DZ1)], 20.0, 41.0, PLASTER_L, top_c=LEAD_D))
    ms.append(extrude([(-NX - 0.3, DZ0 - 0.3), (NX + 0.3, DZ0 - 0.3), (NX + 0.3, DZ1 + 0.3), (-NX - 0.3, DZ1 + 0.3)],
                      40.4, 41.0, TRIM, top_c=LEAD_D))
    # the great tympana of the north and south arches, between the buttress towers
    z0, z1 = TOWERS[0][1], TOWERS[1][0]
    zc, R = (z0 + z1) / 2, (z1 - z0) / 2
    for x, s in ((-NX, -1), (NX, 1)):
        for k in range(7):
            ms.append(arched_win("x", x, s, z0 + 1.7 + k * (z1 - z0 - 3.4) / 6, 27.0, 31.0, 1.7))
        for k in range(5):
            ms.append(arched_win("x", x, s, zc - 6.4 + k * 3.2, 32.5, 36.0, 1.6))
        pts = [(x + s * 0.25, 29.0 + R * 0.62 * math.sin(t), zc + R * math.cos(t)) for t in np.linspace(0, math.pi, 17)]
        ms.append(tube(pts, 0.35, TRIM, 5))
    # the four buttress towers on the north and south sides
    for za, zb in TOWERS:
        for xa, xb in ((MX0, -NX), (NX, MX1)):
            ms.append(extrude([(xa, za), (xb, za), (xb, zb), (xa, zb)], 20.0, 36.0, PLASTER, top_c=LEAD_D))
            ms.append(hip(xa - 0.2, xb + 0.2, za - 0.2, zb + 0.2, 36.0, 2.6, LEAD))
            xo, so = (xa, -1) if xa < 0 else (xb, 1)
            for zz in (za + (zb - za) * 0.3, za + (zb - za) * 0.7):
                ms.append(arched_win("x", xo, so, zz, 27.0, 31.0, 1.2))
    # little domed turrets at the west end of the nave (776067026/7)
    for cx, cz in ((10.35, 35.75), (-10.15, 35.7)):
        ms.append(cyl(1.75, 1.75, 5.6, PLASTER_L, 12).T(cx, 26.0, cz))
        ms.append(rev(dome_prof(1.75, 31.6, 2.4, 4), LEAD, cx, cz, seg=12))
    return ms


def main_dome():
    ms = []
    cx, cz = DOME_C
    R = 16.6
    # the window ring with its 40 windows and 40 little buttresses
    ms.append(cyl(R, R, 5.6, PLASTER_L, 80, caps=False).T(cx, 40.9, cz))
    ms += ring_windows(cx, cz, R, 42.0, 45.0, 40, 1.45)
    for i in range(40):
        a = 2 * math.pi * i / 40
        ms.append(radial_box(cx, cz, a, R - 0.2, R + 2.0, 1.05, 40.9, 46.2, PLASTER, 44.0))
    ms.append(rev([(R + 0.4, 46.2), (R + 0.4, 46.6), (R - 1.0, 46.6)], TRIM, cx, cz, seg=80))
    # the shallow lead cap and its finial
    prof = dome_prof(R - 0.2, 46.6, 9.7, 12, r_top=0.0)
    ms.append(rev(prof, LEAD, cx, cz, seg=80))
    for i in range(40):                                  # the 40 ribs of the lead covering
        a = 2 * math.pi * (i + 0.5) / 40
        pts = [(cx + (r + 0.12) * math.cos(a), y + 0.05, cz + (r + 0.12) * math.sin(a)) for r, y in prof[:-3]]
        ms.append(tube(pts, 0.16, LEAD_D, 4))
    ms.append(cyl(0.9, 0.6, 0.6, LEAD, 10).T(cx, 56.2, cz))
    ms += finial(cx, 56.7, cz, 1.25)
    return ms


def semi_domes():
    ms = []
    R = 16.4
    for cx, cz, s in SEMI:
        a0, a1 = (math.pi, 2 * math.pi) if s < 0 else (0.0, math.pi)
        ms.append(rev([(R, 26.0), (R, 33.0)], PLASTER_L, cx, cz, a0, a1, 36))
        ms += ring_windows(cx, cz, R, 30.4, 32.6, 5, 1.6, a0, a1)
        for i in range(1, 5):
            ms.append(radial_box(cx, cz, a0 + math.pi * i / 5, R - 0.2, R + 1.5, 0.9, 26.0, 33.4, PLASTER, 31.5))
        ms.append(rev([(R + 0.3, 33.0), (R + 0.3, 33.4), (R - 0.8, 33.4)], TRIM, cx, cz, a0, a1, 36))
        ms.append(rev(dome_prof(R - 0.1, 33.4, 6.6, 9), LEAD, cx, cz, a0, a1, 36))
        ms.append(half_wall(R, 26.0, 40.0, PLASTER_L, cx, cz, a0, a1, 36))
    # the exedrae: small half-domes facing out from the semi-dome centres
    r = 6.25
    for ex, ez, sz in EXEDRAE:
        mid = math.atan2(ez - sz, ex - 0.0)
        a0, a1 = mid - math.pi / 2, mid + math.pi / 2
        ms.append(rev([(r, 22.0), (r, 27.6)], PLASTER_L, ex, ez, a0, a1, 16))
        ms += ring_windows(ex, ez, r, 25.9, 27.2, 3, 1.1, a0, a1)
        ms.append(rev([(r + 0.25, 27.6), (r + 0.25, 27.9), (r - 0.6, 27.9)], TRIM, ex, ez, a0, a1, 16))
        ms.append(rev(dome_prof(r - 0.1, 27.9, 3.4, 6), LEAD, ex, ez, a0, a1, 16))
        ms.append(half_wall(r, 22.0, 31.3, PLASTER_L, ex, ez, a0, a1, 16))
    # the apse: its polygonal wall with windows and its half-dome
    ms.append(extrude(APSE, FOUND, 21.0, PLASTER, top_c=LEAD_D))
    for k in range(3):
        a = math.pi * (1.25 + 0.25 * k)
        x, z = 6.6 * math.cos(a), -37.0 + 6.6 * math.sin(a)
        for y0, y1 in ((5.0, 10.0), (13.0, 18.0)):
            ca, sa = math.cos(a), math.sin(a)
            ms.append(oquad([(x + sa * 0.9, y0, z - ca * 0.9), (x - sa * 0.9, y0, z + ca * 0.9),
                             (x - sa * 0.9, y1, z + ca * 0.9), (x + sa * 0.9, y1, z - ca * 0.9)], WIN, (ca, 0, sa)))
    ms.append(rev(dome_prof(6.9, 21.0, 5.6, 7), LEAD, 0.0, -37.0, math.pi, 2 * math.pi, 16))
    return ms


def narthex():
    ms = []
    x0, x1, z0, z1 = NARTHEX
    # inner narthex with the gallery above it: skillion roof falling to the west (OSM roof:direction 302.9)
    ms.append(wedge([(x0, z0), (x1, z0), (x1, z1), (x0, z1)], FOUND, lambda x, z: 25.0 - 3.8 * (z - z0) / (z1 - z0),
                    PLASTER, LEAD))
    for xw in np.arange(x0 + 4.0, x1 - 3.0, 3.6):
        ms.append(arched_win("z", z1, 1, float(xw), 12.5, 18.5, 2.0))
    # exonarthex
    ex0, ex1, ez0, ez1 = EXO
    ms.append(wedge([(ex0, ez0), (ex1, ez0), (ex1, ez1), (ex0, ez1)], FOUND, lambda x, z: 9.0 - 1.0 * (z - ez0) / (ez1 - ez0),
                    PLASTER_L, LEAD))
    for xw in np.arange(ex0 + 2.5, ex1 - 1.5, 3.25):
        if abs(xw + 2.3) < 2.0:
            ms.append(arched_win("z", ez1, 1, float(xw), 0.0, 6.0, 3.0, DOOR))
        else:
            ms.append(arched_win("z", ez1, 1, float(xw), 2.5, 6.0, 1.6))
    # the buttresses of the west front, sloping from the narthex down over the exonarthex
    for xa, xb in WEST_BUTTRESSES:
        ms.append(wedge([(xa, 50.1), (xb, 50.1), (xb, 62.0), (xa, 62.0)], FOUND,
                        lambda x, z: 20.0 - 13.0 * (z - 50.1) / 11.9, PLASTER_D, PLASTER_D))
    return ms


def ottoman_buttresses():
    ms = []
    for ring in BUTTRESSES:
        xs = [p[0] for p in ring]
        xin = MX0 if xs[0] < 0 else MX1
        xout = min(xs) if xs[0] < 0 else max(xs)
        span = abs(xout - xin) or 1.0
        ms.append(wedge(ring, FOUND, lambda x, z, xin=xin, span=span: 24.0 - 7.5 * min(abs(x - xin) / span, 1.0),
                        PLASTER_D, PLASTER_D))
    # the great eastern buttress pair beside the apse and the east block
    return ms


# ── Minarets ─────────────────────────────────────────────────────────────────
def balcony(x, z, y, r_shaft, r_b, c, rail_c=None):
    """A şerefe: corbelled cone under it, the floor, a parapet."""
    rc = c if rail_c is None else rail_c
    return [rev([(r_shaft, y - 1.6), (r_b, y), (r_b, y + 0.15), (r_shaft, y + 0.15)], c, x, z, seg=16),
            rev([(r_b, y + 0.15), (r_b, y + 1.15), (r_b - 0.15, y + 1.15), (r_b - 0.15, y + 0.15)], rc, x, z, seg=16)]


def minaret(key):
    x, z = MINARETS[key]
    ms = []
    if key in ("NW", "SW"):                  # Sinan: square base, fluted shaft, two balconies, ~60 m
        ms.append(B(x - 3.7, x + 3.7, FOUND, 14.0, z - 3.75, z + 3.75, STONE))
        ms.append(B(x - 3.9, x + 3.9, 13.6, 14.3, z - 3.95, z + 3.95, TRIM))
        ms.append(rev([(3.3, 14.3), (2.25, 17.5)], STONE, x, z, seg=20))
        ms.append(rev([(2.0, 17.5), (2.0, 47.5)], STONE, x, z, seg=20))
        for yb in (36.5, 47.5):
            ms += balcony(x, z, yb, 2.0, 3.0, STONE)
        ms.append(rev([(1.75, 47.6), (1.75, 52.6), (2.05, 52.9)], STONE, x, z, seg=20))
        ms.append(rev([(2.05, 52.9), (0.05, 59.6)], LEAD, x, z, seg=20))
        ms += finial(x, 59.4, z, 0.55)
    elif key == "NE":                         # Bayezid II: thicker shaft, one balcony
        ms.append(B(x - 3.7, x + 3.7, FOUND, 12.0, z - 3.8, z + 3.8, STONE))
        ms.append(rev([(3.4, 12.0), (2.5, 15.0)], STONE, x, z, seg=20))
        ms.append(rev([(2.4, 15.0), (2.4, 44.0)], STONE, x, z, seg=20))
        ms += balcony(x, z, 44.0, 2.4, 3.4, STONE)
        ms.append(rev([(2.1, 44.1), (2.1, 47.8), (2.4, 48.1)], STONE, x, z, seg=20))
        ms.append(rev([(2.4, 48.1), (0.05, 54.6)], LEAD, x, z, seg=20))
        ms += finial(x, 54.4, z, 0.55)
    else:                                     # SE, Mehmed II: red brick, one balcony, white stone top
        ms.append(B(x - 3.7, x + 3.7, FOUND, 10.0, z - 3.75, z + 3.75, BRICK))
        ms.append(rev([(3.2, 10.0), (2.2, 13.0)], BRICK, x, z, seg=20))
        ms.append(rev([(2.05, 13.0), (2.05, 42.0)], BRICK, x, z, seg=20))
        for yb in (20.0, 28.0, 36.0):
            ms.append(rev([(2.12, yb), (2.12, yb + 0.35)], STONE, x, z, seg=20))
        ms += balcony(x, z, 42.0, 2.05, 3.0, STONE)
        ms.append(rev([(1.8, 42.1), (1.8, 45.6), (2.1, 45.9)], STONE, x, z, seg=20))
        ms.append(rev([(2.1, 45.9), (0.05, 52.2)], LEAD, x, z, seg=20))
        ms += finial(x, 52.0, z, 0.5)
    return ms


# ── Tombs and the precinct ───────────────────────────────────────────────────
def tomb_dome(cx, cz, y0, R, H, drum_h, c_drum=STONE, n_win=8):
    ms = [rev([(R + 0.5, y0), (R + 0.5, y0 + drum_h)], c_drum, cx, cz, seg=16),
          rev([(R + 0.5, y0 + drum_h), (R + 0.8, y0 + drum_h), (R + 0.8, y0 + drum_h + 0.35), (R, y0 + drum_h + 0.35)],
              TRIM, cx, cz, seg=16)]
    if n_win:
        ms += ring_windows(cx, cz, R + 0.5, y0 + drum_h * 0.25, y0 + drum_h * 0.8, n_win, R * 0.22)
    ms.append(rev(dome_prof(R, y0 + drum_h + 0.35, H, 7), LEAD, cx, cz, seg=24))
    ms += finial(cx, y0 + drum_h + 0.35 + H - 0.1, cz, 0.45)
    return ms


def porch(ring, y_top, c=STONE, n_cols=4):
    """A portico: a lead roof on columns along its outer side, a shaded back wall."""
    ms = [extrude(ring, y_top - 0.8, y_top, STONE, top_c=LEAD_D)]
    P = np.array(ring, float)
    ms.append(extrude(ring, FOUND, TY + 0.4, STONE_D, top_c=STONE_D))
    # columns at the corners and along the long outer edge
    a, b = P[1], P[2]
    for t in np.linspace(0, 1, n_cols):
        p = a + (b - a) * t
        ms.append(cyl(0.35, 0.35, y_top - 0.8 - TY, STONE, 8).T(p[0] + (P.mean(0)[0] - p[0]) * 0.08, TY + 0.4,
                                                                 p[1] + (P.mean(0)[1] - p[1]) * 0.08))
    return ms


def tombs():
    ms = []
    # Selim II (Sinan, 1576–77)
    c = (75.05, -12.4)
    ms.append(extrude(SELIM_BODY, FOUND, TY + 13.0, STONE, top_c=LEAD_D))
    ms += porch(SELIM_PORCH, TY + 7.5)
    ms += tomb_dome(c[0], c[1], TY + 13.0, 8.3, 5.4, 2.4)
    for p in SELIM_BODY:
        ms.append(cyl(0.5, 0.05, 1.6, LEAD, 6).T(p[0] + (c[0] - p[0]) * 0.05, TY + 13.0, p[1] + (c[1] - p[1]) * 0.05))
    # Murad III (Davud Ağa, 1594–99): hexagonal
    c = (88.55, 5.55)
    ms.append(extrude(MURAD_BODY, FOUND, TY + 14.0, STONE, top_c=LEAD_D))
    ms += porch(MURAD_PORCH, TY + 8.0)
    ms += tomb_dome(c[0], c[1], TY + 14.0, 8.2, 5.6, 2.2)
    # Mehmed III (Dalgıç Ahmed Ağa, 1608): octagonal
    c = MEHMED_C
    octo = regular(c[0], c[1], 10.6, 8, math.pi / 8)
    ms.append(extrude(octo, FOUND, TY + 14.0, STONE, top_c=LEAD_D))
    x0, x1, z0, z1 = MEHMED_PORCH
    ms += porch([(x1, z0), (x0, z0), (x0, z1), (x1, z1)], TY + 8.0, n_cols=5)
    ms += tomb_dome(c[0], c[1], TY + 14.0, 8.3, 5.6, 2.2)
    # windows on the tomb walls (two rows)
    for ring, (cx, cz), y_top in ((SELIM_BODY, (75.05, -12.4), 13.0), (MURAD_BODY, (88.55, 5.55), 14.0), (octo, c, 14.0)):
        P = _orient(ring, True)
        for i in range(len(P)):
            (ax, az), (bx, bz) = P[i], P[(i + 1) % len(P)]
            L = math.hypot(bx - ax, bz - az)
            if L < 4.0:
                continue
            mx, mz = (ax + bx) / 2, (az + bz) / 2
            nx, nz = (bz - az) / L, -(bx - ax) / L          # outward normal of a ccw ring
            tx, tz = (bx - ax) / L, (bz - az) / L
            for yy0, yy1 in ((TY + 2.5, TY + 5.5), (TY + 8.0, TY + 11.0)):
                w = 1.2
                px, pz = mx + nx * 0.07, mz + nz * 0.07
                ms.append(oquad([(px - tx * w / 2, yy0, pz - tz * w / 2), (px + tx * w / 2, yy0, pz + tz * w / 2),
                                 (px + tx * w / 2, yy1, pz + tz * w / 2), (px - tx * w / 2, yy1, pz - tz * w / 2)],
                                WIN, (nx, 0, nz)))
    # Şehzadeler türbesi (tomb of the princes)
    cx, cz = poly_center(SEHZADE)
    ms.append(extrude(SEHZADE, FOUND, TY + 8.0, STONE, top_c=LEAD_D))
    ms += tomb_dome(cx, cz, TY + 8.0, 3.9, 2.8, 1.0, n_win=0)
    # the old baptistery, tomb of Mustafa I and İbrahim (plastered like the church)
    cx, cz = poly_center(BAPTISTERY)
    ms.append(extrude(BAPTISTERY, FOUND, 11.0, PLASTER, top_c=LEAD_D))
    ms += tomb_dome(cx, cz, 11.0, 6.6, 4.0, 1.8, c_drum=PLASTER)
    return ms


def precinct():
    ms = []
    # şadırvan of Mahmud I (1740): eight columns, a wide lead roof, the marble basin inside
    cx, cz = SADIRVAN_C
    ms.append(rev([(7.4, -1.0), (7.4, TY + 0.4), (0.0, TY + 0.4)], STONE_D, cx, cz, seg=8))
    for i in range(8):
        a = 2 * math.pi * (i + 0.5) / 8
        ms.append(cyl(0.3, 0.3, 4.6, STONE, 8).T(cx + 5.9 * math.cos(a), TY + 0.4, cz + 5.9 * math.sin(a)))
    ms.append(rev([(3.4, TY + 0.4), (3.4, TY + 2.0), (3.1, TY + 2.0), (3.1, TY + 0.4)], STONE, cx, cz, seg=8))
    ms.append(ring_cap(3.1, 0.0, TY + 1.7, WATER, cx, cz, seg=8))
    ms.append(rev([(0.9, TY + 1.7), (0.6, TY + 3.2), (0.01, TY + 3.4)], STONE, cx, cz, seg=8))
    ms.append(rev([(6.4, TY + 4.9), (8.1, TY + 5.0), (8.1, TY + 5.5), (6.3, TY + 7.4), (2.0, TY + 8.6),
                   (0.01, TY + 8.9)], LEAD, cx, cz, seg=20))
    ms += finial(cx, TY + 8.8, cz, 0.3)
    # sıbyan mektebi (primary school of Mahmud I)
    ms.append(extrude(SCHOOL, FOUND, TY + 8.0, STONE, top_c=LEAD_D))
    sx, sz = poly_center(SCHOOL)
    ms.append(rev(dome_prof(4.2, TY + 8.0, 2.8, 6), LEAD, sx, sz, seg=20))
    ms += finial(sx, TY + 10.7, sz, 0.3)
    # the Ottoman outbuilding beside the şadırvan
    ms.append(extrude(OUTBUILDING, FOUND, TY + 6.0, STONE, top_c=LEAD_D))
    ox, oz = poly_center(OUTBUILDING)
    ms.append(rev(dome_prof(4.3, TY + 6.0, 2.4, 6), LEAD, ox, oz, seg=20))
    # skeuophylakion (round treasury of the church, north side)
    cx, cz = SKEUO_C
    ms.append(cyl(SKEUO_R, SKEUO_R, 9.0 - FOUND, BRICK_D, 24).T(cx, FOUND, cz))
    ms += ring_windows(cx, cz, SKEUO_R, 5.0, 7.5, 8, 1.2)
    ms.append(rev(dome_prof(SKEUO_R + 0.2, 9.0, 2.2, 5), LEAD, cx, cz, seg=24))
    # the railing (on a low wall) and the north wall, with gaps at the gates
    for i in range(len(RAILING) - 1):
        (x0, z0), (x1, z1) = RAILING[i], RAILING[i + 1]
        L = math.hypot(x1 - x0, z1 - z0)
        cuts = [0.0, 1.0]
        for gx, gz, gw in GATES:
            t = ((gx - x0) * (x1 - x0) + (gz - z0) * (z1 - z0)) / (L * L)
            d = abs((gx - x0) * (z1 - z0) - (gz - z0) * (x1 - x0)) / L
            if 0 < t < 1 and d < 2.0:
                cuts = [0.0, t - gw / 2 / L, t + gw / 2 / L, 1.0]
                for s in (-1, 1):
                    px, pz = x0 + (x1 - x0) * (t + s * gw / 2 / L), z0 + (z1 - z0) * (t + s * gw / 2 / L)
                    ms.append(B(px - 0.6, px + 0.6, -2.0, 3.2, pz - 0.6, pz + 0.6, STONE))
        for k in range(0, len(cuts), 2):
            ax, az = x0 + (x1 - x0) * cuts[k], z0 + (z1 - z0) * cuts[k]
            bx, bz = x0 + (x1 - x0) * cuts[k + 1], z0 + (z1 - z0) * cuts[k + 1]
            ms += wall_run(ax, az, bx, bz, -3.0, 0.9, 0.6, WALLC, STONE)
            ms += wall_run(ax, az, bx, bz, 2.05, 2.15, 0.1, RAIL)
            ms += wall_run(ax, az, bx, bz, 1.2, 1.3, 0.1, RAIL)
            Ls = math.hypot(bx - ax, bz - az)
            for t in np.linspace(0, 1, max(2, int(Ls / 2.5) + 1)):
                px, pz = ax + (bx - ax) * t, az + (bz - az) * t
                ms.append(box(0.1, 1.05, 0.1, RAIL).T(px, 1.15, pz))
    for i in range(len(NORTH_WALL) - 1):
        (x0, z0), (x1, z1) = NORTH_WALL[i], NORTH_WALL[i + 1]
        ms += wall_run(x0, z0, x1, z1, -4.0, 3.5, 0.9, WALLC, STONE)
    # trees of the precinct (the gardens round the tombs and the west courtyard)
    spots = [(48.0, 52.0, 0), (52.0, 80.0, 1), (66.0, 84.0, 2), (82.0, 80.0, 0), (30.0, 88.0, 1), (10.0, 90.0, 0),
             (-12.0, 90.0, 1), (-35.0, 90.0, 0), (-58.0, 92.0, 2), (-76.0, 82.0, 1), (-78.0, 60.0, 0),
             (-70.0, 18.0, 1), (-66.0, -10.0, 0), (-64.0, -40.0, 2), (92.0, 24.0, 1), (95.0, -18.0, 0),
             (96.0, 60.0, 2), (68.0, 30.0, 0), (96.0, -36.0, 1), (66.0, 46.0, 0), (-68.0, 38.0, 0)]
    for x, z, k in spots:
        ms += tree(x, z, float(ground_y(x, z)), k)
    return ms


# ── The ground of the precinct ───────────────────────────────────────────
def _terrain():
    """The terrain the game puts here (models/Q12506/terrain.json), as model y (0 at ground_at)."""
    import base64
    t = json.loads((ROOT / "models" / "Q12506" / "terrain.json").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "models" / "Q12506" / "meta.json").read_text(encoding="utf-8"))
    n = t["n"]
    H = np.frombuffer(base64.b64decode(t["h"]), "<i2").reshape(n, n).astype(float) / 10.0 + t["off"]
    th = math.pi - math.radians(meta["heading"]); c, s_ = math.cos(th), math.sin(th)

    def T(x, z):
        x = np.asarray(x, float); z = np.asarray(z, float)
        e = x * c + z * s_; no = -(-x * s_ + z * c)
        fi = np.clip((e - t["x0"]) / t["c"], 0, n - 1.0001); fj = np.clip((t["y0"] - no) / t["c"], 0, n - 1.0001)
        i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int); a = fi - i0; b = fj - j0
        return H[j0, i0] * (1 - a) * (1 - b) + H[j0, i0 + 1] * a * (1 - b) + H[j0 + 1, i0] * (1 - a) * b + H[j0 + 1, i0 + 1] * a * b
    ga = meta.get("ground_at") or [0, 0]
    t0 = float(T(*ga))
    return lambda x, z: T(x, z) - t0


_T_CACHE = []


def ground_y(x, z):
    if not _T_CACHE:
        _T_CACHE.append(_terrain())
    return _T_CACHE[0](x, z)


def _osm_ground():
    j = json.loads((ROOT / "model_sources" / "Q12506_ground_osm.json").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "models" / "Q12506" / "meta.json").read_text(encoding="utf-8"))
    lat0, lon0 = meta["position"]["lat"], meta["position"]["lon"]
    th = math.pi - math.radians(meta["heading"]); c, s_ = math.cos(th), math.sin(th)
    ke = 111320.0 * math.cos(math.radians(lat0))

    def xz(ll):
        A = np.array(ll, float).reshape(-1, 2)
        e = (A[:, 1] - lon0) * ke; so = (lat0 - A[:, 0]) * 110574.0
        return np.stack([c * e - s_ * so, s_ * e + c * so], 1)
    return ([xz(g["ll"]) for g in j["grass"]], [xz(g["ll"]) for g in j["paved_areas"]],
            [xz(g["ll"]) for g in j["paths"]], [tuple(xz(g["ll"])[0]) for g in j["trees"]])


def _inside(px, pz, P):
    P = np.asarray(P, float); ins = np.zeros(len(px), bool)
    for i in range(len(P)):
        (xi, zi), (xj, zj) = P[i], P[i - 1]
        m = (zi > pz) != (zj > pz)
        xc = (xj - xi) * (pz - zi) / np.where(zj - zi == 0, 1e-12, zj - zi) + xi
        ins ^= m & (px < xc)
    return ins


def _dist_lines(px, pz, lines):
    d = np.full(len(px), np.inf)
    for L in lines:
        for k in range(len(L) - 1):
            (ax, az), (bx, bz) = L[k], L[k + 1]
            dx, dz = bx - ax, bz - az; l2 = dx * dx + dz * dz
            if l2 < 1e-9:
                continue
            t = np.clip(((px - ax) * dx + (pz - az) * dz) / l2, 0, 1)
            d = np.minimum(d, np.hypot(px - ax - dx * t, pz - az - dz * t))
    return d


def precinct_ground(cell=2.0):
    """A sheet over the precinct (inside the footprint, outside the buildings) on the terrain, 0.05 m above it,
    paved in the west courtyard and lawn elsewhere, with OSM's grass, paved areas and paths; and OSM's trees."""
    F = np.array(FOOTPRINT, float)
    x0, z0 = F.min(0); x1, z1 = F.max(0)
    xs = np.arange(x0, x1 + cell, cell); zs = np.arange(z0, z1 + cell, cell)
    X, Z = np.meshgrid(xs[:-1] + cell / 2, zs[:-1] + cell / 2)
    cx, cz = X.ravel(), Z.ravel()
    keep = _inside(cx, cz, F)
    for P in [OUTLINE, BAPTISTERY, SELIM_BODY, SELIM_PORCH, MURAD_BODY, MURAD_PORCH, SEHZADE, SCHOOL, OUTBUILDING]:
        keep &= ~_inside(cx, cz, P)
    for (qx, qz), r in ((SKEUO_C, SKEUO_R), (SADIRVAN_C, 7.6), (MEHMED_C, 9.0)):
        keep &= np.hypot(cx - qx, cz - qz) > r
    x0m, x1m, z0m, z1m = MEHMED_PORCH
    keep &= ~((cx > x0m) & (cx < x1m) & (cz > z0m) & (cz < z1m))
    nx = len(xs)
    idx = {}
    V, Fc = [], []

    def vid(i, j):
        k = (i, j)
        if k not in idx:
            idx[k] = len(V); V.append((xs[i], 0.0, zs[j]))
        return idx[k]
    for k in np.nonzero(keep)[0]:
        j, i = divmod(int(k), nx - 1)
        a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
        Fc += [(a, c, b), (a, d, c)]
    V = np.array(V, float)
    V[:, 1] = ground_y(V[:, 0], V[:, 2]) + 0.05
    grass, paved, paths, otrees = _osm_ground()
    vx, vz = V[:, 0], V[:, 2]
    west = vz > EXO[3] + 2.0                                     # the west courtyard, in front of the exonarthex
    col = np.where(west[:, None], np.array(PAVE), np.array(LAWN))
    for P in grass:
        col = np.where(_inside(vx, vz, P)[:, None], np.array(LAWN), col)
    for P in paved:
        col = np.where(_inside(vx, vz, P)[:, None], np.array(PAVE), col)
    col = np.where((_dist_lines(vx, vz, paths) < 1.3)[:, None], np.array(PAVE), col)
    k = fbm(V, 0.07, 5, 2)
    g = Mesh(V, Fc, LAWN)
    g.c = np.clip(col * (0.9 + 0.18 * k[:, None]), 0, 1)
    # make every face look up
    n = np.cross(V[g.f[:, 1]] - V[g.f[:, 0]], V[g.f[:, 2]] - V[g.f[:, 0]])
    flip = n[:, 1] < 0
    g.f[flip] = g.f[flip][:, ::-1]
    trees = []
    hand = [(48.0, 52.0), (52.0, 80.0), (66.0, 84.0), (82.0, 80.0), (30.0, 88.0), (10.0, 90.0), (-12.0, 90.0),
            (-35.0, 90.0), (-58.0, 92.0), (-76.0, 82.0), (-78.0, 60.0), (-70.0, 18.0), (-66.0, -10.0), (-64.0, -40.0),
            (92.0, 24.0), (95.0, -18.0), (96.0, 60.0), (68.0, 30.0), (96.0, -36.0), (66.0, 46.0), (-68.0, 38.0)]
    for i_, (x, z) in enumerate(otrees):
        if not _inside(np.array([x]), np.array([z]), F)[0] or min(math.hypot(x - a, z - b) for a, b in hand) < 4.0:
            continue
        trees += tree(x, z, float(ground_y(x, z)), i_ % 3)
    print(f"precinct ground: {len(Fc) // 2} cells of {cell:g} m, {len(trees) // 2 if trees else 0} OSM trees")
    return [g] + trees


# ── Assembly ─────────────────────────────────────────────────────────────────
def build():
    church = annexes() + main_body() + dome_base() + narthex() + ottoman_buttresses()
    domes = main_dome() + semi_domes()
    mins = []
    for k in MINARETS:
        mins += minaret(k)
    return [("church", merge(church)), ("domes", merge(domes)), ("minarets", merge(mins)),
            ("tombs", merge(tombs())), ("precinct", merge(precinct())), ("ground", merge(precinct_ground()))]


FOOTPRINT = [(-43.4, -57.1), (-63.6, -57.2), (-71.2, -52.0), (-74.4, -42.0), (-84.0, 26.3), (-85.5, 44.0),
             (-87.6, 68.0), (-90.7, 102.1), (-18.0, 100.8), (54.8, 97.0), (97.2, 86.0), (99.3, 50.0),
             (102.8, -43.8), (92.5, -44.3), (88.0, -45.5), (61.4, -46.0), (40.0, -47.1), (40.0, -50.8),
             (-14.2, -52.7), (-13.7, -62.8), (-36.1, -64.3), (-37.1, -57.0)]


def main():
    parts = build()
    out_dir = ROOT / "models" / "Q12506"
    out_dir.mkdir(parents=True, exist_ok=True)
    named, special = [], {"gilt": [], "water": []}
    for name, m in parts:
        for key, col in (("gilt", GOLD), ("water", WATER)):        # faces painted exactly that colour
            hit = np.all(np.abs(m.c[m.f] - np.array(col)).max(axis=2) < 1e-6, axis=1)
            if hit.any():
                special[key].append(Mesh(m.v, m.f[hit], m.c)); m = Mesh(m.v, m.f[~hit], m.c)
        k = 0.95 + 0.05 * fbm(m.v, 0.08, 7, 2)
        m.c = np.clip(m.c * k[:, None], 0, 1)
        used = np.unique(m.f)
        remap = np.full(len(m.v), -1); remap[used] = np.arange(len(used))
        named.append((name, Mesh(m.v[used], remap[m.f], m.c[used])))
    for key in ("gilt", "water"):
        w = merge(special[key])
        used = np.unique(w.f)
        remap = np.full(len(w.v), -1); remap[used] = np.arange(len(used))
        named.append((key, Mesh(w.v[used], remap[w.f], w.c[used]), {"material": key}))
    tris, size = write_glb(out_dir / "model.glb", named, roughness=0.85,
                           materials={"gilt": METAL, "water": {"metallic": 0.0, "roughness": 0.12}})
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        mj["footprint"] = [[round(x, 1), round(z, 1)] for x, z in FOOTPRINT]
        meta.write_bytes((json.dumps(mj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"{tris} triangles, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
