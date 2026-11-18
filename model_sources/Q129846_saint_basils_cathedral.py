"""
Saint Basil's Cathedral, Moscow (Q129846) — the Cathedral of the Intercession on the Moat, with the stretch of
the Kremlin wall across Red Square (Spasskaya, Tsarskaya and Nabatnaya towers), built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Barma and Postnik Yakovlev (1555–1561, for Ivan IV), the chapel of St Basil added 1588, the bell
tower 1680s, the covered stairs and porches of the 17th century. The Spasskaya Tower by Pietro Antonio Solari
(1491, died 1493), its tiered top with the tent by Bazhen Ogurtsov and Christopher Galloway (1624–25); the Nabatnaya
(1495) and Tsarskaya (1680) towers; the ruby star of 1937 is a plain geometric shape. Public domain.

What is modelled:
- The cathedral: nine churches on one base. The central church of the Intercession (octagon, three tiers of keel
  kokoshniks, octagonal drum, the tent (shatyor) with its white lozenges and ribs, the lantern, the small gilded
  onion and the cross); the four large octagonal axial churches (west: Entry into Jerusalem, the tallest; north,
  east, south) and the four small churches on the diagonals, each with its drum, cornices, kokoshniks and its
  own onion dome painted with its own pattern (spirals, chevrons, faceted «pineapple» knobs, scales, stripes);
  the chapel of St Basil (1588) with its green-and-gold dome; the bell tower (octagon, open belfry, green tiled
  tent with coloured ribs and dormers, onion, cross); the podklet and galleries in red brick with white stone
  bands and arched windows, the metal gallery roof, the apses on the east, the two covered stairs on the west
  façade with their tent-roofed porches.
- The Kremlin wall along Red Square from the Spasskaya to the Nabatnaya tower, 12 m high with swallowtail
  merlons; the Spasskaya tower (body, the forward gate «strelnitsa» with the gate arch, the white-stone tiers, the
  clock on four faces, the open octagon, the green tent and the ruby star, 71 m); the little Tsarskaya tower on
  the wall; the Nabatnaya tower.
- Not modelled: the Kremlin interior, Lenin's Mausoleum, GUM, the Minin and Pozharsky monument, Lobnoye Mesto,
  the square itself and the terrain.

Measurements (see meta.json for the sources):
- Plan and heights of the cathedral: OpenStreetMap relation 3030568 (wikidata Q129846) and its ~510 building:part
  pieces (a detailed 3D mapping of the cathedral: every drum, cornice, dome slice, cross and kokoshnik with
  min_height/height). The pieces were grouped by tower and turned into lathes and prisms: central church
  octagon 12.5–29.7 m, kokoshnik tiers 29.7–35.2, drum to 39.4, cornice 40.7, tent 40.7–52.6, lantern to 56.2,
  onion 56.5–59.5, cross to 66.7 m (OSM height 67; ru.wikipedia: 47.5 m to the top of the tent from the floor of
  the church, 65 m from the ground); west church onion 29–39.5 m (Ø 9.2 m), cross to 46 m; the other axial
  churches onion 27–36.5 (Ø 9.6 m), cross to 43 m; diagonal churches onion 21–26 (Ø 6.2 m), cross to 31.5 m;
  chapel of St Basil dome 14.8–19.3, cross 25.5 m; bell tower belfry 13.6–20.5, tent to 30.2, cross 34.9 m;
  galleries to 11.0, cornice 11.8, roof 12.3 m. Widths of drums and octagons from the same pieces (±0.3 m).
- The colours and patterns of the individual domes follow photographs (Wikimedia Commons) in a stylised way;
  which pattern belongs to which church is approximate (±).
- Kremlin: OpenStreetMap relations 1360656 (Spasskaya, height 71), 1360663 (Tsarskaya, 16), 1360661 (Nabatnaya,
  38) and the wall pieces 2470039 and 2470034 (height 12, 4 m thick). The tiers of the Spasskaya tower are
  estimated from photographs scaled to the 71 m total (±10 %): body 0–26 m, white-stone parapet to 27.5, clock
  tier 27–41.5 with the four dials (Ø 6.12 m, en.wikipedia «Kremlin Clock»), octagon 41.5–48.5, open octagon
  48.5–52.5, tent 52.5–66.8, star 67–71 (span 3.75 m). Tsarskaya and Nabatnaya tiers estimated the same way.
- Ground: Red Square is nearly level here; the terrain data (SRTM/ASTER, ±5 m in a city) cannot tell the small
  slope, so y = 0 is the square, the walls go down to y = −5 everywhere and the ground is kept as it is.

Axes (docs/MODEL_SPEC.md): +Y up, +Z points to 338° — up Red Square towards the Historical Museum, the side from
which the cathedral is usually seen; the Kremlin wall runs along Z, about 100 m away on +X (to 248°). Origin on
the ground at the centroid of the cathedral's outline (OSM relation 3030568).

    python model_sources/Q129846_saint_basils_cathedral.py   -> models/Q129846/model.glb (--footprint prints it)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, extrude, walls, cap, offset, ring_area, write_glb  # noqa: E402

# ── Colours ──────────────────────────────────────────────────────────────────
BRICK = (0.62, 0.22, 0.16)
BRICK_D = (0.50, 0.17, 0.13)
WHITE = (0.93, 0.91, 0.85)
STONE = (0.84, 0.81, 0.74)
ROOF = (0.30, 0.40, 0.35)          # painted metal of the gallery roofs
TILE = (0.20, 0.47, 0.30)          # green tiles and painted tents
DARK = (0.08, 0.07, 0.07)
GOLD = (0.90, 0.72, 0.30)
GREEN = (0.10, 0.50, 0.30)
GREEN_D = (0.07, 0.36, 0.22)
YELLOW = (0.95, 0.78, 0.18)
RED = (0.78, 0.13, 0.11)
BLUE = (0.16, 0.38, 0.72)
SKY = (0.45, 0.68, 0.88)
ORANGE = (0.90, 0.47, 0.13)
K_BRICK = (0.64, 0.25, 0.18)       # Kremlin brick
K_TENT = (0.17, 0.40, 0.29)
RUBY = (0.78, 0.04, 0.09)

# ── Plan: OpenStreetMap outlines in the model axes (heading 338°, origin at the centroid of relation 3030568),
#    rounded to 0.1 m (© OpenStreetMap contributors, ODbL) ─────────────────────────────────────────────
GALLERY = [(14.6, 3.4), (19.0, 3.1), (21.6, -0.2), (21.3, -4.3), (18.2, -7.0), (13.6, -6.6), (13.0, -12.8), (5.0, -12.0), (4.7, -16.4), (1.9, -18.9), (-1.9, -18.6), (-4.3, -15.7), (-3.9, -11.2), (-11.9, -10.5), (-11.2, -3.3), (-14.9, -3.0), (-17.2, -0.4), (-16.9, 3.1), (-14.1, 5.3), (-10.4, 4.9), (-9.7, 12.2), (-1.9, 11.5), (-1.5, 15.3), (1.1, 17.6), (4.6, 17.3), (6.9, 14.7), (6.6, 10.7), (15.2, 9.9)]   # way/227681487
GALLERY_ROOF = [(-4.6, -15.8), (-4.3, -11.5), (-12.3, -10.8), (-11.6, -3.7), (-15.1, -3.4), (-17.5, -0.6), (-17.2, 3.2), (-14.2, 5.7), (-10.7, 5.3), (-10.0, 12.6), (-2.2, 11.9), (-1.9, 15.5), (1.0, 18.0), (4.8, 17.7), (7.3, 14.8), (7.0, 11.0), (15.6, 10.2), (15.0, 3.8), (19.2, 3.4), (22.0, -0.1), (21.7, -4.5), (18.3, -7.3), (13.9, -7.0), (13.4, -13.2), (12.8, -13.3), (5.3, -12.4), (5.0, -16.6), (2.0, -19.2), (-2.0, -18.9)]   # relation/3062911
ROOF_NE = [(-1.5, 10.8), (-1.0, 15.9), (-14.2, 17.1), (-14.7, 12.0)]   # relation/3030586
BELL_BASE = [(-18.5, -8.9), (-20.2, -9.2), (-20.1, -9.4), (-25.4, -10.0), (-25.4, -9.8), (-26.9, -10.0), (-26.7, -11.6), (-26.4, -11.6), (-25.8, -16.9), (-26.0, -16.9), (-25.9, -18.0), (-17.5, -17.0)]   # relation/3030576
BELL_ROOF = [(-18.3, -8.7), (-27.1, -9.8), (-26.1, -18.3), (-17.2, -17.2)]   # relation/3062904
APSE_A = [(-17.3, 5.4), (-20.2, 5.8), (-20.9, 6.6), (-21.6, 8.3), (-21.6, 9.8), (-21.3, 11.4), (-20.4, 12.6), (-19.4, 13.2), (-19.7, 13.9), (-19.9, 14.6), (-20.0, 15.7), (-19.7, 16.8), (-19.1, 17.6), (-18.3, 18.2), (-14.0, 17.7), (-14.4, 12.9), (-16.5, 13.1)]   # relation/3030591
APSE_B = [(-12.3, -2.9), (-14.8, -2.6), (-15.4, -1.9), (-16.7, -1.9), (-18.3, -2.4), (-19.3, -3.2), (-19.8, -4.5), (-19.6, -5.8), (-18.7, -7.0), (-17.5, -7.7), (-15.9, -7.9), (-12.8, -8.1)]   # relation/3030583
BASIL_BODY = [(-16.5, 2.9), (-17.3, 5.4), (-16.5, 13.1), (-14.4, 12.9), (-14.5, 12.3), (-10.6, 11.9), (-11.1, 4.6), (-13.9, 4.9)]   # relation/3030595
APSE_C = [(-6.3, 16.2), (-5.8, 21.5), (-13.6, 22.2), (-14.1, 16.9)]   # relation/3030584
APSE_D = [(-17.5, -16.4), (-18.4, -9.1), (-15.8, -8.8), (-14.9, -16.1)]   # relation/3030574
APSE_E = [(-13.5, -16.2), (-13.2, -12.7), (-7.5, -13.3), (-7.8, -16.1)]   # relation/3030575
DIAG_SE = [(-10.8, -3.0), (-12.3, -2.9), (-13.2, -12.7), (-7.5, -13.3), (-7.8, -16.1), (-5.8, -16.4), (-3.9, -15.6), (-3.5, -10.9), (-11.5, -10.1)]   # relation/3030605
DIAG_NE = [(-0.7, 15.7), (-14.1, 16.9), (-14.5, 12.3), (-10.6, 11.9), (-11.1, 4.6), (-10.1, 4.5), (-9.4, 11.8), (-1.5, 11.1), (-1.1, 15.2)]   # relation/3030571
DIAG_SW = [(5.0, -16.6), (7.0, -17.1), (11.8, -17.7), (12.7, -12.4), (4.7, -11.6), (4.3, -16.3)]   # relation/3030592
DIAG_NW = [(9.6, 15.3), (15.1, 14.8), (14.8, 9.5), (6.2, 10.3), (6.5, 14.6)]   # relation/3030593
AX_E = [(-10.8, -3.0), (-14.8, -2.6), (-16.8, -0.3), (-16.5, 2.9), (-13.9, 4.9), (-10.1, 4.5)]   # relation/3030599
AX_S = [(-3.5, -10.9), (4.7, -11.6), (4.3, -16.3), (1.8, -18.5), (-1.7, -18.2), (-3.9, -15.6)]   # relation/3030589
AX_N = [(-1.5, 11.1), (-1.1, 15.2), (-0.7, 15.7), (1.3, 17.2), (4.5, 17.0), (6.5, 14.6), (6.2, 10.3)]   # relation/3030596
AX_W = [(14.2, 3.1), (18.8, 2.8), (21.3, -0.3), (21.0, -4.1), (18.1, -6.6), (13.3, -6.2)]   # relation/3030590
STAIR_S = [(17.4, -12.8), (12.7, -12.4), (13.3, -6.2), (18.1, -6.6)]   # relation/3030594
STAIR_N = [(14.2, 3.1), (18.8, 2.8), (19.6, 4.2), (20.1, 9.1), (14.8, 9.5)]   # relation/3030597
STAIRTOP_S = [(11.8, -17.7), (16.8, -18.2), (17.4, -12.8), (12.7, -12.4)]   # relation/3030585
STAIRTOP_N = [(20.1, 9.1), (20.6, 14.3), (15.1, 14.8), (14.8, 9.5)]   # relation/3030587
STAIRMID_N = [(21.1, 18.2), (15.4, 18.8), (15.1, 14.8), (20.6, 14.3)]   # relation/3030598
STAIRFOOT_S = [(16.4, -21.2), (16.8, -18.2), (11.8, -17.7), (11.5, -20.6)]   # relation/3030602
PORCH_S = [(15.9, -26.2), (11.0, -25.6), (11.5, -20.6), (16.4, -21.2)]   # relation/3030569
PORCH_N = [(21.1, 18.2), (15.4, 18.8), (16.0, 24.6), (21.7, 24.0)]   # relation/3030570
OUTLINE = [(11.6, -20.1), (11.8, -17.7), (7.0, -17.1), (5.0, -16.6), (2.0, -19.2), (-2.0, -18.9), (-4.6, -15.8), (-5.8, -16.4), (-7.8, -16.1), (-13.5, -16.2), (-14.9, -16.1), (-17.3, -16.4), (-17.2, -17.2), (-26.1, -18.3), (-27.1, -9.8), (-19.2, -8.8), (-18.7, -7.0), (-19.6, -5.8), (-19.8, -4.5), (-19.3, -3.2), (-18.3, -2.4), (-16.7, -1.9), (-16.4, -1.9), (-17.5, -0.6), (-17.2, 3.2), (-16.8, 3.6), (-17.3, 5.4), (-20.2, 5.8), (-20.9, 6.6), (-21.6, 8.3), (-21.6, 9.8), (-21.3, 11.4), (-20.4, 12.6), (-19.4, 13.2), (-19.7, 13.9), (-19.9, 14.6), (-20.0, 15.7), (-19.7, 16.8), (-19.1, 17.6), (-18.3, 18.2), (-14.0, 17.7), (-13.6, 22.2), (-5.8, 21.5), (-6.3, 16.4), (-1.4, 15.9), (1.0, 18.0), (4.8, 17.7), (7.3, 14.8), (9.6, 15.3), (15.1, 14.8), (15.4, 18.3), (14.7, 18.3), (14.9, 20.2), (15.3, 20.2), (15.6, 23.2), (15.2, 23.3), (15.4, 25.2), (17.4, 25.0), (17.4, 24.8), (20.3, 24.5), (20.3, 24.8), (22.4, 24.6), (22.2, 22.6), (21.9, 22.7), (21.6, 19.6), (21.9, 19.6), (21.7, 17.7), (21.0, 17.7), (19.6, 4.2), (19.2, 3.4), (22.0, -0.1), (21.7, -4.5), (18.3, -7.3), (18.0, -7.3), (16.5, -20.6), (17.1, -20.7), (16.9, -22.5), (16.6, -22.5), (16.3, -24.8), (16.7, -24.8), (16.5, -26.8), (14.4, -26.6), (14.4, -26.3), (12.4, -26.1), (12.4, -26.2), (10.3, -26.0), (10.5, -24.1), (10.8, -24.1), (11.1, -22.0), (10.8, -21.9), (11.0, -20.1)]   # relation/3030568
R1360656 = [(102.3, 31.8), (102.2, 31.5), (103.8, 30.7), (104.7, 32.4), (104.3, 32.6), (105.3, 34.6), (104.9, 34.8), (109.2, 43.7), (109.6, 43.5), (110.5, 45.3), (110.8, 45.1), (111.7, 46.9), (110.0, 47.7), (109.8, 47.3), (98.1, 52.9), (98.3, 53.3), (96.6, 54.1), (95.8, 52.3), (96.2, 52.2), (95.7, 51.0), (94.5, 51.6), (94.3, 51.2), (88.7, 53.8), (88.9, 54.2), (87.2, 55.0), (86.4, 53.3), (86.7, 53.1), (83.6, 46.6), (83.2, 46.7), (82.4, 45.0), (84.1, 44.2), (84.3, 44.6), (89.7, 42.0), (89.5, 41.6), (90.9, 41.0), (90.1, 39.3), (89.7, 39.4), (88.9, 37.7), (90.6, 36.9), (90.8, 37.2)]
R1360663 = [(98.2, 7.6), (102.2, 7.7), (102.1, 12.5), (98.1, 12.4)]
R1360661 = [(105.0, -35.7), (105.7, -45.5), (95.9, -46.3), (95.2, -36.5)]
R2470039 = [(98.1, 12.4), (98.3, 33.7), (102.0, 31.9), (101.9, 25.4), (102.1, 12.5)]
R2470034 = [(98.2, 7.6), (102.2, 7.7), (102.2, -36.4), (98.2, -36.7)]

C0 = (1.8, -0.2)                   # central church
AX = math.atan2(-1.6, 14.7)        # cathedral axis (centre -> west church), in the (x, z) plane
WEST, EAST, NORTH, SOUTH = (16.5, -1.8), (-12.6, 1.0), (2.6, 13.2), (0.4, -14.2)
DIAG = {"NE": (-6.1, 7.9), "SE": (-7.6, -6.9), "NW": (10.9, 6.4), "SW": (9.5, -8.4)}
BASIL = (-13.9, 8.7)
BELL = (-22.1, -13.5)


# ── Helpers ──────────────────────────────────────────────────────────────────
def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def ccw(P):
    P = [tuple(p) for p in P]
    return P if ring_area(P) > 0 else P[::-1]


def ngon(n, r, rot=0.0, c=(0.0, 0.0)):
    return [(c[0] + r * math.cos(rot + 2 * math.pi * k / n), c[1] + r * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def loft(r0, y0, r1, y1, c):
    """Band between two rings with the same number of points (each side its own quad)."""
    v, f = [], []
    n = len(r0)
    for i in range(n):
        j = (i + 1) % n
        o = len(v)
        v += [(r0[i][0], y0, r0[i][1]), (r0[j][0], y0, r0[j][1]), (r1[j][0], y1, r1[j][1]), (r1[i][0], y1, r1[i][1])]
        f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    return Mesh(v, f, c)


def shrink(P, k):
    cx = sum(p[0] for p in P) / len(P); cz = sum(p[1] for p in P) / len(P)
    return [(cx + (x - cx) * k, cz + (z - cz) * k) for x, z in P]


def roof(P, y0, h, c, k=0.0):
    """Hipped (k > 0: the top is the outline shrunk by k) or pyramidal (k = 0) roof over an outline."""
    P = ccw(P)
    top = shrink(P, k)
    out = [loft(P, y0, top, y0 + h, c)]
    if k > 0:
        out.append(cap(top, y0 + h, c))
    return merge(out)


def oct_body(c, r, y0, y1, col, rot=None, top=True):
    return extrude(ngon(8, r, AX + math.pi / 8 if rot is None else rot, c), y0, y1, col, top=top)


def on_face(m, c, phi, d):
    """Mesh built facing +z (u = x, along the face) -> on the face whose outward normal is at angle phi."""
    return m.R(0, math.pi / 2 - phi, 0).T(c[0] + d * math.cos(phi), 0, c[1] + d * math.sin(phi))


def on_edge(m, p0, p1):
    """Mesh built in (u along the edge, y, w outwards), u from p1 towards p0, for an edge of a ccw ring."""
    ex, ez = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(ex, ez)
    nx, nz = ez / L, -ex / L                          # outward normal of a ccw ring
    phi = math.atan2(nz, nx)
    m.R(0, math.pi / 2 - phi, 0)                      # +z -> outward, +x -> along -edge
    return m.T(p1[0], 0, p1[1])


def prism(pts, w0, w1, c):
    """A convex profile pts [(u, y)] extruded from w0 to w1 (outwards)."""
    n = len(pts)
    if sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n)) < 0:
        pts = pts[::-1]
    V, F = [], []
    for w in (w0, w1):
        V += [(u, y, w) for u, y in pts]
    F += [(n, n + i, n + i + 1) for i in range(1, n - 1)]
    F += [(0, i + 1, i) for i in range(1, n - 1)]
    ms = [Mesh(V, F, c)]
    for i in range(n):
        (u0, y0), (u1, y1) = pts[i], pts[(i + 1) % n]
        ms.append(Mesh([(u0, y0, w0), (u1, y1, w0), (u1, y1, w1), (u0, y0, w1)], [(0, 1, 2), (0, 2, 3)], c))
    return merge(ms)


def arch_pts(u, y0, ys, half, seg=6):
    return [(u - half, y0), (u + half, y0)] + [(u + half * math.cos(a), ys + half * math.sin(a))
                                               for a in np.linspace(0, math.pi, seg + 1)]


def keel_pts(w, h, seg=5):
    """Keel-arch (kokoshnik) outline, half-width w, height h, standing on y = 0 (convex)."""
    ys = h - 1.45 * w
    pts = [(-w, 0.0), (w, 0.0)]
    pts += [(w * math.cos(a), ys + w * math.sin(a)) for a in np.linspace(0, 1.15, seg)]
    pts += [(0.0, h)]
    pts += [(-w * math.cos(a), ys + w * math.sin(a)) for a in np.linspace(1.15, 0, seg)]
    return pts


def kokoshnik(w, h, inner=BRICK, frame=WHITE, t=0.35):
    """A kokoshnik: white keel-arch frame and a red (or dark) field set in it, standing on y = 0, facing +z."""
    out = [prism(keel_pts(w, h), -t, 0.0, frame)]
    pi = [(u * 0.74, y * 0.80 + 0.1) for u, y in keel_pts(w, h)]
    out.append(prism(pi, -t, 0.06, inner))
    return merge(out)


def kok_ring(c, r, n, y0, w, h, rot, inner=BRICK):
    out = []
    for k in range(n):
        phi = rot + 2 * math.pi * k / n
        out.append(on_face(kokoshnik(w, h, inner).T(0, y0, 0), c, phi, r))
    return merge(out)


def cross(x, y, z, h, rot=0.0):
    """Orthodox cross with a ball, gilded: post, top bar, main bar, slanted foot bar."""
    t = max(0.05, h * 0.035)
    out = [B(-t, t, 0, h, -t, t, GOLD),
           B(-0.27 * h, 0.27 * h, 0.66 * h, 0.66 * h + 2 * t, -t, t, GOLD),
           B(-0.14 * h, 0.14 * h, 0.84 * h, 0.84 * h + 2 * t, -t, t, GOLD),
           B(-0.17 * h, 0.17 * h, 0.30 * h, 0.30 * h + 2 * t, -t, t, GOLD).R(0, 0, -0.35).T(0, 0.0, 0),
           lathe([(0.0, -0.11 * h), (0.09 * h, -0.07 * h), (0.11 * h, 0.0), (0.09 * h, 0.07 * h), (0.0, 0.11 * h)], GOLD, 10)]
    return merge(out).R(0, rot, 0).T(x, y, z)


# ── Onion domes with painted patterns ────────────────────────────────────────
def pattern(kind, a, s, n, k, cols):
    """Colour index per vertex: a in [0, 1) around the dome, s in [0, 1] up the dome."""
    if kind == "spiral":
        f = np.mod(a * n + s * k, 1.0)
        idx = (f >= 0.5).astype(int)
        if len(cols) > 2:
            idx = np.where((np.abs(f - 0.5) < 0.07) | (f > 0.93) | (f < 0.0), 2, idx)
    elif kind == "stripes":
        f = np.mod(a * n, 1.0)
        idx = (f >= 0.5).astype(int)
        if len(cols) > 2:
            idx = np.where(np.abs(f - 0.5) < 0.08, 2, idx)
    elif kind == "chevron":
        zz = np.abs(np.mod(a * n, 1.0) - 0.5) * 2
        f = np.mod(s * k + zz * 0.9, 1.0)
        idx = np.floor(f * len(cols)).astype(int) % len(cols)
    elif kind == "facet":
        p, q = a * n, s * k
        idx = ((np.floor(p + q) + np.floor(p - q)) % 2).astype(int)
    elif kind == "scale":
        row = np.floor(s * k)
        p = np.mod(a * n + 0.5 * row, 1.0)
        fs = np.mod(s * k, 1.0)
        idx = (fs < 0.35 + 0.65 * (np.abs(p - 0.5) * 2) ** 1.5).astype(int)
    elif kind == "studs":
        p, q = a * n, s * k
        cen = (1 - np.abs(np.mod(p + q, 1) - 0.5) * 2) * (1 - np.abs(np.mod(p - q, 1) - 0.5) * 2)
        idx = (cen > 0.55).astype(int)
    else:
        idx = np.zeros(len(a), int)
    return np.asarray(cols, float)[np.clip(idx, 0, len(cols) - 1)]


def onion_rs(s, R, h, neck, bulge):
    """Radius and height of the onion at s in [0, 1] (0 = neck, 1 = tip)."""
    s = np.asarray(s, float)
    lo = neck + (R - neck) * np.sin(np.clip(s / bulge, 0, 1) * np.pi / 2)
    u = np.clip((s - bulge) / (1 - bulge), 0, 1)
    hi = R * np.clip(1 - (3 * u * u - 2 * u ** 3), 0, 1) ** 0.9
    return np.where(s <= bulge, lo, hi), s * h


def grid_mesh(A, S, R, h, neck, bulge, c):
    """A surface patch of the onion from a grid of turns A[i, j] (fraction of a turn) and heights S[i, j]."""
    r, y = onion_rs(S, R, h, neck, bulge)
    t = 2 * np.pi * (A + AX / (2 * np.pi))
    V = np.stack([r * np.cos(t), y, r * np.sin(t)], -1).reshape(-1, 3)
    ni, nj = A.shape
    F = []
    for i in range(ni - 1):
        for j in range(nj - 1):
            p, q = i * nj + j, i * nj + j + 1
            F += [(p, p + nj, q + nj), (p, q + nj, q)]
    return Mesh(V, F, c)


def dome(c, y0, R, h, neck, bulge, kind, cols, n=8, k=1.0, bump=0.0, seg=56, rows=28):
    """Onion dome. Spirals, stripes and chevrons are built band by band (each band its own strip of mesh, so the
    edges of the colours are sharp); facets, scales and studs are painted on the vertices (with knobs: bump)."""
    out = []
    if kind in ("spiral", "stripes"):
        # one period = colour A, (edge), colour B, (edge); the bands lean by k/n turns from the neck to the tip
        per = [(0.5, cols[0]), (0.5, cols[1])] if len(cols) < 3 else \
              [(0.42, cols[0]), (0.08, cols[2]), (0.42, cols[1]), (0.08, cols[2])]
        cpp = max(6, round(seg / n))
        Sg = np.linspace(0, 1, rows + 1)
        for b in range(n):
            a0 = 0.0
            for w, col in per:
                m = max(1, round(w * cpp))
                fa = (b + a0 + np.linspace(0, w, m + 1)) / n
                A = fa[None, :] - (k / n) * Sg[:, None]
                S = np.repeat(Sg[:, None], m + 1, 1)
                out.append(grid_mesh(A, S, R, h, neck, bulge, col))
                a0 += w
    elif kind == "chevron":
        amp, q = 0.9, 2
        ncol = 2 * n * q
        Aj = np.linspace(0, 1, ncol + 1)
        zz = np.abs(np.mod(Aj * n, 1.0) - 0.5) * 2
        bw = 1.0 / len(cols)
        nb = int(math.ceil((k + amp) / bw))
        for b in range(nb):
            rr = np.linspace(b * bw, (b + 1) * bw, 4)
            S = np.clip((rr[:, None] - zz[None, :] * amp) / k, 0, 1)
            if S.max() <= 0 or S.min() >= 1:
                continue
            A = np.repeat(Aj[None, :], len(rr), 0)
            out.append(grid_mesh(A, S, R, h, neck, bulge, cols[b % len(cols)]))
    else:
        Aj = np.linspace(0, 1, seg + 1)
        Sg = np.linspace(0, 1, rows + 1)
        A, S = np.meshgrid(Aj, Sg)
        m = grid_mesh(A, S, R, h, neck, bulge, cols[0])
        a, s = A.ravel(), S.ravel()
        m.c = pattern(kind, a, s, n, k, cols)
        if bump:
            p, qq = a * n, s * k
            cen = (1 - np.abs(np.mod(p + qq, 1) - 0.5) * 2) * (1 - np.abs(np.mod(p - qq, 1) - 0.5) * 2)
            f = 1 + bump * (cen - 0.4) * np.sin(np.pi * s) ** 0.5
            m.v[:, 0] *= f; m.v[:, 2] *= f
        out.append(m)
    return merge(out).T(c[0], y0, c[1])


# ── Tents ────────────────────────────────────────────────────────────────────
def face_quad(P, out_dir, c, eps=0.05):
    """A flat quad P (4 points, any winding) pushed eps along its normal and facing out_dir."""
    P = np.asarray(P, float)
    nrm = np.cross(P[1] - P[0], P[2] - P[0])
    nrm /= (np.linalg.norm(nrm) or 1)
    if np.dot(nrm, out_dir) < 0:
        P = P[::-1]; nrm = -nrm
    return Mesh(P + nrm * eps, [(0, 1, 2), (0, 2, 3)], c)


def tent(c, y0, y1, r0, r1, n, rot, col, rib=WHITE, lozenges=None, dormers=None, rib_r=0.09):
    """Tent (shatyor): n-sided pyramid frustum; white ribs; optional lozenge lattice (rows, per face) and dormers."""
    R0, R1 = ngon(n, r0, rot, c), ngon(n, r1, rot, c)
    out = [loft(R0, y0, R1, y1, col)]
    for i in range(n):
        out.append(tube([(R0[i][0], y0, R0[i][1]), (R1[i][0], y1, R1[i][1])], rib_r, rib, 5))
    if lozenges or dormers:
        for i in range(n):
            j = (i + 1) % n
            A0, B0 = np.array([R0[i][0], y0, R0[i][1]]), np.array([R0[j][0], y0, R0[j][1]])
            A1, B1 = np.array([R1[i][0], y1, R1[i][1]]), np.array([R1[j][0], y1, R1[j][1]])
            mid = (A0 + B0 + A1 + B1) / 4
            od = np.array([mid[0] - c[0], 0.0, mid[2] - c[1]])

            def P(u, t):
                return (A0 + (B0 - A0) * u) * (1 - t) + (A1 + (B1 - A1) * u) * t
            if lozenges:
                rows, per, lc = lozenges
                for rr in range(rows):
                    t = (rr + 0.5) / rows * 0.94
                    m_ = per if rr % 2 == 0 else per - 1
                    for q in range(m_):
                        u = (q + 0.5 + (0 if rr % 2 == 0 else 0.5)) / per
                        du, dt = 0.32 / per, 0.42 / rows
                        out.append(face_quad([P(u - du, t), P(u, t - dt), P(u + du, t), P(u, t + dt)], od, lc, 0.04))
            if dormers:
                for t, w, h, dc in dormers:
                    p = P(0.5, t)
                    phi = math.atan2(od[2], od[0])
                    d = math.hypot(p[0] - c[0], p[2] - c[1])
                    out.append(on_face(kokoshnik(w, h, DARK, dc, 0.5).T(0, p[1], 0.3), c, phi, d))
    return merge(out)


# ── Edge decoration of the lower blocks ──────────────────────────────────────
def arcade_edges(P, y0, ys, half, step, frame=WHITE, glass=DARK, min_len=2.2, small=None):
    """Arched windows along every edge of an outline (ccw), with white frames; small = (y0, y1, w) podklet windows."""
    P = ccw(P)
    out = []
    for i in range(len(P)):
        p0, p1 = P[i], P[(i + 1) % len(P)]
        L = math.dist(p0, p1)
        if L < min_len:
            continue
        n = max(1, int(L / step))
        for k in range(n):
            u = (k + 0.5) * L / n
            win = [prism(arch_pts(u, y0 - 0.2, ys, half + 0.22), 0.0, 0.10, frame),
                   prism(arch_pts(u, y0, ys, half), 0.10, 0.14, glass)]
            if small:
                a, b, w = small
                win += [B(u - w / 2 - 0.15, u + w / 2 + 0.15, a - 0.15, b + 0.15, 0.0, 0.08, frame),
                        B(u - w / 2, u + w / 2, a, b, 0.08, 0.12, glass)]
            out.append(on_edge(merge(win), p0, p1))
    return merge(out)


def band(P, y0, y1, d, c):
    return walls(offset(ccw(P), d), y0, y1, c)


def block(P, y0, y1, col=BRICK, top=None, inset=0.05, bands=((5.8, 6.2),), plinth=True):
    """A brick block of the lower building from an OSM outline, white bands, white plinth."""
    Q = offset(ccw(P), -inset)
    out = [extrude(Q, y0, y1, col, top_c=top or ROOF)]
    for a, b in bands:
        if b <= y1:
            out.append(band(Q, a, b, 0.12, WHITE))
    if plinth:
        out.append(band(Q, -0.3, 0.6, 0.12, STONE))
    out.append(band(Q, y1 - 0.45, y1, 0.18, WHITE))
    return merge(out)


# ── The cathedral ────────────────────────────────────────────────────────────
FROT = AX + math.pi / 8             # octagons with a face towards each axis
FACES = [AX + k * math.pi / 4 for k in range(8)]


def lower_building():
    """Podklet and galleries (one brick mass to 11 m), the roofs, apses, chapel of St Basil, bell-tower base,
    covered stairs and porches."""
    gold, out = [], []
    G = ccw(GALLERY)
    out.append(extrude(G, -5.0, 11.0, BRICK, top=False))
    out.append(band(G, -5.0, 0.7, 0.14, STONE))                                   # white-stone plinth
    out.append(band(G, 5.6, 6.1, 0.16, WHITE))                                    # podklet / gallery floor
    out.append(band(G, 10.3, 11.0, 0.22, WHITE))
    out.append(band(G, 11.0, 11.8, 0.32, WHITE))                                  # cornice
    out.append(arcade_edges(G, 7.0, 9.0, 0.62, 2.9, small=(2.2, 3.6, 0.7)))
    GR = ccw(GALLERY_ROOF)
    out.append(extrude(GR, 11.8, 12.3, ROOF, top_c=ROOF))
    # roof between the north-east church and the apse; apses and chapel bodies on the east
    for P in (DIAG_NE, DIAG_SE, DIAG_SW, DIAG_NW):                               # lower bodies of the diagonal churches
        out.append(block(P, -5.0, 10.0, bands=((5.6, 6.1),), inset=0.08))
        out.append(arcade_edges(P, 7.0, 8.8, 0.55, 2.9, min_len=2.6, small=(2.2, 3.6, 0.7)))
        out.append(roof(offset(ccw(P), 0.2), 10.0, 1.6, ROOF, 0.6))
    out.append(block(ROOF_NE, -5.0, 10.0, bands=((5.6, 6.1),)))
    out.append(roof(offset(ccw(ROOF_NE), 0.3), 10.0, 1.8, ROOF, 0.55))
    for P, h, rh in ((APSE_A, 5.5, 1.5), (APSE_B, 5.5, 1.5), (APSE_C, 7.5, 1.5), (APSE_D, 6.5, 1.2), (APSE_E, 3.5, 0.8)):
        out.append(block(P, -5.0, h, bands=()))
        out.append(roof(offset(ccw(P), 0.25), h, rh, ROOF, 0.45))
        out.append(arcade_edges(P, 1.6, 2.8, 0.38, 2.4, min_len=1.6))
    out.append(block(BASIL_BODY, -5.0, 8.5, bands=((4.2, 4.6),)))
    out.append(roof(offset(ccw(BASIL_BODY), 0.25), 8.5, 1.5, ROOF, 0.5))
    out.append(arcade_edges(BASIL_BODY, 5.0, 6.8, 0.5, 2.6))
    # bell tower base (square block, low hipped roof)
    out.append(block(BELL_BASE, -5.0, 9.0, bands=((5.6, 6.0),)))
    out.append(arcade_edges(BELL_BASE, 6.4, 7.8, 0.45, 2.6, small=(2.4, 3.4, 0.6)))
    out.append(roof(BELL_ROOF, 9.0, 2.3, ROOF, 0.6))
    # the covered stairs of the west front and their porches
    for P, h in ((STAIR_S, 10.0), (STAIR_N, 10.0)):
        out.append(block(P, -5.0, h, bands=((5.6, 6.0),)))
        out.append(roof(offset(ccw(P), 0.3), h, 1.0, TILE, 0.6))
        out.append(arcade_edges(P, 1.4, 3.6, 0.7, 2.3, min_len=3.0))
        out.append(arcade_edges(P, 6.6, 8.6, 0.55, 2.3, min_len=3.0))
    for P, h, th in ((STAIRTOP_S, 8.0, 4.2), (STAIRTOP_N, 8.0, 4.2), (STAIRMID_N, 6.0, 3.2), (STAIRFOOT_S, 6.0, 0.0)):
        out.append(block(P, -5.0, h, bands=((4.0, 4.4),)))
        out.append(arcade_edges(P, 1.4, 3.4, 0.7, 2.6, min_len=3.0))
        if th:
            out.append(roof(offset(ccw(P), 0.35), h, th, TILE))
            cx = sum(p[0] for p in P) / len(P); cz = sum(p[1] for p in P) / len(P)
            gold.append(cross(cx, h + th, cz, 1.5, -AX))
        else:
            out.append(roof(offset(ccw(P), 0.3), h, 0.8, TILE, 0.5))
    for P in (PORCH_S, PORCH_N):                                                   # porch on four jug pillars
        Q = ccw(P)
        cx = sum(p[0] for p in Q) / 4; cz = sum(p[1] for p in Q) / 4
        for x, z in shrink(Q, 0.72):
            out.append(lathe([(0.45, -0.5), (0.45, 0.6), (0.62, 1.4), (0.66, 2.0), (0.45, 2.9), (0.36, 3.6), (0.5, 4.0), (0.5, 4.5)],
                             WHITE, 10).T(x, 0, z))
        out.append(extrude(offset(Q, 0.1), 4.5, 6.5, BRICK, top_c=ROOF))
        out.append(band(Q, 6.1, 6.5, 0.3, WHITE))
        out.append(arcade_edges(Q, 4.6, 5.6, 0.9, 3.0, glass=BRICK_D))
        out.append(roof(offset(Q, 0.35), 6.5, 4.0, TILE))
        out.append(B(-0.04, 0.04, 6.5, 6.6, -0.04, 0.04, TILE))
        gold.append(cross(cx, 10.5, cz, 1.4, -AX))
    return merge(out), merge(gold)


def central_church():
    c, out, gold = C0, [], []
    out.append(oct_body(c, 7.7, 11.9, 16.5, BRICK))
    out.append(oct_body(c, 7.85, 16.3, 16.7, WHITE, top=False))
    out.append(oct_body(c, 7.5, 16.5, 22.0, BRICK))
    out.append(oct_body(c, 7.6, 17.6, 18.0, WHITE, top=False))
    out.append(oct_body(c, 7.6, 21.7, 22.05, WHITE))
    out.append(oct_body(c, 6.9, 22.0, 28.5, BRICK))
    out.append(oct_body(c, 7.1, 28.5, 29.2, WHITE))
    out.append(oct_body(c, 7.3, 29.2, 29.7, WHITE))
    ap_lo, ap_hi = 7.5 * math.cos(math.pi / 8), 6.9 * math.cos(math.pi / 8)
    side_lo = 2 * 7.5 * math.sin(math.pi / 8)
    for k, phi in enumerate(FACES):
        # white pilasters at the corners, pointed gables of the middle tier, windows
        out.append(on_face(B(-0.3, 0.3, 11.9, 22.0, -0.2, 0.18, WHITE), c, phi + math.pi / 8, 7.5 * 0.98))
        out.append(on_face(B(-0.28, 0.28, 22.0, 28.5, -0.2, 0.16, WHITE), c, phi + math.pi / 8, 6.9 * 0.98))
        for q in (-1, 0, 1):
            g = kokoshnik(side_lo / 6.6, 4.0).T(q * side_lo / 3.2, 18.0, 0.3)
            out.append(on_face(g, c, phi, ap_lo))
        w = [prism(arch_pts(0, 23.4, 25.6, 0.55), 0.0, 0.12, WHITE), prism(arch_pts(0, 23.6, 25.6, 0.38), 0.12, 0.16, DARK)]
        out.append(on_face(merge(w), c, phi, ap_hi))
    # three-and-a-half tiers of keel kokoshniks at the foot of the drum
    out.append(kok_ring(c, 6.45, 8, 29.7, 1.25, 2.6, AX))
    out.append(kok_ring(c, 5.9, 8, 29.7, 1.15, 3.7, AX + math.pi / 8))
    out.append(kok_ring(c, 5.35, 8, 29.7, 1.05, 4.7, AX))
    out.append(kok_ring(c, 4.95, 8, 29.7, 0.95, 5.5, AX + math.pi / 8))
    # octagonal drum with windows and pilasters
    out.append(oct_body(c, 4.9, 29.7, 39.4, BRICK))
    ap = 4.9 * math.cos(math.pi / 8)
    for phi in FACES:
        out.append(on_face(B(-0.22, 0.22, 33.0, 39.4, -0.2, 0.14, WHITE), c, phi + math.pi / 8, 4.9 * 0.97))
        w = [prism(arch_pts(0, 35.4, 37.6, 0.5), 0.0, 0.1, WHITE), prism(arch_pts(0, 35.6, 37.6, 0.34), 0.1, 0.14, DARK)]
        out.append(on_face(merge(w), c, phi, ap))
    out.append(oct_body(c, 5.4, 39.4, 40.0, WHITE))
    out.append(oct_body(c, 5.6, 39.6, 40.2, WHITE))
    out.append(oct_body(c, 4.75, 40.2, 40.7, WHITE))
    out.append(kok_ring(c, 4.45, 16, 40.6, 0.75, 1.7, AX))
    # the tent: red brick, white ribs, white lozenges
    out.append(tent(c, 40.7, 52.6, 4.6, 1.1, 8, FROT, BRICK, WHITE, lozenges=(11, 3, WHITE), rib_r=0.13))
    out.append(oct_body(c, 1.25, 52.5, 52.9, WHITE))
    out.append(oct_body(c, 1.05, 52.9, 56.0, BRICK))
    for phi in FACES[::2]:
        out.append(on_face(B(-0.18, 0.18, 53.6, 55.3, -0.05, 0.05, DARK), c, phi, 1.05 * math.cos(math.pi / 8)))
    out.append(oct_body(c, 1.2, 55.9, 56.3, WHITE))
    gold.append(dome(c, 56.3, 1.9, 3.4, 1.0, 0.42, "plain", [GOLD], seg=28, rows=14))
    gold.append(cyl(0.2, 0.06, 1.0, GOLD, 8).T(c[0], 59.6, c[1]))
    gold.append(cross(c[0], 60.5, c[1], 6.2, -AX))
    return merge(out), merge(gold)


AXIAL_DOMES = {
    # name: (centre, dome kind, colours, n, k, bump)
    "W": (WEST, "spiral", [GREEN, YELLOW, RED], 10, 3.2, 0.0),          # Entry into Jerusalem
    "N": (NORTH, "spiral", [RED, WHITE, GREEN_D], 12, -3.8, 0.0),     # Ss Cyprian and Justina
    "E": (EAST, "facet", [GREEN, GOLD], 12, 8.0, 0.07),              # Holy Trinity
    "S": (SOUTH, "chevron", [GREEN, WHITE, RED, WHITE], 10, 3.5, 0.0),  # St Nicholas Velikoretsky
}
DIAG_DOMES = {
    "NW": ("scale", [GREEN, YELLOW], 12, 6.0, 0.0),
    "SW": ("spiral", [BLUE, WHITE, YELLOW], 10, 3.0, 0.0),
    "NE": ("chevron", [RED, YELLOW, GREEN], 10, 3.0, 0.0),
    "SE": ("stripes", [SKY, WHITE, GOLD], 12, 0.0, 0.0),
}


def axial_church(name):
    c, kind, cols, n, k, bump = AXIAL_DOMES[name]
    out, gold = [], []
    big = name == "W"
    r = 4.0 if big else 4.2
    top = 19.0 if big else 18.0
    out.append(oct_body(c, r + 0.05, 11.9, 12.6, WHITE))
    out.append(oct_body(c, r, 12.6, top, BRICK))
    ap = r * math.cos(math.pi / 8)
    for phi in FACES:
        out.append(on_face(B(-0.22, 0.22, 12.6, top, -0.2, 0.14, WHITE), c, phi + math.pi / 8, r * 0.97))
        w = [prism(arch_pts(0, 13.6, 16.2, 0.62), 0.0, 0.1, WHITE), prism(arch_pts(0, 13.8, 16.2, 0.44), 0.1, 0.14, DARK)]
        out.append(on_face(merge(w), c, phi, ap))
    out.append(oct_body(c, r + 0.25, top - 0.4, top + 0.3, WHITE))
    out.append(oct_body(c, r + 0.45, top + 0.3, top + 1.0, WHITE))
    if big:
        out.append(kok_ring(c, 3.75, 8, 20.0, 1.2, 2.0, AX))
        out.append(kok_ring(c, 3.35, 8, 20.6, 1.05, 2.9, AX + math.pi / 8))
        d0, d1, dr = 20.0, 28.3, 3.1
    else:
        out.append(kok_ring(c, 3.95, 8, top + 1.0, 1.25, 1.8, AX))
        d0, d1, dr = top + 1.0, 25.6, 3.3
    out.append(cyl(dr, dr, d1 - d0, BRICK, 24).T(c[0], d0, c[1]))
    for kk in range(8):
        phi = AX + kk * math.pi / 4
        out.append(on_face(B(-0.18, 0.18, d0 + 1.0, d1, -0.15, 0.12, WHITE), c, phi + math.pi / 8, dr - 0.02))
        out.append(on_face(merge([prism(arch_pts(0, d1 - 4.2, d1 - 2.0, 0.42), 0.0, 0.08, WHITE),
                                  prism(arch_pts(0, d1 - 4.0, d1 - 2.0, 0.28), 0.08, 0.12, DARK)]), c, phi, dr - 0.03))
    out.append(cyl(dr + 0.15, dr + 0.15, 0.5, WHITE, 24).T(c[0], d1 - 5.4, c[1]))
    if big:
        out.append(cyl(3.3, 3.35, 0.7, WHITE, 24).T(c[0], 28.3, c[1]))
        y0, R, h, nk = 29.0, 4.6, 10.5, 3.15
        tip0, ctop = 39.5, 46.0
    else:
        out.append(cyl(3.6, 3.6, 0.9, WHITE, 24).T(c[0], 25.6, c[1]))
        out.append(cyl(3.9, 3.9, 0.55, WHITE, 24).T(c[0], 26.45, c[1]))
        y0, R, h, nk = 27.0, 4.8, 9.5, 3.4
        tip0, ctop = 36.5, 43.0
    out.append(dome(c, y0, R, h, nk, 0.27, kind, cols, n, k, bump, seg=64, rows=32))
    gold.append(cyl(0.55, 0.12, 2.5, GOLD, 10).T(c[0], tip0 - 0.05, c[1]))
    gold.append(cross(c[0], tip0 + 2.3, c[1], ctop - tip0 - 2.3, -AX))
    return merge(out), merge(gold)


def diagonal_church(name):
    c = DIAG[name]
    kind, cols, n, k, bump = DIAG_DOMES[name]
    out, gold = [], []
    out.append(cyl(3.05, 3.05, 1.0, WHITE, 16).T(c[0], 11.8, c[1]))
    out.append(kok_ring(c, 2.85, 8, 12.8, 1.1, 2.5, AX))
    out.append(kok_ring(c, 2.55, 8, 13.9, 0.95, 2.8, AX + math.pi / 8))
    out.append(cyl(2.3, 2.3, 8.7, BRICK, 20).T(c[0], 11.8, c[1]))
    for kk in range(8):
        phi = AX + kk * math.pi / 4
        out.append(on_face(B(-0.14, 0.14, 15.5, 20.5, -0.12, 0.1, WHITE), c, phi + math.pi / 8, 2.28))
        out.append(on_face(merge([prism(arch_pts(0, 17.0, 18.9, 0.34), 0.0, 0.07, WHITE),
                                  prism(arch_pts(0, 17.15, 18.9, 0.22), 0.07, 0.1, DARK)]), c, phi, 2.27))
    out.append(cyl(2.4, 2.4, 0.4, WHITE, 20).T(c[0], 16.6, c[1]))
    out.append(cyl(2.55, 2.55, 0.5, WHITE, 20).T(c[0], 20.5, c[1]))
    out.append(dome(c, 21.0, 3.1, 5.0, 2.35, 0.33, kind, cols, n, k, bump, seg=56, rows=26))
    gold.append(cyl(0.35, 0.08, 1.6, GOLD, 8).T(c[0], 25.95, c[1]))
    gold.append(cross(c[0], 27.4, c[1], 4.1, -AX))
    return merge(out), merge(gold)


def basil_chapel():
    c, out, gold = BASIL, [], []
    out.append(oct_body(c, 2.2, 8.0, 14.5, BRICK))
    for phi in FACES:
        out.append(on_face(B(-0.13, 0.13, 8.0, 14.5, -0.1, 0.09, WHITE), c, phi + math.pi / 8, 2.15))
        out.append(on_face(merge([prism(arch_pts(0, 10.6, 12.3, 0.3), 0.0, 0.07, WHITE),
                                  prism(arch_pts(0, 10.75, 12.3, 0.2), 0.07, 0.1, DARK)]), c, phi, 2.2 * 0.92))
    out.append(oct_body(c, 2.45, 14.4, 14.9, WHITE))
    out.append(dome(c, 14.9, 2.8, 4.4, 2.2, 0.22, "studs", [GREEN, GOLD], 10, 5.0, 0.05, seg=48, rows=22))
    gold.append(cyl(0.25, 0.06, 1.3, GOLD, 8).T(c[0], 19.2, c[1]))
    gold.append(cross(c[0], 20.4, c[1], 5.1, -AX))
    return merge(out), merge(gold)


def bell_tower():
    c, out, gold = BELL, [], []
    rot = math.atan2(-1.1, -8.8) - math.pi + math.pi / 8
    fac = [rot - math.pi / 8 + k * math.pi / 4 for k in range(8)]
    out.append(oct_body(c, 3.8, 11.0, 13.6, BRICK, rot))
    out.append(oct_body(c, 3.95, 13.3, 13.7, WHITE, rot))
    out.append(oct_body(c, 2.6, 13.6, 18.6, DARK, rot))                            # the open belfry, bells inside
    for x, z in ngon(8, 3.5, rot, c):
        out.append(B(x - 0.38, x + 0.38, 13.6, 18.6, z - 0.38, z + 0.38, BRICK))
    out.append(loft(ngon(8, 3.8, rot, c), 18.5, ngon(8, 3.8, rot, c), 20.5, BRICK))
    out.append(loft(ngon(8, 2.5, rot, c), 18.5, ngon(8, 2.5, rot, c), 20.5, BRICK))
    out.append(cap(ngon(8, 3.8, rot, c), 18.5, BRICK_D, up=False))
    for phi in fac:
        out.append(on_face(prism(arch_pts(0, 18.1, 18.7, 1.1), -0.25, 0.02, WHITE), c, phi, 3.8 * math.cos(math.pi / 8)))
    out.append(oct_body(c, 4.0, 20.5, 21.0, WHITE, rot))
    out.append(kok_ring(c, 3.55, 8, 21.0, 1.15, 1.5, fac[0]))
    out.append(oct_body(c, 3.4, 21.0, 22.5, BRICK, rot))
    out.append(tent(c, 22.5, 30.2, 3.3, 0.55, 8, rot, TILE, YELLOW,
                    dormers=[(0.18, 0.42, 0.9, WHITE), (0.45, 0.32, 0.7, WHITE), (0.7, 0.22, 0.5, WHITE)], rib_r=0.1))
    out.append(oct_body(c, 0.6, 30.1, 30.3, WHITE, rot))
    gold.append(dome(c, 30.25, 0.85, 1.6, 0.55, 0.38, "plain", [GOLD], seg=20, rows=10))
    gold.append(cross(c[0], 31.8, c[1], 3.1, -AX))
    return merge(out), merge(gold)


def cathedral():
    parts, gold = [], []
    for fn in [lower_building, central_church, basil_chapel, bell_tower] + \
              [lambda n=n: axial_church(n) for n in AXIAL_DOMES] + [lambda n=n: diagonal_church(n) for n in DIAG]:
        a, g = fn()
        parts.append(a); gold.append(g)
    return merge(parts), merge(gold)


# ── The Kremlin wall and towers ─────────────────────────────────────────────
def merlon(w=2.3, h=2.4, notch=0.8, t=0.75, c=K_BRICK):
    """Swallowtail («ласточкин хвост») merlon standing on y = 0, outer face at w = 0, thickness inwards (−w)."""
    hw = w / 2
    left = [(-hw, 0.0), (0.0, 0.0), (0.0, h - notch), (-hw, h)]
    right = [(0.0, 0.0), (hw, 0.0), (hw, h), (0.0, h - notch)]
    return merge([prism(left, -t, 0.0, c), prism(right, -t, 0.0, c),
                  B(-0.12, 0.12, 0.5, 1.4, 0.0, 0.04, DARK)])


def merlon_row(p0, p1, y, step=3.6, **kw):
    """Merlons along the outer edge p0 -> p1 of a ccw ring."""
    L = math.dist(p0, p1)
    n = max(1, int(L / step))
    return merge([on_edge(merlon(**kw).T((k + 0.5) * L / n, y, 0), p0, p1) for k in range(n)])


def local(m, origin, ang):
    """Mesh built in a local frame (x = u, z = v) -> rotated by ang (radians, +x -> (cos, sin) in x/z)."""
    return m.R(0, -ang, 0).T(origin[0], 0, origin[1])


def clock_face(r):
    """A Kremlin clock dial facing +z: black face, gold ring, hour marks and hands."""
    out = [cyl(r, r, 0.12, DARK, 32).R(math.pi / 2, 0, 0).T(0, 0, 0.12),
           tube([(r * math.cos(t), r * math.sin(t), 0.26) for t in np.linspace(0, 2 * math.pi, 33)], 0.12, GOLD, 5)]
    for k in range(12):
        t = k * math.pi / 6
        out.append(B(-0.09, 0.09, r * 0.72, r * 0.9, 0.24, 0.3, GOLD).R(0, 0, t))
    out.append(B(-0.1, 0.1, 0, r * 0.62, 0.26, 0.32, GOLD).R(0, 0, -1.0))
    out.append(B(-0.08, 0.08, 0, r * 0.82, 0.3, 0.36, GOLD).R(0, 0, 0.6))
    return merge(out)


def star(span, depth=0.6):
    """Five-pointed ruby star in the x-y plane (a bipyramid), centred at 0."""
    R, r = span / 2, span / 2 * 0.42
    pts = [((R if k % 2 == 0 else r) * math.sin(k * math.pi / 5), (R if k % 2 == 0 else r) * math.cos(k * math.pi / 5)) for k in range(10)]
    v = [(x, y, 0.0) for x, y in pts] + [(0, 0, depth / 2), (0, 0, -depth / 2)]
    f = []
    for k in range(10):
        j = (k + 1) % 10
        f += [(k, j, 10), (j, k, 11)]
    m = Mesh(v, f, RUBY)
    frame = tube([(x * 1.03, y * 1.03, 0.0) for x, y in pts + [pts[0]]], 0.06, GOLD, 4)
    return m, frame


def spasskaya():
    """Built in the tower frame: u from the Kremlin side (0) out into Red Square (25.5), v across."""
    out, gold, ruby = [], [], []
    body = [(0.0, -9.0), (16.0, -9.0), (16.0, 8.4), (0.0, 8.4)]
    strel = [(15.9, -6.8), (25.5, -6.8), (25.5, 3.7), (15.9, 3.7)]
    cu, cv = 8.0, -0.3
    out.append(extrude(body, -5.0, 26.0, K_BRICK))
    out.append(band(body, -5.0, 0.6, 0.15, STONE))
    out.append(band(body, 12.0, 12.4, 0.12, WHITE))
    out.append(band(body, 25.2, 26.0, 0.3, WHITE))
    # white-stone parapet with small pinnacles, and the four tall corner pinnacles
    out.append(band(body, 26.0, 27.5, 0.05, WHITE))
    P = ccw(body)
    for i in range(4):
        p0, p1 = P[i], P[(i + 1) % 4]
        L = math.dist(p0, p1)
        for k in range(1, int(L / 2.0)):
            out.append(on_edge(merge([B(-0.15, 0.15, 27.5, 28.6, -0.3, 0.0, WHITE),
                                      cyl(0.17, 0.0, 0.7, WHITE, 4).T(0, 28.6, -0.15)]).T(k * L / int(L / 2.0), 0, 0), p0, p1))
    for x, z in body:
        x = x + (0.9 if x < cu else -0.9); z = z + (0.9 if z < cv else -0.9)
        out.append(B(x - 0.7, x + 0.7, 26.0, 31.5, z - 0.7, z + 0.7, WHITE))
        out.append(cyl(0.75, 0.0, 4.0, WHITE, 4).R(0, math.pi / 4, 0).T(x, 31.5, z))
        gold.append(cyl(0.05, 0.05, 1.2, GOLD, 4).T(x, 35.4, z))
    # windows of the body, the gate arch through the body and the strelnitsa
    for u in (4.0, 8.0, 12.0):
        out.append(B(u - 0.5, u + 0.5, 15.5, 17.6, 8.4, 8.5, DARK))
        out.append(B(u - 0.5, u + 0.5, 15.5, 17.6, -9.1, -9.0, DARK))
    out.append(extrude(strel, -5.0, 14.5, K_BRICK))
    out.append(band(strel, -5.0, 0.6, 0.15, STONE))
    out.append(band(strel, 13.8, 14.5, 0.25, WHITE))
    out.append(B(25.5, 25.62, 0.0, 6.8, -3.5, 0.4, WHITE))
    out.append(prism(arch_pts(1.55, 0.0, 5.0, 1.8), 0.0, 0.1, DARK).R(0, math.pi / 2, 0).T(25.63, 0, 0))
    S = ccw(strel)
    for i in range(4):
        p0, p1 = S[i], S[(i + 1) % 4]
        if math.dist(p0, p1) > 9:
            out.append(merlon_row(p0, p1, 14.5, step=2.6, w=1.6, h=1.8, notch=0.6, t=0.55))
    for u, v in ((25.0, -6.3), (25.0, 3.2)):
        out.append(B(u - 0.8, u + 0.8, 14.5, 18.0, v - 0.8, v + 0.8, WHITE))
        out.append(cyl(0.8, 0.0, 3.4, WHITE, 4).R(0, math.pi / 4, 0).T(u, 18.0, v))
    out.append(B(19.0, 23.0, 14.5, 17.5, -2.6, -0.4, WHITE))                          # the white aedicule over the gate
    # clock tier
    hw = 5.75
    out.append(B(cu - hw, cu + hw, 26.0, 41.3, cv - hw, cv + hw, K_BRICK))
    out.append(B(cu - hw - 0.2, cu + hw + 0.2, 31.2, 31.6, cv - hw - 0.2, cv + hw + 0.2, WHITE))
    out.append(B(cu - hw - 0.3, cu + hw + 0.3, 40.6, 41.6, cv - hw - 0.3, cv + hw + 0.3, WHITE))
    for a in range(4):
        phi = a * math.pi / 2
        dial = clock_face(3.06).T(0, 36.0, 0)
        out.append(on_face(dial, (cu, cv), phi, hw))
        for s in (-1, 1):
            out.append(on_face(B(s * 4.9 - 0.35, s * 4.9 + 0.35, 26.0, 40.6, -0.2, 0.25, WHITE), (cu, cv), phi, hw))
            out.append(on_face(merge([prism(arch_pts(s * 4.1, 27.8, 29.6, 0.45), 0.0, 0.1, WHITE),
                                      prism(arch_pts(s * 4.1, 28.0, 29.6, 0.3), 0.1, 0.14, DARK)]), (cu, cv), phi, hw))
    for sx in (-1, 1):
        for sz in (-1, 1):
            x, z = cu + sx * (hw - 0.6), cv + sz * (hw - 0.6)
            out.append(B(x - 0.6, x + 0.6, 41.5, 44.0, z - 0.6, z + 0.6, WHITE))
            out.append(cyl(0.65, 0.0, 3.0, WHITE, 4).R(0, math.pi / 4, 0).T(x, 44.0, z))
            gold.append(cyl(0.04, 0.04, 1.0, GOLD, 4).T(x, 46.9, z))
    # octagon with arches, open octagon, tent, star
    oc = (cu, cv)
    out.append(extrude(ngon(8, 5.3, math.pi / 8, oc), 41.5, 48.5, K_BRICK))
    for k in range(8):
        phi = k * math.pi / 4
        out.append(on_face(merge([prism(arch_pts(0, 43.0, 46.0, 0.8), 0.0, 0.12, WHITE),
                                  prism(arch_pts(0, 43.2, 46.0, 0.6), 0.12, 0.16, DARK)]), oc, phi, 5.3 * math.cos(math.pi / 8)))
        out.append(on_face(B(-0.25, 0.25, 41.5, 48.5, -0.2, 0.15, WHITE), oc, phi + math.pi / 8, 5.3 * 0.97))
    out.append(extrude(ngon(8, 5.6, math.pi / 8, oc), 48.3, 48.9, WHITE))
    out.append(extrude(ngon(8, 3.0, math.pi / 8, oc), 48.9, 52.5, DARK))
    for x, z in ngon(8, 4.1, math.pi / 8, oc):
        out.append(B(x - 0.35, x + 0.35, 48.9, 52.5, z - 0.35, z + 0.35, WHITE))
    out.append(extrude(ngon(8, 4.45, math.pi / 8, oc), 52.4, 53.0, WHITE))
    out.append(tent(oc, 53.0, 66.8, 4.0, 0.3, 8, math.pi / 8, K_TENT, WHITE,
                    dormers=[(0.15, 0.5, 1.1, WHITE), (0.42, 0.35, 0.8, WHITE)], rib_r=0.11))
    gold.append(cyl(0.12, 0.1, 1.0, GOLD, 6).T(cu, 66.6, cv))
    st, fr = star(3.75)
    for ang in (0.0, math.pi / 2):                                                  # seen from every side
        ruby.append(st.copy().R(0, ang, 0).T(cu, 69.1, cv))
        gold.append(fr.copy().R(0, ang, 0).T(cu, 69.1, cv))
    return merge(out), merge(gold), merge(ruby)


def kremlin():
    out, gold, ruby = [], [], []
    a, g, r = spasskaya()
    ang = math.radians(158)
    out.append(local(a, (107.7, 38.8), ang)); gold.append(local(g, (107.7, 38.8), ang)); ruby.append(local(r, (107.7, 38.8), ang))
    # the wall: body, walkway, merlons on the Red Square side
    for P in (R2470039, R2470034):
        Q = ccw(P)
        out.append(extrude(Q, -5.0, 9.6, K_BRICK, top_c=(0.55, 0.22, 0.17)))
        out.append(band(Q, -5.0, 0.5, 0.1, STONE))
        out.append(band(Q, 9.2, 9.6, 0.08, K_BRICK))
    out.append(merlon_row((98.25, 33.3), (98.1, 12.6), 9.6))
    out.append(merlon_row((98.15, 7.5), (98.15, -36.4), 9.6))
    out.append(merlon_row((102.05, 12.6), (101.95, 31.7), 9.6, step=4.5, h=1.3, notch=0.4))
    out.append(merlon_row((102.2, -36.2), (102.2, 7.5), 9.6, step=4.5, h=1.3, notch=0.4))
    # Tsarskaya tower: a stone pavilion on the wall
    T = ccw(R1360663)
    tc = (sum(p[0] for p in T) / 4, sum(p[1] for p in T) / 4)
    out.append(extrude(T, -5.0, 11.0, K_BRICK))
    out.append(band(T, 10.6, 11.0, 0.15, WHITE))
    for x, z in shrink(T, 0.78):
        out.append(lathe([(0.32, 0), (0.42, 0.5), (0.45, 1.0), (0.3, 1.8), (0.25, 2.4), (0.38, 2.8)], WHITE, 10).T(x, 11.0, z))
        out.append(cyl(0.55, 0.0, 1.6, K_TENT, 4).R(0, math.pi / 4, 0).T(x, 14.6, z))
        gold.append(cyl(0.04, 0.04, 0.6, GOLD, 4).T(x, 16.2, z))
    out.append(extrude(offset(T, 0.1), 13.8, 14.6, K_BRICK, top_c=K_TENT))
    out.append(band(T, 13.8, 14.1, 0.2, WHITE))
    out.append(tent(tc, 14.6, 16.4, 1.7, 0.12, 8, math.pi / 8, K_TENT, WHITE, rib_r=0.05))
    gold.append(cyl(0.05, 0.05, 0.9, GOLD, 4).T(tc[0], 16.3, tc[1]))
    gold.append(cyl(0.18, 0.18, 0.04, GOLD, 8).R(math.pi / 2, 0, 0).T(tc[0], 17.0, tc[1]))
    # Nabatnaya tower
    N = ccw(R1360661)
    nc = (sum(p[0] for p in N) / 4, sum(p[1] for p in N) / 4)
    na = math.atan2(0.8, 9.8)
    out.append(extrude(N, -5.0, 18.6, K_BRICK, top_c=(0.55, 0.22, 0.17)))
    out.append(band(N, -5.0, 0.6, 0.12, STONE))
    out.append(band(N, 9.4, 9.8, 0.1, WHITE))
    out.append(walls(offset(N, 0.35), 17.4, 18.6, K_BRICK))
    for i in range(4):
        out.append(merlon_row(offset(N, 0.35)[i], offset(N, 0.35)[(i + 1) % 4], 18.6, step=2.4, w=1.6, h=1.9, notch=0.6, t=0.5))
    hw = 3.3
    sq = [local(B(-hw, hw, 18.6, 26.0, -hw, hw, K_BRICK), nc, na)]
    for k in range(4):
        phi = na + k * math.pi / 2
        sq.append(on_face(merge([prism(arch_pts(0, 21.0, 23.6, 0.7), 0.0, 0.1, WHITE),
                                 prism(arch_pts(0, 21.2, 23.6, 0.5), 0.1, 0.14, DARK)]), nc, phi, hw))
    out += sq
    out.append(local(B(-hw - 0.2, hw + 0.2, 25.6, 26.3, -hw - 0.2, hw + 0.2, WHITE), nc, na))
    out.append(extrude(ngon(8, 1.9, na + math.pi / 8, nc), 26.3, 30.0, DARK))
    for x, z in ngon(8, 2.75, na + math.pi / 8, nc):
        out.append(B(x - 0.28, x + 0.28, 26.3, 30.0, z - 0.28, z + 0.28, K_BRICK))
    out.append(extrude(ngon(8, 3.1, na + math.pi / 8, nc), 29.9, 30.7, WHITE))
    out.append(tent(nc, 30.7, 37.3, 2.9, 0.15, 8, na + math.pi / 8, K_TENT, WHITE, dormers=[(0.2, 0.35, 0.7, WHITE)], rib_r=0.07))
    gold.append(cyl(0.05, 0.05, 0.9, GOLD, 4).T(nc[0], 37.2, nc[1]))
    gold.append(B(-0.02, 0.02, 37.6, 38.0, -0.25, 0.25, GOLD).T(nc[0], 0, nc[1]))
    return merge(out), merge(gold), merge(ruby)


def hull(pts):
    pts = sorted(set(pts))
    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def main():
    out = ROOT / "models" / "Q129846"
    out.mkdir(parents=True, exist_ok=True)
    cath, cgold = cathedral()
    kr, kgold, ruby = kremlin()
    gold = merge([cgold, kgold])
    parts = [("saint_basils_cathedral", cath), ("kremlin_wall_spasskaya_tsarskaya_nabatnaya", kr),
             ("gilding", gold, {"material": "gold"}), ("ruby_star", ruby, {"material": "ruby"})]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.85,
                           materials={"gold": {"metallic": 0.75, "roughness": 0.35}, "ruby": {"metallic": 0.2, "roughness": 0.25}})
    allv = np.vstack([p[1].v for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    for p in parts:
        print(f"  {p[0]}: {p[1].tris:,}")
    if "--footprint" in sys.argv:
        pts = [(round(float(x), 1), round(float(z), 1)) for x, z in allv[allv[:, 1] < 12.5][:, [0, 2]]]
        fp = [[round(x, 1), round(z, 1)] for x, z in offset(hull(pts), 1.0)]
        print(json.dumps(fp))


if __name__ == "__main__":
    main()
