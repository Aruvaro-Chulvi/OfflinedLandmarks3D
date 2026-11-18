"""
Leaning Tower of Pisa (Q39054) and the whole Piazza dei Miracoli (Piazza del Duomo, Pisa), built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The tower: begun 1173 (attributed to Bonanno Pisano, Guglielmo and/or Diotisalvi, died before 1200), resumed by
Giovanni di Simone (1272), belfry by Tommaso di Andrea Pisano (finished 1372). The cathedral: Buscheto (begun 1064)
and Rainaldo (façade, 12th c.). The baptistery: Diotisalvi (1152), Nicola and Giovanni Pisano (Gothic upper parts,
finished 1363). The Camposanto: Giovanni di Simone (1278–1464). Public domain.

What is modelled:
- The tower (the model is filed under it and its origin is the centre of its base): the stepped plinth, the ground
  storey with 15 blind arches on engaged columns and grey bands, the six open loggias of 30 columns each with
  their arches and cornices around the striped core, the railing of the terrace, the belfry with 12 arched bell
  openings, its cornice and top railing. The whole tower is ROTATED ABOUT THE CENTRE OF ITS BASE by today's lean,
  3.97°, towards bearing 180° (south): the top stands 3.9 m south of the base, the belfry rim reaches 56.67 m on
  the north side and 55.8 m on the south side (the published 56.67 / 55.86 m).
- The cathedral (Duomo di Santa Maria Assunta): the Latin cross of five naves and a three-nave transept, the
  clerestories, lean-to and gabled lead roofs, the main apse and the two transept apses (with their loggias), the
  elliptical dome on its drum with its gallery of columns, lantern and cross, the façade with the seven blind
  arches, the three bronze doors and the four tiers of open loggias, the statues of the Madonna and the
  evangelists, the blind arcades and windows of the side walls, the white marble with grey bands, the marble
  platform with its step.
- The baptistery: the stepped base, the ground storey with 20 blind arches, half-columns and four portals, the
  loggia of 60 small columns with 20 Gothic gables and pinnacles above it, the third storey with 20 windows, gables
  and pinnacles, the outer dome (lead on the south-east half facing the cathedral, red tiles on the north-west
  half facing the sea), the 12-sided truncated cone that rises out of it, the small cupola and the statue of
  St John the Baptist.
- The Camposanto: the long walled cloister on the north side with its 43 blind arches on the outside (all four
  sides), the lead gallery roofs, the courtyard with its tracery windows (mullions) and three lawns, the doors
  and the Gothic tabernacle of the south front, the Cappella Dal Pozzo with its small dome at the east end.
- The square: the green lawns (prato) as OpenStreetMap draws them, on a paved ground that stands for the paths;
  the iron fence around the tower.
- Not modelled: the Museo dell'Opera, the Palazzo dell'Opera and the other buildings around the square, the city
  walls, interiors (but for the Camposanto galleries), the tower's "catino" (the shallow ring dug round its base)
  and the slight banana curve of its upper storeys (the masons corrected the lean as they built).

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap ways 35667218 (tower base), 22945163 + building:parts 263406583, 265815349–265815382
  (cathedral), 22945162 + parts 252318078–252318327 and relations 3384306–3384315 (baptistery), relation 154289
  (ways 35667219/35667220: Camposanto), 163739528 (Cappella Dal Pozzo), 265818616 (cathedral platform), the grass
  areas 112723669–115600063, 217906953, 237169474–237169476, in the axes of heading 270.3° (the cathedral façade
  faces due west; its nave is squared to the axes within ~0.2 m).
- Tower: 56.67 m high side, 55.86 m low side, base Ø 15.484 m, 8 storeys, lean 3.97° (en.wikipedia); the storey
  levels from the OSM parts (relations 3381971–3382005: ground storey cornice 10.6–13 m, 30-column loggias from
  13 to 48 m, belfry railing 55.8–57 m); axial height 56.37 m (so that the two sides come out at 56.67 / 55.80 m),
  belfry Ø 12.7 m (OSM 252151275). Six loggias evenly spaced, 6.07 m each (the OSM cornice levels are uneven).
- Cathedral: OSM building:parts heights (nave 35 m, aisles 30 m, dome drum 37 m, dome 45 m, lantern 47 m,
  apse 30 m, transept apses 20 m, façade 36.5 m); body 36.4 × 88.7 m, transept 64.7 × 24.9 m, total length
  ≈ 100 m with the apse. Aisle walls 22 m, clerestory 31 m, ridges 35 m, façade gable 35.4 m (published ≈ 34–35 m)
  and the dome to 48.4 m with the cross. Tiers of the façade estimated on photos (±10 %).
- Baptistery: Ø 34.13 m (circumference 107.2 m), 54.86 m high (en.wikipedia), OSM parts: ground order to 13.4 m,
  loggia 13.4–18.8 m (columns), gables to 20.7 m, third order to 32 m, dome 32–45 m, cone to 51.5 m, cupola to
  55 m; the lead and tiled faces of the cone as the OSM roof:material tags give them.
- Camposanto: OSM outline (≈ 136 × 50 m, published 126 × 52 m inside), height of the walls 10.6 m and of the roof
  ridge 13.6 m estimated (±15 %); 43 blind arches on the outside (en.wikipedia).
White Carrara and San Giuliano marble with grey bands, lead roofs, the baptistery dome half lead, half tiles.

Axes (docs/MODEL_SPEC.md): +Y up, +Z points to heading 270.3° (west: the cathedral façade and the baptistery),
+X to 180.3° (south). Origin on the ground at the centre of the base of the tower; the cathedral lies 23–123 m to
the west, the baptistery 196 m, the Camposanto 82–138 m to the north.

    python model_sources/Q39054_leaning_tower_of_pisa.py   -> models/Q39054/model.glb (+ triangles in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, cap, walls, extrude, offset, ring_area, write_glb  # noqa: E402

HEADING = 270.3
LEAN_DEG = 3.97
LEAN_BEARING = 180.0
TOWER_H = 56.37          # axial height of the tower to the top of the belfry

# ── Colours ──────────────────────────────────────────────────────────────────
MARBLE = (0.925, 0.905, 0.855)     # white Carrara / San Giuliano marble
MARBLE_L = (0.95, 0.935, 0.895)    # cornices, columns
MARBLE_SH = (0.80, 0.78, 0.735)    # loggia backs, reveals (in shade)
MARBLE_D = (0.70, 0.69, 0.66)      # plinths, foundations
BAND = (0.56, 0.57, 0.57)          # grey bands
LOZ = (0.45, 0.47, 0.48)           # lozenges, inlays
LEAD = (0.56, 0.58, 0.61)
LEAD_D = (0.48, 0.50, 0.53)
TILE = (0.72, 0.38, 0.25)
DARK = (0.13, 0.12, 0.12)          # openings, windows
GALLERY = (0.40, 0.36, 0.31)       # Camposanto gallery walls (frescoes in the shade)
WOOD = (0.36, 0.25, 0.16)
BRONZE = (0.42, 0.31, 0.17)
GOLD = (0.80, 0.64, 0.28)
IRON = (0.14, 0.14, 0.15)
GRASS = (0.40, 0.59, 0.27)
GRASS_E = (0.34, 0.47, 0.24)
PAVING = (0.73, 0.71, 0.66)
STEP = (0.86, 0.84, 0.80)

# ── Data (OpenStreetMap © OpenStreetMap contributors, ODbL), metres in the model axes ───────────────────
DATA = json.loads(r"""{"platform":[[-54.4,128.1],[-54.4,75.1],[-68.8,75.1],[-68.8,65.5],[-69.6,64.7],[-71.4,61.4],[-72.0,57.6],[-71.4,53.9],[-69.6,50.5],[-68.7,49.6],[-68.6,39.6],[-54.5,39.7],[-54.6,28.8],[-44.6,28.7],[-44.6,28.4],[-43.8,26.1],[-42.5,23.7],[-40.7,21.8],[-38.2,19.8],[-35.6,18.7],[-32.8,18.0],[-29.9,18.0],[-26.6,18.5],[-23.4,20.1],[-20.6,22.5],[-18.9,24.9],[-17.5,28.1],[-17.5,28.9],[-7.4,28.9],[-7.5,39.9],[6.4,40.0],[6.4,50.1],[8.6,52.6],[9.9,56.3],[10.0,60.1],[8.7,63.9],[6.5,66.4],[6.5,75.2],[-7.7,75.2],[-8.0,128.2],[-22.6,128.1],[-31.1,128.1],[-39.6,128.1]],"cathedral":[[1.2,69.9],[-13.0,69.9],[-13.3,122.8],[-19.8,122.8],[-22.6,122.8],[-31.1,122.8],[-39.6,122.8],[-41.9,122.8],[-49.1,122.8],[-49.1,69.9],[-63.5,69.8],[-63.5,63.9],[-63.5,63.0],[-64.1,62.7],[-65.4,61.4],[-66.3,59.6],[-66.7,57.6],[-66.3,55.6],[-65.4,53.8],[-63.9,52.4],[-63.4,52.2],[-63.4,45.0],[-51.7,45.0],[-49.2,45.0],[-49.3,34.0],[-39.3,34.0],[-39.3,29.3],[-38.9,28.2],[-38.2,26.9],[-37.1,25.7],[-35.5,24.4],[-33.9,23.8],[-32.1,23.3],[-30.4,23.3],[-28.2,23.6],[-26.3,24.6],[-24.6,26.1],[-23.5,27.6],[-22.8,29.2],[-22.8,34.2],[-12.7,34.2],[-12.8,45.2],[-10.3,45.2],[1.1,45.2],[1.1,52.4],[1.1,52.6],[2.7,53.6],[4.0,55.3],[4.7,57.2],[4.7,59.3],[4.0,61.3],[2.8,62.9],[1.2,63.9]],"campo_out":[[-94.1,203.4],[-91.0,170.5],[-81.9,75.3],[-99.0,72.7],[-109.0,71.0],[-124.8,68.4],[-128.0,84.5],[-129.1,93.8],[-131.1,117.9],[-135.6,171.1],[-138.3,202.5]],"campo_in":[[-104.9,190.9],[-103.3,174.9],[-102.9,171.4],[-102.7,168.6],[-99.2,127.0],[-98.8,123.0],[-98.6,120.3],[-95.6,85.6],[-115.3,82.8],[-118.6,118.4],[-118.9,121.0],[-119.2,125.3],[-122.9,167.2],[-123.1,169.8],[-123.4,173.6],[-124.8,190.0]],"pozzo":[[-109.0,71.0],[-107.5,61.9],[-97.4,63.7],[-98.6,70.8],[-99.0,72.7]],"lawns":{"112723669":[[-25.5,168.1],[-25.2,132.5],[-5.2,132.8],[-5.0,76.6],[12.3,77.3],[12.4,67.4],[15.1,63.7],[15.8,62.0],[16.4,59.5],[16.2,57.0],[15.7,55.3],[14.8,53.4],[14.1,52.2],[12.6,51.1],[12.7,48.5],[50.6,55.2],[33.8,170.8]],"112723672":[[-86.1,167.0],[-82.3,123.0],[-56.9,123.5],[-57.0,132.1],[-31.6,132.4],[-31.6,167.7]],"112723675":[[-63.9,-3.3],[-27.5,6.1],[-26.0,6.4],[-24.7,6.3],[-23.4,5.9],[-22.3,5.2],[-21.6,4.2],[-21.3,2.8],[-21.3,1.3],[-22.6,-7.6],[-23.3,-16.5],[-23.3,-22.7],[-23.2,-29.5],[-22.6,-36.6],[-21.6,-44.1],[-20.1,-52.5],[-17.5,-63.2],[-57.5,-64.1],[-58.4,-63.7],[-59.3,-63.0],[-59.7,-61.8],[-65.4,-5.9]],"112723676":[[-13.8,-5.2],[-14.0,-15.5],[-13.6,-25.2],[-13.2,-31.5],[-12.6,-38.0],[-11.6,-43.9],[-10.1,-50.7],[-5.5,-71.3],[-0.2,-70.3],[9.6,-10.6],[7.1,-12.5],[4.7,-13.6],[1.9,-14.5],[-1.0,-14.5],[-4.1,-14.0],[-6.7,-13.1],[-8.8,-11.8],[-10.7,-10.0],[-12.2,-7.8]],"112723677":[[-81.9,119.4],[-78.2,76.3],[-56.3,76.4],[-56.8,119.8]],"115600052":[[12.0,-7.5],[13.3,-4.2],[13.8,-1.2],[13.9,1.7],[13.5,4.3],[12.4,6.7],[10.5,9.3],[8.5,11.3],[6.0,12.7],[8.1,14.3],[11.3,15.5],[13.7,15.7],[16.3,15.1],[46.2,2.6]],"115600053":[[-62.7,5.0],[-29.1,14.4],[-37.6,16.7],[-43.0,21.0],[-46.0,26.0],[-49.7,25.6]],"115600055":[[51.4,51.5],[12.8,45.1],[12.8,43.0],[44.4,28.0],[52.9,39.9]],"115600061":[[-55.6,38.1],[-55.7,25.2],[-67.4,7.6],[-70.8,38.1]],"115600063":[[54.8,28.3],[53.8,27.8],[53.2,27.2],[52.3,26.1],[52.2,25.4],[51.9,23.7],[46.1,27.0],[53.8,37.8]],"217906953":[[32.9,177.6],[12.6,298.9],[-52.9,300.3],[-64.4,300.6],[-71.9,300.9],[-76.8,301.0],[-83.2,242.7],[-80.0,242.3],[-80.8,234.3],[-84.5,234.7],[-85.0,227.7],[-87.8,228.1],[-147.6,235.9],[-150.7,236.3],[-150.8,234.0],[-153.1,234.2],[-152.9,228.1],[-150.2,203.4],[-91.5,205.0],[-91.3,201.8],[-89.1,201.3],[-86.7,174.0],[-73.3,173.7],[-60.0,173.6],[-33.8,174.2],[-36.9,176.5],[-42.2,180.0],[-47.4,186.3],[-49.7,193.5],[-49.2,200.3],[-47.2,206.7],[-43.1,212.0],[-37.6,215.9],[-31.1,217.8],[-24.3,217.6],[-17.9,215.3],[-12.6,211.2],[-8.5,205.1],[-6.8,197.9],[-7.5,190.5],[-10.7,183.9],[-15.5,179.0],[-21.2,174.7]],"237169474":[[-123.4,173.6],[-103.3,174.9],[-104.9,190.9],[-124.8,190.0]],"237169475":[[-122.9,167.2],[-102.7,168.6],[-99.2,127.0],[-119.2,125.3]],"237169476":[[-118.6,118.4],[-98.6,120.3],[-95.6,85.6],[-115.3,82.8]]}}""")
FOOTPRINT = [(34, 177), (13, 300), (-77, 302), (-84, 243), (-88, 229), (-148, 236.5), (-150.5, 228), (-150.5, 203.5),
             (-138.8, 203.4), (-136, 171), (-131.7, 118), (-129.6, 94), (-128.6, 84.5), (-125.4, 67.8), (-110, 61.5),
             (-96.5, 61.5), (-82, 73), (-70, 67), (-66, 30), (-66, -66), (-14, -73), (11, -72), (47, -10), (55, 24),
             (57, 55)]


def bearing_dir(b):
    """Unit (x, z) in the model axes of a compass bearing."""
    e, n = math.sin(math.radians(b)), math.cos(math.radians(b))
    c, s = math.cos(math.radians(HEADING)), math.sin(math.radians(HEADING))
    return np.array([-e * c + n * s, e * s + n * c])


def bearing_of(x, z):
    c, s = math.cos(math.radians(HEADING)), math.sin(math.radians(HEADING))
    e, n = -x * c + z * s, x * s + z * c
    return math.degrees(math.atan2(e, n)) % 360


# ── Builders ─────────────────────────────────────────────────────────────────
class B:
    """Free polygons, each wound so that its normal points to `want`."""

    def __init__(self):
        self.v, self.f, self.c = [], [], []

    def poly(self, pts, col, want=None):
        P = [np.asarray(p, float) for p in pts]
        if want is not None:
            n = np.zeros(3)
            for k in range(1, len(P) - 1):
                n += np.cross(P[k] - P[0], P[k + 1] - P[0])
            if np.dot(n, want) < 0:
                P = P[::-1]
        o = len(self.v)
        self.v += P
        self.c += [col] * len(P)
        for k in range(1, len(P) - 1):
            self.f.append((o, o + k, o + k + 1))

    def mesh(self):
        if not self.v:
            return Mesh()
        return Mesh(np.array(self.v), np.array(self.f, np.int64), np.array(self.c, float))


class Circle:
    """Path round a circle (or an ellipse: radii r·sx, r·sz), u in 0..1 from angle a0 to a1."""

    def __init__(self, cx, cz, r, a0=0.0, a1=2 * math.pi, sx=1.0, sz=1.0):
        self.cx, self.cz, self.r, self.a0, self.a1, self.sx, self.sz = cx, cz, r, a0, a1, sx, sz
        a, b = r * sx, r * sz
        per = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
        self.L = per * abs(a1 - a0) / (2 * math.pi)

    def __call__(self, u):
        a = self.a0 + u * (self.a1 - self.a0)
        p = np.array([self.cx + self.r * self.sx * math.cos(a), self.cz + self.r * self.sz * math.sin(a)])
        n = np.array([math.cos(a) / self.sx, math.sin(a) / self.sz])
        return p, n / np.linalg.norm(n)


class Line:
    """Straight path from p0 to p1, its normal pointing away from `inside`."""

    def __init__(self, p0, p1, inside):
        self.p0, self.p1 = np.array(p0, float), np.array(p1, float)
        d = self.p1 - self.p0
        self.L = float(np.linalg.norm(d))
        n = np.array([d[1], -d[0]]) / self.L
        if np.dot(n, (self.p0 + self.p1) / 2 - np.array(inside, float)) < 0:
            n = -n
        self.n, self.t = n, d / self.L

    def __call__(self, u):
        return self.p0 + u * (self.p1 - self.p0), self.n


def at(path, u, off, y):
    p, n = path(u)
    q = p + n * off
    return np.array([q[0], y, q[1]]), np.array([n[0], 0.0, n[1]])


def arcade(b, path, n, y_bot, y_spring, y_top, depth, col, pier, rise=None, seg=6, jambs=True, off=None,
           col_in=None, col_rev=None):
    """A row of n arches along a path: the front face at `off` (default: depth) out of the path, the back face
    `depth` behind it, the soffits and jambs between them. Piers `pier` wide; with jambs the openings go down to
    y_bot, without them the arches spring from y_spring (columns are added apart)."""
    off = depth if off is None else off
    Lb = path.L / n
    hw = max((Lb - pier) / 2, 0.05)
    rs = hw if rise is None else rise
    jambs = jambs and y_bot < y_spring - 1e-6
    yb = y_bot if jambs else y_spring
    prof = [(0.0, yb), (pier / 2, yb)]
    if jambs:
        prof.append((pier / 2, y_spring))
    for k in range(1, seg):
        t = math.pi * k / seg
        prof.append((pier / 2 + hw * (1 - math.cos(t)), y_spring + rs * math.sin(t)))
    if jambs:
        prof.append((Lb - pier / 2, y_spring))
    prof += [(Lb - pier / 2, yb), (Lb, yb)]
    for i in range(n):
        uc = (i + 0.5) / n
        for k in range(len(prof) - 1):
            (s0, b0), (s1, b1) = prof[k], prof[k + 1]
            u0, u1 = (i + s0 / Lb) / n, (i + s1 / Lb) / n
            if s1 - s0 > 1e-6:
                for d, sg in ((0.0, 1), (depth, -1)):
                    a, nn = at(path, u0, off - d, b0)
                    q, _ = at(path, u1, off - d, b1)
                    c, _ = at(path, u1, off - d, y_top)
                    e, _ = at(path, u0, off - d, y_top)
                    b.poly([a, q, c, e], col if sg > 0 else (col_in or col), nn * sg)
            if 1 <= k <= len(prof) - 3:
                a, _ = at(path, u0, off, b0)
                q, _ = at(path, u1, off, b1)
                c, _ = at(path, u1, off - depth, b1)
                e, _ = at(path, u0, off - depth, b0)
                ctr, _ = at(path, uc, off - depth / 2, min((b0 + b1) / 2, y_spring))
                b.poly([a, q, c, e], col_rev or MARBLE_SH, ctr - (a + q + c + e) / 4)


def columns(path, n, r, y0, y1, col, off=0.0, seg=8, closed=True, capital=True, shift=0.0):
    ms = []
    for i in range(n if closed else n + 1):
        p, nn = path((i + shift) / n)
        q = p + nn * off
        ms.append(cyl(r, r, y1 - y0, col, seg=seg, caps=False).T(q[0], y0, q[1]))
        if capital:
            ms.append(box(r * 2.5, min(0.35, (y1 - y0) * 0.12), r * 2.5, MARBLE_L).T(q[0], y1 - min(0.35, (y1 - y0) * 0.12), q[1]))
            ms.append(box(r * 2.4, 0.2, r * 2.4, MARBLE_L).T(q[0], y0, q[1]))
    return merge(ms)


def rect_on(b, path, u, off, y0, y1, w, col):
    """A flat rectangle (door, window) facing out of the path at u."""
    p, n = path(u)
    t = np.array([-n[1], n[0]])
    q = p + n * off
    P = [np.array([q[0] + t[0] * s, y, q[1] + t[1] * s]) for s, y in ((-w / 2, y0), (w / 2, y0), (w / 2, y1), (-w / 2, y1))]
    b.poly(P, col, np.array([n[0], 0, n[1]]))


def diamond_on(b, path, u, off, yc, h, col):
    p, n = path(u)
    t = np.array([-n[1], n[0]])
    q = p + n * off
    P = [np.array([q[0] + t[0] * s, yc + dy, q[1] + t[1] * s]) for s, dy in ((0, -h / 2), (h * 0.38, 0), (0, h / 2), (-h * 0.38, 0))]
    b.poly(P, col, np.array([n[0], 0, n[1]]))


def prism_at(p, t, n, w, d, y0, y1, col):
    """An upright box w (along t) × d (along n) from y0 to y1, centred on p (x, z)."""
    p, t, n = np.asarray(p, float), np.asarray(t, float), np.asarray(n, float)
    ring = [p - t * w / 2 - n * d / 2, p + t * w / 2 - n * d / 2, p + t * w / 2 + n * d / 2, p - t * w / 2 + n * d / 2]
    return extrude([tuple(q) for q in ring], y0, y1, col, bottom=True)


def gable_plate(b, p, n, w, y0, y1, thick, col):
    """A Gothic gable (wimperg): a triangular plate w wide from y0 to its point at y1, facing n."""
    p, n = np.asarray(p, float), np.asarray(n, float)
    t = np.array([-n[1], n[0]])
    F = [p + n * thick / 2 + t * s for s in (-w / 2, w / 2, 0.0)]
    K = [p - n * thick / 2 + t * s for s in (-w / 2, w / 2, 0.0)]
    ys = (y0, y0, y1)
    f3 = [np.array([q[0], y, q[1]]) for q, y in zip(F, ys)]
    k3 = [np.array([q[0], y, q[1]]) for q, y in zip(K, ys)]
    N3 = np.array([n[0], 0, n[1]])
    b.poly(f3, col, N3)
    b.poly(k3, col, -N3)
    for i, j in ((0, 2), (2, 1), (1, 0)):
        mid = (f3[i] + f3[j]) / 2
        out = mid - (f3[0] + f3[1] + f3[2]) / 3
        b.poly([f3[i], f3[j], k3[j], k3[i]], col, out)


def pinnacle(x, z, y0, y1, w, col, tip=1.2):
    return merge([box(w, y1 - y0, w, col).T(x, y0, z), cyl(w * 0.62, 0.0, tip, col, seg=4).R(0, math.pi / 4, 0).T(x, y1, z)])


def revolve(prof, cx, cz, a0, a1, seg, col, sx=1.0, sz=1.0):
    """A body of revolution [(r, y)] over the angles a0..a1 (an apse, a dome, a sector of a dome)."""
    a = np.linspace(a0, a1, seg + 1)
    v = [(cx + r * sx * math.cos(t), y, cz + r * sz * math.sin(t)) for r, y in prof for t in a]
    m = seg + 1
    f = []
    for i in range(len(prof) - 1):
        for j in range(seg):
            p, q = i * m + j, i * m + j + 1
            f += [(p, q + m, p + m), (p, q, q + m)]
    return Mesh(v, f, col)


def striped_rev(cx, cz, r, y0, y1, a0, a1, seg, period=1.25, band=0.28, col=MARBLE, sx=1.0, sz=1.0):
    ms, y = [], y0
    while y < y1 - 1e-6:
        ya = min(y + period - band, y1)
        ms.append(revolve([(r, y), (r, ya)], cx, cz, a0, a1, seg, col, sx, sz))
        if ya < y1 - 1e-6:
            yb = min(ya + band, y1)
            ms.append(revolve([(r, ya), (r, yb)], cx, cz, a0, a1, seg, BAND, sx, sz))
        y += period
    return merge(ms)


def striped_walls(ring, y0, y1, period=1.25, band=0.28, col=MARBLE, hole=False):
    ms, y = [], y0
    while y < y1 - 1e-6:
        ya = min(y + period - band, y1)
        ms.append(walls(ring, y, ya, col, hole=hole))
        if ya < y1 - 1e-6:
            ms.append(walls(ring, ya, min(ya + band, y1), BAND, hole=hole))
        y += period
    return merge(ms)


def rect(x0, x1, z0, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def gable_roof(b, x0, x1, z0, z1, ye, yr, along, col, end_col, ov=0.5, ends=(True, True)):
    """A pitched roof over an axis-aligned rectangle, ridge along 'x' or 'z', with its end gables."""
    if along == "z":
        xm = (x0 + x1) / 2
        drop = ov * (yr - ye) / ((x1 - x0) / 2)
        for xe in (x0 - ov, x1 + ov):
            b.poly([(xe, ye - drop, z0 - 0.3), (xe, ye - drop, z1 + 0.3), (xm, yr, z1 + 0.3), (xm, yr, z0 - 0.3)], col,
                   np.array([xe - xm, 1.5, 0]))
        for k, z in enumerate((z0, z1)):
            if ends[k]:
                b.poly([(x0, ye, z), (x1, ye, z), (xm, yr, z)], end_col, np.array([0, 0, z - (z0 + z1) / 2]))
    else:
        zm = (z0 + z1) / 2
        drop = ov * (yr - ye) / ((z1 - z0) / 2)
        for ze in (z0 - ov, z1 + ov):
            b.poly([(x0 - 0.3, ye - drop, ze), (x1 + 0.3, ye - drop, ze), (x1 + 0.3, yr, zm), (x0 - 0.3, yr, zm)], col,
                   np.array([0, 1.5, ze - zm]))
        for k, x in enumerate((x0, x1)):
            if ends[k]:
                b.poly([(x, ye, z0), (x, ye, z1), (x, yr, zm)], end_col, np.array([x - (x0 + x1) / 2, 0, 0]))


def lean_to(b, x0, x1, z0, z1, y_low, y_high, high, col, end_col, ov=0.4):
    """A one-slope roof over a rectangle, high along the side `high` ('+x', '-x', '+z', '-z')."""
    if high in ("+x", "-x"):
        xl, xh = (x0, x1) if high == "+x" else (x1, x0)
        xlo = xl - ov * np.sign(xh - xl)
        ylo = y_low - ov * (y_high - y_low) / abs(x1 - x0)
        b.poly([(xlo, ylo, z0 - 0.2), (xlo, ylo, z1 + 0.2), (xh, y_high, z1 + 0.2), (xh, y_high, z0 - 0.2)], col,
               np.array([xl - xh, 3, 0]))
        for z in (z0, z1):
            b.poly([(xl, y_low, z), (xh, y_low, z), (xh, y_high, z)], end_col, np.array([0, 0, z - (z0 + z1) / 2]))
    else:
        zl, zh = (z0, z1) if high == "+z" else (z1, z0)
        zlo = zl - ov * np.sign(zh - zl)
        ylo = y_low - ov * (y_high - y_low) / abs(z1 - z0)
        b.poly([(x0 - 0.2, ylo, zlo), (x1 + 0.2, ylo, zlo), (x1 + 0.2, y_high, zh), (x0 - 0.2, y_high, zh)], col,
               np.array([0, 3, zl - zh]))
        for x in (x0, x1):
            b.poly([(x, y_low, zl), (x, y_low, zh), (x, y_high, zh)], end_col, np.array([x - (x0 + x1) / 2, 0, 0]))


def ledge(p0, p1, nrm, y0, y1, proj, col=MARBLE_L, back=0.3):
    p0, p1, n = np.asarray(p0, float), np.asarray(p1, float), np.asarray(nrm, float)
    ring = [p0 - n * back, p1 - n * back, p1 + n * proj, p0 + n * proj]
    return extrude([tuple(q) for q in ring], y0, y1, col, bottom=True)


def figure(h, col=MARBLE_L, r=0.32):
    prof = [(r * 1.1, 0), (r, 0.12 * h), (r * 0.85, 0.55 * h), (r * 0.7, 0.8 * h), (r * 0.45, 0.84 * h),
            (r * 0.5, 0.93 * h), (r * 0.25, h), (0.0, h)]
    return lathe(prof, col, seg=8)


# ── Tower ────────────────────────────────────────────────────────────────────
def lean(m):
    """Rotate about the centre of the base (the origin) by the lean, towards LEAN_BEARING."""
    d = bearing_dir(LEAN_BEARING)
    k = np.array([d[1], 0.0, -d[0]])
    th = math.radians(LEAN_DEG)
    v = m.v
    m.v = v * math.cos(th) + np.cross(k, v) * math.sin(th) + np.outer(v @ k, k) * (1 - math.cos(th))
    return m


def tower():
    ms, b, irn = [], B(), []
    R0 = 7.742                                        # base Ø 15.484 m
    TAU = 2 * math.pi
    # plinth and foundation (the north side lifts 0.54 m when the tower leans)
    ms.append(cyl(8.2, 8.2, 3.35, MARBLE_D, seg=48).T(0, -3.0, 0))
    ms.append(cyl(8.0, 8.0, 0.35, STEP, seg=48).T(0, 0.35, 0))
    # ground storey: core with grey bands behind 15 blind arches on engaged columns
    ms.append(striped_rev(0, 0, 7.36, 0.7, 11.0, 0, TAU, 48, period=1.1, band=0.22))
    g = Circle(0, 0, R0, -math.pi / 2, -math.pi / 2 + TAU)
    arcade(b, g, 15, 0.7, 8.6, 11.0, 0.38, MARBLE, pier=0.7, off=0.0, seg=6, col_rev=MARBLE_SH)
    ms.append(columns(g, 15, 0.3, 0.7, 8.6, MARBLE_L, off=0.0, seg=8))
    for i in range(15):
        diamond_on(b, g, (i + 0.5) / 15, -0.36, 9.55, 1.1, LOZ)
    rect_on(b, g, 0.5 / 15, -0.36, 0.7, 4.6, 1.6, DARK)              # the door (south side, as ground storey bay 0)
    ms.append(cyl(7.98, 7.98, 0.6, MARBLE_L, seg=48).T(0, 11.0, 0))
    # six loggias of 30 columns
    y0, H = 11.6, (48.0 - 11.6) / 6
    for k in range(6):
        yb = y0 + k * H
        ms.append(striped_rev(0, 0, 5.95, yb, yb + H, 0, TAU, 40, period=1.0, band=0.2, col=MARBLE_SH))
        ms.append(cyl(7.5, 7.5, 0.25, MARBLE_L, seg=48).T(0, yb, 0))
        c = Circle(0, 0, 7.2, k * 0.05, k * 0.05 + TAU)
        ys = yb + H - 1.45
        ms.append(columns(c, 30, 0.19, yb + 0.25, ys, MARBLE_L, seg=6, capital=False))
        arcade(b, c, 30, ys, ys, yb + H - 0.42, 0.5, MARBLE, pier=0.42, off=0.25, jambs=False, seg=4)
        ms.append(cyl(7.78, 7.78, 0.42, MARBLE_L, seg=48).T(0, yb + H - 0.42, 0))
    # terrace railing round the belfry
    for k in range(48):
        a = TAU * k / 48
        irn.append(box(0.07, 1.2, 0.07, IRON).T(7.5 * math.cos(a), 48.0, 7.5 * math.sin(a)))
    ring = [(7.5 * math.cos(TAU * k / 48), 49.2, 7.5 * math.sin(TAU * k / 48)) for k in range(49)]
    irn.append(tube(ring, 0.05, IRON, seg=4))
    # belfry: Ø 12.7 m, 12 bell openings, cornice, top railing
    ms.append(cyl(5.7, 5.7, 6.2, DARK, seg=24, caps=False).T(0, 49.2, 0))
    ms.append(cyl(6.35, 6.35, 1.2, MARBLE, seg=48, caps=False).T(0, 48.0, 0))
    bel = Circle(0, 0, 6.35, 0.0, TAU)
    arcade(b, bel, 12, 49.2, 53.2, 55.45, 0.65, MARBLE, pier=1.35, off=0.0, seg=6)
    ms.append(cyl(6.62, 6.62, 0.45, MARBLE_L, seg=48).T(0, 55.45, 0))
    ms.append(cyl(6.36, 6.36, TOWER_H - 55.9, MARBLE, seg=48).T(0, 55.9, 0))
    ms.append(cyl(1.0, 1.0, 1.8, MARBLE, seg=8).T(-2.5, TOWER_H, 1.5))          # stair head on the top terrace
    for k in range(40):
        a = TAU * k / 40
        irn.append(box(0.06, 1.0, 0.06, IRON).T(6.2 * math.cos(a), TOWER_H, 6.2 * math.sin(a)))
    ring = [(6.2 * math.cos(TAU * k / 40), TOWER_H + 1.0, 6.2 * math.sin(TAU * k / 40)) for k in range(41)]
    irn.append(tube(ring, 0.05, IRON, seg=4))
    stone = merge(ms + [b.mesh()])
    return lean(stone), lean(merge(irn))


# ── Cathedral ────────────────────────────────────────────────────────────────
CX = -31.1                       # axis of the nave
XA0, XA1 = -49.2, -12.8          # outer walls of the aisles
XN0, XN1 = -39.5, -22.7          # walls of the nave (clerestory)
ZE, ZF = 34.1, 122.8             # east end of the body, façade
TX0, TX1 = -63.5, 1.2            # ends of the transept
TZ0, TZ1 = 45.0, 69.9            # transept outer walls
TN0, TN1 = 52.3, 63.9            # transept nave
TZC = 58.1
DOME = (-31.4, 58.3, 6.8, 9.3)   # centre and semi-axes of the drum
Y0 = 0.25                        # top of the platform


def side_orders(b, ms, p0, p1, inside, nb, lower=True, upper=True):
    """The two orders of a side wall: blind arches below, pilasters, arches and windows above."""
    ln = Line(p0, p1, inside)
    if lower:
        arcade(b, ln, nb, Y0, 8.6, 11.0, 0.35, MARBLE, pier=1.1, seg=6)
        for i in range(nb):
            diamond_on(b, ln, (i + 0.5) / nb, 0.02, 9.6, 1.1, LOZ)
        ms.append(ledge(ln.p0, ln.p1, ln.n, 11.0, 11.4, 0.55))
    if upper:
        arcade(b, ln, nb, 11.4, 18.4, 21.3, 0.25, MARBLE, pier=1.6, seg=6)
        for i in range(nb):
            rect_on(b, ln, (i + 0.5) / nb, 0.02, 13.6, 17.4, 0.9, DARK)
        ms.append(ledge(ln.p0, ln.p1, ln.n, 21.3, 22.0, 0.7))


def clerestory(b, ms, p0, p1, inside, nb):
    ln = Line(p0, p1, inside)
    arcade(b, ln, nb, 26.2, 28.9, 30.5, 0.2, MARBLE, pier=0.9, seg=4)
    for i in range(nb):
        if i % 2 == 0:
            rect_on(b, ln, (i + 0.5) / nb, 0.02, 26.8, 28.6, 0.6, DARK)
    ms.append(ledge(ln.p0, ln.p1, ln.n, 30.5, 31.0, 0.5))


def cathedral():
    ms, b, roof = [], B(), B()
    # platform with its step
    plat = DATA["platform"]
    ms.append(extrude(plat, -0.6, Y0 - 0.12, STEP, top_c=STEP))
    ms.append(extrude(offset(plat, -0.6), Y0 - 0.12, Y0, STEP, top_c=STEP))
    # main blocks (striped walls)
    ms.append(striped_walls(rect(XA0, XA1, ZE, ZF), -2.0, 22.0))
    ms.append(striped_walls(rect(XN0, XN1, ZE, ZF), 21.0, 31.0))
    ms.append(striped_walls(rect(TX0, TX1, TZ0, TZ1), -2.0, 22.0))
    ms.append(striped_walls(rect(TX0, TX1, TN0, TN1), 21.0, 31.0))
    # roofs (lead)
    gable_roof(roof, XN0, XN1, ZE, ZF, 31.0, 35.0, "z", LEAD, MARBLE, ends=(True, False))
    gable_roof(roof, TX0, TX1, TN0, TN1, 31.0, 35.0, "x", LEAD, MARBLE)
    lean_to(roof, XA0, XN0, ZE, ZF, 22.0, 26.0, "+x", LEAD, MARBLE)
    lean_to(roof, XN1, XA1, ZE, ZF, 22.0, 26.0, "-x", LEAD, MARBLE)
    lean_to(roof, TX0, TX1, TZ0, TN0, 22.0, 26.0, "+z", LEAD, MARBLE)
    lean_to(roof, TX0, TX1, TN1, TZ1, 22.0, 26.0, "-z", LEAD, MARBLE)
    # side walls of the body and of the transept arms
    side_orders(b, ms, (XA0, TZ1), (XA0, ZF), (CX, 80), 10)
    side_orders(b, ms, (XA1, TZ1), (XA1, ZF), (CX, 80), 10)
    side_orders(b, ms, (XA0, ZE), (XA0, TZ0), (CX, 40), 2)
    side_orders(b, ms, (XA1, ZE), (XA1, TZ0), (CX, 40), 2)
    for z, zin in ((TZ0, TZC), (TZ1, TZC)):
        side_orders(b, ms, (TX0, z), (XA0, z), (-56, zin), 3)
        side_orders(b, ms, (XA1, z), (TX1, z), (-6, zin), 3)
    for x in (XN0, XN1):
        clerestory(b, ms, (x, TZ1), (x, ZF), (CX, 80), 18)
        clerestory(b, ms, (x, ZE), (x, TZ0), (CX, 40), 2)
    for z in (TN0, TN1):
        clerestory(b, ms, (TX0, z), (XN0, z), (-50, TZC), 8)
        clerestory(b, ms, (XN1, z), (TX1, z), (-10, TZC), 8)
    # transept end walls: blind arches beside the apses
    for x, xin in ((TX0, -40.0), (TX1, -20.0)):
        for za, zb in ((TZ0, 51.6), (64.6, TZ1)):
            ln = Line((x, za), (x, zb), (xin, TZC))
            arcade(b, ln, 1, Y0, 8.6, 11.0, 0.35, MARBLE, pier=1.1)
            ms.append(ledge(ln.p0, ln.p1, ln.n, 11.0, 11.4, 0.55))
    # main apse (half round, 25 m + half cone) and the bay in front of it
    acx, acz, ar = CX, 31.55, 8.25
    a0, a1 = math.pi, 2 * math.pi
    ms.append(striped_rev(acx, acz, ar, -2.0, 25.0, a0, a1, 24))
    ms.append(revolve([(ar + 0.6, 25.0), (0.0, 30.0)], acx, acz, a0, a1, 24, LEAD))
    ms.append(striped_walls(rect(CX - 8.25, CX + 8.25, acz, ZE), -2.0, 25.0))
    gable_roof(roof, CX - 8.25, CX + 8.25, acz, ZE, 25.0, 30.0, "z", LEAD, MARBLE, ov=0.6, ends=(False, False))
    ap = Circle(acx, acz, ar, a0, a1)
    arcade(b, ap, 7, Y0, 8.6, 11.0, 0.35, MARBLE, pier=1.1)
    ms.append(revolve([(ar + 0.6, 11.0), (ar + 0.6, 11.4)], acx, acz, a0, a1, 24, MARBLE_L))
    arcade(b, ap, 7, 11.4, 17.0, 19.6, 0.3, MARBLE, pier=1.5)
    for i in range(7):
        rect_on(b, ap, (i + 0.5) / 7, 0.02, 12.6, 16.4, 0.9, DARK)
    ms.append(revolve([(ar + 0.6, 19.6), (ar + 0.6, 20.0)], acx, acz, a0, a1, 24, MARBLE_L))
    apl = Circle(acx, acz, ar + 0.2, a0, a1)
    arcade(b, apl, 14, 23.0, 23.0, 24.6, 0.45, MARBLE, pier=0.3, jambs=False, seg=4)
    ms.append(columns(apl, 14, 0.13, 20.0, 23.0, MARBLE_L, off=0.22, seg=6, closed=False, capital=False))
    ms.append(revolve([(ar + 0.5, 24.6), (ar + 0.5, 25.0)], acx, acz, a0, a1, 24, MARBLE_L))
    # transept apses
    for tcx, ta0 in ((-60.5, math.pi / 2), (-1.5, -math.pi / 2)):
        ta1 = ta0 + math.pi
        ms.append(striped_rev(tcx, TZC, 6.2, -2.0, 20.0, ta0, ta1, 20))
        ms.append(revolve([(6.7, 20.0), (0.0, 23.0)], tcx, TZC, ta0, ta1, 20, LEAD))
        tp = Circle(tcx, TZC, 6.2, ta0, ta1)
        arcade(b, tp, 5, Y0, 8.6, 11.0, 0.35, MARBLE, pier=1.1)
        ms.append(revolve([(6.75, 11.0), (6.75, 11.4)], tcx, TZC, ta0, ta1, 20, MARBLE_L))
        arcade(b, tp, 5, 11.4, 16.6, 19.4, 0.3, MARBLE, pier=1.4)
        for i in range(5):
            rect_on(b, tp, (i + 0.5) / 5, 0.02, 12.6, 16.0, 0.8, DARK)
        ms.append(revolve([(6.7, 19.4), (6.7, 20.0)], tcx, TZC, ta0, ta1, 20, MARBLE_L))
    # the elliptical dome: drum with its gallery, dome, lantern, ball and cross
    dx, dz, ax, az = DOME
    TAU = 2 * math.pi
    ms.append(striped_rev(dx, dz, 1.0, 26.0, 36.6, 0, TAU, 40, sx=ax, sz=az))
    dl = Circle(dx, dz, 1.0, 0, TAU, sx=ax + 0.55, sz=az + 0.55)
    ms.append(revolve([(1.0, 32.2), (1.0, 32.6)], dx, dz, 0, TAU, 40, MARBLE_L, sx=ax + 0.9, sz=az + 0.9))
    ms.append(revolve([(1.0, 32.6), (0.0, 32.6)], dx, dz, 0, TAU, 40, MARBLE_L, sx=ax + 0.9, sz=az + 0.9))
    ms.append(columns(dl, 28, 0.14, 32.6, 35.4, MARBLE_L, seg=6, capital=False))
    arcade(b, dl, 28, 35.4, 35.4, 36.6, 0.5, MARBLE, pier=0.3, jambs=False, seg=4, off=0.25)
    ms.append(revolve([(1.0, 36.6), (1.0, 37.0)], dx, dz, 0, TAU, 40, MARBLE_L, sx=ax + 0.9, sz=az + 0.9))
    prof = [(math.cos(t), 37.0 + 7.6 * math.sin(t)) for t in np.linspace(0, math.pi / 2 * 0.86, 9)]
    ms.append(revolve(prof, dx, dz, 0, TAU, 40, LEAD, sx=ax + 0.2, sz=az + 0.2))
    rt = prof[-1][0]
    ms.append(revolve([(rt, prof[-1][1]), (0.0, prof[-1][1])], dx, dz, 0, TAU, 40, LEAD, sx=ax + 0.2, sz=az + 0.2))
    ms.append(cyl(1.4, 1.4, 1.9, MARBLE_L, seg=12).T(dx, 44.3, dz))
    ms.append(lathe([(1.6, 0), (1.3, 0.5), (0.7, 1.0), (0.0, 1.3)], LEAD, seg=12).T(dx, 46.2, dz))
    ms.append(lathe([(0.0, -0.3), (0.3, 0.0), (0.0, 0.3)], GOLD, seg=8).T(dx, 47.8, dz))
    ms.append(box(0.12, 1.1, 0.12, IRON).T(dx, 48.0, dz))
    ms.append(box(0.12, 0.12, 0.6, IRON).T(dx, 48.7, dz))
    # façade: the screen wall with the profile of the church, four tiers of loggias, portals, statues
    fp = [(XA0, -2.0), (XA1, -2.0), (XA1, 22.0), (XN1, 26.0), (XN1, 31.0), (CX, 35.4), (XN0, 31.0), (XN0, 26.0), (XA0, 22.0)]
    scr = extrude(fp, ZF, ZF + 0.6, MARBLE, bottom=False)
    scr.v = scr.v[:, [0, 2, 1]]
    scr.f = scr.f[:, ::-1]
    ms.append(scr)
    FZ = ZF + 0.6
    lo = Line((XA0, FZ), (XA1, FZ), (CX, 80))
    arcade(b, lo, 7, Y0, 8.8, 11.8, 0.45, MARBLE, pier=1.3, seg=6)
    ms.append(columns(lo, 7, 0.32, Y0, 8.8, MARBLE_L, off=0.45, seg=8, closed=False))
    for i in range(7):
        if i in (1, 3, 5):
            w, h = (3.4, 7.4) if i == 3 else (2.6, 6.0)
            rect_on(b, lo, (i + 0.5) / 7, 0.03, Y0, Y0 + h, w, BRONZE)
            rect_on(b, lo, (i + 0.5) / 7, 0.03, Y0 + h, Y0 + h + 0.6, w + 0.6, MARBLE_L)
        diamond_on(b, lo, (i + 0.5) / 7, 0.03, 10.0, 1.2, LOZ)
    ms.append(ledge(lo.p0, lo.p1, lo.n, 11.8, 12.3, 1.6))
    tiers = [((XA0, XA1), 12.3, 17.3, 19), ((XA0, XA1), 17.8, 21.6, 19), ((XN0, XN1), 22.5, 26.6, 8),
             ((XN0 + 0.4, XN1 - 0.4), 27.1, 30.9, 8)]
    for (xa, xb), ya, yt, nt in tiers:
        ln = Line((xa, FZ), (xb, FZ), (CX, 80))
        ys = yt - 1.4
        for yy in np.arange(ya + 0.4, yt, 0.9):              # the striped wall in the shade behind the loggia
            b.poly([(xa, yy, FZ + 0.02), (xb, yy, FZ + 0.02), (xb, min(yy + 0.22, yt), FZ + 0.02),
                    (xa, min(yy + 0.22, yt), FZ + 0.02)], LOZ, np.array([0, 0, 1.0]))
        b.poly([(xa, ya, FZ + 0.01), (xb, ya, FZ + 0.01), (xb, yt, FZ + 0.01), (xa, yt, FZ + 0.01)], MARBLE_SH,
               np.array([0, 0, 1.0]))
        ms.append(columns(ln, nt, 0.16, ya, ys, MARBLE_L, off=1.15, seg=6, closed=False, capital=False))
        arcade(b, ln, nt, ys, ys, yt, 0.4, MARBLE, pier=0.34, jambs=False, seg=4, off=1.35)
        ms.append(ledge(ln.p0, ln.p1, ln.n, yt, yt + 0.5, 1.6, back=0.0))
    # sloping copings of the aisle ends and of the gable
    for (xa, ya), (xb, yb) in (((XA0, 22.5), (XN0, 26.5)), ((XA1, 22.5), (XN1, 26.5)), ((XN0, 31.4), (CX, 35.8)),
                               ((XN1, 31.4), (CX, 35.8))):
        b.poly([(xa, ya - 0.5, FZ + 1.6), (xb, yb - 0.5, FZ + 1.6), (xb, yb, FZ + 1.6), (xa, ya, FZ + 1.6)], MARBLE_L,
               np.array([0, 0, 1.0]))
        b.poly([(xa, ya, ZF), (xb, yb, ZF), (xb, yb, FZ + 1.6), (xa, ya, FZ + 1.6)], MARBLE_L, np.array([0, 1.0, 0]))
        b.poly([(xa, ya - 0.5, ZF + 0.6), (xb, yb - 0.5, ZF + 0.6), (xb, yb - 0.5, FZ + 1.6), (xa, ya - 0.5, FZ + 1.6)],
               MARBLE_L, np.array([0, -1.0, 0]))
    ms.append(figure(2.2).T(CX, 35.7, FZ + 0.6))
    for x, y in ((XA0 + 0.6, 22.3), (XA1 - 0.6, 22.3), (XN0 + 0.4, 31.3), (XN1 - 0.4, 31.3)):
        ms.append(figure(1.7).T(x, y, FZ + 0.8))
    return merge(ms + [b.mesh()]), roof.mesh()


# ── Baptistery ───────────────────────────────────────────────────────────────
BX, BZ = -28.35, 196.45


def lead_side(a):
    """True where the baptistery dome is clad in lead (bearings 37°–187°, the side facing the cathedral)."""
    b = bearing_of(math.cos(a), math.sin(a))
    return 37.0 <= b <= 187.0


def baptistery():
    ms, b, lead, tile = [], B(), [], []
    TAU = 2 * math.pi
    ae = math.atan2(-1.0, 0.0)                         # the east portal faces the cathedral (−z)
    for r, y0, y1 in ((20.0, -2.0, 0.15), (19.3, 0.15, 0.3), (18.6, 0.3, 0.45)):
        ms.append(cyl(r, r, y1 - y0, STEP, seg=72).T(BX, y0, BZ))
    # ground storey: 20 blind arches on half-columns, grey bands, four portals
    ms.append(striped_rev(BX, BZ, 16.6, 0.45, 13.0, 0, TAU, 80, period=1.3, band=0.3))
    g = Circle(BX, BZ, 17.1, ae - math.pi / 20, ae - math.pi / 20 + TAU)
    arcade(b, g, 20, 0.45, 9.8, 12.9, 0.5, MARBLE, pier=1.5, seg=6, off=0.0)
    ms.append(columns(g, 20, 0.42, 0.45, 9.8, MARBLE_L, off=0.15, seg=8))
    for i in range(20):
        if i % 5 == 0:
            p, n = g((i + 0.5) / 20)
            t = np.array([-n[1], n[0]])
            q = p + n * 0.3
            ms.append(prism_at(q, t, n, 5.6, 1.6, 0.45, 9.6, MARBLE_L))
            gable_plate(b, q + n * 0.2, n, 5.8, 9.6, 12.4, 1.2, MARBLE_L)
            rect_on(b, g, (i + 0.5) / 20, 1.12, 0.45, 6.4, 2.8, BRONZE)
            rect_on(b, g, (i + 0.5) / 20, 1.12, 6.4, 8.4, 3.4, LOZ)
        else:
            diamond_on(b, g, (i + 0.5) / 20, -0.48, 11.2, 1.4, LOZ)
    ms.append(cyl(17.45, 17.45, 0.5, MARBLE_L, seg=80).T(BX, 12.9, BZ))
    # second storey: loggia of 60 columns, 20 gables and pinnacles
    ms.append(striped_rev(BX, BZ, 16.0, 13.4, 31.8, 0, TAU, 80, period=1.3, band=0.25))
    lg = Circle(BX, BZ, 16.9, ae, ae + TAU)
    ms.append(columns(lg, 60, 0.13, 13.4, 16.9, MARBLE_L, seg=6, capital=False))
    arcade(b, lg, 60, 16.9, 16.9, 18.3, 0.4, MARBLE, pier=0.3, jambs=False, seg=4, off=0.2)
    ms.append(cyl(17.25, 17.25, 0.4, MARBLE_L, seg=80).T(BX, 18.3, BZ))
    for i in range(20):
        p, n = lg((i + 0.5) / 20)
        gable_plate(b, p + n * 0.2, n, 4.3, 18.7, 22.8, 0.35, MARBLE_L)
        p, n = lg(i / 20)
        q = p + n * 0.2
        ms.append(pinnacle(q[0], q[1], 18.7, 22.4, 0.5, MARBLE_L, tip=1.3))
    ms.append(cyl(16.5, 16.5, 2.0, MARBLE, seg=80, caps=False).T(BX, 18.7, BZ))
    # third storey: 20 windows under gables, pinnacles, cornice
    w3 = Circle(BX, BZ, 16.0, ae + math.pi / 20, ae + math.pi / 20 + TAU)
    arcade(b, w3, 20, 23.6, 27.0, 29.6, 0.3, MARBLE, pier=2.8, seg=6)
    for i in range(20):
        rect_on(b, w3, (i + 0.5) / 20, 0.02, 23.8, 27.6, 1.5, DARK)
        p, n = w3((i + 0.5) / 20)
        gable_plate(b, p + n * 1.0, n, 3.7, 29.4, 34.4, 0.3, MARBLE_L)
        p, n = w3(i / 20)
        q = p + n * 0.9
        ms.append(pinnacle(q[0], q[1], 21.0, 33.0, 0.55, MARBLE_L, tip=1.5))
    ms.append(cyl(16.9, 16.9, 0.4, MARBLE_L, seg=80).T(BX, 31.6, BZ))
    # outer dome: lead on the side of the cathedral, tiles on the side of the sea
    prof = [(16.4 * math.cos(t), 32.0 + 12.4 * math.sin(t)) for t in np.linspace(0, math.acos(5.6 / 16.4), 10)]
    SEG = 72
    for j in range(SEG):
        a0, a1 = TAU * j / SEG, TAU * (j + 1) / SEG
        (lead if lead_side((a0 + a1) / 2) else tile).append(
            revolve(prof, BX, BZ, a0, a1, 1, LEAD if lead_side((a0 + a1) / 2) else TILE))
    # the 12-sided truncated cone that rises out of it, with its ribs
    yc0 = prof[-1][1] - 0.4
    for j in range(12):
        a0, a1 = TAU * j / 12, TAU * (j + 1) / 12
        lf = lead_side((a0 + a1) / 2)
        m = revolve([(5.75, yc0), (3.95, 51.5)], BX, BZ, a0, a1, 1, LEAD if lf else TILE)
        (lead if lf else tile).append(m)
        p0 = (BX + 5.8 * math.cos(a0), yc0, BZ + 5.8 * math.sin(a0))
        p1 = (BX + 4.0 * math.cos(a0), 51.5, BZ + 4.0 * math.sin(a0))
        ms.append(tube([p0, p1], 0.12, MARBLE_L, seg=4))
    ms.append(cyl(4.05, 4.05, 0.3, LEAD_D, seg=24).T(BX, 51.4, BZ))
    lead.append(lathe([(3.7, 0), (3.5, 0.6), (2.7, 1.3), (1.4, 1.75), (0.0, 1.9)], LEAD, seg=24).T(BX, 51.7, BZ))
    ms.append(cyl(0.45, 0.35, 0.5, BRONZE, seg=8).T(BX, 53.55, BZ))
    ms.append(figure(1.3, BRONZE, r=0.22).T(BX, 54.0, BZ))
    return merge(ms + [b.mesh()]), merge(lead), merge(tile)


# ── Camposanto ───────────────────────────────────────────────────────────────
C_OUT = [(-81.92, 75.29), (-94.09, 203.44), (-138.25, 202.47), (-124.81, 68.42)]   # SE, SW, NW, NE corners
C_IN = [(-95.63, 85.64), (-104.93, 190.91), (-124.83, 189.98), (-115.28, 82.77)]
CW, CR = 10.6, 13.6              # walls, ridge


def ccw(r):
    return r if ring_area(r) > 0 else r[::-1]


def camposanto():
    ms, b, roof = [], B(), B()
    O, I = ccw(C_OUT), ccw(C_IN)
    oc = np.mean(np.array(O), 0)
    ms.append(walls(O, -2.0, CW, MARBLE))
    ms.append(walls(offset(O, -0.9), 0.0, CW, GALLERY, hole=True))
    ms.append(cap(O, 0.05, (0.62, 0.60, 0.56), holes=[offset(I, 0.9)]))
    ms.append(cap(O, CW - 0.25, WOOD, holes=[offset(I, 0.9)], up=False))
    # 43 blind arches on the south front, as many as fit on the other sides
    for k in range(4):
        p0, p1 = np.array(O[k]), np.array(O[(k + 1) % 4])
        ln = Line(p0, p1, oc)
        L = ln.L
        nb = 43 if L > 100 and np.mean([p0[0], p1[0]]) > -100 else max(4, int(round(L / 3.0)))
        arcade(b, ln, nb, 0.0, 7.4, CW - 0.4, 0.4, MARBLE, pier=0.95, seg=6)
        ms.append(ledge(ln.p0, ln.p1, ln.n, CW - 0.4, CW, 0.75))
        if nb == 43:                                   # the south front: two doors, the tabernacle over the east one
            for u, tab in ((0.18, True), (0.62, False)):
                rect_on(b, ln, u, 0.02, 0.0, 4.4, 2.2, BRONZE)
                if tab:
                    p, n = ln(u)
                    t = ln.t
                    q = p + n * 0.75
                    ms.append(prism_at(q, t, n, 3.4, 0.7, 4.6, 7.4, MARBLE_L))
                    gable_plate(b, q, n, 3.8, 7.4, 9.9, 0.7, MARBLE_L)
                    for s in (-1.9, 1.9):
                        r = q + t * s
                        ms.append(pinnacle(r[0], r[1], 4.6, 9.2, 0.35, MARBLE_L, tip=0.9))
                    rect_on(b, ln, u, 1.13, 5.0, 7.0, 1.8, DARK)
    # the courtyard front: tracery windows on mullions over a parapet
    ic = np.mean(np.array(I), 0)
    for k in range(4):
        p0, p1 = np.array(I[k]), np.array(I[(k + 1) % 4])
        ln = Line(p0, p1, oc + (oc - ic) * 50)             # normal pointing into the courtyard
        ln.n = (ic - (p0 + p1) / 2) / np.linalg.norm(ic - (p0 + p1) / 2)
        ln.n = ln.n - ln.t * np.dot(ln.n, ln.t)
        ln.n /= np.linalg.norm(ln.n)
        nb = max(5, int(round(ln.L / 4.05)))
        arcade(b, ln, nb, 1.2, 6.2, CW - 0.3, 0.9, MARBLE, pier=1.0, seg=8, off=0.0, col_in=GALLERY)
        ms.append(ledge(ln.p0, ln.p1, ln.n, CW - 0.3, CW, 0.5, back=0.9))
        ms.append(ledge(ln.p0, ln.p1, ln.n, 0.0, 1.2, 0.0, col=MARBLE, back=0.9))
        Lb = ln.L / nb
        for i in range(nb):
            for s in (-0.25, 0.0, 0.25):
                p, n = ln((i + 0.5 + s * (Lb - 1.0) / Lb) / nb)
                q = p - n * 0.45
                ms.append(prism_at(q, ln.t, n, 0.16, 0.25, 1.2, 6.2 + (Lb - 1.0) / 2 * 0.72, MARBLE_L))
    ms.append(cap(I, 0.05, PAVING))
    # gallery roofs: lead, hipped at the corners
    E0, R, E1 = offset(O, 0.8), [], offset(I, 0.6)
    mid = offset(O, -5.6)
    R = [((np.array(E0[k]) + np.array(E1[k])) / 2 + np.array(mid[k])) / 2 for k in range(4)]
    for k in range(4):
        j = (k + 1) % 4
        y_e = CW - 0.15
        P = [(E0[k][0], y_e, E0[k][1]), (E0[j][0], y_e, E0[j][1]), (R[j][0], CR, R[j][1]), (R[k][0], CR, R[k][1])]
        mo = (np.array(E0[k]) + np.array(E0[j])) / 2 - (np.array(R[k]) + np.array(R[j])) / 2
        roof.poly(P, LEAD, np.array([mo[0], 3, mo[1]]))
        P = [(R[k][0], CR, R[k][1]), (R[j][0], CR, R[j][1]), (E1[j][0], y_e, E1[j][1]), (E1[k][0], y_e, E1[k][1])]
        mi = (np.array(E1[k]) + np.array(E1[j])) / 2 - (np.array(R[k]) + np.array(R[j])) / 2
        roof.poly(P, LEAD, np.array([mi[0], 3, mi[1]]))
    # Cappella Dal Pozzo with its small dome (east end)
    pz = DATA["pozzo"]
    ms.append(extrude(pz, -2.0, 11.2, MARBLE, top_c=LEAD_D))
    pc = np.mean(np.array(pz), 0)
    ms.append(cyl(3.3, 3.3, 2.4, MARBLE, seg=16).T(pc[0], 11.2, pc[1]))
    ms.append(lathe([(3.45, 0), (3.2, 1.2), (2.3, 2.4), (1.0, 3.1), (0.0, 3.3)], LEAD, seg=16).T(pc[0], 13.6, pc[1]))
    ms.append(cyl(0.6, 0.6, 1.0, MARBLE_L, seg=8).T(pc[0], 16.8, pc[1]))
    ms.append(cyl(0.75, 0.0, 0.7, LEAD, seg=8).T(pc[0], 17.8, pc[1]))
    return merge(ms + [b.mesh()]), roof.mesh()


# ── The square ───────────────────────────────────────────────────────────────
def square():
    ms = [extrude(FOOTPRINT, -0.4, 0.02, PAVING, top_c=PAVING)]
    for k, ring in DATA["lawns"].items():
        ms.append(extrude(ring, 0.0, 0.12, GRASS_E, top_c=GRASS))
    return merge(ms)


def tower_fence():
    ms = []
    for k in range(64):
        a = 2 * math.pi * k / 64
        ms.append(box(0.08, 1.0, 0.08, IRON).T(12.4 * math.cos(a), 0.0, 12.4 * math.sin(a)))
    ring = [(12.4 * math.cos(2 * math.pi * k / 64), 1.0, 12.4 * math.sin(2 * math.pi * k / 64)) for k in range(65)]
    ms.append(tube(ring, 0.04, IRON, seg=4))
    return merge(ms)


def main():
    out = ROOT / "models" / "Q39054"
    out.mkdir(parents=True, exist_ok=True)
    tw, tw_iron = tower()
    cat, cat_roof = cathedral()
    bap, bap_lead, bap_tile = baptistery()
    cam, cam_roof = camposanto()
    sq = square()
    fence = tower_fence()
    LEADM = {"metallic": 0.45, "roughness": 0.55}
    IRONM = {"metallic": 0.6, "roughness": 0.5}
    parts = [("torre_pendente", tw), ("torre_ringhiere", tw_iron, {"material": "iron"}),
             ("duomo", cat), ("duomo_tetti", cat_roof, {"material": "lead"}),
             ("battistero", bap), ("battistero_cupola_piombo", bap_lead, {"material": "lead"}),
             ("battistero_cupola_tegole", bap_tile),
             ("camposanto", cam), ("camposanto_tetti", cam_roof, {"material": "lead"}),
             ("prato", sq), ("recinto_torre", fence, {"material": "iron"})]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.85, materials={"lead": LEADM, "iron": IRONM})
    allv = np.vstack([p[1].v for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    top = tw.v[np.argmax(tw.v[:, 1])]
    print("tower top", np.round(top, 2), "lean offset of the top centre",
          np.round(lean(Mesh(np.array([[0, TOWER_H, 0]]), np.zeros((0, 3)))).v, 2))
    meta = out / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        m["triangles"] = int(tris)
        m["footprint"] = [[float(x), float(z)] for x, z in FOOTPRINT]
        meta.write_bytes((json.dumps(m, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
