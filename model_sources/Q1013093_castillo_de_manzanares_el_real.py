"""
Castillo nuevo de Manzanares el Real (Q1013093), the castle of the Mendoza, Manzanares el Real (Madrid) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: begun in 1475 for Diego Hurtado de Mendoza, first Duke of the Infantado, on the site of a
Romanesque-Mudéjar church whose apse it kept; the Isabelline Gothic upper works (the southern gallery, the
pearl-studded "caballeros" on the towers) are by Juan Guas (died 1496). Granite. Restored 1965–1975.
Public domain.

What is modelled:
- The square main body (≈32 × 34 m) around its small porticoed courtyard, with flat roof terraces; the
  corbelled machicolation band and the crenellated parapet all round; the three round corner towers (west,
  north and south corners) each with its band of machicolation, crenellated parapet and a narrower round
  "caballero" studded with granite balls, its decorated frieze and merlons; the square keep (Torre del Homenaje)
  at the east-south corner, with its arcaded gallery and its octagonal ball-studded caballero; the southern
  gallery of Juan Guas (an arcaded loggia on top of the south front, under a tile roof); hanging matacanes in
  the middle of the fronts; small windows; the castle door on the west front.
- The eastern body: the thick walls of the ruined upper floors over the chapel (the old church, roofed here
  with a flat cover at the level of its vaults) and the old Romanesque-Mudéjar apse as a round tower.
- The barrier (barrera): the low crenellated outer wall with round bastions concentric with the towers and a
  bulge round the apse, its gate on the west side between two small round turrets with corbelled crowns; the
  liza (the level ground between barrier and castle) as a paved/earth floor; outside, the barrier stands on a
  short earth bank (terraplén) that falls with the knoll.
- Not modelled: interiors, the courtyard arcades in detail, coats of arms, the tracery itself (the openings are
  plain dark arches), the gardens around.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap relation 1266539 (outer way 84686467, courtyard way 84686469), the eastern body way
  84686470 and the barrier way 55261701 (barrier=city_wall) (© OpenStreetMap contributors, ODbL). Round towers
  fitted to the OSM arcs: r ≈ 4.0–4.5 m; barrier bastions r ≈ 9.0 m round the same centres; apse r ≈ 4.8 m.
- Heights: no published figures; measured on Wikimedia Commons photographs, with the 8–9 m diameter of the
  towers (OSM) as the scale (±10–15 %): curtain machicolation 16.6–18.2 m, parapet to 19.2 m, merlons to 20.2 m;
  tower machicolation 18.0–20.0 m, parapet to 21.2 m; tower caballeros r ≈ 2.9 m up to 27.6 m (merlons to
  28.7 m); keep to 22.4 m, its gallery to 25.6 m, octagonal caballero to 32.4 m (merlons to 33.5 m). Barrier
  about 5 m high outside (es.wikipedia: "unos cinco metros"), its merlons 3.3 m above the liza.
- Ground: EU-DEM 25 m (Copernicus, via OpenTopoData) on a 20 m grid round the castle: 921 m at the origin,
  the knoll falls to the south and west (≈916–918 m at 40 m). y = 0 is the liza (≈ the ground at the origin);
  the foot of the barrier outside is the DEM level minus 1.8 m (the bank), the earth bank falls a further
  1.5 m over 6 m.

Axes (docs/MODEL_SPEC.md): +Y up, the south front (with the gallery of Juan Guas, towards the Santillana
reservoir) faces +Z (heading 156.1°, perpendicular to the long axis of the OSM outline), +X points to 66.1°
(the eastern body and the apse). Origin at the centroid of OSM relation 1266539.

    python model_sources/Q1013093_castillo_de_manzanares_el_real.py   -> models/Q1013093/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, cap, walls, extrude, offset, ring_area, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(1475)
HEADING = 156.1

GRANITE = (0.80, 0.70, 0.53)       # warm golden granite rubble of the walls
GRANITE_L = (0.86, 0.80, 0.68)     # ashlar: machicolation, friezes, balls, copings
GRANITE_D = (0.66, 0.58, 0.45)     # foundations, barrier outside
PAVING = (0.70, 0.65, 0.56)        # roof terraces
LIZA = (0.72, 0.67, 0.55)          # the liza: gravel and earth
BANK = (0.55, 0.52, 0.37)          # the earth bank outside the barrier (dry grass)
TILE = (0.62, 0.38, 0.27)          # tile roof of the gallery
COVER = (0.58, 0.55, 0.50)         # flat cover of the chapel
BRICK = (0.74, 0.56, 0.42)         # Mudejar brick of the old apse
DARK = (0.10, 0.09, 0.08)          # openings
DOOR = (0.28, 0.20, 0.13)

# ── Data (OpenStreetMap © OpenStreetMap contributors, ODbL), metres in the model axes ──────────────────
D = json.loads(r"""{"84686467": [[-16.07, 14.46], [-17.0, 14.51], [-18.14, 15.11], [-19.21, 16.45], [-19.61, 18.13], [-19.58, 19.12], [-18.92, 20.47], [-17.6, 21.51], [-16.26, 22.13], [-15.33, 22.29], [-14.18, 22.02], [-13.19, 21.57], [-12.37, 20.77], [-11.9, 20.02], [-11.4, 18.82], [-11.56, 17.51], [4.46, 17.19], [4.69, 19.71], [13.83, 19.61], [13.64, 15.36], [16.49, 15.23], [16.47, 10.44], [16.51, 7.02], [16.25, -6.05], [16.39, -10.09], [16.49, -12.55], [17.96, -12.63], [18.79, -13.08], [19.71, -13.63], [20.46, -15.04], [20.73, -16.29], [20.45, -17.91], [19.81, -18.98], [18.75, -19.99], [17.63, -20.63], [16.17, -21.02], [14.39, -20.89], [12.86, -20.09], [11.96, -18.61], [11.66, -16.89], [-11.78, -16.98], [-11.89, -17.71], [-12.37, -18.72], [-13.78, -19.76], [-15.54, -20.34], [-16.84, -20.23], [-17.78, -19.67], [-18.81, -18.75], [-19.48, -17.37], [-19.62, -15.73], [-19.17, -14.45], [-18.08, -13.03], [-16.84, -12.51], [-15.42, -12.3]], "84686469": [[-8.35, -2.19], [1.62, -2.37], [1.81, 6.82], [-8.16, 6.99]], "55261701": [[-22.74, -22.8], [-24.23, -21.11], [-25.15, -18.99], [-25.49, -16.23], [-25.14, -13.74], [-23.87, -11.23], [-22.49, -9.64], [-20.96, -8.48], [-21.0, -4.84], [-21.05, 0.68], [-21.04, 6.64], [-21.04, 10.04], [-21.72, 10.62], [-23.42, 12.35], [-24.7, 14.96], [-25.2, 18.01], [-24.85, 20.4], [-23.38, 22.98], [-22.28, 24.32], [-20.44, 25.62], [-17.45, 26.69], [-14.59, 26.67], [-11.95, 25.76], [-10.04, 24.32], [-8.33, 22.16], [-4.58, 22.19], [-4.48, 25.46], [20.21, 25.63], [20.3, 14.57], [40.32, 14.74], [40.38, 9.33], [41.57, 8.69], [43.03, 7.52], [43.99, 6.34], [45.03, 4.71], [45.67, 3.11], [46.12, 0.99], [46.22, -0.58], [45.82, -2.73], [45.17, -4.75], [43.96, -6.66], [42.62, -8.01], [41.66, -8.92], [40.54, -9.4], [40.67, -16.53], [24.29, -16.75], [24.11, -18.89], [23.44, -20.64], [21.89, -22.86], [19.21, -24.86], [15.88, -25.77], [12.41, -25.3], [9.49, -23.75], [7.78, -21.78], [-9.49, -22.02], [-10.89, -23.48], [-13.32, -24.88], [-16.08, -25.39], [-18.51, -25.22], [-21.02, -24.34]], "84686470": [[16.39, -10.09], [35.26, -10.37], [35.77, -4.7], [37.38, -4.63], [39.79, -3.99], [41.1, -3.07], [42.21, -1.74], [42.64, 0.34], [42.11, 2.06], [41.01, 3.39], [39.44, 4.42], [37.7, 4.7], [35.79, 4.69], [35.5, 10.58], [16.47, 10.44], [16.51, 7.02], [30.57, 7.01], [30.81, -5.85], [16.25, -6.05]]}""")

# EU-DEM 25 m (Copernicus, via OpenTopoData), metres above sea level on a 20 m grid: rows north (+80 m) to
# south (−80 m), columns west (−80 m) to east (+80 m) of the origin.
DEM = np.array([
    [924.0, 924.1, 924.2, 924.1, 923.6, 922.6, 921.0, 919.0, 917.2],
    [922.2, 922.4, 922.6, 923.0, 923.0, 922.4, 921.0, 919.4, 917.8],
    [920.5, 920.5, 921.0, 921.8, 922.2, 922.1, 921.3, 920.1, 918.6],
    [918.7, 918.9, 919.6, 920.5, 921.4, 921.9, 921.6, 920.8, 919.4],
    [917.0, 917.5, 918.5, 919.7, 921.1, 921.9, 921.8, 921.1, 919.6],
    [915.1, 916.0, 917.1, 918.6, 920.2, 921.4, 921.6, 920.9, 919.3],
    [913.3, 914.2, 915.5, 917.3, 919.1, 920.2, 920.7, 920.2, 918.4],
    [911.8, 912.4, 913.7, 915.8, 917.7, 918.9, 919.2, 918.5, 916.8],
    [910.3, 910.8, 912.1, 914.1, 915.9, 917.0, 917.3, 916.7, 914.9]])


def dem(x, z):
    """Ground (m above the origin's ground) at a model point, bilinear on the DEM grid."""
    h = math.radians(HEADING)
    e = -x * math.cos(h) + z * math.sin(h)
    n = x * math.sin(h) + z * math.cos(h)
    i, j = (e + 80) / 20, (80 - n) / 20
    i0, j0 = int(min(max(math.floor(i), 0), 7)), int(min(max(math.floor(j), 0), 7))
    u, v = min(max(i - i0, 0), 1), min(max(j - j0, 0), 1)
    g = (DEM[j0, i0] * (1 - u) * (1 - v) + DEM[j0, i0 + 1] * u * (1 - v) + DEM[j0 + 1, i0] * (1 - u) * v
         + DEM[j0 + 1, i0 + 1] * u * v)
    return g - DEM[4, 4]


def R(i):
    return [tuple(p) for p in D[str(i)]]


def ccw(P):
    P = [tuple(p) for p in P]
    return P if ring_area(P) > 0 else P[::-1]


def fit_circle(P):
    P = np.asarray(P, float)
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(c[0]), float(c[1]), float(math.sqrt(c[2] + c[0] ** 2 + c[1] ** 2))


def line_x(a, b, c, d):
    """Intersection of the lines ab and cd."""
    a, b, c, d = (np.asarray(p, float) for p in (a, b, c, d))
    r, s = b - a, d - c
    cr = lambda u, v: u[0] * v[1] - u[1] * v[0]  # noqa: E731
    t = cr(c - a, s) / cr(r, s)
    return tuple(a + r * t)


def flip_up(m):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    down = np.cross(b - a, c - a)[:, 1] < 0
    m.f[down] = m.f[down][:, ::-1]
    return m


def sphere(r, c, seg=8, rings=4):
    prof = [(r * math.sin(math.pi * k / rings), -r * math.cos(math.pi * k / rings)) for k in range(rings + 1)]
    return lathe(prof, c, seg)


# ── Plan ─────────────────────────────────────────────────────────────────────
OUT = R(84686467)
TOWERS = {"south": fit_circle(OUT[0:16]), "north": fit_circle(OUT[25:40]), "west": fit_circle(OUT[40:54])}
# the main body: the straight walls, closed through the tower centres
SW = line_x(OUT[53], OUT[0], OUT[15], OUT[16])
NW = line_x(OUT[53], OUT[0], OUT[39], OUT[40])
NE = line_x(OUT[39], OUT[40], OUT[24], OUT[25])
BODY = ccw([SW, OUT[16], OUT[17], OUT[18], OUT[19], OUT[20], OUT[21], OUT[22], OUT[23], OUT[24], NE, NW])
COURT = ccw(R(84686469))
KEEP = ccw([(4.46, 7.0), (16.5, 7.0), OUT[20], OUT[19], OUT[18], OUT[17], OUT[16]])
KEEP_C = (9.6, 13.3)                 # centre of the octagonal caballero on the keep
EAST = R(84686470)
EAST_C = ccw(EAST[0:2] + [(35.77, -4.7), (35.79, 4.69)] + EAST[13:])      # without the apse
APSE = ccw([(35.6, -4.7)] + EAST[3:12] + [(35.6, 4.69)])
CHAPEL = ccw([(16.4, -6.05), (30.81, -5.85), (30.57, 7.01), (16.4, 7.02)])
BARRIER = ccw(R(55261701))

FOUND = -4.0           # foundations of the castle below the liza
WALL_TOP = 16.6        # curtain walls up to the machicolation
MACH = 1.6             # machicolation band
PAR = 1.0              # solid parapet
MER = 1.0              # merlons
T_MACH, T_PAR = 18.0, 21.2       # round towers: machicolation from, parapet top
CAB_R, CAB_TOP = 2.9, 27.6       # tower caballeros
K_TOP, K_GAL, K_OCT_R, K_OCT_TOP = 22.4, 25.6, 4.3, 32.4

OCC = []


def inside(x, z, P):
    ins = False
    n = len(P)
    for i in range(n):
        (xi, zi), (xj, zj) = P[i], P[i - 1]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            ins = not ins
    return ins


# ── Pieces on a wall: local (u along the wall, y up, w outwards) ────────────
class Face:
    def __init__(self, a, t, n):
        self.a, self.t, self.n = np.asarray(a, float), np.asarray(t, float), np.asarray(n, float)

    def P(self, u, y, w):
        p = self.a + self.t * u + self.n * w
        return (p[0], y, p[1])

    def poly(self, pts, c, w=0.05):
        V = [self.P(u, y, w) for u, y in pts]
        F = [(0, i, i + 1) for i in range(1, len(V) - 1)]
        m = Mesh(V, F, c)
        a, b, cc = np.array(V[0]), np.array(V[1]), np.array(V[2])
        nn = np.cross(b - a, cc - a)
        if np.dot([nn[0], nn[2]], self.n) < 0:
            m.f = m.f[:, ::-1]
        return m

    def slab(self, u0, u1, y0, y1, w0, w1, c):
        P = [self.P(u, y, w) for w in (w0, w1) for y in (y0, y1) for u in (u0, u1)]
        quads = [(4, 5, 7, 6), (1, 0, 2, 3), (0, 4, 6, 2), (5, 1, 3, 7), (6, 7, 3, 2), (0, 1, 5, 4)]
        V, F = [], []
        for q in quads:
            o = len(V); V += [P[k] for k in q]; F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        m = Mesh(V, F, c)
        ctr = np.mean(np.array(P), 0)
        for k in range(0, len(F), 2):
            a, b, cc = (np.array(V[i]) for i in F[k])
            if np.dot(np.cross(b - a, cc - a), (a + b + cc) / 3 - ctr) < 0:
                m.f[k] = m.f[k][::-1]; m.f[k + 1] = m.f[k + 1][::-1]
        return m

    def pyramid(self, u0, u1, y, w0, w1, h, c):
        P = [self.P(u0, y, w0), self.P(u1, y, w0), self.P(u1, y, w1), self.P(u0, y, w1)]
        top = self.P((u0 + u1) / 2, y + h, (w0 + w1) / 2)
        V = P + [top]
        return flip_up(Mesh(V, [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)], c))


def edges(P):
    P = ccw(P)
    for i in range(len(P)):
        a, b = np.array(P[i], float), np.array(P[(i + 1) % len(P)], float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-3:
            continue
        t = (b - a) / L
        yield a, b, L, t, np.array([t[1], -t[0]])          # outward normal (ccw ring)


def arch_pts(u, y0, ys, half, seg=6):
    pts = [(u - half, y0), (u + half, y0)]
    pts += [(u + half * math.cos(a), ys + half * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)[1:-1]]
    return pts


def merlons_poly(P, y, w_out=0.0, depth=0.7, h=MER, step=2.2, mw=1.1, c=GRANITE, caps=False, skip=None):
    """Merlons along every edge of the ring P (outer face of the merlon on the ring, w_out further out)."""
    out = []
    for a, b, L, t, n in edges(P):
        if skip and skip(a, b):
            continue
        f = Face(a, t, n)
        k = int(L / step)
        if k == 0:
            if L < mw + 0.2:
                continue
            k = 1
        for j in range(k):
            u = (j + 0.5) * L / k
            out.append(f.slab(u - mw / 2, u + mw / 2, y, y + h, w_out - depth, w_out, c))
            if caps:
                out.append(f.pyramid(u - mw / 2, u + mw / 2, y + h, w_out - depth, w_out, 0.45, c))
    return out


def merlons_circle(cx, cz, r, y, step=2.0, mw=1.0, depth=0.6, h=MER, c=GRANITE):
    """Merlons round a circle of radius r (their outer face on it)."""
    n = max(4, int(round(2 * math.pi * r / step)))
    out = []
    for k in range(n):
        th = 2 * math.pi * (k + 0.5) / n
        nn = np.array([math.cos(th), math.sin(th)])
        f = Face(np.array([cx, cz]) + nn * r, np.array([-nn[1], nn[0]]), nn)
        out.append(f.slab(-mw / 2, mw / 2, y, y + h, -depth, 0.0, c))
    return out


def parapet_ring(P, y0, y1, th, c=GRANITE):
    """A solid parapet whose outer face is the ring P, th thick, from y0 to y1."""
    I = offset(P, -th)
    return merge([walls(P, y0, y1, c), walls(I, y0, y1, c, hole=True), cap(P, y1, GRANITE_L, holes=[I])])


def band(P, y0, h, d, c=GRANITE_L):
    """The corbelled machicolation: a ring d out from P, tapering in below (the corbels), h tall."""
    O = offset(P, d)
    return merge([walls(O, y0 + 0.55, y0 + h, c), cap(O, y0 + h, c, holes=[P]),
                  taper(P, O, y0, y0 + 0.55, GRANITE_D)])


def taper(Pin, Pout, y0, y1, c):
    """Sloping faces from ring Pin at y0 out to ring Pout at y1 (same vertex count), facing down-out."""
    Pin, Pout = ccw(Pin), ccw(Pout)
    V, F = [], []
    n = len(Pin)
    for i in range(n):
        j = (i + 1) % n
        o = len(V)
        V += [(Pin[i][0], y0, Pin[i][1]), (Pin[j][0], y0, Pin[j][1]), (Pout[j][0], y1, Pout[j][1]), (Pout[i][0], y1, Pout[i][1])]
        F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    return Mesh(V, F, c)


def circle(cx, cz, r, n=28):
    return [(cx + r * math.cos(2 * math.pi * k / n), cz + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def windows(P, rows, spacing, only=None, w=0.9, h=1.4):
    out = []
    for a, b, L, t, n in edges(P):
        if L < 4 or (only and not only(a, b, n)):
            continue
        f = Face(a, t, n)
        k = max(1, int((L - 3) // spacing))
        for y in rows:
            for j in range(k):
                u = L / 2 + (j - (k - 1) / 2) * spacing + rng.uniform(-0.6, 0.6)
                p = a + t * u + n * 0.8
                if any(inside(p[0], p[1], Q) for Q in OCC):
                    continue
                out.append(f.poly([(u - w / 2, y), (u + w / 2, y), (u + w / 2, y + h), (u - w / 2, y + h)], DARK, 0.04))
    return out


def matacan(f, u, y):
    """A hanging bartizan in the middle of a front: corbelled box with a pointed base."""
    out = [f.slab(u - 1.3, u + 1.3, y, y + 2.2, -0.1, 1.2, GRANITE_L),
           f.slab(u - 1.0, u + 1.0, y - 1.4, y, 0.0, 0.9, GRANITE_D)]
    return out


# ── The castle ───────────────────────────────────────────────────────────────
def main_body():
    out = [extrude(BODY, FOUND, WALL_TOP, GRANITE, holes=[COURT], top_c=PAVING)]
    out.append(band(BODY, WALL_TOP, MACH, 0.6))
    top = WALL_TOP + MACH
    BO = offset(BODY, 0.6)
    gal = lambda a, b: abs(a[1] - 17.3) < 0.6 and abs(b[1] - 17.3) < 0.6        # noqa: E731  south front
    out.append(parapet_ring(BO, top, top + PAR, 0.6))
    out += merlons_poly(BO, top + PAR, step=2.3, skip=gal)
    # courtyard: walls a little darker, floor, two storeys of arcades as dark bands
    out.append(cap(COURT, 0.3, GRANITE_D))
    for a, b, L, t, n in edges(COURT):
        f = Face(a, -t, -n)                                  # facing into the courtyard
        k = max(2, int(L / 2.6))
        for j in range(k):
            u = -(j + 0.5) * L / k
            hw = L / k * 0.36
            out.append(f.poly(arch_pts(u, 0.3, 3.4, hw, 5), DARK, 0.05))
            out.append(f.poly(arch_pts(u, 5.4, 7.9, hw, 5), DARK, 0.05))
    out.append(merge(windows(BODY, [9.5, 13.4], 7.5)))
    return out


def south_gallery():
    """The gallery of Juan Guas: an arcade on top of the south front, a tile lean-to roof over it."""
    a, b = np.array(OUT[15], float), np.array(OUT[16], float)
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([t[1], -t[0]])
    if n[1] < 0:
        n = -n
    a = a + n * 0.3
    L = float(np.linalg.norm(b - a))
    f = Face(a, t, n)
    y0 = WALL_TOP + MACH
    out = [f.slab(0.3, L, y0, y0 + 1.0, -0.4, 0.3, GRANITE_L)]                   # parapet with tracery
    k = int(L / 2.6)
    for j in range(k + 1):
        u = 0.3 + j * (L - 0.3) / k
        out.append(f.slab(u - 0.25, u + 0.25, y0 + 1.0, y0 + 3.4, -0.3, 0.2, GRANITE_L))    # piers
    out.append(f.slab(0.3, L, y0 + 3.4, y0 + 4.0, -0.4, 0.3, GRANITE_L))                # lintel
    out.append(f.slab(0.3, L, y0 + 1.0, y0 + 3.4, -3.0, -2.8, DARK))                    # the shadow inside
    for j in range(k):
        u = 0.3 + (j + 0.5) * (L - 0.3) / k
        out.append(f.poly(arch_pts(u, y0 + 0.4, y0 + 0.75, 0.9, 4), DARK, 0.32))        # tracery openings
    # lean-to tile roof
    V = [f.P(0.3, y0 + 4.0, 0.5), f.P(L, y0 + 4.0, 0.5), f.P(L, y0 + 5.0, -3.2), f.P(0.3, y0 + 5.0, -3.2)]
    out.append(flip_up(Mesh(V, [(0, 1, 2), (0, 2, 3)], TILE)))
    return out


def round_tower(name):
    cx, cz, r = TOWERS[name]
    P = circle(cx, cz, r)
    OCC.append(circle(cx, cz, r + 0.5, 12))
    out = [cyl(r, r, T_MACH - FOUND, GRANITE, 28, caps=False).T(cx, FOUND, cz)]
    out.append(band(P, T_MACH, 2.0, 0.65))
    PO = circle(cx, cz, r + 0.65)
    out.append(parapet_ring(PO, T_MACH + 2.0, T_PAR, 0.6))
    out.append(cap(circle(cx, cz, r + 0.1), T_MACH + 2.0, PAVING))
    out += merlons_circle(cx, cz, r + 0.65, T_PAR)
    # the caballero: a narrower drum studded with balls, a frieze and merlons
    out.append(cyl(CAB_R, CAB_R, CAB_TOP - T_MACH - 2.0, GRANITE, 24).T(cx, T_MACH + 2.0, cz))
    out.append(band(circle(cx, cz, CAB_R, 24), CAB_TOP - 1.6, 1.4, 0.35))
    PC = circle(cx, cz, CAB_R + 0.35, 24)
    out += merlons_circle(cx, cz, CAB_R + 0.35, CAB_TOP - 0.2, step=1.6, mw=0.9, depth=0.55, h=1.1)
    out += balls_ring(cx, cz, CAB_R, [T_PAR + 0.6, T_PAR + 1.6, T_PAR + 2.6, T_PAR + 3.6, T_PAR + 4.5], 12)
    out += balls_ring(cx, cz, r, [T_MACH - 1.2], 16)
    for k in range(3):
        th = rng.uniform(0, 2 * math.pi)
        nn = np.array([math.cos(th), math.sin(th)])
        f = Face(np.array([cx, cz]) + nn * r, np.array([-nn[1], nn[0]]), nn)
        y = 6 + 4 * k
        out.append(f.poly([(-0.35, y), (0.35, y), (0.35, y + 1.1), (-0.35, y + 1.1)], DARK, 0.06))
    return out


def balls_ring(cx, cz, r, ys, n):
    out = []
    for i, y in enumerate(ys):
        for k in range(n):
            th = 2 * math.pi * (k + 0.5 * (i % 2)) / n
            out.append(sphere(0.32, GRANITE_L).T(cx + (r + 0.12) * math.cos(th), y, cz + (r + 0.12) * math.sin(th)))
    return out


def keep():
    OCC.append(KEEP)
    out = [extrude(KEEP, FOUND, K_TOP - 2.0, GRANITE, top_c=PAVING)]
    out.append(band(KEEP, K_TOP - 2.0, 2.0, 0.7))
    KO = offset(KEEP, 0.7)
    # the arcaded gallery round the top of the keep, under a flat roof
    out.append(extrude(offset(KEEP, -0.6), K_TOP, K_GAL, GRANITE_D, top_c=PAVING))
    for a, b, L, t, n in edges(KO):
        f = Face(a, t, n)
        k = max(1, int(L / 2.4))
        out.append(f.slab(0, L, K_TOP, K_TOP + 0.9, -0.6, 0.0, GRANITE_L))
        for j in range(k + 1):
            u = j * L / k
            out.append(f.slab(u - 0.28, u + 0.28, K_TOP + 0.9, K_GAL - 0.6, -0.6, 0.0, GRANITE_L))
        for j in range(k):
            u = (j + 0.5) * L / k
            out.append(f.poly(arch_pts(u, K_TOP + 0.9, K_GAL - 1.2, L / k / 2 - 0.3, 4), DARK, -0.9))
    out.append(extrude(KO, K_GAL - 0.6, K_GAL, GRANITE_L, top_c=PAVING))
    # the octagonal caballero
    cx, cz = KEEP_C
    O8 = [(cx + K_OCT_R * math.cos(math.pi / 8 + k * math.pi / 4), cz + K_OCT_R * math.sin(math.pi / 8 + k * math.pi / 4))
          for k in range(8)]
    out.append(extrude(O8, K_GAL, K_OCT_TOP - 1.6, GRANITE, top_c=PAVING))
    out.append(band(O8, K_OCT_TOP - 1.6, 1.6, 0.45))
    O8o = offset(O8, 0.45)
    out.append(parapet_ring(O8o, K_OCT_TOP, K_OCT_TOP + 0.2, 0.5))
    out += merlons_poly(O8o, K_OCT_TOP + 0.2, step=1.6, mw=1.0, depth=0.5, h=1.1)
    for a, b, L, t, n in edges(O8):
        f = Face(a, t, n)
        for i, y in enumerate(np.arange(K_GAL + 0.9, K_OCT_TOP - 2.0, 1.0)):
            for u in np.arange(0.45 + 0.4 * (i % 2), L - 0.3, 0.85):
                p = f.P(u, y, 0.12)
                out.append(sphere(0.3, GRANITE_L).T(*p))
    # a window of the keep with tracery (the one on the south front)
    for a, b, L, t, n in edges(KEEP):
        if n[1] > 0.9:
            f = Face(a, t, n)
            out.append(f.poly(arch_pts(L / 2, 12.0, 13.6, 0.9, 6), DARK, 0.05))
            out.append(f.poly([(L / 2 - 0.5, 6.0), (L / 2 + 0.5, 6.0), (L / 2 + 0.5, 7.2), (L / 2 - 0.5, 7.2)], DARK, 0.05))
    return out


def east_body():
    """The ruined upper floors of the eastern body (thick walls, broken tops), the chapel cover, the old apse."""
    E = ccw(EAST_C)
    out = [extrude(E, FOUND, 12.0, GRANITE, top_c=GRANITE_D)]
    # broken stubs of the ruined upper floors
    for m in merlons_poly(E, 12.0, depth=1.4, h=1.0, step=3.6, mw=1.8, c=GRANITE):
        k = rng.uniform(0.3, 2.2)
        m.v[:, 1] = np.where(m.v[:, 1] > 12.5, 12.0 + k, m.v[:, 1])
        out.append(m)
    out.append(cap(CHAPEL, 8.0, COVER))
    out.append(merge(windows(E, [8.5], 6.0, only=lambda a, b, n: abs(n[0]) > 0.5 or abs(n[1]) > 0.5)))
    # the apse: Romanesque-Mudejar, a round tower of brick and granite
    out.append(extrude(APSE, FOUND, 10.5, BRICK, top_c=PAVING))
    out.append(band(APSE, 10.5, 1.0, 0.35))
    AO = offset(APSE, 0.35)
    out += merlons_poly(AO, 11.5, step=2.0, mw=1.0, depth=0.5)
    return out


# ── The barrier, the liza and the bank ───────────────────────────────────────
B_TOP = 2.2          # top of the barrier wall above the liza (the merlons stand on it)
B_TH = 1.8


def barrier():
    P = ccw(BARRIER)
    PI = offset(P, -B_TH)
    yo = [min(-1.8, dem(x, z) - 1.8) for x, z in P]
    out = []
    # outer face from the bank down to foundations, inner face, top
    V, F = [], []
    for i in range(len(P)):
        j = (i + 1) % len(P)
        a, b = P[i], P[j]
        o = len(V)
        V += [(a[0], yo[i] - 2.5, a[1]), (b[0], yo[j] - 2.5, b[1]), (b[0], B_TOP, b[1]), (a[0], B_TOP, a[1])]
        F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    out.append(Mesh(V, F, GRANITE))
    out.append(walls(PI, -0.5, B_TOP, GRANITE, hole=True))
    out.append(cap(P, B_TOP, GRANITE_L, holes=[PI]))
    gate = lambda a, b: abs(a[0] + 21.0) < 0.3 and abs(b[0] + 21.0) < 0.3 and -4 < (a[1] + b[1]) / 2 < 6   # noqa: E731
    out += merlons_poly(P, B_TOP, depth=0.8, h=1.1, step=2.8, mw=1.3, caps=True, skip=gate)
    # the liza
    out.append(cap(PI, 0.0, LIZA))
    # the earth bank outside
    PB = offset(P, 6.0)
    Vb, Fb = [], []
    n = len(P)
    for i in range(n):
        j = (i + 1) % n
        o = len(Vb)
        Vb += [(P[i][0], yo[i], P[i][1]), (P[j][0], yo[j], P[j][1]),
               (PB[j][0], yo[j] - 1.5, PB[j][1]), (PB[i][0], yo[i] - 1.5, PB[i][1])]
        Fb += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    out.append(flip_up(Mesh(Vb, Fb, BANK)))
    # the gate on the west side between two round turrets with corbelled crowns
    gz = 1.0
    f = Face(np.array([-21.03, gz]), np.array([0.0, -1.0]), np.array([-1.0, 0.0]))
    out.append(f.poly(arch_pts(0.0, min(-1.8, dem(-21, gz) - 1.8), 0.9, 1.4, 6), DOOR, 0.06))
    for du in (-2.7, 2.7):
        cxz = f.a + f.t * du + f.n * 0.6
        y0 = min(-1.8, dem(cxz[0], cxz[1]) - 1.8) - 2.0
        out.append(cyl(1.5, 1.5, 4.2 - y0, GRANITE, 16).T(cxz[0], y0, cxz[1]))
        C = circle(cxz[0], cxz[1], 1.5, 16)
        out.append(band(C, 4.2, 1.0, 0.3))
        out += merlons_circle(cxz[0], cxz[1], 1.8, 5.2, step=1.3, mw=0.75, depth=0.45, h=0.8)
    return out


def doors_and_matacanes():
    out = []
    for a, b, L, t, n in edges(BODY):
        if L < 12:
            continue
        f = Face(a, t, n)
        if n[0] < -0.9:                                     # west front: the door
            out.append(f.poly(arch_pts(L / 2, 0.0, 2.6, 1.2, 6), DOOR, 0.05))
            out.append(f.poly(arch_pts(L / 2, -0.1, 2.75, 1.55, 6), GRANITE_L, 0.03))
        if n[1] < 0.9:                                      # not on the south front (the gallery is there)
            out += matacan(f, L / 2, WALL_TOP - 0.6)
    return out


def build():
    parts = [("main_body", merge(main_body() + doors_and_matacanes())),
             ("gallery", merge(south_gallery())),
             ("keep", merge(keep()))]
    for k in TOWERS:
        parts.append((f"tower_{k}", merge(round_tower(k))))
    parts.append(("east_body", merge(east_body())))
    parts.append(("barrier", merge(barrier())))
    return parts


def main():
    parts = build()
    out_dir = ROOT / "models" / "Q1013093"
    out_dir.mkdir(parents=True, exist_ok=True)
    named = []
    for name, m in parts:
        k = 0.94 + 0.08 * fbm(m.v, 0.15, 11, 2)
        m.c = np.clip(m.c * k[:, None], 0, 1)
        named.append((name, m))
    tris, size = write_glb(out_dir / "model.glb", named, roughness=0.9)
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        meta.write_bytes((json.dumps(mj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"{tris} triangles, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
