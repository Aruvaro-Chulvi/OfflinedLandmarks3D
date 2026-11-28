"""
Cuartel del Conde-Duque (Q5191859), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Pedro de Ribera (died 1742), 1717–c. 1730, barracks of the Royal Guardias de Corps; public domain.
Upper floors burnt in 1869; restored as a cultural centre (Julio Cano Lasso from 1969, Carlos de Riaño 2009–2011).

What is modelled:
- The whole block round its three courtyards (the central one the largest), as OpenStreetMap relation 73436 draws
  it: 229 × 85–88 m, ranges 15 m deep, the rear (west) range 15–20 m deep (its outer wall is not quite parallel).
- Red brick walls on a granite plinth, rusticated granite quoins at the outer corners, four rows of openings
  (ground-floor windows, the row of oculi, two upper rows) with granite frames, a moulded cornice at 17 m.
  On the rear (west) the street lies 3–11 m lower: the granite basement shows there, with its small windows.
- Pedro de Ribera's Baroque portal on calle del Conde Duque (east): two stacks of carved piers, the door with its
  lintel, the cartouche with garlands and trophies, the broken pediment and the royal arms with the crown on top,
  stylised (volumes as seen from a few dozen metres up).
- The main courtyard: the white stone central body of the rear range (the old tower, rebuilt after the fire) with
  its giant pilasters, the attic with the arms, the curved shoulders, the urns and the four pinnacles; the stone
  podium with its steps in front of it. Courtyard walls of brick below and cream render on the top floor.
- Roofs: grey zinc (front, side and inner ranges), red tiles on the rear range, hipped at the outer corners.
- Courtyards: the open courtyards get a floor (tools/ground_fill.py), level (they are paved, y = 0), coloured from
  the IGN orthophoto (PNOA, © Instituto Geográfico Nacional de España, CC BY 4.0, scne.es;
  model_sources/Q5191859_pnoa.jpg), its baked shadows taken out.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap relation 73436 (outline and the three courtyards), in the axes of heading 116.7° (the main
  front faces east-south-east onto calle del Conde Duque; long sides at 26.7°). Courtyards: south 41 × 53 m,
  central 87 × 53 m, north 40 × 53 m.
- Heights from photographs on Wikimedia Commons scaled on the plan (±10 %): cornice 17 m, ridges ~21 m; the
  central body of the courtyard 24 m wide, 17.8 m to its cornice, the attic 15 m wide to 26.8 m, pinnacles to
  ~32.5 m; the portal 9.6 m wide and 15.8 m to the cross of the crown, the door 4.4 × 7 m.
- PNOA orthophoto: roofs (zinc / tiles), the attic of the central body, the podium of the main courtyard.
- Terrain: IGN MDT05 (tools/terrain.py): the front street falls from +2 m (north) to −4 m (south), the rear one
  lies at −3 to −11 m. Ground "keep", y = 0 at the street in front of the portal.

Axes (docs/MODEL_SPEC.md): +Y up, the main (east) front faces +Z (heading 116.7°), +X points to 26.7° (north-north-
east). Origin at the centroid of the OSM outline (in the central courtyard).

    python model_sources/Q5191859_cuartel_del_conde_duque.py   -> models/Q5191859/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, triangulate, walls, cap, extrude, offset, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(1717)

# ── Colours ──────────────────────────────────────────────────────────────────
BRICK = np.array((0.70, 0.40, 0.31))
BRICK_D = (0.58, 0.32, 0.25)
GRANITE = (0.64, 0.62, 0.58)
GRANITE_L = (0.76, 0.75, 0.71)          # the carved granite of the portal
WHITE = (0.89, 0.87, 0.82)              # the central body of the courtyard
CREAM = (0.90, 0.84, 0.70)              # rendered top floor of the courtyards
ZINC = (0.56, 0.58, 0.60)
ZINC_D = (0.47, 0.49, 0.51)
TILE = (0.66, 0.39, 0.28)
GLASS = (0.12, 0.13, 0.15)
DARK = (0.07, 0.06, 0.06)
DOOR = (0.30, 0.20, 0.13)
GOLD = (0.78, 0.62, 0.28)
RED = (0.60, 0.15, 0.12)
FOUND = -14.0
EAVE = 16.5                              # top of the brick walls (under the cornice)
CORN = 17.0                              # top of the cornice = eaves of the roofs

# ── Plan (OSM relation 73436, heading 116.7°) ────────────────────────────────
OUTER = [(-114.10, 42.64), (-112.40, -17.77), (-111.63, -44.94), (114.94, -40.41), (115.04, 42.43)]
COURTS = [
    [(-99.03, 27.94), (-98.85, -25.08), (-57.79, -25.13), (-57.51, 27.87)],      # south
    [(-42.46, 27.82), (-42.65, -25.10), (44.59, -25.01), (45.13, 27.85)],        # central
    [(100.37, 27.78), (59.89, 27.89), (59.86, -24.84), (100.27, -25.18)],        # north
]
PX = 1.0                                  # x of the portal and of the central body (axis of the main courtyard)


def rear_z(x):
    """z of the rear (west) wall at x."""
    (x0, z0), (x1, z1) = OUTER[2], OUTER[3]
    return z0 + (x - x0) / (x1 - x0) * (z1 - z0)


def rear_ground(x):
    """The street behind (west), from the IGN MDT05 relative to the front street at the portal."""
    return float(np.interp(x, [-115, -80, -40, 0, 40, 80, 115], [-11.1, -10.7, -9.9, -8.7, -8.3, -5.9, -3.2]))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sphere(r, c, seg=10, rings=6):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def tint(c, k=0.03):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, k)), 0, 1))


def _inside(p, ring):
    x, z = p; ins = False; n = len(ring)
    for i in range(n):
        (x0, z0), (x1, z1) = ring[i], ring[(i + 1) % n]
        if (z0 > z) != (z1 > z) and x < x0 + (z - z0) * (x1 - x0) / (z1 - z0):
            ins = not ins
    return ins


def in_mass(p):
    return _inside(p, OUTER) and not any(_inside(p, c) for c in COURTS)


# ── Edges: elements placed on a wall face ────────────────────────────────────
class Edge:
    def __init__(self, p, q, court):
        self.p = np.array(p, float); q = np.array(q, float)
        d = q - self.p; self.L = float(np.linalg.norm(d)); self.t = d / self.L
        n = np.array([-self.t[1], self.t[0]])
        mid = self.p + d / 2
        if in_mass(tuple(mid + 0.5 * n)):
            n = -n
        self.n = n; self.court = court
        self.flip = (self.t[0] * n[1] - self.t[1] * n[0]) < 0

    def x_at(self, a):
        return float(self.p[0] + a * self.t[0])

    def put(self, m):
        """Local (a along the edge, y, o outwards) -> model axes."""
        v = m.v.copy()
        a, o = v[:, 0].copy(), v[:, 2].copy()
        v[:, 0] = self.p[0] + a * self.t[0] + o * self.n[0]
        v[:, 2] = self.p[1] + a * self.t[1] + o * self.n[1]
        f = m.f[:, ::-1] if self.flip else m.f
        return Mesh(v, f, m.c)

    def box(self, a0, a1, y0, y1, o0, o1, c):
        return self.put(B(a0, a1, y0, y1, o0, o1, c))

    def panel(self, a, yc, w, h, o, c):
        m = Mesh([(a - w / 2, yc - h / 2, o), (a + w / 2, yc - h / 2, o), (a + w / 2, yc + h / 2, o), (a - w / 2, yc + h / 2, o)],
                 [(0, 1, 2), (0, 2, 3)], c)
        return self.put(m)

    def disc(self, a, yc, r, o, c, seg=10):
        ang = np.linspace(0, 2 * math.pi, seg, endpoint=False)
        v = [(a, yc, o)] + [(a + r * math.cos(t), yc + r * math.sin(t), o) for t in ang]
        f = [(0, 1 + i, 1 + (i + 1) % seg) for i in range(seg)]
        return self.put(Mesh(v, f, c))


def edges():
    out = []
    for i in range(len(OUTER)):
        out.append(Edge(OUTER[i], OUTER[(i + 1) % len(OUTER)], False))
    for c in COURTS:
        for i in range(4):
            out.append(Edge(c[i], c[(i + 1) % 4], True))
    return out


def window(e, a, y0, y1, w):
    yc, h = (y0 + y1) / 2, y1 - y0
    return [e.panel(a, yc, w + 0.5, h + 0.45, 0.07, tint(GRANITE if not e.court or y0 < 12 else WHITE, 0.02)),
            e.panel(a, yc, w, h, 0.10, GLASS)]


def oculus(e, a, yc, r=0.5):
    return [e.disc(a, yc, r + 0.22, 0.07, GRANITE), e.disc(a, yc, r, 0.10, GLASS)]


def facade(e):
    """Windows, oculi, plinth details of one face."""
    out = []
    skip = []
    if not e.court and abs(e.n[1] - 1) < 1e-3:                       # front: the portal
        skip.append((PX - 6.2, PX + 6.2))
    xs = (e.p[0], e.x_at(e.L))
    if e.court and e.n[1] > 0.99 and min(xs) < PX < max(xs):
        skip.append((PX - 13.0, PX + 13.0))                              # the central body of the main courtyard
    m0 = 3.4
    n = max(1, int(round((e.L - 2 * m0) / 4.5)))
    for k in range(n + 1):
        a = m0 + (e.L - 2 * m0) * k / n if n else e.L / 2
        x = e.x_at(a)
        if any(lo < x < hi for lo, hi in skip):
            continue
        out += window(e, a, 3.0, 4.9, 1.3)
        if k % 2 == 0:
            out += oculus(e, a, 6.9)
        out += window(e, a, 8.6, 10.8, 1.4)
        out += window(e, a, 12.7, 14.6, 1.35)
        if not e.court and e.n[1] < -0.9:                               # rear: the granite basement
            g = rear_ground(x)
            if g < -4.5:
                out += window(e, a, -3.2, -1.6, 0.9)
            if g < -8.0:
                out += window(e, a, -6.6, -5.0, 0.9)
        if e.court and k % 2 == 0:                                      # low oculi of the courtyard podium floor
            out += oculus(e, a, 1.3, 0.35)
    return out


def quoins(e, at_start):
    """Rusticated granite quoin at one end of an outer face."""
    out = []
    a0 = 0.0 if at_start else e.L
    s = 1 if at_start else -1
    y = 2.3; k = 0
    while y < EAVE - 0.3:
        w = 2.1 if k % 2 == 0 else 1.4
        a1 = a0 + s * w
        out.append(e.box(min(a0, a1), max(a0, a1), y, y + 0.56, 0.0, 0.14, tint(GRANITE, 0.025)))
        y += 0.62; k += 1
    return out


# ── Roofs ────────────────────────────────────────────────────────────────────
def roof(A0, A1, B1, B0, pitch, hmax, c, hip0=True, hip1=True, top_c=None):
    """A roof over a quad: A0→A1 one long eave, B0→B1 the other; hipped or butt ends; truncated at hmax."""
    A0, A1, B1, B0 = (np.array(p, float) for p in (A0, A1, B1, B0))
    L = (np.linalg.norm(A1 - A0) + np.linalg.norm(B1 - B0)) / 2
    W = (np.linalg.norm(B0 - A0) + np.linalg.norm(B1 - A1)) / 2
    h = min(W / 2 * math.tan(math.radians(pitch)), hmax)
    d = h / math.tan(math.radians(pitch))
    tt = min(d / W, 0.5 - 1e-3)
    s0 = d / L if hip0 else 0.0
    s1 = 1 - d / L if hip1 else 1.0

    def P(s, t, y):
        a = A0 + (A1 - A0) * s; b = B0 + (B1 - B0) * s
        q = a + (b - a) * t
        return (q[0], y, q[1])
    y0, y1 = CORN, CORN + h
    base = [P(0, 0, y0), P(1, 0, y0), P(1, 1, y0), P(0, 1, y0)]
    top = [P(s0, tt, y1), P(s1, tt, y1), P(s1, 1 - tt, y1), P(s0, 1 - tt, y1)]
    v = base + top
    f = [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7), (4, 5, 6), (4, 6, 7),
         (0, 2, 1), (0, 3, 2)]
    m = Mesh(v, f, c)
    # make every face point away from the centre of the roof
    cen = np.array(v).mean(0); cen[1] = CORN + h * 0.3
    tri = m.v[m.f]; nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    bad = np.einsum("ij,ij->i", nrm, tri.mean(1) - cen) < 0
    m.f[bad] = m.f[bad][:, ::-1]
    if top_c is not None:
        m.c[4:] = top_c
    return m, h


def roofs():
    out = []
    o = 0.6
    zf_o, zf_i = 42.55 + o, 27.85 - o
    # front range (zinc), hipped at both ends
    out.append(roof((-114.1 - o, zf_o), (115.04 + o, zf_o), (115.04 + o, zf_i), (-114.1 - o, zf_i), 30, 4.6, ZINC)[0])
    # rear range (tiles), hipped
    r0, r1 = rear_z(-111.63) - o, rear_z(114.94) - o
    zr_i = -25.1 + o
    rear, hr = roof((-111.63 - o, r0), (114.94 + o, r1), (114.94 + o, zr_i), (-111.63 - o, zr_i), 24, 4.8, TILE)
    out.append(rear)
    zr_ridge = (rear_z(0) + -25.1) / 2
    # side and inner ranges (zinc), running into the front and rear roofs
    for xa, xb in ((-113.4 - o, -99.0 + o), (-57.6 - o, -42.5 + o), (44.8 - o, 59.9 + o), (100.3 - o, 115.0 + o)):
        out.append(roof((xa, zr_ridge), (xa, 35.2), (xb, 35.2), (xb, zr_ridge), 30, 4.6, ZINC, hip0=False, hip1=False)[0])
    # small dormer vents on the outer slope of the tiled roof
    for x in np.arange(-100, 110, 11.0):
        if abs(x - PX) < 14:
            continue
        z = rear_z(x) + 3.2
        out.append(B(x - 0.5, x + 0.5, CORN + 0.9, CORN + 1.9, z - 0.6, z + 0.6, TILE))
        out.append(B(x - 0.3, x + 0.3, CORN + 1.0, CORN + 1.6, z - 0.66, z - 0.6, DARK))
    # skylight strips on the zinc ridges of the side ranges (seen on the orthophoto)
    for xc in (-106.4, 107.6):
        out.append(B(xc - 0.8, xc + 0.8, CORN + 4.0, CORN + 4.35, -12, 15, ZINC_D))
    return out


# ── Walls ────────────────────────────────────────────────────────────────────
def body():
    out = []
    out.append(extrude(OUTER, FOUND, EAVE, tuple(BRICK), holes=COURTS, top=True))
    # granite plinth outside (down to the foundations: on the west the street lies up to 11 m lower)
    out.append(extrude(offset(OUTER, 0.35), FOUND, 2.0, GRANITE, holes=[], top=False))
    out.append(walls(offset(OUTER, 0.18), 2.0, 2.3, GRANITE))
    out.append(cap(offset(OUTER, 0.35), 2.0, GRANITE, holes=[OUTER]))
    # courtyards: granite socle, cream top floor over a stone band
    for c in COURTS:
        out.append(walls(offset(c, -0.25), FOUND, 0.6, GRANITE, hole=True))
        out.append(walls(offset(c, -0.04), 12.3, EAVE, CREAM, hole=True))
        out.append(walls(offset(c, -0.12), 12.0, 12.3, WHITE, hole=True))
    # cornice
    holes_o = [offset(c, -0.6) for c in COURTS]
    out.append(extrude(offset(OUTER, 0.6), EAVE, CORN, BRICK_D, holes=holes_o, top=True))
    out.append(extrude(offset(OUTER, 0.3), EAVE - 0.35, EAVE, BRICK_D, holes=[offset(c, -0.3) for c in COURTS], top=False))
    es = edges()
    for e in es:
        out += facade(e)
    outer = [e for e in es if not e.court]
    for i, e in enumerate(outer):
        prev = outer[i - 1]
        turn = abs(float(np.dot(prev.t, e.t)))
        if turn < 0.9:                                                   # a real corner, not the small kink
            out += quoins(e, True) + quoins(prev, False)
    return out


# ── The portal of Pedro de Ribera (east front) ───────────────────────────────
def portal():
    out = []
    z0 = 42.55                      # the wall face at the portal
    G = GRANITE_L

    def P(x0, x1, y0, y1, d0, d1, c=None):
        return B(PX + x0, PX + x1, y0, y1, z0 + d0, z0 + d1, tint(c or G, 0.02))

    def mirror(ms):
        res = []
        for m in ms:
            n = m.copy(); n.v[:, 0] = 2 * PX - n.v[:, 0]; n.f = n.f[:, ::-1]; res.append(n)
        return ms + res
    side = []
    side.append(P(2.2, 4.8, -1.0, 0.6, -0.1, 1.45))                                   # granite base
    # the pier stack: alternating wider and narrower carved blocks (estípite-like), up to the lintel
    y, k = 0.6, 0
    while y < 8.0:
        h = 0.55 if k % 2 == 0 else 0.35
        w = (2.45, 4.75) if k % 2 == 0 else (2.7, 4.5)
        dep = 1.3 if k % 2 == 0 else 1.15
        side.append(P(w[0], w[1], y, y + h, 0.0, dep))
        y += h; k += 1
    for yy in (2.0, 3.6, 5.2, 6.8):                                                     # carved fruits and shields on the pier
        side.append(sphere(0.32, G, 10, 6).S(1.1, 1.2, 0.6).T(PX + 3.6, yy, z0 + 1.4))
    for yy in (1.6, 3.0, 4.4, 5.8, 7.2):
        side.append(sphere(0.2, G, 8, 5).T(PX + 2.55, yy, z0 + 1.18))
    side.append(P(2.2, 2.45, 0.0, 7.0, 0.0, 0.9))                                     # inner jamb
    # the cornice over the piers, broken forward
    side.append(P(2.2, 5.05, 8.0, 8.5, -0.1, 1.6))
    side.append(P(2.3, 4.95, 8.5, 8.75, 0.0, 1.35))
    # upper pilaster strips beside the cartouche, curving in
    side.append(P(3.4, 4.6, 8.75, 11.3, 0.0, 0.9))
    for yy in (9.3, 10.3):
        side.append(sphere(0.25, G, 8, 5).S(1, 1.2, 0.6).T(PX + 4.0, yy, z0 + 0.95))
    side.append(sphere(0.45, G, 10, 6).S(1, 0.6, 0.8).T(PX + 4.0, 11.5, z0 + 0.5))   # scroll head
    # broken pediment: raking pieces rising towards the arms
    for t in np.linspace(0, 1, 6):
        x = 4.9 - 2.6 * t; yy = 11.2 + 1.6 * t
        side.append(P(x - 0.5, x + 0.05, yy, yy + 0.4, -0.05, 1.0))
    # trophies / flags and a seated figure beside the cartouche
    side.append(tube([(PX + 2.6, 8.9, z0 + 1.0), (PX + 3.3, 11.4, z0 + 1.0)], 0.06, G, 6))
    side.append(tube([(PX + 2.4, 8.9, z0 + 1.05), (PX + 2.0, 11.6, z0 + 1.05)], 0.06, G, 6))
    side.append(B(PX + 2.75, PX + 3.3, 10.4, 11.2, z0 + 0.95, z0 + 1.1, G))
    side.append(sphere(0.5, G, 10, 6).S(0.8, 1.4, 0.7).T(PX + 2.6, 9.4, z0 + 0.8))
    side.append(sphere(0.22, G, 8, 5).T(PX + 2.6, 10.5, z0 + 0.8))
    out += mirror(side)
    # the door: dark recess, wooden leaves, a granite lintel with eared corners
    out.append(B(PX - 2.2, PX + 2.2, 0.0, 7.0, z0 - 1.5, z0 + 0.02, DARK))
    out.append(B(PX - 2.15, PX + 2.15, 0.0, 6.6, z0 - 1.2, z0 - 1.0, DOOR))
    out.append(P(-2.45, 2.45, 7.0, 8.0, 0.0, 1.1))
    for s in (-1, 1):
        out.append(P(s * 2.2 - 0.25, s * 2.2 + 0.25, 6.4, 7.0, 0.0, 1.0))
    out.append(P(-0.4, 0.4, 6.7, 8.3, 0.0, 1.35))                                     # keystone
    # the cartouche with garlands
    out.append(P(-2.3, 2.3, 8.0, 11.0, -0.05, 0.6))
    out.append(cyl(1.25, 1.25, 0.35, G, 20).R(math.pi / 2, 0, 0).S(1, 1.25, 1).T(PX, 9.7, z0 + 0.95))
    out.append(tube([(PX + 1.35 * math.cos(a), 9.7 + 1.65 * math.sin(a), z0 + 0.98) for a in np.linspace(0, 2 * math.pi, 21)],
                    0.12, G, 6))
    out.append(tube([(PX - 2.0 + 4.0 * t, 8.6 - 0.6 * math.sin(math.pi * t), z0 + 1.15) for t in np.linspace(0, 1, 11)], 0.16, G, 6))
    for s in (-1, 1):
        out.append(tube([(PX + s * 2.1, 10.9, z0 + 1.0), (PX + s * 1.6, 10.2, z0 + 1.1), (PX + s * 1.9, 9.2, z0 + 1.05)], 0.14, G, 6))
    # the royal arms with the crown and cross on the top
    out.append(P(-1.3, 1.3, 11.0, 12.6, 0.0, 0.9))
    out.append(cyl(0.85, 0.85, 0.4, G, 18).R(math.pi / 2, 0, 0).S(1, 1.3, 1).T(PX, 13.3, z0 + 0.85))
    out.append(tube([(PX + 0.95 * math.cos(a), 13.3 + 1.2 * math.sin(a), z0 + 0.9) for a in np.linspace(0, 2 * math.pi, 19)], 0.1, G, 6))
    out.append(cyl(0.6, 0.7, 0.35, G, 14).T(PX, 14.45, z0 + 0.6))
    out.append(lathe([(0.7, 0), (0.5, 0.25), (0.2, 0.45), (0, 0.5)], G, 12).T(PX, 14.8, z0 + 0.6))
    out.append(sphere(0.1, G, 8, 5).T(PX, 15.35, z0 + 0.6))
    out.append(B(PX - 0.05, PX + 0.05, 15.4, 15.85, z0 + 0.55, z0 + 0.65, G))
    out.append(B(PX - 0.18, PX + 0.18, 15.6, 15.7, z0 + 0.55, z0 + 0.65, G))
    for s in (-1, 1):                                                                  # flags behind the arms
        for a in (0.5, 0.9):
            out.append(tube([(PX + s * 0.3, 12.4, z0 + 0.7), (PX + s * (0.3 + 2.0 * math.sin(a)), 12.4 + 2.0 * math.cos(a), z0 + 0.7)],
                            0.05, G, 5))
    # two granite guard stones at the door
    for s in (-1, 1):
        out.append(sphere(0.35, GRANITE, 8, 5).S(1, 1.3, 1).T(PX + s * 2.0, 0.3, z0 + 1.0))
    return out


# ── The central body of the main courtyard (rear range) ──────────────────────
def central_body():
    out = []
    zf = -25.1 + 0.7                  # its face stands 0.7 m into the courtyard
    zb = -31.5
    W = WHITE

    def P(x0, x1, y0, y1, z0, z1, c=None):
        return B(PX + x0, PX + x1, y0, y1, z0, z1, tint(c or W, 0.015))

    def mirror(ms):
        res = []
        for m in ms:
            n = m.copy(); n.v[:, 0] = 2 * PX - n.v[:, 0]; n.f = n.f[:, ::-1]; res.append(n)
        return ms + res
    out.append(P(-12.0, 12.0, -1.0, 17.0, zb, zf))
    out.append(P(-12.4, 12.4, 17.0, 17.9, zb, zf + 0.5))                                 # entablature
    out.append(P(-12.2, 12.2, 17.9, 18.3, zb, zf + 0.3))
    side = []
    for xc in (4.0, 10.9):                                                               # giant pilasters
        side.append(P(xc - 0.9, xc + 0.9, 1.3, 17.0, zf, zf + 0.35))
        side.append(P(xc - 1.05, xc + 1.05, 1.3, 2.2, zf, zf + 0.45))
        side.append(P(xc - 1.05, xc + 1.05, 16.2, 17.0, zf, zf + 0.45))
    for yy in (4.0, 8.6, 12.7):                                                          # windows between them
        side.append(Mesh([(PX + 6.8, yy, zf + 0.03), (PX + 8.1, yy, zf + 0.03), (PX + 8.1, yy + 1.9, zf + 0.03), (PX + 6.8, yy + 1.9, zf + 0.03)],
                         [(0, 1, 2), (0, 2, 3)], GLASS))
    for yy in (8.6, 12.7):
        side.append(Mesh([(PX + 1.4, yy, zf + 0.03), (PX + 2.4, yy, zf + 0.03), (PX + 2.4, yy + 1.9, zf + 0.03), (PX + 1.4, yy + 1.9, zf + 0.03)],
                         [(0, 1, 2), (0, 2, 3)], GLASS))
    side.append(lathe([(0, 0), (0.45, 0), (0.45, 0.3), (0.3, 0.45), (0.6, 1.1), (0.55, 1.6), (0.25, 1.9), (0.3, 2.2), (0, 2.6)], W, 12)
                .T(PX + 11.3, 18.3, zf - 0.5))                                           # urns on the corners
    # the curved shoulder from the attic down to the corner
    sh = [(7.5, 18.3), (11.0, 18.3)] + [(11.0 - 3.5 * t, 18.3 + 7.7 * (t ** 1.7)) for t in np.linspace(0.12, 1, 8)]
    pts, tris = triangulate([(x, y) for x, y in sh])
    for zz, up in ((zf, True), (zf - 6.0, False)):
        v = [(PX + x, y, zz) for x, y in pts]
        side.append(Mesh(v, tris[:, ::-1] if up else tris, tint(W)))
    for i in range(len(sh)):
        (xa, ya), (xb, yb) = sh[i], sh[(i + 1) % len(sh)]
        side.append(Mesh([(PX + xa, ya, zf), (PX + xb, yb, zf), (PX + xb, yb, zf - 6.0), (PX + xa, ya, zf - 6.0)], [(0, 1, 2), (0, 2, 3)], tint(W)))
    side.append(tube([(PX + 11.0 - 3.5 * t, 18.4 + 7.7 * (t ** 1.7), zf + 0.1) for t in np.linspace(0, 1, 10)], 0.18, W, 6))
    # attic pilasters and pinnacles
    for xc in (4.6, 6.6):
        side.append(P(xc - 0.55, xc + 0.55, 18.3, 26.0, zf, zf + 0.25))
        side.append(P(xc - 0.55, xc + 0.55, 26.8, 27.8, zf - 1.4, zf - 0.2))
        side.append(lathe([(0.45, 0), (0.45, 0.4), (0.25, 0.6), (0.5, 1.2), (0.45, 1.7), (0.15, 2.0), (0.22, 2.4), (0.06, 3.6), (0, 4.7)], W, 10)
                    .T(PX + xc, 27.8, zf - 0.8))
    out += mirror(side)
    out.append(P(-7.5, 7.5, 18.3, 26.0, zf - 9.0, zf))                                    # attic
    out.append(P(-7.8, 7.8, 26.0, 26.8, zf - 9.3, zf + 0.35))
    out.append(P(-1.4, 1.4, 1.3, 17.0, zf, zf + 0.25))                                    # central bay frame
    out.append(B(PX - 1.2, PX + 1.2, 1.3, 5.6, zf + 0.26, zf + 0.3, DOOR))               # the door
    out.append(P(-1.6, 1.6, 5.6, 6.2, zf, zf + 0.5))
    for yy in (8.6, 12.7):
        out.append(B(PX - 0.7, PX + 0.7, yy, yy + 2.0, zf + 0.26, zf + 0.3, GLASS))
    # the arms on the attic: oval medallion, gold frame, the shield
    out.append(cyl(2.2, 2.2, 0.3, W, 22).R(math.pi / 2, 0, 0).S(1, 1.2, 1).T(PX, 22.0, zf + 0.6))
    out.append(tube([(PX + 2.25 * math.cos(a), 22.0 + 2.7 * math.sin(a), zf + 0.62) for a in np.linspace(0, 2 * math.pi, 25)], 0.16, GOLD, 6))
    out.append(cyl(1.2, 1.2, 0.15, RED, 16).R(math.pi / 2, 0, 0).S(1, 1.3, 1).T(PX, 21.8, zf + 0.75))
    out.append(cyl(0.7, 0.7, 0.1, GOLD, 12).R(math.pi / 2, 0, 0).T(PX, 21.8, zf + 0.85))
    out.append(cyl(0.7, 0.8, 0.5, GOLD, 12).T(PX, 23.6, zf + 0.6))
    # the stone podium and its steps in front
    S = (0.74, 0.73, 0.70)
    out.append(B(PX - 30, PX + 30, -0.5, 1.3, zf, -15.2, S))
    for k in range(6):
        out.append(B(PX - 5.5, PX + 5.5, -0.5, 1.3 - k * 0.22, -15.2, -15.2 + 0.4 * (k + 1), S))
    for s in (-1, 1):
        out.append(B(PX + s * 5.5 - 0.5, PX + s * 5.5 + 0.5, 1.3, 2.4, -15.6, -14.6, S))
        out.append(sphere(0.4, W, 8, 5).T(PX + s * 5.5, 2.75, -15.1))
    return out


FOOTPRINT = None


def main():
    out_dir = ROOT / "models" / "Q5191859"
    out_dir.mkdir(parents=True, exist_ok=True)
    walls_m = merge(body() + portal() + central_body())
    k = 0.95 + 0.07 * fbm(walls_m.v, 0.08, 3, 2)
    walls_m.c = np.clip(walls_m.c * k[:, None], 0, 1)
    roof_m = merge(roofs())
    k = 0.96 + 0.06 * fbm(roof_m.v, 0.05, 5, 2)
    roof_m.c = np.clip(roof_m.c * k[:, None], 0, 1)
    named = [("cuartel", walls_m), ("roofs", roof_m)]
    from ground_fill import courtyard_floor
    floor = courtyard_floor([walls_m, roof_m], "Q5191859", height=lambda x, z: np.zeros(np.shape(x)))
    if floor is not None:
        named.append(("courtyards", floor))
    tris, size = write_glb(out_dir / "model.glb", named, roughness=0.88)
    V = np.vstack([m.v for _, m in named[:2]])
    lo, hi = V.min(0), V.max(0)
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        meta.write_text(json.dumps(mj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{tris} triangles, {size / 1e6:.2f} MB, extent x {lo[0]:.1f}..{hi[0]:.1f}, y {lo[1]:.1f}..{hi[1]:.1f}, "
          f"z {lo[2]:.1f}..{hi[2]:.1f}")


if __name__ == "__main__":
    main()
