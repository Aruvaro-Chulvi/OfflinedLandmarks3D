"""
Edificio Telefónica (Q2092717), Gran Vía 28, Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Ignacio de Cárdenas Pastor (1898–1979), 1926–1929, for the Compañía Telefónica Nacional de España;
the first skyscraper of Madrid, steel frame clad in Novelda limestone over a granite base. Rights: protected
(the architect died in 1979: free in 2060).

Measurements (see meta.json for the sources):
- OpenStreetMap way 192107181 (outline, 2,286 m²) and its 20 building:part pieces with heights, in model x/z for
  heading 185.8° (the Gran Vía front, z ≈ +23.5, faces south-south-west; +X points east, to Calle de Fuencarral;
  −X is Calle de Valverde). The parts give the stepped massing seen on every photo: a 28–32 m ring on the
  streets (8–9 floors, crowned by the Baroque crests), a set-back ring to 43 m, the core to 50 m, lift
  pavilions to 53 m, the tower on the Gran Vía axis (14 × 14 m) to 73 m and its clock stage (9 × 9 m) to 89 m.
- Floors counted on the photos «Telefónica - Gran Vía 28 - Madrid.jpg», «Madrid - Edificio Telefónica,
  perspectiva.jpg», «Edificio Telefonica - Madrid 01.jpg» and «Parte superior del Edificio Telefonica en
  Madrid.jpg» (Wikimedia Commons): a granite base of shop floor + mezzanine (0–9 m), then floors of ≈ 3.4 m; the
  Baroque portal over the main door rising through three floors; the shaft ends in a cornice with corner
  pinnacles, then the clock stage with buttresses, pinnacles and a curved crest. 89 m (Wikipedia) is taken
  as the top of the crest; the antennas, removed in 2017, are not modelled.
No signs and no logos (the shop fronts are shown as plain dark glazing).

Axes (docs/MODEL_SPEC.md): +Y up, the Gran Vía front faces +Z, origin on the ground at the centroid of the OSM
outline (40.4204031, −3.7019224).

    python model_sources/Q2092717_edificio_telefonica.py   -> models/Q2092717/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import GLASS, Mesh, box, cap, cyl, lathe, merge, write_glb  # noqa: E402

rng = np.random.default_rng(1929)

STONE = np.array((0.87, 0.83, 0.73))     # Novelda limestone (cream)
STONE_B = np.array((0.80, 0.77, 0.70))   # the plainer back façades
GRAN = np.array((0.62, 0.61, 0.58))      # the grey granite base
ORN = (0.90, 0.86, 0.77)                 # carved stone
ROOF = (0.58, 0.57, 0.54)
DARK = (0.14, 0.16, 0.18)
CLOCK = (0.10, 0.10, 0.11)
FOUND = -4.0
BASE = 9.0                               # shop floor + mezzanine (granite)
FL = 3.4                                 # floor height above the base

# OSM way 192107181 in model x/z (heading 185.8°, origin = centroid)
OUTLINE = [(-23.49, 20.49), (-20.33, 23.54), (-9.99, 23.48), (-6.71, 23.53), (-0.61, 23.49), (7.10, 23.43), (10.29, 23.35),
           (21.24, 23.30), (23.90, 21.38), (24.80, 16.63), (29.23, -6.62), (30.40, -12.80), (27.93, -13.35), (27.22, -13.51),
           (23.80, -14.27), (4.11, -18.68), (2.63, -19.01), (3.49, -24.45), (-1.10, -25.30), (-12.51, -27.38), (-14.87, -27.82),
           (-19.36, -28.64), (-20.38, -28.82), (-23.71, -29.29), (-23.71, -25.19), (-23.51, 16.35)]
# building:parts (OSM way id, height, outline). The thin 46 m strip in front of the tower (310671543) is the
# tower's own front (the photos show the tower flush with the street from the ground up): taken as 73 m; the
# tiny 46 m triangle at the back (310671538) is taken as 43 m. The 89 m clock stage is built on top of the tower.
PARTS = [
    (310671535, 43, [(-9.99, 23.48), (-10.03, 20.55), (-17.91, 20.59), (-20.39, 18.10), (-20.47, 16.34), (-20.46, 1.75), (-16.53, 1.72), (-16.36, 16.19), (-6.80, 15.98), (-6.71, 22.91), (-6.71, 23.53)]),
    (310671536, 43, [(7.02, 16.07), (17.57, 16.06), (19.29, 7.70), (21.13, -1.29), (23.80, -14.27), (27.22, -13.51), (27.93, -13.35), (26.66, -7.04), (22.00, 16.06), (21.58, 18.11), (18.43, 20.22), (10.37, 20.48), (10.29, 23.35), (7.10, 23.43), (7.10, 22.82)]),
    (310671537, 50, [(-6.80, 15.98), (-16.36, 16.19), (-16.53, 1.72), (-4.10, 1.55), (-4.14, -0.19), (-3.10, 0.59), (3.02, 0.53), (4.13, -0.51), (4.12, -0.11), (11.20, -0.20), (11.12, -4.20), (4.21, -4.17), (4.11, -18.68), (23.80, -14.27), (21.13, -1.29), (15.55, -1.27), (15.62, 7.68), (19.29, 7.70), (17.57, 16.06), (7.02, 16.07), (6.94, 8.95), (-6.80, 9.08)]),
    (310671538, 43, [(-9.51, -25.48), (-9.50, -24.35), (-4.23, -24.59)]),
    (310671539, 32, [(22.00, 16.06), (24.80, 16.63), (29.23, -6.62), (26.66, -7.04)]),
    (310671540, 53, [(21.13, -1.29), (15.55, -1.27), (15.62, 7.68), (19.29, 7.70)]),
    (310671541, 25, [(29.23, -6.62), (26.66, -7.04), (27.93, -13.35), (30.40, -12.80)]),
    (310671542, 29, [(-23.71, -29.29), (-20.38, -28.82), (-20.39, -27.42), (-20.39, -25.20), (-23.71, -25.19)]),
    (310671544, 53, [(4.12, -0.11), (4.13, -0.51), (4.21, -4.17), (11.12, -4.20), (11.20, -0.20)]),
    (310671545, 32, [(-23.71, -25.19), (-20.39, -25.20), (-20.41, -10.65), (-20.46, 1.75), (-20.47, 16.34), (-23.51, 16.35)]),
    (310671546, 73, [(7.10, 23.43), (7.02, 16.07), (6.94, 8.95), (-6.80, 9.08), (-6.80, 15.98), (-6.71, 23.53), (-0.61, 23.49)]),
    (310671547, 53, [(-4.14, -0.19), (-4.10, 1.55), (-16.53, 1.72), (-20.46, 1.75), (-20.41, -10.65), (-16.61, -10.63), (-4.15, -10.63)]),
    (310671548, 11, [(-4.14, -0.19), (-3.10, 0.59), (3.02, 0.53), (4.13, -0.51), (4.21, -4.17), (4.13, -11.48), (-4.16, -11.43), (-4.15, -10.63)]),
    (310671549, 28, [(-9.99, 23.48), (-10.03, 20.55), (-17.91, 20.59), (-20.39, 18.10), (-20.47, 16.34), (-23.51, 16.35), (-23.49, 20.49), (-20.33, 23.54)]),
    (310671550, 28, [(10.37, 20.48), (18.43, 20.22), (21.58, 18.11), (22.00, 16.06), (24.80, 16.63), (23.90, 21.38), (21.24, 23.30), (10.29, 23.35)]),
    (310671551, 50, [(-16.61, -10.63), (-16.55, -24.63), (-11.93, -24.70), (-11.92, -24.27), (-9.50, -24.35), (-4.23, -24.59), (-4.16, -11.43), (-4.15, -10.63)]),
    (310671552, 43, [(-20.41, -10.65), (-20.39, -25.20), (-20.39, -27.42), (-9.51, -25.48), (-9.50, -24.35), (-11.92, -24.27), (-11.93, -24.70), (-16.55, -24.63), (-16.61, -10.63)]),
    (310671553, 32, [(-4.16, -11.43), (4.13, -11.48), (4.11, -18.68), (2.63, -19.01), (3.49, -24.45), (-1.10, -25.30), (-12.51, -27.38), (-14.87, -27.82), (-19.36, -28.64), (-20.38, -28.82), (-20.39, -27.42), (-9.51, -25.48), (-4.23, -24.59)]),
]
CLOCK_STAGE = (-4.27, 4.52, 11.5, 20.37)       # OSM part 336627008 (x0, x1, z0, z1), 89 m
TOWER = (-6.76, 7.06, 9.0, 23.48)
TOWER_TOP = 73.0
STREET_CREST = (28, 29, 32, 25)                 # the street ring: Baroque crests on its parapet


def jit(c, k=0.012):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, k)), 0, 1))


def inside(pt, ring):
    x, z = pt
    ins = False
    for i in range(len(ring)):
        (xi, zi), (xj, zj) = ring[i], ring[i - 1]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            ins = not ins
    return ins


def h_at(pt):
    """Height of the building at a point (0 outside)."""
    h = 0.0
    for _, hh, ring in PARTS:
        if inside(pt, ring):
            h = max(h, hh)
    return h


def ccw(ring):
    a = 0.0
    for i in range(len(ring)):
        (x0, z0), (x1, z1) = ring[i - 1], ring[i]
        a += x0 * z1 - x1 * z0
    return ring if a > 0 else ring[::-1]


class Run:
    """A straight stretch of wall from a to b with the outward normal n (in x/z)."""

    def __init__(self, a, b, n):
        self.a, self.b = np.array(a, float), np.array(b, float)
        self.L = float(np.linalg.norm(self.b - self.a))
        self.u = (self.b - self.a) / max(self.L, 1e-9)
        self.n = np.array(n, float)

    def put(self, m):
        """m built in local (t along, y up, d out) coordinates -> model coordinates."""
        u3, n3 = np.array([self.u[0], 0, self.u[1]]), np.array([self.n[0], 0, self.n[1]])
        V = m.v
        W = np.array([self.a[0], 0, self.a[1]]) + V[:, :1] * u3 + V[:, 1:2] * np.array([0, 1, 0]) + V[:, 2:3] * n3
        f = m.f if (self.u[0] * self.n[1] - self.u[1] * self.n[0]) > 0 else m.f[:, ::-1]
        return Mesh(W, f, m.c)

    def box(self, t0, t1, y0, y1, d0, d1, c):
        return self.put(box(t1 - t0, y1 - y0, d1 - d0, c).T((t0 + t1) / 2, y0, (d0 + d1) / 2))

    def face(self, t0, t1, y0, y1, c, d=0.0):
        V = [(t0, y0, d), (t1, y0, d), (t1, y1, d), (t0, y1, d)]
        return self.put(Mesh(V, [(0, 1, 2), (0, 2, 3)], c))


def runs_of(ring, top):
    """Split the edges of a part into stretches with the same neighbour height outside: (Run, h_out)."""
    ring = ccw(ring)
    out = []
    for i in range(len(ring)):
        a, b = np.array(ring[i], float), np.array(ring[(i + 1) % len(ring)], float)
        L = float(np.linalg.norm(b - a))
        if L < 0.2:
            continue
        u = (b - a) / L
        n = np.array([u[1], -u[0]])                    # outward for a counter-clockwise ring in (x, z)
        k = max(1, int(round(L / 0.8)))
        hs = [h_at(tuple(a + u * (L * (j + 0.5) / k) + n * 0.35)) for j in range(k)]
        j0 = 0
        for j in range(1, k + 1):
            if j == k or hs[j] != hs[j0]:
                if hs[j0] < top:
                    out.append((Run(a + u * L * j0 / k, a + u * L * j / k, n), hs[j0]))
                j0 = j
    return out


def build_part(ring, top):
    stone, dark = [], []
    crest = top in STREET_CREST
    for r, hout in runs_of(ring, top):
        street = hout == 0
        back = r.n[1] < -0.5                                       # faces north (the back)
        c = STONE_B if back else STONE
        y0 = FOUND if street else hout
        par = 1.1 if top < TOWER_TOP else 0.0
        # wall (granite up to the base on the streets)
        if street and y0 < BASE:
            stone.append(r.face(0, r.L, y0, BASE, jit(GRAN)))
            stone.append(r.box(0, r.L, BASE - 0.5, BASE + 0.1, -0.05, 0.35, jit(GRAN)))      # base cornice
            stone.append(r.face(0, r.L, BASE, top + par, jit(c)))
        else:
            stone.append(r.face(0, r.L, y0, top + par, jit(c)))
        stone.append(r.face(0, r.L, top, top + par, jit(c), d=-0.3))                   # parapet back
        stone.append(r.box(0, r.L, top - 0.7, top - 0.1, -0.02, 0.45, jit(ORN)))       # cornice
        stone.append(r.box(0, r.L, top + par - 0.12, top + par, -0.35, 0.05, jit(ORN)))  # coping
        # windows
        nb = int(r.L // 3.2)
        if nb >= 1:
            sp = r.L / nb
            ts = [sp * (i + 0.5) for i in range(nb)]
            lo = max(hout, BASE)
            nf = int((top - 1.2 - lo) // FL) if top - lo > 3 else 0
            ystart = BASE + math.ceil(max(0.0, lo - BASE) / FL) * FL
            for t in ts:
                y = ystart
                while y + 2.6 < top - 0.8:
                    dark.append(r.face(t - 0.7, t + 0.7, y + 0.8, y + 2.85, DARK, d=0.03))
                    y += FL
                if street:
                    if r.L > 6:
                        dark.append(r.face(t - 1.1, t + 1.1, 0.4, 4.4, DARK, d=0.03))      # shop front
                    dark.append(r.face(t - 0.7, t + 0.7, 5.6, 7.9, DARK, d=0.03))          # mezzanine
            del nf
            # Baroque crests on the street ring: a curved gable over every second bay
            if crest and street:
                for t in ts[::2]:
                    stone.append(crest_m(r, t, top + par))
    roof = cap(ring, top + 0.02, jit(ROOF))
    return stone, dark, [roof]


def crest_m(r, t, y):
    """A small curved gable (Baroque crest) on the parapet, with a ball on top."""
    prof = []
    w = 1.6
    for k in range(9):
        a = math.pi * k / 8
        prof.append((t - w * math.cos(a), y + 1.3 * math.sin(a)))
    V = [(t - w, y, 0.05), (t + w, y, 0.05)] + [(x, yy, 0.05) for x, yy in prof]
    F = [(0, 1, 2 + i) for i in range(0)]
    F = [(0, 2 + i + 1, 2 + i) for i in range(8)]
    F += [(1, 0, 10)]
    front = Mesh(V, F, jit(ORN))
    back = Mesh([(x, yy, -0.35) for x, yy, _ in V], [f[::-1] for f in F], jit(ORN))
    ball = lathe([(0, -0.3), (0.28, -0.2), (0.3, 0), (0.28, 0.2), (0, 0.3)], ORN, 8).T(t, y + 1.6, -0.15)
    return merge([r.put(front), r.put(back), r.put(ball)])


def pinnacle(x, z, y, h, w, c=ORN):
    """A square obelisk pinnacle on a small pedestal."""
    ped = box(w, h * 0.35, w, jit(c)).T(x, y, z)
    sp = cyl(w * 0.42 * math.sqrt(2), 0.0, h * 0.65, jit(c), seg=4).R(0, math.pi / 4, 0).T(x, y + h * 0.35, z)
    return merge([ped, sp])


def tower():
    """The tower's upper parts: shaft cornice and pinnacles, the shield, the clock stage and its crest."""
    stone, dark = [], []
    x0, x1, z0, z1 = TOWER
    cx = (x0 + x1) / 2
    T = TOWER_TOP
    # cornice and balustrade over the shaft
    stone.append(box(x1 - x0 + 0.9, 0.9, z1 - z0 + 0.9, jit(ORN)).T(cx, T - 0.9, (z0 + z1) / 2))
    for (px, pz) in ((x0 + 0.6, z0 + 0.6), (x1 - 0.6, z0 + 0.6), (x0 + 0.6, z1 - 0.6), (x1 - 0.6, z1 - 0.6)):
        stone.append(pinnacle(px, pz, T, 5.0, 1.3))
    # pinnacles at the shaft corners where it leaves the 50 m block, and on the 43 m ring each side of it
    for (px, pz) in ((x0 + 0.4, z1 - 0.4), (x1 - 0.4, z1 - 0.4)):
        stone.append(pinnacle(px, pz, 53.0, 4.2, 1.0))
    # corner pilasters up the shaft and string courses where the blocks around it stop
    for px in (x0 + 0.45, x1 - 0.45):
        stone.append(box(1.3, T - 1.0 - BASE, 0.5, jit(STONE)).T(px, BASE, z1 + 0.2))
    for pz in (z1 - 0.45,):
        for px in (x0 - 0.2, x1 + 0.2):
            stone.append(box(0.5, T - 1.0 - 43.0, 1.3, jit(STONE)).T(px, 43.0, pz))
    for yb in (43.0, 53.0, 63.0):
        stone.append(box(x1 - x0 + 0.5, 0.45, 0.4, jit(ORN)).T(cx, yb, z1 + 0.15))
    # the shield (royal arms, no crown) on the front: an oval cartouche with a scrolled frame
    sh = lathe([(0.0, 0.0), (1.9, 0.0), (2.3, 1.2), (2.4, 3.4), (2.0, 5.6), (1.0, 7.0), (0.0, 7.4)], ORN, 14)
    sh.S(1, 1, 0.12)
    stone.append(sh.T(cx, 63.6, z1 + 0.1))
    fr = lathe([(0.0, 0.0), (2.9, 0.4), (3.2, 3.4), (2.7, 6.6), (1.4, 8.4), (0.0, 8.8)], jit(STONE), 14).S(1, 1, 0.07)
    stone.append(fr.T(cx, 63.0, z1 + 0.05))
    # the clock stage (OSM 89 m part)
    a0, a1, b0, b1 = CLOCK_STAGE
    ccx, ccz = (a0 + a1) / 2, (b0 + b1) / 2
    W, D = a1 - a0, b1 - b0
    H0, H1 = T, 83.0
    stone.append(box(W, H1 - H0, D, jit(STONE)).T(ccx, H0, ccz))
    stone.append(box(W + 0.7, 0.7, D + 0.7, jit(ORN)).T(ccx, H1 - 0.7, ccz))
    # corner buttresses with pinnacles
    for sx in (-1, 1):
        for sz in (-1, 1):
            bx, bz = ccx + sx * (W / 2 - 0.2), ccz + sz * (D / 2 - 0.2)
            stone.append(box(1.8, H1 + 1.6 - H0, 1.8, jit(STONE)).T(bx, H0, bz))
            stone.append(pinnacle(bx, bz, H1 + 1.6, 4.4, 1.5))
    # clock faces on the four sides, and the curved crest over each of them
    for nrm in ("+z", "-z", "+x", "-x"):
        if nrm in ("+z", "-z"):
            s = 1 if nrm == "+z" else -1
            pos = (ccx, ccz + s * (D / 2 + 0.04))
            ang = 0.0 if s > 0 else math.pi
        else:
            s = 1 if nrm == "+x" else -1
            pos = (ccx + s * (W / 2 + 0.04), ccz)
            ang = math.pi / 2 if s > 0 else -math.pi / 2
        ring = lathe([(2.05, 0), (2.05, 0.15), (1.55, 0.15), (1.55, 0)], CLOCK, 24).R(math.pi / 2, 0, 0)
        disc = cyl(1.55, 1.55, 0.06, (0.55, 0.57, 0.58), seg=24).R(math.pi / 2, 0, 0)
        hands = box(0.18, 1.3, 0.05, (0.95, 0.95, 0.93)).T(0, 0, 0.12)
        hands2 = box(0.18, 1.0, 0.05, (0.95, 0.95, 0.93)).R(0, 0, -2.2).T(0, 0, 0.12)
        cl = merge([ring, disc, hands, hands2]).R(0, ang, 0).T(pos[0], 78.6, pos[1])
        dark.append(cl)
        # curved crest (a segmental gable with scrolls) over the clock
        cw = W / 2 - 1.0
        V, F = [(-cw, 0, 0)], []
        for k in range(13):
            a = math.pi * k / 12
            V.append((-cw * math.cos(a), 3.6 * math.sin(a) ** 0.8, 0))
        V.append((cw, 0, 0))
        F = [(0, k + 2, k + 1) for k in range(12)]
        fr = Mesh(V, F, jit(ORN))
        bk = Mesh([(x, y, -0.6) for x, y, _ in V], [f[::-1] for f in F], jit(ORN))
        top = merge([fr, bk]).R(0, ang, 0).T(pos[0], H1, pos[1])
        stone.append(top)
    # central finial (urn) to 89 m
    stone.append(box(2.4, 1.6, 2.4, jit(STONE)).T(ccx, H1, ccz))
    stone.append(lathe([(0, 0), (0.9, 0), (1.1, 0.6), (0.8, 1.4), (1.0, 2.2), (0.5, 2.9), (0.15, 3.6), (0.0, 4.4)], ORN, 10).T(ccx, H1 + 1.6, ccz))
    return stone, dark


def portal():
    """The Baroque portal over the main door, on the tower axis (Gran Vía front)."""
    x0, x1, z0, z1 = TOWER
    cx, zf = (x0 + x1) / 2, z1
    r = Run((cx - 10, zf), (cx + 10, zf), (0, 1))
    out = [r.box(7.0, 13.0, 0.0, 5.6, 0.0, 0.5, jit(GRAN)),                 # door frame
           r.box(7.4, 12.6, 9.0, 20.5, 0.0, 0.6, jit(ORN)),                 # the aedicule through three floors
           r.box(6.8, 13.2, 8.8, 9.4, 0.0, 1.3, jit(ORN)),                  # balcony slab
           r.box(9.0, 11.0, 20.5, 23.5, 0.0, 0.5, jit(ORN))]                # crest
    out.append(r.put(lathe([(0, 0), (0.6, 0.2), (0.4, 1.0), (0, 1.3)], ORN, 8).T(10.0, 23.5, 0.3)))
    dk = [r.face(8.6, 11.4, 0.0, 4.8, DARK, d=0.52), r.face(9.0, 11.0, 9.6, 13.4, DARK, d=0.62)]
    return out, dk


def main():
    out = ROOT / "models" / "Q2092717"
    out.mkdir(parents=True, exist_ok=True)
    stone, dark, roof = [], [], []
    for _, h, ring in PARTS:
        s, d, r = build_part(ring, h)
        stone += s; dark += d; roof += r
    s, d = tower(); stone += s; dark += d
    s, d = portal(); stone += s; dark += d
    # lift-motor pavilions on the roof of the core
    for (x, z, w, dd) in ((-11.0, -17.5, 4.0, 3.5), (13.0, -9.0, 3.5, 3.5)):
        stone.append(box(w, 3.0, dd, jit(STONE_B)).T(x, 50.0, z))
    m_stone, m_dark, m_roof = merge(stone), merge(dark), merge(roof)
    tris, size = write_glb(out / "model.glb", [("edificio_telefonica", m_stone), ("roofs", m_roof),
                                               ("windows", m_dark, {"material": "glass"})],
                           roughness=0.85, materials={"glass": GLASS})
    allv = np.vstack([m_stone.v, m_roof.v, m_dark.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
