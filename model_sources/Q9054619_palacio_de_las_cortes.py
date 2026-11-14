"""
Palacio de las Cortes / Congreso de los Diputados (Q9054619), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Narciso Pascual y Colomer (1814–1870), 1843–1850, public domain. Pediment relief and the two
bronze lions by Ponciano Ponzano (1813–1877); the lions were cast in 1866.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 4545711 (the historic palace only; the modern extensions across Floridablanca are
  separate buildings): outline 55.6 m wide (Carrera de San Jerónimo) × 66 m deep, plus a rear projection
  29.7 × 5.6 m and the portico with its podium and steps (to 7.1 m in front of the façade); long axis at
  19.4°. The portico, the steps (they come down towards 199.4°) and Carrera de San Jerónimo are on the
  south-south-west side, so the main façade faces 199.4°. Four light wells (courtyards), the hemicycle (a
  half circle of ~15 m radius with a glazed fan in its centre), the inner ranges, the glazed corner roofs and
  the skylight behind the portico all come from the OSM building:parts (relative layout; the OSM heights are
  too low, see below) and were checked on the PNOA orthophoto (IGN, CC BY 4.0).
- Heights measured on a near-frontal photo (Wikimedia Commons «Congreso de los Diputados (España) 17.jpg»),
  scale from the 55.6 m façade, ±5 %: rusticated ground floor to the string course 9.6 m, main floor windows
  10.5–13.3 m, main cornice 15.9–16.35 m, attic windows 16.4–17.6 m, attic coping 19.1 m; portico podium
  2.6 m, six Corinthian columns ~1.16 m in diameter to 13.0 m, entablature with «CONGRESO DE LOS DIPUTADOS»,
  pediment apex 20.9 m; lions ~2.1 m high on pedestals beside the steps. Four window bays on each wing of the
  main façade with pilasters between them on the main floor.
- Roofs: low hipped zinc roofs (eaves at the coping, ~22–29° pitch) over the perimeter wings and the inner
  ranges, a low dome over the hemicycle with its glazed fan and the round skylights seen on the orthophoto.
Stone: light limestone/granite (cream-grey), grey granite plinth, zinc-grey roofs, bronze lions and doors.

Axes (docs/MODEL_SPEC.md): +Y up, the main façade (portico) faces +Z, +X points to 109.4° (east-south-east,
calle de Fernanflor), origin on the ground at the centroid of the OSM outline.

    python model_sources/Q9054619_palacio_de_las_cortes.py   -> models/Q9054619/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, cap, walls, offset, write_glb, GLASS  # noqa: E402

rng = np.random.default_rng(1850)

STONE = np.array((0.83, 0.80, 0.73))      # the light stone of the façades
STONE_L = (0.88, 0.86, 0.80)              # columns, surrounds, cornices
GROOVE = (0.62, 0.59, 0.53)               # rustication joints
GRANITE = (0.60, 0.59, 0.56)              # plinth, podium, steps
MARBLE = (0.94, 0.93, 0.89)               # pediment relief
WIN = (0.15, 0.22, 0.23)                  # windows (dark green shutters / glass)
ROOF = (0.62, 0.63, 0.65)                 # zinc (OSM roof:colour #D1D1D1, light grey on the orthophoto)
DECK = (0.50, 0.51, 0.52)                 # flat roofs
GLASS_C = (0.40, 0.50, 0.56)
BRONZE = (0.24, 0.21, 0.16)
DOOR = (0.56, 0.43, 0.20)
INK = (0.18, 0.17, 0.16)
FOUND = -3.0

# Heights (m)
PLINTH_TOP = 1.0
STRING = (9.4, 9.85)
CORN_A = (15.3, 15.9)        # architrave/frieze band of the main cornice
CORN_B = (15.9, 16.35)       # corona
COPE = (18.8, 19.1)          # attic coping = the roof deck level
E = COPE[1]                  # eaves of every roof
POD = 2.6                    # portico podium
COL_TOP = 13.0
PED_BASE, PED_APEX = 15.9, 20.9

# OSM relation 4545711 in model x/z (heading 199.4°), the portico/steps points removed (built separately)
BODY = [(-27.795, 30.695), (-21.307, 30.67), (-13.547, 30.657), (-13.55, 30.493), (-10.294, 30.489),
        (-9.339, 30.494), (9.05, 30.458), (10.128, 30.455), (14.114, 30.452), (14.117, 30.616),
        (21.283, 30.6), (27.795, 30.59), (27.804, 0.108), (27.805, -1.447), (27.817, -30.082), (21.325, -30.068),
        (14.814, -30.059), (14.823, -35.63), (-14.848, -35.565), (-14.856, -30.017), (-21.281, -29.992),
        (-27.768, -29.991), (-27.788, -20.719), (-27.792, -2.282), (-27.798, 2.912)]
COURT_BACK = [[(14.815, -16.615), (14.797, -9.333), (19.401, -9.342), (19.4, -16.628)],
              [(-19.249, -16.274), (-19.25, -8.869), (-14.817, -8.876), (-14.811, -16.271)]]
# front light wells: a courtyard (x 14.2–19.1, z 12–17.7, down to the ground) joined to a lower hall roof
# (x 7.3–14.2, z 3.4–17.7, OSM parts 325705223/4 «height 8» -> 13.5 m here)
HALL_Y = 13.5


def well(s):
    return [(s * 7.3, 3.4), (s * 14.2, 3.4), (s * 14.2, 12.0), (s * 19.1, 12.0), (s * 19.1, 17.7), (s * 7.3, 17.7)]


HOLES = COURT_BACK + [well(1), well(-1)]
CH_Z, CH_R = -10.05, 14.6    # the hemicycle (OSM 1124477176: x ±14.8, z −24.9…4.8)


def sc(c, k=0.025):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, k)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def mirror_x(m):
    n = m.copy(); n.v[:, 0] *= -1; n.f = n.f[:, ::-1]; return n


def sphere(r, c, seg=12, rings=7):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def vquad(a, b, y0, y1, c, inside):
    """Vertical quad from a to b (x, z), facing the point inside."""
    V = [(a[0], y0, a[1]), (b[0], y0, b[1]), (b[0], y1, b[1]), (a[0], y1, a[1])]
    m = Mesh(V, [(0, 1, 2), (0, 2, 3)], c)
    n = np.cross(np.subtract(V[1], V[0]), np.subtract(V[3], V[0]))
    mid = (np.add(a, b)) / 2
    if n[0] * (inside[0] - mid[0]) + n[2] * (inside[1] - mid[1]) < 0:
        m.f = m.f[:, ::-1]
    return m


def band(ring, d, y0, y1, c, bottom=True):
    """A cornice/plinth course: the walls of the ring grown by d, and the strips on top (and below)."""
    o = offset(ring, d)
    out = [walls(o, y0, y1, c), cap(o, y1, c, holes=[ring], up=True)]
    if bottom:
        out.append(cap(o, y0, c, holes=[ring], up=False))
    return merge(out)


# ── Oriented boxes along a façade segment ────────────────────────────────────
class Seg:
    def __init__(self, a, b):
        self.a = np.array(a, float); b = np.array(b, float)
        self.L = float(np.linalg.norm(b - self.a)); self.t = (b - self.a) / self.L
        n = np.array([self.t[1], -self.t[0]])
        mid = (self.a + b) / 2
        self.n = n if np.dot(n, mid) > 0 else -n          # outward (away from the origin)

    def box(self, u0, u1, y0, y1, d0, d1, c):
        m = B(u0, u1, y0, y1, d0, d1, c)                   # local: x = u, z = d
        u, d = m.v[:, 0].copy(), m.v[:, 2].copy()
        P = self.a[None, :] + u[:, None] * self.t[None, :] + d[:, None] * self.n[None, :]
        m.v[:, 0], m.v[:, 2] = P[:, 0], P[:, 1]
        if self.t[0] * self.n[1] - self.t[1] * self.n[0] > 0:                   # keep the winding (mirror otherwise)
            m.f = m.f[:, ::-1]
        return m


def window(s, u, y, w, h, kind):
    """A window: dark opening, stone surround and (main floor) a straight cornice over it."""
    out = [s.box(u - w / 2, u + w / 2, y, y + h, -0.1, 0.035, WIN)]
    f = 0.22
    out += [s.box(u - w / 2 - f, u - w / 2, y - 0.05, y + h, -0.05, 0.09, STONE_L),
            s.box(u + w / 2, u + w / 2 + f, y - 0.05, y + h, -0.05, 0.09, STONE_L),
            s.box(u - w / 2 - f, u + w / 2 + f, y + h, y + h + f, -0.05, 0.09, STONE_L),
            s.box(u - w / 2 - 0.3, u + w / 2 + 0.3, y - 0.2, y - 0.05, -0.05, 0.16, STONE_L)]
    if kind == "main":
        out += [s.box(u - w / 2 - 0.3, u + w / 2 + 0.3, y + h + f, y + h + f + 0.5, -0.05, 0.12, sc(STONE)),
                s.box(u - w / 2 - 0.45, u + w / 2 + 0.45, y + h + f + 0.5, y + h + f + 0.72, -0.05, 0.32, STONE_L)]
    if kind == "main":                                    # balustered balcony sill
        out.append(s.box(u - w / 2 - 0.15, u + w / 2 + 0.15, y - 0.75, y - 0.2, -0.05, 0.12, STONE_L))
    return out


def bay(s, u, door=False):
    out = []
    if door:
        out += [s.box(u - 1.1, u + 1.1, PLINTH_TOP - 1.0 + 0.0, 5.6, -0.1, 0.035, DOOR)]
        out += [s.box(u - 1.45, u - 1.1, 0, 5.9, -0.05, 0.12, STONE_L), s.box(u + 1.1, u + 1.45, 0, 5.9, -0.05, 0.12, STONE_L),
                s.box(u - 1.6, u + 1.6, 5.9, 6.5, -0.05, 0.3, STONE_L)]
    else:
        out += window(s, u, 3.75, 1.5, 3.1, "ground")
    out += window(s, u, 10.5, 1.5, 2.8, "main")
    out += window(s, u, 16.5, 1.25, 1.1, "attic")
    return out


def pilaster(s, u, y0, y1, w=0.75, d=0.12):
    return [s.box(u - w / 2, u + w / 2, y0, y1, -0.05, d, sc(STONE_L, 0.01))]


def facades():
    out = []
    fl = Seg((-27.795, 30.695), (-13.547, 30.657))
    fc = Seg((-13.55, 30.493), (14.114, 30.452))
    fr = Seg((14.117, 30.616), (27.795, 30.59))
    # main façade: four bays a side (x ±11.4, ±16.4, ±20.7, ±25.0), pilasters between them on the main floor
    for x in (-25.0, -20.7, -16.4):
        out += bay(fl, x + 27.795)
    for x in (16.4, 20.7, 25.0):
        out += bay(fr, x - 14.117)
    for x in (-11.4, 11.4):
        out += bay(fc, x + 13.55)
    for x in (-6.6, 6.6):                                 # behind the portico
        out += window(fc, x + 13.55, 3.6, 1.6, 4.2, "ground")
        out += window(fc, x + 13.55, 10.5, 1.5, 2.8, "main")
    out.append(fc.box(13.55 - 1.3, 13.55 + 1.3, POD, POD + 5.2, -0.1, 0.04, DOOR))       # the bronze doors
    out.append(fc.box(13.55 - 1.7, 13.55 + 1.7, POD + 5.2, POD + 5.8, -0.05, 0.25, STONE_L))
    for x in (-22.85, -18.55, 18.55, 22.85):
        s, u = (fl, x + 27.795) if x < 0 else (fr, x - 14.117)
        out += pilaster(s, u, STRING[1], CORN_A[0])
    for x in (-13.75, 13.75):
        out += pilaster(fc, x + 13.55, STRING[1], CORN_A[0], w=0.9)
    for x in (-8.25, -4.95, -1.65, 1.65, 4.95, 8.25):     # pilasters answering the columns
        out += pilaster(fc, x + 13.55, POD, COL_TOP, w=1.0, d=0.15)
    # corner pilasters
    for s, u in ((fl, 0.45), (fr, fr.L - 0.45)):
        out += pilaster(s, u, STRING[1], CORN_A[0], w=0.9, d=0.14)
    # sides (13 bays, the west side has the everyday entrance in the middle), back and the rear projection
    se = Seg((27.795, 30.59), (27.817, -30.082))
    sw = Seg((-27.768, -29.991), (-27.795, 30.695))
    for s, door_side in ((se, False), (sw, True)):
        for i in range(13):
            u = 4.35 + i * 4.33
            out += bay(s, u, door=(door_side and i == 6))
            if i < 12:
                out += pilaster(s, u + 2.165, STRING[1], CORN_A[0])
        out += pilaster(s, 0.45, STRING[1], CORN_A[0], w=0.9, d=0.14)
        out += pilaster(s, s.L - 0.45, STRING[1], CORN_A[0], w=0.9, d=0.14)
    for a, b in (((27.817, -30.082), (14.814, -30.059)), ((-14.856, -30.017), (-27.768, -29.991))):
        s = Seg(a, b)
        for u in (2.6, 6.5, 10.4):
            out += bay(s, u)
    rear = Seg((14.823, -35.63), (-14.848, -35.565))
    for i in range(6):
        out += bay(rear, 2.6 + i * 4.9)
    for a, b in (((14.814, -30.059), (14.823, -35.63)), ((-14.848, -35.565), (-14.856, -30.017))):
        out += bay(Seg(a, b), 2.8)
    return out


def body():
    out = [walls(BODY, FOUND, COPE[0], sc(STONE, 0.0))]
    out.append(band(BODY, 0.12, FOUND, PLINTH_TOP, GRANITE, bottom=False))
    for y in np.arange(PLINTH_TOP + 0.6, STRING[0] - 0.1, 0.62):          # rustication joints
        out.append(walls(offset(BODY, 0.015), y, y + 0.06, GROOVE))
    out.append(band(BODY, 0.2, *STRING, STONE_L))
    out.append(band(BODY, 0.08, *CORN_A, sc(STONE, 0.0)))
    out.append(band(BODY, 0.6, *CORN_B, STONE_L))
    out.append(band(BODY, 0.25, *COPE, STONE_L))
    out.append(cap(BODY, E, DECK, holes=HOLES))                          # the roof deck
    # light wells
    for ring in COURT_BACK:
        out.append(walls(ring, 0.0, E, sc(STONE, 0.0), hole=True))
        out.append(cap(ring, 0.05, GRANITE))
    for s in (1, -1):
        cx = s * 14.2
        pts = well(s)
        segs = [(pts[0], pts[1], HALL_Y), (pts[1], pts[2], HALL_Y), (pts[2], pts[3], 0.0), (pts[3], pts[4], 0.0),
                (pts[4], (cx, 17.7), 0.0), ((cx, 17.7), pts[5], HALL_Y), (pts[5], pts[0], HALL_Y)]
        for a, b, y0 in segs:
            ins = (s * 16.6, 14.9) if y0 == 0.0 else (s * 10.75, 10.5)      # a point inside that part of the well
            out.append(vquad(a, b, y0, E, sc(STONE, 0.0), ins))
        out.append(vquad((cx, 12.0), (cx, 17.7), 0.0, HALL_Y, sc(STONE, 0.0), (s * 16.6, 14.9)))   # step into the court
        hall = [(s * 7.3, 3.4), (cx, 3.4), (cx, 17.7), (s * 7.3, 17.7)]
        out.append(cap(hall, HALL_Y, DECK))
        out.append(cap([(s * 14.2, 12.0), (s * 19.1, 12.0), (s * 19.1, 17.7), (s * 14.2, 17.7)], 0.05, GRANITE))
        # the glazed vault over the hall (Salón de Conferencias / galleries)
        out.append(vault_x(min(s * 8.2, s * 13.3), max(s * 8.2, s * 13.3), 4.4, 16.7, HALL_Y, 1.6, GLASS_C, along="z"))
    return out


# ── Roofs ────────────────────────────────────────────────────────────────────
def hip(x0, x1, z0, z1, y0, k, c, along="x"):
    """Hipped roof over a rectangle, pitch k (rise / run), the ridge along x (or z); a pyramid if square."""
    if along == "z":
        m = hip(z0, z1, x0, x1, y0, k, c, "x")
        m.v = m.v[:, [2, 1, 0]]; m.f = m.f[:, ::-1]
        return m
    W, L = z1 - z0, x1 - x0
    h = k * W / 2
    zc = (z0 + z1) / 2
    r0, r1 = x0 + W / 2, x1 - W / 2
    if r1 < r0:
        r0 = r1 = (x0 + x1) / 2
        h = k * min(W, L) / 2
    V = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (r0, y0 + h, zc), (r1, y0 + h, zc)]
    F = [(0, 4, 5), (0, 5, 1), (3, 2, 5), (3, 5, 4), (0, 3, 4), (1, 5, 2)]
    return Mesh(V, F, c)


def vault_x(x0, x1, z0, z1, y0, h, c, along="x", seg=10):
    """Segmental barrel vault over a rectangle, its axis along x (or z), with its two end lunettes."""
    if along == "z":
        m = vault_x(z0, z1, x0, x1, y0, h, c, "x", seg)
        m.v = m.v[:, [2, 1, 0]]; m.f = m.f[:, ::-1]
        return m
    W = z1 - z0
    R = (W * W / 4 + h * h) / (2 * h)
    a0 = math.asin((W / 2) / R)
    angs = np.linspace(-a0, a0, seg + 1)
    zc, yc = (z0 + z1) / 2, y0 + h - R
    prof = [(zc + R * math.sin(a), yc + R * math.cos(a)) for a in angs]
    V, F = [], []
    for z, y in prof:
        V += [(x0, y, z), (x1, y, z)]
    for i in range(seg):
        a = 2 * i
        F += [(a, a + 3, a + 1), (a, a + 2, a + 3)]
    m = [Mesh(V, F, c)]
    for x, sgn in ((x0, -1), (x1, 1)):
        P = [(x, y, z) for z, y in prof]
        Fl = [(0, i, i + 1) for i in range(1, seg)]
        mm = Mesh(P, Fl, c)
        if sgn < 0:
            mm.f = mm.f[:, ::-1]
        m.append(mm)
    out = merge(m)
    # make the shell face up/out: flip if the first triangle's normal points down
    v = out.v[out.f[0]]
    if np.cross(v[1] - v[0], v[2] - v[0])[1] < 0:
        out.f = out.f[:, ::-1]
    return out


def half_dome(cx, cz, prof, c, seg=24, a0=math.pi, a1=2 * math.pi):
    """A surface of revolution over an angular range (the back half by default: z < cz)."""
    angs = np.linspace(a0, a1, seg + 1)
    V, F = [], []
    for r, y in prof:
        V += [(cx + r * math.cos(t), y, cz + r * math.sin(t)) for t in angs]
    n = seg + 1
    for i in range(len(prof) - 1):
        for j in range(seg):
            p, q = i * n + j, i * n + j + 1
            F += [(p, q + n, p + n), (p, q, q + n)]
    m = Mesh(V, F, c)
    v = m.v[m.f[0]]
    if np.cross(v[1] - v[0], v[2] - v[0])[1] < 0:
        m.f = m.f[:, ::-1]
    return m


def roofs():
    out = []
    out.append(hip(-27.5, 27.5, 17.7, 30.3, E, 0.42, ROOF))                 # front range
    out.append(hip(-27.5, 27.5, -29.7, -24.9, E, 0.42, sc(ROOF, 0.01)))     # back range
    for s in (1, -1):                                                       # side ranges
        out.append(hip(min(s * 19.4, s * 27.5), max(s * 19.4, s * 27.5), -29.7, 30.3, E, 0.55, sc(ROOF, 0.01), along="z"))
    out.append(hip(-14.5, 14.5, -35.3, -24.9, E, 0.38, sc(ROOF, 0.01)))     # rear projection
    out.append(hip(-7.2, 7.2, 0.3, 24.0, E, 0.30, sc(ROOF, 0.01), along="z"))     # central range
    out.append(hip(-23.45, 23.45, -2.6, 3.3, E, 0.50, sc(ROOF, 0.01)))     # cross range in front of the chamber
    # the hemicycle: a barrel across its straight part, a low dome over the half circle, the glazed fan
    out.append(vault_x(-14.6, 14.6, CH_Z, -2.6, E, 1.35, ROOF))
    prof = [(CH_R, E), (12.0, E + 0.55), (9.0, E + 1.0), (6.0, E + 1.3), (4.3, E + 1.45)]
    out.append(half_dome(0, CH_Z, prof, ROOF))
    # the flat face where the half dome meets the barrel (z = CH_Z)
    P = [(r * s, y) for s in (-1,) for r, y in prof] + [(r, y) for r, y in reversed(prof)]
    V = [(x, y, CH_Z) for x, y in P]
    F = [(0, i, i + 1) for i in range(1, len(V) - 1)]
    out.append(Mesh(V, F, ROOF))
    out.append(cap([(4.3, CH_Z), (-4.3, CH_Z)] + [(4.3 * math.cos(t), CH_Z + 4.3 * math.sin(t)) for t in np.linspace(math.pi, 2 * math.pi, 13)][1:-1],
                   E + 1.45, ROOF))
    return out


def glass():
    out = []
    # the fan over the speaker's tribune: a glazed half cone with ribs
    fan = half_dome(0, CH_Z, [(4.3, E + 1.47), (2.0, E + 2.0), (0.2, E + 2.3)], GLASS_C, seg=18)
    out.append(fan)
    ribs = []
    for t in np.linspace(math.pi, 2 * math.pi, 10):
        ribs.append(tube([(4.3 * math.cos(t), E + 1.5, CH_Z + 4.3 * math.sin(t)), (2.0 * math.cos(t), E + 2.04, CH_Z + 2.0 * math.sin(t)),
                          (0.2 * math.cos(t), E + 2.33, CH_Z + 0.2 * math.sin(t))], 0.05, INK, 4))
    # round skylights on the dome
    for t in np.linspace(math.pi + 0.2, 2 * math.pi - 0.2, 11):
        r = 9.6
        y = E + 0.93
        out.append(cyl(0.55, 0.42, 0.5, GLASS_C, 10).T(r * math.cos(t), y, CH_Z + r * math.sin(t)))
    # glazed corner roofs behind the hemicycle (OSM 1124477175/178)
    for s in (1, -1):
        arc = [(s * CH_R * math.cos(t), CH_Z + CH_R * math.sin(t)) for t in np.linspace(math.radians(-66.2), math.radians(-26.7), 8)]
        poly = [(s * 14.75, -16.6), (s * 14.75, -24.85), (s * 5.9, -24.85)] + arc
        out.append(cap(poly, E + 0.06, GLASS_C))
    # skylight behind the portico (OSM 1124495669) and the glass on the halls is in body()
    out.append(B(-3.3, 3.3, E, 19.9, 24.5, 28.0, sc(STONE, 0.0)))
    out.append(hip(-3.3, 3.3, 24.5, 28.0, 19.9, 0.5, GLASS_C))
    return out, ribs


# ── Portico ──────────────────────────────────────────────────────────────────
FONT = {
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
}


def inscription(text, zf, y_mid, px=0.07):
    out = []
    w = (len(text) * 6 - 1) * px
    col = 0
    for ch in text:
        if ch == " ":
            col += 4; continue
        for r_, row in enumerate(FONT[ch]):
            run = 0
            for c_, bit in enumerate(row + "0"):
                if bit == "1":
                    run += 1; continue
                if run:
                    x = -w / 2 + (col + c_ - run / 2) * px
                    y = y_mid + 3.5 * px - r_ * px - px / 2
                    out.append(B(x - run * px * 0.49, x + run * px * 0.49, y - px * 0.49, y + px * 0.49, zf - 0.02, zf + 0.03, INK))
                run = 0
        col += 6
    return out


def corinthian(x, z):
    r = 0.58
    out = [cyl(r * 1.3, r * 1.25, 0.25, STONE_L, 16).T(x, POD, z),
           cyl(r * 1.18, r * 1.08, 0.25, STONE_L, 16).T(x, POD + 0.25, z),
           B(x - 0.8, x + 0.8, POD - 0.001, POD + 0.12, z - 0.8, z + 0.8, STONE_L)]
    y0, y1 = POD + 0.5, COL_TOP - 1.35
    out.append(cyl(r, r * 0.86, y1 - y0, sc(STONE_L, 0.01), 18, caps=False).T(x, y0, z))
    # the capital: an acanthus bell (two rows of leaves as bulges) and the concave abacus
    out.append(lathe([(r * 0.86, 0), (r * 1.02, 0.35), (r * 1.15, 0.75), (r * 1.3, 1.05), (r * 1.25, 1.12), (0, 1.12)], STONE_L, 16).T(x, y1, z))
    for k in range(8):
        t = k * math.pi / 4 + math.pi / 8
        out.append(sphere(0.2, STONE_L, 6, 4).S(1, 1.6, 1).T(x + r * 1.05 * math.cos(t), y1 + 0.42, z + r * 1.05 * math.sin(t)))
    for k in range(4):
        t = k * math.pi / 2 + math.pi / 4
        out.append(sphere(0.18, STONE_L, 6, 4).T(x + r * 1.45 * math.cos(t), y1 + 1.0, z + r * 1.45 * math.sin(t)))
    out.append(B(x - 0.85, x + 0.85, y1 + 1.12, COL_TOP, z - 0.85, z + 0.85, STONE_L))
    return out


def figure(x, y, z, h, c, pose="stand", side=1):
    """A draped figure in high relief (flattened in depth): standing, seated or reclining (head towards side)."""
    out = []
    if pose == "recline":                                  # lying along x, propped on an elbow
        out.append(sphere(1.0, c, 10, 6).S(0.55 * h, 0.16 * h, 0.14 * h).T(-side * 0.1 * h, 0.15 * h, 0))
        out.append(sphere(1.0, c, 8, 6).S(0.16 * h, 0.2 * h, 0.13 * h).T(side * 0.35 * h, 0.3 * h, 0))
        out.append(sphere(0.085 * h, c, 8, 5).T(side * 0.42 * h, 0.52 * h, 0))
        out.append(tube([(side * 0.3 * h, 0.35 * h, 0.05 * h), (side * 0.5 * h, 0.1 * h, 0.08 * h)], 0.045 * h, c, 5))
    elif pose == "sit":
        out.append(B(-0.2 * h, 0.2 * h, 0, 0.32 * h, -0.12 * h, 0.08 * h, c))                       # seat / drapery
        out.append(sphere(1.0, c, 8, 6).S(0.13 * h, 0.12 * h, 0.2 * h).T(0, 0.36 * h, 0.12 * h))     # lap
        out.append(sphere(1.0, c, 10, 6).S(0.14 * h, 0.22 * h, 0.1 * h).T(0, 0.58 * h, -0.02 * h))   # torso
        out.append(sphere(0.08 * h, c, 8, 5).T(0, 0.86 * h, 0))
        out.append(tube([(side * 0.12 * h, 0.7 * h, 0.03 * h), (side * 0.24 * h, 0.5 * h, 0.1 * h), (side * 0.3 * h, 0.62 * h, 0.12 * h)], 0.035 * h, c, 5))
    else:
        out.append(lathe([(0.15 * h, 0), (0.13 * h, 0.3 * h), (0.1 * h, 0.5 * h), (0.11 * h, 0.66 * h), (0.07 * h, 0.76 * h), (0, 0.78 * h)], c, 10).S(1, 1, 0.6))
        out.append(sphere(0.07 * h, c, 8, 5).T(0, 0.85 * h, 0))
        out.append(tube([(side * 0.09 * h, 0.72 * h, 0.02 * h), (side * 0.2 * h, 0.6 * h, 0.08 * h), (side * 0.26 * h, 0.8 * h, 0.1 * h)], 0.03 * h, c, 5))
    m = merge(out)
    return [m.T(x, y, z)]


def portico():
    out = []
    ZF = 30.47                         # the façade behind
    ZC = 32.9                          # column axis
    ZE = 33.7                          # front of the entablature
    # podium, steps, lion pedestals
    out.append(B(-10.2, 10.2, FOUND, POD - 0.25, ZF - 0.3, 34.3, GRANITE))
    out.append(B(-10.25, 10.25, POD - 0.25, POD, ZF - 0.3, 34.35, STONE_L))
    n = 13
    for i in range(n):
        z0 = 34.3 + i * (3.4 / n)
        out.append(B(-7.3, 7.3, -0.6, POD - i * POD / n, 34.36 if i == 0 else z0 - 0.02, z0 + 3.4 / n, sc(GRANITE, 0.015)))
    for s in (1, -1):
        x0, x1 = sorted((s * 7.3, s * 9.3))
        out.append(B(x0, x1, FOUND, POD - 0.3, 34.36, 37.7, GRANITE))
        out.append(B(x0 - 0.08, x1 + 0.08, POD - 0.3, POD + 0.05, 34.36, 37.78, STONE_L))
        out += lion(s * 8.3, POD, 36.0, s)
    # columns
    for x in (-8.25, -4.95, -1.65, 1.65, 4.95, 8.25):
        out += corinthian(x, ZC)
    # entablature: architrave, frieze with the inscription, cornice
    out.append(B(-9.2, 9.2, COL_TOP, 13.8, ZF - 0.2, ZE - 0.1, STONE_L))
    out.append(B(-9.2, 9.2, 13.8, 14.9, ZF - 0.2, ZE, sc(STONE, 0.0)))
    out += inscription("CONGRESO DE LOS DIPUTADOS", ZE, 14.35)
    out.append(B(-9.6, 9.6, 14.9, 15.3, ZF - 0.2, ZE + 0.2, STONE_L))
    out.append(B(-10.1, 10.1, 15.3, PED_BASE, ZF - 0.2, ZE + 0.55, STONE_L))
    # pediment: tympanum, raking cornices, roof running back into the building
    hw = 9.4
    zt = ZE - 0.05
    V = [(-hw, PED_BASE, zt), (hw, PED_BASE, zt), (0, PED_APEX - 0.45, zt)]
    out.append(Mesh(V, [(0, 1, 2)], sc(STONE, 0.0)))
    for s in (1, -1):
        out.append(tube([(s * 10.25, PED_BASE + 0.15, ZE + 0.25), (0, PED_APEX - 0.15, ZE + 0.25)], 0.33, STONE_L, 6))
    rw = 10.2
    zb = 25.6
    V = [(-rw, PED_BASE, zb), (-rw, PED_BASE, ZE + 0.5), (0, PED_APEX, ZE + 0.5), (0, PED_APEX, zb),
         (rw, PED_BASE, zb), (rw, PED_BASE, ZE + 0.5)]
    out.append(Mesh(V, [(0, 1, 2), (0, 2, 3), (4, 3, 2), (4, 2, 5)], ROOF))
    # the relief «España abrazando la Constitución»: a central group, allegories getting lower to the corners
    zr = zt + 0.05
    yb = PED_BASE + 0.25
    out.append(B(-0.75, 0.75, yb, yb + 0.9, zr - 0.3, zr + 0.25, MARBLE))                 # the throne
    out += figure(0, yb + 0.35, zr + 0.1, 3.4, MARBLE, "sit")                         # Spain with the Constitution
    out.append(B(0.35, 1.0, yb + 2.0, yb + 2.7, zr + 0.25, zr + 0.4, MARBLE))           # the open book
    for s_ in (1, -1):
        out += figure(s_ * 1.7, yb, zr, 3.3, MARBLE, "stand", side=-s_)
        out += figure(s_ * 3.0, yb, zr, 2.8, MARBLE, "stand", side=s_)
        out += figure(s_ * 4.4, yb, zr, 2.6, MARBLE, "sit", side=-s_)
        out += figure(s_ * 5.9, yb, zr, 1.9, MARBLE, "sit", side=s_)
        out += figure(s_ * 7.6, yb, zr, 2.6, MARBLE, "recline", side=-s_)
    # the flag on the apex
    out.append(cyl(0.06, 0.05, 4.0, (0.85, 0.85, 0.85), 6).T(0, PED_APEX - 0.1, ZE + 0.1))
    fy = PED_APEX + 3.85
    for (y0, y1), c in (((fy - 1.1, fy - 0.825), (0.67, 0.08, 0.10)), ((fy - 0.825, fy - 0.275), (0.95, 0.75, 0.05)),
                        ((fy - 0.275, fy), (0.67, 0.08, 0.10))):
        out.append(B(0.06, 1.7, y0, y1, ZE + 0.08, ZE + 0.12, c))
    return out


def lion(x, y, z, side):
    """Stylised standing lion facing +Z, one fore paw on a ball (bronze, Ponzano 1866)."""
    c = BRONZE
    out = [sphere(1.0, c, 14, 8).S(0.5, 0.48, 1.25).T(x, y + 1.15, z - 0.1),            # body
           sphere(1.0, c, 12, 8).S(0.58, 0.62, 0.55).T(x, y + 1.45, z + 0.8),           # chest and mane
           sphere(0.34, c, 10, 6).S(1, 1, 1.15).T(x, y + 1.85, z + 1.15),               # head
           sphere(0.16, c, 8, 4).S(1, 0.8, 1.2).T(x, y + 1.75, z + 1.45)]               # muzzle
    for dx in (-0.28, 0.28):
        for dz, lift in ((0.85, 0.0), (-0.95, 0.0)):
            paw_up = (dz > 0 and dx * side < 0)
            yb = y + (0.42 if paw_up else 0.0)
            out.append(cyl(0.15, 0.17, 1.15 - (yb - y), c, 8).T(x + dx, yb, z + dz + (0.15 if paw_up else 0)))
    out.append(sphere(0.3, c, 10, 6).T(x - side * 0.28, y + 0.3, z + 1.05))               # the ball
    out.append(tube([(x, y + 1.25, z - 1.3), (x + 0.15 * side, y + 0.8, z - 1.55), (x + 0.2 * side, y + 0.35, z - 1.45)], 0.06, c, 5))
    return out


def main():
    out = ROOT / "models" / "Q9054619"
    out.mkdir(parents=True, exist_ok=True)
    stone = merge(body() + facades() + portico() + roofs())
    gl, dark = glass()
    gl = merge(gl)
    dark = merge(dark)
    parts = [("palacio_de_las_cortes", stone), ("skylights", gl, {"material": "glass"}), ("skylight_frames", dark)]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.9, materials={"glass": GLASS})
    allv = np.vstack([p[1].v for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
