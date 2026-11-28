"""
Acueducto de Segovia (Q244947) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: the Roman aqueduct bridge of Segovia (late 1st – early 2nd century AD), granite ashlar laid without
mortar; 36 arches rebuilt in the 15th century under the Catholic Monarchs (prior Pedro Mesa); the niches over the
central arches got their statues in the 16th century. Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 1366085092 (waterway=ditch, layer 2, «Acequia del Acueducto de Segovia»): the axis of the
  channel on top of the arcade, 962.7 m from where it comes out of the ground (south-east, Avenida del Padre Claret)
  to where it goes underground again inside the walls; way 503134835 (building, wikidata Q244947) is its outline.
  The axis bends at Plaza de Día Sanz (s = 605.9 m, 52°) and runs north-north-west across the Plaza del Azoguejo to
  the Postigo del Consuelo (s = 873 m), then along a low wall and west along the court to its end.
- Wikipedia (es, en): 167 arches — 75 single ones from the arrival to Plaza de Día Sanz, 44 double arcades (88
  arches), and 4 more single ones; 28 m high at the Azoguejo; channel on 1 % slope; lower piers 1.8 × 2.5 m at the
  top and 2.4 × 3 m at the base, stepping out downwards; two niches over the three tallest arches, a panel where
  the bronze letters of the inscription were; 36 pointed arches rebuilt in the 15th century in the first stretch.
- PNOA orthophoto (shadows of the arches): bays of 6.0 m in the double arcade and the tall single arches next to
  Día Sanz, about 5.2–5.8 m further south-east; the arches begin about 180 m from the start (before that a solid
  wall); no openings after the Postigo (a solid wall).
- Photos on Commons (near-frontal «Aqueduct Segovia statue.jpg», scaled by the 6.0 m bay): upper tier 8.6 m (upper
  piers 1.7 m, arches 4.3 m wide), attic 1.4 m, string course between the tiers, lower arches 4.0 m wide; the panel
  of the inscription 3 bays long and 1.75 m high on the string course, the niche (1.9 × 2.9 m) on the central pier.
- IGN MDT05 along the axis (5 m): 1016.4 m at the start, 982.0 m at the Azoguejo, 1004–1006 m behind the Postigo.
  The top of the channel is a straight slope of 1.005 %: 1018.0 m at the start, 1010.0 m at the Azoguejo (28 m over
  the square), 1008.3 m at the end. The heights of every pier follow from it and the ground.

Axes (docs/MODEL_SPEC.md): +Y up, y = 0 the ground of the Plaza del Azoguejo (982.0 m) under the central pier with
the niche (the origin, on the axis); +Z faces west-south-west (238°, the face with the Virgin towards the Mesón de
Cándido), +X runs along the Azoguejo arcade towards Día Sanz (148°).

    python model_sources/Q244947_acueducto_de_segovia.py   -> models/Q244947/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, lathe, triangulate, fbm, noise3, write_glb  # noqa: E402
from osm import Frame  # noqa: E402

rng = np.random.default_rng(98)

GRANITE = np.array((0.62, 0.595, 0.545))     # weathered granite ashlar, grey-beige
ATTIC = np.array((0.67, 0.615, 0.53))        # the rubble-faced attic, warmer
PANEL = np.array((0.70, 0.665, 0.60))        # the inscription panel and the niches (lighter, restored)
CHANNEL = np.array((0.36, 0.34, 0.31))       # the floor of the specus
WHITE = np.array((0.90, 0.88, 0.83))         # the resin replica of the Virgin

# OSM way 1366085092, south-east -> north-west
NODES = [(40.9444307, -4.1104147), (40.9454186, -4.111396), (40.9454647, -4.1114746), (40.9457961, -4.1120673),
         (40.9462723, -4.113529), (40.9464721, -4.1166551), (40.9479988, -4.1179118), (40.9485251, -4.1183079),
         (40.9485373, -4.1183181), (40.9488486, -4.1185871), (40.9487071, -4.1190656), (40.9487319, -4.1190911)]

# lowest ground (MDT05, m over 1000, across ±2.5 m of the axis) every 4 m of the axis from its south-east end
GROUND = [16.4, 16.1, 16.0, 16.0, 16.0, 15.8, 15.2, 15.1, 15.0, 15.0, 14.6, 14.1, 14.2, 14.0, 14.0, 13.5, 13.2, 13.1,
          13.0, 13.0, 12.9, 12.5, 12.2, 12.2, 12.0, 11.9, 11.5, 11.3, 11.2, 11.1, 10.8, 10.5, 10.4, 10.3, 10.0, 9.8,
          9.8, 9.8, 9.8, 9.5, 9.2, 9.2, 9.0, 8.9, 8.8, 8.6, 8.1, 7.6, 7.3, 7.3, 7.3, 7.0, 6.9, 7.0, 7.0, 6.9, 6.8, 6.6,
          6.4, 6.2, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 5.7, 5.1, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
          5.0, 5.0, 4.9, 5.0, 4.9, 4.9, 4.7, 4.6, 4.7, 4.6, 4.3, 4.4, 4.8, 5.0, 5.0, 5.0, 5.1, 5.4, 5.5, 5.6, 5.7, 5.7,
          5.3, 5.1, 5.1, 5.1, 5.2, 5.1, 5.0, 4.9, 4.5, 4.2, 4.0, 4.0, 4.0, 3.8, 3.3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.8,
          2.4, 2.1, 2.0, 2.0, 2.0, 1.9, 1.2, 1.0, 1.0, 1.0, 1.0, 0.9, 0.6, 0.4, 0.3, 0.1, 0.0, 0.0, -0.1, -0.4, -0.8,
          -0.9, -1.0, -1.0, -1.0, -1.0, -1.1, -1.1, -1.3, -1.9, -2.4, -2.9, -3.1, -3.5, -4.1, -4.6, -5.0, -5.3, -5.7,
          -6.1, -6.5, -6.8, -7.3, -7.9, -8.5, -9.1, -9.5, -10.0, -10.3, -10.8, -11.3, -11.8, -12.1, -12.5, -13.0, -13.6,
          -14.2, -14.7, -15.2, -15.7, -16.0, -16.0, -16.0, -16.2, -16.6, -16.9, -17.0, -17.0, -17.0, -17.3, -17.6,
          -17.9, -18.0, -18.0, -18.0, -18.0, -18.0, -18.0, -18.0, -17.8, -17.1, -16.0, -15.1, -14.4, -13.7, -13.1,
          -12.5, -11.6, -11.0, -10.4, -9.8, -9.1, -8.1, -5.0, 0.3, 3.0, 4.3, 4.6, 4.6, 4.8, 4.3, 4.1, 4.0, 4.2, 4.6,
          4.8, 5.2, 5.6, 5.3, 5.4, 5.6, 5.4, 5.5, 5.9, 5.4, 5.0, 5.0]

TOP0, SLOPE = 1018.0, 0.01005               # top of the channel: m a.s.l. at s = 0, and its fall per metre

# The arcade: pier positions along the axis (s, m)
S_ARCH0 = 180.0                              # the first arch (before it a solid wall)
SINGLE = (list(np.linspace(180.0, 207.7, 7)) + list(np.linspace(207.7, 341.7, 26))[1:]
          + list(np.linspace(341.7, 605.9, 45))[1:])                                  # 75 single arches
BAY_D = 6.0
DOUBLE = [605.9 + BAY_D * k for k in range(45)]                                       # 44 double arcades
LAST = [DOUBLE[-1] + BAY_D * k for k in range(5)]                                     # 4 single arches past them
S_END = 962.7
NICHE_K = 32                                                                          # the pier with the niches
S_ORIGIN = DOUBLE[NICHE_K]
POINTED = 36                                 # the first 36 single arches are the pointed ones of the 15th century

# ── The axis ─────────────────────────────────────────────────────────────────
F0 = Frame(*NODES[0])
EN = [np.array(F0.en(*p)) for p in NODES]
CUM = np.concatenate([[0.0], np.cumsum([np.linalg.norm(b - a) for a, b in zip(EN[:-1], EN[1:])])])
VERT_S = list(CUM)


def at_s(s):
    i = int(np.clip(np.searchsorted(CUM, s) - 1, 0, len(EN) - 2))
    a, b = EN[i], EN[i + 1]
    return a + (b - a) * (s - CUM[i]) / (CUM[i + 1] - CUM[i])


def _bearing(i):
    d = EN[i + 1] - EN[i]
    return math.degrees(math.atan2(d[0], d[1])) % 360


SEG_AZO = 5                                   # the Día Sanz – Azoguejo segment
HEADING = round((_bearing(SEG_AZO) - 90) % 360, 2)
LAT0, LON0 = F0.latlon(*at_s(S_ORIGIN))
FR = Frame(LAT0, LON0, HEADING)


def mxz(s):
    """Model (x, z) of the axis at s."""
    return np.array(FR.xz(*FR.en(*F0.latlon(*at_s(s)))))


def gnd(s):
    return float(np.interp(s, np.arange(len(GROUND)) * 4.0, GROUND)) + 1000.0


E_ORIGIN = gnd(S_ORIGIN)                      # y = 0


def gy(s, half=0.0):
    """Lowest ground (model y) over [s - half, s + half]."""
    return min(gnd(t) for t in np.linspace(s - half, s + half, 5)) - E_ORIGIN


def top(s):
    return TOP0 - SLOPE * s - E_ORIGIN


# ── Mesh helpers (local frame: x along, y up, z across) ──────────────────────
def prism_xy(poly, z0, z1, c):
    """A prism from an outline drawn in the (x, y) plane, from z0 to z1."""
    pts, tris = triangulate([(x, y) for x, y in poly])
    n = len(pts)
    ring = [tuple(p) for p in pts]
    area = sum(ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1] for i in range(n))
    side = []
    for i in range(n):
        j = (i + 1) % n
        (xa, ya), (xb, yb) = ring[i], ring[j]
        q = [(xa, ya, z0), (xb, yb, z0), (xb, yb, z1), (xa, ya, z1)]
        f = [(0, 1, 2), (0, 2, 3)] if area > 0 else [(0, 2, 1), (0, 3, 2)]
        side.append(Mesh(q, f, c))
    cap0 = Mesh([(x, y, z0) for x, y in pts], tris if area > 0 else tris[:, ::-1], c)
    cap1 = Mesh([(x, y, z1) for x, y in pts], tris[:, ::-1] if area > 0 else tris, c)
    return merge(side + [cap0, cap1])


def sbox(x0, x1, yb, yt0, yt1, z0, z1, c):
    """A box from x0 to x1 whose top slopes from yt0 to yt1 (and flat bottom yb)."""
    m = box(x1 - x0, 1.0, z1 - z0, c).T((x0 + x1) / 2, 0, (z0 + z1) / 2)
    t = (m.v[:, 0] - x0) / (x1 - x0)
    m.v[:, 1] = np.where(m.v[:, 1] > 0.5, yt0 + (yt1 - yt0) * t, yb)
    return m


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def place(ms, p, d):
    """Local meshes -> model: local x along the unit vector d (model x/z), origin at p (model x/z)."""
    m = merge(ms)
    v = m.v.copy()
    m.v = np.stack([p[0] + v[:, 0] * d[0] - v[:, 2] * d[1], v[:, 1], p[1] + v[:, 0] * d[1] + v[:, 2] * d[0]], 1)
    return m


def arc(x0, x1, ys, pointed=False, n=12):
    """Intrados from (x1, ys) over the top to (x0, ys)."""
    w = x1 - x0
    if not pointed:
        xc, r = (x0 + x1) / 2, w / 2
        return [(xc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(0, math.pi, n + 1)]
    right = [(x0 + w * math.cos(a), ys + w * math.sin(a)) for a in np.linspace(0, math.pi / 3, n // 2 + 1)]
    left = [(x1 + w * math.cos(a), ys + w * math.sin(a)) for a in np.linspace(2 * math.pi / 3, math.pi, n // 2 + 1)]
    return right + left[1:]


def crown_rise(w, pointed):
    return w * math.sin(math.pi / 3) if pointed else w / 2


def arched_wall(L, xa, xb, ys, yt0, yt1, depth, c, pointed=False, ext=0.3):
    """The wall of one bay over its arch: from the springing ys up to the sloping top, the opening from xa to xb."""
    poly = [(-ext, ys), (-ext, yt0), (L + ext, yt1), (L + ext, ys)] + arc(xa, xb, ys, pointed)
    return prism_xy(poly, -depth / 2, depth / 2, c)


def attic(L, t0, t1, depth, ext=0.3):
    """The attic with the specus: a solid base, two side walls and the floor of the channel between them."""
    ms = [sbox(-ext, L + ext, t0 - 1.4, t0 - 0.6, t1 - 0.6, -depth / 2, depth / 2, ATTIC),
          sbox(-ext, L + ext, t0 - 1.45, t0 - 1.3, t1 - 1.3, -depth / 2 - 0.08, depth / 2 + 0.08, GRANITE * 1.04)]
    for s in (-1, 1):
        z0, z1 = sorted((s * 0.32, s * depth / 2))
        ms.append(sbox(-ext, L + ext, t0 - 0.62, t0, t1, z0, z1, ATTIC))
    ms.append(sbox(-ext, L + ext, t0 - 0.62, t0 - 0.58, t1 - 0.58, -0.32, 0.32, CHANNEL))
    return ms


def pier(yb, yt, w, d, step=3.6, dw=0.1, dd=0.12, impost=True):
    """A pier from yb up to the springing yt, stepping out downwards every `step` metres (centred on x = z = 0)."""
    ms = []
    y1, k = yt, 0
    while y1 > yb + 1e-6:
        y0 = max(yb, y1 - step)
        ww, wd = w + 2 * dw * k, d + 2 * dd * k
        ms.append(B(-ww / 2, ww / 2, y0, y1, -wd / 2, wd / 2, GRANITE * (1 + rng.normal(0, 0.02))))
        if k > 0:                                                        # the little sloping ledge of each step
            ms.append(B(-ww / 2 - 0.03, ww / 2 + 0.03, y1 - 0.18, y1, -wd / 2 - 0.03, wd / 2 + 0.03, GRANITE * 1.05))
        y1, k = y0, k + 1
    if impost:
        ms.append(B(-w / 2 - 0.12, w / 2 + 0.12, yt - 0.28, yt, -d / 2 - 0.1, d / 2 + 0.1, GRANITE * 1.06))
    return ms


# ── Pieces ───────────────────────────────────────────────────────────────────
def bay_frame(sa, sb):
    A, Bp = mxz(sa), mxz(sb)
    L = float(np.linalg.norm(Bp - A))
    return A, (Bp - A) / L, L


def pier_dir(s):
    """Direction of the axis at a pier: the mean of the bays on both sides."""
    a = mxz(s + 1.0) - mxz(s - 1.0)
    return a / np.linalg.norm(a)


def single_bay(sa, sb, wa, wb, depth, pointed):
    A, d, L = bay_frame(sa, sb)
    t0, t1 = top(sa), top(sb)
    w = L - wa / 2 - wb / 2
    ys = min(t0, t1) - 2.4 - crown_rise(w, pointed)
    ms = [arched_wall(L, wa / 2, L - wb / 2, ys, t0 - 1.4, t1 - 1.4, depth, GRANITE, pointed)]
    ms += attic(L, t0, t1, depth)
    return place(ms, A, d), ys


def double_bay(sa, sb):
    A, d, L = bay_frame(sa, sb)
    t0, t1 = top(sa), top(sb)
    t = min(t0, t1)
    ms = []
    # lower tier: arch 4.0 m between piers 2.0 m wide at the top
    ysl = t - 10.3 - (L - 2.0) / 2
    ms.append(arched_wall(L, 1.0, L - 1.0, ysl, t0 - 8.9, t1 - 8.9, 2.5, GRANITE))
    # string course between the tiers
    ms.append(sbox(-0.3, L + 0.3, t0 - 8.95, t0 - 8.55, t1 - 8.55, -1.42, 1.42, GRANITE * 1.05))
    # upper tier: arch 4.3 m between piers 1.7 m
    ysu = t - 2.4 - (L - 1.7) / 2
    ms.append(arched_wall(L, 0.85, L - 0.85, ysu, t0 - 1.4, t1 - 1.4, 2.2, GRANITE))
    ms += attic(L, t0, t1, 2.2)
    return place(ms, A, d), ysl, ysu


def wall_run(s0, s1, depth=2.2, chunk=6.0):
    """The solid wall where the channel runs low (no arches)."""
    out = []
    cuts = sorted({s0, s1} | {v for v in VERT_S if s0 < v < s1})
    for a, b in zip(cuts[:-1], cuts[1:]):
        n = max(1, int(math.ceil((b - a) / chunk)))
        for k in range(n):
            sa, sb = a + (b - a) * k / n, a + (b - a) * (k + 1) / n
            A, d, L = bay_frame(sa, sb)
            t0, t1 = top(sa), top(sb)
            yb = min(gy(sa), gy(sb)) - 2.5
            ext = 0.6 if k in (0, n - 1) else 0.05
            ms = [sbox(-ext, L + ext, yb, t0 - 1.4, t1 - 1.4, -depth / 2 + 0.05, depth / 2 - 0.05,
                       GRANITE)]
            ms += attic(L, t0, t1, depth, ext=ext)
            out.append(place(ms, A, d))
    return out


def corner(s, size, depth):
    """A block over a sharp bend (Día Sanz, the corner by the court) closing the wedge between the two runs."""
    p, d = mxz(s), pier_dir(s)
    t = top(s)
    ms = [B(-size / 2, size / 2, gy(s, 2) - 3.0, t - 1.4, -depth / 2, depth / 2, GRANITE),
          B(-size / 2, size / 2, t - 1.4, t - 0.6, -depth / 2, depth / 2, ATTIC),
          B(-size / 2, size / 2, t - 0.62, t - 0.58, -0.32, 0.32, CHANNEL)]
    for sd in (-1, 1):
        z0, z1 = sorted((sd * 0.32, sd * depth / 2))
        ms.append(B(-size / 2, size / 2, t - 0.62, t, z0, z1, ATTIC))
    return place(ms, p, d)


def virgin():
    """The Virgin with the Child (the resin replica of 2019), about 1.3 m tall, facing local +z."""
    robe = lathe([(0.0, 0.0), (0.34, 0.0), (0.33, 0.25), (0.27, 0.7), (0.2, 0.98), (0.14, 1.05), (0.0, 1.06)], WHITE, 12)
    head = lathe([(0.0, 0.0), (0.09, 0.03), (0.11, 0.13), (0.08, 0.22), (0.0, 0.25)], WHITE, 10).T(0, 1.04, 0)
    crown = lathe([(0.0, 0.0), (0.1, 0.0), (0.12, 0.08), (0.0, 0.1)], WHITE * 0.95, 8).T(0, 1.26, 0)
    child = lathe([(0.0, 0.0), (0.1, 0.0), (0.1, 0.25), (0.06, 0.32), (0.0, 0.34)], WHITE, 8).T(0.12, 0.62, 0.2)
    return [robe, head, crown, child]


def inscription_and_niches(statue_side):
    """The panel of the bronze inscription over the three tallest arches and the two niches on the central pier."""
    p, d = mxz(S_ORIGIN), pier_dir(S_ORIGIN)
    t = top(S_ORIGIN)
    yb = t - 8.55
    ms = [B(-9.0, 9.0, yb, yb + 1.55, -1.45, 1.45, PANEL),
          B(-9.1, 9.1, yb + 1.55, yb + 1.8, -1.55, 1.55, PANEL * 1.03)]
    y0 = yb + 1.8
    for sd in (-1, 1):
        f0, f1 = 1.1, 1.5                                                    # pier face -> front of the niche
        z = lambda a, b: sorted((sd * a, sd * b))                           # noqa: E731
        ms += [B(-0.95, -0.55, y0, y0 + 2.6, *z(f0 - 0.05, f1), PANEL),      # side posts
               B(0.55, 0.95, y0, y0 + 2.6, *z(f0 - 0.05, f1), PANEL),
               B(-0.95, 0.95, y0, y0 + 0.35, *z(f0 - 0.05, f1), PANEL),      # sill
               B(-0.95, 0.95, y0 + 2.2, y0 + 2.6, *z(f0 - 0.05, f1), PANEL),  # lintel
               B(-1.05, 1.05, y0 + 2.6, y0 + 2.8, *z(f0 - 0.05, f1 + 0.08), PANEL * 1.03)]
        if sd == statue_side:
            ms += [m.R(0, 0 if sd > 0 else math.pi, 0).T(0, y0 + 0.35, sd * (f0 + 0.2)) for m in virgin()]
    return place(ms, p, d)


# ── The aqueduct ─────────────────────────────────────────────────────────────
def aqueduct():
    parts = []
    # solid walls at both ends
    parts += wall_run(0.0, S_ARCH0)
    parts += wall_run(LAST[-1], S_END)
    parts.append(corner(VERT_S[9], 3.0, 2.2))                               # the corner by the court (78°)

    # 75 single arches
    WP, DP = 1.8, 2.4
    springs = {}
    for i, (sa, sb) in enumerate(zip(SINGLE[:-1], SINGLE[1:])):
        wb = 2.0 if i == len(SINGLE) - 2 else WP                            # the corner pier at Día Sanz
        m, ys = single_bay(sa, sb, WP, wb, 2.3, pointed=i < POINTED)
        parts.append(m)
        springs[sa] = max(springs.get(sa, -1e9), ys)
        springs[sb] = max(springs.get(sb, -1e9), ys)
    for s in SINGLE[:-1]:
        yb = gy(s, 1.5) - 3.0
        parts.append(place(pier(yb, springs[s], WP, DP, dw=0.08, dd=0.1), mxz(s), pier_dir(s)))

    # 44 double arcades
    lows, ups = {}, {}
    for sa, sb in zip(DOUBLE[:-1], DOUBLE[1:]):
        m, ysl, ysu = double_bay(sa, sb)
        parts.append(m)
        for s in (sa, sb):
            lows[s] = max(lows.get(s, -1e9), ysl)
            ups[s] = max(ups.get(s, -1e9), ysu)
    for s in DOUBLE:
        yb = gy(s, 1.5) - 3.0
        t = top(s)
        ms = []
        if yb < lows[s]:
            ms += pier(yb, lows[s], 2.0, 2.6)
        ms += pier(t - 8.55, ups[s], 1.7, 2.2, step=99)
        parts.append(place(ms, mxz(s), pier_dir(s)))
    parts.append(corner(DOUBLE[0], 3.2, 2.8))                               # Día Sanz: the bend of 52°

    # the 4 single arches over the rock of the Postigo, like the upper tier
    us = {}
    for sa, sb in zip(LAST[:-1], LAST[1:]):
        m, ys = single_bay(sa, sb, 1.7, 1.7, 2.2, pointed=False)
        parts.append(m)
        us[sa] = max(us.get(sa, -1e9), ys); us[sb] = max(us.get(sb, -1e9), ys)
    for s in LAST[1:]:
        yb = gy(s, 1.5) - 3.0
        if yb < us[s]:
            parts.append(place(pier(yb, us[s], 1.7, 2.2), mxz(s), pier_dir(s)))

    # the inscription panel and the niches; the Virgin on the face towards +Z (west-south-west)
    d = pier_dir(S_ORIGIN)
    parts.append(inscription_and_niches(1 if d[0] > 0 else -1))
    return merge(parts)


def weather(m):
    """Granite colour: blotches and a little lichen."""
    v = m.v
    n1 = fbm(v, 0.12, 7, 3) - 0.5
    n2 = noise3(v * np.array([1.0, 2.5, 1.0]), 0.9, 11) - 0.5
    k = (1 + 0.22 * n1 + 0.10 * n2)[:, None]
    lic = np.clip((fbm(v, 0.3, 21, 2) - 0.6) * 2.5, 0, 1)[:, None] * 0.25
    keep = (np.abs(m.c - WHITE).sum(1) < 0.05)[:, None]
    c = np.clip(m.c * k * (1 - lic) + np.array((0.50, 0.52, 0.47)) * lic, 0, 1)
    m.c = np.where(keep, m.c, c)
    return m


def footprint(m, buf=2.6, step=3.0):
    """The ground the model stands on: the axis widened by buf on both sides, in model x/z."""
    ss = np.arange(0.0, S_END + 0.01, step)
    if ss[-1] < S_END:
        ss = np.append(ss, S_END)
    pts = [mxz(s) for s in ss]
    left, right = [], []
    for i, p in enumerate(pts):
        a = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        a = a / np.linalg.norm(a)
        nrm = np.array((-a[1], a[0]))
        left.append(p + nrm * buf); right.append(p - nrm * buf)
    ring = left + right[::-1]
    return [[round(float(x), 1), round(float(z), 1)] for x, z in ring]


def main():
    out = ROOT / "models" / "Q244947"
    out.mkdir(parents=True, exist_ok=True)
    m = weather(aqueduct())
    tris, size = write_glb(out / "model.glb", [("acueducto_de_segovia", m)], roughness=0.92)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    print(f"origin {LAT0:.7f}, {LON0:.7f}  heading {HEADING}  ground {E_ORIGIN:.1f} m  top {top(S_ORIGIN):.2f}")
    print("arches:", len(SINGLE) - 1, "single +", len(DOUBLE) - 1, "double +", len(LAST) - 1, "single =",
          len(SINGLE) - 1 + 2 * (len(DOUBLE) - 1) + len(LAST) - 1)
    if "--footprint" in sys.argv:                                      # for meta.json
        Path(sys.argv[sys.argv.index("--footprint") + 1]).write_text(json.dumps(footprint(m)), "utf-8")


if __name__ == "__main__":
    main()
