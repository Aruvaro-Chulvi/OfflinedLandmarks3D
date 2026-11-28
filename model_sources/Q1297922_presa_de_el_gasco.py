"""
Presa de El Gasco (Q1297922), Torrelodones / Galapagar / Las Rozas de Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: the unfinished dam of the Canal del Guadarrama, project by Carlos Lemaur (died 1785), built
1787–1799 under his sons Carlos and Manuel Lemaur; public domain. Abandoned after the storm of 14 May 1799
brought down the middle of the downstream wall.

Measurements (see meta.json for the sources):
- OpenStreetMap way 724280364 (waterway=dam, the outline of the ruin) and way 672259124 (the downstream wall):
  the downstream face runs straight at 97.9° (± 1°), so it faces 187.9° (down the Guadarrama, which flows
  south through the Estrecho de la Peña). Ways 672259125/…130 are the transverse walls of the cells; the river
  passes under the dam through the vaulted outlet (way 202587589, tunnel=yes).
- IGN MDT05 (terrain.json and a 2.5 m sampling of the 5 m grid): the river below the dam at 644.5 m, the crest
  at 700.5 m (56 m above the river; 53–57 m in the literature), the upstream face at 60° (Wikipedia, and the
  slope of the MDT between the crest and the river), the top 32 m wide (Wikipedia: 32 m at the top, 72 m at
  the base), the ground of the flanks where the wings end.
- Wikipedia (es): two granite masonry walls (upstream 2.8 m, downstream 3.5 m thick) joined by transverse walls
  into cells of 8.5 × 10 m filled with earth and stones; the downstream face almost vertical (no steps,
  no buttresses), the upstream face battered at 60°; 154 m of wall left.
- The breach, measured on «Presa de El Gasco y bóveda de desagüe.jpg» (drone, from downstream) and «Presa de
  El Gasco Molino de la Hoz.jpg», scaled by the cell walls of OSM (9.6 m apart): 38.5 m wide between the cell
  walls at x = −19.8 and +18.7, three transverse walls standing out of the fill, the lower 19 m of the
  downstream wall still standing across it, the fill slumped to about 47 m with a steep face (the MDT05 gives
  686–692 m on it); the vaulted outlet about 7.5 m wide and 10 m high at x = +4.5, in a stone frame.
- Cracks in the east wing, low remains of the cell walls and holm oaks on the top (PNOA orthophoto).
Granite ashlar on the faces, earth and rubble in the core.

The rock of the gorge (a «rock skirt», MODEL_SPEC §3.1) is drawn around the dam following the MDT05, with the
dam itself taken out of it (the MDT05 still carries it as a hump): upstream and downstream of the dam the ground
is taken from the terrain 15–20 m away from the faces.

Axes (docs/MODEL_SPEC.md): +Y up, the downstream face faces +Z (187.9°), the dam runs along X (+X to the
east-south-east, the left bank — Torrelodones), origin at the centre of the OSM outline; y = 0 is the river below
the dam (644.5 m above sea level).

    python model_sources/Q1297922_presa_de_el_gasco.py   -> models/Q1297922/model.glb
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, lathe, triangulate, crumple, noise3, fbm, write_glb  # noqa: E402

OUT = ROOT / "models" / "Q1297922"
rng = np.random.default_rng(1799)

HEADING = 187.9
E0 = 644.5                    # the river below the dam (m a.s.l.) = y 0
YC = 56.0                     # the crest (700.5 m)
ZF_TOP = 24.5                 # downstream face at the crest (OSM way 672259124)
BATTER = 0.06                 # downstream face: almost vertical
ZU_TOP = ZF_TOP - 32.0        # upstream edge of the 32 m top
UP_RUN = 1 / math.tan(math.radians(60))   # upstream face at 60°
T_UP, T_DN = 2.8, 3.5         # thickness of the two walls
X0, X1 = -84.0, 84.0          # the ends, into the hillsides
XB0, XB1 = -19.8, 18.7        # the breach, between two cell walls
RIBS = (-10.0, -0.6, 9.6)     # cell walls standing in the breach
YBAND = 19.0                  # the lower part of the downstream wall left standing in the breach
ARCH_X, ARCH_W, ARCH_SPRING = 4.5, 7.5, 6.25   # the vaulted outlet of the river

FACE = np.array((0.76, 0.72, 0.64))     # granite ashlar, downstream (in the sun)
UPF = np.array((0.62, 0.60, 0.55))     # upstream face, weathered
BAND = np.array((0.66, 0.62, 0.55))    # lower wall in the breach
FILL = np.array((0.66, 0.55, 0.40))    # earth and rubble of the cells
TOP = np.array((0.62, 0.57, 0.42))     # dry grass and earth on the top
ROCK = np.array((0.60, 0.57, 0.52))    # granite of the gorge
SOIL = np.array((0.57, 0.51, 0.38))
SCRUB = np.array((0.27, 0.31, 0.19))
OAK = (0.25, 0.31, 0.17)
DARK = (0.12, 0.12, 0.11)


def zf(y):
    """Downstream face at height y."""
    return ZF_TOP + BATTER * (YC - np.asarray(y, float))


def zu(y):
    """Upstream face at height y."""
    return ZU_TOP - UP_RUN * (YC - np.asarray(y, float))


# ── Terrain (terrain.json, IGN MDT05) ────────────────────────────────────────
def load_terrain():
    t = json.loads((OUT / "terrain.json").read_text("utf-8"))
    n = t["n"]
    q = np.frombuffer(base64.b64decode(t["h"]), "<i2").reshape(n, n).astype(np.float64)
    h = np.where(q == -32768, np.nan, t["off"] + q / 10)
    th = math.pi - math.radians(HEADING)
    cs, sn = math.cos(th), math.sin(th)

    def T(x, z):
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        east = x * cs + z * sn
        north = -(-x * sn + z * cs)
        fi = (east - t["x0"]) / t["c"]
        fj = (t["y0"] - north) / t["c"]
        i0 = np.clip(np.floor(fi).astype(int), 0, n - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
        a = np.clip(fi - i0, 0, 1)
        b = np.clip(fj - j0, 0, 1)
        return (h[j0, i0] * (1 - a) * (1 - b) + h[j0, i0 + 1] * a * (1 - b)
                + h[j0 + 1, i0] * (1 - a) * b + h[j0 + 1, i0 + 1] * a * b)
    return T


T = load_terrain()
ZD_REAL, ZU_REAL = 42.0, -50.0           # beyond the hump of the dam in the MDT


def ground(x, z):
    """The ground of the gorge without the dam, in model y."""
    x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
    dn = T(x, np.maximum(z, ZD_REAL))
    up = T(x, np.minimum(z, ZU_REAL))
    w = np.clip((z - ZU_REAL) / (ZD_REAL - ZU_REAL), 0, 1)
    return dn * w + up * (1 - w) - E0


def ybot(x):
    """Foundation of the dam at x: 4 m below the lower of the two feet."""
    return np.minimum(np.minimum(ground(x, zf(0) + 3), ground(x, zu(0) - 3)) - 4.0, YC - 3)


# ── Mesh helpers ─────────────────────────────────────────────────────────────
def orient(m, out):
    """Flip the winding of all faces if their summed normal points against out."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    if float(np.dot(np.cross(b - a, c - a).sum(0), out)) < 0:
        m.f = m.f[:, ::-1]
    return m


def clean(m):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    m.f = m.f[np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-6]
    return m


def sheet(P, out, col=(0.5, 0.5, 0.5)):
    """A grid surface from points P (nu+1, nv+1, 3), facing out."""
    nu, nv = P.shape[0] - 1, P.shape[1] - 1
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    m = Mesh(P.reshape(-1, 3), np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]), col)
    return clean(orient(m, out))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def poly_x(poly, x, out, col):
    """A flat polygon given in (z, y), lying in the plane x, facing out (±1 along x)."""
    pts, tris = triangulate(poly)
    m = Mesh(np.stack([np.full(len(pts), x), pts[:, 1], pts[:, 0]], 1), tris, col)
    return orient(m, (out, 0, 0))


def prism_zy(poly, x0, x1, col):
    """A prism from an outline in (z, y), from x0 to x1."""
    pts, tris = triangulate(poly)
    n = len(pts)
    ms = [poly_x(poly, x0, -1, col), poly_x(poly, x1, 1, col)]
    cz, cy = pts[:, 0].mean(), pts[:, 1].mean()
    for i in range(n):
        (za, ya), (zb, yb) = pts[i], pts[(i + 1) % n]
        q = Mesh([(x0, ya, za), (x0, yb, zb), (x1, yb, zb), (x1, ya, za)], [(0, 1, 2), (0, 2, 3)], col)
        nz, ny = yb - ya, -(zb - za)                       # right-hand normal of a counter-clockwise ring
        area = sum(pts[k][0] * pts[(k + 1) % n][1] - pts[(k + 1) % n][0] * pts[k][1] for k in range(n))
        if area < 0:
            nz, ny = -nz, -ny
        ms.append(orient(q, (0, ny, nz)))
    return merge(ms)


# ── Colours ──────────────────────────────────────────────────────────────────
def ashlar(base, seed, low=0.0, course=1.0):
    base = np.asarray(base, float)

    def fn(v):
        n1 = fbm(v, 0.07, seed, 3) - 0.5
        n2 = noise3(v, 0.55, seed + 5) - 0.5
        st = fbm(v, (0.3, 0.035, 0.3), seed + 9, 2) - 0.5                 # rain streaks
        row = np.where(np.floor(v[:, 1] / course + 1e-6).astype(int) % 2 == 0, 1.02, 0.97)
        dim = 1 - 0.16 * np.clip((low + 8 - v[:, 1]) / 16, 0, 1)            # damp and darker low down
        k = (1 + 0.20 * n1 + 0.10 * n2 - 0.22 * np.clip(st, 0, 1)) * row * dim
        return np.clip(np.outer(k, base), 0, 1)
    return fn


def earth(base, seed, green=0.0):
    base = np.asarray(base, float)

    def fn(v):
        n1 = fbm(v, 0.1, seed, 3) - 0.5
        n2 = noise3(v, 0.8, seed + 3) - 0.5
        c = np.outer(1 + 0.25 * n1 + 0.15 * n2, base)
        if green:
            g = np.clip((fbm(v, 0.06, seed + 11, 2) - 0.55) * 4, 0, 1)[:, None] * green
            c = c * (1 - g) + SCRUB * g
        return np.clip(c, 0, 1)
    return fn


# ── The faces ────────────────────────────────────────────────────────────────
def xs_between(a, b, step):
    n = max(1, int(math.ceil((b - a) / step)))
    return np.linspace(a, b, n + 1)


def face_down(x0, x1, top, bottom=None, step=2.0, dy=1.0):
    """Downstream face between x0 and x1, from the foundation (or bottom(x)) up to top, rows level every dy."""
    xs = xs_between(x0, x1, step)
    yb = ybot(xs) if bottom is None else bottom(xs)
    lv = np.arange(math.floor(yb.min()), top + dy * 0.5, dy)
    lv = np.append(lv[lv < top], top)
    Y = np.clip(np.maximum(lv[None, :], yb[:, None]), None, top)
    X = np.repeat(xs[:, None], len(lv), 1)
    P = np.stack([X, Y, zf(Y)], -1)
    return sheet(P, (0, 0, 1))


def face_up(step=2.0, dy=1.5):
    xs = xs_between(X0, X1, step)
    yb = ybot(xs)
    lv = np.arange(math.floor(yb.min()), YC + dy * 0.5, dy)
    lv = np.append(lv[lv < YC], YC)
    Y = np.clip(np.maximum(lv[None, :], yb[:, None]), None, YC)
    X = np.repeat(xs[:, None], len(lv), 1)
    P = np.stack([X, Y, zu(Y)], -1)
    return sheet(P, (0, 0.5, -0.87)).paint(ashlar(UPF, 23, low=5, course=1.5))


def ends():
    out = []
    for x, s in ((X0, -1), (X1, 1)):
        yb = float(ybot(np.array([x]))[0])
        poly = [(float(zu(yb)), yb), (float(zf(yb)), yb), (ZF_TOP, YC), (ZU_TOP, YC)]
        out.append(poly_x(poly, x, s, ROCK).paint(earth(ROCK, 31)))
    return out


# ── The breach ───────────────────────────────────────────────────────────────
ZIN = ZU_TOP + T_UP                       # inner side of the upstream wall
ZBAND_IN = float(zf(YBAND)) - T_DN        # inner side of the lower wall
# the slumped fill in the breach, (z, y), from the upstream wall to the top of the lower wall
PROFILE = [(ZIN, 51.0), (3.0, 50.2), (9.0, 48.6), (13.5, 45.6), (17.0, 41.5), (19.8, 36.5), (21.8, 31.0),
           (23.0, 25.0), (ZBAND_IN, 21.0), (ZBAND_IN, YBAND)]


def fill_y(z):
    """Height of the fill of the breach at z (its upper part, where the profile rises with z falling)."""
    zz, yy = [p[0] for p in PROFILE[:-1]], [p[1] for p in PROFILE[:-1]]
    return float(np.interp(z, zz, yy))


def profile_pts(n_per=3):
    pts = []
    for (za, ya), (zb, yb) in zip(PROFILE[:-1], PROFILE[1:]):
        for t in np.linspace(0, 1, n_per, endpoint=False):
            pts.append((za + (zb - za) * t, ya + (yb - ya) * t))
    pts.append(PROFILE[-1])
    return np.array(pts)


def breach():
    ms = []
    pr = profile_pts()
    # the fill: a slight dish in the middle, nothing at the edges (they meet the walls of the wings)
    xs = xs_between(XB0, XB1, 2.0)
    X = np.repeat(xs[:, None], len(pr), 1)
    Z = np.repeat(pr[None, :, 0], len(xs), 0)
    Y = np.repeat(pr[None, :, 1], len(xs), 0)
    u = (X - XB0) / (XB1 - XB0)
    edge = np.sin(np.pi * u)
    P = np.stack([X, Y, Z], -1)
    nz = fbm(P.reshape(-1, 3), 0.25, 41, 2).reshape(X.shape) - 0.5
    top = np.clip((Y - YBAND - 1) / 4, 0, 1)                       # the ledge of the lower wall stays flat
    Y = Y - 1.2 * edge * np.clip((Y - 40) / 6, 0, 1) + 1.6 * nz * edge * top
    Z = Z + 0.8 * (fbm(P.reshape(-1, 3) + 7, 0.3, 43, 2).reshape(X.shape) - 0.5) * edge * top
    ms.append(sheet(np.stack([X, Y, Z], -1), (0, 0.6, 0.8)).paint(earth(FILL, 47, green=0.35)))
    # the walls of the wings where they break off: the cut through the cells
    side = [(ZU_TOP + T_UP, YC), (ZF_TOP, YC)] + [(float(zf(y)), y) for y in (45.0, 35.0, 25.0, YBAND)] \
        + [tuple(p) for p in pr[::-1]]
    ms.append(poly_x(side, XB0, 1, FACE).paint(ashlar(FACE * 0.93, 51)))
    ms.append(poly_x(side, XB1, -1, FACE).paint(ashlar(FACE * 0.93, 52)))
    # the upstream wall seen from the breach, and its top
    ms.append(sheet(np.array([[(XB0, PROFILE[0][1] - 1, ZIN), (XB0, YC, ZIN)],
                              [(XB1, PROFILE[0][1] - 1, ZIN), (XB1, YC, ZIN)]]), (0, 0, 1)).paint(ashlar(BAND, 53)))
    ms.append(sheet(np.array([[(XB0, YC, ZU_TOP), (XB0, YC, ZIN)], [(XB1, YC, ZU_TOP), (XB1, YC, ZIN)]]),
                    (0, 1, 0)).paint(earth(TOP, 54)))
    # the ledge on top of the lower wall
    ms.append(sheet(np.array([[(XB0, YBAND, ZBAND_IN), (XB0, YBAND, float(zf(YBAND)))],
                              [(XB1, YBAND, ZBAND_IN), (XB1, YBAND, float(zf(YBAND)))]]), (0, 1, 0)).paint(earth(BAND, 55)))
    # three cell walls standing out of the fill, their tops a little lower inwards
    for k, x in enumerate(RIBS):
        outer = [(float(zf(y)) - 0.45, y) for y in np.linspace(55.2, 27.5, 8)]
        under = [(z, y - 1.4) for z, y in pr[::-1] if z <= outer[-1][0]]
        under = [p for p in under if p[1] >= 24.0]
        poly = [(ZIN, 53.6 + k * 0.4)] + outer + under
        ms.append(prism_zy(poly, x - 0.85, x + 0.85, BAND).paint(ashlar(BAND * 1.04, 60 + k)))
    return ms


def arch_parts():
    """The vaulted outlet: a dark tunnel 10 m deep, its stone frame."""
    r = ARCH_W / 2
    xa, xb = ARCH_X - r, ARCH_X + r
    ms = []
    deep = 10.0
    z1 = float(zf(0))
    z0 = z1 - deep
    # the side walls and the vault inside
    for x, s in ((xa, 1), (xb, -1)):
        ms.append(sheet(np.array([[(x, -1.0, z0), (x, -1.0, z1 + 0.2)], [(x, ARCH_SPRING, z0), (x, ARCH_SPRING, z1 + 0.2)]]),
                        (s, 0, 0), DARK))
    a = np.linspace(0, math.pi, 13)
    P = np.array([[(ARCH_X + r * math.cos(t), ARCH_SPRING + r * math.sin(t), z) for z in (z0, z1 + 0.2)] for t in a])
    ms.append(sheet(P, (0, -1, 0), DARK))
    ms.append(sheet(np.array([[(xa, -1.0, z0), (xa, YBAND - 6, z0)], [(xb, -1.0, z0), (xb, YBAND - 6, z0)]]), (0, 0, 1),
                    (0.06, 0.06, 0.06)))
    ms.append(sheet(np.array([[(xa, 0.3, z0), (xa, 0.3, z1 + 0.5)], [(xb, 0.3, z0), (xb, 0.3, z1 + 0.5)]]), (0, 1, 0),
                    (0.10, 0.13, 0.11)))
    # voussoirs and jambs, lighter, standing a little out of the face
    lite = FACE * 1.08
    for i in range(13):
        t0, t1 = math.pi * i / 13, math.pi * (i + 1) / 13
        tm = (t0 + t1) / 2
        L = r * (t1 - t0) + 0.05
        vb = box(L, 1.1, 0.6, tuple(np.clip(lite * (0.95 + 0.1 * rng.random()), 0, 1)), bottom=False)
        vb.T(0, 0.55, 0).R(0, 0, tm - math.pi / 2).T(ARCH_X + r * math.cos(tm) * 1.0, ARCH_SPRING + r * math.sin(tm), 0)
        vb.T(0, 0, float(zf(ARCH_SPRING + r)) + 0.1)
        ms.append(vb)
    for x0, x1 in ((xa - 1.1, xa), (xb, xb + 1.1)):
        ms.append(B(x0, x1, -1.0, ARCH_SPRING, float(zf(ARCH_SPRING)) - 0.2, float(zf(0)) + 0.4, tuple(lite)))
    # the frame round it
    zt = float(zf(12.0))
    ms.append(B(xa - 2.0, xb + 2.0, 11.6, 12.3, zt - 0.3, zt + 0.35, tuple(lite * 0.98)))
    for x0, x1 in ((xa - 2.0, xa - 1.1), (xb + 1.1, xb + 2.0)):
        ms.append(B(x0, x1, -1.0, 11.6, float(zf(11.6)) - 0.3, float(zf(0)) + 0.3, tuple(lite * 0.96)))
    return ms


def lower_band():
    """The downstream wall left standing across the breach, with the outlet."""
    r = ARCH_W / 2
    xa, xb = ARCH_X - r, ARCH_X + r
    ms = [face_down(XB0, xa, YBAND), face_down(xb, XB1, YBAND)]

    def arch_bottom(xs):
        return ARCH_SPRING + np.sqrt(np.clip(r * r - (xs - ARCH_X) ** 2, 0, None))
    ms.append(face_down(xa, xb, YBAND, bottom=arch_bottom, step=0.625))
    m = merge(ms)
    return m.paint(ashlar(BAND, 71, low=0))


# ── The top ──────────────────────────────────────────────────────────────────
def top_of_wings():
    ms = []
    for x0, x1, s in ((X0, XB0, 81), (XB1, X1, 82)):
        xs = xs_between(x0, x1, 3.0)
        zs = xs_between(ZU_TOP, ZF_TOP, 3.0)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        ex = np.minimum.reduce([X - x0, x1 - X, Z - ZU_TOP, ZF_TOP - Z])
        bump = np.clip(ex / 3, 0, 1) * 0.35 * fbm(np.stack([X, np.zeros_like(X), Z], -1).reshape(-1, 3), 0.15, s, 2).reshape(X.shape)
        ms.append(sheet(np.stack([X, YC + bump, Z], -1), (0, 1, 0)).paint(earth(TOP, s, green=0.5)))
    # low remains of the cell walls
    for x in [XB0 - 9.6 * k for k in range(1, 6)] + [XB1 + 9.6 * k for k in range(1, 6)]:
        za = ZIN + rng.uniform(0, 4)
        zb = ZF_TOP - T_DN - rng.uniform(0, 6)
        h = rng.uniform(0.4, 0.9)
        ms.append(B(x - 0.6, x + 0.6, YC - 0.2, YC + h, za, zb, tuple(BAND * rng.uniform(0.95, 1.05))))
    for x0, x1 in ((X0 + 8, XB0 - 3), (XB1 + 3, X1 - 8)):
        ms.append(B(x0, x1, YC - 0.2, YC + 0.5, ZIN - 1.2, ZIN, tuple(BAND)))
    return ms


def cracks():
    """The two long cracks of the east wing."""
    ms = []
    for pts in ([(27.0, 56), (26.2, 50), (27.6, 44), (26.8, 37), (28.4, 31), (27.9, 24)],
                [(35.5, 56), (36.4, 49), (35.6, 43), (36.8, 38)]):
        for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
            w = 0.22
            P = np.array([[(xa - w, ya, float(zf(ya)) + 0.06), (xa + w, ya, float(zf(ya)) + 0.06)],
                          [(xb - w, yb, float(zf(yb)) + 0.06), (xb + w, yb, float(zf(yb)) + 0.06)]])
            ms.append(sheet(P, (0, 0, 1), (0.22, 0.21, 0.19)))
    return ms


# ── Holm oaks ────────────────────────────────────────────────────────────────
def sphere(r, c, seg=10, rings=6):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def blob(cx, cy, cz, r, flat, col, seed):
    m = sphere(r, col).S(1, flat, 1)
    crumple(m, r * 0.3, 1.3 / r, seed)
    m.T(cx, cy, cz)
    c = np.asarray(col, float)
    lo = cy - r * flat

    def tint(v):
        t = np.clip((v[:, 1] - lo) / (2 * r * flat + 1e-6), 0, 1)
        k = 0.66 + 0.44 * t + 0.16 * (noise3(v, 0.9, seed) - 0.5)
        return np.clip(np.outer(k, c), 0, 1)
    return m.paint(tint)


def oak(x, y, z, r, seed):
    ms = [B(x - 0.2, x + 0.2, y - 0.3, y + r * 0.6, z - 0.2, z + 0.2, (0.30, 0.25, 0.20))]
    ms.append(blob(x, y + r * 0.95, z, r, 0.75, OAK, seed))
    ms.append(blob(x + r * 0.45, y + r * 0.75, z - r * 0.3, r * 0.6, 0.8, OAK, seed + 1))
    return ms


def oaks():
    ms = []
    spots = [(-74, 2), (-66, 15), (-55, -2), (-47, 9), (-38, 18), (-30, 3), (-26, 13), (-23, -3),
             (24, 6), (31, -1), (40, 15), (52, 4), (63, 12), (71, -2)]
    for i, (x, z) in enumerate(spots):
        ms += oak(x + rng.uniform(-1, 1), YC, z + rng.uniform(-1, 1), rng.uniform(2.0, 3.4), 900 + i)
    for i, (x, z) in enumerate([(-12, 4), (2, 8), (12, 0)]):           # on the fill of the breach
        ms += oak(x, fill_y(z) - 1.0, z, rng.uniform(1.8, 2.6), 950 + i)
    ms += oak(-5.0, fill_y(18.0) - 0.8, 18.0, 2.2, 960)                                # on its face (the drone photo)
    ms += oak(-16.0, fill_y(21.0) - 0.8, 21.0, 1.6, 961)
    return ms


# ── The gorge (rock skirt) ───────────────────────────────────────────────────
def gorge():
    xs = xs_between(-104, 104, 4.0)
    zs = xs_between(-66, 58, 4.0)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    G = ground(X, Z)
    P0 = np.stack([X, G, Z], -1).reshape(-1, 3)
    rough = (fbm(P0, 0.09, 101, 3) - 0.5).reshape(X.shape)
    Y = G + 0.7 + 2.0 * rough
    P = np.stack([X, Y, Z], -1)
    nu, nv = X.shape[0] - 1, X.shape[1] - 1
    idx = np.arange(X.size).reshape(X.shape)

    def wanted(i, j):                                    # only the gorge round the dam
        x, z = (X[i, j] + X[i + 1, j]) / 2, (Z[i, j] + Z[i, j + 1]) / 2
        g = G[i:i + 2, j:j + 2].mean()
        if g > YC + 10 or abs(x) > X1 + 16:
            return False
        return float(zu(min(g, YC))) - 24 < z < float(zf(min(g, YC))) + 30
    faces, used, edge = [], np.zeros(X.shape, bool), np.zeros(X.shape, bool)
    for i in range(nu):
        for j in range(nv):
            if not wanted(i, j):
                edge[i:i + 2, j:j + 2] = True
                continue
            used[i:i + 2, j:j + 2] = True                # kept under the dam too: it closes the bottom
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            faces += [(a, b, c), (a, c, d)]
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    rim = used & edge
    P[..., 1] = np.where(rim, P[..., 1] - 14.0, P[..., 1])   # the rim dips under the ground (a rock skirt)
    m = Mesh(P.reshape(-1, 3), np.array(faces), ROCK)
    m = orient(m, (0, 1, 0))
    n = np.zeros_like(m.v)
    fa, fb, fc = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    fn = np.cross(fb - fa, fc - fa)
    for k in range(3):
        np.add.at(n, m.f[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

    def paint(v):
        steep = np.clip((0.9 - n[:, 1]) / 0.35, 0, 1)                    # bare granite on the steep sides
        slab = np.clip((fbm(v, 0.05, 107, 3) - 0.45) * 3, 0, 1)
        r = np.clip(steep + 0.6 * slab, 0, 1)[:, None]
        c = SOIL * (1 - r) + ROCK * r
        c = c * (1 + 0.22 * (fbm(v, 0.2, 109, 2) - 0.5))[:, None]
        sc = np.clip((fbm(v, 0.07, 113, 3) - 0.52) * 3.5, 0, 1)[:, None] * (1 - 0.5 * r)
        c = c * (1 - sc) + SCRUB * sc
        wet = np.clip(1 - (np.abs(v[:, 0] - ARCH_X - 0.04 * (v[:, 2] - 30)) - 5) / 6, 0, 1)
        wet = (wet * (v[:, 1] < 3.5) * (v[:, 2] > 20))[:, None]          # the river bed below the outlet
        c = c * (1 - wet * 0.55) + np.array((0.16, 0.2, 0.17)) * wet * 0.55
        return np.clip(c, 0, 1)
    m.paint(paint)
    keep = np.unique(m.f)                                 # drop the vertices no face uses
    remap = np.full(len(m.v), -1)
    remap[keep] = np.arange(len(keep))
    return [Mesh(m.v[keep], remap[m.f], m.c[keep])]


def footprint():
    """Seen from above: the dam with 2 m round it (the gorge around keeps its trees)."""
    xs = np.linspace(X0, X1, 13)
    up = [(float(x), round(float(zu(ybot(np.array([x]))[0] + 4)) - 2, 1)) for x in xs]
    pts = [(X0 - 2, round(float(zf(0)) + 3, 1))] + [(X0 - 2, up[0][1])] + up[1:-1] + [(X1 + 2, up[-1][1]),
                                                                                     (X1 + 2, round(float(zf(0)) + 3, 1))]
    return [[round(a, 1), round(b, 1)] for a, b in pts]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wings = merge([face_down(X0, XB0, YC), face_down(XB1, X1, YC)]).paint(ashlar(FACE, 11, low=0))
    dam = merge([wings, face_up(), lower_band()] + ends() + breach() + arch_parts() + top_of_wings() + cracks())
    trees = merge(oaks())
    rock = merge(gorge())
    tris, size = write_glb(OUT / "model.glb", [("presa_de_el_gasco", dam), ("encinas", trees), ("roca", rock)],
                           roughness=0.92)
    allv = np.vstack([dam.v, trees.v, rock.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    print("dam:", dam.tris, "trees:", trees.tris, "rock:", rock.tris)
    print("footprint:", json.dumps(footprint()))
    print("ground at the river below the outlet:", float(ground(ARCH_X, 46.0)) + E0)


if __name__ == "__main__":
    main()
