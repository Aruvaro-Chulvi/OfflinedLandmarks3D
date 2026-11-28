"""
Presa de El Atazar (dam of the Embalse de El Atazar, Q513846), El Atazar / Patones (Madrid) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: the double-curvature concrete arch dam of the Canal de Isabel II on the Lozoya, designed by
Consulpresa, built 1965/1968–1972 (contractor Entrecanales y Távora); a modern work, rights "protected".

Measurements (see meta.json for the sources):
- OpenStreetMap way 288349518 (waterway=dam, the crest line) and way 57907223 (waterway=dam, the outline): the
  arch fits a circle centred 159 m east and 20 m north of the origin, 217 m to the upstream edge of the crest
  (±2 m), spanning bearings 213°–322° from that centre; a straight gravity wing goes on to the south abutment and a
  massive block (the north abutment, "ten times the concrete of the dam") closes the north end. The downstream face
  at the crown looks 82.8° (from the origin to the centre of the arch).
- IGN MDT05 (2.5 m sampling): the crest at 873–874 m all along the arch, the reservoir at 866 m when it was flown,
  the river and the plunge pool below the crown at 754–756 m. The downstream face is in the MDT05: for every
  bearing the radius of its 860 m contour (R860 below) and how the face goes down from there (OFF: −10 m at 765 m,
  +3.5 m at the crest) — a face almost upright at the top and leaning out about 1:9 lower down, a little flared at
  the foot. The ground under the dam is taken from the MDT05 just downstream of it (GROUND below).
- Wikipedia (es/en) and iAgua: 134 m high above the foundation, crest 484 m long at 873 m, 45 m thick at the base
  and 7 m at the crest; one uncontrolled fixed-lip spillway (500 m³/s; normal level 870 m), two bottom outlets.
- Photos (Commons): «Presa de El Atazar - 01.jpg» and «Presa del pantano de El Atazar.jpg» (from downstream: the
  face with its vertical joints and rust streaks, the overhanging crest with the road, lamp posts and parapets, the
  lip of the spillway under the crest at the crown, the north abutment block, the plunge pool);
  «Pie de la Presa del Atazar - panoramio.jpg» (the outlets at the foot), «El Atazar dam view01.jpg» (the crest).
- PNOA orthophoto: the crest road, the plunge pool, the abutments.

Axes (docs/MODEL_SPEC.md): +Y up, the downstream face at the crown faces +Z (82.8°), +X to the north (the left
bank), origin at 40.91229, −3.47295 (OSM way 288349518/57907223; the tour ring), which is in the plunge pool at
the foot of the crown, 36 m downstream of the toe of the face; y = 0 is the MDT05 there (754.5 m).

    python model_sources/Q513846_presa_de_el_atazar.py   -> models/Q513846/model.glb
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
from meshkit import Mesh, merge, box, cyl, lathe, triangulate, crumple, noise3, fbm, write_glb  # noqa: E402

OUT = ROOT / "models" / "Q513846"
rng = np.random.default_rng(1972)

HEADING = 82.8
E0 = 754.5                     # the plunge pool at the origin (m a.s.l.) = y 0
YC = 873.4 - E0                # the crest road
YLIP = 870.0 - E0              # the fixed lip of the spillway (normal level)
CE, CN = 159.0, 20.0           # centre of the arch, metres east / north of the origin
B0, B1 = 208.0, 323.0          # the arch, bearings from the centre (south -> north)
SPILL = (253.0, 271.0)         # the spillway bays at the crown
PIERS = np.linspace(SPILL[0], SPILL[1], 6)

# MDT05: radius of the 860 m contour of the downstream face, by bearing from the centre
R860_B = [200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276,
          280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324]
R860_R = [202, 202, 202.5, 208.5, 206.5, 205, 204.5, 205.5, 203, 205, 206, 205.5, 204, 207.5, 207, 206.5, 205,
          207, 208, 205.5, 202, 200, 199, 199.5, 200, 201, 202, 200.5, 197, 196.5, 197, 197]
# the face below / above its 860 m contour (m, + upstream), by elevation
OFF_Z = [600, 765, 780, 800, 820, 840, 850, 860, 870, 873.4]
OFF_R = [-36, -10, -7.5, -5.5, -3.5, -2, -1, 0, 2.5, 3.5]
# MDT05: the ground just downstream of the face (R 180–190 m), by bearing — the foot of the dam
GROUND_B = [196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 276, 280, 284, 288, 292, 296,
            300, 304, 308, 312, 316, 320, 324, 328]
GROUND_E = [862, 853, 848, 842, 831, 822, 813, 803, 794, 790, 776, 767, 763, 759, 756, 756, 764, 774, 782, 793, 802,
            815, 826, 831, 839, 847, 852, 860, 869]

CONC = np.array((0.80, 0.78, 0.72))     # concrete of the faces
CONC_UP = np.array((0.74, 0.73, 0.69))
ROAD = np.array((0.30, 0.30, 0.31))
WALK = np.array((0.70, 0.68, 0.64))
ROCK = np.array((0.47, 0.43, 0.38))     # slates and schists of the gorge
SOIL = np.array((0.55, 0.49, 0.39))
SCRUB = np.array((0.30, 0.33, 0.21))
PINE = (0.17, 0.26, 0.15)
DARK = (0.08, 0.08, 0.08)

TH = math.pi - math.radians(HEADING)
CS, SN = math.cos(TH), math.sin(TH)


def en2m(e, n):
    """Metres east / north of the origin -> model x, z."""
    e, n = np.asarray(e, float), np.asarray(n, float)
    return e * CS + n * SN, e * SN - n * CS


def polar(b, r):
    """Bearing from the centre of the arch (deg) and radius -> model x, z."""
    t = np.radians(np.asarray(b, float))
    return en2m(CE + np.asarray(r, float) * np.sin(t), CN + np.asarray(r, float) * np.cos(t))


def r860(b):
    raw = np.interp(b, R860_B, R860_R)
    return (np.interp(np.asarray(b) - 3, R860_B, R860_R) + 2 * raw + np.interp(np.asarray(b) + 3, R860_B, R860_R)) / 4


def rd(b, y):
    """Radius of the downstream face at bearing b and model height y."""
    return r860(b) + np.interp(np.asarray(y) + E0, OFF_Z, OFF_R)


def rd_top(b):
    return r860(b) + 3.5


def ru_top(b):
    return r860(b) + 11.0


def ru(b, y):
    """Upstream face: 27 m further upstream at the bed than at the crest (the double curvature)."""
    t = np.clip((YC - np.asarray(y, float)) / 119.0, 0, 1.2)
    return ru_top(b) + 27.0 * t ** 1.5


def foot(b):
    """Model y of the ground at the foot of the dam."""
    return np.interp(b, GROUND_B, GROUND_E) - E0


# ── Terrain (terrain.json, IGN MDT05) ────────────────────────────────────────
def load_terrain():
    t = json.loads((OUT / "terrain.json").read_text("utf-8"))
    n = t["n"]
    q = np.frombuffer(base64.b64decode(t["h"]), "<i2").reshape(n, n).astype(np.float64)
    h = np.where(q == -32768, np.nan, t["off"] + q / 10)

    def T(e, nn):
        e, nn = np.broadcast_arrays(np.asarray(e, float), np.asarray(nn, float))
        fi = (e - t["x0"]) / t["c"]
        fj = (t["y0"] - nn) / t["c"]
        i0 = np.clip(np.floor(fi).astype(int), 0, n - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
        a = np.clip(fi - i0, 0, 1)
        b = np.clip(fj - j0, 0, 1)
        return (h[j0, i0] * (1 - a) * (1 - b) + h[j0, i0 + 1] * a * (1 - b)
                + h[j0 + 1, i0] * (1 - a) * b + h[j0 + 1, i0 + 1] * a * b)
    return T


T = load_terrain()


def ground_en(e, n):
    """Ground without the dam (m a.s.l.); NaN on the reservoir (flat 866 m in the MDT05)."""
    e, n = np.broadcast_arrays(np.asarray(e, float), np.asarray(n, float))
    b = np.degrees(np.arctan2(e - CE, n - CN)) % 360
    R = np.hypot(e - CE, n - CN)
    g = T(e, n)
    inarch = (b > 200) & (b < 326)
    cut = inarch & (R > r860(b) - 16) & (R < ru_top(b) + 4)
    t = np.radians(b)
    rr = r860(b) - 16
    g2 = T(CE + rr * np.sin(t), CN + rr * np.cos(t))
    g = np.where(cut, np.minimum(g, g2), g)
    lake = (np.abs(g - 866.0) < 0.6) & (R > r860(b) - 2)
    return np.where(lake, np.nan, g)


# ── Mesh helpers ─────────────────────────────────────────────────────────────
def orient(m, out):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    if float(np.dot(np.cross(b - a, c - a).sum(0), out)) < 0:
        m.f = m.f[:, ::-1]
    return m


def clean(m):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    m.f = m.f[np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-6]
    return m


def sheet(P, out, col=(0.5, 0.5, 0.5)):
    """A grid surface from points P (nu+1, nv+1, 3); out = a direction it faces, or a callable on the centroid."""
    nu, nv = P.shape[0] - 1, P.shape[1] - 1
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    m = Mesh(P.reshape(-1, 3), np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]), col)
    return clean(orient(m, out))


def bspan(b0, b1, step=1.0):
    n = max(1, int(math.ceil((b1 - b0) / step)))
    return np.linspace(b0, b1, n + 1)


def ppt(b, r, y):
    x, z = polar(b, r)
    return np.stack([x, np.broadcast_to(y, np.shape(x)), z], -1)


def toward_centre(b):
    """Unit vector (model) pointing downstream (to the centre) at bearing b."""
    x0, z0 = polar(b, 200.0)
    x1, z1 = polar(b, 190.0)
    v = np.array([x1 - x0, 0.0, z1 - z0])
    return v / np.linalg.norm(v)


def pbox(b0, b1, r0, r1, y0, y1, col, step=1.0):
    """A block between two bearings, two radii and two heights (curved along the arch)."""
    bs = bspan(b0, b1, step)
    bm = (b0 + b1) / 2
    dn = toward_centre(bm)
    ms = []
    for r, s in ((r0, 1), (r1, -1)):                       # r0: downstream side, r1: upstream side
        P = np.stack([ppt(bs, r, y0), ppt(bs, r, y1)], 1)
        ms.append(sheet(P, dn * s, col))
    for y, s in ((y0, -1), (y1, 1)):
        P = np.stack([ppt(bs, r0, y), ppt(bs, r1, y)], 1)
        ms.append(sheet(P, (0, s, 0), col))
    for b, s in ((b0, -1), (b1, 1)):
        P = np.array([[ppt(b, r0, y0), ppt(b, r0, y1)], [ppt(b, r1, y0), ppt(b, r1, y1)]])
        t = ppt(b + s * 0.5, (r0 + r1) / 2, (y0 + y1) / 2) - ppt(b, (r0 + r1) / 2, (y0 + y1) / 2)
        ms.append(sheet(P, t, col))
    return merge(ms)


# ── Colours ──────────────────────────────────────────────────────────────────
def bearing_of(v):
    e = v[:, 0] * CS + v[:, 2] * SN
    n = v[:, 0] * SN - v[:, 2] * CS
    return np.degrees(np.arctan2(e - CE, n - CN)) % 360


def concrete(base, seed, joints=True, wet_below=None):
    base = np.asarray(base, float)

    def fn(v):
        n1 = fbm(v, 0.03, seed, 3) - 0.5
        n2 = noise3(v, 0.4, seed + 5) - 0.5
        st = fbm(v * np.array([1, 0.04, 1]), 0.35, seed + 9, 2) - 0.5           # vertical streaks
        rust = np.clip((fbm(v * np.array([1, 0.05, 1]), 0.25, seed + 13, 2) - 0.62) * 3, 0, 1)
        k = 1 + 0.10 * n1 + 0.05 * n2 - 0.22 * np.clip(st, 0, 1)
        if joints:                                        # the vertical joints between the blocks, every 4°
            b = bearing_of(v)
            k = k * (1 - 0.13 * (np.abs(((b - 0.5) % 4.0) - 2.0) > 1.85))
        c = np.outer(k, base)
        c = c * (1 - 0.35 * rust[:, None]) + np.array((0.55, 0.36, 0.22)) * 0.35 * rust[:, None]
        if wet_below is not None:
            w = np.clip((wet_below - v[:, 1]) / 1.5, 0, 1)[:, None]
            c = c * (1 - 0.45 * w) + np.array((0.33, 0.33, 0.27)) * 0.45 * w
        return np.clip(c, 0, 1)
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


# ── The arch ─────────────────────────────────────────────────────────────────
def ranges():
    """(b0, b1, top) pieces of the arch: crest road, the spillway bays, crest road."""
    return [(B0, SPILL[0], YC), (SPILL[0], SPILL[1], YLIP), (SPILL[1], B1, YC)]


def face_down(b0, b1, top):
    bs = bspan(b0, b1, 1.0)
    yb = foot(bs) - 8.0
    lv = np.arange(math.floor(yb.min()), top, 4.0)
    lv = np.append(lv, [top - 1.5, top])
    Y = np.clip(np.maximum(lv[None, :], yb[:, None]), None, top)
    B = np.repeat(bs[:, None], len(lv), 1)
    x, z = polar(B, rd(B, Y))
    m = sheet(np.stack([x, Y, z], -1), toward_centre((b0 + b1) / 2))
    m.paint(concrete(CONC, 11, wet_below=1.5))
    if top < YC:                                          # stained under the lip of the spillway
        k = 1 - 0.28 * np.clip((m.v[:, 1] - (YLIP - 45)) / 25, 0, 1) * (m.v[:, 1] < YLIP - 2)
        m.c = m.c * k[:, None]
    return m


def face_up(b0, b1, top):
    bs = bspan(b0, b1, 1.0)
    yb = foot(bs) - 8.0
    lv = np.arange(math.floor(yb.min()), top, 6.0)
    lv = np.append(lv, [YLIP - 0.5, top]) if top > YLIP else np.append(lv, top)
    lv = np.unique(np.clip(lv, None, top))
    Y = np.clip(np.maximum(lv[None, :], yb[:, None]), None, top)
    B = np.repeat(bs[:, None], len(lv), 1)
    x, z = polar(B, ru(B, Y))
    m = sheet(np.stack([x, Y, z], -1), -toward_centre((b0 + b1) / 2))
    return m.paint(concrete(CONC_UP, 12, joints=True, wet_below=YLIP - 0.4))


def top_of(b0, b1, top):
    bs = bspan(b0, b1, 1.0)
    fr = np.linspace(0, 1, 6)
    B = np.repeat(bs[:, None], len(fr), 1)
    R = rd(B, top) + (ru(B, top) - rd(B, top)) * fr[None, :]
    x, z = polar(B, R)
    m = sheet(np.stack([x, np.full_like(x, top), z], -1), (0, 1, 0))
    if top >= YC - 0.01:                                  # the road between two kerbs
        def fn(v):
            b = bearing_of(v)
            R = np.hypot(v[:, 0] * CS + v[:, 2] * SN - CE, v[:, 0] * SN - v[:, 2] * CS - CN)
            u = (R - rd_top(b)) / (ru_top(b) - rd_top(b))
            road = ((u > 0.18) & (u < 0.85))[:, None]
            c = np.where(road, ROAD, WALK) * (1 + 0.06 * (noise3(v, 0.3, 21) - 0.5))[:, None]
            return np.clip(c, 0, 1)
        return m.paint(fn)
    return m.paint(concrete(CONC * 0.92, 13, joints=False))


def ends():
    ms = []
    for b, s in ((B0, -1), (B1, 1)):
        y = np.linspace(foot(b) - 8, YC, 12)
        P = np.stack([ppt(np.full_like(y, b), rd(b, y), y), ppt(np.full_like(y, b), ru(b, y), y)], 1)
        t = ppt(b + s * 0.5, 210, 50) - ppt(b, 210, 50)
        ms.append(sheet(P, t).paint(concrete(CONC, 14, joints=False)))
    return ms


def crest_works():
    ms = []
    lite = tuple(np.clip(CONC * 1.04, 0, 1))
    for b0, b1, top in ranges():
        if top < YC:
            continue
        rt = r860((b0 + b1) / 2)
        # cornice over the downstream face, parapets on both sides
        ms.append(pbox(b0, b1, rt + 2.3, rt + 3.7, YC - 1.3, YC, lite))
        ms.append(pbox(b0, b1, rt + 3.5, rt + 3.9, YC, YC + 1.1, lite))
        ms.append(pbox(b0, b1, rt + 10.6, rt + 11.0, YC, YC + 1.1, lite))
    # lamp posts on the upstream side
    for b in np.arange(B0 + 2, B1, 6.0):
        x, z = polar(b, r860(b) + 10.3)
        ms.append(cyl(0.12, 0.09, 9.0, (0.35, 0.36, 0.37), seg=6).T(float(x), YC, float(z)))
        x2, z2 = polar(b, r860(b) + 9.3)
        ms.append(box(1.2, 0.25, 0.4, (0.85, 0.85, 0.8)).T(float((x + x2) / 2), YC + 8.9, float((z + z2) / 2)))
    return ms


def spillway():
    """The fixed-lip spillway at the crown: the road on a deck over six piers, the lip standing out below."""
    ms = []
    b0, b1 = SPILL
    rt = r860((b0 + b1) / 2)
    deck = tuple(np.clip(CONC * 1.02, 0, 1))
    ms.append(pbox(b0, b1, rt + 2.3, rt + 11.0, YC - 1.4, YC, deck))                   # the deck
    ms.append(top_of(b0, b1, YC))
    ms.append(pbox(b0, b1, rt + 3.5, rt + 3.9, YC, YC + 1.1, deck))
    ms.append(pbox(b0, b1, rt + 10.6, rt + 11.0, YC, YC + 1.1, deck))
    for k, b in enumerate(PIERS):
        w = 0.35 if 0 < k < len(PIERS) - 1 else 0.5
        ms.append(pbox(b - w, b + w, rt - 0.5, rt + 11.5, YLIP - 0.2, YC - 1.4, tuple(CONC * 0.97)))
    # the lip: a nose standing out 3 m from the face under the sill, dark with water stains below
    ms.append(pbox(b0, b1, rt - 3.0, rt + 0.5, YLIP - 2.2, YLIP, tuple(CONC * 0.9)).paint(concrete(CONC * 0.9, 15)))
    return ms


def outlets():
    """The two bottom outlets at the foot of the crown, in their portal."""
    ms = []
    y0 = foot(262) - 1.0
    rf = float(rd(262, y0 + 6))
    ms.append(pbox(258.5, 265.5, rf - 6.0, rf + 1.0, y0, y0 + 14.0, tuple(CONC * 0.95)).paint(concrete(CONC * 0.95, 16)))
    for b in (260.4, 263.6):
        ms.append(pbox(b - 0.55, b + 0.55, rf - 6.15, rf - 5.5, y0 + 3.0, y0 + 7.0, DARK))
    ms.append(pbox(258.0, 266.0, rf - 6.3, rf + 1.0, y0 + 14.0, y0 + 14.6, tuple(CONC)))
    return ms


# ── The abutments (OSM way 57907223) ─────────────────────────────────────────
NORTH = [(-2.7, 142.6), (26.0, 174.9), (57.6, 177.3), (57.7, 182.8), (57.6, 193.6), (11.1, 214.0), (-24.3, 212.8),
         (-21.7, 169.5), (-22.0, 161.8), (-29.0, 149.7), (-22.0, 144.7)]
SOUTH = [(39.7, -162.5), (66.9, -144.9), (128.9, -185.5), (120.5, -199.7), (49.2, -169.5)]


def block(poly_en, seed):
    ms = []
    P = np.array(poly_en, float)
    x, z = en2m(P[:, 0], P[:, 1])
    pts = list(zip(x.tolist(), z.tolist()))
    tp, tris = triangulate(pts)
    top = Mesh(np.stack([tp[:, 0], np.full(len(tp), YC), tp[:, 1]], 1), tris, WALK)
    ms.append(orient(top, (0, 1, 0)).paint(earth(WALK, seed)))
    cx, cz = x.mean(), z.mean()
    n = len(pts)
    for i in range(n):
        (xa, za), (xb, zb) = pts[i], pts[(i + 1) % n]
        L = math.hypot(xb - xa, zb - za)
        ts = np.linspace(0, 1, max(1, int(L / 4)) + 1)
        xs, zs = xa + (xb - xa) * ts, za + (zb - za) * ts
        es, ns = xs * CS + zs * SN, xs * SN - zs * CS
        g = T(es, ns) - E0 - 8.0
        g = np.minimum(np.where(np.isfinite(g), g, YC - 30), YC - 2)
        Pw = np.stack([np.stack([xs, g, zs], -1), np.stack([xs, np.full_like(xs, YC), zs], -1)], 1)
        mx, mz = (xa + xb) / 2 - cx, (za + zb) / 2 - cz
        ms.append(sheet(Pw, (mx, 0, mz)).paint(concrete(CONC * 0.97, seed + i, joints=False)))
        # parapet on the edge
        d = np.array([xb - xa, zb - za]) / max(L, 1e-6)
        nrm = np.array([d[1], -d[0]])
        if nrm @ np.array([mx, mz]) < 0:
            nrm = -nrm
        p0 = np.array([xa, za]) - nrm * 0.2
        ms.append(box(L, 1.1, 0.4, tuple(np.clip(CONC * 1.03, 0, 1)))
                  .R(0, -math.atan2(zb - za, xb - xa), 0).T(float(p0[0] + d[0] * L / 2), YC, float(p0[1] + d[1] * L / 2)))
    return ms


# ── The gorge (rock skirt) and its pines ─────────────────────────────────────
def gorge():
    xs = np.arange(-330, 331, 6.0)
    zs = np.arange(-90, 281, 6.0)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    E, N = X * CS + Z * SN, X * SN - Z * CS
    G = ground_en(E, N) - E0
    P0 = np.stack([X, np.nan_to_num(G), Z], -1).reshape(-1, 3)
    rough = (fbm(P0, 0.08, 201, 3) - 0.5).reshape(X.shape)
    pool = np.clip((2.0 - np.nan_to_num(G, nan=99)) / 2, 0, 1)            # under the water of the pool
    Y = G + 0.4 + 1.8 * rough * (1 - pool) - 1.5 * pool
    P = np.stack([X, Y, Z], -1)
    nu, nv = X.shape[0] - 1, X.shape[1] - 1
    idx = np.arange(X.size).reshape(X.shape)
    dist = np.hypot(X, Z - 40)
    faces, used, edge = [], np.zeros(X.shape, bool), np.zeros(X.shape, bool)
    for i in range(nu):
        for j in range(nv):
            g = G[i:i + 2, j:j + 2]
            ok = np.all(np.isfinite(g)) and g.max() < YC + 25 and dist[i:i + 2, j:j + 2].max() < 300
            if not ok:
                edge[i:i + 2, j:j + 2] = True
                continue
            used[i:i + 2, j:j + 2] = True
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            faces += [(a, b, c), (a, c, d)]
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    rim = used & edge
    P[..., 1] = np.where(rim, P[..., 1] - 20.0, P[..., 1])
    m = Mesh(np.nan_to_num(P).reshape(-1, 3), np.array(faces), ROCK)
    m = orient(m, (0, 1, 0))
    n = np.zeros_like(m.v)
    fa, fb, fc = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    fn = np.cross(fb - fa, fc - fa)
    for k in range(3):
        np.add.at(n, m.f[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

    def paint(v):
        steep = np.clip((0.88 - n[:, 1]) / 0.3, 0, 1)
        slab = np.clip((fbm(v, 0.04, 207, 3) - 0.48) * 3, 0, 1)
        r = np.clip(steep + 0.5 * slab, 0, 1)[:, None]
        c = SOIL * (1 - r) + ROCK * r
        c = c * (1 + 0.24 * (fbm(v, 0.15, 209, 2) - 0.5))[:, None]
        sc = np.clip((fbm(v, 0.05, 213, 3) - 0.45) * 3, 0, 1)[:, None] * (1 - 0.5 * r)
        c = c * (1 - sc) + SCRUB * sc
        wet = (v[:, 1] < 1.0)[:, None]
        c = np.where(wet, np.array((0.20, 0.24, 0.22)), c)
        return np.clip(c, 0, 1)
    m.paint(paint)
    keep = np.unique(m.f)
    remap = np.full(len(m.v), -1)
    remap[keep] = np.arange(len(keep))
    return Mesh(m.v[keep], remap[m.f], m.c[keep])


def pine(x, y, z, h, seed):
    r = h * 0.28
    ms = [box(0.4, h * 0.45, 0.4, (0.33, 0.26, 0.2)).T(x, y - 0.5, z)]
    prof = [(0.0, -r * 0.5), (r, -r * 0.15), (r * 0.95, r * 0.25), (r * 0.55, r * 0.6), (0.0, r * 0.75)]
    m = lathe(prof, PINE, seg=8).S(1, 1.1, 1)
    crumple(m, r * 0.18, 1.5 / r, seed)
    ms.append(m.T(x, y + h * 0.62, z).paint(lambda v: np.clip(np.outer(0.8 + 0.4 * noise3(v, 0.7, seed), PINE), 0, 1)))
    return ms


def pines():
    ms = []
    k = 0
    while len(ms) < 2 * 44 and k < 2000:
        k += 1
        x, z = rng.uniform(-200, 200), rng.uniform(20, 240)
        e, n = x * CS + z * SN, x * SN - z * CS
        g = float(ground_en(e, n))
        if not np.isfinite(g) or g - E0 < 12 or g - E0 > YC or math.hypot(x, z - 40) > 260:
            continue
        b = math.degrees(math.atan2(e - CE, n - CN)) % 360
        if 200 < b < 326 and math.hypot(e - CE, n - CN) > r860(b) - 20:
            continue
        ms += pine(x, g - E0, z, rng.uniform(7, 12), 800 + k)
    return ms


def footprint():
    """Seen from above: the arch, its abutment blocks and 2 m round them (the gorge keeps its trees)."""
    pts = []
    for b in bspan(B0 - 2, B1 + 1, 4.0):
        x, z = polar(b, ru_top(b) + 2)
        pts.append((float(x), float(z)))
    xn, zn = en2m(np.array([p[0] for p in NORTH]), np.array([p[1] for p in NORTH]))
    for i in (6, 5, 4, 2):
        pts.append((float(xn[i]), float(zn[i])))
    for b in bspan(B0 - 2, B1 + 1, 4.0)[::-1]:
        x, z = polar(b, rd(b, foot(b)) - 2)
        pts.append((float(x), float(z)))
    xs_, zs_ = en2m(np.array([p[0] for p in SOUTH]), np.array([p[1] for p in SOUTH]))
    for i in (2, 3):
        pts.append((float(xs_[i]), float(zs_[i])))
    return [[round(a, 1), round(b, 1)] for a, b in pts]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arch = []
    for b0, b1, top in ranges():
        arch += [face_down(b0, b1, top), face_up(b0, b1, top)]
        arch.append(top_of(b0, b1, top))
    dam = merge(arch + ends() + crest_works() + spillway() + outlets() + block(NORTH, 61) + block(SOUTH, 81))
    trees = merge(pines())
    rock = gorge()
    tris, size = write_glb(OUT / "model.glb", [("presa_de_el_atazar", dam), ("pinos", trees), ("roca", rock)],
                           roughness=0.9)
    allv = np.vstack([dam.v, trees.v, rock.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    print("dam:", dam.tris, "trees:", trees.tris, "rock:", rock.tris)
    lo2, hi2 = dam.v.min(0), dam.v.max(0)
    print(f"dam alone: x {lo2[0]:.1f}..{hi2[0]:.1f}, y {lo2[1]:.1f}..{hi2[1]:.1f}, z {lo2[2]:.1f}..{hi2[2]:.1f}")
    print("footprint:", json.dumps(footprint()))
    bs = bspan(B0, B1, 0.5)
    x, z = polar(bs, (rd_top(bs) + ru_top(bs)) / 2)
    print("crest length (centreline of the arch):", round(float(np.hypot(np.diff(x), np.diff(z)).sum()), 1))


if __name__ == "__main__":
    main()
