"""
Presa de Santillana (the dam of the Embalse de Santillana, Q3373807), Manzanares el Real — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The work: the dam of the Canal de Isabel II on the river Manzanares. The first dam, of masonry, was built in
1907–1908; the present one, an embankment of rockfill with an asphalt facing, was built in front of it in twelve
months and is in service since 1971, five metres higher; the old dam is left standing in the reservoir with its
two towers. A 20th-century engineering work, taken as protected.

Measurements (see meta.json for the sources):
- OpenStreetMap way 48848775 (waterway=dam): the embankment in two straight wings joined by a curve — the west
  wing runs at 116° for about 330 m, the east wing at 76° for about 220 m — and the low dikes that close the
  reservoir beyond both ends; way 32261699, the road on the crest; ways 599418082 and 1242506681, the two towers
  of the old dam («Torreón de la presa original») with their footbridges 599418083 and 1242506682; way
  1242506692, the spillway at the west end; way 83397843, the river through the bottom outlet.
- Wikipedia (es, en): rockfill dam with an asphalt facing, 40 m over its foundation (856.5 m to 896.5 m), 1355 m
  long with its dikes; the 1908 dam stands upstream of it.
- IGN MDT05 (terrain.json): the crest at 894.5–895 m (smoothed by the 8 m grid), the reservoir at 892 m, the
  downstream face at about 2 : 1 overall, the river below the dam at 863.4 m (y = 0).
- PNOA orthophoto: the asphalt facing, the crest road between its parapets, the berm on the downstream face, the
  old masonry dam (curved west of the main tower, nearly straight east of it), the two towers and footbridges,
  the spillway at the west end (two gated bays, a road bridge, a chute with a middle wall going south).
- Photos on Commons: the castle-like towers of the old dam (corner turrets, machicolations, merlons), the old
  dam's crest with its row of posts, the light grey rockfill of the downstream face with its berm track, the dark
  asphalt of the upstream face.

Axes (docs/MODEL_SPEC.md): +Y up, the downstream face of the west wing faces +Z (205.9°), +X along the west wing
towards the curve (ESE); origin on the dam at 40.70652, -3.81806 (OSM, the tour stop), just downstream of the crest
road; y = 0 is the river below the dam (863.4 m, ground_at / ground_elev). The reservoir water is drawn by the game.

    python model_sources/Q3373807_presa_de_santillana.py   -> models/Q3373807/model.glb
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
from meshkit import Mesh, merge, box, cyl, triangulate, fbm, noise3, write_glb  # noqa: E402

OUT = ROOT / "models" / "Q3373807"

LAT0, LON0 = 40.70652, -3.81806
HEADING = 205.9
E0 = 863.4                      # the river below the dam = y 0
GA_EN = (-100.2, -26.8)         # where y = 0 meets the ground (east/north of the origin)
Y_C = 895.5 - E0                # the crest road
Y_PAR = 896.5 - E0              # top of the upstream parapet (the highest point of the dam, Wikipedia)
Y_OLD = 892.5 - E0              # crest of the 1908 dam
SU = 1.7                        # upstream face (asphalt), 1.7 : 1
SD = 1.8                        # downstream face (rockfill), 1.8 : 1
CW = 5.0                        # half the crest width
Y_BERM = Y_C - 12.0             # the berm on the downstream face
BW = 4.0

ASPH = np.array((0.29, 0.29, 0.30))
ROCK = np.array((0.62, 0.60, 0.56))
CREST = np.array((0.74, 0.73, 0.69))
ROAD = (0.30, 0.30, 0.31)
TRACK = np.array((0.58, 0.53, 0.45))
OLD = np.array((0.66, 0.61, 0.52))
STONE = np.array((0.64, 0.58, 0.49))
CONC = np.array((0.71, 0.70, 0.66))
STEEL = (0.30, 0.34, 0.36)
DARK = (0.07, 0.07, 0.08)
RAIL = (0.55, 0.57, 0.58)
ROOF = (0.50, 0.32, 0.24)


# ── Geometry of the site ─────────────────────────────────────────────────────────────────────────────
_h = math.radians(HEADING)


def to_model(e, n):
    """Metres east/north -> model x, z (+Z = heading, +X = heading − 90°)."""
    return (-e * math.cos(_h) + n * math.sin(_h), e * math.sin(_h) + n * math.cos(_h))


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


def gnd(x, z):
    return T(x, z) - E0


# ── Mesh helpers ─────────────────────────────────────────────────────────────────────────────────────
def orient(m, out):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    if float(np.dot(np.cross(b - a, c - a).sum(0), out)) < 0:
        m.f = m.f[:, ::-1]
    return m


def clean(m):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    m.f = m.f[np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-6]
    return m


def tone(base, amp=0.08, seed=5, freq=0.2):
    base = np.asarray(base, float)
    return lambda v: np.clip(np.outer(1 + amp * (fbm(v, freq, seed, 2) - 0.5) * 2, base), 0, 1)


def rubble(base, seed):
    """Rockfill: big blotches and a fine grain of blocks."""
    base = np.asarray(base, float)

    def fn(v):
        k = 1 + 0.22 * (fbm(v, 0.06, seed, 3) - 0.5) + 0.16 * (noise3(v, 0.9, seed + 3) - 0.5)
        return np.clip(np.outer(k, base), 0, 1)
    return fn


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def place_en(m):
    """A mesh drawn with x = east, z = north (metres from the origin) -> model axes (a mirror: faces flipped)."""
    e, n = m.v[:, 0].copy(), m.v[:, 2].copy()
    x, z = to_model(e, n)
    m.v = np.stack([x, m.v[:, 1], z], 1)
    m.f = m.f[:, ::-1]
    return m


class Frame:
    """A local frame: w across, u along a direction (both in model metres), y up; a rotation, no mirror."""

    def __init__(self, o_en, bearing):
        self.o = np.array(to_model(*o_en))
        b = math.radians(bearing)
        p = np.array(to_model(math.sin(b), math.cos(b))) - np.array(to_model(0, 0))
        self.U = p / np.linalg.norm(p)
        self.W = np.array([self.U[1], -self.U[0]])

    def place(self, m):
        w, u = m.v[:, 0].copy(), m.v[:, 2].copy()
        m.v = np.stack([self.o[0] + w * self.W[0] + u * self.U[0], m.v[:, 1],
                        self.o[1] + w * self.W[1] + u * self.U[1]], 1)
        return m

    def xz(self, w, u):
        return self.o[0] + w * self.W[0] + u * self.U[0], self.o[1] + w * self.W[1] + u * self.U[1]


# ── A swept line (the axis of the dam, a dike, the old dam) ──────────────────────────────────────────
class Sweep:
    def __init__(self, pts_en, smooth=0):
        P = np.array([to_model(*p) for p in pts_en])
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        ss = np.linspace(0, cum[-1], int(cum[-1]) + 1)
        Q = np.stack([np.interp(ss, cum, P[:, 0]), np.interp(ss, cum, P[:, 1])], 1)
        if smooth:
            pad = np.concatenate([2 * Q[0] - Q[smooth:0:-1], Q, 2 * Q[-1] - Q[-2:-smooth - 2:-1]])
            k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
            Q = np.stack([np.convolve(pad[:, 0], k, "valid"), np.convolve(pad[:, 1], k, "valid")], 1)
        self.s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
        self.L = float(self.s[-1])
        self.Q = Q
        t = np.gradient(Q, axis=0)
        t /= np.linalg.norm(t, axis=1)[:, None]
        self.t = t
        self.n = np.stack([-t[:, 1], t[:, 0]], 1)          # right of the direction of travel: downstream

    def xz(self, s, d):
        s, d = np.broadcast_arrays(np.asarray(s, float), np.asarray(d, float))
        qx = np.interp(s, self.s, self.Q[:, 0]); qz = np.interp(s, self.s, self.Q[:, 1])
        nx = np.interp(s, self.s, self.n[:, 0]); nz = np.interp(s, self.s, self.n[:, 1])
        k = np.hypot(nx, nz)
        return qx + d * nx / k, qz + d * nz / k

    def tangent(self, s):
        return np.array([np.interp(s, self.s, self.t[:, 0]), np.interp(s, self.s, self.t[:, 1])])

    def g(self, s, d):
        return gnd(*self.xz(s, d))


def svals(a, b, step):
    n = max(1, int(math.ceil(abs(b - a) / step)))
    return np.linspace(a, b, n + 1)


def grid_mesh(sw, S, D, Y, out_sign, col):
    """A sheet over stations (rows) × profile points (columns): S, D, Y arrays of the same shape."""
    x, z = sw.xz(S, D)
    P = np.stack([x, Y, z], -1)
    nu, nv = P.shape[0] - 1, P.shape[1] - 1
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    m = Mesh(P.reshape(-1, 3), np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]), col)
    # orientation: the normal must point to out_sign (callers give a point that is outside)
    return clean(m), out_sign


def sweep_prism(sw, ss, poly, col, caps=(True, True)):
    """A prism along the sweep from a closed cross-section polygon [(d, y), …] (anticlockwise in d–y)."""
    poly = list(poly)
    n = len(poly)
    area = sum(poly[k][0] * poly[(k + 1) % n][1] - poly[(k + 1) % n][0] * poly[k][1] for k in range(n))
    sgn = 1.0 if area > 0 else -1.0
    out = []
    ss = np.asarray(ss, float)
    for i in range(n):
        (d0, y0), (d1, y1) = poly[i], poly[(i + 1) % n]
        S = np.stack([ss, ss], 1)
        D = np.stack([np.full(len(ss), d0), np.full(len(ss), d1)], 1)
        Y = np.stack([np.full(len(ss), y0), np.full(len(ss), y1)], 1)
        m, _ = grid_mesh(sw, S, D, Y, None, col)
        # outward normal of the edge in the section: (dy, -dd) for an anticlockwise polygon
        od, oy = sgn * (y1 - y0), -sgn * (d1 - d0)
        a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
        if len(m.f):
            nrm = np.cross(b - a, c - a).sum(0)
            sc = float(ss[len(ss) // 2])
            nn = np.array(sw.xz(sc, 1.0)) - np.array(sw.xz(sc, 0.0))
            want = np.array([od * nn[0], oy, od * nn[1]])
            if float(np.dot(nrm, want)) < 0:
                m.f = m.f[:, ::-1]
        out.append(m)
    pts, tris = triangulate([(d, y) for d, y in poly])
    for k, s in ((0, ss[0]), (1, ss[-1])):
        if not caps[k]:
            continue
        x, z = sw.xz(np.full(len(pts), s), pts[:, 0])
        m = Mesh(np.stack([x, pts[:, 1], z], 1), tris, col)
        t = sw.tangent(s)
        out.append(orient(m, (t[0] * (1 if k else -1), 0, t[1] * (1 if k else -1))))
    return merge(out)


# ── The embankment ───────────────────────────────────────────────────────────────────────────────────
class Section:
    def __init__(self, yc, cw_up, cw_dn, su, sd, berm=None, up_col=ASPH, dn_col=ROCK, top_col=CREST, depth=3.0):
        self.yc, self.cu, self.cd, self.su, self.sd = yc, cw_up, cw_dn, su, sd
        self.berm = berm               # (y, width) or None
        self.up_col, self.dn_col, self.top_col, self.depth = up_col, dn_col, top_col, depth

    def d_up(self, y):
        return -self.cu - self.su * (self.yc - np.asarray(y, float))

    def d_dn(self, y, after_berm=True):
        y = np.asarray(y, float)
        d = self.cd + self.sd * (self.yc - y)
        if self.berm:
            yb, bw = self.berm
            d = d + np.where((y < yb) | ((y == yb) & after_berm), bw, 0.0)
        return d

    def toe(self, sw, s, side):
        """The level where the face meets the ground, walking down from the crest (1 m steps)."""
        ys = np.arange(self.yc, self.yc - 80, -0.5)
        f = self.d_up if side < 0 else self.d_dn
        d = f(ys)
        g = sw.g(np.full(len(ys), s), d)
        below = np.where(ys <= g)[0]
        y = float(ys[below[0]]) if len(below) else float(ys[-1])
        return min(y, self.yc - 0.5)


def embankment(sw, sec, s0, s1, step=4.0, caps=(True, True), seed=1):
    ss = svals(s0, s1, step)
    tu = np.array([sec.toe(sw, s, -1) for s in ss]) - sec.depth
    td = np.array([sec.toe(sw, s, +1) for s in ss]) - sec.depth
    out = []
    K = 10
    # upstream face
    fr = np.linspace(0, 1, K + 1)
    Yu = sec.yc + (tu[:, None] - sec.yc) * fr[None, :]
    Su = np.repeat(ss[:, None], K + 1, 1)
    m = clean(grid_mesh(sw, Su, sec.d_up(Yu), Yu, None, sec.up_col)[0])
    sc = float(ss[len(ss) // 2]); nn = np.array(sw.xz(sc, 1.0)) - np.array(sw.xz(sc, 0.0))
    out.append(orient(m, (-nn[0], 0.6, -nn[1])).paint(tone(sec.up_col, 0.07, seed, 0.05)))
    # downstream face: crest to berm, the berm, berm to the toe
    if sec.berm:
        yb = sec.berm[0]
        ya = np.maximum(td, yb)
        Y1 = sec.yc + (ya[:, None] - sec.yc) * np.linspace(0, 1, 5)[None, :]
        D1 = sec.d_dn(Y1, after_berm=False)
        Y2 = ya[:, None] + (td[:, None] - ya[:, None]) * np.linspace(0, 1, K - 3)[None, :]
        D2 = sec.d_dn(Y2, after_berm=True)
        D2 = np.where(td[:, None] >= yb, D1[:, -1:], D2)
        for Y, D in ((Y1, D1), (Y2, D2)):
            S = np.repeat(ss[:, None], Y.shape[1], 1)
            m = clean(grid_mesh(sw, S, D, Y, None, sec.dn_col)[0])
            out.append(orient(m, (nn[0], 0.6, nn[1])).paint(rubble(sec.dn_col, seed + 7)))
        # the berm: a flat track
        S = np.repeat(ss[:, None], 2, 1)
        D = np.stack([sec.d_dn(np.full(len(ss), yb), False), sec.d_dn(np.full(len(ss), yb), True)], 1)
        Yb = np.where(td[:, None] < yb, yb, np.nan) * np.ones((1, 2))
        ok = ~np.isnan(Yb[:, 0])
        if ok.sum() > 1:
            idx = np.where(ok)[0]
            for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                if len(run) < 2:
                    continue
                m = clean(grid_mesh(sw, S[run], D[run], np.full((len(run), 2), yb + 0.02), None, TRACK)[0])
                out.append(orient(m, (0, 1, 0)).paint(tone(TRACK, 0.06, seed + 9, 0.1)))
    else:
        Y = sec.yc + (td[:, None] - sec.yc) * fr[None, :]
        S = np.repeat(ss[:, None], K + 1, 1)
        m = clean(grid_mesh(sw, S, sec.d_dn(Y), Y, None, sec.dn_col)[0])
        out.append(orient(m, (nn[0], 0.6, nn[1])).paint(rubble(sec.dn_col, seed + 7)))
    # the crest
    S = np.repeat(ss[:, None], 2, 1)
    D = np.stack([np.full(len(ss), -sec.cu), np.full(len(ss), sec.cd)], 1)
    m = clean(grid_mesh(sw, S, D, np.full(S.shape, sec.yc), None, sec.top_col)[0])
    out.append(orient(m, (0, 1, 0)).paint(tone(sec.top_col, 0.05, seed + 4)))
    # end caps: the section at s0 / s1
    for k, (s, a, b) in enumerate(((ss[0], tu[0], td[0]), (ss[-1], tu[-1], td[-1]))):
        if not caps[k]:
            continue
        prof = [(float(sec.d_up(a)), a), (float(sec.d_up(sec.yc)), sec.yc), (sec.cd, sec.yc)]
        if sec.berm and b < sec.berm[0]:
            yb = sec.berm[0]
            prof += [(float(sec.d_dn(yb, False)), yb), (float(sec.d_dn(yb, True)), yb)]
        prof += [(float(sec.d_dn(b)), b)]
        ybase = min(a, b)
        prof = [(prof[0][0], ybase)] + prof[1:-1] + [(prof[-1][0], prof[-1][1]), (prof[-1][0], ybase)] \
            if a != b else prof
        pts, tris = triangulate(prof)
        x, z = sw.xz(np.full(len(pts), s), pts[:, 0])
        m = Mesh(np.stack([x, pts[:, 1], z], 1), tris, CONC * 0.95)
        t = sw.tangent(s)
        sg = 1 if k else -1
        out.append(orient(m, (t[0] * sg, 0, t[1] * sg)))
    return out, ss, tu, td


def crest_furniture(sw, s0, s1, road=True, parapet=True):
    ss = svals(s0, s1, 4.0)
    out = []
    if road:
        out.append(sweep_prism(sw, ss, [(-3.5, Y_C - 0.1), (3.5, Y_C - 0.1), (3.5, Y_C + 0.04), (-3.5, Y_C + 0.04)][::-1], ROAD))
    if parapet:
        out.append(sweep_prism(sw, ss, [(-5.0, Y_C - 0.2), (-5.0, Y_PAR), (-4.55, Y_PAR), (-4.55, Y_C - 0.2)][::-1],
                               CREST).paint(tone(CREST, 0.04, 6)))
        out.append(sweep_prism(sw, ss, [(4.55, Y_C - 0.2), (4.55, Y_C + 0.35), (5.0, Y_C + 0.35), (5.0, Y_C - 0.2)][::-1],
                               CREST * 0.96))
    return out


# ── The main dam, its dikes ──────────────────────────────────────────────────────────────────────────
ROAD_EN = [(-296.0, 157.8), (-286.7, 153.1), (37.6, -11.4), (51.7, -18.2), (68.5, -22.8), (80.1, -25.2),
           (98.2, -27.0), (114.0, -26.4), (130.5, -23.1), (349.2, 30.2), (380.1, 37.0), (403.7, 39.7)]
MAIN = Sweep(ROAD_EN, smooth=8)
SEC = Section(Y_C, CW, CW, SU, SD, berm=(Y_BERM, BW))

WEST_DIKE = Sweep([(-508.0, 262.5), (-352.0, 211.0), (-338.0, 205.5), (-324.5, 196.5), (-315.0, 185.0),
                   (-310.5, 172.0), (-311.0, 163.0)], smooth=4)
EAST_DIKE = Sweep([(394.5, 49.0), (409.0, 85.0), (422.0, 112.0), (440.0, 122.0), (500.0, 144.0),
                   (542.0, 159.0)], smooth=4)
DSEC = Section(Y_C, 3.0, 3.0, 1.7, 1.7, berm=None, up_col=ROCK * 0.85, dn_col=ROCK, top_col=TRACK)


def main_dam():
    out, ss, tu, td = embankment(MAIN, SEC, 0.0, MAIN.L, step=4.0, caps=(True, True), seed=1)
    out += crest_furniture(MAIN, 0.0, MAIN.L)
    return out


def dikes():
    out = []
    for sw, seed in ((WEST_DIKE, 21), (EAST_DIKE, 31)):
        o, *_ = embankment(sw, DSEC, 0.0, sw.L, step=4.0, caps=(True, True), seed=seed)
        out += o
    return out


# ── The 1908 dam and its towers ──────────────────────────────────────────────────────────────────────
OLD_W = Sweep([(-269.0, 190.0), (-220.0, 183.0), (-170.0, 171.0), (-120.0, 159.0), (-70.0, 137.0),
               (-30.0, 116.0), (0.0, 96.5), (25.0, 77.0), (50.0, 57.0), (75.0, 36.0), (96.0, 16.0)], smooth=10)
OLD_E = Sweep([(110.0, 17.5), (120.0, 23.5), (150.0, 33.5), (200.0, 47.0), (250.0, 55.5), (300.0, 59.5),
               (342.5, 61.0)], smooth=6)
Y_OLD_BASE = 881.0 - E0             # (under the water; its real foundation is deeper)


def old_dam(sw, seed):
    ss = svals(0.0, sw.L, 4.0)
    out = [sweep_prism(sw, ss, [(-2.6, Y_OLD_BASE), (2.6 + 0.3 * (Y_OLD - Y_OLD_BASE), Y_OLD_BASE), (2.6, Y_OLD - 0.9),
                                (2.9, Y_OLD - 0.9), (2.9, Y_OLD), (-2.9, Y_OLD), (-2.9, Y_OLD - 0.9),
                                (-2.6, Y_OLD - 0.9)], OLD).paint(tone(OLD, 0.07, seed, 0.15))]
    # the row of posts along the upstream edge, linked by a low wall
    out.append(sweep_prism(sw, ss, [(-2.75, Y_OLD), (-2.25, Y_OLD), (-2.25, Y_OLD + 0.45), (-2.75, Y_OLD + 0.45)][::-1],
                           OLD * 0.95))
    for s in np.arange(3.0, sw.L - 2.0, 7.5):
        x, z = sw.xz(s, -2.5)
        t = sw.tangent(s)
        ang = math.atan2(t[1], t[0])
        p = box(0.75, 1.05, 0.75, OLD * 1.02).R(0, -ang, 0).T(float(x), Y_OLD, float(z))
        out.append(p)
    return out


def castle_tower(cx, cn, w, dn, y0, ytop, turrets, seed, windows=3):
    """A square tower of the old dam (east x north, metres), corner turrets, machicolated crown, merlons."""
    out = [B(cx - w / 2, cx + w / 2, y0, ytop, cn - dn / 2, cn + dn / 2, STONE).paint(tone(STONE, 0.06, seed, 0.25))]
    out.append(B(cx - w / 2 - 0.5, cx + w / 2 + 0.5, ytop - 1.2, ytop, cn - dn / 2 - 0.5, cn + dn / 2 + 0.5,
                 STONE * 1.04))
    for k in range(int(w // 1.6)):
        for side in (-1, 1):
            x = cx - w / 2 + 0.6 + k * 1.6
            out.append(B(x, x + 0.8, ytop, ytop + 1.0, cn + side * (dn / 2 + 0.5) - 0.3, cn + side * (dn / 2 + 0.5) + 0.3,
                         STONE * 1.02))
    for k in range(int(dn // 1.6)):
        for side in (-1, 1):
            z = cn - dn / 2 + 0.6 + k * 1.6
            out.append(B(cx + side * (w / 2 + 0.5) - 0.3, cx + side * (w / 2 + 0.5) + 0.3, ytop, ytop + 1.0, z, z + 0.8,
                         STONE * 1.02))
    for (tx, tz, r) in turrets:
        out.append(cyl(r, r, ytop - y0 + 0.8, STONE * 0.98, seg=12).T(tx, y0, tz))
        out.append(cyl(r + 0.4, r + 0.4, 1.2, STONE * 1.04, seg=12).T(tx, ytop - 0.4, tz))
        for a in np.linspace(0, 2 * math.pi, 7)[:-1]:
            out.append(B(tx + (r + 0.2) * math.cos(a) - 0.35, tx + (r + 0.2) * math.cos(a) + 0.35, ytop + 0.8, ytop + 1.7,
                         tz + (r + 0.2) * math.sin(a) - 0.35, tz + (r + 0.2) * math.sin(a) + 0.35, STONE * 1.02))
    # windows: dark slits on the four faces, one per floor
    ys = np.linspace(y0 + 4.0, ytop - 3.0, windows)
    for y in ys:
        for k in (-1, 1):
            out.append(B(cx + k * w / 4 - 0.45, cx + k * w / 4 + 0.45, y, y + 1.8, cn - dn / 2 - 0.05, cn + dn / 2 + 0.05,
                         DARK))
            out.append(B(cx - w / 2 - 0.05, cx + w / 2 + 0.05, y, y + 1.8, cn + k * dn / 4 - 0.45, cn + k * dn / 4 + 0.45,
                         DARK))
    return out


def towers():
    out = []
    # the main tower on the old dam, at the angle of the new one (OSM way 599418082)
    cx, cn, w = 102.5, 13.5, 11.0
    out += castle_tower(cx, cn, w, w, Y_OLD_BASE, 917.0 - E0, [(cx + k * w / 2, cn + j * w / 2, 2.1)
                                                                for k in (-1, 1) for j in (-1, 1)], 41, windows=4)
    out.append(B(cx - w / 2 - 2, cx + w / 2 + 2, Y_OLD_BASE, Y_OLD, cn - w / 2 - 2, cn + w / 2 + 2, OLD))
    # the east tower (way 1242506681): a lower block with two turrets towards the east
    out += castle_tower(346.5, 58.6, 8.0, 15.0, Y_OLD_BASE, 901.5 - E0, [(351.4, 53.6, 2.6), (352.4, 63.4, 2.6)], 43,
                        windows=1)
    return [place_en(m) for m in out]


def footbridge(a_en, b_en, y, seed, piers=True):
    """A footbridge over the water between the crest and a tower (deck, railings, slender piers)."""
    a, b = np.array(a_en), np.array(b_en)
    L = float(np.linalg.norm(b - a))
    brg = math.degrees(math.atan2(*(b - a)))
    fr = Frame(tuple(a), brg)
    out = [B(-1.3, 1.3, y - 0.5, y, 0, L, CONC), B(-1.35, -1.2, y, y + 1.05, 0, L, RAIL), B(1.2, 1.35, y, y + 1.05, 0, L, RAIL)]
    if piers:
        for u in np.arange(6.0, L - 3.0, 7.0):
            out.append(B(-0.6, 0.6, Y_OLD_BASE, y - 0.5, u - 0.6, u + 0.6, CONC * 0.95))
    return [fr.place(m) for m in out]


# ── The spillway at the west end (two gated bays, a road bridge, the chute) ─────────────────────────
SP_G = (-304.0, 157.5)            # middle of the gate line (east, north)
SP_END = (-311.5, 96.0)
SP_BRG = math.degrees(math.atan2(SP_END[0] - SP_G[0], SP_END[1] - SP_G[1]))
SP_L = math.hypot(SP_END[0] - SP_G[0], SP_END[1] - SP_G[1])
SP_HW = 7.0                       # half the clear width (two bays of 6.4 m and a middle pier)
Y_SILL = 890.0 - E0


def chute_floor():
    u = [-8.0, -1.0, 2.0, 5.0, 9.0, 14.0, 20.0, 40.0, SP_L - 6.0, SP_L - 3.0, SP_L]
    y = [Y_SILL - 1.0, Y_SILL, Y_SILL + 0.3, Y_SILL - 0.4, Y_SILL - 3.0, 881.8 - E0, 881.2 - E0, 880.5 - E0,
         879.4 - E0, 878.6 - E0, 879.6 - E0]
    return u, y


def spillway():
    fr = Frame(SP_G, SP_BRG)
    out = []
    us, ys = chute_floor()
    # the floor: a prism from the profile in (u, y), across the clear width, 3 m thick
    prof = [(u, y) for u, y in zip(us, ys)] + [(u, y - 3.0) for u, y in zip(us, ys)][::-1]
    pts, tris = triangulate(prof)
    for wv, sg in ((-SP_HW, -1), (SP_HW, 1)):
        m = Mesh(np.stack([np.full(len(pts), wv), pts[:, 1], pts[:, 0]], 1), tris, CONC * 0.9)
        out.append(orient(m, (sg, 0, 0)))
    n = len(prof)
    for i in range(n):
        (u0, y0), (u1, y1) = prof[i], prof[(i + 1) % n]
        q = Mesh([(-SP_HW, y0, u0), (SP_HW, y0, u0), (SP_HW, y1, u1), (-SP_HW, y1, u1)], [(0, 1, 2), (0, 2, 3)], CONC)
        area = sum(prof[k][0] * prof[(k + 1) % n][1] - prof[(k + 1) % n][0] * prof[k][1] for k in range(n))
        nu, ny = (y1 - y0), -(u1 - u0)
        if area < 0:
            nu, ny = -nu, -ny
        out.append(orient(q, (0, ny, nu)).paint(tone(CONC, 0.05, 61, 0.1)))
    # side walls and the middle wall: 3.2 m over the floor, rising to the crest at the head
    for wc, th in ((-SP_HW - 0.6, 1.2), (SP_HW + 0.6, 1.2), (0.0, 0.8)):
        top = []
        for u, y in zip(us, ys):
            if u <= 5:
                yt = Y_C
            elif u >= 12:
                yt = y + 3.2
            else:
                yt = y + 3.2 + (12 - u) / 7 * (Y_C - y - 3.2)
            top.append((u, yt))
        wall = [(u, y - 3.0) for u, y in zip(us, ys)] + top[::-1]
        pts, tris = triangulate(wall)
        for xo, sg in ((wc - th / 2, -1), (wc + th / 2, 1)):
            m = Mesh(np.stack([np.full(len(pts), xo), pts[:, 1], pts[:, 0]], 1), tris, CONC * 0.97)
            out.append(orient(m, (sg, 0, 0)))
        nw = len(wall)
        for i in range(nw):
            (u0, y0), (u1, y1) = wall[i], wall[(i + 1) % nw]
            q = Mesh([(wc - th / 2, y0, u0), (wc + th / 2, y0, u0), (wc + th / 2, y1, u1), (wc - th / 2, y1, u1)],
                     [(0, 1, 2), (0, 2, 3)], CONC * 0.97)
            area = sum(wall[k][0] * wall[(k + 1) % nw][1] - wall[(k + 1) % nw][0] * wall[k][1] for k in range(nw))
            nu, ny = (y1 - y0), -(u1 - u0)
            if area < 0:
                nu, ny = -nu, -ny
            out.append(orient(q, (0, ny, nu)))
    # the piers of the gates (head of the walls, up to the road), the gates and the road bridge
    for wc in (-SP_HW - 0.6, 0.0, SP_HW + 0.6):
        out.append(B(wc - 0.9, wc + 0.9, Y_SILL - 1, Y_PAR + 2.2, -4.0, 4.0, CONC).paint(tone(CONC, 0.05, 63, 0.2)))
        out.append(B(wc - 1.1, wc + 1.1, Y_PAR + 2.2, Y_PAR + 2.5, -4.2, 4.2, CONC * 1.03))
    for k in (-1, 1):
        w0, w1 = (0.4, SP_HW) if k > 0 else (-SP_HW, -0.4)
        out.append(B(w0, w1, Y_SILL, Y_C - 0.6, -0.9, -0.5, STEEL))           # the gate leaf
        for j in range(4):
            out.append(B(w0, w1, Y_SILL + 0.8 + 1.1 * j, Y_SILL + 1.0 + 1.1 * j, -0.5, -0.2, (0.25, 0.28, 0.30)))
        # the hoist house over each gate
        out.append(B(w0 + 0.6, w1 - 0.6, Y_PAR + 2.5, Y_PAR + 4.6, -2.0, 1.0, CREST))
        out.append(B(w0 + 0.4, w1 - 0.4, Y_PAR + 4.6, Y_PAR + 4.9, -2.2, 1.2, (0.45, 0.45, 0.45)))
    # the road bridge over the inlet, downstream of the gates
    out.append(B(-SP_HW - 1.5, SP_HW + 1.5, Y_C - 1.0, Y_C, 2.5, 11.0, CONC * 0.95))
    out.append(B(-SP_HW - 1.5, SP_HW + 1.5, Y_C, Y_C + 0.04, 3.5, 10.0, ROAD))
    for z in (2.6, 10.9):
        out.append(B(-SP_HW - 1.5, SP_HW + 1.5, Y_C, Y_C + 1.0, z - 0.2, z + 0.2, CREST))
    # the stilling basin wall at the end
    out.append(B(-SP_HW - 1.2, SP_HW + 1.2, 878.6 - E0 - 1, 880.4 - E0, SP_L - 0.6, SP_L + 0.6, CONC * 0.9))
    return [fr.place(m) for m in out]


# ── Buildings at the foot ────────────────────────────────────────────────────────────────────────────
def outlet_house():
    """The house of the bottom outlet where the river leaves its tunnel (OSM way 83397843)."""
    fr = Frame((-38.0, -64.0), 186.5)
    yb = float(gnd(*fr.xz(0.0, 0.0))) - 2.0
    out = [B(-5.0, 5.0, yb, yb + 7.5, -4.5, 4.5, CREST * 0.95).paint(tone(CREST, 0.05, 71, 0.2)),
           B(-5.3, 5.3, yb + 7.5, yb + 7.9, -4.8, 4.8, CREST),
           B(-4.8, 4.8, yb + 7.9, yb + 8.6, -4.3, 4.3, ROOF)]
    for k in (-2.5, 0, 2.5):
        out.append(B(k - 0.5, k + 0.5, yb + 4.0, yb + 6.0, 4.5, 4.55, DARK))
    out.append(B(-1.8, 1.8, yb + 0.2, yb + 3.0, 4.5, 4.56, DARK))
    return [fr.place(m) for m in out]


# ── Assembly ─────────────────────────────────────────────────────────────────────────────────────────
def footprint():
    """Seen from above: the main dam from the upstream toe to the downstream toe, 2 m round."""
    ups, dns = [], []
    for s in svals(0.0, MAIN.L, 20.0):
        a = SEC.toe(MAIN, s, -1)
        b = SEC.toe(MAIN, s, +1)
        ups.append(MAIN.xz(s, float(SEC.d_up(a)) - 2.0))
        dns.append(MAIN.xz(s, float(SEC.d_dn(b)) + 2.0))
    # and the spillway at the west end: its chute goes south from the end of the dam
    fr = Frame(SP_G, SP_BRG)
    spill = [fr.xz(SP_HW + 2.5, SP_L + 2.0), fr.xz(-SP_HW - 2.5, SP_L + 2.0), fr.xz(-SP_HW - 2.5, -9.0)]
    ring = [(float(x), float(z)) for x, z in ups] + [(float(x), float(z)) for x, z in dns[::-1]] +         [(float(x), float(z)) for x, z in spill]
    return [[round(x, 1), round(z, 1)] for x, z in ring]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = main_dam() + dikes()
    parts += old_dam(OLD_W, 51) + old_dam(OLD_E, 53) + towers()
    parts += footbridge((80.1, -25.2), (86.7, 10.6), Y_C - 0.3, 1) + footbridge((86.7, 10.6), (97.0, 13.3), Y_C - 0.3, 2)
    parts += footbridge((351.9, 30.6), (346.6, 51.1), Y_C - 0.3, 3)
    parts += spillway() + outlet_house()
    dam = merge(parts)
    tris, size = write_glb(OUT / "model.glb", [("presa_de_santillana", dam)], roughness=0.9)
    lo, hi = dam.v.min(0), dam.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    ga = to_model(*GA_EN)
    print(f"ground_at {ga[0]:.1f}, {ga[1]:.1f}: MDT {float(T(*ga)):.2f} (E0 {E0})")
    print(f"main axis {MAIN.L:.1f} m; west dike {WEST_DIKE.L:.1f} m; east dike {EAST_DIKE.L:.1f} m; "
          f"spillway {SP_L:.1f} m at {SP_BRG:.1f}°")
    # how far the origin is from the crest road (perpendicular to the axis)
    k = int(np.argmin(np.hypot(MAIN.Q[:, 0], MAIN.Q[:, 1])))
    print(f"origin: {float(np.hypot(*MAIN.Q[k])):.1f} m from the crest axis; ground there {float(gnd(0, 0)):.2f} m")
    for s in svals(0, MAIN.L, 50):
        print(f"  s {s:6.1f}: toe up {SEC.toe(MAIN, s, -1):6.1f}, toe dn {SEC.toe(MAIN, s, 1):6.1f}, "
              f"MDT on the axis {float(MAIN.g(s, 0)):6.1f}")
    print("footprint:", json.dumps(footprint()))


if __name__ == "__main__":
    main()
