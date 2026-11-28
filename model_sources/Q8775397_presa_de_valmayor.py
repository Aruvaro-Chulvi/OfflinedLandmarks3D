"""
Presa de Valmayor (the dam of the Embalse de Valmayor, Q8775397), Valdemorillo — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The work: the rockfill dam of the Canal de Isabel II on the river Aulencia, finished in 1976 (project and works
directed by Emeterio Cuadrado Díaz, engineer of the Canal); when built, the highest rockfill dam in Spain. A
20th-century engineering work, taken as protected.

Measurements (see meta.json for the sources):
- OpenStreetMap way 26237156 (waterway=dam): the dam, straight, about 870 m from the north abutment to the south
  one; the upstream edge runs at 3.3°, so the downstream face looks east. Way 169766573: the M-510 road on the
  crest (the axis of the model, at 3.1°); way 186665972, the Aulencia through the bottom outlet; way 27263734, the
  spillway channel that leaves the south end of the dam towards the east (79°).
- Wikipedia (es): rockfill dam, 60 m high over the river bed, 1125 m long (with its abutments), 2.1 hm³, 1976.
- IGN MDT05 (terrain.json): the crest at 833–834 m, the reservoir at 826 m, the toe of the downstream face at
  783 m near the river, the river below the dam at 778.3 m (y = 0), the spillway channel at 824–820 m.
- PNOA orthophoto: the crest (a paved platform and parapet on the upstream side, the two-lane road, the kerb on
  the downstream side), the jointed facing of the upstream face, the grey rockfill of the downstream face with
  two berm tracks (at about 29–33 m and 46–50 m from the axis), the round intake tower in the reservoir with its
  footbridge, the gated head of the spillway under the road at the south end and its concrete channel, the
  valve house below the dam (green roof), the outlet works at the foot.
No close photographs of the dam were found on Commons; heights of the tower and buildings are estimated.

Axes (docs/MODEL_SPEC.md): +Y up, the downstream face faces +Z (east, 93.1°), +X along the dam to the north;
origin on the dam at 40.53560, -4.04709 (OSM, the tour stop), on the downstream face 18 m east of the crest
axis; y = 0 is the river below the dam (778.3 m, ground_at / ground_elev). The reservoir water is drawn by the game.

    python model_sources/Q8775397_presa_de_valmayor.py   -> models/Q8775397/model.glb
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

OUT = ROOT / "models" / "Q8775397"

LAT0, LON0 = 40.53560, -4.04709
HEADING = 93.1
E0 = 778.3                      # the river below the dam = y 0
GA_EN = (103.0, 7.2)            # where y = 0 meets the ground: the outlet of the river (east/north of the origin)
Y_C = 834.0 - E0                # the crest road
Y_PAR = Y_C + 1.1               # top of the upstream parapet
SU = 1.6                        # upstream face, 1.6 : 1
SD = 1.3                        # downstream face (rockfill), 1.3 : 1 between the berms
CW_UP = 14.0                    # crest: from the road axis to the upstream edge (platform and parapet)
CW_DN = 8.0                     # … to the downstream edge
BERMS = [(818.0 - E0, 4.0), (808.0 - E0, 4.0)]

FACING = np.array((0.58, 0.58, 0.56))
ROCK = np.array((0.55, 0.56, 0.57))
CREST = np.array((0.74, 0.73, 0.70))
PLAT = np.array((0.66, 0.65, 0.62))
ROAD = (0.28, 0.28, 0.29)
LINE = (0.85, 0.85, 0.82)
TRACK = np.array((0.60, 0.56, 0.49))
CONC = np.array((0.71, 0.70, 0.66))
STEEL = (0.30, 0.34, 0.36)
DARK = (0.07, 0.07, 0.08)
RAIL = (0.55, 0.57, 0.58)
GREEN_ROOF = (0.42, 0.62, 0.55)
WALL = np.array((0.80, 0.78, 0.72))


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
    def __init__(self, yc, cw_up, cw_dn, su, sd, berms=(), up_col=FACING, dn_col=ROCK, top_col=CREST, depth=3.0):
        self.yc, self.cu, self.cd, self.su, self.sd = yc, cw_up, cw_dn, su, sd
        self.berms = sorted(berms, reverse=True)          # [(y, width)], from the top down
        self.up_col, self.dn_col, self.top_col, self.depth = up_col, dn_col, top_col, depth

    def d_up(self, y):
        return -self.cu - self.su * (self.yc - np.asarray(y, float))

    def d_dn(self, y, after_berm=True):
        y = np.asarray(y, float)
        d = self.cd + self.sd * (self.yc - y)
        for yb, bw in self.berms:
            d = d + np.where((y < yb) | ((y == yb) & after_berm), bw, 0.0)
        return d

    def toe(self, sw, s, side):
        """The level where the face meets the ground, walking down from the crest (0.5 m steps)."""
        ys = np.arange(self.yc, self.yc - 90, -0.5)
        d = self.d_up(ys) if side < 0 else self.d_dn(ys)
        g = sw.g(np.full(len(ys), s), d)
        below = np.where(ys <= g)[0]
        y = float(ys[below[0]]) if len(below) else float(ys[-1])
        return min(y, self.yc - 0.5)


def embankment(sw, sec, s0, s1, step=4.0, caps=(True, True), seed=1):
    ss = svals(s0, s1, step)
    tu = np.array([sec.toe(sw, s, -1) for s in ss]) - sec.depth
    td = np.array([sec.toe(sw, s, +1) for s in ss]) - sec.depth
    sc = float(ss[len(ss) // 2])
    nn = np.array(sw.xz(sc, 1.0)) - np.array(sw.xz(sc, 0.0))
    out = []
    # upstream face (rows evenly spaced between the crest and the foundation of each station)
    K = 10
    fr = np.linspace(0, 1, K + 1)
    Yu = sec.yc + (tu[:, None] - sec.yc) * fr[None, :]
    m = grid_mesh(sw, np.repeat(ss[:, None], K + 1, 1), sec.d_up(Yu), Yu, None, sec.up_col)[0]
    out.append(orient(m, (-nn[0], 0.6, -nn[1])).paint(tone(sec.up_col, 0.07, seed, 0.05)))
    # the joints between the slabs of the facing: thin dark strips down the face every 15 m
    for k, s in enumerate(np.arange(s0 + 7.5, s1 - 2.0, 15.0)):
        yb = float(np.interp(s, ss, tu))
        ys = np.linspace(sec.yc - 0.05, yb, 6)
        S = np.stack([np.full(6, s - 0.25), np.full(6, s + 0.25)], 1)
        D = np.repeat((sec.d_up(ys) - 0.06)[:, None], 2, 1)
        Y = np.repeat(ys[:, None], 2, 1)
        m = grid_mesh(sw, S, D, Y, None, sec.up_col * 0.72)[0]
        out.append(orient(m, (-nn[0], 0.6, -nn[1])))
    # downstream face: one sheet between each pair of berms, the berms as flat tracks
    tops = [sec.yc] + [yb for yb, _ in sec.berms]
    for k, ytop in enumerate(tops):
        ylow = tops[k + 1] if k + 1 < len(tops) else -1e9
        a = np.full(len(ss), ytop)
        b = np.maximum(td, ylow)
        if np.all(td >= ytop):
            continue
        n_r = 6
        Y = a[:, None] + (b - a)[:, None] * np.linspace(0, 1, n_r)[None, :]
        Y = np.minimum(Y, ytop)
        D = sec.d_dn(Y, after_berm=True)
        # the top row of a sheet below a berm starts after the berm
        m = grid_mesh(sw, np.repeat(ss[:, None], n_r, 1), D, Y, None, sec.dn_col)[0]
        out.append(orient(m, (nn[0], 0.6, nn[1])).paint(rubble(sec.dn_col, seed + 7 + k)))
    for yb, bw in sec.berms:
        ok = td < yb
        idx = np.where(ok)[0]
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if len(run) < 2:
                continue
            S = np.repeat(ss[run][:, None], 2, 1)
            D = np.stack([np.full(len(run), float(sec.d_dn(yb, False))), np.full(len(run), float(sec.d_dn(yb, True)))], 1)
            m = grid_mesh(sw, S, D, np.full(S.shape, yb + 0.02), None, TRACK)[0]
            out.append(orient(m, (0, 1, 0)).paint(tone(TRACK, 0.06, seed + 9, 0.1)))
    # the crest
    S = np.repeat(ss[:, None], 2, 1)
    D = np.stack([np.full(len(ss), -sec.cu), np.full(len(ss), sec.cd)], 1)
    m = grid_mesh(sw, S, D, np.full(S.shape, sec.yc), None, sec.top_col)[0]
    out.append(orient(m, (0, 1, 0)).paint(tone(sec.top_col, 0.05, seed + 4)))
    # end caps: the section at s0 / s1
    for k, (s, a, b) in enumerate(((ss[0], tu[0], td[0]), (ss[-1], tu[-1], td[-1]))):
        if not caps[k]:
            continue
        prof = [(float(sec.d_up(a)), a), (float(sec.d_up(sec.yc)), sec.yc), (sec.cd, sec.yc)]
        for yb, _ in sec.berms:
            if b < yb:
                prof += [(float(sec.d_dn(yb, False)), yb), (float(sec.d_dn(yb, True)), yb)]
        prof += [(float(sec.d_dn(b)), b)]
        ybase = min(a, b)
        if a != b:
            prof = [(prof[0][0], ybase)] + prof[1:] + [(prof[-1][0], ybase)]
        pts, tris = triangulate(prof)
        x, z = sw.xz(np.full(len(pts), s), pts[:, 0])
        m = Mesh(np.stack([x, pts[:, 1], z], 1), tris, CONC * 0.95)
        t = sw.tangent(s)
        sg = 1 if k else -1
        out.append(orient(m, (t[0] * sg, 0, t[1] * sg)))
    return out


def crest_furniture(sw, s0, s1):
    ss = svals(s0, s1, 6.0)
    out = [sweep_prism(sw, ss, [(-5.0, Y_C - 0.1), (5.0, Y_C - 0.1), (5.0, Y_C + 0.05), (-5.0, Y_C + 0.05)], ROAD),
           sweep_prism(sw, ss, [(-0.08, Y_C + 0.05), (0.08, Y_C + 0.05), (0.08, Y_C + 0.07), (-0.08, Y_C + 0.07)], LINE),
           # the platform on the upstream side and the parapet on its edge
           sweep_prism(sw, ss, [(-8.6, Y_C - 0.1), (-5.0, Y_C - 0.1), (-5.0, Y_C + 0.18), (-8.6, Y_C + 0.18)], PLAT),
           sweep_prism(sw, ss, [(-9.0, Y_C - 0.3), (-8.55, Y_C - 0.3), (-8.55, Y_PAR), (-9.0, Y_PAR)],
                       CREST).paint(tone(CREST, 0.04, 6)),
           # the kerb and the guardrail on the downstream side
           sweep_prism(sw, ss, [(5.0, Y_C - 0.1), (8.0, Y_C - 0.1), (8.0, Y_C + 0.15), (5.0, Y_C + 0.15)], PLAT * 0.97),
           sweep_prism(sw, ss, [(7.4, Y_C + 0.5), (7.55, Y_C + 0.5), (7.55, Y_C + 0.85), (7.4, Y_C + 0.85)], RAIL)]
    for s in np.arange(s0 + 2.0, s1, 4.0):
        x, z = sw.xz(s, 7.48)
        out.append(box(0.15, 0.8, 0.15, RAIL).T(float(x), Y_C + 0.05, float(z)))
    return out


# ── The dam ──────────────────────────────────────────────────────────────────────────────────────────
# the crest road (OSM way 169766573), from the south end of the dam to the north one
AXIS = Sweep([(-41.6, -433.0), (4.9, 430.0)])
SEC = Section(Y_C, CW_UP, CW_DN, SU, SD, berms=BERMS)


def the_dam():
    out = embankment(AXIS, SEC, 0.0, AXIS.L, step=5.0, caps=(True, True), seed=1)
    out += crest_furniture(AXIS, 0.0, AXIS.L)
    return out


def s_d(e, n):
    """Station and offset on the axis of a point (east, north)."""
    x, z = to_model(e, n)
    q = AXIS.Q[0]
    t = AXIS.t[0]
    nrm = AXIS.n[0]
    return float((x - q[0]) * t[0] + (z - q[1]) * t[1]), float((x - q[0]) * nrm[0] + (z - q[1]) * nrm[1])


# ── The intake tower in the reservoir and its footbridge ─────────────────────────────────────────────
TOWER_EN = (-92.0, 168.0)
TOWER_R = 5.6


def intake_tower():
    x, z = to_model(*TOWER_EN)
    y0 = 812.0 - E0
    out = [cyl(TOWER_R, TOWER_R, Y_C + 0.6 - y0, CONC, seg=24).T(x, y0, z).paint(tone(CONC, 0.05, 31, 0.3)),
           cyl(TOWER_R + 0.35, TOWER_R + 0.35, 0.5, CONC * 1.03, seg=24).T(x, Y_C + 0.6, z),
           cyl(TOWER_R + 0.1, TOWER_R + 0.1, 0.9, CONC * 0.98, seg=24, caps=False).T(x, Y_C + 1.1, z),
           cyl(TOWER_R - 0.2, TOWER_R - 0.2, 0.05, (0.66, 0.56, 0.44), seg=24).T(x, Y_C + 1.1, z),
           # the hoist house on top, towards the bridge
           B(-1.8, 1.8, Y_C + 1.1, Y_C + 3.6, -1.6, 1.6, WALL).T(x, 0, z),
           B(-2.0, 2.0, Y_C + 3.6, Y_C + 3.85, -1.8, 1.8, (0.45, 0.45, 0.45)).T(x, 0, z)]
    # a band of darker concrete where the water rises and falls
    out.append(cyl(TOWER_R + 0.03, TOWER_R + 0.03, 826.0 - 812.0, CONC * 0.8, seg=24, caps=False).T(x, y0, z))
    return out


def footbridge():
    a = np.array(to_model(TOWER_EN[0] + TOWER_R - 0.3, TOWER_EN[1]))
    s, _ = s_d(TOWER_EN[0], TOWER_EN[1])
    b = np.array(AXIS.xz(s, -8.6))
    L = float(np.linalg.norm(b - a))
    u = (b - a) / L
    w = np.array([u[1], -u[0]])

    def put(m):
        ww, uu = m.v[:, 0].copy(), m.v[:, 2].copy()
        m.v = np.stack([a[0] + ww * w[0] + uu * u[0], m.v[:, 1], a[1] + ww * w[1] + uu * u[1]], 1)
        return m
    y = Y_C + 0.6
    out = [B(-1.8, 1.8, y - 1.2, y, 0, L, CONC), B(-1.9, -1.75, y, y + 1.1, 0, L, RAIL), B(1.75, 1.9, y, y + 1.1, 0, L, RAIL)]
    for k, uu in enumerate(np.linspace(14.0, L - 6.0, 4)):
        out.append(B(-0.9, 0.9, 812.0 - E0, y - 1.2, uu - 0.9, uu + 0.9, CONC * 0.95))
    return [put(m) for m in out]


# ── The spillway: gated head at the south end of the dam, the channel to the east ───────────────────
SP_G = (-36.0, -447.5)            # middle of the gate line (east, north)
SP_BRG = 79.3                     # OSM way 27263734
SP_L = 375.0
SP_HW = 7.5                       # half the clear width
Y_SILL = 826.0 - E0


def sp_floor(u):
    u = np.asarray(u, float)
    return np.interp(u, [-14.0, -2.0, 2.0, 10.0, SP_L], [Y_SILL - 1.0, Y_SILL, Y_SILL, 824.0 - E0, 819.5 - E0])


def spillway():
    fr = Frame(SP_G, SP_BRG)
    out = []
    us = np.concatenate([[-14.0, -2.0, 2.0, 6.0, 10.0], np.arange(30.0, SP_L + 0.1, 30.0), [SP_L]])
    us = np.unique(us)
    ys = sp_floor(us)
    # the floor, 2 m thick
    prof = [(u, y) for u, y in zip(us, ys)] + [(u, y - 2.0) for u, y in zip(us, ys)][::-1]
    out.append(prism_uy(prof, -SP_HW, SP_HW, CONC * 0.9).paint(tone(CONC * 0.9, 0.06, 61, 0.05)))
    # side walls: 3.5 m over the floor along the channel, up to the crest at the head
    for wc, th in ((-SP_HW - 0.45, 0.9), (SP_HW + 0.45, 0.9)):
        top = [(u, (Y_C if u <= 6 else y + 3.5)) for u, y in zip(us, ys)]
        wall = [(u, y - 2.0) for u, y in zip(us, ys)] + top[::-1]
        out.append(prism_uy(wall, wc - th / 2, wc + th / 2, CONC).paint(tone(CONC, 0.05, 63, 0.08)))
    # the gate structure: piers, steel gates on the reservoir side, the road deck over it
    bays = [(-SP_HW, -2.6), (-1.95, 1.95), (2.6, SP_HW)]
    for wc in (-SP_HW - 0.45, -2.27, 2.27, SP_HW + 0.45):
        out.append(B(wc - 0.7, wc + 0.7, Y_SILL - 1.0, Y_C - 0.6, -9.0, 7.0, CONC).paint(tone(CONC, 0.05, 65, 0.2)))
        out.append(B(wc - 0.5, wc + 0.5, Y_C - 0.6, Y_C + 2.4, -9.0, -5.5, CONC * 1.02))
    for w0, w1 in bays:
        out.append(B(w0, w1, Y_SILL, Y_SILL + 6.2, -6.2, -5.8, STEEL))
        for j in range(4):
            out.append(B(w0, w1, Y_SILL + 0.8 + 1.4 * j, Y_SILL + 1.0 + 1.4 * j, -5.8, -5.5, (0.25, 0.28, 0.30)))
    # the hoist beam over the gates and the road on its deck
    out.append(B(-SP_HW - 1.2, SP_HW + 1.2, Y_C + 2.4, Y_C + 3.0, -9.0, -5.5, CONC * 1.03))
    out.append(B(-SP_HW - 1.2, SP_HW + 1.2, Y_C - 1.2, Y_C - 0.1, -5.5, 7.0, CONC * 0.95))
    out.append(B(-SP_HW - 1.2, SP_HW + 1.2, Y_C - 0.1, Y_C + 0.05, -4.0, 5.5, ROAD))
    for z in (-5.2, 6.7):
        out.append(B(-SP_HW - 1.2, SP_HW + 1.2, Y_C, Y_C + 1.0, z - 0.2, z + 0.2, CREST))
    # the approach on the reservoir side: training walls
    for wc in (-SP_HW - 0.45, SP_HW + 0.45):
        out.append(B(wc - 0.45, wc + 0.45, Y_SILL - 1.0, Y_C - 1.0, -16.0, -9.0, CONC * 0.95))
    return [fr.place(m) for m in out]


def prism_uy(poly, w0, w1, col):
    """A prism from an outline in (u, y) between w0 and w1 (local spillway axes: x = w, z = u)."""
    pts, tris = triangulate(poly)
    n = len(poly)
    ms = []
    for wv, sg in ((w0, -1), (w1, 1)):
        m = Mesh(np.stack([np.full(len(pts), wv), pts[:, 1], pts[:, 0]], 1), tris, col)
        ms.append(orient(m, (sg, 0, 0)))
    area = sum(poly[k][0] * poly[(k + 1) % n][1] - poly[(k + 1) % n][0] * poly[k][1] for k in range(n))
    for i in range(n):
        (u0, y0), (u1, y1) = poly[i], poly[(i + 1) % n]
        q = Mesh([(w0, y0, u0), (w1, y0, u0), (w1, y1, u1), (w0, y1, u1)], [(0, 1, 2), (0, 2, 3)], col)
        nu, ny = (y1 - y0), -(u1 - u0)
        if area < 0:
            nu, ny = -nu, -ny
        ms.append(orient(q, (0, ny, nu)))
    return clean(merge(ms))


# ── Buildings below the dam ──────────────────────────────────────────────────────────────────────────
def valve_house():
    """The big hall with a green roof below the north part of the dam (orthophoto; OSM way 822689949)."""
    yb = float(gnd(*to_model(88.0, 246.0))) - 1.5
    out = []
    hall = [(76.0, 226.0), (100.0, 226.0), (100.0, 266.0), (76.0, 266.0)]
    (e0, n0), (e1, n1) = hall[0], hall[2]
    out.append(B(e0, e1, yb, yb + 11.0, n0, n1, WALL).paint(tone(WALL, 0.04, 71, 0.2)))
    # a gently curved roof (vault along the hall)
    for k in range(6):
        a0, a1 = -0.5 + k / 6, -0.5 + (k + 1) / 6
        h0, h1 = 2.2 * (1 - (2 * a0) ** 2), 2.2 * (1 - (2 * a1) ** 2)
        ea, eb = 88 + a0 * 25, 88 + a1 * 25
        V = [(ea, yb + 11 + h0, n0 - 0.4), (eb, yb + 11 + h1, n0 - 0.4), (eb, yb + 11 + h1, n1 + 0.4), (ea, yb + 11 + h0, n1 + 0.4)]
        out.append(orient(Mesh(V, [(0, 1, 2), (0, 2, 3)], GREEN_ROOF), (0, 1, 0)))
    for n in (n0 - 0.4, n1 + 0.4):
        V = [(88 + a * 25, yb + 11 + 2.2 * (1 - (2 * a) ** 2), n) for a in np.linspace(-0.5, 0.5, 7)]
        V = [(76.0 - 0.5, yb + 11, n), (100.0 + 0.5, yb + 11, n)] + V[::-1]
        F = [(0, 1, 2)] + [(0, 2 + j, 3 + j) for j in range(len(V) - 3)]
        out.append(orient(Mesh(V, F, WALL), (0, 0, 1 if n > 240 else -1)))
    for k in range(7):
        n = n0 + 3 + 5.6 * k
        out.append(B(e0 - 0.05, e0, yb + 6.0, yb + 9.5, n, n + 2.6, DARK))
        out.append(B(e1, e1 + 0.05, yb + 6.0, yb + 9.5, n, n + 2.6, DARK))
    # the lower annex on the dam side, with its flat roof
    yb2 = float(gnd(*to_model(76.0, 240.0))) - 1.5
    out.append(B(63.0, 76.0, yb2, yb2 + 6.0, 228.0, 250.0, WALL * 0.97))
    out.append(B(62.7, 76.0, yb2 + 6.0, yb2 + 6.4, 227.7, 250.3, (0.62, 0.62, 0.60)))
    return [place_en(m) for m in out]


def outlet_house():
    """The portal of the bottom outlet at the foot of the dam (OSM way 186665972)."""
    yb = float(gnd(*to_model(66.0, -3.5))) - 1.5
    out = [B(60.0, 72.0, yb, yb + 6.5, -8.0, 1.0, CREST * 0.95).paint(tone(CREST, 0.05, 73, 0.2)),
           B(59.7, 72.3, yb + 6.5, yb + 6.9, -8.3, 1.3, CREST),
           B(72.0, 72.05, yb + 1.0, yb + 4.0, -5.5, -1.5, DARK)]
    return [place_en(m) for m in out]


# ── Assembly ─────────────────────────────────────────────────────────────────────────────────────────
def footprint():
    """Seen from above: the dam from the upstream toe to the downstream toe, 2 m round, and the spillway head."""
    ups, dns = [], []
    for s in svals(0.0, AXIS.L, 30.0):
        a = SEC.toe(AXIS, s, -1)
        b = SEC.toe(AXIS, s, +1)
        ups.append(AXIS.xz(s, float(SEC.d_up(a)) - 2.0))
        dns.append(AXIS.xz(s, float(SEC.d_dn(b)) + 2.0))
    fr = Frame(SP_G, SP_BRG)
    head = [fr.xz(SP_HW + 2.0, -17.0), fr.xz(SP_HW + 2.0, 40.0)]
    ring = [(float(x), float(z)) for x, z in dns] + [(float(x), float(z)) for x, z in ups[::-1]] + \
        [(float(x), float(z)) for x, z in head]
    return [[round(x, 1), round(z, 1)] for x, z in ring]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = the_dam() + intake_tower() + footbridge() + spillway() + valve_house() + outlet_house()
    dam = merge(parts)
    tris, size = write_glb(OUT / "model.glb", [("presa_de_valmayor", dam)], roughness=0.9)
    lo, hi = dam.v.min(0), dam.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    ga = to_model(*GA_EN)
    print(f"ground_at {ga[0]:.1f}, {ga[1]:.1f}: MDT {float(T(*ga)):.2f} (E0 {E0})")
    s0, d0 = s_d(0.0, 0.0)
    print(f"axis {AXIS.L:.1f} m; origin at s {s0:.1f}, {d0:.1f} m downstream of the crest axis; "
          f"ground there {float(gnd(0, 0)):.2f} m; the face of the model there {float(Y_C - (d0 - CW_DN) / SD):.2f} m")
    for s in svals(0, AXIS.L, 60):
        print(f"  s {s:6.1f}: toe up {SEC.toe(AXIS, s, -1):6.1f}, toe dn {SEC.toe(AXIS, s, 1):6.1f}, "
              f"MDT on the axis {float(AXIS.g(s, 0)):6.1f}, at d 20/40/60: "
              f"{float(AXIS.g(s, 20)):6.1f} {float(AXIS.g(s, 40)):6.1f} {float(AXIS.g(s, 60)):6.1f}")
    print("footprint:", json.dumps(footprint()))


if __name__ == "__main__":
    main()
