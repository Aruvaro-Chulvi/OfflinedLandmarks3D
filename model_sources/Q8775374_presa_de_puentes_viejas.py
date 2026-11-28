"""
Presa de Puentes Viejas (the dam of the Embalse de Puentes Viejas, Q8775374), Puentes Viejas — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The work: the curved gravity dam of the Canal de Isabel II on the river Lozoya, concrete of Portland cement between
facings of artificial ashlar; first stage 1914–1925 (43.5 m), raised 1932–1936, finished 1939, spillway gates of
the early 1960s, mini hydro plant of 1995; a 20th-century engineering work, taken as protected.

Measurements (see meta.json for the sources):
- OpenStreetMap way 57902519 (waterway=dam) and the road M-135 on its crest (way 20677334): the dam is a long arch,
  convex upstream (west), of about 181–187 m radius in its middle and straighter at its south end; the crest road
  turns north at the left abutment over the gated spillway.
- Wikipedia (es, de): gravity dam of curved plan, 66 m high over its foundation, crest at 955 m, 324 m long and
  5 m wide at the top; normal water level about 951 m; a bottom outlet and a gated surface spillway on the left
  bank (north); the mini hydro plant at the foot of the dam (1995) and a new intake tower.
- IGN MDT05 (terrain.json): the reservoir at 952 m, the pool below the dam (the tail of El Villar) at 904 m
  (y = 0), the esplanade at the toe at 909–911 m, the gorge and both abutments.
- PNOA orthophoto: the crest road, the gate piers of the spillway, the curved esplanade at the toe with the power
  house and the valve chamber, the round intake tower in the reservoir and its long footbridge.
- Photos on Commons: the downstream face of artificial ashlar on a curved profile, the arcade of tall blind arches
  on corbels under the crest parapet, the spillway (round-nosed piers, red steel gates, the gantry with its hoists),
  the power house of ashlar with its band of windows, the intake tower with its pillared top and its footbridge.

Axes (docs/MODEL_SPEC.md): +Y up, the downstream face (middle of the arch) faces +Z, +X along the dam towards the
left bank (north); origin on the dam at 40.99317, -3.57107 (OSM, the tour stop); y = 0 is the water of the pool
below the dam (904 m, ground_at / ground_elev). The reservoir water is drawn by the game.

    python model_sources/Q8775374_presa_de_puentes_viejas.py   -> models/Q8775374/model.glb
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
from meshkit import Mesh, merge, box, cyl, triangulate, extrude, fbm, noise3, write_glb  # noqa: E402

OUT = ROOT / "models" / "Q8775374"

LAT0, LON0 = 40.99317, -3.57107
E0 = 904.0                      # the pool below the dam = y 0
Y_C = 955.0 - E0                # the crest road (51 m)
Y_SILL = 946.5 - E0             # sill of the spillway gates
Y_GATE = 951.2 - E0             # top of the gates (normal water level about 951 m)
Y_ESP = 910.5 - E0              # the esplanade at the toe

FACE = np.array((0.70, 0.67, 0.61))     # artificial ashlar of the downstream face
UPC = np.array((0.64, 0.63, 0.59))
CREST = np.array((0.78, 0.75, 0.69))
CONC = np.array((0.70, 0.69, 0.65))
HOUSE = np.array((0.66, 0.60, 0.52))    # the ashlar of the power house
ROAD = (0.31, 0.31, 0.32)
DARK = (0.08, 0.08, 0.08)
RED = (0.62, 0.17, 0.12)
STEEL = (0.36, 0.39, 0.41)
ROOF = (0.50, 0.48, 0.44)

# ── The upstream face of the crest: the road (OSM way 20677334) moved 3 m upstream ─────────────────────
ROAD_PTS = [(-33.2, -136.9), (-37.6, -115.8), (-40.3, -95.8), (-41.6, -79.9), (-42.1, -68.1), (-41.5, -55.7),
            (-39.9, -43.9), (-37.3, -30.0), (-34.3, -18.1), (-27.6, 0.6), (-21.0, 16.7), (-10.1, 34.8),
            (0.5, 49.4), (10.4, 60.0), (21.0, 69.1), (30.7, 76.4), (42.2, 82.2), (54.0, 86.0)]
UP_OFF = 3.0


def _resample(pts, step=1.0, smooth=9):
    P = np.asarray(pts, float)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    s = np.arange(0, L[-1] + 1e-9, step)
    Q = np.stack([np.interp(s, L, P[:, 0]), np.interp(s, L, P[:, 1])], 1)
    if smooth > 1:                                   # moving average, ends kept
        k = smooth
        pad = np.vstack([np.repeat(Q[:1], k, 0), Q, np.repeat(Q[-1:], k, 0)])
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        Q2 = np.stack([np.convolve(pad[:, i], ker, "valid") for i in range(2)], 1)
        w = np.clip(np.minimum(s, s[-1] - s) / (k * step), 0, 1)[:, None]
        Q = Q2 * w + Q * (1 - w)
    return Q


_R = _resample(ROAD_PTS)
_T = np.gradient(_R, axis=0)
_T /= np.linalg.norm(_T, axis=1)[:, None]
_N = np.stack([_T[:, 1], -_T[:, 0]], 1)            # downstream (to the right going north)
CURVE = _R - _N * UP_OFF                           # upstream face at the crest
_L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(CURVE, axis=0), axis=1))])


def _nearest(e, n):
    d2 = (CURVE[:, 0] - e) ** 2 + (CURVE[:, 1] - n) ** 2
    i = int(np.argmin(d2))
    return i


_i0 = _nearest(0.0, 0.0)
S0 = float(_L[_i0])
S_S = -S0                                          # south end (local s)
S_N = float(_L[-1]) - S0                           # north end, at the spillway


def arch_world(s, d):
    s = np.asarray(s, float) + S0
    d = np.asarray(d, float)
    e = np.interp(s, _L, CURVE[:, 0]) + d * np.interp(s, _L, _N[:, 0])
    n = np.interp(s, _L, CURVE[:, 1]) + d * np.interp(s, _L, _N[:, 1])
    return e, n


def local_of(e, n):
    i = _nearest(e, n)
    v = np.array((e, n)) - CURVE[i]
    return float(_L[i] - S0 + np.dot(v, _T[i])), float(np.dot(v, _N[i]))


# heading: the downstream normal halfway along the dam
_im = int(np.searchsorted(_L, _L[-1] / 2))
HEADING = round(math.degrees(math.atan2(_N[_im, 0], _N[_im, 1])) % 360, 1)
_h = math.radians(HEADING)


def to_model(e, n):
    e, n = np.asarray(e, float), np.asarray(n, float)
    return (-e * math.cos(_h) + n * math.sin(_h), e * math.sin(_h) + n * math.cos(_h))


def arch_xz(s, d):
    return to_model(*arch_world(s, d))


def place(m):
    x, z = arch_xz(m.v[:, 0], m.v[:, 2])
    m.v = np.stack([x, m.v[:, 1], z], 1)
    return m


def wplace(m):
    x, z = to_model(m.v[:, 0], m.v[:, 2])
    m.v = np.stack([x, m.v[:, 1], z], 1)
    m.f = m.f[:, ::-1]
    return m


# ── Terrain (terrain.json, IGN MDT05) ────────────────────────────────────────────────────────────────
def load_terrain():
    t = json.loads((OUT / "terrain.json").read_text("utf-8"))
    n = t["n"]
    q = np.frombuffer(base64.b64decode(t["h"]), "<i2").reshape(n, n).astype(np.float64)
    h = np.where(q == -32768, np.nan, t["off"] + q / 10)

    def TW(east, north):
        east, north = np.broadcast_arrays(np.asarray(east, float), np.asarray(north, float))
        fi = (east - t["x0"]) / t["c"]
        fj = (t["y0"] - north) / t["c"]
        i0 = np.clip(np.floor(fi).astype(int), 0, n - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
        a = np.clip(fi - i0, 0, 1)
        b = np.clip(fj - j0, 0, 1)
        return (h[j0, i0] * (1 - a) * (1 - b) + h[j0, i0 + 1] * a * (1 - b)
                + h[j0 + 1, i0] * (1 - a) * b + h[j0 + 1, i0 + 1] * a * b)
    return TW


TW = load_terrain()


def ground(s, d):
    return TW(*arch_world(s, d)) - E0


# ── The cross-section ────────────────────────────────────────────────────────────────────────────────
D0 = 5.6
Y_D = Y_C - 4.6              # the face flares below the arcade
UP_B = 0.04


def d_up(y):
    return -UP_B * (Y_C - np.asarray(y, float))


def d_dn(y):
    hh = np.clip(Y_D - np.asarray(y, float), 0, None)
    hc = np.minimum(hh, 42.0)
    return D0 + 0.45 * hh + 0.0088 * hc * hc + 0.0088 * 2 * 42.0 * (hh - hc)   # 33 m of batter at the esplanade


def ybot(s):
    s = np.atleast_1d(np.asarray(s, float))
    ds = np.linspace(-2, 50.0, 26)
    g = np.min([ground(s, np.full_like(s, d)) for d in ds], axis=0)
    return np.minimum(g - 4.0, Y_C - 6.0)


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


def sheet(P, out, col=(0.5, 0.5, 0.5)):
    nu, nv = P.shape[0] - 1, P.shape[1] - 1
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    m = Mesh(P.reshape(-1, 3), np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]), col)
    return clean(orient(m, out))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def strip(s0, s1, y0, y1, d0, d1, c, step=5.0):
    ss = np.linspace(s0, s1, max(1, int(math.ceil((s1 - s0) / step))) + 1)
    return [B(a, b + 0.02, y0, y1, d0, d1, c) for a, b in zip(ss[:-1], ss[1:])]


def poly_x(poly, x, out, col):
    pts, tris = triangulate(poly)
    m = Mesh(np.stack([np.full(len(pts), x), pts[:, 1], pts[:, 0]], 1), tris, col)
    return orient(m, (out, 0, 0))


def ashlar(base, seed, course=0.8):
    base = np.asarray(base, float)

    def fn(v):
        n1 = fbm(v, 0.05, seed, 3) - 0.5
        n2 = noise3(v, 0.5, seed + 5) - 0.5
        st = fbm(v, (0.25, 0.03, 0.25), seed + 9, 2) - 0.5
        row = np.where(np.floor(v[:, 1] / course + 1e-6).astype(int) % 2 == 0, 1.02, 0.97)
        k = (1 + 0.16 * n1 + 0.08 * n2 - 0.22 * np.clip(st, 0, 1)) * row
        return np.clip(np.outer(k, base), 0, 1)
    return fn


def tone(base, amp=0.08, seed=5, freq=0.2):
    base = np.asarray(base, float)
    return lambda v: np.clip(np.outer(1 + amp * (fbm(v, freq, seed, 2) - 0.5) * 2, base), 0, 1)


def svals(a, b, step):
    n = max(1, int(math.ceil(abs(b - a) / step)))
    return np.linspace(a, b, n + 1)


def arch_open(sc, y0, y1, w, d, col=DARK, seg=7, out=1):
    arc = [(sc + w * math.cos(t), y1 + w * math.sin(t)) for t in np.linspace(0, math.pi, seg)]
    V = [(sc - w, y0, d), (sc + w, y0, d)] + [(x, y, d) for x, y in arc]
    F = [(0, 1, 2)] + [(0, 2 + k, 3 + k) for k in range(len(arc) - 1)]
    return orient(Mesh(V, F, col), (0, 0, out))


# ── The dam body ─────────────────────────────────────────────────────────────────────────────────────
def body(s0, s1, step=3.0, dy=2.5):
    ss = svals(s0, s1, step)
    yb = ybot(ss)
    lv = np.arange(math.floor(yb.min()), Y_C, dy)
    lv = np.unique(np.append(lv, [Y_D, Y_C]))
    Y = np.clip(np.maximum(lv[None, :], yb[:, None]), None, Y_C)
    S = np.repeat(ss[:, None], len(lv), 1)
    up = sheet(np.stack([S, Y, d_up(Y)], -1), (0, 0, -1)).paint(tone(UPC, 0.08, 3, 0.05))
    dn = sheet(np.stack([S, Y, d_dn(Y)], -1), (0, 0, 1)).paint(ashlar(FACE, 11))
    top = sheet(np.stack([np.stack([ss, ss], 1), np.full((len(ss), 2), Y_C),
                          np.stack([np.full(len(ss), float(d_up(Y_C))), np.full(len(ss), float(d_dn(Y_C)))], 1)], -1),
                (0, 1, 0)).paint(tone(CREST, 0.05, 4))
    out = [up, dn, top]
    for k, s in ((0, s0), (1, s1)):
        ybk = float(ybot(np.array([s]))[0])
        lv2 = np.unique(np.append(np.arange(ybk, Y_D, 2.0), [Y_D]))
        prof = [(float(d_up(ybk)), ybk)] + [(float(d_dn(y)), float(y)) for y in lv2] + \
               [(float(d_dn(Y_C)), Y_C), (float(d_up(Y_C)), Y_C)]
        out.append(poly_x(prof, s, -1 if k == 0 else 1, CREST * 0.9))
    return out


def crest_furniture(s0, s1):
    """Road, the upstream parapet, the arcade of tall blind arches on corbels carrying the downstream parapet."""
    out = []
    out += strip(s0, s1, Y_C - 0.02, Y_C + 0.04, 0.6, D0 - 0.4, ROAD)
    out += [m.paint(tone(CREST, 0.05, 6)) for m in strip(s0, s1, Y_C, Y_C + 1.0, -0.2, 0.5, CREST)]
    out += strip(s0, s1, Y_C + 1.0, Y_C + 1.15, -0.3, 0.6, CREST * 1.03)
    zf = D0 + 1.0
    out += strip(s0, s1, Y_C - 0.05, Y_C + 0.2, D0 - 0.4, zf - 0.6, CREST * 0.95)            # footway
    out += strip(s0, s1, Y_C - 4.9, Y_C - 4.4, D0 - 0.2, zf + 0.15, CREST * 0.9)               # corbel course
    out += [m.paint(tone(CREST, 0.05, 7)) for m in strip(s0, s1, Y_C - 4.4, Y_C + 1.05, zf - 0.7, zf, CREST)]
    out += strip(s0, s1, Y_C + 1.05, Y_C + 1.25, zf - 0.8, zf + 0.1, CREST * 1.03)
    out += strip(s0, s1, Y_C - 0.6, Y_C - 0.35, zf, zf + 0.12, CREST * 1.02)                   # string course
    n = int((s1 - s0) // 2.3)
    pitch = (s1 - s0) / max(n, 1)
    for i in range(n):
        sc = s0 + (i + 0.5) * pitch
        out.append(B(sc - 0.2, sc + 0.2, Y_C - 5.6, Y_C - 4.9, D0 - 0.2, zf - 0.2, CREST * 0.85))   # corbel
        out.append(arch_open(sc, Y_C - 4.2, Y_C - 1.6, 0.62, zf + 0.02, col=(0.30, 0.29, 0.27)))     # blind arch
    return out


# ── Works in world coordinates (x = east, z = north) ─────────────────────────────────────────────────
def poly_world(pts, y0, y1, col, top_col=None):
    return wplace(extrude([(e, n) for e, n in pts], y0, y1, col, top_c=top_col))


def seg_box(p, q, w, y0, y1, col):
    p, q = np.asarray(p, float), np.asarray(q, float)
    t = q - p
    t /= float(np.linalg.norm(t))
    nrm = np.array([-t[1], t[0]]) * w / 2
    ring = [tuple(p + nrm), tuple(q + nrm), tuple(q - nrm), tuple(p - nrm)]
    return poly_world(ring, y0, y1, col)


# the spillway: its gate line on the upstream side of the road, from south to north (orthophoto)
G0 = np.array((53.6, 88.5))
G1 = np.array((51.6, 117.5))
_gt = (G1 - G0) / np.linalg.norm(G1 - G0)
_gn = np.array((_gt[1], -_gt[0]))                  # downstream (east)
NBAY = 7


def gp(u, v):
    """A point u m along the gate line from G0, v m downstream of it."""
    return tuple(G0 + _gt * u + _gn * v)


def spillway():
    out = []
    Lg = float(np.linalg.norm(G1 - G0))
    pier = 1.6
    bay = (Lg - (NBAY + 1) * pier) / NBAY
    ylo = Y_SILL - 12.0
    # the sill and the ogee falling to the east, between the training walls
    out.append(poly_world([gp(0, -1), gp(Lg, -1), gp(Lg, 6), gp(0, 6)], ylo, Y_SILL, CONC * 0.95, CONC))
    for k in range(5):
        v0, v1 = 6 + 2.5 * k, 6 + 2.5 * (k + 1)
        out.append(poly_world([gp(0, v0), gp(Lg, v0), gp(Lg, v1), gp(0, v1)], ylo, Y_SILL - 1.2 * (k + 1) ** 1.3,
                              CONC * 0.93, CONC * 0.97))
    # the piers (round nose upstream), the red gates, the deck, the road bridge and the gantry
    for i in range(NBAY + 1):
        u = i * (pier + bay) + pier / 2
        out.append(poly_world([gp(u - pier / 2, 0), gp(u + pier / 2, 0), gp(u + pier / 2, 9.5), gp(u - pier / 2, 9.5)],
                              Y_SILL - 1.0, Y_C - 0.9, CONC).paint(tone(CONC, 0.06, 41 + i, 0.2)))
        cx, cz = to_model(*gp(u, 0))
        out.append(cyl(pier / 2, pier / 2, Y_C - 0.9 - Y_SILL + 1.0, CONC, seg=10).T(float(cx), Y_SILL - 1.0, float(cz)))
        if i < NBAY:
            ua, ub = u + pier / 2, u + pier / 2 + bay
            out.append(poly_world([gp(ua, 0.3), gp(ub, 0.3), gp(ub, 0.7), gp(ua, 0.7)], Y_SILL, Y_GATE, RED))
    out.append(poly_world([gp(-1, -0.6), gp(Lg + 1, -0.6), gp(Lg + 1, 2.6), gp(-1, 2.6)], Y_C - 0.9, Y_C, CREST * 0.92))
    out.append(poly_world([gp(-1, 2.6), gp(Lg + 1, 2.6), gp(Lg + 1, 11.5), gp(-1, 11.5)], Y_C - 1.0, Y_C, CREST * 0.9))
    out.append(poly_world([gp(-1, 3.4), gp(Lg + 1, 3.4), gp(Lg + 1, 10.6), gp(-1, 10.6)], Y_C, Y_C + 0.04, ROAD))
    for v in (2.8, 11.2):
        out.append(seg_box(gp(-1, v), gp(Lg + 1, v), 0.4, Y_C, Y_C + 1.0, CREST))
    # the gantry over the gates: a steel walkway with its railings and the hoists
    out.append(poly_world([gp(-0.5, -0.4), gp(Lg + 0.5, -0.4), gp(Lg + 0.5, 2.2), gp(-0.5, 2.2)], Y_C + 1.6, Y_C + 1.8, STEEL))
    for i in range(NBAY + 1):
        u = i * (pier + bay) + pier / 2
        out.append(poly_world([gp(u - 0.15, -0.3), gp(u + 0.15, -0.3), gp(u + 0.15, 2.1), gp(u - 0.15, 2.1)], Y_C, Y_C + 1.6, STEEL))
        if i < NBAY:
            uc = u + pier / 2 + bay / 2
            out.append(poly_world([gp(uc - 1.0, 0.2), gp(uc + 1.0, 0.2), gp(uc + 1.0, 1.6), gp(uc - 1.0, 1.6)],
                                  Y_C + 1.8, Y_C + 2.9, (0.30, 0.42, 0.50)))
    for v in (-0.35, 2.15):
        out.append(seg_box(gp(-0.5, v), gp(Lg + 0.5, v), 0.06, Y_C + 1.8, Y_C + 2.9, STEEL))
    for u in (2.0, Lg / 2, Lg - 2.0):                                       # the lamp masts
        cx, cz = to_model(*gp(u, 2.0))
        out.append(cyl(0.12, 0.09, 7.0, STEEL, seg=6).T(float(cx), Y_C + 1.8, float(cz)))
    # training walls at both ends, down the rock to the east
    for u in (-1.0, Lg + 1.0):
        out.append(seg_box(gp(u, -1.0), gp(u, 20.0), 1.4, Y_SILL - 12.0, Y_C + 0.9, CREST * 0.9))
    return out


def esplanade():
    """The curved esplanade at the toe, the power house of 1995 and the valve chamber (orthophoto)."""
    out = []
    ESP = [(-12.0, 6.0), (14.0, 3.8), (27.5, -1.2), (32.0, -11.0), (33.5, -20.5), (40.0, -31.0), (31.0, -48.0),
           (20.0, -48.5), (2.0, -44.0), (-12.0, -38.0)]
    out.append(poly_world(ESP, -3.0, Y_ESP, CONC * 0.9, CONC * 0.95))
    # the power house: a block of ashlar with a band of windows under a flat roof
    PH = [(17.5, -22.5), (36.9, -30.0), (27.5, -46.9), (12.5, -40.0)]
    c = np.mean(PH, axis=0)
    out.append(poly_world(PH, -3.0, Y_ESP + 11.0, HOUSE).paint(ashlar(HOUSE, 61, 0.9)))
    PH2 = [tuple(c + (np.asarray(p) - c) * 1.04) for p in PH]
    out.append(poly_world(PH2, Y_ESP + 11.0, Y_ESP + 11.6, CREST))
    PH3 = [tuple(c + (np.asarray(p) - c) * 0.97) for p in PH]
    out.append(poly_world(PH3, Y_ESP + 11.6, Y_ESP + 12.0, ROOF))
    for a, b in zip(PH, PH[1:] + PH[:1]):                                   # the window band all round
        a, b = np.asarray(a), np.asarray(b)
        L = float(np.linalg.norm(b - a))
        t = (b - a) / L
        nrm = np.array([t[1], -t[0]])
        if np.dot(nrm, np.asarray(a) - c) < 0:
            nrm = -nrm
        for k in range(int(L // 2.6)):
            f = (k + 0.5) / int(L // 2.6)
            p = a + (b - a) * f + nrm * 0.05
            out.append(seg_box(p - t * 0.8, p + t * 0.8, 0.1, Y_ESP + 7.0, Y_ESP + 10.0, DARK))
    # the valve chamber near the face
    VC = [(11.2, -0.6), (21.2, -5.6), (18.8, -10.6), (8.8, -5.6)]
    out.append(poly_world(VC, -1.0, Y_ESP + 4.0, CREST * 0.92, (0.6, 0.6, 0.58)))
    return out


def intake_tower():
    """The round intake tower in the reservoir and its long footbridge to the south-west shore (orthophoto)."""
    ce, cn = -94.5, -20.0
    x, z = (float(v) for v in to_model(ce, cn))
    out = [cyl(4.6, 4.6, Y_C + 0.5 - 20.0, CONC, seg=20).T(x, 20.0, z),
           cyl(5.2, 5.2, 0.6, CREST, seg=20).T(x, Y_C + 0.5, z),
           cyl(4.6, 4.6, 3.8, (0.30, 0.31, 0.32), seg=20).T(x, Y_C + 1.1, z),
           cyl(6.6, 6.6, 0.7, CREST, seg=24).T(x, Y_C + 4.9, z)]
    for k in range(10):                                                  # the pillars round the top
        a = 2 * math.pi * k / 10
        out.append(box(0.7, 3.8, 0.7, (0.82, 0.81, 0.78)).T(x + 4.9 * math.cos(a), Y_C + 1.1, z + 4.9 * math.sin(a)))
    p, q = (-97.0, -26.0), (-140.0, -113.0)
    out.append(seg_box(p, q, 2.8, Y_C + 0.0, Y_C + 0.8, CONC))
    t = np.subtract(q, p) / math.dist(p, q)
    nrm = np.array([-t[1], t[0]])
    for side in (-1, 1):
        out.append(seg_box(np.add(p, nrm * 1.35 * side), np.add(q, nrm * 1.35 * side), 0.08, Y_C + 0.8, Y_C + 1.9, STEEL))
    L = math.dist(p, q)
    for f in np.arange(10.0, L - 4.0, 11.0) / L:
        c = np.add(p, f * np.subtract(q, p))
        g = float(TW(*c)) - E0
        px, pz = to_model(*c)
        y0 = min(20.0, g - 2.0)
        out.append(box(1.4, Y_C - y0, 2.4, CONC * 0.95).T(float(px), y0, float(pz)).R(0, 0, 0))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    local = body(S_S, S_N) + crest_furniture(S_S, S_N)
    dam = [place(m) for m in local]
    world = spillway() + esplanade() + intake_tower()
    allm = merge(dam + world)
    tris, size = write_glb(OUT / "model.glb", [("presa_de_puentes_viejas", allm)], roughness=0.9)
    lo, hi = allm.v.min(0), allm.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    print(f"heading {HEADING}, s {S_S:.1f}..{S_N:.1f} ({S_N - S_S:.1f} m), origin local {local_of(0.0, 0.0)}")
    for p in ((44.0, -40.0), (48.0, -36.0)):
        print("pool point", p, [round(float(v), 1) for v in to_model(*p)], TW(*p))


if __name__ == "__main__":
    main()
