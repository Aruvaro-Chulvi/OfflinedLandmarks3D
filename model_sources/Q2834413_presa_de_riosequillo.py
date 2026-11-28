"""
Presa de Riosequillo (the dam of the Embalse de Riosequillo, Q2834413), Buitrago del Lozoya — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The work: the gravity dam of the Canal de Isabel II on the river Lozoya, project approved in 1947, in service
in 1958; a 20th-century engineering work, taken as protected.

Measurements (see meta.json for the sources):
- OpenStreetMap way 57907231 (waterway=dam): the dam in two straight wings meeting at an angle — the main wing,
  about 590 m, runs at 128.2° from the angle down to the left bank, across the gorge of the Lozoya; the west
  wing, about 345 m, runs at 255.6° from the angle over the low ground of the right bank (1060 m of crest in
  all, Wikipedia). The upstream edge of the outline is the upstream face; the crest road is way 199011585; the
  river passes through the dam at way 57907227 (the bottom outlet).
- Wikipedia (es, de): gravity dam of concrete with ashlar facing, 56 m high over its foundation, crest at
  1010 m, 62 m wide at the base and 5 m at the top (the crest road is carried wider on corbels), 203,000 m³; a
  bottom outlet and a gated spillway (386 m³/s).
- IGN MDT05 (terrain.json): the river below the dam at 961 m (y = 0), the crest at 1010–1011 m, the ground along
  both wings and in the gorge.
- PNOA orthophoto: the two spillways on the main wing, each with two gate towers on the upstream face and a
  chute down the downstream face, the valve house between them, the abutment building at the left end of the
  high part, the crest road between its parapets.
- Photos on Commons: the arcaded parapet on corbels along the downstream edge of the crest, the gate towers
  with their cornices standing out of the reservoir, the two chutes ending in flip buckets that throw the water
  into the plunge pool of the river, the ashlar of the downstream face, the abutment building with arched
  windows.
Ashlar granite on the downstream face, concrete upstream and in the chutes, light stone on the crest.

Axes (docs/MODEL_SPEC.md): +Y up, the downstream face of the main wing faces +Z (38.2°), +X along the main wing
towards the angle (NW); origin on the dam at 40.98540, -3.64937 (OSM, the tour stop); y = 0 is the river below
the dam (961 m, ground_at / ground_elev). The reservoir water is drawn by the game.

    python model_sources/Q2834413_presa_de_riosequillo.py   -> models/Q2834413/model.glb
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

OUT = ROOT / "models" / "Q2834413"

LAT0, LON0 = 40.98540, -3.64937
E0 = 961.0                       # the river below the dam = y 0
Y_C = 1010.0 - E0                # the crest road (49 m)
HEADING = 38.2

FACE = np.array((0.63, 0.53, 0.41))     # ashlar of the downstream face
UPC = np.array((0.60, 0.59, 0.55))      # concrete of the upstream face
CREST = np.array((0.76, 0.72, 0.63))    # stone of the parapets and towers
CONC = np.array((0.70, 0.69, 0.65))     # concrete of the chutes
ROAD = (0.31, 0.31, 0.32)
STEEL = (0.33, 0.37, 0.39)
DARK = (0.08, 0.08, 0.08)
ROOF = (0.47, 0.30, 0.22)


# ── Geometry of the site ─────────────────────────────────────────────────────────────────────────────
def world(lat, lon):
    return ((lon - LON0) * 111320 * math.cos(math.radians(LAT0)), (lat - LAT0) * 110950)


_h = math.radians(HEADING)


def to_model(e, n):
    """Metres east/north -> model x, z (+Z = heading, +X = heading − 90°)."""
    return (-e * math.cos(_h) + n * math.sin(_h), e * math.sin(_h) + n * math.cos(_h))


# OSM way 57907231, the upstream edge (metres east/north of the origin)
SE_END = (387.0, -311.5)
SE_A = (348.6, -292.4)
SE_B = (-112.3, 70.2)
W_A = (-132.4, 72.1)
W_END = (-468.1, -13.8)


def _line_x(p1, p2, q1, q2):
    p1, p2, q1, q2 = map(np.array, (p1, p2, q1, q2))
    d1, d2 = p2 - p1, q2 - q1
    def cr(a, b):
        return a[0] * b[1] - a[1] * b[0]
    t = cr(q1 - p1, d2) / cr(d1, d2)
    return p1 + t * d1


KINK_W = _line_x(SE_A, SE_B, W_A, W_END)            # where the two wings meet (world)
_dse = np.array(SE_A) - KINK_W
SE_END = tuple(KINK_W + _dse * float(np.dot(np.array(SE_END) - KINK_W, _dse) / np.dot(_dse, _dse)))  # on the line
KX, KZ = to_model(*KINK_W)


def _unit(a, b):
    v = np.array(to_model(*b)) - np.array(to_model(*a)); return v / np.linalg.norm(v)


U_SE = _unit(SE_END, KINK_W)                        # main wing: s grows towards the angle (≈ +X)
U_W = _unit(KINK_W, W_END)                          # west wing: s grows away from the angle
L_SE = float(np.linalg.norm(np.array(to_model(*SE_END)) - np.array([KX, KZ])))
L_W = float(np.linalg.norm(np.array(to_model(*W_END)) - np.array([KX, KZ])))
DEFL = math.acos(float(np.clip(np.dot(U_SE, U_W), -1, 1)))     # the deflection at the angle
TAN_HALF = math.tan(DEFL / 2)


class Wing:
    def __init__(self, u, sign):
        self.u = np.asarray(u, float)
        self.n = np.array([-self.u[1], self.u[0]])      # downstream (+Z for the main wing)
        self.sign = sign                                # +1: s ≤ 0 runs to the angle at 0 (main); −1: s ≥ 0 (west)

    def xz(self, s, d):
        s, d = np.asarray(s, float), np.asarray(d, float)
        # mitre at the angle: on the outer (downstream) side the wing runs on past the angle, inside it stops short
        w = np.clip(1 - np.abs(s) / 30.0, 0, 1)
        s2 = s + self.sign * d * TAN_HALF * w
        return KX + s2 * self.u[0] + d * self.n[0], KZ + s2 * self.u[1] + d * self.n[1]

    def place(self, m):
        x, z = self.xz(m.v[:, 0], m.v[:, 2])
        m.v = np.stack([x, m.v[:, 1], z], 1)
        return m


SE = Wing(U_SE, +1)
WW = Wing(U_W, -1)
assert SE.n[1] > 0.99, SE.n                         # the main wing's downstream normal is +Z


def s_on_se(e, n):
    x, z = to_model(e, n)
    return float((x - KX) * U_SE[0] + (z - KZ) * U_SE[1]), float((x - KX) * SE.n[0] + (z - KZ) * SE.n[1])


# ── Terrain (terrain.json, IGN MDT05) ────────────────────────────────────────────────────────────────
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


# ── The cross-section (local: s along the wing, d downstream from the upstream edge of the crest) ────
D0 = 10.2                    # the crest: 10.2 m from the upstream face to the top of the downstream face
Y_D = Y_C - 3.0              # where the downstream slope starts, under the corbels
SLOPE = 0.80                 # downstream face, 0.8 : 1
UP_B = 0.05                  # upstream face, 0.05 : 1


def d_up(y):
    return -UP_B * (Y_C - np.asarray(y, float))


def d_dn(y):
    y = np.asarray(y, float)
    return D0 + SLOPE * np.clip(Y_D - y, 0, None)


def ground(wing, s, d):
    x, z = wing.xz(s, d)
    return T(x, z) - E0


def ybot(wing, s):
    """Foundation at s: 4 m below the lowest ground under the section (the face then goes into the ground)."""
    s = np.atleast_1d(np.asarray(s, float))
    ds = np.linspace(-2, D0 + SLOPE * Y_D + 8, 24)
    g = np.min([ground(wing, s, np.full_like(s, d)) for d in ds], axis=0)
    return np.minimum(g - 4.0, Y_C - 4.0)


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
    area = sum(pts[k][0] * pts[(k + 1) % n][1] - pts[(k + 1) % n][0] * pts[k][1] for k in range(n))
    for i in range(n):
        (za, ya), (zb, yb) = pts[i], pts[(i + 1) % n]
        q = Mesh([(x0, ya, za), (x0, yb, zb), (x1, yb, zb), (x1, ya, za)], [(0, 1, 2), (0, 2, 3)], col)
        nz, ny = yb - ya, -(zb - za)
        if area < 0:
            nz, ny = -nz, -ny
        ms.append(orient(q, (0, ny, nz)))
    return merge(ms)


def ashlar(base, seed, course=1.2):
    base = np.asarray(base, float)

    def fn(v):
        n1 = fbm(v, 0.05, seed, 3) - 0.5
        n2 = noise3(v, 0.5, seed + 5) - 0.5
        st = fbm(v, (0.25, 0.03, 0.25), seed + 9, 2) - 0.5
        row = np.where(np.floor(v[:, 1] / course + 1e-6).astype(int) % 2 == 0, 1.02, 0.97)
        k = (1 + 0.18 * n1 + 0.08 * n2 - 0.20 * np.clip(st, 0, 1)) * row
        return np.clip(np.outer(k, base), 0, 1)
    return fn


def tone(base, amp=0.08, seed=5, freq=0.2):
    base = np.asarray(base, float)
    return lambda v: np.clip(np.outer(1 + amp * (fbm(v, freq, seed, 2) - 0.5) * 2, base), 0, 1)


def svals(a, b, step):
    n = max(1, int(math.ceil(abs(b - a) / step)))
    return np.linspace(a, b, n + 1)


# ── The dam body ─────────────────────────────────────────────────────────────────────────────────────
def body(wing, s0, s1, ytop, step=6.0, dy=2.0, caps=(False, False)):
    """Upstream face, downstream face and top of the dam between s0 and s1, up to ytop (local coordinates)."""
    ss = svals(s0, s1, step)
    yb = ybot(wing, ss)
    lv = np.arange(math.floor(yb.min()), ytop + dy * 0.5, dy)
    lv = np.append(lv[lv < ytop], ytop)
    Y = np.clip(np.maximum(lv[None, :], yb[:, None]), None, ytop)
    S = np.repeat(ss[:, None], len(lv), 1)
    up = sheet(np.stack([S, Y, d_up(Y)], -1), (0, 0, -1)).paint(tone(UPC, 0.08, 3, 0.05))
    # the downstream face: rows level, with the change of slope at Y_D kept as a row
    lv2 = np.unique(np.append(lv, Y_D) if Y_D < ytop else lv)
    Y2 = np.clip(np.maximum(lv2[None, :], yb[:, None]), None, ytop)
    S2 = np.repeat(ss[:, None], len(lv2), 1)
    dn = sheet(np.stack([S2, Y2, d_dn(Y2)], -1), (0, 0, 1)).paint(ashlar(FACE, 11))
    top = sheet(np.stack([np.stack([ss, ss], 1), np.full((len(ss), 2), ytop),
                          np.stack([np.full(len(ss), float(d_up(ytop))), np.full(len(ss), float(d_dn(ytop)))], 1)], -1),
                (0, 1, 0)).paint(tone(CREST, 0.05, 4))
    out = [up, dn, top]
    for k, s in ((0, s0), (1, s1)):
        if caps[k]:
            ybk = float(ybot(wing, np.array([s]))[0])
            prof = [(float(d_up(ybk)), ybk), (float(d_dn(ybk)), ybk)]
            if Y_D < ytop:
                prof.append((D0, Y_D))
            prof += [(float(d_dn(ytop)), ytop), (float(d_up(ytop)), ytop)]
            out.append(poly_x(prof, s, -1 if k == 0 else 1, CREST * 0.92))
    return out


def crest_furniture(s0, s1):
    """Road, upstream parapet, the arcaded parapet on corbels over the downstream face (local, s0 < s1)."""
    L = s1 - s0
    out = [B(s0, s1, Y_C - 0.02, Y_C + 0.03, 1.2, D0 - 1.0, ROAD),                          # the road
           B(s0, s1, Y_C, Y_C + 0.18, D0 - 1.0, D0 + 0.4, CREST * 0.95),                     # footway
           B(s0, s1, Y_C, Y_C + 1.05, -0.15, 0.55, CREST).paint(tone(CREST, 0.05, 6)),      # upstream parapet
           B(s0, s1, Y_C + 1.05, Y_C + 1.2, -0.25, 0.65, CREST * 1.03)]
    # downstream: corbel table, the arcade and its coping, projecting over the face
    zf = D0 + 0.9
    out.append(B(s0, s1, Y_C - 2.6, Y_C - 2.2, D0 - 0.2, zf - 0.2, CREST * 0.9))
    out.append(B(s0, s1, Y_C - 2.2, Y_C + 1.0, zf - 0.8, zf, CREST).paint(tone(CREST, 0.05, 7)))
    out.append(B(s0, s1, Y_C + 1.0, Y_C + 1.2, zf - 0.9, zf + 0.1, CREST * 1.03))
    out.append(B(s0, s1, Y_C - 0.3, Y_C, D0 - 0.3, zf - 0.8, CREST * 0.93))                  # closes the gap
    n = int(L // 2.6)
    pitch = L / max(n, 1)
    for i in range(n):
        sc = s0 + (i + 0.5) * pitch
        out.append(B(sc - 0.18, sc + 0.18, Y_C - 2.95, Y_C - 2.6, D0 - 0.2, zf - 0.35, CREST * 0.85))  # corbel
        # a round-arched opening on the outer face
        w, y0, y1 = 0.55, Y_C - 1.7, Y_C - 0.15
        arc = [(sc + w * math.cos(t), y1 + w * math.sin(t)) for t in np.linspace(0, math.pi, 7)]
        V = [(sc - w, y0, zf + 0.02), (sc + w, y0, zf + 0.02)] + [(x, y, zf + 0.02) for x, y in arc]
        F = [(0, 1, 2)] + [(0, 2 + k, 3 + k) for k in range(len(arc) - 1)]
        out.append(orient(Mesh(V, F, DARK), (0, 0, 1)))
    return out


# ── The spillways (main wing) ────────────────────────────────────────────────────────────────────────
# gate towers on the upstream face (PNOA orthophoto, metres east/north of the origin)
TOWERS = [((31.25, -43.75), (41.25, -51.25)), ((65.0, -70.0), (74.25, -77.0))]
SPILL = []
for a, b in TOWERS:
    sa, _ = s_on_se(*a)
    sb, _ = s_on_se(*b)
    SPILL.append(((sa + sb) / 2, abs(sa - sb)))
TW = 4.6                       # tower width along the dam
Y_SILL = Y_C - 6.5             # crest of the spillway (1003.5 m), under the gates
LIPS = (16.0, 11.0)            # the flip bucket lips (y) of the two chutes


def tower(sc, yb):
    out = [B(sc - TW / 2, sc + TW / 2, yb, Y_C + 5.2, -5.2, 0.6, CREST).paint(tone(CREST, 0.05, 31, 0.3))]
    out.append(B(sc - TW / 2 - 0.25, sc + TW / 2 + 0.25, Y_C + 5.2, Y_C + 5.6, -5.45, 0.85, CREST * 1.03))
    out.append(B(sc - TW / 2 + 0.2, sc + TW / 2 - 0.2, Y_C + 5.6, Y_C + 6.4, -5.0, 0.4, CREST * 0.97))
    out.append(B(sc - TW / 2 - 0.1, sc + TW / 2 + 0.1, Y_C + 6.4, Y_C + 6.6, -5.1, 0.5, CREST * 1.03))
    for z in (-5.21, 0.61):
        for k in (-1, 1):
            out.append(B(sc + k * 1.0 - 0.35, sc + k * 1.0 + 0.35, Y_C + 2.6, Y_C + 4.4, z - 0.02, z + 0.02, DARK))
    for x in (sc - TW / 2 - 0.01, sc + TW / 2 + 0.01):
        out.append(B(x - 0.02, x + 0.02, Y_C + 2.6, Y_C + 4.4, -3.2, -1.6, DARK))
    # the band of stone at the water line and the pier under the road on the crest
    out.append(B(sc - TW / 2, sc + TW / 2, yb, Y_C, 0.6, D0, CREST * 0.95))
    return out


def chute_floor(lip):
    """(d, y) points of the floor of a chute: the ogee crest, the slope, the flip bucket and its lip."""
    pts = [(0.0, Y_SILL), (2.5, Y_SILL + 0.3), (5.0, Y_SILL), (D0 + SLOPE * (Y_D - Y_SILL) * 0 + 7.0, Y_SILL - 2.5)]
    y_start = Y_SILL - 4.0
    for y in np.arange(y_start, lip + 3.0, -4.0):
        pts.append((float(d_dn(y)) + 0.4, float(y)))
    # the bucket: a circular arc from the slope (51° down) to a lip turned up 25°
    r = 7.0
    a0, a1 = math.radians(-51.3), math.radians(25)
    dc, yc = pts[-1]
    tx, ty = math.cos(a0), math.sin(a0)
    cx, cy = dc - r * math.sin(a0), yc + r * math.cos(a0)          # centre above the floor
    for a in np.linspace(a0, a1, 7)[1:]:
        pts.append((cx + r * math.sin(a), cy - r * math.cos(a)))
    return pts


def spillway(sc, gap, lip):
    hw = gap / 2 - TW / 2               # half the clear opening between the towers
    out = []
    ytb = float(ybot(SE, np.array([sc]))[0])
    out += tower(sc - gap / 2, ytb) + tower(sc + gap / 2, ytb)
    # the gate (a steel leaf between the towers) and the road bridge over the opening
    out.append(B(sc - hw, sc + hw, Y_SILL, Y_SILL + 5.4, -0.2, 0.25, STEEL))
    for k in range(4):
        out.append(B(sc - hw, sc + hw, Y_SILL + 0.6 + 1.3 * k, Y_SILL + 0.8 + 1.3 * k, 0.25, 0.45, (0.27, 0.30, 0.32)))
    out.append(B(sc - hw, sc + hw, Y_C - 1.1, Y_C, 0.6, D0, CREST * 0.9))
    # the floor of the chute and its walls
    fl = chute_floor(lip)
    floor_poly = list(fl)
    ybk = float(ybot(SE, np.array([sc]))[0])
    floor_poly += [(fl[-1][0], ybk), (0.0, ybk)]
    out.append(prism_zy([(d, y) for d, y in floor_poly], sc - hw, sc + hw, CONC).paint(tone(CONC, 0.06, 41, 0.1)))
    for side in (-1, 1):
        x0 = sc + side * hw
        x1 = x0 + side * 1.0
        top = [(d, y + 2.6) for d, y in fl]
        wall = [(fl[3][0], fl[3][1])] + fl[4:] + [(fl[-1][0], fl[-1][1] + 2.6)] + top[4:-1][::-1] + [(fl[3][0], Y_C)]
        out.append(prism_zy(wall, min(x0, x1), max(x0, x1), CONC * 0.96).paint(tone(CONC, 0.06, 43, 0.1)))
    # the deck at the lip with its railing
    d_end = fl[-1][0]
    out.append(B(sc - hw - 1.0, sc + hw + 1.0, fl[-1][1] + 2.6, fl[-1][1] + 2.9, d_end - 1.0, d_end + 0.5, CONC))
    return out


# ── Buildings on the main wing ───────────────────────────────────────────────────────────────────────
def valve_house():
    """Between the two spillways, low on the downstream face (the brown roof of the orthophoto)."""
    sc, dc = s_on_se(70.0, -35.0)
    ytop = 27.0
    yb = float(ground(SE, np.array([sc]), np.array([dc + 6]))[0]) - 3
    out = [B(sc - 8, sc + 8, yb, ytop, dc - 6, dc + 6, CREST * 0.95).paint(tone(CREST, 0.05, 51, 0.2)),
           B(sc - 8.3, sc + 8.3, ytop, ytop + 0.4, dc - 6.3, dc + 6.3, CREST),
           B(sc - 7.7, sc + 7.7, ytop + 0.4, ytop + 1.2, dc - 5.7, dc + 5.7, ROOF)]
    for k in range(5):
        x = sc - 6 + 3 * k
        out.append(B(x - 0.6, x + 0.6, ytop - 4.5, ytop - 1.5, dc + 6.0, dc + 6.04, DARK))
    return out


def abutment_house():
    """The long building against the downstream face at the left end of the high part (arched windows)."""
    sc, dc = s_on_se(117.5, -83.75)
    ytop = 1001.5 - E0
    yb = float(ground(SE, np.array([sc]), np.array([dc + 5]))[0]) - 3
    out = [B(sc - 17.5, sc + 17.5, yb, ytop, dc - 5, dc + 5, CREST * 0.97).paint(tone(CREST, 0.05, 53, 0.2)),
           B(sc - 17.8, sc + 17.8, ytop, ytop + 0.5, dc - 5.3, dc + 5.3, CREST * 1.03),
           B(sc - 17.2, sc + 17.2, ytop + 0.5, ytop + 1.3, dc - 4.7, dc + 4.7, (0.55, 0.55, 0.53))]
    for row, (y0, h) in enumerate(((ytop - 7.0, 2.4), (ytop - 3.6, 2.0))):
        for k in range(11):
            x = sc - 15 + 3 * k
            w = 0.6
            arc = [(x + w * math.cos(t), y0 + h + w * math.sin(t)) for t in np.linspace(0, math.pi, 6)]
            V = [(x - w, y0, dc + 5.02), (x + w, y0, dc + 5.02)] + [(a, b, dc + 5.02) for a, b in arc]
            F = [(0, 1, 2)] + [(0, 2 + j, 3 + j) for j in range(len(arc) - 1)]
            out.append(orient(Mesh(V, F, DARK), (0, 0, 1)))
    return out


def outlets():
    """The bottom outlets at the foot of the dam where the river passes through (OSM way 57907227)."""
    sc, _ = s_on_se(91.9, -31.5)
    out = []
    d = float(d_dn(0.0)) + 1.5
    out.append(B(sc - 7.5, sc + 7.5, -3.0, 8.0, float(d_dn(8.0)) - 0.5, d, CREST * 0.92).paint(tone(CREST, 0.05, 57, 0.2)))
    out.append(B(sc - 7.8, sc + 7.8, 8.0, 8.5, float(d_dn(8.5)) - 0.5, d + 0.3, CREST))
    for k in (-1, 1):
        x = sc + k * 3.5
        w, y0, y1 = 1.4, 1.0, 4.0
        arc = [(x + w * math.cos(t), y1 + w * math.sin(t)) for t in np.linspace(0, math.pi, 8)]
        V = [(x - w, y0, d + 0.03), (x + w, y0, d + 0.03)] + [(a, b, d + 0.03) for a, b in arc]
        F = [(0, 1, 2)] + [(0, 2 + j, 3 + j) for j in range(len(arc) - 1)]
        out.append(orient(Mesh(V, F, DARK), (0, 0, 1)))
    # the concrete apron of the outlets at the foot
    out.append(B(sc - 9, sc + 9, -3.0, 1.0, float(d_dn(1.0)) - 2, float(d_dn(1.0)) + 9, CONC * 0.9))
    return out


# ── Assembly ─────────────────────────────────────────────────────────────────────────────────────────
def main_wing():
    out = []
    s_end = -L_SE
    cuts = sorted((sc - gap / 2 + TW / 2, sc + gap / 2 - TW / 2) for sc, gap in SPILL)
    a = s_end
    for c0, c1 in cuts:
        out += body(SE, a, c0, Y_C, caps=(a == s_end, True))
        out += body(SE, c0, c1, Y_SILL, step=3.0, caps=(True, True))
        a = c1
    out += body(SE, a, 0.0, Y_C, caps=(False, False))
    out += crest_furniture(s_end, 0.0)
    for (sc, gap), lip in zip(SPILL, LIPS):
        out += spillway(sc, gap, lip)
    out += valve_house() + abutment_house() + outlets()
    return [SE.place(m) for m in out]


def west_wing():
    out = body(WW, 0.0, L_W, Y_C, caps=(False, True)) + crest_furniture(0.0, L_W)
    return [WW.place(m) for m in out]


def footprint():
    """Seen from above: the dam from its upstream face to the toe of its downstream face, 2 m round."""
    pts_up, pts_dn = [], []
    for wing, ss in ((SE, np.linspace(-L_SE, 0, 25)), (WW, np.linspace(0, L_W, 15)[1:])):
        for s in ss:
            g = ground(wing, np.full(40, s), np.linspace(0, 60, 40))
            dd = np.linspace(0, 60, 40)
            face = Y_D - (dd - D0) / SLOPE
            below = np.where((face < g) & (dd > D0))[0]
            dt = float(dd[below[0]]) if len(below) else 60.0
            pts_up.append(wing.xz(s, -6.0))
            pts_dn.append(wing.xz(s, dt + 2.0))
    ring = [(float(x), float(z)) for x, z in pts_up] + [(float(x), float(z)) for x, z in pts_dn[::-1]]
    return [[round(x, 1), round(z, 1)] for x, z in ring]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dam = merge(main_wing() + west_wing())
    tris, size = write_glb(OUT / "model.glb", [("presa_de_riosequillo", dam)], roughness=0.9)
    lo, hi = dam.v.min(0), dam.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    print(f"angle at x {KX:.1f}, z {KZ:.1f}; main wing {L_SE:.1f} m, west wing {L_W:.1f} m, deflection {math.degrees(DEFL):.1f}°")
    print("spillways (s, gap):", [(round(a, 1), round(b, 1)) for a, b in SPILL])
    print("origin from the upstream face:", s_on_se(0.0, 0.0))
    (Path(__file__).resolve().parent.parent / "models" / "Q2834413").mkdir(exist_ok=True)
    print("footprint:", json.dumps(footprint()))


if __name__ == "__main__":
    main()
