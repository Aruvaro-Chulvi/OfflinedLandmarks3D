"""
Presa de El Villar (the dam of the Embalse de El Villar, Q5830463), Puentes Viejas / Robledillo de la Jara — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The work: the curved masonry gravity dam of the Canal de Isabel II on the river Lozoya, designed and directed by the
engineers José Morer and Elzeario Boix, built 1869–1873 (works finished 1882); a 19th-century work, public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 57902521 (waterway=dam) and the road M-127 on its crest (way 204487178): the dam is an arch
  in plan, convex upstream; its upstream face is a circle of about 151 m radius (fitted to the upstream edge of the
  OSM outline and the curve of the crest road), with its centre downstream (south); the crest road leaves the dam
  at the right bank (west) on a bridge over the fixed-lip spillway and runs on to the north.
- Wikipedia (es, de): 107 m of crest, 5 m wide at the top, 50.5 m high over its foundation, crest at 906 m, normal
  water level 904 m; rubble masonry with hydraulic lime and ashlar facings; an intake tower apart from the dam body
  and a fixed-lip spillway on the right bank; a mini hydro plant of 1994 in a cavern with a new intake tower.
- IGN MDT05 (terrain.json): the reservoir at 903 m, the pool below the dam (tail of El Atazar) at 860 m (y = 0),
  the downstream face of the dam itself (it is in the elevation model), the gorge and both abutments, the floor of
  the spillway basin at about 904 m.
- PNOA orthophoto: the crest road and its parapets, the half-round tower on the upstream face, the old square
  intake tower at the left abutment with its hut, the round intake tower of 1994 with its footbridge, the outlets at
  the toe, the spillway (sloping apron upstream of the road, the basin west of the road, the stepped cascade down
  the right bank next to the dam).
- Photos on Commons: the downstream face in ashlar, steep at the top and flaring towards the foot (a curved
  profile), the frieze of small arches on corbels under the crest parapet, the half-round tower with slit windows on
  the upstream face, the stepped walls of the spillway on the right bank, the intake towers.
Pale granite ashlar, light stone on the crest, concrete in the spillway.

Axes (docs/MODEL_SPEC.md): +Y up, the downstream face (middle of the arch) faces +Z (190.4°), +X along the dam
towards the left bank (east); origin on the dam at 40.94744, -3.56323 (OSM, the tour stop); y = 0 is the water
of the pool below the dam (860 m, ground_at / ground_elev). The reservoir water is drawn by the game.

    python model_sources/Q5830463_presa_de_el_villar.py   -> models/Q5830463/model.glb
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

OUT = ROOT / "models" / "Q5830463"

LAT0, LON0 = 40.94744, -3.56323
E0 = 860.0                      # the pool below the dam = y 0
Y_C = 906.0 - E0                # the crest road (46 m)
Y_LIP = 904.0 - E0              # the fixed lip of the spillway (normal water level)

FACE = np.array((0.74, 0.70, 0.62))     # ashlar of the downstream face
UPC = np.array((0.66, 0.64, 0.59))      # upstream face
CREST = np.array((0.80, 0.77, 0.70))    # stone of the parapets and cornices
CONC = np.array((0.70, 0.69, 0.65))     # concrete of the spillway
ROAD = (0.31, 0.31, 0.32)
DARK = (0.08, 0.08, 0.08)
WHITE = (0.88, 0.88, 0.86)
STEEL = (0.35, 0.38, 0.40)

# ── The arch (metres east / north of the origin) ─────────────────────────────────────────────────────────
CE, CN = -9.56, -141.06          # centre of the arch (downstream), fitted to OSM
R_UP = 151.0                     # radius of the upstream face at the crest
W_END = (-37.0, 6.5)             # the dam meets the spillway bridge (right bank, west)
E_END = (66.0, -11.6)            # the left abutment (east)


def bearing(e, n):
    return math.atan2(e - CE, n - CN)


TH0 = bearing(0.0, 0.0)
S_W = (bearing(*W_END) - TH0) * R_UP
S_E = (bearing(*E_END) - TH0) * R_UP
HEADING = (math.degrees((bearing(*W_END) + bearing(*E_END)) / 2) + 180.0) % 360
HEADING = round(HEADING, 1)
_h = math.radians(HEADING)


def to_model(e, n):
    """Metres east/north -> model x, z (+Z = heading, +X = heading − 90°)."""
    e, n = np.asarray(e, float), np.asarray(n, float)
    return (-e * math.cos(_h) + n * math.sin(_h), e * math.sin(_h) + n * math.cos(_h))


def arch_world(s, d):
    """Local (s along the upstream face at the crest, d downstream of it) -> metres east, north."""
    th = TH0 + np.asarray(s, float) / R_UP
    r = R_UP - np.asarray(d, float)
    return CE + r * np.sin(th), CN + r * np.cos(th)


def arch_xz(s, d):
    return to_model(*arch_world(s, d))


def local_of(e, n):
    s = (bearing(e, n) - TH0) * R_UP
    d = R_UP - math.hypot(e - CE, n - CN)
    return s, d


def place(m):
    """Bend a mesh built in local (s, y, d) onto the arch, in model axes."""
    x, z = arch_xz(m.v[:, 0], m.v[:, 2])
    m.v = np.stack([x, m.v[:, 1], z], 1)
    return m


def wplace(m):
    """A mesh built in world metres (x = east, z = north) -> model axes."""
    x, z = to_model(m.v[:, 0], m.v[:, 2])
    m.v = np.stack([x, m.v[:, 1], z], 1)
    m.f = m.f[:, ::-1]                       # east/north/up is left-handed in (x, y, z): keep faces outwards
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


TW = load_terrain()                      # world metres -> elevation


def ground(s, d):
    return TW(*arch_world(s, d)) - E0


# ── The cross-section ────────────────────────────────────────────────────────────────────────────────
D0 = 6.2                     # crest: from the upstream face to the top of the downstream face
Y_D = Y_C - 3.0              # where the downstream face starts to flare, under the frieze
UP_B = 0.03                  # upstream face, nearly vertical


def d_up(y):
    return -UP_B * (Y_C - np.asarray(y, float))


def d_dn(y):
    hh = np.clip(Y_D - np.asarray(y, float), 0, None)
    return D0 + 0.35 * hh + 0.0081 * hh * hh          # a curved profile: 29 m of batter at the pool


def ybot(s):
    s = np.atleast_1d(np.asarray(s, float))
    ds = np.linspace(-2, float(d_dn(-4.0)) + 6, 24)
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


def strip(s0, s1, y0, y1, d0, d1, c, step=4.0):
    """A long box along the arch, cut in pieces so that it follows the curve (local coordinates)."""
    ss = np.linspace(s0, s1, max(1, int(math.ceil((s1 - s0) / step))) + 1)
    return [B(a, b + 0.02, y0, y1, d0, d1, c) for a, b in zip(ss[:-1], ss[1:])]


def poly_x(poly, x, out, col):
    pts, tris = triangulate(poly)
    m = Mesh(np.stack([np.full(len(pts), x), pts[:, 1], pts[:, 0]], 1), tris, col)
    return orient(m, (out, 0, 0))


def ashlar(base, seed, course=0.9):
    base = np.asarray(base, float)

    def fn(v):
        n1 = fbm(v, 0.05, seed, 3) - 0.5
        n2 = noise3(v, 0.5, seed + 5) - 0.5
        st = fbm(v, (0.25, 0.03, 0.25), seed + 9, 2) - 0.5          # vertical streaks of the seepage
        row = np.where(np.floor(v[:, 1] / course + 1e-6).astype(int) % 2 == 0, 1.02, 0.97)
        k = (1 + 0.16 * n1 + 0.08 * n2 - 0.24 * np.clip(st, 0, 1)) * row
        return np.clip(np.outer(k, base), 0, 1)
    return fn


def tone(base, amp=0.08, seed=5, freq=0.2):
    base = np.asarray(base, float)
    return lambda v: np.clip(np.outer(1 + amp * (fbm(v, freq, seed, 2) - 0.5) * 2, base), 0, 1)


def svals(a, b, step):
    n = max(1, int(math.ceil(abs(b - a) / step)))
    return np.linspace(a, b, n + 1)


def arch_open(sc, y0, y1, w, d, col=DARK, seg=7, out=1):
    """A round-headed opening (a dark patch) on a face at d, local coordinates."""
    arc = [(sc + w * math.cos(t), y1 + w * math.sin(t)) for t in np.linspace(0, math.pi, seg)]
    V = [(sc - w, y0, d), (sc + w, y0, d)] + [(x, y, d) for x, y in arc]
    F = [(0, 1, 2)] + [(0, 2 + k, 3 + k) for k in range(len(arc) - 1)]
    return orient(Mesh(V, F, col), (0, 0, out))


# ── The dam body ─────────────────────────────────────────────────────────────────────────────────────
def body(s0, s1, step=2.5, dy=2.0):
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
    """Road, the upstream parapet, the downstream parapet over a frieze of small arches on corbels (local)."""
    out = []
    out += strip(s0, s1, Y_C - 0.02, Y_C + 0.04, 0.7, D0 - 0.6, ROAD)
    out += [m.paint(tone(CREST, 0.05, 6)) for m in strip(s0, s1, Y_C, Y_C + 1.0, -0.2, 0.6, CREST)]
    out += strip(s0, s1, Y_C + 1.0, Y_C + 1.15, -0.3, 0.7, CREST * 1.03)
    zf = D0 + 0.7
    out += strip(s0, s1, Y_C - 0.05, Y_C + 0.2, D0 - 0.6, zf, CREST * 0.95)                 # footway
    out += strip(s0, s1, Y_C - 2.4, Y_C - 1.9, D0 - 0.2, zf + 0.15, CREST * 0.92)            # frieze cornice
    out += [m.paint(tone(CREST, 0.05, 7)) for m in strip(s0, s1, Y_C - 1.9, Y_C + 1.05, zf - 0.7, zf, CREST)]
    out += strip(s0, s1, Y_C + 1.05, Y_C + 1.2, zf - 0.8, zf + 0.1, CREST * 1.03)
    n = int((s1 - s0) // 1.9)
    pitch = (s1 - s0) / max(n, 1)
    for i in range(n):
        sc = s0 + (i + 0.5) * pitch
        out.append(B(sc - 0.16, sc + 0.16, Y_C - 3.1, Y_C - 2.4, D0 - 0.2, zf - 0.1, CREST * 0.85))   # corbel
        out.append(arch_open(sc, Y_C - 1.6, Y_C - 0.4, 0.45, zf + 0.02))                            # small arch
        out.append(B(sc - 0.1, sc + 0.1, Y_C + 0.3, Y_C + 0.9, -0.21, -0.19, DARK))                  # upstream joints
    return out


def half_tower():
    """The half-round tower on the upstream face (orthophoto: 11 m across, at e 29.4, n 7.5)."""
    sc, _ = local_of(29.4, 8.0)
    r = 5.5
    yb = float(ybot(np.array([sc]))[0])
    out = []
    a = np.linspace(0, math.pi, 13)

    def ring(r1, y0, y1, col):
        V, F = [], []
        for t in a:
            V += [(sc + r1 * math.cos(t), y0, -r1 * math.sin(t)), (sc + r1 * math.cos(t), y1, -r1 * math.sin(t))]
        for i in range(len(a) - 1):
            p = 2 * i
            F += [(p, p + 2, p + 3), (p, p + 3, p + 1)]
        top = [(sc + r1 * math.cos(t), -r1 * math.sin(t)) for t in a]
        m = orient(Mesh(V, F, col), (0, 0, -1))
        pts, tris = triangulate([(x, z) for x, z in top])
        cap = Mesh(np.stack([pts[:, 0], np.full(len(pts), y1), pts[:, 1]], 1), tris, col)
        return [m, orient(cap, (0, 1, 0))]
    out += [m.paint(ashlar(FACE * 0.97, 21, 1.0)) for m in ring(r, yb, Y_C - 2.4, FACE)]
    out += ring(r + 0.35, Y_C - 2.4, Y_C - 1.9, CREST * 0.92)
    out += [m.paint(tone(CREST, 0.05, 23)) for m in ring(r + 0.05, Y_C - 1.9, Y_C + 1.0, CREST)]
    out += ring(r + 0.2, Y_C + 1.0, Y_C + 1.2, CREST * 1.03)
    for y in np.arange(Y_C - 6.0, 2.0, -4.2):              # slit windows in a column, three across
        for t in (math.pi / 2 - 0.55, math.pi / 2, math.pi / 2 + 0.55):
            x, z = sc + (r + 0.03) * math.cos(t), -(r + 0.03) * math.sin(t)
            m = box(0.35, 1.2, 0.06, DARK).T(0, y, 0)
            m.R(0, math.atan2(math.cos(t), -math.sin(t)), 0)
            out.append(m.T(x - 0, 0, z))
    return out


def outlets():
    """The bottom outlets at the toe (two dark openings, orthophoto e 19, n −27.5)."""
    sc, _ = local_of(19.0, -27.5)
    out = []
    d = float(d_dn(3.0)) + 0.05
    for k in (-1, 1):
        out.append(arch_open(sc + k * 2.2, 0.6, 2.6, 0.8, d))
    out.append(B(sc - 5, sc + 5, -3.0, 0.8, float(d_dn(0.8)) - 1, float(d_dn(0.8)) + 3.0, CREST * 0.85))
    return out


# ── Works in world coordinates (x = east, z = north) ─────────────────────────────────────────────────
def poly_world(pts, y0, y1, col, top_col=None):
    return wplace(extrude([(e, n) for e, n in pts], y0, y1, col, top_c=top_col))


def seg_box(p, q, w, y0, y1, col):
    """A box from p to q (world e, n), w wide."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    t = q - p
    L = float(np.linalg.norm(t))
    t /= L
    nrm = np.array([-t[1], t[0]]) * w / 2
    ring = [tuple(p + nrm), tuple(q + nrm), tuple(q - nrm), tuple(p - nrm)]
    return poly_world(ring, y0, y1, col)


ROAD_W = [(-36.0, 3.3), (-43.3, 4.1), (-49.0, 5.3), (-54.2, 8.6), (-59.8, 14.0), (-62.9, 18.6), (-65.4, 23.0),
          (-66.7, 28.8), (-66.9, 34.2), (-66.0, 39.6), (-63.6, 47.7), (-59.5, 58.6)]       # OSM way 204487178


def spillway():
    """The fixed-lip spillway on the right bank: the lip under the road bridge, the apron upstream of it, the basin
    west of the road, the stepped cascade down the right bank next to the dam."""
    out = []
    pts = ROAD_W
    for p, q in zip(pts[:-1], pts[1:]):
        out.append(seg_box(p, q, 3.0, Y_LIP - 10.0, Y_LIP, CONC * 0.95))           # the lip (a weir wall)
        out.append(seg_box(p, q, 8.6, Y_C - 0.9, Y_C, CREST * 0.93))                # bridge deck
        out.append(seg_box(p, q, 7.0, Y_C, Y_C + 0.04, ROAD))
        p_, q_ = np.asarray(p), np.asarray(q)
        t = (q_ - p_) / np.linalg.norm(q_ - p_)
        nn = np.array([-t[1], t[0]])
        for side in (-1, 1):                                                          # parapets
            out.append(seg_box(p_ + nn * side * 4.0, q_ + nn * side * 4.0, 0.5, Y_C, Y_C + 1.0, CREST))
    # piers of the bridge over the lip, every ~4.5 m
    L = np.cumsum([0] + [math.dist(a, b) for a, b in zip(pts[:-1], pts[1:])])
    for sv in np.arange(6.0, L[-1] - 2, 4.5):
        k = int(np.searchsorted(L, sv) - 1)
        f = (sv - L[k]) / (L[k + 1] - L[k])
        p = np.asarray(pts[k]) + f * (np.asarray(pts[k + 1]) - np.asarray(pts[k]))
        t = np.asarray(pts[k + 1]) - np.asarray(pts[k]); t /= np.linalg.norm(t)
        out.append(seg_box(p - t * 0.5, p + t * 0.5, 8.2, Y_LIP, Y_C - 0.9, CREST * 0.9))
    # the sloping apron between the road and the reservoir
    APRON = [(-58.5, 58.0), (-62.0, 46.0), (-62.6, 34.0), (-61.0, 22.5), (-57.0, 15.5), (-50.0, 10.0),
             (-38.0, 8.6), (-26.0, 10.0), (-35.0, 34.0), (-44.0, 58.0)]
    out.append(poly_world(APRON, Y_LIP - 4.0, Y_LIP - 0.1, CONC * 0.92, CONC))
    # the basin west of the road, cut into the hillside, with its retaining wall
    BASIN = [(-62.5, 58.0), (-70.5, 32.0), (-87.0, 2.0), (-87.5, -6.5), (-51.4, 2.8), (-46.0, 1.0), (-52.0, 6.0),
             (-58.0, 11.0), (-62.0, 17.0), (-68.0, 29.0), (-68.6, 36.0), (-67.0, 45.0)]
    out.append(poly_world(BASIN, Y_LIP - 3.5, Y_LIP - 0.6, CONC * 0.9, CONC * 0.97))
    for p, q in zip([(-62.5, 58.0), (-70.5, 32.0), (-87.0, 2.0)], [(-70.5, 32.0), (-87.0, 2.0), (-87.5, -6.5)]):
        out.append(seg_box(p, q, 1.2, Y_LIP - 2.0, Y_C + 0.5, CREST * 0.88))
    out.append(seg_box((-87.5, -6.5), (-51.4, 2.8), 1.4, Y_LIP - 4.0, Y_LIP + 0.4, CREST * 0.9))   # the sill
    return out


def cascade():
    """The stepped walls of the spillway down the right bank, next to the dam (photos, orthophoto)."""
    out = []
    # the steps: platforms in the gorge side, each a few metres below the one before (orthophoto + MDT05)
    steps = [((-50.0, 1.0), (-33.0, 4.0), 41.0), ((-46.0, -6.0), (-31.0, -3.0), 36.0),
             ((-44.0, -14.0), (-29.0, -11.0), 30.5), ((-42.0, -22.0), (-27.0, -19.5), 25.0),
             ((-40.0, -30.0), (-25.5, -28.0), 19.0)]
    for (a, b, ytop) in steps:
        (e0, n0), (e1, n1) = a, b
        ring = [(e0, n0), (e1, n1), (e1 + 1.0, n1 - 8.2), (e0 + 1.0, n0 - 8.2)]
        g = float(np.nanmin([TW(e, n) for e, n in ring])) - E0
        out.append(poly_world(ring, min(g - 3.0, ytop - 4.0), ytop, CREST * 0.86, CONC * 0.9))
        out.append(seg_box((e0, n0), (e1, n1), 0.8, ytop, ytop + 1.0, CREST * 0.9))           # its wall
    # the long side wall along the gorge
    for p, q in (((-32.5, 2.0), (-29.0, -25.0)), ((-29.0, -25.0), (-25.0, -50.0))):
        g = float(np.nanmin([TW(*p), TW(*q)])) - E0
        out.append(seg_box(p, q, 1.2, g - 3.0, float(TW(*p) + TW(*q)) / 2 - E0 + 2.5, CREST * 0.85))
    return out


def square_tower():
    """The old intake tower at the left abutment, with the white hut on it (orthophoto e 76.8, n 6.3)."""
    c = np.array((76.8, 6.3))
    hw = 3.7
    sq = [tuple(c + v) for v in ((-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw))]
    out = [poly_world(sq, 15.0, Y_C + 0.5, FACE * 0.97, CREST)]
    out[-1].paint(ashlar(FACE * 0.97, 31, 1.0))
    sq2 = [tuple(c + np.asarray(v) * 1.06) for v in ((-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw))]
    out.append(poly_world(sq2, Y_C + 0.5, Y_C + 0.9, CREST * 1.03))
    h2 = 2.8
    hut = [tuple(c + v) for v in ((-h2, -h2), (h2, -h2), (h2, h2), (-h2, h2))]
    out.append(poly_world(hut, Y_C + 0.9, Y_C + 3.6, WHITE))
    roof = [tuple(c + v) for v in ((-3.3, -3.3), (3.3, -3.3), (3.3, 3.3), (-3.3, 3.3))]
    out.append(poly_world(roof, Y_C + 3.6, Y_C + 3.9, WHITE))
    out.append(seg_box((72.0, 3.0), (72.6, 9.5), 0.06, Y_C + 1.4, Y_C + 2.8, DARK))           # window
    return out


def round_tower():
    """The intake tower of the 1994 mini hydro plant, in the reservoir, and its footbridge (OSM way 1111008470)."""
    ce, cn = 34.7, 79.2
    x, z = to_model(ce, cn)
    x, z = float(x), float(z)
    out = [cyl(5.4, 5.4, Y_C + 0.5 - 15.0, CONC, seg=20).T(x, 15.0, z),
           cyl(6.6, 6.6, 0.6, CREST, seg=24).T(x, Y_C + 0.5, z),
           cyl(6.3, 6.3, 3.4, (0.82, 0.81, 0.78), seg=24).T(x, Y_C + 1.1, z),
           cyl(6.7, 6.7, 0.5, CREST, seg=24).T(x, Y_C + 4.5, z),
           cyl(6.4, 1.5, 1.6, (0.55, 0.57, 0.58), seg=24).T(x, Y_C + 5.0, z)]
    for k in range(12):                                                  # the band of windows
        a = 2 * math.pi * k / 12
        m = box(1.1, 1.6, 0.08, DARK).T(0, Y_C + 2.0, 0).R(0, -a + math.pi / 2, 0)
        out.append(m.T(x + 6.33 * math.cos(a), 0, z + 6.33 * math.sin(a)))
    # the footbridge to the north-west shore (OSM way 1111008471), on three piers
    p, q = (-10.8, 113.4), (29.6, 83.0)
    out.append(seg_box(p, q, 2.4, Y_C + 0.2, Y_C + 0.9, CONC))
    t = np.subtract(q, p) / math.dist(p, q)
    nrm = np.array([-t[1], t[0]])
    for side in (-1, 1):
        out.append(seg_box(np.add(p, nrm * 1.15 * side), np.add(q, nrm * 1.15 * side), 0.08, Y_C + 0.9, Y_C + 2.0, STEEL))
    for f in (0.25, 0.5, 0.75):
        c = np.add(p, f * np.subtract(q, p))
        px, pz = to_model(*c)
        out.append(cyl(0.9, 0.9, Y_C + 0.2 - 20.0, CONC * 0.95, seg=10).T(float(px), 20.0, float(pz)))
    return out


# ── Assembly ─────────────────────────────────────────────────────────────────────────────────────────
def footprint():
    pts_up, pts_dn = [], []
    for s in np.linspace(S_W, S_E, 22):
        dd = np.linspace(0, 50, 60)
        g = ground(np.full(60, s), dd)
        face = Y_D - np.interp(dd - D0, 0.35 * np.linspace(0, 60, 200) + 0.0081 * np.linspace(0, 60, 200) ** 2,
                               np.linspace(0, 60, 200))
        below = np.where((face < g) & (dd > D0))[0]
        dt = float(dd[below[0]]) if len(below) else 45.0
        pts_up.append(arch_xz(s, -7.0))
        pts_dn.append(arch_xz(s, dt + 2.0))
    ring = [(float(x), float(z)) for x, z in pts_up] + [(float(x), float(z)) for x, z in pts_dn[::-1]]
    return [[round(x, 1), round(z, 1)] for x, z in ring]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    local = body(S_W, S_E) + crest_furniture(S_W, S_E) + half_tower() + outlets()
    dam = [place(m) for m in local]
    world = spillway() + cascade() + square_tower() + round_tower()
    allm = merge(dam + world)
    tris, size = write_glb(OUT / "model.glb", [("presa_de_el_villar", allm)], roughness=0.9)
    lo, hi = allm.v.min(0), allm.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    print(f"heading {HEADING}, arc s {S_W:.1f}..{S_E:.1f} ({S_E - S_W:.1f} m), origin local {local_of(0.0, 0.0)}")
    print("pool point (ground_at):", [round(float(v), 1) for v in to_model(12.0, -50.0)], TW(12.0, -50.0))
    print("footprint:", json.dumps(footprint()))


if __name__ == "__main__":
    main()
