"""
Stonehenge (Q39671), Wiltshire — the stone circle as it stands today, stone by stone, with its earthwork, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
A prehistoric monument (the earthwork c. 3000 BC, the sarsens c. 2500 BC, the bluestones rearranged until
c. 1600 BC): public domain.

What is modelled (the monument as it stands today, not a reconstruction):
- The sarsen circle (30 positions on a circle 33 m across outside, 29.6 m inside): the 17 uprights that stand
  (the run 27–30 / 1–7 on the north-east, 10 and the small stone 11 on the south, 16 beside the Great
  Trilithon, 21–23 and 26 on the west), with the six lintels still in place (over 29–30, 30–1 and 1–2 across
  the entrance, over 4–5 and 6–7, and over 21–22), curved to the circle; the fallen and broken uprights where
  OpenStreetMap draws them.
- The sarsen horseshoe of five trilithons opening to the north-east: the three that stand complete with their
  lintels (the two north-eastern ones and the western one), the Great Trilithon with its one upright (7.3 m) and
  its fallen partner broken across the Altar Stone with the fallen lintel, the north-western trilithon with its
  one upright and the fallen upright and lintel.
- The bluestones: the circle (about 25 stones today — standing, leaning, fallen and stumps) and the horseshoe
  inside the trilithons, with the tall standing ones in front of the Great Trilithon.
- The Altar Stone lying under the fallen stones, the Slaughter Stone lying in the entrance, the Heel Stone
  (leaning towards the circle) with its ring ditch, the two surviving Station Stones (91 fallen, 93 standing),
  and the white markers of Aubrey Holes 1–32 that are set in the turf.
- The circular earthwork: the low inner bank, the ditch (~105 m across) and the faint outer counterscarp bank,
  with the wide entrance to the north-east (where the Avenue comes in) and the narrow one to the south, the two
  «barrows» (the ditched enclosures of Station Stones 92 and 94) on the bank, and the grass inside — all on a
  gentle plane that follows the terrain (the site falls ~2 % towards the north-east).
- Not modelled: the Avenue beyond the earthwork, the visitor path and ropes, the surrounding fields.

Measurements (see meta.json for the sources):
- Plan: the stones drawn in OpenStreetMap (ways 99207713–99207761, 227236258, 227605772–776, 422721464–475,
  133262440 Heel Stone, 133262443 Slaughter Stone, 377365047 and 377367000 Station Stones): their centres,
  lengths, widths and directions are used for every fallen stone, the trilithons, the bluestones and the
  singular stones. The centre of the sarsen circle is fitted to the OpenStreetMap stones (r = 15.6 m, residuals
  ≤ 2.7 m, the OSM outlines are rough); the standing sarsen uprights and the lintels are placed on that circle at
  12° steps (the standard plan, stone 1 just right of the axis seen from the centre, numbered clockwise), which
  agrees with the OSM blobs within ~2°. The bluestones that OSM does not draw are filled in from the general
  plan (Cleal, Walker & Montague 1995; English Heritage plan): their positions and states are approximate.
- Sizes (Wikipedia en «Stonehenge», English Heritage): sarsen uprights 4.1 m above ground, 2.1 m wide, 1.1 m
  thick; lintels 3.2 × 1.0 × 0.8 m, so the lintel tops stand at 4.9 m; stone 11 about half the size; trilithon
  uprights 6.0 m (north-east pair), 6.5 m (the two middle ones), 7.3 m (stone 56 of the Great Trilithon), with
  lintels ~4.8 × 1.1 × 1.0 m; the Heel Stone 4.7 m above ground, leaning ~25° towards the circle (estimated);
  the Altar Stone 4.9 × 1.0 × 0.5 m; the Slaughter Stone 6.4 m long, lying; bluestones 1.5–2.5 m above ground.
  Heights of individual fallen stones ±20 %.
- Earthwork: from the OpenStreetMap ditch and embankment ways 377362073–377362116 — the ditch centre line has a
  radius of 52.8 m (its centre ~2 m south of the stone circle's), the inner bank 45–50 m, the counterscarp
  56–61 m, the Aubrey Holes on a circle of 43.5 m; the relief is low (bank ~0.5 m, ditch ~0.7 m, counterscarp
  ~0.25 m, estimated from photographs; the earthwork is much eroded).
- Ground: EU-DEM 25 m (Copernicus, via OpenTopoData): 101.4 m at the centre, ~99.6 m 60 m north-east,
  ~102.1 m 60 m south-west; the plane fitted to 25 samples within 80 m is used for every surface and stone
  base. The grass surfaces stand 0.3 m above that plane and go down in a skirt to ground_low, so that the real
  terrain does not show through.
Sarsen grey with yellow-green and pale grey lichen, darker bluestones, the greenish Altar Stone.

Axes (docs/MODEL_SPEC.md): +Y up, +Z points along the axis of the solstices to the Heel Stone (heading 49.9°,
the midsummer sunrise / midwinter sunset axis of the Avenue), +X points to 319.9°. Origin on the ground at the
centre of the sarsen circle. ground: keep (the site slopes).

    python model_sources/Q39671_stonehenge.py   -> models/Q39671/model.glb (+ triangles and footprint in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, fbm, normals, sstep, write_glb  # noqa: E402

HEADING = 49.9
LAT0, LON0 = 51.1788594, -1.8261942     # centre of the sarsen circle (fitted to the OSM stones)

# ── Colours ──────────────────────────────────────────────────────────────────
SARSEN = (0.57, 0.56, 0.53)
SARSEN_D = (0.40, 0.39, 0.37)
LICHEN_Y = (0.70, 0.69, 0.44)          # yellow-green crustose lichen
LICHEN_W = (0.80, 0.79, 0.75)          # pale grey lichen
BLUE = (0.42, 0.44, 0.47)              # spotted dolerite and rhyolite bluestones
ALTAR = (0.50, 0.52, 0.45)             # the greenish sandstone of the Altar Stone
GRASS = (0.42, 0.55, 0.29)
GRASS_L = (0.50, 0.59, 0.32)           # dry grass on the bank
GRASS_D = (0.35, 0.48, 0.25)           # the damper ditch
CHALK = (0.90, 0.90, 0.86)

# ── Frame: world (east, north) metres from the centre -> model (x, z) ───────
_H = math.radians(HEADING)


def to_model(e, n):
    return -e * math.cos(_H) + n * math.sin(_H), e * math.sin(_H) + n * math.cos(_H)


def polar(theta_deg, r):
    """theta clockwise (seen from above) from the axis (+Z), r in metres -> model (x, z)"""
    t = math.radians(theta_deg)
    return -r * math.sin(t), r * math.cos(t)


def yaw_for(azimuth_deg):
    """Rotation about Y that turns a piece's local +X to the world azimuth given."""
    return -math.radians(azimuth_deg - HEADING + 90.0)


def theta_of(x, z):
    return math.degrees(math.atan2(-x, z)) % 360


# ── Ground: the plane of EU-DEM + the earthwork ──────────────────────────────
# EU-DEM 25 m samples (OpenTopoData, 2026-11-18): centre, then r = 30, 60, 80 m at azimuths 0, 45 … 315.
_DEM = [101.4, 100.9, 100.7, 100.8, 101.1, 101.5, 101.8, 101.8, 101.4, 100.0, 99.6, 100.2, 100.6, 101.3, 102.1,
        101.9, 101.0, 99.2, 98.7, 99.8, 100.1, 100.9, 102.2, 102.0, 100.6]


def _fit_plane():
    pts = [(0.0, 0.0)]
    for r in (30, 60, 80):
        for a in range(0, 360, 45):
            pts.append((r * math.sin(math.radians(a)), r * math.cos(math.radians(a))))
    A = np.array([[e, n, 1.0] for e, n in pts])
    ge, gn, c = np.linalg.lstsq(A, np.array(_DEM), rcond=None)[0]
    # gradient in model axes
    gx = -ge * math.cos(_H) + gn * math.sin(_H)
    gz = ge * math.sin(_H) + gn * math.cos(_H)
    return gx, gz


GX, GZ = _fit_plane()
LIFT = 0.3                                   # the grass stands this much above the fitted plane
EW_C = to_model(0.6, -2.3)                   # centre of the earthwork (OSM ditch and banks)


def plane(x, z):
    return GX * np.asarray(x) + GZ * np.asarray(z)


def _bump(r, c, w):
    t = np.clip(1 - ((r - c) / w) ** 2, 0, 1)
    return t * t


def _gap(az, a0, a1, soft=3.0):
    """1 outside the azimuth window [a0, a1], 0 inside, soft edges"""
    return 1 - sstep(-soft, 0.0, np.minimum(az - a0, a1 - az))


BARROWS = [to_model(-8.2, 36.6), to_model(14.8, -42.3)]     # ditched enclosures of Station Stones 94 and 92


def relief(x, z):
    """Earthwork relief above the plane: bank, ditch, counterscarp, barrows (world azimuths for the gaps)."""
    x = np.asarray(x, float); z = np.asarray(z, float)
    dx, dz = x - EW_C[0], z - EW_C[1]
    r = np.hypot(dx, dz)
    az = (np.degrees(np.arctan2(-dx, dz)) + HEADING) % 360            # world azimuth from the earthwork centre
    mask = _gap(az, 38.0, 60.0, 6.0) * _gap(az, 167.0, 176.0, 4.0)
    p = np.stack([x, np.zeros_like(x), z], 1)
    wob = 0.75 + 0.5 * fbm(p, 0.05, 11, 2)                          # the bank is eroded unevenly
    h = mask * wob * (0.5 * _bump(r, 47.6, 3.4) - 0.7 * _bump(r, 52.8, 3.0) + 0.25 * _bump(r, 58.6, 2.3))
    for bx, bz in BARROWS:
        rb = np.hypot(x - bx, z - bz)
        h = h + 0.22 * _bump(rb, 0.0, 4.6) - 0.3 * _bump(rb, 5.8, 1.6)
    return h


def ground(x, z):
    return plane(x, z) + LIFT + relief(x, z)


def g0(x, z):
    """Ground level at a point, as a float"""
    return float(ground(np.array([x]), np.array([z]))[0])


# ── Rough stones ─────────────────────────────────────────────────────────────
_FACES = [(0, 1, 1, 2), (0, -1, 2, 1), (1, 1, 2, 0), (1, -1, 0, 2), (2, 1, 0, 1), (2, -1, 1, 0)]   # (axis, sign, u, v)


def rock(w, h, d, seed, y0=0.0, rnd=4.0, amp=None, taper=0.0, seg=0.55, lump=0.10):
    """A rough block w (x) × (h - y0) (y) × d (z), from y0 to h, centred on x/z: a subdivided, rounded and
    crumpled box with welded vertices (smooth shading)."""
    dims = np.array([w, h - y0, d])
    n = [int(max(2, min(10, round(s / seg)))) for s in dims]
    P, F = [], []
    for k, s, a, b in _FACES:
        ga = np.linspace(-1, 1, n[a] + 1); gb = np.linspace(-1, 1, n[b] + 1)
        o = len(P)
        for i in range(n[a] + 1):
            for j in range(n[b] + 1):
                q = [0.0, 0.0, 0.0]; q[k] = s; q[a] = ga[i]; q[b] = gb[j]
                P.append(q)
        W = n[b] + 1
        for i in range(n[a]):
            for j in range(n[b]):
                p0 = o + i * W + j
                F += [(p0, p0 + W, p0 + W + 1), (p0, p0 + W + 1, p0 + 1)]
    P = np.array(P); F = np.array(F)
    key = np.round(P * 1000).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    P = P[first]; F = inv.ravel()[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    # rounded box (superellipsoid)
    P = P / (np.sum(np.abs(P) ** rnd, axis=1, keepdims=True) ** (1 / rnd))
    t = (P[:, 1] + 1) / 2
    v = np.stack([P[:, 0] * w / 2, y0 + t * (h - y0), P[:, 2] * d / 2], 1)
    if taper:
        k = 1 - taper * t
        v[:, 0] *= k; v[:, 2] *= k
    m = Mesh(v, F, SARSEN)
    a = (0.06 * min(w, d) + 0.03) if amp is None else amp
    big = min(w, d)
    d1 = np.stack([fbm(v, 0.33, seed + 31 * q, 2) - 0.5 for q in range(3)], 1) * 2 * lump * big
    d2 = np.stack([fbm(v, 1.1, seed + 7 + 13 * q, 2) - 0.5 for q in range(3)], 1) * 2 * a
    keep_base = sstep(y0, y0 + 0.4, v[:, 1])[:, None]
    m.v = v + (d1 + d2) * np.array([1.0, 0.35, 1.0]) * keep_base
    return m


def paint_stone(m, base, seed, lichen=1.0):
    n = normals(m)
    p = m.v
    k = 0.88 + 0.22 * fbm(p, 0.8, seed, 3)
    c = np.array(base)[None, :] * k[:, None]
    ly = sstep(0.50, 0.68, fbm(p, 0.9, seed + 5, 3)) * (0.55 + 0.45 * sstep(-0.2, 0.8, n[:, 1])) * lichen
    lw = sstep(0.54, 0.68, fbm(p, 1.6, seed + 9, 2)) * 0.85 * lichen
    st = sstep(0.55, 0.75, fbm(p, (0.9, 0.12, 0.9), seed + 13, 2)) * 0.45         # dark weathering streaks
    c = c * (1 - st[:, None]) + np.array(SARSEN_D) * st[:, None]
    c = c * (1 - ly[:, None]) + np.array(LICHEN_Y) * ly[:, None]
    c = c * (1 - lw[:, None]) + np.array(LICHEN_W) * lw[:, None]
    dark = sstep(-0.1, -0.7, n[:, 1]) * 0.35                     # overhangs and the under-sides
    c = c * (1 - dark[:, None]) + np.array(SARSEN_D) * dark[:, None]
    m.c = np.clip(c, 0, 1)
    return m


STONES, BLUES, SPECIAL = [], [], []
_seed = [100]


def nseed():
    _seed[0] += 1
    return _seed[0]


def place(m, x, z, yaw, y=None, lean_x=0.0, lean_z=0.0):
    m.R(lean_x, 0, lean_z).R(0, yaw, 0)
    return m.T(x, g0(x, z) if y is None else y, z)


def upright(theta, r, w, h, d, base=SARSEN, lean=(0.0, 0.0), taper=0.08, dest=STONES, bury=1.2, rnd=5.0, seed=None):
    """A standing stone whose wide face faces the centre (tangential), at polar (theta, r)."""
    x, z = polar(theta, r)
    s = nseed() if seed is None else seed
    m = rock(w, h, d, s, y0=-bury, taper=taper, rnd=rnd, amp=0.07 * d + 0.03, lump=0.09)
    paint_stone(m, base, s)
    yaw = yaw_for(HEADING + theta + 90)
    dest.append(place(m, x, z, yaw, lean_x=lean[0], lean_z=lean[1]))
    return x, z


def lying(e, n, L, W, ori, t, base=SARSEN, dest=STONES, sink=0.25, tilt=0.0, y=None, model_xy=False):
    """A fallen stone lying on the turf: length L along the world azimuth ori, width W, thickness t above the
    ground (sunk a little into it). e, n in world metres from the centre (or model x, z if model_xy)."""
    x, z = (e, n) if model_xy else to_model(e, n)
    s = nseed()
    m = rock(L, t, W, s, y0=-sink, rnd=3.5, amp=0.06 * min(W, t) + 0.03, lump=0.08)
    paint_stone(m, base, s)
    dest.append(place(m, x, z, yaw_for(ori), y=y, lean_z=tilt))


# ── OpenStreetMap stones: centre (east, north from the circle centre), length, width, long axis azimuth ─────
OSM = {
    99207713: (0.81, -7.31, 5.69, 3.48, 59),       # south-eastern trilithon (stands, with lintel)
    99207718: (-7.87, -3.68, 2.98, 1.88, 136),     # stump of the fallen upright of the Great Trilithon
    99207720: (-10.61, 11.88, 2.48, 1.85, 50),     # sarsen 23 (stands)
    99207722: (-3.54, -2.91, 3.90, 2.38, 32),      # fallen upright of the Great Trilithon (piece)
    99207723: (-7.32, 2.54, 5.88, 2.80, 30),       # western trilithon (stands, with lintel)
    99207724: (11.08, 9.55, 12.16, 3.13, 142),     # sarsens 29, 30, 1, 2 with their lintels (one outline)
    99207725: (-13.85, -8.03, 3.40, 2.13, 150),    # sarsen 16 (stands)
    99207726: (-7.07, 15.53, 4.17, 1.83, 89),      # fallen sarsen (24/25)
    99207727: (-13.83, 8.42, 5.29, 2.11, 37),      # sarsens 21, 22 with lintel 122
    99207728: (6.91, -2.37, 5.50, 2.91, 40),       # north-eastern trilithon (stands, with lintel)
    99207729: (4.33, 14.80, 2.39, 2.36, 21),       # sarsen 28
    99207730: (14.54, -1.54, 4.94, 2.07, 2),       # sarsens 4, 5 with lintel
    99207731: (-7.48, -10.52, 5.11, 2.49, 48),     # fallen sarsen (14/15), lying inwards
    99207732: (12.37, -7.52, 5.77, 2.38, 33),      # sarsens 6, 7 with lintel
    99207733: (8.54, -15.97, 1.63, 1.28, 137),     # piece of sarsen 9
    99207734: (3.67, -13.54, 2.93, 2.07, 82),      # sarsen 10
    99207735: (-6.13, -5.72, 4.12, 2.36, 48),      # fallen lintel of the Great Trilithon
    99207736: (-1.90, 15.32, 3.00, 1.63, 87),      # sarsen 26
    99207738: (7.47, -13.90, 2.56, 1.20, 58),      # fallen sarsen 9
    99207741: (-2.97, 6.04, 4.23, 2.49, 135),      # fallen upright of the north-western trilithon
    99207745: (-1.66, 9.15, 2.85, 2.18, 37),       # standing upright of the north-western trilithon
    99207747: (-2.77, -16.17, 5.14, 2.45, 169),    # fallen sarsen 12
    99207750: (1.15, 15.36, 2.50, 2.35, 114),      # sarsen 27
    99207752: (14.58, 3.01, 2.55, 1.77, 171),      # sarsen 3
    99207754: (-9.21, -8.02, 1.97, 1.56, 57),      # broken piece
    99207756: (11.14, -13.26, 2.23, 2.02, 102),    # broken sarsen 8
    99207759: (-19.21, 1.16, 3.63, 2.35, 16),      # fallen sarsen 19, lying outwards
    99207761: (-4.32, -0.54, 4.75, 1.39, 25),      # fallen upright of the Great Trilithon, across the Altar Stone
    133262440: (60.09, 49.42, 3.13, 2.63, 117),    # Heel Stone (96)
    133262443: (34.46, 27.67, 6.12, 2.29, 48),     # Slaughter Stone (95)
    227236258: (-3.00, -1.12, 4.06, 0.83, 147),    # Altar Stone (80)
    227605772: (-0.82, 3.87, 2.23, 0.76, 73),      # bluestone horseshoe (fallen)
    227605773: (0.01, 2.76, 1.92, 0.79, 58),       # bluestone horseshoe (fallen)
    227605774: (3.03, 1.90, 1.37, 1.12, 150),      # bluestone horseshoe (stump)
    227605775: (1.93, 4.21, 1.28, 1.08, 77),       # bluestone horseshoe
    227605776: (0.38, 5.40, 1.10, 0.99, 149),      # bluestone horseshoe
    377365047: (39.75, -20.00, 2.99, 1.76, 124),   # Station Stone 91 (fallen)
    377367000: (-38.80, 19.09, 1.72, 1.31, 17),    # Station Stone 93 (stands)
    422721464: (3.57, 11.34, 0.85, 0.56, 138),     # bluestone circle …
    422721465: (5.18, 10.82, 0.85, 0.84, 94),
    422721466: (6.12, 9.11, 0.82, 0.69, 98),
    422721467: (7.58, 8.60, 0.84, 0.80, 110),
    422721468: (9.33, 6.51, 0.96, 0.59, 134),
    422721469: (10.68, 1.70, 1.95, 0.84, 88),      # bluestone 32 (fallen)
    422721470: (11.08, 3.02, 0.74, 0.74, 86),      # bluestone 150
    422721471: (4.89, -10.40, 0.88, 0.79, 54),
    422721472: (0.98, -15.68, 1.82, 1.35, 82),     # sarsen 11 (the small one)
    422721473: (-13.37, -2.22, 2.18, 0.66, 35),    # bluestone (fallen)
    422721474: (-4.18, 11.82, 2.04, 0.75, 164),    # bluestone (fallen)
    422721475: (-3.27, -10.55, 1.35, 0.98, 117),   # bluestone (stump)
}


def osm_lying(i, t, base=SARSEN, dest=STONES, **kw):
    e, n, L, W, ori = OSM[i]
    lying(e, n, L, W, ori, t, base, dest, **kw)


def osm_theta(i):
    e, n = OSM[i][:2]
    x, z = to_model(e, n)
    return theta_of(x, z), math.hypot(x, z)


# ── The sarsen circle ────────────────────────────────────────────────────────
R_SC = 15.6
STANDING = {1, 2, 3, 4, 5, 6, 7, 10, 11, 16, 21, 22, 23, 26, 27, 28, 29, 30}
LINTELS = [(29, 30), (30, 1), (1, 2), (4, 5), (6, 7), (21, 22)]
H_SC = 4.1


def sc_theta(k):
    return (k - 0.5) * 12.0


def sarsen_circle():
    rng = np.random.default_rng(2500)
    with_lintel = {a for a, b in LINTELS} | {b for a, b in LINTELS}
    for k in sorted(STANDING):
        th = sc_theta(k) + (rng.uniform(-0.6, 0.6) if k not in with_lintel else 0.0)
        if k == 11:
            upright(th, R_SC, 1.2, 2.6, 0.9, lean=(0.0, 0.03))
            continue
        h = H_SC if k in with_lintel else H_SC + rng.uniform(-0.25, 0.15)
        lean = (0.0, 0.0) if k in with_lintel else (rng.uniform(-0.03, 0.03), rng.uniform(-0.02, 0.02))
        if k == 26:
            lean = (-0.06, 0.0)                                      # leans a little outwards
        upright(th, R_SC + rng.uniform(-0.15, 0.15), 2.1 + rng.uniform(-0.15, 0.2), h, 1.1, lean=lean, taper=0.06)
    # lintels: each spans from the middle of one upright to the middle of the next, curved to the circle
    for a, b in LINTELS:
        t0 = sc_theta(a); t1 = sc_theta(b)
        if t1 < t0:
            t1 += 360
        tc = (t0 + t1) / 2
        L = math.radians(t1 - t0) * R_SC - 0.06
        s = nseed()
        m = rock(L, 0.8, 1.0, s, y0=0.0, rnd=6.0, amp=0.03, lump=0.03)
        paint_stone(m, SARSEN, s)
        tcr = math.radians(tc)
        lx, ly, lz = m.v[:, 0], m.v[:, 1], m.v[:, 2]
        phi = tcr + lx / R_SC
        rr = R_SC - lz
        yb = g0(*polar(tc, R_SC)) + H_SC
        m.v = np.stack([-rr * np.sin(phi), ly + yb, rr * np.cos(phi)], 1)
        STONES.append(m)
    # fallen and broken uprights (OpenStreetMap)
    osm_lying(99207747, 1.0)                    # 12
    osm_lying(99207731, 1.0)                    # 14/15
    osm_lying(99207754, 0.7)
    osm_lying(99207759, 0.95)                   # 19
    osm_lying(99207726, 0.9)                    # 24/25
    osm_lying(99207738, 0.8)                    # 9
    osm_lying(99207733, 0.6)
    osm_lying(99207756, 1.2, tilt=0.1)          # 8, broken
    # stumps of the missing uprights (low remnants in the turf)
    for k in (13, 17, 18, 20):
        upright(sc_theta(k) + rng.uniform(-2, 2), R_SC, 1.6, 0.35, 0.9, bury=0.5, taper=0.0)


# ── The trilithons ───────────────────────────────────────────────────────────
def trilithon(i, h, lintel=True):
    """A standing trilithon on an OSM outline: two uprights side by side along the outline, a lintel on top."""
    e, n, L, W, ori = OSM[i]
    x, z = to_model(e, n)
    yaw = yaw_for(ori)
    gy = g0(x, z)
    ca, sa = math.cos(yaw), -math.sin(yaw)                                  # local +X in model (x, z)
    for side in (-1, 1):
        s = nseed()
        m = rock(2.2, h, 1.15, s, y0=-1.5, taper=0.07, rnd=6.0, amp=0.04, lump=0.04)
        paint_stone(m, SARSEN, s)
        m.R(0, yaw, 0).T(x + side * 1.25 * ca, gy, z + side * 1.25 * sa)
        STONES.append(m)
    if lintel:
        s = nseed()
        m = rock(4.8, 1.0, 1.1, s, y0=0.0, rnd=6.0, amp=0.03, lump=0.03)
        paint_stone(m, SARSEN, s)
        STONES.append(m.R(0, yaw, 0).T(x, gy + h * 0.995, z))


def horseshoe():
    trilithon(99207728, 6.0)                    # north-east
    trilithon(99207713, 6.5)                    # south-east
    trilithon(99207723, 6.5)                    # west
    # the Great Trilithon: stone 56 stands (7.3 m), its partner lies broken across the Altar Stone
    th, r = osm_theta(99207718)
    th56 = th - 15.5
    upright(th56, r - 0.4, 2.4, 7.3, 1.2, taper=0.05, bury=2.0, rnd=6.0)
    upright(th, r, 2.2, 0.9, 1.2, bury=0.8, taper=0.0)                     # stump of 55
    osm_lying(99207761, 1.1, y=g0(*to_model(*OSM[99207761][:2])) + 0.2, tilt=0.04)
    osm_lying(99207722, 1.0)
    e, n, L, W, ori = OSM[99207735]                                         # lintel 156, fallen
    lying(e - 0.5 * math.sin(math.radians(HEADING)), n - 0.5 * math.cos(math.radians(HEADING)), 4.6, 1.1, ori, 1.0)
    # the north-western trilithon: one upright stands, the other and the lintel lie inside
    th, r = osm_theta(99207745)
    upright(th, r, 2.2, 6.0, 1.15, bury=1.5, rnd=6.0, taper=0.07)
    osm_lying(99207741, 1.05)
    lx, lz = polar(276.0, 9.6)
    lying(lx, lz, 4.6, 1.1, (HEADING + 276 + 90) % 360 + 8, 0.95, model_xy=True)


# ── Bluestones ───────────────────────────────────────────────────────────────
R_BC = 11.6


def bluestones():
    rng = np.random.default_rng(1600)
    done = []
    # those drawn in OSM
    for i in (422721464, 422721465, 422721466, 422721467, 422721468, 422721470, 422721471):
        th, r = osm_theta(i)
        upright(th, r, 0.85, rng.uniform(1.7, 2.2), 0.65, base=BLUE, dest=BLUES, bury=0.8, taper=0.12,
                lean=(rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)))
        done.append(th)
    for i in (422721469, 422721473, 422721474):
        osm_lying(i, 0.55, BLUE, BLUES); done.append(osm_theta(i)[0])
    th, r = osm_theta(422721475)
    upright(th, r, 1.2, 0.5, 0.9, base=BLUE, dest=BLUES, bury=0.4, taper=0.0); done.append(th)
    # the rest of the circle, from the general plan (approximate): (theta, state)
    plan = [(72, "s"), (84, "s"), (93, "l"), (117, "s"), (124, "t"), (132, "s"), (141, "l"), (156, "t"), (164, "s"),
            (172, "t"), (186, "l"), (196, "t"), (218, "s"), (228, "t"), (238, "l"), (250, "s"), (262, "t"),
            (272, "s"), (282, "t"), (300, "l"), (311, "s"), (318, "t")]
    for th, st in plan:
        if min(abs((th - d + 180) % 360 - 180) for d in done) < 4:
            continue
        r = R_BC + rng.uniform(-0.3, 0.3)
        if st == "s":                                       # standing
            upright(th, r, rng.uniform(0.7, 1.0), rng.uniform(1.6, 2.1), 0.6, base=BLUE, dest=BLUES, bury=0.8,
                    taper=0.12, lean=(rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)))
        elif st == "l":                                     # fallen
            x, z = polar(th, r)
            lying(x, z, rng.uniform(1.8, 2.4), 0.8, (HEADING + th + 90 + rng.uniform(-30, 30)) % 360, 0.5,
                  BLUE, BLUES, model_xy=True)
        else:                                               # stump
            upright(th, r, rng.uniform(0.7, 1.1), rng.uniform(0.25, 0.6), 0.6, base=BLUE, dest=BLUES, bury=0.4,
                    taper=0.0)
    # the bluestone horseshoe: OSM stones + the tall ones in front of the trilithons (approximate)
    for i in (227605772, 227605773):
        osm_lying(i, 0.5, BLUE, BLUES)
    for i, h in ((227605775, 1.9), (227605776, 2.0)):
        th, r = osm_theta(i)
        upright(th, r, 0.9, h, 0.7, base=BLUE, dest=BLUES, bury=0.8, taper=0.15)
    th, r = osm_theta(227605774)
    upright(th, r, 1.1, 0.5, 0.9, base=BLUE, dest=BLUES, bury=0.4, taper=0.0)
    for th, r, h in ((96, 4.3, 1.8), (112, 4.9, 0.4), (128, 5.3, 2.1), (146, 5.6, 2.3), (160, 5.9, 2.4),
                     (226, 5.4, 2.2), (240, 5.0, 0.4), (254, 4.6, 2.0), (268, 4.3, 1.7)):
        upright(th, r, 0.8, h, 0.6, base=BLUE, dest=BLUES, bury=0.8 if h > 1 else 0.4, taper=0.15 if h > 1 else 0.0,
                lean=(rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)))


# ── Singular stones ──────────────────────────────────────────────────────────
def singular():
    # Altar Stone, lying partly buried under the fallen stones
    e, n, L, W, ori = OSM[227236258]
    lying(e, n, 4.9, 1.0, ori, 0.25, ALTAR, SPECIAL, sink=0.3)
    # Slaughter Stone, lying flat in the entrance, half sunk in the turf
    osm_lying(133262443, 0.45, dest=SPECIAL, sink=0.3)
    # Heel Stone: 4.7 m above ground, unworked, leaning towards the circle
    e, n, L, W, ori = OSM[133262440]
    x, z = to_model(e, n)
    s = nseed()
    m = rock(2.5, 5.1, 2.0, s, y0=-1.2, rnd=3.0, amp=0.18, lump=0.16, taper=0.25)
    paint_stone(m, SARSEN, s)
    # lean ~25° towards the circle (-Z in the model, the circle lies behind it)
    m.R(-math.radians(25), 0, 0)
    m.R(0, math.radians(8), 0)
    SPECIAL.append(m.T(x, g0(x, z), z))
    # Station Stones: 91 lies fallen, 93 stands (~1.2 m)
    osm_lying(377365047, 0.7, dest=SPECIAL)
    e, n, L, W, ori = OSM[377367000]
    x, z = to_model(e, n)
    s = nseed()
    m = rock(1.4, 1.2, 1.0, s, y0=-0.8, rnd=3.5, amp=0.08, lump=0.1)
    paint_stone(m, SARSEN, s)
    SPECIAL.append(m.R(0, yaw_for(ori), 0).T(x, g0(x, z), z))


# ── Ground ───────────────────────────────────────────────────────────────────
R_OUT = 63.0
GROUND_LOW = -2.6


def grass_colour(v):
    x, z = v[:, 0], v[:, 2]
    rel = relief(x, z)
    p = v.copy()
    k = 0.93 + 0.12 * fbm(p, 0.04, 3, 3)
    c = np.array(GRASS)[None, :] * k[:, None]
    up = sstep(0.05, 0.4, rel)[:, None]
    dn = sstep(-0.05, -0.45, rel)[:, None]
    c = c * (1 - up) + np.array(GRASS_L) * up * k[:, None]
    c = c * (1 - dn) + np.array(GRASS_D) * dn * k[:, None]
    return c


def polar_grid(cx, cz, radii, seg, hfn):
    a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    V = [[cx, 0.0, cz]]
    for r in radii[1:]:
        V += [[cx + r * math.sin(t), 0.0, cz + r * math.cos(t)] for t in a]
    V = np.array(V)
    V[:, 1] = hfn(V[:, 0], V[:, 2])
    F = []
    for j in range(seg):                                     # fan around the centre
        F.append((0, 1 + (j + 1) % seg, 1 + j))
    for i in range(len(radii) - 2):
        o0, o1 = 1 + i * seg, 1 + (i + 1) * seg
        for j in range(seg):
            k = (j + 1) % seg
            F += [(o0 + j, o0 + k, o1 + k), (o0 + j, o1 + k, o1 + j)]
    m = Mesh(V, F, GRASS)
    # skirt down to below ground_low around the edge
    o = 1 + (len(radii) - 2) * seg
    rim = V[o:o + seg]
    low = rim.copy(); low[:, 1] = GROUND_LOW - 0.3
    base = len(V)
    V2 = np.vstack([V, low])
    for j in range(seg):
        k = (j + 1) % seg
        F += [(o + j, base + k, o + k), (o + j, base + j, base + k)]
    return Mesh(V2, F, GRASS)


def earthwork():
    radii = [0.0] + list(np.arange(3.0, 27.1, 3.0)) + list(np.arange(28.0, R_OUT + 0.01, 0.6))
    m = polar_grid(EW_C[0], EW_C[1], radii, 320, ground)
    m.c = grass_colour(m.v)
    sk = m.v[:, 1] < GROUND_LOW
    m.c[sk] = np.array(GRASS_D) * 0.8
    # the Heel Stone with its ring ditch, on its own patch of turf
    e, n = OSM[133262440][:2]
    hx, hz = to_model(e, n)

    def heel_h(x, z):
        r = np.hypot(np.asarray(x) - hx, np.asarray(z) - hz)
        return plane(x, z) + LIFT - 0.25 * _bump(r, 4.4, 1.0)
    h = polar_grid(hx, hz, [0.0, 1.5, 3.0, 3.6, 4.0, 4.4, 4.8, 5.2, 6.0, 7.0], 48, heel_h)
    h.c = np.tile(GRASS, (len(h.v), 1)) * (0.93 + 0.12 * fbm(h.v, 0.04, 3, 3))[:, None]
    hr = np.hypot(h.v[:, 0] - hx, h.v[:, 2] - hz)
    dd = _bump(hr, 4.4, 1.0)[:, None]
    h.c = h.c * (1 - dd) + np.array(GRASS_D) * dd
    # Aubrey Holes 1–32: white markers in the turf, on a circle of 43.5 m
    discs = []
    for k in range(1, 33):
        az = 212.0 + (k - 26) * 360 / 56
        t = math.radians(az - HEADING)
        x, z = EW_C[0] - 43.5 * math.sin(t), EW_C[1] + 43.5 * math.cos(t)
        a = np.linspace(0, 2 * math.pi, 10, endpoint=False)
        V = np.array([[x, 0, z]] + [[x + 0.5 * math.sin(q), 0, z + 0.5 * math.cos(q)] for q in a])
        V[:, 1] = ground(V[:, 0], V[:, 2]) + 0.04
        discs.append(Mesh(V, [(0, 1 + (j + 1) % 10, 1 + j) for j in range(10)], CHALK))
    return m, h, merge(discs)


# ── Footprint ────────────────────────────────────────────────────────────────
def footprint():
    e, n = OSM[133262440][:2]
    hx, hz = to_model(e, n)
    th_h = theta_of(hx - EW_C[0], hz - EW_C[1])
    R = R_OUT + 1.0
    half = math.degrees(math.asin(7.5 / R))
    pts = []
    for a in np.arange(0, 360, 6.0):
        d = (a - th_h + 180) % 360 - 180
        if abs(d) < half:
            continue
        pts.append((a, R))
    out = []
    for a, r in sorted(pts, key=lambda p: (p[0] - th_h) % 360):
        out.append((EW_C[0] - r * math.sin(math.radians(a)), EW_C[1] + r * math.cos(math.radians(a))))
    # the lobe out to the Heel Stone
    ux, uz = (hx - EW_C[0]), (hz - EW_C[1])
    L = math.hypot(ux, uz); ux, uz = ux / L, uz / L
    px, pz = -uz, ux
    near = math.sqrt(R * R - 7.5 * 7.5)
    lobe = [(EW_C[0] + ux * near - px * 7.5, EW_C[1] + uz * near - pz * 7.5),
            (EW_C[0] + ux * (L + 7.5) - px * 7.5, EW_C[1] + uz * (L + 7.5) - pz * 7.5),
            (EW_C[0] + ux * (L + 7.5) + px * 7.5, EW_C[1] + uz * (L + 7.5) + pz * 7.5),
            (EW_C[0] + ux * near + px * 7.5, EW_C[1] + uz * near + pz * 7.5)]
    ring = out + lobe
    # keep it a simple polygon: order by angle around the earthwork centre
    ring.sort(key=lambda p: math.atan2(p[0] - EW_C[0], p[1] - EW_C[1]))
    return ring


def build():
    sarsen_circle()
    horseshoe()
    bluestones()
    singular()
    grass, heel, aubrey = earthwork()
    return [("sarsens", merge(STONES)), ("bluestones", merge(BLUES)), ("altar_heel_slaughter_station", merge(SPECIAL)),
            ("earthwork", grass), ("heel_stone_ditch", heel), ("aubrey_holes", aubrey)]


def main():
    parts = build()
    out_dir = ROOT / "models" / "Q39671"
    out_dir.mkdir(parents=True, exist_ok=True)
    tris, size = write_glb(out_dir / "model.glb", parts, roughness=0.95)
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        mj["footprint"] = [[round(x, 1), round(z, 1)] for x, z in footprint()]
        meta.write_text(json.dumps(mj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{tris} triangles, {size / 1e6:.2f} MB; plane gradient x {GX:.4f} z {GZ:.4f}")


if __name__ == "__main__":
    main()
