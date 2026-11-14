"""
Estadio Santiago Bernabéu (Q164027), Madrid — built by code, as renovated in 2019–2024.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Manuel Muñoz Monasterio and Luis Alemany Soler (1947), enlarged 1982 and 1992–94; renovation
2019–2024 by gmp Architekten (von Gerkan, Marg und Partner), L35 Arquitectos and Ribas & Ribas -> protected.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 21041481 (building=stadium, multipolygon): outer way 1507411898 (leisure=stadium,
  wikidata Q164027) 242.7 × 220.1 m, area 47,372 m²; inner way 1446479375 = the pitch, 105.1 × 70.4 m with
  its long axis at 4.2° (north–south). The outer line is drawn round the new steel skin: straight along the
  Paseo de la Castellana (west), rounded corners, and a wider, slanting east side (Calle del Padre Damián),
  where the old «La Esquina del Bernabéu» building stood. The orthophoto (Esri World Imagery, looked at only)
  agrees: the skin is a rounded square, the east side a broad sloping panel.
- Heights: total height 60 m (Wikipedia en/es: the stands were raised 12 m for the roof); the retractable roof
  runs 55 m above the ground (Real Madrid); the 1994 building was 45 m high. The split of the 60 m (skin up to
  ~46 m, a dark recessed band, then the roof «cushion» clad with the same strips rising to the rim of the
  opening) is measured on aerial photos (±2–3 m).
- Roof: fixed part 28,000 m² + retractable part 8,000 m² (12 moving trusses) -> an opening of about
  108 × 74 m over the 105 × 68 m pitch. Modelled OPEN (the moving panels stowed, not shown).
- 360° video ring hanging under the roof round the pitch: 6.5 m high on the sides and corners, 11 m at the
  ends, its screens tilted 8° (a truncated cone) (Wikipedia es).
- Stands: four tiers (anfiteatros) of white/grey seats, here stepped bands (no single seats); the pitch is
  at street level (y = +0.3 m, so that the game's terrain does not cover it).
- The four corner towers (stairs and helical ramps, 1992–94) are wrapped inside the new skin: they are the
  rounded corners of the drum and are not seen as separate volumes any more.
- The skin: curved stainless-steel louvres (lamas) with horizontal gaps; here ~1.55 m louvres, each a tilted
  band and a dark gap (sawtooth), on the walls; concentric strips on the roof.
No club crest, name or sponsor is modelled.

Axes (docs/MODEL_SPEC.md): +Y up, the main (Castellana, west) façade faces +Z (heading 274.2°), +X points
along the pitch to the south (184.2°), origin = centre of the pitch, y = 0 its ground level (street level).

    python model_sources/Q164027_estadio_santiago_bernabeu.py   -> models/Q164027/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cap, normals, write_glb, METAL, GLASS  # noqa: E402

# ── OSM (relation 21041481, heading 274.2°, origin at the centre of the pitch), model x/z in metres ───────
OUTLINE = [(-67.31, 106.82), (-60.34, 107.43), (-42.85, 108.14), (-35.01, 108.17), (-28.56, 108.04), (-19.53, 107.94),
           (11.52, 107.40), (11.91, 107.39), (42.88, 106.76), (65.70, 106.21), (72.57, 105.76), (76.98, 105.30),
           (81.07, 104.12), (85.52, 102.56), (88.59, 100.53), (92.69, 97.67), (95.13, 95.72), (98.67, 91.44),
           (100.46, 88.81), (102.90, 84.51), (105.57, 77.40), (106.62, 70.90), (107.65, 64.01), (108.07, 57.76),
           (108.39, 49.40), (108.67, 41.89), (109.13, 19.32), (109.62, -17.34), (109.36, -73.56), (108.93, -79.45),
           (108.14, -83.55), (106.79, -87.35), (105.01, -90.99), (102.78, -94.33), (100.27, -97.05), (97.09, -99.70),
           (93.69, -101.96), (89.88, -103.62), (86.48, -104.57), (82.45, -105.36), (78.25, -106.21), (73.04, -107.09),
           (66.65, -107.99), (57.91, -108.99), (43.34, -112.49), (32.53, -115.68), (26.85, -118.80), (22.68, -124.55),
           (18.60, -127.44), (15.33, -128.87), (12.14, -130.27), (4.84, -132.24), (-3.46, -133.80), (-10.79, -134.63),
           (-21.85, -132.11), (-25.75, -130.37), (-38.75, -124.58), (-56.43, -115.97), (-73.94, -107.64),
           (-90.69, -99.45), (-98.17, -95.75), (-100.90, -94.38), (-103.46, -92.16), (-105.85, -89.52),
           (-106.90, -88.15), (-107.72, -85.75), (-108.91, -80.26), (-109.49, -66.52), (-110.04, -45.17),
           (-110.36, -19.04), (-110.41, 4.26), (-109.59, 31.22), (-108.36, 52.18), (-106.45, 69.33), (-105.72, 72.77),
           (-104.54, 76.54), (-102.57, 81.05), (-99.54, 86.62), (-95.75, 91.92), (-92.18, 95.38), (-88.36, 98.69),
           (-83.50, 101.70), (-76.42, 104.90), (-70.36, 106.36)]

N = 240                                   # points round every ring
FOUND = -12.0                             # foundations: the streets around fall several metres to the east
Y_PITCH = 0.30                            # the pitch, at street level (kept above the game's terrain)
Y_SKIN0 = 5.0                             # the louvred skin starts above a glazed ground storey
SKIN_SIDE, SKIN_CORNER = 42.0, 50.0       # top of the skin: it sweeps up at the corners, dips along the sides
BAND_SIDE, BAND_CORNER = 47.0, 50.6       # top of the dark recessed band under the roof edge
Y_RIM = 60.0                              # top of the roof at the rim of the opening (total height 60 m)
Y_UNDER_IN = 53.5                         # underside of the roof at the opening
Y_UNDER_OUT = 46.5                        # … and at the back of the top tier
OPEN_HX, OPEN_HZ, OPEN_R = 54.0, 37.0, 12.0          # roof opening (≈ 8,000 m²)
PLAT_HX, PLAT_HZ, PLAT_R = 59.5, 41.0, 6.0           # pitch + run-off, where the first row stands
SCREEN_SIDE, SCREEN_END, SCREEN_TILT = 6.5, 11.0, math.radians(8)

STEEL = (0.80, 0.81, 0.83)
STEEL_HI = (0.88, 0.89, 0.90)
STEEL_D = (0.62, 0.63, 0.66)
GAP = (0.20, 0.21, 0.23)
ROOF_GAP = (0.47, 0.48, 0.50)
DARK = (0.10, 0.105, 0.115)
GLAZE = (0.16, 0.19, 0.22)
CONCRETE = (0.62, 0.61, 0.58)
UNDER = (0.74, 0.75, 0.76)
SCREEN = (0.05, 0.06, 0.08)
SEAT = (0.86, 0.87, 0.88)
SEAT_D = (0.58, 0.59, 0.61)
FASCIA = (0.25, 0.27, 0.30)
GRASS = (0.29, 0.53, 0.21)
GRASS_D = (0.25, 0.47, 0.18)
RUNOFF = (0.24, 0.42, 0.20)
LINE = (0.95, 0.95, 0.93)


# ── Rings: every surface of revolution-like piece is a loft of N points along the same rays ─────────────────
def resample(P, n):
    """n points equally spaced along the closed polyline P."""
    P = np.asarray(P, float)
    Q = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(t, cum, Q[:, 0]), np.interp(t, cum, Q[:, 1])], 1)


def ray_hit(P, u):
    """Distance from the origin along the unit vector u to the polygon P (largest hit)."""
    P = np.asarray(P, float)
    best = 0.0
    for i in range(len(P)):
        a, b = P[i - 1], P[i]
        d = b - a
        den = u[0] * d[1] - u[1] * d[0]
        if abs(den) < 1e-12:
            continue
        t = (a[0] * d[1] - a[1] * d[0]) / den            # along the ray
        s = (a[0] * u[1] - a[1] * u[0]) / den            # along the edge
        if t > 0 and -1e-9 <= s <= 1 + 1e-9:
            best = max(best, t)
    return best


def rrect(hx, hz, r, n=12):
    pts = []
    for cx, cz, a0 in ((hx - r, hz - r, 0), (-hx + r, hz - r, 90), (-hx + r, -hz + r, 180), (hx - r, -hz + r, 270)):
        for a in np.linspace(a0, a0 + 90, n):
            pts.append((cx + r * math.cos(math.radians(a)), cz + r * math.sin(math.radians(a))))
    return pts


# the rays: through N points equally spaced along the OSM outline (fine at the rounded corners)
_O = resample(OUTLINE, N)
ANG = np.arctan2(_O[:, 1], _O[:, 0])
if np.any(np.diff(np.unwrap(ANG)) <= 0):
    ANG = ANG[::-1]
    _O = _O[::-1]
U = np.stack([np.cos(ANG), np.sin(ANG)], 1)
R_O = np.linalg.norm(_O, axis=1)                                         # outer face of the skin (OSM line)
R_I = np.array([ray_hit(rrect(OPEN_HX, OPEN_HZ, OPEN_R), u) for u in U])   # rim of the roof opening
R_P = np.array([ray_hit(rrect(PLAT_HX, PLAT_HZ, PLAT_R), u) for u in U])   # front of the stands
# the outward normal of the outline at each ray, to turn perpendicular set-backs into distances along the ray
_T = np.roll(_O, -1, 0) - np.roll(_O, 1, 0)
_NRM = np.stack([_T[:, 1], -_T[:, 0]], 1)
_NRM /= np.linalg.norm(_NRM, axis=1)[:, None]
if np.mean(np.sum(_NRM * U, 1)) < 0:
    _NRM = -_NRM
COS_O = np.clip(np.sum(_NRM * U, 1), 0.45, 1.0)
END = np.clip((np.abs(U[:, 0]) - 0.72) / 0.2, 0, 1)                      # 1 at the ends (behind the goals)
_c = np.clip((2 * np.abs(U[:, 0] * U[:, 1]) - 0.45) / 0.45, 0, 1)
CORN = _c * _c * (3 - 2 * _c)                                            # 1 at the four rounded corners
Y_SKIN1 = SKIN_SIDE + (SKIN_CORNER - SKIN_SIDE) * CORN
Y_BAND1 = BAND_SIDE + (BAND_CORNER - BAND_SIDE) * CORN


def o(d):
    """Radius d metres outside (d > 0) or inside the OSM line, measured square to the wall."""
    return R_O + d / COS_O


def i(d):
    return R_I + d


def p(d):
    return R_P + d


def f(t, a=None, b=None):
    """Fraction t of the way from radius array a (default the opening) to b (default the skin)."""
    a = R_I if a is None else a
    b = R_O if b is None else b
    return a + t * (b - a)


def Y(y):
    return np.full(N, float(y)) if np.isscalar(y) else np.asarray(y, float)


def band(r0, y0, r1, y1, c, face):
    """A strip between two rings (radius arrays, heights), closed round; face = 'out', 'in', 'up' or 'down'."""
    y0, y1 = Y(y0), Y(y1)
    A = np.stack([U[:, 0] * r0, y0, U[:, 1] * r0], 1)
    B = np.stack([U[:, 0] * r1, y1, U[:, 1] * r1], 1)
    v = np.vstack([A, B])
    k = np.arange(N)
    k1 = (k + 1) % N
    F = np.concatenate([np.stack([k, k1, N + k1], 1), np.stack([k, N + k1, N + k], 1)])
    m = Mesh(v, F, c)
    want = {"out": None, "in": None, "up": np.array([0, 1, 0]), "down": np.array([0, -1, 0])}[face]
    nn = np.cross(v[F[:, 1]] - v[F[:, 0]], v[F[:, 2]] - v[F[:, 0]])
    if want is None:
        cen = (v[F[:, 0]] + v[F[:, 1]] + v[F[:, 2]]) / 3
        rad = cen * np.array([1, 0, 1])
        s = np.sum(nn * rad, 1).sum() * (1 if face == "out" else -1)
    else:
        s = (nn @ want).sum()
    if s < 0:
        m.f = m.f[:, ::-1]
    return m


def path(pts, c, face):
    """Consecutive bands through a list of (radius array, y)."""
    return [band(pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1], c, face) for k in range(len(pts) - 1)]


def ring_poly(r):
    return [(float(U[k, 0] * r[k]), float(U[k, 1] * r[k])) for k in range(N)]


# ── The skin ─────────────────────────────────────────────────────────────────────────────────────────────
def bulge(s):
    """Set-back of the skin from the OSM line (<= 0) at the fraction s of its height: tucked in at the foot,
    the louvres stepping out as they rise (the skin overhangs), widest at about 3/4 and rounding in at the top."""
    s = np.asarray(s, float)
    return -4.0 * np.clip(1 - s / 0.72, 0, 1) ** 2 - 1.8 * np.clip((s - 0.86) / 0.14, 0, 1) ** 2


def skin():
    ms = []
    # foundations, glazed ground storey (set back), soffit under the skin
    ms.append(band(o(-5.0), FOUND, o(-5.0), 0.4, CONCRETE, "out"))
    ms.append(band(o(-5.0), 0.4, o(-5.0), Y_SKIN0 - 0.3, GLAZE, "out"))
    ms.append(band(o(-5.0), Y_SKIN0 - 0.3, o(-5.0), Y_SKIN0, STEEL_D, "out"))
    ms.append(band(o(-5.0), Y_SKIN0, o(bulge(0.0) - 0.35), Y_SKIN0, STEEL_D, "down"))
    # louvres: each a band tilted out at its top and a dark gap stepping back to the next one (a sawtooth)
    n = 28
    H = Y_SKIN1 - Y_SKIN0
    for k in range(n):
        s0, s1 = k / n, (k + 1) / n
        y0, y1 = Y_SKIN0 + s0 * H, Y_SKIN0 + s1 * H
        sg = s1 - 0.28 / H
        yg = Y_SKIN0 + sg * H
        c = STEEL if k % 2 else STEEL_HI
        ms.append(band(o(bulge(s0) - 0.35), y0, o(bulge(sg)), yg, c, "out"))
        ms.append(band(o(bulge(sg)), yg, o(bulge(s1) - 0.35), y1, GAP, "out"))
    # top of the skin: a lip back to the recessed dark band
    ms.append(band(o(bulge(1.0) - 0.35), Y_SKIN1, o(-6.5), Y_SKIN1, STEEL_D, "up"))
    ms.append(band(o(-6.5), Y_SKIN1, o(-6.5), Y_BAND1, DARK, "out"))
    return merge(ms)


# ── The roof: a cushion of concentric strips from the edge to the rim of the opening ───────────────────
def roof():
    ms = []
    # control profile from the outer edge to the rim: (radius array, y)
    ctrl = [(o(-4.6), Y_BAND1 - 0.5)]
    for a in np.linspace(0, 90, 9)[1:]:                  # rounded edge, centre 12 m in, radii 7.4 × 7 m
        t = math.radians(a)
        ctrl.append((o(-12.0 + 7.4 * math.cos(t)), Y_BAND1 - 0.5 + 7.0 * math.sin(t)))
    y_top0 = Y_BAND1 + 6.5
    for t in (0.35, 0.6, 0.8):                           # gently up towards the opening
        ctrl.append((f(t, o(-12.0), i(5.0)), y_top0 + (Y_RIM - 0.4 - y_top0) * t ** 0.8))
    ctrl.append((i(5.0), Y_RIM - 0.4))
    ctrl.append((i(2.0), Y_RIM))
    ctrl.append((i(0.6), Y_RIM - 0.6))
    # strips: walk the profile with a mean length, light strips with a thin dark joint every `pitch` metres
    mean = [float(np.mean(np.hypot(ctrl[k + 1][0] - ctrl[k][0], Y(ctrl[k + 1][1]) - Y(ctrl[k][1]))))
            for k in range(len(ctrl) - 1)]
    cum = np.concatenate([[0], np.cumsum(mean)])
    total = cum[-1]
    curve_len = cum[8]
    bps = set(cum.tolist())
    s = 0.0
    while s < total:                                     # finer strips on the rounded edge
        step = 1.6 if s < curve_len else 3.2
        bps.add(min(s + step - 0.35, total)); bps.add(min(s + step, total))
        s += step
    bps = sorted(b for b in bps if 0 <= b <= total)

    def at(sv):
        k = min(int(np.searchsorted(cum, sv, side="right")) - 1, len(ctrl) - 2)
        t = (sv - cum[k]) / (cum[k + 1] - cum[k])
        (r0, y0), (r1, y1) = ctrl[k], ctrl[k + 1]
        return r0 + t * (r1 - r0), Y(y0) + t * (Y(y1) - Y(y0))

    s0 = 0.0
    for b in bps[1:]:
        if b - s0 < 1e-6:
            continue
        mid = 0.5 * (s0 + b)
        step = 1.6 if mid < curve_len else 3.2
        joint = ((mid % step) > step - 0.35)
        ra, ya = at(s0)
        rb, yb = at(b)
        ms.append(band(ra, ya, rb, yb, ROOF_GAP if joint else (STEEL if (int(mid // step) % 2) else STEEL_HI), "up"))
        s0 = b
    # under the outer edge, the rim of the opening (the ring truss) and the underside
    ms.append(band(o(-6.5), Y_BAND1, o(-4.6), Y_BAND1 - 0.5, STEEL_D, "down"))
    ms.append(band(i(0.6), Y_RIM - 0.6, i(0.0), Y_UNDER_IN, STEEL_D, "in"))
    ms.append(band(i(0.0), Y_UNDER_IN, f(0.5, i(0.0), o(-7.0)), 0.5 * (Y_UNDER_IN + Y_UNDER_OUT) + 1.0, UNDER, "down"))
    ms.append(band(f(0.5, i(0.0), o(-7.0)), 0.5 * (Y_UNDER_IN + Y_UNDER_OUT) + 1.0, o(-7.0), Y_UNDER_OUT, UNDER, "down"))
    ms.append(band(o(-7.0), Y_UNDER_OUT, o(-6.5), Y_BAND1, DARK, "down"))
    return merge(ms)


def screen():
    """The 360° video ring: a truncated cone hanging under the roof, its screens facing the stands."""
    h = SCREEN_SIDE + (SCREEN_END - SCREEN_SIDE) * END
    r_top = i(4.0)
    y_top = Y_UNDER_IN - 1.0 - 0.6
    r_bot = r_top - h * math.tan(SCREEN_TILT)
    y_bot = y_top - h
    th = 1.2
    ms = [band(r_bot, y_bot, r_top, y_top, SCREEN, "out"),                     # the screens
          band(r_bot - th, y_bot, r_top - th, y_top, DARK, "in"),              # back towards the pitch
          band(r_bot - th, y_bot, r_bot, y_bot, DARK, "down"),
          band(r_top - th, y_top, r_top, y_top, DARK, "up"),
          band(r_top - th * 0.5, y_top, r_top - th * 0.5, Y_UNDER_IN - 0.4, STEEL_D, "out")]   # hangers (a thin skirt)
    return merge(ms)


# ── Stands: four tiers of stepped bands ─────────────────────────────────────────────────────────────────
TIERS = [  # (fraction of the way from the front row to the back wall at start, at end, y start, y end, rows)
    (0.00, 0.30, Y_PITCH + 1.0, 13.0, 9),
    (0.27, 0.47, 17.0, 27.0, 6),
    (0.44, 0.66, 30.5, 38.5, 6),
    (0.62, 0.90, 41.5, 45.0, 4),
]


def stands():
    ms = []
    back = o(-7.0)
    prev_top = None
    for n_t, (fa, fb, ya, yb, rows) in enumerate(TIERS):
        ra, rb = f(fa, R_P, back), f(fb, R_P, back)
        if prev_top is not None:                         # fascia: front of this tier, under it the boxes' glass
            pr, py = prev_top
            ms.append(band(pr, py, ra, py, SEAT_D, "up"))
            ms.append(band(ra, py, ra, ya - 1.1, GLAZE, "in"))
            ms.append(band(ra, ya - 1.1, ra, ya, FASCIA, "in"))
            ms.append(band(ra, ya, ra - 0.5, ya, FASCIA, "up"))
            ms.append(band(ra - 0.5, ya, ra - 0.5, ya - 1.1, FASCIA, "in"))
        else:
            ms.append(band(R_P, Y_PITCH, R_P, ya, FASCIA, "in"))
        for k in range(rows):
            r0 = ra + (rb - ra) * k / rows
            r1 = ra + (rb - ra) * (k + 1) / rows
            y0 = ya + (yb - ya) * k / rows
            y1 = ya + (yb - ya) * (k + 1) / rows
            ms.append(band(r0, y0, r0, y1, SEAT_D, "in"))   # riser
            ms.append(band(r0, y1, r1, y1, SEAT, "up"))     # tread with the seats
        prev_top = (rb, yb)
    pr, py = prev_top
    ms.append(band(pr, py, back, py, SEAT_D, "up"))
    ms.append(band(back, py, back, Y_UNDER_OUT, DARK, "in"))   # back wall up to the roof
    return merge(ms)


# ── Pitch ─────────────────────────────────────────────────────────────────────────────────────────────
def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def pitch():
    ms = [cap(ring_poly(R_P), Y_PITCH, RUNOFF)]
    L, W = 105.0, 68.0
    nb = 18
    for k in range(nb):                                   # mowing stripes
        x0 = -L / 2 + L * k / nb
        ms.append(B(x0, x0 + L / nb, Y_PITCH, Y_PITCH + 0.03, -W / 2, W / 2, GRASS if k % 2 else GRASS_D))
    y0, y1, w = Y_PITCH + 0.03, Y_PITCH + 0.05, 0.12

    def line(xa, za, xb, zb):
        if abs(za - zb) < 1e-6:
            ms.append(B(min(xa, xb) - w / 2, max(xa, xb) + w / 2, y0, y1, za - w / 2, za + w / 2, LINE))
        else:
            ms.append(B(xa - w / 2, xa + w / 2, y0, y1, min(za, zb) - w / 2, max(za, zb) + w / 2, LINE))

    hx, hz = L / 2, W / 2
    for z in (-hz, hz):
        line(-hx, z, hx, z)
    for x in (-hx, 0.0, hx):
        line(x, -hz, x, hz)
    for sx in (-1, 1):
        for d, half in ((16.5, 20.16), (5.5, 9.16)):       # penalty and goal areas
            line(sx * hx, -half, sx * (hx - d), -half)
            line(sx * hx, half, sx * (hx - d), half)
            line(sx * (hx - d), -half, sx * (hx - d), half)
        gx = sx * (hx + 0.06)                              # goals 7.32 × 2.44 m
        for z in (-3.66, 3.66):
            ms.append(B(gx - 0.06, gx + 0.06, Y_PITCH, Y_PITCH + 2.44, z - 0.06, z + 0.06, LINE))
        ms.append(B(gx - 0.06, gx + 0.06, Y_PITCH + 2.32, Y_PITCH + 2.44, -3.66, 3.66, LINE))
    seg = 48
    for k in range(seg):                                   # centre circle
        a0, a1 = 2 * math.pi * k / seg, 2 * math.pi * (k + 1) / seg
        r_in, r_out = 9.15 - w / 2, 9.15 + w / 2
        P = [(r_in * math.cos(a0), r_in * math.sin(a0)), (r_out * math.cos(a0), r_out * math.sin(a0)),
             (r_out * math.cos(a1), r_out * math.sin(a1)), (r_in * math.cos(a1), r_in * math.sin(a1))]
        ms.append(cap(P, y1, LINE))
    return merge(ms)


def main():
    out = ROOT / "models" / "Q164027"
    out.mkdir(parents=True, exist_ok=True)
    steel = merge([skin(), roof()])
    inner = merge([stands(), screen()])
    field = pitch()
    parts = [("skin_and_roof", steel, {"material": "steel"}), ("stands_and_screen", inner), ("pitch", field)]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.9, materials={"steel": METAL})
    allv = np.vstack([steel.v, inner.v, field.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
