"""
Estadio Metropolitano (Q1369687), Madrid — the Atlético de Madrid stadium, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Cruz y Ortiz Arquitectos (Antonio Cruz and Antonio Ortiz, both alive): the athletics stadium
«La Peineta» (competition 1989, opened 1994) and its conversion 2011–2017 into the present football stadium,
roof engineered by schlaich bergermann partner -> protected.

What is modelled:
- The roof, a spoked wheel: an outer double compression ring standing on the back of the stands, an inner
  double tension ring round the opening over the pitch (floodlights along it), radial cables between them with
  a zig-zag PTFE membrane (96 radial panels, white on top), a lower perforated PTFE skin underneath, and the
  membrane cantilevered outside the compression ring down to the outer edge of the old building, with a red
  gutter along the rim. The rim is high along the two long sides and sweeps down at the two ends («wavy shape
  and variable height», e-architect), so the whole stadium reads as a white lens.
- Under the roof: the curved white concrete band at the back of the upper tier (the crest of La Peineta, now
  carried all round), the dark zone of the raking beams under the stands, and the beige concrete blocks of the
  outer buildings (stairs, lifts, offices) with their narrow horizontal slits, taller along the old Peineta
  stand (west).
- Inside: three tiers of red seats (lower, middle with the box level, upper), the pitch 105 × 68 m with
  mowing stripes, lines and goals.
- Not modelled: names, crests, sponsors, the giant flag, the outer plazas, ramps, car parks, the separate kiosks.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 7525256 (building=stadium, multipolygon): outer ring 287.0 × 249.1 m = the roof seen
  from above (e-architect: roof about 286 m north–south by 248 m east–west, 46,500 m²); its inner ring = the roof
  opening, 141 × 101 m. Way 4409945 (leisure=pitch) gives the centre of the pitch and its long axis, 170.6°.
  Way 458731140 (leisure=stadium, wikidata Q1369687) is the stadium grounds, larger (plazas included).
- Roof: compression and tension rings 57 m apart (Arquitectura Viva, e-architect): so the compression ring is
  placed 57 m outside the opening and the membrane cantilevers the remaining ~15 m to the OSM outer line.
  Height 45–57 m above the pitch (Atlético de Madrid, e-architect).
- The pitch was sunk about 10 m when the stadium was converted (Arquitectura Viva; e-architect: the middle tier
  is entered at level 0.00). Offlined and the viewers draw the ground at y = 0 over the whole footprint, so a
  sunken pitch would be hidden: the pitch is drawn at street level (y = +0.3) and the bowl is shallower than
  the real one. The outside — what is seen from the air and from the streets — keeps its heights above the
  street: inner ring 46 m (39 m at its foot), compression ring 34–41 m, rim of the roof 12 m at the ends and
  38 m along the sides; the beige blocks 16–25 m. These come from the published 45–57 m above the pitch minus
  ~10 m and from photos (Wikimedia Commons «Wanda Metropolitano - 2017-09.jpg» from the west,
  «Madrid - Estadio Wanda Metropolitano 03.jpg», «Estadio Metropolitano - Madrid 001.jpg»), ±3–4 m.

Axes (docs/MODEL_SPEC.md): +Y up, the west side (La Peineta, the old main stand) faces +Z, heading 260.6°;
+X points along the pitch to 170.6° (south-south-east). Origin = centre of the pitch, y = 0 = street level.

    python model_sources/Q1369687_estadio_metropolitano.py   -> models/Q1369687/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cap, write_glb  # noqa: E402

# ── OSM relation 7525256, heading 260.6°, origin at the centre of the pitch (way 4409945), model x/z in metres ─
OUTLINE = [(-115.57, 82.75), (-124.82, 71.43), (-135.21, 52.89), (-140.15, 39.69), (-143.12, 28.25), (-144.33, 14.64),
           (-145.05, -1.08), (-145.47, -14.36), (-142.38, -27.99), (-139.18, -42.93), (-135.10, -53.13), (-128.03, -64.88),
           (-120.32, -76.21), (-109.06, -88.51), (-96.09, -99.30), (-79.39, -109.06), (-67.44, -113.83), (-48.03, -117.95),
           (-32.51, -120.59), (-12.10, -122.38), (8.32, -122.26), (26.54, -121.65), (56.56, -117.15), (71.11, -112.65),
           (84.87, -106.55), (99.05, -97.44), (112.51, -84.12), (122.54, -72.35), (129.96, -60.89), (136.44, -47.52),
           (139.92, -34.24), (141.13, -21.43), (141.85, -7.74), (141.72, 8.85), (140.82, 22.47), (139.12, 33.54),
           (136.11, 44.89), (132.51, 55.01), (125.36, 66.76), (115.26, 80.95), (98.48, 96.78), (81.99, 107.82),
           (66.32, 114.69), (44.91, 121.47), (29.81, 124.52), (18.91, 125.73), (7.07, 126.22), (-3.76, 126.58),
           (-15.93, 126.35), (-31.98, 123.35), (-46.49, 120.66), (-69.43, 113.80), (-90.81, 104.27), (-105.69, 92.63)]
OPENING = [(59.66, 30.37), (64.39, 23.47), (69.00, 14.01), (71.26, 1.75), (70.62, -8.91), (66.31, -20.17), (60.61, -28.80),
           (54.39, -35.49), (47.32, -40.14), (32.95, -46.49), (21.88, -49.16), (1.54, -50.93), (-10.73, -49.93),
           (-19.88, -49.03), (-33.87, -45.00), (-42.24, -41.45), (-49.24, -37.62), (-53.82, -34.20), (-61.57, -25.19),
           (-67.54, -12.66), (-69.65, 2.81), (-66.41, 17.08), (-58.91, 29.70), (-52.62, 34.69), (-44.35, 40.59),
           (-31.19, 45.70), (-12.00, 49.42), (4.59, 50.19), (22.07, 48.08), (39.05, 44.11), (50.14, 38.49)]

N = 192                      # rays = 96 radial membrane panels, each a ridge and a valley
SPAN = 57.0                  # tension ring -> compression ring
Y_PITCH = 0.30
FOUND = -4.0
Y_IN_TOP, Y_IN_BOT = 46.0, 39.0          # inner (tension) ring
PLEAT, PLEAT_OUT = 1.0, 0.6              # half-depth of the zig-zag membrane (main span, cantilever)

WHITE = (0.93, 0.93, 0.91)
WHITE_D = (0.87, 0.88, 0.87)
UNDER = (0.86, 0.86, 0.84)
UNDER_OUT = (0.78, 0.76, 0.72)
RED = (0.72, 0.09, 0.11)
RING = (0.80, 0.81, 0.82)
LAMP = (0.97, 0.96, 0.90)
CONC = (0.90, 0.89, 0.85)          # the white concrete band
BEIGE = (0.80, 0.75, 0.64)
BEIGE_D = (0.70, 0.66, 0.56)
SLIT = (0.25, 0.23, 0.21)
DARK = (0.33, 0.31, 0.29)
BEAM = (0.62, 0.59, 0.53)
SEAT = (0.70, 0.11, 0.13)
SEAT_D = (0.50, 0.08, 0.10)
FASCIA = (0.78, 0.78, 0.77)
GLAZE = (0.16, 0.19, 0.22)
GRASS = (0.29, 0.53, 0.21)
GRASS_D = (0.25, 0.47, 0.18)
RUNOFF = (0.24, 0.42, 0.20)
LINE = (0.95, 0.95, 0.93)


def resample(P, n):
    P = np.asarray(P, float)
    Q = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(t, cum, Q[:, 0]), np.interp(t, cum, Q[:, 1])], 1)


def ray_hit(P, u):
    P = np.asarray(P, float)
    best = 0.0
    for k in range(len(P)):
        a, b = P[k - 1], P[k]
        d = b - a
        den = u[0] * d[1] - u[1] * d[0]
        if abs(den) < 1e-12:
            continue
        t = (a[0] * d[1] - a[1] * d[0]) / den
        s = (a[0] * u[1] - a[1] * u[0]) / den
        if t > 0 and -1e-9 <= s <= 1 + 1e-9:
            best = max(best, t)
    return best


def rrect(hx, hz, r, n=12):
    pts = []
    for cx, cz, a0 in ((hx - r, hz - r, 0), (-hx + r, hz - r, 90), (-hx + r, -hz + r, 180), (hx - r, -hz + r, 270)):
        for a in np.linspace(a0, a0 + 90, n):
            pts.append((cx + r * math.cos(math.radians(a)), cz + r * math.sin(math.radians(a))))
    return pts


# the rays: through N points equally spaced along the rim of the opening
_I = resample(OPENING, N)
ANG = np.unwrap(np.arctan2(_I[:, 1], _I[:, 0]))
if ANG[-1] < ANG[0]:
    _I = _I[::-1]
    ANG = np.unwrap(np.arctan2(_I[:, 1], _I[:, 0]))
U = np.stack([np.cos(ANG), np.sin(ANG)], 1)
R_I = np.linalg.norm(_I, axis=1)
R_O = np.array([ray_hit(OUTLINE, u) for u in U])
# the span is measured square to the rings: along the ray it is a little longer where the rim is slanted
_T = np.roll(_I, -1, 0) - np.roll(_I, 1, 0)
_NRM = np.stack([_T[:, 1], -_T[:, 0]], 1)
_NRM /= np.linalg.norm(_NRM, axis=1)[:, None]
if np.mean(np.sum(_NRM * U, 1)) < 0:
    _NRM = -_NRM
COS_I = np.clip(np.sum(_NRM * U, 1), 0.6, 1.0)
R_C = np.minimum(R_I + SPAN / COS_I, R_O - 8.0)
R_P = np.array([ray_hit(rrect(58.5, 40.5, 10.0), u) for u in U])          # front row of the lower tier

SIDE = U[:, 1] ** 2                                   # 1 along the long sides (±z), 0 at the ends (±x)
WEST = np.clip(U[:, 1], 0, 1) ** 2                    # La Peineta
Y_C = 34.0 + 7.0 * SIDE                               # top of the compression ring
Y_EDGE = 12.0 + 26.0 * SIDE ** 1.3                    # rim of the roof
RIDGE = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)    # zig-zag: ridge / valley rays


def Y(y):
    return np.full(N, float(y)) if np.isscalar(y) else np.asarray(y, float)


def loft(prof, col, face, flat=False):
    """Surface through a profile [(radius array, y array), …] swept round all the rays.
    col: one colour, or fn(k, j) -> colour for the quad between rays k, k+1 and profile points j, j+1.
    face: 'up', 'down', 'out' or 'in'. flat: every quad its own vertices (sharp pleats)."""
    P = [np.stack([U[:, 0] * r, Y(y), U[:, 1] * r], 1) for r, y in prof]
    v, f, c = [], [], []
    for j in range(len(P) - 1):
        for k in range(N):
            k1 = (k + 1) % N
            q = [P[j][k], P[j][k1], P[j + 1][k1], P[j + 1][k]]
            cc = col(k, j) if callable(col) else col
            o = len(v)
            v += q
            c += [cc] * 4
            f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    m = Mesh(np.array(v), np.array(f), np.array(c, float))
    vv = m.v
    nn = np.cross(vv[m.f[:, 1]] - vv[m.f[:, 0]], vv[m.f[:, 2]] - vv[m.f[:, 0]])
    if face in ("up", "down"):
        want = 1 if face == "up" else -1
        flip = nn[:, 1] * want < 0
    else:
        cen = (vv[m.f[:, 0]] + vv[m.f[:, 1]] + vv[m.f[:, 2]]) / 3
        s = nn[:, 0] * cen[:, 0] + nn[:, 2] * cen[:, 2]
        flip = s < 0 if face == "out" else s > 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


# ── Roof ─────────────────────────────────────────────────────────────────────────────────────────────────
def roof():
    ms = []
    # top membrane, tension ring -> compression ring: zig-zag pleats (each ray a ridge or a valley cable)
    nseg = 10
    prof = []
    for j in range(nseg + 1):
        t = j / nseg
        r = R_I + t * (R_C - R_I)
        y = Y_IN_TOP + t * (Y_C - Y_IN_TOP) + PLEAT * math.sin(math.pi * t) * RIDGE
        prof.append((r, y))
    ms.append(loft(prof, lambda k, j: WHITE if k % 2 == 0 else WHITE_D, "up", flat=True))
    # cantilever outside the compression ring, curving down to the rim
    nseg = 6
    prof = []
    for j in range(nseg + 1):
        t = j / nseg
        r = R_C + t * (R_O - R_C)
        y = Y_C - (Y_C - Y_EDGE) * t ** 1.7 + PLEAT_OUT * math.sin(math.pi * t) * RIDGE
        prof.append((r, y))
    ms.append(loft(prof, lambda k, j: WHITE if k % 2 == 0 else WHITE_D, "up", flat=True))
    # the red gutter along the rim, and the underside of the cantilever back to the ring
    ms.append(loft([(R_O, Y_EDGE - 1.1), (R_O, Y_EDGE + 0.05)], RED, "out"))
    under = []
    for j in range(6 + 1):
        t = j / 6
        r = R_O - t * (R_O - (R_C - 1.0))
        y = (Y_EDGE - 1.1) + ((Y_C - 4.0) - (Y_EDGE - 1.1)) * (1 - (1 - t) ** 1.7)
        under.append((r, y))
    ms.append(loft(under, lambda k, j: UNDER_OUT if k % 4 else BEAM, "down"))
    # the compression ring truss (its outer face shows under the cantilever)
    ms.append(loft([(R_C - 1.0, Y_C - 4.0), (R_C - 1.0, Y_C - 0.6)], DARK, "out"))
    # the lower PTFE skin, from the compression ring to the foot of the tension ring
    prof = [(R_C - 1.0, Y_C - 4.0)]
    for t in (0.3, 0.6, 0.85):
        prof.append((R_C - 1.0 + t * (R_I - (R_C - 1.0)), (Y_C - 4.0) + t * (Y_IN_BOT - (Y_C - 4.0)) - 1.2 * math.sin(math.pi * t)))
    prof.append((R_I, Y_IN_BOT))
    ms.append(loft(prof, lambda k, j: UNDER if k % 2 == 0 else WHITE_D, "down"))
    # the tension ring: its face to the pitch with the floodlights, a thin top edge
    ms.append(loft([(R_I, Y_IN_BOT), (R_I, Y_IN_BOT + 1.0)], LAMP, "in"))
    ms.append(loft([(R_I, Y_IN_BOT + 1.0), (R_I, Y_IN_TOP)], RING, "in"))
    ms.append(loft([(R_I - 0.6, Y_IN_TOP + 0.4), (R_I + 1.5, Y_IN_TOP + 0.4)], RING, "up"))
    ms.append(loft([(R_I - 0.6, Y_IN_TOP - 0.2), (R_I - 0.6, Y_IN_TOP + 0.4)], RING, "in"))
    return merge(ms)


# ── Outer buildings under the roof ──────────────────────────────────────────────────────────────────────
Y_BAND_TOP = Y_C - 9.0                                  # top of the white concrete band (crest of the upper tier)
Y_BAND_BOT = Y_BAND_TOP - 3.2
R_F = R_O - 13.0                                        # face of the beige blocks, 13 m under the rim
_t = 13.0 / (R_O - R_C + 1.0)
_Y_UNDER_F = (Y_EDGE - 1.1) + ((Y_C - 4.0) - (Y_EDGE - 1.1)) * (1 - (1 - _t) ** 1.7)
Y_BLOCK = np.minimum(16.0 + 9.0 * WEST, _Y_UNDER_F - 3.0)  # the beige blocks: up to 25 m along La Peineta


def outside():
    ms = []
    rb = R_C - 2.0                                       # the band
    ms.append(loft([(rb, Y_BAND_BOT), (rb, Y_BAND_TOP)], CONC, "out"))
    ms.append(loft([(rb, Y_BAND_TOP), (rb - 4.0, Y_BAND_TOP)], CONC, "up"))
    ms.append(loft([(rb, Y_BAND_BOT), (rb - 3.0, Y_BAND_BOT)], CONC, "down"))
    # the raking beams under the upper tier: a dark recess with light beams every 4th ray
    rr = R_F - 1.5
    ms.append(loft([(rr, Y_BLOCK), (rb - 3.0, Y_BAND_BOT)], lambda k, j: BEAM if k % 4 == 0 else DARK, "down"))
    # the beige blocks: a wall with slits, interrupted by dark gaps (stairs and entrances) every ~12 rays
    rf = R_F

    def blk(k, j):
        if k % 12 in (5, 6):
            return SLIT
        if j % 2 == 1:
            return SLIT if k % 3 == 1 else BEIGE_D
        return BEIGE

    prof = [(rf, FOUND), (rf, 0.0)]
    nb = 6
    for j in range(nb):
        y0 = Y_BLOCK * j / nb
        prof += [(rf, y0 + Y_BLOCK / nb * 0.78), (rf, y0 + Y_BLOCK / nb)]
    ms.append(loft(prof, lambda k, j: BEIGE_D if j == 0 else blk(k, j - 1), "out"))
    ms.append(loft([(rf, Y_BLOCK), (rr, Y_BLOCK)], BEIGE_D, "up"))
    return merge(ms)


# ── Stands: three tiers ─────────────────────────────────────────────────────────────────────────────────
def stands():
    ms = []
    back = R_C - 10.0
    H = Y_C - 10.5
    tiers = [(0.00, 0.34, Y(Y_PITCH + 1.0), 0.36 * H, 12),
             (0.37, 0.52, 0.45 * H, 0.62 * H, 6),
             (0.55, 1.00, 0.69 * H, H, 14)]
    prev = None
    for fa, fb, ya, yb, rows in tiers:
        ra, rb = R_P + fa * (back - R_P), R_P + fb * (back - R_P)
        ya, yb = Y(ya), Y(yb)
        if prev is None:
            ms.append(loft([(R_P, Y_PITCH), (R_P, ya)], FASCIA, "in"))
        else:
            pr, py = prev
            ms.append(loft([(pr, py), (ra, py)], SEAT_D, "up"))
            ms.append(loft([(ra, py), (ra, ya - 1.2)], GLAZE, "in"))
            ms.append(loft([(ra, ya - 1.2), (ra, ya)], FASCIA, "in"))
        for k in range(rows):
            r0 = ra + (rb - ra) * k / rows
            r1 = ra + (rb - ra) * (k + 1) / rows
            y0 = ya + (yb - ya) * k / rows
            y1 = ya + (yb - ya) * (k + 1) / rows
            ms.append(loft([(r0, y0), (r0, y1)], SEAT_D, "in"))
            ms.append(loft([(r0, y1), (r1, y1)], lambda kk, j: FASCIA if kk % 16 == 0 else SEAT, "up"))
        prev = (rb, yb)
    pr, py = prev
    ms.append(loft([(pr, py), (pr, Y_C - 4.0)], DARK, "in"))
    return merge(ms)


# ── Pitch ─────────────────────────────────────────────────────────────────────────────────────────────
def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def pitch():
    ring = [(float(U[k, 0] * R_P[k]), float(U[k, 1] * R_P[k])) for k in range(N)]
    ms = [cap(ring, Y_PITCH, RUNOFF)]
    L, W = 105.0, 68.0
    nb = 18
    for k in range(nb):
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
        for d, half in ((16.5, 20.16), (5.5, 9.16)):
            line(sx * hx, -half, sx * (hx - d), -half)
            line(sx * hx, half, sx * (hx - d), half)
            line(sx * (hx - d), -half, sx * (hx - d), half)
        gx = sx * (hx + 0.06)
        for z in (-3.66, 3.66):
            ms.append(B(gx - 0.06, gx + 0.06, Y_PITCH, Y_PITCH + 2.44, z - 0.06, z + 0.06, LINE))
        ms.append(B(gx - 0.06, gx + 0.06, Y_PITCH + 2.32, Y_PITCH + 2.44, -3.66, 3.66, LINE))
    seg = 48
    for k in range(seg):
        a0, a1 = 2 * math.pi * k / seg, 2 * math.pi * (k + 1) / seg
        r_in, r_out = 9.15 - w / 2, 9.15 + w / 2
        P = [(r_in * math.cos(a0), r_in * math.sin(a0)), (r_out * math.cos(a0), r_out * math.sin(a0)),
             (r_out * math.cos(a1), r_out * math.sin(a1)), (r_in * math.cos(a1), r_in * math.sin(a1))]
        ms.append(cap(P, y1, LINE))
    return merge(ms)


def main():
    out = ROOT / "models" / "Q1369687"
    out.mkdir(parents=True, exist_ok=True)
    r = roof()
    body = merge([outside(), stands()])
    field = pitch()
    parts = [("roof", r, {"material": "membrane"}), ("stands_and_buildings", body), ("pitch", field)]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.9,
                           materials={"membrane": {"metallic": 0.0, "roughness": 0.6}})
    allv = np.vstack([r.v, body.v, field.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    print("compression ring - outline (min/mean/max):", np.round([np.min(R_O - R_C), np.mean(R_O - R_C), np.max(R_O - R_C)], 1))


if __name__ == "__main__":
    main()
