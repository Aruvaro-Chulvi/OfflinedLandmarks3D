"""
Puente romano de Mérida (Q588535), over the Guadiana — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: the Roman bridge of Augusta Emerita (late 1st century BC), 60 granite arches, rebuilt many times in
its middle (Visigothic repairs in 483, the central stretch of 1603–1611, 19th-century repairs after 1812, 1823,
1860). Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap ways 25024396, 1075066789 and 391413815 (the footway on the bridge, 783.7 m from end to end, with
  29 nodes tagged bridge:support=pier) and way 770843480 (man_made=bridge: the deck 7.1 m wide, the two round
  balconies over the pier at x = −157, the descendedero del Humilladero down to the island at x = −258 and the
  pointed descendedero de San Antonio at x = +113, both on the upstream side). The axis runs at 51.45° (SW–NE);
  the downstream face (north-west) is the main face, +Z (321.45°). +X runs to the south-west (the Barrio side), so
  the city (Alcazaba) end is at x = −392. The axis bends by about 0.3° at San Antonio (OSM), no more.
- Wikipedia (es): 790 m, 60 arches, 12 m above the mean water; three stretches: from the city to the Humilladero
  the Roman one (square piers with semicircular cutwaters upstream up to the springing, relief arches over the
  piers, rusticated ashlar), the middle one rebuilt in 1611 (five larger arches, rectangular piers with sharp
  cutwaters with pyramidal caps upstream and buttresses downstream) and repaired in the 19th century, and from San
  Antonio to the end the Roman stretch over the dry bank, without relief arches, the deck falling towards the end.
- OSM pier spacing: the five arches of 1611 (bays of 18–21 m), the rest 11–16 m; two gaps of 27 and 33 m split in
  two; the 29 arches of the south-western stretch (no piers in OSM) spaced evenly (10.7 m) so the bridge has 60.
- IGN MDT05 (terrain.json): the water at 202.0 m (y = 0), the city end at 209–211 m, the south-western bank at
  204–206 m, the end at 207 m. Heights of the piers and arches by proportion on the Commons photos (±10 %).

Axes (docs/MODEL_SPEC.md): +Y up, y = 0 the water (202.0 m); the bridge along X; origin on the axis at the middle
of the footway (783.7 m / 2).

    python model_sources/Q588535_puente_romano_de_merida.py   -> models/Q588535/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, triangulate, fbm, noise3, normals, write_glb  # noqa: E402

GRANITE = np.array((0.73, 0.63, 0.50))     # rusticated granite ashlar, warm grey-brown
GRANITE_M = np.array((0.71, 0.65, 0.56))   # the 1611 and 19th-century work, a little greyer
PAVING = np.array((0.58, 0.56, 0.52))      # granite setts of the footway
COPING = np.array((0.72, 0.68, 0.60))
DAMP = np.array((0.36, 0.33, 0.27))        # the waterline

W = 7.1                                    # width of the deck between the outer faces (OSM)
HW = W / 2
X_NE, X_SW = -391.8, 391.8                 # the ends of the bridge (OSM footway)
E0 = 202.0                                 # the water, m a.s.l. = y 0

# piers from OSM (bridge:support=pier on the footway), x along the axis
PIERS_OSM = [-370.72, -356.81, -341.78, -326.1, -310.58, -295.45, -283.35, -255.92, -235.0, -214.81, -195.44,
             -175.89, -157.73, -124.37, -111.85, -99.43, -87.1, -74.4, -59.2, -43.14, -29.12, -14.85, -1.0, 11.3,
             23.91, 36.38, 47.68, 62.32, 74.52]
X_HUM, X_SANT, X_BALC = -258.6, 113.2, -157.2      # Humilladero, San Antonio, the round balconies
NE_FACE, SW_FACE = -384.6, 385.0                   # the abutments' faces

# lowest ground across the bridge (MDT05, z −15..+15), metres above the water
GROUND = [(-440, 9.6), (-420, 7.9), (-410, 7.1), (-400, 5.4), (-390, 2.6), (-380, 1.0), (-370, 0.0), (80, 0.0),
          (90, 1.0), (115, 1.0), (125, 1.7), (130, 2.0), (240, 2.0), (250, 1.7), (260, 1.2), (275, 1.1), (290, 1.4),
          (300, 1.7), (310, 2.1), (320, 2.6), (330, 2.7), (345, 2.3), (360, 2.3), (370, 3.2), (380, 4.1),
          (390, 4.9), (400, 5.0), (420, 6.2)]
# the deck: from the city end up to a gentle hump over the river, then falling over the south-western bank
DECK = [(-392, 8.8), (-370, 9.6), (-300, 10.8), (-250, 11.4), (-150, 11.8), (0, 12.0), (100, 11.8), (180, 10.9),
        (260, 9.4), (330, 7.6), (392, 5.3)]
# the axis (OSM way 770843480: the middle of its two sides)
ZC = [(-392, 1.6), (112, 0.65), (392, 1.5)]


def ground(x):
    xs, ys = zip(*GROUND)
    return float(np.interp(x, xs, ys))


def deck(x):
    xs, ys = zip(*DECK)
    return float(np.interp(x, xs, ys))


def zc(x):
    xs, zs = zip(*ZC)
    return np.interp(x, xs, zs)


def found(x):
    return min(ground(x), 0.0) - 3.0 if ground(x) < 0.5 else ground(x) - 3.0


def section(x):
    if x < X_HUM - 3:
        return "ne"            # Roman, from the city to the Humilladero
    if x <= X_SANT:
        return "mid"           # 1611 and the 19th-century repairs
    return "sw"                # Roman, over the dry bank


# ── Mesh helpers ─────────────────────────────────────────────────────────────
def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def prism_xy(poly, z0, z1, c):
    """A prism from an outline drawn in the (x, y) plane (an elevation), from z0 to z1."""
    pts, tris = triangulate([(x, y) for x, y in poly])
    n = len(pts)
    ring = [tuple(p) for p in pts]
    area = sum(ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1] for i in range(n))
    side = []
    for i in range(n):
        j = (i + 1) % n
        (xa, ya), (xb, yb) = ring[i], ring[j]
        q = [(xa, ya, z0), (xb, yb, z0), (xb, yb, z1), (xa, ya, z1)]
        f = [(0, 1, 2), (0, 2, 3)] if area > 0 else [(0, 2, 1), (0, 3, 2)]
        side.append(Mesh(q, f, c))
    cap0 = Mesh([(x, y, z0) for x, y in pts], tris if area > 0 else tris[:, ::-1], c)
    cap1 = Mesh([(x, y, z1) for x, y in pts], tris[:, ::-1] if area > 0 else tris, c)
    return merge(side + [cap0, cap1])


def prism_zy(poly, x0, x1, c):
    """The same with the outline in the (z, y) plane, extruded along x."""
    m = prism_xy(poly, x0, x1, c)
    m.v = m.v[:, [2, 1, 0]]
    m.f = m.f[:, ::-1]
    return m


def half_round(cx, zf, r, y0, y1, c, side=-1, cap=0.0, seg=10):
    """Half a cylinder (a semicircular cutwater, a balcony) standing out of the face z = zf towards side,
    from y0 to y1, with a half-cone cap of height cap (or a flat top)."""
    a = np.linspace(0, math.pi, seg + 1)
    px, pz = cx + r * np.cos(a), zf + side * r * np.sin(a)
    v, f = [], []
    for i in range(seg):
        o = len(v)
        v += [(px[i], y0, pz[i]), (px[i + 1], y0, pz[i + 1]), (px[i + 1], y1, pz[i + 1]), (px[i], y1, pz[i])]
        f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    m = Mesh(v, f, c)
    ap = (cx, y1 + cap, zf)
    v2, f2 = [], []
    for i in range(seg):
        o = len(v2)
        v2 += [(px[i], y1, pz[i]), (px[i + 1], y1, pz[i + 1]), ap]
        f2 += [(o, o + 2, o + 1)]
    m2 = Mesh(v2, f2, c)
    out = merge([m, m2])
    # wind outwards
    n = normals(out)
    d = out.v - np.array([cx, (y0 + y1) / 2, zf])
    if float((n * d).sum()) < 0:
        out.f = out.f[:, ::-1]
    return out


def pointed(cx, zf, w, y0, y1, c, side=-1, nose=None, cap=1.6):
    """A sharp cutwater w wide standing out of the face z = zf, its nose `nose` out, a pyramid cap on top."""
    nose = w * 0.75 if nose is None else nose
    tri = [(cx - w / 2, zf), (cx, zf + side * nose), (cx + w / 2, zf)]
    v, f = [], []
    for (xa, za), (xb, zb) in zip(tri[:-1], tri[1:]):
        o = len(v)
        v += [(xa, y0, za), (xb, y0, zb), (xb, y1, zb), (xa, y1, za)]
        f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    ap = (cx, y1 + cap, zf)
    for (xa, za), (xb, zb) in zip(tri[:-1], tri[1:]):
        o = len(v)
        v += [(xa, y1, za), (xb, y1, zb), ap]
        f += [(o, o + 2, o + 1)]
    m = Mesh(v, f, c)
    n = normals(m)
    d = m.v - np.array([cx, (y0 + y1) / 2, zf])
    if float((n * d).sum()) < 0:
        m.f = m.f[:, ::-1]
    return m


def stone(base, seed, course=0.62, amp=0.22):
    base = np.asarray(base, float)

    def fn(v):
        n1 = fbm(v, 0.12, seed, 3) - 0.5
        n2 = noise3(v * np.array([1.0, 1.6, 1.0]), 0.9, seed + 5) - 0.5            # blocks
        row = np.where(np.floor(v[:, 1] / course + 1e-6).astype(int) % 2 == 0, 1.03, 0.96)
        k = (1 + amp * n1 + 0.10 * n2) * row
        c = np.outer(k, base)
        wet = np.clip((1.3 - v[:, 1]) / 1.8, 0, 1)[:, None] * 0.8                      # damp at the waterline
        return np.clip(c * (1 - wet) + DAMP * wet, 0, 1)
    return fn


def paint(m, base, seed):
    return m.paint(stone(base, seed))


# ── The arches ───────────────────────────────────────────────────────────────
def pier_list():
    """[(x, half width, kind)] from the city end to the far end, abutments included."""
    xs = list(PIERS_OSM)
    for a, b in ((-283.35, -255.92), (-157.73, -124.37)):              # gaps of 27 and 33 m: one pier each
        xs.append((a + b) / 2)
    n_sw = 29
    bay = (SW_FACE - 74.52) / n_sw
    xs += [74.52 + bay * k for k in range(1, n_sw)]
    xs.sort()
    out = [(X_NE, NE_FACE - X_NE, "abut")]
    for x in xs:
        s = section(x)
        if s == "ne":
            hw = 2.1
        elif s == "sw":
            hw = 1.8
        elif -256.5 <= x <= -157.0:
            hw = 2.5                                                       # the big piers of 1611
        else:
            hw = 2.0
        out.append((x, hw, s))
    out.append((X_SW, X_SW - SW_FACE, "abut"))
    return out


def semi(xa, xb, ys, n):
    xc, r = (xa + xb) / 2, (xb - xa) / 2
    return [(xc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(math.pi, 0, n + 1)]


def bridge():
    P = pier_list()
    narch = len(P) - 1
    spring, rel = [], {}
    for i in range(narch):
        (xa, ha, _), (xb, hb, _) = P[i], P[i + 1]
        fa, fb = xa + ha, xb - hb
        r = (fb - fa) / 2
        spring.append(deck((fa + fb) / 2) - 1.7 - r)
    # relief arches over the piers (not over the dry bank, not at the abutments, not at the balcony pier)
    for i in range(1, len(P) - 1):
        x, hw, kind = P[i]
        if kind == "sw" or abs(x - X_BALC) < 2:
            continue
        a = min(1.1, 0.5 * hw)
        yb = max(spring[i - 1], spring[i]) + 0.8
        yc = min(deck(x) - 2.0, yb + 2 * a + 2.0)
        if yc - a - yb >= 0.8:
            rel[i] = (a, yb, yc - a)

    parts = []
    for i in range(narch):
        (xa, ha, ka), (xb, hb, kb) = P[i], P[i + 1]
        ys = spring[i]
        fa, fb = xa + ha, xb - hb
        n = int(np.clip((fb - fa) * 1.1, 10, 18))
        poly = [(xa, deck(xa))]
        if i in rel:
            a, yb, y0 = rel[i]
            poly += [(xa + a * math.sin(t), y0 + a * math.cos(t)) for t in np.linspace(0, math.pi / 2, 5)]
            poly += [(xa + a, yb), (xa, yb)]
        poly += [(xa, ys)]
        poly += semi(fa, fb, ys, n)
        poly += [(xb, ys)]
        if i + 1 in rel:
            a, yb, y0 = rel[i + 1]
            poly += [(xb, yb), (xb - a, yb)]
            poly += [(xb - a * math.sin(t), y0 + a * math.cos(t)) for t in np.linspace(math.pi / 2, 0, 5)]
        poly += [(xb, deck(xb))]
        sec = section((xa + xb) / 2)
        col = GRANITE_M if sec == "mid" else GRANITE
        m = prism_xy(poly, -HW, HW, col)
        paint(m, col, 3 + i % 7)
        nn = normals(m)
        up = (nn[:, 1] > 0.8) & (m.v[:, 1] > ys + 1.0)
        m.c[up] = np.clip(np.outer(1 + 0.1 * (fbm(m.v[up], 0.3, 9, 2) - 0.5), PAVING), 0, 1)
        parts.append(m)
        # the voussoirs, standing a little proud of both faces
        ring_o = semi(fa - 0.85, fb + 0.85, ys, n)
        ring_i = semi(fa, fb, ys, n)[::-1]
        parts.append(paint(prism_xy(ring_o + ring_i, -HW - 0.1, HW + 0.1, col * 1.04), col * 1.04, 11))

    # piers: shaft, footing, cutwater upstream (−Z), buttress downstream (+Z) in the middle stretch
    for i, (x, hw, kind) in enumerate(P):
        yf = found(x)
        if kind == "abut":
            x0, x1 = sorted((x, x + (hw if x < 0 else -hw)))
            top = spring[0] if x < 0 else spring[-1]
            parts.append(paint(B(x0, x1, yf, top, -HW, HW, GRANITE), GRANITE, 21))
            continue
        top = max(spring[i - 1], spring[i])
        low = min(spring[i - 1], spring[i])
        col = GRANITE_M if kind == "mid" else GRANITE
        parts.append(paint(B(x - hw, x + hw, yf, top, -HW, HW, col), col, 22))
        if ground(x) < 1.5:                                                   # stepped footing in the water
            parts.append(paint(B(x - hw - 0.45, x + hw + 0.45, yf, 0.9, -HW - 0.45, HW + 0.45, col * 0.97), col, 23))
        if abs(x - X_BALC) < 2:
            continue
        if kind == "mid":
            parts.append(paint(pointed(x, -HW, 2 * hw, yf, low, col, -1, nose=1.5 * hw, cap=2.2), col, 24))
            bt = [(HW, yf), (HW + 1.5, yf), (HW + 1.5, low - 1.6), (HW, low - 0.2)]
            parts.append(paint(prism_zy(bt, x - hw + 0.5, x + hw - 0.5, col), col, 25))
        else:
            parts.append(paint(half_round(x, -HW, hw, yf, low, col, -1, cap=hw * 0.8), col, 26))

    # cornice band and parapets with their coping, along both faces
    xs = np.array([p[0] for p in P])
    for s in (-1, 1):
        for xa, xb in zip(xs[:-1], xs[1:]):
            band = [(xa, deck(xa) - 0.45), (xb, deck(xb) - 0.45), (xb, deck(xb) - 0.1), (xa, deck(xa) - 0.1)]
            parts.append(prism_xy(band, *sorted((s * (HW - 0.2), s * (HW + 0.18))), GRANITE * 1.05))
            wall = [(xa, deck(xa) - 0.05), (xb, deck(xb) - 0.05), (xb, deck(xb) + 0.85), (xa, deck(xa) + 0.85)]
            parts.append(paint(prism_xy(wall, *sorted((s * (HW - 0.6), s * HW)), GRANITE * 1.02), GRANITE * 1.02, 30))
            cope = [(xa, deck(xa) + 0.85), (xb, deck(xb) + 0.85), (xb, deck(xb) + 1.0), (xa, deck(xa) + 1.0)]
            parts.append(prism_xy(cope, *sorted((s * (HW - 0.65), s * (HW + 0.05))), COPING))
    body = merge(parts)
    body.v[:, 2] += zc(body.v[:, 0])                                          # the axis of OSM (bends 0.3°)
    return body


# ── Balconies and descendederos (built in the model's own axes) ──────────────
def balconies():
    parts = []
    x = X_BALC
    z0 = float(zc(x))
    yd = deck(x)
    yf = found(x)
    for s in (-1, 1):
        zf = z0 + s * HW
        parts.append(paint(half_round(x, zf, 3.6, yf, yd, GRANITE_M, s, cap=0.0, seg=14), GRANITE_M, 40 + s))
        # floor and parapet of the balcony
        a = np.linspace(0, math.pi, 15)
        ring_o = [(x + 3.6 * math.cos(t), zf + s * 3.6 * math.sin(t)) for t in a]
        for t0, t1 in zip(a[:-1], a[1:]):
            tm = (t0 + t1) / 2
            L = 3.3 * (t1 - t0) + 0.05
            seg = box(L, 1.0, 0.6, GRANITE_M * 1.02).R(0, -math.atan2(s * math.cos(tm), -math.sin(tm)) + 0, 0)
            seg.T(x + 3.3 * math.cos(tm), yd - 0.05, zf + s * 3.3 * math.sin(tm))
            parts.append(paint(seg, GRANITE_M * 1.02, 42))
        disc_pts = [(px, pz) for px, pz in ring_o]
        pts, tris = triangulate(disc_pts)
        parts.append(Mesh(np.stack([pts[:, 0], np.full(len(pts), yd - 0.02), pts[:, 1]], 1), tris[:, ::-1], PAVING))
    return parts


def ramp(xa, xb, z_root, z_end, y_root, y_end, nose=0.0, col=GRANITE_M):
    """A descendedero on the upstream side: a walled ramp from the deck (z_root) down to z_end, with parapets;
    with nose > 0 it ends in a sharp point like a cutwater."""
    parts = []
    xm = (xa + xb) / 2
    yf = min(found(xm), -3.0 if ground(xm) < 0.5 else ground(xm) - 3.0)
    prof = [(z_root, yf), (z_root, y_root), (z_end, y_end), (z_end, yf)]
    parts.append(paint(prism_zy(prof, xa, xb, col), col, 50))
    if nose > 0:
        parts.append(paint(pointed(xm, z_end, xb - xa, yf, y_end, col, -1, nose=nose, cap=0.0), col, 51))
    # surface of the ramp
    for s, xw in ((-1, xa), (1, xb)):
        x0, x1 = sorted((xw, xw - s * 0.55))
        wall = [(z_root, y_root - 0.05), (z_end, y_end - 0.05), (z_end, y_end + 0.9), (z_root, y_root + 0.9)]
        parts.append(paint(prism_zy(wall, x0, x1, col * 1.03), col * 1.03, 52))
    m = merge(parts)
    nn = normals(m)
    up = (nn[:, 1] > 0.7) & (m.v[:, 1] > min(y_root, y_end) - 0.2) & (m.v[:, 0] > xa + 0.5) & (m.v[:, 0] < xb - 0.5)
    m.c[up] = PAVING
    return m


def descendederos():
    out = []
    # Humilladero: 47 m down to the island (OSM x −262.1…−255.8, out to z −49)
    z0 = float(zc(X_HUM)) - HW + 0.3
    out.append(ramp(-262.0, -255.4, z0, -48.5, deck(X_HUM) - 0.05, 4.4))
    # San Antonio: 6 m wide, 19 m of ramp, then a sharp end out to z −28 (OSM)
    z0 = float(zc(X_SANT)) - HW + 0.3
    out.append(ramp(110.3, 116.3, z0, -22.0, deck(X_SANT) - 0.05, 4.6, nose=6.0))
    return out


def main():
    out = ROOT / "models" / "Q588535"
    out.mkdir(parents=True, exist_ok=True)
    m = merge([bridge()] + balconies() + descendederos())
    tris, size = write_glb(out / "model.glb", [("puente_romano_de_merida", m)], roughness=0.9)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
