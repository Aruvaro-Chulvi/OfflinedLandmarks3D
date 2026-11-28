"""
Circo romano de Mérida (Q1984155) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: the circus of Augusta Emerita, built outside the walls in the early 1st century AD (time of Tiberius),
restored under the sons of Constantine (337–340), in use until the 6th century; about 440 × 115 m, for some 30,000
spectators. What is left are its foundations: the track, the spina, the low remains of the stands, the carceres.
Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap: the site (way 152837480, 436 × 119 m), the wall of the track (ways 260581680 north and 260581678
  south), the two halves of the spina (ways 260581757 and 260581758, 104 m each, 8 m wide, a gap of 13 m between them)
  and the western meta (way 260581753), the block of the tribune on the north side (ways 260581677, 260581755), the
  outer wall of the southern stands with its buttresses (ways 260581675…261) and the short paired walls of the
  passages in the eastern curve (ways 260581739…750). All coordinates below are those ways in the model's axes.
- The heading follows the spina (267.8°): the façade with the carceres and the porta pompae faces west (+Z).
- The eastern meta is not in OSM: placed from the PNOA orthophoto, symmetric to the western one.
- Heights from photos on Wikimedia Commons: the spina a low platform about 0.8 m high, the wall of the track 1–2 m,
  a corridor behind it and the rubble core of the stands 2–3 m, the carceres only foundations.
- The ground of the stands follows the IGN MDT05 (terrain.json): the circus lies on a gentle slope that rises
  2–5 m to the south and to the eastern curve, where the stands were laid on the hillside.

Axes (docs/MODEL_SPEC.md): +Y up, the west end (carceres) faces +Z, +X to the south; origin on the sand of the track
in the gap in the middle of the spina.

    python model_sources/Q1984155_circo_romano_de_merida.py   -> models/Q1984155/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, triangulate, cap, extrude, offset, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(337)

SAND = (0.74, 0.64, 0.47)          # the track, dry grass on sand
EARTH = np.array((0.62, 0.50, 0.38))
RUBBLE = np.array((0.60, 0.54, 0.45))
GRANITE = np.array((0.66, 0.62, 0.55))
SPINA_TOP = (0.58, 0.57, 0.53)
FOUND = -2.5
WEST = 238.0                        # beyond this z the west end: carceres, no stands

OSM = {
    152837480: [(-39.26, 251.85), (-23.88, 261.77), (3.68, 263.16), (22.23, 262.81), (28.15, 259.98), (61.00, 244.27), (66.56, 229.37), (63.82, 193.65), (64.38, 160.97), (63.25, 96.40), (51.24, -136.64), (21.70, -164.31), (-2.09, -172.67), (-19.03, -166.35), (-43.26, -150.98), (-58.18, -124.55), (-60.44, -102.29), (-57.38, 25.92), (-56.67, 55.75), (-53.15, 202.93), (-52.74, 221.14), (-51.09, 230.32), (-48.65, 238.65), (-44.95, 246.28), (-39.26, 251.85)],
    260581680: [(39.41, 245.68), (31.27, 246.95), (23.12, 247.06), (12.83, 247.10), (3.39, 246.95), (-10.16, 245.36), (-22.70, 242.51), (-32.95, 238.34), (-37.86, 235.03), (-40.33, 229.55), (-41.91, 199.79), (-43.89, 139.59), (-44.00, 133.84), (-44.21, 123.41), (-44.31, 118.42), (-44.41, 113.51), (-44.39, 108.66), (-44.43, 102.85), (-45.01, 27.12), (-45.71, -116.65), (-44.38, -123.44), (-39.84, -131.19), (-33.35, -138.55), (-26.62, -143.93), (-21.19, -147.07)],
    260581678: [(39.41, 245.68), (46.71, 241.72), (52.12, 234.92), (54.06, 227.85), (52.11, 133.95), (48.91, 6.70), (45.63, -109.14), (42.64, -122.04), (37.95, -130.29), (32.88, -136.58), (24.81, -143.44), (16.64, -148.02), (6.45, -151.00)],
    260581757: [(-3.81, -7.76), (-4.29, -109.95), (-1.62, -108.17), (0.50, -108.09), (3.61, -110.26), (4.29, -5.62), (2.25, -7.68), (-1.37, -8.27), (-3.81, -7.76)],
    260581758: [(-4.37, 110.50), (-3.91, 6.70), (-1.07, 7.57), (1.21, 7.66), (4.00, 5.94), (3.90, 111.00), (1.54, 109.54), (-1.03, 109.29), (-4.37, 110.50)],
    260581753: [(-4.23, 117.11), (3.99, 116.62), (4.21, 119.76), (2.75, 121.57), (0.29, 122.51), (-2.23, 121.95), (-3.79, 120.73), (-4.23, 117.11)],
    260581677: [(-43.89, 139.59), (-44.00, 133.84), (-44.11, 128.63), (-44.21, 123.41), (-44.31, 118.42), (-44.41, 113.51), (-44.39, 108.66), (-44.43, 102.85), (-50.68, 102.98), (-53.50, 103.04), (-52.77, 139.77), (-49.79, 139.71), (-43.89, 139.59)],
    260581755: [(-49.79, 139.71), (-49.93, 134.07), (-50.06, 128.74), (-50.19, 123.24), (-50.31, 118.47), (-50.42, 113.54), (-50.55, 108.42), (-50.68, 102.98)],
    260581675: [(60.51, 227.01), (60.48, 225.39), (60.41, 222.42), (60.35, 219.41), (60.29, 216.44), (60.21, 213.08), (60.17, 211.22), (60.14, 210.05), (60.07, 206.75), (59.99, 203.10)],
    260581676: [(59.89, 189.29), (59.84, 187.95), (59.77, 185.46), (59.73, 184.35), (59.64, 181.68), (59.60, 180.36), (59.54, 178.37), (59.45, 175.67), (59.34, 172.04), (59.25, 169.20), (59.18, 167.17)],
    260581679: [(55.63, 97.07), (58.19, 97.18), (58.05, 93.66), (57.98, 91.78), (57.91, 90.11), (57.82, 87.71), (57.72, 85.41), (57.62, 82.62), (57.36, 76.09), (57.27, 73.55), (57.23, 72.76), (57.17, 71.38), (57.08, 68.92), (56.88, 63.93), (56.86, 63.50), (56.73, 60.01), (59.43, 59.69)],
    260581754: [(59.07, 152.31), (59.02, 150.27), (58.98, 148.72), (58.93, 147.01), (58.84, 143.88), (58.82, 143.07), (58.76, 140.81), (58.71, 139.13)],
    260581760: [(58.79, 135.77), (58.76, 134.71), (58.72, 132.37), (58.70, 131.40), (58.64, 128.59), (58.61, 126.77), (58.57, 125.43), (58.51, 122.06), (58.49, 120.97)],
    260581761: [(58.64, 116.98), (58.62, 116.10), (58.53, 112.20), (58.48, 110.25), (58.46, 109.33), (58.38, 106.35), (58.31, 103.25), (58.24, 100.32), (55.42, 100.27)],
    260581751: [(61.59, 187.04), (61.55, 185.36), (61.42, 180.35), (61.33, 177.23)],
    260581752: [(59.17, 162.86), (59.14, 159.64), (59.11, 156.04), (56.48, 156.19)],
    260581737: [(57.30, 206.88), (60.07, 206.75), (62.39, 206.73)],
    260581741: [(55.47, 106.35), (58.38, 106.35), (61.25, 106.40)],
    260581743: [(57.43, 169.13), (59.25, 169.20), (61.18, 169.22)],
    260581739: [(-18.76, -159.61), (-15.40, -160.61), (-11.78, -161.50)],
    260581740: [(-19.53, -160.93), (-16.04, -162.26), (-11.72, -163.06)],
    260581741: [(55.47, 106.35), (58.38, 106.35), (61.25, 106.40)],
    260581742: [(-34.32, -148.31), (-31.81, -150.75), (-29.39, -152.43)],
    260581743: [(57.43, 169.13), (59.25, 169.20), (61.18, 169.22)],
    260581744: [(-56.14, -47.08), (-56.72, -63.55), (-54.39, -64.82)],
    260581745: [(-28.26, -156.85), (-24.55, -158.65), (-21.55, -159.77)],
    260581746: [(-27.61, -155.48), (-24.28, -157.24), (-21.00, -158.62)],
    260581747: [(-37.07, -150.63), (-34.31, -153.00), (-30.78, -155.40)],
    260581748: [(-35.82, -149.61), (-33.22, -151.88), (-30.20, -153.86)],
    260581749: [(-17.78, -157.42), (-14.25, -158.58), (-11.44, -159.23)],
    260581750: [(-26.44, -153.94), (-23.64, -155.49), (-20.65, -156.62)],
}


def dens(pts, step=3.0, closed=False):
    P = [tuple(p) for p in pts]
    if closed and P[0] == P[-1]:
        P = P[:-1]
    out = []
    n = len(P)
    rng_ = range(n) if closed else range(n - 1)
    for i in rng_:
        a, b = np.array(P[i]), np.array(P[(i + 1) % n])
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        for t in range(k):
            out.append(tuple(a + (b - a) * t / k))
    if not closed:
        out.append(P[-1])
    return out


def ccol(base, k=0.05):
    return tuple(np.clip(np.asarray(base) * (1 + rng.normal(0, k)), 0, 1))


def seg_wall(a, b, t, y0, y1, c, side=0.0):
    """A wall t thick along the segment a-b (x/z), shifted sideways by side (left of a->b > 0)."""
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    ang = math.atan2(d[1], d[0])
    m = box(L + 0.2, y1 - y0, t, c).T(0, y0, side)
    m.R(0, -ang, 0)
    mid = (a + b) / 2
    return m.T(mid[0], 0, mid[1])


def ruin_wall(pts, t, ylo, top, c, side=0.0, step=3.0):
    """A ruined wall along a polyline: each stretch of about 3 m a little higher or lower; top(x, z) -> height."""
    P = dens(pts, step)
    out = []
    for a, b in zip(P[:-1], P[1:]):
        x, z = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        h = top(x, z) + rng.uniform(-0.35, 0.35)
        out.append(seg_wall(a, b, t, ylo, h, ccol(c, 0.06), side))
    return out


# ── the ground: MDT05 around the model ───────────────────────────────────────
def terrain_fn():
    t = json.loads((ROOT / "models" / "Q1984155" / "terrain.json").read_text("utf-8"))
    meta = json.loads((ROOT / "models" / "Q1984155" / "meta.json").read_text("utf-8"))
    n = t["n"]
    import base64
    q = np.frombuffer(base64.b64decode(t["h"]), "<i2").reshape(n, n).astype(float)
    H = np.where(q == -32768, np.nan, t["off"] + q / 10)
    h = math.radians(meta["heading"]); c, s = math.cos(h), math.sin(h)

    def at(x, z):
        e, nn = -x * c + z * s, x * s + z * c
        fi = (e - t["x0"]) / t["c"]; fj = (t["y0"] - nn) / t["c"]
        i0, j0 = int(fi), int(fj); tx, tz = fi - i0, fj - j0
        return float(H[j0, i0] * (1 - tx) * (1 - tz) + H[j0, i0 + 1] * tx * (1 - tz)
                     + H[j0 + 1, i0] * (1 - tx) * tz + H[j0 + 1, i0 + 1] * tx * tz)
    T0 = at(0.0, 0.0)
    return lambda x, z: at(x, z) - T0, T0


def height_mesh(outer, holes, hfn, colfn):
    """A small surface triangulated over outer (with holes) whose vertices take the height hfn(x, z)."""
    pts, tris = triangulate(outer, holes)
    V = np.array([(x, hfn(x, z), z) for x, z in pts])
    m = Mesh(V, tris, (0.5, 0.5, 0.5))
    return up(m, colfn)


def up(m, colfn):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    flip = np.cross(b - a, c - a)[:, 1] < 0
    m.f[flip] = m.f[flip][:, ::-1]
    m.c = colfn(m.v)
    return m


def fan(ring, centre, hfn, colfn):
    """A star-shaped surface: a fan of triangles from centre to the ring."""
    V = [(centre[0], hfn(*centre), centre[1])] + [(x, hfn(x, z), z) for x, z in ring]
    n = len(ring)
    F = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    return up(Mesh(V, F), colfn)


def strip(r0, r1, h0, h1, colfn):
    """Quads between two rings with the same number of points (r0[i] faces r1[i])."""
    n = len(r0)
    V = [(x, h0(x, z), z) for x, z in r0] + [(x, h1(x, z), z) for x, z in r1]
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [(i, j, n + j), (i, n + j, n + i)]
    return up(Mesh(V, F), colfn)


def cast(ring, poly):
    """For each point of ring, the point where the outward normal meets the closed polyline poly."""
    R = np.array(ring); P = np.array(poly)
    n = len(R)
    out = []
    A, Bq = P, np.roll(P, -1, axis=0)
    for i in range(n):
        t = R[(i + 1) % n] - R[i - 1]
        nr = np.array((t[1], -t[0])); nr /= np.linalg.norm(nr)
        if ring_area_xz(ring) < 0:
            nr = -nr
        o = R[i]
        e = Bq - A
        den = nr[0] * e[:, 1] - nr[1] * e[:, 0]
        w = A - o
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / den
            uu = (w[:, 0] * nr[1] - w[:, 1] * nr[0]) / den
        ok = (np.abs(den) > 1e-9) & (tt > 0) & (uu >= 0) & (uu <= 1)
        k = np.argmin(np.where(ok, tt, np.inf))
        out.append(tuple(o + nr * tt[k]))
    return out


def ring_area_xz(P):
    return 0.5 * sum(P[i - 1][0] * P[i][1] - P[i][0] * P[i - 1][1] for i in range(len(P)))


def skirt(ring, hfn, y0, c):
    """Vertical faces along a closed ring from y0 up to hfn(x, z) (the outer edge of the stands)."""
    V, F, C = [], [], []
    n = len(ring)
    for i in range(n):
        (xa, za), (xb, zb) = ring[i], ring[(i + 1) % n]
        o = len(V)
        V += [(xa, y0, za), (xb, y0, zb), (xb, hfn(xb, zb), zb), (xa, hfn(xa, za), za)]
        F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        k = ccol(c, 0.06)
        C += [k] * 4
    return Mesh(V, F, np.array(C))


def circus():
    G, T0 = terrain_fn()
    parts = []
    north, south = OSM[260581680], OSM[260581678]
    ringA = dens(north + south[::-1][:-1], 3.0, closed=True)          # the track
    ringB = offset(ringA, 4.7)                                         # the inner wall of the stands
    outline = dens(OSM[152837480], 3.0, closed=True)

    # the track
    sand = lambda v: np.tile(SAND, (len(v), 1))  # noqa: E731  (long fan triangles: one colour)
    # nested rings shrinking towards the axis of the track (x = 3.5, z −105…+200), so no triangle is long
    def towards_axis(P, k):
        out = []
        for x, z in P:
            cz = min(max(z, -105.0), 200.0)
            out.append((3.5 + (x - 3.5) * k, cz + (z - cz) * k))
        return out
    rings = [towards_axis(ringA, 1.0 - i / 8) for i in range(9)]
    for r0, r1 in zip(rings[:-1], rings[1:]):
        parts.append(strip(r0, r1, lambda x, z: 0.03, lambda x, z: 0.03, sand))
    # the corridor between the wall of the track and the stands (at the west end: the floor of the carceres)
    parts.append(strip(ringA, ringB, lambda x, z: 0.05, lambda x, z: 0.05,
                       lambda v: np.tile(EARTH * 1.05, (len(v), 1)) * (0.93 + 0.12 * fbm(v, 0.1, 6, 2))[:, None]))

    # the stands: an earth and rubble bank from the inner wall to the outer edge, which follows the hillside
    def h_out(x, z):
        if z > WEST + 2:
            return 0.05
        return float(np.clip(G(x, z) + 0.6, 1.4, 5.5))

    def h_in(x, z):
        return 0.03 if z > WEST - 2 else 2.3

    def bank_col(v):
        k = 0.9 + 0.2 * fbm(v, 0.12, 9, 3)
        mix = np.clip(v[:, 1] / 3.0, 0, 1)[:, None]
        base = EARTH * (1 - mix) + RUBBLE * mix
        return base * k[:, None]
    outer = cast(ringB, outline)
    parts.append(strip(ringB, outer, h_in, h_out, bank_col))
    parts.append(skirt(outer, h_out, FOUND, RUBBLE * 0.95))

    # the wall of the track (podium) and the inner wall of the stands, both ruined, not at the west end
    def not_west(P, lim=WEST):
        runs, cur = [], []
        for p in P:
            if p[1] <= lim:
                cur.append(p)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        return [r for r in runs if len(r) > 1]
    for run in not_west(ringA + ringA[:1]):
        parts += ruin_wall(run, 1.0, FOUND, lambda x, z: 1.6, RUBBLE, side=0.0)
    ringB_in = offset(ringA, 4.1)
    for run in not_west(ringB_in + ringB_in[:1], WEST - 3):
        parts += ruin_wall(run, 1.3, FOUND, lambda x, z: 2.7, RUBBLE * 0.97, side=0.0)

    # the spina and its metae
    for k in (260581757, 260581758):
        ring = OSM[k][:-1]
        parts.append(extrude(ring, FOUND, 0.55, tuple(GRANITE * 0.92), top_c=SPINA_TOP))
        parts.append(extrude(offset(ring, -0.6), 0.55, 0.8, tuple(GRANITE), top_c=SPINA_TOP))
    west = OSM[260581753][:-1]
    east = [(x, 0.8 - z) for x, z in west][::-1]                       # mirrored about the middle of the spina
    for ring in (west, east):
        parts.append(extrude(ring, FOUND, 1.1, tuple(GRANITE * 0.85), top_c=(0.5, 0.48, 0.44)))
        cx = sum(p[0] for p in ring) / len(ring); cz = sum(p[1] for p in ring) / len(ring)
        for dx in (-2.4, 0.0, 2.4):                                     # the three cones of the meta (stumps)
            parts.append(cyl(0.9, 0.55, 1.0, ccol(GRANITE * 0.8), 10).T(cx + dx, 1.1, cz))

    # the tribune on the north side
    trib = OSM[260581677][:-1]
    parts.append(extrude(trib, FOUND, 2.8, tuple(RUBBLE * 0.92), top_c=tuple(EARTH * 0.95)))
    parts += ruin_wall(OSM[260581755], 1.0, 2.5, lambda x, z: 3.6, RUBBLE)
    for (x0, x1) in ((-44.6, -43.6), (-53.6, -52.6)):
        for z in np.arange(106.0, 139.0, 5.5):                          # its piers towards the track and outside
            parts.append(box(1.0, 3.4, 1.4, ccol(GRANITE * 0.9)).T((x0 + x1) / 2, 0.0, z))

    # the outer wall of the southern stands with its buttresses
    for k in (260581675, 260581676, 260581679, 260581754, 260581760, 260581761, 260581751, 260581752,
              260581737, 260581741, 260581743):
        parts += ruin_wall(OSM[k], 1.1, FOUND, lambda x, z: h_out(x, z) + 0.8, RUBBLE * 0.95)
    # the passages in the eastern curve: pairs of short ashlar walls
    for k in range(260581739, 260581751):
        if k in OSM:
            parts += ruin_wall(OSM[k], 0.8, FOUND, lambda x, z: max(2.3, h_out(x, z)) + 0.7, GRANITE * 0.9, step=10)

    # the porta triumphalis in the middle of the eastern curve: its two ashlar jambs and the passage walls
    a, b = np.array((-6.5, -150.5)), np.array((-2.2, -171.5))
    d = (b - a) / np.linalg.norm(b - a)
    nrm = np.array((d[1], -d[0]))
    for s in (-1, 1):
        p0, p1 = a + nrm * s * 2.6, b + nrm * s * 2.6
        parts += ruin_wall([tuple(p0), tuple(p1)], 1.4, FOUND, lambda x, z: 3.6, GRANITE * 0.95, step=4)
        q = b + nrm * s * 3.4
        parts.append(seg_wall(tuple(q - d * 1.2), tuple(q + d * 0.8), 2.4, FOUND, 4.6, ccol(GRANITE)))
    quadp = [tuple(a + nrm * 1.9), tuple(b + nrm * 1.9 + d * 0.8), tuple(b - nrm * 1.9 + d * 0.8), tuple(a - nrm * 1.9)]
    parts.append(up(Mesh([(x, 0.05, z) for x, z in quadp], [(0, 1, 2), (0, 2, 3)]),
                    lambda v: np.tile(EARTH * 1.08, (len(v), 1))))

    # the carceres at the west end: twelve stalls, six on each side of the porta pompae
    wl = sorted(north[:8] + south[:4], key=lambda p: p[0])
    xs, zs = [p[0] for p in wl], [p[1] for p in wl]

    def zf(x):
        return float(np.interp(x, xs, zs)) + 0.6
    depth = 7.5
    stalls = list(np.linspace(-28.0, 5.0, 7)) + list(np.linspace(11.0, 44.0, 7))
    for x in stalls:                                                   # piers in front and the walls between stalls
        z0 = zf(x)
        parts.append(box(1.3, 1.1 + rng.uniform(-0.3, 0.3) - FOUND, 1.3, ccol(GRANITE * 0.9)).T(x, FOUND, z0 + 0.65))
        parts.append(seg_wall((x, z0 + 1.3), (x, z0 + depth), 0.8, FOUND, 0.8 + rng.uniform(-0.2, 0.2), ccol(RUBBLE)))
    back = [(x, zf(x) + depth) for x in np.linspace(-28.6, 44.6, 25)]
    parts += ruin_wall(back, 1.1, FOUND, lambda x, z: 1.2, RUBBLE)
    for x in (5.0, 11.0):                                              # the porta pompae: two big piers
        parts.append(seg_wall((x, zf(x) - 0.2), (x, zf(x) + depth + 0.5), 2.2, FOUND, 1.9, ccol(GRANITE)))
    return parts


def main():
    out = ROOT / "models" / "Q1984155"
    out.mkdir(parents=True, exist_ok=True)
    m = merge([p for p in circus() if p is not None])
    tris, size = write_glb(out / "model.glb", [("circo_romano", m)], roughness=0.95)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
