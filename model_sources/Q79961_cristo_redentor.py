"""
Christ the Redeemer / Cristo Redentor (Q79961), summit of Corcovado, Rio de Janeiro — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: Art Deco statue of Christ designed by the engineer Heitor da Silva Costa (1873–1947) with the
painter Carlos Oswald (1882–1971), sculpted by Paul Landowski (1875–1961), the face by Gheorghe Leonida
(1892–1942), the reinforced-concrete structure computed with Albert Caquot (1881–1976); clad in a mosaic of
pale soapstone tiles. Built 1922–1931, inaugurated 12 October 1931. The chapel of Nossa Senhora Aparecida inside
the pedestal was consecrated in 2006; the escalators, walkways and lifts of the visit were installed in 2003.
The statue is still under copyright in Brazil (Landowski died in 1961): rights "protected".

Measurements (see meta.json for the sources):
- Wikipedia (pt, en): the statue 30 m tall without the pedestal, 38 m with it (pedestal 8 m), arms 28 m from
  fingertip to fingertip; reinforced concrete clad in soapstone; faces east, over Guanabara Bay, with its back
  to the Tijuca Forest; summit of Corcovado ~709–710 m.
- Widely published figures (Santuário Cristo Redentor, press): head 3.75 m, each hand 3.2 m long.
- The pedestal is clad in pale grey-green stone close to the soapstone of the statue, with a lighter cornice.
- OpenStreetMap: way 105667349 (the chapel = the pedestal: an octagon, a 7.87 m square with cut corners whose
  sides face 80.9° / 170.9° / 260.9° / 350.9°, so heading = 80.9°, the statue facing east as all sources say);
  relation 16147269 (the viewpoint: the paved platform around the statue, outer way 141708228, with its long
  terrace in front of the statue along the ridge, 50 m to its rounded end); steps 1021319650/52/53, escalators
  1021319646 and 1021319648, footways 1021319647/49/51, 1027035331, 274557377, 274556230 (the restaurant
  terrace); buildings from the data.rio survey with their base elevations and heights: lift tower 274554624
  (ele 667.8 m, 24.3 m tall; the lift node 9468600313 is inside it), its link building 1021319645 (679.4 m,
  11.5 m), the shop 274554625 (686.2 m, 5.3 m), the restaurant 274554621 (688.2 m, 2.9 m).
- Levels: y = 0 is the platform around the pedestal, taken as ≈701 m (estimate ±3 m: Corcovado peak 710 m in
  OSM, the data.rio building bases 668–688 m, and the escalators at ~30° over their OSM length). The walkway
  behind the platform −2.5 m, the walkway to the lift −12 m (689 m), the restaurant terrace −13 m, the foot of
  the lift tower −33.2 m (667.8 m), its top −8.9 m.
- The statue's outline measured on photos (Wikimedia Commons «Christ the Redeemer-(Corcovado) front view.jpg»
  and «Christ on Corcovado mountain.JPG»), scaled by the 28 m arm span: the pleated tunic 5.3 m wide, the hem
  of the mantle ~7 m above the feet in front and rising diagonally across the back, the body 6.7–7.2 m wide
  over the mantle, arms horizontal ~23.6 m above the feet, wide sleeves hanging 3.5–4.5 m below the arms, the
  head bowed. Depths (front to back) are estimated from side views (±15 %).

The statue is lofted through superelliptic sections with relief: 26 vertical pleats of the tunic, the mantle
standing out at its hem with broad folds, the hanging sleeves lofted along the arms, open hands with fingers,
the head bowed ~10° with hair to the shoulders and beard, the Sacred Heart on the chest.

Axes (docs/MODEL_SPEC.md): +Y up, the statue faces +Z (heading 80.9°, east), +X is its left (north). Origin at
the centre of the pedestal on the platform.

    python model_sources/Q79961_cristo_redentor.py   -> models/Q79961/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import (Mesh, merge, box, cyl, lathe, extrude, walls, cap, offset, write_glb,  # noqa: E402
                     fbm, sstep, normals)

SOAP = np.array((0.80, 0.81, 0.77))        # soapstone tiles, pale grey-white with a green hint
PED = (0.66, 0.67, 0.61)                   # the pale grey-green stone of the pedestal (chapel)
PED_L = (0.80, 0.80, 0.75)                 # its lighter cornice
PAVE = (0.71, 0.69, 0.65)
WALL = (0.73, 0.72, 0.68)                  # the pale stone of the platform walls
WALL_D = (0.62, 0.61, 0.57)
ROCK = np.array((0.47, 0.45, 0.42))        # the granite-gneiss of Corcovado
DARK = (0.10, 0.10, 0.10)
STEP = (0.66, 0.64, 0.60)
ESC = (0.30, 0.31, 0.32)
GREEN_GLASS = (0.30, 0.55, 0.44)
TOWER = (0.70, 0.72, 0.68)
ROOF = (0.55, 0.54, 0.50)

YS = 8.0                                   # top of the pedestal = feet of the statue
H_PED = 3.935                              # half-width of the pedestal square (OSM 7.87 m)
CH = 1.4                                   # its cut corners

# OSM way 141708228 (outer ring of the viewpoint, relation 16147269) in model x/z (heading 80.9, origin
# -22.9519171, -43.2104631)
PLAT = [(10.0, -5.3), (3.6, -11.2), (1.9, -12.1), (-0.1, -12.5), (-2.1, -12.2), (-4.0, -11.2), (-9.9, -5.4),
        (-10.5, -0.7), (-12.0, -0.9), (-13.0, -1.0), (-13.1, 0.3), (-12.4, 2.2), (-10.5, 5.0), (-6.5, 9.0),
        (-4.4, 10.9), (-2.9, 12.9), (-2.4, 14.3), (-2.2, 15.7), (-2.3, 24.0), (-3.3, 24.4), (-4.1, 25.0),
        (-4.7, 26.4), (-4.8, 33.7), (-4.4, 34.6), (-3.8, 35.4), (-2.2, 36.2), (-2.3, 48.5), (-1.9, 49.5),
        (-1.1, 50.3), (0.0, 50.6), (1.2, 50.3), (2.1, 49.4), (2.4, 48.2), (2.4, 35.4), (4.7, 32.4), (4.9, 26.8),
        (4.6, 26.0), (4.2, 25.3), (3.3, 24.6), (2.2, 24.3), (2.2, 15.1), (4.1, 15.1), (6.1, 13.0), (11.0, 7.8),
        (12.2, 0.9), (12.4, -1.7), (11.1, -1.8), (10.0, -2.0)]
# walkway behind the platform (footways 274557377, 1021319651) and on to the top of the escalator 1021319648
T1_PATH = [(-12.2, -4.2), (-11.4, -5.8), (-7.0, -10.2), (-3.3, -13.6), (-0.6, -14.3), (0.9, -14.2), (4.9, -11.3),
           (7.6, -13.4), (9.3, -16.4)]
Y_T1 = -2.5
N_WALK = [(6.1, 13.0), (8.1, 14.4), (13.1, 8.2)]                       # footway 1021319647
L2_PATH = [(5.8, -33.1), (5.3, -38.6), (5.3, -43.8), (8.1, -44.8), (10.5, -45.6)]   # footway 1021319649
L2B_PATH = [(5.3, -43.8), (1.8, -41.6), (1.2, -33.4), (-3.7, -32.6)]                # footway 1027035331
Y_L2 = -12.0
TOWER_RING = [(11.7, -47.6), (11.7, -55.3), (8.1, -55.4), (8.1, -57.5), (1.5, -57.5), (1.5, -45.5), (8.1, -45.5),
              (8.1, -47.6)]                                                      # way 274554624
LINK_RING = [(4.0, -41.7), (3.9, -45.2), (0.1, -42.6), (0.2, -39.1)]             # way 1021319645
SHOP_RING = [(5.1, -26.4), (3.6, -32.7), (0.9, -32.3), (-0.4, -27.4)]            # way 274554625
REST_RING = [(-9.1, -29.2), (-11.8, -33.2), (-17.7, -29.0), (-14.9, -25.1)]      # way 274554621
REST_TERR = [(-11.5, -43.8), (-13.2, -44.9), (-19.5, -47.3), (-26.1, -48.7), (-28.4, -46.9), (-28.7, -42.2),
             (-27.5, -36.8), (-24.7, -33.6), (-21.1, -26.1), (-18.8, -28.2), (-20.6, -33.6), (-12.6, -37.1),
             (-9.8, -30.6), (-5.4, -33.7), (-7.9, -39.3)]                        # footway loop 274556230
Y_REST = -13.0
FOUND = -9.0                               # the platform walls go down to here, then the rock


def jit(c, s=0.015, rng=np.random.default_rng(1931)):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, s)), 0, 1))


def _fix(m, centre):
    """Flip triangles whose normals point towards centre."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    n = np.cross(b - a, c - a)
    bad = np.einsum("ij,ij->i", n, (a + b + c) / 3 - np.asarray(centre, float)) < 0
    m.f[bad] = m.f[bad][:, ::-1]
    return m


def sphere(r, c, seg=16, rings=10):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def ell(rx, ry, rz, c, seg=16, rings=10):
    return sphere(1.0, c, seg, rings).S(rx, ry, rz)


def sweep(pts, radii, c, seg=12, sy=1.0):
    """A tube of varying radius along a polyline, capped, faces outwards; sy squashes the section vertically."""
    P = np.asarray(pts, float)
    n = len(P)
    T = np.array([P[min(i + 1, n - 1)] - P[max(i - 1, 0)] for i in range(n)])
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    up = np.array([0.0, 1.0, 0.0])
    v = []
    for i in range(n):
        a = np.cross(T[i], up) if abs(T[i][1]) < 0.95 else np.cross(T[i], [1.0, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(a, T[i])
        for k in range(seg):
            th = 2 * math.pi * k / seg
            v.append(P[i] + radii[i] * (math.cos(th) * a + sy * math.sin(th) * b))
    f = []
    for i in range(n - 1):
        for k in range(seg):
            j = (k + 1) % seg
            p, q = i * seg + k, i * seg + j
            f += [(p, q, q + seg), (p, q + seg, p + seg)]
    V = np.array(v)
    for i in (0, n - 1):
        o = len(V); V = np.vstack([V, P[i]])
        for k in range(seg):
            j = (k + 1) % seg
            f.append((o, i * seg + k, i * seg + j))
    m = Mesh(V, f, c)
    a_, b_, c_ = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    nrm = np.cross(b_ - a_, c_ - a_)
    cen = (a_ + b_ + c_) / 3
    near = np.array([P[np.argmin(np.linalg.norm(P - q, axis=1))] for q in cen])
    bad = np.einsum("ij,ij->i", nrm, cen - near) < 0
    m.f[bad] = m.f[bad][:, ::-1]
    return m


def pbox(p0, p1, w, h, c, down=0.0):
    """A beam from p0 to p1 (the middle of its top edge), w wide (horizontal, across), from h above the line
    down to `down` below it; its ends are vertical (stair flights, escalators, sloping parapets)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    s = np.array([-d[2], 0.0, d[0]]); s /= (np.linalg.norm(s) or 1)
    s *= w / 2
    up = np.array([0.0, 1.0, 0.0])
    P = [p0 - s - up * down, p1 - s - up * down, p1 + s - up * down, p0 + s - up * down,
         p0 - s + up * h, p1 - s + up * h, p1 + s + up * h, p0 + s + up * h]
    quads = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    v, f = [], []
    for q in quads:
        o = len(v); v += [P[k] for k in q]; f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    m = Mesh(v, f, c)
    return _fix(m, np.mean(P, 0))


def strip(path, w):
    """Outline of a band w wide along a polyline (a walkway)."""
    P = np.asarray(path, float)
    n = len(P)
    L, R = [], []
    for i in range(n):
        a, b = P[max(i - 1, 0)], P[min(i + 1, n - 1)]
        d = b - a; d /= np.linalg.norm(d)
        nn = np.array([-d[1], d[0]])
        if 0 < i < n - 1:                                          # mitre
            d0 = P[i] - P[i - 1]; d0 /= np.linalg.norm(d0)
            n0 = np.array([-d0[1], d0[0]])
            k = 1 / max(float(np.dot(nn, n0)), 0.5)
        else:
            k = 1.0
        L.append(P[i] + nn * w / 2 * k); R.append(P[i] - nn * w / 2 * k)
    return [tuple(p) for p in L] + [tuple(p) for p in R[::-1]]


def path_len(path):
    P = np.asarray(path, float)
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))])


# ── The statue ───────────────────────────────────────────────────────────────
# (y above the feet, half-width x, half-depth z, centre z): measured on the photos, depths estimated
BODY = [(-0.3, 2.62, 1.80, 0.0), (0.0, 2.62, 1.80, 0.0), (3.0, 2.64, 1.82, 0.0), (6.6, 2.80, 1.90, 0.0),
        (7.4, 3.00, 2.08, 0.05), (9.0, 3.05, 2.14, 0.05), (12.0, 3.12, 2.18, 0.05), (15.0, 3.18, 2.20, 0.05),
        (18.0, 3.24, 2.20, 0.05), (20.0, 3.30, 2.16, 0.08), (21.8, 3.30, 2.08, 0.10), (23.0, 3.05, 1.95, 0.08),
        (24.0, 2.55, 1.75, 0.05), (24.8, 1.95, 1.50, 0.0), (25.5, 1.45, 1.25, 0.0), (26.0, 1.05, 1.00, 0.05)]


def _hem(X, Z, ax):
    """Height of the lower edge of the mantle: ~7 m in front, a diagonal across the back from ~7 m on the
    statue's left (+X) up to ~14 m on its right (−X)."""
    xn = np.clip(X / ax, -1.2, 1.2)
    back = 7.0 + 7.0 * sstep(0.6, -1.0, xn)
    w = sstep(-0.5, 0.3, Z / 2.0)                    # 1 in front, 0 at the back
    front = 7.0 + 0.6 * sstep(0.5, 1.1, np.abs(xn))
    return w * front + (1 - w) * back


def body(N=128, dy=0.25, p=2.3):
    T = np.array(BODY)
    ys = np.arange(T[0, 0], T[-1, 0] + 1e-6, dy)
    ax_, az_, cz_ = (np.interp(ys, T[:, 0], T[:, k]) for k in (1, 2, 3))
    t = np.linspace(0, 2 * math.pi, N, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    S = lambda u: np.sign(u) * np.abs(u) ** (2 / p)
    rings, cols = [], []
    for i, y in enumerate(ys):
        ax, az, cz = ax_[i], az_[i], cz_[i]
        X = ax * S(ct); Z = cz + az * S(st)
        tx = np.roll(X, -1) - np.roll(X, 1); tz = np.roll(Z, -1) - np.roll(Z, 1)
        nx, nz = tz, -tx
        ln = np.hypot(nx, nz) + 1e-9; nx /= ln; nz /= ln
        flip = (nx * X + nz * (Z - cz)) < 0
        nx[flip] *= -1; nz[flip] *= -1
        h = _hem(X, Z - cz, ax)
        mant = sstep(h - 0.08, h + 0.12, y)                          # 1 on the mantle
        noise = fbm(np.stack([X, np.full(N, y), Z], 1), 0.3, 7, 2)
        # tunic: 26 rounded vertical pleats
        pl = (np.abs(np.cos(13 * t + 0.15 * np.sin(y * 0.4))) ** 0.6 - 0.62) * 0.30
        # mantle: broad folds; on the front a V of folds on the chest, a vertical edge on the statue's left;
        # on the back diagonal folds
        front = sstep(-0.3, 0.4, st)
        vf = np.sin(2 * math.pi * (y * 0.55 - 0.75 * np.abs(X)) / 3.0 + 1.5 * noise) * 0.22 * sstep(15, 19, y) * sstep(25.5, 23.5, y)
        lf = np.sin(5 * t + 1.8 * noise + y * 0.08) * 0.18 * sstep(19, 15, y)
        edge = 0.38 * np.exp(-((X - 0.62 * ax) / 0.35) ** 2) * sstep(7.0, 8.0, y) * sstep(23.5, 21.0, y)
        bf = np.sin(2 * math.pi * (y + 0.9 * X) / 3.6 + 2.5 * noise + 0.8 * np.sin(t * 3)) * 0.14
        md = 0.20 + front * (vf + lf + edge) + (1 - front) * bf
        md = md * sstep(25.8, 24.0, y)
        fold = (1 - mant) * pl + mant * md
        X = X + nx * fold; Z = Z + nz * fold
        rings.append(np.stack([X, np.full(N, y), Z], 1))
        val = np.clip(0.55 + fold * 2.2 - 0.18 * (1 - mant), 0, 1)
        lip = sstep(h - 0.05, h + 0.02, y) * (1 - sstep(h + 0.02, h + 0.25, y))
        k = (0.82 + 0.25 * val) * (1 - 0.18 * lip)
        cols.append(np.clip(np.outer(k, SOAP), 0, 1))
    V = np.vstack(rings); C = np.vstack(cols)
    f = []
    for i in range(len(rings) - 1):
        for j in range(N):
            k = (j + 1) % N
            a, b = i * N + j, i * N + k
            f += [(a, a + N, b + N), (a, b + N, b)]
    for i, up in ((0, False), (len(rings) - 1, True)):
        o = len(V); V = np.vstack([V, rings[i].mean(0)]); C = np.vstack([C, SOAP])
        for j in range(N):
            k = (j + 1) % N
            f.append((o, i * N + k, i * N + j) if up else (o, i * N + j, i * N + k))
    m = Mesh(V, f, C)
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    n = np.cross(b - a, c - a); cen = (a + b + c) / 3
    d = cen * np.array([1, 0, 1])
    side = np.abs(n[:, 1]) < 0.7 * np.linalg.norm(n, axis=1)
    if ((np.einsum("ij,ij->i", n, d) < 0) & side).mean() > 0.3:
        m.f = m.f[:, ::-1]
    return m


ARM_Y, ARM_Z = 23.6, 0.15


def sleeve(side):
    """The arm with its wide hanging sleeve, lofted along x from inside the shoulder to the cuff."""
    xs = np.linspace(2.4, 10.3, 22)
    M1, M2, M3 = 12, 8, 5
    secs = []
    for x in xs:
        u = (x - 2.4) / 7.9
        r = 1.05 - 0.30 * u - 0.10 * u * u                  # the arm inside (thicker at the shoulder)
        yb = ARM_Y - (4.6 - 1.3 * u - 0.5 * u * u)           # the lowest point of the hanging cloth
        wz = 0.55 - 0.25 * u
        zc = ARM_Z
        pts = []
        for a in np.linspace(0, math.pi, M1 + 1)[:-1]:
            pts.append((zc + r * math.cos(a), ARM_Y + r * math.sin(a)))
        for s in np.linspace(0, 1, M2 + 1)[:-1]:
            pts.append((zc - r + (r - wz) * s ** 1.5, ARM_Y + (yb - ARM_Y) * s))
        for s in np.linspace(0, 1, M3 + 1)[:-1]:
            pts.append((zc - wz + 2 * wz * s, yb - 0.25 * math.sin(math.pi * s)))
        for s in np.linspace(0, 1, M2 + 1)[:-1]:
            pts.append((zc + wz + (r - wz) * s ** 0.67, yb + (ARM_Y - yb) * s))
        P = np.array([(x, yy, zz) for zz, yy in pts])
        # drapery: the cloth below the arm ripples in folds hanging from the arm
        dep = np.clip((ARM_Y - P[:, 1]) / (ARM_Y - yb), 0, 1)
        P[:, 2] += np.sign(P[:, 2] - zc) * 0.10 * np.sin(x * 1.6 + P[:, 1] * 1.1) * dep
        P[:, 1] += 0.18 * np.sin(x * 2.2) * dep ** 2
        secs.append(P)
    n = len(secs[0])
    V = np.vstack(secs)
    f = []
    for i in range(len(secs) - 1):
        for j in range(n):
            k = (j + 1) % n
            a, b = i * n + j, i * n + k
            f += [(a, a + n, b + n), (a, b + n, b)]
    o = len(V); V = np.vstack([V, secs[-1].mean(0)])
    for j in range(n):
        f.append((o, (len(secs) - 1) * n + j, (len(secs) - 1) * n + (j + 1) % n))
    m = Mesh(V, f, SOAP)
    # outward faces: away from the section centre
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    nrm = np.cross(b - a, c - a); cen = (a + b + c) / 3
    ref = np.stack([cen[:, 0], np.full(len(cen), ARM_Y - 1.2), np.full(len(cen), ARM_Z)], 1)
    ref[cen[:, 0] > 10.25, 0] = 9.0
    bad = np.einsum("ij,ij->i", nrm, cen - ref) < 0
    m.f[bad] = m.f[bad][:, ::-1]
    sh = 0.88 + 0.16 * np.clip((m.v[:, 1] - (ARM_Y - 4.5)) / 5.5, 0, 1)
    m.c = np.clip(np.outer(sh, SOAP), 0, 1)
    if side < 0:
        m.v[:, 0] *= -1
        m.f = m.f[:, ::-1]
    return m


def hand(side):
    """An open hand, palm forward, fingers along the arm (hand 3.2 m from the wrist at |x| 10.8 to 14.0)."""
    out = [sweep([(10.0, ARM_Y, ARM_Z), (10.9, ARM_Y, ARM_Z + 0.05)], [0.62, 0.55], SOAP, 12)]   # wrist
    out.append(ell(0.95, 0.72, 0.30, SOAP, 14, 8).T(11.75, ARM_Y + 0.02, ARM_Z + 0.08))         # palm
    for k, (dy, L) in enumerate(((0.45, 1.35), (0.15, 1.55), (-0.15, 1.5), (-0.45, 1.2))):
        p0 = np.array([12.5, ARM_Y + dy * 0.9, ARM_Z + 0.1])
        p1 = p0 + np.array([L, dy * 0.35, 0.12])
        out.append(sweep([p0, (p0 + p1) / 2 + [0, 0, 0.05], p1], [0.17, 0.15, 0.11], SOAP, 8))
    out.append(sweep([(11.2, ARM_Y + 0.45, ARM_Z + 0.15), (11.65, ARM_Y + 0.85, ARM_Z + 0.3),
                      (12.0, ARM_Y + 1.15, ARM_Z + 0.35)], [0.24, 0.2, 0.15], SOAP, 8))        # thumb
    m = merge(out)
    if side < 0:
        m.v[:, 0] *= -1
        m.f = m.f[:, ::-1]
    return m


def head():
    out = []
    c = tuple(SOAP)
    hc = np.array([0.0, 27.85, 0.25])
    sk = ell(1.18, 1.85, 1.38, c, 22, 14).T(*hc)

    def jaw(v):
        t = np.clip((hc[1] - v[:, 1]) / 1.85, 0, 1)
        v[:, 0] = v[:, 0] * (1 - 0.25 * t ** 2)
        return v
    out.append(sk.warp(jaw))
    hair = tuple(SOAP * 0.95)
    out.append(ell(1.36, 1.55, 1.36, hair, 20, 12).T(0, 28.35, -0.05))                   # hair over the crown
    out.append(ell(1.22, 1.75, 0.85, hair, 18, 10).T(0, 26.7, -0.45))                     # hair down the back
    for s in (-1, 1):                                                                     # hair at the sides
        out.append(sweep([(s * 1.05, 28.6, 0.3), (s * 1.2, 27.3, 0.05), (s * 1.3, 25.9, -0.15),
                          (s * 1.4, 25.0, -0.3)], [0.45, 0.48, 0.45, 0.3], hair, 10))
    out.append(ell(1.0, 0.25, 0.32, c, 14, 6).T(0, 28.45, 1.42))                          # brow
    for s in (-1, 1):
        out.append(ell(0.30, 0.13, 0.12, tuple(SOAP * 0.72), 10, 6).T(s * 0.45, 28.1, 1.5))  # eyes
    out.append(sweep([(0, 28.35, 1.6), (0, 27.5, 1.82)], [0.14, 0.22], c, 8))              # nose
    out.append(ell(0.62, 0.22, 0.25, c, 12, 6).T(0, 27.15, 1.45))                         # moustache
    out.append(ell(0.78, 0.85, 0.62, tuple(SOAP * 0.96), 14, 8).T(0, 26.55, 1.05))        # beard
    out.append(cyl(0.95, 0.9, 1.8, c, 14).T(0, 25.2, 0.15))                               # neck
    m = merge(out)
    # bowed: about 10° forward around the base of the neck
    m.v = m.v - [0, 25.6, 0.1]
    m.R(math.radians(11), 0, 0)
    m.v = m.v + [0, 25.6, 0.1]
    return m


def statue():
    out = [body()]
    for s in (-1, 1):
        out += [sleeve(s), hand(s)]
        # the mantle over the shoulder, sloping from the neck into the sleeve
        out.append(ell(2.5, 0.75, 1.7, tuple(SOAP * 0.98), 18, 8).R(0, 0, -s * math.radians(16)).T(s * 2.4, 24.15, 0.12))
    out.append(head())
    out.append(ell(0.42, 0.40, 0.14, tuple(SOAP * 0.98), 12, 6).R(0, 0, math.radians(45)).T(0, 21.4, 2.32))  # heart
    for s in (-1, 1):                                                                     # toes under the hem
        out.append(ell(0.55, 0.32, 0.75, tuple(SOAP * 0.95), 10, 6).T(s * 0.75, 0.25, 1.75))
    return merge(out).T(0, YS, 0)


# ── The pedestal and its chapel ──────────────────────────────────────────────
def octo(h, ch):
    return [(h, -h + ch), (h, h - ch), (h - ch, h), (-h + ch, h), (-h, h - ch), (-h, -h + ch), (-h + ch, -h),
            (h - ch, -h)]


def pedestal():
    out = [extrude(octo(H_PED + 0.35, CH + 0.15), FOUND, 0.45, jit(PED), top_c=PED),
           extrude(octo(H_PED, CH), 0.45, 7.25, jit(PED), top=False),
           extrude(octo(H_PED + 0.22, CH + 0.09), 7.25, 7.75, jit(PED_L), top_c=PED_L, bottom=True),
           extrude(octo(H_PED - 0.25, CH - 0.1), 7.75, YS, PED_L, top_c=jit(PED_L))]
    # thin shadow line under the cornice and between the stone courses
    for y in (2.6, 4.9):
        out.append(walls(octo(H_PED + 0.012, CH), y, y + 0.05, (0.50, 0.51, 0.47)))
    # the door of the chapel at the back (−Z), with its frame
    out.append(box(2.9, 3.6, 0.2, PED_L).T(0, 0.45, -H_PED - 0.08))
    out.append(box(2.0, 2.9, 0.06, DARK).T(0, 0.45, -H_PED - 0.2))
    # the plinth step on the platform
    out.append(extrude(octo(H_PED + 0.9, CH + 0.4), -0.2, 0.15, jit(PAVE), top_c=(0.66, 0.64, 0.60)))
    return out


# ── The summit: platform, terraces, stairs, escalators, the lift tower ──────
def parapet(ring, y, c, gaps=(), h=1.1, t=0.3, closed=True):
    """A low wall on the edge of a terrace, skipping the edges whose middle lies near a point in gaps."""
    out = []
    P = list(ring)
    n = len(P) if closed else len(P) - 1
    for i in range(n):
        a, b = np.array(P[i], float), np.array(P[(i + 1) % len(P)], float)
        mid = (a + b) / 2
        if any(np.hypot(*(mid - np.array(g))) < r for g, r in gaps):
            continue
        L = np.linalg.norm(b - a)
        if L < 0.05:
            continue
        d = (b - a) / L
        a2, b2 = a - d * t / 2, b + d * t / 2
        out.append(pbox((a2[0], y + h, a2[1]), (b2[0], y + h, b2[1]), t, 0.0, c, down=h))
        out.append(pbox((a2[0], y + h + 0.08, a2[1]), (b2[0], y + h + 0.08, b2[1]), t + 0.12, 0.0, WALL_D, down=0.1))
    return out


def resample(ring, step):
    P = np.asarray(ring, float)
    out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        k = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * s / k for s in range(k)]
    return np.array(out)


def hull(pts):
    pts = sorted(set(map(tuple, np.round(np.asarray(pts, float), 3))))

    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cr(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def rock(top_ring, y_top, levels):
    """The bare rock of the summit under the platform: rings growing outwards as they go down, roughened."""
    base = resample(top_ring, 2.0)
    n = len(base)
    cen = base.mean(0)
    rings = [np.column_stack([base[:, 0], np.full(n, y_top), base[:, 1]])]
    for y, d in levels:
        R = base - cen
        L = np.linalg.norm(R, axis=1, keepdims=True)
        P = base + R / L * d
        rings.append(np.column_stack([P[:, 0], np.full(n, y), P[:, 1]]))
    V = np.vstack(rings)
    # roughen everything below the top ring
    nz = np.stack([fbm(V, 0.12, 31 + k, 3) - 0.5 for k in range(3)], 1) * 3.0
    w = np.clip((y_top - V[:, 1]) / 6.0, 0, 1)
    nz[:, 1] *= 0.5
    V = V + nz * w[:, None]
    f = []
    for i in range(len(rings) - 1):
        for j in range(n):
            k = (j + 1) % n
            a, b = i * n + j, i * n + k
            f += [(a, a + n, b + n), (a, b + n, b)]
    m = Mesh(V, f, ROCK)
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    nrm = np.cross(b - a, c - a)
    d = (a + b + c) / 3 - np.array([cen[0], 0, cen[1]])
    d[:, 1] = 0
    if (np.einsum("ij,ij->i", nrm, d) < 0).mean() > 0.5:
        m.f = m.f[:, ::-1]
    s = fbm(m.v, (0.25, 0.05, 0.25), 5, 3)
    g = fbm(m.v, 0.08, 9, 2)
    col = ROCK * (0.78 + 0.35 * s[:, None])
    moss = sstep(0.6, 0.75, g)[:, None] * 0.55
    m.c = np.clip(col * (1 - moss) + np.array((0.30, 0.37, 0.24)) * moss, 0, 1)
    topcap = cap([tuple(p) for p in base], y_top, (0.40, 0.41, 0.35))
    return merge([m, topcap])


def flight(path, y0, y1, w, c=STEP, rail=True):
    """A flight of steps along a polyline from level y0 to y1, with a sloping parapet on each side."""
    P = np.asarray(path, float)
    L = path_len(P)
    tot = L[-1]
    nst = max(2, int(round(abs(y1 - y0) / 0.17)))
    out = []
    for k in range(nst):
        s0, s1 = tot * k / nst, tot * (k + 1) / nst
        p0 = np.array([np.interp(s0, L, P[:, 0]), np.interp(s0, L, P[:, 1])])
        p1 = np.array([np.interp(s1, L, P[:, 0]), np.interp(s1, L, P[:, 1])])
        yk = y0 + (y1 - y0) * (k + 1) / nst if y1 > y0 else y0 + (y1 - y0) * k / nst
        out.append(pbox((p0[0], yk, p0[1]), (p1[0], yk, p1[1]), w, 0.0, jit(c, 0.02), down=0.45))
    if rail:
        ys = y0 + (y1 - y0) * L / tot
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            d = (b - a) / np.linalg.norm(b - a)
            nn = np.array([-d[1], d[0]])
            for s in (-1, 1):
                q0, q1 = a + nn * s * (w / 2 + 0.12), b + nn * s * (w / 2 + 0.12)
                out.append(pbox((q0[0], ys[i] + 1.0, q0[1]), (q1[0], ys[i + 1] + 1.0, q1[1]), 0.24, 0.0, WALL,
                                down=1.6))
    return out


def escalator(p0, y0, p1, y1, canopy=False):
    """Two escalators side by side between their balustrades, from (p0, y0) up to (p1, y1)."""
    out = []
    a = (p0[0], y0, p0[1]); b = (p1[0], y1, p1[1])
    out.append(pbox(a, b, 2.6, 0.0, ESC, down=1.0))                                  # truss
    out.append(pbox(a, b, 2.3, 0.05, (0.22, 0.22, 0.23)))                            # the steps
    d = np.array(p1) - np.array(p0); d = d / np.linalg.norm(d)
    nn = np.array([-d[1], d[0]])
    for s in (-1.2, 0.0, 1.2):
        q0, q1 = np.array(p0) + nn * s, np.array(p1) + nn * s
        out.append(pbox((q0[0], y0 + 1.0, q0[1]), (q1[0], y1 + 1.0, q1[1]), 0.12, 0.0, (0.55, 0.60, 0.58), down=1.0))
    glass = []
    if canopy:
        for s in (-1, 1):
            q0, q1 = np.array(p0) + nn * s * 1.45, np.array(p1) + nn * s * 1.45
            glass.append(pbox((q0[0], y0 + 3.0, q0[1]), (q1[0], y1 + 3.0, q1[1]), 0.08, 0.0, GREEN_GLASS, down=2.0))
        glass.append(pbox((p0[0], y0 + 3.2, p0[1]), (p1[0], y1 + 3.2, p1[1]), 3.2, 0.0, (0.24, 0.42, 0.33), down=0.2))
    return out, glass


def summit():
    stone, glass = [], []
    # the platform round the statue and the terrace along the ridge in front of it
    stone.append(extrude(PLAT, FOUND, 0.0, jit(WALL), top_c=PAVE))
    for y in (-3.0, -6.0):                                                        # courses of the retaining wall
        stone.append(walls(offset(PLAT, 0.03), y, y + 0.08, WALL_D))
    gaps = [((11.6, -1.8), 1.2), ((-12.5, -0.95), 1.0), ((8.55, 10.4), 3.0), ((10.6, -3.6), 1.4)]
    stone += parapet(PLAT, 0.0, jit(WALL), gaps)
    # the paved ring of the inner path round the pedestal (footway 1447517218): a slightly darker band
    ring = [(8.7 * math.cos(a), 8.7 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
    inner = [(7.2 * math.cos(a), 7.2 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
    stone.append(cap(ring, 0.02, (0.64, 0.62, 0.58), holes=[inner]))
    # walkway behind the platform, its stairs up to the platform on both sides
    t1 = strip(T1_PATH, 3.0)
    stone.append(extrude(t1, FOUND - 3, Y_T1, jit(WALL), top_c=PAVE))
    outer_edge = strip(T1_PATH, 3.0)[len(T1_PATH):]                                 # the outer side
    stone += parapet(outer_edge, Y_T1, jit(WALL), closed=False)
    stone += flight([(4.9, -11.3), (8.6, -8.1), (11.2, -5.8), (11.15, -3.2)], Y_T1, -0.1, 2.2)
    stone.append(extrude(strip([(11.15, -3.4), (11.1, -1.2)], 2.4), FOUND, 0.0, jit(WALL), top_c=PAVE))
    stone += flight([(-11.4, -5.8), (-11.9, -2.0)], Y_T1, -0.1, 2.0)
    # the long stair down to the restaurant terrace (steps 1021319650)
    stone += flight([(-4.8, -14.8), (-10.1, -17.4)], Y_T1, -4.4, 2.2)
    stone.append(extrude(strip([(-9.6, -17.0), (-10.7, -19.0)], 2.6), -16, -4.4, jit(WALL), top_c=PAVE))
    stone += flight([(-10.7, -19.0), (-10.0, -25.8)], -4.4, Y_REST, 2.2)
    # the north walkway to the covered escalator (footway 1021319647) and the escalator 1021319646
    stone.append(extrude(strip(N_WALK, 2.6), FOUND, 0.0, jit(WALL), top_c=PAVE))
    stone += parapet(strip(N_WALK, 2.6)[len(N_WALK):], 0.0, jit(WALL), closed=False)
    s, g = escalator((16.3, -8.6), -9.6, (13.3, 7.4), 0.0, canopy=True)
    stone += s; glass += g
    stone.append(box(4.0, 3.4, 3.4, GREEN_GLASS).R(0, math.radians(-10.7), 0).T(13.1, 0.0, 8.3))   # its glass head
    stone.append(box(4.3, 0.3, 3.7, (0.24, 0.42, 0.33)).R(0, math.radians(-10.7), 0).T(13.1, 3.4, 8.3))
    stone.append(extrude([(14.6, -8.0), (18.2, -8.6), (17.8, -11.2), (14.2, -10.6)], -16, -9.6, jit(WALL), top_c=PAVE))
    # the escalator 1021319648 from the walkway to the lift up to the walkway behind the platform
    s, g = escalator((5.8, -33.1), Y_L2, (9.3, -16.7), Y_T1)
    stone += s; glass += g
    stone.append(extrude(strip([(9.3, -16.4), (9.9, -18.0)], 3.4), -12, Y_T1, jit(WALL), top_c=PAVE))
    # walkways to the lift tower and to the restaurant, on piers
    for path in (L2_PATH, L2B_PATH):
        stone.append(extrude(strip(path, 2.8), Y_L2 - 0.5, Y_L2, jit(WALL_D), top_c=PAVE, bottom=True))
        for side in (strip(path, 2.8)[:len(path)], strip(path, 2.8)[len(path):]):
            stone += parapet(side, Y_L2, jit(WALL), closed=False)
        for p in path[1:-1]:
            stone.append(box(0.8, 10, 0.8, WALL_D).T(p[0], Y_L2 - 10.4, p[1]))
    stone.append(extrude([(4.6, -32.0), (7.2, -31.6), (7.6, -34.6), (4.4, -34.6)], -16, Y_L2, jit(WALL), top_c=PAVE))
    # the lift tower (data.rio: base 667.8 m, 24.3 m tall) and its link building (679.4 m, 11.5 m)
    yb, yt = -33.2, -8.9
    stone.append(extrude(TOWER_RING, yb, yt - 3.0, jit(TOWER), top=False))
    glass.append(extrude(offset(TOWER_RING, -0.05), yt - 3.0, yt, GREEN_GLASS, top=False))
    stone.append(extrude(offset(TOWER_RING, 0.25), yt, yt + 0.4, (0.24, 0.42, 0.33), top_c=(0.26, 0.45, 0.35)))
    for x in (3.3, 6.3):                                                          # tall window strips
        stone.append(box(1.2, 16.0, 0.06, (0.32, 0.40, 0.38)).T(x, yb + 4.0, -45.5 + 0.04))
    for z in (-50.0, -53.0):
        stone.append(box(0.06, 16.0, 1.2, (0.32, 0.40, 0.38)).T(11.7 + 0.04, yb + 4.0, z))
    stone.append(extrude(LINK_RING, -21.6, -10.1, jit(TOWER), top_c=ROOF))
    # the shop (686.2 m, 5.3 m) and the restaurant on its terrace (688.2 m, 2.9 m)
    stone.append(extrude(SHOP_RING, -20.0, -9.5, jit((0.78, 0.76, 0.70)), top_c=ROOF))
    stone.append(extrude(REST_TERR, -24.0, Y_REST, jit(WALL), top_c=PAVE))
    stone += parapet(REST_TERR, Y_REST, jit(WALL))
    stone.append(extrude(REST_RING, Y_REST, Y_REST + 2.9, jit((0.80, 0.78, 0.72)), top_c=ROOF))
    # the bare rock of the summit under it all
    top = hull(PLAT + t1 + strip(N_WALK, 2.6))
    stone.append(rock(offset(top, 0.4), -5.0,
                      [(-10.0, 1.5), (-14.0, 4.0), (-20.0, 7.0), (-27.0, 10.0), (-36.0, 14.0)]))
    return stone, glass, top


def main():
    out = ROOT / "models" / "Q79961"
    out.mkdir(parents=True, exist_ok=True)
    stone, glass, top = summit()
    st = statue()
    ped = merge(pedestal())
    sm = merge(stone)
    gl = merge(glass)
    parts = [("summit", sm), ("pedestal", ped, {"material": "polished"}), ("statue", st, {"material": "soapstone"})]
    if len(gl.f):
        parts.append(("glass", gl, {"material": "glass"}))
    tris, size = write_glb(out / "model.glb", parts, roughness=0.9,
                           materials={"soapstone": {"metallic": 0.0, "roughness": 0.75},
                                      "polished": {"metallic": 0.1, "roughness": 0.45},
                                      "glass": {"metallic": 0.35, "roughness": 0.18}})
    allv = np.vstack([p[1].v for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles (statue {st.tris:,}), {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    sv = st.v
    print(f"statue: x {sv[:, 0].min():.2f}..{sv[:, 0].max():.2f}, y {sv[:, 1].min():.2f}..{sv[:, 1].max():.2f}")
    fp_pts = list(top) + TOWER_RING + REST_TERR + SHOP_RING + LINK_RING + [(18.2, -8.6), (17.8, -11.2), (14.2, -10.6)]
    fp = [[round(x, 1), round(z, 1)] for x, z in offset(hull(fp_pts), 2.0)]
    print("footprint:", json.dumps(fp))


if __name__ == "__main__":
    main()
