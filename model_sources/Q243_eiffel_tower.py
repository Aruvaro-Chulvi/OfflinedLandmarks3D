"""
Eiffel Tower / Tour Eiffel (Q243), Paris — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The tower: Compagnie des Établissements Eiffel — Gustave Eiffel (died 1923), engineers Maurice Koechlin (died
1946) and Émile Nouguier (died 1897), architect Stephen Sauvestre (died 1919); built 1887–1889. Public domain
in France since 2017; «protected» under the project's 80-year rule until 1 January 2027 (Koechlin).

Measurements (see meta.json for the sources):
- OpenStreetMap way 5013364 (the base square, 123.6 × 123.5 m, sides facing 134.2° / 314.2°, height 330) and its
  building:parts: the four piers (ways 69034127/135/179/180, each a ~24 m square, outer edges at ±61.8 m, inner
  edges at ±37.6 m), the 1st-floor pavilions (ways 308145239/258/259, 57.63–64 m), the 3rd floor (way 308021389,
  ±8.5 m at 276.13 m).
- fr.wikipedia «Tour Eiffel»: base square 125 m, 74.24 m between the piers (inside) and 124.90 m (outside), each pier
  in a 25 m square; 1st floor at 57.63 m, 70.69 m side, 4 200 m²; 2nd floor at 115.73 m, 40.96 m side; 3rd floor at
  276.13 m, 18.65 m side, upper deck 279.11 m; arches 74 m across rising to 39 m; 330 m with the antennas (2022).
- Proportions measured on Wikimedia Commons «Tour Eiffel Wikimedia Commons.jpg» (frontal, from the Champ de Mars),
  scaled on the known widths of the 1st, 2nd and 3rd floors at their own heights (±1 m): the outer edge of the legs
  (≈ 50 m half-width at 25 m, 42 at 42 m, 37 at 51 m, 28.5 at 64.5 m, 19 at 110 m), the gap between the legs
  above the 1st floor (≈ 17 m → 13 m), the narrower shaft above the 2nd floor (half-width 14.6 m, 9.2 m at 178 m,
  8.6 m at 205 m, 7.6 m under the 3rd floor), the four upper columns (each a third of the face at the 2nd floor)
  whose inner edges meet at the middle of each face at ≈ 178 m, the bands under the platforms, the head and the
  antenna mast.
- The lattice: each leg is a box girder (four corner chords) whose faces carry a middle rail, horizontal struts
  and big X braces (below the 1st floor one X across the whole face every two panels; between the 1st and 2nd
  floors one X per half-face); above the 2nd floor each face is split by the inner chords of the columns (three
  strips, then two above 178 m), one X per panel. Members are thin square bars (chords 1–2.3 m, braces 0.4–0.7 m).

Axes (docs/MODEL_SPEC.md): +Y up, +Z faces 314.2° (north-west: the face towards the Seine, the Pont d'Iéna and the
Trocadéro), +X points 224.2° (south-west). The tower is square, so the main façade is a choice: the NW face is
the one seen in the classic view from the Trocadéro and the one the Champ de Mars axis ends on. Origin on the
ground at the centre of the base square (= centroid of the OSM outline).

    python model_sources/Q243_eiffel_tower.py   -> models/Q243/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, extrude, write_glb, GLASS, METAL  # noqa: E402

BROWN_LO = np.array((0.50, 0.385, 0.30))       # «brun Tour Eiffel», darkest at the bottom
BROWN_HI = np.array((0.63, 0.515, 0.41))       # … lightest at the top
DECK = (0.52, 0.48, 0.43)
STONE = (0.80, 0.75, 0.66)
GLASS_C = (0.36, 0.45, 0.49)
DARK = (0.20, 0.18, 0.16)
GRILLE = (0.38, 0.33, 0.29)                   # fine mesh fences, seen against the light
WHITE = (0.86, 0.86, 0.84)
FOUND = -3.0

Y1, Y2, Y3 = 57.63, 115.73, 276.13            # floors
Y_JOIN = 178.0                                # the inner chords of the four upper columns meet


# ── Profiles (centre lines of the chords, half-widths from the axis) ─────────
def pchip(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    h = np.diff(xs); d = np.diff(ys) / h
    m = np.zeros_like(ys)
    for k in range(1, len(xs) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    m[0], m[-1] = d[0], d[-1]

    def f(x):
        x = np.clip(x, xs[0], xs[-1])
        k = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(h) - 1))
        t = (x - xs[k]) / h[k]
        h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
        return float(h00 * ys[k] + h10 * h[k] * m[k] + h01 * ys[k + 1] + h11 * h[k] * m[k + 1])
    return f


W_LOW = pchip([-3, 0, 25, 45, Y1, 64.5, 80, 110, Y2],
              [61.1, 59.6, 48.3, 38.3, 31.6, 27.5, 24.0, 18.2, 17.6])         # outer corner chord
I_LOW = pchip([-3, 0, 30, Y1, 87, Y2], [41.1, 39.6, 24.6, 11.0, 8.6, 7.0])     # inner chords
W_UP = pchip([Y2, 140, Y_JOIN, 205, 240, Y3, 280], [14.3, 11.9, 9.1, 8.3, 7.6, 7.0, 6.95])


def U_UP(h):                                                                   # inner chords of the upper columns
    return max(0.0, 4.75 * (Y_JOIN - h) / (Y_JOIN - Y2))


def brown(y):
    t = np.clip(np.asarray(y) / 300.0, 0, 1)[..., None]
    return BROWN_LO + (BROWN_HI - BROWN_LO) * t


# ── Bars: thin square members ────────────────────────────────────────────────
class Acc:
    def __init__(self):
        self.v, self.f = [], []
        self.n = 0

    def bar(self, p0, p1, w, d=None, nrm=(0, 0, 1)):
        """A square-section member from p0 to p1: w wide in the face plane, d deep along nrm (no end caps)."""
        d = w if d is None else d
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        t = p1 - p0
        L = np.linalg.norm(t)
        if L < 1e-3:
            return
        t /= L
        n = np.asarray(nrm, float)
        n = n - np.dot(n, t) * t
        if np.linalg.norm(n) < 1e-6:
            n = np.array([1.0, 0, 0]) - t[0] * t
        n /= np.linalg.norm(n)
        b = np.cross(t, n)
        off = [b * w / 2 + n * d / 2, -b * w / 2 + n * d / 2, -b * w / 2 - n * d / 2, b * w / 2 - n * d / 2]
        sharp = max(w, d) >= 0.9
        if sharp:
            for k in range(4):
                a, c = off[k], off[(k + 1) % 4]
                o = self.n
                self.v += [p0 + a, p0 + c, p1 + c, p1 + a]
                self.f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
                self.n += 4
        else:
            o = self.n
            self.v += [p0 + q for q in off] + [p1 + q for q in off]
            for k in range(4):
                j = (k + 1) % 4
                self.f += [(o + k, o + 4 + j, o + j), (o + k, o + 4 + k, o + 4 + j)]
            self.n += 8

    def poly(self, pts, w, nrm):
        for a, b in zip(pts[:-1], pts[1:]):
            self.bar(a, b, w, w, nrm)

    def mesh(self):
        v = np.array(self.v, float).reshape(-1, 3)
        return Mesh(v, np.array(self.f, np.int64).reshape(-1, 3), brown(v[:, 1]))


def breaks(h0, h1, step, even=False):
    """Panel heights from h0 to h1, each panel about step(h) tall."""
    hs = np.linspace(h0, h1, 400)
    dens = np.array([1.0 / step(h) for h in hs])
    N = np.concatenate([[0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(hs))])
    n = max(1, int(round(N[-1])))
    if even and n % 2:
        n += 1
    return [float(np.interp(k * N[-1] / n, N, hs)) for k in range(n + 1)]


def lattice(acc, R1, R2, ys, mode, nrm, hw=0.6, dw=0.5, mw=0.8, rails=False, rw=1.0, first=True, last=True):
    """Bracing of one face strip between two rails R1(h), R2(h) (functions -> 3D points).
    mode 'X2': middle rail, a strut at every break, one X across the whole strip every two panels;
    'Xh': middle rail, one X per half-strip and panel; 'X1': one X across the strip per panel."""
    M = lambda h: (np.asarray(R1(h)) + np.asarray(R2(h))) / 2                 # noqa: E731
    if rails:
        acc.poly([R1(y) for y in ys], rw, nrm)
        acc.poly([R2(y) for y in ys], rw, nrm)
    for y in ys[(0 if first else 1):(None if last else -1)]:                  # struts (deepest)
        acc.bar(R1(y), R2(y), hw, hw + 0.12, nrm)
    if mode in ("X2", "Xh"):
        acc.poly([M(y) for y in ys], mw, nrm)
    for k in range(len(ys) - 1):
        a, b = ys[k], ys[k + 1]
        if mode == "X2":
            if k % 2 or k + 2 >= len(ys):
                continue
            b = ys[k + 2]
            acc.bar(R1(a), R2(b), dw, dw, nrm); acc.bar(R2(a), R1(b), dw, dw + 0.07, nrm)
        elif mode == "Xh":
            for R in (R1, R2):
                acc.bar(R(a), M(b), dw, dw, nrm); acc.bar(M(a), R(b), dw, dw + 0.07, nrm)
        else:
            acc.bar(R1(a), R2(b), dw, dw, nrm); acc.bar(R2(a), R1(b), dw, dw + 0.07, nrm)


def band(acc, y0, y1, half, z, pitch, nrm=(0, 0, 1), rw=0.55, dw=0.35, rows=1):
    """A flat lattice girder on the face +Z at depth z (a number or z(y)): rails, posts, X braces of `pitch`; half(y) = half-length."""
    Z = z if callable(z) else (lambda y: z)
    ys = list(np.linspace(y0, y1, rows + 1))
    for y in ys:
        hx = half(y)
        acc.bar((-hx, y, Z(y)), (hx, y, Z(y)), rw, rw + 0.1, nrm)
    for r in range(rows):
        a, b = ys[r], ys[r + 1]
        hx = min(half(a), half(b))
        n = max(2, int(round(2 * hx / pitch)))
        xs = np.linspace(-hx, hx, n + 1)
        for i in range(n):
            acc.bar((xs[i], a, Z(a)), (xs[i + 1], b, Z(b)), dw, dw, nrm)
            acc.bar((xs[i + 1], a, Z(a)), (xs[i], b, Z(b)), dw, dw + 0.06, nrm)
        for x in xs[::2]:
            acc.bar((x, a, Z(a)), (x, b, Z(b)), dw + 0.05, dw + 0.14, nrm)


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def rot4(m):
    """The same piece on the four faces (90° turns about the axis)."""
    return merge([m.copy().R(0, k * math.pi / 2, 0) for k in range(4)])


# ── One leg (quadrant +X +Z) ─────────────────────────────────────────────────
def leg():
    acc = Acc()
    A = lambda h: (W_LOW(h), h, W_LOW(h))                                     # noqa: E731  outer corner
    Bq = lambda h: (I_LOW(h), h, W_LOW(h))                                    # noqa: E731  on the +Z face
    C = lambda h: (W_LOW(h), h, I_LOW(h))                                     # noqa: E731  on the +X face
    D = lambda h: (I_LOW(h), h, I_LOW(h))                                     # noqa: E731  inner corner
    lo = breaks(1.2, Y1, lambda h: 0.27 * (W_LOW(h) - I_LOW(h)), even=True)
    hi = breaks(Y1, Y2, lambda h: 0.50 * (W_LOW(h) - I_LOW(h)))
    faces = [(A, Bq, (0, 0, 1)), (A, C, (1, 0, 0)), (Bq, D, (1, 0, 0)), (C, D, (0, 0, 1))]
    for R1, R2, n in faces:
        lattice(acc, R1, R2, lo, "X2", n, hw=0.75, dw=0.6, mw=0.9)
        lattice(acc, R1, R2, hi, "Xh", n, hw=0.65, dw=0.5, mw=0.85, first=False)
    ys = sorted(set([FOUND + 0.4] + lo + hi + list(np.arange(4, Y2, 4.0))))
    for R, s in ((A, 2.3), (Bq, 2.0), (C, 2.0), (D, 1.8)):                     # chords
        pts = [R(y) for y in ys]
        for a, b in zip(pts[:-1], pts[1:]):
            w = s - 0.6 * a[1] / Y2
            acc.bar(a, b, w, w, (1, 0, 0))
    piers = [B(x - 2.3, x + 2.3, FOUND, 1.2, z - 2.3, z + 2.3, STONE)
             for x in (I_LOW(0), W_LOW(0)) for z in (I_LOW(0), W_LOW(0))]
    return acc.mesh(), merge(piers)


# ── First floor (face +Z, turned four times) ─────────────────────────────────
R_ARCH, YC_ARCH, T_ARCH = 37.3, 1.5, 3.0      # intrados 74.6 m across, rising to 38.8 m


def ZF(y):
    """The inclined outer face of the legs (+Z), where the arch and the girders under the 1st floor lie."""
    return max(W_LOW(y) + 0.25, 35.9)


def first_floor_face():
    acc = Acc()
    solid = []
    # lattice girders under the frieze, from leg to leg, on the inclined face
    band(acc, 45.5, 51.6, lambda y: W_LOW(y) - 0.4, ZF, 3.05, rows=2)
    band(acc, 42.3, 45.5, lambda y: W_LOW(y) - 0.4, lambda y: ZF(y) + 0.2, 3.2, rows=1, rw=0.42)
    # the decorative arch: a solid ribbon lying on the inclined face, its feet on the inner piers
    n = 64
    th = np.concatenate([[None], np.linspace(0, math.pi, n + 1), [None]])
    r0, r1 = R_ARCH, R_ARCH + T_ARCH

    def P(r, t, dz):
        if t is None:                                           # straight foot down to the pier
            x, y = None, 1.2
        else:
            x, y = r * math.cos(t), YC_ARCH + r * math.sin(t)
        return x, y, dz

    def W3(r, t, dz, side):
        x, y, dz = P(r, t, dz)
        if x is None:
            x = side * r
        return (x, y, ZF(y) + 0.35 + dz)
    v, f = [], []

    def quad(a, b, c, d):
        o = len(v); v.extend([a, b, c, d]); f.extend([(o, o + 1, o + 2), (o, o + 2, o + 3)])
    for i in range(len(th) - 1):
        a, b = th[i], th[i + 1]
        sa = 1 if i == 0 else -1
        z0, z1 = -0.55, 0.55
        q = lambda r, t, dz: W3(r, t, dz, sa)                    # noqa: E731
        quad(q(r0, a, z1), q(r1, a, z1), q(r1, b, z1), q(r0, b, z1))             # front
        quad(q(r0, a, z0), q(r0, b, z0), q(r1, b, z0), q(r1, a, z0))             # back
        quad(q(r1, a, z0), q(r1, b, z0), q(r1, b, z1), q(r1, a, z1))             # extrados
        quad(q(r0, a, z0), q(r0, a, z1), q(r0, b, z1), q(r0, b, z0))             # intrados
    arch = Mesh(v, f).paint(lambda vv: brown(vv[:, 1]))
    # spandrel ring with round openings (radial posts) outside the arch, under the girders
    rr = r1 + 3.6
    on = lambda r, t: (r * math.cos(t), YC_ARCH + r * math.sin(t), ZF(YC_ARCH + r * math.sin(t)) + 0.35)  # noqa: E731
    for t in np.linspace(0.12, math.pi - 0.12, 46):
        p0, p1 = on(r1, t), on(rr, t)
        if p1[1] > 42.3:
            continue
        acc.bar(p0, p1, 0.45, 0.6, (0, 0, 1))
    ring = []
    for t in np.linspace(0.05, math.pi - 0.05, 97):
        p = on(rr, t)
        ring.append(p if p[1] <= 42.3 else None)
    for a, b in zip(ring[:-1], ring[1:]):
        if a is not None and b is not None:
            acc.bar(a, b, 0.55, 0.7, (0, 0, 1))
    # frieze with the brackets, cornice (pinwheel: each covers one corner)
    s0, s1 = -35.05, 35.55
    solid.append(B(s0, s1, 51.6, 57.5, 35.05, 35.55, (0, 0, 0)))
    solid.append(B(-34.9, 35.95, 57.5, 58.3, 34.9, 35.95, (0, 0, 0)))
    for x in np.arange(-33.6, 34.0, 2.95):
        solid.append(B(x - 0.25, x + 0.25, 52.0, 57.5, 35.55, 35.85, (0, 0, 0)))
    # gallery: parapet, posts, rails, top beam
    solid.append(B(-34.8, 35.5, 58.3, 59.4, 35.05, 35.45, (0, 0, 0)))
    for x in np.arange(-33.0, 34.0, 3.0):
        acc.bar((x, 59.4, 35.25), (x, 63.9, 35.25), 0.3, 0.3)
    acc.bar((-35.0, 61.6, 35.25), (35.25, 61.6, 35.25), 0.18, 0.18)
    solid.append(B(-34.8, 35.5, 63.9, 64.5, 34.95, 35.55, (0, 0, 0)))
    for m in solid:
        m.paint(lambda vv: brown(vv[:, 1]))
    return merge([acc.mesh(), arch] + solid)


def first_floor_deck():
    sq = lambda a: [(-a, -a), (a, -a), (a, a), (-a, a)]                       # noqa: E731
    deck = extrude(sq(34.9), 56.8, Y1, brown(56.8), holes=[sq(14.0)], top_c=DECK, bottom=True)
    glass, roofs = [], []
    pav = [(-31.6, -18.2, -12.9, 12.9), (18.2, 31.6, -12.9, 12.9), (-13.1, 13.1, 16.5, 31.0)]
    for x0, x1, z0, z1 in pav:                                                 # OSM ways 308145258/259/239
        glass.append(B(x0, x1, Y1, 63.4, z0, z1, GLASS_C))
        roofs.append(B(x0 - 0.2, x1 + 0.2, 63.4, 64.0, z0 - 0.2, z1 + 0.2, DECK))
    return merge([deck] + roofs), merge(glass)


# ── Second floor ─────────────────────────────────────────────────────────────
def second_floor_face():
    acc = Acc()
    zb = W_LOW(107) + 0.7
    band(acc, 104.7, 109.9, lambda y: W_LOW(y) - 0.2, zb, 5.2, rows=1, rw=0.6, dw=0.45)
    band(acc, 101.8, 104.7, lambda y: W_LOW(y) - 0.2, zb + 0.15, 2.9, rows=1, rw=0.55)
    zg = W_LOW(96) - 3.0                                                        # set-back girder over the gap
    band(acc, 90.8, 101.8, lambda y: I_LOW(y) + 1.5, zg, 2.75, rows=4, rw=0.5, dw=0.3)
    solid = [B(-20.25, 20.7, 109.9, 115.6, 20.25, 20.7, (0, 0, 0)),
             B(-20.1, 21.05, 115.6, 116.3, 20.1, 21.05, (0, 0, 0)),
             B(-19.9, 20.9, 116.3, 117.2, 20.5, 20.9, (0, 0, 0))]
    for x in np.arange(-19.0, 19.5, 2.5):
        solid.append(B(x - 0.22, x + 0.22, 110.2, 115.6, 20.7, 20.98, (0, 0, 0)))
        acc.bar((x, 117.2, 20.7), (x, 119.6, 20.7), 0.16, 0.16)
    acc.bar((-20.6, 119.6, 20.7), (20.6, 119.6, 20.7), 0.2, 0.2)
    for m in solid:
        m.paint(lambda vv: brown(vv[:, 1]))
    fence = B(-20.4, 20.7, 117.2, 119.5, 20.68, 20.72, GRILLE)                  # anti-fall grille
    return merge([acc.mesh()] + solid), fence


def second_floor_deck():
    sq = lambda a: [(-a, -a), (a, -a), (a, a), (-a, a)]                       # noqa: E731
    deck = extrude(sq(20.1), 114.9, Y2, brown(114.9), top_c=DECK, bottom=True)
    kiosk = B(-5.5, 5.5, Y2, 119.8, -5.5, 5.5, brown(118))
    return merge([deck, kiosk])


# ── Upper shaft: four columns that merge at Y_JOIN, then one shaft ───────────
def upper_quarter():
    acc = Acc()
    ys_lo = breaks(Y2, Y_JOIN, lambda h: 0.95 * (W_UP(h) - U_UP(h)))
    ys_hi = breaks(Y_JOIN, Y3, lambda h: 0.95 * W_UP(h))
    A = lambda h: (W_UP(h), h, W_UP(h))                                       # noqa: E731
    Bq = lambda h: (U_UP(h), h, W_UP(h))                                      # noqa: E731
    Bm = lambda h: (-U_UP(h), h, W_UP(h))                                     # noqa: E731
    C = lambda h: (W_UP(h), h, U_UP(h))                                       # noqa: E731
    D = lambda h: (U_UP(h), h, U_UP(h))                                       # noqa: E731
    Ctr = lambda h: (0.0, h, W_UP(h))                                         # noqa: E731
    Edge = lambda h: (-W_UP(h), h, W_UP(h))                                   # noqa: E731
    # face +Z below the join: column strip, middle strip; above: two halves
    lattice(acc, A, Bq, ys_lo, "X1", (0, 0, 1), hw=0.55, dw=0.42)
    lattice(acc, Bq, Bm, ys_lo, "X1", (0, 0, 1), hw=0.5, dw=0.4)
    lattice(acc, A, Ctr, ys_hi, "X1", (0, 0, 1), hw=0.5, dw=0.4, first=False)
    lattice(acc, Ctr, Edge, ys_hi, "X1", (0, 0, 1), hw=0.5, dw=0.4, first=False)
    # inner faces of the (+X +Z) column, up to the join
    lattice(acc, Bq, D, ys_lo, "X1", (1, 0, 0), hw=0.45, dw=0.35, last=False)
    lattice(acc, C, D, ys_lo, "X1", (0, 0, 1), hw=0.45, dw=0.35, last=False)
    # chords: corner, inner chords of the face (+Z), centre rail above the join, inner corner
    ys = sorted(set(ys_lo + ys_hi + list(np.arange(Y2, Y3, 6.0)) + [Y3]))
    for a, b in zip(ys[:-1], ys[1:]):
        w = 1.55 - 0.65 * (a - Y2) / (Y3 - Y2)
        acc.bar(A(a), A(b), w, w, (1, 0, 0))
    yl = [y for y in ys if y <= Y_JOIN + 1e-6]
    yh = [y for y in ys if y >= Y_JOIN - 1e-6]
    acc.poly([Bq(y) for y in yl], 1.1, (0, 0, 1))
    acc.poly([D(y) for y in yl], 0.9, (1, 0, 0))
    acc.poly([Ctr(y) for y in yh], 0.95, (0, 0, 1))
    return acc.mesh()


def spine():
    """The lifts between the 2nd and 3rd floors run up the middle: a dark core seen through the lattice."""
    return B(-1.1, 1.1, 117.0, 275.0, -1.1, 1.1, (0, 0, 0)).paint(lambda vv: brown(vv[:, 1]) * 0.85)


# ── Head: 3rd floor, campanile, antennas ─────────────────────────────────────
def head():
    metal, glass, dark, white = [], [], [], []
    # corbelled floor under the 3rd floor
    metal.append(B(-8.0, 8.0, 273.8, 275.4, -8.0, 8.0, (0, 0, 0)))
    metal.append(B(-9.3, 9.3, 275.4, Y3, -9.3, 9.3, (0, 0, 0)))
    # enclosed level (glass) with mullions
    glass.append(B(-8.6, 8.6, Y3, 280.5, -8.6, 8.6, GLASS_C))
    for s in np.arange(-8.4, 8.5, 2.1):
        for (x, z, w, d) in ((s, 8.72, 0.28, 0.24), (8.72, s, 0.24, 0.28)):
            for sx in (1, -1):
                metal.append(B(sx * x - w / 2, sx * x + w / 2, Y3, 280.5, sx * z - d / 2, sx * z + d / 2, (0, 0, 0)))
    # upper deck and its cornice
    metal.append(B(-9.3, 9.3, 280.5, 282.6, -9.3, 9.3, (0, 0, 0)))
    # open upper level: posts, rail, grille
    for s in np.arange(-8.85, 9.0, 2.95):
        for sx in (1, -1):
            metal.append(B(s - 0.14, s + 0.14, 282.6, 285.6, sx * 9.0 - 0.14, sx * 9.0 + 0.14, (0, 0, 0)))
            metal.append(B(sx * 9.0 - 0.14, sx * 9.0 + 0.14, 282.6, 285.6, s - 0.14, s + 0.14, (0, 0, 0)))
    for sx in (1, -1):
        dark.append(B(-9.05, 9.05, 282.6, 285.4, sx * 9.0 - 0.03, sx * 9.0 + 0.03, GRILLE))
        dark.append(B(sx * 9.0 - 0.03, sx * 9.0 + 0.03, 282.6, 285.4, -8.95, 8.95, GRILLE))
    for sx in (1, -1):                                                          # top rail of the open terrace
        metal.append(B(-9.2, 9.2, 285.6, 285.85, sx * 9.0 - 0.16, sx * 9.0 + 0.16, (0, 0, 0)))
        metal.append(B(sx * 9.0 - 0.15, sx * 9.0 + 0.15, 285.62, 285.87, -8.84, 8.84, (0, 0, 0)))
    # campanile block with windows
    metal.append(B(-6.2, 6.2, 282.0, 293.0, -6.2, 6.2, (0, 0, 0)))
    for s in np.arange(-4.5, 4.6, 1.8):
        for sx in (1, -1):
            dark.append(B(s - 0.45, s + 0.45, 288.0, 291.6, sx * 6.2 - 0.04, sx * 6.2 + 0.04, DARK))
            dark.append(B(sx * 6.2 - 0.04, sx * 6.2 + 0.04, 288.0, 291.6, s - 0.45, s + 0.45, DARK))
    metal.append(B(-6.6, 6.6, 293.0, 293.6, -6.6, 6.6, (0, 0, 0)))
    # lantern (the beacon) and crown platform
    for sx in (1, -1):
        for sz in (1, -1):
            metal.append(B(sx * 3.4 - 0.25, sx * 3.4 + 0.25, 293.6, 298.4, sz * 3.4 - 0.25, sz * 3.4 + 0.25, (0, 0, 0)))
    glass.append(cyl(2.9, 2.9, 3.6, GLASS_C, 12).T(0, 294.2, 0))
    metal.append(cyl(3.2, 1.8, 1.0, (0, 0, 0), 12).T(0, 297.8, 0))
    metal.append(B(-4.6, 4.6, 298.4, 299.2, -4.6, 4.6, (0, 0, 0)))
    for x, z, w, h in ((2.6, 2.4, 1.6, 2.2), (-2.8, 1.5, 1.2, 1.6), (1.2, -3.0, 1.4, 2.6), (-2.2, -2.6, 1.8, 1.4)):
        white.append(B(x - w / 2, x + w / 2, 299.2, 299.2 + h, z - w / 2, z + w / 2, WHITE))   # equipment boxes
    # antenna mast: lattice 299.2–312, tube to 325.5, whip to 330
    acc = Acc()
    a = 1.2
    ys = list(np.linspace(299.2, 312.0, 8))
    for sx in (1, -1):
        for sz in (1, -1):
            acc.bar((sx * a, 299.2, sz * a), (sx * a * 0.8, 312.0, sz * a * 0.8), 0.26, 0.26, (1, 0, 0))
    for y0, y1 in zip(ys[:-1], ys[1:]):
        k0, k1 = a * (1 - 0.2 * (y0 - 299.2) / 12.8), a * (1 - 0.2 * (y1 - 299.2) / 12.8)
        for s in range(4):
            r = s * math.pi / 2
            ca, sa = math.cos(r), math.sin(r)
            q = lambda u, y, k: (ca * u * k + sa * k, y, -sa * u * k + ca * k)  # noqa: E731
            acc.bar(q(-1, y0, k0), q(1, y1, k1), 0.14, 0.14, (sa, 0, ca))
            acc.bar(q(-1, y1, k1), q(1, y1, k1), 0.16, 0.16, (sa, 0, ca))
    mast = Mesh(np.array(acc.v).reshape(-1, 3), np.array(acc.f).reshape(-1, 3), WHITE)
    white.append(mast)
    white.append(cyl(0.95, 0.8, 13.5, WHITE, 10).T(0, 312.0, 0))
    white.append(cyl(1.15, 1.15, 0.7, WHITE, 10).T(0, 318.5, 0))
    white.append(cyl(0.25, 0.12, 4.5, WHITE, 8).T(0, 325.5, 0))
    for y in (326.5, 328.0):
        white.append(B(-1.0, 1.0, y, y + 0.12, -0.06, 0.06, WHITE))
        white.append(B(-0.06, 0.06, y + 0.2, y + 0.32, -1.0, 1.0, WHITE))
    for m in metal:
        m.paint(lambda vv: brown(vv[:, 1]))
    return merge(metal), merge(glass), merge(dark), merge(white)


def main():
    out = ROOT / "models" / "Q243"
    out.mkdir(parents=True, exist_ok=True)
    leg_m, piers = leg()
    ff_face = first_floor_face()
    sf_face, sf_fence = second_floor_face()
    up = upper_quarter()
    ff_deck, ff_glass = first_floor_deck()
    h_metal, h_glass, h_dark, h_white = head()
    iron = merge([rot4(leg_m), rot4(ff_face), rot4(sf_face), rot4(up), ff_deck, second_floor_deck(), spine(), h_metal])
    glass = merge([ff_glass, h_glass])
    other = merge([rot4(piers), rot4(sf_fence), h_dark, h_white])
    parts = [("iron", iron, {"material": "iron"}), ("glass", glass, {"material": "glass"}), ("other", other)]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.8, materials={"glass": GLASS, "iron": METAL})
    allv = np.vstack([iron.v, glass.v, other.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
