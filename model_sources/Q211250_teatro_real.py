"""
Teatro Real (Q211250), Madrid — the opera house between the Plaza de Oriente and the Plaza de Isabel II, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: designed by Antonio López Aguado (1764–1831) from 1818, finished by Custodio Teodoro Moreno
(1780–1854) and opened in 1850; rebuilt inside 1925–1966; refurbished as an opera house again 1988–1997 by
José Manuel González-Valcárcel (died 1992) and Francisco Rodríguez de Partearroyo, the stage block led by Jaime
González-Valcárcel: that work rebuilt the roofs and the stage tower, which is most of what is seen from the air.
Rights: protected (the 19th-century façades are in the public domain, the roofs and the stage tower are of the 1990s).

Measurements (see meta.json for the sources):
- OpenStreetMap way 4328975 and its building:parts, in model x/z for heading 272° (the Plaza de Oriente front,
  z ≈ +54, faces west, towards the Royal Palace; +X points south, to Calle de Carlos III; the Plaza de Isabel II
  front is at z ≈ −52). Parts: the hexagonal outer range (relation 4099455, 30 m with a 5 m rounded roof), the
  inner block (27 m, flat), the auditorium roof (31 m, low gable), the stage tower (40 m, gabled, its ridge
  running east–west), a raised strip along its ridge (42 m), the arcaded porch on the Plaza de Oriente (10 m) and
  the shallow central projection of the east front.
- Elevations measured on near-frontal photos (Wikimedia Commons «Teatro Real-fachada plaza de Oriente.jpg»,
  «Fachada Este - Teatro Real - Madrid 01.jpg», «Teatro Real (5184908481).jpg», ±5 %): granite ground storey with
  the arcades to ≈ 9.5 m, main floor to ≈ 17.5 m, upper storey and cornice to ≈ 26 m; on the east and side
  fronts the top storey is a continuous attic colonnade; on the west front it has large windows between
  pilasters under a balustrade. The stage tower and the curved lead roofs on «Madrid cathedral 2020 - view from
  the dome.jpg»: a pedimented box with pilasters and a big arched window in each gable.

Axes (docs/MODEL_SPEC.md): +Y up, the Plaza de Oriente front faces +Z, origin on the ground at the centroid of
the OSM outline (40.4182254, −3.7105376).

    python model_sources/Q211250_teatro_real.py   -> models/Q211250/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import GLASS, METAL, Mesh, box, cap, cyl, merge, write_glb  # noqa: E402

rng = np.random.default_rng(1850)

GRAN = np.array((0.66, 0.66, 0.645))     # the grey granite of the ground storey
STONE = np.array((0.80, 0.80, 0.78))     # the light stone above
ORN = (0.86, 0.86, 0.83)
TOWER_C = np.array((0.86, 0.82, 0.74))   # the stage tower (warm stone colour on the photos)
LEAD = (0.50, 0.53, 0.56)                # lead/zinc roofs
FLATR = (0.60, 0.60, 0.58)
DARK = (0.13, 0.15, 0.17)
IRON = (0.12, 0.12, 0.13)
FOUND = -5.0

G1 = 9.5        # top of the ground storey
F1 = 17.5       # top of the main floor
AT0 = 20.6      # bottom of the attic colonnade
TOP = 26.0      # top of the cornice of the outer range
CREST = 30.0    # crest of the rounded roof of the outer range
INNER_H = 27.0

# relation 4099455 (outer range) and the parts, in model x/z (heading 272°, origin = centroid of way 4328975)
OUTER = [(-34.95, -18.95), (-18.75, -52.17), (-13.22, -51.99), (-13.21, -52.57), (16.29, -51.33), (16.27, -50.71), (23.16, -50.39),
         (37.26, -16.73), (16.58, 48.74), (14.77, 54.48), (12.62, 54.20), (10.61, 53.95), (8.29, 53.72), (5.98, 53.48), (3.35, 53.27),
         (0.62, 53.14), (-3.11, 53.14), (-6.42, 53.15), (-10.62, 53.30), (-14.53, 53.64), (-17.54, 53.94), (-20.25, 54.31), (-21.34, 48.87)]
INNER = [(-14.15, 47.59), (-12.82, 47.41), (-9.38, 47.17), (-6.79, 47.12), (-0.66, 47.12), (4.04, 47.41), (7.79, 47.69), (26.57, -10.59),
         (26.64, -14.39), (26.67, -16.32), (18.89, -34.99), (-16.35, -36.23), (-25.11, -19.52), (-25.12, -15.60), (-25.14, -7.02)]
STAGE = [(-21.64, 7.57), (-20.47, -18.60), (18.00, -16.86), (16.84, 9.32)]          # 40 m, gabled, ridge along z
RIDGE = [(-6.37, 5.03), (-5.50, -14.65), (2.86, -14.27), (1.98, 5.41)]              # 42 m
AUDIT = [(-18.24, 25.84), (-16.98, 2.51), (14.49, 4.25), (13.23, 27.57)]            # 31 m, low gable
PORCH_BACK = [(-20.25, 54.31), (-17.54, 53.94), (-14.53, 53.64), (-10.62, 53.30), (-6.42, 53.15), (-3.11, 53.14), (0.62, 53.14),
              (3.35, 53.27), (5.98, 53.48), (8.29, 53.72), (10.61, 53.95), (12.62, 54.20), (14.77, 54.48)]
PORCH_FRONT = [(-19.05, 60.29), (-16.75, 59.94), (-13.69, 59.48), (-10.84, 59.25), (-7.06, 58.96), (-3.76, 58.94), (0.61, 58.94),
               (4.73, 59.17), (8.96, 59.65), (12.94, 60.29)]
PORCH_H = 9.5


def jit(c, k=0.012):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, k)), 0, 1))


def ccw(ring):
    a = 0.0
    for i in range(len(ring)):
        (x0, z0), (x1, z1) = ring[i - 1], ring[i]
        a += x0 * z1 - x1 * z0
    return list(ring) if a > 0 else list(ring)[::-1]


def inside(pt, ring):
    x, z = pt
    ins = False
    for i in range(len(ring)):
        (xi, zi), (xj, zj) = ring[i], ring[i - 1]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            ins = not ins
    return ins


def seg_dist(P, ring):
    """Distance of points P (N,2) to the boundary of ring, and the nearest boundary points."""
    R = np.array(ring, float)
    A, B = R, np.roll(R, -1, 0)
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    Q = A[None] + t[..., None] * AB[None]
    d = np.linalg.norm(P[:, None, :] - Q, axis=2)
    k = d.argmin(1)
    return d[np.arange(len(P)), k], Q[np.arange(len(P)), k]


class Run:
    """A straight wall from a to b with outward normal n: local (t along, y up, d out)."""

    def __init__(self, a, b, n):
        self.a, self.b = np.array(a, float), np.array(b, float)
        self.L = float(np.linalg.norm(self.b - self.a))
        self.u = (self.b - self.a) / max(self.L, 1e-9)
        self.n = np.array(n, float)

    def put(self, m):
        u3, n3 = np.array([self.u[0], 0, self.u[1]]), np.array([self.n[0], 0, self.n[1]])
        V = m.v
        W = np.array([self.a[0], 0, self.a[1]]) + V[:, :1] * u3 + V[:, 1:2] * np.array([0, 1, 0]) + V[:, 2:3] * n3
        f = m.f if (self.u[0] * self.n[1] - self.u[1] * self.n[0]) > 0 else m.f[:, ::-1]
        return Mesh(W, f, m.c)

    def box(self, t0, t1, y0, y1, d0, d1, c):
        return self.put(box(t1 - t0, y1 - y0, d1 - d0, c).T((t0 + t1) / 2, y0, (d0 + d1) / 2))

    def face(self, t0, t1, y0, y1, c, d=0.0):
        V = [(t0, y0, d), (t1, y0, d), (t1, y1, d), (t0, y1, d)]
        return self.put(Mesh(V, [(0, 1, 2), (0, 2, 3)], c))

    def col(self, t, y0, y1, r, c, d):
        return self.put(cyl(r, r * 0.88, y1 - y0, c, seg=8).T(t, y0, d))

    def arch(self, t, w, y0, ys, c, d, seg=8):
        """A flat arched opening (dark) of width w, springing at ys."""
        r = w / 2
        V = [(t - r, y0, d), (t + r, y0, d)] + [(t + r * math.cos(a), ys + r * math.sin(a), d) for a in np.linspace(0, math.pi, seg + 1)]
        F = [(0, 1, 2)] + [(0, 2 + i, 3 + i) for i in range(seg)]
        return self.put(Mesh(V, F, c))


def runs(ring, hole=False):
    P = ccw(ring)
    if hole:
        P = P[::-1]
    out = []
    for i in range(len(P)):
        a, b = np.array(P[i], float), np.array(P[(i + 1) % len(P)], float)
        L = float(np.linalg.norm(b - a))
        if L < 0.05:
            continue
        u = (b - a) / L
        n = np.array([u[1], -u[0]])           # outward for ccw (x, z)
        out.append(Run(a, b, n))
    return out


def outer_range():
    stone, dark, orn = [], [], []
    for r in runs(OUTER):
        west = r.n[1] > 0.85 and r.a[1] > 45          # the Plaza de Oriente front (behind the porch)
        east = r.n[1] < -0.85 and r.a[1] < -45        # the Plaza de Isabel II front
        top = TOP + (1.6 if west else 0.0)
        stone.append(r.face(0, r.L, FOUND, G1, jit(GRAN)))
        stone.append(r.face(0, r.L, G1, top, jit(STONE)))
        stone.append(r.box(0, r.L, G1 - 0.3, G1 + 0.35, -0.05, 0.4, jit(GRAN)))        # string course
        stone.append(r.box(0, r.L, F1 - 0.3, F1 + 0.3, -0.05, 0.35, jit(ORN)))
        stone.append(r.box(0, r.L, TOP - 0.9, TOP, -0.05, 0.85, jit(ORN)))            # main cornice
        if r.L < 1.0:
            continue
        if west:
            continue                                                                 # done in west_front()
        nb = max(1, int(r.L // 4.3))
        sp = r.L / nb
        for i in range(nb):
            t = sp * (i + 0.5)
            if east and abs((r.a + r.u * t)[0] - 1.5) < 13.5:
                continue                                                             # done in east_front()
            dark.append(r.face(t - 0.7, t + 0.7, 2.6, 5.6, DARK, d=0.03))            # ground storey
            dark.append(r.face(t - 0.85, t + 0.85, 10.6, 14.6, DARK, d=0.03))        # main floor
            stone.append(r.box(t - 1.25, t + 1.25, 14.7, 15.2, -0.02, 0.35, jit(ORN)))  # pediment (flat)
            stone.append(r.box(t - 1.2, t + 1.2, 10.3, 10.5, 0.0, 0.7, IRON))          # balcony
            dark.append(r.face(t - 0.6, t + 0.6, 18.2, 19.8, DARK, d=0.03))          # upper storey
        # attic colonnade: a dark recess with columns
        dark.append(r.face(0.4, r.L - 0.4, AT0 + 0.4, TOP - 1.0, DARK, d=0.02))
        nc = max(1, int(r.L // 2.6))
        for i in range(nc + 1):
            t = 0.4 + (r.L - 0.8) * i / nc
            stone.append(r.col(t, AT0 + 0.3, TOP - 0.9, 0.3, jit(ORN), 0.15))
        stone.append(r.box(0, r.L, AT0, AT0 + 0.35, -0.02, 0.45, jit(ORN)))
    return stone, dark


def east_front():
    """Plaza de Isabel II: five arches between columns, five giant-order bays above, the attic colonnade."""
    stone, dark = [], []
    r = Run((-13.22, -52.57), (16.29, -51.33), (0, 0))
    r.u = (r.b - r.a) / r.L
    r.n = np.array([r.u[1], -r.u[0]])
    if r.n[1] > 0:
        r.n = -r.n
    # projecting central body over the five arches
    stone.append(r.box(-0.5, r.L + 0.5, FOUND, TOP + 0.2, -0.6, 0.3, jit(STONE)))
    stone.append(r.box(-0.5, r.L + 0.5, FOUND, G1, -0.6, 0.35, jit(GRAN)))
    stone.append(r.box(-0.7, r.L + 0.7, TOP - 0.9, TOP + 0.4, -0.6, 1.0, jit(ORN)))
    stone.append(r.box(-0.7, r.L + 0.7, G1 - 0.2, G1 + 0.5, -0.6, 0.8, jit(ORN)))
    w = r.L / 5
    for i in range(5):
        t = w * (i + 0.5)
        dark.append(r.arch(t, 3.6, 0.0, 4.6, DARK, 0.37))
        dark.append(r.face(t - 1.3, t + 1.3, 11.0, 16.2, DARK, d=0.33))
    for i in range(6):
        t = w * i
        stone.append(r.col(t, 0.0, G1 - 0.2, 0.42, jit(GRAN), 0.75))
        stone.append(r.col(t, G1 + 0.5, F1 + 2.4, 0.42, jit(ORN), 0.75))
    stone.append(r.box(-0.7, r.L + 0.7, F1 + 2.4, F1 + 3.1, -0.6, 1.0, jit(ORN)))
    dark.append(r.face(4.0, r.L - 4.0, AT0 + 0.8, TOP - 1.0, DARK, d=0.32))
    for i in range(14):
        t = 4.0 + (r.L - 8.0) * i / 13
        stone.append(r.col(t, AT0 + 0.7, TOP - 0.9, 0.28, jit(ORN), 0.5))
    return stone, dark


def west_front():
    """Plaza de Oriente: the arcaded porch carrying a balcony, two storeys between pilasters, a balustrade."""
    stone, dark, iron = [], [], []
    a, b = np.array(PORCH_BACK[0]), np.array(PORCH_BACK[-1])
    r = Run(a, b, (0, 1))
    r.n = np.array([-r.u[1], r.u[0]])
    if r.n[1] < 0:
        r.n = -r.n
    # the porch (10 m part): a solid block with five arches on the front
    porch = PORCH_BACK + PORCH_FRONT[::-1]
    porch = ccw(porch)
    stone.append(cap(porch, PORCH_H, jit(STONE)))
    for rr in runs(porch):
        if rr.n[1] < -0.5:
            continue
        stone.append(rr.face(0, rr.L, FOUND, PORCH_H, jit(GRAN)))
        stone.append(rr.box(0, rr.L, PORCH_H - 0.7, PORCH_H, -0.02, 0.5, jit(ORN)))
        stone.append(rr.box(0, rr.L, PORCH_H + 1.2, PORCH_H + 1.3, -0.12, 0.0, IRON))   # balcony railing (rail)
        for k in range(int(rr.L // 2.2) + 1):
            stone.append(rr.box(k * rr.L / max(1, int(rr.L // 2.2)) - 0.25, k * rr.L / max(1, int(rr.L // 2.2)) + 0.25, PORCH_H, PORCH_H + 1.3, -0.4, 0.0, jit(ORN)))
    fr = Run(np.array(PORCH_FRONT[0]), np.array(PORCH_FRONT[-1]), (0, 1))
    if fr.u[0] < 0:
        fr = Run(np.array(PORCH_FRONT[-1]), np.array(PORCH_FRONT[0]), (0, 1))
    zf = lambda t: np.interp(fr.a[0] + fr.u[0] * t, [p[0] for p in PORCH_FRONT], [p[1] for p in PORCH_FRONT])  # noqa: E731
    w = fr.L / 5
    for i in range(5):
        t = w * (i + 0.5)
        dz = float(zf(t) - (fr.a[1] + fr.u[1] * t))
        dark.append(fr.arch(t, 3.9, 0.0, 5.0, DARK, dz + 0.05))
    for i in range(6):
        t = w * i
        dz = float(zf(t) - (fr.a[1] + fr.u[1] * t))
        stone.append(fr.col(t, 0.0, PORCH_H - 0.7, 0.45, jit(GRAN), dz + 0.5))
    # pilasters, oculi and windows of the two upper storeys, on the chord of the arc
    nbays = 6
    for i in range(nbays + 1):
        t = r.L * i / nbays
        dz = float(np.interp(r.a[0] + r.u[0] * t, [p[0] for p in PORCH_BACK], [p[1] for p in PORCH_BACK]) - (r.a[1] + r.u[1] * t))
        stone.append(r.box(t - 0.55, t + 0.55, PORCH_H + 1.2, TOP + 0.6, dz - 0.1, dz + 0.45, jit(ORN)))
    for i in range(nbays):
        t = r.L * (i + 0.5) / nbays
        dz = float(np.interp(r.a[0] + r.u[0] * t, [p[0] for p in PORCH_BACK], [p[1] for p in PORCH_BACK]) - (r.a[1] + r.u[1] * t))
        dark.append(r.face(t - 0.9, t + 0.9, PORCH_H + 1.3, PORCH_H + 4.6, DARK, d=dz + 0.04))      # balcony doors
        dark.append(r.put(cyl(0.45, 0.45, 0.05, DARK, seg=10).R(math.pi / 2, 0, 0).T(t, 15.6, dz + 0.06)))  # oculus
        dark.append(r.face(t - 1.25, t + 1.25, 19.8, 23.9, DARK, d=dz + 0.04))      # upper windows
    # balustrade and posts along the top
    stone.append(r.box(-0.5, r.L + 0.5, TOP + 0.6, TOP + 1.6, -0.6, 0.1, jit(ORN)))
    for i in range(nbays + 1):
        t = r.L * i / nbays
        stone.append(r.box(t - 0.5, t + 0.5, TOP + 1.6, TOP + 2.5, -0.6, 0.2, jit(ORN)))
    return [m for m in stone if m is not None], [m for m in dark if m is not None]


def range_roof():
    """The rounded lead roof of the outer range: from the cornice (26 m) up to its crest (30 m) and down to the
    flat roof of the inner block (27 m), whatever the width of the range."""
    O, I = ccw(OUTER), ccw(INNER)
    xs = np.arange(-36.0, 38.5, 0.8)
    zs = np.arange(-53.5, 55.5, 0.8)
    X, Z = np.meshgrid(xs, zs)
    P = np.stack([X.ravel(), Z.ravel()], 1)
    nx = len(xs)
    F = []
    for j in range(len(zs) - 1):
        for i in range(nx - 1):
            a = j * nx + i
            F += [(a, a + nx, a + 1), (a + 1, a + nx, a + nx + 1)]
    F = np.array(F)
    cen = P[F].mean(1)
    keep = np.array([inside(tuple(c), O) and not inside(tuple(c), I) for c in cen])
    F = F[keep]
    used = np.unique(F)
    # snap vertices outside the range onto its nearest edge
    Q = P.copy()
    for k in used:
        p = tuple(P[k])
        if not inside(p, O):
            _, q = seg_dist(P[k:k + 1], O); Q[k] = q[0]
        elif inside(p, I):
            _, q = seg_dist(P[k:k + 1], I); Q[k] = q[0]
    d_o, _ = seg_dist(Q, O)
    d_i, _ = seg_dist(Q, I)
    s = d_o / np.maximum(d_o + d_i, 1e-6)
    W = d_o + d_i
    rise = np.clip(W / 12.0, 0.3, 1.0) * (CREST - TOP)
    Y = TOP + (INNER_H - TOP) * s + rise * np.sin(np.pi * s) ** 0.7
    V = np.stack([Q[:, 0], Y + 0.05, Q[:, 1]], 1)
    remap = np.full(len(V), -1)
    remap[used] = np.arange(len(used))
    return Mesh(V[used], remap[F][:, ::-1], LEAD)


def gable_block(ring, wall_top, ridge, c, roof_c, ridge_axis="z", pilasters=True):
    """A block on a 4-point outline with a gabled roof whose ridge runs along the model z axis (as OSM says)."""
    R = np.array(ring, float)
    ctr = R.mean(0)
    # local axes from the outline: e1 ≈ x, e2 ≈ z
    e1 = R[2] - R[1]; e1 /= np.linalg.norm(e1)
    e2 = np.array([-e1[1], e1[0]])
    loc = (R - ctr) @ np.stack([e1, e2], 1)
    hx, hz = np.abs(loc[:, 0]).max(), np.abs(loc[:, 1]).max()
    ang = math.atan2(e1[1], e1[0])
    parts = []
    body = box(2 * hx, wall_top - 20.0, 2 * hz, jit(c)).T(0, 20.0, 0)
    parts.append(body)
    # roof: two slopes falling to ±x, gables at ±z
    V = [(-hx - 0.4, wall_top, -hz - 0.4), (0, ridge, -hz - 0.4), (0, ridge, hz + 0.4), (-hx - 0.4, wall_top, hz + 0.4),
         (hx + 0.4, wall_top, -hz - 0.4), (hx + 0.4, wall_top, hz + 0.4)]
    roof = Mesh(V, [(0, 2, 1), (0, 3, 2), (4, 1, 2), (4, 2, 5)], roof_c)
    gab = Mesh([(-hx, wall_top, hz), (hx, wall_top, hz), (0, ridge - 0.3, hz), (-hx, wall_top, -hz), (hx, wall_top, -hz), (0, ridge - 0.3, -hz)],
               [(0, 1, 2), (3, 5, 4)], jit(c))
    parts += [roof, gab]
    return parts, (ctr, ang, hx, hz)


def place(m, ctr, ang):
    """local (x along e1, z along e2) -> model: rotate about y so +x goes to e1."""
    return m.R(0, -ang, 0).T(ctr[0], 0, ctr[1])


def stage_tower():
    stone, dark, roof = [], [], []
    parts, (ctr, ang, hx, hz) = gable_block(STAGE, 35.0, 40.0, TOWER_C, LEAD)
    wall_top, ridge = 35.0, 40.0
    loc_st = [parts[0], parts[2]]
    loc_rf = [parts[1]]
    loc_dk = []
    # cornice, pilasters on all four sides, the raking cornices of the pediments, the arched windows
    loc_st.append(box(2 * hx + 1.0, 0.9, 2 * hz + 1.0, jit(ORN)).T(0, wall_top - 0.9, 0))
    for sgn in (-1, 1):
        for k in range(9):
            x = -hx + 2 * hx * k / 8
            loc_st.append(box(1.0, wall_top - 0.9 - 27.5, 0.35, jit(ORN)).T(x, 27.5, sgn * (hz + 0.15)))
        for k in range(7):
            z = -hz + 2 * hz * k / 6
            loc_st.append(box(0.35, wall_top - 0.9 - 27.5, 1.0, jit(ORN)).T(sgn * (hx + 0.15), 27.5, z))
        # raking cornices
        L = math.hypot(hx, ridge - wall_top)
        a = math.atan2(ridge - wall_top, hx)
        for sx in (-1, 1):
            rc = box(L + 0.6, 0.6, 0.8, jit(ORN)).T(0, -0.3, 0).R(0, 0, sx * -a if sx > 0 else a)
            rc.T(sx * hx / 2, wall_top + (ridge - wall_top) / 2, sgn * (hz + 0.1))
            loc_st.append(rc)
        # big arched (thermal) window in the gable wall
        r = 3.2
        V = [(-r, 29.5, 0), (r, 29.5, 0)] + [(r * math.cos(t), 33.0 + r * math.sin(t), 0) for t in np.linspace(0, math.pi, 11)]
        F = [(0, 1, 2)] + [(0, 2 + i, 3 + i) for i in range(10)]
        w = Mesh(V, F, DARK).T(0, 0, sgn * (hz + 0.05))
        if sgn < 0:
            w.f = w.f[:, ::-1]
        loc_dk.append(w)
        # small windows along the long sides
        for k in range(8):
            x = -hx + 2 * hx * (k + 0.5) / 8
            loc_dk.append(box(1.1, 2.4, 0.06, DARK).T(x, 30.0, sgn * (hz + 0.04)))
    # the raised strip along the ridge (42 m part): a low lantern
    R2 = np.array(RIDGE, float)
    c2 = R2.mean(0)
    e1 = np.array([math.cos(ang), math.sin(ang)])
    loc2 = (R2 - c2) @ np.stack([e1, np.array([-e1[1], e1[0]])], 1)
    w2, d2 = np.abs(loc2[:, 0]).max(), np.abs(loc2[:, 1]).max()
    off = (c2 - ctr) @ np.stack([e1, np.array([-e1[1], e1[0]])], 1)
    loc_rf.append(box(w2 * 0.55, 1.6, 2 * d2, LEAD).T(off[0], ridge - 0.6, off[1]))
    loc_rf.append(box(w2 * 0.35, 0.5, 2 * d2 - 0.4, LEAD).T(off[0], ridge + 1.0, off[1]))
    stone += [place(m, ctr, ang) for m in loc_st]
    roof += [place(m, ctr, ang) for m in loc_rf]
    dark += [place(m, ctr, ang) for m in loc_dk]
    return stone, dark, roof


def auditorium():
    parts, (ctr, ang, hx, hz) = gable_block(AUDIT, 30.4, 31.0, STONE, LEAD)
    return [place(parts[0], ctr, ang), place(parts[2], ctr, ang)], [place(parts[1], ctr, ang)]


def main():
    out = ROOT / "models" / "Q211250"
    out.mkdir(parents=True, exist_ok=True)
    stone, dark, roof, flat = [], [], [], []
    s, d = outer_range(); stone += s; dark += d
    s, d = east_front(); stone += s; dark += d
    s, d = west_front(); stone += s; dark += d
    # inner block: walls up to the flat roof (seen above the curved roof on its inner side), flat roof
    for r in runs(INNER):
        stone.append(r.face(0, r.L, TOP - 1.0, INNER_H + 0.6, jit(STONE)))
    flat.append(cap(INNER, INNER_H + 0.02, jit(FLATR)))
    for (x, z, w, dd) in ((-12.0, 38.0, 5.0, 4.0), (8.0, 36.0, 4.0, 3.0), (-15.0, -28.0, 6.0, 4.0), (12.0, -29.0, 5.0, 3.5)):
        flat.append(box(w, 2.2, dd, jit((0.70, 0.70, 0.68))).T(x, INNER_H, z))          # plant rooms
    roof.append(range_roof())
    s, d, r = stage_tower(); stone += s; dark += d; roof += r
    s, r = auditorium(); stone += s; roof += r
    m_stone, m_dark, m_roof, m_flat = merge(stone), merge(dark), merge(roof), merge(flat)
    tris, size = write_glb(out / "model.glb", [("teatro_real", m_stone), ("flat_roofs", m_flat),
                                               ("lead_roofs", m_roof, {"material": "lead"}),
                                               ("windows", m_dark, {"material": "glass"})],
                           roughness=0.85, materials={"glass": GLASS, "lead": {"metallic": 0.45, "roughness": 0.5}})
    allv = np.vstack([m.v for m in (m_stone, m_dark, m_roof, m_flat)])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
