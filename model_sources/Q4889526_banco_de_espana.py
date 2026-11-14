"""
Banco de España (Q4889526), Madrid — the headquarters at Cibeles, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Eduardo Adaro (1848–1906) and Severiano Sainz de la Lastra (died 1884), 1884–1891, the eclectic
block at the corner of Paseo del Prado and Calle de Alcalá; extended along Alcalá by José Yárnoz Larrosa
(1927–1934, died 1966), along Paseo del Prado / Los Madrazo by José Yárnoz Orcoyen (1969–1975), and the block
was closed at Alcalá / Marqués de Cubas by Rafael Moneo (2003–2006, living), all keeping the exterior lines of
the 1891 building. Because of the later architects the whole is still under copyright (status «protected»).

Measurements (see meta.json for the sources):
- OpenStreetMap relation 2614014 «Banco de España» (outline + 6 courtyards, ±0.5 m): 167 × 140 m. The Cibeles
  corner is a 12.4 m chamfer whose outward normal points to 44.3°: that is the main façade and the heading,
  so +Z points from the block to the Plaza de Cibeles, and the Paseo del Prado front (160.3 m, facing 102°) is
  on −X, the Calle de Alcalá front (61.0 + 62.6 m, facing 348° / 344°) on +X.
- building:parts: the inner 6–7 storey block (south-west of the middle), the octagonal glazed roof of the
  main hall and three smaller glazed roofs (roof:shape mansard, roof:material glass).
- PNOA orthophoto (IGN, CC BY 4.0): mansard ranges about 12 m deep along the streets, the corner pavilion, the
  big glass pyramid, the courtyards.
- Heights and bays measured on photos (Wikimedia Commons «Banco de España (Madrid) 01/02/06/07.jpg», «Madrid.
  Bank of Spain building. Alcalá street…jpg»), ±5 %: rusticated ground storey 6.8 m, main floor with arched
  windows between engaged columns, second floor, cornice 17.0 m, balustrade 18.2, mansards with dormers and
  iron cresting to 21.4 (pavilions 22.0), the corner crest with the clock and the gilded globe on its iron
  stand (top ≈ 24.6 m), the slate dome over the Paseo del Prado pavilion (spire ≈ 27.5 m). Bays ≈ 4.25 m.
Simplified: the plan inside the outline (one wall line, flat roofs inside, 6–7 storey inner block from its
parts), the sculpture and the crests (stylised), the window rhythm of the later wings (same as the old ones).

Axes (docs/MODEL_SPEC.md): +Y up, the Cibeles chamfer faces +Z, +X points north-west (along Alcalá), origin on
the ground at the centroid of the OSM outline.

    python model_sources/Q4889526_banco_de_espana.py   -> models/Q4889526/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import GLASS, METAL, Mesh, box, cap, cyl, lathe, merge, offset, ring_area, tube, walls, write_glb  # noqa: E402

rng = np.random.default_rng(1891)

STONE = np.array((0.86, 0.85, 0.815))     # the white limestone of the upper floors
GRAN = np.array((0.63, 0.63, 0.62))       # the grey granite of the rusticated ground storey
ORN = (0.91, 0.90, 0.87)                  # sculpture and carved stone
SLATE = (0.31, 0.34, 0.39)
FLAT = (0.64, 0.64, 0.62)
DARK = (0.11, 0.12, 0.14)
IRON = (0.13, 0.14, 0.15)
GOLD = (0.83, 0.66, 0.24)
GLASS_C = (0.45, 0.56, 0.60)
FOUND = -5.0

G1 = 6.8          # top of the ground storey
TOP = 17.0        # top of the cornice
BAL = 18.2        # top of the balustrade

# OSM relation 2614014 in model x/z (heading 44.3°, origin = centroid 40.4182604, −3.6948029)
OUTLINE = [(63.77, -38.58), (64.81, -39.47), (77.06, -27.11), (80.14, -27.77), (82.70, -12.41), (79.60, -12.18),
           (74.03, -2.37), (43.09, 52.09), (9.16, 102.73), (-3.24, 102.73), (-88.96, -32.75), (-57.61, -52.77),
           (-53.55, -46.60), (-48.02, -50.23), (-50.22, -53.59), (-50.39, -54.25), (-50.07, -54.99), (-7.55, -82.72),
           (-7.44, -83.37), (-5.02, -83.30), (-5.16, -82.82), (0.91, -82.54), (1.07, -83.02), (3.23, -83.07),
           (3.20, -82.64)]
COURTS = [
    [(-21.70, -51.94), (-17.31, -45.00), (-24.50, -40.50), (-23.89, -39.37), (-27.01, -37.38), (-26.32, -36.28), (-27.48, -35.58),
     (-19.09, -22.48), (-18.05, -23.13), (-17.44, -22.08), (-17.36, -21.93), (-18.60, -21.05), (-17.40, -19.07), (-12.71, -22.03),
     (-13.23, -22.98), (-6.34, -27.40), (-7.02, -28.36), (-6.91, -28.44), (4.64, -24.28), (-8.51, -0.02), (-16.03, 4.77),
     (-36.69, -27.72), (-37.81, -28.91), (-39.94, -29.95), (-42.59, -30.35), (-45.41, -29.47), (-47.18, -27.41), (-48.20, -24.33),
     (-50.41, -24.81), (-55.76, -33.22), (-49.95, -36.85), (-49.56, -36.53), (-48.36, -37.23), (-48.38, -37.76), (-49.03, -38.78),
     (-49.44, -38.71), (-49.77, -39.23), (-43.49, -43.28), (-41.30, -40.12), (-39.97, -39.98)],
    [(-3.84, -30.44), (-6.42, -34.53), (-6.72, -35.01), (-8.40, -37.66), (-3.57, -40.76), (-2.49, -39.28), (-1.86, -38.98),
     (-1.06, -38.94), (-0.23, -39.21), (0.45, -39.91), (0.73, -40.85), (0.51, -41.56), (-2.90, -47.16), (-3.45, -47.93),
     (-4.16, -48.42), (-4.99, -48.53), (-5.81, -48.47), (-6.50, -47.91), (-6.96, -47.04), (-6.78, -46.27), (-6.66, -45.74),
     (-6.14, -44.78), (-10.91, -41.69), (-13.54, -45.86), (-13.97, -45.65), (-19.07, -53.69), (-7.41, -61.25), (-5.61, -59.84),
     (-4.50, -59.81), (0.21, -59.66), (2.36, -60.98), (52.61, -24.39), (47.60, -16.36), (16.66, -33.82), (14.71, -34.06),
     (12.36, -33.82), (10.23, -32.99), (8.11, -30.85), (6.09, -26.89)],
    [(6.20, 24.06), (-0.83, 13.53), (-7.79, 17.80), (1.40, 32.36)],
    [(-8.89, 54.38), (-10.01, 53.59), (-14.88, 56.64), (-9.83, 64.48), (-8.96, 64.22)],
    [(21.46, 56.75), (16.48, 53.54), (15.67, 53.96), (15.64, 64.17), (16.38, 64.52)],
    [(-6.19, 50.27), (-2.83, 48.09), (-5.54, 43.60), (-9.01, 45.66)],
]
# building:parts (OSM ways 5959282xx and 13470391xx): the inner block (levels) and the glazed roofs
INNER = [  # (levels, outline)
    (6, [(-22.37, -34.46), (-24.77, -38.43), (-15.20, -44.43), (-12.76, -40.51), (-10.91, -41.69), (-13.54, -45.86), (-13.97, -45.65),
         (-16.65, -43.92), (-23.89, -39.37), (-27.01, -37.38), (-26.32, -36.28), (-20.18, -26.54), (-19.98, -26.65), (-24.16, -33.22)]),
    (7, [(-8.40, -37.66), (-3.57, -40.76), (-2.49, -39.28), (-1.86, -38.98), (-1.06, -38.94), (-0.23, -39.21), (0.45, -39.91),
         (0.73, -40.85), (0.51, -41.56), (-2.90, -47.16), (-3.45, -47.93), (-4.16, -48.42), (-4.99, -48.53), (-5.81, -48.47),
         (-6.50, -47.91), (-6.96, -47.04), (-6.78, -46.27), (-6.66, -45.74), (-6.14, -44.78), (-10.91, -41.69), (-12.76, -40.51),
         (-13.18, -40.24), (-13.90, -41.29), (-22.44, -35.88), (-21.83, -34.83), (-22.37, -34.46), (-24.16, -33.22), (-19.98, -26.65),
         (-17.47, -28.21), (-16.83, -27.18), (-8.43, -32.48), (-9.06, -33.49), (-6.72, -35.01)]),
    (7, [(-12.76, -40.51), (-15.20, -44.43), (-24.77, -38.43), (-22.37, -34.46), (-21.83, -34.83), (-22.44, -35.88), (-13.90, -41.29),
         (-13.18, -40.24)]),
    (7, [(-8.43, -32.48), (-16.83, -27.18), (-17.47, -28.21), (-19.98, -26.65), (-20.18, -26.54), (-19.93, -26.07), (-17.91, -27.37),
         (-15.41, -23.40), (-14.09, -24.26), (-13.88, -23.91), (-7.02, -28.36), (-6.91, -28.44), (-5.73, -29.22), (-8.34, -33.27),
         (-6.42, -34.53), (-6.72, -35.01), (-9.06, -33.49)]),
    (7, [(-13.88, -23.91), (-14.09, -24.26), (-15.41, -23.40), (-17.44, -22.08), (-17.36, -21.93), (-18.60, -21.05), (-17.40, -19.07),
         (-12.71, -22.03), (-13.23, -22.98)]),
    (6, [(-5.73, -29.22), (-3.84, -30.44), (-6.42, -34.53), (-8.34, -33.27)]),
    (6, [(-15.41, -23.40), (-17.91, -27.37), (-19.93, -26.07), (-18.05, -23.13), (-17.44, -22.08)]),
]
HALL = [(10.16, 8.91), (27.51, 18.61), (35.23, 16.49), (44.82, -0.92), (42.72, -8.70), (25.30, -18.21), (17.57, -16.07), (7.93, 1.30)]
SKY = [[(-27.61, 17.60), (-35.88, 4.78), (-47.73, 12.45), (-39.46, 25.27)],
       [(-10.58, 43.91), (-18.85, 31.10), (-30.70, 38.76), (-22.43, 51.58)],
       [(10.82, 75.76), (-4.96, 76.02), (-5.19, 57.96), (10.60, 57.72)]]

P = OUTLINE


def jit(c, k=0.015):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, k)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def inside(pt, ring):
    x, z = pt
    ins = False
    for i in range(len(ring)):
        (xi, zi), (xj, zj) = ring[i], ring[i - 1]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            ins = not ins
    return ins


class Side:
    """One straight wall from a to b: t along it (metres from a), d out of the building, y up."""

    def __init__(self, a, b, ring=OUTLINE, hole=False):
        """ring: the outline the wall belongs to; hole: it is a courtyard (the outside is inside the ring)."""
        self.a, self.b = np.array(a, float), np.array(b, float)
        self.L = float(np.linalg.norm(self.b - self.a))
        self.u = (self.b - self.a) / self.L
        n = np.array([self.u[1], -self.u[0]])
        probe = inside(tuple((self.a + self.b) / 2 + n * 0.05), ring)
        self.n = n if probe == hole else -n

    def p(self, t, d=0.0):
        return self.a + self.u * t + self.n * d

    def put(self, m):
        """m built in local (t, y, d) coordinates -> model coordinates."""
        u3, n3 = np.array([self.u[0], 0, self.u[1]]), np.array([self.n[0], 0, self.n[1]])
        V = m.v
        W = np.array([self.a[0], 0, self.a[1]]) + V[:, :1] * u3 + V[:, 1:2] * np.array([0, 1, 0]) + V[:, 2:3] * n3
        f = m.f if (self.u[0] * self.n[1] - self.u[1] * self.n[0]) > 0 else m.f[:, ::-1]
        return Mesh(W, f, m.c)

    def box(self, t0, t1, y0, y1, d0, d1, c):
        return self.put(B(t0, t1, y0, y1, d0, d1, c))


def arch_face(w, ys, c, seg=12):
    r = w / 2
    V = [(-r, 0, 0), (r, 0, 0)] + [(r * math.cos(a), ys + r * math.sin(a), 0) for a in np.linspace(0, math.pi, seg + 1)]
    F = [(0, 1, 2)] + [(0, 2 + i, 3 + i) for i in range(seg)]
    return Mesh(V, F, c)


def figure(h, c):
    prof = [(0.2 * h, 0), (0.2 * h, 0.25 * h), (0.14 * h, 0.5 * h), (0.11 * h, 0.66 * h), (0.06 * h, 0.72 * h), (0, 0.74 * h)]
    head = lathe([(0, -0.08 * h), (0.07 * h, -0.05 * h), (0.075 * h, 0), (0.07 * h, 0.05 * h), (0, 0.08 * h)], c, 8).T(0, 0.82 * h, 0)
    return merge([lathe(prof, c, 10), head])


def _fix_up(m):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    n = np.cross(b - a, c - a)
    ctr = m.v.mean(0)
    out = (a + b + c) / 3 - ctr
    out[:, 1] = np.maximum(out[:, 1], 0) + 0.01
    flip = np.einsum("ij,ij->i", n, out) < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def frustum(poly, y0, inset, y1, c, top_c=None, top=True):
    """Sloped roof from an outline at y0 to the outline shrunk by inset at y1 (mansard, glass pyramid)."""
    P0 = poly if ring_area(poly) > 0 else poly[::-1]
    P1 = offset(P0, -inset)
    V, F = [], []
    for i in range(len(P0)):
        j = (i + 1) % len(P0)
        o = len(V)
        V += [(P0[i][0], y0, P0[i][1]), (P0[j][0], y0, P0[j][1]), (P1[j][0], y1, P1[j][1]), (P1[i][0], y1, P1[i][1])]
        F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    ms = [_fix_up(Mesh(V, F, c))]
    if top:
        ms.append(cap(P1, y1, c if top_c is None else top_c))
    return merge(ms), P1


# ── Façade pieces (local t, y, d) ───────────────────────────────────────────
def arcade_bay(s, t, w, out):
    """One bay of the main fronts: ground window + mezzanine, arched main-floor window between engaged columns,
    balcony, second-floor window with its hood."""
    out.append(s.box(t - 0.8, t + 0.8, 1.3, 4.5, -0.05, 0.08, DARK))
    out.append(s.box(t - 1.0, t + 1.0, 4.5, 4.75, -0.05, 0.16, jit(GRAN)))
    out.append(s.box(t - 0.7, t + 0.7, 5.15, 6.05, -0.05, 0.08, DARK))
    out.append(s.put(arch_face(1.8, 2.9, DARK).T(t, 7.5, 0.07)))
    out.append(s.box(t - 1.0, t + 1.0, 7.15, 7.55, -0.05, 0.55, jit(STONE)))          # balcony slab
    out.append(s.box(t - 0.95, t + 0.95, 7.55, 8.35, 0.4, 0.5, ORN))                  # its balustrade (as a band)
    out.append(s.box(t - 0.8, t + 0.8, 12.6, 15.0, -0.05, 0.08, DARK))
    out.append(s.box(t - 1.05, t + 1.05, 15.0, 15.3, -0.05, 0.3, jit(STONE)))         # hood
    out.append(s.box(t - 0.95, t + 0.95, 12.3, 12.6, -0.05, 0.4, jit(STONE)))         # small balcony
    for e in (-w / 2, w / 2):                                                       # engaged columns between bays
        c0 = s.p(t + e, 0.18)
        out.append(cyl(0.3, 0.3, 0.3, ORN, 12).T(c0[0], 7.15, c0[1]))
        out.append(cyl(0.25, 0.22, 4.15, jit(STONE), 12, caps=False).T(c0[0], 7.45, c0[1]))
        out.append(cyl(0.22, 0.34, 0.45, ORN, 12).T(c0[0], 11.6, c0[1]))
    return out


def plain_bay(s, t, out):
    out.append(s.box(t - 0.75, t + 0.75, 1.4, 4.4, -0.05, 0.08, DARK))
    out.append(s.box(t - 0.8, t + 0.8, 7.6, 11.2, -0.05, 0.08, DARK))
    out.append(s.box(t - 1.0, t + 1.0, 7.3, 7.6, -0.05, 0.35, jit(STONE)))
    out.append(s.box(t - 0.75, t + 0.75, 12.6, 15.0, -0.05, 0.08, DARK))
    out.append(s.box(t - 1.0, t + 1.0, 15.0, 15.3, -0.05, 0.25, jit(STONE)))
    return out


def crest(s, t, w, h, out, clock=False):
    """The carved crest over a pavilion: a raised block, a cartouche (or the clock) with two figures, finials."""
    out.append(s.box(t - w / 2, t + w / 2, TOP, TOP + h, -1.2, 0.25, jit(STONE)))
    out.append(s.box(t - w / 2 - 0.15, t + w / 2 + 0.15, TOP + h, TOP + h + 0.3, -1.35, 0.4, ORN))
    yc = TOP + h * 0.55
    disc = cyl(1.05 if clock else 0.8, 1.05 if clock else 0.8, 0.3, ORN, 20).R(math.pi / 2, 0, 0)
    out.append(s.put(disc.T(t, yc, 0.55)))
    if clock:
        out.append(s.put(cyl(0.85, 0.85, 0.05, (0.97, 0.97, 0.95), 20).R(math.pi / 2, 0, 0).T(t, yc, 0.85)))
        out.append(s.box(t - 0.05, t + 0.05, yc - 0.05, yc + 0.6, 0.88, 0.94, DARK))
        out.append(s.box(t - 0.05, t + 0.45, yc - 0.05, yc + 0.05, 0.88, 0.94, DARK))
    for e in (-1, 1):
        f = figure(1.9, ORN)
        out.append(s.put(f.T(t + e * (1.9 if clock else 1.6), TOP + h + 0.25, -0.2)))
        fx = t + e * (w / 2 - 0.3)
        out.append(s.box(fx - 0.35, fx + 0.35, TOP + h + 0.3, TOP + h + 0.9, -0.6, 0.1, ORN))
        p0 = s.p(fx, -0.25)
        out.append(cyl(0.28, 0.02, 2.0, ORN, 4).T(p0[0], TOP + h + 0.9, p0[1]))         # obelisk finial


def big_arch_pavilion(s, t0, t1, out, clock=False, door=True):
    """Pavilion bay: projecting, pilasters, door between columns, the great arched window with its balcony."""
    t = (t0 + t1) / 2
    w = t1 - t0
    out.append(s.box(t0, t1, FOUND, G1, -0.3, 0.35, jit(GRAN)))
    out.append(s.box(t0, t1, G1, TOP - 1.2, -0.3, 0.35, jit(STONE)))
    for e in (t0 + 0.9, t1 - 0.9):                                                  # carved pilasters
        out.append(s.box(e - 0.75, e + 0.75, G1 + 0.03, TOP - 1.15, 0.3, 0.55, ORN))
    if door:
        out.append(s.box(t - 1.7, t + 1.7, 0.0, 5.6, 0.3, 0.42, DARK))
        for e in (-2.35, 2.35):
            c0 = s.p(t + e, 0.75)
            out.append(cyl(0.38, 0.33, 5.6, jit(GRAN), 14).T(c0[0], 0.0, c0[1]))
        out.append(s.box(t - 2.9, t + 2.9, 5.6, 6.3, 0.3, 1.2, jit(GRAN)))
    out.append(s.put(arch_face(4.3, 4.7, (0.16, 0.2, 0.22)).T(t, 7.9, 0.4)))
    out.append(s.put(arch_face(3.9, 4.6, GLASS_C).T(t, 8.0, 0.45)))                  # the glazed window
    for k in (-0.65, 0.65):
        out.append(s.box(t + k - 0.06, t + k + 0.06, 8.0, 14.4, 0.45, 0.52, (0.2, 0.22, 0.24)))
    arc = [s.p(t + 2.4 * math.cos(a), 0.45) for a in np.linspace(0, math.pi, 15)]
    out.append(tube([(p[0], 12.6 + 2.4 * math.sin(a), p[1]) for p, a in zip(arc, np.linspace(0, math.pi, 15))], 0.18, ORN, 6))
    out.append(s.box(t - 2.8, t + 2.8, 7.0, 7.5, 0.3, 1.25, jit(STONE)))                # balcony
    for k in np.arange(t - 2.6, t + 2.65, 0.4):
        c0 = s.p(k, 1.05)
        out.append(cyl(0.09, 0.09, 0.85, ORN, 6, caps=False).T(c0[0], 7.5, c0[1]))
    out.append(s.box(t - 2.8, t + 2.8, 8.35, 8.55, 0.85, 1.25, ORN))
    crest(s, t, min(w - 2.0, 8.5), 2.2, out, clock=clock)


def balustrade(s, t0, t1, out):
    t0, t1 = max(t0, 0.6), min(t1, s.L - 0.6)          # the corners carry round pedestals (corner_posts)
    if t1 - t0 < 0.5:
        return
    out.append(s.box(t0, t1, TOP, TOP + 0.25, -0.6, -0.1, jit(STONE)))
    out.append(s.box(t0, t1, BAL - 0.25, BAL, -0.62, -0.08, jit(STONE)))
    for k in np.arange(t0 + 0.25, t1 - 0.1, 0.42):
        out.append(s.box(k - 0.08, k + 0.08, TOP + 0.25, BAL - 0.25, -0.43, -0.27, ORN))
    for k in np.arange(t0 + 0.2, t1, 4.25):
        out.append(s.box(k - 0.25, k + 0.25, TOP, BAL + 0.1, -0.65, -0.05, jit(STONE)))


def mansard(s, t0, t1, roof, dormers, ridge, pitch=4.25):
    """Mansard range along a street front: steep slate slope with dormers, flatter top, iron cresting."""
    prof = [(-0.9, TOP), (-2.4, 20.4), (-6.5, 21.4), (-12.0, TOP)]
    V, F = [], []
    for k in range(3):
        (d0, y0), (d1, y1) = prof[k], prof[k + 1]
        o = len(V)
        V += [(t0 + (1.5 if k == 0 else 0), y0, d0), (t1 - (1.5 if k == 0 else 0), y0, d0), (t1, y1, d1), (t0, y1, d1)]
        F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    for t, sg in ((t0, -1), (t1, 1)):                                                  # end walls
        o = len(V)
        V += [(t, TOP, -0.9), (t, 20.4, -2.4), (t, 21.4, -6.5), (t, TOP, -12.0)]
        F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    roof.append(s.put(_fix_up(Mesh(V, F, SLATE))))
    ridge.append(s.box(t0, t1, 21.35, 21.85, -6.55, -6.45, IRON))
    for k in np.arange(t0 + 0.3, t1, 0.9):
        ridge.append(s.box(k - 0.04, k + 0.04, 21.4, 22.05, -6.53, -6.47, IRON))
    for t in np.arange(t0 + pitch / 2, t1 - 1.0, pitch):                                # dormers
        dormers.append(s.box(t - 0.75, t + 0.75, 18.0, 19.7, -2.6, -1.05, jit((0.72, 0.72, 0.71))))
        dormers.append(s.box(t - 0.5, t + 0.5, 18.2, 19.4, -1.1, -1.0, DARK))
        tri = Mesh([(t - 0.9, 19.7, -0.95), (t + 0.9, 19.7, -0.95), (t, 20.5, -0.95),
                    (t - 0.9, 19.7, -2.6), (t + 0.9, 19.7, -2.6), (t, 20.5, -2.6)],
                   [(0, 1, 2), (3, 5, 4), (0, 3, 4), (0, 4, 1), (1, 4, 5), (1, 5, 2), (2, 5, 3), (2, 3, 0)], ORN)
        dormers.append(s.put(tri))


def pav_roof(poly, roof, ridge, h=22.0):
    """Pavilion roof: a steep slate frustum with a flat top fenced by iron cresting."""
    m, top = frustum(poly, TOP, 2.6, h, SLATE, top_c=SLATE)
    roof.append(m)
    for i in range(len(top)):
        a, b = np.array(top[i]), np.array(top[(i + 1) % len(top)])
        L = np.linalg.norm(b - a)
        if L < 0.3:
            continue
        s = Side(a, b, ring=top)
        ridge.append(s.box(0, L, h, h + 0.45, -0.05, 0.05, IRON))
        for k in np.arange(0.3, L, 0.8):
            ridge.append(s.box(k - 0.04, k + 0.04, h, h + 0.7, -0.03, 0.03, IRON))
    return top


def rect_poly(s, t0, t1, depth):
    return [tuple(s.p(t0, 0)), tuple(s.p(t1, 0)), tuple(s.p(t1, -depth)), tuple(s.p(t0, -depth))]


def line_x(p, u, q, v):
    """Intersection of the lines p + a u and q + b v."""
    A = np.array([[u[0], -v[0]], [u[1], -v[1]]])
    a, _ = np.linalg.solve(A, np.asarray(q) - np.asarray(p))
    return np.asarray(p) + a * np.asarray(u)


# ── The building ─────────────────────────────────────────────────────────────
def body():
    out = []
    out.append(merge([walls(OUTLINE, FOUND, G1, jit(GRAN))] + [walls(h, FOUND, G1, jit(GRAN), hole=True) for h in COURTS]))
    out.append(merge([walls(OUTLINE, G1, TOP - 1.2, jit(STONE))] + [walls(h, G1, TOP, jit(STONE), hole=True) for h in COURTS]))
    out.append(walls(OUTLINE, TOP - 1.2, TOP, jit(STONE)))
    out.append(cap(OUTLINE, TOP - 0.02, FLAT, COURTS))
    # rustication grooves of the ground storey, string courses, frieze, the cornice
    for y in (1.5, 2.9, 4.3):
        out.append(walls(offset(OUTLINE, 0.04), y, y + 0.12, (0.5, 0.5, 0.5)))
    for d, y0, y1, c in ((0.25, G1 - 0.3, G1 + 0.1, GRAN), (0.12, 11.95, 12.3, STONE), (0.2, 15.55, 15.8, STONE),
                         (0.45, 15.8, 16.4, STONE), (0.8, 16.4, TOP, STONE)):
        o = offset(OUTLINE, d)
        out.append(merge([walls(o, y0, y1, jit(c)), cap(o, y0, jit(c), [offset(OUTLINE, -0.5)], up=False)]))
    o = offset(OUTLINE, 0.8)
    out.append(cap(o, TOP, jit(STONE), [offset(OUTLINE, -0.6)]))
    return out


SIDES = {
    "paseo": Side(P[9], P[10]),
    "alcala_e": Side(P[8], P[7]),
    "alcala_w": Side(P[7], P[6]),
    "alcala_nw": Side(P[6], P[5]),
    "chamfer_nw": Side(P[4], P[3]),
    "cubas_n": Side(P[2], P[1]),
    "cubas": Side(P[0], P[24]),
    "madrazo_e": Side(P[10], P[11]),
    "madrazo": Side(P[16], P[17]),
    "cibeles": Side(P[8], P[9]),
}
for i, j in ((11, 12), (12, 13), (13, 14), (3, 2), (5, 4)):
    SIDES[f"short_{i}"] = Side(P[i], P[j])


def fronts():
    out, roof, dormers, ridge = [], [], [], []
    S = SIDES
    # programmes: (kind, t0, t1)
    prog = {
        "paseo": [("corner", 0, 8.5), ("arcade", 8.5, 34.0), ("dome", 34.0, 46.0), ("arcade", 46.0, 101.0),
                  ("pav", 101.0, 113.0), ("arcade", 113.0, 150.5), ("endpav", 150.5, 160.3)],
        "alcala_e": [("corner", 0, 8.5), ("arcade", 8.5, 49.0), ("pav", 49.0, 61.0)],
        "alcala_w": [("arcade", 0, 50.6), ("pav", 50.6, S["alcala_w"].L)],
        "alcala_nw": [("plain", 0, S["alcala_nw"].L)],
        "chamfer_nw": [("endpav", 0, S["chamfer_nw"].L)],
        "cubas_n": [("plain", 0, S["cubas_n"].L)],
        "cubas": [("plain", 0, S["cubas"].L)],
        "madrazo_e": [("plain", 0, S["madrazo_e"].L)],
        "madrazo": [("plain", 0, S["madrazo"].L)],
    }
    pav_polys = []
    for name, items in prog.items():
        s = S[name]
        for kind, t0, t1 in items:
            if kind == "arcade":
                n = max(1, int(round((t1 - t0) / 4.25)))
                w = (t1 - t0) / n
                for k in range(n):
                    arcade_bay(s, t0 + (k + 0.5) * w, w, out)
                mansard(s, t0, t1, roof, dormers, ridge, pitch=w)
            elif kind == "plain":
                n = max(1, int(round((t1 - t0) / 4.0)))
                w = (t1 - t0) / n
                for k in range(n):
                    plain_bay(s, t0 + (k + 0.5) * w, out)
                if t1 - t0 > 6:
                    mansard(s, t0 + 1.0, t1 - 1.0, roof, dormers, ridge, pitch=w)
            elif kind in ("pav", "dome", "endpav"):
                big_arch_pavilion(s, t0, t1, out, door=(kind != "endpav"))
                pav_polys.append(rect_poly(s, t0, t1, 12.0))
                top = pav_roof(pav_polys[-1], roof, ridge, h=22.0)
                if kind == "dome":
                    dome(s, (t0 + t1) / 2, out, roof, ridge)
            balustrade(s, t0, t1, out)
    for name in [k for k in S if k.startswith("short_")]:
        s = S[name]
        if s.L > 2.5:
            plain_bay(s, s.L / 2, out)
        balustrade(s, 0, s.L, out)
    # courtyard walls: plain windows
    for h in COURTS:
        for i in range(len(h)):
            a, b = h[i], h[(i + 1) % len(h)]
            L = math.dist(a, b)
            if L < 4.0:
                continue
            s = Side(a, b, ring=h, hole=True)
            n = int(L // 3.8)
            for k in range(n):
                t = (L - n * 3.8) / 2 + (k + 0.5) * 3.8
                for y0, y1 in ((1.4, 4.2), (7.6, 10.8), (12.4, 14.8)):
                    out.append(s.box(t - 0.65, t + 0.65, y0, y1, -0.05, 0.07, DARK))
    corner(out, roof, ridge)
    for x, z in OUTLINE:                                                                # corner pedestals
        out.append(cyl(0.6, 0.6, BAL + 0.2 - TOP, jit(STONE), 10).T(x, TOP - 0.02, z))
    return out, roof, dormers, ridge


def corner(out, roof, ridge):
    """The Cibeles chamfer: door between columns, the great glazed arch, the clock crest with the gilded globe."""
    s = SIDES["cibeles"]
    L = s.L
    big_arch_pavilion(s, 0.0, L, out, clock=True)
    # the iron stand and the globe over the clock
    c0 = s.p(L / 2, -0.4)
    yb = TOP + 2.5
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        dx, dz = math.cos(a) * 0.55, math.sin(a) * 0.55
        out.append(cyl(0.07, 0.05, 2.6, IRON, 6).T(c0[0] + dx, yb, c0[1] + dz))
    for y in (yb + 0.9, yb + 1.8, yb + 2.6):
        out.append(cyl(0.7, 0.7, 0.08, IRON, 12).T(c0[0], y, c0[1]))
    out.append(lathe([(0.0, -0.75), (0.38, -0.65), (0.65, -0.38), (0.75, 0.0), (0.65, 0.38), (0.38, 0.65), (0.0, 0.75)], GOLD, 16)
               .T(c0[0], yb + 3.35, c0[1]))
    # the corner roof: chamfer plus the two return bays, 12 m deep
    sp, sa = SIDES["paseo"], SIDES["alcala_e"]
    A, Bp = sp.p(8.5, 0), sa.p(8.5, 0)
    A2, B2 = sp.p(8.5, -12.0), sa.p(8.5, -12.0)
    C = line_x(A2, -sp.u, B2, -sa.u)
    poly = [tuple(A), tuple(P[9]), tuple(P[8]), tuple(Bp), tuple(B2), tuple(C), tuple(A2)]
    pav_roof(poly, roof, ridge, h=22.4)
    for nm in ("paseo", "alcala_e"):                                                   # the return bays
        s2 = SIDES[nm]
        out.append(s2.box(0.0, 8.5, FOUND, G1, -0.3, 0.35, jit(GRAN)))
        out.append(s2.box(0.0, 8.5, G1, TOP - 1.2, -0.3, 0.35, jit(STONE)))
        out.append(s2.box(7.4, 8.5, G1 + 0.03, TOP - 1.15, 0.3, 0.55, ORN))
        out.append(s2.box(3.0, 6.0, 0.0, 5.2, 0.3, 0.42, DARK))
        out.append(s2.put(arch_face(3.0, 3.6, DARK).T(4.5, 0.0, 0.42)))
        out.append(s2.box(2.6, 6.4, 7.6, 14.8, 0.3, 0.42, DARK))                           # columned window
        for k in (3.1, 4.5, 5.9):
            c1 = s2.p(k, 0.6)
            out.append(cyl(0.2, 0.18, 7.0, jit(STONE), 10).T(c1[0], 7.6, c1[1]))
        out.append(s2.box(2.4, 6.6, 7.1, 7.6, 0.3, 1.0, jit(STONE)))


def dome(s, t, out, roof, ridge):
    """The slate dome with lantern and spire over the Paseo del Prado pavilion."""
    c = s.p(t, -6.0)
    y0 = 22.0
    roof.append(cyl(4.3, 4.3, 0.9, jit(STONE), 8).T(c[0], y0 - 0.2, c[1]))
    prof = [(4.4, 0.7), (4.2, 1.6), (3.6, 2.6), (2.6, 3.4), (1.3, 3.9), (0.9, 4.0)]
    roof.append(lathe(prof, SLATE, 8).T(c[0], y0, c[1]))
    roof.append(cyl(0.9, 0.9, 1.0, jit(STONE), 8).T(c[0], y0 + 4.0, c[1]))
    roof.append(cyl(1.1, 0.0, 0.9, SLATE, 8).T(c[0], y0 + 5.0, c[1]))
    ridge.append(cyl(0.07, 0.02, 2.2, IRON, 6).T(c[0], y0 + 5.8, c[1]))
    for k in range(8):                                                                  # oculi
        a = k * math.pi / 4 + math.pi / 8
        out.append(B(-0.35, 0.35, 0, 0.7, -0.05, 0.05, DARK).R(0, -a + math.pi / 2, 0)
                   .T(c[0] + math.cos(a) * 4.33, y0 + 0.75, c[1] + math.sin(a) * 4.33))


def inner_roofs():
    roof, glass, out = [], [], []
    # the inner 6–7 storey block
    for k, (lv, ring) in enumerate(INNER):
        h = (26.0 if lv == 7 else 22.6) + 0.05 * k
        out.append(merge([walls(ring, TOP - 0.5, h, jit(STONE, 0.03)), cap(ring, h, FLAT)]))
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            L = math.dist(a, b)
            if L < 3.0:
                continue
            s = Side(a, b, ring=ring)
            n = int(L // 3.0)
            for j in range(n):
                t = (L - n * 3.0) / 2 + (j + 0.5) * 3.0
                for y in np.arange(TOP + 0.9, h - 1.0, 3.2):
                    out.append(s.box(t - 0.6, t + 0.6, y, y + 1.6, -0.05, 0.06, DARK))
    # the glazed roof of the main hall (octagonal) and the three smaller ones
    m, top = frustum(HALL, TOP - 0.05, 5.0, 21.0, GLASS_C, top=False)
    glass.append(m)
    ctr = np.mean(np.array(top), 0)
    V = [(ctr[0], 22.6, ctr[1])] + [(x, 21.0, z) for x, z in top]
    glass.append(_fix_up(Mesh(V, [(0, i + 1, (i + 1) % len(top) + 1) for i in range(len(top))], GLASS_C)))
    roof.append(walls(offset(HALL, 0.3), TOP - 0.05, TOP + 0.6, jit(STONE)))
    for sk in SKY:
        m, top = frustum(sk, TOP - 0.05, 3.0, 19.6, GLASS_C, top=False)
        glass.append(m)
        ctr = np.mean(np.array(top), 0)
        V = [(ctr[0], 20.6, ctr[1])] + [(x, 19.6, z) for x, z in top]
        glass.append(_fix_up(Mesh(V, [(0, i + 1, (i + 1) % len(top) + 1) for i in range(len(top))], GLASS_C)))
    return out, roof, glass


def main():
    out = ROOT / "models" / "Q4889526"
    out.mkdir(parents=True, exist_ok=True)
    stone = body()
    f, roof, dormers, ridge = fronts()
    stone += f + dormers
    i_out, i_roof, glass = inner_roofs()
    stone += i_out
    roof += i_roof
    m_stone, m_roof, m_glass, m_iron = merge(stone), merge(roof), merge(glass), merge(ridge)
    tris, size = write_glb(out / "model.glb", [("banco_de_espana", m_stone), ("roofs", m_roof, {"material": "slate"}),
                                               ("cresting", m_iron, {"material": "metal"}),
                                               ("skylights", m_glass, {"material": "glass"})],
                           roughness=0.88, materials={"glass": GLASS, "metal": METAL, "slate": {"metallic": 0.1, "roughness": 0.55}})
    allv = np.vstack([m_stone.v, m_roof.v, m_glass.v, m_iron.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
