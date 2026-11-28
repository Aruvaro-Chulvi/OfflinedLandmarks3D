"""
Puente de Alcántara, Toledo (Q581532) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: Roman bridge over the Tagus (founding of Toletum), rebuilt in the 10th century (when its third arch
was walled up, leaving a small horseshoe postern) and again under Alfonso X (13th century); the west gate-tower
was rebuilt in 1484 under the Catholic Monarchs (their arms on it), and the east tower was replaced in 1721 by a
Baroque triumphal arch. Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 660415643 (man_made=bridge): the deck 83 m between the two gates and 5.2 m wide, axis at
  81.0° (the footway, way 724145214, from the west tower to the east arch), the two trapezoidal buttresses of the
  pier between the arches (4.9 m out from the deck, both sides) and the two cantilevered refuges near the east end
  (x 21.6–27.5, 1.6 m out). Way 105336454: the west gate-tower (8.8 × 10–14 m with its base on the slope).
  Way 724145213: the Baroque east arch (3.5 × 9.4 m).
- Wikipedia (de, after C. Fernández Casado 1979): 83 m between the gates, 5.40 m wide, main semicircular arch of
  28.30 m span, pier of 8.40 m, smaller semicircular arch of 16 m over the slope up to the city, the walled-up third
  arch with a horseshoe postern 1.72 m wide.
- IGN MDT05 (tools/terrain.py -> terrain.json): the Tagus at 450.0 m, the street west of the tower at 471 m, the
  road at the east arch at 468–469 m. So the deck is set at +21.0 (west) down to +19.0 (east arch).
- Photos on Commons (heights by proportion, ±10 %): main arch springing about 5 m over the water; the west tower
  about 13.5 m above the street (merlons included), 10.6 m wide, with two corner masses on its city face; the
  Baroque arch about 9.3 m wide, 4.2 m opening, cornice at 9.8 m, its attic with the imperial eagle, a shell crest
  and ball finials up to about 15.5 m.

Axes (docs/MODEL_SPEC.md): +Y up, y = 0 the river (450.0 m); the bridge runs along X (+X to the east, 81.0°), the
downstream (south) elevation faces +Z (171.0°); origin on the water under the middle of the main arch, on the
centre line of the deck.

    python model_sources/Q581532_puente_de_alcantara_toledo.py   -> models/Q581532/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, extrude, triangulate, fbm, normals, write_glb  # noqa: E402

rng = np.random.default_rng(1484)

GRANITE = np.array((0.70, 0.67, 0.61))     # ashlar of the arches and piers
RUBBLE = np.array((0.72, 0.66, 0.56))      # spandrel walls (rubble and ashlar)
TOWER = np.array((0.73, 0.62, 0.52))       # Toledan masonry: rubble boxed by brick courses
BRICK = np.array((0.70, 0.44, 0.34))       # the Baroque arch
LIGHT = np.array((0.80, 0.78, 0.73))       # carved stone, coats of arms
PAVING = np.array((0.62, 0.60, 0.57))
DARK = np.array((0.16, 0.14, 0.13))
IRON = (0.12, 0.11, 0.10)

W = 2.6                                    # half width of the bridge (OSM 5.2 m)
X_TW, X_EA = -38.4, 44.3                   # east face of the west tower, west face of the east arch
SPRING_M, R_M, XC_M = 5.0, 14.15, -0.45    # main arch: springing, radius (28.3 m span), centre
SPRING_S, R_S, XC_S = 11.2, 8.0, -30.4     # small arch (16 m span)
PIER = (-22.4, -14.6)

# MDT05 ground along the axis (lowest across the width), metres above the river (450.0 m)
GROUND = [(-62, 21.0), (-58, 20.7), (-54, 19.6), (-50, 17.3), (-46, 14.3), (-42, 10.8), (-38, 7.6), (-34, 4.9),
          (-30, 3.1), (-26, 1.8), (-22, 0.9), (-18, 0.4), (-14, 0.0), (10, 0.0), (14, 1.1), (18, 1.9), (22, 2.9),
          (26, 4.2), (30, 5.8), (34, 7.3), (38, 8.9), (42, 11.3), (46, 14.3), (50, 16.7), (54, 18.5)]


def ground(x):
    xs, ys = zip(*GROUND)
    return float(np.interp(x, xs, ys))


def deck(x):
    """Walking level of the deck: level from the west tower to the middle, then down to the east arch."""
    return 21.0 if x <= 0 else 21.0 - 2.0 * min(x, X_EA) / X_EA


def tint(c, v, seed, amp=0.10, freq=0.35):
    n = fbm(v, freq, seed, 3)
    return np.clip(np.asarray(c)[None, :] * (1 - amp / 2 + amp * n[:, None]), 0, 1)


def paint(m, c, seed, amp=0.10):
    m.c = tint(c, m.v, seed, amp)
    return m


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def mirror_z(m):
    n = m.copy(); n.v[:, 2] *= -1; n.f = n.f[:, ::-1]; return n


def both_z(ms):
    ms = [m for m in ms if m is not None]
    return ms + [mirror_z(m) for m in ms]


def sphere(r, c, seg=12, rings=7):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


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
    """The same with the outline in the (z, y) plane, extruded along x (a wall across the bridge axis)."""
    m = prism_xy(poly, x0, x1, c)
    m.v = m.v[:, [2, 1, 0]]
    m.f = m.f[:, ::-1]
    return m


def semi(xa, xb, ys, n=24, extra=0.0):
    """Intrados of a semicircular arch from xa to xb (radius + extra), left to right."""
    xc, r = (xa + xb) / 2, (xb - xa) / 2 + extra
    return [(xc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(math.pi, 0, n + 1)]


def arch_ring(xa, xb, ys, t, z0, z1, c, n=24):
    """The voussoirs: a half ring t thick, standing a little proud of both faces."""
    outer = semi(xa, xb, ys, n, t)
    inner = semi(xa, xb, ys, n)[::-1]
    return prism_xy(outer + inner, z0, z1, c)


def horseshoe(xc, w, y0, h, n=14):
    """A horseshoe opening w wide, h high from y0 (the arch rises over 3/4 of a circle)."""
    r = w / 2 * 1.12
    yc = y0 + h - r
    a0 = -0.25 * math.pi
    pts = [(xc + w / 2, y0)]
    for a in np.linspace(a0, math.pi - a0, n + 1):
        pts.append((xc + r * math.cos(a), yc + r * math.sin(a)))
    pts.append((xc - w / 2, y0))
    return pts


def merlons_line(p0, p1, y0, c, w=0.6, h=0.9, t=0.55, gap=0.55):
    """Square merlons along a straight top edge from p0 to p1 (x, z), standing on y0."""
    (xa, za), (xb, zb) = p0, p1
    L = math.hypot(xb - xa, zb - za)
    n = max(1, int((L + gap) / (w + gap)))
    pitch = L / n
    ang = math.atan2(-(zb - za), xb - xa)
    out = []
    for k in range(n):
        s = (k + 0.5) * pitch / L
        m = box(w, h, t, c).R(0, ang, 0).T(xa + (xb - xa) * s, y0, za + (zb - za) * s)
        out.append(m)
    return out


# ── The bridge body ──────────────────────────────────────────────────────────
def body():
    parts = []
    top = [(x, deck(x)) for x in np.linspace(X_TW, X_EA, 24)]
    bottom = []
    # east abutment: down the slope from the east arch to the main arch, with the walled-up third arch's postern
    for x in np.linspace(X_EA, 31.2, 6):
        bottom.append((x, ground(x) - 2.5))
    g30 = ground(30.0)
    post = horseshoe(30.0, 1.72, g30 - 2.5, 2.5 + 3.0)
    bottom += post
    for x in np.linspace(28.8, 13.7, 6):
        bottom.append((x, ground(x) - 2.5))
    bottom.append((13.7, SPRING_M))
    bottom += semi(-14.6, 13.7, SPRING_M, 32)[::-1][1:-1]
    bottom += [(-14.6, SPRING_M), (-14.6, -3.0), (-22.4, -3.0), (-22.4, SPRING_S)]
    bottom += semi(-38.4, -22.4, SPRING_S, 20)[::-1][1:-1]
    bottom += [(-38.4, SPRING_S), (-38.4, ground(-38.4) - 2.5)]
    # top runs west -> east, bottom east -> west: one simple outline
    poly = top + bottom
    m = prism_xy(poly, -W, W, RUBBLE)
    n = normals(m)
    paint(m, RUBBLE, 3)
    up = n[:, 1] > 0.8
    m.c[up] = tint(PAVING, m.v[up], 4, 0.06)
    parts.append(m)

    # voussoirs of the two arches and the impost blocks
    parts.append(paint(arch_ring(-14.6, 13.7, SPRING_M, 1.35, -W - 0.06, W + 0.06, GRANITE), GRANITE, 5))
    parts.append(paint(arch_ring(-38.4, -22.4, SPRING_S, 1.05, -W - 0.06, W + 0.06, GRANITE), GRANITE, 6))
    for x0, x1, y in ((-15.2, -14.6, SPRING_M), (13.7, 14.3, SPRING_M), (-23.0, -22.4, SPRING_S)):
        parts.append(B(x0, x1, y - 0.35, y, -W - 0.08, W + 0.08, GRANITE * 1.04))
    # the pier in granite ashlar, with its two trapezoidal buttresses (OSM) up to the deck: refuges on top
    py = deck(-18.5)
    parts.append(paint(B(PIER[0], PIER[1], -3.0, SPRING_S, -W - 0.04, W + 0.04, GRANITE), GRANITE, 7))
    for s in (-1, 1):
        tr = [(-21.9, s * W), (-20.4, s * 7.3), (-17.0, s * 7.3), (-14.9, s * W)]
        parts.append(paint(extrude(tr, -3.0, py, GRANITE, top_c=PAVING), GRANITE, 8 + s))
        tr2 = [(-22.2, s * W), (-20.6, s * 7.6), (-16.8, s * 7.6), (-14.6, s * W)]
        parts.append(paint(extrude(tr2, -3.0, 7.0, GRANITE * 0.97), GRANITE * 0.97, 10 + s))   # the battered foot
        for k in range(7):                                                  # putlog holes under the set-back
            x = -20.2 + k * 0.5
            parts.append(B(x - 0.12, x + 0.12, 8.0, 8.3, s * 7.32 - 0.02, s * 7.32 + 0.02, DARK))
        # parapet of the refuge on top of the buttress
        for (xa, za), (xb, zb) in zip(tr[:-1], tr[1:]):
            L = math.hypot(xb - xa, zb - za)
            ang = math.atan2(-(zb - za), xb - xa)
            parts.append(paint(box(L, 1.05, 0.45, GRANITE * 1.03).R(0, ang, 0).T((xa + xb) / 2, py, (za + zb) / 2),
                               GRANITE * 1.03, 12))
        parts.append(sphere(0.32, LIGHT, 10, 6).T(-20.4, py + 1.4, s * 7.2))
        parts.append(sphere(0.32, LIGHT, 10, 6).T(-17.0, py + 1.4, s * 7.2))
        parts.append(B(-20.6, -20.2, py + 1.05, py + 1.12, s * 7.2 - 0.2, s * 7.2 + 0.2, LIGHT))
        parts.append(B(-17.2, -16.8, py + 1.05, py + 1.12, s * 7.2 - 0.2, s * 7.2 + 0.2, LIGHT))

    # the two cantilevered refuges near the east end (OSM x 21.6–27.5)
    for s in (-1, 1):
        yd = deck(24.5)
        parts.append(B(21.6, 27.5, yd - 0.55, yd, *sorted((s * W, s * 4.3)), GRANITE))
        for x in (22.1, 23.6, 25.5, 27.0):
            parts.append(B(x - 0.25, x + 0.25, yd - 1.4, yd - 0.55, *sorted((s * W, s * 4.0)), GRANITE * 0.95))
        parts.append(B(21.6, 27.5, yd, yd + 1.0, *sorted((s * 3.85, s * 4.3)), GRANITE * 1.03))
        for x0 in (21.6, 27.05):
            parts.append(B(x0, x0 + 0.45, yd, yd + 1.0, *sorted((s * W, s * 4.3)), GRANITE * 1.03))

    # cornice band along both faces, and the parapets with their coping
    for s in (-1, 1):
        for xa, xb in ((X_TW, PIER[0] + 0.5), (PIER[1] - 0.5, 21.6), (27.5, X_EA)):
            xs = np.linspace(xa, xb, 6)
            band = [(x, deck(x) - 0.55) for x in xs] + [(x, deck(x) - 0.25) for x in xs[::-1]]
            parts.append(prism_xy(band, *sorted((s * W, s * (W + 0.12))), GRANITE * 1.02))
        for xa, xb in ((X_TW, -21.9), (-14.9, 21.6), (27.5, X_EA)):
            xs = np.linspace(xa, xb, 8)
            wall = [(x, deck(x) - 0.05) for x in xs] + [(x, deck(x) + 0.95) for x in xs[::-1]]
            parts.append(paint(prism_xy(wall, *sorted((s * (W - 0.45), s * W)), GRANITE * 1.02), GRANITE * 1.02, 14))
            cope = [(x, deck(x) + 0.95) for x in xs] + [(x, deck(x) + 1.12) for x in xs[::-1]]
            parts.append(prism_xy(cope, *sorted((s * (W - 0.52), s * (W + 0.05))), LIGHT * 0.96))
    # big stone balls at the ends of the parapets (as on the photos), and lamp posts
    for x in (X_TW + 0.5, X_EA - 0.5):
        for s in (-1, 1):
            parts.append(B(x - 0.3, x + 0.3, deck(x) + 1.1, deck(x) + 1.5, s * (W - 0.25) - 0.3, s * (W - 0.25) + 0.3, LIGHT))
            parts.append(sphere(0.42, LIGHT, 12, 7).T(x, deck(x) + 1.92, s * (W - 0.25)))
    for x in (-5.0, 10.0):
        yd = deck(x)
        parts.append(cyl(0.06, 0.05, 3.4, IRON, 6).T(x, yd + 1.1, W - 0.25))
        parts.append(B(x - 0.15, x + 0.15, yd + 4.4, yd + 4.85, W - 0.4, W - 0.1, IRON))
    return parts


# ── The west gate-tower (1484) ───────────────────────────────────────────────
OSM_TOWER = [(-45.9, 7.8), (-46.4, 3.6), (-45.3, 3.4), (-45.6, 0.5), (-46.0, -1.8), (-46.7, -1.7), (-47.2, -6.2),
             (-42.3, -6.4), (-38.8, -3.9), (-38.4, 3.6), (-41.1, 7.6)]


def west_tower():
    parts = []
    D = deck(X_TW)
    TOP = D + 12.0                         # wall head; merlons 0.95 m on it (13.5 m above the street with caps)
    X0, X1, ZW = -46.4, -38.4, 5.3
    PW, PY = 1.55, D + 4.6                 # passage half width, springing (crown D + 6.15)
    base_y = ground(X1) - 2.5
    # base on the slope (OSM outline, with the masses on its south-west), up to the street
    parts.append(paint(extrude(OSM_TOWER, base_y, D - 0.3, TOWER * 0.96), TOWER * 0.96, 20))
    parts.append(paint(extrude([(X0 - 0.6, -ZW - 0.3), (X1 + 0.3, -ZW - 0.3), (X1 + 0.3, ZW + 0.3), (X0 - 0.6, ZW + 0.3)],
                               base_y, D - 6.0, TOWER * 0.93), TOWER * 0.93, 21))
    # the two side bodies beside the passage, and the vault over it
    for s in (-1, 1):
        parts.append(paint(B(X0, X1, D, TOP, *sorted((s * PW, s * ZW)), TOWER), TOWER, 22 + s))
    arch = [(-PW, TOP), (-PW, PY)] + [(z, y) for z, y in [(-PW * math.cos(a), PY + PW * math.sin(a)) for a in np.linspace(0, math.pi, 13)]][1:-1] + \
           [(PW, PY), (PW, TOP)]
    parts.append(paint(prism_zy(arch, X0, X1, TOWER), TOWER, 24))
    parts.append(B(X0, X1, D - 0.02, D + 0.02, -PW, PW, PAVING))
    # roof platform
    parts.append(B(X0 + 0.5, X1 - 0.5, TOP - 0.5, TOP - 0.3, -ZW + 0.5, ZW - 0.5, TOWER * 0.85))
    # city face: two corner masses standing 0.6 m forward, the recess between them with the gate
    for s in (-1, 1):
        parts.append(paint(B(X0 - 0.6, X0 + 3.0, D, TOP + 0.9, *sorted((s * 1.95, s * ZW)), TOWER), TOWER, 25 + s))
        parts += merlons_line((X0 - 0.35, s * 2.2), (X0 - 0.35, s * (ZW - 0.25)), TOP + 0.9, TOWER, t=0.5)
        parts += merlons_line((X0 - 0.35, s * (ZW - 0.25)), (X0 + 2.75, s * (ZW - 0.25)), TOP + 0.9, TOWER, t=0.5)
        parts += merlons_line((X0 + 2.75, s * (ZW - 0.25)), (X0 + 2.75, s * 2.2), TOP + 0.9, TOWER, t=0.5)
    # brick courses (Toledan masonry): thin lighter bands all round
    for k in range(1, 9):
        y = D + k * 1.4
        for s in (-1, 1):
            parts.append(B(X0, X1, y, y + 0.12, *sorted((s * ZW, s * (ZW + 0.03))), TOWER * 1.1))
        parts.append(B(X1, X1 + 0.03, y, y + 0.12, -ZW, -PW - 0.4, TOWER * 1.1))
        parts.append(B(X1, X1 + 0.03, y, y + 0.12, PW + 0.4, ZW, TOWER * 1.1))
    # merlons on the wall head (east, north, south faces; the recess of the west face lower)
    parts += merlons_line((X1 - 0.3, -ZW + 0.3), (X1 - 0.3, ZW - 0.3), TOP, TOWER)
    for s in (-1, 1):
        parts += merlons_line((X0 + 3.3, s * (ZW - 0.3)), (X1 - 0.3, s * (ZW - 0.3)), TOP, TOWER)
    parts += merlons_line((X0 + 0.3, -1.6), (X0 + 0.3, 1.6), TOP, TOWER)
    # gate arch mouldings on both faces
    for xf, sgn in ((X1, 1), (X0, -1)):
        ring = [(z, y) for z, y in [(-(PW + 0.45) * math.cos(a), PY + (PW + 0.45) * math.sin(a)) for a in np.linspace(0, math.pi, 13)]]
        inner = [(z, y) for z, y in [(-PW * math.cos(a), PY + PW * math.sin(a)) for a in np.linspace(0, math.pi, 13)]][::-1]
        parts.append(prism_zy(ring + inner, *sorted((xf, xf + sgn * 0.1)), GRANITE * 1.05))
        for s in (-1, 1):
            parts.append(B(*sorted((xf, xf + sgn * 0.1)), D, PY, *sorted((s * PW, s * (PW + 0.45))), GRANITE * 1.05))
    # bridge face: the arms of the Catholic Monarchs in a framed panel, a box machicolation over them
    parts.append(B(X1, X1 + 0.12, D + 7.0, D + 9.4, -0.9, 0.9, LIGHT * 0.95))
    parts.append(cyl(0.6, 0.6, 0.18, LIGHT, 14).R(math.pi / 2, 0, 0).R(0, math.pi / 2, 0).S(1, 1.25, 1).T(X1 + 0.12, D + 8.1, 0))
    parts.append(B(X1, X1 + 0.85, D + 9.9, D + 11.3, -1.0, 1.0, TOWER))
    for z in (-0.7, 0.0, 0.7):
        parts.append(B(X1, X1 + 0.85, D + 9.45, D + 9.9, z - 0.15, z + 0.15, TOWER * 0.95))
    # city face: inscription and two sculpture niches, framed
    parts.append(B(X0 - 0.12, X0, D + 5.9, D + 9.0, -1.85, 1.85, TOWER * 1.06))
    for z in (-1.1, 1.1):
        parts.append(B(X0 - 0.16, X0 - 0.12, D + 7.4, D + 8.8, z - 0.45, z + 0.45, DARK * 2.2))
        parts.append(cyl(0.22, 0.18, 1.1, LIGHT, 8).T(X0 - 0.3, D + 7.5, z))
        parts.append(sphere(0.16, LIGHT, 8, 5).T(X0 - 0.3, D + 8.75, z))
    parts.append(B(X0 - 0.16, X0 - 0.12, D + 6.1, D + 7.2, -1.6, 1.6, LIGHT * 0.92))
    parts.append(B(X0 - 0.16, X0 - 0.12, D + 7.3, D + 9.6, -0.35, 0.35, LIGHT * 0.92))
    # arrow slits
    for y in (D + 5.5, D + 9.0):
        for s in (-1, 1):
            parts.append(B(X0 + 2.5, X0 + 2.7, y, y + 0.9, s * (ZW + 0.02) - 0.01, s * (ZW + 0.02) + 0.01, DARK))
            parts.append(B(X1 - 2.7, X1 - 2.5, y + 1.2, y + 2.1, s * (ZW + 0.02) - 0.01, s * (ZW + 0.02) + 0.01, DARK))
    for z in (-3.6, 3.6):
        parts.append(B(X1 + 0.01, X1 + 0.03, D + 10.0, D + 10.9, z - 0.1, z + 0.1, DARK))
    return parts


# ── The Baroque triumphal arch at the east end (1721) ───────────────────────
def east_arch():
    parts = []
    D = deck(X_EA)
    X0, X1, ZW = X_EA, 47.8, 4.65
    PW, PY = 2.1, D + 4.6                  # opening 4.2 m, crown D + 6.7
    CORN = D + 9.8
    base_y = ground(X1) - 2.5
    parts.append(paint(B(X0, X1, base_y, D, -ZW, ZW, GRANITE * 0.95), GRANITE * 0.95, 30))
    for s in (-1, 1):
        parts.append(paint(B(X0, X1, D, CORN, *sorted((s * PW, s * ZW)), BRICK), BRICK, 31 + s))
        # granite quoins at the outer corners and around the opening (alternating blocks)
        for k in range(10):
            y = D + k * 0.98
            wq = 0.75 if k % 2 == 0 else 0.45
            parts.append(B(X0 - 0.04, X1 + 0.04, y, y + 0.9, *sorted((s * (ZW + 0.04), s * (ZW - wq))), GRANITE * 1.06))
            parts.append(B(X0 - 0.04, X1 + 0.04, y, y + 0.9, *sorted((s * PW, s * (PW + wq))), GRANITE * 1.06)) if y + 0.9 <= PY else None
        parts.append(B(X0 - 0.05, X1 + 0.05, D, D + 0.6, *sorted((s * PW, s * (ZW + 0.1))), GRANITE))
    arch = [(-PW, CORN), (-PW, PY)] + [(-PW * math.cos(a), PY + PW * math.sin(a)) for a in np.linspace(0, math.pi, 15)][1:-1] + \
           [(PW, PY), (PW, CORN)]
    parts.append(paint(prism_zy(arch, X0, X1, BRICK), BRICK, 33))
    ring = [(-(PW + 0.55) * math.cos(a), PY + (PW + 0.55) * math.sin(a)) for a in np.linspace(0, math.pi, 15)]
    inner = [(-PW * math.cos(a), PY + PW * math.sin(a)) for a in np.linspace(0, math.pi, 15)][::-1]
    parts.append(prism_zy(ring + inner, X0 - 0.08, X1 + 0.08, GRANITE * 1.07))
    parts.append(B(X0 - 0.12, X1 + 0.12, PY + PW + 0.25, PY + PW + 0.85, -0.3, 0.3, LIGHT))     # keystone
    for z in np.linspace(-PW + 0.25, PW - 0.25, 9):                                            # the portcullis, raised
        yt = PY + math.sqrt(max(PW ** 2 - z ** 2, 0)) - 0.05
        parts.append(B(X0 + 1.05, X0 + 1.15, PY - 0.9, yt, z - 0.04, z + 0.04, IRON))
    for y in (PY - 0.85, PY - 0.3, PY + 0.3, PY + 0.9):
        parts.append(B(X0 + 1.04, X0 + 1.16, y, y + 0.08, -PW + 0.05, PW - 0.05, IRON))
    parts.append(B(X0, X1, D - 0.02, D + 0.02, -PW, PW, PAVING))
    # pilasters with hanging garlands, inscription tablets
    for xf, sg in ((X0, -1), (X1, 1)):
        for s in (-1, 1):
            parts.append(B(*sorted((xf, xf + sg * 0.15)), D + 0.6, CORN, *sorted((s * 3.05, s * 3.75)), BRICK * 1.04))
            parts.append(B(*sorted((xf, xf + sg * 0.22)), D + 6.0, D + 8.9, *sorted((s * 3.2, s * 3.6)), LIGHT * 0.95))
            parts.append(B(*sorted((xf, xf + sg * 0.08)), D + 3.9, D + 5.2, *sorted((s * 2.85, s * 3.45)), LIGHT))
    # entablature and cornice
    parts.append(B(X0 - 0.15, X1 + 0.15, CORN, CORN + 0.45, -ZW - 0.15, ZW + 0.15, GRANITE * 1.05))
    parts.append(B(X0 - 0.35, X1 + 0.35, CORN + 0.45, CORN + 0.8, -ZW - 0.35, ZW + 0.35, GRANITE * 1.08))
    # attic: the central panel with curved shoulders, the imperial eagle, the shell crest
    A = CORN + 0.8
    att = [(-3.0, A), (3.0, A), (3.0, A + 0.6)] + [(3.0 - 1.0 * (1 - math.cos(t)) - 0.6 * t, A + 0.6 + 2.4 * math.sin(t))
                                                   for t in np.linspace(0.15, math.pi / 2, 7)] + \
          [(1.25, A + 3.6), (1.25, A + 4.3), (-1.25, A + 4.3), (-1.25, A + 3.6)] + \
          [(-(3.0 - 1.0 * (1 - math.cos(t)) - 0.6 * t), A + 0.6 + 2.4 * math.sin(t)) for t in np.linspace(math.pi / 2, 0.15, 7)] + \
          [(-3.0, A + 0.6)]
    parts.append(paint(prism_zy(att, X0 + 0.5, X1 - 0.5, BRICK * 1.02), BRICK * 1.02, 34))
    for xf, sg in ((X0 + 0.5, -1), (X1 - 0.5, 1)):
        parts.append(B(*sorted((xf, xf + sg * 0.25)), A + 0.7, A + 3.2, -1.0, 1.0, LIGHT))                  # eagle panel
        parts.append(sphere(0.55, LIGHT, 10, 6).S(0.35, 1.0, 1.5).T(xf + sg * 0.3, A + 2.4, 0))              # spread wings
        parts.append(cyl(0.3, 0.36, 0.45, LIGHT, 10).T(xf + sg * 0.25, A + 3.2, 0))                         # crown
        parts.append(sphere(0.75, LIGHT, 12, 6).S(0.3, 0.8, 1.0).T(xf + sg * 0.2, A + 4.0, 0))              # shell
    parts.append(B(X0 + 0.4, X1 - 0.4, A + 4.3, A + 4.9, -0.7, 0.7, GRANITE * 1.05))
    parts.append(B(X0 + 0.7, X1 - 0.7, A + 4.9, A + 5.4, -0.45, 0.45, GRANITE * 1.05))
    parts.append(sphere(0.42, GRANITE * 1.08, 12, 7).T((X0 + X1) / 2, A + 5.8, 0))
    # side pedestals with pairs of balls
    for s in (-1, 1):
        parts.append(B(X0 + 0.2, X1 - 0.2, A, A + 1.9, *sorted((s * 3.15, s * 4.6)), GRANITE * 1.03))
        parts.append(B(X0 + 0.1, X1 - 0.1, A + 1.9, A + 2.2, *sorted((s * 3.05, s * 4.7)), GRANITE * 1.07))
        for xb in (X0 + 1.05, X1 - 1.05):
            parts.append(sphere(0.42, GRANITE * 1.08, 12, 7).T(xb, A + 2.62, s * 3.9))
    return parts


def main():
    out = ROOT / "models" / "Q581532"
    out.mkdir(parents=True, exist_ok=True)
    parts = body() + west_tower() + east_arch()
    m = merge(parts)
    tris, size = write_glb(out / "model.glb", [("puente_de_alcantara", m)], roughness=0.9)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
