"""
Puente de San Martín, Toledo (Q1518670) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: medieval bridge over the Tagus on the west side of Toledo, first built in the 13th century, burnt
about 1355 and damaged in 1368, rebuilt about 1390 by archbishop Pedro Tenorio, who made the great central arch
and the two crenellated gate-towers at its ends; the approaches widened under Carlos II, the outer tower and the
inner face of the city tower bear 16th-century arms (Charles V's imperial eagle between two seated kings).
Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 19735463 (man_made=bridge): the deck 126.6 m long between the towers and 6.4 m wide,
  axis at 42.4° (the footway, way 23495005); the piers seen from above: the big pier on the city side of the main
  arch (x −33.0 to −20.4, 12.6 m) with its two rounded buttresses (3.3 m out downstream, 4.2 m upstream), and the
  two rectangular buttresses of the piers at x 20.1–25.4 and −51.3 to −46.6 (1–1.6 m out). So the main arch
  spans 40.5 m (Wikipedia: 40 m). Way 257932774: the city (north-east) tower, an irregular hexagon of 15 × 17 m
  with the gate house on the bridge axis; way 257932776: the outer (south-west) tower, 13 × 17 m.
- IGN MDT05 (tools/terrain.py -> terrain.json): the Tagus at 443.0 m; the road beyond the outer tower at
  464–465 m (+21.5 m), the street along the city tower at 467–468 m (+24.5 m). So the deck runs from +21.5 m at
  the outer tower up to +24.5 m at the city tower.
- Photos on Commons (heights by proportion, ±10 %): the pointed central arch rises about 21 m over the water
  from springings just above it, the deck about 24–25 m; on the south-west side a stilted semicircular arch of
  about 17 m and a small one of about 7 m next to the tower; on the city side two semicircular arches of about
  13.6 and 10 m, also stilted; the outer tower about 16 m over the deck, with a pointed horseshoe arch, a statue
  niche, a gallery of seven small arches and pyramid-capped merlons; the city tower about 22 m over the street,
  the gate house on its bridge side about 13.5 m to the eaves with a hipped tiled roof.

Axes (docs/MODEL_SPEC.md): +Y up, y = 0 the river (443.0 m); the bridge runs along X (+X to the south-west,
222.4°, towards the outer tower), the downstream (north-west) elevation faces +Z (312.4°); origin on the water
under the middle of the main arch, on the line x = 0 across the deck (deck centre at z = 0.2).

    python model_sources/Q1518670_puente_de_san_martin_toledo.py   -> models/Q1518670/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, extrude, triangulate, fbm, normals, offset, write_glb  # noqa: E402

GRANITE = np.array((0.75, 0.70, 0.61))     # ashlar of arches, piers, parapets
SPANDREL = np.array((0.72, 0.66, 0.55))
TOWER = np.array((0.71, 0.65, 0.55))       # granite ashlar and rubble of the towers
PLASTER = np.array((0.80, 0.70, 0.57))
TILE = np.array((0.60, 0.35, 0.25))
LIGHT = np.array((0.84, 0.81, 0.74))
PAVING = np.array((0.62, 0.61, 0.59))
DARK = np.array((0.15, 0.13, 0.12))
IRON = (0.12, 0.11, 0.10)

ZC, W = 0.2, 3.2                           # deck centre line and half width (OSM z −3.0 to 3.4)
X_NE, X_SW = -65.3, 60.9                   # the bridge faces of the city tower and of the outer tower

# MDT05 ground along the axis (lowest across the deck), metres above the river (443.0 m)
GROUND = [(-96, 27.2), (-88, 24.0), (-84, 22.2), (-80, 20.0), (-76, 17.7), (-72, 15.2), (-68, 12.3), (-64, 9.5),
          (-60, 6.9), (-56, 4.4), (-52, 2.0), (-48, 1.0), (-44, 0.4), (-40, 0.0), (32, 0.0), (36, 0.5), (40, 1.6),
          (44, 4.0), (48, 6.2), (52, 8.9), (56, 10.5), (60, 12.3), (64, 14.5), (68, 16.7), (72, 18.5), (76, 19.9),
          (80, 20.7), (84, 21.1), (88, 21.5)]
DECK = [(-80.0, 24.5), (X_NE, 24.5), (0.0, 24.4), (X_SW, 22.0), (78.0, 21.5)]

# arches from the city to the outer tower: (west end, east end, springing, kind)
MAIN = (-20.4, 20.1, 1.0, 21.0)            # pointed: xa, xb, springing, radius of the two arcs (rise 21 m)
ARCHES = [(-61.3, -51.3, 17.4), (-46.6, -33.0, 15.6), (25.4, 42.4, 12.6), (46.4, 53.4, 17.1)]


def ground(x):
    xs, ys = zip(*GROUND)
    return float(np.interp(x, xs, ys))


def deck(x):
    xs, ys = zip(*DECK)
    return float(np.interp(x, xs, ys))


def tint(c, v, seed, amp=0.10, freq=0.35):
    n = fbm(v, freq, seed, 3)
    return np.clip(np.asarray(c)[None, :] * (1 - amp / 2 + amp * n[:, None]), 0, 1)


def paint(m, c, seed, amp=0.10):
    m.c = tint(c, m.v, seed, amp)
    return m


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


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
    xc, r = (xa + xb) / 2, (xb - xa) / 2 + extra
    return [(xc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(math.pi, 0, n + 1)]


def pointed(xa, xb, ys, R, n=16, extra=0.0):
    """Intrados (or, with extra, extrados) of a pointed arch of two arcs of radius R, left to right."""
    xm = (xa + xb) / 2
    cl, cr, r = xa + R, xb - R, R + extra
    at = math.acos(max(-1.0, min(1.0, (xm - cl) / r)))
    left = [(cl + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(math.pi, at, n + 1)]
    right = [(cr + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(math.pi - at, 0, n + 1)]
    return left + right[1:]


def ring_prism(outer, inner, z0, z1, c):
    return prism_xy(outer + inner[::-1], z0, z1, c)


def clip(P, lo=None, hi=None):
    """Sutherland–Hodgman: the part of polygon P (x, z) with lo <= z <= hi."""
    def cut(P, keep, zc):
        out = []
        for i in range(len(P)):
            a, b = P[i - 1], P[i]
            ia, ib = keep(a[1]), keep(b[1])
            if ia != ib:
                t = (zc - a[1]) / (b[1] - a[1])
                out.append((a[0] + t * (b[0] - a[0]), zc))
            if ib:
                out.append(b)
        return out
    if lo is not None:
        P = cut(P, lambda z: z >= lo, lo)
    if hi is not None and P:
        P = cut(P, lambda z: z <= hi, hi)
    return P


def edge_box(p0, p1, t0, t1, y0, y1, d0, d1, c):
    """A box lying along the edge p0 -> p1 (x, z) of a plan, between fractions t0..t1 of it, from y0 to y1, and
    from d0 to d1 metres off the edge (positive to the left of the edge direction)."""
    (xa, za), (xb, zb) = p0, p1
    L = math.hypot(xb - xa, zb - za)
    ux, uz = (xb - xa) / L, (zb - za) / L
    nx, nz = -uz, ux
    tm, dm = (t0 + t1) / 2, (d0 + d1) / 2
    ang = math.atan2(-uz, ux)
    m = box(L * (t1 - t0), y1 - y0, abs(d1 - d0), c).R(0, ang, 0)
    return m.T(xa + ux * L * tm + nx * dm, y0, za + uz * L * tm + nz * dm)


def merlons_ring(P, y0, c, w=0.75, h=1.0, t=0.55, gap=0.6, inset=0.3, caps=True):
    """Merlons with little pyramid caps all along a closed plan P (x, z), set inset metres inside its edge."""
    Q = offset(P, -inset)
    out = []
    for i in range(len(Q)):
        (xa, za), (xb, zb) = Q[i], Q[(i + 1) % len(Q)]
        L = math.hypot(xb - xa, zb - za)
        n = max(1, int((L + gap) / (w + gap)))
        ang = math.atan2(-(zb - za), xb - xa)
        for k in range(n):
            s = (k + 0.5) / n
            x, z = xa + (xb - xa) * s, za + (zb - za) * s
            out.append(box(w, h, t, c).R(0, ang, 0).T(x, y0, z))
            if caps:
                out.append(cyl(w * 0.62, 0.0, 0.32, c, 4).R(0, ang + math.pi / 4, 0).T(x, y0 + h, z))
    return out


def parapet_ring(P, y0, y1, t, c):
    return extrude(P, y0, y1, c, holes=[offset(P, -t)])


def hip_roof(P, y0, h, c, over=0.45):
    """A hipped roof over a convex-ish plan: every eave edge rises to a ridge along the long axis."""
    Q = np.array(offset(P, over))
    cen = Q.mean(0)
    d = Q - cen
    cov = d.T @ d
    w, v = np.linalg.eigh(cov)
    ax = v[:, 1]
    proj = d @ ax
    across = d @ v[:, 0]
    half = max(0.0, (proj.max() - proj.min()) / 2 - (across.max() - across.min()) / 2)
    mid = cen + ax * (proj.max() + proj.min()) / 2
    R0, R1 = mid - ax * half, mid + ax * half
    V, F = [], []
    for i in range(len(Q)):
        a, b = Q[i], Q[(i + 1) % len(Q)]
        ra = R0 if np.dot(a - mid, ax) < 0 else R1
        rb = R0 if np.dot(b - mid, ax) < 0 else R1
        o = len(V)
        V += [(a[0], y0, a[1]), (b[0], y0, b[1]), (rb[0], y0 + h, rb[1])]
        F.append((o, o + 2, o + 1))
        if not np.allclose(ra, rb):
            o = len(V)
            V += [(a[0], y0, a[1]), (rb[0], y0 + h, rb[1]), (ra[0], y0 + h, ra[1])]
            F.append((o, o + 2, o + 1))
    m = Mesh(V, F, c)
    a, b, cc = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    down = np.cross(b - a, cc - a)[:, 1] < 0                    # wind every face upwards
    m.f[down] = m.f[down][:, ::-1]
    return m


def passage_block(P, y0, y1, zlo, zhi, spring, c, seed):
    """A tower body from plan P (x, z) between y0 and y1, pierced along x by an arched passage zlo..zhi
    (semicircular, springing at spring, floor at y0)."""
    parts = []
    for part in (clip(P, hi=zlo), clip(P, lo=zhi)):
        if len(part) >= 3:
            parts.append(paint(extrude(part, y0, y1, c), c, seed))
    mid = clip(P, lo=zlo, hi=zhi)
    xs = [p[0] for p in mid]
    x0, x1 = min(xs), max(xs)
    r = (zhi - zlo) / 2
    zc = (zlo + zhi) / 2
    arch = [(zlo, y1), (zlo, spring)] + [(zc - r * math.cos(a), spring + r * math.sin(a)) for a in np.linspace(0, math.pi, 13)][1:-1] + \
           [(zhi, spring), (zhi, y1)]
    parts.append(paint(prism_zy(arch, x0, x1, c), c, seed + 1))
    parts.append(B(x0, x1, y0 - 0.02, y0 + 0.02, zlo, zhi, PAVING))
    return parts, (x0, x1)


def shape(zc, y0, w, h, top):
    """Outline of an opening or panel, from its bottom-left corner up and over to its bottom-right corner."""
    pts = [(zc - w / 2, y0)]
    if top == "round":
        r = w / 2
        pts += [(zc + r * math.cos(a), y0 + h - r + r * math.sin(a)) for a in np.linspace(math.pi, 0, 11)]
    elif top == "pointed":
        R = w * 0.8
        pts += pointed(zc - w / 2, zc + w / 2, y0 + h - math.sqrt(R ** 2 - (R - w / 2) ** 2), R, 6)
    else:
        pts += [(zc - w / 2, y0 + h), (zc + w / 2, y0 + h)]
    pts.append((zc + w / 2, y0))
    return pts


def x_plate(xf, sg, zc, y0, w, h, c, d=0.06, top="round", hole=None):
    """A thin plate on a face normal to x (sg = +1 for a +x face): an opening, a panel, an arched niche; with
    hole = (w, h, top) it becomes a frame around an opening that reaches its foot (a gate)."""
    pts = shape(zc, y0, w, h, top)
    if hole:
        pts += shape(zc, y0, *hole)[::-1]
    xa, xb = sorted((xf, xf + sg * d))
    return prism_zy(pts, xa, xb, c)


def curved_wall(pts, y0, y1, t, c):
    return merge([edge_box(p0, p1, 0.0, 1.0, y0, y1, -t / 2, t / 2, c) for p0, p1 in zip(pts[:-1], pts[1:])])


# ── The bridge body ──────────────────────────────────────────────────────────
def body():
    parts = []
    top = [(x, deck(x)) for x in np.linspace(X_NE, X_SW, 32)]
    b = []

    def down_to_ground(x0, x1, n=3):
        for x in np.linspace(x0, x1, n):
            b.append((x, min(ground(x) - 2.5, -3.0) if ground(x) < 0.3 else ground(x) - 2.5))

    down_to_ground(X_SW, 53.4)
    a1 = ARCHES[3]
    b.append((53.4, a1[2])); b += semi(a1[0], a1[1], a1[2], 14)[::-1][1:-1]; b.append((46.4, a1[2]))
    down_to_ground(46.4, 42.4, 2)
    a2 = ARCHES[2]
    b.append((42.4, a2[2])); b += semi(a2[0], a2[1], a2[2], 24)[::-1][1:-1]; b.append((25.4, a2[2]))
    b += [(25.4, -3.0), (20.1, -3.0), (20.1, MAIN[2])]
    b += pointed(MAIN[0], MAIN[1], MAIN[2], MAIN[3], 18)[::-1][1:-1]
    b += [(-20.4, MAIN[2]), (-20.4, -3.0), (-33.0, -3.0)]
    a4 = ARCHES[1]
    b.append((-33.0, a4[2])); b += semi(a4[0], a4[1], a4[2], 20)[::-1][1:-1]; b.append((-46.6, a4[2]))
    down_to_ground(-46.6, -51.3, 2)
    a5 = ARCHES[0]
    b.append((-51.3, a5[2])); b += semi(a5[0], a5[1], a5[2], 16)[::-1][1:-1]; b.append((-61.3, a5[2]))
    down_to_ground(-61.3, X_NE, 2)
    m = prism_xy(top + b, ZC - W, ZC + W, SPANDREL)
    n = normals(m)
    paint(m, SPANDREL, 3)
    up = n[:, 1] > 0.8
    m.c[up] = tint(PAVING, m.v[up], 4, 0.06)
    parts.append(m)

    # voussoirs and imposts
    z0, z1 = ZC - W - 0.07, ZC + W + 0.07
    parts.append(paint(ring_prism(pointed(*MAIN, 18, 1.6), pointed(*MAIN, 18), z0, z1, GRANITE), GRANITE, 5))
    for k, (xa, xb, ys) in enumerate(ARCHES):
        t = 1.2 if xb - xa > 12 else 0.95
        parts.append(paint(ring_prism(semi(xa, xb, ys, 18, t), semi(xa, xb, ys, 18), z0, z1, GRANITE), GRANITE, 6 + k))
        for x0, x1 in ((xa - 0.5, xa), (xb, xb + 0.5)):
            parts.append(B(x0, x1, ys - 0.4, ys, z0 - 0.05, z1 + 0.05, GRANITE * 1.05))
    # the piers in ashlar (lighter than the spandrels)
    for xa, xb in ((-51.3, -46.6), (-33.0, -20.4), (20.1, 25.4), (42.4, 46.4)):
        yb = -3.0 if ground((xa + xb) / 2) < 0.3 else ground((xa + xb) / 2) - 2.5
        top_y = {(-51.3): 17.4, (-33.0): 15.6, 20.1: 12.6, 42.4: 12.6}[xa]
        parts.append(paint(B(xa, xb, yb, top_y, z0 + 0.03, z1 - 0.03, GRANITE * 0.98), GRANITE * 0.98, 15))

    # the big pier on the city side: rounded buttresses, stepped upstream, a round balcony downstream
    xc, ax = -26.7, 6.3
    yd = deck(xc)

    def half_ellipse(zb, bz, a=ax, n=14):
        s = 1 if bz > 0 else -1
        return [(xc + a * math.cos(t), zb + bz * math.sin(t)) for t in np.linspace(0, math.pi, n + 1)][::s]

    up_b = half_ellipse(ZC + W, 3.3)
    parts.append(paint(extrude(up_b, -3.0, yd, GRANITE, top_c=PAVING), GRANITE, 20))
    for k, (y1, gr) in enumerate(((3.0, 1.6), (7.0, 0.8))):
        parts.append(paint(extrude(half_ellipse(ZC + W, 3.3 + gr, ax + gr * 0.6), -3.0, y1, GRANITE * 0.96), GRANITE * 0.96, 21 + k))
    parts.append(paint(curved_wall(half_ellipse(ZC + W, 3.05, ax - 0.25, 18), yd, yd + 1.0, 0.45, GRANITE * 1.02),
                       GRANITE * 1.02, 23))
    parts.append(paint(extrude(half_ellipse(ZC + W, 3.55, ax + 0.25), yd - 0.9, yd - 0.4, GRANITE * 1.06), GRANITE * 1.06, 24))
    for k in range(8):                                                     # the steps of the upstream buttress
        y1 = 1.5 + 2.4 * k
        g = (8 - k) * 0.42
        parts.append(paint(extrude(half_ellipse(ZC - W, -(4.2 + g), ax + g * 0.5), -3.0, y1, GRANITE * (0.95 + 0.01 * (k % 2))),
                           GRANITE * 0.96, 30 + k))
    parts.append(paint(extrude(half_ellipse(ZC - W, -4.2), -3.0, yd, GRANITE, top_c=PAVING), GRANITE, 40))
    parts.append(paint(curved_wall(half_ellipse(ZC - W, -3.95, ax - 0.25, 18), yd, yd + 1.0, 0.45, GRANITE * 1.02),
                       GRANITE * 1.02, 41))

    # rectangular buttresses (OSM) with pilaster strips; corbelled refuges at the top
    for xa, xb, out_n, out_s in ((20.1, 25.4, 1.2, 1.1), (-51.3, -46.6, 1.4, 1.6)):
        ybot = -3.0 if ground(xa) < 0.3 else ground(xb) - 2.5
        yd = deck((xa + xb) / 2)
        for s, o in ((1, out_n), (-1, out_s)):
            zf = ZC + s * W
            parts.append(paint(B(xa, xb, ybot, yd - 1.3, *sorted((zf, zf + s * o)), GRANITE), GRANITE, 42))
            for xp in (xa + 0.9, xb - 0.9):
                parts.append(B(xp - 0.35, xp + 0.35, ybot + 4, yd - 1.3, *sorted((zf + s * o, zf + s * (o + 0.18))), GRANITE * 1.04))
            parts.append(B(xa - 0.3, xb + 0.3, yd - 0.55, yd, *sorted((zf, zf + s * (o + 0.6))), GRANITE * 1.04))
            for xp in np.linspace(xa, xb, 5):
                parts.append(B(xp - 0.22, xp + 0.22, yd - 1.3, yd - 0.55, *sorted((zf + s * o, zf + s * (o + 0.5))), GRANITE))
            parts.append(B(xa - 0.3, xb + 0.3, yd, yd + 1.0, *sorted((zf + s * (o + 0.15), zf + s * (o + 0.6))), GRANITE * 1.02))
            for x0 in (xa - 0.3, xb - 0.15):
                parts.append(B(x0, x0 + 0.45, yd, yd + 1.0, *sorted((zf, zf + s * (o + 0.6))), GRANITE * 1.02))

    # cornice band, parapets with coping, balls on pedestals, lamps
    gaps = [(-33.0, -20.4), (-51.3, -46.6), (20.1, 25.4)]
    for s in (-1, 1):
        zf = ZC + s * W
        runs = [(X_NE, -51.3), (-46.6, -33.0), (-20.4, 20.1), (25.4, X_SW)]
        for xa, xb in runs:
            xs = np.linspace(xa, xb, max(3, int((xb - xa) / 6)))
            band = [(x, deck(x) - 0.6) for x in xs] + [(x, deck(x) - 0.3) for x in xs[::-1]]
            parts.append(prism_xy(band, *sorted((zf, zf + s * 0.14)), GRANITE * 1.03))
            wall = [(x, deck(x) - 0.05) for x in xs] + [(x, deck(x) + 0.9) for x in xs[::-1]]
            parts.append(paint(prism_xy(wall, *sorted((zf - s * 0.5, zf)), GRANITE * 1.03), GRANITE * 1.03, 44))
            cope = [(x, deck(x) + 0.9) for x in xs] + [(x, deck(x) + 1.08) for x in xs[::-1]]
            parts.append(prism_xy(cope, *sorted((zf - s * 0.56, zf + s * 0.06)), LIGHT * 0.95))
            for x in np.arange(xa + 3.0, xb - 2.0, 8.0):
                zb = zf - s * 0.25
                parts.append(B(x - 0.22, x + 0.22, deck(x) + 1.08, deck(x) + 1.35, zb - 0.22, zb + 0.22, LIGHT))
                parts.append(sphere(0.3, LIGHT, 10, 6).T(x, deck(x) + 1.63, zb))
        for x in (-40.0, -8.0, 12.0, 34.0, 50.0) if s > 0 else ():
            zl = zf - 0.25
            parts.append(cyl(0.07, 0.05, 3.3, IRON, 6).T(x, deck(x) + 1.08, zl))
            parts.append(B(x - 0.18, x + 0.18, deck(x) + 4.35, deck(x) + 4.9, zl - 0.18, zl + 0.18, IRON))
    return parts


# ── The outer (south-west) gate-tower ───────────────────────────────────────
SW_PLAN = [(X_SW, -8.6), (72.4, -9.1), (73.8, -7.8), (73.8, 2.6), (69.0, 7.6), (X_SW, 7.9)]
SW_BASTION = [(73.8, -7.8), (77.2, -4.6), (77.0, -1.5), (73.8, -1.7)]


def sw_tower():
    parts = []
    D = deck(X_SW)
    HEAD = D + 14.9                        # wall head; pyramid-capped merlons to about D + 16.2
    base = ground(X_SW) - 3.0
    zlo, zhi = ZC - 1.5, ZC + 1.5
    parts.append(paint(extrude(offset(SW_PLAN, 0.35), base, D - 0.3, TOWER * 0.93), TOWER * 0.93, 50))
    parts.append(paint(extrude(SW_BASTION, base, D + 1.0, TOWER * 0.95), TOWER * 0.95, 51))
    ps, (x0, x1) = passage_block(SW_PLAN, D, HEAD, zlo, zhi, D + 3.0, TOWER, 52)
    parts += ps
    parts += merlons_ring(SW_PLAN, HEAD, TOWER)
    # string course at D + 8.8 all round
    parts.append(parapet_ring(offset(SW_PLAN, 0.12), D + 8.7, D + 8.95, 0.3, GRANITE * 1.05))
    # bridge face (x = X_SW, facing −x): the great pointed horseshoe arch, with the statue niche and the gate
    xf, sg = X_SW, -1
    gate = (3.0, 4.5, "round")
    parts.append(x_plate(xf, sg, ZC, D, 3.9, 8.2, GRANITE * 1.08, 0.2, "pointed", hole=(3.3, 7.8, "pointed")))
    parts.append(x_plate(xf, sg, ZC, D, 3.3, 7.8, TOWER * 1.04, 0.08, "pointed", hole=gate))
    parts.append(x_plate(xf, sg, ZC, D, 3.7, 4.9, GRANITE * 1.06, 0.14, "round", hole=gate))
    parts.append(x_plate(xf, sg, ZC, D + 4.9, 0.9, 1.7, DARK * 2.0, 0.24, "round"))          # niche
    parts.append(cyl(0.22, 0.18, 1.1, LIGHT, 8).T(xf - 0.3, D + 5.0, ZC))                     # San Ildefonso
    parts.append(sphere(0.16, LIGHT, 8, 5).T(xf - 0.3, D + 6.25, ZC))
    parts.append(x_plate(xf, sg, ZC, D + 6.9, 1.6, 0.45, LIGHT * 0.95, 0.24, "flat"))
    parts.append(x_plate(xf, sg, ZC, D + 9.0, 3.4, 1.5, LIGHT * 0.95, 0.12, "round"))          # carved tympanum
    for z in (ZC - 1.9, ZC + 1.9):
        parts.append(B(xf - 0.3, xf, D + 8.4, D + 10.6, z - 0.15, z + 0.15, LIGHT))           # its little pinnacles
    for k in range(7):                                                                        # the gallery
        z = ZC - 4.8 + k * 1.6
        parts.append(x_plate(xf, sg, z, D + 11.0, 0.75, 1.3, DARK, 0.05, "round"))
        parts.append(x_plate(xf, sg, z, D + 10.9, 1.05, 1.55, TOWER * 1.12, 0.03, "round"))
    # outer face (+x, towards the road): pointed arch and the arms of Toledo
    xo = 73.8
    parts.append(x_plate(xo, 1, ZC, D, 3.9, 7.2, GRANITE * 1.08, 0.16, "pointed", hole=gate))
    parts.append(x_plate(xo, 1, ZC, D + 8.0, 1.6, 2.0, LIGHT, 0.15, "round"))
    for k in range(5):
        z = ZC - 3.2 + k * 1.6
        parts.append(x_plate(xo, 1, z, D + 11.0, 0.75, 1.3, DARK, 0.05, "round"))
    # arrow slits on the side faces
    for zf, sgn in ((-8.6, -1), (7.9, 1)):
        for xx in (64.0, 70.0):
            for yy in (D + 4.0, D + 9.5):
                parts.append(B(xx - 0.1, xx + 0.1, yy, yy + 1.0, zf + sgn * 0.02 - 0.03, zf + sgn * 0.02 + 0.03, DARK))
    return parts


# ── The city (north-east) gate-tower: the polygonal keep and the gate house ─
NE_KEEP = [(-65.3, -2.4), (-70.6, -0.1), (-78.2, 1.4), (-79.5, -5.4), (-76.6, -8.8), (-69.7, -10.0), (-66.7, -7.9)]
NE_GATE = [(-65.3, -2.4), (-64.7, 1.0), (-64.8, 3.9), (-65.2, 5.1), (-69.0, 7.1), (-70.0, 2.8), (-70.6, -0.1)]


def ne_tower():
    parts = []
    D = deck(X_NE)
    base = ground(X_NE) - 3.0
    HEAD = D + 20.6
    keep = NE_KEEP[::-1]
    parts.append(paint(extrude(offset(NE_KEEP, 0.4), base, D - 2.0, TOWER * 0.92), TOWER * 0.92, 60))
    parts.append(paint(extrude(NE_KEEP, D - 2.0, HEAD, TOWER), TOWER, 61))
    parts.append(paint(parapet_ring(offset(NE_KEEP, 0.15), D + 13.0, D + 13.3, 0.3, GRANITE * 1.05), GRANITE * 1.05, 62))
    parts += merlons_ring(NE_KEEP, HEAD, TOWER, w=0.8, h=1.2)
    # arrow slits and a cross-shaped loophole on the keep's faces
    for i in range(len(NE_KEEP)):
        p0, p1 = NE_KEEP[i], NE_KEEP[(i + 1) % len(NE_KEEP)]
        if math.dist(p0, p1) < 3.0 or i == 0:
            continue
        for yy in (D + 6.0, D + 15.0):
            parts.append(edge_box(p0, p1, 0.47, 0.53, yy, yy + 1.2, -0.03, 0.03, DARK))
        parts.append(edge_box(p0, p1, 0.44, 0.56, D + 16.0, D + 16.15, -0.03, 0.03, DARK))
    # the gate house on the bridge axis, pierced by the passage, under a hipped tiled roof
    EAVES = D + 13.5
    parts.append(paint(extrude(offset(NE_GATE, 0.3), base, D - 0.3, TOWER * 0.92), TOWER * 0.92, 64))
    ps, (x0, x1) = passage_block(NE_GATE, D, EAVES, ZC - 1.5, ZC + 1.5, D + 3.3, PLASTER, 65)
    parts += ps
    parts.append(paint(hip_roof(NE_GATE, EAVES, 2.3, TILE), TILE, 66, 0.08))
    parts.append(parapet_ring(offset(NE_GATE, 0.2), EAVES - 0.3, EAVES, 0.5, GRANITE * 1.05))
    # its bridge face: ashlar frame, arch, Charles V's eagle between two seated kings, two tablets
    xf, sg = -64.7, 1
    parts.append(paint(B(xf - 0.2, xf + 0.15, D + 5.4, D + 9.0, ZC - 3.0, ZC + 3.2, TOWER * 1.03), TOWER * 1.03, 67))
    parts.append(x_plate(xf, sg, ZC, D, 4.2, 5.6, GRANITE * 1.1, 0.25, "round", hole=(3.0, 4.8, "round")))
    parts.append(x_plate(xf + 0.15, sg, ZC, D + 5.8, 1.9, 2.6, LIGHT, 0.22, "round"))         # the eagle
    parts.append(sphere(0.7, LIGHT, 10, 6).S(0.3, 0.9, 1.5).T(xf + 0.45, D + 7.2, ZC))
    parts.append(cyl(0.3, 0.36, 0.4, LIGHT, 10).T(xf + 0.4, D + 8.3, ZC))
    for z in (ZC - 1.9, ZC + 1.9):
        parts.append(x_plate(xf + 0.15, sg, z, D + 6.6, 1.0, 1.6, LIGHT * 0.95, 0.16, "round"))   # the kings
        parts.append(cyl(0.28, 0.2, 0.9, LIGHT, 8).T(xf + 0.42, D + 6.75, z))
        parts.append(sphere(0.15, LIGHT, 8, 5).T(xf + 0.42, D + 7.8, z))
        parts.append(x_plate(xf + 0.15, sg, z, D + 5.0, 1.15, 1.45, LIGHT * 1.05, 0.12, "flat"))  # tablets
    # its north-west face: the window and the open loggia over the street (seen from the city)
    p0, p1 = NE_GATE[4], NE_GATE[3]
    parts.append(edge_box(p0, p1, 0.3, 0.7, D + 9.6, D + 11.6, -0.04, 0.04, DARK))
    parts.append(edge_box(p0, p1, 0.2, 0.8, D + 5.6, D + 8.4, -0.05, 0.05, DARK * 1.4))
    parts.append(edge_box(p0, p1, 0.15, 0.85, D + 5.4, D + 5.6, -0.5, 0.05, GRANITE))
    parts.append(edge_box(p0, p1, 0.15, 0.85, D + 9.2, D + 9.4, -0.3, 0.05, GRANITE))
    return parts


def main():
    out = ROOT / "models" / "Q1518670"
    out.mkdir(parents=True, exist_ok=True)
    parts = body() + sw_tower() + ne_tower()
    m = merge(parts)
    tris, size = write_glb(out / "model.glb", [("puente_de_san_martin", m)], roughness=0.9)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
