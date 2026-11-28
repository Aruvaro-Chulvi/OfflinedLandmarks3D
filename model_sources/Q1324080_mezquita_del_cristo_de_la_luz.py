"""
Mezquita del Cristo de la Luz, or Mosque of Bab al-Mardum (Q1324080), Toledo — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: a small neighbourhood mosque finished in December 999 (foundation inscription: built for Ahmad ibn
Hadidi under the direction of Musa ibn Ali and Sa'ada); the Mudéjar apse added to the east about 1187, when it
became the hermitage of the Holy Cross (later Cristo de la Luz); public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 114522030: the square mosque 8.66 × 8.8 m (Wikipedia: about 9 m) and, to the north-east,
  the 12th-century addition, a straight bay 3.8 m long and a semicircular apse as wide as the mosque; long axis
  at 57.4°.
- Main façade: the south-west one on Calle Cristo de la Luz, with the foundation inscription, facing 237.4°.
- Heights measured on a frontal photo of that façade (Wikimedia Commons, «Fachada sudoeste de la Mezquita del
  Cristo de la Luz (Toledo).jpg», scaled by the 8.66 m of its width, ±5 %): three doorways (the central one with a
  horseshoe arch, the side ones lobed) up to 2.6–2.9 m, the band of interlaced blind arches 3.1–4.7 m, the sebka
  lattice 5.2–5.8 m, the brick inscription and the corbelled cornice from 6.8 to 7.4 m. The north-west façade on
  the patio has three horseshoe arches inside round ones and polylobed arches with two-colour voussoirs above;
  the apse two tiers of blind arcades (round arches below, horseshoe arches above) on a masonry plinth (photos
  «Apse of Cristo de la Luz, Toledo, Spain.jpg», «Mezquita de Bab al-Mardum (Toledo).jpg»).
- The roofs are seldom seen: low tiled hipped roofs, the central bay of the nine raised a little (as its dome is
  inside), and a half cone over the apse — estimated.
- The garden of Cristo de la Luz (OSM way 257671838) and the patio between it and the mosque go down the slope to
  the north-west: three terraces (levels from the IGN MDT05), the patio with the well, the crossing paths with a
  small pool (chahar bagh, Wikipedia), trees and a palm from the orthophoto and the photos.
Brick and Toledo masonry (rubble with brick courses), Arab tiles.

Axes (docs/MODEL_SPEC.md): +Y up, the south-west façade faces +Z, +X to the south-east (heading − 90°), origin at the
centroid of the OSM outline; y = 0 is the street in front of the middle of the south-west façade (ground_at). The
street rises to the south-east and the ground falls about 8 m to the bottom of the garden ("ground": "keep").

    python model_sources/Q1324080_mezquita_del_cristo_de_la_luz.py   -> models/Q1324080/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, quad, extrude, offset, crumple, noise3, write_glb  # noqa: E402

rng = np.random.default_rng(999)

BRICK = np.array((0.73, 0.50, 0.37))
BRICK_L = np.array((0.84, 0.70, 0.55))
MASON = np.array((0.60, 0.56, 0.50))
RECESS = np.array((0.60, 0.43, 0.33))
ROOF = np.array((0.60, 0.45, 0.36))
DARK = (0.10, 0.09, 0.08)
IRON = (0.08, 0.09, 0.08)
SAND = (0.74, 0.62, 0.46)
PAVE = (0.70, 0.66, 0.60)
GRASS = (0.38, 0.47, 0.24)
WATER = (0.22, 0.36, 0.42)
FOUND = -4.0

XW = 4.33                     # half-width of the mosque and of the apse
ZF, ZM = 8.13, -0.68          # the south-west façade and the old qibla wall (start of the 12th-century bay)
ZA = -4.6                     # centre of the apse half circle
EAVE = 6.8                    # top of the walls, under the corbelled cornice
CORN = 7.4
PATIO = -1.6
TERR = [(-18.0, -3.4), (-25.0, -5.2), (-40.0, -7.2)]     # (x of the upper edge, level) of the garden terraces
GARDEN_RING = [(-8.22, -29.98), (-14.31, -2.13), (-33.34, -6.05), (-26.48, -33.63)]   # OSM way 257671838


def jit(c, k=1.0, s=0.03):
    return tuple(np.clip(np.asarray(c) * k * (1 + rng.normal(0, s)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def frustum(ax, az, y0, bx, bz, y1, c, top=True):
    lo = [(-ax, y0, -az), (ax, y0, -az), (ax, y0, az), (-ax, y0, az)]
    hi = [(-bx, y1, -bz), (bx, y1, -bz), (bx, y1, bz), (-bx, y1, bz)]
    F = []
    for i in range(4):
        j = (i + 1) % 4
        F += [(i, 4 + j, j), (i, 4 + i, 4 + j)]
    if top and bx > 0 and bz > 0:
        F += [(4, 6, 5), (4, 7, 6)]
    return Mesh(lo + hi, F, c)


def arch_face(w, h, c, seg=10, horseshoe=0.0):
    """A flat arched shape facing +z, foot at y = 0, centred on x = 0 (horseshoe > 0 closes the arch below its
    springing, as a fraction of the radius)."""
    r = w / 2 * (1 + horseshoe)
    yc = h - r
    t0 = -math.asin(min(0.95, horseshoe)) if horseshoe else 0.0
    pts = [(r * math.cos(t), yc + r * math.sin(t)) for t in np.linspace(t0, math.pi - t0, seg + 1)]
    V = [(0.0, 0.0, 0.0), (w / 2, 0.0, 0.0)] + [(x, y, 0.0) for x, y in pts] + [(-w / 2, 0.0, 0.0)]
    F = [(0, i, i + 1) for i in range(1, len(V) - 1)]
    return Mesh(V, F, c)


def arch_ring(w, h, c, r_tube=0.09, seg=10, horseshoe=0.0):
    """The brick voussoirs round an arch, as a tube along its curve (facing +z, foot at y = 0)."""
    r = w / 2 * (1 + horseshoe) + r_tube
    yc = h - w / 2 * (1 + horseshoe)
    t0 = -math.asin(min(0.95, horseshoe)) if horseshoe else 0.0
    return tube([(r * math.cos(t), yc + r * math.sin(t), 0.0) for t in np.linspace(t0, math.pi - t0, seg + 1)], r_tube, c, 5)


def place(ms, x, y, z, ang):
    """Turn features built facing +z by ang about y (Ry), then move them to (x, y, z)."""
    return [m.copy().R(0, ang, 0).T(x, y, z) for m in ms]


def blind_arch(w, h, depth=0.05, horseshoe=0.0, ring=BRICK, fill=RECESS):
    return [arch_face(w, h, jit(fill), horseshoe=horseshoe).T(0, 0, depth),
            arch_ring(w, h, jit(ring, 1.05), horseshoe=horseshoe).T(0, 0, depth)]


def cornice_band(ring_pts, y0):
    """The brick cornice: a row of corbels (modillions) under a projecting brick crown, on a closed outline."""
    out = [extrude(offset(ring_pts, 0.05), y0, y0 + 0.25, jit(BRICK_L, 0.95), top=False),
           extrude(offset(ring_pts, 0.32), y0 + 0.4, y0 + 0.6, jit(BRICK, 1.05))]
    P = ring_pts + ring_pts[:1]
    for (xa, za), (xb, zb) in zip(P, P[1:]):
        L = math.hypot(xb - xa, zb - za)
        if L < 1e-6:
            continue
        nx, nz = (zb - za) / L, -(xb - xa) / L          # outward normal (the ring is counter-clockwise in x, z)
        n = max(1, int(L / 0.42))
        for k in range(n):
            t = (k + 0.5) / n
            x, z = xa + (xb - xa) * t, za + (zb - za) * t
            out.append(B(x - 0.08, x + 0.08, y0 + 0.25, y0 + 0.4, z - 0.08, z + 0.08, jit(BRICK, 0.9)).T(nx * 0.14, 0, nz * 0.14))
    return out


def ccw(ring):
    a = sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1] for i in range(len(ring)))
    return ring if a > 0 else ring[::-1]


def apse_ring(n=14):
    """Outline of the 12th-century addition: the straight bay and the half circle."""
    return [(XW, ZM)] + [(XW * math.cos(t), ZA - XW * math.sin(t)) for t in np.linspace(0, math.pi, n + 1)] + [(-XW, ZM)]


# ── The mosque ───────────────────────────────────────────────────────────────
def mosque():
    sq = [(-XW, ZM), (XW, ZM), (XW, ZF), (-XW, ZF)]
    parts = [extrude(sq, FOUND, EAVE, jit(BRICK)),
             extrude(offset(sq, 0.04), FOUND, 0.9, jit(MASON))]                    # masonry plinth
    for y in np.arange(1.3, EAVE - 0.3, 0.55):                                     # brick courses, light and dark
        parts.append(extrude(offset(sq, 0.02), y, y + 0.05, jit(BRICK_L, 0.97), top=False))
    parts += cornice_band(ccw(sq), EAVE)
    # roof: low tiled hip over the square, the central bay raised a little
    parts.append(frustum(XW + 0.3, (ZF - ZM) / 2 + 0.3, CORN - 0.2, 0.6, 0.6, CORN + 1.1, jit(ROOF)).T(0, 0, (ZF + ZM) / 2))
    zc = (ZF + ZM) / 2
    parts.append(B(-1.45, 1.45, CORN, CORN + 1.35, zc - 1.45, zc + 1.45, jit(BRICK)))
    parts += cornice_band(ccw([(-1.45, zc - 1.45), (1.45, zc - 1.45), (1.45, zc + 1.45), (-1.45, zc + 1.45)]), CORN + 1.35)[:2]
    parts.append(frustum(1.75, 1.75, CORN + 1.95, 0.0, 0.0, CORN + 2.6, jit(ROOF)).T(0, 0, zc))

    # South-west façade (+z): three doorways, interlaced arches, the sebka lattice, the inscription band
    f = []
    for x, w, h, hs in ((0.0, 1.45, 2.9, 0.2), (-2.55, 1.0, 2.6, 0.25), (2.55, 1.0, 2.6, 0.25)):
        f.append(arch_face(w, h, DARK, horseshoe=hs).T(x, 0.25, 0.08))
        f.append(arch_ring(w, h, jit(BRICK_L, 1.05), 0.1, horseshoe=hs).T(x, 0.25, 0.06))
        f.append(B(x - w / 2 - 0.35, x + w / 2 + 0.35, h + 0.3, h + 0.42, 0, 0.1, jit(BRICK_L)))   # alfiz
        for s in (-1, 1):
            f.append(B(x + s * (w / 2 + 0.35) - 0.06, x + s * (w / 2 + 0.35) + 0.06, h - 0.9, h + 0.42, 0, 0.1, jit(BRICK_L)))
    xs = np.linspace(-3.4, 3.4, 6)
    for x in xs:                                                            # the interlaced blind arches
        f += [m.T(x, 3.1, 0.0) for m in blind_arch(1.05, 1.6, depth=0.02, fill=MASON, horseshoe=0.15)]
    for x in np.linspace(-3.4 + 0.68, 3.4 - 0.68, 5):
        f.append(arch_ring(1.35, 1.75, jit(BRICK, 1.08), 0.07).T(x, 3.0, 0.1))
    f.append(B(-3.9, 3.9, 4.75, 4.9, 0, 0.12, jit(BRICK_L)))
    f.append(quad(0, 5.5, 0, 7.2, 0.62, "+z", DARK, eps=0.02))              # the sebka lattice
    for k in range(25):
        x = -3.6 + 7.2 * k / 24
        for s in (-1, 1):
            a = (max(-3.6, x - 0.31), 5.19) if s > 0 else (min(3.6, x + 0.31), 5.19)
            f.append(tube([(a[0], a[1], 0.06), (x, 5.5, 0.06)], 0.035, jit(BRICK_L, 1.08), 4))
            f.append(tube([(x, 5.5, 0.06), (x + 0.31 * s, 5.81, 0.06)], 0.035, jit(BRICK_L, 1.08), 4))
    f.append(B(-3.9, 3.9, 5.12, 5.2, 0, 0.1, jit(BRICK_L)))
    f.append(B(-3.9, 3.9, 5.8, 5.9, 0, 0.1, jit(BRICK_L)))
    f.append(B(-4.0, 4.0, 6.15, 6.5, 0, 0.08, jit(BRICK_L, 1.06)))           # the inscription in brick
    for k in range(18):
        x = -3.8 + 7.6 * k / 17
        f.append(B(x - 0.08, x + 0.08, 6.22, 6.43, 0.08, 0.12, jit(BRICK, 0.85)))
    parts += [m.T(0, 0, ZF) for m in f]
    # the iron railing on a stone kerb in front of the façade (as on the photos)
    zr = ZF + 1.0
    parts.append(B(-XW - 0.2, XW + 0.2, FOUND, 0.35, ZF, zr + 0.2, jit(MASON, 1.1)))
    for x in np.linspace(-XW, XW, 30):
        parts.append(B(x - 0.02, x + 0.02, 0.35, 2.0, zr - 0.02, zr + 0.02, IRON))
    for y in (0.5, 1.9):
        parts.append(B(-XW, XW, y, y + 0.04, zr - 0.03, zr + 0.03, IRON))

    # North-west façade (−x), on the patio: three horseshoe arches inside round ones, polylobed arches above
    n = []
    for z in (ZM + 1.47, (ZM + ZF) / 2, ZF - 1.47):
        n.append(arch_face(1.5, 3.1, DARK, horseshoe=0.2).T(0, PATIO + 0.1, 0.08).R(0, -math.pi / 2, 0).T(-XW, 0, z))
        n.append(arch_ring(1.5, 3.1, jit(BRICK_L), 0.11, horseshoe=0.2).T(0, PATIO + 0.1, 0.06).R(0, -math.pi / 2, 0).T(-XW, 0, z))
        n.append(arch_ring(2.2, 3.7, jit(BRICK, 1.08), 0.12).T(0, PATIO + 0.1, 0.12).R(0, -math.pi / 2, 0).T(-XW, 0, z))
        for dz in (-0.75, 0.0, 0.75):
            for k, col in enumerate((BRICK_L, (0.70, 0.38, 0.28))):
                n.append(arch_face(0.55, 1.25, jit(RECESS), horseshoe=0.2).T(0, 4.6, 0.04).R(0, -math.pi / 2, 0).T(-XW, 0, z + dz))
                n.append(arch_ring(0.55, 1.25, jit(col), 0.06 + 0.03 * k, horseshoe=0.2).T(0, 4.6, 0.06).R(0, -math.pi / 2, 0).T(-XW, 0, z + dz))
    parts += n
    # South-east façade (+x): a row of blind arches and two small windows
    for z in np.linspace(ZM + 1.0, ZF - 1.0, 5):
        parts += place(blind_arch(0.9, 1.8, horseshoe=0.15), XW, 3.4, z, math.pi / 2)
    for z in (1.5, 5.0):
        parts += place([arch_face(0.45, 1.0, DARK)], XW + 0.06, 5.4, z, math.pi / 2)
    return parts


# ── The Mudéjar apse (about 1187) ────────────────────────────────────────────
def apse():
    ring = ccw(apse_ring())
    parts = [extrude(ring, FOUND, EAVE - 0.2, jit(BRICK, 0.98)),
             extrude(offset(ring, 0.05), FOUND, 1.3, jit(MASON))]
    for y in np.arange(1.6, EAVE - 0.4, 0.5):
        parts.append(extrude(offset(ring, 0.02), y, y + 0.05, jit(BRICK_L, 0.97), top=False))
    parts += cornice_band(ring, EAVE - 0.2)
    # roof: a gable over the straight bay and a half cone over the apse, to one ridge
    top = EAVE + 0.4
    ridge = top + 1.6
    o = 0.3
    cone = []
    for t0, t1 in zip(np.linspace(0, math.pi, 13)[:-1], np.linspace(0, math.pi, 13)[1:]):
        a = ((XW + o) * math.cos(t0), top, ZA - (XW + o) * math.sin(t0))
        b = ((XW + o) * math.cos(t1), top, ZA - (XW + o) * math.sin(t1))
        cone.append(Mesh([a, (0, ridge, ZA), b], [(0, 2, 1)], jit(ROOF)))
    parts += cone
    for s in (-1, 1):
        V = [(s * (XW + o), top, ZA), (s * (XW + o), top, ZM + 0.2), (0, ridge, ZM + 0.2), (0, ridge, ZA)]
        F = [(0, 1, 2), (0, 2, 3)] if s < 0 else [(0, 2, 1), (0, 3, 2)]
        parts.append(Mesh(V, F, jit(ROOF)))
    parts.append(Mesh([(-XW - o, top, ZM + 0.2), (XW + o, top, ZM + 0.2), (0, ridge, ZM + 0.2)], [(0, 1, 2)], jit(BRICK)))
    # two tiers of blind arcades round the apse and the bay
    def around(fn, y, n, w, windows=()):
        out = []
        for k in range(n):
            t = (k + 0.5) / n * math.pi
            x, z = XW * math.cos(t), ZA - XW * math.sin(t)
            ang = math.atan2(math.cos(t), -math.sin(t))          # Ry so that +z points outwards
            ms = fn(w) if k not in windows else [arch_face(w * 0.45, 1.2, DARK).T(0, 0.3, 0.1)] + fn(w)[1:]
            out += place(ms, x, y, z, ang)
        for z in (ZM - 1.0, ZM - 2.6):
            for s in (-1, 1):
                out += place(fn(w), s * XW, y, z, s * math.pi / 2)
        return out
    parts += around(lambda w: blind_arch(w, 2.0), 1.5, 9, 1.1)
    parts += around(lambda w: blind_arch(w, 1.7, horseshoe=0.2, ring=BRICK_L), 3.9, 11, 0.85, windows=(3, 7))
    return parts


# ── The patio and the garden on the slope ───────────────────────────────────
def clip_x(poly, xlo, xhi):
    def clip(P, keep, inter):
        out = []
        for i in range(len(P)):
            a, b = P[i - 1], P[i]
            if keep(b):
                if not keep(a):
                    out.append(inter(a, b))
                out.append(b)
            elif keep(a):
                out.append(inter(a, b))
        return out

    def at_x(xv):
        return lambda a, b: (xv, a[1] + (b[1] - a[1]) * (xv - a[0]) / (b[0] - a[0]))
    P = clip(poly, lambda p: p[0] >= xlo, at_x(xlo))
    return clip(P, lambda p: p[0] <= xhi, at_x(xhi)) if P else []


def blob(cx, cy, cz, r, flat, col, seed, seg=9, rings=6):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    m = lathe(prof, col, seg).S(1, flat, 1)
    crumple(m, r * 0.28, 1.3 / r, seed)
    m.T(cx, cy, cz)
    c = np.asarray(col, float)
    lo = cy - r * flat

    def tint(v):
        t = np.clip((v[:, 1] - lo) / (2 * r * flat + 1e-6), 0, 1)
        return np.clip(np.outer(0.68 + 0.42 * t + 0.14 * (noise3(v, 0.9, seed) - 0.5), c), 0, 1)
    return m.paint(tint)


def tree(x, z, y, h, crown, seed):
    rr = np.random.default_rng(seed)
    out = [cyl(0.3, 0.2, h * 0.5, (0.38, 0.30, 0.24), 7).T(x, y, z)]
    for k in range(8):
        a = rr.uniform(0, 2 * math.pi); d = crown * 0.28 * rr.random()
        out.append(blob(x + d * math.cos(a), y + h * (0.58 + 0.2 * rr.random()), z + d * math.sin(a),
                        crown * (0.28 + 0.1 * rr.random()), 0.8, (0.22, 0.34, 0.16), seed + k))
    return out


def cypress(x, z, y, h, seed):
    return [cyl(0.2, 0.15, 1.0, (0.35, 0.26, 0.2), 7).T(x, y, z),
            blob(x, y + 0.8 + (h - 0.8) / 2, z, 0.9, (h - 0.8) / 1.8, (0.14, 0.24, 0.13), seed, seg=10, rings=8)]


def palm(x, z, y, h, seed):
    """A Canary palm: a thick straight trunk and a round crown of arching fronds."""
    rr = np.random.default_rng(seed)
    out = [cyl(0.45, 0.38, h, (0.45, 0.36, 0.27), 9).T(x, y, z)]
    top = np.array([x, y + h, z])
    for k in range(16):
        a = k * 2 * math.pi / 16 + rr.normal(0, 0.1)
        d = np.array([math.cos(a), 0, math.sin(a)])
        pts = [top + d * 3.2 * t + np.array([0, 1.4 * math.sin(math.pi * t * 0.8) - 0.6 * t, 0]) for t in np.linspace(0, 1, 5)]
        out.append(tube(pts, 0.22, (0.28, 0.42, 0.18), 4))
    out.append(blob(*top, 0.7, 0.8, (0.30, 0.42, 0.2), seed))
    return out


def railing(pts, y, h=1.1):
    out = []
    for (xa, za), (xb, zb) in zip(pts, pts[1:]):
        L = math.hypot(xb - xa, zb - za)
        n = max(1, int(L / 1.6))
        for k in range(n + 1):
            x, z = xa + (xb - xa) * k / n, za + (zb - za) * k / n
            out.append(B(x - 0.03, x + 0.03, y, y + h, z - 0.03, z + 0.03, IRON))
        for yy in (y + 0.1, y + h):
            out.append(tube([(xa, yy, za), (xb, yy, zb)], 0.025, IRON, 4))
    return out


def garden():
    parts = []
    # the patio along the north-west façade, with the well; a retaining wall down to the garden
    patio = [(-15.43, 3.0), (-12.15, -12.0), (-1.0, -12.0), (-1.0, 3.0)]
    parts.append(extrude(patio, FOUND - 4.0, PATIO, jit(MASON, 0.95), top_c=SAND))
    parts.append(B(-8.4, -4.33, PATIO, PATIO + 0.05, ZM - 0.2, 2.9, PAVE))
    parts.append(lathe([(0.0, 0), (0.75, 0), (0.75, 0.75), (0.55, 0.75), (0.55, 0.1), (0.0, 0.1)], jit(MASON, 1.15), 18).T(-9.0, PATIO, -5.0))
    parts.append(B(-9.04, -8.96, PATIO + 0.75, PATIO + 2.0, -5.7, -4.3, IRON))
    parts.append(B(-9.04, -8.96, PATIO + 1.95, PATIO + 2.02, -5.7, -4.3, IRON))
    parts += railing([(-15.2, 2.0), (-12.15, -12.0)], PATIO)
    parts += palm(-6.5, -9.5, PATIO, 6.0, 51)
    # steps down from the patio to the garden
    for k in range(9):
        y = PATIO - (k + 1) * (PATIO - TERR[0][1]) / 9
        parts.append(B(-12.7 - 0.35 * (k + 1), -12.7 - 0.35 * k, FOUND - 6, y + 0.2, -10.5, -8.5, jit(MASON, 1.1)))
    # three terraces with their retaining walls
    hi = -8.0
    for xlo, lv in TERR:
        piece = clip_x(GARDEN_RING, xlo, hi)
        if len(piece) >= 3:
            parts.append(extrude(piece, FOUND - 8.0, lv, jit(MASON, 0.93), top_c=SAND))
            parts.append(extrude(offset(piece, -0.6), lv - 0.1, lv + 0.05, GRASS))
        hi = xlo
    # the chahar bagh in the upper terrace: crossing paths and a small pool in the middle
    cx, cz, y = -13.2, -16.0, TERR[0][1]
    parts.append(B(cx - 0.8, cx + 0.8, y, y + 0.1, cz - 8.0, cz + 8.0, PAVE))
    parts.append(B(cx - 3.2, cx + 1.9, y, y + 0.1, cz - 0.8, cz + 0.8, PAVE))
    parts.append(B(cx - 1.3, cx + 1.3, y, y + 0.35, cz - 1.3, cz + 1.3, jit(MASON, 1.15)))
    parts.append(B(cx - 1.05, cx + 1.05, y + 0.1, y + 0.3, cz - 1.05, cz + 1.05, WATER))
    # trees from the orthophoto (big crowns in the lower garden) and cypresses
    for i, (x, z, kind, h, c) in enumerate([(-21.0, -12.0, "t", 11, 8), (-22.0, -24.0, "t", 10, 7), (-29.0, -16.0, "t", 9, 7),
                                            (-14.5, -26.0, "t", 8, 6), (-11.5, -6.5, "c", 9, 0), (-16.5, -20.5, "c", 10, 0)]):
        lv = TERR[0][1] if x > -18 else (TERR[1][1] if x > -25 else TERR[2][1])
        if i == 4:
            lv = PATIO                                   # the cypress on the patio
        parts += tree(x, z, lv, h, c, 300 + i) if kind == "t" else cypress(x, z, lv, h, 300 + i)
    return parts


def main():
    out = ROOT / "models" / "Q1324080"
    out.mkdir(parents=True, exist_ok=True)
    m = merge(mosque() + apse() + garden())
    tris, size = write_glb(out / "model.glb", [("cristo_de_la_luz", m)], roughness=0.9)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
