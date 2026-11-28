"""
Santa María la Blanca, the old Main Synagogue of Toledo (Q1568115) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: synagogue of about 1180 (date on a beam; rebuilt in the 13th century), Mudéjar work of Muslim
builders for the Jewish community; church from 1411, chapels at the head by Alonso de Covarrubias (died 1570) in
the 16th century; public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 150415814: a slightly skewed block 21.3 m wide and 25–28.5 m long, long axis at 73.8°. The five
  naves run along it, from the foot (west, on the garden) to the head with the chapels (east). The garden courtyard
  in front of the foot and along the north side is the OSM area way 150415800 (tourism=attraction), entered from
  the street through a gate up a few steps.
- Main façade: the foot of the naves on the garden, facing 253.8° (west-south-west).
- Wikipedia: five naves on 32 octagonal brick pillars with horseshoe arches, white plain walls, a sober exterior.
  The naves step up to the central one, as on the photos of the foot and on the PNOA orthophoto (the parallel
  ridges of the roofs): outer naves 4.1 m wide with walls 6 m high and lean-to roofs, the inner ones 4.0 m, raised
  to 8.2–9.6 m, the central one 5.0 m wide, eaves 10.6 m, ridge 12.2 m (estimated on the photos, ±10 %).
  A brick plinth below, whitewashed plaster above, big modern glazed windows in the foot, Arab tiles.

Axes (docs/MODEL_SPEC.md): +Y up, the foot of the naves faces +Z, +X to the south-south-east (heading − 90°),
origin at the centroid of the OSM outline; y = 0 is the ground at the door in the foot (ground_at). The ground
falls about 3 m from the head to the street in front of the garden ("ground": "keep").

    python model_sources/Q1568115_santa_maria_la_blanca.py   -> models/Q1568115/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, quad, extrude, offset, crumple, noise3, write_glb  # noqa: E402

rng = np.random.default_rng(1180)

PLASTER = np.array((0.90, 0.88, 0.82))
BRICK = np.array((0.76, 0.54, 0.40))
ROOF = np.array((0.62, 0.46, 0.37))
GLASS = (0.16, 0.18, 0.20)
FRAME = (0.25, 0.20, 0.16)
WOOD = (0.45, 0.28, 0.16)
SAND = (0.78, 0.70, 0.58)
PAVE = (0.72, 0.66, 0.58)
STONE = (0.64, 0.58, 0.50)
FOUND = -4.0
GARDEN = -0.4                 # the garden floor, a step below the door

# OSM way 150415814 in model metres (osm.py --heading 253.8): the foot (+z) and the head (−z) edges
FRONT_EDGE = [(-10.5, 14.2), (-2.3, 13.8), (9.6, 13.1)]
BACK_EDGE = [(-10.5, -11.5), (10.8, -15.3)]
X0, X1 = -10.5, 10.8
# the garden courtyard (OSM way 150415800, the area round the building)
GARDEN_RING = [(-27.8, 35.9), (-17.7, 35.7), (-5.9, 35.6), (-5.9, 30.4), (-2.1, 30.6), (17.8, 31.1), (13.2, 12.9),
               (9.6, 13.1), (-2.3, 13.8), (-10.5, 14.2), (-10.5, -11.5), (-22.4, -9.4), (-20.8, -1.4), (-20.3, 6.2),
               (-19.3, 15.3), (-17.7, 17.5), (-17.5, 23.3), (-17.4, 25.0), (-22.9, 25.0), (-23.0, 29.3), (-28.5, 28.9)]
# naves across x: (x0, x1, height at x0, height at x1, ridge or None)
NAVES = [
    (-10.5, -6.4, 6.0, 7.4, None),
    (-6.4, -2.4, 8.2, 9.6, None),
    (-2.4, 2.6, 10.6, 10.6, 12.2),
    (2.6, 6.6, 9.6, 8.2, None),
    (6.6, 10.8, 7.4, 6.0, None),
]


def jit(c, k=1.0, s=0.025):
    return tuple(np.clip(np.asarray(c) * k * (1 + rng.normal(0, s)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def zfront(x):
    return float(np.interp(x, [p[0] for p in FRONT_EDGE], [p[1] for p in FRONT_EDGE]))


def zback(x):
    return float(np.interp(x, [p[0] for p in BACK_EDGE], [p[1] for p in BACK_EDGE]))


def profile_prism(prof, wall_c, roof_c, zf=zfront, zb=zback, over=0.0):
    """A solid whose cross-section in (x, y) is the convex polygon prof (counter-clockwise seen from +z), running
    from the head edge zb(x) to the foot edge zf(x). Vertical sides take wall_c, sloping tops roof_c."""
    V, F, C = [], [], []
    n = len(prof)
    for i in range(n):
        (xa, ya), (xb, yb) = prof[i], prof[(i + 1) % n]
        o = len(V)
        V += [(xa, ya, zb(xa) - over), (xb, yb, zb(xb) - over), (xb, yb, zf(xb) + over), (xa, ya, zf(xa) + over)]
        F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        c = wall_c if abs(xb - xa) < 1e-6 else (roof_c if yb > FOUND + 0.01 or ya > FOUND + 0.01 else wall_c)
        C += [c] * 4
    for front in (True, False):
        o = len(V)
        for x, y in prof:
            V.append((x, y, zf(x) if front else zb(x)))
            C.append(wall_c)
        for k in range(1, n - 1):
            F.append((o, o + k, o + k + 1) if front else (o, o + k + 1, o + k))
    m = Mesh(V, F)
    m.c = np.array(C, float)
    return m


def on_front(ms, x, eps=0.0):
    """Place features built facing +z at x = 0, z = 0 on the (slightly skewed) foot façade."""
    a = math.atan2(FRONT_EDGE[-1][1] - FRONT_EDGE[0][1], FRONT_EDGE[-1][0] - FRONT_EDGE[0][0])
    return [m.copy().R(0, -a, 0).T(x, 0, zfront(x) + eps) for m in ms]


def grid_window(w, h):
    """A tall modern window of square panes in a dark frame, facing +z, sill at y = 0."""
    out = [quad(0, h / 2, 0, w + 0.3, h + 0.3, "+z", jit(PLASTER, 0.93), eps=0.02),
           quad(0, h / 2, 0, w, h, "+z", GLASS, eps=0.04)]
    nx, ny = max(2, int(round(w / 0.55))), max(2, int(round(h / 0.55)))
    for k in range(1, nx):
        x = -w / 2 + w * k / nx
        out.append(B(x - 0.03, x + 0.03, 0, h, 0.03, 0.08, FRAME))
    for k in range(1, ny):
        y = h * k / ny
        out.append(B(-w / 2, w / 2, y - 0.03, y + 0.03, 0.03, 0.08, FRAME))
    out.append(B(-w / 2 - 0.1, w / 2 + 0.1, -0.12, 0.0, 0.0, 0.2, jit(BRICK, 1.05)))
    return out


def door(w=2.4, h=4.2):
    """The door of the foot: two carved wooden leaves under a glazed transom, a brick frame, a stone step."""
    out = [quad(0, (h + 1.2) / 2, 0, w + 0.7, h + 1.5, "+z", jit(BRICK, 1.0), eps=0.03),
           quad(0, h / 2, 0, w, h, "+z", WOOD, eps=0.06),
           quad(0, h / 2, 0, 0.06, h, "+z", (0.3, 0.18, 0.1), eps=0.07),
           quad(0, h + 0.6, 0, w, 0.9, "+z", GLASS, eps=0.06)]
    for k in range(1, 4):
        out.append(B(-w / 2 + w * k / 4 - 0.03, -w / 2 + w * k / 4 + 0.03, h + 0.15, h + 1.05, 0.06, 0.1, FRAME))
    for y in (h * 0.3, h * 0.7):
        out.append(B(-w / 2 + 0.1, w / 2 - 0.1, y - 0.04, y + 0.04, 0.06, 0.1, (0.36, 0.22, 0.12)))
    out.append(B(-w / 2 - 0.6, w / 2 + 0.6, FOUND, 0.0, 0.0, 0.9, STONE))
    out.append(B(-w / 2 - 0.9, w / 2 + 0.9, FOUND, GARDEN + 0.2, 0.9, 1.5, STONE))
    return out


def building():
    parts = []
    for x0, x1, h0, h1, ridge in NAVES:
        if ridge is None:
            prof = [(x0, FOUND), (x1, FOUND), (x1, h1), (x0, h0)]
        else:
            xm = (x0 + x1) / 2
            prof = [(x0, FOUND), (x1, FOUND), (x1, h1), (xm, ridge), (x0, h0)]
        parts.append(profile_prism(prof, jit(PLASTER), jit(ROOF)))
        # the tiled roof itself: a thin slab over the slopes with a small overhang at the foot and the head
        top = prof[2:]
        for (xa, ya), (xb, yb) in zip(top, top[1:]):
            if xa > xb:
                (xa, ya), (xb, yb) = (xb, yb), (xa, ya)
            slab = [(xa, ya - 0.12), (xb, yb - 0.12), (xb, yb + 0.06), (xa, ya + 0.06)]
            parts.append(profile_prism(slab, jit(ROOF, 0.9), jit(ROOF), over=0.35))
        # ridge or top line of the tiles
        hi = max(prof[2:], key=lambda p: p[1])
        parts.append(tube([(hi[0], hi[1] + 0.08, zback(hi[0]) - 0.3), (hi[0], hi[1] + 0.08, zfront(hi[0]) + 0.3)], 0.12,
                          jit(ROOF, 1.08), 6))
    # eaves over the outer walls
    for x, h in ((X0, 6.0), (X1, 6.0)):
        s = -1 if x == X0 else 1
        parts.append(B(*sorted((x, x + s * 0.45)), h - 0.15, h + 0.02, zback(x) - 0.3, zfront(x) + 0.3, jit(ROOF, 0.92)))
    # the brick plinth all round (below the whitewash) and a brick band at the foot
    outline = [(X0, zback(X0)), (X1, zback(X1)), (X1, zfront(X1)), (-2.3, 13.8), (X0, zfront(X0))]
    parts.append(extrude(offset(outline, 0.06), FOUND, 3.2, jit(BRICK)))
    parts.append(extrude(offset(outline, 0.1), 3.2, 3.35, jit(BRICK, 1.08), top=False))
    for y in np.arange(0.4, 3.2, 0.5):
        parts.append(extrude(offset(outline, 0.07), y, y + 0.04, jit(BRICK, 0.85), top=False))
    # the foot: the low brick porch-wall on the north part, the door, the big grid windows
    parts += on_front([B(-3.0, 3.0, FOUND, 4.0, 0.0, 0.7, jit(BRICK))], -7.7)
    parts += on_front([quad(0, 2.4, 0.7, 1.6, 1.8, "+z", jit(PLASTER, 0.8), eps=0.02)], -7.7)
    parts += on_front(door(), 0.1)
    for x, y0, w, h in ((0.1, 6.0, 2.6, 3.6), (-4.4, 4.2, 2.4, 3.4), (4.6, 4.2, 2.4, 3.4), (-8.4, 4.4, 1.0, 1.2),
                        (8.7, 4.4, 1.0, 1.2)):
        parts += on_front([m.T(0, y0, 0) for m in grid_window(w, h)], x)
    # the clerestory of the raised naves: small windows on the walls standing above the lower roofs
    for z in np.arange(-9.0, 12.0, 4.0):
        for x, y, side in ((-6.4, 7.55, -1), (6.6, 7.55, 1), (-2.4, 9.85, -1), (2.6, 9.85, 1)):
            q = quad(x, y, z, 0.7, 0.5, "+x" if side > 0 else "-x", GLASS, eps=0.03)
            parts.append(q)
    # small windows and a side door on the long walls
    for z in (-6.0, 0.0, 6.0):
        for x, nrm in ((X0, "-x"), (X1, "+x")):
            parts.append(quad(x, 4.6, z, 0.9, 1.1, nrm, GLASS, eps=0.1))
    parts.append(quad(X0, 1.6, 3.0, 1.6, 3.0, "-x", WOOD, eps=0.1))
    # the head: three small windows of the chapels of Covarrubias in the east wall
    for x in (-4.4, 0.1, 4.6):
        parts.append(quad(x, 6.5 if abs(x) < 1 else 5.2, zback(x), 0.8, 1.2, "-z", GLASS, eps=0.1))
    return parts


# ── The garden courtyard ─────────────────────────────────────────────────────
def blob(cx, cy, cz, r, flat, col, seed, seg=9, rings=6):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    m = lathe(prof, col, seg).S(1, flat, 1)
    crumple(m, r * 0.25, 1.3 / r, seed)
    m.T(cx, cy, cz)
    c = np.asarray(col, float)
    lo = cy - r * flat

    def tint(v):
        t = np.clip((v[:, 1] - lo) / (2 * r * flat + 1e-6), 0, 1)
        k = 0.68 + 0.42 * t + 0.14 * (noise3(v, 0.9, seed) - 0.5)
        return np.clip(np.outer(k, c), 0, 1)
    return m.paint(tint)


def cypress(x, z, y, h, seed):
    """An Italian cypress: a short trunk and a tall dense spindle of dark foliage."""
    r = 0.9 + 0.25 * np.random.default_rng(seed).random()
    out = [cyl(0.22, 0.16, 1.2, (0.35, 0.26, 0.2), 7).T(x, y, z)]
    out.append(blob(x, y + 0.9 + (h - 0.9) / 2, z, r, (h - 0.9) / (2 * r), (0.14, 0.24, 0.13), seed, seg=10, rings=8))
    return out


def broadleaf(x, z, y, h, crown, seed):
    rr = np.random.default_rng(seed)
    out = [cyl(0.3, 0.2, h * 0.45, (0.38, 0.3, 0.24), 7).T(x, y, z)]
    for k in range(7):
        a = rr.uniform(0, 2 * math.pi); d = crown * 0.25 * rr.random()
        out.append(blob(x + d * math.cos(a), y + h * (0.6 + 0.15 * rr.random()), z + d * math.sin(a),
                        crown * (0.3 + 0.1 * rr.random()), 0.8, (0.24, 0.36, 0.17), seed + k))
    return out


TREES = [  # cypresses and a few broadleaf trees, from the PNOA orthophoto and the photos (positions ±2 m)
    (cypress, -14.0, 18.0, 14), (cypress, -14.0, 24.0, 13), (cypress, -11.0, 27.0, 15), (cypress, -4.0, 22.0, 13),
    (cypress, -16.0, 8.0, 12), (cypress, -16.5, 0.0, 13), (cypress, -15.5, -6.0, 11), (cypress, 6.0, 20.0, 12),
    (cypress, 12.0, 24.0, 12), (cypress, -21.0, 31.0, 12),
]


def garden():
    parts = [extrude(GARDEN_RING, FOUND, GARDEN, jit(STONE, 0.95), top_c=SAND)]
    # a paved path from the gate to the door and a strip along the foot
    parts.append(B(-1.3, 1.5, GARDEN, GARDEN + 0.05, 14.5, 25.5, PAVE))
    parts.append(B(-8.4, 1.5, GARDEN, GARDEN + 0.05, 23.0, 25.5, PAVE))
    parts.append(B(-8.4, -5.8, GARDEN, GARDEN + 0.05, 23.0, 35.5, PAVE))
    parts.append(B(-10.5, 13.0, GARDEN, GARDEN + 0.05, 13.0, 17.5, PAVE))
    # low brick borders of the beds
    for x0, x1, z0, z1 in ((-17.0, -9.5, 19.0, 33.0), (3.5, 15.0, 19.0, 29.0), (-18.5, -12.0, -8.0, 14.0)):
        ring = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
        parts.append(extrude(ring, GARDEN - 0.05, GARDEN + 0.25, jit(BRICK), holes=[offset(ring, -0.25)]))
        parts.append(extrude(offset(ring, -0.25), GARDEN - 0.05, GARDEN + 0.1, (0.42, 0.40, 0.26)))
    # the wall round the garden (brick and masonry, 3 m), with the gate on the street and its steps
    ring = GARDEN_RING
    skip = {(9.6, 13.1), (-2.3, 13.8), (-10.5, 14.2), (-10.5, -11.5)}
    for a, b in zip(ring, ring[1:] + ring[:1]):
        if a in skip and b in skip:
            continue
        if a == (13.2, 12.9) and b == (9.6, 13.1):
            continue
        if a == (-10.5, -11.5):
            continue
        (xa, za), (xb, zb) = a, b
        L = math.hypot(xb - xa, zb - za)
        ang = math.atan2(zb - za, xb - xa)
        segs = [(0, L)]
        if a == (-17.7, 35.7) and b == (-5.9, 35.6):
            segs = [(0, 9.6), (11.6, L)]
        for s0, s1 in segs:
            w = B(s0, s1, FOUND, GARDEN + 3.0, -0.25, 0.25, jit(BRICK, 0.97))
            parts.append(w.R(0, -ang, 0).T(xa, 0, za))
            cap = B(s0, s1, GARDEN + 3.0, GARDEN + 3.15, -0.3, 0.3, jit(ROOF, 1.0))
            parts.append(cap.R(0, -ang, 0).T(xa, 0, za))
    # the gate: a brick frame with a lintel, and steps down to the street (about 0.9 m)
    gx = -17.7 + 10.6
    parts.append(B(gx - 1.4, gx + 1.4, GARDEN + 3.0, GARDEN + 3.6, 35.25, 35.95, jit(BRICK, 1.05)))
    parts.append(B(gx - 1.6, gx + 1.6, GARDEN + 3.6, GARDEN + 3.75, 35.2, 36.0, jit(ROOF)))
    for k in range(5):
        parts.append(B(gx - 1.5, gx + 1.5, FOUND, GARDEN - 0.18 * k, 35.4 + 0.32 * k, 35.75 + 0.32 * k, jit(STONE, 1.05)))
    for i, (kind, x, z, h) in enumerate(TREES):
        parts += kind(x, z, GARDEN, h, 700 + 13 * i)
    parts += broadleaf(9.0, 27.0, GARDEN, 9, 7, 901)
    parts += broadleaf(-20.0, 20.0, GARDEN, 8, 6, 902)
    return parts


def main():
    out = ROOT / "models" / "Q1568115"
    out.mkdir(parents=True, exist_ok=True)
    m = merge(building() + garden())
    tris, size = write_glb(out / "model.glb", [("santa_maria_la_blanca", m)], roughness=0.9)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
