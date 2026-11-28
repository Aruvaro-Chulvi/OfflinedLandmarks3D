"""
Anfiteatro de Mérida — Amphitheatre of Augusta Emerita, Mérida (Q2360379) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: planned together with the theatre beside it and inaugurated in 8 BC (inscriptions of the tribunes);
abandoned in the 4th century, half buried until the excavations from 1919; the lower seats were partly rebuilt
by José Menéndez-Pidal in the mid 20th century. Architects unknown. The model shows the ruin as it stands today.
Public domain.

Measurements (see meta.json for the sources):
- Plan from OpenStreetMap relation 1316311: an ellipse fitted to the outer way 89067879 gives 124 × 105 m, one
  fitted to the arena walls (ways 89067885, 1206491700, 1206491716) 68 × 40 m (with the stubs of the gates), both
  with the long axis at 134–137° / 314–317° and the same centre (38.916176, −6.337929, the origin). Published
  (Wikipedia es/en): 126 × 102 m, arena 64 × 41 m. The model uses 126 × 102 m and draws every ring as an offset
  curve of the outer ellipse (the arena wall at 30.75 m from the outer face: 64.5 × 40.5 m).
- The main gate is at the north-west end of the long axis («the western end»), so that end is +Z (heading 316°).
  Two long galleries on the long axis lead from the outside to the arena, open to the sky with brick transverse
  arches; fourteen more doors in the façade (sixteen in all) open into radial passages between pairs of walls
  (OSM ways 1206491701–1206491714 show some of them).
- The arena with its podium wall (2.6 m), the cross-shaped pit (fossa bestiaria) in the middle: the central
  basin and the short arm towards the south-west open (2.8 m deep), the long arms covered by wooden decks as
  today (outline from OSM ways 261488526, 260417196–260417199 and the PNOA orthophoto).
- The ima cavea: nine rows (0.8 m deep, 0.42 m high), rebuilt on the parts near the gates and the tribunes,
  rough on the rest; the praecinctio and its balteus at about 21 m from the outer face (OSM way 261488524).
  The two tribunes on the minor axis (the east one, of the editor, with two stairs to the arena; the west one over
  the passage to the outside, OSM way 1206491715).
- Media and summa cavea: lost; a grassy slope of rubble with the stubs of the radial walls, higher on the east
  (built against the hill of San Albín) than on the west. The outer wall stands only in parts. Heights checked
  against the IGN MDT05: the arena is the lowest point; the ground round the outer wall is 7–9 m above it on the
  north-east and the ends, 3–4 m on the south-west.
Colours: sand of the arena, grey-brown diorite and granite masonry, light granite of the rebuilt seats, red brick
of the arches, grass on the ruined cavea.

Axes (docs/MODEL_SPEC.md): +Y up, the main gate (north-west end of the long axis) faces +Z = heading 316°, +X to
the south-west (226°, towards the theatre). Origin at the centre of the arena, y = 0 on the arena. Walls go down
to −3 m (the pit to −2.8 m).

    python model_sources/Q2360379_anfiteatro_de_merida.py   -> models/Q2360379/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, cap, walls, extrude, triangulate, fbm, noise3, write_glb  # noqa: E402
from meshkit import _orient  # noqa: E402

rng = np.random.default_rng(8)

# ── Colours ──────────────────────────────────────────────────────────────────
SAND = (0.80, 0.70, 0.56)
STONE = (0.60, 0.53, 0.43)          # diorite / granite masonry
STONE_D = (0.50, 0.44, 0.36)
SEAT = (0.77, 0.73, 0.65)           # rebuilt granite seats
RUBBLE = (0.58, 0.52, 0.43)
GRASS = (0.50, 0.52, 0.32)
EARTH = (0.55, 0.47, 0.36)
BRICK = (0.70, 0.43, 0.31)
WOOD = (0.56, 0.45, 0.33)
PIT = (0.38, 0.45, 0.26)            # grass at the bottom of the fossa
COPING = (0.80, 0.77, 0.70)
DARK = (0.16, 0.14, 0.12)

A, BX = 63.0, 51.0                  # semi-axes of the outer face: along z (long axis), along x
D_ARENA = 30.75                     # arena wall face, from the outer face
D_POD = 29.9
D_IMA0, ROW_D, ROW_Y, N_ROWS = 28.9, 0.8, 0.42, 9
Y_POD = 2.6
Y_IMA = Y_POD + N_ROWS * ROW_Y      # 6.38
D_PRAEC = D_IMA0 - N_ROWS * ROW_D   # 21.7
D_BALT, Y_BALT = 20.5, 7.6
D_WALL = 2.4                        # thickness of the outer wall
GAL = 2.5                           # half width of the long-axis galleries
FOUND = -3.0


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def mirror_x(m):
    n = m.copy(); n.v[:, 0] *= -1; n.f = n.f[:, ::-1]; return n


def mirror_z(m):
    n = m.copy(); n.v[:, 2] *= -1; n.f = n.f[:, ::-1]; return n


def tint(m, amp=0.08, freq=0.35, seed=0):
    k = 1 + amp * (2 * fbm(m.v, freq, seed, 2) - 1)
    m.c = np.clip(m.c * k[:, None], 0, 1)
    return m


def prism_xy(poly, z0, z1, c):
    """A polygon in the (x, y) plane extruded from z0 to z1."""
    pts, tris = triangulate(poly)
    n = len(pts)
    V = [(p[0], p[1], z1) for p in pts] + [(p[0], p[1], z0) for p in pts]
    F = [tuple(t) for t in tris] + [(n + t[0], n + t[2], n + t[1]) for t in tris]
    ms = [Mesh(V, F, c)]
    R = _orient(poly, True)
    sv, sf = [], []
    for i in range(len(R)):
        (u0, y0), (u1, y1) = R[i], R[(i + 1) % len(R)]
        o = len(sv)
        sv += [(u0, y0, z0), (u1, y1, z0), (u1, y1, z1), (u0, y0, z1)]
        sf += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    ms.append(Mesh(sv, sf, c))
    return merge(ms)


# ── The ellipse and its offset curves ───────────────────────────────────────
def curve(t, d):
    """Point at parameter t on the curve d metres inside the outer face (x = BX cos t, z = A sin t)."""
    nx, nz = A * math.cos(t), BX * math.sin(t)
    ln = math.hypot(nx, nz)
    return BX * math.cos(t) - d * nx / ln, A * math.sin(t) - d * nz / ln


def tangent(t):
    tx, tz = -BX * math.sin(t), A * math.cos(t)
    ln = math.hypot(tx, tz)
    return tx / ln, tz / ln


def t_edge(d, half=GAL):
    """Parameter where the curve d meets the side x = half of the long-axis gallery (0 < t < π/2)."""
    lo, hi = 0.0, math.pi / 2
    for _ in range(50):
        m = (lo + hi) / 2
        if curve(m, d)[0] > half:
            lo = m
        else:
            hi = m
    return lo


def tu(d, u):
    """u ∈ [0, 1] along the +x half of the cavea, from the south-east gallery (u = 0) to the main one (u = 1)."""
    te = t_edge(d)
    return -te + u * 2 * te


def pt(d, u, y):
    x, z = curve(tu(d, u), d)
    return (x, y, z)


def strip(d0, d1, y, c, n=64, u0=0.0, u1=1.0):
    us = np.linspace(u0, u1, n + 1)
    V = [pt(d0, u, y) for u in us] + [pt(d1, u, y) for u in us]
    F = []
    for i in range(n):
        F += [(i, n + 2 + i, i + 1), (i, n + 1 + i, n + 2 + i)]
    return Mesh(V, F, c)


def riser(d, y0, y1, c, n=64, u0=0.0, u1=1.0):
    us = np.linspace(u0, u1, n + 1)
    V = [pt(d, u, y0) for u in us] + [pt(d, u, y1) for u in us]
    F = []
    for i in range(n):
        F += [(i, i + 1, n + 2 + i), (i, n + 2 + i, n + 1 + i)]
    return Mesh(V, F, c)


def slope_h(t, d, side=-1):
    """Height of the ruined media and summa cavea: higher on the hill side (x < 0) than on the south-west."""
    k = max(0.0, math.cos(t)) ** 1.5 if side > 0 else 0.0     # 1 in the middle of the south-west side (x > 0)
    f = (D_BALT - d) / (D_BALT - D_WALL)
    return Y_BALT + f * (1.2 - 4.2 * k)


def both(ms):
    ms = [m for m in ms if m is not None]
    return ms + [mirror_x(m) for m in ms]


# ── Arena and fossa ──────────────────────────────────────────────────────────
FOSSA = [(-9.0, -13.0), (11.0, -13.0), (11.0, -2.5), (17.4, -2.5), (17.4, 0.7), (11.0, 0.7), (11.0, 13.0),
         (-9.0, 13.0)]


def arena():
    ms = []
    n = 96
    ring = [curve(2 * math.pi * k / n, D_ARENA - 0.4) for k in range(n)]
    ms.append(tint(cap(ring, 0.0, SAND, holes=[FOSSA]), 0.05, 0.15, 1))
    ms.append(walls(ring, FOUND, 0.0, SAND))
    ms.append(tint(walls(FOSSA, -2.8, 0.0, STONE, hole=True), 0.12, 0.5, 2))
    ms.append(tint(cap(FOSSA, -2.8, PIT), 0.15, 0.3, 3))
    # the long arms of the cross, covered by wooden decks
    for z0, z1 in ((13.0, 25.5), (-25.5, -13.0)):
        ms.append(B(-2.0, 5.0, 0.0, 0.14, z0, z1, WOOD))
        for k in range(int((z1 - z0) / 1.2)):
            z = z0 + 0.6 + 1.2 * k
            ms.append(B(-2.0, 5.0, 0.14, 0.16, z - 0.02, z + 0.02, (0.42, 0.34, 0.25)))
    # loose stones in the pit
    for k in range(6):
        x, z = rng.uniform(-7, 9), rng.uniform(-11, 11)
        ms.append(B(x - 0.5, x + 0.5, -2.8, -2.3, z - 0.4, z + 0.4, STONE_D))
    return merge(ms)


# ── Podium, ima cavea, praecinctio and balteus (one half; mirrored) ─────────
RESTORED = [(0.0, 0.22), (0.40, 0.60), (0.78, 1.0)]


def half_cavea(side):
    """The +x half; side = +1 for the south-west half, −1 for the north-east one (mirrored afterwards)."""
    seat, stone, light, dark = [], [], [], []
    stone.append(riser(D_ARENA, -0.3, Y_POD, STONE))
    light.append(strip(D_ARENA, D_POD, Y_POD, COPING))
    stone.append(strip(D_POD, D_IMA0, Y_POD, STONE_D))
    y = Y_POD
    for k in range(N_ROWS):
        d0 = D_IMA0 - k * ROW_D
        for u0, u1 in RESTORED + [(0.22, 0.40), (0.60, 0.78)]:
            restored = (u0, u1) in RESTORED
            n = max(4, int((u1 - u0) * 64))
            dy = 0.0 if restored else -0.12 * ((k * 7 + int(u0 * 10)) % 3)
            c = SEAT if restored else RUBBLE
            seat.append(riser(d0, y, y + ROW_Y + dy, c, n, u0, u1))
            seat.append(strip(d0, d0 - ROW_D, y + ROW_Y + dy, c, n, u0, u1))
        y += ROW_Y
    seat.append(strip(D_PRAEC, D_BALT, Y_IMA, SEAT))
    stone.append(riser(D_BALT, Y_IMA, Y_BALT, STONE))
    # the slope of the lost media and summa cavea
    n, nd = 64, 6
    us = np.linspace(0, 1, n + 1)
    ds = np.linspace(D_BALT, D_WALL, nd + 1)
    V, F, C = [], [], []
    for d in ds:
        for u in us:
            t = tu(d, u)
            x, z = curve(t, d)
            h = slope_h(t, d, side) + 0.35 * (noise3(np.array([[x, 0.0, z]]), 0.18, 5)[0] - 0.5) * (d < D_BALT - 0.1)
            V.append((x, h, z))
            g = noise3(np.array([[x, 1.0, z]]), 0.12, 9)[0]
            C.append(GRASS if g > 0.42 else (EARTH if g > 0.3 else RUBBLE))
    W = n + 1
    for j in range(nd):
        for i in range(n):
            a = j * W + i
            F += [(a, a + 1, a + W), (a + 1, a + W + 1, a + W)]
    stone.append(Mesh(V, F, np.array(C)))
    # radial walls of the passages from the seven doors of this half
    for k in range(1, 8):
        t = -math.pi / 2 + k * math.pi / 8
        tx, tz = tangent(t)
        for d0, d1 in ((D_WALL, 11.0), (11.0, D_BALT)):
            hh = slope_h(t, (d0 + d1) / 2, side) + rng.uniform(0.5, 1.6)
            for o0, o1 in ((1.5, 2.6), (-2.6, -1.5)):
                p0, p1 = curve(t, d0), curve(t, d1)
                poly = [(p0[0] + tx * o0, p0[1] + tz * o0), (p1[0] + tx * o0, p1[1] + tz * o0),
                        (p1[0] + tx * o1, p1[1] + tz * o1), (p0[0] + tx * o1, p0[1] + tz * o1)]
                stone.append(extrude(poly, FOUND, hh, STONE))
            # the floor of the passage, earth between the walls
            p0, p1 = curve(t, d0), curve(t, d1)
            hm = min(slope_h(t, d0, side), slope_h(t, d1, side)) + 0.05
            poly = [(p0[0] + tx * 1.5, p0[1] + tz * 1.5), (p1[0] + tx * 1.5, p1[1] + tz * 1.5),
                    (p1[0] - tx * 1.5, p1[1] - tz * 1.5), (p0[0] - tx * 1.5, p0[1] - tz * 1.5)]
            stone.append(extrude(poly, hm - 0.3, hm, EARTH))
        # its mouth in the balteus
        p = curve(t, D_BALT)
        dark.append(B(p[0] - 0.05, p[0] + 0.05, Y_IMA, Y_BALT - 0.1, p[1] - 1.4, p[1] + 1.4, DARK)
                    if abs(math.cos(t)) > 0.7 else
                    B(p[0] - 1.4, p[0] + 1.4, Y_IMA, Y_BALT - 0.1, p[1] - 0.05, p[1] + 0.05, DARK))
    # the outer wall, standing only in parts (per stretch between doors: kept height above the slope)
    keep = [0.4, 2.6, 0.6, 3.4, 1.2, 0.5, 2.2, 4.0] if side > 0 else [3.0, 0.8, 1.6, 0.4, 2.4, 0.6, 1.8, 3.6]
    for k in range(8):
        ta = -math.pi / 2 + k * math.pi / 8
        tb = ta + math.pi / 8
        if k == 0:
            ta = -t_edge(0.0, GAL + 1.2)
        if k == 7:
            tb = t_edge(0.0, GAL + 1.2)
        ts = np.linspace(ta, tb, 6)
        # a door at each passage: the wall is cut 3 m round it
        for j in range(5):
            t0, t1 = ts[j], ts[j + 1]
            tm = (t0 + t1) / 2
            near_door = any(abs(tm - (-math.pi / 2 + q * math.pi / 8)) < 0.035 for q in range(1, 8))
            top = slope_h(tm, D_WALL, side) + keep[k] * (0.6 + 0.4 * math.sin(j * 1.7 + k))
            if near_door:
                top = min(top, slope_h(tm, D_WALL, side) + 0.3)
            poly = [curve(t, 0.0) for t in np.linspace(t0, t1, 3)] + \
                   [curve(t, D_WALL) for t in np.linspace(t1, t0, 3)]
            stone.append(extrude(poly, FOUND, top, STONE))
    return merge([tint(merge(seat), 0.06, 0.4, 11), tint(merge(stone), 0.1, 0.3, 12), merge(light), merge(dark)])


# ── Long-axis galleries ──────────────────────────────────────────────────────
def gallery_profile(d):
    """Top of the side walls of the galleries at depth d (they follow the cavea)."""
    if d > D_POD:
        return Y_POD + 0.3
    if d > D_PRAEC:
        return Y_POD + (D_IMA0 - min(d, D_IMA0)) / ROW_D * ROW_Y + 0.4
    if d > D_BALT:
        return Y_IMA + 0.4
    return slope_h(math.pi / 2, d) + 0.6


def floor_y(d):
    return 3.4 * (D_ARENA - d) / D_ARENA


def gallery():
    ms, brick = [], []
    ds = list(np.linspace(D_ARENA, 0.0, 16))
    prof = [(A - d, gallery_profile(d)) for d in ds]
    prof += [(A, FOUND), (A - D_ARENA, FOUND)]
    for w0, w1 in ((-GAL - 1.2, -GAL), (GAL, GAL + 1.2)):
        m = prism_xy(prof, -w1, -w0, STONE).R(0, -math.pi / 2, 0)
        ms.append(m)
    # ramped floor, paved
    V = [(-GAL, floor_y(D_ARENA), A - D_ARENA), (GAL, floor_y(D_ARENA), A - D_ARENA), (GAL, floor_y(0.0), A),
         (-GAL, floor_y(0.0), A)]
    ms.append(Mesh(V, [(0, 2, 1), (0, 3, 2)], (0.62, 0.58, 0.52)))
    # brick transverse arches over the outer part
    for d in (3.0, 7.0, 11.0, 15.0, 19.0):
        ys = floor_y(d) + 3.6
        arch = [(-GAL - 1.2, ys), (-GAL, ys)] + \
            [(GAL * math.cos(a), ys + GAL * math.sin(a)) for a in np.linspace(math.pi, 0, 11)[1:-1]] + \
            [(GAL, ys), (GAL + 1.2, ys), (GAL + 1.2, ys + GAL + 0.8), (-GAL - 1.2, ys + GAL + 0.8)]
        brick.append(prism_xy(arch, A - d - 0.45, A - d + 0.45, BRICK))
    # the gladiators' rooms on both sides by the arena: low walls
    for sx in (1, -1):
        x0, x1 = sorted((sx * (GAL + 1.2), sx * (GAL + 5.5)))
        ms.append(B(x0, x1, FOUND, Y_POD + 0.6, A - D_ARENA - 0.2, A - D_ARENA + 0.9, STONE_D))
    g = merge([tint(merge(ms), 0.1, 0.3, 21), tint(merge(brick), 0.08, 0.8, 22)])
    return [g, mirror_z(g)]


# ── Tribunes on the minor axis ───────────────────────────────────────────────
def tribunes():
    ms = []
    for sx in (1, -1):
        xa, xb = BX - D_POD, BX - 23.5
        x0, x1 = sorted((sx * xa, sx * xb))
        ms.append(B(x0, x1, FOUND, 5.2, -3.6, 3.6, STONE))
        ms.append(B(x0, x1, 5.2, 5.35, -3.6, 3.6, COPING))
        xp0, xp1 = sorted((sx * xa, sx * (xa + 0.5)))
        ms.append(B(xp0, xp1, 5.35, 6.3, -3.6, 3.6, STONE_D))          # parapet towards the arena
        if sx < 0:                                                     # the editor's tribune: two stairs
            for zc in (-5.0, 5.0):
                for k in range(6):
                    xe = sx * (BX - D_ARENA + 0.2 + 0.55 * k)
                    a, b = sorted((xe, xe + sx * 0.55))
                    ms.append(B(a, b, 0.0, 0.43 * (k + 1), zc - 0.7, zc + 0.7, SEAT))
        else:                                                          # passage under it to the outside
            xd = BX - D_ARENA
            ms.append(B(xd - 0.06, xd + 0.02, 0.0, 2.3, -1.3, 1.3, DARK))
    return tint(merge(ms), 0.08, 0.4, 31)


def main():
    out = ROOT / "models" / "Q2360379"
    out.mkdir(parents=True, exist_ok=True)
    m = merge([arena(), half_cavea(1), mirror_x(half_cavea(-1))] + gallery() + [tribunes()])
    tris, size = write_glb(out / "model.glb", [("anfiteatro_de_merida", m)], roughness=0.92)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
