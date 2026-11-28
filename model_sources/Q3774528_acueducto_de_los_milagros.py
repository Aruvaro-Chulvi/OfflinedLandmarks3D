"""
Acueducto de los Milagros, Mérida (Q3774528) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: the arcades that carried the aqueduct from Proserpina over the valley of the Albarregas into Augusta
Emerita (1st century AD, rebuilt around 300 AD). A ruin; public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 202213492 (man_made=bridge, bridge=aqueduct: the standing arcade, 188.6 m long and 3.66 m
  wide, its axis at 8.75°) and way 202213494 (the row of lower piers south of the railway, 45 m, turning towards
  the city at 47°); way 37859139 is the whole course of the aqueduct. The east face is taken as the main face (+Z,
  98.75°), so +X runs north along the arcade.
- Wikipedia (es, en): the arcade over the Albarregas, 830 m long with up to 25 m of height at the bottom of the
  valley; robust piers of opus caementicium faced with five courses of granite ashlar alternating with five courses
  of brick; brick arches at different heights joining the piers (granite voussoirs over the river); 73 piers left
  more or less ruined (38 arched pillars standing).
- PNOA orthophoto: about 27 piers on the 188.6 m (bays of 7.14 m), buttresses on both faces of each pier; the
  southern row of 7 lower piers. Photos on Commons: pier 2.9 m along the line, arches of about 4.2 m, three tiers
  of arches (springings at about 7.5, 14.5 and 20.5 m above the river bed), granite bands about 2.2 m and brick
  bands about 0.4 m high, storks' nests on the pier tops (heights by proportion, ±10 %).
- IGN MDT05 (terrain.json): the bed of the Albarregas at 204.0 m under the arcade (x ≈ +20), 206 m at the origin
  side, 208 m at both ends; the southern row stands on the slope at 214–216 m.

Axes (docs/MODEL_SPEC.md): +Y up, y = 0 the ground at the origin (205.0 m); the arcade along X (north); the origin
on its axis at the middle of the standing arcade.

    python model_sources/Q3774528_acueducto_de_los_milagros.py   -> models/Q3774528/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, triangulate, fbm, noise3, write_glb  # noqa: E402

rng = np.random.default_rng(1993)

GRANITE = np.array((0.68, 0.64, 0.56))     # granite ashlar, grey-beige with lichen
BRICK = np.array((0.64, 0.33, 0.23))       # Roman brick, deep red
CORE = np.array((0.58, 0.53, 0.45))        # the broken opus caementicium at the tops
NEST = np.array((0.36, 0.30, 0.22))        # storks' nests (twigs)
WHITE = np.array((0.92, 0.91, 0.88))

E0 = 205.0                                 # ground at the origin, m a.s.l. = y 0
X0, X1 = -94.3, 94.3                       # ends of the arcade (OSM way 202213492)
ZW = 1.83                                  # half width of the piers (OSM 3.66 m)
PW = 2.9                                   # piers along the line
N_PIERS = 27
BAND, BRICK_H = 2.6, 0.4                   # granite 2.2 m + brick 0.4 m
TIERS = (7.5, 14.5, 20.5)                  # springing of the three tiers of arches
TOP = 24.5                                 # the highest pier tops (229.5 m, 25.5 m over the river bed)

# lowest ground across the arcade (MDT05, z −6..+6), metres over y 0
GROUND = [(-230, 12.0), (-210, 11.0), (-200, 10.3), (-190, 9.7), (-180, 9.0), (-170, 8.3), (-160, 7.7), (-150, 6.4),
          (-130, 5.3), (-110, 3.8), (-100, 3.1), (-90, 2.3), (-80, 2.0), (-60, 1.6), (-50, 1.0), (-15, 1.0),
          (-5, 0.2), (0, -0.3), (5, -0.7), (10, -0.9), (40, -1.0), (45, -0.8), (50, -0.4), (55, 0.2), (60, 0.8),
          (70, 1.0), (80, 1.2), (85, 1.5), (90, 2.0), (95, 2.2), (100, 2.8), (110, 3.4), (120, 4.2)]


def ground(x):
    xs, ys = zip(*GROUND)
    return float(np.interp(x, xs, ys))


# ── Mesh helpers ─────────────────────────────────────────────────────────────
def prism_xy(poly, z0, z1, c):
    """A prism from an outline drawn in the (x, y) plane, from z0 to z1."""
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


def bands(y0, y1):
    """Cut [y0, y1] at the granite/brick course changes -> [(ya, yb, is_brick)]."""
    cuts = {y0, y1}
    k0 = math.floor((y0 + 1.0) / BAND) - 1
    for k in range(k0, k0 + int((y1 - y0) / BAND) + 3):
        for y in (k * BAND - 1.0, k * BAND - 1.0 + BAND - BRICK_H):
            if y0 < y < y1:
                cuts.add(y)
    ys = sorted(cuts)
    out = []
    for a, b in zip(ys[:-1], ys[1:]):
        m = (a + b) / 2
        out.append((a, b, ((m + 1.0) % BAND) >= BAND - BRICK_H))
    return out


def banded(x0, x1, z0, z1, y0, y1, seed, top=True, brick_out=0.0):
    """A box faced with courses of granite ashlar and brick (sides only, plus a top)."""
    ms = []
    for ya, yb, brick in bands(y0, y1):
        d = -brick_out if brick else 0.0
        P = [(x0 - d, z0 - d), (x1 + d, z0 - d), (x1 + d, z1 + d), (x0 - d, z1 + d)]
        v, f = [], []
        for i in range(4):
            (xa, za), (xb, zb) = P[i], P[(i + 1) % 4]
            o = len(v)
            v += [(xa, ya, za), (xb, ya, zb), (xb, yb, zb), (xa, yb, za)]
            f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        m = Mesh(v, f, BRICK if brick else GRANITE)
        ms.append(m.paint(stone(BRICK if brick else GRANITE, seed + (7 if brick else 0))))
    if top:
        ms.append(box(x1 - x0, 0.01, z1 - z0, CORE).T((x0 + x1) / 2, y1 - 0.01, (z0 + z1) / 2).paint(stone(CORE, seed + 3)))
    m = merge(ms)
    # the side quads were wound for a ring that runs counter-clockwise seen from below: flip to face out
    c = np.array([(x0 + x1) / 2, 0, (z0 + z1) / 2])
    a, b, cc = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    nrm = np.cross(b - a, cc - a)
    out = (a + b + cc) / 3 - c
    out[:, 1] = 0
    side = np.abs(nrm[:, 1]) < 1e-6
    flip = side & ((nrm * out).sum(1) < 0)
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def stone(base, seed):
    base = np.asarray(base, float)

    def fn(v):
        n1 = fbm(v, 0.15, seed, 3) - 0.5
        n2 = noise3(v * np.array([1.0, 2.2, 1.0]), 1.1, seed + 5) - 0.5
        lic = np.clip((fbm(v, 0.35, seed + 9, 2) - 0.55) * 3, 0, 1)[:, None] * 0.35      # lichen, grey-green
        c = np.outer(1 + 0.22 * n1 + 0.12 * n2, base)
        return np.clip(c * (1 - lic) + np.array((0.55, 0.56, 0.50)) * lic, 0, 1)
    return fn


def semi(xa, xb, ys, n, extra=0.0):
    xc, r = (xa + xb) / 2, (xb - xa) / 2 + extra
    return [(xc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(math.pi, 0, n + 1)]


# ── The arcade ───────────────────────────────────────────────────────────────
def pier(x, yb, yt, seed, buttress=True):
    """One pier from yb to yt (ragged top), with its two buttresses."""
    ms = []
    h0 = x - PW / 2, x + PW / 2
    # the top is broken: one half a little lower than the other
    cut = yt - rng.uniform(0.4, 1.8)
    side = rng.integers(2)
    xm = x + rng.uniform(-0.5, 0.5)
    ms.append(banded(h0[0], h0[1], -ZW, ZW, yb, cut, seed))
    if side:
        ms.append(banded(h0[0], xm, -ZW, ZW, cut, yt, seed + 1))
    else:
        ms.append(banded(xm, h0[1], -ZW, ZW, cut, yt, seed + 1))
    if buttress:
        for s in (-1, 1):
            b1 = min(TIERS[1], yt - 1.0)
            b2 = min(TIERS[2], yt - 0.5)
            ms.append(banded(x - 0.8, x + 0.8, *sorted((s * ZW, s * (ZW + 1.0))), yb, b1, seed + 4, brick_out=0.02))
            if b2 > b1 + 0.5:
                ms.append(banded(x - 0.8, x + 0.8, *sorted((s * ZW, s * (ZW + 0.5))), b1, b2, seed + 5, brick_out=0.02))
    return ms


def arch(xa, xb, ys, granite=False, depth=2.4, filled=False):
    """A brick arch (or granite voussoirs over the river) joining two piers, springing at ys."""
    n = 12
    col = GRANITE * 1.02 if granite else BRICK
    t = 0.75 if granite else 0.55
    if filled:
        r = (xb - xa) / 2
        poly = [(xa, ys), (xa, ys + r + t + 0.4), (xb, ys + r + t + 0.4), (xb, ys)] + semi(xa, xb, ys, n)[::-1][1:-1]
        m = prism_xy(poly, -depth / 2, depth / 2, col)
    else:
        m = prism_xy(semi(xa, xb, ys, n, t) + semi(xa, xb, ys, n)[::-1], -depth / 2, depth / 2, col)
    return m.paint(stone(col, 41 if granite else 43))


def nest(x, y):
    ms = [cyl(0.85, 1.0, 0.55, NEST, 10).T(x, y, 0).paint(stone(NEST, 51)),
          cyl(0.65, 0.0, 0.12, NEST * 0.8, 10).T(x, y + 0.55, 0)]
    if rng.random() < 0.6:                                     # a stork standing on it
        ms.append(cyl(0.16, 0.12, 0.55, WHITE, 6).R(0, 0, 0.35).T(x + 0.2, y + 0.6, 0.1))
        ms.append(cyl(0.05, 0.05, 0.5, (0.1, 0.1, 0.1), 4).T(x + 0.15, y + 0.55, -0.1))
    return ms


def arcade():
    parts = []
    xs = np.linspace(X0 + PW / 2, X1 - PW / 2, N_PIERS)
    tops = []
    for i, x in enumerate(xs):
        g = ground(x)
        t = TOP - rng.uniform(0.0, 1.5)
        if i in (0, N_PIERS - 1):
            t = min(t, 17.0 + rng.uniform(0, 2))              # the ends stand lower
        if rng.random() < 0.08:
            t = rng.uniform(10.0, 15.0)                       # a few broken lower
        tops.append(t)
        parts += pier(x, g - 3.0, t, 3 + i)
    river = (10.0, 40.0)
    for i in range(N_PIERS - 1):
        xa, xb = xs[i] + PW / 2, xs[i + 1] - PW / 2
        low = min(tops[i], tops[i + 1])
        over_river = river[0] <= (xa + xb) / 2 <= river[1]
        for k, ys in enumerate(TIERS):
            crown = ys + (xb - xa) / 2 + 0.6
            if crown > low + 0.3:
                continue
            keep = (0.95, 0.85, 0.8)[k]
            if rng.random() > keep:
                continue
            parts.append(arch(xa, xb, ys, granite=(over_river and k == 0), filled=(k == 2)))
    # storks' nests on some of the tops
    for i in rng.choice(N_PIERS, 9, replace=False):
        parts += nest(xs[i] + rng.uniform(-0.4, 0.4), tops[i])
    return parts


def south_row():
    """The row of lower piers south of the railway (OSM way 202213494), turning towards the city."""
    parts = []
    a = np.array((-176.2, -0.2))
    b = np.array((-208.7, -26.8))
    d = b - a
    L = float(np.linalg.norm(d))
    u = d / L
    ang = math.atan2(-u[1], u[0])                     # rotation about y that takes +x onto u
    for k in range(7):
        p = a + u * (L * k / 6)
        g = ground(float(p[0]))
        h = rng.uniform(3.0, 7.5)
        ms = merge(pier(0.0, g - 3.0, g + h, 61 + k, buttress=(h > 5)))
        ms.R(0, ang, 0).T(float(p[0]), 0, float(p[1]))
        parts.append(ms)
    return parts


def main():
    out = ROOT / "models" / "Q3774528"
    out.mkdir(parents=True, exist_ok=True)
    m = merge(arcade() + south_row())
    tris, size = write_glb(out / "model.glb", [("acueducto_de_los_milagros", m)], roughness=0.92)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
