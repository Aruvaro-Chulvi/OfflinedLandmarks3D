"""
Templo de Debod (Q1140249), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: an ancient Egyptian temple from Lower Nubia (begun c. 200–180 BC under Adikhalamani, enlarged
under the Ptolemies and Augustus), given by Egypt to Spain in 1968 and rebuilt in Madrid in 1970–1972 on the
site of the Cuartel de la Montaña (reconstruction directed by Martín Almagro Basch). Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 6432785 (the temple): main block 12.6 m wide × 20.0 m long, its long axis at 77.6°;
  a lower annex 5.9 × 4.5 m on the south side near the front. Relation 7875419: the two stone gateways on the
  axis, 3.6 × 2.4 m and 3.9 × 3.1 m, 28.5 and 48.5 m in front of the temple. Way 549109296: the raised
  terrace around the temple (≈33 × 35 m) and the causeway (≈13.4–16.2 m wide) that runs east between the
  two basins of the reflecting pool and carries the gateways.
- The pool (not mapped as water in OSM): measured on the PNOA orthophoto (IGN, CC BY 4.0): ≈41 m across
  (x −19.6…21.6) from the terrace to 87 m in front of the temple, wrapping round the east end of the causeway
  (which stops at 72 m).
- The entrance (pronaos with four columns with composite floral capitals and screen walls between them)
  faces east, towards the gateways and the pool: the main façade faces 77.6°.
- Heights from photos on Wikimedia Commons (people and benches for scale, and counting the ~0.5 m stone
  courses), ±10 %: terrace 1.3 m above the promenade; temple walls 6.4 m above the terrace with a cavetto
  cornice on the front; the annex ~4.7 m; gateways ~5.5 and ~6.0 m with openings ~1.3 × 3.7 m.
- The low grey hipped roof that protects the temple (seen on the orthophoto) is kept just above the parapet.
Sandstone (golden beige, the restored parts a little lighter), grey granite terrace and pool kerbs, water
as a flat blue-grey surface 0.2 m below the kerbs.

Axes (docs/MODEL_SPEC.md): +Y up, the entrance (east) façade faces +Z, the gateways and the pool lie
towards +Z, +X points to 347.6° (north-north-west). Origin on the ground (promenade level) at the centroid
of the OSM outline of the temple; the temple axis is at x = +0.9.

    python model_sources/Q1140249_templo_de_debod.py   -> models/Q1140249/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, cap, walls, extrude, offset, write_glb, GLASS  # noqa: E402

rng = np.random.default_rng(1972)

SAND = np.array((0.79, 0.66, 0.46))
SAND_L = (0.84, 0.73, 0.54)
JOINT = (0.66, 0.54, 0.37)
DARK = (0.30, 0.24, 0.17)
GRANITE = (0.56, 0.56, 0.54)
PAVE = (0.80, 0.79, 0.75)
KERB = (0.70, 0.70, 0.68)
WATER = (0.36, 0.46, 0.48)
ROOF = (0.68, 0.69, 0.70)
FOUND = -3.0

PT = 1.3                       # top of the terrace
TOP = PT + 6.4                 # top of the temple walls
CW = 0.25                      # causeway and kerb tops
WL = 0.05                      # water level
AX = 0.9                       # the temple axis (x)

# OSM relation 6432785 in model x/z (heading 77.6°, origin its centroid)
TEMPLE = [(7.2, -10.6), (-5.4, -10.6), (-5.4, 9.4), (7.2, 9.4)]
ANNEX = [(-5.4, 3.8), (-11.3, 3.8), (-11.3, 8.3), (-5.4, 8.3)]
# way 549109296: the terrace (front-corner stair notches simplified) and the causeway
TERRACE = [(15.3, -16.4), (17.6, 14.7), (17.9, 18.6), (-15.6, 18.7), (-15.2, 14.8), (-13.5, -16.4)]
CAUSEWAY = [(-5.7, 18.85), (7.7, 18.85), (9.2, 72.4), (-7.0, 72.2)]
POOL = [(-19.6, 18.75), (21.6, 18.75), (21.6, 87.0), (-19.6, 87.0)]
# relation 7875419: the gateways (x0, x1, z0, z1)
GATES = [(-0.7, 2.9, 37.9, 40.3), (-0.8, 3.1, 57.9, 61.0)]


def sc(c, k=0.03):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, k)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def rect(x0, x1, z0, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def courses(ring, y0, y1, step=0.5, c=JOINT, eps=0.012):
    """Thin joints between the stone courses (horizontal bands just proud of the wall)."""
    out = []
    o = offset(ring, eps)
    for y in np.arange(y0 + step, y1 - 0.1, step):
        out.append(walls(o, y - 0.015, y + 0.015, c))
    return out


def banded(ring, y0, y1, step=0.5, base=SAND, k=0.045):
    """Walls built course by course, each course a slightly different sandstone (old and restored blocks)."""
    out = []
    ys = list(np.arange(y0, y1, step)) + [y1]
    for a, b in zip(ys[:-1], ys[1:]):
        if b - a > 1e-6:
            out.append(walls(ring, a, b, sc(base, k)))
    return merge(out)


def cavetto(x0, x1, z, y0, h, proj, c, seg=6, side=1):
    """Egyptian gorge cornice along x on the face z (facing side): a quarter-round flaring out, with a flat
    top slab; ends closed."""
    prof = [(proj * (1 - math.cos(t * math.pi / 2)), y0 + h * math.sin(t * math.pi / 2)) for t in np.linspace(0, 1, seg + 1)]
    V, F = [], []
    for d, y in prof:
        V += [(x0 - d, y, z + side * d), (x1 + d, y, z + side * d)]
    for i in range(seg):
        a = 2 * i
        F += [(a, a + 1, a + 3), (a, a + 3, a + 2)]
    m = Mesh(V, F, c)
    v = m.v[m.f[0]]
    n = np.cross(v[1] - v[0], v[2] - v[0])
    if n[2] * side < 0:
        m.f = m.f[:, ::-1]
    return m


def gorge_box(x0, x1, z0, z1, y0, h, proj, c):
    """A cavetto cornice running round a rectangle (four flaring faces + the top)."""
    prof = [(proj * (1 - math.cos(t * math.pi / 2)), y0 + h * math.sin(t * math.pi / 2)) for t in np.linspace(0, 1, 7)]
    rings = [[(x0 - d, z0 - d), (x1 + d, z0 - d), (x1 + d, z1 + d), (x0 - d, z1 + d)] for d, _ in prof]
    V, F = [], []
    for (d, y), r in zip(prof, rings):
        V += [(x, y, z) for x, z in r]
    for i in range(len(prof) - 1):
        for j in range(4):
            a, b = i * 4 + j, i * 4 + (j + 1) % 4
            F += [(a, b, b + 4), (a, b + 4, a + 4)]
    m = Mesh(V, F, c)
    v = m.v[m.f[0]]; ctr = np.array([(x0 + x1) / 2, 0, (z0 + z1) / 2])
    n = np.cross(v[1] - v[0], v[2] - v[0])
    if np.dot(n, v[0] - ctr) < 0:
        m.f = m.f[:, ::-1]
    top = prof[-1]
    slab = B(x0 - proj - 0.05, x1 + proj + 0.05, top[1], top[1] + 0.18, z0 - proj - 0.05, z1 + proj + 0.05, c)
    return [m, slab]


def torus_edges(x0, x1, z0, z1, y0, y1, c, r=0.13):
    """The round mouldings on the four vertical edges."""
    return [cyl(r, r, y1 - y0, c, 8).T(x, y0, z) for x in (x0, x1) for z in (z0, z1)]


# ── The terrace, the causeway and the pool ──────────────────────────────────
def site():
    out = []
    out.append(extrude(TERRACE, FOUND, PT - 0.25, GRANITE, top=False))
    out.append(extrude(offset(TERRACE, 0.05), PT - 0.25, PT, sc(KERB, 0.0), top_c=PAVE))
    # steps down from the terrace to the causeway
    n = 5
    for i in range(n):
        z0 = 18.62 + i * 0.466
        out.append(B(-5.8 + 0.03 * i, 7.75 + 0.03 * i, -0.5, PT - (i + 1) * (PT - CW) / n, z0, z0 + 0.466 + (0.01 if i < n - 1 else 0), sc(KERB, 0.01)))
    # the causeway (starts after the steps)
    cw = [(-5.75, 20.95), (7.75, 20.95), (9.2, 72.4), (-7.0, 72.2)]
    out.append(extrude(cw, -1.0, CW, GRANITE, top_c=PAVE))
    # the kerb round the pool and the water
    out.append(extrude(offset(POOL, 0.6), -1.0, CW + 0.05, KERB, holes=[POOL]))
    return out


def water():
    U = [(-19.6, 18.75), (-5.7, 18.75), (-7.0, 72.2), (9.2, 72.4), (7.7, 18.75), (21.6, 18.75), (21.6, 87.0), (-19.6, 87.0)]
    return cap(U, WL, WATER)


# ── The temple ───────────────────────────────────────────────────────────────
ZP = 6.9          # back of the pronaos (front of the closed rooms)
ZF = 9.4          # front of the temple


def temple():
    out = []
    x0, x1, z0, z1 = -5.4, 7.2, -10.6, ZF
    # the closed body (to the back of the pronaos) and the two front piers (antae) of the pronaos
    body = rect(x0, x1, z0, ZP)
    out.append(walls(body, PT - 0.3, PT, sc(SAND, 0.0)))
    out.append(banded(body, PT, TOP))
    out += courses(body, PT, TOP)
    for a, b in ((x0, -3.5), (5.3, x1)):
        out.append(B(a, b, PT - 0.3, TOP, ZP - 0.01, z1, sc(SAND, 0.01)))
        out.append(banded(offset(rect(a, b, ZP + 0.02, z1), 0.006), PT, TOP))
        out += courses(rect(a, b, ZP + 0.02, z1), PT, TOP)
    out += torus_edges(x0 - 0.02, x1 + 0.02, z0 - 0.02, z1 + 0.02, PT, TOP - 0.05, SAND_L)
    # pronaos: dark back wall with the doorway, four columns with floral capitals, screen walls between them
    out.append(B(-3.5, 5.3, PT, TOP - 0.6, ZP - 0.2, ZP + 0.01, DARK))
    out.append(B(AX - 0.8, AX + 0.8, PT, PT + 3.6, ZP + 0.01, ZP + 0.06, (0.16, 0.13, 0.10)))
    cols = [AX - 3.3, AX - 1.1, AX + 1.1, AX + 3.3]
    zc = ZF - 0.5
    for x in cols:
        out.append(B(x - 0.5, x + 0.5, PT, PT + 0.15, zc - 0.5, zc + 0.5, SAND_L))
        out.append(cyl(0.42, 0.40, 4.2, sc(SAND_L, 0.02), 14).T(x, PT + 0.15, zc))
        out.append(lathe([(0.40, 0), (0.5, 0.35), (0.68, 0.75), (0.74, 0.95), (0, 0.95)], SAND_L, 14).T(x, PT + 4.35, zc))   # bell capital
        for k in range(8):
            t = k * math.pi / 4
            out.append(box(0.2, 0.7, 0.12, SAND).R(0.25 * math.sin(t), 0, -0.25 * math.cos(t)).T(x + 0.55 * math.cos(t), PT + 4.55, zc + 0.55 * math.sin(t)))
        out.append(B(x - 0.45, x + 0.45, PT + 5.3, TOP - 0.7, zc - 0.45, zc + 0.45, SAND))               # abacus block
    # screen walls (half height) between the columns and the antae, except the doorway in the middle
    for a, b in ((-3.5, cols[0]), (cols[0], cols[1]), (cols[2], cols[3]), (cols[3], 5.3)):
        out.append(B(a, b, PT, PT + 2.7, zc - 0.25, zc + 0.25, sc(SAND, 0.01)))
        out.append(cavetto(a + 0.05, b - 0.05, zc + 0.25, PT + 2.7, 0.35, 0.18, SAND_L))
        out.append(B(a, b, PT + 2.7, PT + 2.95, zc - 0.3, zc + 0.25, SAND_L))
    # the doorway between the two middle columns: jambs and a broken lintel (the usual Ptolemaic screen door)
    for a, b in ((cols[1] + 0.35, AX - 0.6), (AX + 0.6, cols[2] - 0.35)):
        out.append(B(a, b, PT, PT + 2.95, zc - 0.25, zc + 0.25, sc(SAND, 0.01)))
    # architrave over the columns, cavetto cornice across the front, its top slab
    out.append(B(-3.5, 5.3, TOP - 0.7, TOP, ZP, zc + 0.5, sc(SAND, 0.01)))
    out.append(B(x0 - 0.15, x1 + 0.15, TOP - 0.85, TOP - 0.7, z1 - 0.05, z1 + 0.12, SAND_L))       # torus band
    out.append(cavetto(x0 - 0.15, x1 + 0.15, z1 + 0.12, TOP, 0.85, 0.55, sc(SAND_L, 0.0)))
    out.append(B(x0 - 0.75, x1 + 0.75, TOP + 0.85, TOP + 1.05, z1 - 2.5, z1 + 0.72, SAND_L))
    out.append(B(x0, x1, TOP, TOP + 0.85, ZP, z1 + 0.11, sc(SAND, 0.0)))                          # behind the cornice
    # winged sun disc over the doorway
    out.append(cyl(0.32, 0.32, 0.1, (0.62, 0.48, 0.28), 14).R(math.pi / 2, 0, 0).T(AX, TOP - 0.35, zc + 0.62))
    for s in (-1, 1):
        out.append(B(AX + s * 0.3 - (1.3 if s < 0 else 0), AX + s * 0.3 + (1.3 if s > 0 else 0), TOP - 0.45, TOP - 0.25, zc + 0.5, zc + 0.56, (0.62, 0.48, 0.28)))
    # coping round the rest of the top, the low protective roof
    out.append(B(x0 - 0.12, x1 + 0.12, TOP, TOP + 0.35, z0 - 0.12, z1 - 2.5, SAND_L))
    out.append(hip(x0 + 0.3, x1 - 0.3, z0 + 0.3, z1 - 2.8, TOP + 0.2, 0.2, ROOF))
    out.append(B(AX - 0.6, AX + 0.6, TOP + 1.0, TOP + 1.25, -6.0, -4.8, (0.90, 0.91, 0.92)))       # skylights
    out.append(B(AX - 0.6, AX + 0.6, TOP + 1.0, TOP + 1.25, 0.6, 1.8, (0.90, 0.91, 0.92)))
    # the annex on the south side (Roman-period rooms), lower
    ax0, ax1, az0, az1 = -11.3, -5.4, 3.8, 8.3
    at = PT + 4.7
    out.append(walls(rect(ax0, ax1 + 0.01, az0, az1), PT - 0.3, PT, sc(SAND_L, 0.0)))
    out.append(banded(rect(ax0, ax1 + 0.01, az0, az1), PT, at, base=np.array(SAND_L)))
    out += courses(rect(ax0, ax1, az0, az1), PT, at)
    out.append(B(ax0 - 0.1, ax1 - 0.02, at, at + 0.3, az0 - 0.1, az1 + 0.1, SAND_L))
    out.append(cap(rect(ax0 + 0.05, ax1 - 0.05, az0 + 0.05, az1 - 0.05), at + 0.31, ROOF))
    out += torus_edges(ax0 - 0.02, ax0 - 0.02, az0 - 0.02, az1 + 0.02, PT, at - 0.05, SAND_L)[:2]
    return out


def hip(x0, x1, z0, z1, y0, k, c):
    """Hipped roof, ridge along the longer side, pitch k."""
    if (x1 - x0) < (z1 - z0):
        m = hip(z0, z1, x0, x1, y0, k, c)
        m.v = m.v[:, [2, 1, 0]]; m.f = m.f[:, ::-1]
        return m
    W = z1 - z0
    h = k * W / 2
    zc = (z0 + z1) / 2
    r0, r1 = x0 + W / 2, x1 - W / 2
    V = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (r0, y0 + h, zc), (r1, y0 + h, zc)]
    F = [(0, 4, 5), (0, 5, 1), (3, 2, 5), (3, 5, 4), (0, 3, 4), (1, 5, 2)]
    return Mesh(V, F, c)


# ── The gateways ─────────────────────────────────────────────────────────────
def gateway(x0, x1, z0, z1, h, ow, oh, broken=False):
    out = []
    xc = (x0 + x1) / 2
    y0 = CW
    body = h - 0.65
    for a, b in ((x0, xc - ow / 2), (xc + ow / 2, x1)):
        out.append(B(a, b, FOUND, y0 + body, z0, z1, sc(SAND, 0.02)))
        out.append(banded(offset(rect(a, b, z0, z1), 0.006), y0, y0 + body, step=0.52))
        out += courses(rect(a, b, z0, z1), y0, y0 + body, step=0.52)
    out.append(B(xc - ow / 2 - 0.01, xc + ow / 2 + 0.01, y0 + oh, y0 + body, z0, z1, sc(SAND, 0.02)))  # lintel
    out.append(banded(offset(rect(xc - ow / 2, xc + ow / 2, z0, z1), 0.009), y0 + oh, y0 + body, step=0.52))
    out += courses(rect(xc - ow / 2, xc + ow / 2, z0, z1), y0 + oh, y0 + body, step=0.52)
    out.append(B(xc - ow / 2 + 0.01, xc + ow / 2 - 0.01, y0 - 0.02, y0 + 0.08, z0 + 0.01, z1 - 0.01, SAND_L))  # threshold
    out += torus_edges(x0 - 0.03, x1 + 0.03, z0 - 0.03, z1 + 0.03, y0, y0 + body - 0.1, SAND_L, r=0.1)
    out.append(B(x0 - 0.08, x1 + 0.08, y0 + body - 0.12, y0 + body - 0.01, z0 - 0.08, z1 + 0.08, SAND_L))  # torus band
    g = gorge_box(x0, x1, z0, z1, y0 + body, 0.45, 0.35, sc(SAND_L, 0.0))
    if broken:                     # a missing block at one end of the cornice slab (as on the eastern gate)
        g[-1] = B(x0 - 0.4, x1 - 0.9, y0 + body + 0.45, y0 + body + 0.63, z0 - 0.4, z1 + 0.4, SAND_L)
    out += g
    # the sun disc on the gorge over the passage, both faces
    for z, s in ((z1 + 0.2, 1), (z0 - 0.2, -1)):
        out.append(cyl(0.22, 0.22, 0.08, (0.66, 0.52, 0.32), 12).R(math.pi / 2, 0, 0).T(xc, y0 + body + 0.25, z - s * 0.04))
    return out


def main():
    out = ROOT / "models" / "Q1140249"
    out.mkdir(parents=True, exist_ok=True)
    stone = merge(site() + temple()
                  + gateway(*GATES[0], h=5.5, ow=1.3, oh=3.6)
                  + gateway(*GATES[1], h=6.0, ow=1.4, oh=3.8, broken=True))
    w = water()
    parts = [("templo_de_debod", stone), ("pool", w, {"material": "water"})]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.9, materials={"water": GLASS})
    allv = np.vstack([p[1].v for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    # footprint: convex hull of the terrace, the annex and the pool kerb, grown by 1 m
    pts = np.array(TERRACE + ANNEX + offset(POOL, 0.6))
    hull = convex_hull(pts)
    fp = [(round(x, 2), round(z, 2)) for x, z in offset(hull, 1.0)]
    print("footprint:", fp)


def convex_hull(P):
    P = sorted(map(tuple, P))
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    lower, upper = half(P), half(P[::-1])
    return lower[:-1] + upper[:-1]


if __name__ == "__main__":
    main()
