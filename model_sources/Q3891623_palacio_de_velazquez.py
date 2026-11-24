"""
Palacio de Velázquez (Q3891623), Parque del Buen Retiro, Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Ricardo Velázquez Bosco (1843–1923), 1881–1883, for the Exposición Nacional de Minería of 1883;
two-tone brick (red and buff bands), glazed ceramics by Daniel Zuloaga, iron and glass roofs. Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 23595002: outline, 76.2 × 40.2 m (smallest rectangle), long axis at 74.8°. The main front
  (the triple-arched portico with its stair and two winged lions) faces 164.8° (SSE): that is +Z. The north
  front has the same portico, recessed between the two north wings. Origin: centroid of the outline.
- PNOA orthophoto (IGN): roof layout — a glass barrel vault 18 m wide running north–south over the central hall;
  on each wing three parallel metal barrel vaults running east–west with glazed strips on top; four square
  corner pavilions with metal cloister vaults and a skylight; a flat terrace with skylights along the north.
- Heights measured on frontal photos (Wikimedia Commons «037 … façana sud», «053 … façana nord», «051 … angle
  sud-oest»), scaled by the 1.7 m visitors and the ±10 m pylons of the portico (±8 %): wing cornice 10.5 m,
  central pylons and cornice 13.0 m with a balustrade to 14.0 m, glass vault to about 21.5 m, pavilion cornice
  12.5 m with their metal vaults to about 17 m.

Axes (docs/MODEL_SPEC.md): +Y up, the south portico faces +Z, +X points to 74.8° (east-north-east), origin on
the ground at the centroid of the outline.

    python model_sources/Q3891623_palacio_de_velazquez.py   -> models/Q3891623/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import (GLASS, Mesh, box, cap, cyl, extrude, merge, offset, ring_area, triangulate,  # noqa: E402
                     tube, walls, write_glb)

rng = np.random.default_rng(1883)

RED = np.array((0.68, 0.34, 0.25))          # red brick
BUFF = np.array((0.86, 0.70, 0.52))         # buff (yellow) brick
STONE = np.array((0.88, 0.80, 0.67))        # cornices, plinth, balustrades (artificial stone)
TILE_B = (0.22, 0.34, 0.62)                 # Zuloaga's blue-and-white glazed tiles
TILE_W = (0.90, 0.90, 0.86)
METAL_C = (0.56, 0.59, 0.61)                # zinc / sheet-metal vaults
GLASS_C = (0.56, 0.69, 0.75)
IRON = (0.22, 0.25, 0.26)
WIN = (0.13, 0.15, 0.16)
FLAT = (0.66, 0.64, 0.60)
PAVE = (0.84, 0.83, 0.79)
FOUND = -3.0

WING = 10.5          # wing cornice
CEN = 13.0           # central pylons and pavilions cornice band
PAV = 12.5           # corner pavilions cornice

# ── OpenStreetMap way 23595002 (© OpenStreetMap contributors, ODbL), in model x/z for heading 164.8 ─────────
OUTLINE = [(-36.379, -19.784), (-10.275, -19.526), (-10.323, -13.717), (-5.682, -13.675), (-5.688, -12.745),
           (5.59, -12.627), (5.596, -13.718), (9.958, -13.684), (10.011, -19.768), (36.31, -19.515),
           (36.238, -10.445), (38.072, -10.43), (37.994, -0.693), (35.522, -0.72), (35.461, 6.754), (38.002, 6.789),
           (37.923, 17.503), (9.878, 17.236), (9.85, 20.382), (5.568, 20.346), (5.575, 18.99), (-5.95, 18.873),
           (-5.963, 20.377), (-10.122, 20.341), (-10.095, 17.298), (-38.094, 17.021), (-38.006, 6.597),
           (-35.64, 6.63), (-35.585, -0.858), (-38.137, -0.884), (-38.055, -10.734), (-36.449, -10.723)]

# derived pieces (orthophoto + outline)
CENTRAL = [(-10.3, -12.63), (10.0, -12.63), (9.9, 18.95), (-10.1, 18.95)]          # pylons + central hall
PYL_S = [[(5.57, 17.2), (9.88, 17.2), (9.85, 20.38), (5.57, 20.35)],
         [(-10.12, 17.2), (-5.96, 17.2), (-5.96, 20.38), (-10.12, 20.34)]]
PYL_N = [[(5.6, -13.72), (9.96, -13.68), (9.96, -12.0), (5.6, -12.0)],
         [(-10.32, -13.72), (-5.68, -13.68), (-5.68, -12.0), (-10.32, -12.0)]]
PAVS = [[(25.6, -10.45), (38.07, -10.43), (37.99, -0.69), (25.6, -0.72)],
        [(25.6, 6.75), (38.0, 6.79), (37.92, 17.50), (25.6, 17.40)],
        [(-38.06, -10.73), (-25.6, -10.72), (-25.6, -0.86), (-38.14, -0.88)],
        [(-38.01, 6.60), (-25.6, 6.62), (-25.6, 17.13), (-38.09, 17.02)]]
VAULT_HW = 8.6       # half span of the central glass vault
WING_VAULTS = [(-6.2, 4.3), (2.6, 4.3), (11.4, 4.3)]    # (z centre, half span) of the three vaults of each wing


def jit(c, s=0.015):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, s)), 0, 1))


def ccw(P):
    P = [tuple(map(float, p)) for p in P]
    return P if ring_area(P) > 0 else P[::-1]


def edges(P):
    out = []
    for i in range(len(P)):
        a, b = np.array(P[i]), np.array(P[(i + 1) % len(P)])
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        u = (b - a) / L
        out.append((a, b, u, np.array([u[1], -u[0]]), L))
    return out


def on_wall(poly_sy, p0, u, n, c, eps=0.05):
    pts, tris = triangulate(poly_sy)
    v = np.array([(p0[0] + u[0] * s + n[0] * eps, y, p0[1] + u[1] * s + n[1] * eps) for s, y in pts])
    m = Mesh(v, tris, c)
    a, b, cc = v[tris[0, 0]], v[tris[0, 1]], v[tris[0, 2]]
    if np.dot(np.cross(b - a, cc - a), (n[0], 0, n[1])) < 0:
        m.f = m.f[:, ::-1]
    return m


def rect(s0, s1, y0, y1):
    return [(s0, y0), (s1, y0), (s1, y1), (s0, y1)]


def arch(s0, s1, y0, ys, seg=8):
    r, sc = (s1 - s0) / 2, (s0 + s1) / 2
    return [(s0, y0), (s1, y0)] + [(sc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)][:-1] + [(s0, ys)]


def slab(p0, u, n, s0, s1, y0, y1, d_out, d_in, c):
    q = [np.array(p0) + u * s0 + n * d_out, np.array(p0) + u * s1 + n * d_out,
         np.array(p0) + u * s1 - n * d_in, np.array(p0) + u * s0 - n * d_in]
    return extrude([tuple(x) for x in q], y0, y1, c, bottom=True)


def banded(P, y0, y1, hole=False):
    """Walls in bands of red and buff brick (the polychrome brickwork)."""
    out, y, k = [], y0, 0
    while y < y1 - 1e-6:
        h = 0.55 if k % 2 == 0 else 0.3
        y2 = min(y1, y + h)
        out.append(walls(P, y, y2, jit(RED if k % 2 == 0 else BUFF, 0.01), hole=hole))
        y, k = y2, k + 1
    return out


def vault_z(xc, hw, z0, z1, y0, rise, c, seg=14):
    """A barrel vault along z (x from xc − hw to xc + hw), springing at y0."""
    t = np.linspace(0, math.pi, seg + 1)
    xs, ys = xc + hw * np.cos(t), y0 + rise * np.sin(t)
    v, f = [], []
    for i in range(seg):
        o = len(v)
        v += [(xs[i], ys[i], z0), (xs[i + 1], ys[i + 1], z0), (xs[i + 1], ys[i + 1], z1), (xs[i], ys[i], z1)]
        f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    m = Mesh(v, f, c)
    return m, list(zip(xs, ys))


def vault_x(zc, hw, x0, x1, y0, rise, c, seg=10, t0=0.0, t1=math.pi, eps=0.0):
    t = np.linspace(t0, t1, seg + 1)
    zs, ys = zc + (hw + eps) * np.cos(t), y0 + (rise + eps) * np.sin(t)
    v, f = [], []
    for i in range(seg):
        o = len(v)
        v += [(x0, ys[i], zs[i]), (x0, ys[i + 1], zs[i + 1]), (x1, ys[i + 1], zs[i + 1]), (x1, ys[i], zs[i])]
        f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    return Mesh(v, f, c)


def end_x(zc, hw, x, y0, rise, c, seg=10):
    t = np.linspace(0, math.pi, seg + 1)
    poly = [(zc + hw * math.cos(a), y0 + rise * math.sin(a)) for a in t]
    pts, tris = triangulate(poly)
    return Mesh(np.array([(x, y, z) for z, y in pts]), tris, c)


def balustrade(p0, u, n, s0, s1, y0, h=1.0, c=STONE):
    out = [slab(p0, u, n, s0, s1, y0, y0 + 0.18, 0.05, 0.25, jit(c)),
           slab(p0, u, n, s0, s1, y0 + h - 0.16, y0 + h, 0.05, 0.25, jit(c))]
    k = max(1, int((s1 - s0) / 0.45))
    for i in range(k):
        s = s0 + (i + 0.5) * (s1 - s0) / k
        q = np.array(p0) + u * s - n * 0.1
        out.append(box(0.16, h - 0.34, 0.16, jit(c)).T(q[0], y0 + 0.18, q[1]))
    return out


def lion(x, z, y0, c, facing):
    """A winged lion on a pedestal, stylised: a crouching body and two raised wings."""
    out = [box(1.1, 1.3, 1.6, c).T(x, y0, z),
           box(0.75, 0.75, 1.3, c).T(x, y0 + 1.3, z),
           box(0.55, 0.6, 0.55, c).T(x, y0 + 2.05, z + 0.45 * facing)]
    for sx in (-1, 1):
        out.append(box(0.12, 1.2, 0.8, c).R(0, 0, -0.35 * sx).T(x + 0.3 * sx, y0 + 1.9, z - 0.2 * facing))
    return out


# ── The building ─────────────────────────────────────────────────────────────
def palace():
    brick, stone, roofs, glass, dark, tiles = [], [], [], [], [], []
    O = ccw(OUTLINE)

    # plinth and the low brick body everywhere (wing height), flat roof inside
    stone.append(extrude(offset(O, 0.15), FOUND, 1.0, jit(STONE), top=False))
    brick += banded(O, 1.0, WING)
    roofs.append(cap(O, WING, jit(FLAT)))
    for a, b, u, n, L in edges(O):
        stone.append(slab(a, u, n, 0, L, WING - 0.9, WING - 0.55, 0.12, 0.0, jit(STONE)))       # tile band edge
        tiles.append(on_wall(rect(0.05, L - 0.05, WING - 1.6, WING - 0.95), a, u, n, TILE_B, 0.04))
        stone.append(slab(a, u, n, 0, L, WING - 0.5, WING, 0.45, 0.0, jit(STONE)))              # cornice
        stone += balustrade(a, u, n, 0, L, WING, 0.9)

    # central block: hall walls and the four pylons rising to 13 m
    C = ccw(CENTRAL)
    brick += banded(C, WING, CEN)
    for P in PYL_S + PYL_N:
        P = ccw(P)
        brick += banded(P, 1.0, CEN)
        stone.append(extrude(offset(P, 0.35), CEN - 0.6, CEN, jit(STONE), bottom=True))
        stone.append(extrude(offset(P, 0.1), CEN, CEN + 1.1, jit(STONE)))
        for a, b, u, n, L in edges(P):
            tiles.append(on_wall(rect(0.2, L - 0.2, CEN - 2.0, CEN - 0.9), a, u, n, TILE_B, 0.04))
    for a, b, u, n, L in edges(C):
        stone.append(slab(a, u, n, 0, L, CEN - 0.6, CEN, 0.45, 0.0, jit(STONE)))
        tiles.append(on_wall(rect(0.05, L - 0.05, CEN - 1.6, CEN - 0.95), a, u, n, TILE_B, 0.04))
        stone += balustrade(a, u, n, 0.3, L - 0.3, CEN, 1.0)
    roofs.append(cap(C, CEN, jit(FLAT)))

    # the glass barrel vault over the central hall, north–south, with its iron ribs and glazed ends
    z0, z1 = -12.4, 17.6
    g, prof = vault_z(0.0, VAULT_HW, z0, z1, CEN + 0.6, 7.9, GLASS_C, 16)
    glass.append(g)
    stone.append(extrude([(-VAULT_HW - 0.3, z0), (VAULT_HW + 0.3, z0), (VAULT_HW + 0.3, z1), (-VAULT_HW - 0.3, z1)],
                         CEN, CEN + 0.6, jit(STONE)))
    pts, tris = triangulate(prof)
    for z, flip in ((z0, True), (z1, False)):
        m = Mesh(np.array([(x, y, z) for x, y in pts]), tris if not flip else tris[:, ::-1], GLASS_C)
        glass.append(m)
    for z in np.linspace(z0, z1, 11):                                     # transverse iron ribs
        rib = [(x * 1.006, y + 0.05, z) for x, y in prof]
        dark.append(tube(rib, 0.09, IRON, 4))
    roofs.append(box(1.6, 0.45, z1 - z0, METAL_C).T(0, CEN + 0.6 + 7.8, (z0 + z1) / 2))   # ridge walkway

    # wings: three metal barrel vaults each, east–west, with glazed strips on top
    for sgn in (-1, 1):
        xa, xb = sorted((sgn * 10.0, sgn * 25.6))
        for zc, hw in WING_VAULTS:
            roofs.append(vault_x(zc, hw, xa, xb, WING, 3.6, jit(METAL_C, 0.02), 10))
            for x in (xa, xb):
                glass.append(end_x(zc, hw, x, WING, 3.6, GLASS_C))
            glass.append(vault_x(zc, hw, xa + 2.2, xb - 2.2, WING, 3.6, GLASS_C, 6, math.pi * 0.3, math.pi * 0.7, 0.05))
        # the flat terrace along the north, with two skylights
        for xs in (0.33, 0.7):
            x = xa + (xb - xa) * xs
            glass.append(box(5.5, 0.6, 4.0, GLASS_C).T(x, WING, -15.2))

    # corner pavilions: taller brick blocks with metal cloister vaults and a skylight
    for P in PAVS:
        P = ccw(P)
        brick += banded(P, 1.0, PAV)
        stone.append(extrude(offset(P, 0.45), PAV - 0.6, PAV, jit(STONE), bottom=True))
        stone.append(extrude(offset(P, 0.1), PAV, PAV + 0.9, jit(STONE), holes=[offset(P, -0.25)]))
        for a, b, u, n, L in edges(P):
            tiles.append(on_wall(rect(0.2, L - 0.2, PAV - 1.9, PAV - 0.85), a, u, n, TILE_B, 0.04))
        cx, cz = np.mean(np.array(P), 0)
        w = max(p[0] for p in P) - min(p[0] for p in P)
        d = max(p[1] for p in P) - min(p[1] for p in P)
        prof = [(1.0, 0.0), (0.93, 1.4), (0.78, 2.7), (0.55, 3.7), (0.3, 4.2)]
        ring = lambda s: [(cx - (w / 2 - 0.6) * s, cz - (d / 2 - 0.6) * s), (cx + (w / 2 - 0.6) * s, cz - (d / 2 - 0.6) * s),
                          (cx + (w / 2 - 0.6) * s, cz + (d / 2 - 0.6) * s), (cx - (w / 2 - 0.6) * s, cz + (d / 2 - 0.6) * s)]
        for (s0, h0), (s1, h1) in zip(prof[:-1], prof[1:]):
            lo, hi = ring(s0), ring(s1)
            v, f = [], []
            for i in range(4):
                j = (i + 1) % 4
                o = len(v)
                v += [(lo[i][0], PAV + h0, lo[i][1]), (lo[j][0], PAV + h0, lo[j][1]), (hi[j][0], PAV + h1, hi[j][1]), (hi[i][0], PAV + h1, hi[i][1])]
                f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
            m = Mesh(v, f, jit(METAL_C, 0.02))
            if np.dot(np.cross(m.v[1] - m.v[0], m.v[2] - m.v[0]), [0, 1, 0]) < 0:
                m.f = m.f[:, ::-1]
            roofs.append(m)
        top = ring(0.3)
        glass.append(cap(top, PAV + 4.2, GLASS_C))
        glass.append(extrude(ring(0.22), PAV + 4.2, PAV + 4.9, GLASS_C))

    # façades: arched windows in the bays of the wings and pavilions, triple-arched porticos
    def openings(P, H, skip=()):
        for ei, (a, b, u, n, L) in enumerate(edges(P)):
            if L < 3.0:
                continue
            mid = a + u * L / 2
            if any(x0 <= mid[0] <= x1 and z0 <= mid[1] <= z1 for x0, x1, z0, z1 in skip):
                continue
            nb = max(1, int(L / 3.2))
            bay = L / nb
            for i in range(nb):
                s = (i + 0.5) * bay
                w = min(1.7, bay * 0.55)
                dark.append(on_wall(arch(s - w / 2, s + w / 2, 1.6, min(H - 3.6, 6.4) - w / 2), a, u, n, WIN))
                stone.append(on_wall(arch(s - w / 2 - 0.35, s + w / 2 + 0.35, 1.5, min(H - 3.6, 6.4) - w / 2), a, u, n, STONE * 1.02, 0.03))
                stone.append(slab(a, u, n, s - w / 2 - 0.2, s + w / 2 + 0.2, 1.3, 1.6, 0.15, 0.0, jit(STONE)))

    skip = [(-10.5, 10.5, -14.0, -11.0), (-10.5, 10.5, 16.0, 21.0)]
    openings(O, WING, skip + [(24.0, 39.0, -12.0, 19.0), (-39.0, -24.0, -12.0, 19.0)])
    for P in PAVS:
        P = ccw(P)
        for a, b, u, n, L in edges(P):
            mid = a + u * L / 2
            if abs(mid[0]) < 26.0 and abs(n[0]) > 0.9:              # inner side, hidden by the wing
                continue
            s = L / 2
            dark.append(on_wall(arch(s - 1.3, s + 1.3, 1.6, 6.8), a, u, n, WIN))
            tiles.append(on_wall(arch(s - 2.6, s + 2.6, 1.4, 6.9), a, u, n, TILE_W, 0.03))
            tiles.append(on_wall(rect(s - 3.0, s + 3.0, 7.0, 10.0), a, u, n, TILE_B, 0.03))

    # the two porticos (south and north): three great arches on columns, tiled spandrels, a stair on the south
    for zf, nz in ((19.02, 1), (-12.78, -1)):
        p0 = np.array((-5.95 if nz > 0 else 5.6, zf)) if nz > 0 else np.array((5.6, zf))
        u = np.array((1.0, 0.0)) if nz > 0 else np.array((-1.0, 0.0))
        n = np.array((0.0, float(nz)))
        L = 11.55
        tiles.append(on_wall(rect(0.0, L, 7.2, 10.3), p0, u, n, TILE_B, 0.03))
        for k in range(3):
            s0, s1 = 0.3 + k * L / 3 + 0.3, (k + 1) * L / 3 - 0.3
            dark.append(on_wall(arch(s0, s1, 1.0, 8.4 - (s1 - s0) / 2), p0, u, n, WIN, 0.06))
            stone.append(on_wall(arch(s0 - 0.35, s1 + 0.35, 1.0, 8.4 - (s1 - s0) / 2), p0, u, n, (0.80, 0.55, 0.45), 0.045))
        for k in range(4):
            q = p0 + u * (0.3 + k * L / 3 - (0.0 if 0 < k < 3 else 0.0)) + n * 0.55
            if 0 < k < 3:
                stone.append(cyl(0.32, 0.28, 6.0, jit(STONE), 10).T(q[0], 1.0, q[1]))
                stone.append(box(0.9, 0.6, 0.9, jit(STONE)).T(q[0], 7.0, q[1]))
        stone.append(slab(p0, u, n, 0.0, L, 10.3, 10.8, 0.2, 0.0, jit(STONE)))
        if nz > 0:                                       # the stair and the two winged lions
            for k in range(6):                           # six steps from the portico (1.2 m) down to the path
                stone.append(box(24.0, 1.2 - 0.2 * k - FOUND, 0.7, jit(PAVE, 0.01)).T(0, FOUND, 20.75 + 0.7 * k))
            for sx in (-1, 1):
                stone += lion(sx * 11.0, 22.0, 1.2, jit(TILE_W), 1)
        else:                                            # the north forecourt with its balustrades
            stone.append(box(20.0, 0.4, 6.0, jit(PAVE, 0.01)).T(0, 0.0, -16.0))

    return brick, stone, roofs, glass, dark, tiles


def main():
    out = ROOT / "models" / "Q3891623"
    out.mkdir(parents=True, exist_ok=True)
    brick, stone, roofs, glass, dark, tiles = palace()
    B, S, R, G, D, T_ = (merge(x) for x in (brick, stone, roofs, glass, dark, tiles))
    tris, size = write_glb(out / "model.glb", [("brick", B), ("stone", S), ("roofs", R, {"material": "zinc"}),
                                               ("glass", G, {"material": "glass"}), ("iron_windows", D),
                                               ("tiles", T_, {"material": "tile"})],
                           roughness=0.9, materials={"glass": GLASS, "zinc": {"metallic": 0.35, "roughness": 0.5},
                                                     "tile": {"metallic": 0.0, "roughness": 0.35}})
    allv = np.vstack([m.v for m in (B, S, R, G, D, T_)])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
