"""
Puerta de Europa — Torres KIO (Q1141262), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Philip Johnson (1906–2007) and John Burgee (born 1933), 1989–1996; structure by Leslie E.
Robertson Associates. Protected.

One Wikidata item (and one Wikipedia article) for the pair, so one model with both towers, the Paseo de la
Castellana running between them.

Measurements (see meta.json for the sources):
- OpenStreetMap ways 11103227 (east tower, «Puerta de Europa Este») and 965429323 (west tower, «Oeste»), both
  tagged wikidata=Q1141262, height 114.7 m, 27 levels, with one building:part per floor that shows each floor
  square shifting towards the Castellana. Ground floor squares 35.6 × 36.0 m; their sides run at 101.5°/191.5°,
  the bases are 144 m apart.
- Height 114 m (Wikidata; es.wikipedia 113.8 m), 26 office floors; each tower leans 14.3° towards the other
  (es.wikipedia; en.wikipedia rounds to 15°), so the roof overhangs the base by 29 m.
- Façades (photos on Wikimedia Commons: «Puerta de Europa.001 - Madrid.JPG», «Torres Kio.JPG», «Madrid - Torres
  KIO (35260741083).jpg», «Torres KIO (2008).JPG»): black-blue glass with a fine silver grid; the two vertical
  façades (north and south, parallelograms) carry a central stainless-steel band and three horizontal bands
  (four tiers); the two leaning façades carry the perimeter bracing as a great stainless X crossing at a third
  of the height, with horizontal bands at one and two thirds; thin red lines cross every panel; stainless
  corner columns. A 20 × 20 m heliport on each roof (es.wikipedia), outlined in red on the east tower and in
  blue on the west tower (en.wikipedia).

Axes (docs/MODEL_SPEC.md): +Y up, +Z faces 191.5° (south-south-west, down the Castellana towards the Plaza de
Castilla: the view of the "gate" everyone knows), +X points 101.5° (towards the east tower). Origin on the
ground halfway between the two bases, on the axis of the Castellana.

    python model_sources/Q1141262_puerta_de_europa.py   -> models/Q1141262/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, write_glb, GLASS, METAL  # noqa: E402

GLASS_C = (0.09, 0.11, 0.16)          # black-blue glass
GRID = (0.29, 0.32, 0.37)             # the fine silver grid, seen from afar
STEEL = (0.80, 0.81, 0.82)            # stainless steel bands and bracing
RED = (0.80, 0.10, 0.10)
BLUE = (0.15, 0.35, 0.80)
ROOF = (0.38, 0.38, 0.39)
PAD = (0.28, 0.29, 0.30)
WHITE = (0.92, 0.92, 0.90)
FOUND = -3.0

H = 114.0                             # roof
LEAN = math.radians(14.3)
A, Bz = 17.8, 18.0                    # half sizes of the floor square: across (x) and along (z) the Castellana
TOWERS = [  # base centre (x, z) from the OSM ground-floor parts, lean direction, heliport colour
    ("east", (89.95, 0.46), -1, RED),
    ("west", (-89.94, -0.55), +1, BLUE),
]


# ── Pieces in the tower's own, upright frame (sheared afterwards) ────────────
def prism_z(poly, z0, z1, c):
    """Prism from a convex polygon in (x, y), between z0 and z1 (pieces on the ±Z façades)."""
    P = list(poly); n = len(P)
    v, f = [], []
    for z in (z0, z1):
        v += [(x, y, z) for x, y in P]
    f += [(0, i + 1, i) for i in range(1, n - 1)]                 # back (−z)
    f += [(n, n + i, n + i + 1) for i in range(1, n - 1)]         # front (+z)
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i)]
    m = Mesh(v, f, (0, 0, 0))
    return m


def orient_out(m):
    """Flip faces whose normal points towards the piece's centre (keeps the prism helpers simple)."""
    c = m.v.mean(0)
    a, b, d = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    n = np.cross(b - a, d - a)
    bad = np.einsum("ij,ij->i", n, (a + b + d) / 3 - c) < 0
    m.f[bad] = m.f[bad][:, ::-1]
    return m


def piece_z(poly, side, d0, d1, c):
    """A flat piece on the ±Z façade (z = ±Bz), from d0 to d1 out of the glass."""
    d0 = d0 - 0.02 if d0 == 0 else d0                             # sunk a little into the glass: no coplanar faces
    z0, z1 = (Bz + d0, Bz + d1) if side > 0 else (-Bz - d1, -Bz - d0)
    m = orient_out(prism_z(poly, z0, z1, c)); m.c[:] = c
    return m


def piece_x(poly, side, d0, d1, c):
    """A flat piece on the ±X façade (x = ±A), poly in (z, y)."""
    d0 = d0 - 0.02 if d0 == 0 else d0
    m = prism_z(poly, 0, 1, c)
    zz, yy, tt = m.v[:, 0].copy(), m.v[:, 1].copy(), m.v[:, 2].copy()
    x = (A + d0 + (d1 - d0) * tt) * side
    m.v = np.stack([x, yy, zz], 1)
    m = orient_out(m); m.c[:] = c
    return m


def rect(u0, u1, y0, y1):
    return [(u0, y0), (u1, y0), (u1, y1), (u0, y1)]


def strip(p0, p1, w):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-t[1], t[0]]) * w / 2
    return [tuple(p0 - n), tuple(p1 - n), tuple(p1 + n), tuple(p0 + n)]


def tower(heli_c):
    glass, grid, steel, red, misc = [], [], [], [], []
    glass.append(box(2 * A, H - FOUND, 2 * Bz, GLASS_C).T(0, FOUND, 0))
    # ── the two vertical façades (±Z): central band, three horizontal bands, red cross in every panel
    tiers = [0.0, H * 0.25, H * 0.5, H * 0.75, H]
    for s in (1, -1):
        for x in np.arange(-A + 1.5, A - 0.5, 1.5):                 # fine grid
            grid.append(piece_z(rect(x - 0.05, x + 0.05, 0.0, H), s, 0.0, 0.03, GRID))
        for y in np.arange(1.4, H - 0.5, 1.4):
            grid.append(piece_z(rect(-A, A, y - 0.05, y + 0.05), s, 0.0, 0.03, GRID))
        for x in (-A / 2, A / 2):                                    # red lines
            red.append(piece_z(rect(x - 0.12, x + 0.12, 0.0, H), s, 0.0, 0.07, RED))
        for k in range(4):
            y = (tiers[k] + tiers[k + 1]) / 2
            red.append(piece_z(rect(-A, A, y - 0.12, y + 0.12), s, 0.0, 0.07, RED))
        steel.append(piece_z(rect(-0.7, 0.7, 0.0, H), s, 0.0, 0.34, STEEL))
        for y in tiers[1:4]:
            steel.append(piece_z(rect(-A, A, y - 0.65, y + 0.65), s, 0.0, 0.28, STEEL))
    # ── the two leaning façades (±X): the great X, bands at a third and two thirds
    yc = H / 3
    for s in (1, -1):
        for z in np.arange(-Bz + 1.5, Bz - 0.5, 1.5):
            grid.append(piece_x(rect(z - 0.05, z + 0.05, 0.0, H), s, 0.0, 0.03, GRID))
        for y in np.arange(1.4, H - 0.5, 1.4):
            grid.append(piece_x(rect(-Bz, Bz, y - 0.05, y + 0.05), s, 0.0, 0.03, GRID))
        red.append(piece_x(rect(-0.12, 0.12, 0.0, H), s, 0.0, 0.07, RED))
        for y0, y1 in ((0, yc), (yc, 2 * yc), (2 * yc, H)):
            y = (y0 + y1) / 2
            red.append(piece_x(rect(-Bz, Bz, y - 0.12, y + 0.12), s, 0.0, 0.07, RED))
        for y in (yc, 2 * yc):
            steel.append(piece_x(rect(-Bz, Bz, y - 0.65, y + 0.65), s, 0.0, 0.28, STEEL))
        zc = Bz - 0.9
        for sz in (1, -1):
            steel.append(piece_x(strip((sz * zc, H - 0.6), (0.0, yc), 1.5), s, 0.0, 0.4, STEEL))
            steel.append(piece_x(strip((0.0, yc), (sz * zc, 0.6), 1.5), s, 0.0, 0.4, STEEL))
    # ── stainless corner columns (an L on each vertical edge) and the crown frame
    for sx in (1, -1):
        for sz in (1, -1):
            x0, x1 = sorted((sx * (A - 1.1), sx * (A + 0.46)))
            z0, z1 = sorted((sz * (Bz - 1.1), sz * (Bz + 0.46)))
            steel.append(box(x1 - x0, H + 0.9 - FOUND, z1 - z0, STEEL).T((x0 + x1) / 2, FOUND, (z0 + z1) / 2))
    for s in (1, -1):
        steel.append(piece_z(rect(-A, A, H - 1.3, H + 0.9), s, 0.0, 0.42, STEEL))
        steel.append(piece_x(rect(-Bz, Bz, H - 1.3, H + 0.9), s, 0.0, 0.42, STEEL))
    # ── roof: plant room and the 20 × 20 m heliport with its coloured outline
    misc.append(box(2 * A - 0.2, 0.6, 2 * Bz - 0.2, ROOF).T(0, H - 0.05, 0))
    misc.append(box(14.0, 2.4, 14.0, ROOF).T(0, H + 0.5, 0))
    misc.append(box(20.0, 0.35, 20.0, PAD).T(0, H + 2.9, 0))
    yp = H + 3.25
    for x0, x1, z0, z1 in ((-9.6, 9.6, 8.8, 9.6), (-9.6, 9.6, -9.6, -8.8), (8.8, 9.6, -8.8, 8.8), (-9.6, -8.8, -8.8, 8.8)):
        misc.append(box(x1 - x0, 0.05, z1 - z0, heli_c).T((x0 + x1) / 2, yp, (z0 + z1) / 2))
    for x0, x1, z0, z1 in ((-2.1, -1.3, -3.0, 3.0), (1.3, 2.1, -3.0, 3.0), (-1.3, 1.3, -0.4, 0.4)):
        misc.append(box(x1 - x0, 0.05, z1 - z0, WHITE).T((x0 + x1) / 2, yp, (z0 + z1) / 2))
    for sx, sz in ((1, 1), (-1, -1)):                                 # two masts on the corners
        misc.append(cyl(0.12, 0.07, 6.0, STEEL, 6).T(sx * (A - 0.6), H + 0.9, sz * (Bz - 0.6)))
    return merge(glass), merge(grid), merge(steel), merge(red), merge(misc)


def shear(m, s, cx, cz):
    """Lean the tower: x moves s·tan(lean)·y above the ground (floors stay level), then place it."""
    v = m.v.copy()
    v[:, 0] += s * math.tan(LEAN) * np.maximum(v[:, 1], 0.0)
    v[:, 0] += cx; v[:, 2] += cz
    m.v = v
    return m


def footprint():
    """Two rectangles (each tower seen from above, + 1 m) joined by a thin slit along the Castellana's axis,
    so the street and the Plaza de Castilla between the towers are not cleared (even-odd polygon)."""
    sh = math.tan(LEAN) * (H + 0.9)
    (ex, ez), (wx, wz) = TOWERS[0][1], TOWERS[1][1]
    m = 0.46 + 1.0
    e_in, e_out = ex - A - sh - m, ex + A + m
    w_in, w_out = wx + A + sh + m, wx - A - m
    ez0, ez1 = ez - Bz - m, ez + Bz + m
    wz0, wz1 = wz - Bz - m, wz + Bz + m
    k = 0.05
    ring = [(e_in, -k), (e_in, ez0), (e_out, ez0), (e_out, ez1), (e_in, ez1), (e_in, k),
            (w_in, k), (w_in, wz1), (w_out, wz1), (w_out, wz0), (w_in, wz0), (w_in, -k)]
    return [[round(x, 2), round(z, 2)] for x, z in ring]


def main():
    out = ROOT / "models" / "Q1141262"
    out.mkdir(parents=True, exist_ok=True)
    G, R, S, X = [], [], [], []
    for name, (cx, cz), s, heli_c in TOWERS:
        glass, grid, steel, red, misc = tower(heli_c)
        G.append(shear(glass, s, cx, cz))
        S.append(shear(steel, s, cx, cz))
        X += [shear(grid, s, cx, cz), shear(red, s, cx, cz), shear(misc, s, cx, cz)]
    glass, steel, rest = merge(G), merge(S), merge(X)
    parts = [("glass", glass, {"material": "glass"}), ("stainless", steel, {"material": "steel"}), ("lines_roofs", rest)]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.8, materials={"glass": GLASS, "steel": METAL})
    allv = np.vstack([glass.v, steel.v, rest.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    print("footprint:", json.dumps(footprint()))


if __name__ == "__main__":
    main()
