"""
Caleido (Torre Caleido, «la quinta torre», Q22970105), Cuatro Torres Business Area, Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Fenwick Iribarren Architects with Serrano-Suñer Arquitectura, 2017–2021 (opened 19 October 2021),
for Grupo Villar Mir and Megaworld; IE University. Protected.

Measurements (see meta.json for the sources):
- OpenStreetMap way 762323937 and its building:parts: the whole complex is an inverted T — a podium 69.6 m wide and
  256 m long (north–south, 4 floors, 20 m) with two lower parts of 15 m inside it, and across its middle the tower,
  69.4 × 16.9 m (1 : 4 proportion, Fenwick Iribarren: «a span of 17 m»), 181 m high, 36 floors. On both broad faces
  a smooth glass block stands out 5 m (with a 1.2 m step at the face) over the middle 52 m of the face.
- The blocks: OSM maps them from 40 to 150 m; on photos (Wikimedia Commons «Madrid - CTBA, Torre Caleido 01.jpg»,
  «Torre Caleido - CTBA - Madrid 01/02/03.jpg») their top is at about 132 m (±5 %), which is used here.
- Façade: black glass (a double-skin curtain wall); the main body is ribbed by thin vertical fins about 1.15 m apart
  that rise above the roof as a serrated edge; the blocks are smooth, with a fine grid. The IE logo is left out.

Heading: like the Torre Moeve next to it, the main façade (+Z) is the south broad face (191.6°); +X points east
(101.6°). Origin = centroid of the OSM outline of the whole complex (podium and tower).

    python model_sources/Q22970105_torre_caleido.py   -> models/Q22970105/model.glb
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import merge, box, extrude, cap, walls, quad, write_glb, GLASS  # noqa: E402

FOUND = -4.0
BLACK = (0.15, 0.16, 0.19)          # black glass of the main body
FIN = (0.29, 0.30, 0.33)            # the vertical fins
BLOCK = (0.21, 0.24, 0.29)          # the smooth glass blocks, a little more reflective
BLOCK_LINE = (0.13, 0.14, 0.16)
ROOF = (0.20, 0.21, 0.22)
POD_GLASS = (0.26, 0.30, 0.33)
POD_WHITE = (0.86, 0.86, 0.84)
POD_BROWN = (0.40, 0.30, 0.24)      # the brown cladding of the podium
GREEN = (0.42, 0.53, 0.31)          # gardens on the podium
GREEN_L = (0.50, 0.58, 0.36)
PATH = (0.72, 0.71, 0.67)

# Plan in model x/z (osm.py way 762323937 --heading 191.6)
TX, TZ0, TZ1 = 34.7, -2.2, 14.65    # tower: |x| < TX, TZ0 < z < TZ1
H_TOP, FIN_UP = 181.0, 1.6
BX, BX_IN, BD, STEP = 26.1, 24.7, 5.1, 1.2    # blocks: half width, half width of the step, depth, step depth
B_Y0, B_Y1 = 40.0, 132.0
POD_S = [(-34.54, 96.95), (-34.56, 14.62), (34.56, 14.66), (34.4, 92.88), (34.33, 128.27), (-34.78, 128.24)]
POD_N = [(34.62, -112.51), (34.85, -128.1), (-34.26, -128.13), (-34.85, -14.06), (-34.84, -2.09), (34.79, -2.3)]
LOW_S = [(-12.34, 88.71), (-12.18, 18.23), (10.28, 18.25), (10.13, 88.72)]          # the 15 m parts
LOW_N = [(-11.89, -38.89), (-11.72, -109.37), (10.74, -109.35), (10.58, -38.88)]
H_POD, H_LOW = 20.0, 15.0


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def tower():
    glass, fins = [], []
    glass.append(B(-TX, TX, FOUND, H_TOP, TZ0, TZ1, BLACK))
    fins.append(B(-TX + 0.4, TX - 0.4, H_TOP, H_TOP + 1.0, TZ0 + 0.4, TZ1 - 0.4, ROOF))          # plant floor
    # fins on the broad faces (north z = TZ0, south z = TZ1); behind the blocks only above and below them
    n = round(2 * TX / 1.15)
    for k in range(n + 1):
        x = -TX + 0.06 + (2 * TX - 0.12) * k / n
        spans = [(0.0, H_TOP + FIN_UP)] if abs(x) > BX else [(0.0, B_Y0), (B_Y1, H_TOP + FIN_UP)]
        for y0, y1 in spans:
            fins.append(B(x - 0.06, x + 0.06, y0, y1, TZ1, TZ1 + 0.45, FIN))
            fins.append(B(x - 0.06, x + 0.06, y0, y1, TZ0 - 0.45, TZ0, FIN))
    # fins on the narrow ends (x = ±TX)
    m = round((TZ1 - TZ0) / 1.15)
    for k in range(m + 1):
        z = TZ0 + 0.06 + (TZ1 - TZ0 - 0.12) * k / m
        for sx in (-1, 1):
            x0 = TX if sx > 0 else -TX - 0.45
            fins.append(B(x0, x0 + 0.45, 0.0, H_TOP + FIN_UP, z - 0.06, z + 0.06, FIN))
    return merge(glass), merge(fins)


def blocks():
    out = []
    for sz, zf in ((1, TZ1), (-1, TZ0)):
        z_step = zf + sz * STEP
        z_out = zf + sz * BD
        za, zb = sorted((zf, z_step))
        out.append(B(-BX_IN, BX_IN, B_Y0, B_Y1, za, zb, BLOCK))
        za, zb = sorted((z_step, z_out))
        out.append(B(-BX, BX, B_Y0, B_Y1, za, zb, BLOCK))
        # fine grid: floor lines every 3.83 m (24 floors) and mullions every 1.5 m on the outer face
        nrm = "+z" if sz > 0 else "-z"
        for k in range(1, 24):
            y = B_Y0 + (B_Y1 - B_Y0) * k / 24
            out.append(quad(0.0, y, z_out, 2 * BX, 0.12, nrm, BLOCK_LINE, eps=0.03))
        nm = round(2 * BX / 1.5)
        for k in range(1, nm):
            x = -BX + 2 * BX * k / nm
            out.append(quad(x, (B_Y0 + B_Y1) / 2, z_out, 0.08, B_Y1 - B_Y0, nrm, BLOCK_LINE, eps=0.03))
    return merge(out)


def podium():
    solid, glass = [], []
    for outer, low in ((POD_S, LOW_S), (POD_N, LOW_N)):
        # ground floor glazed, then three floors: brown cladding and white bands with windows
        glass.append(walls(outer, FOUND, 5.0, POD_GLASS))
        for y0 in (5.0, 10.0, 15.0):
            solid.append(walls(outer, y0, y0 + 1.4, POD_WHITE))
            glass.append(walls(outer, y0 + 1.4, y0 + 4.0, POD_GLASS))
            solid.append(walls(outer, y0 + 4.0, y0 + 5.0, POD_BROWN))
        solid.append(walls(outer, H_POD, H_POD + 1.1, POD_WHITE))                             # garden parapet
        # roof: gardens, with the lower part (15 m) as a sunken garden court
        solid.append(cap(outer, H_POD, GREEN, holes=[low]))
        solid.append(walls(low, H_LOW, H_POD, POD_WHITE, hole=True))
        solid.append(cap(low, H_LOW, GREEN_L))
    # paths along the gardens (light paving), on both halves
    for z0, z1 in ((20.0, 125.0), (-125.0, -5.0)):
        for x in (-23.0, 22.0):
            solid.append(B(x - 1.5, x + 1.5, H_POD, H_POD + 0.08, z0, z1, PATH))
    return merge(solid), merge(glass)


def main():
    out = ROOT / "models" / "Q22970105"
    out.mkdir(parents=True, exist_ok=True)
    t, f = tower()
    b = blocks()
    ps, pg = podium()
    tris, size = write_glb(out / "model.glb", [("tower_glass", t, {"material": "glass"}), ("fins", f),
                                                ("blocks", b, {"material": "glass"}),
                                                ("podium", ps), ("podium_glass", pg, {"material": "glass"})],
                           materials={"glass": GLASS}, roughness=0.85)
    m = merge([t, f, b, ps, pg])
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
