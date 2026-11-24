"""
Torres de Colón (Q2919969), Plaza de Colón, Madrid — built by code, as they are after the 2020–2024 reform.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Antonio Lamela (1926–2017), 1967–1976, a hung structure by Carlos Fernández Casado, Leonardo
Fernández Troyano and Javier Manterola; crown («el enchufe») by Carlos Lamela, 1990s; rebuilt 2020–2024 by Luis Vidal
+ Architects, who replaced the green crown with a double prismatic volume of four floors. Protected.

Measurements (see meta.json for the sources):
- OpenStreetMap way 336755231 and its building:parts: two towers, each a rectangle of 17.5 × 19.4 m (one turned 90°
  to the other), set 7 m apart and 9.6 m out of line; their outlines zigzag every 1.8 m where the hangers stand
  proud of the glass (18 hangers per tower, es.wikipedia); a concrete core of 8 × 8 m in each; the low podium covers
  the whole plot. OSM still maps the 1990s crowns (107–125 m), not the present top.
- Heights: 116–117 m to the top of the new crown (es.wikipedia, press on the reform: the height was kept). The rest
  measured on photos (Wikimedia Commons «Torres de Colon - Plaza de Colon 2 - Madrid 01.jpg», «Torres de Colón -
  Madrid - 001/013/015.jpg», 2025–2026), ±5 %: podium 13 m (recessed ground floor and a band of green glass), the
  hung glass shafts from 14 m to the head beams at 86 m (the structural height of 86 m, El Mundo 2024), the head
  slabs with their coffered soffit and a frosted-glass parapet to 92.5 m, a dark recessed floor, then the two
  glass cubes of four floors (96.5–116.5 m) joined by a lower glass link.
- Colours: light blue-green glass on the shafts, the hangers in light concrete, dark blue reflective glass on the
  cubes (photos).

Heading: the main façade (+Z) is the south-south-west one (197.5°), on the Calle de Génova (the address, Génova 31),
where the two towers stand side by side; +X points east-south-east (107.5°), to the Paseo de la Castellana and the
Plaza de Colón. Origin = centroid of the OSM outline (the podium).

    python model_sources/Q2919969_torres_de_colon.py   -> models/Q2919969/model.glb
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import merge, box, extrude, cap, walls, offset, quad, write_glb, GLASS  # noqa: E402

FOUND = -3.0
SHAFT_GLASS = (0.47, 0.62, 0.64)       # light blue-green glass of the hung floors
SHAFT_BAND = (0.36, 0.45, 0.48)        # floor edges
HANGER = (0.80, 0.77, 0.70)            # the concrete hangers
SOFFIT = (0.56, 0.55, 0.52)            # coffered underside of the head slabs
PARAPET = (0.80, 0.86, 0.86)           # frosted glass parapet of the terraces
DARK = (0.16, 0.17, 0.19)              # recessed plant floor under the cubes, the tower bottoms
CUBE_GLASS = (0.24, 0.31, 0.42)        # dark blue reflective glass of the crown cubes
CUBE_LINE = (0.15, 0.18, 0.23)
LINK_GLASS = (0.18, 0.24, 0.31)
POD_GLASS = (0.27, 0.40, 0.38)         # podium: band of green glass
POD_LOW = (0.20, 0.24, 0.26)           # recessed ground floor
ROOF = (0.42, 0.43, 0.44)

# Plan in model x/z (osm.py way 336755231 --heading 197.5, the zigzag averaged)
TOWERS = [  # (x0, x1, z0, z1): A = the west tower (in front), B = the east tower (set back 9.6 m)
    (-19.2, 0.2, 1.2, 18.7),
    (7.3, 24.8, -8.2, 11.2),
]
SITE = [(4.6, 12.5), (4.6, 19.2), (-27.8, 19.2), (-24.1, 6.2), (-21.2, -3.6), (-21.1, -4.0), (-18.1, -3.2),
        (-13.0, -21.5), (13.4, -17.3), (18.9, -16.4), (25.0, -15.5), (25.0, 11.2)]
Y_POD, Y_SHAFT0, Y_HEAD = 13.0, 14.0, 86.0
Y_DECK, Y_PARAPET = 89.5, 92.5
Y_CUBE0, Y_TOP = 96.5, 116.5
INSET = 0.3                            # glass plane behind the line of the hangers


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def rect(x0, x1, z0, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def floor_lines(x0, x1, z0, z1, ys, h, c, eps=0.03):
    """Thin horizontal bands (floor edges) on the four faces of a box."""
    out = []
    w, d = x1 - x0, z1 - z0
    for y in ys:
        out += [quad((x0 + x1) / 2, y, z1, w, h, "+z", c, eps), quad((x0 + x1) / 2, y, z0, w, h, "-z", c, eps),
                quad(x1, y, (z0 + z1) / 2, d, h, "+x", c, eps), quad(x0, y, (z0 + z1) / 2, d, h, "-x", c, eps)]
    return out


def hanger_positions(L):
    """18 hangers per tower: 5 on the long faces (19.4 m), 4 on the short ones (17.5 m), 3.6 m apart."""
    n = 5 if L > 18.5 else 4
    return [(k - (n - 1) / 2) * 3.6 for k in range(n)]


def tower(x0, x1, z0, z1):
    glass, solid, cube = [], [], []
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    # the hung glass shaft
    g = (x0 + INSET, x1 - INSET, z0 + INSET, z1 - INSET)
    glass.append(B(g[0], g[1], Y_SHAFT0 + 1.2, Y_HEAD, g[2], g[3], SHAFT_GLASS))
    ys = [Y_SHAFT0 + 1.2 + 3.42 * k for k in range(1, 21)]
    glass += floor_lines(*g, [y + 0.25 for y in ys], 0.5, SHAFT_BAND)
    solid.append(B(g[0], g[1], Y_SHAFT0, Y_SHAFT0 + 1.2, g[2], g[3], DARK))          # bottom slab of the hung block
    # the hangers, standing proud of the glass, from the bottom slab up to the head beam
    for a in hanger_positions(x1 - x0):
        for z in (z0, z1):
            solid.append(B(cx + a - 0.35, cx + a + 0.35, Y_SHAFT0, Y_HEAD, z - 0.35 if z == z0 else z - 0.35,
                           z + 0.35, HANGER))
    for a in hanger_positions(z1 - z0):
        for x in (x0, x1):
            solid.append(B(x - 0.35, x + 0.35, Y_SHAFT0, Y_HEAD, cz + a - 0.35, cz + a + 0.35, HANGER))
    # the concrete core, seen under the hung block (8 × 8 m), down into the podium
    solid.append(B(cx - 4.0, cx + 4.0, FOUND, Y_SHAFT0, cz - 4.0, cz + 4.0, (0.62, 0.61, 0.58)))
    # the head: slab 2.5 m out all round, coffered underside, terrace with a frosted parapet
    deck = rect(x0 - 2.5, x1 + 2.5, z0 - 2.5, z1 + 2.5)
    solid.append(extrude(deck, Y_HEAD, Y_DECK, SOFFIT, top=True, bottom=True, top_c=ROOF))
    for k in range(1, 6):                                                    # beams of the coffers (both ways)
        t = k / 6
        xx = x0 - 2.5 + (x1 - x0 + 5) * t
        zz = z0 - 2.5 + (z1 - z0 + 5) * t
        solid.append(B(xx - 0.25, xx + 0.25, Y_HEAD - 0.6, Y_HEAD, z0 - 2.4, z1 + 2.4, (0.48, 0.47, 0.45)))
        solid.append(B(x0 - 2.4, x1 + 2.4, Y_HEAD - 0.6, Y_HEAD, zz - 0.25, zz + 0.25, (0.48, 0.47, 0.45)))
    par = offset(deck, 0.15)
    glass.append(walls(par, Y_DECK - 1.2, Y_PARAPET, PARAPET))
    glass.append(walls(offset(deck, -0.05), Y_DECK, Y_PARAPET, PARAPET, hole=True))
    glass.append(cap(par, Y_PARAPET, PARAPET, holes=[offset(deck, -0.05)]))
    # the dark recessed floor under the cube, then the four-storey glass cube, a little wider than the shaft
    solid.append(B(x0 + 2.0, x1 - 2.0, Y_DECK, Y_CUBE0 + 0.5, z0 + 2.0, z1 - 2.0, DARK))
    c0 = (x0 - 0.8, x1 + 0.8, z0 - 0.8, z1 + 0.8)
    solid.append(B(c0[0], c0[1], Y_CUBE0, Y_CUBE0 + 0.9, c0[2], c0[3], DARK))               # dark soffit band
    cube.append(B(c0[0], c0[1], Y_CUBE0 + 0.9, Y_TOP, c0[2], c0[3], CUBE_GLASS))
    cube += floor_lines(*c0, [Y_CUBE0 + 0.9 + 2.4 * k for k in range(1, 8)], 0.18, CUBE_LINE)
    # vertical mullions of the cube every 1.5 m
    for face, (a0, a1) in (("z", (c0[0], c0[1])), ("x", (c0[2], c0[3]))):
        n = int((a1 - a0) / 1.5)
        for k in range(1, n):
            a = a0 + (a1 - a0) * k / n
            h = Y_TOP - Y_CUBE0 - 0.9
            yc = (Y_CUBE0 + 0.9 + Y_TOP) / 2
            if face == "z":
                cube += [quad(a, yc, c0[3], 0.12, h, "+z", CUBE_LINE), quad(a, yc, c0[2], 0.12, h, "-z", CUBE_LINE)]
            else:
                cube += [quad(c0[1], yc, a, 0.12, h, "+x", CUBE_LINE), quad(c0[0], yc, a, 0.12, h, "-x", CUBE_LINE)]
    solid.append(B(c0[0] + 0.3, c0[1] - 0.3, Y_TOP, Y_TOP + 0.6, c0[2] + 0.3, c0[3] - 0.3, ROOF))     # roof parapet
    solid.append(B(cx - 3.0, cx + 3.0, Y_TOP + 0.6, Y_TOP + 2.4, cz - 2.0, cz + 2.0, (0.55, 0.56, 0.57)))  # plant
    return glass, solid, cube


def building():
    glass, solid, cube = [], [], []
    for t in TOWERS:
        g, s, c = tower(*t)
        glass += g; solid += s; cube += c
    # the glass link between the towers (stairs and lifts of the 1990s, rebuilt), set back from both faces
    lz0, lz1 = 1.8, 10.6
    xa, xb = TOWERS[0][1], TOWERS[1][0]
    glass.append(B(xa - 0.5, xb + 0.5, Y_POD, Y_HEAD, lz0, lz1, LINK_GLASS))
    glass += floor_lines(xa - 0.5, xb + 0.5, lz0, lz1, [Y_SHAFT0 + 1.2 + 3.42 * k + 0.25 for k in range(1, 21)],
                         0.4, CUBE_LINE)
    # the link between the cubes: a narrower glass box, a floor lower than the cubes
    cube.append(B(xa - 0.5, xb + 0.5, Y_HEAD, Y_TOP - 3.0, lz0 + 0.4, lz1 - 0.4, LINK_GLASS))
    cube += floor_lines(xa - 0.5, xb + 0.5, lz0 + 0.4, lz1 - 0.4, [Y_CUBE0 + 0.9 + 2.4 * k for k in range(0, 7)],
                        0.18, CUBE_LINE)
    # the podium over the whole plot: recessed ground floor, then a band of green glass
    site_in = offset(SITE, -1.5)
    solid.append(extrude(site_in, FOUND, 5.5, POD_LOW, top=False))
    solid.append(cap(SITE, 5.5, DARK, up=False))
    glass.append(extrude(SITE, 5.5, Y_POD, POD_GLASS, top=False))
    glass += [walls(SITE, 8.9, 9.2, CUBE_LINE)]
    solid.append(cap(SITE, Y_POD, ROOF))
    solid.append(walls(offset(SITE, 0.05), Y_POD - 0.6, Y_POD + 0.9, (0.55, 0.57, 0.57)))
    return merge(glass), merge(solid), merge(cube)


def main():
    out = ROOT / "models" / "Q2919969"
    out.mkdir(parents=True, exist_ok=True)
    g, s, c = building()
    tris, size = write_glb(out / "model.glb", [("glass", g, {"material": "glass"}), ("structure", s),
                                                ("crown_glass", c, {"material": "glass"})],
                           materials={"glass": GLASS}, roughness=0.85)
    m = merge([g, s, c])
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
