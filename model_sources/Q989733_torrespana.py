"""
Torrespaña («El Pirulí», Q989733), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Emilio Fernández Martínez de Velasco (1940–2014), built 1981–1982 by Dragados and Agroman for RTVE,
opened 7 June 1982. Protected.

Measurements (see meta.json for the sources):
- Height 231–232 m to the tip of the antenna (es/en.wikipedia; 220 m without it); antenna mast about 45 m
  (es.wikipedia); shaft about 120 m with the main platform «at 120 m» (es.wikipedia, tourist guides).
- The head, measured on a straight-on telephoto (Wikimedia Commons «Torrespaña - 2008.jpg») and checked on
  «Torrespaña (Madrid) 01.jpg» and «Torrespana-Tower-Madrid.jpg», scaled so that the tip is at 231 m: a concrete
  bowl from 125 to 129.8 m, the open gallery with the dish antennas (129.8–134.8 m), a recessed band, the dark
  metal-clad drum of the control centre (138–145.3 m, Ø 31.7 m, a strip of windows), a light grey shallow dome,
  four antenna platforms (155.6, 162.1, 168.8, 173.2 m) and a ring at 150.6 m round the core, a cylinder in a lattice
  cage to 184 m and the red-and-white lattice mast to 231 m. Shaft Ø 12 m at the foot, 8.4 m under the bowl.
  OpenStreetMap way 309746187 has the same parts but every height and width about 20–25 % smaller (drum Ø 24 m,
  tip at 200 m): the published heights were preferred, and the photos fix the proportions.
- Plan: round; OSM way 309746187 gives the centre.

Heading: the tower is round; +Z faces south (heading 180), towards the RTVE studios.
Axes (docs/MODEL_SPEC.md): +Y up, origin on the ground at the axis of the shaft.

    python model_sources/Q989733_torrespana.py   -> models/Q989733/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import merge, box, cyl, lathe, tube, write_glb, METAL  # noqa: E402

FOUND = -4.0
CONCRETE = (0.70, 0.69, 0.66)
CONCRETE_D = (0.60, 0.59, 0.56)
DRUM = (0.24, 0.21, 0.20)            # dark brown metal cladding of the control centre
DRUM_LOW = (0.30, 0.27, 0.25)
WINDOW = (0.72, 0.78, 0.80)
DOME = (0.78, 0.78, 0.76)
RAIL = (0.45, 0.20, 0.16)            # red-brown steel of the gallery and railings
DISH = (0.92, 0.92, 0.90)
STEEL = (0.62, 0.63, 0.64)
RED = (0.78, 0.13, 0.12)
WHITE = (0.93, 0.93, 0.92)
H_TIP = 231.0


def ring_of(n, r, y, make, phase=0.0):
    """n copies of make() (built facing +z at the origin) set round a circle of radius r at height y."""
    out = []
    for k in range(n):
        a = phase + 2 * math.pi * k / n
        m = make().R(0, a, 0)
        out.append(m.T(r * math.sin(a), y, r * math.cos(a)))
    return out


def dish(rad):
    """A dish antenna facing out (+z): a shallow white bowl with a dark back."""
    m = lathe([(0.0, -0.1), (rad * 0.6, 0.05), (rad, 0.25), (rad * 0.98, 0.32)], DISH, seg=10, axis="z")
    back = cyl(rad * 0.3, rad * 0.3, 0.5, STEEL, seg=6).R(math.pi / 2, 0, 0).T(0, 0, -0.5)
    return merge([m, back])


def shaft():
    prof = [(6.0, FOUND), (6.0, 0.0), (5.6, 30.0), (5.1, 70.0), (4.6, 105.0), (4.2, 125.0)]
    s = lathe(prof, CONCRETE, seg=32)
    # the bowl under the head
    bowl = lathe([(4.2, 125.0), (5.4, 126.4), (7.2, 128.2), (9.1, 129.8)], CONCRETE_D, seg=32)
    door = box(3.0, 3.2, 0.3, (0.25, 0.26, 0.27)).T(0, 0, 5.92)
    return merge([s, bowl, door])


def head():
    solid, metal, white = [], [], []
    # gallery floor and the open steel gallery with the dish antennas
    solid.append(cyl(9.1, 15.2, 0.6, CONCRETE_D, seg=32).T(0, 129.8, 0))
    solid.append(cyl(12.4, 12.4, 5.0, (0.33, 0.30, 0.28), seg=32, caps=False).T(0, 130.4, 0))     # inner wall
    metal.append(cyl(15.0, 15.0, 1.3, RAIL, seg=32, caps=False).T(0, 130.4, 0))                  # parapet
    for k in range(24):                                                                         # posts
        a = 2 * math.pi * k / 24
        metal.append(box(0.25, 4.4, 0.25, RAIL).T(14.9 * math.sin(a), 130.4, 14.9 * math.cos(a)))
    white += ring_of(14, 15.6, 132.2, lambda: dish(1.25), phase=0.2)
    white += ring_of(9, 15.4, 130.9, lambda: dish(0.8), phase=0.5)
    # recessed band and the drum of the control centre with its strip of windows
    solid.append(cyl(13.2, 13.2, 3.3, DRUM_LOW, seg=32).T(0, 134.7, 0))
    solid.append(cyl(15.3, 15.85, 0.6, DRUM, seg=36).T(0, 138.0, 0))
    solid.append(cyl(15.85, 15.85, 6.7, DRUM, seg=36).T(0, 138.6, 0))
    white += ring_of(40, 15.9, 141.0, lambda: box(1.7, 0.85, 0.08, WINDOW), phase=0.04)
    # the shallow light-grey dome
    solid.append(lathe([(15.4, 145.3), (14.4, 146.6), (12.6, 148.4), (9.6, 149.7), (5.0, 150.3), (0.0, 150.4)],
                       DOME, seg=36))
    # the core and the antenna platforms
    solid.append(cyl(2.8, 2.8, 34.0, (0.42, 0.42, 0.42), seg=16).T(0, 150.0, 0))
    solid.append(cyl(4.6, 4.6, 24.5, (0.30, 0.30, 0.31), seg=16).T(0, 150.4, 0))                 # equipment rooms
    for y, r in ((150.6, 10.5), (155.6, 8.9), (162.1, 8.8), (168.8, 7.6), (173.2, 7.3)):
        solid.append(cyl(r - 1.2, r, 0.8, CONCRETE_D, seg=28).T(0, y - 0.8, 0))
        metal.append(tube([(r * math.sin(a), y + 1.0, r * math.cos(a)) for a in
                           [2 * math.pi * k / 28 for k in range(29)]], 0.06, RAIL, seg=4))
        n = int(r * 1.3)
        white += ring_of(n, r - 1.4, y + 1.2, lambda: dish(0.75), phase=y)
        white += ring_of(max(4, n // 2), r - 1.6, y + 2.8, lambda: dish(1.05), phase=y * 0.7)
    # the cylinder in its lattice cage, the red cap
    solid.append(cyl(3.2, 3.2, 9.6, (0.46, 0.46, 0.47), seg=16).T(0, 174.4, 0))
    for k in range(8):
        a = 2 * math.pi * k / 8
        metal.append(tube([(4.4 * math.sin(a), 174.0, 4.4 * math.cos(a)), (4.4 * math.sin(a), 184.0, 4.4 * math.cos(a))],
                          0.08, STEEL, seg=4))
    for y in (176.5, 179.5, 182.5):
        metal.append(tube([(4.4 * math.sin(a), y, 4.4 * math.cos(a)) for a in
                           [2 * math.pi * k / 16 for k in range(17)]], 0.06, STEEL, seg=4))
    metal.append(cyl(3.3, 2.6, 1.2, RED, seg=16).T(0, 184.0, 0))
    return merge(solid), merge(metal), merge(white)


def mast():
    """The lattice mast, 185–231 m: four legs tapering, X bracing, painted in red and white sections; a solid red
    section (the antenna panels) 209.6–219 m and a thin top whip."""
    ms = []
    y0, y1 = 185.2, 209.6
    w0, w1 = 1.5, 0.8
    n = 12
    for i in range(n):
        ya, yb = y0 + (y1 - y0) * i / n, y0 + (y1 - y0) * (i + 1) / n
        wa, wb = w0 + (w1 - w0) * i / n, w0 + (w1 - w0) * (i + 1) / n
        c = RED if (i // 2) % 2 == 0 else WHITE
        P = lambda sx, sz, y, w: (sx * w, y, sz * w)  # noqa: E731
        corners = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        for k, (sx, sz) in enumerate(corners):
            tx, tz = corners[(k + 1) % 4]
            ms.append(tube([P(sx, sz, ya, wa), P(sx, sz, yb, wb)], 0.18, c, seg=4))
            ms.append(tube([P(sx, sz, ya, wa), P(tx, tz, yb, wb)], 0.05, STEEL, seg=3))
        # side antennas sticking out of the lattice (thin horizontal bars)
        if i % 2 == 1:
            for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ms.append(tube([(sx * wb, yb, sz * wb), (sx * (wb + 1.6), yb, sz * (wb + 1.6))], 0.05, STEEL, seg=3))
    ms.append(cyl(0.55, 0.4, y1 - y0, (0.70, 0.70, 0.70), seg=6).T(0, y0, 0))     # feeders inside the lattice
    ms.append(box(1.6, 9.4, 1.6, RED).T(0, 209.6, 0))
    for k, (ya, yb) in enumerate(((219.0, 222.0), (222.0, 225.0), (225.0, 228.0), (228.0, H_TIP))):
        ms.append(cyl(0.45 - 0.07 * k, 0.4 - 0.07 * k, yb - ya, WHITE if k % 2 else RED, seg=8).T(0, ya, 0))
    return merge(ms)


def main():
    out = ROOT / "models" / "Q989733"
    out.mkdir(parents=True, exist_ok=True)
    s = shaft()
    h, m, w = head()
    a = mast()
    tris, size = write_glb(out / "model.glb", [("shaft", s), ("head", h), ("steel", merge([m, a]), {"material": "metal"}),
                                                ("antennas", w)],
                           materials={"metal": METAL}, roughness=0.85)
    allm = merge([s, h, m, w, a])
    lo, hi = allm.v.min(0), allm.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
