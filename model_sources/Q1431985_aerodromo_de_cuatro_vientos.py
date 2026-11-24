"""
Aeródromo de Cuatro Vientos (Q1431985), Madrid — its two historic buildings, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The aerodrome (1911, the oldest in Spain) has no single building with its own Wikidata item, so the model
carries the two historic pieces that identify it, at their real places (1.2 km apart), and nothing else:
1. The hangar of the Real Aero Club de España (1949), engineer Eduardo Torroja Miret (died 1961), steelwork by
   OMES: 54 × 45 m, 6.5 m clear height, a vault of crossing three-hinged steel arches of variable depth (a
   lamella-like triangulated vault) springing from brackets on the side columns, with lean-tos along both
   sides. Today it houses the air unit of the Policía Nacional (OSM way 394043499), next to the Real Aero Club.
   The model origin is its centre.
2. The Torre de Señales (signal tower, 1919–1920), «the first aerodrome control tower in the world»: a white
   reinforced-concrete tower like a lighthouse, by the Army engineer Leopoldo Jiménez García, its precast
   pieces made in Enrique Sierra's workshop; Bien de Interés Cultural since 2021. A square base (basement and
   main floor) and a tower of five stages, 16.86 m above the ground: octagonal at the foot, then cylindrical,
   an observation terrace and a glazed dome. It stands in the Air Force base, north-west of the runway
   (OSM node 13654243055).
Rights: protected (Torroja's hangar until the end of 2041, 80 years after 1961 for authors who died before 1987).
Not modelled: runways, taxiways, aprons, the other hangars and buildings, the 1982 and 2005 control towers.

Measurements (see meta.json for the sources):
- Hangar: the OSM outline (way 394043499) 60.0 × 46.0 m, its long side facing 28.3° (the apron and the runway):
  the door front. CEDEX/CEHOPU archive ETM-346-002: «Hangar de 54 × 45 × 6,50 metros para el Aero Club, en
  Cuatro Vientos» (1949) -> the vault spans 54 m across the door front, 45 m deep, 6.5 m clear under the door
  beam, and the lean-tos fill the OSM outline to 60 m (3 m each side). IETcc (CSIC), «Hangares»: arches of
  variable depth that cross to form a rigid triangulated beam, resting on brackets of the side columns. The
  vault is seen in the PNOA orthophoto (looked at only) as a smooth barrel with its axis square to the door
  front. Its rise is NOT published in the sources found: springing at 7.5 m and crown at 16 m are estimates
  from the archive photo of the interior (rise ≈ 1/6 of the span).
  (The brief mentioned arches of ~115 m span; the archive records give 54 m, and the IETcc poster puts a 115 m
  figure under another structure, so 54 m is used.)
- Tower: 16.86 m high, square base, five stages, octagonal foot, glass dome, «Mediterranean white» (ACAMI,
  J. A. Toledano, 2025). Plan and orientation from the PNOA orthophoto: base terrace ≈ 11 × 9 m with its sides
  at 61°/151°, the tower on its north-west half; the stage heights are estimates (±1 m).

Axes (docs/MODEL_SPEC.md): +Y up, the hangar doors face +Z, heading 28.3°; +X points to 298.3°. Origin = centre
of the hangar (centroid of OSM way 394043499), y = 0 = the apron at the doors.

    python model_sources/Q1431985_aerodromo_de_cuatro_vientos.py   -> models/Q1431985/model.glb   (--footprint)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, quad, triangulate, write_glb, GLASS  # noqa: E402

HEADING = 28.3
ORIGIN = (40.3678143, -3.7781218)              # centroid of OSM way 394043499
TOWER_LL = (40.3753633, -3.7884384)            # OSM node 13654243055, Torre de Señales
TOWER_FRONT = 151.0                            # bearing the front of the tower base faces (orthophoto)

SPAN, DEPTH = 54.0, 45.6                       # vault span (across the doors) and depth (OSM 46 m)
LEAN = 3.0                                     # lean-tos on both sides (OSM 60 m overall)
Y_SPRING, Y_CROWN = 7.5, 16.0
Y_DOOR = 6.5
FOUND = -3.0

ROOF = (0.80, 0.81, 0.80)
ROOF_D = (0.74, 0.75, 0.74)
WALL = (0.86, 0.85, 0.81)
WALL_D = (0.70, 0.69, 0.66)
DOOR = (0.62, 0.64, 0.66)
DOOR_L = (0.45, 0.47, 0.50)
GLAZE = (0.22, 0.27, 0.31)
FRAME = (0.30, 0.31, 0.32)
LEAN_ROOF = (0.66, 0.67, 0.66)
WHITE = (0.95, 0.94, 0.90)
TERRACE = (0.66, 0.44, 0.36)
DOME = (0.30, 0.42, 0.44)
IRON = (0.25, 0.26, 0.27)


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def local_xz(lat, lon):
    """lat/lon -> model x/z (metres) around ORIGIN, heading HEADING."""
    n = (lat - ORIGIN[0]) * 111320.0
    e = (lon - ORIGIN[1]) * 111320.0 * math.cos(math.radians(ORIGIN[0]))
    h = math.radians(HEADING)
    z = e * math.sin(h) + n * math.cos(h)
    x = e * math.sin(h - math.pi / 2) + n * math.cos(h - math.pi / 2)
    return x, z


# ── The hangar ────────────────────────────────────────────────────────────────────────────────────
R_V = (SPAN / 2) ** 2 / (2 * (Y_CROWN - Y_SPRING)) + (Y_CROWN - Y_SPRING) / 2     # radius of the vault
Y_C0 = Y_CROWN - R_V                                                              # centre of the circle


def vault_y(x):
    return Y_C0 + np.sqrt(np.maximum(R_V ** 2 - np.asarray(x, float) ** 2, 0))


def hangar():
    ms = []
    hd = DEPTH / 2
    nx = 36
    xs = np.linspace(-SPAN / 2, SPAN / 2, nx + 1)
    # the vault: strips of roof sheeting along the depth (flat-shaded facets, a slightly polyhedral look)
    v, f, c = [], [], []
    for k in range(nx):
        x0, x1 = xs[k], xs[k + 1]
        y0, y1 = float(vault_y(x0)), float(vault_y(x1))
        o = len(v)
        v += [(x0, y0, -hd - 0.4), (x1, y1, -hd - 0.4), (x1, y1, hd + 0.4), (x0, y0, hd + 0.4)]
        c += [ROOF if k % 2 else ROOF_D] * 4
        f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    m = Mesh(np.array(v), np.array(f), np.array(c, float))
    nn = np.cross(m.v[m.f[:, 1]] - m.v[m.f[:, 0]], m.v[m.f[:, 2]] - m.v[m.f[:, 0]])
    flip = nn[:, 1] < 0
    m.f[flip] = m.f[flip][:, ::-1]
    ms.append(m)
    # eaves: a thin edge strip along both sides
    for sx in (-1, 1):
        ms.append(B(sx * SPAN / 2 - 0.4, sx * SPAN / 2 + 0.4, Y_SPRING - 0.6, Y_SPRING + 0.05, -hd - 0.4, hd + 0.4, WALL_D))
    # side walls under the springing, and the lean-tos
    for sx in (-1, 1):
        xa = sx * SPAN / 2
        ms.append(B(min(xa, xa - sx * 0.4), max(xa, xa - sx * 0.4), FOUND, Y_SPRING - 0.6, -hd, hd, WALL))
        xl0, xl1 = sorted((xa, xa + sx * LEAN))
        ms.append(B(xl0, xl1, FOUND, 4.2, -hd + 0.6, hd - 0.6, WALL))
        ms.append(B(xl0 - 0.15, xl1 + 0.15, 4.2, 4.45, -hd + 0.45, hd - 0.45, LEAN_ROOF))
        for zz in np.arange(-hd + 4.0, hd - 3.0, 5.0):           # windows of the lean-tos
            ms.append(quad(xa + sx * LEAN, 2.4, zz, 2.4, 1.2, "+x" if sx > 0 else "-x", GLAZE))
    # the two gables: wall up to the door beam, then the glazed tympanum under the vault
    for sz in (-1, 1):
        zf = sz * hd
        nrm = "+z" if sz > 0 else "-z"
        ms.append(gable(zf, sz))
        if sz > 0:
            # sliding doors: almost the whole span, in leaves of ~4.9 m
            nleaf = 10
            w = (SPAN - 2.0) / nleaf
            for k in range(nleaf):
                xc = -SPAN / 2 + 1.0 + (k + 0.5) * w
                ms.append(quad(xc, Y_DOOR / 2, zf, w - 0.12, Y_DOOR, nrm, DOOR if k % 2 else DOOR_L, eps=0.08))
            ms.append(quad(0, Y_DOOR + 0.5, zf, SPAN, 1.0, nrm, WALL_D, eps=0.1))           # the door beam
        else:
            for k in range(8):
                ms.append(quad(-SPAN / 2 + 3.4 + k * 6.75, 3.0, zf, 2.6, 1.6, nrm, GLAZE, eps=0.06))
        ms.append(tympanum(zf, sz))
    # floor slab edge / plinth
    ms.append(B(-SPAN / 2 - LEAN - 0.2, SPAN / 2 + LEAN + 0.2, FOUND, 0.15, -hd - 0.2, hd + 0.2, WALL_D))
    return merge(ms)


def gable(zf, sz):
    """The end wall from the ground to the vault, as one flat shape (its upper edge follows the arch)."""
    pts = [(-SPAN / 2, FOUND), (SPAN / 2, FOUND)]
    for x in np.linspace(SPAN / 2, -SPAN / 2, 37):
        pts.append((float(x), float(vault_y(x))))
    P, T = triangulate(pts)
    v = np.stack([P[:, 0], P[:, 1], np.full(len(P), zf)], 1)
    m = Mesh(v, T, WALL)
    nn = np.cross(v[m.f[:, 1]] - v[m.f[:, 0]], v[m.f[:, 2]] - v[m.f[:, 0]])
    flip = nn[:, 2] * sz < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def tympanum(zf, sz):
    """The glazed band under the arch of each gable, with mullions."""
    ms = []
    x_lim = SPAN / 2 - 2.0
    y_bot = Y_DOOR + 1.0 if sz > 0 else Y_SPRING
    n = 24
    xs = np.linspace(-x_lim, x_lim, n + 1)
    for k in range(n):
        xa, xb = xs[k], xs[k + 1]
        ya, yb = float(vault_y(xa)) - 1.0, float(vault_y(xb)) - 1.0
        if min(ya, yb) <= y_bot + 0.3:
            continue
        pts = [(xa, y_bot), (xb, y_bot), (xb, yb), (xa, ya)]
        P, T = triangulate(pts)
        v = np.stack([P[:, 0], P[:, 1], np.full(len(P), zf + sz * 0.07)], 1)
        m = Mesh(v, T, GLAZE)
        nn = np.cross(v[m.f[:, 1]] - v[m.f[:, 0]], v[m.f[:, 2]] - v[m.f[:, 0]])
        flip = nn[:, 2] * sz < 0
        m.f[flip] = m.f[flip][:, ::-1]
        ms.append(m)
        ms.append(B(xb - 0.07, xb + 0.07, y_bot, yb, zf + sz * 0.05 - 0.05, zf + sz * 0.05 + 0.05, FRAME))
    ms.append(B(-x_lim, x_lim, y_bot + 2.4, y_bot + 2.55, zf + sz * 0.05 - 0.05, zf + sz * 0.05 + 0.05, FRAME))
    return merge(ms)


# ── The Torre de Señales ──────────────────────────────────────────────────────────────────────────────
def tower():
    """Built facing +Z at the origin; placed and turned in main()."""
    ms = []
    # the square base: basement (half out of the ground) and main floor, a flat terrace with a parapet
    bw, bd, bh = 10.6, 8.6, 4.6
    ms.append(B(-bw / 2, bw / 2, -8.0, bh, -bd / 2, bd / 2, WHITE))
    ms.append(B(-bw / 2 - 0.1, bw / 2 + 0.1, bh, bh + 0.08, -bd / 2 - 0.1, bd / 2 + 0.1, TERRACE))
    for (x0, x1, z0, z1) in ((-bw / 2, bw / 2, bd / 2 - 0.3, bd / 2), (-bw / 2, bw / 2, -bd / 2, -bd / 2 + 0.3),
                             (-bw / 2, -bw / 2 + 0.3, -bd / 2, bd / 2), (bw / 2 - 0.3, bw / 2, -bd / 2, bd / 2)):
        ms.append(B(x0, x1, bh, bh + 0.9, z0, z1, WHITE))
    # door and windows of the main floor on the front, windows round the other sides
    ms.append(quad(0, 1.6, bd / 2, 1.6, 2.6, "+z", (0.32, 0.25, 0.20)))
    for x in (-3.4, 3.4):
        ms.append(quad(x, 2.3, bd / 2, 1.2, 1.6, "+z", GLAZE))
    for z in (-2.2, 2.2):
        ms.append(quad(bw / 2, 2.3, z, 1.2, 1.6, "+x", GLAZE))
        ms.append(quad(-bw / 2, 2.3, z, 1.2, 1.6, "-x", GLAZE))
    for x in (-3.0, 0.0, 3.0):
        ms.append(quad(x, 2.3, -bd / 2, 1.2, 1.6, "-z", GLAZE))
    # the tower on the back half: octagonal foot, cylinder, a second narrower cylinder, terrace, glass dome
    tz = -1.8
    oct_ = cyl(2.05, 1.9, 8.2 - bh, WHITE, seg=8).R(0, math.pi / 8, 0).T(0, bh, tz)
    ms.append(oct_)
    ms.append(cyl(2.05, 2.05, 0.35, WHITE, seg=8).R(0, math.pi / 8, 0).T(0, 8.2, tz))           # cornice
    ms.append(cyl(1.55, 1.5, 11.6 - 8.55, WHITE, seg=20).T(0, 8.55, tz))
    ms.append(cyl(1.75, 1.75, 0.3, WHITE, seg=20).T(0, 11.6, tz))
    ms.append(cyl(1.3, 1.25, 13.4 - 11.9, WHITE, seg=20).T(0, 11.9, tz))
    ms.append(cyl(2.1, 2.1, 0.25, WHITE, seg=24).T(0, 13.4, tz))                              # observation terrace
    ring = []
    for k in range(24):
        a = 2 * math.pi * k / 24
        ring.append(B(2.0 * math.cos(a) - 0.04, 2.0 * math.cos(a) + 0.04, 13.65, 14.65, 2.0 * math.sin(a) - 0.04,
                      2.0 * math.sin(a) + 0.04, IRON).T(0, 0, tz))
    ms += ring
    ms.append(lathe([(2.02, 14.6), (2.02, 14.68)], IRON, seg=24).T(0, 0, tz))
    ms.append(cyl(1.0, 1.0, 15.4 - 13.65, DOME, seg=16).T(0, 13.65, tz))                      # glazed lantern
    dome = [(1.08, 15.4)] + [(1.08 * math.cos(t), 15.4 + 1.46 * math.sin(t)) for t in np.linspace(0.2, math.pi / 2, 7)]
    dome[-1] = (0.0, 16.86)
    ms.append(lathe(dome, DOME, seg=16).T(0, 0, tz))
    ms.append(cyl(0.05, 0.03, 1.6, IRON, seg=6).T(0, 16.8, tz))                              # windsock mast
    return merge(ms)


def footprints(tx, tz, ry):
    """Hangar outline + 1 m and the tower base + 1.2 m, joined by a 0.1 m slit (two separate clearings)."""
    hx, hz = SPAN / 2 + LEAN + 1.0, DEPTH / 2 + 1.0
    c, s = math.cos(ry), math.sin(ry)
    # tower base corners (local, around the tower piece's centre) -> model, counter-clockwise in (x, z)
    cx, cz = tx + 1.8 * s, tz + 1.8 * c
    loc = [(-6.5, -5.5), (6.5, -5.5), (6.5, 5.5), (-6.5, 5.5)]
    tw = [(cx + x * c + z * s, cz - x * s + z * c) for x, z in loc]
    area = 0.5 * sum(tw[i - 1][0] * tw[i][1] - tw[i][0] * tw[i - 1][1] for i in range(4))
    if area < 0:
        tw = tw[::-1]
    a = math.atan2(cz, cx)                      # direction from the hangar to the tower
    k = min(range(4), key=lambda i: math.hypot(tw[i][0], tw[i][1]))
    t_ring = tw[k:] + tw[:k]                    # starts at the corner nearest the hangar
    ux, uz = -math.sin(a) * 0.05, math.cos(a) * 0.05      # half-width of the slit, square to it
    hang = [(-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz)]   # counter-clockwise in (x, z)
    ring = hang[:2] + [(hx, -0.05), (t_ring[0][0] - ux, t_ring[0][1] - uz)] + t_ring[1:] +         [(t_ring[0][0] + ux, t_ring[0][1] + uz), (hx, 0.05)] + hang[2:]
    return [[round(x, 2), round(z, 2)] for x, z in ring]


def main():
    out = ROOT / "models" / "Q1431985"
    out.mkdir(parents=True, exist_ok=True)
    h = hangar()
    tx, tz = local_xz(*TOWER_LL)
    ry = -math.radians(TOWER_FRONT - HEADING)
    t = tower().R(0, ry, 0)
    t.T(tx, 0, tz)
    # the tower OSM node marks the tower itself: move the piece so the tower (at local z = -1.8) sits on it
    t.T(1.8 * math.sin(ry), 0, 1.8 * math.cos(ry))
    parts = [("hangar", h, {"material": "zinc"}), ("torre_de_senales", t)]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.9, materials={"zinc": {"metallic": 0.35, "roughness": 0.55}})
    allv = np.vstack([h.v, t.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}; tower at x {tx:.1f}, z {tz:.1f}")
    if "--footprint" in sys.argv:
        print(json.dumps(footprints(tx, tz, ry)))


if __name__ == "__main__":
    main()
