"""
Torre Moeve (ex Torre Cepsa, also «Torre Foster»; Q519568), Cuatro Torres Business Area, Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Foster + Partners (Norman Foster, b. 1935), 2004–2008, opened 2009 — protected.

Measurements (see meta.json for the sources):
- OpenStreetMap way 188396764 and its 15 building:part ways (mapped with heights from the CTBUH data):
  footprint 53.3 × 42.9 m; two concrete service cores at the east and west ends (each 13.3 × 23.5 m, clad in
  stainless steel, a black strip 14.3 m wide down the middle of each end face), height 248.3 m (CTBUH, 2013).
- Office floors hang between the cores in three blocks: 25.6–81.9, 92.1–153.5 and 163.7–220 m (11–12 floors each),
  with the façade set back 2.7 m at the two transfer platforms between them (81.9–92.1 and 153.5–163.7 m), where
  the steel trusses show. Below, the floors step back over the 13.85–15.3 m high entrance hall (atrium) between the
  cores, which is glazed on both broad faces.
- The «arch»: a void 19 m high between the top office floor (220 m) and the steel frame that ties the two cores
  (239–248.3 m) — the gate-like silhouette seen from all over Madrid. The wind turbines in the void are omitted.
Dark grey-blue glass with light floor lines on the offices, stainless steel on the cores and the frame.

Heading: the two broad glass faces look north and south (Foster: the north–south orientation turns the cores to the
low east and west sun and frames the views to the Sierra and to the city); the end faces towards the Paseo de la
Castellana (west) are the blind steel cores. The main façade (+Z) is the south face, the one that faces the city
centre and the approach along the Castellana: heading 191.6° (OSM long axis 101.6°).

Axes (docs/MODEL_SPEC.md): +Y up, the south façade faces +Z, +X points east; origin = centroid of the OSM outline.

    python model_sources/Q519568_torre_moeve.py   -> models/Q519568/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, cap, tube, write_glb, GLASS, METAL, _orient  # noqa: E402

FOUND = -3.0
H_TOP = 248.3
GLASS_C = (0.36, 0.43, 0.50)          # dark grey-blue office glass
BAND_C = (0.56, 0.59, 0.61)           # floor edge / spandrel line
LOBBY_C = (0.42, 0.50, 0.55)          # the clearer glass of the entrance hall
STEEL = (0.74, 0.75, 0.76)            # stainless steel cladding of the cores and the frame
STEEL_J = (0.69, 0.70, 0.71)          # joints between the steel plates
BLACK = (0.19, 0.21, 0.24)            # the black strip on the core ends
TRUSS = (0.36, 0.37, 0.39)            # transfer platforms
SOFFIT = (0.55, 0.56, 0.57)

# ── helpers (shared pattern: horizontal floor bands cut out of planar faces) ──
def clip_y(poly, y, above):
    out, n = [], len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ia = a[1] >= y if above else a[1] <= y
        ib = b[1] >= y if above else b[1] <= y
        if ia:
            out.append(a)
        if ia != ib:
            t = (y - a[1]) / (b[1] - a[1])
            out.append(tuple(a[k] + (b[k] - a[k]) * t for k in range(3)))
    return out


def fan(poly, c):
    if len(poly) < 3:
        return None
    return Mesh(poly, [(0, i, i + 1) for i in range(1, len(poly) - 1)], c)


def bands(lo, hi, levels, band):
    """[(y0, y1, is_band)] covering lo..hi: a band of `band` m at the foot of each floor level."""
    cuts = sorted({lo, hi} | {y for L in levels for y in (L, L + band) if lo < y < hi})
    out = []
    for a, b in zip(cuts, cuts[1:]):
        m = (a + b) / 2
        out.append((a, b, any(L <= m < L + band for L in levels)))
    return out


def banded(poly, levels, band, cg, cb):
    ys = [p[1] for p in poly]
    ms = []
    for a, b, isb in bands(min(ys), max(ys), levels, band):
        q = clip_y(clip_y(poly, a, True), b, False)
        ms.append(fan(q, cb if isb else cg))
    return merge(ms)


def wall_quads(ring, y0, y1):
    """Outward-facing vertical quads along a ring (counter-clockwise in x/z)."""
    P = _orient(ring, True)
    out = []
    for i in range(len(P)):
        (x0, z0), (x1, z1) = P[i], P[(i + 1) % len(P)]
        out.append((((x0, y0, z0), (x0, y1, z0), (x1, y1, z1), (x1, y0, z1)), (x0, z0, x1, z1)))
    return out


def prism(ring, y0, y1, cg, levels=(), band=0.0, cb=None, top=None, bottom=None, colour_of=None):
    """Walls with floor bands; top/bottom = cap colour or None. colour_of(x0,z0,x1,z1) -> wall colour override."""
    ms = []
    for q, e in wall_quads(ring, y0, y1):
        c = colour_of(*e) if colour_of else None
        c = cg if c is None else c
        ms.append(banded(list(q), levels, band, c, cb or c) if levels else fan(list(q), c))
    if top is not None:
        ms.append(cap(ring, y1, top, up=True))
    if bottom is not None:
        ms.append(cap(ring, y0, bottom, up=False))
    return merge(ms)


def rect(x0, x1, z0, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def floors(y0, y1, n):
    return [y0 + (y1 - y0) * k / n for k in range(n)]


# ── the building ─────────────────────────────────────────────────────────────
CX0, CX1 = 13.35, 26.65          # cores: |x| from the inner face to the end face
CZ = 11.77                       # cores: half depth
XI = 13.5                        # office floors reach 15 cm into the cores (no coplanar faces)
ZO = 21.45                       # office blocks: outer face
ZM = 18.8                        # set-back face (atrium storeys, transfer platforms)
XB, XM = 16.3, 14.38             # half width of the office blocks / of the set-back storeys
ZS = 11.74                       # where the office wings start beside the cores (3 cm inside them)
BLOCKS = [(25.6, 81.9, 12), (92.1, 153.5, 13), (163.7, 220.0, 12)]
PLATFORMS = [(81.9, 92.1), (153.5, 163.7)]


def office_ring(xw, zf):
    """Plan of an office storey: between the cores (|x| < XI) and, beyond them, wings to ±xw out to ±zf."""
    return [(XI, -ZS), (xw, -ZS), (xw, -zf), (-xw, -zf), (-xw, -ZS), (-XI, -ZS),
            (-XI, ZS), (-xw, ZS), (-xw, zf), (xw, zf), (xw, ZS), (XI, ZS)]


def core(sx):
    """One service core; sx = +1 east, -1 west. The middle of the end face is a black strip, 10 cm back."""
    ring = [(CX0, -CZ), (CX1, -CZ), (CX1, -7.15), (CX1 - 0.1, -7.15), (CX1 - 0.1, 7.15), (CX1, 7.15), (CX1, CZ), (CX0, CZ)]
    ring = [(sx * x, z) for x, z in ring]

    def col(x0, z0, x1, z1):
        if abs(abs(x0) - (CX1 - 0.1)) < 1e-6 and abs(abs(x1) - (CX1 - 0.1)) < 1e-6:
            return BLACK
        return None
    lv = floors(0.0, H_TOP, 53)
    return prism(ring, FOUND, H_TOP, STEEL, lv, 0.25, STEEL_J, top=STEEL, colour_of=col)


def trusses(y0, y1):
    """The steel diagonals of a transfer platform on both broad faces (a W of four panels)."""
    ms = []
    xs = np.linspace(-XB + 0.4, XB - 0.4, 5)
    for sz in (-1, 1):
        z = sz * (ZO - 0.3)
        pts = [(x, y0 + 0.4 if k % 2 == 0 else y1 - 0.4, z) for k, x in enumerate(xs)]
        ms.append(tube(pts, 0.35, TRUSS, 6))
        ms.append(tube([(-XB + 0.3, (y0 + y1) / 2, z), (XB - 0.3, (y0 + y1) / 2, z)], 0.18, TRUSS, 6))
    return ms


def building():
    glass, steel = [], []
    steel += [core(1), core(-1)]
    # entrance hall between the cores, glazed on both broad faces (two tall storeys)
    glass.append(prism(rect(-XI, XI, -11.1, 11.1), FOUND, 15.3, LOBBY_C, [0.0, 7.65], 0.35, BAND_C))
    # set-back storeys over the hall, then the three hanging blocks with the transfer platforms between them
    ring_m, ring_b = office_ring(XM, ZM), office_ring(XB, ZO)
    glass.append(prism(ring_m, 15.3, 25.6, GLASS_C, floors(15.3, 25.6, 2), 0.5, BAND_C, bottom=SOFFIT))
    for y0, y1, n in BLOCKS:
        glass.append(prism(ring_b, y0, y1, GLASS_C, floors(y0, y1, n), 0.55, BAND_C, top=TRUSS, bottom=SOFFIT))
    for y0, y1 in PLATFORMS:
        steel.append(prism(ring_m, y0, y1, TRUSS, floors(y0, y1, 2), 0.6, (0.28, 0.29, 0.31)))
        steel += trusses(y0, y1)
    # the frame that closes the «arch»: a slab under it and the deep steel beam tying the tops of the cores
    steel.append(prism(rect(-XI, XI, -11.1, 11.1), 239.0, 240.4, (0.30, 0.31, 0.33), bottom=SOFFIT))
    steel.append(prism(rect(-XI, XI, -11.7, 11.7), 240.4, H_TOP - 0.05, STEEL, [240.4, 244.3], 0.25, STEEL_J,
                       top=STEEL, bottom=SOFFIT))
    return merge(glass), merge(steel)


def main():
    out = ROOT / "models" / "Q519568"
    out.mkdir(parents=True, exist_ok=True)
    g, s = building()
    tris, size = write_glb(out / "model.glb", [("offices_glass", g, {"material": "glass"}),
                                                ("cores_steel", s, {"material": "steel"})],
                           materials={"glass": GLASS, "steel": METAL}, roughness=0.85)
    m = merge([g, s])
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
