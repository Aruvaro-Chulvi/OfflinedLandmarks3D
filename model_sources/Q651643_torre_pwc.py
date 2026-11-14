"""
Torre PwC (ex Torre Sacyr Vallehermoso; Q651643), Cuatro Torres Business Area, Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Carlos Rubio Carvajal and Enrique Álvarez-Sala Walther, 2004–2008 — protected.

Measurements (see meta.json for the sources):
- OpenStreetMap way 98280725: the plan, 47.3 × 47.2 m, a rounded triangle of three convex lobes («three arches
  enclosing three cylinders», Wikipedia) split by three narrow vertical fissures (1.6 m wide, 3.4 m deep) that
  divide every floor in three. Height 236 m (OSM, Wikipedia), 52 floors for calculation, 58 levels in all.
- OpenStreetMap way 970617526: the hexagonal roof deck at 236 m inside the crown.
- Façade (Wikipedia, photos on Wikimedia Commons): a double skin of glass «scales» — one tilted pane per floor and
  bay, overlapping like fish scales, bronze-grey and reflective. Hotel Eurostars Madrid Tower on floors 6–27,
  offices above (PwC on 34–50), a two-storey restaurant near the top; floors 51–58 hold the plant, hidden in an
  open crown where the scaly skin goes on above the last floor as separate panels with sky between them.
Simplified: one scale per floor and bay of about 2.4 m (staggered from floor to floor), a recessed glazed ground
storey, the plant as one grey hexagonal block; no wind turbines, no logo.

Heading: the tower has no front; the fissure that faces the Paseo de la Castellana (west), where the hotel has its
address and entrance (Pº de la Castellana 259 B), is taken as the main façade: heading 281.4° = the bearing from the
centre of the plan to the middle of that fissure.

Axes (docs/MODEL_SPEC.md): +Y up, the west fissure faces +Z, +X points south-south-west (191.4°);
origin = centroid of the OSM outline.

    python model_sources/Q651643_torre_pwc.py   -> models/Q651643/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, cap, extrude, offset, write_glb, GLASS, METAL, _orient  # noqa: E402

rng = np.random.default_rng(2008)

FOUND = -3.0
H_TOP = 236.0
Y_SKIN0 = 7.6                         # the scales start over the ground storey
Y_ROOF = 212.0                        # last enclosed floor: the crown above is open
N_FLOORS = 54
SCALE_C = np.array((0.52, 0.52, 0.48))   # silver-grey reflective glass with a warm tint
FISSURE_C = (0.13, 0.14, 0.15)
LOBBY_C = (0.22, 0.24, 0.25)
REST_C = np.array((0.58, 0.58, 0.54))    # the lighter glass of the top restaurant storeys
PLANT_C = (0.38, 0.39, 0.41)
ROOF_C = (0.33, 0.34, 0.35)

# OSM way 98280725 in model x/z (heading 281.4), simplified (Douglas–Peucker 0.10 m)
OUTLINE = [(0.97, 25.07), (0.62, 24.32), (0.67, 20.95), (-0.94, 20.88), (-0.92, 24.37), (-1.40, 25.15), (-2.09, 25.27),
           (-3.75, 24.88), (-5.37, 24.26), (-7.81, 22.81), (-10.55, 20.87), (-13.74, 17.93), (-15.77, 15.68),
           (-17.85, 12.86), (-19.86, 9.36), (-21.08, 6.65), (-22.21, 3.48), (-23.06, -0.11), (-23.67, -5.26),
           (-23.48, -8.87), (-22.66, -11.30), (-22.31, -11.62), (-21.64, -11.70), (-18.44, -9.90), (-17.62, -11.18),
           (-20.62, -12.96), (-21.01, -13.58), (-21.02, -14.15), (-20.65, -14.67), (-18.61, -16.56), (-17.24, -17.43),
           (-12.06, -19.78), (-9.89, -20.50), (-6.78, -21.28), (-2.89, -21.80), (3.12, -21.79), (7.34, -21.13),
           (12.21, -19.71), (15.46, -18.27), (18.53, -16.59), (19.90, -15.44), (21.03, -14.20), (21.26, -13.65),
           (20.85, -12.80), (17.84, -10.99), (18.53, -9.69), (21.98, -11.50), (22.53, -11.30), (22.93, -10.81),
           (23.63, -8.14), (23.63, -4.29), (23.31, -1.30), (22.29, 3.25), (20.36, 8.43), (18.85, 11.23), (17.25, 13.61),
           (13.98, 17.61), (12.05, 19.49), (9.07, 21.87), (7.04, 23.23), (4.98, 24.35), (2.80, 25.08), (1.46, 25.29)]
FISSURES = [(-0.14, 20.92), (-18.03, -10.54), (18.19, -10.34)]       # middle of the bottom of each fissure
HEXAGON = [(-7.32, 14.30), (9.36, 14.14), (17.54, -0.43), (9.05, -14.85), (-7.63, -14.70), (-15.81, -0.12)]  # way 970617526


def fan(poly, c):
    if len(poly) < 3:
        return None
    return Mesh(poly, [(0, i, i + 1) for i in range(1, len(poly) - 1)], c)


def in_fissure(x, z, r=3.6):
    return any(math.hypot(x - fx, z - fz) < r for fx, fz in FISSURES)


def subdivide(ring, step, phase):
    """Points along the ring every ~step m (each side split evenly), shifted by phase (0 or 0.5) of a bay;
    the corners are kept so the plan stays exact. Returns [(x, z, corner?)]."""
    out = []
    n = len(ring)
    for i in range(n):
        a, b = np.array(ring[i]), np.array(ring[(i + 1) % n])
        k = max(1, round(np.linalg.norm(b - a) / step))
        out.append((a[0], a[1], True))
        ts = [(j + phase) / k for j in range(k)] if phase else [j / k for j in range(1, k)]
        for t in ts:
            if 1e-6 < t < 1 - 1e-6:
                p = a + (b - a) * t
                out.append((p[0], p[1], False))
    return out


def scales_row(pts, y0, y1, tilt0=-0.10, tilt1=0.16, gap=0.0, colour=None):
    """One storey of scales: a quad per bay, its foot tucked in by tilt0 and its head out by tilt1 (outward),
    so every row catches the light differently. gap > 0 leaves an open slot between the panels (the crown)."""
    xz = [(p[0], p[1]) for p in pts]
    lo, hi = offset(xz, tilt0), offset(xz, tilt1)
    ms = []
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        a0, b0, a1, b1 = np.array(lo[i]), np.array(lo[j]), np.array(hi[i]), np.array(hi[j])
        mx, mz = (xz[i][0] + xz[j][0]) / 2, (xz[i][1] + xz[j][1]) / 2
        if gap:
            if in_fissure(mx, mz, 4.2):
                continue
            s = gap / 2
            a0, b0 = a0 + (b0 - a0) * s, b0 - (b0 - a0) * s
            a1, b1 = a1 + (b1 - a1) * s, b1 - (b1 - a1) * s
        if in_fissure(mx, mz):
            c = FISSURE_C
        else:
            base = SCALE_C if colour is None else colour
            c = tuple(np.clip(base * (1 + rng.normal(0, 0.05)), 0, 1))
        quad = [(a0[0], y0, a0[1]), (a1[0], y1, a1[1]), (b1[0], y1, b1[1]), (b0[0], y0, b0[1])]
        ms.append(fan(quad, c))
    return merge(ms)


def tower():
    ring = _orient(OUTLINE, True)                    # counter-clockwise: offset() keeps the point order
    rows_a = subdivide(ring, 2.4, 0.0)
    rows_b = subdivide(ring, 2.4, 0.5)
    glass, solid = [], []
    # ground storey: glazed and set back 1 m under the first scales, foundations to -3 m
    core = offset(ring, -1.0)
    glass.append(extrude(core, FOUND, Y_SKIN0, LOBBY_C, top=False))
    solid.append(cap(ring, Y_SKIN0 - 0.02, ROOF_C, up=False))                # soffit of the first floor
    # the scaly shaft
    fh = (Y_ROOF - Y_SKIN0) / N_FLOORS
    for k in range(N_FLOORS):
        y0 = Y_SKIN0 + k * fh
        col = REST_C if k >= N_FLOORS - 2 else None
        glass.append(scales_row(rows_b if k % 2 else rows_a, y0, y0 + fh, colour=col))
    # inner face of the skin (seen through the slots of the crown) and the roof over the last floor
    inner = offset(ring, -0.35)
    solid.append(extrude(inner, Y_ROOF - 1.0, Y_ROOF, ROOF_C, top=True))
    # the open crown: the scales go on as separate panels with sky between them, tied by a ring beam on top
    for k in range(6):
        y0 = Y_ROOF + k * fh
        glass.append(scales_row(rows_b if k % 2 else rows_a, y0, y0 + fh, gap=0.38))
    y_beam = Y_ROOF + 6 * fh
    solid.append(extrude(offset(ring, 0.12), y_beam, y_beam + 0.9, PLANT_C, holes=[offset(ring, -0.45)], bottom=True))
    # the plant inside the crown and the hexagonal roof deck at 236 m (OSM way 970617526)
    solid.append(extrude(HEXAGON, Y_ROOF, H_TOP - 1.2, PLANT_C, top_c=ROOF_C))
    solid.append(extrude(offset(HEXAGON, 0.6), H_TOP - 1.2, H_TOP, (0.50, 0.51, 0.52), bottom=True))
    return merge(glass), merge([m for m in solid if m is not None])


def main():
    out = ROOT / "models" / "Q651643"
    out.mkdir(parents=True, exist_ok=True)
    g, s = tower()
    tris, size = write_glb(out / "model.glb", [("scales_glass", g, {"material": "glass"}),
                                                ("roof_plant", s, {"material": "metal"})],
                           materials={"glass": GLASS, "metal": METAL}, roughness=0.85)
    m = merge([g, s])
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
