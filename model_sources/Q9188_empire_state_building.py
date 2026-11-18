"""
Empire State Building (Q9188), Midtown Manhattan, New York — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Shreve, Lamb & Harmon (William F. Lamb, chief designer, died 1952; Richmond H. Shreve, died 1946;
Arthur Loomis Harmon, died 1958), built 1930–1931 for Empire State Inc. (John J. Raskob, Alfred E. Smith);
opened 1 May 1931. Rights: public domain in the United States (buildings finished before 1 December 1990 have no
architectural-work copyright, Architectural Works Copyright Protection Act 1990); marked "protected" in meta.json
because the repository counts 80 years from the architects' deaths (Lamb 1952, Harmon 1958).

Measurements (see meta.json for the sources):
- OpenStreetMap way 34633854 (the lot, height 443.2, 102 levels) and the building:part ways of the tower, in
  model x/z for heading 299.0 — the Fifth Avenue front (the short side of the lot) faces west-north-west, at
  right angles to the long axis of the lot (119.0°, the Manhattan street grid):
  · relation 10872054: the five-storey base filling the whole lot, 59.5 × 129.5 m (197 × 424 ft);
  · way 137425131: floors 6–20 (setback at the 6th floor, about 17.7 m back from Fifth Avenue and from the east
    end), with the side wings towards 33rd and 34th Streets; top 75 m;
  · way 137425133: floors 21–24, top 90 m;   way 137425128: floors 25–29, top 115 m;
  · way 137425127: the tower shaft, floors 30–71 (setback at the 30th floor), 115 → 255 m, about 41 × 56 m with
    the deep notches in the middle of the 33rd/34th Street faces;
  · way 137425129: floors 72–80, 255 → 290 m;   way 137425125: floors 81–85, 290 → 310 m;
  · way 137425144: the 86th floor, to 320 m (its roof is the open-air observatory, 1,050 ft = 320 m);
  · way 137425142: the glazed 86th-floor observatory block, to 330 m;
  · ways 265932618 / 137425145: the mooring mast and the antenna (octagons about 13 m and 8 m across).
  OSM's setbacks (6th, 21st, 25th, 30th, 72nd, 81st and 86th floors) agree with the published ones (6th, 25th,
  30th, 72nd and 81st floors, en.wikipedia).
- Published heights (en.wikipedia, Empire State Building): roof 1,250 ft = 381 m, top floor (102nd) 1,224 ft =
  373 m, antenna tip 1,454 ft = 443.2 m, 86th-floor observatory 320 m. The OSM tag of the base (17 m) is too low
  for five storeys with the three-storey-high lobby: the base is modelled 21 m high (ground floor 6 m + 4 ×
  3.75 m), an estimate (±15 %). The mast (330 → 381 m) is modelled from photos: an octagonal shaft with four
  stepped Art Deco buttresses on its diagonals, the glazed upper floors, the 102nd-floor observatory ring and a
  stepped cap; its widths follow the OSM octagons (±1 m). The antenna (381 → 443.2 m) is a tapering steel mast
  with two small platforms (diameters estimated ±30 %).
- Cladding (from the photos, Wikimedia Commons): Indiana limestone piers, the window columns with aluminium
  spandrels and the bright chrome-nickel steel mullions running up the whole height, modelled economically as one
  aluminium strip per window column with a dark pane per floor and two thin bright mullion lines — not 6,514
  separate windows. The mast is aluminium, chrome-nickel steel and glass (material METAL).

Axes (docs/MODEL_SPEC.md): +Y up, the Fifth Avenue front faces +Z (heading 299.0°), +X points to 209° (towards
33rd Street), origin = centroid of the OSM outline of the lot, y = 0 the ground there.

    python model_sources/Q9188_empire_state_building.py   -> models/Q9188/model.glb (+ footprint, triangles)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import (GLASS, METAL, Mesh, box, cap, cyl, merge, offset, ring_area,  # noqa: E402
                     walls, write_glb)

# ── OpenStreetMap (© OpenStreetMap contributors, ODbL), model metres for heading 299.0 and origin
#    40.748442, -73.9856588 (tools/osm.py way 34633854 --heading 299) ───────────────────────────────────────
LOT = [(29.73, 64.86), (29.75, -64.64), (-29.73, -64.84), (-29.75, 64.64), (-0.7, 64.73), (4.53, 64.75),
       (26.74, 64.85)]
P131 = [(-18.19, 46.76), (-18.18, 35.49), (-22.59, 35.46), (-22.55, 14.0), (-18.14, 14.03), (-18.12, 8.94),
        (-18.09, -9.2), (-18.08, -14.14), (-22.59, -14.16), (-22.55, -35.6), (-18.07, -35.58), (-18.05, -47.39),
        (23.15, -47.19), (23.13, -35.4), (27.88, -35.38), (27.84, -13.94), (23.14, -13.96), (23.13, -8.9),
        (23.1, 9.22), (23.09, 14.21), (27.89, 14.23), (27.85, 35.71), (23.02, 35.68), (23.0, 46.96)]
P133 = [(-22.55, -35.6), (-18.07, -35.58), (-4.46, -35.51), (-4.47, -33.92), (10.35, -33.85), (10.36, -35.46),
        (23.13, -35.4), (27.88, -35.38), (27.84, -13.94), (23.14, -13.96), (23.13, -8.9), (23.1, 9.22),
        (23.09, 14.21), (27.89, 14.23), (27.85, 35.71), (23.02, 35.68), (9.8, 35.62), (9.8, 34.09), (-4.59, 34.01),
        (-4.59, 35.56), (-18.18, 35.49), (-22.59, 35.46), (-22.55, 14.0), (-18.14, 14.03), (-18.12, 8.94),
        (-18.09, -9.2), (-18.08, -14.14), (-22.59, -14.16)]
P128 = [(-18.05, -27.86), (-11.69, -27.82), (-11.68, -33.97), (-4.47, -33.92), (10.35, -33.85), (16.59, -33.81),
        (16.58, -27.65), (23.17, -27.62), (23.14, -13.96), (23.13, -8.9), (23.1, 9.22), (23.09, 14.21),
        (23.05, 28.12), (16.41, 28.09), (16.4, 34.12), (9.8, 34.09), (-11.87, 33.97), (-11.86, 27.92),
        (-18.18, 27.89), (-18.14, 14.03), (-18.12, 8.94), (-18.09, -9.2), (-18.08, -14.14)]
P127 = [(-18.18, 27.89), (-18.14, 14.03), (-18.12, 8.94), (-11.83, 8.98), (-11.79, -9.16), (-18.09, -9.2),
        (-18.08, -14.14), (-18.05, -27.86), (-11.69, -27.82), (16.58, -27.65), (23.17, -27.62), (23.14, -13.96),
        (23.13, -8.9), (16.98, -8.94), (16.94, 9.19), (23.1, 9.22), (23.09, 14.21), (23.05, 28.12), (16.41, 28.09),
        (-11.86, 27.92)]
P129 = [(16.93, 11.19), (21.09, 11.21), (21.09, 13.99), (21.07, 26.09), (16.42, 26.07), (-11.86, 25.93),
        (-16.16, 25.91), (-16.14, 10.92), (-11.84, 10.94), (-11.83, 8.98), (-11.79, -9.16), (-11.79, -11.2),
        (-16.09, -11.23), (-16.08, -14.13), (-16.06, -25.83), (-11.7, -25.8), (16.58, -25.66), (21.16, -25.64),
        (21.14, -13.81), (21.14, -10.92), (16.99, -10.97), (16.98, -8.94), (16.94, 9.19)]
P125 = [(-9.97, 22.33), (14.98, 22.52), (16.9, 20.68), (16.93, 11.19), (16.94, 9.19), (16.98, -8.94),
        (16.99, -10.97), (17.02, -20.74), (15.36, -22.64), (-9.4, -22.82), (-11.76, -20.34), (-11.79, -11.2),
        (-11.79, -9.16), (-11.83, 8.98), (-11.84, 10.94), (-11.86, 20.24)]
P144 = [(-9.37, 21.34), (14.55, 21.5), (15.9, 20.19), (16.02, -20.16), (14.74, -21.64), (-8.09, -21.8),
        (-10.76, -19.13), (-10.88, 19.71)]
P142 = [(-6.29, 18.36), (11.61, 18.49), (12.92, 17.1), (13.02, -17.12), (11.5, -18.68), (-6.26, -18.79),
        (-7.78, -17.3), (-7.87, 16.82)]
MAST_C = (2.56, -0.45)               # centre of the mast octagons (ways 265932618, 137425145)

# ── Heights ──────────────────────────────────────────────────────────────────
BASE = 21.0                          # five-storey base (estimate, see the docstring)
G0 = 6.0                             # ground floor with the lobby
FOUND = -4.0
OBS86 = 320.0                        # 86th-floor open-air observatory
ROOF = 381.0                         # roof of the mast (published)
F102 = 373.0                         # 102nd floor (published)
TIP = 443.2                          # antenna tip (published, OSM)

# (name, outline, bottom, top, floors) — OSM building:part ways
TOWER = [("w131", P131, BASE, 75.0, 15), ("w133", P133, 75.0, 90.0, 4), ("w128", P128, 90.0, 115.0, 5),
         ("w127", P127, 115.0, 255.0, 42), ("w129", P129, 255.0, 290.0, 9), ("w125", P125, 290.0, 310.0, 5),
         ("w144", P144, 310.0, OBS86, 1)]

# ── Colours ──────────────────────────────────────────────────────────────────
LIME = np.array((0.80, 0.775, 0.715))      # Indiana limestone
LIME_L = np.array((0.86, 0.84, 0.79))      # copings, cornices
LIME_D = np.array((0.70, 0.67, 0.61))      # granite-ish plinth, shadows
TERR = (0.56, 0.52, 0.48)                  # setback roofs (OSM roof:colour #A98A7B / #b0a5a3)
ALU = (0.63, 0.64, 0.65)                  # aluminium spandrels
CHROME = (0.90, 0.91, 0.92)                # chrome-nickel mullions
PANE = (0.15, 0.18, 0.22)                  # window glass
SHOP = (0.12, 0.14, 0.16)
MAST = (0.78, 0.79, 0.80)                  # aluminium / steel of the mast
MAST_D = (0.60, 0.62, 0.64)
LIT = (0.93, 0.88, 0.70)                   # the lit glazing of the upper floors of the mast
ANT = (0.68, 0.69, 0.70)

PITCH = 2.85                               # window columns (one column = a pair of sashes; stylised)


# ── Geometry helpers ─────────────────────────────────────────────────────────
def ccw(P):
    P = [tuple(map(float, p)) for p in P]
    return P if ring_area(P) > 0 else P[::-1]


def pip(pts, poly):
    P = np.asarray(poly, float)
    x, z = pts[:, 0][:, None], pts[:, 1][:, None]
    xi, zi = P[:, 0][None, :], P[:, 1][None, :]
    xj, zj = np.roll(P[:, 0], 1)[None, :], np.roll(P[:, 1], 1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        hit = ((zi > z) != (zj > z)) & (x < (xj - xi) * (z - zi) / (zj - zi) + xi)
    return (np.sum(hit, 1) % 2) == 1


class Solids:
    def __init__(self):
        self.items = []

    def add(self, poly, y0, y1):
        self.items.append((np.asarray(poly, float), float(y0), float(y1)))

    def inside(self, pts, y):
        pts = np.asarray(pts, float).reshape(-1, 2)
        occ = np.zeros(len(pts), bool)
        for poly, y0, y1 in self.items:
            if y0 <= y < y1:
                occ |= pip(pts, poly)
        return occ


def edges(P):
    """(a, unit direction, outward normal, length) of a counter-clockwise ring, merging straight runs."""
    P = [np.array(p, float) for p in P]
    keep = []
    for i in range(len(P)):
        d0 = P[i] - P[i - 1]; d1 = P[(i + 1) % len(P)] - P[i]
        l0, l1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if l0 < 1e-6 or l1 < 1e-6:
            continue
        if abs(d0[0] * d1[1] - d0[1] * d1[0]) / (l0 * l1) > 0.02 or np.dot(d0, d1) < 0:
            keep.append(P[i])
    out = []
    for i in range(len(keep)):
        a, b = keep[i], keep[(i + 1) % len(keep)]
        L = float(np.linalg.norm(b - a))
        if L > 1e-6:
            e = (b - a) / L
            out.append((a, e, np.array([e[1], -e[0]]), L))
    return out


def wall_quad(a, e, n, s0, s1, y0, y1, off, c):
    p0 = a + e * s0 + n * off; p1 = a + e * s1 + n * off
    V = [(p0[0], y0, p0[1]), (p1[0], y0, p1[1]), (p1[0], y1, p1[1]), (p0[0], y1, p0[1])]
    return Mesh(V, [(0, 2, 1), (0, 3, 2)], c)


def edge_box(a, e, n, s0, s1, y0, y1, o0, o1, c):
    m = box(s1 - s0, y1 - y0, o1 - o0, c)
    m.R(0, math.atan2(-e[1], e[0]), 0)
    cx = a + e * (s0 + s1) / 2 + n * (o0 + o1) / 2
    return m.T(cx[0], y0, cx[1])


def runs(flags):
    out, i0 = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and i0 is None:
            i0 = i
        elif not f and i0 is not None:
            out.append((i0, i)); i0 = None
    return out


def column_positions(a, e, L, margin):
    """Window columns on a global lattice (so that they line up across setbacks); local ones on chamfers."""
    if abs(e[0]) > 0.98 or abs(e[1]) > 0.98:
        ax = 0 if abs(e[0]) > 0.98 else 1
        org = MAST_C[ax]
        c0, c1 = sorted((a[ax] + margin * np.sign(e[ax]), a[ax] + (L - margin) * np.sign(e[ax])))
        k0, k1 = math.ceil((c0 - org) / PITCH - 0.5), math.floor((c1 - org) / PITCH - 0.5)
        cs = org + (np.arange(k0, k1 + 1) + 0.5) * PITCH
        return np.sort((cs - a[ax]) / e[ax])
    if L < 2.6:
        return np.zeros(0)
    n = max(1, int((L - 2 * margin * 0.5) // PITCH))
    return (np.arange(n) + 0.5) * L / n


# ── The building ─────────────────────────────────────────────────────────────
def facade_columns(stone, metal, glass, a, e, n, L, y0, y1, nfl, solids, margin=1.1):
    """Window columns of one wall: aluminium strip, a pane per floor, two chrome mullion lines."""
    fh = (y1 - y0) / nfl
    for s in column_positions(a, e, L, margin):
        if s < 0.9 or s > L - 0.9:
            continue
        q = a + e * s + n * 0.5
        if solids.inside(q[None], (y0 + y1) / 2)[0]:
            continue
        w = 1.35
        metal.append(wall_quad(a, e, n, s - w / 2, s + w / 2, y0 + 0.5, y1 - 0.7, 0.04, ALU))
        for side in (-1, 1):
            metal.append(wall_quad(a, e, n, s + side * (w / 2 + 0.06) - 0.06, s + side * (w / 2 + 0.06) + 0.06,
                                   y0 + 0.3, y1 - 0.5, 0.06, CHROME))
        sp = s + PITCH / 2                              # the limestone pier between two columns, in relief
        if sp < L - 1.0 and not solids.inside((a + e * sp + n * 0.5)[None], (y0 + y1) / 2)[0]:
            stone.append(edge_box(a, e, n, sp - 0.22, sp + 0.22, y0 + 0.2, y1 - 0.5, -0.05, 0.16, LIME_L))
        for k in range(nfl):
            fy = y0 + k * fh
            if fy + 2.3 > y1 - 0.6:
                continue
            glass.append(wall_quad(a, e, n, s - 0.52, s + 0.52, fy + 0.85, fy + min(2.5, fh - 0.6), 0.08, PANE))


def tower(solids):
    stone, metal, glass = [], [], []
    for name, P, y0, y1, nfl in TOWER:
        G = ccw(P)
        stone.append(walls(G, y0, y1, LIME))
        stone.append(cap(G, y1, TERR))
    # the 86th-floor observatory block (way 137425142): glazed all round
    G = ccw(P142)
    stone.append(walls(G, OBS86, 330.0, LIME))
    stone.append(cap(G, 330.0, TERR))

    for name, P, y0, y1, nfl in TOWER:
        G = ccw(P)
        for a, e, n, L in edges(G):
            if L < 0.8:
                continue
            # coping / parapet on the open roof edges
            ns = max(1, int(round(L / 0.7)))
            ss = (np.arange(ns) + 0.5) * L / ns
            pts = a + np.outer(ss, e) - n * 0.3
            open_top = ~solids.inside(pts, y1 + 0.5)
            for i0, i1 in runs(open_top):
                s0, s1 = i0 * L / ns - (0.25 if i0 == 0 else 0), i1 * L / ns + (0.25 if i1 == ns else 0)
                h = 1.6 if name == "w144" else 1.1
                stone.append(edge_box(a, e, n, s0, s1, y1 - 0.5, y1 + h, -0.35, 0.25, LIME_L))
            # Art Deco band under the coping
            for i0, i1 in runs(open_top):
                s0, s1 = i0 * L / ns, i1 * L / ns
                stone.append(edge_box(a, e, n, s0, s1, y1 - 1.6, y1 - 0.5, -0.05, 0.12, LIME_D))
            if name != "w144":
                facade_columns(stone, metal, glass, a, e, n, L, y0, y1, nfl, solids)
            else:
                facade_columns(stone, metal, glass, a, e, n, L, y0, y1, 2, solids)
    # observatory block windows
    for a, e, n, L in edges(ccw(P142)):
        cs = column_positions(a, e, L, 0.8)
        for s in cs:
            if 0.8 < s < L - 0.8:
                glass.append(wall_quad(a, e, n, s - 1.1, s + 1.1, OBS86 + 1.0, 328.6, 0.05, PANE))
        stone.append(edge_box(a, e, n, -0.2, L + 0.2, 329.4, 330.8, -0.3, 0.2, LIME_L))
    return stone, metal, glass


def deck_fence(stone, metal):
    """The parapet and the curved steel safety fence of the 86th-floor deck (posts and rails)."""
    G = ccw(offset(ccw(P144), -0.5))
    for a, e, n, L in edges(G):
        k = max(2, int(L // 1.8) + 1)
        for s in np.linspace(0.2, L - 0.2, k):
            p = a + e * s
            metal.append(box(0.12, 2.6, 0.12, MAST_D).T(p[0], OBS86 + 1.4, p[1]))
            # the inward curving top of the fence
            q = p - n * 0.6
            metal.append(box(0.1, 0.1, 0.1, MAST_D).T(q[0], OBS86 + 3.9, q[1]))
        for y in (OBS86 + 2.6, OBS86 + 3.9):
            off = -0.6 if y > OBS86 + 3 else 0.0
            metal.append(edge_box(a, e, n, 0.0, L, y, y + 0.08, off - 0.05, off + 0.05, MAST_D))
    return stone, metal


def base(solids):
    """The five-storey base filling the lot, with the Fifth Avenue entrance and the shop fronts."""
    stone, metal, glass = [], [], []
    G = ccw(LOT)
    stone.append(walls(G, FOUND, BASE, LIME))
    stone.append(cap(G, BASE, TERR))
    for a, e, n, L in edges(G):
        # granite plinth, cornice, parapet
        stone.append(edge_box(a, e, n, -0.3, L + 0.3, FOUND, 0.9, -0.2, 0.15, LIME_D))
        stone.append(edge_box(a, e, n, -0.35, L + 0.35, G0 - 0.4, G0 + 0.4, -0.1, 0.35, LIME_L))
        stone.append(edge_box(a, e, n, -0.4, L + 0.4, BASE - 0.6, BASE + 1.2, -0.3, 0.4, LIME_L))
        front = n[1] > 0.9
        side = abs(n[0]) > 0.9
        # shop fronts between granite piers
        pitch = 4.6
        k = int((L - 2.0) // pitch)
        for i in range(k):
            s = 1.0 + (L - 2.0 - k * pitch) / 2 + (i + 0.5) * pitch
            p = a + e * s
            if front and -6.5 < p[0] < 10.5:
                continue
            if side and abs(p[1]) < 4.0:
                continue
            glass.append(wall_quad(a, e, n, s - 1.7, s + 1.7, 0.9, G0 - 0.9, 0.05, SHOP))
            metal.append(wall_quad(a, e, n, s - 1.7, s + 1.7, G0 - 1.4, G0 - 1.1, 0.08, CHROME))
        # floors 2-5: window columns
        for s in column_positions(a, e, L, 1.6):
            if s < 1.2 or s > L - 1.2:
                continue
            p = a + e * s
            if front and -4.6 < p[0] < 8.6:
                continue
            metal.append(wall_quad(a, e, n, s - 0.75, s + 0.75, G0 + 0.8, BASE - 1.0, 0.04, ALU))
            for f in range(4):
                fy = G0 + f * 3.75
                glass.append(wall_quad(a, e, n, s - 0.6, s + 0.6, fy + 0.9, fy + 2.9, 0.08, PANE))
        # entrances on 33rd and 34th Streets, midblock
        if side:
            # the point of the edge at z = 0
            t = (0.0 - a[1]) / e[1] if abs(e[1]) > 1e-6 else L / 2
            stone.append(edge_box(a, e, n, t - 4.0, t + 4.0, 0.0, 13.0, 0.0, 0.5, LIME_L))
            glass.append(wall_quad(a, e, n, t - 2.6, t + 2.6, 0.0, 9.5, 0.55, SHOP))
            metal.append(edge_box(a, e, n, t - 3.4, t + 3.4, 5.6, 6.0, 0.5, 2.4, CHROME))
    # the Fifth Avenue entrance: a projecting limestone frontispiece rising above the base, the tall recessed
    # portal (OSM marks it between x = -0.7 and 4.53) with its transom window, and the steel canopy
    zf = 64.74
    cx = 1.9
    stone.append(box(14.0, BASE + 4.5 - FOUND, 0.9, LIME_L).T(cx, FOUND, zf + 0.45))
    stone.append(box(9.0, 1.2, 0.5, LIME).T(cx, BASE + 4.5, zf + 0.25))
    glass.append(box(6.0, 16.5, 0.2, SHOP).T(cx, 0.0, zf + 0.82))
    for dx in (-2.0, 0.0, 2.0):                              # mullions of the tall window
        metal.append(box(0.14, 9.5, 0.1, CHROME).T(cx + dx, 7.0, zf + 0.95))
    metal.append(box(7.2, 0.45, 2.6, CHROME).T(cx, 5.2, zf + 2.2))         # canopy
    metal.append(box(12.0, 0.6, 0.12, CHROME).T(cx, 18.4, zf + 0.95))     # lettering band
    for dx in (-5.6, 5.6):                                   # flanking pylons
        stone.append(box(1.6, BASE + 6.0, 1.3, LIME_L).T(cx + dx, 0.0, zf + 0.65))
        metal.append(cyl(0.08, 0.05, 6.0, CHROME, 6).T(cx + dx, BASE + 6.0, zf + 0.65))    # flagpoles
    return stone, metal, glass


def octagon(r_flat, y0, y1, c, caps=True):
    R = r_flat / math.cos(math.pi / 8)
    m = cyl(R, R, y1 - y0, c, 8, caps)
    return m.R(0, math.pi / 8, 0).T(MAST_C[0], y0, MAST_C[1])


def mast():
    """The mooring mast: octagonal shaft with four stepped buttresses, lit upper floors, 102nd floor, cap."""
    mt, gl = [], []
    cx, cz = MAST_C
    mt.append(octagon(6.6, 330.0, 352.0, MAST))
    mt.append(octagon(5.9, 352.0, 362.0, MAST))
    mt.append(octagon(5.2, 362.0, F102, MAST))
    # vertical ribs on the eight faces of the shaft
    for i in range(8):
        a = i * math.pi / 4
        for dr, y0, y1, rf in ((0.3, 330.0, 352.0, 6.6), (0.25, 352.0, 362.0, 5.9), (0.2, 362.0, F102, 5.2)):
            for t in (-1.6, 0.0, 1.6):
                if rf < 5.5 and t:
                    t *= 0.8
                px = cx + math.cos(a) * rf + math.cos(a + math.pi / 2) * t
                pz = cz + math.sin(a) * rf + math.sin(a + math.pi / 2) * t
                rib = box(0.35, y1 - y0 - 0.6, 0.35 + 2 * dr, MAST_D).R(0, -a + math.pi / 2, 0)
                mt.append(rib.T(px, y0 + 0.3, pz))
            # the windows between the ribs: lit glazing of the upper floors
            for t in (-0.8, 0.8):
                px = cx + math.cos(a) * (rf + 0.06) + math.cos(a + math.pi / 2) * t * (rf / 6.6)
                pz = cz + math.sin(a) * (rf + 0.06) + math.sin(a + math.pi / 2) * t * (rf / 6.6)
                col = LIT if y0 >= 352.0 else PANE
                pane = box(1.0 * rf / 6.6, y1 - y0 - 2.0, 0.08, col).R(0, -a + math.pi / 2, 0)
                gl.append(pane.T(px, y0 + 1.0, pz))
    # four stepped buttresses on the diagonals (the Art Deco wings of the crown)
    for i in range(4):
        a = math.pi / 4 + i * math.pi / 2
        for out, y1, w in ((9.5, 333.0, 3.4), (8.6, 338.0, 3.0), (7.8, 344.0, 2.6), (7.1, 351.0, 2.2),
                           (6.5, 356.5, 1.8)):
            b = box(out, y1 - 326.0, w, MAST).R(0, -a, 0)
            mt.append(b.T(cx + math.cos(a) * out / 2, 326.0, cz + math.sin(a) * out / 2))
        # small stepped finials on the shoulders
        for out, y0, y1, w in ((9.0, 333.0, 334.2, 1.2), (8.1, 338.0, 339.2, 1.1), (7.3, 344.0, 345.2, 1.0)):
            b = box(1.0, y1 - y0, w, MAST_D).R(0, -a, 0)
            mt.append(b.T(cx + math.cos(a) * out, y0, cz + math.sin(a) * out))
    # cornice rings
    for y, r in ((352.0, 7.0), (362.0, 6.3)):
        mt.append(octagon(r, y - 0.5, y + 0.4, MAST_D))
    # 102nd-floor observatory: a glazed ring and the stepped cap up to the roof (381 m)
    mt.append(octagon(5.6, F102 - 0.4, F102 + 0.3, MAST_D))
    gl.append(octagon(4.3, F102 + 0.3, F102 + 3.6, PANE, caps=False))
    for i in range(8):                                       # mullions of the 102nd floor
        a = i * math.pi / 4 + math.pi / 8
        R = 4.3 / math.cos(math.pi / 8) + 0.05
        mt.append(box(0.18, 3.3, 0.18, CHROME).T(cx + math.cos(a) * R, F102 + 0.3, cz + math.sin(a) * R))
    mt.append(octagon(4.6, F102 + 3.6, F102 + 4.4, MAST))
    mt.append(octagon(3.7, F102 + 4.4, F102 + 6.4, MAST))
    mt.append(octagon(2.8, F102 + 6.4, ROOF, MAST))
    return mt, gl


def antenna():
    cx, cz = MAST_C
    out = []
    for r0, r1, y0, y1, seg in ((2.3, 2.0, ROOF, 385.0, 12), (1.5, 1.3, 385.0, 398.0, 10), (1.0, 0.85, 398.0, 412.0, 10),
                                (0.65, 0.5, 412.0, 426.0, 8), (0.38, 0.25, 426.0, 436.0, 8), (0.16, 0.05, 436.0, TIP, 6)):
        out.append(cyl(r0, r1, y1 - y0, ANT, seg).T(cx, y0, cz))
    for y, r in ((394.0, 2.4), (409.0, 1.8), (423.0, 1.2)):         # small platforms / antenna arrays
        out.append(cyl(r, r, 0.4, MAST_D, 12).T(cx, y, cz))
        for i in range(4):
            a = i * math.pi / 2 + math.pi / 4
            out.append(box(0.5, 3.0, 0.3, MAST_D).R(0, -a, 0).T(cx + math.cos(a) * r * 0.8, y + 0.4, cz + math.sin(a) * r * 0.8))
    return out


FOOTPRINT = offset(ccw(LOT), 1.0)


def main():
    out = ROOT / "models" / "Q9188"
    out.mkdir(parents=True, exist_ok=True)
    solids = Solids()
    solids.add(LOT, FOUND, BASE)
    for name, P, y0, y1, nfl in TOWER:
        solids.add(P, y0, y1)
    solids.add(P142, OBS86, 330.0)
    bs, bm, bg = base(solids)
    ts, tm, tg = tower(solids)
    ds, dm = deck_fence([], [])
    ms, mg = mast()
    an = antenna()
    stone = merge(bs + ts + ds)
    metal = merge(bm + tm + dm + ms + an)
    glass = merge(bg + tg + mg)
    parts = [("limestone", stone), ("metal", metal, {"material": "metal"}), ("glass", glass, {"material": "glass"})]
    tris, size = write_glb(out / "model.glb", parts, materials={"glass": GLASS, "metal": METAL},
                           roughness=0.85)
    allv = np.vstack([stone.v, metal.v, glass.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.1f}..{hi[0]:.1f}, "
          f"y {lo[1]:.1f}..{hi[1]:.1f}, z {lo[2]:.1f}..{hi[2]:.1f}")
    meta = out / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = tris
        mj["footprint"] = [[round(x, 1), round(z, 1)] for x, z in FOOTPRINT]
        meta.write_bytes((json.dumps(mj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
