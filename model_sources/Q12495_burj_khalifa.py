"""
Burj Khalifa (Q12495), Downtown Dubai — the tower, its spire and the podium at its foot, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Skidmore, Owings & Merrill (Adrian Smith, design partner; William F. Baker, structural engineer;
George J. Efstathiou; Marshall Strabala), 2004–2010, for Emaar Properties — protected (modern work).

What is modelled:
- The tower: the Y-shaped «Hymenocallis» plan — three wings at 120° round a buttressed hexagonal core — built
  parametrically: every wing is a straight arm about 18–25 m wide with a rounded, slightly pointed end, and
  between the wings the faces of the core. Each wing steps back in a spiral as it rises (clockwise seen from
  above: the north-west wing first, then the north-east one, then the south one), the 27 setbacks at the
  heights mapped in OpenStreetMap; above the last wing setbacks (520 m) the six-lobed core goes on stepping back
  in the same spiral, and from the last office/maintenance tiers the slender steel spire rises to 828 m.
- The cladding: silver-blue reflective glass with the vertical stainless-steel fins (as light vertical stripes,
  one every 1.6 m), the darker louvred bands of the mechanical floors (17–18, 40–42, 73–75, 109–111, 136–138,
  155, 160–163), the observation levels (At the Top, floors 124–125 at 452 m; At the Top Sky, floor 148 at
  555 m), light grey setback terraces. Above the last occupied floor (585.4 m) the stepped crown and spire are
  steel and stainless cladding (material METAL); below, the curtain wall is GLASS.
- The podium at the foot: the three low glass pavilions that prolong the wings to their points (the Armani
  Hotel, residential and corporate entrances, 15 m high per OSM) and the small pieces between the wings.
- Not modelled: the Dubai Fountain, the Burj Khalifa Lake and the reflecting pools of the Burj Park (they lie
  80–300 m away to the south-east, part of the plaza, irregular shapes; left to the map), the Dubai Mall, the
  interiors, the window-cleaning machines.

Measurements (see meta.json for the sources):
- Wikipedia (en): architectural height 828 m, tip 829.8 m, roof 739.4 m, top floor 585.4 m, 154 usable floors +
  9 maintenance = 163; At the Top floor 124 at 452 m, At the Top Sky floor 148 at 555 m, lounge 152–154 at
  585 m; 27 setbacks in a spiral; Y-shaped tripartite floor plan with a buttressed central core; reflective glass
  with stainless-steel spandrels and vertical tubular fins; mechanical floors 17–18, 40–42, 73–75, 109–111,
  136–138, 155 and 160–163; the plan inspired by the Hymenocallis flower.
- OpenStreetMap relation 7584462 (type=building), outline way 446646206 and its 36 building:part ways: the
  crescent of every setback with its height (wing tips, measured from the axis of the spire):
    south wing (heading)   62.4 m to 35 m, 52.0 to 130, 42.9 to 235, 34.6 to 360, 25.4 to 520, 15.0 to 625
    north-west wing        62.7 m to 35 m, 52.3 to 70,  43.1 to 160, 34.8 to 270, 25.5 to 405, 15.1 to 645
    north-east wing        63.3 m to 35 m, 52.9 to 105, 43.8 to 200, 35.4 to 315, 26.2 to 460, 15.7 to 605,
                           9.3 to 710 (the other two wings' last tier, missing in OSM, assumed the same)
    core lobes between the wings (north, south-east, south-west): 18.6 / 18.5 / 18.2 m to 545 / 560 / 580 m,
                           11.6 / 11.5 / 11.2 m to 660 / 680 / 700 m, 4.9 / 5.2 / 5.0 m to 720 / 740 / 760 m
  The OSM heights are round figures (±5 m); the shapes of the lobes (width of the wings growing from ≈ 18 m at
  the base to ≈ 25 m in the upper tiers, the pointed rounded ends) are fitted to the OSM crescents (±1 m).
- Floor heights between the published ones are interpolated (0 → 452 m over 124 floors, 452 → 555 m over 24,
  555 → 585.4 m over 6, 4 m per maintenance floor above): the bands of the mechanical floors are ±3 m.
- Spire: from the top of the core tiers (760 m) to 828 m, a tapering steel tube (the central pinnacle pipe,
  Wikipedia); its diameter (5 m at the foot, 0.5 m at the top) is an estimate (±30 %).
- Podium: the six 15 m building:part ways of the same relation, as mapped.

Heading: the main façade (+Z) is the end of the south wing, which points to 185.7° — towards the Burj Khalifa
Lake, the Dubai Fountain and the Dubai Mall (the lake lies 170–320 m to the south-south-east). The wing bearing
is the mean of the 15 wing-tip points of the OSM setbacks taken modulo 120° (65.7° / 185.7° / 305.7°, all within
±0.7°). Origin = the axis of the spire (centroid of the 828 m part, way 137147715), 1.4 m north of the centroid
of the whole outline.

Axes (docs/MODEL_SPEC.md): +Y up, the south wing points to +Z (heading 185.7°), +X points to 95.7° (east).

    python model_sources/Q12495_burj_khalifa.py   -> models/Q12495/model.glb (+ footprint, triangles in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, lathe, cyl, cap, walls, offset, write_glb, GLASS, METAL  # noqa: E402

HEADING = 185.7
FOUND = -3.0

# ── Colours ──────────────────────────────────────────────────────────────────
GLASS_C = (0.54, 0.63, 0.71)          # reflective silver-blue glass
FIN_C = (0.78, 0.80, 0.82)            # stainless-steel fins
MECH_C = (0.45, 0.49, 0.53)           # louvred mechanical floors
DECK_C = (0.33, 0.42, 0.52)           # observation levels (clear glass, darker)
TERRACE_C = (0.70, 0.70, 0.68)        # setback terraces
CROWN_C = (0.66, 0.71, 0.76)          # crown panels above the last floor
CROWN_FIN = (0.86, 0.87, 0.89)
CROWN_MECH = (0.56, 0.59, 0.62)      # louvres of the maintenance floors in the crown
SPIRE_C = (0.80, 0.82, 0.84)
PODIUM_C = (0.42, 0.52, 0.58)         # podium glass
PODIUM_ROOF = (0.80, 0.78, 0.73)      # podium roofs (light stone, terraces)

TOP_FLOOR = 585.4

# ── Plan: lobes with their setbacks [(tip radius, up to height)] ─────────────
# rel = direction of the lobe, degrees clockwise (seen from above) from +Z (the south wing).
WINGS = [
    (0.0, [(62.4, 35), (52.0, 130), (42.9, 235), (34.6, 360), (25.4, 520), (15.0, 625), (9.3, 710)]),     # S
    (120.0, [(62.7, 35), (52.3, 70), (43.1, 160), (34.8, 270), (25.5, 405), (15.1, 645), (9.3, 710)]),    # NW
    (240.0, [(63.3, 35), (52.9, 105), (43.8, 200), (35.4, 315), (26.2, 460), (15.7, 605), (9.3, 710)]),   # NE
]
LOBES = [
    (180.0, [(18.6, 545), (11.6, 660), (4.9, 720)]),       # N, between the NW and NE wings
    (300.0, [(18.5, 560), (11.5, 680), (5.2, 740)]),       # SE
    (60.0, [(18.2, 580), (11.2, 700), (5.0, 760)]),        # SW
]
CORE = [(8.0, 710), (4.6, 760)]                            # the hexagonal core under the lobes
SPIRE_FOOT, SPIRE_TOP = 760.0, 828.0


def direction(rel):
    a = math.radians(rel)
    return np.array([-math.sin(a), math.cos(a)])           # (x, z): bearing HEADING + rel


def lobe_poly(rel, R, b, a, p, n=10):
    """A convex arm from the core to a rounded, slightly pointed end at radius R: half-width b, end length a,
    end profile w = b·sin(π/2·t)^p (t = distance from the tip / a)."""
    a = min(a, R + 2.0)
    us = [R - a * (1 - k / n) for k in range(n)]           # from the start of the end to just before the tip
    right = [(-2.0, -b)] + [(u, -b * math.sin(math.pi / 2 * (R - u) / a) ** p) for u in us] + [(R, 0.0)]
    left = [(u, -w) for u, w in reversed(right[:-1])]
    loc = np.array(right + left)
    d = direction(rel)
    perp = np.array([d[1], -d[0]])
    return loc[:, :1] * d + loc[:, 1:] * perp


def wing_shape(R):
    """Half-width and end length of a wing whose tip is at R (fitted to the OSM crescents)."""
    b = min(8.9 + (62.8 - R) * 0.1, 0.62 * R)
    return b, min(13.0, 0.75 * R)


def ray_r(poly, phi):
    """Distance from the origin along the direction phi to where it leaves the convex polygon (0 if it misses)."""
    d = np.array([math.cos(phi), math.sin(phi)])
    A, B = poly, np.roll(poly, -1, 0)
    E = B - A
    den = d[0] * E[:, 1] - d[1] * E[:, 0]
    ok = np.abs(den) > 1e-12
    t = np.where(ok, (A[:, 0] * E[:, 1] - A[:, 1] * E[:, 0]) / np.where(ok, den, 1), -1)
    s = np.where(ok, (A[:, 0] * d[1] - A[:, 1] * d[0]) / np.where(ok, den, 1), -1)
    t = t[(s >= -1e-9) & (s <= 1 + 1e-9)]
    return float(t.max()) if len(t) else 0.0


def envelope(polys):
    """Outline of the union of convex polygons that all contain the origin (a star-shaped ring), exact:
    every vertex of a lobe on the outside, and the crossings of the lobes' edges."""
    def best(phi):
        rs = [ray_r(P, phi) for P in polys]
        k = int(np.argmax(rs))
        return k, rs[k]

    def line_pt(k, phi):
        r = ray_r(polys[k], phi)
        return np.array([math.cos(phi), math.sin(phi)]) * r

    angs = sorted({math.atan2(p[1], p[0]) % (2 * math.pi) for P in polys for p in P})
    angs.append(angs[0] + 2 * math.pi)
    out = []
    eps = 1e-7

    def refine(a0, a1, depth):
        k0, _ = best(a0 + eps)
        k1, _ = best(a1 - eps)
        if k0 == k1 or depth > 8:
            return
        p0, p1 = line_pt(k0, a0 + eps), line_pt(k0, a1 - eps)
        q0, q1 = line_pt(k1, a0 + eps), line_pt(k1, a1 - eps)
        d1, d2 = p1 - p0, q1 - q0
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-12:
            return
        t = ((q0[0] - p0[0]) * d2[1] - (q0[1] - p0[1]) * d2[0]) / den
        X = p0 + t * d1
        ax = math.atan2(X[1], X[0]) % (2 * math.pi)
        if ax < a0 - 1e-9:
            ax += 2 * math.pi
        if not a0 < ax < a1:
            return
        refine(a0, ax, depth + 1)
        out.append(X)
        refine(ax, a1, depth + 1)

    for a0, a1 in zip(angs, angs[1:]):
        _, r = best(a0)
        out.append(np.array([math.cos(a0), math.sin(a0)]) * r)
        refine(a0, a1, 0)
    # drop collinear points
    P = out
    changed = True
    while changed:
        changed = False
        Q = []
        n = len(P)
        for i in range(n):
            a, b, c = P[i - 1], P[i], P[(i + 1) % n]
            ab, ac = b - a, c - a
            L = np.linalg.norm(ac)
            if L > 1e-9 and abs(ab[0] * ac[1] - ab[1] * ac[0]) / L < 0.02 and np.dot(ab, ac) > 0 \
                    and np.linalg.norm(ab) < L:
                changed = True
                continue
            if np.linalg.norm(b - a) < 0.05:
                changed = True
                continue
            Q.append(b)
        P = Q
    return [tuple(map(float, p)) for p in P]


def radius_at(sched, y1):
    for r, top in sched:
        if top >= y1 - 1e-6:
            return r
    return None


def plan(y1):
    """The outline of the tower in the tier that ends at height y1."""
    polys = []
    for rel, sched in WINGS:
        R = radius_at(sched, y1)
        if R:
            b, a = wing_shape(R)
            polys.append(lobe_poly(rel, R, b, a, 0.8))
    for rel, sched in LOBES:
        R = radius_at(sched, y1)
        if R:
            s = R / 18.4
            polys.append(lobe_poly(rel, R, 12.3 * s, 10.0 * s, 1.0))
    c = radius_at(CORE, y1)
    if c:
        polys.append(np.array([(c * math.cos(math.radians(30 + 60 * k)), c * math.sin(math.radians(30 + 60 * k)))
                               for k in range(6)]))
    return envelope(polys)


# ── Floors and bands ─────────────────────────────────────────────────────────
def floor_y(n):
    """Height of the floor of storey n (interpolated between the published levels)."""
    pts = [(0, 0.0), (124, 452.0), (148, 555.0), (154, TOP_FLOOR)]
    if n >= 154:
        return TOP_FLOOR + 4.0 * (n - 154)
    for (n0, y0), (n1, y1) in zip(pts, pts[1:]):
        if n <= n1:
            return y0 + (y1 - y0) * (n - n0) / (n1 - n0)


MECH = [(17, 18), (40, 42), (73, 75), (109, 111), (136, 138), (155, 155), (160, 163)]
DECKS = [(124, 125), (148, 148)]
BANDS = sorted([(floor_y(a), floor_y(b + 1), MECH_C) for a, b in MECH] +
               [(floor_y(a), floor_y(b + 1), DECK_C) for a, b in DECKS])


def band_colour(y0, y1):
    m = (y0 + y1) / 2
    for a, b, c in BANDS:
        if a <= m < b:
            return c
    return None


def facade(ring, y0, y1, crown=False, fin_step=1.6, fin_w=0.32):
    """The curtain wall of one tier: glass panels with a stainless fin every fin_step metres, cut by the bands."""
    cuts = sorted({y0, y1} | {y for a, b, _ in BANDS for y in (a, b) if y0 < y < y1})
    pts = []                     # subdivided ring with a flag: True where the segment that starts here is a fin
    n = len(ring)
    for i in range(n):
        p, q = np.array(ring[i]), np.array(ring[(i + 1) % n])
        L = float(np.linalg.norm(q - p))
        k = max(1, int(round(L / fin_step)))
        if L < 2 * fin_w + 0.3:
            pts.append((p, False)); continue
        for j in range(k):
            t0 = j / k
            t1 = (j + 1) / k - fin_w / L
            pts.append((p + (q - p) * t0, False))
            pts.append((p + (q - p) * t1, True))
    glass_c, fin_c = (CROWN_C, CROWN_FIN) if crown else (GLASS_C, FIN_C)
    V, F, C = [], [], []
    for ya, yb in zip(cuts, cuts[1:]):
        bc = band_colour(ya, yb)
        for i in range(len(pts)):
            (p, fin), (q, _) = pts[i], pts[(i + 1) % len(pts)]
            o = len(V)
            V += [(p[0], ya, p[1]), (q[0], ya, q[1]), (q[0], yb, q[1]), (p[0], yb, p[1])]
            if bc is not None and crown:
                bc = CROWN_MECH
            col = bc if bc is not None else (fin_c if fin else glass_c)
            C += [col] * 4
            F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]       # the ring runs clockwise in (x, z): faces out
    return Mesh(V, F, np.array(C))


def fan_cap(ring, y, c):
    """A flat top over a star-shaped ring, as a fan from the axis."""
    V = [(0.0, y, 0.0)] + [(x, y, z) for x, z in ring]
    n = len(ring)
    F = [(0, 1 + (i + 1) % n, 1 + i) for i in range(n)]
    m = Mesh(V, F, c)
    # make it face up
    a, b, d = m.v[m.f[0]]
    if np.cross(b - a, d - a)[1] < 0:
        m.f = m.f[:, ::-1]
    return m


def tower():
    tops = sorted({t for _, s in WINGS + LOBES for _, t in s} | {t for _, t in CORE})
    glass, crown, terr = [], [], []
    y0 = FOUND
    for y1 in tops:
        ring = plan(y1)
        if ring_area_xz(ring) > 0:          # facade() wants the ring clockwise in (x, z)
            ring = ring[::-1]
        if y0 < TOP_FLOOR < y1:
            glass.append(facade(ring, y0, TOP_FLOOR))
            crown.append(facade(ring, TOP_FLOOR, y1, crown=True))
        elif y1 <= TOP_FLOOR:
            glass.append(facade(ring, y0, y1))
        else:
            crown.append(facade(ring, y0, y1, crown=True, fin_step=1.2))
        if y1 < TOP_FLOOR:
            terr.append(fan_cap(ring, y1, TERRACE_C))
        else:
            crown.append(fan_cap(ring, y1, CROWN_FIN))
        y0 = y1
    return merge(glass), merge(crown), merge(terr)


def ring_area_xz(P):
    P = np.asarray(P)
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def spire():
    """The steel pinnacle from the top of the core tiers to 828 m, with a few collars."""
    h = SPIRE_TOP - SPIRE_FOOT
    prof = [(2.5, 0.0), (2.2, 8.0), (1.7, 22.0), (1.2, 38.0), (0.8, 52.0), (0.45, 62.0), (0.25, h)]
    ms = [lathe(prof, SPIRE_C, seg=16).T(0, SPIRE_FOOT, 0),
          cyl(0.08, 0.02, 1.8, SPIRE_C, seg=6).T(0, SPIRE_TOP, 0)]
    for y, r in ((8.0, 2.45), (22.0, 1.95), (38.0, 1.45)):
        ms.append(cyl(r, r, 0.8, CROWN_FIN, seg=16).T(0, SPIRE_FOOT + y - 0.4, 0))
    return merge(ms)


# ── Podium (OpenStreetMap building:parts of 15 m, heading 185.7, origin on the spire axis) ─────────
OUTLINE = [(-15.73, 20.18), (-12.39, 71.67), (0.68, 94.78), (3.98, 88.47), (12.90, 71.41), (15.18, 19.06), (16.06, 19.35), (18.48, 28.86), (22.68, 27.39), (26.48, 25.10), (29.73, 22.08), (32.30, 18.46), (34.06, 14.40), (34.97, 10.06), (34.95, 5.64), (34.02, 1.31), (24.93, 4.25), (24.47, 3.52), (66.07, -20.82), (81.17, -42.69), (54.71, -43.30), (9.30, -23.43), (9.04, -24.32), (15.64, -31.13), (11.63, -34.16), (7.06, -36.24), (2.13, -37.28), (-2.90, -37.23), (-7.82, -36.08), (-12.35, -33.90), (-16.30, -30.79), (-8.64, -23.91), (-8.91, -23.05), (-51.70, -44.74), (-78.33, -45.86), (-64.83, -23.23), (-24.57, 4.29), (-25.35, 5.12), (-34.18, 2.19), (-35.11, 6.64), (-35.04, 11.20), (-33.98, 15.63), (-31.93, 19.70), (-29.46, 23.45), (-26.23, 26.57), (-22.38, 28.92), (-18.11, 30.36), (-16.71, 20.85)]
PODIUM = [
    ("way/137153127", [(-15.73, 20.18), (-12.39, 71.67), (0.68, 94.78), (3.98, 88.47), (12.90, 71.41), (15.18, 19.06), (12.49, 18.34), (12.86, 19.52), (13.07, 21.06), (12.98, 22.95), (12.32, 24.92), (11.05, 27.26), (11.68, 29.64), (11.45, 31.97), (10.69, 34.28), (9.21, 36.52), (10.14, 38.97), (10.26, 40.47), (9.86, 42.30), (8.97, 44.16), (7.75, 45.87), (8.35, 47.09), (8.52, 49.26), (8.04, 51.68), (6.39, 54.90), (3.56, 58.64), (-0.83, 62.38), (-5.08, 58.85), (-7.80, 55.04), (-9.33, 51.76), (-9.73, 49.31), (-9.47, 47.17), (-8.84, 45.97), (-10.01, 44.22), (-10.84, 42.34), (-11.17, 40.49), (-11.01, 38.99), (-10.00, 36.58), (-11.41, 34.28), (-12.07, 31.95), (-12.24, 29.62), (-11.52, 27.27), (-12.71, 24.89), (-13.31, 22.88), (-13.33, 21.00), (-13.07, 19.48), (-12.67, 18.30)]),
    ("way/1094648913", [(34.02, 1.31), (24.93, 4.25), (24.47, 3.52), (22.66, 1.15), (21.44, 2.90), (19.81, 4.33), (17.77, 5.35), (16.35, 5.88), (16.25, 8.94), (13.56, 10.31), (13.92, 12.07), (13.98, 14.34), (13.47, 16.43), (12.49, 18.34), (15.18, 19.06), (16.06, 19.35), (18.48, 28.86), (22.68, 27.39), (26.48, 25.10), (29.73, 22.08), (32.30, 18.46), (34.06, 14.40), (34.97, 10.06), (34.95, 5.64)]),
    ("way/1094648914", [(-31.93, 19.70), (-29.46, 23.45), (-26.23, 26.57), (-22.38, 28.92), (-18.11, 30.36), (-16.71, 20.85), (-15.73, 20.18), (-12.67, 18.30), (-13.59, 16.37), (-14.02, 14.26), (-13.89, 12.00), (-13.64, 10.49), (-15.60, 9.36), (-15.74, 6.71), (-17.45, 6.18), (-19.47, 5.12), (-21.07, 3.65), (-22.25, 1.88), (-24.57, 4.29), (-25.35, 5.12), (-34.18, 2.19), (-35.11, 6.64), (-35.04, 11.20), (-33.98, 15.63)]),
    ("way/1094648915", [(-8.91, -23.05), (-51.70, -44.74), (-78.33, -45.86), (-64.83, -23.23), (-24.57, 4.29), (-22.25, 1.88), (-23.47, 1.61), (-24.93, 1.04), (-26.53, 0.04), (-27.95, -1.49), (-29.38, -3.73), (-31.78, -4.33), (-33.70, -5.68), (-35.35, -7.45), (-36.59, -9.84), (-39.20, -10.21), (-40.57, -10.87), (-41.99, -12.10), (-43.18, -13.77), (-44.09, -15.67), (-45.45, -15.75), (-47.43, -16.64), (-49.32, -18.25), (-51.34, -21.24), (-53.23, -25.52), (-54.36, -31.17), (-49.20, -33.14), (-44.50, -33.62), (-40.89, -33.36), (-38.56, -32.52), (-36.80, -31.25), (-36.06, -30.11), (-33.95, -30.28), (-31.89, -30.07), (-30.11, -29.46), (-28.86, -28.59), (-27.24, -26.54), (-24.54, -26.64), (-22.17, -26.09), (-20.05, -25.09), (-18.32, -23.32), (-15.65, -23.18), (-13.61, -22.73), (-11.95, -21.83), (-10.74, -20.86), (-9.90, -19.93)]),
    ("way/1094648916", [(-16.30, -30.79), (-8.64, -23.91), (-8.91, -23.05), (-9.90, -19.93), (-7.76, -19.79), (-5.70, -19.15), (-3.77, -17.91), (-2.59, -16.98), (0.03, -18.65), (2.63, -17.46), (3.99, -18.64), (5.93, -19.84), (8.02, -20.45), (10.16, -20.54), (9.30, -23.43), (9.04, -24.32), (15.64, -31.13), (11.63, -34.16), (7.06, -36.24), (2.13, -37.28), (-2.90, -37.23), (-7.82, -36.08), (-12.35, -33.90)]),
    ("way/1094648917", [(24.47, 3.52), (66.07, -20.82), (81.17, -42.69), (54.71, -43.30), (9.30, -23.43), (10.16, -20.54), (11.03, -21.45), (12.25, -22.41), (13.93, -23.28), (15.99, -23.70), (18.64, -23.77), (20.41, -25.52), (22.56, -26.46), (24.94, -26.98), (27.63, -26.82), (29.30, -28.84), (30.55, -29.69), (32.36, -30.27), (34.42, -30.43), (36.52, -30.23), (37.28, -31.35), (39.06, -32.60), (41.43, -33.39), (45.05, -33.59), (49.72, -33.02), (55.18, -31.10), (54.22, -25.66), (52.23, -21.42), (50.15, -18.46), (48.23, -16.89), (46.22, -16.05), (44.87, -15.99), (43.92, -14.10), (42.69, -12.45), (41.26, -11.24), (39.86, -10.64), (37.25, -10.29), (35.97, -7.93), (34.27, -6.19), (32.32, -4.90), (29.89, -4.32), (28.43, -2.10), (26.98, -0.59), (25.36, 0.37), (23.89, 0.90), (22.66, 1.15)]),
]


def podium():
    w, r = [], []
    for _, ring in PODIUM:
        w.append(walls(ring, FOUND, 15.0, PODIUM_C))
        r.append(cap(ring, 15.0, PODIUM_ROOF))
        w.append(walls(offset(ring, -0.6), 15.0, 15.9, PODIUM_ROOF))     # parapet (outer face)
    return merge(w), merge(r)


FOOTPRINT = offset(OUTLINE, 1.5)


def main():
    out = ROOT / "models" / "Q12495"
    out.mkdir(parents=True, exist_ok=True)
    g, c, t = tower()
    s = spire()
    pw, pr = podium()
    parts = [("tower_glass", g, {"material": "glass"}), ("crown", c, {"material": "metal"}),
             ("spire", s, {"material": "metal"}), ("terraces", t),
             ("podium_glass", pw, {"material": "glass"}), ("podium_roofs", pr)]
    tris, size = write_glb(out / "model.glb", parts, materials={"glass": GLASS, "metal": METAL},
                           roughness=0.7, metallic=0.1)
    m = merge([p[1] for p in parts])
    lo, hi = m.v.min(0), m.v.max(0)
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
