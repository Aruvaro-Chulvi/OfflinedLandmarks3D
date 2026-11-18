"""
Golden Gate Bridge (Q44440), San Francisco — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The bridge: chief engineer Joseph Strauss (died 1938); principal engineer Charles Alton Ellis (died 1949), who did
the structural calculations; Leon Moisseiff (died 1943), who championed the final suspension design; consulting
architect Irving Morrow (died 1952): the Art Deco towers, pylons, lamps, railings and the International Orange
colour. Built 1933–1937, opened 27 May 1937. Free to reproduce in the US (no copyright on architecture built
before December 1990), but marked "protected" in meta.json because Ellis (1949) and Morrow (1952) died less than
80 years ago (the project's rule).

Measurements (see meta.json for the sources):
- OpenStreetMap relation 451971 (Golden Gate Bridge, Q44440) and its building:parts. The two tower piers
  (ways 1329971884 and 1330832681, 42.1 × 20.2 m, concrete to 13 m) are 1280.6 m apart on a bearing of 354.70°
  — the main span of 4,200 ft / 1,280 m. The legs (ways 1329558939… 1330832689): base 9.6 × 15.6 m from 13 to
  21 m, then 7.8 × 14.0 m to 74 m, 7.8 × 11.8 m to 123 m, 7.8 × 10.0 m to 162 m, 5.6 × 10.0 m to 195 m and
  3.7 × 7.6 m to 225 m (across × along the bridge), their centres 27.5 m apart; the struts between the legs at
  24–26, 50–52, 111–121, 151–161, 185–193 and 215–223 m; the roadway at 74.65 m and the sidewalks at 75 m
  passing round the outside of the legs. The outline gives the ends of the bridge and their curves (the San
  Francisco end bends east towards the toll plaza, the Marin end west), the pylons S1/S2 at the Fort Point
  arch (z −976…−986, −1090…−1100) and N1/N2 with the north anchorage between them (z 980…1090), the south
  anchorage (z −1100…−1200). Fort Point Light (way 1358282761) marks the fort under the arch.
- Golden Gate Bridge Highway & Transportation District, «Design & construction stats»: total length 8,981 ft
  (2,737 m) abutment to abutment; main span 4,200 ft (1,280 m); side spans 1,125 ft (343 m); towers 746 ft
  (227 m) above the water and 500 ft (152 m) above the roadway; legs 33 × 54 ft (10 × 16 m) at the base;
  main cables 36 3/8 in (0.92 m) across, 27,572 wires; 250 pairs of suspender ropes 50 ft (15.24 m) apart;
  deck 90 ft (27 m) wide, roadway 62 ft (19 m) between curbs, sidewalks 10 ft (3 m); clearance 220 ft (67 m)
  above mean higher high water.
- Wikipedia (en) «Golden Gate Bridge»: International Orange chosen by Irving Morrow; the steel arch over Fort
  Point; the bottom lateral bracing added 1953–54.
- Estimated (±): the sag of the main cables, 144.8 m (475 ft as usually quoted; the cable low point 6 m above
  the roadway, ±3 m); the stiffening truss 7.6 m (25 ft) deep with 7.62 m panels (±10 %); the side-span sag
  from the same horizontal pull (≈ 10 m); the pylons 15 m above the roadway, the Fort Point arch (springings
  6 m, crown of the lower chord 58 m) and the anchorage housings (±20 %); the south tower's concrete fender
  (≈ 92 × 48 m outside, 4.5 m above the water, ±30 %); the grade of the approaches (roadway 68.5 m at the San
  Francisco abutment and 72 m at the Marin one, ±4 m); the ground under the approach viaducts from SRTM 30 m
  (OpenTopoData), the bents taken 8 m below it.

Axes (docs/MODEL_SPEC.md): +Y up; +Z points along the bridge towards Marin (heading 354.7°, the bearing from
the south tower to the north tower), +X points 264.7° (west, the Pacific side). A bridge has no main façade:
the axis was chosen so that the model runs along +Z. Origin at the middle of the main span, on the water:
y = 0 is sea level (the water is the ground here). meta.json gives "ground": "keep", so the terrain at the
two ends (the Presidio bluffs and the Marin headlands, 15–70 m) stays as it is and the piers, pylons,
anchorages and bents go down into it.

    python model_sources/Q44440_golden_gate_bridge.py   -> models/Q44440/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, extrude, tube, write_glb, normals  # noqa: E402

ORANGE = np.array((0.76, 0.27, 0.17))          # International Orange in daylight
ORANGE_D = np.array((0.58, 0.19, 0.12))        # the same in the recesses
CONC = (0.74, 0.72, 0.67)
CONC_D = (0.63, 0.61, 0.57)
ASPH = (0.29, 0.29, 0.30)
WALK = (0.62, 0.60, 0.56)
LINE = (0.86, 0.86, 0.82)
BARRIER = (0.72, 0.71, 0.68)
RED = (0.85, 0.10, 0.08)
LAMP = (0.95, 0.85, 0.55)

# ── Main figures ─────────────────────────────────────────────────────────────
ZT = 640.3                  # towers at z = ±ZT (main span 1280.6 m)
XL = 13.75                  # leg centres and cable planes at x = ±XL (27.5 m)
Y_SADDLE = 225.5            # centre of the main cable over the saddles
SAG = 144.8                 # main span sag (475 ft, estimated)
Y_LOW = Y_SADDLE - SAG      # 80.7: cable low point at mid-span
Z_S1, Z_S2, Z_N1, Z_N2 = -981.0, -1095.0, 985.0, 1085.0     # pylons (OSM)
Y_CBENT = 84.0              # cable where it enters pylons S1 / N1
PANEL = 7.62                # 25 ft truss panels; suspenders every 2 panels (50 ft)
TRUSS = 7.7                 # truss depth below the roadway (67 m clearance at mid-span)
FOUND = 8.0                 # bents and piers go this far below the ground estimate

# Centre line (OSM outline, model x/z): straight between the towers and the anchorages, curving at both ends.
CTRL = [(-58, -1305), (-38, -1270), (-19, -1230), (-5, -1175), (0, -1130), (0, -1000), (0, 0), (0, 1000),
        (0, 1180), (3, 1240), (13, 1290), (27, 1335), (50, 1380), (66, 1404)]
# Roadway level (m above the sea) by z — level on the suspended spans (OSM 74.65), estimated grades at the ends.
ROAD = [(-1305, 68.5), (-1100, 74.65), (1090, 74.65), (1404, 72.0)]
# Ground along the centre line (SRTM 30 m via OpenTopoData), by z.
GROUND = [(-1500, 54), (-1400, 47), (-1300, 40), (-1250, 36), (-1200, 24), (-1150, 15), (-1100, 14), (-1050, 6),
          (-1000, 2), (-950, 4), (-900, 17), (-850, 5), (-800, 0), (800, 0), (850, 24), (900, 62), (950, 60),
          (1000, 60), (1050, 60), (1100, 49), (1150, 32), (1200, 22), (1250, 23), (1300, 28), (1350, 40),
          (1400, 56), (1450, 69)]


def road_y(z):
    return np.interp(z, [p[0] for p in ROAD], [p[1] for p in ROAD])


def ground_y(z):
    return float(np.interp(z, [p[0] for p in GROUND], [p[1] for p in GROUND]))


# ── The centre line, sampled every 0.5 m (Catmull-Rom through CTRL) ─────────
def _path():
    P = np.array(CTRL, float)
    pts = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)]
        L = np.linalg.norm(p2 - p1)
        n = max(2, int(L / 0.5))
        m1 = (p2 - p0) / 2 * (L / max(np.linalg.norm(p2 - p0) / 2, 1e-9)) if i > 0 else p2 - p1
        m2 = (p3 - p1) / 2 * (L / max(np.linalg.norm(p3 - p1) / 2, 1e-9)) if i < len(P) - 2 else p2 - p1
        for t in np.linspace(0, 1, n, endpoint=False):
            h00, h10, h01, h11 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2
            pts.append(h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2)
    pts.append(P[-1])
    pts = np.array(pts)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return pts, s


PTS, SS = _path()
S_END = SS[-1]


def s_of_z(z):
    return float(np.interp(z, PTS[:, 1], SS))


def frame(s):
    """(x, z) of the centre line at s, and the unit normal pointing to +x (left of the direction of travel north)."""
    s = np.asarray(s, float)
    x, z = np.interp(s, SS, PTS[:, 0]), np.interp(s, SS, PTS[:, 1])
    ds = 1.0
    x1, z1 = np.interp(s + ds, SS, PTS[:, 0]), np.interp(s + ds, SS, PTS[:, 1])
    x0, z0 = np.interp(s - ds, SS, PTS[:, 0]), np.interp(s - ds, SS, PTS[:, 1])
    tx, tz = x1 - x0, z1 - z0
    ln = np.hypot(tx, tz)
    return x, z, tz / ln, -tx / ln


def W(u, dy, s):
    """Local (u across, dy above the roadway, s along) -> world points (N,3)."""
    u, dy, s = np.broadcast_arrays(np.asarray(u, float), np.asarray(dy, float), np.asarray(s, float))
    x, z, nx, nz = frame(s)
    return np.stack([x + u * nx, road_y(z) + dy, z + u * nz], -1)


def WA(u, y, s):
    """Like W, with an absolute height y."""
    u, y, s = np.broadcast_arrays(np.asarray(u, float), np.asarray(y, float), np.asarray(s, float))
    x, z, nx, nz = frame(s)
    return np.stack([x + u * nx, y, z + u * nz], -1)


# ── Builders ─────────────────────────────────────────────────────────────────
def bars(P0, P1, w, h, c, caps=False, up=(0.0, 1.0, 0.0)):
    """Many square bars at once, from P0[i] to P1[i], w wide (horizontal) and h high, faces wound outwards."""
    P0, P1 = np.asarray(P0, float).reshape(-1, 3), np.asarray(P1, float).reshape(-1, 3)
    n = len(P0)
    if n == 0:
        return Mesh()
    w = np.broadcast_to(np.asarray(w, float), (n,)); h = np.broadcast_to(np.asarray(h, float), (n,))
    d = P1 - P0
    dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    upv = np.broadcast_to(np.asarray(up, float), (n, 3))
    a = np.cross(dn, upv)
    bad = np.linalg.norm(a, axis=1) < 0.2
    a[bad] = np.cross(dn[bad], np.array([0.0, 0.0, 1.0]))
    bad2 = np.linalg.norm(a, axis=1) < 0.2
    a[bad2] = np.cross(dn[bad2], np.array([1.0, 0.0, 0.0]))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(a, dn)
    corners = []
    for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        off = a * (sa * w / 2)[:, None] + b * (sb * h / 2)[:, None]
        corners.append(off)
    V = np.zeros((n, 8, 3))
    for k in range(4):
        V[:, k] = P0 + corners[k]; V[:, 4 + k] = P1 + corners[k]
    F = []
    for k in range(4):
        j = (k + 1) % 4
        F += [(k, j, 4 + j), (k, 4 + j, 4 + k)]
    if caps:
        F += [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    F = np.array(F)
    faces = (F[None, :, :] + (np.arange(n) * 8)[:, None, None]).reshape(-1, 3)
    verts = V.reshape(-1, 3)
    m = Mesh(verts, faces, c)
    centre = np.repeat((P0 + P1) / 2, len(F), axis=0)
    return orient_out(m, centre)


def orient_out(m, centre):
    """Flip every triangle whose normal points towards centre[i] (a point inside the solid)."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    fn = np.cross(b - a, c - a)
    flip = np.einsum("ij,ij->i", fn, (a + b + c) / 3 - centre) < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def facing(m, hint):
    """Flip triangles so that their normals have a positive dot with hint (a vector or (M,3))."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    fn = np.cross(b - a, c - a)
    hint = np.broadcast_to(np.asarray(hint, float), fn.shape)
    flip = np.einsum("ij,ij->i", fn, hint) < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def strip(A, B, c, hint):
    """Quads between two polylines A[i], B[i] (N,3), facing hint."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    n = len(A)
    V = np.vstack([A, B])
    F = []
    for i in range(n - 1):
        F += [(i, i + 1, n + i + 1), (i, n + i + 1, n + i)]
    return facing(Mesh(V, F, c), hint)


def notched(hx, hz, k):
    """A rectangle 2hx × 2hz with each corner notched back k (the stepped corners of Morrow's towers)."""
    return [(-hx + k, -hz), (hx - k, -hz), (hx - k, -hz + k), (hx, -hz + k), (hx, hz - k), (hx - k, hz - k),
            (hx - k, hz), (-hx + k, hz), (-hx + k, hz - k), (-hx, hz - k), (-hx, -hz + k), (-hx + k, -hz + k)]


def prism(ring, y0, y1, c, cx=0.0, cz=0.0, top=True):
    return extrude([(x + cx, z + cz) for x, z in ring], y0, y1, c, top=top)


def stripe(x0, x1, y0, y1, z0, z1, c):
    """A thin box used as a recessed (darker) line on a face."""
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


# ── Towers ───────────────────────────────────────────────────────────────────
LEG = [(13.0, 21.0, 4.8, 7.8), (21.0, 74.0, 3.9, 7.0), (74.0, 123.0, 3.9, 5.9), (123.0, 162.0, 3.9, 5.0),
       (162.0, 195.0, 2.8, 5.0), (195.0, 225.0, 1.85, 3.8)]
STRUTS_UP = [(111.0, 121.0), (151.0, 161.0), (185.0, 193.0), (215.0, 223.0)]


def leg_size(y):
    for y0, y1, hx, hz in LEG:
        if y0 <= y < y1:
            return hx, hz
    return LEG[-1][2], LEG[-1][3]


def tower(zc, fender=False):
    steel, dark, conc = [], [], []
    # pier (OSM 42.1 × 20.2 m, concrete to 13 m)
    conc.append(B(-21.05, 21.05, -6.0, 13.0, zc - 10.1, zc + 10.1, CONC))
    conc.append(B(-20.0, 20.0, 13.0, 13.4, zc - 9.2, zc + 9.2, CONC_D))
    if fender:                                   # the concrete fender round the south pier (estimated)
        ring_o = [(46 * math.cos(t), 24 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
        ring_o = [(max(min(x, 46.0), -46.0) * 1.0, z) for x, z in ring_o]
        ring_i = [(x * 43.0 / 46.0, z * 21.0 / 24.0) for x, z in ring_o]
        fm = extrude([(x, z + zc) for x, z in ring_o], -6.0, 4.5, CONC, holes=[[(x, z + zc) for x, z in ring_i]])
        conc.append(fm)
    for sx in (-1, 1):
        xc = sx * XL
        for i, (y0, y1, hx, hz) in enumerate(LEG):
            k = min(hx, hz) * 0.2
            steel.append(prism(notched(hx, hz, k), y0, y1, ORANGE, xc, zc))
            # setback shoulder: a thin cap of the next size, giving a stepped transition
            # vertical recesses: three on the broad faces (±x), one on the narrow faces (±z)
            if y1 - y0 > 6:
                for f in (-1, 1):
                    xf = xc + f * hx
                    for t in (-0.45, 0.0, 0.45):
                        zz = zc + t * hz
                        dark.append(stripe(xf - 0.06, xf + 0.06, y0 + 1.5, y1 - 1.5, zz - 0.35, zz + 0.35, ORANGE_D))
                    zf = zc + f * hz
                    dark.append(stripe(xc - 0.22 * hx, xc + 0.22 * hx, y0 + 1.5, y1 - 1.5,
                                       zf - 0.06, zf + 0.06, ORANGE_D))
        # saddle cover on top of the leg, rounded along the cable, and the aircraft beacon
        hx, hz = LEG[-1][2], LEG[-1][3]
        sad = cyl(1.6, 1.6, 2 * hz - 0.6, ORANGE, seg=14).R(math.pi / 2, 0, 0).T(xc, 225.8, zc - hz + 0.3)
        sad.v[:, 0] = xc + (sad.v[:, 0] - xc) * (hx / 1.6)
        sad.v[:, 1] = np.maximum(sad.v[:, 1], 224.9)
        steel.append(sad)
        steel.append(cyl(0.35, 0.3, 1.4, RED, seg=8).T(xc, 226.4, zc))
    # struts between the legs above the deck: Art Deco portals with fluting and stepped corbels
    for y0, y1 in STRUTS_UP:
        hx, hz = leg_size(y0)
        xi = XL - hx                               # inner face of the legs
        d = hz - 0.9
        steel.append(B(-xi, xi, y0, y1, zc - d, zc + d, ORANGE))
        for f in (-1, 1):
            zf = zc + f * d
            for t in np.linspace(-0.8, 0.8, 7):
                dark.append(stripe(t * xi - 0.25, t * xi + 0.25, y0 + 0.8, y1 - 0.8, zf - 0.06, zf + 0.06, ORANGE_D))
        for sx in (-1, 1):                         # two-step corbels under the strut ends
            x_in = sx * xi
            steel.append(B(min(x_in, x_in - sx * 1.6), max(x_in, x_in - sx * 1.6), y0 - 1.6, y0, zc - d, zc + d, ORANGE))
            steel.append(B(min(x_in, x_in - sx * 0.8), max(x_in, x_in - sx * 0.8), y0 - 3.4, y0 - 1.6, zc - d, zc + d, ORANGE))
    # below the deck: horizontal struts and X bracing between the legs, on both faces of the tower
    hx, hz = leg_size(30.0)
    xi = XL - hx
    for y0, y1 in ((24.0, 26.0), (50.0, 52.0)):
        steel.append(B(-xi, xi, y0, y1, zc - hz + 0.6, zc + hz - 0.6, ORANGE))
    steel.append(B(-xi, xi, 64.5, 73.6, zc - hz + 0.8, zc + hz - 0.8, ORANGE))      # deck strut
    P0, P1 = [], []
    for y0, y1 in ((26.0, 50.0), (52.0, 64.5)):
        for zf in (zc - hz + 1.2, zc + hz - 1.2):
            P0 += [(-xi, y0, zf), (xi, y0, zf)]
            P1 += [(xi, y1, zf), (-xi, y1, zf)]
    steel.append(bars(P0, P1, 1.0, 1.2, ORANGE, caps=True, up=(0, 0, 1)))
    # the sidewalks pass round the outside of the legs (OSM: deck parts to x ±21.4)
    walk = []
    for sx in (-1, 1):
        x0, x1 = sorted((sx * (XL + LEG[1][2]), sx * 21.4))
        walk.append(B(x0, x1, 74.1, 75.0, zc - 10.0, zc + 10.0, WALK))
        xr = sx * 21.3
        steel.append(B(xr - 0.08, xr + 0.08, 75.0, 76.3, zc - 10.0, zc + 10.0, ORANGE))
        steel.append(B(min(x0, x1), max(x0, x1), 72.6, 74.1, zc - 10.0, zc + 10.0, ORANGE))
    return merge(steel), merge(dark), merge(conc + walk)


# ── Main cables and suspenders ──────────────────────────────────────────────
def cable_y(z):
    """Height of the cable centre at z (main span, side spans, over the Fort Point arch, into the anchorages)."""
    z = np.asarray(z, float)
    y = np.empty_like(z)
    a = np.abs(z)
    main = a <= ZT
    y[main] = Y_LOW + SAG * (z[main] / ZT) ** 2
    ssag = SAG * (343.0 / (2 * ZT)) ** 2                # same horizontal pull: sag ∝ span²  (≈ 10.4 m)
    for sgn, zb in ((-1, Z_S1 + 5.0), (1, Z_N1 - 5.0)):
        m = (np.sign(z) == sgn) & (a > ZT)
        t = (a[m] - ZT) / (abs(zb) - ZT)
        y[m] = Y_SADDLE + (Y_CBENT - Y_SADDLE) * t - 4 * ssag * t * (1 - t)
    return y


def cables():
    out = []
    for sx in (-1, 1):
        zs = np.concatenate([np.linspace(Z_S1 + 5.0, -ZT, 44), np.linspace(-ZT, ZT, 161)[1:],
                             np.linspace(ZT, Z_N1 - 5.0, 44)[1:]])
        pts = np.stack([np.full(len(zs), sx * XL), cable_y(zs), zs], 1)
        # south: over the arch from S1 to S2, then down into the south anchorage
        south = [(sx * XL, Y_CBENT - 0.5, Z_S1 - 5.0), (sx * XL, Y_CBENT - 1.6, Z_S1 - 30.0), (sx * XL, Y_CBENT - 2.6, -1038.0),
                 (sx * XL, Y_CBENT - 2.4, Z_S2 + 30.0), (sx * XL, Y_CBENT - 1.5, Z_S2 + 5.0)]
        down_s = [(sx * XL, Y_CBENT - 1.5, Z_S2 - 5.0), (sx * XL, 76.0, -1130.0)]
        down_n = [(sx * XL, Y_CBENT, Z_N1 + 5.0), (sx * XL, 76.0, 1020.0)]
        for P in (pts, south, down_s, down_n):
            out.append(tube(P, 0.46, ORANGE, seg=8))
        # cable bands every 15.24 m (with the suspenders)
    return merge(out)


def suspenders():
    """Pairs of ropes every 50 ft from the cable bands to the top chord, in both cable planes."""
    s0 = s_of_z(0.0)
    P0, P1 = [], []
    k = np.arange(-70, 71)
    z = k * 15.24
    ok = ((np.abs(z) < ZT - 9.0) | ((np.abs(z) > ZT + 9.0) & (z > Z_S1 + 8.0) & (z < Z_N1 - 8.0)))
    for zz in z[ok]:
        yc = float(cable_y(np.array([zz]))[0]) - 0.45
        for sx in (-1, 1):
            for dz in (-0.28, 0.28):
                P0.append((sx * XL, yc, zz + dz)); P1.append((sx * XL, float(road_y(zz)) + 0.9, zz + dz))
    ropes = bars(P0, P1, 0.16, 0.16, ORANGE)
    # cable bands: short collars
    Q0, Q1 = [], []
    for zz in z[ok]:
        yc = float(cable_y(np.array([zz]))[0])
        for sx in (-1, 1):
            Q0.append((sx * XL, yc, zz - 0.6)); Q1.append((sx * XL, yc, zz + 0.6))
    bands = bars(Q0, Q1, 1.15, 1.15, ORANGE_D * 1.1, caps=True, up=(0, 1, 0))
    return merge([ropes, bands])


# ── The deck: roadway, sidewalks, railings, lamps and the truss ─────────────
def in_any(z, ranges):
    return any(a <= z <= b for a, b in ranges)


TOWER_GAPS = [(-ZT - 10.2, -ZT + 10.2), (ZT - 10.2, ZT + 10.2)]
BLOCKS = [(-1200.0, -1100.0), (990.0, 1080.0)]           # anchorages: the deck lies on the concrete
ARCH = (Z_S1 - 5.0, Z_S2 + 5.0)


def deck():
    s_start, s_end = 0.0, S_END
    st = np.arange(s_start, s_end, 4.0)
    st = np.append(st, s_end)
    meshes_road, meshes_walk, steel, white, grey = [], [], [], [], []
    # roadway (6 lanes, 19 m between curbs)
    meshes_road.append(strip(W(-9.5, 0, st), W(9.5, 0, st), ASPH, (0, 1, 0)))
    # lane lines and the movable median barrier
    for u in (-6.33, -3.17, 3.17, 6.33):
        white.append(strip(W(u - 0.08, 0.03, st), W(u + 0.08, 0.03, st), LINE, (0, 1, 0)))
    for u in (-0.3, 0.3):
        grey.append(strip(W(u, 0, st), W(u, 0.8, st), BARRIER, (np.sign(u), 0, 0)))
    grey.append(strip(W(-0.3, 0.8, st), W(0.3, 0.8, st), BARRIER, (0, 1, 0)))
    # sidewalks (3 m), curbs, slab edges and soffit
    for sx in (-1, 1):
        a, b = sx * 9.5, sx * 13.4
        meshes_walk.append(strip(W(a, 0.35, st), W(b, 0.35, st), WALK, (0, 1, 0)))
        meshes_walk.append(strip(W(a, 0, st), W(a, 0.35, st), WALK, (-sx, 0, 0)))
        # outer edge of the slab
        meshes_walk.append(strip(W(sx * 13.9, -0.6, st), W(sx * 13.9, 0.35, st), ORANGE, (sx, 0, 0)))
    meshes_road.append(strip(W(-13.9, -0.6, st), W(13.9, -0.6, st), CONC_D, (0, -1, 0)))
    # fix normals of the vertical strips: outward (curb faces towards the road)
    road = merge(meshes_road)
    walk = merge(meshes_walk)
    # railings (Morrow's railing, 1.3 m), on both outer edges, broken at the towers
    rail = []
    for sx in (-1, 1):
        for seg in _runs(st, TOWER_GAPS):
            A, Bv = W(sx * 13.85, 0.35, seg), W(sx * 13.85, 1.65, seg)
            rail.append(strip(A, Bv, ORANGE, (sx, 0, 0)))
            rail.append(strip(W(sx * 13.7, 1.65, seg), W(sx * 14.0, 1.65, seg), ORANGE, (0, 1, 0)))
            rail.append(strip(W(sx * 13.55, 0.35, seg), W(sx * 13.55, 1.25, seg), ORANGE_D, (-sx, 0, 0)))
    steel.append(merge(rail))
    # lamp posts every 30.48 m (100 ft) on both curbs
    s0 = s_of_z(0.0)
    sl = s0 + np.arange(-45, 46) * 30.48
    sl = sl[(sl > 10) & (sl < S_END - 10)]
    sl = np.array([s for s in sl if not in_any(float(frame(s)[1]), TOWER_GAPS)])
    lamps0, lamps1, heads0, heads1 = [], [], [], []
    for sx in (-1, 1):
        lamps0.append(W(sx * 9.8, 0.35, sl)); lamps1.append(W(sx * 9.8, 8.5, sl))
        heads0.append(W(sx * 9.8, 8.3, sl)); heads1.append(W(sx * 8.4, 8.3, sl))
    steel.append(bars(np.vstack(lamps0), np.vstack(lamps1), 0.3, 0.3, ORANGE))
    lamp_heads = bars(np.vstack(heads0), np.vstack(heads1), 0.45, 0.4, LAMP, caps=True)
    # the truss, panel by panel (nodes aligned on the suspenders)
    sn = s0 + np.arange(-400, 401) * PANEL
    sn = sn[(sn >= 1.0) & (sn <= S_END - 1.0)]
    zn = frame(sn)[1]
    keep = np.array([not in_any(z, TOWER_GAPS + BLOCKS) for z in zn])
    T0, T1, Wd, Hd, cols = [], [], [], [], []

    def add(p0, p1, w, h):
        T0.append(p0); T1.append(p1); Wd.append(np.full(len(p0), w)); Hd.append(np.full(len(p0), h))
    pairs = [(i, i + 1) for i in range(len(sn) - 1) if keep[i] and keep[i + 1]]
    ia = np.array([p[0] for p in pairs]); ib = np.array([p[1] for p in pairs])
    sa, sb = sn[ia], sn[ib]
    for sx in (-1, 1):
        u = sx * XL
        add(W(u, -0.35, sa), W(u, -0.35, sb), 0.8, 0.9)                     # top chord
        add(W(u, -TRUSS + 0.45, sa), W(u, -TRUSS + 0.45, sb), 0.8, 0.9)     # bottom chord
        add(W(u, -TRUSS + 0.45, sa), W(u, -0.35, sa), 0.45, 0.5)            # verticals
        odd = (ia % 2 == 0)
        add(np.where(odd[:, None], W(u, -TRUSS + 0.45, sa), W(u, -0.35, sa)),
            np.where(odd[:, None], W(u, -0.35, sb), W(u, -TRUSS + 0.45, sb)), 0.4, 0.45)   # diagonals
    lastv = [p[1] for p in pairs if (p[1] + 1 >= len(sn)) or not keep[p[1] + 1]]
    if lastv:
        sv = sn[np.array(lastv)]
        for sx in (-1, 1):
            add(W(sx * XL, -TRUSS + 0.45, sv), W(sx * XL, -0.35, sv), 0.45, 0.5)
    # floor beams and the bottom lateral bracing (1953–54)
    add(W(-XL, -1.4, sa), W(XL, -1.4, sa), 0.5, 1.2)
    ev = ia % 2 == 0
    add(W(-XL, -TRUSS + 0.45, sa[ev]), W(XL, -TRUSS + 0.45, sb[ev]), 0.35, 0.35)
    add(W(XL, -TRUSS + 0.45, sa[ev]), W(-XL, -TRUSS + 0.45, sb[ev]), 0.35, 0.35)
    truss = bars(np.vstack(T0), np.vstack(T1), np.concatenate(Wd), np.concatenate(Hd), ORANGE)
    steel.append(truss)
    return road, walk, merge(white), merge(grey), merge(steel), lamp_heads


def _runs(st, gaps):
    """Split the stations st into runs that avoid the z ranges in gaps."""
    z = frame(st)[1]
    ok = np.array([not in_any(v, gaps) for v in z])
    runs, cur = [], []
    for s, k in zip(st, ok):
        if k:
            cur.append(s)
        elif cur:
            runs.append(np.array(cur)); cur = []
    if cur:
        runs.append(np.array(cur))
    return [r for r in runs if len(r) > 1]


# ── Fort Point arch, pylons, anchorages, approach bents, abutments ──────────
def arch():
    za, zb = Z_S1 - 5.5, Z_S2 + 5.5                      # springings at the faces of the pylons
    zm, half = (za + zb) / 2, (za - zb) / 2
    n = 14
    zs = np.linspace(za, zb, n + 1)
    t = (zs - zm) / half
    y_lo = 6.0 + 52.0 * (1 - t ** 2)                    # lower chord: crown 58 m
    y_hi = 15.0 + 48.5 * (1 - t ** 2)                   # upper chord: crown 63.5 m
    P0, P1, w = [], [], []
    for sx in (-1, 1):
        u = sx * 10.0
        for i in range(n):
            P0 += [(u, y_lo[i], zs[i]), (u, y_hi[i], zs[i])]
            P1 += [(u, y_lo[i + 1], zs[i + 1]), (u, y_hi[i + 1], zs[i + 1])]
            w += [1.4, 1.4]
            P0.append((u, y_lo[i], zs[i])); P1.append((u, y_hi[i + 1] if i % 2 == 0 else y_hi[i], zs[i + 1] if i % 2 == 0 else zs[i])); w.append(0.6)
            if i % 2:
                P0.append((u, y_lo[i + 1], zs[i + 1])); P1.append((u, y_hi[i], zs[i])); w.append(0.6)
            P0.append((u, y_lo[i], zs[i])); P1.append((u, y_hi[i], zs[i])); w.append(0.7)
        # spandrel columns up to the truss
        for i in range(1, n):
            P0.append((u, y_hi[i], zs[i])); P1.append((u, road_y(zs[i]) - TRUSS, zs[i])); w.append(0.8)
    for i in range(0, n + 1, 2):                          # struts between the two ribs
        P0.append((-10.0, y_hi[i], zs[i])); P1.append((10.0, y_hi[i], zs[i])); w.append(0.6)
    ribs = bars(P0, P1, np.array(w), np.array(w) * 1.1, ORANGE, caps=True)
    # the deck over the arch: a stringer frame on the spandrel columns
    return ribs


def pylon(zc, ground):
    """Morrow's concrete pylons: a block under the deck, two stepped towers flanking the roadway."""
    out = []
    yr = float(road_y(zc))
    out.append(B(-17.5, 17.5, ground - FOUND, yr - TRUSS, zc - 5.0, zc + 5.0, CONC))
    for sx in (-1, 1):
        xc = sx * 15.75
        for y0, y1, hx, hz in ((yr - TRUSS, yr + 9.4, 2.75, 5.0), (yr + 9.4, yr + 12.6, 2.35, 4.4),
                               (yr + 12.6, yr + 15.3, 1.9, 3.7)):
            out.append(prism(notched(hx, hz, 0.45), y0, y1, CONC, xc, zc))
        for f in (-1, 1):
            for t in (-0.5, 0.0, 0.5):
                out.append(stripe(xc - 0.35 + t * 3.0, xc + 0.35 + t * 3.0, yr - TRUSS + 2.0, yr + 8.6,
                                  zc + f * 5.0 - 0.06, zc + f * 5.0 + 0.06, CONC_D))
    return merge(out)


def anchorage(z0, z1, ground_lo, south):
    out = []
    yr0, yr1 = float(road_y(z0)), float(road_y(z1))
    top = min(yr0, yr1) - 0.6
    ring = [(-21.0, z0), (21.0, z0), (21.0, z1), (-21.0, z1)]
    out.append(extrude(ring, ground_lo - FOUND, top, CONC, top_c=CONC_D))
    for x in np.linspace(-21.0, 21.0, 2):                 # ribs on the long sides (Art Deco buttresses)
        for zz in np.linspace(z0 + 8, z1 - 8, 6):
            out.append(B(x - 0.7, x + 0.7, ground_lo - 2.0, top - 1.0, zz - 1.2, zz + 1.2, CONC_D))
    # cable housings over the sidewalks, where the cables go down into the anchorage
    for sx in (-1, 1):
        if south:
            zz0, zz1, ya, yb = Z_S2 - 5.0, -1135.0, 82.5, 75.4
        else:
            zz0, zz1, ya, yb = Z_N1 + 5.0, 1025.0, 85.8, 75.4
        h = box(2.6, 1.0, abs(zz1 - zz0), CONC).T(sx * XL, 0, (zz0 + zz1) / 2)
        lo_y = float(road_y((zz0 + zz1) / 2)) + 0.3

        def wf(v, zz0=zz0, zz1=zz1, ya=ya, yb=yb, lo_y=lo_y):
            t = np.clip((v[:, 2] - zz0) / (zz1 - zz0), 0, 1)
            v[:, 1] = np.where(v[:, 1] > 0.5, ya + (yb - ya) * t, lo_y)
            return v
        out.append(h.warp(wf))
    return merge(out)


def bent(s, kind="steel"):
    z = float(frame(s)[1])
    g = ground_y(z) - FOUND
    yt = float(road_y(z)) - TRUSS
    out = []
    P0 = [W(-XL, 0, s)[()], W(XL, 0, s)[()]]
    P0 = [np.array([p[0], g, p[2]]) for p in P0]
    P1 = [W(-XL, -TRUSS, s)[()], W(XL, -TRUSS, s)[()]]
    out.append(bars(P0, P1, 1.8, 2.6, ORANGE, caps=True))
    # cross bracing every ~12 m and a cap beam
    levels = np.linspace(g + FOUND, yt, max(2, int((yt - g - FOUND) / 12) + 1))
    Q0, Q1 = [], []
    for i in range(len(levels) - 1):
        a, b = levels[i], levels[i + 1]
        Q0 += [WA(-XL, a, s), WA(XL, a, s)]
        Q1 += [WA(XL, b, s), WA(-XL, b, s)]
        Q0.append(WA(-XL, b, s)); Q1.append(WA(XL, b, s))
    if Q0:
        out.append(bars(np.array(Q0), np.array(Q1), 0.6, 0.7, ORANGE, caps=True))
    footing = WA(0, g, s)
    x, z0, nx, nz = frame(s)
    f = box(34.0, FOUND + 1.0, 6.0, CONC)
    ang = math.atan2(float(nz), float(nx))
    f.R(0, -ang, 0).T(footing[0], g, footing[2])
    out.append(f)
    return merge(out)


def abutment(s_end_side):
    s = s_end_side
    z = float(frame(s)[1])
    g = ground_y(z) - FOUND
    x, zc, nx, nz = frame(s)
    yr = float(road_y(z))
    m = box(30.0, yr - 0.6 - g, 10.0, CONC)
    ang = math.atan2(float(nz), float(nx))
    m.R(0, -ang, 0).T(float(x), g, float(zc))
    return m


def footprint():
    """A thin strip along the deck, widened at the piers, the pylons and the anchorages, and pinched to a 2 m neck
    over Fort Point so the fort under the arch is not cleared."""
    st = np.arange(0.0, S_END, 5.0)
    st = np.append(st, S_END)
    left, right = [], []
    for s in st:
        z = float(frame(s)[1])
        w = 15.0
        if abs(abs(z) - ZT) < 12.0:
            w = 22.5
        if abs(z + ZT) < 25.0:
            w = 47.0                              # south pier fender
        if in_any(z, [(Z_S1 - 6, Z_S1 + 6), (Z_S2 - 6, Z_S2 + 6), (Z_N1 - 6, Z_N2 + 6), (-1201.0, -1094.0)]):
            w = 22.5
        if ARCH[1] + 1.0 < z < ARCH[0] - 1.0:
            w = 1.0
        left.append(W(w, 0, s)[()]); right.append(W(-w, 0, s)[()])
    ring = [(round(float(p[0]), 1), round(float(p[2]), 1)) for p in left] + \
           [(round(float(p[0]), 1), round(float(p[2]), 1)) for p in right[::-1]]
    out = []
    for p in ring:                                   # drop repeated points
        if not out or out[-1] != list(p):
            out.append(list(p))
    changed = True                                   # drop points within 0.2 m of the line through their neighbours
    while changed:
        changed = False
        k = 1
        while k < len(out) - 1:
            (x0, z0), (x1, z1), (x2, z2) = out[k - 1], out[k], out[k + 1]
            L = math.hypot(x2 - x0, z2 - z0)
            if L > 0 and abs((x2 - x0) * (z1 - z0) - (z2 - z0) * (x1 - x0)) / L < 0.2:
                out.pop(k); changed = True
            else:
                k += 1
    return out


def main():
    out = ROOT / "models" / "Q44440"
    out.mkdir(parents=True, exist_ok=True)
    s_steel, s_dark, s_conc = tower(-ZT, fender=True)
    n_steel, n_dark, n_conc = tower(ZT)
    road, walk, white, grey, d_steel, lamp_heads = deck()
    conc = [s_conc, n_conc]
    for zc in (Z_S1, Z_S2, Z_N1, Z_N2):
        conc.append(pylon(zc, ground_y(zc)))
    conc.append(anchorage(-1200.0, -1100.0, 14.0, True))
    conc.append(anchorage(990.0, 1080.0, 50.0, False))
    steel = [s_steel, n_steel, d_steel, cables(), suspenders(), arch()]
    for z in (-1235.0, -1270.0, 1130.0, 1175.0, 1220.0, 1265.0, 1310.0, 1355.0):
        steel.append(bent(s_of_z(z)))
    conc.append(abutment(3.0))
    conc.append(abutment(S_END - 3.0))
    steel_m = merge(steel)
    other = merge([road, walk, white, grey] + conc + [s_dark, n_dark, lamp_heads])
    parts = [("steel", steel_m, {"material": "steel"}), ("concrete_and_deck", other)]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.85,
                           materials={"steel": {"metallic": 0.25, "roughness": 0.55}})
    allv = np.vstack([steel_m.v, other.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.1f}..{hi[0]:.1f}, "
          f"y {lo[1]:.1f}..{hi[1]:.1f}, z {lo[2]:.1f}..{hi[2]:.1f}")
    fp = footprint()
    meta = out / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        m["footprint"] = fp
        m["triangles"] = tris
        meta.write_bytes((json.dumps(m, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print("meta.json: footprint", len(fp), "points, triangles", tris)


if __name__ == "__main__":
    main()
