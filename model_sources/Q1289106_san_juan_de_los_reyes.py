"""
Monasterio de San Juan de los Reyes (Q1289106), Toledo — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Franciscan monastery founded by the Catholic Monarchs Isabel and Fernando (1477) to commemorate the
battle of Toro and meant as their burial place; church and cloister by Juan Guas (master builder 1479–1496, with Egas
Cueman), finished by Enrique and Antón Egas (church 1495, cloister c. 1504–1510); the north portal designed by Alonso
de Covarrubias (1553) and finished in 1610. Burnt by the French in 1808, restored by Arturo Mélida (from 1883) and
after 1940. Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 21282452 (2759 m²) and its three building:parts, in model x/z for heading 333.3°
  (python tools/osm.py relation 21282452 --heading 333.3): the church (way 1552419301: 67 × 22.6 m, the single nave
  with its chapels, the crossing and the head with its buttresses), the east range of the cloister (way 1552419298,
  1 level) and the south and west ranges (way 1552419318, 2 levels), and the cloister courtyard (20 × 21 m). The
  north wall of the church is straight along the long axis 63.3°/243.3° (±0.2°).
- Wikipedia (es/en): single nave with chapels between the buttresses, raised choir at the west end, the crossing
  with its octagonal cimborrio over the head, two portals (north and west), the iron chains of the Christian
  captives freed at Málaga hanging on the walls of the head, pinnacles and cresting along the roofline, the
  two-storey cloister (pointed arches with tracery below, mixtilinear arches and balustrades above) with its garden
  in four parts around a well.
- Heights from photographs on Wikimedia Commons (see meta.json), scaled by the OSM lengths (±10 %): the nave wall
  23.3 m to its cornice (buttress pinnacles to 28.5–29 m), bays of 8.6–8.9 m with windows from 14.2 to 19.8 m and
  the cresting at their sills; the head 23.6 m with its balustrade to 25 m; the octagonal lantern to 31.2 m, its
  balustrade to 32 m, its pinnacles to 36 m, the dome and little lantern with the cross to 38.3 m; the north portal
  12.2 m with its gable and cross to 17 m; the cloister 7.6 m (lower floor) + 5 m (upper floor), eaves 12.9 m,
  pinnacles to 16.5 m, five bays a side on the garden.
- PNOA orthophoto (IGN): the octagonal lantern over the crossing at the east end, the deep recesses between the
  buttresses on the south side of the nave above the cloister, the tiled roofs of the ranges, the garden.
- Terrain: IGN MDT05. The ground falls about 8 m from the east end (Calle de los Reyes Católicos, +2.5 m) to the
  west end (−5 m); y = 0 is the Plaza de San Juan de los Reyes in front of the north portal (ground_at); every wall
  goes down to 1.5 m below the real ground under it.

Axes (docs/MODEL_SPEC.md): +Y up, the north façade (portal, chains) faces +Z (heading 333.3°), +X points to 243.3°
(west-south-west: the west end of the church), origin at the centroid of the OSM outline (in the cloister garden).

    python model_sources/Q1289106_san_juan_de_los_reyes.py   -> models/Q1289106/model.glb
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import (Mesh, merge, box, cyl, lathe, tube, extrude, cap, walls, offset, triangulate, ring_area,  # noqa: E402
                     crumple, noise3, write_glb)

rng = np.random.default_rng(1477)

# ── Colours ──────────────────────────────────────────────────────────────────
GRAN = (0.56, 0.52, 0.46)          # the grey-brown granite ashlar of the church (white mortar lines)
GRAN_D = (0.47, 0.43, 0.38)
LIME = (0.82, 0.77, 0.66)          # the white stone of the head, the lantern, the portal and the cresting
CLOI = (0.84, 0.76, 0.62)          # the warm limestone of the cloister
RENDER = (0.79, 0.67, 0.54)        # rendered walls of the convent ranges
ROOF = (0.56, 0.43, 0.35)          # old Arab tiles
LEAD = (0.46, 0.46, 0.47)
SLATE = (0.30, 0.32, 0.35)
DARK = (0.10, 0.10, 0.12)
GLASS = (0.17, 0.19, 0.21)
IRON = (0.06, 0.06, 0.06)
WOOD = (0.36, 0.22, 0.13)
PAVE = (0.62, 0.58, 0.52)
EARTH = (0.50, 0.42, 0.32)
GRASS = (0.30, 0.40, 0.20)
HEDGE = (0.17, 0.29, 0.14)

FOUND = -3.0
HEADING = 333.3
GROUND_AT = (3.0, 33.0)

# ── OpenStreetMap (osm.py relation 21282452 --heading 333.3), model metres ──
CHURCH = [  # way/1552419301
    (16.85, 6.48), (-6.42, 6.09), (-15.96, 6.01), (-17.20, 6.01), (-22.81, 5.92), (-24.48, 5.83), (-25.16, 6.98),
    (-25.01, 7.05), (-23.60, 7.88), (-23.53, 7.92), (-23.02, 8.05), (-23.00, 9.93), (-25.21, 9.93), (-25.86, 8.74),
    (-27.42, 9.57), (-26.78, 10.76), (-28.64, 11.81), (-30.17, 11.04), (-31.05, 12.59), (-29.03, 13.73),
    (-29.03, 20.57), (-29.48, 20.71), (-30.52, 21.43), (-29.72, 22.71), (-28.61, 22.02), (-26.97, 23.71),
    (-27.62, 25.00), (-26.07, 25.54), (-25.62, 24.27), (-22.98, 24.35), (-23.16, 26.62), (-25.12, 26.51),
    (-25.23, 28.44), (-24.85, 28.64), (-23.27, 28.56), (-8.07, 28.62), (-6.63, 28.60), (0.14, 28.65), (1.79, 28.61),
    (3.32, 28.63), (9.05, 28.66), (10.62, 28.58), (11.18, 28.61), (17.68, 28.65), (19.05, 28.59), (21.13, 28.64),
    (25.90, 28.68), (27.24, 28.11), (28.86, 27.56), (34.73, 23.30), (35.41, 20.04), (35.97, 17.22), (36.14, 16.31),
    (36.57, 13.17), (31.04, 12.75), (31.05, 10.64), (29.29, 10.50), (29.52, 7.76), (27.38, 6.68), (22.89, 6.66),
    (19.49, 6.57), (18.05, 6.58)]
EAST_RANGE = [  # way/1552419298
    (-27.36, -24.83), (-26.47, -9.46), (-25.61, 5.01), (-24.48, 5.83), (-22.81, 5.92), (-17.20, 6.01), (-15.96, 6.01),
    (-15.79, 4.35), (-16.78, -25.10)]
RANGES = [  # way/1552419318
    (13.40, -34.65), (13.34, -35.68), (7.33, -35.72), (7.31, -34.73), (7.51, -34.75), (7.64, -31.58), (-17.90, -30.88),
    (-17.90, -30.51), (-27.64, -30.19), (-27.36, -24.83), (-16.78, -25.10), (14.80, -26.00), (16.85, 6.48),
    (18.05, 6.58), (19.49, 6.57), (22.89, 6.66), (22.86, 6.52), (20.60, -23.82), (25.24, -24.10), (25.01, -29.93),
    (20.42, -29.75), (18.37, -29.67), (14.61, -29.44), (13.52, -29.37)]
COURT = [(8.71, 1.14), (-11.35, 1.62), (-11.86, -19.27), (8.20, -19.74)]      # courtyard (inner ring)
GALLERY = [(-15.96, 6.01), (-16.78, -25.10), (14.80, -26.00), (16.85, 6.48)]   # outer edge of the cloister walks

# the church, from the photographs: bays and levels
X_HEAD = -10.0                    # the crossing and the head lie east of this line (x < X_HEAD)
Z_NAVE_S = 11.5                   # south wall of the nave above the chapels between the south buttresses
NAVE_TOP, HEAD_TOP = 23.3, 23.6
BUTTS = [-9.6, -1.5, 7.4, 15.7, 24.6]
BAYS = [-5.45, 2.95, 11.55, 20.15]
LC = (-16.5, 19.6)                # centre of the lantern over the crossing
LR = 6.6                          # its circumradius


# ── Terrain (IGN MDT05 in models/Q1289106/terrain.json): foundations follow the real ground ──
def _terrain():
    p = ROOT / "models" / "Q1289106" / "terrain.json"
    if not p.exists():
        return None
    t = json.loads(p.read_text("utf-8"))
    n = t["n"]
    q = np.frombuffer(base64.b64decode(t["h"]), "<i2").reshape(n, n).astype(np.float64)
    h = np.where(q == -32768, np.nan, t["off"] + q / 10)
    a = math.radians(HEADING)
    ez = np.array([math.sin(a), math.cos(a)])
    ex = np.array([math.sin(a - math.pi / 2), math.cos(a - math.pi / 2)])

    def at(x, z):
        e = np.asarray(x) * ex[0] + np.asarray(z) * ez[0]
        nn = np.asarray(x) * ex[1] + np.asarray(z) * ez[1]
        fi = (e - t["x0"]) / t["c"]; fj = (t["y0"] - nn) / t["c"]
        i0 = np.clip(np.floor(fi).astype(int), 0, n - 2); j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
        tx = np.clip(fi - i0, 0, 1); tz = np.clip(fj - j0, 0, 1)
        return (h[j0, i0] * (1 - tx) * (1 - tz) + h[j0, i0 + 1] * tx * (1 - tz)
                + h[j0 + 1, i0] * (1 - tx) * tz + h[j0 + 1, i0 + 1] * tx * tz)
    g0 = float(at(np.array([GROUND_AT[0]]), np.array([GROUND_AT[1]]))[0])
    return lambda x, z: at(x, z) - g0


TERRAIN = _terrain()


def sink(m):
    """Foundations: every vertex at FOUND goes down to 1.5 m below the real ground under it (and never above FOUND)."""
    if TERRAIN is None or not len(m.v):
        return m
    k = m.v[:, 1] <= FOUND + 1e-6
    if k.any():
        g = TERRAIN(m.v[k, 0], m.v[k, 2])
        g = np.where(np.isnan(g), 0.0, g)
        m.v[k, 1] = np.minimum(FOUND, np.floor((g - 1.5) * 2) / 2)
    return m


def ground(x, z):
    if TERRAIN is None:
        return 0.0
    return float(TERRAIN(np.array([x]), np.array([z]))[0])


# ── Small helpers (as in Q1123180_catedral_de_toledo.py) ─────────────────────
def tone(c, k=1.0, s=0.015):
    return tuple(np.clip(np.asarray(c) * k * (1 + rng.normal(0, s)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sphere(r, c, seg=8, rings=5):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def obox(p, q, y0, y1, w, c):
    """A box along the segment p→q (plan), w thick, from y0 to y1."""
    (x0, z0), (x1, z1) = p, q
    L = math.hypot(x1 - x0, z1 - z0)
    a = math.atan2(-(z1 - z0), x1 - x0)
    return box(L, y1 - y0, w, c).R(0, a, 0).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def ccw(ring):
    return list(ring) if ring_area(ring) > 0 else list(ring)[::-1]


def ngon(cx, cz, r, n=8, rot=0.0):
    return [(cx + r * math.cos(rot + 2 * math.pi * k / n), cz + r * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def clip(P, axis, a, keep_low):
    """Sutherland–Hodgman: the part of a polygon with p[axis] <= a (keep_low) or >= a."""
    def inside(p):
        return p[axis] <= a if keep_low else p[axis] >= a
    out = []
    n = len(P)
    for i in range(n):
        p, q = P[i], P[(i + 1) % n]
        ip, iq = inside(p), inside(q)
        if ip:
            out.append(p)
        if ip != iq:
            t = (a - p[axis]) / (q[axis] - p[axis])
            out.append(tuple(p[k] + t * (q[k] - p[k]) for k in range(2)))
    clean = []
    for p in out:
        if not clean or math.dist(clean[-1], p) > 0.05:
            clean.append(p)
    return clean


def fr(p, n):
    """A wall frame at plan point p with outward normal n: (origin, tangent, normal); u runs along the tangent."""
    nn = np.array([n[0], 0.0, n[1]], float)
    nn /= np.linalg.norm(nn)
    return np.array([p[0], 0.0, p[1]], float), np.array([nn[2], 0.0, -nn[0]]), nn


def plane_poly(pts_uv, origin, t, n, c):
    pts, tris = triangulate(pts_uv)
    o = np.asarray(origin, float)
    V = np.array([o + u * np.asarray(t) + np.array([0, v, 0]) for u, v in pts])
    m = Mesh(V, tris, c)
    if np.dot(np.cross(t, [0.0, 1.0, 0.0]), n) < 0:
        m.f = m.f[:, ::-1]
    return m


def poly_on(F, pts, eps, c):
    o, t, n = F
    return plane_poly(pts, o + n * eps, t, n, c)


def rect_on(F, s, y0, w, h, eps, c):
    return poly_on(F, [(s - w / 2, y0), (s + w / 2, y0), (s + w / 2, y0 + h), (s - w / 2, y0 + h)], eps, c)


def arc_pts(s, ys, w, sharp=0.8, seg=5):
    """A pointed arch from the right springing over the apex to the left springing (springing at ys)."""
    R = max(sharp, 0.5) * w
    cr = s + w / 2 - R
    amax = math.acos(max(-1.0, min(1.0, (s - cr) / R)))
    right = [(cr + R * math.cos(a), ys + R * math.sin(a)) for a in np.linspace(0, amax, seg + 1)]
    left = [(2 * s - u, v) for u, v in right[::-1]]
    pts = right + left[1:]
    return pts


def round_arc(s, ys, w, seg=8):
    r = w / 2
    return [(s + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)]


def lancet_on(F, s, y0, w, h, eps, c, sharp=0.8, seg=5):
    R = max(sharp, 0.5) * w
    rise = math.sqrt(max(R * R - (R - w / 2) ** 2, 0.0))
    ys = y0 + h - rise
    pts = [(s - w / 2, y0), (s + w / 2, y0)] + arc_pts(s, ys, w, sharp, seg)
    return poly_on(F, pts, eps, c)


def round_on(F, s, y0, w, h, eps, c, seg=8):
    ys = y0 + h - w / 2
    return poly_on(F, [(s - w / 2, y0), (s + w / 2, y0)] + round_arc(s, ys, w, seg), eps, c)


def slab(F, pts_uv, thick, c, off=0.0):
    """A solid plate: a polygon drawn in a wall plane (u, height), thick along the frame's normal."""
    o, t, n = F
    pts, tris = triangulate(pts_uv)
    k = len(pts)

    def P(u, v, w):
        return o + t * u + np.array([0, v, 0]) + n * w

    V = [P(u, v, off + thick / 2) for u, v in pts] + [P(u, v, off - thick / 2) for u, v in pts]
    f = [tuple(tr) for tr in tris] + [(a + k, c_ + k, b + k) for a, b, c_ in tris]
    m = Mesh(V, f, c)
    if np.dot(np.cross(m.v[f[0][1]] - m.v[f[0][0]], m.v[f[0][2]] - m.v[f[0][0]]), n) < 0:
        m.f = m.f[:, ::-1]
    R = ccw(pts.tolist())
    rim = []
    for i in range(len(R)):
        (u0, v0), (u1, v1) = R[i], R[(i + 1) % len(R)]
        a, b = P(u0, v0, off + thick / 2), P(u1, v1, off + thick / 2)
        cc, d = P(u1, v1, off - thick / 2), P(u0, v0, off - thick / 2)
        rim.append(Mesh([a, b, cc, d], [(0, 2, 1), (0, 3, 2)], c))
    return merge([m] + rim)


def hip(x0, x1, z0, z1, y0, y1, c):
    """Hipped roof over a rectangle."""
    dx, dz = x1 - x0, z1 - z0
    if dx >= dz:
        h = dz / 2
        a, b = (x0 + h, y1, (z0 + z1) / 2), (x1 - h, y1, (z0 + z1) / 2)
        V = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), a, b]
        f = [(0, 4, 5), (0, 5, 1), (1, 5, 2), (2, 5, 4), (2, 4, 3), (3, 4, 0)]
    else:
        h = dx / 2
        a, b = ((x0 + x1) / 2, y1, z0 + h), ((x0 + x1) / 2, y1, z1 - h)
        V = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), a, b]
        f = [(0, 4, 1), (1, 4, 5), (1, 5, 2), (2, 5, 3), (3, 5, 4), (3, 4, 0)]
    return Mesh(V, f, c)


def strip(A, Bp, c):
    """Quad strip between two polylines of 3D points (a roof slope, a lean-to)."""
    A, Bp = np.asarray(A, float), np.asarray(Bp, float)
    n = len(A)
    V = np.vstack([A, Bp])
    f = []
    for i in range(n - 1):
        f += [(i, i + 1, n + i + 1), (i, n + i + 1, n + i)]
    return Mesh(V, f, c)


def tent_roof(ring, ra, rb, y0, y1, c):
    """A hipped roof over any outline: every eave point rises to its nearest point on the ridge ra-rb."""
    R = ccw(ring)
    a, b = np.asarray(ra, float), np.asarray(rb, float)
    ab = b - a

    def top(p):
        t = np.clip(np.dot(np.asarray(p) - a, ab) / np.dot(ab, ab), 0, 1)
        q = a + t * ab
        return (q[0], y1, q[1])
    V, f = [], []
    for i in range(len(R)):
        p, q = R[i], R[(i + 1) % len(R)]
        o = len(V)
        V += [(p[0], y0, p[1]), (q[0], y0, q[1]), top(q), top(p)]
        f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    m = Mesh(V, f, c)
    return m


def pinnacle(x, z, y0, h, r, c=LIME, seg=4, crockets=True, rot=math.pi / 4):
    """A Gothic pinnacle: a little shaft and a spirelet with crockets and a finial."""
    ms = [cyl(r, r, h * 0.35, c, seg=seg).R(0, rot, 0).T(x, y0, z),
          cyl(r * 1.15, r * 1.15, h * 0.04, c, seg=seg).R(0, rot, 0).T(x, y0 + h * 0.33, z),
          cyl(r * 1.0, 0.03, h * 0.65, c, seg=seg).R(0, rot, 0).T(x, y0 + h * 0.35, z)]
    if crockets:
        for j, t in enumerate((0.45, 0.65)):
            for k in range(seg):
                a = rot + 2 * math.pi * (k + 0.5 * j) / seg
                rr = r * (1.0 - (t - 0.35) / 0.65) + 0.12
                ms.append(cyl(0.1 * r / 0.3, 0.01, h * 0.12, c, seg=3).T(x + math.cos(a) * rr, y0 + h * t, z + math.sin(a) * rr))
    ms.append(sphere(r * 0.25, c, 5, 3).T(x, y0 + h * 0.98, z))
    return merge(ms)


def cresting(p, q, y, c=LIME, w=0.35, h=0.45, pitch=0.75, spike=0.5):
    """The Isabelline cresting: a pierced band along p→q topped with little fleurons."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = float(np.linalg.norm(q - p))
    ms = [obox(p, q, y, y + h, w, c)]
    n = max(1, int(L / pitch))
    for k in range(n):
        x, z = p + (q - p) * (k + 0.5) / n
        ms.append(cyl(0.16, 0.02, spike, c, seg=4).R(0, math.pi / 4, 0).T(x, y + h, z))
    return merge(ms)


def figure(x, y, z, h, c, r=None):
    """A standing statue on its bracket: a tapering body, a head."""
    r = r or h * 0.15
    body = lathe([(r * 1.1, 0), (r * 1.1, h * 0.06), (r, h * 0.08), (r * 0.95, h * 0.45), (r * 0.8, h * 0.78),
                  (r * 0.55, h * 0.82), (0.0, h * 0.82)], c, 6)
    head = sphere(r * 0.62, c, 6, 4).T(0, h * 0.82 + r * 0.5, 0)
    return merge([body, head]).T(x, y, z)


def chains(F, u0, u1, y0, rows=2, pitch=0.42, eps=0.08):
    """The iron chains of the captives: hanging lengths in rows, as on the walls of the head."""
    ms = []
    n = max(1, int((u1 - u0) / pitch))
    for r in range(rows):
        for k in range(n):
            u = u0 + (k + 0.5) * (u1 - u0) / n
            ln = 0.9 + 0.25 * rng.random()
            y = y0 + r * 1.35
            ms.append(rect_on(F, u, y + 1.15 - ln, 0.12, ln, eps, IRON))
            ms.append(rect_on(F, u, y + 1.1 - ln - 0.12, 0.22, 0.14, eps + 0.01, IRON))
    return ms


def festoon(F, u0, u1, y, eps=0.12, n=6, sag=0.35):
    o, t, nrm = F
    pts = []
    for k in range(n * 6 + 1):
        u = u0 + (u1 - u0) * k / (n * 6)
        ph = (k % 6) / 6.0
        pts.append(o + t * u + np.array([0, y - sag * math.sin(math.pi * ph), 0]) + nrm * eps)
    return tube(pts, 0.07, IRON, 4)


# ── The church ───────────────────────────────────────────────────────────────
def head():
    S, D = [], []
    H = clip(ccw(CHURCH), 0, X_HEAD, keep_low=True)
    S.append(extrude(H, FOUND, HEAD_TOP - 0.6, tone(GRAN)))
    S.append(extrude(offset(H, 0.06), 21.2, HEAD_TOP - 0.6, tone(LIME, 0.97), top=False))     # inscription band
    S.append(extrude(offset(H, 0.3), HEAD_TOP - 0.6, HEAD_TOP, tone(LIME)))                  # cornice
    S.append(cap(H, HEAD_TOP + 0.02, LEAD))
    S.append(extrude(offset(H, 0.12), FOUND, 0.6, tone(GRAN, 0.9)))                           # plinth
    R = ccw(H)
    for i in range(len(R)):                                                                    # balustrade/cresting
        p, q = R[i], R[(i + 1) % len(R)]
        if math.dist(p, q) < 1.6 or (p[0] > X_HEAD - 0.5 and q[0] > X_HEAD - 0.5):
            continue
        S.append(cresting(p, q, HEAD_TOP, LIME, w=0.35, h=0.8, pitch=0.8, spike=0.6))
    # the north face of the crossing: the great window, the chains, the frieze, the blind arcade
    F = fr((0.0, 28.62), (0, 1))
    D.append(rect_on(F, -16.6, 10.4, 12.6, 10.8, 0.04, tone(GRAN, 1.05)))
    D.append(lancet_on(F, -16.6, 11.8, 4.8, 9.0, 0.08, tone(LIME, 1.02), 0.75))
    D.append(lancet_on(F, -16.6, 12.4, 3.6, 7.8, 0.12, GLASS, 0.75))
    D.append(rect_on(F, -16.6, 12.4, 0.18, 6.0, 0.14, LIME))
    D.append(rect_on(F, -16.6, 10.5, 12.6, 0.8, 0.12, tone(LIME, 0.95)))                     # frieze
    D.append(festoon(F, -22.8, -10.4, 11.2))
    D += chains(F, -22.6, -19.4, 13.2, rows=3)
    D += chains(F, -13.8, -10.6, 13.2, rows=3)
    for u in np.linspace(-22.4, -10.8, 12):
        D.append(lancet_on(F, u, 5.4, 0.8, 4.6, 0.05, tone(GRAN_D), 0.8, 3))
        D.append(rect_on(F, u, 6.6, 0.1, 1.0, 0.09, IRON))
    for u in (-22.7, -10.5):
        D.append(figure(u, 14.0, 29.3, 1.8, LIME))
        D.append(B(u - 0.5, u + 0.5, 13.6, 14.0, 28.6, 29.5, LIME))
    # the east face of the head (the chancel), chains and blind arcades
    F = fr((-29.03, 17.15), (-1, 0))
    D.append(rect_on(F, 0, 11.2, 6.4, 0.7, 0.12, tone(LIME, 0.95)))
    D.append(festoon(F, -3.2, 3.2, 11.9, n=3))
    D += chains(F, -3.0, 3.0, 13.0, rows=2)
    for u in np.linspace(-2.6, 2.6, 5):
        D.append(lancet_on(F, u, 14.8, 1.15, 5.8, 0.05, tone(GRAN_D), 0.8, 3))
        D.append(lancet_on(F, u, 4.8, 1.0, 5.6, 0.05, tone(GRAN_D), 0.8, 3))
    for u, z0 in ((-24.0, 6.4), (-24.4, 27.9)):                                                # chains on the side faces
        nz = -1 if z0 < 10 else 1
        F2 = fr((0.0, z0), (0, nz))
        uu = u * (1 if nz > 0 else -1)
        D += chains(F2, uu - 0.4, uu + 0.4, 13.0, rows=2)
    # big pinnacles on the corners of the head
    for x, z in ((-10.2, 28.3), (-24.3, 28.0), (-30.3, 21.7), (-30.3, 12.3), (-24.5, 6.4), (-10.2, 6.4)):
        D.append(B(x - 0.75, x + 0.75, HEAD_TOP, HEAD_TOP + 1.6, z - 0.75, z + 0.75, tone(LIME)))
        D.append(pinnacle(x, z, HEAD_TOP + 1.6, 7.2, 0.75, tone(LIME)))
    return S, D


def lantern():
    """The octagonal cimborrio over the crossing: windows, corner buttresses with pinnacles, balustrade, dome."""
    S = []
    cx, cz = LC
    ring = ngon(cx, cz, LR, 8, rot=math.pi / 8)
    Y0, Y1 = HEAD_TOP, 31.2
    S.append(extrude(ring, Y0 - 0.5, Y1 - 0.4, tone(LIME, 1.0)))
    S.append(extrude(offset(ring, 0.3), Y1 - 0.4, Y1, tone(LIME, 1.04)))
    for i in range(8):
        p, q = ring[i], ring[(i + 1) % 8]
        m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        nrm = (m[0] - cx, m[1] - cz)
        F = fr(m, nrm)
        S.append(lancet_on(F, 0, 25.2, 3.0, 4.8, 0.05, tone(LIME, 0.92), 0.75))
        S.append(lancet_on(F, 0, 25.6, 2.1, 4.0, 0.09, GLASS, 0.75))
        S.append(rect_on(F, 0, 25.6, 0.12, 2.6, 0.11, LIME))
        S.append(cresting(p, q, Y1, LIME, w=0.3, h=0.7, pitch=0.7, spike=0.45))
    for i, (x, z) in enumerate(ngon(cx, cz, LR + 0.25, 8, rot=math.pi / 8)):
        S.append(cyl(0.6, 0.6, Y1 + 0.6 - Y0, tone(LIME), seg=4).R(0, math.pi / 4 + i * math.pi / 4 + math.pi / 8, 0).T(x, Y0, z))
        S.append(pinnacle(x, z, Y1 + 0.6, 4.6, 0.55, tone(LIME), rot=math.pi / 4 + i * math.pi / 4 + math.pi / 8))
    # the dome with its ribs, the little lantern and the cross
    prof = [(LR * 0.86 * math.cos(t) ** 0.8, Y1 - 0.4 + 3.4 * math.sin(t)) for t in np.linspace(0, math.pi / 2, 7)]
    prof[-1] = (0.0, prof[-1][1])
    S.append(lathe(prof, tone(LIME, 0.95), 16).T(cx, 0, cz))
    for i in range(8):
        a = math.pi / 8 + i * math.pi / 4
        pts = [(cx + r * math.cos(a), y + 0.12, cz + r * math.sin(a)) for r, y in prof[:-1]]
        S.append(tube(pts, 0.14, LIME, 4))
    yt = Y1 - 0.4 + 3.4
    S.append(cyl(0.95, 0.95, 1.3, SLATE, seg=8).T(cx, yt - 0.2, cz))
    S.append(cyl(1.05, 0.05, 1.0, SLATE, seg=8).T(cx, yt + 1.1, cz))
    S.append(sphere(0.2, IRON, 6, 4).T(cx, yt + 2.2, cz))
    S.append(B(cx - 0.06, cx + 0.06, yt + 2.2, yt + 3.6, cz - 0.06, cz + 0.06, IRON))
    S.append(B(cx - 0.45, cx + 0.45, yt + 3.0, yt + 3.12, cz - 0.06, cz + 0.06, IRON))
    return S


def nave():
    S, D = [], []
    N = clip(ccw(CHURCH), 0, X_HEAD, keep_low=False)
    body = clip(N, 1, Z_NAVE_S, keep_low=False)
    chapels = clip(N, 1, Z_NAVE_S, keep_low=True)
    S.append(extrude(body, FOUND, NAVE_TOP - 0.4, tone(GRAN)))
    S.append(extrude(offset(body, 0.25), NAVE_TOP - 0.4, NAVE_TOP, tone(LIME, 0.98)))
    S.append(extrude(offset(body, 0.12), FOUND, 0.6, tone(GRAN, 0.9)))
    S.append(tent_roof(offset(body, 0.1), (X_HEAD, 20.07), (27.4, 20.07), NAVE_TOP, 26.0, ROOF))
    R = ccw(body)
    for i in range(len(R)):
        p, q = R[i], R[(i + 1) % len(R)]
        if math.dist(p, q) < 1.2 or (p[0] < X_HEAD + 0.3 and q[0] < X_HEAD + 0.3):
            continue
        D.append(cresting(p, q, NAVE_TOP, LIME))
    # the chapels between the south buttresses (lower, under the cloister roof)
    S.append(extrude(chapels, FOUND, 13.6, tone(GRAN, 0.97), top_c=ROOF))
    # buttresses with pinnacles and statues
    for x in BUTTS:
        S.append(B(x - 0.9, x + 0.9, FOUND, NAVE_TOP, 28.4, 29.5, tone(GRAN)))
        S.append(B(x - 1.0, x + 1.0, NAVE_TOP - 0.4, NAVE_TOP, 28.3, 29.6, tone(LIME)))
        D.append(pinnacle(x, 29.0, NAVE_TOP, 5.4, 0.62, tone(LIME, 0.97)))
        if x > -9:
            D.append(B(x - 0.5, x + 0.5, 13.3, 13.7, 29.5, 30.2, LIME))
            D.append(figure(x, 13.7, 29.85, 1.8, tone(LIME, 0.95)))
        if x > -9:
            S.append(B(x - 0.9, x + 0.9, FOUND, NAVE_TOP, 6.3, Z_NAVE_S, tone(GRAN)))
            D.append(pinnacle(x, 7.4, NAVE_TOP, 5.8, 0.62, tone(LIME, 0.97)))
    # windows: north (with the cresting along their sills) and south (above the chapels)
    Fn = fr((0.0, 28.62), (0, 1))
    Fs = fr((0.0, Z_NAVE_S), (0, -1))
    for s in BAYS:
        D.append(lancet_on(Fn, s, 13.9, 3.8, 6.3, 0.05, tone(LIME, 0.95), 0.72))
        D.append(lancet_on(Fn, s, 14.2, 3.0, 5.6, 0.09, GLASS, 0.72))
        D.append(rect_on(Fn, s, 14.2, 0.14, 3.6, 0.11, LIME))
        D.append(rect_on(Fn, s, 21.4, 0.7, 0.75, 0.05, DARK))
        if abs(s - 2.95) > 1:
            D.append(cresting((s - 3.4, 29.0), (s + 3.4, 29.0), 13.2, LIME, w=0.6, h=0.4, pitch=0.6, spike=0.45))
        us = -s                                                                       # south frame runs along −x
        D.append(lancet_on(Fs, us, 15.2, 3.6, 5.6, 0.05, tone(LIME, 0.95), 0.72))
        D.append(lancet_on(Fs, us, 15.5, 2.8, 4.9, 0.09, GLASS, 0.72))
    D.append(round_on(Fn, 13.4, 5.2, 0.9, 1.6, 0.05, DARK))
    D.append(round_on(Fn, 21.8, 5.2, 0.9, 1.6, 0.05, DARK))
    D += north_portal()
    D += west_end()
    return S, D


def north_portal():
    """The north portal (Alonso de Covarrubias, 1553–1610): clustered pilasters with statues, a round-arched door,
    the frontispiece with its niche and gable, the cross."""
    ms = []
    F = fr((2.95, 28.62), (0, 1))
    xc, zf = 2.95, 28.62
    pts = [(-3.6, 0.0), (3.6, 0.0), (3.6, 12.2), (2.4, 12.6), (1.4, 13.6), (0.6, 14.9), (0.0, 15.4),
           (-0.6, 14.9), (-1.4, 13.6), (-2.4, 12.6), (-3.6, 12.2)]
    ms.append(slab(F, pts, 0.5, tone(LIME, 0.98), off=0.25))
    for u in (-3.3, -2.3, 2.3, 3.3):
        ms.append(B(xc + u - 0.38, xc + u + 0.38, 0.0, 12.9, zf, zf + 0.95, tone(LIME, 1.02)))
        ms.append(pinnacle(xc + u, zf + 0.6, 12.9, 2.6, 0.32, tone(LIME, 1.02)))
    ms.append(round_on(F, 0, 0.0, 4.4, 7.5, 0.52, tone(LIME, 1.06), 10))
    ms.append(round_on(F, 0, 0.0, 3.4, 6.6, 0.56, WOOD, 10))
    ms.append(rect_on(F, 0, 0.0, 0.06, 4.8, 0.58, (0.22, 0.13, 0.08)))
    ms.append(rect_on(F, 0, 8.0, 5.6, 0.5, 0.56, tone(LIME, 1.04)))                         # entablature
    ms.append(round_on(F, 0, 9.2, 1.6, 2.8, 0.52, tone(LIME, 0.85), 8))                     # niche
    ms.append(figure(xc, 9.3, zf + 0.75, 1.9, tone(LIME, 1.05)))
    for u in (-2.8, 2.8):
        for y in (1.8, 4.9):
            ms.append(figure(xc + u, y, zf + 1.15, 1.35, tone(LIME, 1.05)))
    ms.append(B(xc - 0.12, xc + 0.12, 15.3, 17.0, zf + 0.25, zf + 0.5, LIME))
    ms.append(B(xc - 0.6, xc + 0.6, 16.1, 16.35, zf + 0.25, zf + 0.5, LIME))
    ms.append(B(xc - 4.4, xc + 4.4, FOUND, 0.15, zf, zf + 2.2, tone(PAVE)))                # step
    return ms


def west_end():
    """The west end: the west door at the level of the street below, and the stair turret on the north-west corner."""
    ms = []
    p = (35.80, 18.0)
    nrm = (0.98, 0.19)
    F = fr(p, nrm)
    ys = -4.0
    ms.append(slab(F, [(-2.8, ys), (2.8, ys), (2.8, ys + 8.4), (0.0, ys + 9.6), (-2.8, ys + 8.4)], 0.5, tone(LIME, 0.98), off=0.25))
    ms.append(round_on(F, 0, ys, 2.7, 5.0, 0.53, WOOD, 10))
    ms.append(lancet_on(F, 0, 13.0, 3.2, 6.2, 0.05, tone(LIME, 0.95), 0.72))
    ms.append(lancet_on(F, 0, 13.3, 2.4, 5.5, 0.09, GLASS, 0.72))
    o, t, n = F
    c = o + n * 1.2
    ms.append(box(4.0, ys + 0.15 - FOUND, 2.4, tone(PAVE)).R(0, math.atan2(-t[2], t[0]), 0).T(c[0], FOUND, c[2]))
    # the stair turret
    tx, tz = 28.6, 25.4
    ms.append(B(tx - 1.6, tx + 1.6, NAVE_TOP - 1.0, 27.2, tz - 1.6, tz + 1.6, tone(GRAN)))
    ms.append(B(tx - 1.8, tx + 1.8, 27.2, 27.6, tz - 1.8, tz + 1.8, tone(LIME)))
    for dx in (-1.4, 1.4):
        for dz in (-1.4, 1.4):
            ms.append(pinnacle(tx + dx, tz + dz, 27.6, 2.8, 0.35, tone(LIME)))
    ms.append(cyl(1.6, 0.05, 2.2, tone(LIME, 0.95), seg=4).R(0, math.pi / 4, 0).T(tx, 27.6, tz))
    for u in (-0.8, 0.8):
        ms.append(round_on(fr((tx, tz + 1.6), (0, 1)), u - tx * 0, 24.4, 0.7, 1.8, 0.04, DARK))
    return ms


# ── The cloister ─────────────────────────────────────────────────────────────
Y_FLOOR = 0.3
Y_MID0, Y_MID1 = 7.6, 8.4          # the vault/floor between the two galleries
Y_EAVE = 12.9                      # eaves of the lean-to roofs over the walks
Y_TOP = 13.6                       # where they meet the walls of the ranges


def cloister():
    S, D = [], []
    c0 = np.mean(np.asarray(COURT), axis=0)
    # walks: floor, the vault between the floors, the outer walls of the walks, the lean-to roofs
    S.append(cap(GALLERY, Y_FLOOR, tone(PAVE), holes=[COURT]))
    S.append(cap(GALLERY, Y_MID0, tone(CLOI, 0.8), holes=[COURT], up=False))
    S.append(cap(GALLERY, Y_MID1, tone(PAVE, 1.05), holes=[COURT]))
    S.append(walls(GALLERY, FOUND, Y_TOP, tone(CLOI, 0.88)))
    G = ccw(GALLERY)
    Cc = ccw(COURT)
    # match each courtyard corner with the nearest gallery corner for the roof strips
    gm = [min(G, key=lambda g: math.dist(g, c)) for c in Cc]
    for i in range(4):
        a, b = Cc[i], Cc[(i + 1) % 4]
        ga, gb = gm[i], gm[(i + 1) % 4]
        S.append(strip([(a[0], Y_EAVE, a[1]), (b[0], Y_EAVE, b[1])], [(ga[0], Y_TOP, ga[1]), (gb[0], Y_TOP, gb[1])], ROOF))
    # the arcades on the garden, five bays a side
    for i in range(4):
        p, q = np.array(Cc[i]), np.array(Cc[(i + 1) % 4])
        L = float(np.linalg.norm(q - p))
        e = (q - p) / L
        nrm = np.array([e[1], -e[0]])
        if np.dot(nrm, c0 - (p + q) / 2) < 0:
            nrm = -nrm
        mid = (p + q) / 2
        F = fr(mid, nrm)
        o, t, n = F
        sgn = 1.0 if np.dot([t[0], t[2]], e) > 0 else -1.0
        nb = 5
        pitch = L / nb
        for k in range(nb + 1):                                          # the piers with their pinnacles
            pt = p + e * k * pitch
            a0 = pt - nrm * 0.2
            a1 = pt + nrm * 0.95
            S.append(obox(a0, a1, FOUND, Y_EAVE - 0.2, 1.05, tone(CLOI, 1.02)))
            tip = pt + nrm * 0.55
            D.append(pinnacle(tip[0], tip[1], Y_EAVE - 0.2, 3.8, 0.42, tone(CLOI, 1.04)))
            D.append(B(tip[0] - 0.3, tip[0] + 0.3, 4.6, 5.6, tip[1] - 0.3, tip[1] + 0.3, tone(CLOI, 1.08)))
        for k in range(nb):
            s = sgn * (-L / 2 + (k + 0.5) * pitch)
            w = pitch - 1.05
            # lower: a pointed arch with tracery
            ys = 4.9
            pts = arc_pts(s, ys, w, 0.72, 6) + [(s - w / 2, Y_MID1), (s + w / 2, Y_MID1)]
            S.append(slab(F, pts, 0.6, tone(CLOI), off=0.0))
            D.append(slab(F, [(s - 0.09, Y_FLOOR), (s + 0.09, Y_FLOOR), (s + 0.09, ys + 1.0), (s - 0.09, ys + 1.0)], 0.25, CLOI))
            # the tracery in the head of the arch: two small pointed lights under a solid tympanum
            tym = arc_pts(s, ys, w - 0.2, 0.72, 6)
            yb = ys + 0.55
            D.append(slab(F, [(u, max(v, yb)) for u, v in tym], 0.14, tone(CLOI, 1.03)))
            for sx in (-1, 1):
                D.append(lancet_on(F, s + sx * w / 4.2, yb, w / 2 - 0.7, 1.05, 0.08, DARK, 0.7, 3))
            D.append(rect_on(F, s, Y_FLOOR, w, 0.5, 0.05, tone(CLOI, 0.95)))                  # low wall (podium)
            # upper: a mixtilinear arch over a balustrade
            ys2 = 10.9
            pts2 = round_arc(s, ys2, w, 8)
            pts2[len(pts2) // 2] = (s, ys2 + w / 2 + 0.35)                    # the little point of the mixtilinear arch
            S.append(slab(F, pts2 + [(s - w / 2, Y_EAVE - 0.2), (s + w / 2, Y_EAVE - 0.2)], 0.5, tone(CLOI), off=0.0))
            D.append(slab(F, [(s - w / 2, Y_MID1), (s + w / 2, Y_MID1), (s + w / 2, Y_MID1 + 1.0), (s - w / 2, Y_MID1 + 1.0)],
                          0.3, tone(CLOI, 1.04), off=0.05))
            for j in range(int(w / 0.35)):
                u = s - w / 2 + (j + 0.5) * w / int(w / 0.35)
                D.append(rect_on(F, u, Y_MID1 + 0.15, 0.12, 0.7, 0.22, DARK))
        # cornice and cresting along the arcade
        D.append(obox(p + nrm * 0.3, q + nrm * 0.3, Y_EAVE - 0.2, Y_EAVE + 0.1, 0.8, tone(CLOI, 1.05)))
        D.append(cresting(p + nrm * 0.45, q + nrm * 0.45, Y_EAVE + 0.1, tone(CLOI, 1.06), w=0.3, h=0.4, pitch=0.6, spike=0.4))
        D.append(obox(p + nrm * 0.1, q + nrm * 0.1, Y_MID0, Y_MID1, 0.9, tone(CLOI, 1.05)))
    return S, D


def ranges():
    """The convent ranges around the cloister: rendered walls, windows, tiled roofs."""
    S, D = [], []
    S.append(extrude(RANGES, FOUND, 13.0, tone(RENDER)))
    S.append(extrude(offset(RANGES, 0.08), FOUND, 0.8, tone(GRAN, 0.9), top=False))
    S.append(hip(-27.9, 25.5, -31.0, -24.8, 13.0, 15.8, ROOF))
    S.append(hip(14.6, 23.2, -25.2, 6.8, 13.0, 15.8, ROOF))
    S.append(hip(7.1, 13.6, -35.9, -30.6, 13.0, 15.0, ROOF))
    S.append(extrude(EAST_RANGE, FOUND, 7.6, tone(RENDER, 0.97)))
    S.append(extrude(offset(EAST_RANGE, 0.08), FOUND, 0.8, tone(GRAN, 0.9), top=False))
    S.append(hip(-27.9, -15.6, -25.3, 6.0, 7.6, 10.2, ROOF))
    # windows on the outer faces
    R = ccw(RANGES)
    for i in range(len(R)):
        p, q = np.array(R[i]), np.array(R[(i + 1) % len(R)])
        L = float(np.linalg.norm(q - p))
        if L < 6:
            continue
        e = (q - p) / L
        nrm = (e[1], -e[0])
        m = (p + q) / 2
        if (abs(m[0]) < 16 and m[1] > -26.5 and m[1] < 7) or (m[0] > 14 and m[0] < 17.5 and m[1] > -26):
            continue                                                   # inner faces (the walks)
        n = int(L / 3.6)
        for k in range(n):
            F = fr(p + e * (k + 0.5) * L / n, nrm)
            for y in (3.2, 8.6):
                D.append(rect_on(F, 0, y, 0.95, 1.6, 0.05, DARK))
    R = ccw(EAST_RANGE)
    for i in range(len(R)):
        p, q = np.array(R[i]), np.array(R[(i + 1) % len(R)])
        L = float(np.linalg.norm(q - p))
        if L < 12:
            continue
        e = (q - p) / L
        nrm = (e[1], -e[0])
        if nrm[0] > 0:
            continue
        n = int(L / 4.0)
        for k in range(n):
            D.append(rect_on(fr(p + e * (k + 0.5) * L / n, nrm), 0, 4.6, 0.9, 1.4, 0.05, DARK))
    return S, D


# ── The garden of the cloister (four beds around the well) ──────────────────
def blob(cx, cy, cz, r, flat, col, seed, seg=9, rings=5):
    m = sphere(r, col, seg, rings).S(1, flat, 1)
    crumple(m, r * 0.32, 1.3 / r, seed)
    m.T(cx, cy, cz)
    c = np.asarray(col, float)
    lo = cy - r * flat

    def tint(v):
        t = np.clip((v[:, 1] - lo) / (2 * r * flat + 1e-6), 0, 1)
        k = 0.66 + 0.44 * t + 0.16 * (noise3(v, 0.9, seed) - 0.5)
        return np.clip(np.outer(k, c), 0, 1)
    return m.paint(tint)


def limb(p0, p1, r0, r1, col, seg=6, bend=0.0, n=3):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    out = []
    pts = [p0 + (p1 - p0) * t + np.array([0, bend * math.sin(math.pi * t), 0]) for t in np.linspace(0, 1, n + 1)]
    for k in range(n):
        r = r0 + (r1 - r0) * (k + 0.5) / n
        out.append(tube([pts[k], pts[k + 1]], r, col, seg))
    return out


def fir(x, z, y, h, crown, seed):
    """A tall conifer (the fir of the cloister): straight trunk, tiers of drooping clumps narrowing to the top."""
    rr = np.random.default_rng(seed)
    BK = (0.36, 0.27, 0.20); FOL = (0.14, 0.26, 0.15)
    out = limb((x, y, z), (x, y + h, z), 0.35, 0.08, BK, 6, n=3)
    for k in range(26):
        t = 0.18 + 0.8 * k / 25
        R = crown / 2 * (1.0 - t) ** 0.9 + 0.3
        a = k * 2.4 + rr.normal(0, 0.2)
        out.append(blob(x + R * 0.55 * math.cos(a), y + h * t, z + R * 0.55 * math.sin(a), max(0.5, R * 0.7), 0.55, FOL, seed + k, 8, 4))
    out.append(blob(x, y + h * 0.98, z, 0.45, 1.5, FOL, seed + 99, 7, 4))
    return out


def leafy(x, z, y, h, crown, seed, FOL=(0.20, 0.33, 0.16), n=12):
    """A broad-leaved tree (magnolia, orange): a short trunk and a round crown of clumps."""
    rr = np.random.default_rng(seed)
    BK = (0.34, 0.27, 0.21)
    out = limb((x, y, z), (x, y + h * 0.5, z), 0.22 * h / 8 + 0.08, 0.12, BK, 6, n=2)
    cy = y + h * 0.62
    for k in range(n):
        u = rr.normal(size=3); u /= np.linalg.norm(u)
        u[1] = u[1] * 0.6
        p = (x + u[0] * crown * 0.32, cy + u[1] * h * 0.3, z + u[2] * crown * 0.32)
        out += limb((x, y + h * 0.45, z), p, 0.1, 0.05, BK, 4, n=1)
        out.append(blob(*p, crown * (0.24 + 0.06 * rr.random()), 0.8, FOL, seed + k, 8, 4))
    return out


def garden():
    G = []
    c0 = np.mean(np.asarray(COURT), axis=0)
    S0 = Y_FLOOR - 0.15
    G.append(cap(COURT, S0, tone(EARTH)))
    G.append(walls(COURT, FOUND, S0, tone(PAVE), hole=True))
    ux = np.array(COURT[0]) - np.array(COURT[1]); ux /= np.linalg.norm(ux)       # the garden's own axes
    uz = np.array([-ux[1], ux[0]])
    ang = math.atan2(-ux[1], ux[0])

    def at(a, b):
        return c0 + ux * a + uz * b

    hw = 9.6
    for a in (-1, 1):
        for b in (-1, 1):
            cxz = at(a * hw / 2 + a * 0.4, b * hw / 2 + b * 0.4)
            bed = box(hw - 2.4, 0.25, hw - 2.4, tone(GRASS)).R(0, ang, 0).T(cxz[0], S0, cxz[1])
            G.append(bed)
            for side in range(4):                                       # low box hedges round the bed
                d = (hw - 2.4) / 2
                for (u0, v0, u1, v1) in (((-d, -d, d, -d), (d, -d, d, d), (d, d, -d, d), (-d, d, -d, -d))[side],):
                    pa = cxz + ux * u0 + uz * v0
                    pb = cxz + ux * u1 + uz * v1
                    G.append(obox(pa, pb, S0, S0 + 0.55, 0.45, HEDGE))
    # the well in the middle
    G.append(cyl(1.25, 1.25, 0.15, tone(PAVE), seg=8).T(c0[0], S0, c0[1]))
    G.append(cyl(0.85, 0.85, 0.8, tone(CLOI, 1.02), seg=8).T(c0[0], S0, c0[1]))
    G.append(cyl(0.65, 0.65, 0.02, (0.12, 0.15, 0.17), seg=8).T(c0[0], S0 + 0.79, c0[1]))
    for sx in (-1, 1):
        G.append(B(c0[0] + sx * 0.75 - 0.06, c0[0] + sx * 0.75 + 0.06, S0 + 0.8, S0 + 2.6, c0[1] - 0.06, c0[1] + 0.06, IRON))
    G.append(B(c0[0] - 0.8, c0[0] + 0.8, S0 + 2.5, S0 + 2.62, c0[1] - 0.06, c0[1] + 0.06, IRON))
    # trees: the tall fir, a magnolia, two orange trees
    p = at(-5.2, -5.0); G += fir(p[0], p[1], S0 + 0.25, 15.0, 5.5, 11)
    p = at(5.0, 5.0); G += leafy(p[0], p[1], S0 + 0.25, 8.5, 7.0, 21, (0.16, 0.30, 0.14), 14)
    p = at(5.0, -5.0); G += leafy(p[0], p[1], S0 + 0.25, 4.2, 3.6, 31, (0.22, 0.36, 0.15), 8)
    p = at(-5.0, 5.0); G += leafy(p[0], p[1], S0 + 0.25, 4.0, 3.4, 41, (0.22, 0.36, 0.15), 8)
    return G


def main():
    out = ROOT / "models" / "Q1289106"
    out.mkdir(parents=True, exist_ok=True)
    hs, hd = head()
    ns, nd = nave()
    cs, cd = cloister()
    rs, rd = ranges()
    church = sink(merge(hs + hd + lantern() + ns + nd))
    cl = sink(merge(cs + cd + rs + rd))
    gd = sink(merge(garden()))
    parts = [("church", church), ("cloister", cl), ("garden", gd)]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.9)
    allv = np.vstack([m.v for _, m in parts])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
