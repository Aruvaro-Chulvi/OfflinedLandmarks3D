"""
Alcázar de Toledo (Q1326589), Toledo — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: the fortress-palace on the highest point of Toledo, rebuilt for Charles V from 1535: the north façade
and the courtyard by Alonso de Covarrubias (with Francisco de Villalpando, the staircase), the south façade by Juan
de Herrera (1570s), the east façade with its medieval round turrets; burnt in 1710, 1810 and 1887, destroyed in the
siege of 1936 and rebuilt as it was (1939–1961). Today the Museo del Ejército (whose modern building, 2010, is a
separate building north and east of it and is not included). Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 83998985 (6506 m², the four corner towers and the ranges as building:parts), in model x/z for
  heading 1.6° (python tools/osm.py way 83998985 --heading 1.6): the main body 68 × 89.5 m between the façades, the
  four towers (ways 1173930253–56, about 12 × 12 m, 48 m with their slate roofs, the needles 60 m: ways 1552714420,
  1552714601–03), the courtyard (way 194447882, 30.5 × 38.8 m, paved), the ranges round it (way 1552568908), the
  staircase range (way 1552568907) and the south range (way 1552568906), the three round turrets of the east façade
  (ways 1552710873–75).
- Wikipedia (es): «a square building of 60 m a side with four towers of 60 m, each crowned by a slate spire»; the
  courtyard with two floors of arches on columns; north façade by Covarrubias (three floors of windows, the portal
  with the imperial arms), south façade by Herrera; the statue of Charles V in the courtyard.
- Heights from photographs on Wikimedia Commons scaled by the OSM widths of the towers (±10 %): façade cornice 28.5 m
  above the north esplanade, ridge 34.5 m, tower cornice 33 m, slate spire to 41.5 m, its lantern to 46 m and the
  needle to 50.5 m (about 61 m above the street on the south, the «60 m» of Wikipedia).
- PNOA orthophoto (IGN): the esplanade in front of the north façade, the walk between the west towers above the
  Cuesta de Carlos V, the terrace on the east side (car park) above the gardens, the street along the south façade.
- Terrain: IGN MDT05. y = 0 is the north esplanade and the courtyard (ground_at, about 546.5 m); the ground falls to
  about −8 m on the west (Cuesta de Carlos V), −10 m on the south (Cuesta de los Capuchinos) and −10.5 to −13 m on the
  east (car park and gardens): every wall goes down to 1.5 m below the real ground under it.

Axes (docs/MODEL_SPEC.md): +Y up, the north façade faces +Z (heading 1.6°), +X points to 271.6° (west), origin at the
centroid of the OSM outline (in the courtyard).

    python model_sources/Q1326589_alcazar_de_toledo.py   -> models/Q1326589/model.glb
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
from meshkit import Mesh, merge, box, cyl, lathe, tube, extrude, cap, walls, offset, triangulate, ring_area, write_glb  # noqa: E402

rng = np.random.default_rng(1535)

# ── Colours ──────────────────────────────────────────────────────────────────
ASHLAR = (0.82, 0.78, 0.70)        # the pale granite/limestone ashlar of the façades and towers
ASHLAR_D = (0.72, 0.68, 0.61)
RUBBLE = (0.71, 0.61, 0.49)        # Toledo masonry: rubble between brick courses (west and east façades)
BRICK = (0.70, 0.42, 0.30)         # the red panels of Herrera's south façade
TILE = (0.65, 0.44, 0.34)
SLATE = (0.24, 0.26, 0.30)
DARK = (0.10, 0.10, 0.12)
GLASS = (0.17, 0.19, 0.22)
IRON = (0.07, 0.07, 0.08)
WOOD = (0.33, 0.20, 0.12)
PAVE = (0.80, 0.77, 0.71)
BRONZE = (0.30, 0.26, 0.20)

FOUND = -3.0
HEADING = 1.6
GROUND_AT = (0.0, 50.0)

# ── OpenStreetMap (osm.py way 83998985 --heading 1.6), model metres ──────────
T_NE = [(-27.94, 43.84), (-27.94, 40.66), (-34.02, 40.60), (-36.42, 40.55), (-40.34, 40.45), (-40.35, 52.21), (-29.16, 52.22), (-27.94, 52.22)]
T_NW = [(34.31, 41.14), (28.32, 41.20), (28.46, 43.89), (28.45, 48.52), (28.45, 52.00), (32.49, 52.00), (39.75, 52.01), (39.64, 48.10), (39.52, 40.99)]
T_SE = [(-33.13, -34.73), (-27.47, -34.55), (-27.02, -45.79), (-26.97, -47.01), (-38.96, -47.45), (-39.47, -35.00), (-35.84, -34.84)]
T_SW = [(25.30, -45.47), (25.12, -33.25), (27.95, -33.10), (34.12, -33.05), (34.63, -33.07), (38.83, -33.07), (39.06, -46.19), (25.36, -46.36)]
TOWERS = [T_NE, T_NW, T_SE, T_SW]
CUBO_N = [(-33.96, 35.62), (-35.48, 36.00), (-36.64, 37.13), (-37.09, 38.70), (-36.68, 40.28), (-36.42, 40.55), (-34.02, 40.60)]
CUBO_S = [(-33.19, -30.05), (-33.13, -34.73), (-35.84, -34.84), (-35.90, -34.79), (-36.34, -33.22), (-35.94, -31.64), (-34.80, -30.48), (-33.23, -30.05)]
CUBO_M = [(-33.65, 8.91), (-33.57, 2.65), (-35.15, 3.05), (-36.31, 4.19), (-36.76, 5.75), (-36.35, 7.32), (-35.22, 8.48)]
BODY = [(-33.13, -45.75), (25.30, -45.47), (34.55, -45.40), (34.31, 43.89), (-27.94, 43.84), (-33.98, 43.84)]
COURT = [(15.80, -13.90), (15.60, 24.85), (-14.95, 24.75), (-14.35, -14.20)]          # the open courtyard
GAL = [(21.16, -18.81), (20.30, 29.87), (-20.12, 29.55), (-19.09, -19.24)]            # outer edge of its galleries

Z_N, Z_S, X_W, X_E = 43.87, -45.6, 34.4, -33.6     # the four façades
Y_CORN, Y_RIDGE, Y_IN = 28.5, 34.5, 27.6
Y_TOWER = 33.0


# ── Terrain (IGN MDT05 in models/Q1326589/terrain.json): foundations follow the real ground ──
def _terrain():
    p = ROOT / "models" / "Q1326589" / "terrain.json"
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


# ── Helpers (as in Q1289106_san_juan_de_los_reyes.py) ────────────────────────
def tone(c, k=1.0, s=0.015):
    return tuple(np.clip(np.asarray(c) * k * (1 + rng.normal(0, s)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sphere(r, c, seg=8, rings=5):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def obox(p, q, y0, y1, w, c):
    (x0, z0), (x1, z1) = p, q
    L = math.hypot(x1 - x0, z1 - z0)
    a = math.atan2(-(z1 - z0), x1 - x0)
    return box(L, y1 - y0, w, c).R(0, a, 0).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def ccw(ring):
    return list(ring) if ring_area(ring) > 0 else list(ring)[::-1]


def centroid(ring):
    P = np.asarray(ring, float)
    return float(P[:, 0].mean()), float(P[:, 1].mean())


def fr(p, n):
    nn = np.array([n[0], 0.0, n[1]], float)
    nn /= np.linalg.norm(nn)
    return np.array([p[0], 0.0, p[1]], float), np.array([nn[2], 0.0, -nn[0]]), nn


def poly_on(F, pts, eps, c):
    o, t, n = F
    P, tris = triangulate(pts)
    oo = o + n * eps
    V = np.array([oo + u * t + np.array([0, v, 0]) for u, v in P])
    m = Mesh(V, tris, c)
    if np.dot(np.cross(t, [0.0, 1.0, 0.0]), n) < 0:
        m.f = m.f[:, ::-1]
    return m


def rect_on(F, s, y0, w, h, eps, c):
    return poly_on(F, [(s - w / 2, y0), (s + w / 2, y0), (s + w / 2, y0 + h), (s - w / 2, y0 + h)], eps, c)


def round_on(F, s, y0, w, h, eps, c, seg=8):
    ys = y0 + h - w / 2
    arc = [(s + w / 2 * math.cos(a), ys + w / 2 * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)]
    return poly_on(F, [(s - w / 2, y0), (s + w / 2, y0)] + arc, eps, c)


def pediment_on(F, s, y0, w, h, eps, c):
    return poly_on(F, [(s - w / 2, y0), (s + w / 2, y0), (s, y0 + h)], eps, c)


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


def quad3(a, b, c, d, col):
    return Mesh([a, b, c, d], [(0, 1, 2), (0, 2, 3)], col)


def railing(p, q, y, h=1.0, c=IRON, pitch=0.25):
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = float(np.linalg.norm(q - p))
    ms = [obox(p, q, y + h - 0.06, y + h, 0.07, c), obox(p, q, y + 0.1, y + 0.15, 0.05, c)]
    n = max(1, int(L / pitch))
    for k in range(n + 1):
        x, z = p + (q - p) * k / n
        ms.append(B(x - 0.02, x + 0.02, y, y + h, z - 0.02, z + 0.02, c))
    return merge(ms)


def balustrade(p, q, y, h=1.0, c=ASHLAR, pitch=0.4):
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = float(np.linalg.norm(q - p))
    ms = [obox(p, q, y + h - 0.15, y + h, 0.4, c), obox(p, q, y, y + 0.15, 0.4, c)]
    n = max(1, int(L / pitch))
    for k in range(n):
        x, z = p + (q - p) * (k + 0.5) / n
        ms.append(cyl(0.1, 0.1, h - 0.3, c, seg=5).T(x, y + 0.15, z))
    return merge(ms)


# ── The body: walls, façades, roofs ──────────────────────────────────────────
def edge_frame(a, b, out):
    """A frame on the wall a→b (outward normal close to out), its origin where the origin of the model projects."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    e = (b - a) / np.linalg.norm(b - a)
    n = np.array([e[1], -e[0]])
    if np.dot(n, out) < 0:
        n = -n
    p = a + e * np.dot(-a, e)
    return fr(tuple(p), tuple(n))


def facade_frames():
    """The four façades follow the OSM walls (they are not quite square to each other)."""
    return {"N": edge_frame((-33.98, 43.84), (34.31, 43.89), (0, 1)),
            "S": edge_frame((-33.13, -45.75), (34.55, -45.40), (0, -1)),
            "W": edge_frame((34.55, -45.40), (34.31, 43.89), (1, 0)),
            "E": edge_frame((-33.13, -45.75), (-33.98, 43.84), (-1, 0))}


def u_of(face, x, z):
    """Position along a façade frame for a model point."""
    return {"N": x, "S": -x, "W": -z, "E": z}[face]


def body():
    S, D = [], []
    S.append(extrude(BODY, FOUND, Y_CORN - 0.8, tone(ASHLAR), holes=[GAL]))
    S.append(extrude(offset(BODY, 0.5), Y_CORN - 0.8, Y_CORN, tone(ASHLAR, 1.03), top=False))
    # hipped roofs of the four ranges: outer eaves, ridge half-way, inner eaves over the galleries
    O = ccw([(-33.13, -45.75), (34.55, -45.40), (34.31, 43.89), (-33.98, 43.84)])
    I = ccw(GAL)

    def start_se(R):                                   # both rings counter-clockwise from the south-east corner
        k = min(range(4), key=lambda i: R[i][0] + R[i][1])
        return R[k:] + R[:k]
    O, I = start_se(O), start_se(I)
    Oe = offset(O, 0.6)
    Oe = start_se(ccw(Oe))
    Rr = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(Oe, I)]
    for i in range(4):
        j = (i + 1) % 4
        S.append(quad3((Oe[i][0], Y_CORN, Oe[i][1]), (Oe[j][0], Y_CORN, Oe[j][1]),
                       (Rr[j][0], Y_RIDGE, Rr[j][1]), (Rr[i][0], Y_RIDGE, Rr[i][1]), TILE))
        S.append(quad3((I[i][0], Y_IN, I[i][1]), (Rr[i][0], Y_RIDGE, Rr[i][1]),
                       (Rr[j][0], Y_RIDGE, Rr[j][1]), (I[j][0], Y_IN, I[j][1]), tone(TILE, 0.95)))
    S.append(extrude(GAL, Y_IN - 0.6, Y_IN, tone(ASHLAR, 1.03), top=False))
    F = facade_frames()
    D += north_facade(F["N"]) + south_facade(F["S"]) + west_facade(F["W"]) + east_facade(F["E"])
    return S, D


def window(F, s, y, w=1.3, h=2.5, ped=False, balcony=False, frame=ASHLAR, eps=0.05, arched=False):
    ms = [rect_on(F, s, y - 0.25, w + 0.6, h + 0.5, eps, tone(frame, 1.05))]
    ms.append((round_on if arched else rect_on)(F, s, y, w, h, eps + 0.04, GLASS))
    if ped:
        ms.append(rect_on(F, s, y + h + 0.25, w + 1.0, 0.25, eps + 0.08, tone(frame, 1.08)))
        ms.append(pediment_on(F, s, y + h + 0.5, w + 1.0, 0.8, eps + 0.08, tone(frame, 1.08)))
    if balcony:
        o, t, n = F
        p = o + t * (s - w / 2 - 0.3) + n * (eps + 0.5)
        q = o + t * (s + w / 2 + 0.3) + n * (eps + 0.5)
        ms.append(slab(F, [(s - w / 2 - 0.35, y - 0.35), (s + w / 2 + 0.35, y - 0.35), (s + w / 2 + 0.35, y - 0.1),
                           (s - w / 2 - 0.35, y - 0.1)], 1.0, tone(frame, 1.05), off=eps + 0.45))
        ms.append(railing((p[0], p[2]), (q[0], q[2]), y - 0.1, 0.9, IRON, 0.2))
    return ms


def north_facade(F):
    """Covarrubias's façade (from 1537): ashlar, three floors of windows, the arcaded top floor, the cresting of
    candlesticks, the portal with the imperial arms."""
    ms = []
    u0, u1 = -27.9, 28.4
    xs = [x for x in np.linspace(u0 + 2.6, u1 - 2.6, 12) if abs(x - 0.3) > 3.5]
    for x in xs:
        ms += window(F, x, 1.6, 1.3, 2.4)
        ms += window(F, x, 9.0, 1.4, 2.8, ped=True, balcony=True)
        ms += window(F, x, 15.6, 1.3, 2.6, ped=True, balcony=True)
    for x in np.linspace(u0 + 1.6, u1 - 1.6, 20):
        ms.append(round_on(F, x, 21.2, 1.25, 2.4, 0.05, GLASS))
        ms.append(rect_on(F, x, 20.9, 1.5, 0.25, 0.08, tone(ASHLAR, 1.06)))
    for y in (7.4, 14.2, 20.3):                                           # string courses
        ms.append(rect_on(F, (u0 + u1) / 2, y, u1 - u0, 0.35, 0.1, tone(ASHLAR, 1.07)))
    for x in np.linspace(u0 + 0.6, u1 - 0.6, 25):                         # the candlesticks along the parapet
        ms.append(B(x - 0.22, x + 0.22, Y_CORN, Y_CORN + 0.6, Z_N + 0.05, Z_N + 0.5, tone(ASHLAR, 1.05)))
        ms.append(cyl(0.16, 0.02, 1.2, tone(ASHLAR, 1.05), seg=4).R(0, math.pi / 4, 0).T(x, Y_CORN + 0.6, Z_N + 0.27))
    ms.append(B(u0, u1, Y_CORN, Y_CORN + 0.3, Z_N - 0.2, Z_N + 0.5, tone(ASHLAR, 1.05)))
    # the portal
    xc = 0.3
    ms.append(slab(F, [(xc - 3.4, 0), (xc + 3.4, 0), (xc + 3.4, 8.6), (xc - 3.4, 8.6)], 0.6, tone(ASHLAR, 1.06), off=0.3))
    for u in (-2.7, 2.7):
        ms.append(cyl(0.32, 0.28, 7.4, tone(ASHLAR, 1.1), seg=10).T(xc + u, 0.0, Z_N + 0.9))
    ms.append(round_on(F, xc, 0.0, 3.6, 6.6, 0.62, tone(ASHLAR, 1.1), 10))
    ms.append(round_on(F, xc, 0.0, 3.0, 6.0, 0.66, WOOD, 10))
    ms.append(slab(F, [(xc - 2.6, 9.2), (xc + 2.6, 9.2), (xc + 2.6, 13.6), (xc, 14.6), (xc - 2.6, 13.6)], 0.7, tone(ASHLAR, 1.1), off=0.35))
    ms.append(rect_on(F, xc, 10.0, 2.4, 3.0, 0.72, tone(ASHLAR, 0.8)))                      # the imperial arms
    ms.append(sphere(0.5, tone(ASHLAR, 0.75), 8, 4).T(xc, 11.5, Z_N + 0.9))
    for u in (-2.0, 2.0):
        ms.append(cyl(0.3, 0.25, 1.7, tone(ASHLAR, 1.05), seg=6).T(xc + u, 9.2, Z_N + 0.8))
    return ms


def south_facade(F):
    """Herrera's façade: stone pilasters and red panels, rows of windows, the open gallery under the cornice."""
    ms = []
    u0, u1 = -25.3, 27.0                       # u = −x on the south face
    nb = 12
    for k in range(nb + 1):
        u = u0 + (u1 - u0) * k / nb
        ms.append(rect_on(F, u, -2.0, 0.9, 21.8, 0.08, tone(ASHLAR, 1.05)))
    for k in range(nb):
        u = u0 + (u1 - u0) * (k + 0.5) / nb
        ms.append(rect_on(F, u, -2.0, (u1 - u0) / nb - 0.9, 21.8, 0.05, tone(BRICK)))
        for y in (-7.2, -1.2):
            ms += window(F, u, y, 1.2, 1.8, frame=ASHLAR, eps=0.1)
        for y in (3.6, 9.2):
            ms += window(F, u, y, 1.3, 2.4, frame=ASHLAR, eps=0.1)
        ms += window(F, u, 14.8, 1.3, 2.2, frame=ASHLAR, eps=0.1)
        for j in (-0.25, 0.25):                                        # the gallery: two arches a bay
            uu = u + j * (u1 - u0) / nb
            ms.append(round_on(F, uu, 21.6, 1.45, 3.1, 0.07, DARK))
    for y in (-3.0, 19.8, 21.1):
        ms.append(rect_on(F, (u0 + u1) / 2, y, u1 - u0, 0.45, 0.12, tone(ASHLAR, 1.08)))
    o, t, n = F
    a = o + t * u0 + n * 0.4
    b = o + t * u1 + n * 0.4
    ms.append(balustrade((a[0], a[2]), (b[0], b[2]), 21.4, 1.0, tone(ASHLAR, 1.08)))
    ms.append(rect_on(F, (u0 + u1) / 2, -11.0, u1 - u0, 8.0, 0.04, tone(ASHLAR, 0.94)))       # rusticated base
    return ms


def west_facade(F):
    """The west façade: Toledo masonry between ashlar bands, plain windows in rows."""
    ms = []
    u0, u1 = -41.0, 33.0                       # u = −z on the west face
    ms.append(rect_on(F, (u0 + u1) / 2, -12.0, u1 - u0, 39.6, 0.03, tone(RUBBLE)))
    for y in (-4.0, 6.2, 18.0):
        ms.append(rect_on(F, (u0 + u1) / 2, y, u1 - u0, 0.35, 0.06, tone(ASHLAR, 1.03)))
    xs = np.linspace(u0 + 2.2, u1 - 2.2, 22)
    for i, x in enumerate(xs):
        ms += window(F, x, 7.2, 0.8, 1.2, eps=0.05) if i % 3 else window(F, x, 2.0, 1.1, 2.0, ped=True, eps=0.05)
        ms.append(rect_on(F, x, 12.0, 0.75, 1.0, 0.06, GLASS))
        ms.append(rect_on(F, x, 21.0, 0.75, 1.0, 0.06, GLASS))
        ms.append(rect_on(F, x, 24.8, 0.75, 0.9, 0.06, GLASS))
    for x in xs[1::3]:
        ms += window(F, x, 2.0, 1.1, 2.0, ped=True, eps=0.05)
    return ms


def east_facade(F):
    """The east façade (the medieval front, rebuilt): masonry, few windows, the three round turrets."""
    ms = []
    u0, u1 = -34.0, 40.0                       # u = z on the east face
    ms.append(rect_on(F, (u0 + u1) / 2, -14.0, u1 - u0, 41.6, 0.03, tone(RUBBLE, 0.97)))
    for y in (6.2, 18.0):
        ms.append(rect_on(F, (u0 + u1) / 2, y, u1 - u0, 0.35, 0.06, tone(ASHLAR, 1.03)))
    for x in np.linspace(u0 + 4, u1 - 4, 14):
        if abs(x - 5.8) < 3.5:
            continue
        for y in (8.0, 13.0, 21.5):
            ms.append(rect_on(F, x, y, 0.9, 1.5, 0.06, GLASS))
        ms.append(rect_on(F, x, 1.5, 0.7, 1.2, 0.06, GLASS))
    return ms


def cubos():
    S = []
    for ring, top in ((CUBO_N, 11.4), (CUBO_S, 11.4), (CUBO_M, 1.4)):
        S.append(extrude(ring, FOUND, top - 0.6, tone(RUBBLE, 1.02)))
        S.append(extrude(offset(ring, 0.25), top - 0.6, top, tone(ASHLAR)))
        cx, cz = centroid(ring)
        R = ccw(ring)
        for i in range(len(R)):                                        # merlons
            p, q = R[i], R[(i + 1) % len(R)]
            if math.dist(p, q) < 1.0 or min(p[0], q[0]) > -33.8:
                continue
            m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            S.append(B(m[0] - 0.35, m[0] + 0.35, top, top + 0.9, m[1] - 0.35, m[1] + 0.35, tone(ASHLAR)))
    return S


# ── The towers with their slate spires ──────────────────────────────────────
def tower(ring, north):
    S = []
    cx, cz = centroid(ring)
    S.append(extrude(ring, FOUND, Y_TOWER - 0.7, tone(ASHLAR)))
    if north:                                                          # rubble with ashlar quoins below
        S.append(extrude(offset(ring, 0.03), 2.0, 17.0, tone(RUBBLE), top=False))
    S.append(extrude(offset(ring, 0.5), Y_TOWER - 0.7, Y_TOWER, tone(ASHLAR, 1.05)))
    R = ccw(ring)
    xs = [p[0] for p in R]; zs = [p[1] for p in R]
    x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
    for face, (p, nrm, L) in {"N": ((cx, z1), (0, 1), x1 - x0), "S": ((cx, z0), (0, -1), x1 - x0),
                              "W": ((x1, cz), (1, 0), z1 - z0), "E": ((x0, cz), (-1, 0), z1 - z0)}.items():
        F = fr(p, nrm)
        for y in (4.0, 11.5, 19.5):
            S += window(F, 0, y, 1.3, 2.4, ped=True, eps=0.06)
        for s in (-1.6, 1.6):
            S.append(round_on(F, s, 27.0, 1.1, 2.0, 0.06, GLASS))
        for s in (-L / 2 + 0.5, L / 2 - 0.5):                          # quoins
            S.append(rect_on(F, s, 0.0, 1.0, Y_TOWER - 0.7, 0.05, tone(ASHLAR, 1.06)))
    # the slate spire: pyramid, dormers, lantern, needle
    e = 0.6
    base = [(x0 - e, z0 - e), (x1 + e, z0 - e), (x1 + e, z1 + e), (x0 - e, z1 + e)]
    yb, yt, r = Y_TOWER, 39.6, 1.6
    top = [(cx - r, cz - r), (cx + r, cz - r), (cx + r, cz + r), (cx - r, cz + r)]
    for i in range(4):
        j = (i + 1) % 4
        S.append(quad3((base[i][0], yb, base[i][1]), (top[i][0], yt, top[i][1]), (top[j][0], yt, top[j][1]),
                       (base[j][0], yb, base[j][1]), tone(SLATE)))
    for (dx, dz, a) in ((0, -1, 0.0), (0, 1, math.pi), (1, 0, -math.pi / 2), (-1, 0, math.pi / 2)):
        hx = (x1 - x0) / 2 + e; hz = (z1 - z0) / 2 + e
        px, pz = cx + dx * hx * 0.62, cz + dz * hz * 0.62
        d = merge([box(1.4, 1.6, 1.6, tone(SLATE, 1.1)).T(0, 0, 0),
                   Mesh([(-0.85, 1.6, -0.8), (0.85, 1.6, -0.8), (0, 2.4, -0.8), (-0.85, 1.6, 0.8), (0.85, 1.6, 0.8), (0, 2.4, 0.8)],
                        [(0, 2, 1), (3, 4, 5), (0, 3, 5), (0, 5, 2), (1, 2, 5), (1, 5, 4)], tone(SLATE, 1.05)),
                   quad(0.0, 0.8, -0.82, 0.7, 1.0, DARK)])
        S.append(d.R(0, a, 0).T(px, 35.2, pz))
    S.append(cyl(1.5, 1.5, 0.3, tone(SLATE, 0.9), seg=8).T(cx, yt, cz))
    S.append(cyl(1.2, 1.2, 2.6, tone(SLATE, 1.15), seg=8).T(cx, yt + 0.3, cz))
    for k in range(8):
        a = 2 * math.pi * (k + 0.5) / 8
        S.append(B(cx + 1.21 * math.cos(a) - 0.2, cx + 1.21 * math.cos(a) + 0.2, yt + 0.8, yt + 2.3,
                   cz + 1.21 * math.sin(a) - 0.2, cz + 1.21 * math.sin(a) + 0.2, DARK))
    S.append(cyl(1.5, 0.5, 1.2, tone(SLATE), seg=8).T(cx, yt + 2.9, cz))
    S.append(cyl(0.55, 0.55, 1.0, tone(SLATE, 1.1), seg=8).T(cx, yt + 4.1, cz))
    S.append(sphere(0.6, tone(SLATE, 1.05), 8, 4).T(cx, yt + 5.4, cz))
    S.append(cyl(0.45, 0.04, 4.6, tone(SLATE), seg=8).T(cx, yt + 5.8, cz))
    S.append(sphere(0.22, BRONZE, 6, 4).T(cx, yt + 10.3, cz))
    S.append(B(cx - 0.03, cx + 0.03, yt + 10.5, yt + 11.4, cz - 0.03, cz + 0.03, IRON))
    S.append(B(cx - 0.02, cx + 0.5, yt + 10.9, yt + 11.3, cz - 0.02, cz + 0.02, IRON))
    for (bx, bz) in ((x0 - e, z0 - e), (x1 + e, z0 - e), (x1 + e, z1 + e), (x0 - e, z1 + e)):   # corner finials
        S.append(sphere(0.3, BRONZE, 6, 4).T(bx * 0.97 + cx * 0.03, yb + 0.3, bz * 0.97 + cz * 0.03))
    return S


def quad(cx, cy, cz, w, h, c):
    P = [(cx - w / 2, cy - h / 2, cz), (cx - w / 2, cy + h / 2, cz), (cx + w / 2, cy + h / 2, cz), (cx + w / 2, cy - h / 2, cz)]
    return Mesh(P, [(0, 1, 2), (0, 2, 3)], c)


# ── The courtyard: two floors of arches on columns, paving, Charles V ────────
def courtyard():
    S, D = [], []
    S.append(cap(COURT, 0.2, tone(PAVE)))
    S.append(cap(GAL, 0.25, tone(PAVE, 0.95), holes=[COURT]))
    S.append(cap(GAL, 7.2, tone(ASHLAR, 0.85), holes=[COURT], up=False))
    S.append(cap(GAL, 7.9, tone(PAVE, 0.95), holes=[COURT]))
    S.append(cap(GAL, 14.6, tone(ASHLAR, 0.85), holes=[COURT], up=False))
    C, G = ccw(COURT), ccw(GAL)
    gm = [min(G, key=lambda g: math.dist(g, c)) for c in C]
    for i in range(4):                                              # lean-to roofs over the galleries
        a, b = C[i], C[(i + 1) % 4]
        ga, gb = gm[i], gm[(i + 1) % 4]
        S.append(quad3((a[0], 15.4, a[1]), (b[0], 15.4, b[1]), (gb[0], 16.6, gb[1]), (ga[0], 16.6, ga[1]), TILE))
    c0 = np.mean(np.asarray(COURT), axis=0)
    for i in range(4):
        p, q = np.array(C[i]), np.array(C[(i + 1) % 4])
        L = float(np.linalg.norm(q - p))
        e = (q - p) / L
        nrm = np.array([e[1], -e[0]])
        if np.dot(nrm, c0 - (p + q) / 2) < 0:
            nrm = -nrm
        F = fr((p + q) / 2, nrm)
        o, t, n = F
        sgn = 1.0 if np.dot([t[0], t[2]], e) > 0 else -1.0
        nb = 9 if L > 35 else 7
        pitch = L / nb
        for k in range(nb + 1):
            pt = p + e * k * pitch
            for y0, h in ((0.25, 4.9), (7.9, 4.5)):                  # Corinthian columns, two floors
                S.append(cyl(0.3, 0.26, h, tone(ASHLAR, 1.06), seg=10).T(pt[0], y0, pt[1]))
                S.append(B(pt[0] - 0.42, pt[0] + 0.42, y0 + h, y0 + h + 0.45, pt[1] - 0.42, pt[1] + 0.42, tone(ASHLAR, 1.08)))
        for k in range(nb):
            s = sgn * (-L / 2 + (k + 0.5) * pitch)
            w = pitch - 0.84
            for ys, ytop in ((5.6, 7.9), (12.85, 15.4)):
                arc = [(s + w / 2 * math.cos(a), ys + w / 2 * math.sin(a)) for a in np.linspace(0, math.pi, 9)]
                S.append(slab(F, arc + [(s - w / 2, ytop), (s + w / 2, ytop)], 0.55, tone(ASHLAR, 1.02)))
            mid = p + e * (k + 0.5) * pitch                           # the balustrade of the upper gallery
            S.append(balustrade(mid - e * w / 2, mid + e * w / 2, 7.9, 1.0, tone(ASHLAR, 1.05)))
        D.append(obox(p, q, 7.2, 7.9, 0.6, tone(ASHLAR, 1.08)))
        D.append(obox(p, q, 14.9, 15.45, 0.7, tone(ASHLAR, 1.08)))
        # the walls of the ranges above the gallery roofs, with their windows
        gp, gq = np.array(gm[i]), np.array(gm[(i + 1) % 4])
        Lg = float(np.linalg.norm(gq - gp))
        eg = (gq - gp) / Lg
        Fg = fr((gp + gq) / 2, nrm)
        sg = 1.0 if np.dot([Fg[1][0], Fg[1][2]], eg) > 0 else -1.0
        for k in range(nb + 1):
            s = sg * (-Lg / 2 + (k + 0.5) * Lg / (nb + 1))
            D.append(rect_on(Fg, s, 18.0, 1.1, 1.9, 0.05, GLASS))
            D.append(rect_on(Fg, s, 22.6, 1.1, 1.9, 0.05, GLASS))
    # Charles V on his pedestal, in the middle
    cx, cz = c0
    S.append(B(cx - 1.6, cx + 1.6, 0.2, 0.6, cz - 1.6, cz + 1.6, tone(ASHLAR, 0.95)))
    S.append(B(cx - 1.0, cx + 1.0, 0.6, 3.2, cz - 1.0, cz + 1.0, tone(ASHLAR, 0.9)))
    S.append(B(cx - 1.2, cx + 1.2, 3.2, 3.5, cz - 1.2, cz + 1.2, tone(ASHLAR, 0.95)))
    body_ = lathe([(0.5, 0), (0.45, 0.9), (0.42, 1.5), (0.35, 1.9), (0.0, 2.0)], BRONZE, 8)
    S.append(body_.T(cx, 3.5, cz))
    S.append(sphere(0.22, BRONZE, 6, 4).T(cx, 5.7, cz))
    return S, D


# ── The esplanades, terraces and retaining walls round it ───────────────────
def terraces():
    S = []
    # north esplanade at the level of the courtyard, between and in front of the north towers
    north = [(-46.0, 43.8), (46.0, 43.8), (46.0, 64.0), (-46.0, 64.0)]
    S.append(extrude(north, FOUND, 0.0, tone(RUBBLE, 0.95), top_c=tone(PAVE)))
    for p, q in (((-46.0, 64.0), (46.0, 64.0)), ((46.0, 64.0), (46.0, 52.5)), ((-46.0, 52.5), (-46.0, 64.0))):
        S.append(balustrade(p, q, 0.0, 1.0, tone(ASHLAR, 1.03), 0.45))
    for x in np.linspace(-20, 20, 5):                                    # lamp posts
        S.append(cyl(0.08, 0.06, 3.6, IRON, 6).T(x, 0.0, 58.0))
        S.append(B(x - 0.25, x + 0.25, 3.6, 4.2, 57.75, 58.25, (0.25, 0.25, 0.22)))
    # west walk between the west towers above the Cuesta de Carlos V, with its railing
    west = [(34.3, -33.1), (39.6, -33.1), (39.6, 41.0), (34.3, 41.0)]
    S.append(extrude(west, FOUND, 0.0, tone(RUBBLE), top_c=tone(PAVE)))
    S.append(extrude(offset(west, 0.05), -5.0, -4.4, tone(BRICK, 1.05), top=False))
    S.append(railing((39.4, -33.1), (39.4, 41.0), 0.0, 1.1))
    # east terrace (car park) at the street level, above the gardens, with its retaining wall
    ye = -10.3
    east = [(-60.0, -47.5), (-33.6, -47.5), (-33.6, 40.0), (-60.0, 40.0)]
    S.append(extrude(east, FOUND, ye, tone(RUBBLE, 0.97), top_c=tone(PAVE, 0.92)))
    S.append(railing((-59.8, -47.5), (-59.8, 40.0), ye, 1.1))
    # the pavement along the south façade
    south = [(-39.0, -54.0), (39.0, -54.0), (39.0, -45.4), (-39.0, -45.4)]
    S.append(extrude(south, FOUND, -9.6, tone(RUBBLE, 0.95), top_c=tone(PAVE, 0.9)))
    return S


def main():
    out = ROOT / "models" / "Q1326589"
    out.mkdir(parents=True, exist_ok=True)
    bs, bd = body()
    cs, cd = courtyard()
    tw = []
    for ring in TOWERS:
        tw += tower(ring, north=centroid(ring)[1] > 0)
    parts = [("alcazar", sink(merge(bs + bd + cubos()))), ("towers", sink(merge(tw))),
             ("courtyard", sink(merge(cs + cd))), ("terraces", sink(merge(terraces())))]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.9)
    allv = np.vstack([m.v for _, m in parts])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
