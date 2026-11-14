"""
Catedral de Santa María la Real de la Almudena (Q849711), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Francisco de Cubas (1826–1899), first stone 1883 (crypt, neo-Romanesque; apse, neo-Gothic);
Enrique María Repullés (1845–1922) finished the crypt; Fernando Chueca Goitia (1911–2004) and Carlos Sidro
completed it after the 1950 competition (the classical north façade, the dome, the nave); consecrated 1993.
Protected (see meta.json).

Measurements (see meta.json for the sources):
- OpenStreetMap relation 75658 (outline, two courtyards) and its building:part ways (nave, aisles, transept
  arms, crossing, drum, apse ring, radiating chapels, flying-buttress walls, towers, portico columns), in model
  x/z for heading 4.9° (python tools/osm.py relation 75658 --heading 4.9; parts read from the OSM API). The
  church is a Latin cross running north–south: façade + towers at z ≈ 34.5–41.7, nave to z −4.3, crossing to
  z −32, apse to z −49.7, radiating chapels to z −59.7. Nave axis x ≈ 8.7.
- es.wikipedia: nave 82 m long, 12.5 m wide, aisles and chapels 6 m, transept 68 m, nave 25.8 m high (inside),
  dome «73 m» and towers «60 m» (both look like heights from a lower datum — see below).
- Heights measured on a near-orthographic photo of the north façade (Commons «Almudena 2022 - front view.jpg»;
  scale from the OSM width between the towers' outer faces, 39.1 m, and the 1.4 m columns): portico floor 2.4
  (steps), lower Tuscan order to 12.3, entablature + cornice 15.1, balcony 16.5, upper Ionic order to 24.8,
  main cornice 27.3, attic balustrade 28.5, evangelists to ~32.3, central crest with the Virgin to 39;
  towers: second stage to 31.9, belfry to 39.4, attic 42, slate cupola, spire 48.4, ball, cross 52.8 m.
  Wings along the Plaza de la Armería: cornice 15 m, balustrade, three rows of windows.
- Dome on «Almudena 2022 - overview.jpg» (camera ~10 m up, horizon line, the 20.9 m drum from OSM as scale):
  crossing block 29 (OSM), platform balustrade 30.2, octagonal drum to 38.3, dome to ~47.3, lantern 52.4,
  slate spire 58, cross top ~62.4 m. Octagonal drum (OSM part) with four big arched windows on the cardinal
  faces; ribbed octagonal slate dome with ox-eye dormers; twelve bronze apostles on the platform.
- Nave clerestory: cream wall with big oculi between buttress piers, cornice ~24.3, slate roof ridge ~28.3
  (photos from the NE and NW); aisles + chapels 12 m (OSM), lean-to roofs; transept arms with pediments.
- Ground (EU-DEM 25 m, approximate): the Plaza de la Armería and Calle de Bailén are level with the origin, the
  ground drops ~10–13 m to the west (Cuesta de la Vega, Campo del Moro) and some metres to the south (Calle
  Mayor, the Moorish wall park): foundations go from −3 m on the east to about −16 m at the west end.
Cream Novelda limestone (nave, transept, crossing, drum, apse), grey Zarzalejo granite and white limestone
(façade, towers, wings), dark slate roofs, a red tile roof over the east wing.

Axes (docs/MODEL_SPEC.md): +Y up, the main (north) façade faces +Z, +X points west, origin on the ground at the
centroid of the OSM outline.

    python model_sources/Q849711_catedral_de_la_almudena.py   -> models/Q849711/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, extrude, offset, cap, walls, triangulate, ring_area, write_glb  # noqa: E402

rng = np.random.default_rng(1993)

# ── Colours ──────────────────────────────────────────────────────────────────
GRANITE = np.array((0.66, 0.68, 0.70))     # Zarzalejo granite, blue-grey
WHITE = (0.90, 0.885, 0.85)                # white limestone: cornices, capitals, balustrades
MARBLE = (0.94, 0.93, 0.90)                # statues
CREAM = np.array((0.85, 0.78, 0.60))       # Novelda limestone
CREAM_L = (0.91, 0.86, 0.72)               # its mouldings
SLATE = (0.31, 0.35, 0.40)
DOME = (0.35, 0.39, 0.46)
RIB = (0.55, 0.58, 0.62)
TILE = (0.70, 0.44, 0.35)
ROOF_FLAT = (0.55, 0.55, 0.54)
GLASS = (0.13, 0.14, 0.17)
BRONZE = (0.27, 0.23, 0.18)
BRONZE_ST = (0.22, 0.21, 0.19)             # the dark apostles on the dome
GOLD = (0.86, 0.68, 0.26)
IRON = (0.15, 0.15, 0.16)
PLINTH = (0.62, 0.62, 0.60)

XC = 8.7            # the axis of the church (x of the nave, façade, crossing)
ZF = 41.65          # front of the towers and of the portico


def granite(k=1.0):
    return tuple(np.clip(GRANITE * k * (1 + rng.normal(0, 0.012)), 0, 1))


def cream(k=1.0):
    return tuple(np.clip(CREAM * k * (1 + rng.normal(0, 0.012)), 0, 1))


def found(x, z):
    """Foundation depth below the origin's ground: the ground falls to the west and a little to the south."""
    return -3.0 - 0.17 * max(0.0, x) - 0.12 * max(0.0, -z - 25.0)


def depth(ring):
    return min(found(x, z) for x, z in ring)


# ── OSM outlines (model x/z, heading 4.9°, origin 40.4157623, -3.7144271) ───
OUTLINE = [(75.54, 41.10), (63.52, 41.09), (63.52, 36.65), (28.38, 36.56), (28.32, 41.74), (21.03, 41.66),
           (15.39, 41.67), (8.08, 41.64), (5.02, 41.64), (1.69, 41.62), (-3.73, 41.61), (-10.77, 41.58),
           (-10.74, 36.37), (-46.64, 36.34), (-46.65, 41.12), (-58.36, 41.12), (-58.35, 27.30), (-58.34, 2.09),
           (-58.33, -5.64), (-25.43, -5.69), (-25.52, -11.26), (-25.61, -24.95), (-25.65, -25.56), (-25.73, -31.19),
           (-25.76, -35.14), (-23.92, -35.16), (-23.91, -34.38), (-18.73, -34.40), (-18.81, -36.52), (-18.12, -36.51),
           (-18.08, -44.97), (-15.54, -47.57), (-11.86, -47.71), (-9.80, -45.86), (-9.03, -46.02), (-8.18, -48.77),
           (-5.58, -50.11), (-5.70, -50.93), (-4.86, -52.18), (-4.12, -52.23), (-3.51, -55.11), (-0.64, -57.19),
           (1.40, -56.94), (2.18, -56.85), (2.48, -57.54), (4.01, -58.04), (4.71, -57.78), (6.93, -59.63),
           (10.37, -59.73), (12.51, -57.76), (13.06, -58.12), (14.64, -57.65), (15.06, -56.98), (18.11, -57.27),
           (20.74, -55.19), (21.30, -52.36), (22.23, -52.11), (23.03, -51.07), (22.89, -50.23), (25.44, -48.62),
           (26.40, -46.09), (27.16, -45.83), (29.23, -47.84), (32.76, -47.76), (35.38, -44.95), (35.35, -36.79),
           (36.00, -36.75), (35.93, -32.40), (36.59, -32.41), (36.58, -32.24), (39.21, -32.25), (44.25, -32.28),
           (44.29, -24.97), (44.26, -11.27), (44.24, -4.19), (43.08, -4.19), (43.09, -1.38), (34.09, -1.37),
           (34.09, -4.30), (27.44, -4.29), (27.58, 25.56), (30.11, 25.57), (30.11, 27.04), (63.48, 27.15),
           (63.69, 25.86), (75.68, 25.98)]
COURT_1 = [(-12.46, 27.32), (-25.42, 27.34), (-25.43, -4.21), (-10.40, -4.23), (-10.38, 24.76), (-12.47, 24.76)]
COURT_2 = [(-32.14, 20.23), (-32.19, 1.86), (-50.66, 1.93), (-50.61, 20.29)]

WING_W = [(30.11, 27.04), (28.39, 27.05), (28.40, 34.68), (28.38, 36.56), (63.52, 36.65), (63.52, 41.10),
          (75.54, 41.10), (75.68, 25.99), (63.69, 25.87), (63.48, 27.16)]
WING_E = [(-10.73, 27.33), (-12.46, 27.33), (-25.42, 27.34), (-25.43, -5.69), (-58.33, -5.63), (-58.36, 41.12),
          (-46.65, 41.12), (-46.64, 36.34), (-10.74, 36.37)]
AISLE_E = [(-3.85, -11.41), (-3.74, -4.24), (1.88, -4.28), (1.92, 34.53), (-3.69, 34.52), (-10.73, 34.49),
           (-10.73, 27.33), (-12.46, 27.33), (-12.47, 24.76), (-10.38, 24.76), (-10.40, -4.22)]
AISLE_W = [(21.05, -11.17), (21.19, -4.46), (15.63, -4.42), (15.73, 34.58), (21.12, 34.60), (28.40, 34.68),
           (28.39, 27.05), (30.11, 27.04), (30.12, 25.57), (27.59, 25.56), (27.44, -4.28)]
NAVE = [(15.73, 34.56), (15.63, -4.35), (1.88, -4.35), (1.92, 34.56)]
CROSSING = [(21.19, -4.40), (-3.74, -4.30), (-3.85, -11.41), (-3.96, -32.05), (20.84, -32.35), (21.05, -11.17)]
TRANS_W = [(20.88, -24.82), (21.05, -11.17), (44.26, -11.26), (44.29, -24.96)]
TRANS_E = [(-3.85, -11.41), (-3.93, -25.10), (-25.61, -24.94), (-25.52, -11.25)]
DRUM = [(15.36, -7.93), (19.06, -11.31), (19.00, -25.02), (15.36, -28.78), (1.58, -28.60), (-1.83, -24.93),
        (-1.69, -11.37), (1.93, -7.94)]
APSE = [(0.45, -31.96), (0.28, -37.48), (0.42, -42.38), (1.02, -45.14), (1.89, -46.71), (5.20, -48.84),
        (7.36, -49.69), (10.50, -49.72), (12.39, -49.12), (14.88, -47.13), (15.96, -45.54), (16.74, -43.13),
        (16.98, -32.12)]
# 12 m bodies around the crossing and the apse (sacristies, chapels): flat roofs
SIDE_FLAT = [
    [(-3.85, -11.41), (-10.40, -4.22), (-25.43, -4.21), (-25.43, -5.69), (-25.52, -11.25)],
    [(21.05, -11.17), (27.44, -4.28), (34.09, -4.29), (34.09, -1.36), (43.09, -1.38), (43.09, -4.18), (44.24, -4.18), (44.26, -11.26)],
    [(-3.93, -25.10), (-3.96, -32.05), (-9.97, -32.01), (-18.71, -31.96), (-18.72, -34.39), (-23.91, -34.37), (-23.92, -35.15), (-25.76, -35.14), (-25.73, -31.18), (-25.65, -25.55), (-25.61, -24.94)],
    [(44.25, -32.28), (44.29, -24.96), (20.88, -24.82), (20.84, -32.35), (28.29, -32.38), (35.93, -32.39), (36.59, -32.40), (36.58, -32.24), (39.21, -32.25)],
    [(0.45, -31.96), (0.28, -37.48), (0.42, -42.38), (-4.14, -43.15), (-4.50, -42.17), (-6.75, -42.20), (-9.03, -46.01), (-9.80, -45.86), (-9.97, -32.01), (-3.96, -32.05)],
    [(16.98, -32.12), (16.74, -43.13), (23.25, -42.98), (26.40, -46.09), (27.16, -45.83), (27.45, -33.20), (28.27, -33.23), (28.29, -32.38), (20.84, -32.35)],
]
# radiating and corner chapels: pyramidal roofs
CHAPELS = [
    [(35.93, -32.39), (28.29, -32.38), (28.27, -33.23), (27.45, -33.20), (27.16, -45.83), (29.23, -47.83), (32.77, -47.76), (35.38, -44.95), (35.35, -36.79), (36.00, -36.74)],
    [(-18.72, -34.39), (-18.71, -31.96), (-9.97, -32.01), (-9.80, -45.86), (-11.86, -47.71), (-15.53, -47.56), (-18.08, -44.97), (-18.11, -36.51), (-18.81, -36.51)],
    [(10.50, -49.72), (12.51, -57.76), (10.37, -59.72), (6.93, -59.63), (4.72, -57.78), (7.36, -49.69)],
    [(26.40, -46.09), (23.25, -42.98), (16.74, -43.13), (15.96, -45.54), (20.92, -49.00), (22.89, -50.22), (25.44, -48.61)],
    [(14.88, -47.13), (21.31, -52.36), (20.74, -55.18), (18.11, -57.27), (15.06, -56.97), (12.39, -49.12)],
    [(5.20, -48.84), (2.18, -56.85), (1.41, -56.94), (-0.63, -57.19), (-3.51, -55.10), (-4.12, -52.23), (1.89, -46.71)],
    [(-5.57, -50.11), (1.02, -45.14), (0.42, -42.38), (-4.14, -43.15), (-4.50, -42.17), (-6.75, -42.20), (-9.03, -46.01), (-8.18, -48.76)],
]
# flying-buttress walls between the radiating chapels: 21 m at the apse, 10 m at the outer end
BUTTRESS_WALLS = [
    [(15.96, -45.54), (20.92, -49.00), (22.89, -50.22), (23.03, -51.07), (22.23, -52.10), (21.31, -52.36), (14.88, -47.13)],
    [(1.89, -46.71), (-4.12, -52.23), (-4.86, -52.18), (-5.69, -50.92), (-5.57, -50.11), (1.02, -45.14)],
    [(7.36, -49.69), (5.20, -48.84), (2.18, -56.85), (2.48, -57.53), (4.01, -58.03), (4.72, -57.78)],
    [(12.39, -49.12), (10.50, -49.72), (12.51, -57.76), (13.06, -58.12), (14.64, -57.65), (15.06, -56.97)],
]
TOWER_HX, TOWER_HZ, TOWER_DX, TOWER_ZC = 3.61, 3.55, 15.99, 38.10     # half sizes, offset from the axis, centre z
COLS = (2.3, 6.4, 10.4)                                                  # portico columns, ± from the axis
WALL_Z = 38.2                                                            # the wall behind the portico


# ── Small helpers ────────────────────────────────────────────────────────────
def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sphere(r, c, seg=10, rings=6):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def obox(p, q, y0, y1, w, c):
    """A box along the segment p→q (plan), w thick, from y0 to y1."""
    (x0, z0), (x1, z1) = p, q
    L = math.hypot(x1 - x0, z1 - z0)
    a = math.atan2(-(z1 - z0), x1 - x0)
    return box(L, y1 - y0, w, c).R(0, a, 0).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def ccw(ring):
    return ring if ring_area(ring) > 0 else ring[::-1]


def centroid(ring):
    P = np.asarray(ring, float)
    x, z = P[:, 0], P[:, 1]
    k = x * np.roll(z, -1) - np.roll(x, -1) * z
    a = k.sum() / 2
    return float(((x + np.roll(x, -1)) * k).sum() / (6 * a)), float(((z + np.roll(z, -1)) * k).sum() / (6 * a))


def inside(x, z, P):
    ins = False
    for i in range(len(P)):
        (xi, zi), (xj, zj) = P[i], P[i - 1]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            ins = not ins
    return ins


def plane_poly(pts_uv, origin, t, n, c):
    """A flat polygon given in wall coordinates (u along t, v = height) facing n."""
    pts, tris = triangulate(pts_uv)
    o = np.asarray(origin, float)
    V = np.array([o + u * np.asarray(t) + np.array([0, v, 0]) for u, v in pts])
    m = Mesh(V, tris, c)
    up = np.array([0.0, 1.0, 0.0])
    if np.dot(np.cross(t, up), n) < 0:              # keep the front face towards n
        m.f = m.f[:, ::-1]
    return m


def wall_frame(p0, p1, ccw_ring=True):
    """(origin, unit tangent, outward normal, length) of a ring edge in 3D."""
    p0, p1 = np.array([p0[0], 0, p0[1]], float), np.array([p1[0], 0, p1[1]], float)
    t = p1 - p0
    L = float(np.linalg.norm(t)); t /= L
    n = np.array([t[2], 0, -t[0]])                  # outward for a counter-clockwise ring in (x, z)
    if not ccw_ring:
        n = -n
    return p0, t, n, L


def rect_on(fr, s, y0, w, h, eps, c):
    o, t, n, _ = fr
    return plane_poly([(s - w / 2, y0), (s + w / 2, y0), (s + w / 2, y0 + h), (s - w / 2, y0 + h)], o + n * eps, t, n, c)


def arch_on(fr, s, y0, w, h, eps, c, pointed=False, seg=8):
    """An arched (or pointed) opening: dark glass on the wall."""
    o, t, n, _ = fr
    r = w / 2
    ys = y0 + h - (r * 1.2 if pointed else r)
    pts = [(s - r, y0), (s + r, y0), (s + r, ys)]
    if pointed:
        pts += [(s, y0 + h)]
    else:
        pts += [(s + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)[1:-1]]
    pts += [(s - r, ys)]
    return plane_poly(pts, o + n * eps, t, n, c)


def disc_on(fr, s, y, r, eps, c, seg=14):
    o, t, n, _ = fr
    pts = [(s + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, seg, endpoint=False)]
    return plane_poly(pts, o + n * eps, t, n, c)


def free_edges(ring, others, probe=0.7):
    """Edges of a ring as frames, with a test telling whether a point on the edge is free (not against another body)."""
    R = ccw(ring)
    out = []
    for i in range(len(R)):
        fr = wall_frame(R[i], R[(i + 1) % len(R)])
        o, t, n, L = fr

        def free(s, o=o, t=t, n=n):
            p = o + t * s + n * probe
            return not any(inside(p[0], p[2], P) for P in others)
        out.append((fr, free))
    return out


def window_rows(ring, others, rows, spacing, margin=1.6, min_len=4.0, ring_is_hole=False):
    """Windows along every free wall of a ring. rows = [(y0, h, w, kind)], kind 'rect' | 'arch' | 'pointed'."""
    ms = []
    R = ccw(ring)
    for i in range(len(R)):
        fr = wall_frame(R[i], R[(i + 1) % len(R)], ccw_ring=not ring_is_hole)   # a courtyard faces inwards
        o, t, n, L = fr
        if L < min_len:
            continue
        k = max(1, int((L - 2 * margin) // spacing) + 1)
        span = (k - 1) * spacing
        for j in range(k):
            s = L / 2 - span / 2 + j * spacing
            p = fr[0] + fr[1] * s + fr[2] * 0.7
            if any(inside(p[0], p[2], P) for P in others):
                continue
            for y0, h, w, kind in rows:
                if kind == "rect":
                    ms.append(rect_on(fr, s, y0 - 0.1, w + 0.36, h + 0.2, 0.03, WHITE))
                    ms.append(rect_on(fr, s, y0, w, h, 0.06, GLASS))
                elif kind == "small":
                    ms.append(rect_on(fr, s, y0, w, h, 0.05, GLASS))
                else:
                    ms.append(arch_on(fr, s, y0, w, h, 0.05, GLASS, pointed=(kind == "pointed")))
    return ms


def roof_ridge(ring, y0, a, b, y1, c_roof, c_wall, eave=0.0):
    """A roof from a ring (at y0, grown by eave) up to the ridge segment a→b (at y1): gables, hips, cones, lean-tos.
    Faces that come out vertical (gable ends) take the wall colour."""
    R = ccw(offset(ring, eave) if eave else ring)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    dd = float(d @ d)

    def proj(p):
        tt = 0.0 if dd == 0 else min(1.0, max(0.0, float((np.asarray(p) - a) @ d) / dd))
        return a + tt * d
    V, F, C = [], [], []
    for i in range(len(R)):
        p, q = np.asarray(R[i]), np.asarray(R[(i + 1) % len(R)])
        rp, rq = proj(p), proj(q)
        quad = [(p[0], y0, p[1]), (q[0], y0, q[1]), (rq[0], y1, rq[1]), (rp[0], y1, rp[1])]
        if np.allclose(rp, rq):
            quad = quad[:3]
        P = np.array(quad)
        nrm = np.cross(P[1] - P[0], P[2] - P[0])
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        col = c_wall if abs(nrm[1] / ln) < 0.2 else c_roof
        o = len(V)
        V += quad
        C += [col] * len(quad)
        F += [(o, o + 2, o + 1)] + ([(o, o + 3, o + 2)] if len(quad) == 4 else [])
    return Mesh(V, F, np.array(C))


def figure(x, y, z, h, c, seated=False):
    """A stylised standing statue: robe, shoulders, head."""
    prof = [(0.17 * h, 0), (0.15 * h, 0.3 * h), (0.12 * h, 0.55 * h), (0.13 * h, 0.72 * h), (0.06 * h, 0.8 * h), (0, 0.81 * h)]
    return [lathe(prof, c, 10).T(x, y, z), sphere(0.075 * h, c, 8, 5).T(x, y + 0.88 * h, z)]


def pedestal_statue(x, y, z, h, c_ped, c_st, pw=0.9, ph=1.0):
    return [B(x - pw / 2, x + pw / 2, y, y + ph, z - pw / 2, z + pw / 2, c_ped)] + figure(x, y + ph, z, h, c_st)


def balusters(p, q, y0, h, c, spacing=0.42, w=0.32):
    """A balustrade along p→q: plinth, rail and balusters."""
    out = [obox(p, q, y0, y0 + 0.22, w + 0.06, c), obox(p, q, y0 + h - 0.2, y0 + h, w + 0.1, c)]
    L = math.dist(p, q)
    n = max(1, int(L / spacing))
    for i in range(n):
        f = (i + 0.5) / n
        x, z = p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f
        out.append(cyl(0.1, 0.07, h - 0.42, c, 6).T(x, y0 + 0.22, z))
    return out


def parapet(ring, y0, y1, c, out=0.15, inn=0.2):
    return extrude(offset(ring, out), y0, y1, c, holes=[offset(ring, -inn)])


def plinth(ring, k, top=0.9):
    """The base course and the foundations (to the depth the slope needs); offsets differ per piece (no z-fighting)."""
    k = k % 11
    return extrude(offset(ring, 0.12 + 0.013 * k), depth(ring), top + 0.029 * k, PLINTH, bottom=True)


def cornice(ring, y0, y1, d, c):
    """A moulded band standing out d round a ring (a ring-shaped prism, its top 2 cm over the wall top)."""
    return extrude(offset(ring, d), y0, y1 + 0.02, c, holes=[offset(ring, -0.06)], bottom=True)


# ── Wings along the Plaza de la Armería and Calle de Bailén ────────────────
WING_TOP = 15.0


def wings(bodies):
    ms = []
    for k, (R, holes) in enumerate(((WING_W, []), (WING_E, [COURT_2]))):
        others = [b for b in bodies if b is not R]
        ms.append(plinth(R, 1 + k, 1.2))
        ms.append(extrude(R, 0.0, WING_TOP, granite(1.0), holes=holes, top=True, top_c=ROOF_FLAT))
        for y0, y1, d in ((6.0, 6.25, 0.08), (8.9, 9.15, 0.08)):                        # string courses
            ms.append(extrude(offset(R, d + 0.01 * k), y0, y1, WHITE, holes=[offset(h, -d) for h in holes], bottom=True))
        ms.append(extrude(offset(R, 0.4), 14.35, WING_TOP + 0.02, WHITE, holes=[offset(R, -0.06)], bottom=True))
        for h in holes:
            ms.append(extrude(offset(h, 0.06), 14.35, WING_TOP + 0.02, WHITE, holes=[offset(h, -0.4)], bottom=True))
        ms.append(parapet(R, WING_TOP, WING_TOP + 1.25, WHITE, out=0.05, inn=0.35))
        for h in holes:
            ms.append(extrude(offset(h, 0.35), WING_TOP, WING_TOP + 1.0, WHITE, holes=[offset(h, -0.05)]))
        rows = [(2.0, 3.0, 1.45, "rect"), (6.55, 0.75, 1.1, "small"), (9.6, 2.5, 1.45, "rect")]
        ms += window_rows(R, others, rows, 4.8)
        for h in holes:
            ms += window_rows(h, [], [(2.4, 2.6, 1.3, "rect"), (9.6, 2.4, 1.3, "rect")], 4.2, ring_is_hole=True)
    # the red tile roof round the east courtyard
    outer = [(-57.6, -4.95), (-26.1, -4.95), (-26.1, 26.65), (-57.6, 26.65)]
    inner = [(-51.25, 1.25), (-31.5, 1.25), (-31.5, 20.9), (-51.25, 20.9)]
    mid = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(outer, inner)]
    V, F = [], []
    for ringA, ringB in ((outer, mid), (inner, mid)):
        for i in range(4):
            j = (i + 1) % 4
            o = len(V)
            V += [(ringA[i][0], WING_TOP + 0.03, ringA[i][1]), (ringA[j][0], WING_TOP + 0.03, ringA[j][1]),
                  (ringB[j][0], WING_TOP + 1.7, ringB[j][1]), (ringB[i][0], WING_TOP + 1.7, ringB[i][1])]
            F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    ms.append(Mesh(V, F, TILE))
    # the corner pavilions on the plaza: a slightly higher attic over the balustrade line
    for x0, x1 in ((63.52, 75.54), (-58.36, -46.65)):
        ms.append(B(x0 + 0.5, x1 - 0.5, WING_TOP + 1.25, WING_TOP + 1.9, 36.9, 40.6, WHITE))
    # Bailén front of the east wing: a relief panel and a door
    ms.append(B(-58.75, -58.36, 8.6, 12.4, 10.0, 17.0, MARBLE))
    ms.append(B(-58.6, -58.36, 0.0, 5.2, 12.2, 14.8, BRONZE))
    return ms


# ── North façade and towers ──────────────────────────────────────────────────
def tower(xc):
    zc, hx, hz = TOWER_ZC, TOWER_HX, TOWER_HZ
    ms = []
    R = [(xc - hx, zc - hz), (xc + hx, zc - hz), (xc + hx, zc + hz), (xc - hx, zc + hz)]
    ms.append(plinth(R, 5, 2.4))
    ms.append(B(xc - hx, xc + hx, 0, 26.6, zc - hz, zc + hz, granite()))
    for sx in (-1, 1):                                        # corner pilasters (L-shaped, standing out 0.18)
        for sz in (-1, 1):
            x0, x1 = sorted((xc + sx * (hx - 0.95), xc + sx * (hx + 0.18)))
            z0, z1 = sorted((zc + sz * (hz - 0.95), zc + sz * (hz + 0.18)))
            ms.append(B(x0, x1, 2.4, 14.5, z0, z1, granite(1.05)))
            ms.append(B(x0, x1, 15.1, 26.6, z0, z1, granite(1.05)))
            ms.append(B(x0 - 0.04, x1 + 0.04, 13.9, 14.5, z0 - 0.04, z1 + 0.04, WHITE))
            ms.append(B(x0 - 0.04, x1 + 0.04, 26.0, 26.6, z0 - 0.04, z1 + 0.04, WHITE))
    ms.append(B(xc - hx - 0.3, xc + hx + 0.3, 14.45, 15.15, zc - hz - 0.3, zc + hz + 0.3, WHITE))
    ms.append(B(xc - hx - 0.42, xc + hx + 0.42, 26.6, 27.36, zc - hz - 0.42, zc + hz + 0.42, WHITE))
    zf = zc + hz
    # front: royal arms relief, an angel in a niche, a window, an oculus
    ms.append(B(xc - 2.2, xc + 2.2, 9.2, 12.6, zf, zf + 0.14, MARBLE))
    ms.append(cyl(0.8, 0.8, 0.2, MARBLE, 12).R(math.pi / 2, 0, 0).T(xc, 10.9, zf + 0.33))
    ms.append(B(xc - 1.3, xc + 1.3, 2.6, 3.0, zf, zf + 0.3, MARBLE))
    ms += figure(xc, 3.0, zf + 0.35, 4.0, MARBLE)
    ms.append(B(xc - 1.05, xc + 1.05, 16.8, 20.7, zf, zf + 0.05, WHITE))
    ms.append(B(xc - 0.85, xc + 0.85, 17.0, 20.5, zf, zf + 0.09, GLASS))
    ms.append(cyl(0.95, 0.95, 0.12, WHITE, 14).R(math.pi / 2, 0, 0).T(xc, 23.6, zf + 0.12))
    ms.append(cyl(0.65, 0.65, 0.1, GLASS, 14).R(math.pi / 2, 0, 0).T(xc, 23.6, zf + 0.17))
    # outer side: same window and oculus above the wing
    sx = 1 if xc > XC else -1
    xf = xc + sx * hx
    for y0, y1, c, d in ((16.8, 20.7, WHITE, 0.05), (17.0, 20.5, GLASS, 0.09)):
        x0, x1 = sorted((xf, xf + sx * d))
        ms.append(B(x0, x1, y0, y1, zc - 1.05 + (0.2 if c == GLASS else 0), zc + 1.05 - (0.2 if c == GLASS else 0), c))
    # second stage
    ms.append(B(xc - 3.3, xc + 3.3, 27.3, 31.5, zc - 3.25, zc + 3.25, granite(1.02)))
    ms.append(B(xc - 3.58, xc + 3.58, 31.5, 31.9, zc - 3.53, zc + 3.53, WHITE))
    for s in (-1, 1):
        ms.append(B(xc - 0.3, xc + 0.3, 28.6, 29.8, zc + s * 3.25 - (0.05 if s < 0 else 0), zc + s * 3.25 + (0.05 if s > 0 else 0), GLASS))
    # belfry: core, corner pilasters, an arched opening on each face, pediments
    ms.append(B(xc - 2.7, xc + 2.7, 31.9, 38.6, zc - 2.65, zc + 2.65, granite(1.04)))
    for sx_ in (-1, 1):
        for sz in (-1, 1):
            ms.append(B(*sorted((xc + sx_ * 2.0, xc + sx_ * 3.0)), 31.9, 38.6, *sorted((zc + sz * 1.95, zc + sz * 2.95)), granite(1.08)))
    Rb = [(xc - 2.7, zc - 2.65), (xc + 2.7, zc - 2.65), (xc + 2.7, zc + 2.65), (xc - 2.7, zc + 2.65)]
    for i in range(4):
        fr = wall_frame(Rb[i], Rb[(i + 1) % 4])
        L = fr[3]
        ms.append(arch_on(fr, L / 2, 33.0, 2.1, 3.9, 0.04, GLASS))
        ms.append(rect_on(fr, L / 2, 33.0, 2.3, 0.25, 0.08, WHITE))                     # sill / little balcony
        o, t, n, _ = fr                                                                  # pediment over the opening
        ms.append(plane_poly([(L / 2 - 2.0, 38.9), (L / 2 + 2.0, 38.9), (L / 2, 39.9)], o + n * 0.36, t, n, WHITE))
    ms.append(B(xc - 3.15, xc + 3.15, 38.6, 38.95, zc - 3.1, zc + 3.1, WHITE))
    for i in range(4):                                                                   # pediment roofs (slate)
        fr = wall_frame(Rb[i], Rb[(i + 1) % 4])
        o, t, n, L = fr
        a0, a1 = o + t * (L / 2 - 2.0) + n * 0.36 + [0, 38.9, 0], o + t * (L / 2 + 2.0) + n * 0.36 + [0, 38.9, 0]
        ap = o + t * (L / 2) + n * 0.36 + [0, 39.9, 0]
        b0, b1, bp = a0 - n * 0.7, a1 - n * 0.7, ap - n * 0.7
        ms.append(Mesh([a0, ap, bp, b0, ap, a1, b1, bp], [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)], WHITE))
    # attic with balustrade
    ms.append(B(xc - 2.4, xc + 2.4, 38.95, 41.2, zc - 2.35, zc + 2.35, granite(1.04)))
    ms.append(B(xc - 2.58, xc + 2.58, 41.2, 41.6, zc - 2.53, zc + 2.53, WHITE))
    for sx_ in (-1, 1):
        for sz in (-1, 1):
            ms.append(sphere(0.22, WHITE, 8, 5).T(xc + sx_ * 2.4, 41.85, zc + sz * 2.35))
    # slate cupola (a concave square bell), dormers, white band, slate lantern, spire, ball, cross
    k = math.sqrt(2)
    prof = [(2.32 * k, 41.6), (1.9 * k, 42.15), (1.5 * k, 42.85), (1.25 * k, 43.45), (1.12 * k, 43.85), (0, 43.86)]
    ms.append(lathe(prof, SLATE, 4).R(0, math.pi / 4, 0).T(xc, 0, zc))
    for i in range(4):
        a = i * math.pi / 2
        dx, dz = math.sin(a), math.cos(a)
        ms.append(box(0.7, 0.75, 0.5, WHITE).R(0, a, 0).T(xc + dx * 1.75, 42.25, zc + dz * 1.75))
        ms.append(box(0.4, 0.45, 0.06, GLASS).R(0, a, 0).T(xc + dx * 2.02, 42.35, zc + dz * 2.02))
    ms.append(B(xc - 1.25, xc + 1.25, 43.85, 44.2, zc - 1.25, zc + 1.25, WHITE))
    r8 = 1.0 / math.cos(math.pi / 8)
    ms.append(lathe([(1.05 * r8, 44.2), (1.05 * r8, 45.0), (1.3 * r8, 45.15), (0.95 * r8, 46.0), (0.6 * r8, 47.0),
                     (0.25 * r8, 48.0), (0.05, 48.45), (0, 48.5)], SLATE, 8).R(0, math.pi / 8, 0).T(xc, 0, zc))
    ms.append(sphere(0.58, GOLD, 12, 7).T(xc, 49.1, zc))
    ms.append(B(xc - 0.07, xc + 0.07, 49.6, 52.8, zc - 0.07, zc + 0.07, IRON))
    ms.append(B(xc - 0.65, xc + 0.65, 51.5, 51.64, zc - 0.06, zc + 0.06, IRON))
    ms.append(B(xc - 0.06, xc + 0.06, 51.5, 51.64, zc - 0.65, zc + 0.65, IRON))
    return ms


def tuscan(x, z, y0, y1, r):
    return [B(x - r - 0.2, x + r + 0.2, y0, y0 + 0.3, z - r - 0.2, z + r + 0.2, WHITE),
            cyl(r * 1.1, r * 1.05, 0.3, WHITE, 16).T(x, y0 + 0.3, z),
            cyl(r, r * 0.88, y1 - y0 - 1.2, granite(1.03), 16, caps=False).T(x, y0 + 0.6, z),
            cyl(r * 0.92, r * 1.12, 0.35, WHITE, 16).T(x, y1 - 0.6, z),
            B(x - r - 0.2, x + r + 0.2, y1 - 0.25, y1, z - r - 0.2, z + r + 0.2, WHITE)]


def ionic(x, z, y0, y1, r):
    out = tuscan(x, z, y0, y1, r)
    for s in (-1, 1):
        out.append(cyl(0.2, 0.2, 2 * r + 0.3, WHITE, 8).R(math.pi / 2, 0, 0).T(x + s * (r + 0.05), y1 - 0.45, z - r - 0.15))
    return out


def facade():
    ms = []
    x0, x1 = XC - (TOWER_DX - TOWER_HX), XC + (TOWER_DX - TOWER_HX)       # between the towers
    ms.append(B(x0, x1, -3.0, 26.9, 34.55, WALL_Z, granite()))           # the wall and the block behind it
    ms.append(B(x0, x1, -3.0, 2.4, WALL_Z, ZF, PLINTH))                  # portico floor
    for i in range(12):                                                  # the steps down to the plaza
        yt = 2.4 - 0.2 * i
        ms.append(B(XC - 20.0, XC + 20.0, yt - 0.2 if i < 11 else -1.0, yt, ZF, ZF + 0.39 * (i + 1), PLINTH))
    for s in COLS:                                                       # two orders of columns
        for x in (XC - s, XC + s):
            ms += tuscan(x, 40.9, 2.4, 12.3, 0.7)
            ms += ionic(x, 40.9, 16.5, 24.8, 0.58)
    for sx in (-1, 1):                                                   # half-columns against the towers
        x = XC + sx * (x1 - XC - 0.45)
        ms += tuscan(x, 40.9, 2.4, 12.3, 0.45)
        ms += ionic(x, 40.9, 16.5, 24.8, 0.4)
    ms.append(B(x0, x1, 12.3, 14.5, WALL_Z, ZF, granite(0.98)))          # lower entablature
    ms.append(B(x0, x1, 14.5, 15.1, WALL_Z, ZF + 0.27, WHITE))
    ms += balusters((x0 + 0.1, ZF - 0.05), (x1 - 0.1, ZF - 0.05), 15.1, 1.4, WHITE)
    for s in COLS:
        for x in (XC - s, XC + s):
            ms.append(B(x - 0.55, x + 0.55, 15.1, 16.6, ZF - 0.55, ZF + 0.05, WHITE))
    ms.append(B(x0, x1, 24.8, 26.6, WALL_Z, ZF, granite(0.98)))          # upper entablature
    ms.append(B(x0, x1, 26.62, 27.3, WALL_Z - 3.7, ZF + 0.38, WHITE))
    ms += balusters((x0 + 0.1, ZF + 0.1), (x1 - 0.1, ZF + 0.1), 27.3, 1.2, WHITE)
    for s in COLS:                                                       # pedestals; the evangelists on the outer four
        for x in (XC - s, XC + s):
            ms.append(B(x - 0.55, x + 0.55, 27.3, 28.6, ZF - 0.45, ZF + 0.42, WHITE))
            if s > 3:
                ms += figure(x, 28.6, ZF, 3.7, MARBLE)
    # the wall behind the portico: bronze doors and reliefs below, saints in niches and reliefs above
    zw = WALL_Z
    ms.append(B(XC - 1.65, XC + 1.65, 2.4, 8.4, zw, zw + 0.06, WHITE))
    ms.append(B(XC - 1.4, XC + 1.4, 2.4, 8.0, zw, zw + 0.1, BRONZE))
    for sx in (-1, 1):
        x = XC + sx * 8.4
        ms.append(B(x - 1.2, x + 1.2, 2.4, 7.2, zw, zw + 0.06, WHITE))
        ms.append(B(x - 1.0, x + 1.0, 2.4, 6.9, zw, zw + 0.1, BRONZE))
    for s in (0.0, -4.3, 4.3, -8.4, 8.4):
        ms.append(B(XC + s - 1.2, XC + s + 1.2, 9.2, 11.2, zw, zw + 0.08, BRONZE))
        if s:
            ms.append(B(XC + s - 0.6, XC + s + 0.6, 17.0, 19.8, zw, zw + 0.05, GLASS))
            ms += figure(XC + s, 17.0, zw + 0.35, 2.4, MARBLE)
        ms.append(B(XC + s - 1.25, XC + s + 1.25, 20.7, 22.7, zw, zw + 0.1, MARBLE))
    ms.append(B(XC - 0.85, XC + 0.85, 16.6, 19.8, zw, zw + 0.06, GLASS))
    ms += crest()
    for xc in (XC - TOWER_DX, XC + TOWER_DX):
        ms += tower(xc)
    return ms


def slab(profile, z0, z1, c):
    """A profile drawn in (x, y) and extruded along z from z0 to z1."""
    m = extrude(profile, z0, z1, c, bottom=True)
    v = m.v.copy()
    m.v = np.stack([v[:, 0], v[:, 2], v[:, 1]], 1)
    m.f = m.f[:, ::-1]
    return m


def crest():
    """The central frontispiece over the attic: concave sides, an arched niche with the Virgin, a crown and a ball."""
    y0 = 28.5
    half = [(5.5, y0)] + [(3.6 + 1.9 * (1 - t) ** 2, y0 + 6.0 * t) for t in np.linspace(0.1, 1, 8)] + [(3.6, 34.9)]
    prof = half + [(-x, y) for x, y in half[::-1]]
    ms = [slab(prof, 39.95, 41.1, WHITE)]
    ms.append(B(XC - 3.9, XC + 3.9, 34.9, 35.6, 39.75, 41.35, WHITE))     # cornice
    up = [(2.5, 35.6), (2.5, 37.0), (1.3, 37.4), (1.5, 38.1), (0.0, 38.4)]
    ms.append(slab(up + [(-x, y) for x, y in up[::-1]], 40.1, 40.95, WHITE))
    for m in ms[:1] + ms[2:]:
        m.T(XC, 0, 0)
    ms.append(sphere(0.38, WHITE, 10, 6).T(XC, 38.8, 40.5))
    for sx in (-1, 1):
        ms.append(sphere(0.3, WHITE, 8, 5).T(XC + sx * 2.3, 37.35, 40.5))
    # the niche (dark) and the Virgin
    pts = [(-1.3, 28.7), (1.3, 28.7), (1.3, 32.6)] + [(1.3 * math.cos(a), 32.6 + 1.3 * math.sin(a)) for a in np.linspace(0, math.pi, 9)[1:-1]] + [(-1.3, 32.6)]
    ms.append(plane_poly(pts, np.array([XC, 0, 41.14]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), GLASS))
    ms.append(B(XC - 0.6, XC + 0.6, 28.5, 29.0, 40.9, 41.7, WHITE))
    ms += figure(XC, 29.0, 41.3, 2.6, MARBLE)
    ms.append(tube([(XC + 0.45 * math.cos(a), 31.4 + 0.45 * math.sin(a), 41.3) for a in np.linspace(0, 2 * math.pi, 13)], 0.05, GOLD, 5))
    return ms


# ── Nave, aisles, transept, crossing ─────────────────────────────────────────
NAVE_TOP, RIDGE = 24.3, 28.3


def nave(bodies):
    ms = []
    # aisles + side chapels: lean-to slate roofs from the outer wall (11 m) up to the clerestory (13.4 m)
    for k, (R, xin) in enumerate(((AISLE_E, 1.9), (AISLE_W, 15.7))):
        others = [b for b in bodies if b is not R]
        ms.append(plinth(R, 7 + k))
        ms.append(extrude(R, 0.0, 11.0, cream(), top=False))
        ms.append(cornice(R, 10.55, 11.0, 0.25 + 0.02 * k, CREAM_L))
        ms.append(roof_ridge(R, 11.0, (xin, -4.3), (xin, 34.5), 13.4, SLATE, cream(), eave=0.25 + 0.02 * k))
        ms += window_rows(R, others, [(3.4, 5.4, 1.7, "arch")], 6.47)
    # the nave: clerestory with big oculi between buttress piers, cornice, slate roof
    ms.append(extrude(NAVE, 10.5, NAVE_TOP, cream(), top=False))
    ms.append(cornice(NAVE, 23.75, NAVE_TOP, 0.32, CREAM_L))
    ms.append(roof_ridge(NAVE, NAVE_TOP, (XC, -4.35), (XC, 34.56), RIDGE, SLATE, cream(), eave=0.32))
    z0, z1 = -4.35, 34.56
    n = 6
    bay = (z1 - z0) / n
    for x, s in ((1.88, -1), (15.68, 1)):
        for i in range(n):
            zc = z0 + (i + 0.5) * bay
            xf = x + s * 0.02
            ms.append(cyl(1.9, 1.9, 0.12, CREAM_L, 16).R(0, 0, math.pi / 2).T(xf + s * 0.12 if s > 0 else xf, 18.6, zc))
            ms.append(cyl(1.45, 1.45, 0.08, GLASS, 16).R(0, 0, math.pi / 2).T(xf + s * 0.2 if s > 0 else xf - 0.08, 18.6, zc))
            if 0 < i:
                zb = z0 + i * bay
                ms.append(B(*sorted((x, x + s * 0.9)), 13.0, NAVE_TOP + 0.6, zb - 0.55, zb + 0.55, cream(1.03)))
                ms += figure(x + s * 0.45, NAVE_TOP + 0.6, zb, 2.0, MARBLE)
    # dormers on the nave roof
    for i in range(1, n):
        zc = z0 + i * bay
        for s in (-1, 1):
            xd = XC + s * 4.0
            ms.append(B(xd - 0.6, xd + 0.6, 25.6, 26.6, zc - 0.5, zc + 0.5, WHITE))
    return ms


def transept():
    ms = []
    for k, (R, xe, xi) in enumerate(((TRANS_E, -25.57, -3.9), (TRANS_W, 44.28, 20.95))):
        ms.append(plinth(R, 11 + k))
        ms.append(extrude(R, 0.0, NAVE_TOP, cream(), top=False))
        ms.append(cornice(R, 23.75, NAVE_TOP, 0.34 + 0.02 * k, CREAM_L))
        zc = -18.1
        ms.append(roof_ridge(R, NAVE_TOP, (xe, zc), (xi, zc), RIDGE, SLATE, cream(), eave=0.34 + 0.02 * k))
        s = -1 if k == 0 else 1
        xf = xe + s * 0.36
        # the end front: pilasters, a raking cornice, a big arched window, an oculus, a portal
        for zp in (-24.4, -20.6, -15.6, -11.8):
            ms.append(B(*sorted((xe, xe + s * 0.35)), 0.9, 23.75, zp - 0.55, zp + 0.55, cream(1.04)))
        for zz in (-25.3, -10.9):
            ms.append(tube([(xf, NAVE_TOP + 0.05, zz), (xf, RIDGE + 0.2, zc)], 0.24, CREAM_L, 6))
        fr = (np.array([xe + s * 0.01, 0, -11.25 if s < 0 else -24.96]), np.array([0, 0, -1.0 if s < 0 else 1.0]),
              np.array([float(s), 0, 0]), 13.7)
        L = 13.7
        ms.append(arch_on(fr, L / 2, 12.5, 4.6, 8.5, 0.05, GLASS))
        ms.append(disc_on(fr, L / 2, 25.6, 0.9, 0.05, GLASS))
        ms.append(rect_on(fr, L / 2, 0.0, 4.0, 7.2, 0.04, WHITE))
        ms.append(rect_on(fr, L / 2, 0.0, 3.2, 6.4, 0.08, BRONZE))
        ms.append(rect_on(fr, L / 2, 7.4, 5.4, 0.6, 0.3, CREAM_L))
        for zz in (-22.5, -13.7):                                               # oculi in the side bays
            ms.append(disc_on(fr, L / 2 + (zz + 18.1) * (-1 if s < 0 else 1), 16.5, 1.2, 0.05, GLASS))
        # the long walls above the side bodies: oculi
        for zw, sg in ((-11.26, 1), (-24.95, -1)):
            for xx in np.linspace(min(xe, xi) + 3.5, max(xe, xi) - 3.5, 3):
                ms.append(cyl(1.3, 1.3, 0.1, GLASS, 14).R(math.pi / 2, 0, 0).T(xx, 18.4, zw + (0.02 if sg > 0 else -0.12)))
    # 12 m bodies in the corners: flat roofs, parapets, windows
    bodies = [AISLE_E, AISLE_W, NAVE, CROSSING, TRANS_E, TRANS_W, APSE] + SIDE_FLAT + CHAPELS
    for k, R in enumerate(SIDE_FLAT):
        ms.append(plinth(R, 13 + k))
        ms.append(extrude(R, 0.0, 12.0, cream(0.98), top=True, top_c=ROOF_FLAT))
        ms.append(parapet(R, 12.0, 12.9 + 0.03 * k, CREAM_L, out=0.1 + 0.015 * k, inn=0.25))
        others = [b for b in bodies if b is not R]
        ms += window_rows(R, others, [(3.0, 5.5, 1.5, "pointed")], 5.0, min_len=3.5)
    return ms


def crossing():
    ms = []
    R = CROSSING
    ms.append(extrude(R, 10.0, 29.0, cream(1.02), top=True, top_c=ROOF_FLAT))
    ms.append(cornice(R, 28.35, 29.0, 0.4, CREAM_L))
    ms += window_rows(R, [TRANS_E, TRANS_W, NAVE, APSE], [(25.4, 1.7, 1.0, "small")], 4.0)   # small windows over the roofs
    # the platform balustrade
    P = ccw(offset(R, -0.25))
    for i in range(len(P)):
        ms += balusters(P[i], P[(i + 1) % len(P)], 29.0, 1.2, WHITE, spacing=0.5)
    return ms


# ── Dome ─────────────────────────────────────────────────────────────────────
DC = centroid(DRUM)
DRUM_TOP, DOME_Y0 = 38.3, 39.3


def dome():
    ms = []
    cx, cz = DC
    R = ccw(DRUM)
    ms.append(extrude(R, 29.0, DRUM_TOP, cream(1.04), top=True))
    ms.append(cornice(R, 37.75, 38.5, 0.45, CREAM_L))
    ms.append(extrude(offset(R, -0.25), 38.5, DOME_Y0, cream(1.06), top=True))
    for i in range(len(R)):
        fr = wall_frame(R[i], R[(i + 1) % len(R)])
        o, t, n, L = fr
        if L > 9:                                                    # the four big arched windows
            ms.append(arch_on(fr, L / 2, 30.6, 7.6, 7.0, 0.04, CREAM_L))
            ms.append(arch_on(fr, L / 2, 30.9, 6.8, 6.5, 0.08, GLASS))
            for du in (-1.15, 1.15):
                ms.append(rect_on(fr, L / 2 + du, 30.9, 0.22, 5.2, 0.11, CREAM_L))
            ms.append(rect_on(fr, L / 2, 33.9, 6.8, 0.22, 0.11, CREAM_L))
            for du in (-5.6, 5.6):                                    # paired pilasters beside the window
                ms.append(rect_on(fr, L / 2 + du, 29.0, 1.0, 8.75, 0.25, CREAM_L))
        else:                                                        # the diagonal faces: a small arched window
            ms.append(arch_on(fr, L / 2, 31.2, 1.5, 3.6, 0.05, GLASS))
        # scroll buttress at each corner of the drum
        p = R[i]
        ms.append(obox((p[0] - (p[0] - cx) * 0.06, p[1] - (p[1] - cz) * 0.06), (p[0] + (p[0] - cx) * 0.1, p[1] + (p[1] - cz) * 0.1),
                       29.0, 37.75, 1.1, cream(1.08)))
        ms.append(sphere(0.35, WHITE, 8, 5).T(p[0], DOME_Y0 + 0.35, p[1]))
        # apostles (bronze, dark) on pedestals at the corners, and at the middle of the diagonal faces
        dx, dz = p[0] - cx, p[1] - cz
        dl = math.hypot(dx, dz)
        sx_, sz_ = p[0] + dx / dl * 1.4, p[1] + dz / dl * 1.4
        ms += pedestal_statue(sx_, 29.0, sz_, 2.5, CREAM_L, BRONZE_ST)
        if L < 9:
            mx, mz = (R[i][0] + R[(i + 1) % len(R)][0]) / 2, (R[i][1] + R[(i + 1) % len(R)][1]) / 2
            dx, dz = mx - cx, mz - cz
            dl = math.hypot(dx, dz)
            ms += pedestal_statue(mx + dx / dl * 1.3, 29.0, mz + dz / dl * 1.3, 2.5, CREAM_L, BRONZE_ST)
    # the dome: octagonal (flats to the cardinal points), slate, with ribs and ox-eye dormers
    A, H = 9.6, 8.2
    k8 = 1 / math.cos(math.pi / 8)
    ts = np.linspace(0, math.acos(2.4 / A), 12)
    prof = [(A * k8, 38.9)] + [(A * math.cos(t) * k8, DOME_Y0 + H * math.sin(t)) for t in ts] + [(0, DOME_Y0 + H * math.sin(ts[-1]) + 0.01)]
    ytop = DOME_Y0 + H * math.sin(ts[-1])
    ms.append(lathe(prof, DOME, 8).R(0, math.pi / 8, 0).T(cx, 0, cz))
    for j in range(8):
        a = math.pi / 8 + j * math.pi / 4
        pts = [(cx + (A * math.cos(t) * k8 + 0.05) * math.cos(a), DOME_Y0 + H * math.sin(t) + 0.03, cz + (A * math.cos(t) * k8 + 0.05) * math.sin(a)) for t in ts]
        ms.append(tube(pts, 0.16, RIB, 5))
    for j in range(4):
        a = j * math.pi / 2
        t = 0.55
        r = A * math.cos(t) + 0.5
        x, y, z = cx + r * math.cos(a), DOME_Y0 + H * math.sin(t), cz + r * math.sin(a)
        e = cyl(0.75, 0.75, 0.6, WHITE, 12, sx=1.25).R(0, 0, math.pi / 2).R(0, -a, 0).T(x, y, z)
        g = cyl(0.45, 0.45, 0.62, GLASS, 12, sx=1.25).R(0, 0, math.pi / 2).R(0, -a, 0).T(x + 0.05 * math.cos(a), y, z + 0.05 * math.sin(a))
        ms += [e, g]
    # lantern, cornice, gold balls, slate spire, ball and cross
    ms.append(cyl(3.0 * k8, 3.0 * k8, 0.5, WHITE, 8).R(0, math.pi / 8, 0).T(cx, ytop - 0.25, cz))
    ms.append(cyl(2.0 * k8, 2.0 * k8, 4.2, cream(1.06), 8).R(0, math.pi / 8, 0).T(cx, ytop + 0.25, cz))
    for j in range(8):
        a = j * math.pi / 4
        x, z = cx + 2.0 * math.cos(a), cz + 2.0 * math.sin(a)
        ms.append(box(0.95, 2.8, 0.08, GLASS).R(0, -a + math.pi / 2, 0).T(x + 0.03 * math.cos(a), ytop + 0.9, z + 0.03 * math.sin(a)))
        ac = a + math.pi / 8
        ms.append(box(0.45, 4.2, 0.45, WHITE).R(0, -ac, 0).T(cx + 2.05 * k8 * math.cos(ac), ytop + 0.25, cz + 2.05 * k8 * math.sin(ac)))
        ms.append(sphere(0.2, GOLD, 8, 5).T(cx + 2.4 * k8 * math.cos(ac), ytop + 4.95, cz + 2.4 * k8 * math.sin(ac)))
    ms.append(cyl(2.45 * k8, 2.45 * k8, 0.4, WHITE, 8).R(0, math.pi / 8, 0).T(cx, ytop + 4.45, cz))
    y = ytop + 4.85
    ms.append(lathe([(2.1 * k8, y), (1.55 * k8, y + 0.9), (0.95 * k8, y + 2.2), (0.5 * k8, y + 3.8), (0.13 * k8, y + 5.6), (0, y + 5.7)],
                    SLATE, 8).R(0, math.pi / 8, 0).T(cx, 0, cz))
    ms.append(sphere(0.5, GOLD, 12, 7).T(cx, y + 6.15, cz))
    ms.append(B(cx - 0.08, cx + 0.08, y + 6.6, 62.4, cz - 0.08, cz + 0.08, IRON))
    ms.append(B(cx - 0.6, cx + 0.6, 61.1, 61.25, cz - 0.07, cz + 0.07, IRON))
    return ms


# ── Apse (Cubas's neo-Gothic chevet) ─────────────────────────────────────────
APSE_TOP = 22.5


def apse(bodies):
    ms = []
    R = APSE
    ms.append(plinth(R, 20))
    ms.append(extrude(R, 0.0, APSE_TOP, cream(), top=False))
    ms.append(cornice(R, 22.0, APSE_TOP, 0.3, CREAM_L))
    ms.append(roof_ridge(R, APSE_TOP, (XC, -32.1), (XC, -40.6), RIDGE, SLATE, cream(), eave=0.3))
    # buttresses with pinnacles round the curved end, tall pointed windows between them
    cxa, cza = XC, -40.6
    Rc = ccw(R)
    for i in range(len(Rc)):
        p = Rc[i]
        if p[1] > -36.0:
            continue
        dx, dz = p[0] - cxa, p[1] - cza
        dl = math.hypot(dx, dz)
        ux, uz = dx / dl, dz / dl
        ms.append(obox((p[0] - ux * 0.3, p[1] - uz * 0.3), (p[0] + ux * 1.6, p[1] + uz * 1.6), 11.0, APSE_TOP + 0.4, 1.0, cream(1.04)))
        px, pz = p[0] + ux * 0.9, p[1] + uz * 0.9
        ms.append(B(px - 0.4, px + 0.4, APSE_TOP + 0.4, APSE_TOP + 1.6, pz - 0.4, pz + 0.4, CREAM_L))
        ms.append(cyl(0.42, 0.0, 2.6, CREAM_L, 6).T(px, APSE_TOP + 1.6, pz))
    for i in range(len(Rc)):
        fr = wall_frame(Rc[i], Rc[(i + 1) % len(Rc)])
        if fr[3] < 2.0 or (Rc[i][1] > -36 and Rc[(i + 1) % len(Rc)][1] > -36):
            continue
        ms.append(arch_on(fr, fr[3] / 2, 14.0, 1.3, 6.0, 0.05, GLASS, pointed=True))
    # radiating chapels with pyramid roofs, pinnacles on their corners
    for k, C in enumerate(CHAPELS):
        ms.append(plinth(C, 21 + k))
        yt = 11.4 + 0.03 * k
        ms.append(extrude(C, 0.0, yt, cream(0.98), top=False))
        ms.append(cornice(C, yt - 0.4, yt, 0.22 + 0.015 * k, CREAM_L))
        c = centroid(C)
        ms.append(roof_ridge(C, yt, c, c, 14.6, SLATE, cream(), eave=0.22 + 0.015 * k))
        for x, z in C:                                              # pinnacles on the outer corners
            if math.hypot(x - cxa, z - cza) > 9.0 and not any(inside(x + (x - c[0]) * 0.05, z + (z - c[1]) * 0.05, P) for P in bodies if P is not C):
                ms.append(B(x - 0.3, x + 0.3, yt, yt + 1.0, z - 0.3, z + 0.3, CREAM_L))
                ms.append(cyl(0.32, 0.0, 2.0, CREAM_L, 6).T(x, yt + 1.0, z))
        others = [b for b in bodies if b is not C]
        ms += window_rows(C, others, [(3.5, 5.6, 1.2, "pointed")], 4.0, margin=1.0, min_len=2.4)
    # flying-buttress walls: their tops fall from 21 m at the apse to 10 m at the outer end
    for k, W in enumerate(BUTTRESS_WALLS):
        m = extrude(W, depth(W), 10.0, cream(1.03), bottom=True)
        d = [math.hypot(x - cxa, z - cza) for x, z in W]
        d0, d1 = min(d), max(d)

        def top(v, d0=d0, d1=d1):
            dd = np.hypot(v[:, 0] - cxa, v[:, 2] - cza)
            f = np.clip((dd - d0) / (d1 - d0), 0, 1)
            v[:, 1] = np.where(v[:, 1] > 9.99, 21.0 - 11.0 * f, v[:, 1])
            return v
        ms.append(m.warp(top))
        far = max(W, key=lambda p: math.hypot(p[0] - cxa, p[1] - cza))
        ms.append(cyl(0.45, 0.0, 3.0, CREAM_L, 6).T(far[0], 10.0, far[1]))
    return ms


# ── Assembly ─────────────────────────────────────────────────────────────────
FOOTPRINT_STEPS = [(28.38, 36.56), (28.75, 36.56), (28.75, 46.35), (-11.25, 46.35), (-11.25, 36.37), (-10.74, 36.37)]


def footprint():
    """The OSM outline with the front steps, grown by 1 m (the default convex hull would clear the plazas
    and the gardens beside the apse)."""
    i0 = OUTLINE.index((28.38, 36.56))
    i1 = OUTLINE.index((-10.74, 36.37))
    ring = OUTLINE[:i0] + FOOTPRINT_STEPS + OUTLINE[i1 + 1:]
    return [[round(x, 2), round(z, 2)] for x, z in offset(ring, 1.0)]


def build():
    towers = [[(XC + s * TOWER_DX - TOWER_HX, TOWER_ZC - TOWER_HZ), (XC + s * TOWER_DX + TOWER_HX, TOWER_ZC - TOWER_HZ),
               (XC + s * TOWER_DX + TOWER_HX, TOWER_ZC + TOWER_HZ), (XC + s * TOWER_DX - TOWER_HX, TOWER_ZC + TOWER_HZ)] for s in (-1, 1)]
    fac = [(XC - 12.4, 34.5), (XC + 12.4, 34.5), (XC + 12.4, ZF), (XC - 12.4, ZF)]
    bodies = [WING_W, WING_E, AISLE_E, AISLE_W, NAVE, CROSSING, TRANS_E, TRANS_W, APSE, fac] + towers + SIDE_FLAT + CHAPELS + BUTTRESS_WALLS
    parts = {
        "wings": merge(wings(bodies)),
        "facade_towers": merge(facade()),
        "nave": merge(nave(bodies)),
        "transept": merge(transept()),
        "crossing": merge(crossing()),
        "dome": merge(dome()),
        "apse": merge(apse(bodies)),
    }
    return parts


def main():
    out = ROOT / "models" / "Q849711"
    out.mkdir(parents=True, exist_ok=True)
    parts = build()
    tris, size = write_glb(out / "model.glb", [(k, m) for k, m in parts.items()], roughness=0.9)
    allv = np.vstack([m.v for m in parts.values()])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    for k, m in parts.items():
        print(f"  {k}: {m.tris:,}")
    if "--footprint" in sys.argv:
        import json
        print(json.dumps(footprint()))


if __name__ == "__main__":
    main()
