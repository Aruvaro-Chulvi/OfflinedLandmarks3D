"""
Monasterio de Santa María de El Paular (Q1752772), Rascafría (Madrid) — the charterhouse, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Carthusian monastery founded in 1390 by King John I of Castile; the church, its portal and atrium
and the templete of the great cloister are attributed to Juan Guas (died 1496); the Baroque Sagrario and
Transparente behind the high altar by Francisco Hurtado Izquierdo (died 1725), begun 1718; the bell tower
and the roofs of the church were rebuilt in the 18th–20th centuries. Benedictine monastery since 1954.
Public domain.

What is modelled:
- The church: the long granite nave with its gabled slate roof, buttresses with pinnacles, the Gothic west front
  (the portal of Juan Guas) towards the atrium; the bell tower rising from the east end of the nave (square
  belfry, bell-shaped slate roof with dormers, lantern); the round drum of the Transparente with its slate cone;
  the Sagrario beyond it: a cross of low wings with slate roofs round an octagonal drum with a lead dome and
  lantern.
- The rest of the monastery as OpenStreetMap draws it (one multipolygon): the ranges round the great cloister
  (the monks' cloister, ~43 m square, with the octagonal Gothic templete in its centre), the atrium, the
  smaller courtyards and the south-east wing with its cloister (the claustrillo), all as two-storey stone
  ranges with hipped roofs of red tile; the courtyards with grass or paving.
- Not modelled: interiors, the cloister arcades in detail (dark arches only), the cells' gardens, the outer
  precinct walls, the Puente del Perdón, the farm buildings.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap relation 531354 (outer ways 53011735 and 928933129 — the church —, courtyards 53011748,
  53011750, 53011739, 53011747, 53011745, the octagonal templete 928933130) and relation 531355 (the
  claustrillo: ways 53011736 and 53011738) (© OpenStreetMap contributors, ODbL). The church is 12.7 m wide and
  40 m long to the drum of the Transparente (r ≈ 4.7 m); the Sagrario fills the 19 × 13 m bay east of it.
- Heights: no published figures; measured on Wikimedia Commons photographs (the 12.7 m nave and the 6.5 m
  tower as scale, ±10–15 %): nave eaves 17 m, ridge 20.5 m; tower cornice 27 m, slate roof to 33 m, lantern to
  36.5 m; drum of the Transparente eaves 17.5 m, cone to 21.5 m; Sagrario wings 11 m, drum 14.5 m, dome to
  19.5 m, lantern to 22 m; the other ranges 8.5 m to the eaves (two storeys), tile roofs pitched ~29°.
- Ground: EU-DEM 25 m (Copernicus, via OpenTopoData): 1159 m at the origin, the valley floor falls ~5 m
  to the east (to the Lozoya) across the 160 m of the complex. y = 0 is the ground at the origin; floors of the
  courtyards at y = 0, foundations 5 m below.

Axes (docs/MODEL_SPEC.md): +Y up, the west front of the church (the portal of Juan Guas, towards the atrium)
faces +Z (heading 253.1°, along the axis of the nave); +X points to 163.1° (south-south-east). Origin at the
centroid of OSM relation 531354.

    python model_sources/Q1752772_monasterio_de_el_paular.py   -> models/Q1752772/model.glb (+ footprint in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, cap, walls, extrude, offset, ring_area, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(1390)
HEADING = 253.1

ASHLAR = (0.83, 0.79, 0.69)        # light granite ashlar of the church
ASHLAR_L = (0.90, 0.87, 0.79)      # cornices, pinnacles, frames
RUBBLE = (0.72, 0.67, 0.58)        # rubble masonry of the ranges
SLATE = (0.38, 0.41, 0.46)         # slate roofs of the church and tower
LEAD = (0.47, 0.50, 0.54)          # the dome (lead)
TILE = (0.70, 0.40, 0.29)          # red tile roofs
GRASS = (0.42, 0.52, 0.28)
PAVING = (0.68, 0.65, 0.58)
DARK = (0.08, 0.08, 0.09)
DOOR = (0.28, 0.20, 0.13)

# ── Data (OpenStreetMap © OpenStreetMap contributors, ODbL), metres in the model axes ──────────────────
D = json.loads(r"""{"53011735": [[0.45, 86.48], [-1.52, 83.38], [-3.92, 80.16], [20.87, 62.36], [12.04, 12.94], [-4.83, 13.34], [-4.98, 53.82], [-16.95, 58.13], [-17.72, 35.62], [-64.05, 35.02], [-64.16, 44.44], [-68.31, 44.57], [-67.82, 40.63], [-74.91, 41.08], [-75.43, -22.62], [-79.81, -22.34], [-79.8, -23.11], [-79.72, -27.11], [-79.58, -34.62], [-77.36, -34.6], [-15.43, -34.08], [-15.24, -45.06], [-17.56, -45.03], [-23.59, -44.97], [-23.28, -58.81], [-19.26, -59.04], [-13.13, -59.41], [-13.18, -61.49], [-4.89, -61.45], [-4.9, -63.67], [-4.07, -63.67], [2.65, -63.72], [2.73, -70.37], [2.8, -76.16], [2.82, -76.92], [22.0, -76.82], [21.58, -63.48], [32.44, -63.15], [32.12, -45.76], [31.98, -38.3], [31.67, -26.08], [35.42, -26.07], [60.57, -25.27], [68.51, -25.02], [69.26, -25.0], [71.4, -25.09], [68.93, 12.04], [66.15, 12.14], [61.1, 17.0], [63.94, 22.15], [75.42, 41.75], [30.11, 66.32]], "928933129": [[4.93, -9.72], [17.7, -10.36], [17.68, -49.64], [14.1, -53.69], [15.64, -55.72], [15.66, -58.36], [13.5, -61.98], [9.27, -62.33], [6.47, -59.75], [6.15, -55.17], [7.9, -53.24], [5.47, -50.86]], "53011736": [[35.42, -26.07], [31.67, -26.08], [31.98, -38.3], [32.12, -45.76], [35.32, -45.8], [35.42, -49.39], [60.89, -48.66], [60.86, -45.38], [68.31, -45.67], [68.51, -25.02], [60.57, -25.27]], "53011738": [[40.95, -28.89], [41.21, -43.78], [56.18, -43.55], [55.91, -28.66]], "53011748": [[-62.21, 22.01], [-61.98, -21.01], [-19.5, -20.39], [-19.21, 22.31]], "53011750": [[-6.62, -10.46], [-6.52, -1.2], [4.07, -1.58], [4.17, -7.11], [4.12, -9.29], [-2.37, -8.92], [-2.57, -10.78]], "53011739": [[19.33, 6.5], [19.16, -14.51], [34.3, -14.5], [35.26, 3.8]], "53011747": [[52.09, -3.0], [52.29, -10.47], [57.6, -10.34], [57.41, -2.87]], "53011745": [[34.88, 52.6], [28.92, 19.67], [56.32, 15.04], [60.23, 38.72]], "928933130": [[-43.13, 4.36], [-43.28, 0.73], [-40.33, -1.81], [-37.64, -1.46], [-35.29, 0.51], [-35.12, 3.52], [-37.55, 5.9], [-40.74, 6.18]]}""")


def R(i):
    return [tuple(p) for p in D[str(i)]]


def ccw(P):
    P = [tuple(map(float, p)) for p in P]
    return P if ring_area(P) > 0 else P[::-1]


def flip_up(m):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    down = np.cross(b - a, c - a)[:, 1] < 0
    m.f[down] = m.f[down][:, ::-1]
    return m


def inside(x, z, P):
    ins = False
    n = len(P)
    for i in range(n):
        (xi, zi), (xj, zj) = P[i], P[i - 1]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            ins = not ins
    return ins


def inside_v(X, Z, P):
    """Vectorised even-odd test of points (X, Z) in ring P."""
    P = np.asarray(P, float)
    ins = np.zeros(X.shape, bool)
    for i in range(len(P)):
        xi, zi = P[i]; xj, zj = P[i - 1]
        if zi == zj:
            continue
        m = (zi > Z) != (zj > Z)
        xc = (xj - xi) * (Z - zi) / (zj - zi) + xi
        ins ^= m & (X < xc)
    return ins


# ── Plan ─────────────────────────────────────────────────────────────────────
MASS = ccw(R(53011735))
COURTS = {"great_cloister": ccw(R(53011748)), "atrium": ccw(R(53011750)), "court_e": ccw(R(53011739)),
          "court_small": ccw(R(53011747)), "court_n": ccw(R(53011745))}
CL2 = ccw(R(53011736))
CL2_C = ccw(R(53011738))
CH = R(928933129)
NAVE = ccw([CH[0], CH[1], CH[2], CH[11]])                      # 12.7 m wide, the west front at z ≈ −10
DRUM_C = (10.9, -57.7)
DRUM_R = 4.75
SAG_C = (12.3, -70.2)                                          # the Sagrario fills the bay x 2.8–22, z −63.5…−76.9
TEMPLETE = ccw(R(928933130))

FOUND = -5.0
RANGE_EAVE = 8.5
NAVE_EAVE, NAVE_RIDGE = 17.0, 20.5
TOWER_C = (11.3, -44.5)
TOWER_S = 6.5


# ── Roofs on any outline: a hipped roof as a height field ───────────────────
def seg_dist(X, Z, rings):
    d = np.full(X.shape, np.inf)
    for P in rings:
        P = np.asarray(P, float)
        A, B = P, np.roll(P, -1, 0)
        for (ax, az), (bx, bz) in zip(A, B):
            ex, ez = bx - ax, bz - az
            L2 = ex * ex + ez * ez or 1e-9
            t = np.clip(((X - ax) * ex + (Z - az) * ez) / L2, 0, 1)
            d = np.minimum(d, np.hypot(X - ax - t * ex, Z - az - t * ez))
    return d


def nearest_on(X, Z, rings):
    """The nearest point on the rings' edges to each point."""
    best = np.full(X.shape, np.inf); NX = X.copy(); NZ = Z.copy()
    for P in rings:
        P = np.asarray(P, float)
        for (ax, az), (bx, bz) in zip(P, np.roll(P, -1, 0)):
            ex, ez = bx - ax, bz - az
            L2 = ex * ex + ez * ez or 1e-9
            t = np.clip(((X - ax) * ex + (Z - az) * ez) / L2, 0, 1)
            qx, qz = ax + t * ex, az + t * ez
            dd = np.hypot(X - qx, Z - qz)
            m = dd < best
            best = np.where(m, dd, best); NX = np.where(m, qx, NX); NZ = np.where(m, qz, NZ)
    return NX, NZ


def hip_field(outer, holes, eave, slope, rise, c, step=1.0):
    """A roof over an outline with courtyards: y = eave + slope × distance to the nearest wall (≤ rise).
    Grid vertices just outside are snapped onto the walls, so the roof meets the eaves all round."""
    rings = [outer] + list(holes)
    xs = np.arange(min(p[0] for p in outer) - step, max(p[0] for p in outer) + 2 * step, step)
    zs = np.arange(min(p[1] for p in outer) - step, max(p[1] for p in outer) + 2 * step, step)
    X, Z = np.meshgrid(xs, zs)
    ins = inside_v(X, Z, outer)
    for h in holes:
        ins &= ~inside_v(X, Z, h)
    near = np.zeros_like(ins)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            near |= np.roll(np.roll(ins, dj, 0), di, 1)
    snap = near & ~ins
    NX, NZ = nearest_on(X, Z, rings)
    X = np.where(snap, NX, X); Z = np.where(snap, NZ, Z)
    d = np.where(snap, 0.0, seg_dist(X, Z, rings))
    Y = eave + np.minimum(d * slope, rise)
    ok = ins | snap
    nz, nx = X.shape
    idx = np.arange(nz * nx).reshape(nz, nx)
    F = []
    for j in range(nz - 1):
        for i in range(nx - 1):
            a, b, cc, dd = idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]
            q = [ok[j, i], ok[j, i + 1], ok[j + 1, i + 1], ok[j + 1, i]]
            if all(q):
                if not (ins[j, i] or ins[j, i + 1] or ins[j + 1, i + 1] or ins[j + 1, i]):
                    continue
                F += [(a, b, cc), (a, cc, dd)]
            elif sum(q) == 3 and any((ins[j, i], ins[j, i + 1], ins[j + 1, i + 1], ins[j + 1, i])):
                F.append(tuple(v for v, k in zip((a, b, cc, dd), q) if k))
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    F = np.array(F, np.int64)
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    m = Mesh(V[used], remap[F], c)
    # drop slivers that cross a courtyard (all three vertices on walls, centroid outside)
    a, b, cc = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    cen = (a + b + cc) / 3
    keep = inside_v(cen[:, 0], cen[:, 2], outer)
    for h in holes:
        keep &= ~inside_v(cen[:, 0], cen[:, 2], h)
    m.f = m.f[keep]
    return flip_up(m)


# ── Pieces on a wall: local (u along, y up, w outwards) ─────────────────────
class Face:
    def __init__(self, a, t, n):
        self.a, self.t, self.n = np.asarray(a, float), np.asarray(t, float), np.asarray(n, float)

    def P(self, u, y, w):
        p = self.a + self.t * u + self.n * w
        return (p[0], y, p[1])

    def poly(self, pts, c, w=0.05):
        V = [self.P(u, y, w) for u, y in pts]
        F = [(0, i, i + 1) for i in range(1, len(V) - 1)]
        m = Mesh(V, F, c)
        a, b, cc = np.array(V[0]), np.array(V[1]), np.array(V[2])
        nn = np.cross(b - a, cc - a)
        if np.dot([nn[0], nn[2]], self.n) < 0:
            m.f = m.f[:, ::-1]
        return m

    def slab(self, u0, u1, y0, y1, w0, w1, c):
        P = [self.P(u, y, w) for w in (w0, w1) for y in (y0, y1) for u in (u0, u1)]
        quads = [(4, 5, 7, 6), (1, 0, 2, 3), (0, 4, 6, 2), (5, 1, 3, 7), (6, 7, 3, 2), (0, 1, 5, 4)]
        V, F = [], []
        for q in quads:
            o = len(V); V += [P[k] for k in q]; F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        m = Mesh(V, F, c)
        ctr = np.mean(np.array(P), 0)
        for k in range(0, len(F), 2):
            a, b, cc = (np.array(V[i]) for i in F[k])
            if np.dot(np.cross(b - a, cc - a), (a + b + cc) / 3 - ctr) < 0:
                m.f[k] = m.f[k][::-1]; m.f[k + 1] = m.f[k + 1][::-1]
        return m


def edges(P):
    P = ccw(P)
    for i in range(len(P)):
        a, b = np.array(P[i], float), np.array(P[(i + 1) % len(P)], float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-3:
            continue
        t = (b - a) / L
        yield a, b, L, t, np.array([t[1], -t[0]])


def arch_pts(u, y0, ys, half, seg=6):
    pts = [(u - half, y0), (u + half, y0)]
    pts += [(u + half * math.cos(a), ys + half * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)[1:-1]]
    return pts


def pyramid(P, y0, y1, c, apex=None):
    cx = sum(p[0] for p in P) / len(P) if apex is None else apex[0]
    cz = sum(p[1] for p in P) / len(P) if apex is None else apex[1]
    V = [(x, y0, z) for x, z in P] + [(cx, y1, cz)]
    n = len(P)
    return flip_up(Mesh(V, [(i, (i + 1) % n, n) for i in range(n)], c))


def rect_hip(x0, x1, z0, z1, eave, rise, c):
    """A hipped roof on an axis-aligned rectangle."""
    w, d = x1 - x0, z1 - z0
    if w >= d:
        h = d / 2
        V = [(x0, eave, z0), (x1, eave, z0), (x1, eave, z1), (x0, eave, z1),
             (x0 + h, eave + rise, (z0 + z1) / 2), (x1 - h, eave + rise, (z0 + z1) / 2)]
        F = [(0, 1, 5), (0, 5, 4), (1, 2, 5), (2, 3, 4), (2, 4, 5), (3, 0, 4)]
    else:
        h = w / 2
        V = [(x0, eave, z0), (x1, eave, z0), (x1, eave, z1), (x0, eave, z1),
             ((x0 + x1) / 2, eave + rise, z0 + h), ((x0 + x1) / 2, eave + rise, z1 - h)]
        F = [(0, 1, 4), (1, 2, 5), (1, 5, 4), (2, 3, 5), (3, 0, 4), (3, 4, 5)]
    return flip_up(Mesh(V, F, c))


def ngon(cx, cz, r, n, a0=0.0):
    return [(cx + r * math.cos(a0 + 2 * math.pi * k / n), cz + r * math.sin(a0 + 2 * math.pi * k / n)) for k in range(n)]


def cornice(P, y, h=0.45, d=0.35, c=ASHLAR_L):
    O = offset(P, d)
    return merge([walls(O, y, y + h, c), cap(O, y + h, c, holes=[P]), cap(O, y, c, holes=[P], up=False)])


# ── The ranges ───────────────────────────────────────────────────────────────
OCC = [NAVE]


def ranges():
    holes = list(COURTS.values())
    out = [extrude(MASS, FOUND, RANGE_EAVE, RUBBLE, holes=holes, top=False)]
    out.append(hip_field(MASS, holes, RANGE_EAVE + 0.02, 0.5, 3.2, TILE))
    out.append(extrude(CL2, FOUND, RANGE_EAVE - 1.0, RUBBLE, holes=[CL2_C], top=False))
    out.append(hip_field(CL2, [CL2_C], RANGE_EAVE - 0.98, 0.5, 3.0, TILE))
    # courtyards: grass in the great cloister and the claustrillo, paving elsewhere
    for k, C in COURTS.items():
        out.append(cap(C, 0.05, GRASS if k in ("great_cloister", "court_n") else PAVING))
    out.append(cap(CL2_C, 0.05, GRASS))
    # cloister arcades: dark arches on the inner faces of the two cloisters
    for C, yt in ((COURTS["great_cloister"], 4.2), (CL2_C, 3.8)):
        for a, b, L, t, n in edges(C):
            f = Face(a, -t, -n)
            k = max(2, int(L / 3.6))
            for j in range(k):
                u = -(j + 0.5) * L / k
                out.append(f.poly(arch_pts(u, 0.05, yt - 1.4, 1.2, 5), DARK, 0.05))
    # windows on the outer walls of the ranges
    for a, b, L, t, n in edges(MASS):
        if L < 6:
            continue
        f = Face(a, t, n)
        k = int((L - 2) // 5.0)
        for j in range(k):
            u = L / 2 + (j - (k - 1) / 2) * 5.0
            p = a + t * u + n * 1.0
            if any(inside(p[0], p[1], Q) for Q in OCC + [CL2]):
                continue
            for y in (2.0, 5.6):
                out.append(f.poly([(u - 0.5, y), (u + 0.5, y), (u + 0.5, y + 1.4), (u - 0.5, y + 1.4)], DARK, 0.04))
    return out


def templete():
    """The Gothic templete in the great cloister: an octagon with pinnacles and a stone cupola."""
    P = TEMPLETE
    cx = sum(p[0] for p in P) / len(P); cz = sum(p[1] for p in P) / len(P)
    out = [extrude(P, -0.5, 6.0, ASHLAR, top_c=ASHLAR_L), cornice(P, 6.0, 0.4, 0.3)]
    for a, b, L, t, n in edges(P):
        f = Face(a, t, n)
        out.append(f.poly(arch_pts(L / 2, 0.0, 3.0, L / 2 - 0.45, 6), DARK, 0.04))
    out.append(pyramid(offset(P, 0.3), 6.4, 9.5, ASHLAR_L))
    for x, z in P:
        out.append(cyl(0.22, 0.0, 2.2, ASHLAR_L, 4).T(x, 6.4, z))
    out.append(cyl(0.12, 0.0, 1.2, ASHLAR_L, 6).T(cx, 9.5, cz))
    return out


# ── The church ───────────────────────────────────────────────────────────────
def nave():
    P = NAVE
    out = [extrude(P, FOUND, NAVE_EAVE, ASHLAR, top_c=ASHLAR)]
    out.append(cornice(P, NAVE_EAVE - 0.45))
    # gabled slate roof, ridge along z; the west gable wall rises above it (the Gothic front)
    (x0, z0), (x1, z1) = (4.93, -9.72), (17.7, -10.36)
    (x3, z3), (x2, z2) = (5.47, -50.86), (17.68, -49.64)
    mx0, mz0 = (x0 + x1) / 2, (z0 + z1) / 2
    mx1, mz1 = (x2 + x3) / 2, (z2 + z3) / 2
    o = 0.5
    E = NAVE_EAVE
    # hipped at the east end (against the drum of the Transparente), as on the photographs
    ax, az = mx0 - mx1, mz0 - mz1
    al = math.hypot(ax, az)
    rx, rz = mx1 + ax / al * 6.4, mz1 + az / al * 6.4
    V = [(x0 - o, E - 0.25, z0), (mx0, NAVE_RIDGE, mz0), (rx, NAVE_RIDGE, rz), (x3 - o, E - 0.25, z3 - o),
         (x1 + o, E - 0.25, z1), (x2 + o, E - 0.25, z2 - o)]
    out.append(flip_up(Mesh(V, [(0, 1, 2), (0, 2, 3), (4, 5, 2), (4, 2, 1), (3, 2, 5)], SLATE)))
    # west gable with its crocketed coping, standing a little above the roof
    f = Face(np.array([x0, z0]), np.array([x1 - x0, z1 - z0]) / math.hypot(x1 - x0, z1 - z0), np.array([0.05, 1.0]))
    W = math.hypot(x1 - x0, z1 - z0)
    out.append(f.slab(-0.6, W + 0.6, FOUND, E + 0.3, -0.6, 0.6, ASHLAR))
    G = [(-0.6, E + 0.3), (W + 0.6, E + 0.3), (W / 2, NAVE_RIDGE + 1.6)]
    out.append(f.poly(G, ASHLAR, 0.6)); out.append(f.poly(G, ASHLAR, -0.6))
    for (ua, ya), (ub, yb) in ((G[0], G[2]), (G[2], G[1])):
        L = math.hypot(ub - ua, yb - ya)
        tx, ty = (ub - ua) / L, (yb - ya) / L
        nxx, nyy = -ty, tx
        if nyy < 0:
            nxx, nyy = -nxx, -nyy
        Q = [(ua, ya), (ub, yb), (ub + nxx * 0.5, yb + nyy * 0.5), (ua + nxx * 0.5, ya + nyy * 0.5)]
        for w in (0.65, -0.65):
            out.append(f.poly(Q, ASHLAR_L, w))
    out.append(cyl(0.3, 0.0, 2.2, ASHLAR_L, 4).T(*f.P(W / 2, NAVE_RIDGE + 1.6, 0.0)))
    # the portal of Juan Guas: a pointed arch in a light frame, a rose window above
    out.append(f.poly(arch_pts(W / 2, 0.0, 4.2, 2.3, 8), ASHLAR_L, 0.66))
    out.append(f.poly(arch_pts(W / 2, 0.0, 3.8, 1.7, 8), DOOR, 0.7))
    out.append(f.poly([(W / 2 + 1.6 * math.cos(a), 11.5 + 1.6 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)], DARK, 0.66))
    for u in (-0.6, W + 0.6):
        out.append(f.slab(u - 0.7, u + 0.7, FOUND, E + 2.5, 0.6, 1.6, ASHLAR))
        out.append(cyl(0.45, 0.0, 2.4, ASHLAR_L, 4).T(*f.P(u, E + 2.5, 1.1)))
    # side walls: buttresses with pinnacles and tall windows
    for a, b, L, t, n in edges(P):
        if L < 30:
            continue
        f = Face(a, t, n)
        k = 5
        for j in range(k + 1):
            u = 1.5 + j * (L - 3.0) / k
            out.append(f.slab(u - 0.6, u + 0.6, FOUND, E - 1.0, 0.0, 1.4, ASHLAR))
            out.append(cyl(0.4, 0.0, 2.0, ASHLAR_L, 4).T(*f.P(u, E - 1.0, 0.7)))
        for j in range(k):
            u = 1.5 + (j + 0.5) * (L - 3.0) / k
            out.append(f.poly([(u - 0.8, 10.5), (u + 0.8, 10.5), (u + 0.8, 13.6), (u - 0.8, 13.6)], DARK, 0.05))
    return out


def tower():
    cx, cz = TOWER_C
    h = TOWER_S / 2
    S = [(cx - h, cz - h), (cx + h, cz - h), (cx + h, cz + h), (cx - h, cz + h)]
    out = [extrude(S, NAVE_EAVE - 2.0, 27.0, ASHLAR, top_c=ASHLAR), cornice(S, 22.0, 0.35, 0.2), cornice(S, 26.6, 0.5, 0.35)]
    for a, b, L, t, n in edges(S):
        f = Face(a, t, n)
        out.append(f.poly(arch_pts(L / 2, 22.8, 25.0, 0.85, 6), DARK, 0.04))
        out.append(f.slab(-0.05, 0.6, NAVE_RIDGE - 1.0, 26.6, -0.1, 0.25, ASHLAR_L))      # pilasters at the corners
        out.append(f.slab(L - 0.6, L + 0.05, NAVE_RIDGE - 1.0, 26.6, -0.1, 0.25, ASHLAR_L))
    # bell-shaped slate roof: concave hip from the cornice to a neck, then the lantern
    S1 = [(cx - h - 0.3, cz - h - 0.3), (cx + h + 0.3, cz - h - 0.3), (cx + h + 0.3, cz + h + 0.3), (cx - h - 0.3, cz + h + 0.3)]
    lev = [(1.0, 27.1), (0.78, 28.6), (0.52, 30.6), (0.30, 32.4), (0.22, 33.0)]
    rings = [[(cx + (x - cx) * s, cz + (z - cz) * s) for x, z in S1] for s, _ in lev]
    V, F = [], []
    for i in range(len(lev) - 1):
        for k in range(4):
            o = len(V)
            a, b = rings[i][k], rings[i][(k + 1) % 4]
            c2, d2 = rings[i + 1][(k + 1) % 4], rings[i + 1][k]
            V += [(a[0], lev[i][1], a[1]), (b[0], lev[i][1], b[1]), (c2[0], lev[i + 1][1], c2[1]), (d2[0], lev[i + 1][1], d2[1])]
            F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    out.append(flip_up(Mesh(V, F, SLATE)))
    out.append(cap(rings[-1], 33.0, SLATE))
    for a, b, L, t, n in edges(S):                       # small dormers
        f = Face(a, t, n)
        out.append(f.slab(L / 2 - 0.5, L / 2 + 0.5, 28.0, 29.3, -0.9, 0.1, ASHLAR_L))
        out.append(f.poly([(L / 2 - 0.3, 28.2), (L / 2 + 0.3, 28.2), (L / 2 + 0.3, 29.0), (L / 2 - 0.3, 29.0)], DARK, 0.12))
    out.append(cyl(0.75, 0.75, 2.2, SLATE, 8).T(cx, 33.0, cz))
    out.append(cyl(0.95, 0.0, 1.4, SLATE, 8).T(cx, 35.2, cz))
    out.append(cyl(0.06, 0.06, 2.6, DARK, 4).T(cx, 36.4, cz))
    return out


def transparente():
    """The round drum of the Transparente behind the high altar, with its slate cone."""
    cx, cz = DRUM_C
    out = [cyl(DRUM_R, DRUM_R, 17.5 - FOUND, ASHLAR, 24).T(cx, FOUND, cz)]
    out.append(cyl(DRUM_R + 0.35, DRUM_R + 0.35, 0.45, ASHLAR_L, 24).T(cx, 17.1, cz))
    out.append(cyl(DRUM_R + 0.4, 0.3, 4.0, SLATE, 16).T(cx, 17.5, cz))
    out.append(cyl(0.06, 0.06, 2.0, DARK, 4).T(cx, 21.3, cz))
    for k in range(6):                                  # round-arched windows high up
        th = math.radians(-150 + k * 60)
        nn = np.array([math.cos(th), math.sin(th)])
        f = Face(np.array([cx, cz]) + nn * DRUM_R, np.array([-nn[1], nn[0]]), nn)
        out.append(f.poly(arch_pts(0, 12.5, 14.6, 0.75, 5), DARK, 0.06))
    return out


def sagrario():
    """The Sagrario of Hurtado Izquierdo: a cross of low slate-roofed wings round an octagonal drum and dome."""
    cx, cz = SAG_C
    out = []
    for w, d in ((18.8, 6.4), (6.4, 13.2)):            # the two arms of the cross, filling the OSM bay
        P = [(cx - w / 2, cz - d / 2), (cx + w / 2, cz - d / 2), (cx + w / 2, cz + d / 2), (cx - w / 2, cz + d / 2)]
        out.append(extrude(P, FOUND, 11.0, ASHLAR, top_c=SLATE))
        out.append(cornice(P, 10.6, 0.4, 0.3))
        out.append(rect_hip(cx - w / 2 - 0.3, cx + w / 2 + 0.3, cz - d / 2 - 0.3, cz + d / 2 + 0.3, 11.0, 2.4, SLATE))
    O8 = ngon(cx, cz, 4.9, 8, math.pi / 8)
    out.append(extrude(O8, 10.0, 14.5, ASHLAR, top_c=ASHLAR))
    out.append(cornice(O8, 14.1, 0.4, 0.3))
    for a, b, L, t, n in edges(O8):
        f = Face(a, t, n)
        out.append(f.poly([(L / 2 + 0.7 * math.cos(q), 12.6 + 0.7 * math.sin(q)) for q in np.linspace(0, 2 * math.pi, 8, endpoint=False)], DARK, 0.05))
    prof = [(4.8, 0.0), (4.6, 1.4), (4.0, 2.8), (3.0, 3.9), (1.6, 4.7), (0.9, 4.9)]
    dome = [lathe([(r, 14.5 + y) for r, y in prof], LEAD, 16).T(cx, 0, cz), cyl(1.1, 0.0, 1.2, LEAD, 8).T(cx, 20.8, cz)]
    out.append(cyl(0.9, 0.9, 1.4, ASHLAR_L, 8).T(cx, 19.4, cz))
    out.append(cyl(0.06, 0.06, 1.8, DARK, 4).T(cx, 22.0, cz))
    return out, dome


def build():
    global SAG
    SAG = sagrario()
    return [("ranges", merge(ranges())), ("templete", merge(templete())), ("nave", merge(nave())),
            ("tower", merge(tower())), ("transparente", merge(transparente()))] +         [("sagrario", merge(SAG[0])), ("dome", merge(SAG[1]))]


def main():
    parts = build()
    out_dir = ROOT / "models" / "Q1752772"
    out_dir.mkdir(parents=True, exist_ok=True)
    named = []
    for name, m in parts:
        k = 0.94 + 0.08 * fbm(m.v, 0.12, 5, 2)
        m.c = np.clip(m.c * k[:, None], 0, 1)
        named.append((name, m, {}))
    tris, size = write_glb(out_dir / "model.glb", named, roughness=0.9,
                           materials={"lead": {"metallic": 0.15, "roughness": 0.6}})
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        meta.write_bytes((json.dumps(mj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"{tris} triangles, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
