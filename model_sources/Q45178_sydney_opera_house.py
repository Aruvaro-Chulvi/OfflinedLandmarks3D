"""
Sydney Opera House (Q45178) — the shells, the podium, the monumental steps and Bennelong Point, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Jørn Utzon (design 1957, the shells and the podium, died 2008) with Ove Arup & Partners
(engineering); finished 1967–1973 by Peter Hall (died 1995) with Lionel Todd and David Littlemore (glass walls,
interiors). Opened 1973. Protected by copyright (Utzon died less than 70 years ago): status "protected".

What is modelled:
- The shells, built the way Utzon's «spherical solution» (1961–62) defines them: every half-shell is a
  spherical triangle cut from a sphere of the same radius, 75.2 m (246 ft). It stands on a vertex, the
  pedestal on the podium, which is a pole of its sphere: the concrete ribs are great circles of the sphere
  through that pole, 3.65° apart, so they fan out from the pedestal and get wider up the shell; the ridge is
  the cut of the sphere by the vertical plane of the hall axis (Lewis 1973, Arup). Each half-shell is given by
  its pedestal P, its tip T and the rear end of its ridge B; the sphere of radius 75.2 m through P, T and B
  gives the centre, and the surface is swept rib by rib from P to the ridge. The other half is its mirror image.
  · Concert Hall (west): three north-facing main shells A1–A3 rising to the highest point of the building
    (67 m above sea level) and the big south-facing shell A4 over the southern foyer, with the small louvre
    shells between the backs of A3 and A4, and the low side shells between the pedestals.
  · Joan Sutherland Theatre (east): the same family, smaller, on an axis splayed 22.4° from the Concert Hall's.
  · Bennelong restaurant (south-west): two small shells back to back.
  · The open ends: the great glass walls (topaz glass on steel mullions) that hang from the north mouths and
    lean out over the podium, the glass of the south and restaurant mouths, the glazed gaps between nested
    shells; the ribs show on the underside of every shell (the inner surface is ribbed, one rib every 3.65°).
  · The tiles: glossy white and matt cream, the cream lid borders along the ribs and the chevrons between
    them, as vertex colours.
- The podium (pink-brown granite-aggregate panels from Tarana, paved top): the upper podium on which the shells
  stand, its stepped northern end under the glass walls, the lower podium of the south-west and south-east,
  the monumental steps (about 90 m wide) down to the forecourt, and Bennelong Point with its broadwalk at
  water level and the sea walls going down into the harbour.
- Not modelled: the interiors, the Utzon Room colonnade, the forecourt furniture, the city streets.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap relation 9596872 (outline of the building, wikidata Q45178), relations 9596870 (Upper
  Podium), 9596871 (Lower Podium), 4645750 (Monumental Steps), 9599723/9599724 (Northern/Western Broadwalk),
  19933113 (eastern broadwalk), 17582116 (Forecourt), the shell building:part relations 9596873–9677509 (the
  ridge lines of both halls) and the glass-wall ways 690624100/690624152. Tips and mouths of the shells were
  placed on an Esri World Imagery orthophoto (reference only). Hall axes from the OSM ridge lines: Concert Hall
  at 2.0° true (towards the mouths), Joan Sutherland Theatre at 24.4°; the steps face 193.3°.
- Sphere radius 75.2 m (246 ft), rib spacing 3.65°, the largest shell 54.6 m (179 ft) from its springing
  point to its top, the longest rib 64 m (210 ft) (M. Lewis, «Roof cladding of the Sydney Opera House»,
  J. Proc. Royal Society of NSW 106, 1973). Highest point 67 m above sea level (en.wikipedia), so the podium
  where the shells spring is at ≈ 12.4 m above sea level; the forecourt and the broadwalk at ≈ 3.5 m
  (en.wikipedia gives 4 m for the site) — the steps are «two storeys high» (UNESCO nomination), 9 m here.
- Building 183 m long and 120 m wide (en.wikipedia); 1,056,006 tiles in two colours, glossy white and matt
  cream, in a chevron pattern (en.wikipedia, Lewis 1973).
- The heights of the other shells were measured on photographs (Wikimedia Commons: «Aerial view of the Sydney
  Opera House.jpg», from the Harbour Bridge; «Sydney Opera House, botanic gardens 1.jpg», from the east),
  scaled on the 54.6 m of the highest shell: Concert Hall A1/A2/A3/louvre/A4 ≈ 34/44/54.6/40/31 m above the
  podium, Joan Sutherland Theatre ≈ 30/39/48/35/27 m, Bennelong ≈ 11.5 m (north shell, over the upper podium)
  and 18 m (south shell, over the lower podium) (±10 %). The pedestals are estimated from the same photographs
  (±3 m); those of A1 sit where the north glass walls (OSM ways 690624100/690624152) meet the podium.

Axes (docs/MODEL_SPEC.md): +Y up, the monumental steps (the main approach from the city) face +Z, heading
193.3°; +X points to 103.3° (east-south-east). Origin at the centroid of the OSM outline of the building
(relation 9596872), y = 0 at the level of the forecourt and the broadwalk (≈ 3.5 m above sea level).

    python model_sources/Q45178_sydney_opera_house.py   -> models/Q45178/model.glb   (--footprint prints it)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, extrude, walls, cap, offset, write_glb, GLASS as GLASS_MAT  # noqa: E402

rng = np.random.default_rng(1973)

# ── Colours ──────────────────────────────────────────────────────────────────
TILE_W = (0.955, 0.950, 0.925)      # glossy white tiles
TILE_C = (0.880, 0.850, 0.760)      # matt cream tiles (lid borders, chevrons)
RIB = (0.835, 0.815, 0.765)         # concrete ribs under the shells
RIB_D = (0.700, 0.680, 0.640)       # the deep grooves between the ribs
RIM = (0.900, 0.880, 0.830)         # edge beams of the shells
GRANITE = (0.600, 0.495, 0.430)     # podium cladding (pink granite aggregate, Tarana)
GRANITE_D = (0.520, 0.425, 0.370)
PAVE = (0.760, 0.665, 0.580)        # podium paving (pink granite slabs)
PAVE_L = (0.800, 0.715, 0.635)      # broadwalk, forecourt
STEP = (0.735, 0.650, 0.570)
SEAWALL = (0.560, 0.530, 0.480)
GLASS_C = (0.300, 0.300, 0.285)     # topaz glass walls
MULLION = (0.430, 0.410, 0.370)
DARK = (0.120, 0.115, 0.110)        # deep openings, shadowed glazing
RAIL = (0.300, 0.290, 0.270)

# ── Levels (y = 0: forecourt and broadwalk, ≈ RL 3.5 m) ──────────────────────
Y_POD = 9.0            # upper podium, springing of the shells (≈ RL 12.4)
Y_LOW = 4.5            # lower podium (south-west and south-east) and the landing of the steps
Y_SEA = -3.4           # mean sea level
Y_FOUND = -6.0         # sea walls and podium walls go down to here
R_SPH = 75.2           # the sphere of every shell (Lewis 1973: 246 ft)
RIB_DEG = 3.65         # rib spacing


def tint(c, k=0.012):
    return tuple(np.clip(np.array(c) * (1 + rng.normal(0, k)), 0, 1))


def quadstrip(A, B, c):
    """Surface between two polylines A and B (same number of points), two triangles per segment."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    n = len(A)
    V = np.vstack([A, B])
    F = []
    for i in range(n - 1):
        F += [(i, i + 1, n + i + 1), (i, n + i + 1, n + i)]
    return Mesh(V, F, c)


def grid_mesh(P, C):
    """P (K+1, J+1, 3) grid of points, C (K+1, J+1, 3) colours -> triangles."""
    K, J = P.shape[0] - 1, P.shape[1] - 1
    V = P.reshape(-1, 3); Cc = C.reshape(-1, 3)
    F = []
    for k in range(K):
        for j in range(J):
            a = k * (J + 1) + j
            b = a + J + 1
            if (k + j) % 2:
                F += [(a, b, b + 1), (a, b + 1, a + 1)]
            else:
                F += [(a, b, a + 1), (b, b + 1, a + 1)]
    return Mesh(V, F, Cc)


# ── The spherical solution ──────────────────────────────────────────────────
def sphere_centre(P, T, B, R, out):
    """Centre of the sphere of radius R through P, T, B, on the side opposite to the direction out."""
    a, b = T - P, B - P
    n = np.cross(a, b)
    nn = np.dot(n, n)
    c0 = P + (np.dot(b, b) * np.cross(n, a) + np.dot(a, a) * np.cross(b, n)) / (2 * nn)   # circumcentre
    r0 = np.linalg.norm(c0 - P)
    h = math.sqrt(max(R * R - r0 * r0, 0.0))
    n = n / math.sqrt(nn)
    C1, C2 = c0 + h * n, c0 - h * n
    return C1 if np.dot(C1 - c0, out) < 0 else C2


def rodrigues(v, k, ang):
    return v * math.cos(ang) + np.cross(k, v) * math.sin(ang) + k * np.dot(k, v) * (1 - math.cos(ang))


class HalfShell:
    """A half-shell in hall coordinates (u along the hall axis towards the mouth of north-facing shells, y up,
    w across; the ridge is the plane w = 0). P pedestal, T tip, B rear end of the ridge."""

    def __init__(self, P, T, B, R=R_SPH):
        self.P, self.T, self.B = (np.array(p, float) for p in (P, T, B))
        self.sig = 1.0 if self.P[2] > 0 else -1.0
        out = np.array([0.0, 1.0, self.sig])
        self.C = sphere_centre(self.P, self.T, self.B, R, out)
        self.R = R
        self.n = (self.C - self.P) / R
        dT = self._tdir(self.T); dB = self._tdir(self.B)
        axis = np.cross(dT, dB); s = np.linalg.norm(axis)
        self.axis = axis / s
        self.phi = math.atan2(s, float(np.dot(dT, dB)))
        self.dT = dT

    def _tdir(self, X):
        v = X - self.P
        v = v - np.dot(v, self.n) * self.n
        return v / np.linalg.norm(v)

    def rib(self, frac, ts):
        """Points of the rib at fraction frac (0 = front edge through T, 1 = back edge through B), at the
        fractions ts of its length from the pedestal to the ridge."""
        d = rodrigues(self.dT, self.axis, frac * self.phi)
        tt = np.linspace(0, math.pi, 721)
        X = self.C[None, :] - self.R * np.outer(np.cos(tt), self.n) + self.R * np.outer(np.sin(tt), d)
        w = X[:, 2] * self.sig
        i = int(np.argmax(w <= 0)) if np.any(w <= 0) else len(tt) - 1
        t0, t1 = tt[max(i - 1, 0)], tt[i]
        for _ in range(30):                                   # bisection on w = 0
            tm = 0.5 * (t0 + t1)
            xm = self.C - self.R * np.cos(tm) * self.n + self.R * np.sin(tm) * d
            if xm[2] * self.sig > 0:
                t0 = tm
            else:
                t1 = tm
        tr = 0.5 * (t0 + t1)
        t = np.asarray(ts) * tr
        return self.C[None, :] - self.R * np.outer(np.cos(t), self.n) + self.R * np.outer(np.sin(t), d)

    def ribs_count(self):
        return max(4, int(round(math.degrees(self.phi) / RIB_DEG)))

    def mesh(self, J=22, thick=1.3, web=0.55, colour=True):
        """Outer tiled surface, ribbed underside and the edge beams (front, back, ridge)."""
        nr = self.ribs_count()
        K = 2 * nr                                    # rib lines on even k, middle of the lids on odd k
        ts = np.linspace(0, 1, J + 1) ** 0.85         # a little denser near the pedestal
        P = np.stack([self.rib(k / K, ts) for k in range(K + 1)])           # (K+1, J+1, 3)
        Nrm = (P - self.C[None, None, :]) / self.R                          # outward normals of the sphere
        # tiles: cream lid borders on the rib lines, white fields with cream chevrons
        kk = np.arange(K + 1)[:, None] * np.ones((1, J + 1))
        jj = np.ones((K + 1, 1)) * np.arange(J + 1)[None, :]
        col = np.empty(P.shape)
        border = (kk % 2 == 0)
        chev = (kk % 2 == 1) & (jj % 2 == 1)
        col[...] = TILE_W
        col[chev] = (np.array(TILE_W) * 0.55 + np.array(TILE_C) * 0.45)
        col[border] = TILE_C
        col[:, 0] = TILE_C
        outer = grid_mesh(P, col)
        # underside: the ribs (rib lines deep, the lids between them thin), so they show at the open ends
        depth = np.where(kk % 2 == 0, thick, web) * (0.55 + 0.45 * (1 - jj / J))
        depth[0, :] = thick; depth[-1, :] = thick
        Q = P - Nrm * depth[..., None]
        qc = np.empty(P.shape); qc[...] = RIB_D; qc[kk % 2 == 0] = RIB
        inner = grid_mesh(Q, qc); inner.f = inner.f[:, ::-1]
        # edge beams: front rib (k = 0), back rib (k = K), ridge (j = J)
        front = quadstrip(P[0], Q[0], RIM)
        back = quadstrip(P[-1], Q[-1], RIM)
        ridge = quadstrip(P[:, -1], Q[:, -1], RIM)
        self.grid = P; self.under = Q
        return merge([outer, inner, front, back, ridge])

    def front_edge(self, n=24):
        return self.rib(0.0, np.linspace(0, 1, n + 1))

    def back_edge(self, n=24):
        return self.rib(1.0, np.linspace(0, 1, n + 1))


def mirror_w(m):
    m = m.copy(); m.v[:, 2] *= -1; m.f = m.f[:, ::-1]; return m


def shell_pair(spec, y0):
    """spec: dict(uT, hT, uB, hB, uP, a) in hall coordinates, heights above the springing level y0.
    Returns (mesh of both halves, the half-shell object of the w < 0 side)."""
    hs = HalfShell((spec["uP"], y0, -spec["a"]), (spec["uT"], y0 + spec["hT"], 0.0),
                   (spec["uB"], y0 + spec["hB"], 0.0))
    m = hs.mesh(J=spec.get("J", 22))
    return merge([m, mirror_w(m)]), hs


def mouth_glass(hs, y0, lean, n=24, strips=12, foot_y=None):
    """Glass wall closing the open end of a shell pair: from the front edge (both halves) down to the podium,
    its foot pushed forward by lean × height (the glass walls of the Opera House lean out)."""
    fy = y0 if foot_y is None else foot_y
    E = hs.under[0].copy()                                # underside of the front rib, pedestal -> tip
    E[-1, 2] = 0.0
    E2 = E[::-1].copy(); E2[:, 2] *= -1
    edge = np.vstack([E, E2[1:]])                         # pedestal -> tip -> other pedestal
    fwd = 1.0 if hs.T[0] > hs.P[0] else -1.0
    foot = edge.copy()
    foot[:, 0] += fwd * lean * (edge[:, 1] - fy)
    foot[:, 1] = fy
    # vertical-ish strips (mullions every other strip)
    ms = []
    m = len(edge)
    for i in range(m - 1):
        c = MULLION if i % 3 == 0 else GLASS_C
        ms.append(Mesh([edge[i], edge[i + 1], foot[i + 1], foot[i]], [(0, 1, 2), (0, 2, 3)], c))
    # the sloping sides of the foot (the glass is a hanging, folded wall; close it at the podium)
    return merge(ms)


# ── Halls ───────────────────────────────────────────────────────────────────
# hall frames in model axes: origin at the plan point of the A1 tip, s = towards the north mouths, l across.
def hall_frame(origin, s):
    s = np.array(s, float); s /= np.linalg.norm(s)
    l = np.array([-s[1], s[0]])
    return np.array(origin, float), s, l


def to_model(m, frame):
    o, s, l = frame
    u, y, w = m.v[:, 0].copy(), m.v[:, 1].copy(), m.v[:, 2].copy()
    m.v = np.stack([o[0] + u * s[0] + w * l[0], y, o[1] + u * s[1] + w * l[1]], 1)    # det = +1: winding kept
    return m


CH_FRAME = hall_frame((-28.0, -69.1), (-0.195, -0.981))
JST_FRAME = hall_frame((37.4, -64.9), (0.194, -0.981))
BEN_FRAME = hall_frame((-28.9, 48.3), (0.0, -1.0))

# u: metres along the hall axis from the A1 tip (negative = towards the steps); heights above the springing
# level. "dir": +1 the mouth faces the harbour (north), -1 it faces the steps.
CH = {
    "A1": dict(uT=0.0, hT=34.0, uB=-36.0, hB=24.0, uP=-5.0, a=24.0, dir=1),
    "A2": dict(uT=-27.1, hT=44.0, uB=-60.0, hB=33.0, uP=-32.0, a=24.0, dir=1),
    "A3": dict(uT=-48.7, hT=54.6, uB=-81.2, hB=42.0, uP=-54.0, a=22.0, dir=1),
    "C": dict(uT=-84.0, hT=40.0, uB=-63.0, hB=33.0, uP=-73.0, a=17.0, dir=-1, J=16),
    "A4": dict(uT=-112.0, hT=31.0, uB=-81.2, hB=25.0, uP=-100.0, a=14.0, dir=-1),
}
JST = {k: dict(v, uT=v["uT"] * 0.80, hT=v["hT"] * 0.88, uB=v["uB"] * 0.80, hB=v["hB"] * 0.88,
               uP=v["uP"] * 0.80, a=v["a"] * 0.78) for k, v in CH.items()}
# Bennelong: the shell over the restaurant facing the harbour stands on the upper podium, the one facing the
# steps springs from the lower podium (y0 given per shell)
BEN = {
    "N": dict(uT=15.0, hT=11.5, uB=-3.0, hB=10.5, uP=10.0, a=12.5, dir=1, y0=Y_POD, J=16),
    "S": dict(uT=-27.0, hT=18.0, uB=-1.0, hB=15.5, uP=-17.0, a=15.0, dir=-1, y0=Y_LOW, J=18),
}


def tiles_tri(A, B, X, n, sig, bulge, y0):
    """A curved triangle A-B-X (a side shell): barycentric grid, pushed out by bulge (×4·a·b·c… shape) towards
    w·sig; tiled with cream borders, dark glazing along its foot."""
    V, C, idx = [], [], {}
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            a, b, c = i / n, j / n, k / n
            p = a * A + b * B + c * X
            p = p + np.array([0, 0, sig]) * bulge * 6.75 * a * b * c ** 0.5 * (a + b)
            idx[(i, j)] = len(V); V.append(p)
            if p[1] < y0 + 3.2:
                C.append(DARK)
            elif (i + j) % 3 == 0:
                C.append(TILE_C)
            else:
                C.append(TILE_W)
    F = []
    for i in range(n):
        for j in range(n - i):
            F.append((idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
            if i + j < n - 1:
                F.append((idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
    m = Mesh(V, F, np.array(C))
    if sig < 0:
        m.f = m.f[:, ::-1]
    return m


def crossing(E1, E2, y_min):
    """Indices (i, j) of the points of E1 and E2 nearest to each other in the u–y plane (where the two edges
    cross seen from the side), above y_min."""
    best, bd = (len(E1) - 1, len(E2) - 1), 1e9
    for j, p in enumerate(E2):
        if p[1] < y_min:
            continue
        d = np.hypot(E1[:, 0] - p[0], E1[:, 1] - p[1])
        i = int(np.argmin(d))
        if d[i] < bd:
            bd, best = d[i], (i, j)
    return best


def resample(L, n):
    L = np.asarray(L, float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, L[:, k]) for k in range(3)], 1)


def side_shells(objs, spec, y0s, n=18, m=7, bulge=1.2):
    """Side shells between consecutive shells: the warped surface spanning from the back edge of one shell
    (from its pedestal up to where the next shell's edge crosses it, seen from the side) to that edge of the
    next shell (from its pedestal up to the same place). Tiled; dark glazing along the foot (the foyers behind)."""
    names = list(spec)
    ms = []
    for a, b in zip(names, names[1:]):
        hi, hj = objs[a], objs[b]
        e1 = hi.back_edge(60) if spec[a]["dir"] > 0 else hi.front_edge(60)     # edge of a facing the steps
        e2 = hj.front_edge(60) if spec[b]["dir"] > 0 else hj.back_edge(60)     # edge of b facing the harbour
        y0 = min(y0s[a], y0s[b])
        i, j = crossing(e1, e2, y0 + 4)
        L1 = resample(e1[:i + 1], n); L2 = resample(e2[:j + 1], n)
        L2[-1] = L1[-1]
        G = np.zeros((m + 1, n, 3)); C = np.zeros((m + 1, n, 3))
        for r in range(m + 1):
            f = r / m
            row = L1 * (1 - f) + L2 * f
            row[:, 2] -= bulge * math.sin(math.pi * f) * np.linspace(1, 0, n)          # bulge outwards (w < 0)
            G[r] = row
            for q in range(n):
                C[r, q] = DARK if row[q, 1] < y0 + 4.5 else (TILE_C if (q + r) % 3 == 0 else TILE_W)
        g = grid_mesh(G, C)
        ms += [g, mirror_w(g)]
    return merge(ms)


def hall(spec, frame, y0, glass_lean):
    parts, glass = [], []
    objs, y0s = {}, {}
    for name, sp in spec.items():
        yy = sp.get("y0", y0)
        m, hs = shell_pair(sp, yy)
        objs[name] = hs; y0s[name] = yy
        parts.append(m)
        first_or_last = name in (list(spec)[0], list(spec)[-1])
        glass.append(mouth_glass(hs, yy, glass_lean if first_or_last else 0.0))
    parts.append(side_shells(objs, spec, y0s))
    shells = to_model(merge(parts), frame)
    gl = to_model(merge(glass), frame)
    return shells, gl, objs


# ── Podium, steps, Bennelong Point (model axes; OSM rings in metres) ────────
UPPER = [(-46.2, 45.2), (-46.2, -21.6), (-46.8, -22.9), (-48.8, -27.3), (-52.8, -36.2), (-52.4, -48.4),
         (-52.3, -55.2), (-57.6, -55.1), (-57.7, -74.1), (-54.0, -76.5), (-47.6, -79.3), (-42.4, -81.5),
         (-36.1, -83.5), (-29.4, -84.8), (-23.2, -85.7), (-12.7, -85.7), (-4.1, -84.7), (-2.6, -81.2),
         (0.1, -74.9), (2.8, -66.8), (-0.5, -65.5), (2.1, -58.8), (4.6, -52.3), (9.2, -52.1), (19.2, -74.7),
         (22.6, -75.0), (26.9, -75.9), (33.5, -76.0), (38.6, -75.2), (43.9, -74.1), (48.6, -72.6), (52.5, -71.1),
         (56.6, -68.2), (56.6, -38.8), (50.5, -23.8), (50.3, 45.2)]
LOW_W = [(-46.5, 83.8), (-46.5, 45.2), (-12.8, 45.2), (-12.8, 83.8)]           # Bennelong platform
LOW_E = [(39.7, 83.8), (39.7, 45.2), (50.3, 45.2), (49.8, 60.5), (40.1, 60.6), (40.1, 75.8)]
STAIR_X = (-12.8, 39.7)
# the profile of the monumental steps (z, y): flights and landings from the upper podium down to the forecourt
STAIR_PROFILE = [(45.2, Y_POD), (58.0, 6.6), (62.0, 6.6), (79.5, Y_LOW), (83.8, Y_LOW), (101.4, 0.0)]
SITE = [(-71.1, -72.9), (-71.1, -84.7), (-71.0, -88.7), (-70.3, -90.2), (-69.5, -91.3), (-67.5, -91.7),
        (-65.6, -91.8), (-62.8, -91.5), (-62.8, -89.6), (-62.7, -87.6), (-52.4, -87.7), (-52.6, -91.2),
        (-51.0, -92.5), (-43.6, -97.1), (-32.5, -103.7), (-27.1, -106.7), (-19.7, -109.5), (-13.9, -111.0),
        (-5.7, -112.4), (3.6, -112.9), (14.6, -112.5), (22.2, -111.4), (27.8, -109.6), (33.5, -106.9),
        (38.9, -103.2), (44.2, -97.8), (48.1, -92.4), (57.1, -74.5), (61.2, -74.6), (61.1, -43.0), (52.3, -23.2),
        (52.6, -9.7), (53.4, 71.5), (56.2, 71.6), (63.0, 88.9), (64.3, 92.2), (67.8, 101.3), (72.1, 112.8),
        (75.5, 116.3), (77.6, 118.5), (81.5, 123.4), (78.0, 126.5), (-71.2, 126.5), (-71.2, 53.1), (-71.3, -10.2)]


def stairs(x0, x1, profile, rise=0.16):
    """Steps across x0..x1 following profile [(z, y)] (flights where y changes, landings where it does not),
    with granite cheek walls and a solid body down to the foundations."""
    tops, cols = [], []
    pts = [profile[0]]
    for (za, ya), (zb, yb) in zip(profile, profile[1:]):
        if abs(yb - ya) < 1e-6:
            pts.append((zb, yb)); continue
        n = max(1, int(round(abs(ya - yb) / rise)))
        dz = (zb - za) / n; dy = (yb - ya) / n
        for i in range(n):
            z = za + dz * (i + 1)
            pts.append((z, ya + dy * i))        # tread end
            pts.append((z, ya + dy * (i + 1)))  # riser foot
    ms = []
    for i, ((za, ya), (zb, yb)) in enumerate(zip(pts, pts[1:])):
        c = STEP if abs(yb - ya) < 1e-6 else GRANITE_D
        if abs(yb - ya) < 1e-6 and (zb - za) > 2.5:
            c = PAVE
        ms.append(Mesh([(x0, ya, za), (x1, ya, za), (x1, yb, zb), (x0, yb, zb)], [(0, 2, 1), (0, 3, 2)], c))
    ms.append(Mesh([(x0, pts[-1][1], pts[-1][0]), (x1, pts[-1][1], pts[-1][0]), (x1, Y_FOUND, pts[-1][0]),
                    (x0, Y_FOUND, pts[-1][0])], [(0, 1, 2), (0, 2, 3)], GRANITE_D))
    # cheek walls: the profile closed down to the foundations, seen from the sides
    prof = [(z, y) for z, y in pts] + [(pts[-1][0], Y_FOUND), (pts[0][0], Y_FOUND)]
    for x, up in ((x0, False), (x1, True)):
        m = cap([(z, y) for z, y in prof], 0.0, GRANITE, up=up)
        m.v = np.stack([np.full(len(m.v), x), m.v[:, 2], m.v[:, 0]], 1)
        ms.append(m)
    return merge(ms)


NORTH_ARCS = [  # the north faces of the podium under the glass walls (OSM outline), with the hall centres
    ([(-57.7, -74.1), (-54.0, -76.5), (-47.6, -79.3), (-42.4, -81.5), (-36.1, -83.5), (-29.4, -84.8), (-23.2, -85.7),
      (-12.7, -85.7), (-4.1, -84.7), (-2.6, -81.2)], (-25.0, -40.0)),
    ([(19.2, -74.7), (22.6, -75.0), (26.9, -75.9), (33.5, -76.0), (38.6, -75.2), (43.9, -74.1), (48.6, -72.6),
      (52.5, -71.1), (56.6, -68.2)], (35.0, -35.0)),
]
TIERS = [(1.6, 6.6), (3.2, 4.2), (4.8, 1.8)]       # (offset out from the podium face, top) of the stepped tiers


def tiers():
    """The stepped granite tiers at the north ends of the halls (the northern foyers step down to the broadwalk)."""
    ms = []
    for arc, c in NORTH_ARCS:
        A = np.array(arc, float)
        out = A - np.array(c)
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        d_prev = 0.0
        for d, y in TIERS:
            inner = [tuple(p) for p in (A + out * d_prev)]
            outer = [tuple(p) for p in (A + out * d)]
            ms.append(extrude(inner + outer[::-1], Y_FOUND, y, GRANITE, top_c=PAVE))
            d_prev = d - 0.02
    return merge(ms)


def podium():
    ms = [tiers(),extrude(UPPER, Y_FOUND, Y_POD, GRANITE, top_c=PAVE),
          extrude(LOW_W, Y_FOUND, Y_LOW, GRANITE, top_c=PAVE),
          extrude(LOW_E, Y_FOUND, Y_LOW, GRANITE, top_c=PAVE)]
    # parapet along the edge of the upper podium (a granite upstand 1.1 m)
    P = UPPER
    for (x0, z0), (x1, z1) in zip(P, P[1:]):
        if z0 > 44 and z1 > 44:
            continue
        L = math.hypot(x1 - x0, z1 - z0)
        if L < 0.5:
            continue
        b = box(L, 1.1, 0.45, GRANITE_D).R(0, -math.atan2(z1 - z0, x1 - x0), 0)
        ms.append(b.T((x0 + x1) / 2, Y_POD, (z0 + z1) / 2))
    # dark strip windows of the podium walls (the foyers and rehearsal rooms behind the cladding)
    for (x0, z0), (x1, z1) in zip(UPPER, UPPER[1:]):
        L = math.hypot(x1 - x0, z1 - z0)
        if L < 8 or (z0 > 44 and z1 > 44):
            continue
        d = np.array([x1 - x0, z1 - z0]) / L
        nrm = np.array([d[1], -d[0]])                 # outward for a counter-clockwise ring in (x, z)?
        if np.dot(nrm, np.array([(x0 + x1) / 2, (z0 + z1) / 2]) - np.array([0, -10])) < 0:
            nrm = -nrm
        a = np.array([x0, z0]) + d * 1.5 + nrm * 0.05
        e = np.array([x1, z1]) - d * 1.5 + nrm * 0.05
        ms.append(Mesh([(a[0], 4.0, a[1]), (e[0], 4.0, e[1]), (e[0], 5.4, e[1]), (a[0], 5.4, a[1])],
                       [(0, 1, 2), (0, 2, 3)], DARK))
    ms.append(stairs(STAIR_X[0], STAIR_X[1], STAIR_PROFILE))
    ms.append(stairs(-46.5, STAIR_X[0], [(83.8, Y_LOW), (101.4, 0.0)]))
    return merge(ms)


def point():
    """Bennelong Point: the broadwalk and forecourt at y = 0 and the sea walls down into the harbour."""
    top = cap(SITE, 0.05, PAVE_L)
    wall = walls(SITE, Y_FOUND, 0.0, SEAWALL)
    # a low granite kerb along the water
    kerb = walls(offset(SITE, -0.5), 0.0, 0.55, GRANITE_D)
    kerb_t = []
    return merge([top, wall, kerb] + kerb_t)


def main():
    out = ROOT / "models" / "Q45178"
    out.mkdir(parents=True, exist_ok=True)
    ch, ch_g, _ = hall(CH, CH_FRAME, Y_POD, 0.28)
    js, js_g, _ = hall(JST, JST_FRAME, Y_POD, 0.28)
    bn, bn_g, _ = hall(BEN, BEN_FRAME, Y_POD, 0.20)
    parts = [("concert_hall_shells", ch), ("joan_sutherland_theatre_shells", js), ("bennelong_shells", bn),
             ("glass_walls", merge([ch_g, js_g, bn_g]), {"material": "glass"}),
             ("podium_and_steps", podium()), ("bennelong_point", point())]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.85, materials={"glass": GLASS_MAT})
    allv = np.vstack([p[1].v for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.1f}..{hi[0]:.1f}, "
          f"y {lo[1]:.1f}..{hi[1]:.1f}, z {lo[2]:.1f}..{hi[2]:.1f}")
    for name, m, *rest in parts:
        print(f"  {name}: {m.tris:,}")
    if "--footprint" in sys.argv:
        print(json.dumps([[round(x, 1), round(z, 1)] for x, z in offset(SITE, 1.0)]))


if __name__ == "__main__":
    main()
