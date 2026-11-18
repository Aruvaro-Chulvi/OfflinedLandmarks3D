"""
Angkor Wat (Q43473), Siem Reap, Cambodia — the temple, its enclosure and its moat, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: built for King Suryavarman II (died c. 1150), first half of the 12th century; the architect is
not known (tradition names Divakarapandita as the king's priest). Public domain.

What is modelled:
- The moat (190 m of water around a rectangle about 1.5 × 1.3 km) with its stepped laterite and sandstone banks,
  the sandstone causeway from the west with its stair landings and the lion terrace at its end, and the earthen
  causeway from the east.
- The island: the 30–40 m apron of lawn, the laterite outer enclosure wall (4.5 m) with its four gopuras — the
  235 m long west gopura with its three ruined towers, its colonnade and the «elephant gates» at both ends — and
  the woods of the enclosure (simple trees, tall as the dipterocarps there are).
- The raised processional causeway (9.5 m wide) from the west gopura to the temple, its naga balustrades with the
  multi-headed naga hoods, the side stair landings, the two «libraries» and the two ponds in front of the temple,
  the cruciform Terrace of Honour.
- The temple on its terrace: the first gallery (the bas-relief gallery) with its outer pillared half-gallery,
  the west gopura with three towers, the cruciform corner pavilions and the gopuras of the other sides; the
  cruciform cloister (Preah Poan) with its four basins; the two libraries of the first court; the second gallery
  on its high moulded base with corner towers and axial gopuras; the two small libraries of the second level;
  the Bakan (third level) on its 11–12 m base with the twelve steep stairs, its gallery and cross galleries,
  the four corner towers and the central sanctuary with its porches and its lotus-bud tower, quincunx, the
  central tower 65 m above the ground. The towers are built as tiers of a redented plan along a bud-shaped
  profile, with antefixes on every tier and a lotus finial.
- Not modelled: the reliefs, carvings and devatas, the interiors, the modern buildings (the Buddhist monasteries
  north and south of the causeway), the floating pontoon bridge, the roads round the moat.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap — relations 1535019 (first gallery), 12094941 (second gallery), 1648455 (third gallery),
  ways 888649880 (central sanctuary), 886790393 … 886790447 (towers, as tiers with heights), 886790450 …
  886790468 (gallery parts with min_height/height), 888569856 (cruciform cloister), 642066967 (Terrace of
  Honour), 120320892/3 and 120324355/7 (inner libraries), 240011220 and 39842240 (outer libraries), 39842237/8
  (ponds), 1092292990 (west gopura, 235 m), 1092279918, 502932535, 1462271693 (north, south and east gopuras),
  513679283 … 513679288 (outer wall), 1113380295 (the island: inner edge of the moat), 160675481 and relation
  10930344 (the moat), 471925578 (west causeway), 1462276418 (east causeway), 446335238 … 513685094 (retaining
  walls of the temple terrace and of the Terrace of Honour). Taken to model axes and squared; the small
  differences between the two halves (1–3 m) were made symmetric about the axis of the central sanctuary.
  Sizes found: moat outside 1281 × 1500 m, island 897 × 1118 m (moat 190–193 m wide), outer wall 815 × 1033 m,
  first gallery 178 × 211 m (outer face; Wikipedia gives 187 × 215 m), second gallery 102 × 118 m (100 × 115),
  Bakan 60 m square (plus its stairs), central tower 19.4 m across at its base, corner towers 13 m.
- Heights: Wikipedia (en): outer wall 4.5 m; central tower 65 m above the ground, 43 m above the Bakan
  (Bakan floor 22 m). OpenStreetMap parts: first gallery floor 4 m and roofs 10 m, second gallery 11–17 m,
  Bakan galleries 23–29/30 m, corner towers 23 → 43 m, central tower 23 → 62 m (scaled to the published 65 m:
  corner towers 46 m). Estimated (±15–30 %): the processional causeway 1.5 m above the lawn, the temple
  terrace 1.2 m, the first court 3.2 m, the second court 10 m, the gopura and pavilion roofs and the ruined
  towers of the west gopura (14–20 m), the libraries, the Terrace of Honour (2.6 m), the laterite steps of the
  moat banks.
- Water: in the game the ground inside the footprint is levelled flat, so the moat water is put at y = +0.15 (just
  over the levelled ground) and the island is raised by 1.6 m with its stepped banks going down to the water —
  the moat's banks read as in reality (the water about 1.5–2 m below the island) and nothing is hidden under the
  levelled ground. y = 0 is the plain around the moat; every height above is given from the island (y = 1.6).
Grey and ochre sandstone with dark lichen weathering, laterite (red-brown) for the outer wall, the moat banks
and the cores of the terraces, grey corbelled stone roofs.

Axes (docs/MODEL_SPEC.md): +Y up, the main (west) front faces +Z (heading 270.6°, from the outlines of the moat,
the island, the galleries and the axis central sanctuary → west gopura), +X points south (180.6°). Origin at
the centre of the central sanctuary (OSM way 888649880).

    python model_sources/Q43473_angkor_wat.py   -> models/Q43473/model.glb (+ triangles and footprint in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, cap, walls, extrude, triangulate, fbm, sstep, write_glb  # noqa: E402

rng = np.random.default_rng(1113)
HEADING = 270.6

# ── Colours ─────────────────────────────────────────────────────────────
SAND = (0.60, 0.57, 0.50)              # grey-ochre sandstone
SAND_L = (0.68, 0.64, 0.55)            # sunlit sandstone, paving
SAND_O = (0.66, 0.58, 0.45)            # ochre sandstone
SAND_D = (0.40, 0.38, 0.34)            # doors, windows, shadowed openings
ROOF = (0.49, 0.47, 0.42)              # corbelled stone roofs
ROOF_D = (0.40, 0.39, 0.35)
LAT = (0.56, 0.36, 0.25)               # laterite
LAT_D = (0.46, 0.30, 0.22)
LAWN = (0.43, 0.55, 0.27)
LAWN_D = (0.34, 0.46, 0.22)
DRY = (0.60, 0.58, 0.38)
EARTH = (0.63, 0.53, 0.39)
TERR = (0.50, 0.56, 0.31)              # grass of the temple terrace
COURT = (0.55, 0.55, 0.36)             # worn grass of the first court
WATER = (0.26, 0.37, 0.31)
TREES = [(0.18, 0.33, 0.15), (0.22, 0.38, 0.17), (0.16, 0.29, 0.14), (0.26, 0.41, 0.19), (0.20, 0.35, 0.13)]

G = 1.6                                 # island ground above the levelled plain (see the docstring)
WL = 0.15                               # water of the moat and the ponds' level offset
# moat and island (model axes, symmetric about x = 0)
MX, MZ0, MZ1 = 640.0, -712.0, 788.0     # outside edge of the moat
IX, IZ0, IZ1 = 448.0, -523.0, 595.0     # island (inner edge of the moat)
WX, WZ0, WZ1 = 407.0, -478.0, 555.0     # outer enclosure wall (centre line)
WALL_H, WALL_T = 4.5, 1.4
# temple levels above the island ground
H_CW = 1.5                              # processional causeway
H_PL = 1.2                              # temple terrace
H_C1 = 3.2                              # first court
F1, E1, R1 = 4.0, 7.6, 10.0             # first gallery: floor, eaves, ridge
H_C2 = 10.0                             # second court
F2, E2, R2 = 10.6, 14.6, 17.0           # second gallery
F3, E3, R3 = 22.5, 27.0, 29.6           # Bakan galleries
TOP_C, TOP_K = 65.0, 46.0               # central and corner towers (above the ground)


# ── Helpers ─────────────────────────────────────────────────────────────
def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def rect(x0, x1, z0, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def ccw(P):
    P = [tuple(map(float, p)) for p in P]
    a = sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))
    return P if a > 0 else P[::-1]


def cross_ring(cx, cz, lx, lz, wx, wz):
    """A cross: a bar along x (half-length lx, half-width wz) and a bar along z (half-length lz, half-width wx)."""
    P = [(wx, lz), (-wx, lz), (-wx, wz), (-lx, wz), (-lx, -wz), (-wx, -wz), (-wx, -lz), (wx, -lz), (wx, -wz),
         (lx, -wz), (lx, wz), (wx, wz)]
    return ccw([(cx + x, cz + z) for x, z in P])


def redent(cx, cz, h, r):
    return cross_ring(cx, cz, h, h, h - r, h - r)


def loft(r0, y0, r1, y1, c, top=False):
    """Side surface between two rings with the same number of points (both counter-clockwise), and a top cap."""
    r0, r1 = ccw(r0), ccw(r1)
    n = len(r0)
    v, f = [], []
    for i in range(n):
        j = (i + 1) % n
        o = len(v)
        v += [(r0[i][0], y0, r0[i][1]), (r0[j][0], y0, r0[j][1]), (r1[j][0], y1, r1[j][1]), (r1[i][0], y1, r1[i][1])]
        f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    ms = [Mesh(v, f, c)]
    if top:
        ms.append(cap(r1, y1, c))
    return merge(ms)


def run(p0, p1, prof, c, seg=8.0, caps=True):
    """A prism along the segment p0 → p1 (x, z) with the closed profile prof = [(u, y)] (u to the left of the
    direction), cut every `seg` metres so that the vertex colours can vary along it. Faces turned outwards."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    t = (p1 - p0) / L
    nrm = np.array([-t[1], t[0]])
    prof = [tuple(map(float, q)) for q in prof]
    m = len(prof)
    k = max(1, int(math.ceil(L / seg)))
    uc = sum(q[0] for q in prof) / m; yc = sum(q[1] for q in prof) / m
    v, f = [], []
    for s in range(k + 1):
        base = p0 + t * (L * s / k)
        for u, y in prof:
            p = base + nrm * u
            v.append((p[0], y, p[1]))
    for s in range(k):
        for i in range(m):
            j = (i + 1) % m
            a, b, c2, d = s * m + i, s * m + j, (s + 1) * m + j, (s + 1) * m + i
            f += [(a, b, c2), (a, c2, d)]
    V = np.array(v); F = np.array(f)
    # outward: compare each face normal with the direction from the axis of the prism
    A, Bv, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(Bv - A, C - A)
    cen = (A + Bv + C) / 3
    sc = (cen[:, [0, 2]] - p0) @ t
    ref = np.stack([p0[0] + t[0] * sc + nrm[0] * uc, np.full(len(sc), yc), p0[1] + t[1] * sc + nrm[1] * uc], 1)
    flip = np.einsum("ij,ij->i", fn, cen - ref) < 0
    F[flip] = F[flip][:, ::-1]
    ms = [Mesh(V, F, c)]
    if caps:
        pts, tr = triangulate(prof)
        for s, sgn in ((0, -1), (k, 1)):
            base = p0 + t * (L * s / k)
            P = np.array([(base[0] + nrm[0] * u, y, base[1] + nrm[1] * u) for u, y in pts])
            tr2 = tr.copy()
            a, b, c2 = P[tr2[:, 0]], P[tr2[:, 1]], P[tr2[:, 2]]
            n2 = np.cross(b - a, c2 - a)
            fl = (n2[:, [0, 2]] @ t) * sgn < 0
            tr2[fl] = tr2[fl][:, ::-1]
            ms.append(Mesh(P, tr2, c))
    return merge(ms)


def roof_prof(w, eave, ridge, over=0.45):
    """Corbelled stone roof: two convex slopes from the eaves to the ridge, and the ridge crest."""
    h = ridge - eave
    a = w / 2 + over
    return [(-a, eave), (a, eave), (a, eave + 0.22), (a * 0.72, eave + 0.55 * h), (a * 0.40, eave + 0.86 * h),
            (0.0, ridge), (-a * 0.40, eave + 0.86 * h), (-a * 0.72, eave + 0.55 * h), (-a, eave + 0.22)]


def place_runs(p0, p1, pieces):
    return [run(p0, p1, prof, c, caps=cp) for prof, c, cp in pieces]


def gallery(p0, p1, fl, w, eave, ridge, half=0.0, half_side=1, pillars=True, plinth=None, win="both"):
    """A gallery along p0 → p1: plinth, body, window bands, corbelled roof, and on one side (half_side, +1 = to
    the left of the direction) a half-gallery: a lean-to roof on a row of square pillars."""
    ms = []
    if plinth is not None:
        ms.append(run(p0, p1, [(-w / 2 - 0.6, plinth), (w / 2 + 0.6, plinth), (w / 2 + 0.6, fl), (-w / 2 - 0.6, fl)], SAND))
    ms.append(run(p0, p1, [(-w / 2, fl), (w / 2, fl), (w / 2, eave), (-w / 2, eave)], SAND_O))
    ms.append(run(p0, p1, [(-w / 2 - 0.25, eave - 0.45), (w / 2 + 0.25, eave - 0.45), (w / 2 + 0.25, eave + 0.05),
                           (-w / 2 - 0.25, eave + 0.05)], SAND_L, caps=False))                       # cornice
    sides = (-1, 1) if win == "both" else ((half_side,) if win == "half" else (-half_side,))
    for s in sides:
        u = s * (w / 2 + 0.04)
        ms.append(run(p0, p1, [(u - 0.03, fl + 1.1), (u + 0.03, fl + 1.1), (u + 0.03, eave - 0.9), (u - 0.03, eave - 0.9)],
                      SAND_D, seg=40.0, caps=False))
    ms.append(run(p0, p1, roof_prof(w, eave, ridge), ROOF))
    if half > 0:
        s = half_side
        uo = s * (w / 2 + half)
        e2 = eave - 1.0
        ms.append(run(p0, p1, [(s * w / 2, e2 + 0.55), (uo + s * 0.4, e2 - 0.75), (uo + s * 0.4, e2 - 1.05),
                               (s * w / 2, e2 + 0.2)], ROOF_D))
        if pillars:
            p0a, p1a = np.asarray(p0, float), np.asarray(p1, float)
            L = float(np.linalg.norm(p1a - p0a)); t = (p1a - p0a) / L; nrm = np.array([-t[1], t[0]])
            n = int(L // 3.2)
            for i in range(n):
                q = p0a + t * (L * (i + 0.5) / n) + nrm * uo
                ms.append(B(q[0] - 0.3, q[0] + 0.3, fl, e2 - 0.9, q[1] - 0.3, q[1] + 0.3, SAND_L))
    return ms


def gallery_ring(x0, x1, z0, z1, fl, w, eave, ridge, half=0.0, plinth=None, win="both", skip=()):
    """Four galleries on the centre line of a rectangle; the x-runs fit between the z-runs. The half-gallery
    goes on the outside. skip: sides to leave out ('n', 's', 'e', 'w' as -x, +x, -z, +z)."""
    ms = []
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    segs = {"n": ((x0, z0 - w / 2), (x0, z1 + w / 2)), "s": ((x1, z0 - w / 2), (x1, z1 + w / 2)),
            "e": ((x0 + w / 2, z0), (x1 - w / 2, z0)), "w": ((x0 + w / 2, z1), (x1 - w / 2, z1))}
    for k, (p0, p1) in segs.items():
        if k in skip:
            continue
        t = np.subtract(p1, p0); t = t / np.linalg.norm(t); nrm = np.array([-t[1], t[0]])
        mid = (np.add(p0, p1)) / 2
        side = 1 if np.dot(nrm, mid - np.array([cx, cz])) > 0 else -1
        ms += gallery(p0, p1, fl, w, eave, ridge, half=half, half_side=side, plinth=plinth, win=win)
    return ms


def bud(t):
    t = np.clip(t, 0, 1)
    return (1 - t ** 2.1) ** 0.6 * (1 + 0.10 * np.sin(math.pi * t))


def prasat(cx, cz, y0, y1, w0, ntier=8, c=SAND, finial=True, trunc=1.0):
    """A Khmer tower: tiers of a redented plan following a lotus-bud profile, antefixes on every ledge, a lotus
    finial. trunc < 1 stops it lower (a ruined tower)."""
    ms = []
    H = y1 - y0
    body = 0.9 * H
    for i in range(ntier):
        t0, t1 = i / ntier, (i + 0.72) / ntier
        if t0 >= trunc:
            break
        wa, wb = w0 * bud(t0), w0 * bud(t1) * 0.96
        ya, yb = y0 + body * t0, y0 + body * t1
        tn = (i + 1) / ntier
        yc = y0 + body * tn
        ms.append(loft(redent(cx, cz, wa, wa * 0.24), ya, redent(cx, cz, wb, wb * 0.24), yb, c))
        wl = wb * 1.04                                         # the ledge (cornice) of the tier
        ms.append(loft(redent(cx, cz, wl, wl * 0.24), yb, redent(cx, cz, wl, wl * 0.24), yc, SAND_L if i % 2 else c,
                       top=True))
        # antefixes: little pointed stones on the corners and in the middle of every side
        a = wl * 0.92
        hh = (yc - yb) * 1.9 + 0.3
        for px, pz in ((a, a), (-a, a), (a, -a), (-a, -a), (0, wl), (0, -wl), (wl, 0), (-wl, 0)):
            r = max(0.18, wl * 0.11)
            ms.append(cyl(r, 0.0, hh, c, seg=4, caps=False).R(0, math.pi / 4, 0).T(cx + px, yc, cz + pz))
    if finial and trunc >= 1.0:
        wt = w0 * bud(0.97)
        ms.append(lathe([(wt * 0.9, y0 + body), (wt * 1.05, y0 + body + 0.3 * (H - body)), (wt * 0.5, y0 + body + 0.7 * (H - body)),
                         (0.05, y1)], SAND_L, seg=8).T(cx, 0, cz))
    elif trunc < 1.0:
        k = min(ntier, int(math.ceil(trunc * ntier)))
        wl = w0 * bud((k - 1 + 0.72) / ntier) * 0.96 * 1.04
        ms.append(cap(redent(cx, cz, wl * 0.8, wl * 0.2), y0 + body * k / ntier + 0.01, SAND_D))
    return ms


def gable_roof_cross(cx, cz, lx, lz, wx, wz, eave, ridge, c=ROOF):
    """Two crossing gabled roofs over a cross-shaped body."""
    return [run((cx - lx - 0.4, cz), (cx + lx + 0.4, cz), roof_prof(2 * wz, eave, ridge), c),
            run((cx, cz - lz - 0.4), (cx, cz + lz + 0.4), roof_prof(2 * wx, eave, ridge + 0.05), c)]


def pavilion(cx, cz, fl, lx, lz, wx, wz, eave, top, tower=True, trunc=1.0, base=None, c=SAND_O):
    """A cruciform pavilion or gopura: body, crossing roofs, and a stepped tower over the crossing."""
    ms = []
    if base is not None:
        ms.append(extrude(cross_ring(cx, cz, lx + 0.8, lz + 0.8, wx + 0.8, wz + 0.8), base, fl, SAND, top_c=SAND_L))
    ms.append(extrude(cross_ring(cx, cz, lx, lz, wx, wz), fl, eave, c, top=False))
    # doors at the four ends
    for sx, sz, dx, dz in ((1, 0, 0.05, 0), (-1, 0, -0.05, 0), (0, 1, 0, 0.05), (0, -1, 0, -0.05)):
        if sx:
            ms.append(B(cx + sx * lx + dx - 0.05, cx + sx * lx + dx + 0.05, fl, fl + min(3.4, eave - fl - 0.6),
                        cz - 1.0, cz + 1.0, SAND_D))
        else:
            ms.append(B(cx - 1.0, cx + 1.0, fl, fl + min(3.4, eave - fl - 0.6), cz + sz * lz + dz - 0.05,
                        cz + sz * lz + dz + 0.05, SAND_D))
    rr = eave + 0.55 * (wx + wz)
    ms += gable_roof_cross(cx, cz, lx, lz, wx, wz, eave, rr)
    if tower and top > rr + 1:
        w0 = max(wx, wz) * 1.15
        ms.append(extrude(redent(cx, cz, w0, w0 * 0.25), eave, rr + 0.6, SAND_O))
        ms += prasat(cx, cz, rr + 0.6, top, w0, ntier=6 if top - rr < 12 else 8, trunc=trunc)
    return ms


def stairs(cx, cz, ax, length, width, y0, y1, n=None, c=SAND):
    """A flight of steps going up from (cx, cz) along ax ('+z', '-z', '+x', '-x') over `length` metres."""
    n = n or max(3, int(abs(y1 - y0) / 0.45))
    ms = []
    for i in range(n):
        d0 = length * i / n
        yt = y0 + (y1 - y0) * (i + 1) / n
        if ax in ("+z", "-z"):
            s = 1 if ax == "+z" else -1
            za, zb = sorted((cz + s * d0, cz + s * length))
            ms.append(B(cx - width / 2, cx + width / 2, y0, yt, za, zb, c))
        else:
            s = 1 if ax == "+x" else -1
            xa, xb = sorted((cx + s * d0, cx + s * length))
            ms.append(B(xa, xb, y0, yt, cz - width / 2, cz + width / 2, c))
    return ms


def basin_ring(ring_outer, ring_inner, y, c):
    return cap(ring_outer, y, c, holes=[ring_inner])


# ── The moat and the island ─────────────────────────────────────────────
def band(h0, y0, h1, y1, c, seg=25.0, ext=(IX, IZ0, IZ1)):
    """A band round the rectangle of the island grown by h0 (at height y0) and by h1 (at y1): a step tread when
    y0 == y1, a vertical riser when h0 == h1, a slope otherwise; cut every `seg` metres along the sides."""
    X, Z0, Z1 = ext
    def R(h):
        return [(-X - h, Z0 - h), (X + h, Z0 - h), (X + h, Z1 + h), (-X - h, Z1 + h)]
    A, Bq = R(h0), R(h1)
    v, f = [], []
    for i in range(4):
        a0, a1, b0, b1 = np.array(A[i]), np.array(A[(i + 1) % 4]), np.array(Bq[i]), np.array(Bq[(i + 1) % 4])
        k = max(1, int(math.ceil(np.linalg.norm(a1 - a0) / seg)))
        o = len(v)
        for j in range(k + 1):
            t = j / k
            pa, pb = a0 + (a1 - a0) * t, b0 + (b1 - b0) * t
            v += [(pa[0], y0, pa[1]), (pb[0], y1, pb[1])]
        for j in range(k):
            q = o + 2 * j
            f += [(q, q + 1, q + 3), (q, q + 3, q + 2)]
    m = Mesh(v, f, c)
    # turn the faces up (treads, slopes) or away from the band's middle (risers)
    V, F = m.v, m.f
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if abs(h0 - h1) > 1e-6:
        flip = nrm[:, 1] < 0
    else:
        cen = V[F].mean(1)
        outward = np.stack([np.sign(cen[:, 0]) * (np.abs(cen[:, 0]) > X + h0 - 0.01),
                            np.zeros(len(F)),
                            ((cen[:, 2] > Z1 + h0 - 0.01) * 1.0 - (cen[:, 2] < Z0 - h0 + 0.01) * 1.0)], 1)
        flip = np.einsum("ij,ij->i", nrm, outward) < 0
        if y1 < y0:
            flip = ~flip
    F[flip] = F[flip][:, ::-1]
    return m


def moat():
    banks, water = [], []
    # water: strips round the island, cut in 40 m cells
    for (x0, x1, z0, z1) in ((-MX, MX, IZ1, MZ1), (-MX, MX, MZ0, IZ0), (-MX, -IX, IZ0, IZ1), (IX, MX, IZ0, IZ1)):
        from meshkit import grid
        nx, nz = max(1, int(round((x1 - x0) / 40))), max(1, int(round((z1 - z0) / 40)))
        water.append(grid(x0, x1, z0, z1, nx, nz, lambda X_, Z_: np.full_like(X_, WL), WATER))
    # inner bank: a sandstone kerb and three laterite steps from the island down to the water
    banks += [band(0.0, -0.8, 0.0, G, SAND), band(0.0, G - 0.55, 1.6, G - 0.55, LAT_D),
              band(1.6, -0.8, 1.6, G - 0.55, LAT), band(1.6, G - 1.05, 3.0, G - 1.05, LAT),
              band(3.0, -0.8, 3.0, G - 1.05, LAT)]
    # outer bank: the same steps facing the island, a kerb walk and a grass slope down to the plain
    O = (MX, MZ0, MZ1)
    banks += [band(0.0, -0.8, 0.0, G, SAND, ext=O), band(-1.6, G - 0.55, 0.0, G - 0.55, LAT_D, ext=O),
              band(-1.6, -0.8, -1.6, G - 0.55, LAT, ext=O), band(-3.0, G - 1.05, -1.6, G - 1.05, LAT, ext=O),
              band(-3.0, -0.8, -3.0, G - 1.05, LAT, ext=O), band(0.0, G, 3.5, G, SAND_L, ext=O),
              band(3.5, G, 14.0, -0.4, DRY, ext=O)]
    return banks, water


def island_ground():
    """The lawn of the island at y = G, a height field cut in 16 m cells so that it can be painted: lawn, the
    shade of the woods, worn earth along the causeway and round the temple terrace."""
    from meshkit import grid
    m = grid(-IX, IX, IZ0, IZ1, 56, 70, lambda X, Z: np.full_like(X, G), LAWN)
    p = m.v
    n = fbm(p, 0.02, 31, 3)
    col = np.array(LAWN)[None, :] * (1 - n[:, None] * 0.35) + np.array(DRY)[None, :] * (n[:, None] * 0.35)
    wood = ((np.abs(p[:, 0]) > 160) & (np.abs(p[:, 0]) < WX - 4) & (p[:, 2] > WZ0 + 4) & (p[:, 2] < WZ1 - 4)) | \
           ((p[:, 2] < -175) & (np.abs(p[:, 0]) < WX - 4) & (p[:, 2] > WZ0 + 4))
    col[wood] = col[wood] * 0.55 + np.array(LAWN_D) * 0.45
    path = (np.abs(p[:, 0]) < 14) & (p[:, 2] > 190) & (p[:, 2] < WZ1)
    col[path] = col[path] * 0.3 + np.array(EARTH) * 0.7
    m.c = col
    # the apron of the moat (outside the wall): mown lawn edge, a ring of worn earth along the wall
    return [m]


def trees():
    """Simple trees in the woods of the enclosure (tall: the dipterocarps of Angkor reach 30–40 m)."""
    ms = []
    sp = 23.0
    for x in np.arange(-WX + 12, WX - 8, sp):
        for z in np.arange(WZ0 + 12, WZ1 - 10, sp):
            px, pz = x + rng.uniform(-8, 8), z + rng.uniform(-8, 8)
            if abs(px) < 150 and -160 < pz < 215:
                continue                                     # the temple terrace and around it
            if abs(px) < 22 and pz > 150:
                continue                                     # the processional causeway
            if abs(px) < 130 and 200 < pz < 400:
                p = 0.10                                     # the open lawns in front of the temple
            elif abs(px) < 140 and pz >= 400:
                p = 0.25
            elif abs(px) < 30 and pz < WZ0 + 40:
                continue                                     # the east gopura
            elif abs(abs(px) - WX) < 30 and abs(pz) < 40:
                continue                                     # the north and south gopuras
            else:
                p = 0.72
            if rng.random() > p:
                continue
            h = rng.uniform(14, 30); r = rng.uniform(4.5, 8.5)
            c = TREES[rng.integers(len(TREES))]
            ms.append(cyl(0.5, 0.35, h * 0.5, (0.36, 0.30, 0.22), seg=3, caps=False).T(px, G, pz))
            ms.append(lathe([(0.0, h * 0.45), (r, h * 0.62), (r * 0.75, h * 0.88), (0.0, h)], c, seg=6)
                      .R(0, rng.uniform(0, 1), 0).S(1, 1, rng.uniform(0.8, 1.2)).T(px, G, pz))
    return ms


# ── Enclosure wall and gopuras ─────────────────────────────────────────
def outer_wall():
    ms = []
    gaps = {"w": [(-118, 118)], "e": [(-12, 18)], "n": [(-16, 21)], "s": [(-20, 14)]}
    t = WALL_T / 2

    def wall_x(z, a, b):                                    # along x
        ms.append(run((a, z), (b, z), [(-t, -0.5), (t, -0.5), (t, WALL_H), (-t, WALL_H)], LAT, seg=12.0))
        ms.append(run((a, z), (b, z), [(-t - 0.15, WALL_H), (t + 0.15, WALL_H), (t + 0.15, WALL_H + 0.45),
                                       (-t - 0.15, WALL_H + 0.45)], SAND, seg=12.0))

    def wall_z(x, a, b):
        ms.append(run((x, a), (x, b), [(-t, -0.5), (t, -0.5), (t, WALL_H), (-t, WALL_H)], LAT, seg=12.0))
        ms.append(run((x, a), (x, b), [(-t - 0.15, WALL_H), (t + 0.15, WALL_H), (t + 0.15, WALL_H + 0.45),
                                       (-t - 0.15, WALL_H + 0.45)], SAND, seg=12.0))

    def split(lo, hi, gaps_):
        out, cur = [], lo
        for a, b in sorted(gaps_):
            out.append((cur, a)); cur = b
        out.append((cur, hi))
        return out

    for a, b in split(-WX - t, WX + t, gaps["w"]):
        wall_x(WZ1, a, b)
    for a, b in split(-WX - t, WX + t, [(-10, 15)]):
        wall_x(WZ0, a, b)
    for a, b in split(WZ0 + t, WZ1 - t, [(-16, 21)]):
        wall_z(-WX, a, b)
    for a, b in split(WZ0 + t, WZ1 - t, [(-21, 14)]):
        wall_z(WX, a, b)
    return ms


def west_gopura():
    """The west entrance: a 235 m long gallery with a pillared front, three pavilions with ruined towers and the
    two «elephant gates» at the ends (OSM way 1092292990)."""
    ms = []
    zc = 554.5
    fl = 1.1
    ms.append(B(-118, 118, -0.5, fl, zc - 5.6, zc + 6.8, SAND))                     # base
    ms += gallery((-117.0, zc), (117.0, zc), fl, 7.0, 5.6, 8.2, half=2.8, half_side=1, win="both")
    for xp in (-117.0, 117.0):                                                       # elephant gates
        ms += pavilion(math.copysign(106.5, xp), zc, fl, 3.6, 12.0, 3.2, 4.0, 6.0, 0, tower=False, base=-0.3)
    for xp in (-72.0, 72.0):
        ms.append(B(xp - 2.5, xp + 2.5, fl, 5.2, zc - 4.3, zc + 7.6, SAND_O))
        ms += gable_roof_cross(xp, zc, 2.5, 4.5, 2.5, 2.5, 5.2, 7.0)
    ms += pavilion(0.0, zc, fl, 9.0, 19.0, 4.8, 5.2, 7.2, 20.0, trunc=0.55, base=-0.3)
    for xp in (-31.0, 31.0):
        ms += pavilion(xp, zc, fl, 6.5, 12.0, 3.6, 4.2, 6.6, 16.0, trunc=0.5, base=-0.3)
    # steps to the causeway across the moat, and to the processional causeway
    ms += stairs(0.0, zc + 19.0, "-z", 3.0, 8.0, 0.0, fl, 3)
    return ms


def side_gopura(cx, cz, along_x):
    """The smaller gopuras of the north, south and east sides: cruciform, a low tower (ruined top)."""
    if along_x:
        return pavilion(cx, cz, 0.9, 23.0, 8.0, 3.6, 4.4, 5.8, 13.0, trunc=0.5, base=-0.3)
    return pavilion(cx, cz, 0.9, 8.0, 21.0, 4.4, 3.6, 5.8, 13.0, trunc=0.5, base=-0.3)


# ── Causeways across the moat and the processional causeway ─────────────
def naga_rail(p0, p1, y, heads=(True, True)):
    """A naga balustrade: the serpent's body on short posts, the raised multi-headed hood at its ends."""
    ms = [tube([(p0[0], y + 0.85, p0[1]), (p1[0], y + 0.85, p1[1])], 0.32, SAND, seg=6)]
    d = math.dist(p0, p1)
    n = max(2, int(d // 2.6))
    for i in range(n + 1):
        x = p0[0] + (p1[0] - p0[0]) * i / n; z = p0[1] + (p1[1] - p0[1]) * i / n
        ms.append(B(x - 0.22, x + 0.22, y, y + 0.7, z - 0.22, z + 0.22, SAND_L))
    for end, (q, o) in enumerate(((p0, p1), (p1, p0))):
        if not heads[end]:
            continue
        dz = np.subtract(q, o); dz = dz / np.linalg.norm(dz)
        ang = math.atan2(dz[0], dz[1])
        hood = []
        hood.append(tube([(0, y + 0.85, -0.6), (0, y + 1.6, 0.3), (0, y + 2.4, 0.5)], 0.38, SAND, seg=6))
        prof = [(0.0, y + 1.6)] + [(1.25 * math.sin(math.pi * (k / 8 - 0.5)), y + 1.6 + 1.6 * math.cos(math.pi * (k / 8 - 0.5)))
                                   for k in range(9)]
        hood.append(run((0.0, 0.35), (0.0, 0.65), prof, SAND))                   # the fan of seven heads
        for h in hood:
            ms.append(h.R(0, ang, 0).T(q[0], 0, q[1]))
    return ms


def west_causeway():
    """The sandstone causeway across the moat (OSM way 471925578): 12.4 m wide, a stair landing on both sides in the
    middle and the cruciform lion terrace at its west end."""
    ms = []
    y = G + 0.5
    z0, z1 = IZ1 - 2.0, MZ1 + 1.0
    ms.append(run((0.0, z0), (0.0, z1), [(-6.2, -0.8), (6.2, -0.8), (6.2, y), (-6.2, y)], SAND, seg=6.0))
    ms.append(run((0.0, z0), (0.0, z1), [(-6.6, -0.8), (6.6, -0.8), (6.6, WL + 0.6), (-6.6, WL + 0.6)], LAT, seg=12.0))                            # laterite footing in the water
    for s in (-1, 1):
        ms.append(B(min(s * 6.2, s * 17.0), max(s * 6.2, s * 17.0), -0.8, y - 0.2, 685.0, 696.0, SAND))
        ms += stairs(s * 17.0, 690.5, "-x" if s > 0 else "+x", 0.01, 11.0, WL, WL, 1)
        for zz in (z0, 685.0), (696.0, z1):
            ms.append(B(s * 6.0 - 0.3, s * 6.0 + 0.3, y, y + 0.9, zz[0], zz[1], SAND_L))   # low balustrade
    ms.append(extrude(cross_ring(0.0, 797.5, 12.5, 9.3, 7.3, 5.2), -0.6, y, SAND, top_c=SAND_L))
    ms += stairs(0.0, 806.8, "-z", 2.4, 8.0, 0.0, y, 4)
    for s in (-1, 1):                                                                # the lions of the terrace
        ms.append(B(s * 5.6 - 0.6, s * 5.6 + 0.6, y, y + 1.0, 800.5, 802.5, SAND_L))
        ms.append(B(s * 5.6 - 0.45, s * 5.6 + 0.45, y + 1.0, y + 2.0, 801.7, 802.6, SAND))
    return ms


def east_causeway():
    """The earthen causeway from the east (OSM way 1462276418, about 56 m wide)."""
    xa, xb = -27.0, 37.5
    ms = [loft(rect(xa - 4, xb + 4, MZ0 - 2, IZ0 + 2), -0.8, rect(xa, xb, MZ0 - 2, IZ0 + 2), G, LAT_D, top=False),
          cap(rect(xa, xb, MZ0 - 2, IZ0 + 2), G, LAWN)]
    ms.append(cap(rect(-4.0, 9.0, MZ0 - 2, IZ0 + 2), G + 0.03, EARTH))
    return ms


def processional():
    """The raised causeway from the west gopura to the Terrace of Honour (350 m long, 9.5 m wide), its naga
    balustrades, side landings, the two libraries and the two ponds."""
    ms, water = [], []
    z0, z1 = 186.0, 549.0
    w = 4.75
    ms.append(run((0.0, z0), (0.0, z1), [(-w, -0.4), (w, -0.4), (w, H_CW), (-w, H_CW)], SAND, seg=6.0))
    ms.append(run((0.0, z0), (0.0, z1), [(-w + 0.4, H_CW), (w - 0.4, H_CW), (w - 0.4, H_CW + 0.02), (-w + 0.4, H_CW + 0.02)],
                  SAND_L, seg=6.0))
    lands = [480.0, 425.0, 370.0, 315.0, 260.0]
    for s in (-1, 1):
        cuts = [z1 - 1.0] + [v for zl in lands for v in (zl + 2.6, zl - 2.6)] + [z0 + 2.0]
        for i in range(0, len(cuts) - 1, 2):
            ms += naga_rail((s * (w - 0.35), cuts[i]), (s * (w - 0.35), cuts[i + 1]), H_CW,
                            heads=(i > 0, True))
        for zl in lands:                                                            # side landings with steps
            ms.append(B(min(s * w, s * (w + 4.2)), max(s * w, s * (w + 4.2)), -0.3, H_CW - 0.05, zl - 2.4, zl + 2.4, SAND))
            ms += stairs(s * (w + 4.2), zl, "-x" if s > 0 else "+x", 2.2, 4.0, 0.0, H_CW - 0.05, 3)
    # the outer libraries (OSM ways 240011220, 39842240): cruciform, on a base, with gabled roofs
    for s in (-1, 1):
        cx, cz = s * 32.0, 369.0
        ms.append(extrude(cross_ring(cx, cz, 13.6, 19.2, 5.5, 5.4), -0.3, 1.9, SAND, top_c=SAND_L))
        ms.append(extrude(cross_ring(cx, cz, 11.8, 17.2, 4.2, 4.0), 1.9, 6.0, SAND_O, top=False))
        ms += gable_roof_cross(cx, cz, 11.8, 17.2, 4.2, 4.0, 6.0, 9.2)
        ms += prasat(cx, cz, 9.0, 12.5, 2.8, ntier=3)
        ms += stairs(cx, cz + 19.2 + 2.6, "-z", 2.6, 3.0, 0.0, 1.9, 4)
        ms += stairs(cx - s * 13.6 - s * 2.6, cz, "+x" if s > 0 else "-x", 2.6, 3.0, 0.0, 1.9, 4)
    # the ponds (OSM ways 39842237/8): stone rims and water
    for s in (-1, 1):
        x0, x1 = (30.0, 80.5) if s > 0 else (-80.5, -30.0)
        za, zb = 218.0, 294.5
        ms.append(walls(rect(x0, x1, za, zb), -0.2, 0.35, LAT))
        ms.append(cap(rect(x0, x1, za, zb), 0.35, SAND, holes=[rect(x0 + 2.0, x1 - 2.0, za + 2.0, zb - 2.0)]))
        ms.append(walls(rect(x0 + 2.0, x1 - 2.0, za + 2.0, zb - 2.0), -0.6, 0.35, LAT_D, hole=True))
        water.append(cap(rect(x0 + 2.0, x1 - 2.0, za + 2.0, zb - 2.0), 0.12, WATER))
    return ms, water


# ── The temple ───────────────────────────────────────────────────────────
def terrace():
    """The temple terrace (OSM retaining walls 446335238 …) and the cruciform Terrace of Honour."""
    ms = []
    ms.append(extrude(rect(-133.0, 133.0, -145.0, 200.0), -0.6, H_PL, LAT, top_c=TERR))
    ms.append(walls(rect(-133.3, 133.3, -145.3, 200.3), H_PL - 0.35, H_PL + 0.05, SAND))
    ms += stairs(0.0, 200.0 + 3.0, "-z", 3.0, 9.5, 0.0, H_PL, 3)
    # stairs at the middle of the other sides
    ms += stairs(0.0, -145.0 - 3.0, "+z", 3.0, 9.0, 0.0, H_PL, 3)
    for s in (-1, 1):
        ms += stairs(s * 136.0, 8.0, "-x" if s > 0 else "+x", 3.0, 9.0, 0.0, H_PL, 3)
    # Terrace of Honour (OSM 642066967, 513685091–94): a lower cross on short columns and an upper cross
    lo = cross_ring(0.0, 159.7, 22.7, 28.4, 6.5, 6.4)
    hi = cross_ring(0.0, 156.5, 10.3, 18.3, 4.0, 3.9)
    ms.append(extrude(lo, H_PL, 1.9, SAND_D, top=False))
    ms.append(extrude(lo, 1.9, 2.25, SAND, top_c=SAND_L))
    ms.append(extrude(hi, 2.25, 2.6, SAND_O, top_c=SAND_L))
    for s in (-1, 1):
        ms += stairs(s * 22.7 + s * 2.2, 159.7, "-x" if s > 0 else "+x", 2.2, 4.5, H_PL, 2.25, 3)
    ms += naga_rail((-6.2, 186.0), (-6.2, 144.0), 2.25, heads=(True, False))
    ms += naga_rail((6.2, 186.0), (6.2, 144.0), 2.25, heads=(True, False))
    ms.append(B(-3.2, 3.2, H_PL, 2.25, 186.0, 189.0, SAND))
    # steps from the Terrace of Honour up to the west gopura of the first gallery
    ms += stairs(0.0, 131.0, "-z", 4.2, 8.0, 2.25, F1, 4)
    return ms


def first_gallery():
    ms = []
    # the block of the first enclosure: from the terrace to the first court
    ms.append(extrude(rect(-89.5, 89.5, -92.5, 119.5), 0.0, H_C1, LAT, top_c=COURT))
    ms.append(walls(rect(-90.2, 90.2, -93.2, 120.2), H_PL - 0.2, H_C1 - 0.4, SAND))          # moulded plinth
    ms.append(walls(rect(-89.9, 89.9, -92.9, 119.9), H_C1 - 0.4, H_C1, SAND_L))
    # galleries: centre line of a 5.2 m gallery, a 2.6 m half-gallery outside (outer face 178 × 211 m)
    xg, z0, z1 = 84.0, -87.0, 114.0
    ms += gallery_ring(-xg, xg, z0, z1, F1, 5.2, E1, R1, half=2.6, plinth=H_C1)
    # corner pavilions (cruciform, stepped roofs)
    for sx in (-1, 1):
        for zc in (z0, z1):
            ms += pavilion(sx * xg, zc, F1, 9.5, 9.5, 3.4, 3.4, E1 + 0.6, 17.0, trunc=1.0, base=H_C1)
    # gopuras of the north, south and east sides
    for sx in (-1, 1):
        ms += pavilion(sx * xg, 13.5, F1, 9.0, 8.0, 3.8, 3.6, E1 + 0.9, 16.5, base=H_C1)
        ms += stairs(sx * (xg + 9.0), 13.5, "-x" if sx > 0 else "+x", 3.0, 5.0, H_PL, F1, 6)
    ms += pavilion(0.0, z0, F1, 8.0, 9.0, 3.6, 3.8, E1 + 0.9, 16.5, base=H_C1)
    ms += stairs(0.0, z0 - 9.0, "+z", 3.0, 5.0, H_PL, F1, 6)
    # the west gopura: three pavilions with towers, the central one the largest
    ms += pavilion(0.0, z1, F1, 8.5, 11.0, 4.6, 4.2, E1 + 1.4, 22.0, base=H_C1, trunc=0.8)
    for sx in (-1, 1):
        ms += pavilion(sx * 17.5, z1, F1, 5.0, 8.5, 3.2, 3.6, E1 + 1.0, 17.5, base=H_C1, trunc=0.75)
    return ms


def cloister():
    """Preah Poan, the cruciform cloister between the first and second galleries (OSM way 888569856): galleries on
    a 田 plan round four basins."""
    ms = []
    fl, e, r = F1 + 0.6, E1 + 0.6, R1 + 0.8
    za, zb = 72.0, 111.5
    for x in (-24.4, 0.0, 24.4):
        ms += gallery((x, za), (x, zb), fl, 6.0 if x == 0 else 5.0, e, r, plinth=H_C1, win="both")
    ms += gallery((-34.5, 89.0), (34.5, 89.0), fl, 6.0, e, r, plinth=H_C1, win="both")
    ms += pavilion(0.0, 89.0, fl, 5.0, 5.0, 3.3, 3.3, e + 0.6, r + 4.0, tower=False)
    for sx in (-1, 1):
        for zc, (z0b, z1b) in ((95.0, (93.0, 104.6)), (80.0, (73.8, 85.4))):
            x0b, x1b = sorted((sx * 3.5, sx * 21.6))
            ms.append(walls(rect(x0b, x1b, z0b, z1b), 1.6, H_C1 + 0.02, SAND, hole=True))
            ms.append(cap(rect(x0b, x1b, z0b, z1b), 1.6, SAND_D))
            ms.append(cap(rect(x0b, x1b, z0b, z1b), H_C1 + 0.03, SAND_L, holes=[rect(x0b + 0.8, x1b - 0.8, z0b + 0.8, z1b - 0.8)]))
    return ms


def first_court_libraries():
    """The two libraries of the first court (OSM ways 120320892/3)."""
    ms = []
    for s in (-1, 1):
        cx, cz = s * 66.0, 89.0
        ms.append(extrude(cross_ring(cx, cz, 10.0, 17.0, 4.6, 4.4), H_C1, 5.0, SAND, top_c=SAND_L))
        ms.append(extrude(cross_ring(cx, cz, 8.4, 15.2, 3.5, 3.4), 5.0, 8.8, SAND_O, top=False))
        ms += gable_roof_cross(cx, cz, 8.4, 15.2, 3.5, 3.4, 8.8, 11.6)
        ms += stairs(cx, cz + 17.0 + 2.2, "-z", 2.2, 2.6, H_C1, 5.0, 4)
    return ms


def second_gallery():
    ms = []
    x0, x1, z0, z1 = -51.0, 51.0, -53.0, 65.0
    # the high moulded base from the first court to the second level
    ms.append(extrude(rect(x0 - 1.2, x1 + 1.2, z0 - 1.2, z1 + 1.2), H_C1 - 0.2, H_C1 + 1.4, SAND))
    ms.append(loft(rect(x0 - 0.9, x1 + 0.9, z0 - 0.9, z1 + 0.9), H_C1 + 1.4, rect(x0 - 0.3, x1 + 0.3, z0 - 0.3, z1 + 0.3), H_C1 + 2.4, SAND_L))
    ms.append(extrude(rect(x0 - 0.3, x1 + 0.3, z0 - 0.3, z1 + 0.3), H_C1 + 2.4, H_C2, SAND, top_c=SAND_L))
    w = 6.0
    ms += gallery_ring(x0 + w / 2, x1 - w / 2, z0 + w / 2, z1 - w / 2, F2, w, E2, R2, plinth=H_C2, win="half")
    # corner towers (low, partly ruined)
    for sx in (-1, 1):
        for zc in (z0 + w / 2, z1 - w / 2):
            cx = sx * (x1 - w / 2)
            ms += pavilion(cx, zc, F2, 6.6, 6.6, 3.6, 3.6, E2 + 0.4, 27.0, base=H_C2)
    # axial gopuras
    for sx in (-1, 1):
        ms += pavilion(sx * (x1 - w / 2), 6.0, F2, 6.5, 6.5, 3.8, 3.4, E2 + 0.8, 22.0, base=H_C2)
        ms += stairs(sx * (x1 + 7.0), 6.0, "-x" if sx > 0 else "+x", 7.0, 4.0, H_C1, F2, 14)
    ms += pavilion(0.0, z0 + w / 2, F2, 6.5, 6.5, 3.4, 3.8, E2 + 0.8, 22.0, base=H_C2)
    ms += stairs(0.0, z0 - 7.0, "+z", 7.0, 4.0, H_C1, F2, 14)
    ms += pavilion(0.0, z1 - w / 2, F2, 8.0, 7.0, 4.2, 3.8, E2 + 1.0, 23.0, base=H_C2)
    # the west gopura is reached through the cloister: steps from its galleries
    ms += stairs(0.0, 72.0, "-z", 3.0, 5.0, F1 + 0.6, F2, 12)
    # the small libraries of the second court (OSM ways 120324355/7)
    for s in (-1, 1):
        cx, cz = s * 14.5, 48.0
        ms.append(extrude(cross_ring(cx, cz, 6.3, 9.1, 3.0, 2.9), H_C2, H_C2 + 1.2, SAND, top_c=SAND_L))
        ms.append(extrude(cross_ring(cx, cz, 5.0, 7.8, 2.4, 2.3), H_C2 + 1.2, H_C2 + 4.0, SAND_O, top=False))
        ms += gable_roof_cross(cx, cz, 5.0, 7.8, 2.4, 2.3, H_C2 + 4.0, H_C2 + 6.2)
    # raised walk from the west gopura of the second gallery to the Bakan (OSM way 888649878)
    ms.append(B(-7.5, 7.5, H_C2, H_C2 + 1.0, 36.0, z1 - w, SAND))
    return ms


def bakan():
    """The third level: the base from the second court to 22.5 m, its twelve steep stairs, the galleries, the cross
    galleries, the four corner towers and the central sanctuary."""
    ms = []
    h = 30.0
    # base in three moulded tiers
    tiers = [(H_C2, 13.5, 36.0), (13.5, 17.5, 34.0), (17.5, F3 - 0.4, 32.0)]
    for y0, y1, a in tiers:
        ms.append(extrude(rect(-a, a, -a, a), y0, y1, SAND))
        ms.append(loft(rect(-a, a, -a, a), y1, rect(-a + 0.9, a - 0.9, -a + 0.9, a - 0.9), y1 + 0.35, SAND_L, top=True))
    ms.append(extrude(rect(-31.0, 31.0, -31.0, 31.0), F3 - 0.4, F3, SAND_L))
    # twelve stairs: three on each side (axis and corners), 70° steep
    for ax in ("+z", "-z", "+x", "-x"):
        for off in (-25.0, 0.0, 25.0):
            wd = 3.6 if off == 0 else 2.8
            if ax in ("+z", "-z"):
                s = 1 if ax == "+z" else -1
                ms += stairs(off, s * 40.5, "-z" if s > 0 else "+z", 9.0, wd, H_C2, F3, 22)
            else:
                s = 1 if ax == "+x" else -1
                ms += stairs(s * 40.5, off, "-x" if s > 0 else "+x", 9.0, wd, H_C2, F3, 22)
    # galleries round the Bakan and the cross galleries to the central sanctuary
    a, w = 27.4, 5.2
    ms += gallery_ring(-a, a, -a, a, F3, w, E3, R3, win="both")
    for p0, p1 in (((-a + w / 2, 0.0), (-9.8, 0.0)), ((9.8, 0.0), (a - w / 2, 0.0)), ((0.0, -a + w / 2), (0.0, -9.8)),
                   ((0.0, 9.8), (0.0, a - w / 2))):
        ms += gallery(p0, p1, F3, w, E3, R3, win="both")
    for cx, cz in ((a, 0.0), (-a, 0.0), (0.0, a), (0.0, -a)):                     # axial porches of the gallery
        ms += pavilion(cx, cz, F3, 4.6 if cz == 0 else 3.4, 4.6 if cx == 0 else 3.4, 3.0, 3.0, E3 + 0.7, 0, tower=False)
    # four courtyards (dry basins)
    for sx in (-1, 1):
        for sz in (-1, 1):
            x0, x1 = sorted((sx * (w / 2), sx * (a - w / 2)))
            z0, z1 = sorted((sz * (w / 2), sz * (a - w / 2)))
            ms.append(cap(rect(x0, x1, z0, z1), F3 + 0.02, SAND_D))
    # corner towers on the corners of the gallery (OSM 886790393 …: 13 m at the base)
    for sx in (-1, 1):
        for sz in (-1, 1):
            cx, cz = sx * 25.0, sz * 24.5
            ms.append(extrude(redent(cx, cz, 6.6, 1.5), F3, R3 + 0.4, SAND_O))
            ms.append(loft(redent(cx, cz, 6.6, 1.5), R3 + 0.4, redent(cx, cz, 6.9, 1.6), R3 + 1.0, SAND_L, top=True))
            ms += prasat(cx, cz, R3 + 1.0, TOP_K, 6.2, ntier=8)
    # central sanctuary: cella with four porches, then the tower to 65 m
    cw = 9.7
    ms.append(extrude(redent(0, 0, cw, 2.0), F3, 35.0, SAND_O))
    ms.append(loft(redent(0, 0, cw, 2.0), 35.0, redent(0, 0, cw + 0.5, 2.1), 35.8, SAND_L, top=True))
    for ang in (0, 90, 180, 270):
        c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        px, pz = s * (cw + 2.0), c * (cw + 2.0)
        dx, dz = (3.4, 2.4) if s == 0 else (2.4, 3.4)
        ms.append(B(px - (2.4 if c else 3.0), px + (2.4 if c else 3.0), F3, 31.5, pz - (3.0 if c else 2.4), pz + (3.0 if c else 2.4), SAND_O))
        ms.append(B(px - 1.2 if c else px - 0.1 + s * 3.0, px + 1.2 if c else px + 0.1 + s * 3.0, F3, F3 + 3.6,
                    pz - 0.1 + c * 3.0 if c else pz - 1.2, pz + 0.1 + c * 3.0 if c else pz + 1.2, SAND_D))
        # stepped pediment roofs of the porches
        if c:
            ms.append(run((px, pz - c * 3.0), (px, pz + c * 3.4), roof_prof(4.8, 31.5, 34.5), ROOF))
        else:
            ms.append(run((px - s * 3.0, pz), (px + s * 3.4, pz), roof_prof(4.8, 31.5, 34.5), ROOF))
    ms += prasat(0.0, 0.0, 35.8, TOP_C, 9.6, ntier=10)
    return ms


# ── Build ────────────────────────────────────────────────────────────────
def subdiv(m, L=40.0):
    """Split the long triangles (moat, banks, walls, causeways) by their longest edge until every edge is shorter
    than L: the vertex colours then vary along them, and no triangle is a kilometre long."""
    V, F, C = [*m.v], m.f.tolist(), [*m.c]
    out = []
    while F:
        nxt = []
        for a, b, c in F:
            pa, pb, pc = V[a], V[b], V[c]
            e = [np.linalg.norm(pb - pa), np.linalg.norm(pc - pb), np.linalg.norm(pa - pc)]
            k = int(np.argmax(e))
            if e[k] <= L:
                out.append((a, b, c)); continue
            i, j, o = ((a, b, c), (b, c, a), (c, a, b))[k]
            V.append((V[i] + V[j]) / 2); C.append((C[i] + C[j]) / 2)
            n = len(V) - 1
            nxt += [(i, n, o), (n, j, o)]
        F = nxt
    return Mesh(np.array(V), np.array(out), np.array(C))


def weather(m, seed, amount=0.5):
    """Grey-ochre stone with patches, dark lichen streaks running down, darker low down."""
    p = m.v
    k = 0.88 + 0.22 * fbm(p, 0.11, seed, 3)
    st = sstep(0.50, 0.74, fbm(p, (0.30, 0.035, 0.30), seed + 5, 3))
    pa = sstep(0.55, 0.80, fbm(p, 0.045, seed + 9, 2))
    a = np.clip(amount * (0.65 * st + 0.45 * pa), 0, 0.75)[:, None]
    dark = np.array([0.25, 0.25, 0.22])
    m.c = np.clip(m.c * k[:, None] * (1 - a) + dark * a, 0, 1)
    return m


def build():
    banks, water = moat()
    ground = island_ground()
    tr = merge(trees())
    encl = outer_wall() + west_gopura()
    encl += side_gopura(-WX, 2.3, False) + side_gopura(WX, -4.0, False) + side_gopura(2.5, WZ0 - 3.0, True)
    proc, pond_w = processional()
    temple = terrace() + first_gallery() + cloister() + first_court_libraries() + second_gallery() + bakan()

    def lift(ms):
        return merge([m.copy().T(0, G, 0) for m in ms])

    m_encl = weather(lift(encl), 3, 0.75)
    m_proc = weather(lift(proc), 5, 0.7)
    m_temple = weather(lift(temple), 7, 0.85)
    m_cw = weather(merge(west_causeway()), 11, 0.45)
    m_banks = weather(merge(banks + [subdiv(m, 30.0) for m in east_causeway()]), 13, 0.35)
    m_water = merge(water + [subdiv(m.copy().T(0, G, 0), 20.0) for m in pond_w])
    return [("temple", m_temple), ("enclosure", m_encl), ("processional_causeway", m_proc),
            ("west_causeway", m_cw), ("moat_banks", m_banks), ("island", merge(ground)), ("trees", tr),
            ("water", m_water, {"material": "water"})]


FOOTPRINT = [(-MX - 15, MZ0 - 15), (MX + 15, MZ0 - 15), (MX + 15, MZ1 + 15), (16.0, MZ1 + 15), (16.0, 812.0),
             (-16.0, 812.0), (-16.0, MZ1 + 15), (-MX - 15, MZ1 + 15)]


def main():
    parts = build()
    out_dir = ROOT / "models" / "Q43473"
    out_dir.mkdir(parents=True, exist_ok=True)
    named = []
    for p in parts:
        m = p[1]
        used = np.unique(m.f)
        remap = np.full(len(m.v), -1); remap[used] = np.arange(len(used))
        named.append((p[0], Mesh(m.v[used], remap[m.f], m.c[used])) + tuple(p[2:]))
    tris, size = write_glb(out_dir / "model.glb", named, roughness=0.9,
                           materials={"water": {"metallic": 0.0, "roughness": 0.15}})
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        mj["footprint"] = [[round(x, 1), round(z, 1)] for x, z in FOOTPRINT]
        meta.write_bytes((json.dumps(mj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    for p in named:
        print(f"  {p[0]:24s} {len(p[1].f):7d}")
    print(f"{tris} triangles, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
