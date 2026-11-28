"""
Palacio Real de El Pardo (Q1368571), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: the Habsburg palace begun in 1547 for Charles V to Luis de Vega's design (died 1562), a square
alcázar with four corner towers and an arcaded courtyard, slate roofs put on under Philip II by Juan Bautista de
Toledo (died 1567), rebuilt after the fire of 1604 by Francisco de Mora (died 1610); towers lowered and given
mansard caps by François Carlier (1739–1743); doubled to the east by Francesco Sabatini for Charles III
(1772–1773, died 1797): a second, identical block round the Patio de los Borbones, joined to the old one by a
central body round the Patio Central. Public domain.

What is modelled:
- The whole palace: the ring of two-storey ranges round the three courtyards (Austrias to the west, Central,
  Borbones to the east), all three left open with their paved floors, as asked for Wiki Flight (today they are
  covered by iron-and-glass roofs of 1983 and 2001–2002); arcades on the ground floor of the two big courtyards.
- Cream render with white trim (pilaster strips at the corners, string course, cornice, window frames with
  hoods), granite plinth, white shutters or dark glass, iron balconies on the main floor, the window rhythm
  (10 bays on each half of the long fronts, 3.7 m apart).
- Slate mansard roofs with one dormer per bay (with its little finial) and tall brick chimneys.
- The four corner towers (four floors, 17 m to the cornice) with their slate bell caps, the oculus dormers on
  each side and the finial (24.4 m); the lower two-bay bodies beside them on the long fronts.
- The two central pavilions (Sabatini) with five bays, the attic with oculi, the pediment with the royal arms
  and the hipped roof; the gate of the south front with its two sentry boxes.
- The Habsburg portal of the west front (Charles V, 1547) with its granite frame and balcony.
- The dry moat all round (4 m deep, granite walls and parapet), its three bridges with two arches, granite piers
  and iron railings (south, north, west), and the head of the covered gallery that crosses it towards the
  Capilla Real and the Casa de Oficios (only the part over the moat).

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap relation 18324181 (outline, the Austrias and Borbones courtyards) and its building:part
  ways 1337369925–1337369929 (the four towers and the south central pavilion), in the axes of heading 159.7°
  (the long sides at 69.7°; the main front, with Sabatini's central pavilion and the gate, faces the forecourt
  and the parterres to the south-south-east). Main body 113.4 × 48.6 m, towers 8.3 × 8.4 m projecting 3 m,
  pavilions 20.4 m wide projecting 3.3 m. The Patio Central (not in OSM) from the PNOA orthophoto.
- Heights from a frontal panorama of the south front on Wikimedia Commons, scaled by the 119.8 m between the
  outer faces of the towers (±8 %): main cornice 9.4 m above the forecourt, mansard break 13 m, ridges 14.9 m,
  chimneys 17.5 m; pavilion attic 12.8 m, pediment 12.1 m, ridge 16.8 m; towers 17 m to the cornice, bell cap
  to 20.2 m, upper cap to 22.3 m, finial 24.4 m.
- Moat: IGN MDT05 (forecourt ≈ 611 m, moat floor 4 m lower) and the orthophoto (outer wall ≈ 7.3 m from the
  long fronts, 4 m from the towers).

Axes (docs/MODEL_SPEC.md): +Y up, the south front faces +Z (heading 159.7°), +X points to 69.7° (east-north-east).
Origin at the centroid of the OSM outline; y = 0 is the forecourt (the ground outside the moat), the moat floor
and the foot of the palace are at −4 m.

    python model_sources/Q1368571_palacio_real_de_el_pardo.py   -> models/Q1368571/model.glb (+ footprint in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, lathe, cap, extrude, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(1772)

# ── Colours ──────────────────────────────────────────────────────────────────
RENDER = (0.91, 0.83, 0.61)           # the cream render of the Bourbon reform
TRIM = (0.95, 0.93, 0.86)             # pilaster strips, frames, cornices
GRANITE = (0.62, 0.61, 0.58)
GRANITE_D = (0.53, 0.52, 0.50)
SLATE = (0.33, 0.35, 0.40)
SLATE_D = (0.27, 0.29, 0.33)
SHUTTER = (0.84, 0.84, 0.82)          # white louvred shutters
GLASS = (0.15, 0.16, 0.18)
DARK = (0.10, 0.09, 0.09)             # arcades, openings
DOOR = (0.12, 0.11, 0.10)
IRON = (0.09, 0.09, 0.10)
BRICK = (0.72, 0.50, 0.38)            # the chimneys
LEAD = (0.45, 0.46, 0.48)
SAND = (0.79, 0.72, 0.58)             # the floor of the moat
PAVE = (0.73, 0.71, 0.66)             # courtyards
APRON = (0.70, 0.68, 0.63)
HEDGE = (0.20, 0.33, 0.16)

FOUND = -4.0      # moat floor and foot of the palace
E = 9.4           # main cornice
XF, ZF = 56.7, 24.3                   # outer faces of the ranges
TX0, TX1, TZ0, TZ1 = 51.4, 59.7, 19.1, 27.5   # corner towers (±)
PX, PZ = 10.2, 27.6                   # central pavilions
LX = 46.9                             # the two-bay bodies beside the towers: |x| from LX to TX0
MX, MZ = 63.8, 31.8                   # inner face of the outer moat wall
COURTS = {"W": (-41.8, -22.9, -10.7, 10.8), "C": (-7.0, 7.0, -10.5, 10.5), "E": (22.3, 41.6, -10.9, 10.8)}
ZI = 10.7                             # courtyard side of the long ranges (±)
M_A, M_R1, M_R2 = 2.0, 3.6, 1.9       # mansard: lower slope inset, its rise, rise of the upper slope


def tint(c, k=0.015):
    return tuple(np.clip(np.array(c) * (1 + rng.normal(0, k)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    x0, x1 = min(x0, x1), max(x0, x1); z0, z1 = min(z0, z1), max(z0, z1)
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sphere(r, c, seg=8, rings=5):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def outward(m, ctr):
    """Turn every triangle so that it faces away from the point ctr."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    n = np.cross(b - a, c - a)
    flip = np.einsum("ij,ij->i", n, (a + b + c) / 3 - np.asarray(ctr)) < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def quads_mesh(quads, c):
    V, F = [], []
    for q in quads:
        o = len(V); V += list(q)
        F += [(o, o + 1, o + 2)] + ([(o, o + 2, o + 3)] if len(q) == 4 else [])
    return Mesh(V, F, c)


class Face:
    """A wall plane: a = start point (x, z), t = direction along the wall, n = outward normal; local (u, y, w)."""

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


def face_of(x0, z0, x1, z1, n):
    a, b = np.array([x0, z0], float), np.array([x1, z1], float)
    L = float(np.linalg.norm(b - a))
    return Face(a, (b - a) / L, n), L


def rect_faces(x0, x1, z0, z1, inward=False):
    """The four sides of a rectangle as Faces: +z, +x, −z, −x (normals outwards, or into the rectangle)."""
    s = -1 if inward else 1
    return {"+z": face_of(x0, z1, x1, z1, (0, s)), "+x": face_of(x1, z1, x1, z0, (s, 0)),
            "-z": face_of(x1, z0, x0, z0, (0, -s)), "-x": face_of(x0, z0, x0, z1, (-s, 0))}


def arch_pts(u, y0, ys, half, seg=6):
    pts = [(u - half, y0), (u + half, y0)]
    pts += [(u + half * math.cos(a), ys + half * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)]
    return pts


def rect_pts(u, y, w, h):
    return [(u - w / 2, y - h / 2), (u + w / 2, y - h / 2), (u + w / 2, y + h / 2), (u - w / 2, y + h / 2)]


def circle_pts(u, y, r, n=10):
    return [(u + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n + 1)[:-1]]


def bays(L, n, margin=0.0):
    s = (L - 2 * margin) / n
    return [margin + s * (k + 0.5) for k in range(n)], s


# ── Roofs ────────────────────────────────────────────────────────────────────
def hip_roof(x0, x1, z0, z1, E0, R, c):
    L, W = x1 - x0, z1 - z0
    R = min(R, 0.5 * min(L, W) * 0.9)
    if L >= W:
        a, b = (x0 + W / 2, (z0 + z1) / 2), (x1 - W / 2, (z0 + z1) / 2)
    else:
        a, b = ((x0 + x1) / 2, z0 + L / 2), ((x0 + x1) / 2, z1 - L / 2)
    Y = E0 + R
    c00, c10, c11, c01 = (x0, E0, z0), (x1, E0, z0), (x1, E0, z1), (x0, E0, z1)
    A, Bp = (a[0], Y, a[1]), (b[0], Y, b[1])
    if L >= W:
        faces = [(c00, c10, Bp, A), (c11, c01, A, Bp), (c01, c00, A), (c10, c11, Bp)]
    else:
        faces = [(c10, c11, Bp, A), (c01, c00, A, Bp), (c00, c10, A), (c11, c01, Bp)]
    return outward(quads_mesh(faces, c), ((x0 + x1) / 2, E0, (z0 + z1) / 2))


def mansard(x0, x1, z0, z1, E0=E, c=None):
    """A slate mansard: the steep lower slope (inset M_A, rise M_R1) and a hipped upper slope (rise M_R2)."""
    c = tint(SLATE, 0.02) if c is None else c
    a, y1 = M_A, E0 + M_R1
    lo = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
    hi = [(x0 + a, z0 + a), (x1 - a, z0 + a), (x1 - a, z1 - a), (x0 + a, z1 - a)]
    quads = []
    for i in range(4):
        j = (i + 1) % 4
        quads.append(((lo[i][0], E0, lo[i][1]), (lo[j][0], E0, lo[j][1]), (hi[j][0], y1, hi[j][1]), (hi[i][0], y1, hi[i][1])))
    low = outward(quads_mesh(quads, c), ((x0 + x1) / 2, E0, (z0 + z1) / 2))
    eave = [B(x0 - 0.15, x1 + 0.15, E0 - 0.05, E0 + 0.12, z0 - 0.15, z1 + 0.15, LEAD)]
    return [low, hip_roof(x0 + a, x1 - a, z0 + a, z1 - a, y1, M_R2, tint(SLATE_D, 0.02))] + eave


def dormer(f, u, E0=E, w=1.4):
    """A dormer on the lower slope of a mansard, its front 0.6 m in from the eave line of face f."""
    d0 = 0.6
    yb = E0 + M_R1 * d0 / M_A - 0.3
    yt = yb + 1.9
    out = [f.slab(u - w / 2, u + w / 2, yb - 0.4, yt, -d0 - 1.9, -d0, tint(SLATE, 0.02)),
           f.poly(rect_pts(u, yb + 0.95, w - 0.25, 1.65), TRIM, -d0 + 0.03),
           f.poly(rect_pts(u, yb + 0.9, w - 0.6, 1.25), GLASS, -d0 + 0.06)]
    c = [f.P(u - w / 2 - 0.1, yt, -d0 + 0.12), f.P(u + w / 2 + 0.1, yt, -d0 + 0.12),
         f.P(u + w / 2 + 0.1, yt, -d0 - 2.0), f.P(u - w / 2 - 0.1, yt, -d0 - 2.0)]
    ap = f.P(u, yt + 0.85, -d0 - 0.75)
    tris = [(c[0], c[1], ap), (c[1], c[2], ap), (c[2], c[3], ap), (c[3], c[0], ap)]
    out.append(outward(quads_mesh(tris, tint(SLATE_D, 0.02)), f.P(u, yt + 0.2, -d0 - 0.9)))
    out.append(B(ap[0] - 0.05, ap[0] + 0.05, ap[1] - 0.1, ap[1] + 0.55, ap[2] - 0.05, ap[2] + 0.05, LEAD))
    return out


def chimney(x, z, top=17.6, base=12.0, sx=1.1, sz=0.75):
    return [B(x - sx / 2, x + sx / 2, base, top, z - sz / 2, z + sz / 2, tint(BRICK, 0.04)),
            B(x - sx / 2 - 0.12, x + sx / 2 + 0.12, top, top + 0.22, z - sz / 2 - 0.12, z + sz / 2 + 0.12, tint(BRICK, 0.04)),
            B(x - sx / 2 + 0.1, x + sx / 2 - 0.1, top + 0.22, top + 0.45, z - sz / 2 + 0.1, z + sz / 2 - 0.1, GRANITE_D)]


# ── Walls and windows ────────────────────────────────────────────────────────
def window(f, u, y0, w, h, balcony=False, hood=False, rail=False, glass=None):
    if glass is None:
        glass = rng.random() < 0.45
    out = [f.poly(rect_pts(u, y0 + h / 2, w + 0.5, h + 0.45), TRIM, 0.03),
           f.poly(rect_pts(u, y0 + h / 2, w, h), GLASS if glass else tint(SHUTTER, 0.02), 0.06)]
    if hood:
        out.append(f.slab(u - w / 2 - 0.4, u + w / 2 + 0.4, y0 + h + 0.22, y0 + h + 0.45, -0.02, 0.3, TRIM))
    if balcony:
        out.append(f.slab(u - w / 2 - 0.35, u + w / 2 + 0.35, y0 - 0.22, y0, -0.02, 0.65, GRANITE))
        out.append(f.slab(u - w / 2 - 0.3, u + w / 2 + 0.3, y0, y0 + 0.95, 0.58, 0.62, IRON))
    elif rail:
        out.append(f.slab(u - w / 2 - 0.1, u + w / 2 + 0.1, y0, y0 + 0.75, 0.14, 0.18, IRON))
    return out


def dress(f, L, top=E, plinth_from=FOUND, corners=True, impost=True):
    """Granite plinth, string course, cornice and corner pilaster strips of a rendered wall."""
    out = [f.slab(-0.15, L + 0.15, plinth_from, 0.35, -0.05, 0.14, tint(GRANITE, 0.02)),
           f.slab(-0.3, L + 0.3, top - 0.45, top, -0.05, 0.5, TRIM),
           f.slab(-0.2, L + 0.2, top - 0.75, top - 0.45, -0.05, 0.22, TRIM)]
    if impost:
        out.append(f.slab(-0.1, L + 0.1, 4.35, 4.6, -0.05, 0.12, TRIM))
    if corners:
        out += [f.slab(-0.05, 0.9, 0.35, top - 0.75, -0.05, 0.1, TRIM), f.slab(L - 0.9, L + 0.05, 0.35, top - 0.75, -0.05, 0.1, TRIM)]
    return out


def front(f, L, n, moat=True, skip=(), dormers=True, ground="win", corners=True, us=None):
    """A range front with n bays (or the bay centres us): basement windows in the moat, ground floor, main
    floor with balconies, cornice, and one dormer per bay on the mansard."""
    out = dress(f, L, corners=corners, plinth_from=FOUND if moat else -0.5)
    if us is None:
        us, s = bays(L, n)
    else:
        s = 3.8
    for u in us:
        if any(a <= u <= b for a, b in skip):
            continue
        if moat:
            out.append(f.poly(rect_pts(u, -2.0, 0.9, 1.1), DARK, 0.18))
            out.append(f.slab(u - 0.5, u + 0.5, -2.6, -1.4, 0.14, 0.2, IRON))
        if ground == "arcade":
            out.append(f.poly(arch_pts(u, 0.0, 2.55, min(1.25, s * 0.36)), DARK, 0.05))
            out.append(f.poly(arch_pts(u, -0.05, 2.55, min(1.25, s * 0.36) + 0.3), TRIM, 0.025))
        else:
            out += window(f, u, 0.9, 1.25, 2.2, rail=True)
        out += window(f, u, 5.0, 1.35, 2.75, balcony=True, hood=True)
        if dormers:
            out += dormer(f, u)
    return out


# ── The palace ───────────────────────────────────────────────────────────────
def body_outline():
    return [(-XF, -ZF), (-PX, -ZF), (-PX, -PZ), (PX, -PZ), (PX, -ZF), (XF, -ZF),
            (XF, ZF), (PX, ZF), (PX, PZ), (-PX, PZ), (-PX, ZF), (-XF, ZF)]


def court_ring(k):
    x0, x1, z0, z1 = COURTS[k]
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def palace():
    out = [extrude(body_outline(), FOUND, E, tint(RENDER, 0.01), holes=[court_ring(k) for k in COURTS], top=True,
                   top_c=SLATE_D)]
    # ── roofs: the ring of mansards
    xw, xe = (-XF + COURTS["W"][0]) / 2, (XF + COURTS["E"][1]) / 2
    zs, zn = (ZF + ZI) / 2, -(ZF + ZI) / 2
    out += mansard(xw, xe, ZI, ZF)                                       # south range
    out += mansard(xw, xe, -ZF, -ZI)                                     # north range
    out += mansard(-XF, COURTS["W"][0], -ZF, ZF)                         # west range
    out += mansard(COURTS["E"][1], XF, -ZF, ZF)                          # east range
    out += mansard(COURTS["W"][1], COURTS["C"][0], zn, zs)               # the two ranges between the courtyards
    out += mansard(COURTS["C"][1], COURTS["E"][0], zn, zs)
    # ── outer fronts
    for sgn in (1, -1):                                                  # south (+1) and north (−1)
        n = (0, sgn)
        for xa, xb in ((-LX, -PX), (PX, LX)):
            a, b = (xa, xb) if sgn > 0 else (xb, xa)
            f, L = face_of(a, sgn * ZF, b, sgn * ZF, n)
            out += front(f, L, 10)
        out += pavilion(sgn)
        for xa, xb in ((-TX0, -LX), (LX, TX0)):
            out += side_body(xa, xb, sgn)
    f, L = face_of(-XF, -TZ0, -XF, 15.1, (-1, 0))                       # west front, with the Habsburg portal
    us = [TZ0 + 3.8 * k for k in range(-4, 4) if k != 0]                  # bays 3.8 m apart round the portal (z = 0)
    out += front(f, L, 0, us=us)
    out += habsburg_portal(f, TZ0)
    f, L = face_of(XF, TZ0, XF, -TZ0, (1, 0))                           # east front
    out += front(f, L, 10)
    # ── courtyards
    for k, (x0, x1, z0, z1) in COURTS.items():
        for side, (f, L) in rect_faces(x0, x1, z0, z1, inward=True).items():
            if k == "C" and side in ("+z", "-z"):
                continue                                                 # the pavilions stand there (below)
            n = max(1, int(round(L / 3.7)))
            out += front(f, L, n, moat=False, ground="win" if k == "C" else "arcade", corners=False)
        out.append(cap(court_ring(k), 0.02, tint(PAVE, 0.01)))
    # central courtyard: the inner walls of the two pavilions
    for sgn in (1, -1):
        z = sgn * COURTS["C"][3]
        f, L = face_of(7.0, z, -7.0, z, (0, -sgn)) if sgn > 0 else face_of(-7.0, z, 7.0, z, (0, -sgn))
        out += dress(f, L, top=12.8, plinth_from=-0.5, corners=False)
        for u in bays(L, 3)[0]:
            out += window(f, u, 0.9, 1.25, 2.2)
            out += window(f, u, 5.0, 1.35, 2.75, balcony=True, hood=True)
            out += window(f, u, 10.3, 1.0, 1.4)
    # ── the corner towers
    for sx in (-1, 1):
        for sz in (-1, 1):
            out += tower(sx, sz)
    # ── the head of the gallery on the west (between the SW tower and the west front)
    out.append(B(-TX1, -XF, FOUND, E, 15.1, TZ0, tint(RENDER, 0.01)))
    f, L = face_of(-TX1, 15.1, -XF, 15.1, (0, -1))
    out += dress(f, L, corners=False)
    out += mansard(-TX1, -XF + 0.5, 15.1, TZ0 + 0.5)
    out += gallery()
    # ── chimneys
    ch = []
    for z in (zs - 1.2, zn + 1.2):
        for x in (-44.0, -35.0, -27.0, -18.5, 18.5, 27.0, 35.0, 44.0):
            ch.append((x, z))
    for x in (xw - 1.2, xe + 1.2):
        for z in (-14.0, -4.0, 6.0, 14.5):
            ch.append((x, z))
    for x in ((COURTS["W"][1] + COURTS["C"][0]) / 2, (COURTS["C"][1] + COURTS["E"][0]) / 2):
        for z in (-6.0, 6.0):
            ch.append((x + 1.0, z))
    for x, z in ch:
        out += chimney(x, z)
    for sgn in (1, -1):
        for x in (-5.0, 5.0):
            out += chimney(x, sgn * 16.0, top=18.8, base=14.0)
    return out


def side_body(xa, xb, sgn):
    """The narrow body beside each tower on the long fronts: four floors with its own small hipped roof."""
    z0, z1 = (ZF - 7.0, ZF) if sgn > 0 else (-ZF, -ZF + 7.0)
    out = [B(xa, xb, FOUND, 12.0, z0, z1, tint(RENDER, 0.01))]
    a, b = (xa, xb) if sgn > 0 else (xb, xa)
    f, L = face_of(a, sgn * ZF, b, sgn * ZF, (0, sgn))
    out += dress(f, L, top=12.0)
    out.append(f.slab(-0.2, L + 0.2, E - 0.4, E - 0.1, -0.05, 0.2, TRIM))
    u = L / 2
    out.append(f.poly(rect_pts(u, -2.0, 0.9, 1.1), DARK, 0.18))
    out += window(f, u, 0.9, 1.15, 2.2, rail=True)
    out += window(f, u, 5.0, 1.2, 2.75, balcony=True, hood=True)
    out += window(f, u, 9.9, 1.0, 1.4)
    out.append(hip_roof(xa - 0.3, xb + 0.3, z0 - 0.3, z1 + 0.3, 12.0, 2.4, tint(SLATE, 0.02)))
    out.append(B(xa - 0.35, xb + 0.35, 11.9, 12.1, min(z0, z1) - 0.35, max(z0, z1) + 0.35, LEAD))
    return out


def pavilion(sgn):
    """Sabatini's central pavilion: five bays, the gate, attic with oculi, pediment with the arms, hipped roof."""
    z0, z1 = (COURTS["C"][3], PZ) if sgn > 0 else (-PZ, -COURTS["C"][3])
    out = [B(-PX, PX, E - 0.5, 12.8, z0, z1, tint(RENDER, 0.01))]
    a, b = (-PX, PX) if sgn > 0 else (PX, -PX)
    f, L = face_of(a, sgn * PZ, b, sgn * PZ, (0, sgn))
    out += dress(f, L, top=E)
    out += [f.slab(-0.6, 1.1, 0.35, E - 0.75, -0.05, 0.16, TRIM), f.slab(L - 1.1, L + 0.6, 0.35, E - 0.75, -0.05, 0.16, TRIM)]
    out.append(f.slab(-0.3, L + 0.3, 12.35, 12.8, -0.05, 0.45, TRIM))     # attic cornice
    us = [L / 2 + k * 3.7 for k in (-2, -1, 0, 1, 2)]
    for i, u in enumerate(us):
        out.append(f.poly(rect_pts(u, -2.0, 0.9, 1.1), DARK, 0.18))
        if i == 2:                                                       # the gate
            out.append(f.poly(arch_pts(u, 0.0, 2.95, 1.35), DOOR, 0.06))
            out.append(f.poly(arch_pts(u, -0.05, 2.95, 1.75), GRANITE, 0.03))
        else:
            out += window(f, u, 0.9, 1.25, 2.2, rail=True)
        out += window(f, u, 5.0, 1.35, 2.75, balcony=True, hood=True)
    for u in (L / 2 - 6.6, L / 2 + 6.6):                                 # oculi of the attic
        out.append(f.poly(circle_pts(u, 11.0, 0.85, 12), TRIM, 0.03))
        out.append(f.poly(circle_pts(u, 11.0, 0.6, 12), GLASS, 0.06))
    # pediment over the three middle bays
    hw, yb, ya = 5.3, E, 12.1
    out.append(f.poly([(L / 2 - hw, yb), (L / 2 + hw, yb), (L / 2, ya)], tint(RENDER, 0.01), 0.35))
    for (u0, y0), (u1, y1) in (((L / 2 - hw - 0.3, yb - 0.1), (L / 2, ya + 0.25)), ((L / 2, ya + 0.25), (L / 2 + hw + 0.3, yb - 0.1))):
        out.append(sloped_moulding(f, (u0, y0), (u1, y1), 0.5))
    out.append(f.slab(L / 2 - 0.75, L / 2 + 0.75, 9.6, 11.2, 0.3, 0.6, TRIM))   # the royal arms
    out.append(f.slab(L / 2 - 0.5, L / 2 + 0.5, 11.2, 11.6, 0.3, 0.55, TRIM))
    # sides of the projection
    for xs, nn in ((-PX, (-1, 0)), (PX, (1, 0))):
        za, zb = (ZF, PZ) if sgn > 0 else (-PZ, -ZF)
        g, L2 = face_of(xs, za, xs, zb, nn)
        out += dress(g, L2, corners=False)
        out.append(g.slab(-0.3, L2 + 0.3, 12.35, 12.8, -0.05, 0.45, TRIM))
    out.append(hip_roof(-PX - 0.3, PX + 0.3, min(z0, z1) - (0.3 if sgn < 0 else 0), max(z0, z1) + (0.3 if sgn > 0 else 0),
                        12.8, 4.6, tint(SLATE, 0.02)))
    g = Face(f.a - f.n * 1.6, f.t, f.n)                                  # two dormers on the front slope
    for u in (L / 2 - 4.5, L / 2 + 4.5):
        out += dormer(g, u, E0=12.8, w=1.3)
    # the iron finial on the ridge
    zc = (z0 + z1) / 2
    out.append(B(-0.25, 0.25, 17.2, 17.9, zc - 0.25, zc + 0.25, LEAD))
    out.append(sphere(0.3, LEAD).T(0, 18.2, zc))
    if sgn > 0:                                                          # the sentry boxes at the gate
        for x in (-7.2, 7.2):
            out += sentry_box(x, MZ + 1.4)
    return out


def sloped_moulding(f, a, b, w):
    (u0, y0), (u1, y1) = a, b
    P0, P1 = np.array(f.P(u0, y0, 0.0)), np.array(f.P(u1, y1, 0.0))
    d = P1 - P0
    nrm = np.array([f.n[0], 0, f.n[1]])
    up = np.cross(nrm, d); up = up / np.linalg.norm(up) * 0.35
    if up[1] < 0:
        up = -up
    o = nrm * (w + 0.15)
    V = [P0, P1, P1 + up, P0 + up, P0 + o, P1 + o, P1 + up + o, P0 + up + o]
    quads = [(V[0], V[1], V[2], V[3]), (V[4], V[5], V[6], V[7]), (V[3], V[2], V[6], V[7]), (V[0], V[1], V[5], V[4]),
             (V[0], V[3], V[7], V[4]), (V[1], V[2], V[6], V[5])]
    return outward(quads_mesh(quads, TRIM), np.mean(V, 0))


def sentry_box(x, z):
    out = [B(x - 0.7, x + 0.7, 0, 0.3, z - 0.7, z + 0.7, GRANITE),
           B(x - 0.6, x + 0.6, 0.3, 2.5, z - 0.6, z + 0.6, TRIM),
           B(x - 0.7, x + 0.7, 2.5, 2.7, z - 0.7, z + 0.7, GRANITE)]
    tris = [((x - 0.7, 2.7, z - 0.7), (x + 0.7, 2.7, z - 0.7), (x, 3.5, z)), ((x + 0.7, 2.7, z - 0.7), (x + 0.7, 2.7, z + 0.7), (x, 3.5, z)),
            ((x + 0.7, 2.7, z + 0.7), (x - 0.7, 2.7, z + 0.7), (x, 3.5, z)), ((x - 0.7, 2.7, z + 0.7), (x - 0.7, 2.7, z - 0.7), (x, 3.5, z))]
    out.append(outward(quads_mesh(tris, GRANITE), (x, 2.9, z)))
    out.append(sphere(0.18, GRANITE).T(x, 3.6, z))
    out.append(B(x - 0.35, x + 0.35, 0.3, 2.0, z + 0.6, z + 0.62, DARK) if z > 0 else B(x - 0.35, x + 0.35, 0.3, 2.0, z - 0.62, z - 0.6, DARK))
    return out


def habsburg_portal(f, u):
    """The granite portal of Charles V (1547) on the west front: door, entablature, the window above with its
    scrolls and pediment."""
    out = [f.slab(u - 1.9, u + 1.9, 0.0, 4.3, -0.05, 0.3, GRANITE),
           f.poly(rect_pts(u, 1.75, 1.9, 3.5), DOOR, 0.33),
           f.slab(u - 2.2, u + 2.2, 4.3, 5.0, -0.05, 0.45, GRANITE),
           f.slab(u - 1.5, u + 1.5, 5.0, 8.0, -0.05, 0.25, GRANITE),
           f.poly(rect_pts(u, 6.4, 1.2, 2.4), tint(SHUTTER, 0.02), 0.28),
           f.slab(u - 1.0, u + 1.0, 5.0, 5.9, 0.4, 0.45, IRON),
           f.slab(u - 1.1, u + 1.1, 4.85, 5.0, -0.05, 0.7, GRANITE)]
    for s in (-1, 1):
        out.append(f.slab(u + s * 1.5 - 0.2, u + s * 1.5 + 0.2, 5.0, 7.6, 0.2, 0.42, GRANITE))   # the scrolls
    out.append(f.slab(u - 1.8, u + 1.8, 8.0, 8.3, -0.05, 0.45, GRANITE))
    out.append(f.poly([(u - 1.8, 8.3), (u + 1.8, 8.3), (u, 9.3)], GRANITE, 0.3))
    return out


def sq_stack(cx, cz, prof, c):
    """A square-sectioned body of flat faces from a profile [(half width, y)] (the caps of the towers)."""
    quads = []
    for (h0, y0), (h1, y1) in zip(prof[:-1], prof[1:]):
        r0 = [(cx - h0, cz - h0), (cx + h0, cz - h0), (cx + h0, cz + h0), (cx - h0, cz + h0)]
        r1 = [(cx - h1, cz - h1), (cx + h1, cz - h1), (cx + h1, cz + h1), (cx - h1, cz + h1)]
        for i in range(4):
            j = (i + 1) % 4
            q = ((r0[i][0], y0, r0[i][1]), (r0[j][0], y0, r0[j][1]), (r1[j][0], y1, r1[j][1]), (r1[i][0], y1, r1[i][1]))
            if h1 == 0:
                q = q[:3]
            quads.append(q)
    m = quads_mesh(quads, c)
    # orient each face away from the axis of the tower at its own height
    a, b, cc = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    n = np.cross(b - a, cc - a)
    ctr = (a + b + cc) / 3
    ax = np.stack([np.full(len(ctr), cx), ctr[:, 1] - 5.0, np.full(len(ctr), cz)], 1)
    flip = np.einsum("ij,ij->i", n, ctr - ax) < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def tower(sx, sz):
    """A corner tower: four floors to 17 m, string courses, the slate bell cap with oculus dormers, finial."""
    x0, x1 = sorted((sx * TX0, sx * TX1)); z0, z1 = sorted((sz * TZ0, sz * TZ1))
    cx, cz, hw = (x0 + x1) / 2, (z0 + z1) / 2, (x1 - x0) / 2
    top = 17.0
    out = [B(x0, x1, FOUND, top, z0, z1, tint(RENDER, 0.01))]
    faces = rect_faces(x0, x1, z0, z1)
    outer = {("+x" if sx > 0 else "-x"), ("+z" if sz > 0 else "-z")}
    for k, (f, L) in faces.items():
        out += dress(f, L, top=top)
        out.append(f.slab(-0.2, L + 0.2, E - 0.4, E - 0.1, -0.05, 0.22, TRIM))
        out.append(f.slab(-0.2, L + 0.2, 13.0, 13.25, -0.05, 0.18, TRIM))
        u = L / 2
        if k in outer:
            out.append(f.poly(rect_pts(u, -2.0, 0.9, 1.1), DARK, 0.18))
            out += window(f, u, 0.9, 1.25, 2.2, rail=True)
            out += window(f, u, 5.0, 1.35, 2.75, balcony=True, hood=True)
            out += window(f, u, 10.3, 1.1, 1.6)
        out += window(f, u, 14.0, 1.0, 1.7)
        for s in (-1, 1):                                                # blind arches beside the top window
            out.append(f.poly(arch_pts(u + s * 2.5, 13.7, 15.6, 0.55), tint(RENDER, 0.03), 0.04))
    # the cap
    h = hw
    prof = [(h + 0.55, top), (h + 0.55, top + 0.18), (h + 0.35, top + 0.45), (h - 0.05, top + 1.3), (h - 0.5, top + 2.4),
            (h - 0.75, top + 3.2), (h - 0.6, top + 3.2), (h - 0.6, top + 3.45), (h - 0.85, top + 3.55),
            (h - 1.75, top + 4.45), (h - 2.7, top + 4.95), (0.75, top + 5.3), (0.0, top + 5.32)]
    out.append(sq_stack(cx, cz, prof[:6], tint(SLATE, 0.02)))
    out.append(sq_stack(cx, cz, prof[5:9], LEAD))
    out.append(sq_stack(cx, cz, prof[8:], tint(SLATE_D, 0.02)))
    for k, (f, L) in faces.items():                                      # oculus dormers on the bell
        u = L / 2
        out.append(f.slab(u - 0.85, u + 0.85, top + 0.6, top + 2.6, -1.6, -0.15, tint(SLATE, 0.02)))
        out.append(f.poly(circle_pts(u, top + 1.6, 0.72, 12), TRIM, -0.12))
        out.append(f.poly(circle_pts(u, top + 1.6, 0.45, 12), GLASS, -0.09))
        out.append(f.slab(u - 0.95, u + 0.95, top + 2.6, top + 2.85, -1.7, -0.05, LEAD))
    out.append(lathe([(0.0, 0.0), (0.42, 0.0), (0.42, 0.25), (0.22, 0.45), (0.38, 0.8), (0.3, 1.05), (0.12, 1.2),
                      (0.1, 1.5), (0.22, 1.65), (0.1, 1.85), (0.05, 2.9), (0.0, 3.0)], LEAD, 8).T(cx, top + 5.3, cz))
    return out


def gallery():
    """The head of the covered gallery to the Capilla Real, where it crosses the west moat."""
    x0, x1, z0, z1 = -MX - 3.0, -TX1, 15.1, TZ0
    out = [B(x0, x1, 0.0, 8.2, z0, z1, tint(RENDER, 0.01))]
    for xa, xb in ((x0, x0 + 1.4), (x1 - 1.0, x1)):
        out.append(B(xa, xb, FOUND, 0.0, z0, z1, tint(GRANITE, 0.02)))
    out.append(B(x0, x1, -1.0, 0.0, z0, z1, tint(GRANITE, 0.02)))
    for zz, nn in ((z1, (0, 1)), (z0, (0, -1))):
        f, L = face_of(x0, zz, x1, zz, nn) if nn[1] > 0 else face_of(x1, zz, x0, zz, nn)
        out += [f.slab(-0.1, L + 0.1, -0.2, 0.4, -0.05, 0.12, GRANITE), f.slab(-0.2, L + 0.2, 7.8, 8.2, -0.05, 0.35, TRIM),
                f.slab(-0.1, L + 0.1, 3.8, 4.0, -0.05, 0.1, TRIM)]
        out.append(f.poly(arch_pts(L / 2, 0.4, 2.3, 1.3), DARK, 0.05))
        out += window(f, L / 2, 4.6, 1.2, 2.4, balcony=False, hood=True)
    out += mansard(x0 - 1.0, x1, z0, z1, E0=8.2, c=tint(SLATE, 0.02))
    return out


# ── The moat ─────────────────────────────────────────────────────────────────
def plan_union():
    """Outline of everything standing in the moat (palace, towers, gallery head), for the moat floor."""
    return [(-TX1, TZ1), (-TX0, TZ1), (-TX0, ZF), (-PX, ZF), (-PX, PZ), (PX, PZ), (PX, ZF), (TX0, ZF), (TX0, TZ1),
            (TX1, TZ1), (TX1, TZ0), (XF, TZ0), (XF, -TZ0), (TX1, -TZ0), (TX1, -TZ1), (TX0, -TZ1), (TX0, -ZF),
            (PX, -ZF), (PX, -PZ), (-PX, -PZ), (-PX, -ZF), (-TX0, -ZF), (-TX0, -TZ1), (-TX1, -TZ1), (-TX1, -TZ0),
            (-XF, -TZ0), (-XF, 15.1), (-TX1, 15.1)]


def bridge_along(a0, a1, c0, hw, axis):
    """A stone bridge over the moat with two arches, deck at y = 0, granite piers with balls and iron railings.
    axis 'z': it runs along z from a0 to a1 at x = c0; axis 'x': along x at z = c0."""
    L = a1 - a0
    sgn = 1 if L > 0 else -1
    L = abs(L)
    out = []
    # the two side walls with their arches, as solid blocks with dark arch openings drawn on both sides
    def P(t, y, s):            # t along the bridge, s across
        return (c0 + s, y, a0 + sgn * t) if axis == "z" else (a0 + sgn * t, y, c0 + s)

    lo, hi = min(a0, a1), max(a0, a1)
    if axis == "z":
        out.append(B(c0 - hw, c0 + hw, FOUND, 0.0, lo, hi, tint(GRANITE, 0.02)))
        out.append(B(c0 - hw - 0.15, c0 + hw + 0.15, -0.35, 0.02, lo, hi, GRANITE_D))
    else:
        out.append(B(lo, hi, FOUND, 0.0, c0 - hw, c0 + hw, tint(GRANITE, 0.02)))
        out.append(B(lo, hi, -0.35, 0.02, c0 - hw - 0.15, c0 + hw + 0.15, GRANITE_D))
    for s in (-1, 1):
        n = (s, 0) if axis == "z" else (0, s)
        pa, pb = P(0, 0, s * hw), P(L, 0, s * hw)
        f, LL = face_of(pa[0], pa[2], pb[0], pb[2], n)
        for k in (0.25, 0.75):
            out.append(f.poly(arch_pts(LL * k, FOUND, -1.6, min(1.1, LL * 0.2)), DARK, 0.03))
        # parapet: granite piers with balls, iron railing between
        for t in (0.25, L - 0.25):
            p = P(t, 0, s * (hw - 0.25))
            out.append(B(p[0] - 0.25, p[0] + 0.25, 0.0, 1.15, p[2] - 0.25, p[2] + 0.25, GRANITE))
            out.append(sphere(0.2, GRANITE).T(p[0], 1.35, p[2]))
        r0, r1 = P(0.5, 0, s * (hw - 0.25)), P(L - 0.5, 0, s * (hw - 0.25))
        if axis == "z":
            out.append(B(r0[0] - 0.04, r0[0] + 0.04, 0.0, 1.0, min(r0[2], r1[2]), max(r0[2], r1[2]), IRON))
        else:
            out.append(B(min(r0[0], r1[0]), max(r0[0], r1[0]), 0.0, 1.0, r0[2] - 0.04, r0[2] + 0.04, IRON))
    return out


def moat():
    out = []
    outer = [(-MX, -MZ), (MX, -MZ), (MX, MZ), (-MX, MZ)]
    out.append(cap(outer, FOUND, tint(SAND, 0.01), holes=[plan_union()]))
    # the outer retaining wall and its parapet, with gaps for the bridges
    t = 0.6
    out += [B(-MX - t, MX + t, FOUND - 1.0, 0.0, MZ, MZ + t, tint(GRANITE, 0.02)),
            B(-MX - t, MX + t, FOUND - 1.0, 0.0, -MZ - t, -MZ, tint(GRANITE, 0.02)),
            B(MX, MX + t, FOUND - 1.0, 0.0, -MZ, MZ, tint(GRANITE, 0.02)),
            B(-MX - t, -MX, FOUND - 1.0, 0.0, -MZ, MZ, tint(GRANITE, 0.02))]
    gaps_z = (-2.75, 2.75)
    for zz in (MZ, -MZ - t):
        for xa, xb in ((-MX - t, gaps_z[0]), (gaps_z[1], MX + t)):
            out.append(B(xa, xb, 0.0, 0.8, zz, zz + t, tint(GRANITE, 0.02)))
            out.append(B(xa, xb, 0.8, 0.9, zz - 0.05, zz + t + 0.05, GRANITE_D))
    out.append(B(MX, MX + t, 0.0, 0.8, -MZ, MZ, tint(GRANITE, 0.02)))
    out.append(B(MX - 0.05, MX + t + 0.05, 0.8, 0.9, -MZ, MZ, GRANITE_D))
    for za, zb in ((-MZ, -2.2), (2.2, 15.1), (TZ0, MZ)):
        out.append(B(-MX - t, -MX, 0.0, 0.8, za, zb, tint(GRANITE, 0.02)))
        out.append(B(-MX - t - 0.05, -MX + 0.05, 0.8, 0.9, za, zb, GRANITE_D))
    # paving round the moat, and the low box hedges along the south front
    out.append(cap([(-MX - 3.6, -MZ - 3.6), (MX + 3.6, -MZ - 3.6), (MX + 3.6, MZ + 3.6), (-MX - 3.6, MZ + 3.6)], 0.0,
                   tint(APRON, 0.01), holes=[[(-MX - t, -MZ - t), (MX + t, -MZ - t), (MX + t, MZ + t), (-MX - t, MZ + t)]]))
    for xa, xb in ((-MX + 2.0, -9.0), (9.0, MX - 2.0)):
        out.append(B(xa, xb, 0.0, 0.7, MZ + t + 0.3, MZ + t + 1.3, tint(HEDGE, 0.05)))
    # the three bridges
    out += bridge_along(PZ, MZ, 0.0, 2.75, "z")
    out += bridge_along(-PZ, -MZ, 0.0, 2.75, "z")
    out += bridge_along(-XF, -MX, 0.0, 2.2, "x")
    return out


FOOTPRINT = [(-MX - 3.6, -MZ - 3.6), (MX + 3.6, -MZ - 3.6), (MX + 3.6, MZ + 3.6), (-MX - 3.6, MZ + 3.6)]


def main():
    pal = merge(palace())
    mo = merge(moat())
    named = []
    for name, m in (("palace", pal), ("moat", mo)):
        k = 0.96 + 0.04 * fbm(m.v, 0.08, 11, 2)
        m.c = np.clip(m.c * k[:, None], 0, 1)
        named.append((name, m))
    out = ROOT / "models" / "Q1368571"
    out.mkdir(parents=True, exist_ok=True)
    tris, size = write_glb(out / "model.glb", named, roughness=0.88)
    allv = np.vstack([m.v for _, m in named])
    lo, hi = allv.min(0), allv.max(0)
    meta = out / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        mj["footprint"] = [[round(x, 1), round(z, 1)] for x, z in FOOTPRINT]
        meta.write_text(json.dumps(mj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline=chr(10))
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
