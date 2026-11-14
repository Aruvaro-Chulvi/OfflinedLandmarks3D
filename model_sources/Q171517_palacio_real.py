"""
Palacio Real de Madrid (Q171517) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Filippo Juvarra (design 1735, died 1736), Giovanni Battista Sacchetti (built 1738–1755/1764,
died 1764), Ventura Rodríguez, Francesco Sabatini (south-east corner «de San Gil», died 1797); the galleries
closing the Plaza de la Armería by Enrique María Repullés (1890s). Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 30399 (wikidata Q171517), traced from aerial imagery (±0.5–1 m). Edges run at 4.6°
  from north (length-weighted histogram of the outline); the main façade on the Plaza de la Armería faces
  184.6°. The relation is a U: the palace block (about 130 × 129 m with corner pavilions, the Patio del
  Príncipe 51 × 51 m in the middle) and the two galleries closing the Plaza de la Armería to the south
  (27 m deep, 145 m long). The outline below is that polygon squared to the model axes (kept within ~1 m;
  the 1–1.4 m bumps of the column bodies are modelled as the columns themselves).
- Heights measured on near-orthographic photos of the south façade (Wikimedia Commons, «Plaza de la Armeria -
  Palacio Real - Madrid 01.jpg»), scale from the façade width (OSM), ±5 %: rusticated granite base with two rows
  of windows 10.5 m, giant order (Ionic columns on the central body and the corner pavilions, pilasters
  elsewhere) 11.3–23.2 m with three rows of windows (main floor with balconies and alternating pediments,
  a mezzanine, the second floor), entablature to 25.0, cornice top 26.3, balustrade to 28.7, vases and statues
  to 30.5–31.5, the clock attic of the main façade to 34.6 and its bell frame to ~41 m. Galleries of the plaza:
  one tall arcaded storey, 12.9 m to the top of their balustrade.
- West façade over the Campo del Moro (photos «Palacio Real, Campo del Moro, fachada, Madrid, España, 2015.jpg»,
  «Royal Palace Madrid and Campo del Moro.JPG»): the same orders, and below the plaza level about 12 m of
  rusticated granite with small windows and a further ~11 m of substructure; the walls go down to y = −24 m
  on every side (buried where the ground is higher), the west gallery to −20 m.
- The chapel dome (Capilla Real, Sacchetti/Ventura Rodríguez) rises over the middle of the north wing;
  its size is estimated (drum Ø 14 m, lantern top ~47 m), not measured.
Roofs: low lead-grey hips behind the balustrade; the galleries have flat terraces.

Axes (docs/MODEL_SPEC.md): +Y up, the south façade (Plaza de la Armería) faces +Z, +X points east (94.6°).
Origin on the ground (plaza level) at the centroid of the palace block (not of the whole U, whose centroid
falls on the south wing), y = 0 at the level of the Plaza de la Armería and of the Patio del Príncipe.

    python model_sources/Q171517_palacio_real.py   -> models/Q171517/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, quad, tube, extrude, walls, cap, offset, write_glb  # noqa: E402

rng = np.random.default_rng(1764)

# ── Colours ──────────────────────────────────────────────────────────────────
GRANITE = (0.72, 0.70, 0.66)        # rusticated base (Guadarrama granite)
GRANITE_D = (0.62, 0.605, 0.575)       # foundations, substructure
STONE = (0.88, 0.855, 0.79)           # Colmenar limestone walls
WHITE = (0.93, 0.915, 0.87)           # orders, cornices, frames, sculpture
LEAD = (0.45, 0.48, 0.51)             # roofs
LEAD_D = (0.38, 0.41, 0.44)           # flat lead between the roofs
TERRACE = (0.70, 0.69, 0.66)          # flat terraces of the galleries
PAVING = (0.74, 0.72, 0.68)           # Patio del Príncipe
GLASS = (0.15, 0.17, 0.20)            # windows
DARK = (0.09, 0.09, 0.10)             # open arcades
DOOR = (0.80, 0.78, 0.72)            # the great doors are painted white
IRON = (0.12, 0.12, 0.13)
RED, YELLOW = (0.70, 0.10, 0.10), (0.95, 0.75, 0.10)

# ── Heights (y = 0 at the Plaza de la Armería) ───────────────────────────────
FOUND = -24.0          # deepest foundations (west side over the Campo del Moro)
Y_BASE = 10.5          # top of the rusticated base
Y_BAND = 11.3          # top of the band where the giant order starts
Y_CAP = 22.4           # pilaster capitals
Y_ENT = 23.2           # entablature
Y_BED = 25.0
Y_COR = 25.5
Y_TOP = 26.3           # top of the cornice
Y_ROOF = 26.25         # flat lead under the hips
Y_PL, Y_BAL, Y_RAIL = 26.9, 28.3, 28.7      # balustrade: plinth top, balusters top, rail top
Y_RIDGE = 29.0
# galleries of the plaza
A_BAND, A_COR0, A_TOP = 9.4, 10.6, 11.2
A_PL, A_BAL, A_RAIL = 11.6, 12.6, 12.9

# ── Plan (OSM relation 30399 in the axes of heading 184.6, origin at the centroid of the whole relation;
#    shifted to the centroid of the palace block at the end) ────────────────────────────────────────────
SHIFT = (2.324, 42.892)
MAIN = [(-67.4, -108.0), (-43.9, -108.0), (-43.9, -102.6), (-19.1, -102.6), (-19.1, -108.0), (13.8, -108.0),
        (13.8, -102.8), (38.6, -102.8), (38.6, -108.2), (62.2, -108.2), (62.2, -82.5), (55.9, -82.5),
        (55.9, -1.2), (62.6, -1.2), (62.6, 21.0), (40.5, 21.0), (40.5, 20.0), (-45.0, 20.0), (-45.0, 21.0),
        (-67.1, 21.0), (-67.1, -0.7), (-60.5, -0.7), (-60.5, -82.0), (-67.4, -82.0)]
COURT = [(-28.1, -72.6), (23.1, -72.6), (23.1, -21.6), (-28.1, -21.6)]          # Patio del Príncipe
EAST_ARM = [(52.7, 19.7), (76.4, 19.7), (76.4, 21.3), (80.0, 21.3), (80.0, 166.2),
            (52.7, 166.3)]
WEST_ARM = [(-84.2, 20.4), (-56.6, 20.4),
            (-56.6, 166.6), (-83.6, 166.6), (-83.6, 122.0), (-64.2, 122.0), (-64.2, 82.0), (-84.2, 82.0)]
FOOTPRINT = [(-67.4, -108.0), (-43.9, -108.0), (-43.9, -102.6), (-19.1, -102.6), (-19.1, -108.0), (13.8, -108.0),
             (13.8, -102.8), (38.6, -102.8), (38.6, -108.2), (62.2, -108.2), (62.2, -82.5), (55.9, -82.5),
             (55.9, -1.2), (62.6, -1.2), (62.6, 19.7), (76.4, 19.7), (76.4, 21.3), (80.0, 21.3), (80.0, 166.2),
             (52.7, 166.3), (52.7, 21.0), (40.5, 21.0), (40.5, 20.0), (-45.0, 20.0), (-45.0, 21.0), (-56.6, 21.0),
             (-56.6, 166.6), (-83.6, 166.6), (-83.6, 122.0), (-64.2, 122.0), (-64.2, 82.0), (-84.2, 82.0),
             (-84.2, 20.4), (-67.1, 20.4), (-67.1, -0.7), (-60.5, -0.7), (-60.5, -82.0), (-67.4, -82.0)]
AXIS_X = -2.25         # axis of the south façade (centre of its four columns)
DOME = (-2.6, -88.0)   # Capilla Real, middle of the north wing


def tint(c, k=0.012):
    return tuple(np.clip(np.array(c) * (1 + rng.normal(0, k)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    """Axis-aligned box from its extents."""
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sphere(r, c, seg=10, rings=6):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def prism(pts, w0, w1, c):
    """A convex profile pts [(u, y)] in the façade plane, extruded from w0 to w1 (outwards)."""
    n = len(pts)
    if sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n)) < 0:
        pts = pts[::-1]                                                  # counter-clockwise in (u, y)
    V, F = [], []
    for w in (w0, w1):
        V += [(u, y, w) for u, y in pts]
    F += [(n, n + i, n + i + 1) for i in range(1, n - 1)]                # front (w1), facing +w
    F += [(0, i + 1, i) for i in range(1, n - 1)]                        # back (w0), facing -w
    m = Mesh(V, F, c)
    sides = []
    for i in range(n):
        (u0, y0), (u1, y1) = pts[i], pts[(i + 1) % n]
        sides.append(Mesh([(u0, y0, w0), (u1, y1, w0), (u1, y1, w1), (u0, y0, w1)], [(0, 1, 2), (0, 2, 3)], c))
    return merge([m] + sides)


def arch_pts(u, y0, ys, half, seg=8):
    """Opening with a semicircular head: bottom y0, springing ys, half-width half (convex, counter-clockwise)."""
    pts = [(u - half, y0), (u + half, y0)]
    pts += [(u + half * math.cos(a), ys + half * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)]
    return pts


def flat(pts, w, c):
    """A flat convex polygon in the plane w, facing +w."""
    n = len(pts)
    V = [(u, y, w) for u, y in pts]
    F = [(0, i, i + 1) for i in range(1, n - 1)]
    m = Mesh(V, F, c)
    if sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n)) < 0:
        m.f = m.f[:, ::-1]
    return m


# ── Façade frame: build in local (u along the façade, y up, w outwards), then place ─────────────────────
def place(m, face, d):
    if face == "S":
        return m.T(0, 0, d)
    if face == "N":
        return m.R(0, math.pi, 0).T(0, 0, d)
    if face == "E":
        return m.R(0, math.pi / 2, 0).T(d, 0, 0)
    return m.R(0, -math.pi / 2, 0).T(d, 0, 0)                       # "W"


def local_range(face, a, b):
    return (a, b) if face in ("S", "W") else (-b, -a)


def to_u(face, s):
    return s if face in ("S", "W") else -s


# ── Pieces of the elevations (local coordinates) ────────────────────────────
def window(u, y0, y1, w, fc=WHITE, t=0.28, depth=0.24, pane_w=0.04, sill=True):
    """Window seen as recessed: a projecting stone surround and the dark glass behind it."""
    hw = w / 2
    out = [quad(u, (y0 + y1) / 2, 0.0, w, y1 - y0, "+z", GLASS, eps=pane_w),
           B(u - hw - t, u - hw, y0, y1 + t, -0.05, depth, fc),
           B(u + hw, u + hw + t, y0, y1 + t, -0.05, depth, fc),
           B(u - hw, u + hw, y1, y1 + t, -0.05, depth, fc)]
    if sill:
        out.append(B(u - hw - t - 0.1, u + hw + t + 0.1, y0 - 0.22, y0, -0.05, depth + 0.1, fc))
    return out


def railing(u0, u1, y0, y1, w, n=7):
    """Iron balcony railing: top and bottom bars and a few balusters, as flat strips facing out."""
    out = [quad((u0 + u1) / 2, y1 - 0.04, w, u1 - u0, 0.08, "+z", IRON, eps=0.0),
           quad((u0 + u1) / 2, y0 + 0.1, w, u1 - u0, 0.07, "+z", IRON, eps=0.0)]
    for u in np.linspace(u0 + 0.05, u1 - 0.05, n):
        out.append(quad(u, (y0 + y1) / 2, w, 0.06, y1 - y0, "+z", IRON, eps=0.0))
    return out


def small_window(u, y0, y1, w, fc=WHITE, pane_w=0.11):
    return [B(u - w / 2 - 0.18, u + w / 2 + 0.18, y0 - 0.18, y1 + 0.18, -0.05, 0.08, fc),
            quad(u, (y0 + y1) / 2, 0.0, w, y1 - y0, "+z", GLASS, eps=pane_w)]


def pediment(u, yb, w, kind, c=WHITE):
    hw = w / 2
    if kind == 0:                                              # triangular
        pts = [(u - hw, yb), (u + hw, yb), (u, yb + 0.95)]
    else:                                                      # segmental
        pts = [(u - hw, yb), (u + hw, yb)] + [(u + hw * math.cos(a), yb + 0.75 * math.sin(a))
                                               for a in np.linspace(0.25, math.pi - 0.25, 7)]
    return [prism(pts, -0.05, 0.42, c), B(u - hw - 0.08, u + hw + 0.08, yb - 0.18, yb, -0.05, 0.48, c)]


def pilaster(u, y0, y1, w=1.1, c=STONE, cap_c=WHITE):
    return [B(u - w / 2, u + w / 2, y0, y1, -0.05, 0.32, c),
            B(u - w / 2 - 0.1, u + w / 2 + 0.1, y0, y0 + 0.45, -0.05, 0.42, cap_c),
            B(u - w / 2 - 0.12, u + w / 2 + 0.12, y1, y1 + 0.8, -0.05, 0.46, cap_c)]


R_COL = 0.62


def column(u, y0=Y_BAND, ytop=Y_ENT):
    """Engaged Ionic column of the giant order with the entablature breaking forward over it."""
    wc = R_COL + 0.05
    out = [cyl(R_COL * 1.3, R_COL * 1.3, 0.25, WHITE, 10).T(u, y0, wc),
           cyl(R_COL * 1.18, R_COL * 1.1, 0.3, WHITE, 10).T(u, y0 + 0.25, wc),
           cyl(R_COL, R_COL * 0.88, ytop - 0.95 - y0 - 0.55, tint(STONE, 0.01), 12, caps=False).T(u, y0 + 0.55, wc),
           cyl(R_COL * 0.95, R_COL * 1.12, 0.3, WHITE, 10).T(u, ytop - 0.95, wc),
           B(u - 0.85, u + 0.85, ytop - 0.65, ytop, wc - 0.85, wc + 0.85, WHITE)]
    for s in (-1, 1):                                                            # volutes
        out.append(cyl(0.2, 0.2, 1.5, WHITE, 8).R(math.pi / 2, 0, 0).T(u + s * 0.7, ytop - 0.78, wc + 0.75))
    out.append(B(u - 0.95, u + 0.95, Y_ENT, Y_BED, -0.05, 1.5, tint(STONE, 0.005)))   # entablature break
    out.append(B(u - 1.15, u + 1.15, Y_BED, Y_COR + 0.02, -0.05, 1.75, WHITE))
    out.append(B(u - 1.3, u + 1.3, Y_COR, Y_TOP + 0.03, -0.05, 2.05, WHITE))
    return out


def figure(u, y, w, h, c=WHITE):
    """A stylised standing statue (robe, head, an arm) on the balustrade."""
    prof = [(0.17 * h, 0), (0.15 * h, 0.3 * h), (0.11 * h, 0.55 * h), (0.1 * h, 0.72 * h), (0.05 * h, 0.79 * h), (0, 0.8 * h)]
    return [lathe(prof, c, 10).T(u, y, w), sphere(0.075 * h, c, 8, 5).T(u, y + 0.87 * h, w),
            tube([(u + 0.08 * h, y + 0.7 * h, w), (u + 0.17 * h, y + 0.55 * h, w + 0.08 * h)], 0.035 * h, c, 6)]


def vase(u, y, w, c=WHITE, s=1.0):
    prof = [(0.0, 0), (0.3, 0), (0.22, 0.18), (0.38, 0.6), (0.24, 1.0), (0.13, 1.12), (0.2, 1.32), (0.0, 1.45)]
    return [lathe([(r * s, h * s) for r, h in prof], c, 8).T(u, y, w)]


def royal_arms(u, y, w, c=WHITE, k=1.0):
    """Coat of arms with its crown, with a lion on each side (stylised)."""
    out = [cyl(1.0 * k, 1.0 * k, 0.5, c, 16).R(math.pi / 2, 0, 0).S(1, 1.3, 1).T(u, y + 1.4 * k, w + 0.5),
           cyl(0.7 * k, 0.8 * k, 0.5 * k, c, 12).T(u, y + 2.75 * k, w), sphere(0.55 * k, c, 10, 5).S(1, 0.7, 1).T(u, y + 3.3 * k, w),
           B(u - 0.06, u + 0.06, y + 3.6 * k, y + 4.2 * k, w - 0.06, w + 0.06, c),
           B(u - 0.25 * k, u + 0.25 * k, y + 3.9 * k, y + 4.0 * k, w - 0.06, w + 0.06, c),
           B(u - 1.6 * k, u + 1.6 * k, y, y + 0.4, w - 0.6, w + 0.6, c)]
    for s in (-1, 1):
        out.append(lathe([(0.45 * k, 0), (0.4 * k, 0.6 * k), (0.25 * k, 1.1 * k), (0, 1.25 * k)], c, 8).S(1, 1, 0.8).T(u + s * 1.5 * k, y + 0.4, w))
        out.append(sphere(0.3 * k, c, 8, 5).T(u + s * 1.5 * k, y + 1.75 * k, w + 0.1))
    return out


def baluster_row(u0, u1, y0, y1, w, step=0.42, c=WHITE):
    n = max(1, int((u1 - u0) / step))
    out = []
    for i in range(n):
        u = u0 + (i + 0.5) * (u1 - u0) / n
        out.append(cyl(0.12, 0.12, y1 - y0, c, 4, caps=False).R(0, math.pi / 4, 0).T(u, y0, w))
    return out


# ── Elevations ───────────────────────────────────────────────────────────────
def bays(ua, ub, n):
    step = (ub - ua) / n
    return [ua + step * (i + 0.5) for i in range(n)], [ua + step * i for i in range(n + 1)]


def upper_windows(u, k, main=True):
    """Main floor (balcony, pediment), mezzanine and second floor of one bay."""
    out = []
    if main:
        out += window(u, 11.9, 15.9, 2.0)
        out += pediment(u, 16.25, 2.9, k % 2)
        out.append(B(u - 1.5, u + 1.5, 11.32, 11.62, -0.05, 0.85, WHITE))                     # balcony
        out += railing(u - 1.42, u + 1.42, 11.62, 12.55, 0.8)                                  # its railing
    out += small_window(u, 17.5, 18.7, 1.35)
    out += window(u, 19.5, 21.8, 1.7, t=0.24)
    return out


def base_windows(u, deep=False, door=False):
    out = []
    if door:
        out += [flat(arch_pts(u, 0.0, 4.6, 1.7), 0.14, DOOR), prism(arch_pts(u, 0.0, 4.6, 2.05), -0.05, 0.11, WHITE)]
        out += [B(u - 0.35, u + 0.35, 6.2, 7.2, -0.05, 0.4, WHITE)]                             # keystone
    else:
        out += window(u, 2.4, 5.6, 1.9, fc=GRANITE, pane_w=0.11)
        out += small_window(u, 7.0, 8.3, 1.4, fc=GRANITE, pane_w=0.11)
        out.append(B(u - 0.85, u + 0.85, 2.4, 5.5, 0.12, 0.16, IRON))                           # grille
    if deep:                                                                                     # below the plaza
        out += small_window(u, -4.6, -2.8, 1.3, fc=GRANITE_D, pane_w=0.11)
        out += small_window(u, -9.4, -7.6, 1.3, fc=GRANITE_D, pane_w=0.11)
    return out


def top_items(u, kind, face, statue=False):
    """Pedestal on the balustrade over a pilaster or column, with its vase or statue."""
    out = [B(u - 0.5, u + 0.5, Y_TOP, Y_RAIL + 0.3, -0.25, 0.55, WHITE)]
    if statue:
        out += figure(u, Y_RAIL + 0.3, 0.15, 2.5)
    else:
        out += vase(u, Y_RAIL + 0.3, 0.15)
    return out


def facade(face, d, a, b, kind, n=None, deep=False, doors=(), top=True):
    """One straight run of the palace's outer elevations."""
    ua, ub = local_range(face, a, b)
    L = ub - ua
    out = []
    if kind == "ret":                                          # short return of a pavilion: one window column
        c = (ua + ub) / 2
        out += base_windows(c, deep) + upper_windows(c, 1)
        out += baluster_row(ua + 0.6, ub - 0.6, Y_PL, Y_BAL, 0.1)
        return place(merge(out), face, d)
    if kind in ("pav", "centre"):
        e = 0.8 if kind == "pav" else 0.0
        if kind == "pav":
            out += pilaster(ua + e, Y_BAND, Y_CAP, 1.3) + pilaster(ub - e, Y_BAND, Y_CAP, 1.3)
            cols = list(np.linspace(ua + 2.4, ub - 2.4, (n or 3) + 1))
        else:
            cols = list(np.linspace(ua + 1.4, ub - 1.4, (n or 3) + 1))
        centres = [(cols[i] + cols[i + 1]) / 2 for i in range(len(cols) - 1)]
        for c in cols:
            out += column(c)
        for k, c in enumerate(centres):
            out += base_windows(c, deep, door=(k in doors)) + upper_windows(c, k)
        ped = cols + ([ua + e, ub - e] if kind == "pav" else [])
        stat = (face == "S") or kind == "centre"
        for c in ped:
            out += top_items(c, kind, face, statue=stat and c in cols)
        posts = sorted(ped + [ua, ub])
    else:                                                      # body: pilasters between the bays
        n = n or max(1, round(L / 5.3))
        centres, edges = bays(ua, ub, n)
        edges = [ua + 0.75] + edges[1:-1] + [ub - 0.75]
        for c in edges:
            out += pilaster(c, Y_BAND, Y_CAP)
            out += top_items(c, kind, face)
        for k, c in enumerate(centres):
            out += base_windows(c, deep) + upper_windows(c, k)
        posts = sorted(edges + [ua, ub])
    for p, q in zip(posts[:-1], posts[1:]):                    # balusters between the pedestals
        if q - p > 1.2:
            out += baluster_row(p + 0.5, q - 0.5, Y_PL, Y_BAL, 0.1)
    return place(merge(out), face, d)


def court_facade(face, d, a, b, n=9):
    """Patio del Príncipe: arcades on the ground floor, the glazed arcade of the main floor, two rows above."""
    ua, ub = local_range(face, a, b)
    centres, edges = bays(ua, ub, n)
    out = []
    for k, c in enumerate(centres):
        out += [flat(arch_pts(c, 0.0, 4.4, 1.95), 0.05, DARK),
                B(c - 2.35, c - 1.95, 4.25, 4.55, -0.05, 0.2, WHITE), B(c + 1.95, c + 2.35, 4.25, 4.55, -0.05, 0.2, WHITE)]
        out.append(B(c - 0.3, c + 0.3, 6.0, 6.7, -0.05, 0.3, WHITE))                         # keystone
        out += [flat(arch_pts(c, 12.0, 14.9, 1.55), 0.05, GLASS), B(c - 1.9, c + 1.9, 11.5, 12.0, -0.05, 0.3, WHITE)]
        out += small_window(c, 17.5, 18.6, 1.3)
        out += window(c, 19.5, 21.7, 1.6, t=0.22)
    for c in edges[1:-1]:
        out += pilaster(c, Y_BAND, Y_CAP, 1.0)
        out.append(B(c - 0.45, c + 0.45, 0.0, 7.4, -0.05, 0.25, GRANITE))                      # pier strip
        out.append(B(c - 0.45, c + 0.45, Y_TOP, Y_RAIL + 0.3, -0.25, 0.55, WHITE))
    posts = [ua] + edges[1:-1] + [ub]
    for p, q in zip(posts[:-1], posts[1:]):
        out += baluster_row(p + 0.6, q - 0.6, Y_PL, Y_BAL, 0.1)
    return place(merge(out), face, d)


def gallery_facade(face, d, a, b, kind, deep=0.0, step=6.1):
    """Galleries of the Plaza de la Armería: an arcade on the plaza side, windows outside."""
    ua, ub = local_range(face, a, b)
    L = ub - ua
    n = max(1, round(L / step))
    centres, edges = bays(ua, ub, n)
    out = []
    for k, c in enumerate(centres):
        if kind == "arcade":
            hw = min(2.15, (ub - ua) / n / 2 - 0.9)
            out += [flat(arch_pts(c, 0.0, 6.0, hw), 0.05, DARK), B(c - 0.3, c + 0.3, 6.0 + hw - 0.1, 6.0 + hw + 0.9, -0.05, 0.3, WHITE)]
            out.append(B(c - hw - 0.25, c + hw + 0.25, 5.8, 6.05, -0.05, 0.16, WHITE))
        else:
            if (ub - ua) / n > 3.2:
                out += window(c, 1.8, 4.6, 1.7, fc=WHITE, pane_w=0.04)
                out += window(c, 6.0, 8.4, 1.6, fc=WHITE, pane_w=0.04, t=0.22)
            if deep > 4:
                out += small_window(c, -4.4, -2.6, 1.3, fc=GRANITE_D)
            if deep > 9:
                out += small_window(c, -9.4, -7.4, 1.3, fc=GRANITE_D)
    for c in edges[1:-1]:
        out.append(B(c - 0.45, c + 0.45, 0.0, A_BAND, -0.05, 0.3, WHITE if kind == "arcade" else GRANITE))
        out.append(B(c - 0.4, c + 0.4, A_TOP, A_RAIL + 0.2, -0.2, 0.4, WHITE))
    posts = [ua] + edges[1:-1] + [ub]
    for p, q in zip(posts[:-1], posts[1:]):
        out += baluster_row(p + 0.5, q - 0.5, A_PL, A_BAL, 0.1)
    return place(merge(out), face, d)


# ── Volumes ──────────────────────────────────────────────────────────────────
_RING = [0]


def ring(outer_d, inner_d, y0, y1, c, poly=MAIN, top=True):
    """A band around the building between offset(poly, outer_d) and offset(poly, inner_d). The inner edge
    (hidden in the wall) is staggered by 1 cm per band so that no two hidden faces coincide."""
    _RING[0] += 1
    inner_d -= 0.01 * (_RING[0] % 40)
    return extrude(offset(poly, outer_d), y0, y1, c, holes=[offset(poly, inner_d)], top=top)


def court_ring(grow, shrink, y0, y1, c, top=True):
    """A band around the courtyard, projecting into it: between offset(COURT, +grow) and offset(COURT, -shrink)."""
    return extrude(offset(COURT, grow), y0, y1, c, holes=[offset(COURT, -shrink)], top=top)


def quad_roof(P, Q, y0, y1, c):
    """Planar faces between an outer rectangle P (at y0) and a ridge rectangle Q (at y1), both [(x, z)]×4, ccw."""
    out = []
    for i in range(4):
        a, b, cq, dq = P[i], P[(i + 1) % 4], Q[(i + 1) % 4], Q[i]
        V = [(a[0], y0, a[1]), (b[0], y0, b[1]), (cq[0], y1, cq[1]), (dq[0], y1, dq[1])]
        m = Mesh(V, [(0, 2, 1), (0, 3, 2)], c)
        p = m.v
        if np.cross(p[2] - p[0], p[1] - p[0])[1] < 0:            # keep every roof face facing up
            m.f = m.f[:, ::-1]
        out.append(m)
    return merge(out)


def rect(x0, x1, z0, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def hip(x0, x1, z0, z1, y0, y1, c):
    """Hipped roof over a rectangle, ridge along its longer side."""
    if x1 - x0 >= z1 - z0:
        d = (z1 - z0) / 2
        Q = rect(x0 + d * 0.999, x1 - d * 0.999, z0 + d * 0.999, z1 - d * 0.999)
    else:
        d = (x1 - x0) / 2
        Q = rect(x0 + d * 0.999, x1 - d * 0.999, z0 + d * 0.999, z1 - d * 0.999)
    return quad_roof(rect(x0, x1, z0, z1), Q, y0, y1, c)


def palace():
    parts = []
    # walls: granite base (with the substructure below the plaza), limestone above
    parts.append(extrude(MAIN, FOUND, 0.0, GRANITE_D, top=False))
    parts.append(extrude(MAIN, 0.0, Y_BASE, GRANITE, holes=[COURT], top=False))
    parts.append(extrude(MAIN, Y_BASE, Y_ROOF, STONE, holes=[COURT], top_c=LEAD_D))
    parts.append(cap(COURT, 0.03, PAVING))
    # rustication: courses of the base, below the plaza on the west and north too
    for y in list(np.arange(-12.0, -0.9, 0.86)) + list(np.arange(0.12, Y_BASE - 0.5, 0.86)):
        parts.append(ring(0.07, -0.2, y, y + 0.72, tint(GRANITE if y > 0 else GRANITE_D, 0.02), top=True))
    parts.append(ring(0.25, -0.2, -12.6, -12.0, GRANITE_D))                     # string course of the substructure
    parts.append(ring(0.18, -0.2, -0.5, 0.0, GRANITE_D))                        # plinth at the plaza level
    parts.append(ring(0.32, -0.2, Y_BASE, Y_BAND, WHITE))                       # band under the order
    parts.append(ring(0.2, -0.2, Y_ENT, Y_BED, tint(STONE, 0.004)))             # architrave and frieze
    parts.append(ring(0.55, -0.2, Y_BED, Y_COR, WHITE))                         # bed mould
    parts.append(ring(1.0, -0.3, Y_COR, Y_TOP, WHITE))                          # corona
    parts.append(ring(0.42, -0.2, Y_TOP, Y_PL, WHITE))                          # balustrade plinth
    parts.append(ring(0.45, -0.22, Y_BAL, Y_RAIL, WHITE))                       # its rail
    # courtyard bands
    parts.append(court_ring(0.2, 0.3, Y_BASE, Y_BAND, WHITE))
    parts.append(court_ring(0.2, 0.2, Y_ENT, Y_BED, tint(STONE, 0.004)))
    parts.append(court_ring(0.3, 0.8, Y_BED, Y_TOP, WHITE))
    parts.append(court_ring(0.2, 0.42, Y_TOP, Y_PL, WHITE))
    parts.append(court_ring(0.22, 0.45, Y_BAL, Y_RAIL, WHITE))

    # outer elevations — (face, plane, from, to, kind, bays)
    S, N, E, W = "S", "N", "E", "W"
    runs = [
        (N, -108.0, -67.4, -43.9, "pav", 3), (E, -43.9, -108.0, -102.6, "ret", None),
        (N, -102.6, -43.9, -19.1, "body", 5), (W, -19.1, -108.0, -102.6, "ret", None),
        (N, -108.0, -19.1, 13.8, "centre", 5), (E, 13.8, -108.0, -102.8, "ret", None),
        (N, -102.8, 13.8, 38.6, "body", 5), (W, 38.6, -108.2, -102.8, "ret", None),
        (N, -108.2, 38.6, 62.2, "pav", 3),
        (E, 62.2, -108.2, -82.5, "pav", 3), (S, -82.5, 55.9, 62.2, "ret", None),
        (E, 55.9, -82.5, -51.4, "body", 6), (E, 55.9, -51.4, -32.4, "centre", 3), (E, 55.9, -32.4, -1.2, "body", 6),
        (N, -1.2, 55.9, 62.6, "ret", None), (E, 62.6, -1.2, 19.7, "pav", 3),
        (S, 21.0, 40.5, 62.6, "pav", 3), (S, 20.0, 7.9, 40.5, "body", 6), (S, 20.0, -45.0, -12.4, "body", 6),
        (S, 21.0, -67.1, -45.0, "pav", 3),
        (W, -67.1, -0.7, 21.0, "pav", 3), (N, -0.7, -67.1, -60.5, "ret", None),
        (W, -60.5, -31.7, -0.7, "body", 6), (W, -60.5, -51.1, -31.7, "centre", 3), (W, -60.5, -82.0, -51.1, "body", 6),
        (S, -82.0, -67.4, -60.5, "ret", None), (W, -67.4, -108.0, -82.0, "pav", 3),
    ]
    for face, d, a, b, kind, n in runs:
        deep = face in ("W", "N")
        parts.append(facade(face, d, a, b, kind, n, deep=deep))
    # the central body of the main façade: four columns on the bumps of the OSM outline, three doors
    cent = []
    cols = [-11.0, -5.45, 1.0, 6.55]
    for c in cols:
        cent += column(c)
        cent += [B(c - 0.5, c + 0.5, Y_TOP, Y_RAIL + 0.3, -0.25, 0.55, WHITE)] + figure(c, Y_RAIL + 0.3, 0.15, 2.6)
    for k, c in enumerate([-8.2, -2.2, 3.8]):
        cent += base_windows(c, door=True)
        cent += upper_windows(c, 1 if k != 1 else 0)
    cent += baluster_row(-10.4, -6.05, Y_PL, Y_BAL, 0.1) + baluster_row(1.6, 5.95, Y_PL, Y_BAL, 0.1)
    for c in (-12.4 + 0.55, 7.9 - 0.55):
        cent += pilaster(c, Y_BAND, Y_CAP)
    cent += clock_attic(AXIS_X)
    parts.append(place(merge(cent), "S", 20.0))
    # sculptural groups over the south pavilions, arms over the other central bodies
    for x in (-56.05, 51.55):
        parts.append(place(merge(royal_arms(x, Y_RAIL + 0.3, 0.0, k=0.9)), "S", 21.0))
    parts.append(place(merge(royal_arms(to_u("E", -41.9), Y_RAIL + 0.3, 0.0)), "E", 55.9))
    parts.append(place(merge(royal_arms(to_u("W", -41.4), Y_RAIL + 0.3, 0.0)), "W", -60.5))
    parts.append(place(merge(royal_arms(to_u("N", -2.65), Y_RAIL + 0.3, 0.0)), "N", -108.0))

    # Patio del Príncipe
    parts.append(court_facade("N", -21.6, -28.1, 23.1))       # south wing, its wall faces north into the courtyard
    parts.append(court_facade("S", -72.6, -28.1, 23.1))
    parts.append(court_facade("W", 23.1, -72.6, -21.6))
    parts.append(court_facade("E", -28.1, -72.6, -21.6))
    for x, z in ((-24.6, -69.1), (19.6, -69.1), (-24.6, -25.1), (19.6, -25.1)):     # the four emperors
        parts.append(B(x - 0.7, x + 0.7, 0.0, 1.7, z - 0.7, z + 0.7, WHITE))
        parts.append(merge(figure(0, 1.7, 0, 2.3)).T(x, 0, z))

    # roofs: a ring of low hips round the courtyard, hips over the pavilions and projections
    P = rect(-59.4, 54.8, -101.5, 18.9)[::-1]
    Q = rect(-44.0, 39.5, -87.4, -0.9)[::-1]
    Ci = rect(-29.2, 24.2, -73.7, -20.5)[::-1]
    parts.append(quad_roof(P, Q, Y_ROOF + 0.02, Y_RIDGE, LEAD))
    parts.append(quad_roof(Ci, Q, Y_ROOF + 0.02, Y_RIDGE, LEAD))
    for r in ((-66.3, -45.0, -106.9, -83.1), (39.7, 61.1, -107.1, -83.6), (-18.0, 12.7, -106.9, -101.0),
              (-66.0, -46.1, -1.8, 19.9), (41.6, 61.5, -2.3, 19.9)):
        parts.append(hip(*r, Y_ROOF + 0.04, Y_RIDGE - 0.3, tint(LEAD, 0.02)))
    parts.append(chapel_dome(*DOME))
    return merge(parts)


def clock_attic(u0):
    """The attic over the main door: clock, arms and statues, the bell frame and the flag."""
    out = []
    a0, a1 = u0 - 8.0, u0 + 8.0
    out.append(B(a0, a1, Y_TOP, 32.6, -5.0, 0.6, STONE))
    out.append(B(a0 - 0.3, a1 + 0.3, 32.6, 33.4, -5.3, 0.9, WHITE))
    for u in (a0 + 0.6, a0 + 4.6, a1 - 4.6, a1 - 0.6):                                       # pilasters
        out.append(B(u - 0.5, u + 0.5, Y_TOP + 0.02, 32.62, -0.05, 0.85, WHITE))
    out.append(cyl(1.7, 1.7, 0.3, WHITE, 24).R(math.pi / 2, 0, 0).T(u0, 30.0, 0.9))           # clock
    out.append(cyl(1.35, 1.35, 0.1, (0.95, 0.94, 0.9), 24).R(math.pi / 2, 0, 0).T(u0, 30.0, 1.0))
    out += [B(u0 - 0.05, u0 + 0.05, 30.0, 31.0, 1.05, 1.1, IRON), B(u0, u0 + 0.75, 29.95, 30.05, 1.05, 1.1, IRON)]
    out += baluster_row(a0 + 0.3, a1 - 0.3, 33.4, 34.3, 0.4)
    out.append(B(a0, a1, 34.3, 34.6, 0.1, 0.7, WHITE))
    for u in (a0 + 0.3, a1 - 0.3):
        out += figure(u, 33.4, 0.4, 2.2)
    out += royal_arms(u0, 33.4, -0.4, k=1.0)
    # the bell frame: two iron posts and an arch, the bells, a crown on top
    for s in (-1, 1):
        out.append(B(u0 + s * 1.6 - 0.12, u0 + s * 1.6 + 0.12, 37.5, 40.0, -1.62, -1.38, IRON))
    out.append(tube([(u0 + 1.6 * math.cos(t), 40.0 + 1.0 * math.sin(t), -1.5) for t in np.linspace(0, math.pi, 9)], 0.1, IRON, 6))
    out.append(lathe([(0.55, 0), (0.45, 0.25), (0.32, 0.7), (0.0, 0.85)], (0.35, 0.3, 0.2), 10).T(u0, 38.2, -1.5))
    out.append(sphere(0.35, IRON, 8, 5).T(u0, 41.3, -1.5))
    out.append(B(u0 - 0.05, u0 + 0.05, 41.5, 42.4, -1.55, -1.45, IRON))
    # flag on the attic
    fu = u0 - 6.0
    out.append(cyl(0.07, 0.06, 9.0, IRON, 6).T(fu, 33.4, -2.0))
    out += [B(fu + 0.05, fu + 2.7, 40.5, 40.95, -2.03, -1.97, RED), B(fu + 0.05, fu + 2.7, 39.6, 40.5, -2.03, -1.97, YELLOW),
            B(fu + 0.05, fu + 2.7, 39.15, 39.6, -2.03, -1.97, RED)]
    return out


def chapel_dome(x, z):
    """Capilla Real: drum with windows, lead dome, lantern."""
    out = [cyl(7.0, 7.0, 34.0 - Y_ROOF, STONE, 32).T(x, Y_ROOF, z),
           cyl(7.5, 7.5, 0.8, WHITE, 32).T(x, 34.0, z)]
    for k in range(8):
        a = k * math.pi / 4 + math.pi / 8
        win = merge([B(-0.75, 0.75, 29.6, 32.6, -0.2, 0.25, GLASS), B(-1.05, 1.05, 32.6, 33.0, -0.2, 0.35, WHITE)])
        out.append(win.R(0, math.pi / 2 - a, 0).T(x + 7.0 * math.cos(a), 0, z + 7.0 * math.sin(a)))
        pil = B(-0.4, 0.4, Y_ROOF, 34.0, -0.2, 0.3, WHITE)
        b = k * math.pi / 4
        out.append(pil.R(0, math.pi / 2 - b, 0).T(x + 7.0 * math.cos(b), 0, z + 7.0 * math.sin(b)))
    prof = [(7.2 * math.cos(t), 34.8 + 7.0 * math.sin(t) * 1.02) for t in np.linspace(0, math.pi / 2 - 0.2, 9)]
    out.append(lathe(prof + [(1.5, prof[-1][1] + 0.05)], LEAD, 32).T(x, 0, z))
    top = prof[-1][1]
    for k in range(8):                                                                       # ribs
        a = k * math.pi / 4
        out.append(tube([(x + (r + 0.08) * math.cos(a), y, z + (r + 0.08) * math.sin(a)) for r, y in prof], 0.13, WHITE, 5))
    out.append(cyl(1.5, 1.5, 3.2, WHITE, 12).T(x, top, z))
    out.append(lathe([(1.7, 0), (1.2, 0.5), (0.4, 1.3), (0.0, 1.5)], LEAD, 12).T(x, top + 3.2, z))
    out.append(sphere(0.3, (0.75, 0.62, 0.3), 8, 5).T(x, top + 4.9, z))
    out.append(B(x - 0.06, x + 0.06, top + 5.1, top + 6.2, z - 0.06, z + 0.06, IRON))
    out.append(B(x - 0.35, x + 0.35, top + 5.75, top + 5.87, z - 0.06, z + 0.06, IRON))
    return merge(out)


def galleries():
    parts = []
    for poly, found in ((EAST_ARM, -4.0), (WEST_ARM, -20.0)):
        parts.append(extrude(poly, found, A_BAND, GRANITE, top=False))
        parts.append(extrude(poly, A_BAND, A_TOP - 0.05, STONE, top_c=TERRACE))
        parts.append(ring(0.2, -0.2, A_BAND, A_COR0, WHITE, poly=poly))
        parts.append(ring(0.6, -0.25, A_COR0, A_TOP, WHITE, poly=poly))
        parts.append(ring(0.3, -0.2, A_TOP, A_PL, WHITE, poly=poly))
        parts.append(ring(0.32, -0.22, A_BAL, A_RAIL, WHITE, poly=poly))
        parts.append(ring(0.15, -0.2, -0.5, 0.0, GRANITE_D, poly=poly))
    if True:
        for y in np.arange(-12.0, -0.9, 0.86):
            parts.append(ring(0.07, -0.2, y, y + 0.72, tint(GRANITE_D, 0.02), poly=WEST_ARM))
    S, N, E, W = "S", "N", "E", "W"
    runs = [
        (W, 52.7, 21.0, 166.3, "arcade", 0), (E, -56.6, 21.0, 166.6, "arcade", 0),             # plaza sides
        (N, 19.7, 62.6, 76.4, "plain", 0), (E, 80.0, 21.3, 166.2, "plain", 0), (S, 166.3, 52.7, 80.0, "plain", 0),
        (N, 20.4, -84.2, -67.1, "plain", 12), (W, -84.2, 20.4, 82.0, "plain", 12), (S, 82.0, -84.2, -64.2, "plain", 12),
        (W, -64.2, 82.0, 122.0, "plain", 12), (N, 122.0, -83.6, -64.2, "plain", 12), (W, -83.6, 122.0, 166.6, "plain", 12),
        (S, 166.6, -83.6, -56.6, "plain", 6),
    ]
    for face, d, a, b, kind, deep in runs:
        parts.append(gallery_facade(face, d, a, b, kind, deep))
    return merge(parts)


def main():
    out = ROOT / "models" / "Q171517"
    out.mkdir(parents=True, exist_ok=True)
    pal = palace().T(SHIFT[0], 0, SHIFT[1])
    gal = galleries().T(SHIFT[0], 0, SHIFT[1])
    tris, size = write_glb(out / "model.glb", [("palacio_real", pal), ("galerias_plaza_de_la_armeria", gal)], roughness=0.9)
    allv = np.vstack([pal.v, gal.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    if "--footprint" in sys.argv:
        fp = [[round(x + SHIFT[0], 1), round(z + SHIFT[1], 1)] for x, z in offset(FOOTPRINT, 1.0)]
        print(json.dumps(fp))


if __name__ == "__main__":
    main()
