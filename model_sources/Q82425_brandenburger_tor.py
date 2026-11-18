"""
Brandenburger Tor / Brandenburg Gate (Q82425), Berlin — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: Carl Gotthard Langhans (1732–1808), 1788–1791; Quadriga by Johann Gottfried Schadow (1764–1850),
1793; side colonnades by Heinrich Strack (1805–1880), 1868; gatehouses rebuilt after 1945. Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 7288267 / way 518071791 and their building:parts (traced from aerial imagery, ±0.5 m):
  gate 33.9 × 12.9 m (cornice), long axis at 173.5°, so the east façade (Pariser Platz, where the Quadriga
  looks) faces 83.5°; the twelve columns at z ≈ ±4.3; transverse walls 1.85 m thick and 6.4 m long; the end
  walls reach x ±17; entablature to 16 m, cornice crown 16.8 m (copper-covered), side attics 16.8–19.7 m and
  6.4 m deep, central attic 16.8–20 m and the full depth; Strack's colonnades (x 17–33, z ±5.5) and the two
  gatehouses (x 20–31, z 5.6–23.3, towards Pariser Platz) with every column position (Ø 0.9 m); whole ensemble
  65.8 m wide (Wikipedia: 62.5–65.5 m).
- de.wikipedia «Brandenburger Tor» (section Abmessungen): building 20.3 m high, about 26 m to the top of the
  Quadriga, 11 m deep; two rows of six fluted Doric columns with bases, 13.5 m high, 1.73 m at the foot; middle
  passage 18 Fuß (5.65 m), side passages 12 Fuß 1 Zoll (3.80 m); 16 metopes (≈1 m², Centauromachy) on each
  side; the relief «Folgen der Tapferkeit» on the attic towards the city (east), its west counterpart never
  made; Quadriga horses 12 Fuß (3.77 m); Elbe sandstone (Cotta), plinths of Lusatian granite.
  (The brief said 15 m columns and 1.75 m; the published 13.5 m / 1.73 m and OSM's 13.5 m are used.)
- Column axes from the published widths: ±3.69, ±9.22, ±14.75 m (OSM agrees within 0.5 m).
- Estimated (±10–15 %): the order's subdivision (base, capital, architrave/frieze/cornice heights), the steps of
  the side attics, the gatehouse/colonnade heights (columns 8 m, cornice 10 m, attic to 11.2 m; OSM gives 10 and
  12 m), the Quadriga's figure sizes. Sculpture (Quadriga, metopes, relief) is stylised.

Axes (docs/MODEL_SPEC.md): +Y up, the east façade (Pariser Platz) faces +Z, the gate runs along X (+X to the
north-north-west), origin on the ground in the middle of the central passage.

    python model_sources/Q82425_brandenburger_tor.py   -> models/Q82425/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, write_glb  # noqa: E402

rng = np.random.default_rng(1791)

SAND = np.array((0.87, 0.80, 0.63))     # Elbe sandstone, pale yellowish
SAND_L = (0.91, 0.86, 0.72)            # sculpture in the stone (metopes, relief)
SAND_D = (0.66, 0.59, 0.45)            # grooves, shadowed recesses
GRANITE = (0.60, 0.58, 0.56)           # Lusatian granite plinths
COPPER = np.array((0.36, 0.62, 0.53))  # the Quadriga's patina
ROOF = (0.42, 0.63, 0.56)              # copper roofs of the gate (OSM: copper, mediumaquamarine)
ROOF_W = (0.50, 0.56, 0.54)            # flat roofs of the colonnades and gatehouses (colour estimated)
DOOR = (0.24, 0.21, 0.18)
FOUND = -3.0                           # foundations below the ground (MODEL_SPEC §3)


def sand(k=1.0):
    return tuple(np.clip(SAND * k * (1 + rng.normal(0, 0.015)), 0, 1))


def copper(k=1.0):
    return tuple(np.clip(COPPER * k * (1 + rng.normal(0, 0.03)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    """Axis-aligned box from its extents."""
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def mirror_x(m):
    n = m.copy(); n.v[:, 0] *= -1; n.f = n.f[:, ::-1]; return n


def mirror_z(m):
    n = m.copy(); n.v[:, 2] *= -1; n.f = n.f[:, ::-1]; return n


def both_x(ms):
    ms = [m for m in ms if m is not None]
    return ms + [mirror_x(m) for m in ms]


def sphere(r, c, seg=12, rings=7):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


# ── The Doric order ──────────────────────────────────────────────────────────
def fluted(r0, r1, y0, y1, c, flutes=20, depth=0.07, rings=3):
    """A fluted shaft: per flute an arris and a scalloped hollow; slight entasis between the rings."""
    per = 4
    n = flutes * per
    V, F = [], []
    for i in range(rings + 1):
        t = i / rings
        y = y0 + (y1 - y0) * t
        r = r0 + (r1 - r0) * t + 0.03 * r0 * math.sin(math.pi * t)          # entasis
        for k in range(n):
            a = 2 * math.pi * k / n
            ph = (k % per) / per                                              # 0 = arris
            rr = r * (1 - depth * math.sin(math.pi * ph))
            V.append((rr * math.cos(a), y, rr * math.sin(a)))
    for i in range(rings):
        for k in range(n):
            j = (k + 1) % n
            p, q = i * n + k, i * n + j
            F += [(p, p + n, q + n), (p, q + n, q)]
    return Mesh(V, F, c)


def doric_column(x, z, r, y0, h, abacus_top, base=True, flutes=20):
    """Langhans' Doric column with an Attic base: plinth, torus, fluted shaft, echinus and abacus."""
    out = []
    yb = y0
    if base:
        out.append(B(x - r * 1.32, x + r * 1.32, y0, y0 + 0.16 * r, z - r * 1.32, z + r * 1.32, sand(0.98)))
        out.append(lathe([(r * 1.26, 0), (r * 1.3, 0.12 * r), (r * 1.22, 0.3 * r), (r * 1.08, 0.34 * r),
                          (r * 1.12, 0.46 * r), (r * 1.02, 0.56 * r)], sand(1.02), 20).T(x, y0 + 0.16 * r, z))
        yb = y0 + 0.72 * r
    y_neck = abacus_top - 0.7 * r
    out.append(fluted(r, r * 0.82, yb, y_neck, sand(1.0), flutes).T(x, 0, z))
    out.append(lathe([(r * 0.84, 0), (r * 0.86, 0.06 * r), (r * 1.18, 0.34 * r), (r * 1.22, 0.4 * r)], sand(1.02), 20).T(x, y_neck, z))
    out.append(B(x - r * 1.25, x + r * 1.25, y_neck + 0.4 * r, abacus_top, z - r * 1.25, z + r * 1.25, sand(1.03)))
    return out


def triglyph(x, y0, y1, zf, side, w, c=None, along="x"):
    """Three vertical bars with two grooves, on a face at zf (or xf when along='z')."""
    c = c or sand(1.03)
    out = []
    bw = w / 3 * 0.72
    for k in (-1, 0, 1):
        cx = x + k * w / 3
        if along == "x":
            out.append(B(cx - bw / 2, cx + bw / 2, y0, y1, min(zf, zf + 0.12 * side), max(zf, zf + 0.12 * side), c))
        else:
            out.append(B(min(zf, zf + 0.12 * side), max(zf, zf + 0.12 * side), y0, y1, cx - bw / 2, cx + bw / 2, c))
    # the band over it and the regula with its guttae under the taenia
    if along == "x":
        out.append(B(x - w / 2, x + w / 2, y1 - 0.08, y1, min(zf, zf + 0.14 * side), max(zf, zf + 0.14 * side), c))
        out.append(B(x - w / 2, x + w / 2, y0 - 0.32, y0 - 0.22, min(zf, zf + 0.12 * side), max(zf, zf + 0.12 * side), c))
    else:
        out.append(B(min(zf, zf + 0.14 * side), max(zf, zf + 0.14 * side), y1 - 0.08, y1, x - w / 2, x + w / 2, c))
        out.append(B(min(zf, zf + 0.12 * side), max(zf, zf + 0.12 * side), y0 - 0.32, y0 - 0.22, x - w / 2, x + w / 2, c))
    return out


def metope_relief(x, yc, zf, side):
    """Centaur and Lapith, stylised: the horse body of the centaur, his torso, and the man fighting him."""
    k = 0.03 * side
    body = sphere(1, SAND_L, 8, 5).S(0.28, 0.17, 0.08).T(x - 0.1, yc - 0.08, zf + k)
    torso = cyl(0.08, 0.06, 0.32, SAND_L, 6).R(0, 0, -0.3).T(x + 0.1, yc, zf + k)
    head = sphere(0.07, SAND_L, 6, 4).T(x + 0.2, yc + 0.34, zf + k)
    man = cyl(0.09, 0.06, 0.62, SAND_L, 6).R(0, 0, 0.25).T(x + 0.32, yc - 0.42, zf + k)
    mhead = sphere(0.07, SAND_L, 6, 4).T(x + 0.18, yc + 0.24, zf + k)
    return [m.S(1, 1, 1) for m in (body, torso, head, man, mhead)]


# ── The gate ─────────────────────────────────────────────────────────────────
COL_X = (3.69, 9.22, 14.75)       # column axes (half), from the passages 5.65 / 3.80 and Ø 1.73
COL_Z = 4.3
R = 0.865
Y_ENT = 13.5                      # top of the columns = bottom of the architrave
WALL_Z = 3.22                     # transverse walls run between the rows of columns
END_X = 16.95                     # outer face of the end walls


def gate():
    parts = []
    half = []
    # granite plinths under each pair of columns and its wall
    for i, x in enumerate(COL_X):
        x0, x1 = x - 1.12, (END_X + 0.05 if i == 2 else x + 1.12)
        half.append(B(x0, x1, FOUND, 0.42, -5.35, 5.35, GRANITE))
    # transverse walls (inner ones between the passages, the end walls wider)
    for i, x in enumerate(COL_X):
        x0, x1 = x - 0.92, (END_X if i == 2 else x + 0.92)
        half.append(B(x0, x1, 0.42, Y_ENT, -WALL_Z, WALL_Z, sand()))
        half.append(B(x0 - 0.06, x1 + (0.0 if i == 2 else 0.06), Y_ENT - 0.75, Y_ENT - 0.45, -WALL_Z - 0.06, WALL_Z + 0.06, sand(1.04)))  # impost band
        half.append(B(x0 - 0.04, x1 + (0.0 if i == 2 else 0.04), 0.42, 0.9, -WALL_Z - 0.04, WALL_Z + 0.04, sand(0.95)))       # socle
        # relief panels in the passages (the deeds of Hercules)
        for sx in ((-1, 1) if i < 2 else (-1,)):
            xf = x + sx * 0.92
            for zc in (-1.55, 1.55):
                half.append(B(min(xf, xf + 0.05 * sx), max(xf, xf + 0.05 * sx), 7.6, 9.9, zc - 1.25, zc + 1.25, sand(1.05)))
                half.append(B(min(xf, xf + 0.09 * sx), max(xf, xf + 0.09 * sx), 8.0, 9.5, zc - 0.5, zc + 0.5, SAND_L))
    # the end walls' outer faces: antae with capitals at both ends
    for s in (-1, 1):
        half.append(B(END_X - 0.02, END_X + 0.22, 0.42, Y_ENT - 0.5, s * WALL_Z - 0.55 if s > 0 else -WALL_Z - 0.0, s * WALL_Z if s > 0 else -WALL_Z + 0.55, sand(1.02)))
    # columns
    for x in COL_X:
        for z in (-COL_Z, COL_Z):
            half += doric_column(x, z, R, 0.42, Y_ENT - 0.42, Y_ENT)
    parts += both_x(half)

    # Entablature: architrave, triglyph frieze, cornice; the crown covered with copper
    AZ = 5.12                                    # architrave face
    parts.append(B(-END_X, END_X, Y_ENT, 14.65, -AZ, AZ, sand(0.99)))
    parts.append(B(-END_X - 0.08, END_X + 0.08, 14.55, 14.7, -AZ - 0.08, AZ + 0.08, sand(1.04)))     # taenia
    parts.append(B(-END_X + 0.05, END_X - 0.05, 14.7, 15.85, -AZ + 0.06, AZ - 0.06, sand(0.97)))     # frieze
    parts.append(B(-END_X - 0.25, END_X + 0.25, 15.85, 16.1, -AZ - 0.3, AZ + 0.3, sand(1.03)))      # bed moulding
    parts.append(B(-END_X - 0.55, END_X + 0.55, 16.1, 16.55, -6.0, 6.0, sand(1.0)))                 # corona
    parts.append(B(-END_X - 0.6, END_X + 0.6, 16.55, 16.75, -6.05, 6.05, sand(1.05)))               # cymatium
    parts.append(B(-END_X - 0.55, END_X + 0.55, 16.75, 16.82, -6.0, 6.0, ROOF))                     # copper crown
    # mutules under the corona
    s_tri = COL_X[2] / 8
    xs_tri = [k * s_tri for k in range(-8, 9)] + [-END_X + 0.45, END_X - 0.45]
    for s in (-1, 1):
        zf = s * (AZ - 0.06)
        for x in xs_tri:
            parts += triglyph(x, 14.75, 15.85, zf, s, 0.84)
            parts.append(B(x - 0.42, x + 0.42, 16.02, 16.1, min(s * 5.4, s * 5.95), max(s * 5.4, s * 5.95), sand(0.93)))
        for k in range(-8, 8):                       # 16 metopes with centaurs and Lapiths
            xm = (k + 0.5) * s_tri
            parts += metope_relief(xm, 15.3, zf, s)
    for sx in (-1, 1):                               # triglyphs on the short ends
        xf = sx * (END_X - 0.05)
        for zc in (-AZ + 0.45, -2.6, -0.85, 0.85, 2.6, AZ - 0.45):
            parts += triglyph(zc, 14.75, 15.85, xf, sx, 0.84, along="z")

    # Attic: stepped side parts, the central block with the relief, the pedestal of the Quadriga
    CX = 4.65
    for sx in (-1, 1):
        steps = ((16.82, 18.45, 3.2, 16.3), (18.45, 19.08, 2.5, 15.6), (19.08, 19.7, 1.8, 14.9))
        for y0, y1, dz, xe in steps:
            x0, x1 = sorted((sx * CX, sx * xe))
            parts.append(B(x0, x1, y0, y1, -dz, dz, sand(1.01)))
            x0r, x1r = sorted((sx * CX, sx * (xe - 0.06)))
            parts.append(B(x0r, x1r, y1, y1 + 0.05, -dz + 0.06, dz - 0.06, ROOF))
        x0, x1 = sorted((sx * CX, sx * 16.36))
        parts.append(B(x0, x1, 16.82, 17.15, -3.28, 3.28, sand(1.04)))                 # base moulding
    parts.append(B(-CX - 0.15, CX + 0.15, 16.82, 17.25, -5.6, 5.6, sand(1.04)))         # central block: base
    parts.append(B(-CX, CX, 17.25, 19.55, -5.42, 5.42, sand()))
    parts.append(B(-CX - 0.2, CX + 0.2, 19.55, 20.0, -5.62, 5.62, sand(1.05)))          # its cornice
    parts.append(B(-CX - 0.15, CX + 0.15, 20.0, 20.05, -5.57, 5.57, ROOF))
    parts.append(B(-3.9, 3.9, 20.05, 20.32, -4.3, 4.3, sand(1.02)))                     # Quadriga pedestal
    parts.append(B(-3.85, 3.85, 20.32, 20.36, -4.25, 4.25, ROOF))
    for s in (-1, 1):                                                                   # relief panels
        zf = s * 5.42
        parts.append(B(-4.15, 4.15, 17.42, 19.4, min(zf, zf + 0.08 * s), max(zf, zf + 0.08 * s), sand(1.04)))
        parts.append(B(-4.0, 4.0, 17.52, 19.3, min(zf, zf + 0.1 * s), max(zf, zf + 0.1 * s), SAND_D if s < 0 else sand(0.95)))
        if s > 0:
            parts += attic_relief(zf + 0.1)
    # «Folgen der Tapferkeit»: only on the east (city) side; the west panel stays empty
    parts += quadriga(20.36)
    return merge(parts)


def attic_relief(zf):
    """«The consequences of bravery»: a procession of figures with Victory in the middle, stylised."""
    out = []
    xs = np.linspace(-3.5, 3.5, 13)
    for i, x in enumerate(xs):
        h = 1.45 if i != 6 else 1.6
        y0 = 17.6
        w = 1 + 0.25 * rng.random()
        fig = lathe([(0.24 * w, 0), (0.2 * w, 0.4 * h), (0.15 * w, 0.62 * h), (0.12, 0.74 * h), (0.05, 0.8 * h), (0, 0.8 * h)], SAND_L, 8)
        out.append(fig.S(1, 1, 0.45).T(x + rng.normal(0, 0.05), y0, zf))
        out.append(sphere(0.1, SAND_L, 6, 4).S(1, 1, 0.6).T(x, y0 + 0.88 * h, zf + 0.02))
        if i == 6:                                              # Victory's wings
            for s in (-1, 1):
                out.append(B(-0.05, 0.05, -0.4, 0.4, -0.03, 0.03, SAND_L).R(0, 0, s * 0.6).T(x + s * 0.32, y0 + 1.15, zf + 0.04))
    return out


# ── The Quadriga ─────────────────────────────────────────────────────────────
def horse(c, raise_leg=0, turn=0.0):
    """A horse facing +Z standing on y = 0 (head ~3.75 m, as the 12 Fuß of the design)."""
    out = [sphere(1, c, 12, 8).S(0.47, 0.6, 1.4).T(0, 2.05, 0),
           sphere(1, c, 10, 6).S(0.48, 0.62, 0.62).T(0, 2.12, 1.05),
           sphere(1, c, 10, 6).S(0.5, 0.6, 0.6).T(0, 2.1, -1.08),
           tube([(0, 2.3, 1.25), (0, 2.85, 1.62), (0, 3.35, 1.85)], 0.3, c, 8),
           tube([(0, 3.55, 1.85), (0, 3.25, 2.28), (0, 2.95, 2.55)], 0.19, c, 8),
           sphere(1, c, 8, 5).S(0.2, 0.24, 0.24).T(0, 3.5, 1.85),
           sphere(1, c, 8, 5).S(0.16, 0.16, 0.18).T(0, 2.95, 2.55),
           tube([(0, 2.62, 1.12), (0, 3.15, 1.5), (0, 3.68, 1.72)], 0.11, c, 5),               # mane
           tube([(0, 2.3, -1.6), (0, 1.75, -1.95), (0, 1.0, -1.95)], 0.12, c, 6)]
    for sx in (-1, 1):
        out.append(cyl(0.06, 0.0, 0.28, c, 5).T(sx * 0.1, 3.7, 1.82))                       # ears
        for zl, front in ((1.05, True), (-1.1, False)):
            xl = sx * 0.24
            if front and raise_leg == sx:
                pts = [(xl, 1.75, zl), (xl, 1.1, zl + 0.5), (xl, 0.75, zl + 0.2)]
            elif front:
                pts = [(xl, 1.75, zl), (xl, 0.95, zl + 0.05), (xl, 0.15, zl)]
            else:
                pts = [(xl, 1.8, zl), (xl, 1.0, zl - 0.25), (xl, 0.15, zl - 0.05)]
            out.append(tube(pts, 0.14, c, 6))
            out.append(cyl(0.13, 0.15, 0.16, c, 6).T(*pts[-1]).T(0, -0.15, 0))
    m = merge(out)
    return m.R(0, turn, 0)


def victoria(c):
    """Victoria standing in the chariot: robe, wings, raised staff with the oak wreath, Iron Cross and eagle."""
    out = [lathe([(0.52, 0), (0.46, 0.8), (0.36, 1.7), (0.32, 2.35), (0.36, 2.65), (0.2, 2.95), (0, 3.0)], c, 14),
           sphere(0.25, c, 10, 6).T(0, 3.25, 0.05),
           sphere(0.27, c, 10, 6).S(1, 0.7, 1).T(0, 3.42, -0.02)]                            # helmet / hair
    for s in (-1, 1):                                                                       # wings
        out.append(B(-0.06, 0.06, 0, 1.9, -0.45, 0.45, c).R(0, 0, s * 0.42).T(s * 0.38, 2.35, -0.35))
        out.append(B(-0.05, 0.05, 0, 1.3, -0.3, 0.3, c).R(0, 0, s * 0.75).T(s * 0.5, 2.3, -0.42))
    out.append(tube([(0.3, 2.6, 0.0), (0.55, 3.15, 0.1), (0.62, 3.75, 0.15)], 0.09, c, 6))  # right arm up
    out.append(tube([(-0.3, 2.55, 0.0), (-0.45, 2.2, 0.4), (-0.35, 2.0, 0.75)], 0.09, c, 6))  # left arm: the reins
    out.append(cyl(0.05, 0.05, 4.6, c, 6).T(0.62, 0.8, 0.15))                               # the staff
    # oak wreath, Iron Cross, eagle (top ~26 m)
    wy, wz = 4.75, 0.15
    ring = [(0.62 + 0.42 * math.cos(a), wy + 0.42 * math.sin(a), wz) for a in np.linspace(0, 2 * math.pi, 17)]
    out.append(tube(ring, 0.08, c, 6))
    out.append(B(0.62 - 0.09, 0.62 + 0.09, wy - 0.33, wy + 0.33, wz - 0.04, wz + 0.04, c))
    out.append(B(0.62 - 0.33, 0.62 + 0.33, wy - 0.09, wy + 0.09, wz - 0.04, wz + 0.04, c))
    out.append(sphere(1, c, 8, 5).S(0.14, 0.24, 0.12).T(0.62, 5.4, wz))
    out.append(sphere(0.08, c, 6, 4).T(0.62, 5.66, wz + 0.04))
    for s in (-1, 1):
        out.append(B(-0.04, 0.04, 0, 0.55, -0.12, 0.12, c).R(0, 0, s * 0.95).T(0.62 + s * 0.1, 5.38, wz))
    return merge(out)


def quadriga(y0):
    out = [B(-3.4, 3.4, y0, y0 + 0.35, -3.3, 3.75, sand(1.02)),                     # the bronze group's base
           B(-3.35, 3.35, y0 + 0.35, y0 + 0.38, -3.25, 3.7, copper(0.85))]
    yb = y0 + 0.38
    for i, x in enumerate((-2.3, -0.77, 0.77, 2.3)):
        turn = (-0.13 if x < -1 else 0.13 if x > 1 else (-0.03 if x < 0 else 0.03))
        raise_leg = -1 if i == 0 else (1 if i == 3 else 0)
        out.append(horse(copper(), raise_leg, turn).T(x, yb, 0.95))
    # the chariot: a curved box with two wheels
    cz = -2.1
    out.append(cyl(0.95, 0.95, 1.05, copper(0.95), 16, sx=1.0).S(1, 1, 0.55).T(0, yb + 0.55, cz))
    out.append(B(-0.9, 0.9, yb + 0.5, yb + 0.6, cz - 0.55, cz + 0.5, copper(0.9)))
    for s in (-1, 1):
        out.append(cyl(0.72, 0.72, 0.12, copper(0.8), 16).R(0, 0, math.pi / 2).T(s * 1.08 + (0.06 if s < 0 else -0.06), yb + 0.72, cz - 0.15))
        out.append(cyl(0.12, 0.12, 0.3, copper(0.8), 8).R(0, 0, math.pi / 2).T(s * 1.2 + (0.15 if s < 0 else -0.15), yb + 0.72, cz - 0.15))
    out.append(tube([(0, yb + 0.9, cz + 0.5), (0, yb + 1.6, cz + 1.8), (0, yb + 2.0, cz + 2.6)], 0.07, copper(0.9), 6))  # pole
    out.append(victoria(copper(1.05)).S(0.9).T(0, yb + 0.55, cz))
    # reins from her left hand to the bits
    for x in (-2.3, -0.77, 0.77, 2.3):
        out.append(tube([(-0.32, yb + 2.35, cz + 0.65), (x, yb + 2.9, 0.95 + 1.2), (x, yb + 3.0, 0.95 + 2.35)], 0.025, copper(0.85), 4))
    return out


# ── Strack's colonnades and the gatehouses (towards Pariser Platz) ───────────
r_s = 0.45
COLON_X = (17.76, 21.23, 24.8, 28.37, 31.95)
COLON_Z_OUT, COLON_Z_IN = 4.6, 1.54
HOUSE_X = (20.85, 30.2)                       # side rows of the gatehouse
HOUSE_Z = (8.16, 11.74, 15.3, 18.88, 22.44)
FRONT_X = (20.85, 23.97, 27.08, 30.2)
Y_C = 8.0                                     # top of the small columns (estimated)


def wings():
    half = []
    # colonnade hall: x 16.95–32.9, z ±5.5
    cols = [(x, z) for x in COLON_X for z in (-COLON_Z_OUT, COLON_Z_OUT)]
    cols += [(x, z) for x in (21.23, 28.37, 31.95) for z in (-COLON_Z_IN, COLON_Z_IN)]
    # gatehouse peristyle
    cols += [(x, z) for x in HOUSE_X for z in HOUSE_Z]
    cols += [(x, HOUSE_Z[-1]) for x in FRONT_X[1:3]]
    for x, z in cols:
        half += doric_column(x, z, r_s, 0.25, Y_C - 0.25, Y_C, base=True, flutes=16)
    half.append(B(16.95, 32.4, FOUND, 0.25, -5.05, 5.05, GRANITE))           # stylobates
    half.append(B(20.3, 30.75, FOUND, 0.25, 5.05, 22.95, GRANITE))
    half.append(B(16.95, 18.25, 0.25, Y_C, -3.2, 3.2, sand()))               # wall joining the gate's end wall
    # gatehouse core («Raum der Stille» in the north, an information room in the south)
    half.append(B(21.75, 29.3, 0.25, Y_C, 9.0, 21.6, sand(0.98)))
    half.append(B(21.7, 29.35, 0.25, 0.85, 8.95, 21.65, sand(0.93)))
    half.append(B(21.68, 21.75, 0.25, 3.6, 13.9, 15.7, DOOR))                  # door to the passage side
    half.append(B(24.3, 26.8, 0.25, 3.8, 21.6, 21.67, DOOR))                   # door to Pariser Platz
    for x0 in (22.5, 27.6):
        half.append(B(x0, x0 + 1.2, 1.8, 4.4, 21.6, 21.66, SAND_D))           # blind windows
    # entablature over the colonnade and over the gatehouse (estimated heights)
    Y_A, Y_F, Y_K = Y_C + 0.62, Y_C + 1.32, Y_C + 2.0
    half.append(B(17.2, 32.5, Y_C, Y_A, -5.05, 5.05, sand(0.99)))
    half.append(B(20.3, 30.75, Y_C, Y_A, 5.05, 22.95, sand(0.99)))
    half.append(B(17.15, 32.55, Y_A - 0.07, Y_A + 0.04, -5.12, 5.12, sand(1.04)))
    half.append(B(20.25, 30.8, Y_A - 0.07, Y_A + 0.04, 5.05, 23.02, sand(1.04)))
    half.append(B(17.25, 32.45, Y_A, Y_F, -5.0, 5.0, sand(0.97)))
    half.append(B(20.35, 30.7, Y_A, Y_F, 5.0, 22.9, sand(0.97)))
    half.append(B(17.0, 32.9, Y_F, Y_K, -5.5, 5.5, sand(1.03)))               # cornice
    half.append(B(19.9, 31.3, Y_F, Y_K, 5.5, 23.35, sand(1.03)))
    # low attic and copper flat roofs (top 11.2 m, OSM: 10–12 m)
    half.append(B(17.3, 32.6, Y_K, Y_K + 1.05, -5.2, 5.2, sand(1.0)))
    half.append(B(20.2, 31.0, Y_K, Y_K + 1.05, 5.2, 23.05, sand(1.0)))
    half.append(B(17.25, 32.65, Y_K + 0.95, Y_K + 1.12, -5.25, 5.25, sand(1.05)))
    half.append(B(20.15, 31.05, Y_K + 0.95, Y_K + 1.12, 5.25, 23.1, sand(1.05)))
    half.append(B(17.4, 32.5, Y_K + 1.12, Y_K + 1.18, -5.1, 5.1, ROOF_W))
    half.append(B(20.3, 30.9, Y_K + 1.12, Y_K + 1.18, 5.1, 22.95, ROOF_W))
    # triglyphs: Pariser Platz front of the gatehouse, the outer side, the colonnade fronts
    for x in np.linspace(20.85, 30.2, 7):
        half += triglyph(x, Y_A + 0.05, Y_F, 22.9, 1, 0.5)
    for z in np.linspace(8.16, 22.44, 9):
        half += triglyph(z, Y_A + 0.05, Y_F, 30.7, 1, 0.5, along="z")
        half += triglyph(z, Y_A + 0.05, Y_F, 20.35, -1, 0.5, along="z")
    for s in (-1, 1):
        for x in np.linspace(17.76, 31.95, 9):
            if s > 0 and 20.2 < x < 30.8:
                continue
            half += triglyph(x, Y_A + 0.05, Y_F, s * 5.0, s, 0.5)
    for z in np.linspace(-4.6, 4.6, 7):
        half += triglyph(z, Y_A + 0.05, Y_F, 32.45, 1, 0.5, along="z")
    return both_x(half)


def main():
    out = ROOT / "models" / "Q82425"
    out.mkdir(parents=True, exist_ok=True)
    stone = merge([gate()] + wings())
    tris, size = write_glb(out / "model.glb", [("brandenburger_tor", stone)], roughness=0.9)
    lo, hi = stone.v.min(0), stone.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
