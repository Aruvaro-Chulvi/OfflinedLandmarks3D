"""
Puerta de Hierro (Q962107), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: Francisco Moradillo (died 1776), sculpture by Juan Domingo Olivieri (died 1762), 1751–1753,
public domain. Dismantled stone by stone in 1991 and set up again on a traffic island near its first place.

Measurements (see meta.json for the sources):
- OpenStreetMap way 568626354: outline 22.18 × 3.78 m, long axis at 67.7°, so the south façade (towards Madrid)
  faces 157.7° and the north one (towards the Monte de El Pardo) 337.7°.
- Heights and widths measured on a near-orthographic photo of the south façade (Wikimedia Commons,
  «Puerta de Hierro (Madrid) 03.jpg», scaled by the 22.18 m of the outline, ±3–5 %): central body 11.7 m wide,
  arch 3.7 m wide with its crown at 7.0 m, entablature 7.6–9.4 m, pediment apex 11.2 m, the royal arms with
  its crown and cross up to 14.7 m; the two loose end piers 1.6 m wide, 5.6 m to their cornice and 7.4 m to
  the top of their flaming vases; iron railings about 2.5 m high between them and the central body.
- Both façades are alike: fluted Tuscan pilasters beside the arch, hunting and military trophies in relief on
  the side panels, curved buttresses (scrolls) against the upper body, a triangular pediment with a relief in
  the tympanum, two reclining sphinxes on pedestals, and the royal arms of Fernando VI between flags on top.
White Colmenar limestone for the body, grey granite for the plinths, black wrought iron for the gates.
Sculpture is stylised (volumes and silhouettes, as seen from a few dozen metres up).

Axes (docs/MODEL_SPEC.md): +Y up, the south façade faces +Z, the gate runs along X (+X to the east-north-east),
origin on the ground in the middle of the arch.

    python model_sources/Q962107_puerta_de_hierro.py   -> models/Q962107/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, triangulate, write_glb  # noqa: E402

rng = np.random.default_rng(1753)

STONE = np.array((0.87, 0.855, 0.81))   # Colmenar limestone
GRANITE = (0.62, 0.61, 0.585)
CARVE = (0.93, 0.92, 0.885)             # sculpture
IRON = (0.10, 0.10, 0.11)
GILT = (0.55, 0.47, 0.25)               # the gilded tips of the gate
FOUND = -2.5


def stone(k=1.0):
    return tuple(np.clip(STONE * k * (1 + rng.normal(0, 0.015)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def mirror_x(m):
    n = m.copy(); n.v[:, 0] *= -1; n.f = n.f[:, ::-1]; return n


def mirror_z(m):
    n = m.copy(); n.v[:, 2] *= -1; n.f = n.f[:, ::-1]; return n


def both_x(ms):
    ms = [m for m in ms if m is not None]
    return ms + [mirror_x(m) for m in ms]


def both_z(ms):
    ms = [m for m in ms if m is not None]
    return ms + [mirror_z(m) for m in ms]


def sphere(r, c, seg=12, rings=7):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def prism_xy(poly, z0, z1, c):
    """A prism from an outline drawn in the (x, y) plane (a façade outline), from z0 to z1."""
    pts, tris = triangulate([(x, y) for x, y in poly])
    n = len(pts)
    # triangulate() orients counter-clockwise in its own (first, second) axes; make the side walls face outwards
    ring = [tuple(p) for p in pts]
    area = sum(ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1] for i in range(n))
    side = []
    for i in range(n):
        j = (i + 1) % n
        (xa, ya), (xb, yb) = ring[i], ring[j]
        q = [(xa, ya, z0), (xb, yb, z0), (xb, yb, z1), (xa, ya, z1)]
        f = [(0, 1, 2), (0, 2, 3)] if area > 0 else [(0, 2, 1), (0, 3, 2)]
        side.append(Mesh(q, f, c))
    # the caps: the z1 cap must face +z, the z0 cap −z
    cap0 = Mesh([(x, y, z0) for x, y in pts], tris if area > 0 else tris[:, ::-1], c)
    cap1 = Mesh([(x, y, z1) for x, y in pts], tris[:, ::-1] if area > 0 else tris, c)
    return merge(side + [cap0, cap1])


# ── Arch ─────────────────────────────────────────────────────────────────────
def arch_block(x0, x1, ys, ytop, z0, z1, c, seg=24):
    """The wall over a semicircular arch: both faces (rectangle minus the half circle) and the soffit."""
    r, xc = (x1 - x0) / 2, (x0 + x1) / 2
    arc = [(xc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(math.pi, 0, seg + 1)]
    out = []
    for z, flip in ((z0, True), (z1, False)):
        V = [(x, y, z) for x, y in arc] + [(x0, ytop, z), (x1, ytop, z)]
        TL, TR, h = seg + 1, seg + 2, seg // 2
        F = [(TL, i, i + 1) for i in range(h)] + [(TL, h, TR)] + [(TR, i, i + 1) for i in range(h, seg)]
        F = np.array(F)
        out.append(Mesh(V, F if not flip else F[:, ::-1], c))
    V, F = [], []
    for x, y in arc:
        V += [(x, y, z0), (x, y, z1)]
    for i in range(seg):
        a = 2 * i; F += [(a, a + 3, a + 1), (a, a + 2, a + 3)]
    out.append(Mesh(V, F, c))
    return merge(out)


# ── Ornament ─────────────────────────────────────────────────────────────────
def vase(x, y, z, s=1.0):
    """The urn with a flaming plume that crowns each pier."""
    prof = [(0.0, 0), (0.30, 0), (0.30, 0.12), (0.16, 0.22), (0.14, 0.38), (0.34, 0.62), (0.40, 0.85), (0.32, 1.05),
            (0.20, 1.15), (0.24, 1.22), (0.10, 1.30), (0.16, 1.42), (0.12, 1.62), (0.05, 1.80), (0.0, 1.88)]
    return lathe([(r * s, h * s) for r, h in prof], CARVE, 14).T(x, y, z)


def trophy_relief(xc, yc, zf, side, w=0.95, h=2.9):
    """Hunting / military trophy in low relief on a side panel: crossed staffs, a helmet, a shield, a horn."""
    out = [B(xc - w / 2 - 0.08, xc + w / 2 + 0.08, yc - h / 2 - 0.08, yc + h / 2 + 0.08,
             *sorted((zf, zf + 0.06 * side)), stone(0.96))]
    zr = zf + 0.08 * side
    for s in (-1, 1):
        out.append(tube([(xc - s * 0.35, yc - 1.25, zr), (xc + s * 0.35, yc + 1.1, zr)], 0.05, CARVE, 6))
    out.append(sphere(0.22, CARVE, 10, 6).S(1, 1.15, 0.45).T(xc, yc + 0.95, zr))           # helmet
    out.append(cyl(0.07, 0.0, 0.4, CARVE, 6).T(xc, yc + 1.15, zr))                          # its plume
    out.append(cyl(0.32, 0.32, 0.1, CARVE, 14).R(math.pi / 2, 0, 0).S(1, 1.2, 1).T(xc, yc + 0.15, zr - 0.04 * side + (0.1 if side > 0 else 0)))
    out.append(tube([(xc - 0.3 + 0.6 * t, yc - 0.65 - 0.25 * math.sin(math.pi * t), zr) for t in np.linspace(0, 1, 7)], 0.07, CARVE, 6))
    out.append(tube([(xc + 0.3 * math.cos(a), yc - 1.05 + 0.18 * math.sin(a), zr) for a in np.linspace(0, math.pi, 7)], 0.04, CARVE, 6))
    return out


def sphinx(x, y, z, out_dir):
    """A reclining sphinx lying along x, its head towards the outside of the gate."""
    s = out_dir
    body = sphere(0.5, CARVE, 12, 7).S(1.35, 0.5, 0.42).T(x - 0.1 * s, y + 0.3, z)
    chest = sphere(0.36, CARVE, 10, 6).S(0.8, 1.0, 0.75).T(x + 0.45 * s, y + 0.55, z)
    head = sphere(0.2, CARVE, 10, 6).T(x + 0.6 * s, y + 0.98, z)
    headdress = cyl(0.24, 0.16, 0.35, CARVE, 10).T(x + 0.57 * s, y + 0.68, z)
    paws = [B(min(x + 0.5 * s, x + 1.0 * s), max(x + 0.5 * s, x + 1.0 * s), y, y + 0.16, z + dz - 0.08, z + dz + 0.08, CARVE)
            for dz in (-0.15, 0.15)]
    tail = tube([(x - 0.7 * s, y + 0.2, z + 0.1), (x - 0.9 * s, y + 0.1, z + 0.25), (x - 0.75 * s, y + 0.05, z + 0.35)], 0.05, CARVE, 6)
    return [body, chest, head, headdress, tail] + paws


def flags(xc, y0, z, side):
    """A fan of furled flags and staffs beside the royal arms."""
    out = []
    for k, a in enumerate((0.35, 0.65, 0.95)):
        tip = (xc + side * math.sin(a) * 2.2, y0 + math.cos(a) * 2.2)
        out.append(tube([(xc, y0, z), (tip[0], tip[1], z)], 0.045, CARVE, 6))
        mx, my = xc + side * math.sin(a) * 1.5, y0 + math.cos(a) * 1.5
        cloth = merge([B(-0.04, 0.42, -0.55, 0.45, -0.06, 0.06, CARVE),                 # a flag hanging from its staff
                       B(0.05, 0.5, -0.75, -0.5, -0.05, 0.05, CARVE).R(0, 0, 0.35)])
        out.append(cloth.S(side, 1, 1).R(0, 0, -side * a).T(mx, my, z + 0.08 * (k - 1)))
        if cloth.f is not None and side < 0:
            out[-1].f = out[-1].f[:, ::-1]
    out.append(sphere(0.22, CARVE, 8, 5).T(xc + side * 0.55, y0 - 0.1, z + 0.2))          # a drum / cannonball heap
    return out


def royal_arms(y0):
    """The royal arms of Fernando VI on the apex of the pediment, with its crown, cross and flags."""
    out = [B(-1.0, 1.0, y0 - 0.55, y0 + 0.25, -0.65, 0.65, stone(1.02)),                       # pedestal
           B(-1.15, 1.15, y0 + 0.25, y0 + 0.42, -0.75, 0.75, stone(1.05))]
    yc = y0 + 1.5
    shield = cyl(0.62, 0.62, 0.5, CARVE, 20).R(math.pi / 2, 0, 0).S(1, 1.38, 1).T(0, yc, -0.25)
    rim = tube([(0.66 * math.cos(a), yc + 0.9 * math.sin(a), 0.0) for a in np.linspace(0, 2 * math.pi, 25)], 0.09, CARVE, 6)
    cartouche = [tube([(sx * (0.62 + 0.18 * math.sin(t * 3)), yc - 0.9 + 1.8 * t, 0.0) for t in np.linspace(0, 1, 9)], 0.1, CARVE, 6)
                 for sx in (-1, 1)]
    crown = [cyl(0.48, 0.55, 0.25, CARVE, 16).T(0, yc + 0.95, 0),
             lathe([(0.55, 0), (0.42, 0.18), (0.18, 0.32), (0, 0.36)], CARVE, 12).T(0, yc + 1.2, 0),
             sphere(0.08, CARVE, 8, 5).T(0, yc + 1.6, 0)]
    cross = [B(-0.04, 0.04, yc + 1.62, yc + 1.95, -0.04, 0.04, CARVE), B(-0.13, 0.13, yc + 1.78, yc + 1.85, -0.04, 0.04, CARVE)]
    out += [shield, rim] + cartouche + crown + cross
    for side in (-1, 1):
        out += flags(side * 0.75, y0 + 0.45, 0.0, side)
    return out


# ── Iron work ────────────────────────────────────────────────────────────────
def railing(x0, x1, h, z=0.0, pitch=0.13):
    """Wrought-iron railing on a low stone curb: bars with spear tips, two rails."""
    out = [B(x0, x1, FOUND, 0.3, z - 0.28, z + 0.28, GRANITE)]
    n = max(2, int((x1 - x0) / pitch))
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        out.append(B(x - 0.015, x + 0.015, 0.3, h, z - 0.015, z + 0.015, IRON))
        out.append(cyl(0.035, 0.0, 0.14, IRON, 4).T(x, h, z))
    for y in (0.45, h - 0.25):
        out.append(B(x0, x1, y, y + 0.06, z - 0.03, z + 0.03, IRON))
    return out


def arch_gate(r, ys, z=0.0):
    """The double gate in the arch: two leaves with bars, rails, and a scrolled crest with gilded tips."""
    out = []
    top = ys - 0.95                                    # the leaves end with a low arc under the crest
    xs = np.linspace(-r + 0.06, r - 0.06, 27)
    for x in xs:
        yt = top + 0.25 * math.cos(x / r * math.pi / 2)
        out.append(B(x - 0.018, x + 0.018, 0.0, yt, z - 0.018, z + 0.018, IRON))
    for y in (0.3, 1.1, top - 0.35):
        out.append(B(-r, r, y, y + 0.07, z - 0.035, z + 0.035, IRON))
    out.append(tube([(x, top + 0.25 * math.cos(x / r * math.pi / 2) + 0.05, z) for x in np.linspace(-r, r, 13)], 0.05, IRON, 6))
    for x in (-r + 0.04, -0.03, 0.03, r - 0.04):       # the frames of the leaves
        out.append(B(x - 0.04, x + 0.04, 0.0, top + 0.1, z - 0.04, z + 0.04, IRON))
    # crest: scrolls rising to a central finial, filling the arch up to about 1.4 m above the springing
    for s in (-1, 1):
        for k in range(3):
            cx, cy, rr = s * (0.45 + 0.5 * k), top + 0.6 - 0.12 * k, 0.28 - 0.04 * k
            out.append(tube([(cx + rr * math.cos(a) * s, cy + rr * math.sin(a), z) for a in np.linspace(-0.3, 2 * math.pi - 0.6, 14)], 0.03, IRON, 5))
        out.append(tube([(s * 1.6, top + 0.35, z), (s * 0.9, top + 0.95, z), (s * 0.25, top + 1.45, z), (0, top + 1.95, z)], 0.035, IRON, 5))
    out.append(B(-0.03, 0.03, top, top + 2.3, z - 0.03, z + 0.03, IRON))
    out.append(sphere(0.12, GILT, 8, 5).T(0, top + 2.38, z))
    out.append(sphere(0.25, IRON, 10, 6).S(1, 1, 0.25).T(0, top + 1.65, z))           # the crowned cipher in the middle
    return out


# ── The gate ─────────────────────────────────────────────────────────────────
def gate():
    parts = []
    R, YS = 1.85, 5.17                    # arch radius and springing: crown at 7.02 m
    DL, DU = 1.30, 1.10                   # half-depths of the lower and the upper body
    half = []

    # Lower body (x 1.85–4.65) and its end pier (4.65–5.85) with the vase
    half.append(B(R, 4.65, FOUND, 5.4, -DL, DL, stone()))
    half.append(B(R - 0.02, 4.67, 0.0, 0.55, -DL - 0.08, DL + 0.08, GRANITE))           # granite plinth course
    half.append(B(4.65, 5.85, FOUND, 5.5, -DL - 0.15, DL + 0.15, stone(1.02)))
    half.append(B(4.6, 5.9, 0.0, 0.6, -DL - 0.22, DL + 0.22, GRANITE))
    half.append(B(4.55, 5.95, 5.35, 5.7, -DL - 0.25, DL + 0.25, stone(1.05)))            # cornice of the pier
    half.append(B(4.75, 5.75, 5.7, 5.95, -0.5, 0.5, stone()))                             # vase plinth
    half.append(vase(5.25, 5.95, 0.0, 0.85))
    half.append(B(R, 4.7, 5.15, 5.42, -DL - 0.1, DL + 0.1, stone(1.06)))                 # impost band
    for k in range(9):                                                                     # rustication joints (light grooves)
        y = 0.55 + k * 0.54
        for s in (-1, 1):
            half.append(B(4.65, 5.85, y, y + 0.03, *sorted((s * (DL + 0.15), s * (DL + 0.17))), stone(0.86)))

    # Upper body (x 1.85–3.95) and the curved buttress (scroll) leaning on it
    half.append(B(R, 3.95, 5.4, 7.6, -DU, DU, stone()))
    scroll = [(3.95, 7.6), (3.95, 5.4), (4.65, 5.4)] + [
        (3.95 + 0.7 * (1 - t) ** 1.8, 5.4 + 2.2 * t) for t in np.linspace(0.1, 0.9, 7)]
    half.append(prism_xy(scroll, -DU + 0.1, DU - 0.1, stone(0.98)))
    for s in (-1, 1):
        zf = s * (DU - 0.1)
        half.append(tube([(3.95 + 0.7 * (1 - t) ** 1.8 + 0.04, 5.4 + 2.2 * t, zf) for t in np.linspace(0, 1, 9)], 0.07, stone(1.05), 6))
        half.append(sphere(0.17, stone(1.05), 8, 5).S(1, 1, 0.5).T(4.6, 5.55, zf))

    for s in (-1, 1):
        zL, zU = s * DL, s * DU
        # fluted Tuscan pilaster beside the arch (x 2.2–3.35), on its pedestal, up to the architrave
        half.append(B(2.15, 3.4, 0.0, 1.3, *sorted((zL, zL + 0.24 * s)), stone(1.03)))
        half.append(B(2.1, 3.45, 1.2, 1.35, *sorted((zL, zL + 0.3 * s)), stone(1.06)))
        half.append(B(2.25, 3.3, 1.35, 7.35, *sorted((zL, zL + 0.2 * s)), stone(1.04)))
        for k in range(5):
            x = 2.4 + k * 0.19
            half.append(B(x - 0.045, x + 0.045, 1.6, 7.0, *sorted((zL + 0.2 * s, zL + 0.205 * s)), stone(0.82)))
        half.append(B(2.15, 3.4, 7.3, 7.6, *sorted((zL, zL + 0.3 * s)), stone(1.07)))     # capital
        if abs(zU) < abs(zL):                                                               # carry it up the upper body face
            half.append(B(2.25, 3.3, 5.4, 7.35, *sorted((zU, zL + 0.2 * s)), stone(1.0)))
        # trophy panel between pilaster and pier
        half += trophy_relief(4.05, 3.1, zL, s)
        # archivolt
        half.append(tube([(R * 1.08 * math.cos(a), YS + R * 1.08 * math.sin(a), zL + 0.06 * s)
                          for a in np.linspace(math.pi / 2, 0, 9)], 0.12, stone(1.07), 6))
    parts += both_x(half)

    # The arch itself through both bodies
    parts.append(arch_block(-R, R, YS, 7.6, -DL, DL, stone()))
    for s in (-1, 1):
        zL = s * DL
        lo, hi = sorted((zL, zL + 0.25 * s))
        parts.append(B(-0.3, 0.3, YS + R - 0.15, YS + R + 0.6, lo, hi, stone(1.08)))      # keystone
        parts.append(sphere(0.32, CARVE, 10, 6).S(1.2, 0.9, 0.5).T(0, YS + R + 0.45, zL + 0.25 * s))   # its shell

    # Entablature: architrave, frieze with triglyphs, cornice (widening upwards)
    parts.append(B(-4.0, 4.0, 7.6, 7.95, -DL - 0.05, DL + 0.05, stone(1.03)))
    parts.append(B(-4.05, 4.05, 7.95, 8.65, -DL - 0.02, DL + 0.02, stone()))
    for s in (-1, 1):
        for k in range(-7, 8):
            x = k * 0.55
            parts.append(B(x - 0.13, x + 0.13, 8.0, 8.6, *sorted((s * (DL + 0.02), s * (DL + 0.08))), stone(1.07)))
    parts.append(B(-4.35, 4.35, 8.65, 8.95, -DL - 0.3, DL + 0.3, stone(1.05)))
    parts.append(B(-4.75, 4.75, 8.95, 9.4, -1.9, 1.9, stone(1.08)))

    # Pediment: tympanum, raking cornices, relief, and the roof slab
    PA, PH = 4.05, 1.8                                    # half span and height above 9.4 (apex 11.2)
    parts.append(prism_xy([(-PA + 0.25, 9.4), (PA - 0.25, 9.4), (0.0, 9.4 + PH - 0.25)], -DL + 0.1, DL - 0.1, stone()))
    roof = []
    for s in (-1, 1):
        roof.append(prism_xy([(0, 9.4 + PH), (s * PA, 9.4), (s * (PA - 0.55), 9.4), (0, 9.4 + PH - 0.4)], -1.55, 1.55, stone(1.07)))
    parts += roof
    for s in (-1, 1):                                     # tympanum relief: a radiating trophy of arms
        zf = s * (DL - 0.1)
        for a in np.linspace(-1.2, 1.2, 7):
            parts.append(tube([(0, 9.55, zf + 0.05 * s), (1.3 * math.sin(a), 9.55 + 1.05 * math.cos(a), zf + 0.05 * s)], 0.05, CARVE, 5))
        parts.append(sphere(0.33, CARVE, 10, 6).S(1, 0.9, 0.4).T(0, 9.95, zf))

    # Pedestals of the sphinxes over the ends of the cornice, and the sphinxes
    for sx in (-1, 1):
        parts.append(B(min(sx * 2.3, sx * 3.85), max(sx * 2.3, sx * 3.85), 9.4, 10.35, -0.95, 0.95, stone()))
        parts.append(B(min(sx * 2.25, sx * 3.9), max(sx * 2.25, sx * 3.9), 10.3, 10.45, -1.0, 1.0, stone(1.06)))
        parts += sphinx(sx * 3.05, 10.45, 0.0, sx)

    # The royal arms on the apex
    parts += royal_arms(9.4 + PH + 0.05)

    # Iron: the gate in the arch, and the railings between the central body and the end piers
    parts += arch_gate(R, YS)
    for sx in (-1, 1):
        rail = railing(5.85, 9.3, 2.5)
        parts += rail if sx > 0 else [mirror_x(m) for m in rail]

    # The two loose end piers (x 9.3–10.9 shaft), with their flaming vases
    pier = [B(9.2, 11.05, FOUND, 0.7, -0.95, 0.95, GRANITE),
            B(9.3, 10.9, 0.7, 5.25, -0.8, 0.8, stone(1.02)),
            B(9.2, 11.0, 5.2, 5.6, -0.9, 0.9, stone(1.06)),
            B(9.55, 10.65, 5.6, 5.85, -0.55, 0.55, stone()),
            vase(10.1, 5.85, 0.0, 0.85)]
    for k in range(8):
        y = 0.7 + k * 0.57
        for s in (-1, 1):
            pier.append(B(9.3, 10.9, y, y + 0.03, *sorted((s * 0.8, s * 0.82)), stone(0.86)))
    parts += both_x(pier)
    return merge(parts)


def main():
    out = ROOT / "models" / "Q962107"
    out.mkdir(parents=True, exist_ok=True)
    m = gate()
    tris, size = write_glb(out / "model.glb", [("puerta_de_hierro", m)], roughness=0.9)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
