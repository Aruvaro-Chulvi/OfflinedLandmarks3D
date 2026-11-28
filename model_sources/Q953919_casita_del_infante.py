"""
Casita del Infante, or Casita de Arriba (Q953919), San Lorenzo de El Escorial — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Juan de Villanueva (1739–1811), 1771–1773, for the Infante Gabriel de Borbón; public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 315210758: a square block 16.98 × 14.48 m with a central body 7.45 m wide standing out
  1.07 m on both long façades (total depth 16.61 m); long sides at 58.3°, so the façades face 328.3° and 148.3°.
- The PNOA orthophoto (IGN) shows the road and the entrance gate on the north-west and the boxwood parterre with
  its round fountain on the south-east: the entrance façade, with the porch and the sphinxes, faces 328.3°.
- Heights measured on near-frontal photos of both façades (Wikimedia Commons, «Casita de Arriba.jpg» and
  «Casa del Infante Don Gabriel-Escorial-II.JPG», scaled by the OSM widths, ±5 %): walls and granite frieze up
  to the cornice at 5.45 m, the granite attic of the central body to 6.6 m, slate-and-lead mansard roofs with
  dormers on the side bodies, the hipped roof of the central body to 7.8 m and the lantern with its oculus to
  about 9.1 m, weather vane to 10.2 m.
- Entrance (north-west): a recessed porch with two columns and an iron grille, an arched door on each side, a
  round plaque on the attic, steps up to the floor (0.8 m) between two Egyptian-style sphinxes on pedestals.
- Garden (south-east): granite central body with the arched door under a small pediment and two windows;
  one window with its granite lintel on each side body.
Granite (the stone of El Escorial) for the plinth, the central bodies, corners, frieze and cornice; the walls of
the side bodies are rendered in a pale pink; roofs of slate and lead.

Axes (docs/MODEL_SPEC.md): +Y up, the entrance façade faces +Z, +X to the north-east (heading − 90°),
origin on the ground in the middle of the building; y = 0 is the ground in front of the entrance steps.

    python model_sources/Q953919_casita_del_infante.py   -> models/Q953919/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, quad, extrude, offset, crumple, noise3, write_glb  # noqa: E402

rng = np.random.default_rng(1771)

GRANITE = np.array((0.77, 0.74, 0.68))
STUCCO = (0.87, 0.775, 0.70)
SLATE = (0.33, 0.32, 0.34)
LEAD = (0.45, 0.43, 0.42)
SHUTTER = (0.93, 0.93, 0.90)
DOOR = (0.80, 0.80, 0.78)
GLASS = (0.13, 0.15, 0.17)
IRON = (0.10, 0.10, 0.11)
SCULPT = (0.74, 0.70, 0.63)
FOUND = -2.5

XW, ZW = 8.49, 7.24          # half-size of the square block
XC, ZC = 3.72, 8.30          # half-width of the central bodies, and how far they reach
FLOOR = 0.8                  # floor of the house above the ground at the entrance
EAVE = 5.45                  # top of the cornice
ATTIC = 6.6                  # top of the granite attic of the central body


def granite(k=1.0):
    return tuple(np.clip(GRANITE * k * (1 + rng.normal(0, 0.02)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
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


def frustum(ax, az, y0, bx, bz, y1, c, top=True):
    """A roof from a rectangle ±ax × ±az at y0 up to ±bx × ±bz at y1 (bx or bz may be 0: a ridge)."""
    lo = [(-ax, y0, -az), (ax, y0, -az), (ax, y0, az), (-ax, y0, az)]
    hi = [(-bx, y1, -bz), (bx, y1, -bz), (bx, y1, bz), (-bx, y1, bz)]
    V = lo + hi
    F = []
    for i in range(4):
        j = (i + 1) % 4
        F += [(i, 4 + j, j), (i, 4 + i, 4 + j)]
    if top and bx > 0 and bz > 0:
        F += [(4, 6, 5), (4, 7, 6)]
    return Mesh(V, F, c)


def rect_ring(x, z):
    return [(-x, -z), (x, -z), (x, z), (-x, z)]


# the plan: square block with the two central bodies standing out
PLAN = [(-XW, -ZW), (-XC, -ZW), (-XC, -ZC), (XC, -ZC), (XC, -ZW), (XW, -ZW),
        (XW, ZW), (XC, ZW), (XC, ZC), (-XC, ZC), (-XC, ZW), (-XW, ZW)]


# ── Openings ─────────────────────────────────────────────────────────────────
def window(x, y0, y1, zf, side, w=1.15, lintel=True, axis="z"):
    """A window with white louvred shutters closed, a granite frame and sill, and a cornice lintel."""
    out = []
    n = ("+z" if side > 0 else "-z") if axis == "z" else ("+x" if side > 0 else "-x")
    h = y1 - y0

    def Q(cx, cy, ww, hh, c, eps):
        if axis == "z":
            return quad(cx, cy, zf, ww, hh, n, c, eps=eps)
        return quad(zf, cy, cx, ww, hh, n, c, eps=eps)

    def BX(a0, a1, b0, b1, d0, d1, c):            # a box in "along the façade, up, out of it" coordinates
        o0, o1 = sorted((zf + d0 * side, zf + d1 * side))
        if axis == "z":
            return B(a0, a1, b0, b1, o0, o1, c)
        return B(o0, o1, b0, b1, a0, a1, c)

    out.append(BX(x - w / 2 - 0.16, x + w / 2 + 0.16, y0 - 0.12, y1 + 0.12, 0, 0.06, granite(1.04)))   # frame
    out.append(Q(x, (y0 + y1) / 2, w, h, SHUTTER, 0.08))
    for k in range(1, 12):                                                            # louvres
        out.append(Q(x, y0 + h * k / 12, w * 0.98, 0.025, (0.72, 0.72, 0.70), 0.09))
    out.append(Q(x, (y0 + y1) / 2, 0.03, h, (0.6, 0.6, 0.58), 0.095))
    out.append(BX(x - w / 2 - 0.25, x + w / 2 + 0.25, y0 - 0.2, y0 - 0.05, 0, 0.18, granite(1.06)))  # sill
    if lintel:
        out.append(BX(x - w / 2 - 0.35, x + w / 2 + 0.35, y1 + 0.3, y1 + 0.5, 0, 0.2, granite(1.06)))
        out.append(BX(x - w / 2 - 0.2, x + w / 2 + 0.2, y1 + 0.12, y1 + 0.3, 0, 0.1, granite(1.02)))
    return out


def arched_door(x, y0, w, h, zf, side, c=DOOR, glazed=False):
    """A door with a round head: the leaf (rectangle + half disc) and a granite frame."""
    r = w / 2
    n = "+z" if side > 0 else "-z"
    out = [quad(x, y0 + (h - r) / 2, zf, w, h - r, n, c, eps=0.06)]
    seg = 12
    V = [(x, y0 + h - r, zf + 0.06 * side)] + [(x + r * math.cos(a), y0 + h - r + r * math.sin(a), zf + 0.06 * side)
                                              for a in np.linspace(0, math.pi, seg + 1)]
    F = [(0, i + 1, i + 2) for i in range(seg)]
    m = Mesh(V, F, GLASS if glazed else c)
    if side < 0:
        m.f = m.f[:, ::-1]
    out.append(m)
    out.append(tube([(x + (r + 0.1) * math.cos(a), y0 + h - r + (r + 0.1) * math.sin(a), zf + 0.04 * side)
                     for a in np.linspace(0, math.pi, 10)], 0.1, granite(1.06), 6))
    for s in (-1, 1):
        lo, hi = sorted((zf, zf + 0.08 * side))
        out.append(B(x + s * (r + 0.02) - 0.1, x + s * (r + 0.02) + 0.1, y0, y0 + h - r, lo, hi, granite(1.06)))
    if glazed:
        out.append(quad(x, y0 + (h - r) / 2, zf, w * 0.75, (h - r) * 0.8, n, GLASS, eps=0.07))
    return out


def sphinx(x, y, z):
    """A seated Egyptian-style sphinx facing +z: lion body, upright chest and a head with its nemes."""
    c = SCULPT
    out = [sphere(0.5, c, 12, 7).S(0.55, 0.55, 1.0).T(x, y + 0.3, z - 0.1),                   # body
           sphere(0.3, c, 10, 6).S(1.0, 1.4, 0.8).T(x, y + 0.6, z + 0.3),                      # chest
           sphere(0.2, c, 10, 6).S(1.0, 1.15, 1.0).T(x, y + 1.1, z + 0.35),                    # head
           lathe([(0.32, 0), (0.28, 0.3), (0.16, 0.5), (0, 0.55)], c, 10).S(1.0, 1.0, 0.6).T(x, y + 0.8, z + 0.28)]  # nemes
    for s in (-1, 1):
        out.append(B(x + s * 0.13 - 0.08, x + s * 0.13 + 0.08, y, y + 0.14, z + 0.2, z + 0.62, c))  # forepaws
    return out


def dormer(x, z0, side, yb=5.55):
    """A small dormer on the mansard, its front face at z0 (side +1 faces +z)."""
    w, d, h = 1.05, 0.9, 0.85
    zb = z0 - d * side
    lo, hi = sorted((z0, zb))
    out = [B(x - w / 2, x + w / 2, yb, yb + h, lo, hi, LEAD)]
    roof = frustum(w / 2 + 0.1, d / 2 + 0.05, 0.0, w / 2 + 0.1, 0.0, 0.32, LEAD).T(x, yb + h, (z0 + zb) / 2 + 0.05 * side)
    out.append(roof)
    out.append(quad(x, yb + h / 2, z0, 0.6, 0.5, "+z" if side > 0 else "-z", GLASS, eps=0.02))
    out.append(quad(x, yb + h / 2, z0, 0.72, 0.62, "+z" if side > 0 else "-z", SHUTTER, eps=0.01))
    return out


# ── The house ────────────────────────────────────────────────────────────────
def house():
    parts = []
    # Walls: rendered side bodies, granite central bodies, granite plinth, corners, frieze.
    # The square block is one prism; the central bodies are granite boxes in front of it (the entrance one
    # built round its porch).
    core = rect_ring(XW, ZW)
    parts.append(extrude(core, FOUND, EAVE - 0.4, STUCCO))
    parts.append(extrude(offset(core, 0.06), FOUND, FLOOR + 0.25, granite(0.96)))                # plinth
    for s in (-1, 1):
        lo, hi = sorted((s * (ZW - 0.1), s * ZC))
        if s < 0:
            parts.append(B(-XC, XC, FOUND, EAVE - 0.4, lo, hi, granite()))
        else:
            parts.append(B(-XC, -1.3, FOUND, EAVE - 0.4, lo, hi, granite()))
            parts.append(B(1.3, XC, FOUND, EAVE - 0.4, lo, hi, granite()))
            parts.append(B(-1.3, 1.3, 4.3, EAVE - 0.4, lo, hi, granite()))
            parts.append(B(-1.3, 1.3, FOUND, FLOOR - 0.05, lo, hi, granite()))
        for k in range(1, 9):                                                                 # ashlar joints
            y = FLOOR + 0.25 + k * 0.42
            if y > 4.3:
                break
            for x0, x1 in (((-XC - 0.02, XC + 0.02),) if s < 0 else ((-XC - 0.02, -1.3), (1.3, XC + 0.02))):
                parts.append(B(x0, x1, y, y + 0.025, *sorted((s * ZC, s * (ZC + 0.02))), granite(0.86)))
    parts.append(extrude(offset(PLAN, 0.04), 4.35, EAVE - 0.4, granite()))                        # frieze
    parts.append(extrude(offset(PLAN, 0.2), EAVE - 0.4, EAVE - 0.2, granite(1.06)))              # cornice, two steps
    parts.append(extrude(offset(PLAN, 0.36), EAVE - 0.2, EAVE, granite(1.08)))
    for s in (-1, 1):                                                                          # granite corners
        for sx in (-1, 1):
            parts.append(B(*sorted((sx * (XW - 0.6), sx * (XW + 0.05))), FLOOR + 0.25, 4.35, *sorted((s * (ZW - 0.6), s * (ZW + 0.05))), granite(1.02)))
            parts.append(B(*sorted((sx * XC, sx * (XC + 0.55))), FLOOR + 0.25, 4.35, *sorted((s * (ZW - 0.1), s * (ZW + 0.05))), granite(1.02)))

    # Entrance façade (+z): the porch, the doors, the windows, the steps and the sphinxes
    zf = ZC
    parts.append(B(-1.32, 1.32, FLOOR - 0.05, FLOOR, ZC - 0.95, ZC + 0.04, granite(1.05)))      # its floor
    parts.append(quad(0, (FLOOR + 4.3) / 2, ZC - 0.94, 2.6, 4.3 - FLOOR, "+z", (0.42, 0.40, 0.37), eps=0.0))
    for k in range(13):                                                                       # the grille
        x = -0.75 + 1.5 * k / 12
        parts.append(B(x - 0.02, x + 0.02, FLOOR, 4.25, ZC - 0.62, ZC - 0.58, IRON))
    for y in (FLOOR + 0.3, 2.1, 3.0, 4.15):
        parts.append(B(-0.78, 0.78, y, y + 0.05, ZC - 0.63, ZC - 0.57, IRON))
    for x in (-0.98, 0.98):                                                                   # two columns
        parts.append(cyl(0.27, 0.27, 0.25, granite(1.06), 14).T(x, FLOOR, ZC - 0.25))
        parts.append(cyl(0.21, 0.18, 4.3 - FLOOR - 0.6, granite(1.04), 14, caps=False).T(x, FLOOR + 0.25, ZC - 0.25))
        parts.append(cyl(0.2, 0.26, 0.2, granite(1.06), 14).T(x, 3.95, ZC - 0.25))
        parts.append(B(x - 0.3, x + 0.3, 4.15, 4.3, ZC - 0.55, ZC + 0.05, granite(1.06)))
        for sx in (-1, 1):
            parts.append(cyl(0.08, 0.08, 0.5, granite(1.08), 8).R(math.pi / 2, 0, 0).T(x + sx * 0.24, 4.05, ZC - 0.5))
    parts.append(B(-1.45, 1.45, 4.3, 4.45, ZC - 0.02, ZC + 0.08, granite(1.08)))             # lintel band of the porch
    for sx in (-1, 1):
        parts += arched_door(sx * 2.48, FLOOR, 1.0, 2.6, zf, +1)
        parts.append(B(sx * 2.48 - 0.45, sx * 2.48 + 0.45, 3.65, 4.2, zf, zf + 0.08, granite(1.08)))   # framed panel
        parts.append(B(sx * 2.48 - 0.12, sx * 2.48 + 0.12, 3.75, 4.05, zf + 0.08, zf + 0.22, (0.25, 0.22, 0.18)))  # lantern
        parts += window(sx * 6.0, 1.45, 3.3, ZW, +1)
    # steps up to the porch, and the sphinxes on their pedestals
    for k in range(4):
        parts.append(B(-3.5, 3.5, FOUND, FLOOR - k * 0.2, ZC, ZC + 0.3 * (k + 1), granite(1.0 + 0.02 * k)))
    for sx in (-1, 1):
        parts.append(B(sx * 4.3 - 0.5, sx * 4.3 + 0.5, FOUND, 1.4, ZC + 0.2, ZC + 1.2, granite()))
        parts.append(B(sx * 4.3 - 0.58, sx * 4.3 + 0.58, 1.4, 1.55, ZC + 0.12, ZC + 1.28, granite(1.07)))
        parts += sphinx(sx * 4.3, 1.55, ZC + 0.65)

    # Garden façade (−z)
    zg = -ZC
    gs = [arched_door(0.0, FLOOR, 1.35, 2.75, zg, -1, c=SHUTTER, glazed=True)]
    for p in gs:
        parts += p
    ped = [(-1.2, 3.75), (1.2, 3.75), (0.0, 4.25)]                                            # small pediment
    for a, b in ((0, 1), (1, 2), (2, 0)):
        parts.append(tube([(ped[a][0], ped[a][1], zg - 0.12), (ped[b][0], ped[b][1], zg - 0.12)], 0.09, granite(1.08), 6))
    parts.append(B(-1.2, 1.2, 3.62, 3.75, zg - 0.22, zg, granite(1.08)))
    for k in range(2):
        parts.append(B(-1.4, 1.4, FOUND, FLOOR - 0.4 * k, -ZC - 0.5 * (k + 1), -ZC, granite(1.02)))
    for sx in (-1, 1):
        parts += window(sx * 2.3, 1.45, 3.1, zg, -1, w=1.0, lintel=False)
        parts += window(sx * 5.75, 1.45, 3.1, -ZW, -1)
    # Side façades (±x): two windows each
    for sx in (-1, 1):
        for z in (-3.6, 3.6):
            parts += window(z, 1.45, 3.1, sx * XW, sx, axis="x")

    # Roofs: mansard of the whole block, then the low hip over it
    parts.append(frustum(XW + 0.2, ZW + 0.2, EAVE, XW - 0.85, ZW - 0.85, EAVE + 1.25, SLATE, top=False))
    parts.append(frustum(XW - 0.85, ZW - 0.85, EAVE + 1.25, 3.0, 2.0, EAVE + 1.85, LEAD))
    for s in (-1, 1):
        for sx in (-1, 1):
            parts += dormer(sx * 5.9, s * (ZW - 0.05), s)
        for z in (0.0,):
            m = dormer(0.0, 0.0, 1)
            for d in m:
                d.R(0, s * math.pi / 2, 0).T(s * (XW - 0.05), 0, z)
            parts += m
    # Granite attic of the central body, across the whole depth, with its small windows and cornice
    parts.append(B(-XC + 0.1, XC - 0.1, EAVE, ATTIC, -ZC + 0.25, ZC - 0.25, granite()))
    parts.append(B(-XC - 0.05, XC + 0.05, ATTIC - 0.18, ATTIC, -ZC + 0.08, ZC - 0.08, granite(1.08)))
    for s in (-1, 1):
        zf = s * (ZC - 0.25)
        if s < 0:
            for x in (-2.4, 0.0, 2.4):
                parts.append(quad(x, 6.0, zf, 0.75, 0.48, "-z", GLASS, eps=0.02))
                parts.append(B(x - 0.48, x + 0.48, 5.68, 6.33, *sorted((zf, zf - 0.05)), granite(1.07)))
        else:
            parts.append(cyl(0.48, 0.48, 0.08, granite(1.1), 20).R(math.pi / 2, 0, 0).T(0, 6.0, zf))   # round plaque
            parts.append(B(-2.9, -1.5, 5.75, 6.3, zf, zf + 0.05, granite(1.05)))
            parts.append(B(1.5, 2.9, 5.75, 6.3, zf, zf + 0.05, granite(1.05)))
    # hipped roof of the central body
    parts.append(frustum(XC + 0.1, ZC - 0.1, ATTIC, 1.4, 2.3, 7.8, LEAD).T(0, 0, 0))
    # the lantern with its oculus, lead cap, ball and weather vane
    parts.append(B(-0.95, 0.95, 7.3, 8.6, -0.95, 0.95, LEAD))
    for s in (-1, 1):
        parts.append(cyl(0.45, 0.45, 0.05, GLASS, 18).R(math.pi / 2, 0, 0).T(0, 7.95, s * 0.95 - (0.0 if s > 0 else 0.05)))
        parts.append(tube([(0.55 * math.cos(a), 7.95 + 0.55 * math.sin(a), s * 0.98) for a in np.linspace(0, 2 * math.pi, 21)], 0.07, granite(1.1), 6))
    cap = lathe([(1.05, 0), (0.95, 0.2), (0.6, 0.42), (0.2, 0.52), (0, 0.55)], LEAD, 16).T(0, 8.6, 0)
    parts.append(cap)
    parts.append(sphere(0.12, (0.5, 0.45, 0.3), 8, 5).T(0, 9.25, 0))
    parts.append(B(-0.02, 0.02, 9.25, 10.2, -0.02, 0.02, IRON))
    parts.append(B(-0.35, 0.35, 9.85, 9.88, -0.015, 0.015, IRON))
    parts.append(B(-0.02, 0.02, 9.6, 9.63, -0.25, 0.25, IRON))
    parts.append(Mesh([(0.05, 10.0, 0), (0.45, 10.05, 0), (0.05, 10.15, 0)], [(0, 1, 2), (0, 2, 1)], IRON))  # the vane
    return merge(parts)


# ── The gardens ──────────────────────────────────────────────────────────────
# Italian terraced gardens, in model metres. The grid of paths, the round fountains, the walls of the central
# terrace, the steps and the trees come from OpenStreetMap (ways 315210748, 905236755/58, 9052367xx, 1010227xxx;
# tree nodes 9220720713–16, 93210859xx, 93223483xx); the shape of the parterres from the PNOA orthophoto; the
# levels of the terraces from the IGN MDT05 (central terrace 0, side terraces −1, the upper ones −2).
SAND = (0.84, 0.78, 0.66)
PAVE = (0.80, 0.77, 0.71)
BOX = (0.19, 0.31, 0.14)          # boxwood hedges
BOX_D = (0.15, 0.26, 0.12)
GRASS = (0.40, 0.50, 0.25)
FLOWER = (0.62, 0.30, 0.30)
WATER = (0.22, 0.36, 0.42)
BARK = (0.36, 0.27, 0.20)
GFOUND = -6.0

# terraces: (x0, x1, z0, z1, level)
TERRACES = [
    (-17.7, 16.9, -52.0, 14.0, 0.0),      # central terrace with the house
    (-17.7, 16.9, 14.0, 37.0, -1.0),      # entrance garden
    (-45.5, -17.7, -40.0, 22.0, -1.0),    # side terraces
    (16.9, 45.6, -40.0, 22.0, -1.0),
    (-45.5, -17.7, -70.4, -40.0, -2.0),   # upper terraces
    (-17.7, 16.9, -70.4, -52.0, -2.0),
    (16.9, 45.6, -70.4, -40.0, -2.0),
]


def level_at(x, z):
    for x0, x1, z0, z1, lv in TERRACES:
        if x0 <= x <= x1 and z0 <= z <= z1:
            return lv
    return -2.0


def ngon(cx, cz, r, n=20):
    return [(cx + r * math.cos(a), cz + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def chamfer_rect(x0, x1, z0, z1, ch=(1.6, 1.6, 1.6, 1.6)):
    """A rectangle with its corners cut (−x−z, +x−z, +x+z, −x+z)."""
    a, b, c, d = ch
    return [(x0 + a, z0), (x1 - b, z0), (x1, z0 + b), (x1, z1 - c), (x1 - c, z1), (x0 + d, z1), (x0, z1 - d), (x0, z0 + a)]


def notched_rect(x0, x1, z0, z1, ncx, ncz, nr, ch=1.4):
    """A rectangle whose +x edge has a round notch around the plaza centred at (ncx, ncz), radius nr."""
    t0 = math.acos(min(1.0, (ncx - x1) / nr))
    arc = [(ncx - nr * math.cos(t), ncz + nr * math.sin(t)) for t in np.linspace(-t0, t0, 11)]
    return [(x0, z0 + ch), (x0 + ch, z0), (x1, z0)] + arc + [(x1, z1), (x0 + ch, z1), (x0, z1 - ch)]


def hedge_ring(ring, y, h=0.55, w=0.45, c=BOX):
    return extrude(ring, y - 0.05, y + h, c, holes=[offset(ring, -w)])


def parterre(ring, y, kind=0, seed=0):
    """A boxwood parterre: an outer hedge, an inner hedge 0.9 m inside, grass in the middle and a pattern."""
    r = np.random.default_rng(seed)
    out = [hedge_ring(ring, y), hedge_ring(offset(ring, -1.35), y, h=0.4, w=0.35, c=BOX_D)]
    inner = offset(ring, -1.7)
    out.append(extrude(inner, y - 0.05, y + 0.06, GRASS))
    xs = [p[0] for p in inner]; zs = [p[1] for p in inner]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    w, d = (max(xs) - min(xs)) / 2, (max(zs) - min(zs)) / 2
    if kind == 0:                                   # a clipped yew in the middle
        out.append(lathe([(0.0, 0), (0.9, 0.3), (1.1, 1.2), (0.8, 2.2), (0.3, 2.9), (0, 3.1)], BOX_D, 10).T(cx, y, cz))
    elif kind == 1:                                 # four little hedge knots (the cross pattern)
        for sx in (-1, 1):
            for sz in (-1, 1):
                k = ngon(cx + sx * w * 0.45, cz + sz * d * 0.45, min(w, d) * 0.32, 10)
                out.append(hedge_ring(k, y, h=0.35, w=0.25, c=BOX_D))
        out.append(extrude(ngon(cx, cz, 0.6, 8), y, y + 0.5, FLOWER))
    else:                                           # flower beds along a cross of hedge
        out.append(B(cx - 0.18, cx + 0.18, y, y + 0.4, cz - d * 0.8, cz + d * 0.8, BOX_D))
        out.append(B(cx - w * 0.8, cx + w * 0.8, y, y + 0.4, cz - 0.18, cz + 0.18, BOX_D))
        for sx in (-1, 1):
            for sz in (-1, 1):
                out.append(B(cx + sx * w * 0.5 - 0.6, cx + sx * w * 0.5 + 0.6, y, y + 0.15,
                             cz + sz * d * 0.5 - 0.9, cz + sz * d * 0.5 + 0.9, FLOWER if r.random() < 0.5 else (0.75, 0.62, 0.35)))
    return out


def fountain(cx, cz, y, r, bowl=True, pave=0.0):
    out = []
    if pave:
        out.append(extrude(ngon(cx, cz, pave, 28), y - 0.05, y + 0.02, PAVE))
    out.append(extrude(ngon(cx, cz, r, 24), y, y + 0.5, granite(1.05), holes=[ngon(cx, cz, r - 0.3, 24)]))
    out.append(extrude(ngon(cx, cz, r - 0.3, 24), y - 0.05, y + 0.32, WATER))
    if bowl:
        out.append(lathe([(0.35, 0), (0.25, 0.8), (0.3, 1.0), (1.0, 1.15), (1.05, 1.3), (0.0, 1.3)], granite(1.06), 16).T(cx, y + 0.3, cz))
        out.append(cyl(0.12, 0.05, 0.6, granite(1.08), 8).T(cx, y + 1.6, cz))
    return out


def stairs(x0, x1, z0, z1, y_hi, y_lo, axis):
    """Steps going down from y_hi to y_lo, from z0 towards z1 (axis 'z') or from x0 towards x1 (axis 'x')."""
    out = []
    n = max(2, int(round((y_hi - y_lo) / 0.17)))
    for k in range(n):
        top = y_hi - (k + 1) * (y_hi - y_lo) / n
        if axis == "z":
            za, zb = z0 + (z1 - z0) * k / n, z0 + (z1 - z0) * (k + 1) / n
            out.append(B(x0, x1, GFOUND, top + (y_hi - y_lo) / n, *sorted((za, zb)), granite(1.03)))
        else:
            xa, xb = x0 + (x1 - x0) * k / n, x0 + (x1 - x0) * (k + 1) / n
            out.append(B(*sorted((xa, xb)), GFOUND, top + (y_hi - y_lo) / n, z0, z1, granite(1.03)))
    return out


def railing(pts, y, h=1.2, c=IRON):
    out = []
    for (xa, za), (xb, zb) in zip(pts, pts[1:]):
        L = math.hypot(xb - xa, zb - za)
        n = max(1, int(L / 2.5))
        for k in range(n + 1):
            x, z = xa + (xb - xa) * k / n, za + (zb - za) * k / n
            out.append(B(x - 0.04, x + 0.04, y, y + h, z - 0.04, z + 0.04, c))
        for yy in (y + 0.15, y + h - 0.1):
            out.append(tube([(xa, yy, za), (xb, yy, zb)], 0.025, c, 4))
    return out


# ── Trees (positions, heights and crowns from OpenStreetMap) ────────────────
# Each crown is built from many irregular clumps of foliage (spheres dented with 3D noise), darker underneath
# and lighter on top, carried by visible boughs, with the habit of each species.
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
    """A bough or trunk from p0 to p1, tapering from r0 to r1, with a slight bend upwards."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    out = []
    pts = [p0 + (p1 - p0) * t + np.array([0, bend * math.sin(math.pi * t), 0]) for t in np.linspace(0, 1, n + 1)]
    for k in range(n):
        r = r0 + (r1 - r0) * (k + 0.5) / n
        out.append(tube([pts[k], pts[k + 1]], r, col, seg))
    return out


def bark(col, seed):
    rr = np.random.default_rng(seed)
    return tuple(np.clip(np.asarray(col) * (0.92 + 0.16 * rr.random()), 0, 1))


def sequoia(x, z, y, h, crown, seed):
    """Coast redwood: a straight red trunk with a flared foot, short drooping boughs all the way up, a narrow
    irregular spire."""
    rr = np.random.default_rng(seed)
    BK = bark((0.50, 0.28, 0.18), seed)
    out = [cyl(1.6, 1.05, 2.2, BK, 10).T(x, y, z)]
    out += limb((x, y + 2.0, z), (x, y + h * 0.97, z), 1.05, 0.15, BK, 8, n=4)
    FOL = (0.15, 0.27, 0.13)
    for k in range(100):
        t = 0.12 + 0.86 * rr.random() ** 0.85
        R = crown / 2 * (1 - t) ** 0.7 + 0.7
        yb = y + h * t
        a = rr.uniform(0, 2 * math.pi)
        d = R * (0.3 + 0.55 * rr.random())
        tip = (x + d * math.cos(a), yb - 0.4 - 0.6 * rr.random(), z + d * math.sin(a))
        if k % 2 == 0:
            out += limb((x, yb + 0.4, z), tip, 0.14, 0.07, BK, 4, bend=0.25, n=1)
        out.append(blob(tip[0], tip[1], tip[2], max(0.8, R * (0.45 + 0.25 * rr.random())), 0.75, FOL, seed + k, seg=8, rings=5))
    out.append(blob(x, y + h * 0.985, z, 0.7, 1.4, FOL, seed + 999))
    return out


def cedar(x, z, y, h, crown, seed):
    """Lebanon cedar: a massive trunk that splits into great ascending limbs, carrying wide, flat, separate
    tables of blue-green foliage."""
    rr = np.random.default_rng(seed)
    BK = bark((0.38, 0.33, 0.29), seed)
    FOL = (0.19, 0.29, 0.25)
    out = [cyl(1.8, 1.3, 1.5, BK, 10).T(x, y, z)]
    fork = (x, y + h * 0.28, z)
    out += limb((x, y + 1.4, z), fork, 1.3, 1.1, BK, 8, n=2)
    tops = []
    for j in range(5):
        a = j * 2 * math.pi / 5 + rr.normal(0, 0.3)
        d = crown * (0.16 + 0.08 * rr.random())
        top = (x + d * math.cos(a), y + h * (0.55 + 0.25 * rr.random()), z + d * math.sin(a))
        out += limb(fork, top, 0.75, 0.35, BK, 7, n=3)
        tops.append(top)
    out += limb(fork, (x, y + h * 0.9, z), 0.7, 0.25, BK, 7, n=3)
    tops.append((x, y + h * 0.9, z))
    tiers = [(0.30, 7, 0.46), (0.42, 7, 0.44), (0.54, 6, 0.40), (0.65, 6, 0.34), (0.76, 5, 0.27), (0.86, 4, 0.2), (0.95, 2, 0.1)]
    for ti, (t, npad, rad) in enumerate(tiers):
        for j in range(npad):
            a = j * 2 * math.pi / npad + ti * 0.7 + rr.normal(0, 0.25)
            d = crown * rad * (0.55 + 0.45 * rr.random())
            py = y + h * t + rr.normal(0, 0.5)
            px, pz = x + d * math.cos(a), z + d * math.sin(a)
            src = min(tops, key=lambda p: (p[0] - px) ** 2 + (p[2] - pz) ** 2 + 0.3 * (p[1] - py) ** 2)
            base = (src[0] + (x - src[0]) * 0.3, min(src[1], py - 0.5), src[2] + (z - src[2]) * 0.3)
            out += limb(base, (px, py - 0.4, pz), 0.32, 0.12, BK, 5, n=1)
            pr = crown * (0.11 + 0.04 * rr.random()) * (1 - 0.35 * t)
            for b in range(3):                                   # each table is a few flat clumps
                ba = rr.uniform(0, 2 * math.pi); bd = pr * 0.6 * rr.random()
                out.append(blob(px + bd * math.cos(ba), py + rr.normal(0, 0.15), pz + bd * math.sin(ba),
                                pr * (0.75 + 0.3 * rr.random()), 0.26, FOL, seed + ti * 31 + j * 5 + b))
    return out


def pine(x, z, y, h, crown, seed, kind="jeffrey"):
    """Pines. Jeffrey: a straight red-brown trunk with whorls of boughs and an open, irregular cone of tufts.
    Sabine (grey pine): a trunk that forks into leaning stems and a thin, open, grey-green crown."""
    rr = np.random.default_rng(seed)
    if kind == "sabine":
        BK = bark((0.36, 0.33, 0.30), seed); FOL = (0.36, 0.43, 0.32)
        fork = (x + rr.normal(0, 0.4), y + h * 0.38, z + rr.normal(0, 0.4))
        out = limb((x, y, z), fork, 0.65, 0.5, BK, 8, n=3)
        stems = []
        for j in range(3):
            a = j * 2.1 + rr.normal(0, 0.3)
            lean = crown * (0.12 + 0.08 * rr.random())
            top = (x + lean * math.cos(a), y + h * (0.82 + 0.15 * rr.random()), z + lean * math.sin(a))
            out += limb(fork, top, 0.42, 0.12, BK, 6, n=3)
            stems.append(top)
        for k in range(34):
            s = stems[k % 3]
            t = 0.45 + 0.55 * rr.random()
            sp = (fork[0] + (s[0] - fork[0]) * t, fork[1] + (s[1] - fork[1]) * t, fork[2] + (s[2] - fork[2]) * t)
            a = rr.uniform(0, 2 * math.pi); d = crown * (0.1 + 0.3 * rr.random())
            tip = (sp[0] + d * math.cos(a), sp[1] + rr.uniform(-0.5, 1.5), sp[2] + d * math.sin(a))
            out += limb(sp, tip, 0.12, 0.05, BK, 4, n=1)
            out.append(blob(*tip, crown * (0.08 + 0.05 * rr.random()), 0.6, FOL, seed + k))
        return out
    BK = bark((0.47, 0.30, 0.21), seed); FOL = (0.20, 0.32, 0.18)
    out = limb((x, y, z), (x + rr.normal(0, 0.3), y + h * 0.97, z + rr.normal(0, 0.3)), 0.55, 0.12, BK, 8, n=4)
    n = 9
    for k in range(n):
        t = 0.38 + 0.58 * k / (n - 1)
        R = crown / 2 * (1.05 - t) / 0.65 * (0.75 + 0.4 * rr.random())
        yb = y + h * t
        for j in range(3):
            a = k * 1.9 + j * 2.09 + rr.normal(0, 0.3)
            tip = (x + R * math.cos(a), yb + 0.4 + 0.3 * rr.random(), z + R * math.sin(a))
            out += limb((x, yb, z), tip, 0.18, 0.07, BK, 5, bend=0.4, n=1)
            out.append(blob(*tip, max(0.8, R * 0.5), 0.6, FOL, seed + k * 5 + j))
    out.append(blob(x, y + h * 0.97, z, 1.0, 1.1, FOL, seed + 77))
    return out


def sabine(x, z, y, h, crown, seed):
    return pine(x, z, y, h, crown, seed, "sabine")


def conifer(x, z, y, h, crown, seed):
    """A big needle-leaved tree of unrecorded species (an old cedar or pine on the orthophoto): a short trunk
    and a broad, dense, rounded crown of clumps."""
    rr = np.random.default_rng(seed)
    BK = bark((0.38, 0.30, 0.24), seed); FOL = (0.17, 0.29, 0.17)
    out = limb((x, y, z), (x, y + h * 0.75, z), 0.7, 0.25, BK, 8, n=3)
    cy, ry, rxz = y + h * 0.58, h * 0.4, crown / 2
    for k in range(34):
        u = rr.normal(size=3); u /= np.linalg.norm(u)
        u[1] = abs(u[1]) * 0.9 - 0.3 if rr.random() < 0.8 else u[1]
        f = 0.7 + 0.3 * rr.random()
        p = (x + u[0] * rxz * f, cy + u[1] * ry * f, z + u[2] * rxz * f)
        out += limb((x, min(p[1], y + h * 0.7), z), p, 0.2, 0.08, BK, 4, n=1)
        out.append(blob(*p, crown * (0.14 + 0.06 * rr.random()), 0.7, FOL, seed + k))
    return out


TREES = [  # (x, z, kind, height, crown)  — OSM nodes
    (26.0, -48.2, cedar, 27, 30),        # Cedro del Líbano de la Casita de Arriba
    (36.1, 11.6, sabine, 25, 17),        # Pino de Sabine de la Casita de Arriba I
    (23.7, -9.7, pine, 15, 8),           # Pino de Jeffrey I
    (46.4, -35.4, pine, 14, 8),          # Pino de Jeffrey II
    (5.0, 17.9, sequoia, 30, 9),         # sequoias (height not mapped: about 30 m on the photos)
    (15.2, 19.3, sequoia, 28, 8),
    (-14.0, 18.8, sequoia, 30, 9),
    (-14.8, 33.1, sequoia, 28, 8),
    (-8.0, -43.9, conifer, 20, 12),      # needle-leaved trees without species
    (21.6, 13.3, conifer, 16, 8),
    (-25.7, -49.7, conifer, 22, 16),
]


def garden():
    parts = []
    # Terraces: solid platforms of sand with granite retaining walls, down into the ground
    for x0, x1, z0, z1, lv in TERRACES:
        parts.append(extrude(rect_ring((x1 - x0) / 2, (z1 - z0) / 2), GFOUND, lv, granite(0.95), top_c=SAND)
                     .T((x0 + x1) / 2, 0, (z0 + z1) / 2))
    # Parapets (0.4 m) on the sides of the central terrace, with gaps for the steps
    for x in (-17.7, 16.9):
        gap = (-1.8, 0.6) if x < 0 else (2.7, 4.9)
        for za, zb in ((-52.0, gap[0]), (gap[1], 14.0)):
            parts.append(B(x - 0.25, x + 0.25, -0.05, 0.4, za, zb, granite(1.04)))
    parts.append(B(-17.7, -1.6, -0.05, 0.4, -52.25, -51.75, granite(1.04)))
    parts.append(B(1.6, 16.9, -0.05, 0.4, -52.25, -51.75, granite(1.04)))
    parts.append(B(-17.7, -2.2, -0.05, 0.4, 13.75, 14.25, granite(1.04)))
    parts.append(B(2.2, 16.9, -0.05, 0.4, 13.75, 14.25, granite(1.04)))
    # Steps between the terraces
    parts += stairs(-1.6, 1.6, -52.0, -56.8, 0.0, -2.0, "z")
    parts += stairs(-2.2, 2.2, 14.0, 16.2, 0.0, -1.0, "z")
    parts += stairs(-17.7, -20.0, -1.8, 0.6, 0.0, -1.0, "x")
    parts += stairs(16.9, 19.2, 2.7, 4.9, 0.0, -1.0, "x")
    for sx in (-1, 1):
        parts += stairs(sx * 30.7 - 1.5, sx * 30.7 + 1.5, -40.0, -42.0, -1.0, -2.0, "z")

    # Central terrace: the fountain plaza in front of the garden door, the upper fountain, six parterres
    parts += fountain(0.0, -16.5, 0.0, 2.6, pave=4.3)
    parts += fountain(0.0, -36.7, 0.0, 2.2, bowl=False, pave=3.0)
    parts.append(cyl(0.1, 0.05, 1.2, granite(1.08), 8).T(0, 0.3, -36.7))
    for sx in (-1, 1):
        rings = (notched_rect(-14.6, -2.0, -21.6, -11.6, 0.0, -16.5, 5.0),
                 chamfer_rect(-14.6, -2.0, -35.2, -24.6, (1.4, 3.6, 1.4, 1.4)),
                 chamfer_rect(-14.6, -2.0, -50.5, -40.8))
        for k, ring in enumerate(rings):
            rg = ring if sx < 0 else [(-x, z) for x, z in ring]
            parts += parterre(rg, 0.0, kind=(0, 0, 2)[k], seed=k + 10 * (sx + 1))
    # Side terraces: two columns of three parterres each side, round fountains and hedge rings at the crossings
    for sx in (-1, 1):
        cols = ((-43.2, -32.6), (-28.8, -19.6))
        rows = ((-35.3, -19.6), (-16.5, -2.2), (0.8, 19.2))
        for ci, (xa, xb) in enumerate(cols):
            for ri, (za, zb) in enumerate(rows):
                x0, x1 = (xa, xb) if sx < 0 else (-xb, -xa)
                ring = chamfer_rect(x0, x1, za, zb, (2.2, 2.2, 2.2, 2.2))
                parts += parterre(ring, -1.0, kind=1 if (ci + ri) % 2 == 0 else 2, seed=100 + ci * 7 + ri * 3 + sx)
        cx = sx * 30.7
        parts += fountain(cx, -0.5, -1.0, 2.1, bowl=True, pave=3.2)
        parts += fountain(cx, -36.7, -1.0, 2.1, bowl=True, pave=3.0)
        parts.append(hedge_ring(ngon(cx, -18.1, 2.4, 16), -1.0, h=0.6, w=0.45))
        parts.append(extrude(ngon(cx, -18.1, 1.95, 16), -1.05, -0.7, FLOWER))
        # upper terraces: a large lawn parterre each side, under the big trees
        x0, x1 = (-43.0, -21.0) if sx < 0 else (20.0, 43.0)
        ring = chamfer_rect(x0, x1, -62.0, -43.0, (2.5, 2.5, 2.5, 2.5))
        parts.append(hedge_ring(ring, -2.0))
        parts.append(extrude(offset(ring, -0.45), -2.05, -1.94, GRASS))
    # Entrance garden: the path to the gate and two small walled flower gardens
    parts.append(B(-1.6, 1.6, -1.05, -0.97, 16.2, 37.0, PAVE))
    for sx in (-1, 1):
        for xa, xb in ((4.5, 9.8), (10.8, 16.2)):
            x0, x1 = (xa, xb) if sx > 0 else (-xb, -xa)
            ring = chamfer_rect(x0, x1, 29.0, 36.5, (1.0, 1.0, 1.0, 1.0))
            parts += parterre(ring, -1.0, kind=2, seed=200 + int(xa) + sx)
        # iron fences beside the house front (OSM ways 1010227621/22)
        parts += railing([(sx * 8.45, 7.4), (sx * 8.45, 14.0)], 0.0)
    # The garden fence on its granite curb (OSM way 1010371116), simplified to the rectangle of the terraces
    fence = [(-45.5, 22.0), (-45.5, -70.4), (45.6, -70.4), (45.6, 22.0)]
    for (xa, za), (xb, zb) in zip(fence, fence[1:]):
        x0, x1 = sorted((xa, xb)); z0, z1 = sorted((za, zb))
        lv = -2.0 if max(z0, z1) <= -40 else -1.0
        if abs(x1 - x0) < 0.01:            # along z: two pieces at their two levels
            parts.append(B(x0 - 0.3, x0 + 0.3, GFOUND, -1.6, -70.4, -40.0, granite(0.98)))
            parts.append(B(x0 - 0.3, x0 + 0.3, GFOUND, -0.6, -40.0, 22.0, granite(0.98)))
            parts += railing([(x0, -70.4), (x0, -40.0)], -1.6)
            parts += railing([(x0, -40.0), (x0, 22.0)], -0.6)
        else:
            parts.append(B(x0, x1, GFOUND, lv + 0.4, z0 - 0.3, z0 + 0.3, granite(0.98)))
            parts += railing([(x0, z0), (x1, z0)], lv + 0.4)
    # Trees
    for i, (x, z, kind, h, crown) in enumerate(TREES):
        parts += kind(x, z, level_at(x, z), h, crown, 500 + 37 * i)
    return merge(parts)


def main():
    out = ROOT / "models" / "Q953919"
    out.mkdir(parents=True, exist_ok=True)
    m = merge([house(), garden()])
    tris, size = write_glb(out / "model.glb", [("casita_del_infante", m)], roughness=0.9)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
