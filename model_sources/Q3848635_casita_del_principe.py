"""
Casita del Príncipe, or Casita de Abajo (Q3848635), El Escorial — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Juan de Villanueva (1739–1811), 1771–1775 for the Prince of Asturias, the future Carlos IV, enlarged
with the rear wing in 1781–1784; public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 86425434 and its building:parts (891230443–50), plus way 891450370: a long block of 83.8 m,
  long axis at 6.6°, so the main façade faces 96.6° (east, towards the round fountain and the gate). The main
  block 28.7 m × 7.9 m (Wikipedia: «una fachada de 27 metros»), the portico 6.3 m (7.8 m measured on the photos)
  standing out 1.8 m, the rear wing 7.6 m wide reaching 18.2 m back with its porch to 20.2 m, low bridges of
  4 m over the passages to the rear garden, and the two long one-floor end wings (23.5 × 9 m).
- The PNOA orthophoto (IGN) confirms the plan, the two hipped roofs of the side bodies, the tower in the middle,
  the hipped roof of the rear wing, and shows the gardens.
- Heights measured on near-frontal photos (Wikimedia Commons «Casita del Príncipe 1.jpg» and «El Escorial - Casa
  del Príncipe Don Carlos (Casita del Príncipe) 04.jpg» for the main façade, «… 02.jpg» for the rear porch),
  scaled by the widths above, ±5–8 %: the floor of the house 1.0 m above the walk in front; ground-floor
  cornice 5.0 m above the floor, the low upper floor of the side bodies to an eave at 6.4 m; four Tuscan
  columns 4.3 m carrying the entablature and the balcony at 5.4 m; the central tower to its cornice at 9.1 m,
  its lead roof to 10.3 m and the spire with its ball and vane to about 14.8 m. End wings: eave about 4 m.
  Rear porch (west): two Tuscan columns in antis between broad piers, arched door, attic to 5.7 m.
Granite ashlar (the warm, pale stone of El Escorial) for all the house; the end wings rendered pale ochre with
granite plinth, pilasters and frames; roofs of slate, the tower roof and spire of lead; white louvred shutters;
the balcony railing of painted iron.

Axes (docs/MODEL_SPEC.md): +Y up, the main façade faces +Z (east), +X to the north (heading − 90° = 6.6°),
origin in the middle of the main block on the axis of the portico; y = 0 is the walk in front of the house.

The gardens: the front garden (east) with the round fountain and eight parterres between radiating paths; the
rear garden (west) with its grid of twelve parterres, the small fountain and the two boxwood broderie
parterres at the end of the axis. Hedges, paths, fountains and walls from OpenStreetMap, the planting of the
parterres from the PNOA orthophoto, the levels from the IGN MDT05: the rear garden and the floor of the house
1.0 m above the walk, the front garden going down to −1.0 m at the fountain and −2.2 m at its far end, the
broderie terrace rising to +2.6 m. Trees: only those mapped in OpenStreetMap, each with the habit of its species.

    python model_sources/Q3848635_casita_del_principe.py   -> models/Q3848635/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, quad, extrude, offset, crumple, noise3, write_glb  # noqa: E402

rng = np.random.default_rng(1772)

GRANITE = np.array((0.80, 0.75, 0.64))
RENDER = (0.88, 0.81, 0.66)
SLATE = (0.36, 0.37, 0.40)
SLATE_L = (0.50, 0.51, 0.53)        # the paler roofs of the end wings
LEAD = (0.47, 0.48, 0.50)
SHUTTER = (0.94, 0.94, 0.91)
DOOR = (0.90, 0.90, 0.87)
GLASS = (0.13, 0.15, 0.17)
IRON = (0.10, 0.10, 0.11)
RAIL = (0.33, 0.43, 0.45)           # painted iron of the balcony
FOUND = -2.5

FLOOR = 1.0                          # floor of the house (and the rear garden) above the walk
XB, ZF, ZR = 14.4, 5.6, -2.3         # main block: half-width, front and rear façades
XT = 3.8                             # half-width of the central tower and of the rear wing
GF_CORN = 6.0                        # top of the ground-floor cornice
EAVE = 7.4                           # eave of the side bodies
T_CORN = 10.1                        # top of the cornice of the tower
ZW_END, ZW_PORCH = -18.2, -20.2      # rear wing and its porch
XE0, XE1, ZE_F = 18.4, 41.9, 6.7     # end wings
E_EAVE = 5.0


def granite(k=1.0):
    return tuple(np.clip(GRANITE * k * (1 + rng.normal(0, 0.02)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    x0, x1 = sorted((x0, x1)); z0, z1 = sorted((z0, z1)); y0, y1 = sorted((y0, y1))
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def mirror_x(m):
    n = m.copy(); n.v[:, 0] *= -1; n.f = n.f[:, ::-1]; return n


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


def hip(x0, x1, z0, z1, y0, y1, c, over=0.35):
    """A hipped roof over the rectangle, with its eaves standing out `over` m; ridge along the long side."""
    w, d = (x1 - x0) / 2 + over, (z1 - z0) / 2 + over
    if w >= d:
        r = frustum(w, d, y0, w - d, 0.0, y1, c)
    else:
        r = frustum(w, d, y0, 0.0, d - w, y1, c)
    return r.T((x0 + x1) / 2, 0, (z0 + z1) / 2)


def rect_ring(x, z):
    return [(-x, -z), (x, -z), (x, z), (-x, z)]


def rect(x0, x1, z0, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


# ── Openings ─────────────────────────────────────────────────────────────────
def window(x, y0, y1, zf, side, w=1.3, hood=True, axis="z", shutters=True):
    """A window with white louvred shutters closed, a granite frame and sill, and a hood moulding."""
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

    out.append(BX(x - w / 2 - 0.15, x + w / 2 + 0.15, y0 - 0.1, y1 + 0.1, 0, 0.05, granite(1.05)))   # frame
    if shutters:
        out.append(Q(x, (y0 + y1) / 2, w, h, SHUTTER, 0.07))
        for k in range(1, 12):                                                        # louvres
            out.append(Q(x, y0 + h * k / 12, w * 0.98, 0.025, (0.74, 0.74, 0.72), 0.08))
        out.append(Q(x, (y0 + y1) / 2, 0.03, h, (0.6, 0.6, 0.58), 0.085))
    else:
        out.append(Q(x, (y0 + y1) / 2, w, h, GLASS, 0.07))
    out.append(BX(x - w / 2 - 0.25, x + w / 2 + 0.25, y0 - 0.2, y0 - 0.06, 0, 0.16, granite(1.07)))   # sill
    if hood:
        out.append(BX(x - w / 2 - 0.4, x + w / 2 + 0.4, y1 + 0.22, y1 + 0.38, 0, 0.24, granite(1.08)))
        out.append(BX(x - w / 2 - 0.25, x + w / 2 + 0.25, y1 + 0.1, y1 + 0.22, 0, 0.1, granite(1.03)))
    return out


def door(x, y0, w, h, zf, side, arched=False, c=DOOR):
    """A glazed door (white frame) in a granite frame; arched=True gives it a round head."""
    n = "+z" if side > 0 else "-z"
    out = []
    hr = h - (w / 2 if arched else 0)
    out.append(quad(x, y0 + hr / 2, zf, w, hr, n, c, eps=0.06))
    out.append(quad(x - w * 0.22, y0 + hr * 0.55, zf, w * 0.3, hr * 0.7, n, GLASS, eps=0.07))
    out.append(quad(x + w * 0.22, y0 + hr * 0.55, zf, w * 0.3, hr * 0.7, n, GLASS, eps=0.07))
    if arched:
        r = w / 2; seg = 12
        V = [(x, y0 + hr, zf + 0.06 * side)] + [(x + r * math.cos(a), y0 + hr + r * math.sin(a), zf + 0.06 * side)
                                               for a in np.linspace(0, math.pi, seg + 1)]
        m = Mesh(V, [(0, i + 1, i + 2) for i in range(seg)], (0.96, 0.96, 0.94))
        if side < 0:
            m.f = m.f[:, ::-1]
        out.append(m)
        out.append(tube([(x + (r + 0.12) * math.cos(a), y0 + hr + (r + 0.12) * math.sin(a), zf + 0.05 * side)
                         for a in np.linspace(0, math.pi, 10)], 0.12, granite(1.07), 6))
    for s in (-1, 1):
        out.append(B(x + s * (w / 2 + 0.12) - 0.12, x + s * (w / 2 + 0.12) + 0.12, y0, y0 + hr, zf, zf + 0.09 * side, granite(1.07)))
    if not arched:
        out.append(B(x - w / 2 - 0.24, x + w / 2 + 0.24, y0 + h, y0 + h + 0.24, zf, zf + 0.09 * side, granite(1.07)))
    return out


def column(x, z, y0, y1, r=0.28, c=None):
    """A Tuscan column: square plinth, torus base, tapering shaft, echinus and abacus."""
    c = c or granite(1.06)
    out = [B(x - r - 0.08, x + r + 0.08, y0, y0 + 0.18, z - r - 0.08, z + r + 0.08, c),
           cyl(r + 0.06, r + 0.02, 0.12, c, 14).T(x, y0 + 0.18, z),
           cyl(r, r * 0.86, y1 - y0 - 0.62, c, 14, caps=False).T(x, y0 + 0.3, z),
           cyl(r * 0.9, r + 0.08, 0.16, c, 14).T(x, y1 - 0.32, z),
           B(x - r - 0.1, x + r + 0.1, y1 - 0.16, y1, z - r - 0.1, z + r + 0.1, c)]
    return out


def joints(x0, x1, y0, y1, zf, side, step=0.46, axis="z"):
    """Horizontal joints of the ashlar: thin darker bands on a façade."""
    out = []
    y = y0 + step
    while y < y1 - 0.1:
        if axis == "z":
            out.append(B(x0, x1, y, y + 0.03, zf, zf + 0.015 * side, granite(0.85)))
        else:
            out.append(B(zf, zf + 0.015 * side, y, y + 0.03, x0, x1, granite(0.85)))
        y += step
    return out


def chimney(x, z, y0, h=1.4):
    return [B(x - 0.4, x + 0.4, y0, y0 + h, z - 0.3, z + 0.3, granite(0.98)),
            B(x - 0.5, x + 0.5, y0 + h, y0 + h + 0.15, z - 0.4, z + 0.4, granite(1.05))]


# ── The house ────────────────────────────────────────────────────────────────
def main_block():
    parts = []
    # side bodies (x from ±XT to ±XB): two floors of granite ashlar
    for s in (-1, 1):
        xa, xb = s * XT, s * XB
        parts.append(B(xa, xb, FOUND, EAVE - 0.3, ZR, ZF, granite()))
        parts.append(B(xa, xb + s * 0.08, FOUND, FLOOR + 0.3, ZR - 0.08, ZF + 0.08, granite(0.94)))        # plinth
        parts.append(B(xa, xb + s * 0.3, GF_CORN - 0.35, GF_CORN, ZR - 0.3, ZF + 0.3, granite(1.07)))     # cornice
        parts.append(B(xa, xb + s * 0.12, GF_CORN - 0.6, GF_CORN - 0.35, ZR - 0.12, ZF + 0.12, granite(1.02)))
        parts.append(B(xa, xb + s * 0.4, EAVE - 0.3, EAVE, ZR - 0.4, ZF + 0.4, granite(1.08)))             # eave
        for zf, sd in ((ZF, 1), (ZR, -1)):
            parts += joints(min(xa, xb), max(xa, xb), FLOOR + 0.3, GF_CORN - 0.6, zf, sd)
        parts += joints(ZR, ZF, FLOOR + 0.3, GF_CORN - 0.6, xb, s, axis="x")
        # corners: rusticated quoins
        for zc in (ZR, ZF):
            for k in range(9):
                y = FLOOR + 0.3 + k * 0.52
                ww = 0.75 if k % 2 == 0 else 0.5
                parts.append(B(xb - s * ww, xb + s * 0.05, y, y + 0.48, zc, zc + (0.05 if zc > 0 else -0.05), granite(1.04)))
        # windows: three bays each side, both façades; the upper floor has small square windows
        for x in (5.6, 9.4, 13.0):
            for zf, sd in ((ZF, 1), (ZR, -1)):
                parts += window(s * x, 2.05, 4.0, zf, sd)
                parts += window(s * x, 6.3, 7.0, zf, sd, w=1.05, hood=False)
        # end façades (±x): two windows
        for z in (-0.2, 3.5):
            parts += window(z, 2.05, 4.0, xb, s, axis="x")
            parts += window(z, 6.3, 7.0, xb, s, w=1.05, hood=False, axis="x")
        # hipped roof of the side body, against the tower
        r = hip(min(xa, xb), max(xa, xb), ZR, ZF, EAVE, EAVE + 1.15, SLATE, over=0.4)
        parts.append(r)
        parts += chimney(s * 9.0, ZR + 2.2, EAVE + 0.4, 1.5)
        parts += chimney(s * 9.0, ZF - 2.2, EAVE + 0.4, 1.5)
    # the central tower, full depth, rising above the side roofs
    parts.append(B(-XT, XT, FOUND, T_CORN - 0.45, ZR, ZF, granite()))
    parts.append(B(-XT - 0.08, XT + 0.08, FOUND, FLOOR + 0.3, ZR - 0.08, ZF + 0.08, granite(0.94)))
    parts.append(B(-XT - 0.12, XT + 0.12, GF_CORN - 0.6, GF_CORN, ZR - 0.12, ZF + 0.12, granite(1.04)))
    parts.append(B(-XT - 0.2, XT + 0.2, T_CORN - 0.45, T_CORN - 0.2, ZR - 0.2, ZF + 0.2, granite(1.05)))
    parts.append(B(-XT - 0.42, XT + 0.42, T_CORN - 0.2, T_CORN, ZR - 0.42, ZF + 0.42, granite(1.09)))
    for zf, sd in ((ZF, 1), (ZR, -1)):
        parts += joints(-XT, XT, EAVE, T_CORN - 0.45, zf, sd)
    for s in (-1, 1):
        parts += joints(ZR, ZF, EAVE + 1.2, T_CORN - 0.45, s * XT, s, axis="x")
        for k in range(4):                                     # quoins of the tower
            y = EAVE + 0.1 + k * 0.6
            for zc, sz in ((ZF, 1), (ZR, -1)):
                parts.append(B(s * (XT - (0.7 if k % 2 == 0 else 0.45)), s * (XT + 0.04), y, y + 0.55, zc, zc + 0.04 * sz, granite(1.04)))
    # upper windows of the tower: front, the door to the balcony between two small columns
    for x in (-2.45, 2.45):
        parts += window(x, 6.75, 8.75, ZF, 1, w=1.15)
        parts += window(x, 6.75, 8.75, ZR, -1, w=1.15)
    parts += door(0.0, 6.4, 1.3, 2.4, ZF, 1)
    for x in (-1.15, 1.15):
        parts += column(x, ZF + 0.25, 6.4, 8.9, r=0.16)
    parts += window(0.0, 6.75, 8.75, ZR, -1, w=1.15)
    # the lead roof of the tower (a low hip with a flat top), the spire with its ball and vane
    parts.append(frustum(XT + 0.5, (ZF - ZR) / 2 + 0.5, T_CORN, XT - 1.1, (ZF - ZR) / 2 - 1.1, T_CORN + 0.85, LEAD).T(0, 0, (ZF + ZR) / 2))
    parts.append(frustum(XT - 1.1, (ZF - ZR) / 2 - 1.1, T_CORN + 0.85, 1.2, 1.2, T_CORN + 1.2, LEAD).T(0, 0, (ZF + ZR) / 2))
    zc = (ZF + ZR) / 2
    parts.append(B(-0.55, 0.55, T_CORN + 1.0, T_CORN + 1.9, zc - 0.55, zc + 0.55, LEAD))
    parts.append(lathe([(0.5, 0), (0.42, 0.15), (0.3, 0.5), (0.16, 1.5), (0.07, 2.6), (0.0, 2.75)], LEAD, 10).T(0, T_CORN + 1.9, zc))
    parts.append(sphere(0.14, (0.55, 0.48, 0.3), 8, 5).T(0, T_CORN + 4.75, zc))
    parts.append(B(-0.02, 0.02, T_CORN + 4.85, T_CORN + 5.7, zc - 0.02, zc + 0.02, IRON))
    parts.append(B(-0.3, 0.3, T_CORN + 5.25, T_CORN + 5.28, zc - 0.015, zc + 0.015, IRON))
    parts.append(B(-0.015, 0.015, T_CORN + 5.25, T_CORN + 5.28, zc - 0.3, zc + 0.3, IRON))
    parts.append(Mesh([(0.03, T_CORN + 5.4, zc), (0.45, T_CORN + 5.47, zc), (0.03, T_CORN + 5.58, zc)], [(0, 1, 2), (0, 2, 1)], IRON))
    for x, z in ((-2.6, ZR + 0.8), (2.6, ZR + 0.8), (-2.6, ZF - 0.8), (2.6, ZF - 0.8)):
        parts += chimney(x, z, T_CORN + 0.3, 1.0)

    # ── The portico (east): four Tuscan columns, entablature, balcony ──
    zp = ZF + 1.8
    parts.append(B(-3.95, 3.95, FOUND, FLOOR, ZF, zp, granite(1.03)))                              # its floor
    for x in (-3.3, -1.35, 1.35, 3.3):
        parts += column(x, zp - 0.4, FLOOR, 5.3)
    parts.append(B(-3.95, 3.95, 5.3, 5.85, ZF, zp, granite(1.05)))                                  # architrave
    parts.append(B(-4.05, 4.05, 5.85, 6.15, ZF, zp + 0.1, granite(1.02)))                           # frieze
    parts.append(B(-4.2, 4.2, 6.15, 6.4, ZF, zp + 0.25, granite(1.09)))                             # cornice / floor
    for x in (-3.0, 3.0):                                                                           # pilasters on the wall
        parts.append(B(x - 0.3, x + 0.3, FLOOR, 5.3, ZF, ZF + 0.12, granite(1.04)))
    parts += door(0.0, FLOOR, 1.6, 3.0, ZF, 1)
    for x in (-2.05, 2.05):
        parts += window(x, 2.0, 3.7, ZF, 1, w=0.95, hood=False, shutters=False)
        parts.append(B(x - 0.55, x + 0.55, 4.2, 4.7, ZF, ZF + 0.06, granite(1.1)))                  # carved panels
    # balcony: posts, rails and the balusters of painted iron
    rail_pts = [(-4.05, ZF + 0.05), (-4.05, zp + 0.1), (4.05, zp + 0.1), (4.05, ZF + 0.05)]
    for (xa, za), (xb, zb) in zip(rail_pts, rail_pts[1:]):
        L = math.hypot(xb - xa, zb - za)
        n = max(2, int(L / 0.14))
        for k in range(n + 1):
            x, z = xa + (xb - xa) * k / n, za + (zb - za) * k / n
            parts.append(B(x - 0.015, x + 0.015, 6.4, 7.35, z - 0.015, z + 0.015, RAIL))
        for y in (6.5, 7.0, 7.35):
            parts.append(tube([(xa, y, za), (xb, y, zb)], 0.035 if y > 7.3 else 0.02, RAIL, 5))
    # steps from the portico down to the walk (two flights with a landing)
    for k in range(3):
        parts.append(B(-4.2, 4.2, FOUND, FLOOR - 0.17 * (k + 1) + 0.17, zp, zp + 0.35 * (k + 1), granite(1.0 + 0.02 * k)))
    for k in range(3):
        top = 0.5 - 0.17 * k
        parts.append(B(-5.0, 5.0, FOUND, top, zp + 1.05, zp + 2.1 + 0.4 * k, granite(1.0 + 0.02 * k)))
    # granite curbs beside the steps
    for s in (-1, 1):
        parts.append(B(s * 4.95, s * 5.4, FOUND, 1.15, ZF, zp + 1.0, granite(1.04)))
    return merge(parts)


def rear_wing():
    """The rear wing of 1781–84 (west): one tall floor under a hipped roof, and its porch in antis."""
    parts = []
    z0, z1 = ZW_END, ZR
    parts.append(B(-XT, XT, FOUND, 5.3, z0, z1, granite()))
    parts.append(B(-XT - 0.08, XT + 0.08, FOUND, FLOOR + 0.3, z0, z1, granite(0.94)))
    parts.append(B(-XT - 0.3, XT + 0.3, 5.3, 5.6, z0 - 0.3, z1, granite(1.08)))
    parts.append(hip(-XT, XT, z0, z1, 5.6, 7.0, SLATE, over=0.35))
    for s in (-1, 1):
        parts += joints(z0, z1, FLOOR + 0.3, 5.2, s * XT, s, axis="x")
        for z in (-5.4, -9.0, -12.6, -16.0):
            parts += window(z, 1.9, 4.1, s * XT, s, w=1.2, axis="x")
    # the porch: two broad rusticated piers, two Tuscan columns, entablature and attic
    zp = ZW_PORCH
    for s in (-1, 1):
        parts.append(B(s * 2.35, s * XT, FOUND, 4.4, zp, z0, granite(1.02)))
        parts += joints(2.35, XT, FLOOR, 4.3, zp, -1, step=0.4) if s > 0 else joints(-XT, -2.35, FLOOR, 4.3, zp, -1, step=0.4)
        parts += column(s * 1.05, zp + 0.45, FLOOR, 4.4, r=0.26)
        parts.append(B(s * 1.45, s * 2.3, 1.4, 3.3, z0, z0 - 0.07, SHUTTER))                       # white panels
    parts.append(B(-XT, XT, FOUND, FLOOR, zp, z0, granite(1.03)))                                   # floor of the porch
    parts.append(B(-XT - 0.05, XT + 0.05, 4.4, 5.2, zp - 0.05, z0, granite(1.05)))                  # entablature
    parts.append(B(-XT - 0.25, XT + 0.25, 5.2, 5.45, zp - 0.25, z0, granite(1.09)))
    parts.append(B(-XT + 0.1, XT - 0.1, 5.45, 6.55, zp + 0.05, z0, granite()))                      # attic
    parts.append(B(-XT + 0.1, XT - 0.1, 6.55, 6.75, zp - 0.1, z0 + 0.15, granite(1.08)))
    parts.append(B(-2.4, 2.4, 5.7, 6.3, zp + 0.05, zp - 0.0, granite(1.1)))                        # its panel
    parts.append(B(-XT + 0.1, XT - 0.1, 6.75, 6.95, zp + 0.1, z0, LEAD))
    parts += door(0.0, FLOOR, 1.5, 3.3, z0, -1, arched=True)
    for k in range(2):
        parts.append(B(-2.3, 2.3, FOUND, FLOOR - 0.33 * (k + 1) + 0.33, zp - 0.4 * (k + 1), zp, granite(1.01 + 0.02 * k)))
    return merge(parts)


def end_wing(s):
    """A long one-floor end wing (s = +1 north, −1 south): ochre render, granite plinth and pilasters."""
    parts = []
    x0, x1 = s * XE0, s * XE1
    xa, xb = sorted((x0, x1))
    parts.append(B(xa, xb, FOUND, E_EAVE - 0.3, ZR, ZE_F, RENDER))
    parts.append(B(xa - 0.06, xb + 0.06, FOUND, FLOOR + 0.25, ZR - 0.06, ZE_F + 0.06, granite(0.95)))
    parts.append(B(xa - 0.35, xb + 0.35, E_EAVE - 0.3, E_EAVE, ZR - 0.35, ZE_F + 0.35, granite(1.07)))
    parts.append(hip(xa, xb, ZR, ZE_F, E_EAVE, E_EAVE + 1.9, SLATE_L, over=0.45))
    bays = np.linspace(xa + 1.7, xb - 1.7, 7)
    for k, x in enumerate(bays):
        for zf, sd in ((ZE_F, 1), (ZR, -1)):
            parts += window(x, 1.6, 3.6, zf, sd, w=1.15)
            if k < len(bays) - 1:                                                                  # pilasters
                xm = (x + bays[k + 1]) / 2
                parts.append(B(xm - 0.3, xm + 0.3, FLOOR + 0.25, E_EAVE - 0.3, zf, zf + 0.06 * sd, granite(1.02)))
    for xc in (xa, xb):                                                                            # corner pilasters
        for zf, sd in ((ZE_F, 1), (ZR, -1)):
            parts.append(B(xc - 0.45, xc + 0.45, FLOOR + 0.25, E_EAVE - 0.3, zf - 0.02 * sd, zf + 0.06 * sd, granite(1.02)))
    xend = x1
    for z in (0.0, 4.2):
        parts += window(z, 1.6, 3.6, xend, s, w=1.15, axis="x")
    parts += chimney(s * 26.0, 2.2, E_EAVE + 0.8, 1.4)
    parts += chimney(s * 35.0, 2.2, E_EAVE + 0.8, 1.4)
    return merge(parts)


def link(s):
    """The low bridge over the passage between the main block and an end wing (OSM min_level 1)."""
    parts = []
    xa, xb = sorted((s * XB, s * XE0))
    z0, z1 = ZR, 1.0
    parts.append(B(xa, xb, 3.7, 4.9, z0, z1, granite()))
    parts.append(B(xa, xb, 4.9, 5.1, z0 - 0.2, z1 + 0.2, granite(1.08)))
    r = frustum((xb - xa) / 2, (z1 - z0) / 2 + 0.25, 5.1, (xb - xa) / 2, 0.0, 5.7, SLATE).T((xa + xb) / 2, 0, (z0 + z1) / 2)
    parts.append(r)
    # the arch under it, seen from the walk
    for zf in (z0, z1):
        sd = 1 if zf > 0 else -1
        parts.append(tube([((xa + xb) / 2 + 1.75 * math.cos(a), 3.0 + 0.7 * math.sin(a), zf + 0.04 * sd)
                           for a in np.linspace(0, math.pi, 10)], 0.16, granite(1.07), 6))
        parts.append(B(xa, xb, 3.0, 3.7, zf, zf + 0.02 * sd, granite(0.98)))
    # steps up from the walk (0) to the rear garden (FLOOR) under it
    for k in range(5):
        parts.append(B(xa, xb, FOUND, 0.2 * (k + 1), z1 - 0.6 * (k + 1) + 0.6, z1 - 0.6 * (k + 1), granite(1.0 + 0.01 * k)))
    return merge(parts)


def house():
    return merge([main_block(), rear_wing(), end_wing(1), end_wing(-1), link(1), link(-1)])


# ── The gardens ──────────────────────────────────────────────────────────────
SAND = (0.84, 0.79, 0.67)
PAVE = (0.80, 0.77, 0.70)
BOX = (0.19, 0.31, 0.14)          # boxwood hedges
BOX_D = (0.15, 0.26, 0.12)
GRASS = (0.40, 0.50, 0.25)
LAVENDER = (0.50, 0.50, 0.58)
FLOWER = (0.70, 0.36, 0.34)
FLOWER_Y = (0.86, 0.70, 0.32)
WATER = (0.22, 0.34, 0.38)
GFOUND = -5.0


def ngon(cx, cz, r, n=20, rz=None):
    rz = r if rz is None else rz
    return [(cx + r * math.cos(a), cz + rz * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def level(x, z):
    """Ground level of the gardens (MDT05): the walk 0, the rear garden FLOOR, the front garden going down."""
    if z < -90.0:
        return FLOOR + min(1.6, (-90.0 - z) * 0.05)
    if z <= 5.0:
        return FLOOR
    if z <= 16.5:
        return 0.0
    if z <= 33.5:
        return -0.5 * (z - 16.5) / 17.0
    if z <= 36.0:
        return -0.5 - 0.5 * (z - 33.5) / 2.5
    if z <= 58.0:
        return -1.0
    return -1.0 - 1.2 * min(1.0, (z - 58.0) / 16.0)


def inside(px, pz, ring):
    """Point in polygon (even–odd)."""
    P = np.asarray(ring, float)
    x, z = P[:, 0], P[:, 1]
    x2, z2 = np.roll(x, -1), np.roll(z, -1)
    px = np.asarray(px, float)[..., None]; pz = np.asarray(pz, float)[..., None]
    c = ((z > pz) != (z2 > pz)) & (px < (x2 - x) * (pz - z) / (z2 - z + 1e-12) + x)
    return (np.sum(c, axis=-1) % 2) == 1


def hedge_ring(ring, y, h=0.6, w=0.45, c=BOX):
    return extrude(ring, y - 0.3, y + h, c, holes=[offset(ring, -w)])


def hedge_line(pts, h=0.8, w=0.5, c=BOX, closed=True, lv=None, piece=3.0):
    """A clipped hedge along a polyline whose foot follows the ground (lv(x, z)), cut into short pieces."""
    lv = lv or level
    P = list(pts) + ([pts[0]] if closed else [])
    out = []
    for (xa, za), (xb, zb) in zip(P, P[1:]):
        L = math.hypot(xb - xa, zb - za)
        if L < 0.05:
            continue
        n = max(1, int(math.ceil(L / piece)))
        ux, uz = (xb - xa) / L, (zb - za) / L
        nx, nz = -uz * w / 2, ux * w / 2
        for k in range(n):
            t0, t1 = k / n, (k + 1) / n
            ax, az = xa + (xb - xa) * t0 - ux * w / 2, za + (zb - za) * t0 - uz * w / 2
            bx, bz = xa + (xb - xa) * t1 + ux * w / 2, za + (zb - za) * t1 + uz * w / 2
            y = level((ax + bx) / 2, (az + bz) / 2) if lv is level else lv((ax + bx) / 2, (az + bz) / 2)
            ring = [(ax - nx, az - nz), (bx - nx, bz - nz), (bx + nx, bz + nz), (ax + nx, az + nz)]
            m = extrude(ring, y - 0.35, y + h, c)
            out.append(m)
    return out


def shrub(x, z, y, r, h, c, seed):
    m = lathe([(0.0, 0.0), (r * 0.8, 0.05), (r, h * 0.4), (r * 0.75, h * 0.85), (0.0, h)], c, 7)
    k = 0.9 + 0.2 * np.random.default_rng(seed).random()
    m.c = np.clip(m.c * k, 0, 1)
    return m.T(x, y, z)


def parterre(ring, y, fill, seed, hedge_h=0.6):
    """A parterre on flat ground: its box hedge and what grows inside (seen on the orthophoto)."""
    r = np.random.default_rng(seed)
    out = [hedge_ring(ring, y, h=hedge_h)]
    inner = offset(ring, -0.45)
    xs = [p[0] for p in inner]; zs = [p[1] for p in inner]
    x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
    if fill == "lavender":
        out.append(extrude(inner, y - 0.05, y + 0.08, (0.42, 0.44, 0.33)))
        k = 0
        z = z0 + 0.9
        while z < z1 - 0.6:                                     # rows of lavender
            pts = [(x, z) for x in np.arange(x0 + 0.8, x1 - 0.5, 0.6)]
            run = [p for p in pts if inside(p[0], p[1], inner)]
            if len(run) > 1:
                out.append(B(run[0][0] - 0.25, run[-1][0] + 0.25, y, y + 0.45, z - 0.3, z + 0.3,
                             LAVENDER if k % 2 == 0 else (0.45, 0.47, 0.42)))
            z += 1.0; k += 1
    else:
        out.append(extrude(inner, y - 0.05, y + 0.07, GRASS))
        if fill in ("shrubs", "trees"):
            step = 2.6 if fill == "shrubs" else 3.4
            for x in np.arange(x0 + step / 2, x1, step):
                for z in np.arange(z0 + step / 2, z1, step):
                    if inside(x, z, offset(inner, -0.6)):
                        if fill == "shrubs":
                            out.append(shrub(x, z, y, 0.55, 1.1, (0.24, 0.36, 0.17), seed + int(x * 7 + z)))
                        else:
                            out.append(shrub(x, z, y, 0.9, 2.2, (0.22, 0.34, 0.16), seed + int(x * 7 + z)))
        elif fill == "flowers":
            for k in range(6):
                cx, cz = r.uniform(x0 + 1.5, x1 - 1.5), r.uniform(z0 + 1.5, z1 - 1.5)
                if inside(cx, cz, inner):
                    out.append(extrude(ngon(cx, cz, 0.9, 8), y, y + 0.3, FLOWER if k % 2 else FLOWER_Y))
    return out


def fountain(cx, cz, y, r, finial=True):
    """A round granite basin with a thick rounded rim, its water, and the stone pine cone in the middle."""
    out = [lathe([(r - 0.45, 0.0), (r - 0.45, 0.5), (r - 0.35, 0.68), (r - 0.15, 0.72), (r, 0.6), (r + 0.05, 0.35),
                  (r, 0.0)], granite(1.0), 32).T(cx, y, cz)]
    out.append(cyl(r - 0.45, r - 0.45, 0.42, WATER, 32).T(cx, y, cz))
    if finial:
        out.append(B(cx - 0.3, cx + 0.3, y, y + 0.75, cz - 0.3, cz + 0.3, granite(1.02)))
        out.append(lathe([(0.0, 0.0), (0.3, 0.1), (0.36, 0.4), (0.3, 0.7), (0.16, 0.95), (0.0, 1.05)], granite(0.98), 10).T(cx, y + 0.75, cz))
    return out


def stairs_z(x0, x1, z0, z1, y_hi, y_lo):
    """Steps going down from y_hi at z0 to y_lo at z1."""
    out = []
    n = max(2, int(round(abs(y_hi - y_lo) / 0.16)))
    for k in range(n):
        za, zb = z0 + (z1 - z0) * k / n, z0 + (z1 - z0) * (k + 1) / n
        out.append(B(x0, x1, GFOUND, y_hi - k * (y_hi - y_lo) / n, za, zb, granite(1.03)))
    return out


def railing(pts, y, h=1.1, c=IRON):
    out = []
    for (xa, za), (xb, zb) in zip(pts, pts[1:]):
        L = math.hypot(xb - xa, zb - za)
        n = max(1, int(L / 2.5))
        for k in range(n + 1):
            x, z = xa + (xb - xa) * k / n, za + (zb - za) * k / n
            out.append(B(x - 0.04, x + 0.04, y, y + h, z - 0.04, z + 0.04, c))
        for yy in (y + 0.12, y + h - 0.08):
            out.append(tube([(xa, yy, za), (xb, yy, zb)], 0.025, c, 4))
    return out


def wall_line(pts, top_above, c_k=0.98, w=0.5, cap=True):
    """A granite garden wall along a polyline, its top `top_above` m over the ground inside, down to GFOUND."""
    out = []
    for (xa, za), (xb, zb) in zip(pts, pts[1:]):
        L = math.hypot(xb - xa, zb - za)
        n = max(1, int(math.ceil(L / 6.0)))
        ux, uz = (xb - xa) / L, (zb - za) / L
        nx, nz = -uz * w / 2, ux * w / 2
        for k in range(n):
            ax, az = xa + (xb - xa) * k / n, za + (zb - za) * k / n
            bx, bz = xa + (xb - xa) * (k + 1) / n, za + (zb - za) * (k + 1) / n
            y = max(level(ax, az), level(bx, bz)) + top_above
            ring = [(ax - nx, az - nz), (bx - nx, bz - nz), (bx + nx, bz + nz), (ax + nx, az + nz)]
            out.append(extrude(ring, GFOUND, y, granite(c_k)))
            if cap:
                ring2 = [(ax - nx * 1.4 - ux * 0.05, az - nz * 1.4 - uz * 0.05), (bx - nx * 1.4 + ux * 0.05, bz - nz * 1.4 + uz * 0.05),
                         (bx + nx * 1.4 + ux * 0.05, bz + nz * 1.4 + uz * 0.05), (ax + nx * 1.4 - ux * 0.05, az + nz * 1.4 - uz * 0.05)]
                out.append(extrude(ring2, y, y + 0.15, granite(1.07)))
    return out


def heightfield(x0, x1, z0, z1, step, colour_fn):
    """The ground of a sloping garden, its colour painted per vertex (gravel, lawn, flower beds)."""
    nx, nz = int(math.ceil((x1 - x0) / step)), int(math.ceil((z1 - z0) / step))
    xs, zs = np.linspace(x0, x1, nx + 1), np.linspace(z0, z1, nz + 1)
    X, Z = np.meshgrid(xs, zs)
    Y = np.vectorize(level)(X, Z)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    W = nx + 1
    F = []
    for j in range(nz):
        for i in range(nx):
            a = j * W + i
            F += [(a, a + W, a + 1), (a + 1, a + W, a + W + 1)]
    m = Mesh(V, F, SAND)
    m.c = np.asarray(colour_fn(V[:, 0], V[:, 2]), float)
    return m


# Front garden (east): OpenStreetMap ways 996507112–19 (hedges, 0.8 m), 231916287 (the fountain),
# 231916258 (its lawn ring), 231916268 and the radiating paths, 231916286 (wall), 231916327 (hedge).
FRONT_PARTERRES = [
    [(-3.0, 18.2), (-26.8, 18.1), (-5.9, 36.0), (-4.5, 35.0), (-2.8, 34.5)],
    [(-10.0, 40.9), (-42.1, 41.1), (-41.5, 18.0), (-32.4, 17.7), (-9.1, 37.6), (-9.9, 39.2)],
    [(3.4, 17.9), (4.3, 33.8), (6.0, 35.0), (25.1, 17.8)],
    [(41.7, 18.0), (29.8, 17.9), (7.9, 37.5), (8.9, 39.0), (9.5, 40.6), (42.0, 40.2)],
    [(-8.0, 46.8), (-8.8, 44.4), (-41.5, 45.0), (-39.8, 51.7), (-37.4, 57.9), (-34.3, 62.7), (-30.2, 66.6)],
    [(-2.9, 68.3), (-3.1, 51.8), (-4.5, 51.1), (-5.8, 49.9), (-27.0, 68.9)],
    [(28.9, 68.4), (1.3, 68.2), (1.7, 52.8), (4.3, 51.9), (6.6, 50.4)],
    [(42.2, 44.2), (9.9, 44.8), (9.8, 46.2), (9.1, 48.3), (31.6, 66.4), (36.0, 61.9), (39.6, 56.4), (41.6, 51.1)],
]
FOUNTAIN_F = (0.25, 43.3)
FRONT_EDGE = [(-31.0, 73.4), (-33.8, 69.4), (-40.3, 62.6), (-43.6, 57.9), (-45.5, 49.6), (-46.4, 42.4), (-45.6, 16.0),
              (-45.6, 5.6), (46.4, 5.6), (47.0, 10.9), (47.8, 35.1), (48.2, 44.9), (46.0, 55.1), (40.6, 64.3), (30.1, 73.2)]
WALK_HEDGES = [(-41.5, -29.6), (-29.0, -17.8), (-13.3, -3.2), (3.2, 13.3), (17.8, 29.0), (29.6, 41.5)]

# Rear garden (west): OSM hedges 891230428–39 and the fountain 891230440; planting from the orthophoto
REAR_PARTERRES = [
    ([(-16.5, -23.0), (-16.3, -6.2), (-39.7, -6.0), (-39.9, -22.9)], "shrubs"),
    ([(-7.0, -23.0), (-6.8, -6.1), (-14.7, -6.1), (-14.8, -23.0)], "lavender"),
    ([(7.1, -23.1), (7.5, -6.2), (15.3, -6.2), (15.2, -23.1)], "lavender"),
    ([(17.1, -23.0), (17.4, -5.8), (39.9, -6.5), (40.0, -23.1)], "shrubs"),
    ([(-16.5, -24.6), (-39.8, -24.7), (-39.8, -53.2), (-16.5, -53.4)], "grass"),
    ([(-14.8, -24.6), (-5.9, -24.7), (-5.3, -27.0), (-4.5, -28.4), (-3.5, -29.2), (-2.1, -29.5), (-2.1, -48.1),
      (-4.2, -48.7), (-5.8, -49.7), (-7.1, -51.5), (-7.8, -53.4), (-14.9, -53.4)], "lavender"),
    ([(15.1, -24.5), (6.2, -24.4), (5.7, -26.8), (4.9, -28.1), (3.8, -28.9), (2.5, -29.3), (2.6, -47.9), (4.5, -48.3),
      (6.3, -49.5), (7.8, -51.5), (8.1, -53.7), (14.9, -53.8)], "lavender"),
    ([(17.0, -24.7), (40.1, -25.1), (39.9, -54.3), (16.9, -54.0)], "flowers"),
    ([(-16.5, -58.6), (-39.4, -58.6), (-39.3, -86.9), (-16.5, -86.9)], "grass"),
    ([(-15.0, -87.0), (-14.9, -58.5), (-7.9, -58.5), (-7.0, -60.8), (-5.6, -62.8), (-3.8, -63.7), (-1.8, -64.2),
      (-2.0, -87.0)], "shrubs"),
    ([(15.0, -87.0), (15.0, -58.7), (8.1, -58.7), (7.7, -60.8), (6.0, -62.8), (4.1, -63.9), (2.3, -64.2), (2.1, -87.0)], "shrubs"),
    ([(17.0, -59.1), (39.9, -59.0), (40.0, -87.3), (17.2, -87.5)], "trees"),
]
FOUNTAIN_R = (0.0, -56.4)
BRODERIE = [  # OSM ways 996177394/95 (Buxus sempervirens, 0.6 m)
    [(-1.7, -116.4), (-1.4, -95.4), (-12.4, -95.0), (-11.5, -119.3), (-3.6, -119.2), (-3.0, -117.7)],
    [(12.2, -119.4), (12.2, -95.6), (1.6, -95.1), (1.5, -116.4), (3.3, -117.6), (4.3, -119.2)],
]


def front_colour(x, z):
    c = np.tile(np.array(SAND), (len(x), 1))
    for ring in FRONT_PARTERRES:
        m = inside(x, z, offset(ring, -0.3))
        c[m] = GRASS
    d = np.hypot(x - FOUNTAIN_F[0], z - FOUNTAIN_F[1])
    ring_m = (d > 3.2) & (d < 5.0)
    k = (np.floor(np.degrees(np.arctan2(z - FOUNTAIN_F[1], x - FOUNTAIN_F[0])) / 30) % 2) == 0
    c[ring_m & k] = FLOWER_Y
    c[ring_m & ~k] = FLOWER
    out = ~inside(x, z, FRONT_EDGE)
    c[out & (z > 5.0)] = GRASS
    return c


def broderie_colour(x, z):
    c = np.tile(np.array(SAND), (len(x), 1))
    for ring in BRODERIE:
        c[inside(x, z, offset(ring, -0.3))] = (0.66, 0.52, 0.40)       # red gravel under the box scrolls
    return c


def broderie(ring, seed):
    """The box embroidery of a broderie parterre: outer hedge, two long ovals, and scrolls around them."""
    out = hedge_line(ring, h=0.6, w=0.45, c=BOX)
    xs = [p[0] for p in ring]; zs = [p[1] for p in ring]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    w, d = (max(xs) - min(xs)) / 2 - 1.2, (max(zs) - min(zs)) / 2 - 1.2
    for sx in (-1, 1):                                    # two long ovals with a round eye
        ox = cx + sx * w * 0.45
        out += hedge_line(ngon(ox, cz, w * 0.42, 16, d * 0.85), h=0.4, w=0.3, c=BOX_D, piece=2.0)
        out += hedge_line(ngon(ox, cz, w * 0.18, 10, w * 0.18), h=0.4, w=0.3, c=BOX_D, piece=2.0)
        for sz in (-1, 1):                               # scrolls
            pts = [(ox + w * 0.25 * math.cos(a), cz + sz * d * (0.45 + 0.3 * math.sin(a))) for a in np.linspace(0.3, 2.8, 7)]
            out += hedge_line(pts, h=0.35, w=0.25, c=BOX, closed=False, piece=2.0)
    return out


# ── Trees (positions, heights and crowns from OpenStreetMap) ────────────────
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


def giant_sequoia(x, z, y, h, crown, seed, girth=8.5):
    """Giant sequoia: a huge, fluted, cinnamon-red trunk flaring at the foot, bare for its lower part, then a
    dense, broad cone of short up-curving boughs with clumps of foliage, rounded at the top in old trees."""
    rr = np.random.default_rng(seed)
    BK = bark((0.55, 0.30, 0.18), seed)
    FOL = (0.20, 0.31, 0.15)
    r0 = girth / (2 * math.pi)
    out = [lathe([(r0 * 1.7, 0.0), (r0 * 1.25, 1.0), (r0 * 1.05, 2.5), (r0, 3.5)], BK, 12).T(x, y - 0.3, z)]
    out += limb((x, y + 3.3, z), (x, y + h * 0.96, z), r0, 0.2, BK, 10, n=5)
    t0 = 0.2
    for k in range(95):
        t = t0 + (1 - t0) * rr.random() ** 0.9
        R = crown / 2 * ((1 - t) / (1 - t0)) ** 0.75 + 0.8
        yb = y + h * t
        a = rr.uniform(0, 2 * math.pi)
        d = R * (0.45 + 0.5 * rr.random())
        tip = (x + d * math.cos(a), yb + 0.8 + 0.6 * rr.random(), z + d * math.sin(a))
        if k % 3 == 0 and t < 0.8:
            out += limb((x, yb, z), tip, 0.28, 0.1, BK, 5, bend=-0.4, n=2)
        out.append(blob(tip[0], tip[1], tip[2], max(1.0, R * (0.38 + 0.22 * rr.random())), 0.8, FOL, seed + k, seg=7, rings=4))
    out.append(blob(x, y + h * 0.95, z, 1.6, 1.2, FOL, seed + 999))
    return out


def fir(x, z, y, h, crown, seed, col=(0.13, 0.24, 0.14), stiff=False):
    """Firs (Abies): a straight grey trunk hidden in a dense, narrow, regular cone of whorled tiers of boughs that
    reach almost to the ground, a pointed leader at the top. Pinsapo (stiff=True): bluer, the boughs short and
    stiff, the cone broader and fuller."""
    rr = np.random.default_rng(seed)
    BK = bark((0.42, 0.38, 0.35), seed)
    out = limb((x, y, z), (x, y + h * 0.97, z), 0.5, 0.08, BK, 8, n=4)
    tiers = 22
    base = 0.04
    for k in range(tiers):
        t = base + (0.93 - base) * k / (tiers - 1)
        R = crown / 2 * (1 - (t - base) / (1 - base)) ** (0.85 if not stiff else 0.7) + 0.3
        yb = y + h * t
        n = 4 if R > 1.5 else 3
        for j in range(n):
            a = j * 2 * math.pi / n + k * 1.3 + rr.normal(0, 0.3)
            d = R * (0.45 + 0.3 * rr.random())
            tip = (x + d * math.cos(a), yb - (0.25 if not stiff else 0.1) * R, z + d * math.sin(a))
            out.append(blob(*tip, max(0.55, R * 0.55), 0.6, col, seed + k * 13 + j, seg=7, rings=4))
        if k % 2 == 0:
            out.append(blob(x, yb + 0.3, z, max(0.5, R * 0.5), 0.8, col, seed + k * 13 + 11, seg=7, rings=4))
    out.append(blob(x, y + h * 0.965, z, 0.45, 2.4, col, seed + 777, seg=6, rings=4))
    return out


def abies_nordmanniana(x, z, y, h, crown, seed):
    return fir(x, z, y, h, crown, seed, (0.12, 0.23, 0.13))


def abies_pinsapo(x, z, y, h, crown, seed):
    return fir(x, z, y, h, crown, seed, (0.20, 0.30, 0.25), stiff=True)


TREES = [  # (x, z, kind, height, crown, girth)  — OSM nodes
    (7.1, 22.1, giant_sequoia, 39, 14, 8.55),            # Secuoya Gigante de la Casita del Príncipe I (7895989409)
    (-37.0, -61.4, giant_sequoia, 41.5, 20, 9.2),         # … II (7895989408)
    (20.3, 21.5, giant_sequoia, 34, 11, 6.0),             # sequoias without height (9201302787, 9201302786): ~34 m
    (-4.3, 65.3, giant_sequoia, 34, 12, 6.0),
    (-38.2, 38.2, abies_nordmanniana, 31, 10.5, 3.45),    # Abeto del Cáucaso I (7895989406)
    (-38.2, 25.8, abies_nordmanniana, 32, 9.5, 2.85),     # … II (7895989407)
    (32.4, 61.1, abies_pinsapo, 29, 13.5, 3.7),           # Pinsapo de la Casita del Príncipe (7895737867)
]


def garden():
    parts = []
    # ── Ground ──
    # the rear garden and the house, on a platform at FLOOR with its retaining walls
    parts.append(extrude(rect(-44.6, 45.6, -90.0, ZR + 0.01), GFOUND, FLOOR, granite(0.95), top_c=SAND))
    # the walk in front of the house (and under the end wings), level 0
    parts.append(extrude(rect(-45.4, 46.4, ZR + 0.01, 16.5), GFOUND, 0.0, granite(0.95), top_c=PAVE))
    # the sloping front garden and the rising broderie terrace: height fields painted per vertex
    parts.append(heightfield(-46.5, 48.3, 16.5, 73.0, 1.2, front_colour))
    parts.append(heightfield(-16.0, 16.0, -122.0, -90.0, 1.6, broderie_colour))
    # skirts of the height fields down to GFOUND
    for x0, x1, z0, z1 in ((-46.5, 48.3, 16.5, 73.0), (-16.0, 16.0, -122.0, -90.0)):
        for zz in (z0, z1):
            xs = np.linspace(x0, x1, 40)
            V, F = [], []
            for i, x in enumerate(xs):
                V += [(x, GFOUND, zz), (x, level(x, zz + (0.01 if zz == z0 else -0.01)), zz)]
            for i in range(len(xs) - 1):
                a = 2 * i
                F += [(a, a + 2, a + 1), (a + 1, a + 2, a + 3)] if zz == z1 else [(a, a + 1, a + 2), (a + 1, a + 3, a + 2)]
            parts.append(Mesh(V, F, granite(0.9)))
        for xx in (x0, x1):
            zs = np.linspace(z0, z1, 30)
            V, F = [], []
            for i, z in enumerate(zs):
                V += [(xx, GFOUND, z), (xx, level(xx, z), z)]
            for i in range(len(zs) - 1):
                a = 2 * i
                F += [(a, a + 1, a + 2), (a + 1, a + 3, a + 2)] if xx == x1 else [(a, a + 2, a + 1), (a + 1, a + 2, a + 3)]
            parts.append(Mesh(V, F, granite(0.9)))

    # ── Front garden ──
    for i, ring in enumerate(FRONT_PARTERRES):
        parts += hedge_line(ring, h=0.8, w=0.5)
        xs = [p[0] for p in ring]; zs = [p[1] for p in ring]
        rr = np.random.default_rng(300 + i)
        for k in range(5):                                   # rose and flower beds inside (orthophoto)
            cx, cz = rr.uniform(min(xs), max(xs)), rr.uniform(min(zs), max(zs))
            if inside(cx, cz, offset(ring, -2.0)):
                parts.append(extrude(ngon(cx, cz, 1.1, 8), level(cx, cz) - 0.2, level(cx, cz) + 0.45,
                                     FLOWER if k % 2 else (0.80, 0.62, 0.62)))
    fx, fz = FOUNTAIN_F
    parts += fountain(fx, fz, -1.0, 3.0)
    parts += hedge_line(ngon(fx, fz, 5.1, 24), h=0.25, w=0.25, c=granite(0.95), piece=2.0)      # stone curb of the ring
    # the walk: low hedges along it, the steps on the axis and the side flights
    for xa, xb in WALK_HEDGES:
        parts.append(B(xa, xb, -0.3, 0.6, 13.4, 14.6, BOX))
    parts += stairs_z(-1.6, 1.6, 33.5, 36.0, -0.5, -1.0)
    # walls and hedges of the enclosure (OSM 231916286, 231916327), railing and gate at the east end
    parts += wall_line([(-46.4, 42.4), (-45.5, 49.6), (-43.6, 57.9), (-40.3, 62.6), (-33.8, 69.4), (-31.0, 73.4)], 1.3)
    parts += wall_line([(-45.8, 5.5), (-46.4, 42.4)], 1.3)
    parts += hedge_line([(47.0, 10.9), (47.8, 35.1), (48.2, 44.9), (46.0, 55.1), (40.6, 64.3), (30.1, 73.2)], h=2.2, w=1.2, closed=False)
    # the gate on the axis between the two lodges (separate buildings, not modelled): piers and iron gate
    for x in (-2.55, 1.75):
        y = level(x, 73.0)
        parts.append(B(x - 0.35, x + 0.35, GFOUND, y + 2.6, 72.4, 73.2, granite(1.03)))
        parts.append(sphere(0.3, granite(1.08), 10, 6).T(x, y + 2.9, 72.8))
    parts += railing([(-2.2, 72.8), (1.4, 72.8)], level(0, 72.8), h=2.2)
    # the gates at both ends of the walk, in the side walls (granite piers, iron gate)
    for x in (-45.8, 46.2):
        for z in (7.0, 12.5):
            parts.append(B(x - 0.45, x + 0.45, GFOUND, 2.6, z - 0.45, z + 0.45, granite(1.03)))
            parts.append(sphere(0.3, granite(1.08), 10, 6).T(x, 2.95, z))
        parts += railing([(x, 7.4), (x, 12.1)], 0.0, h=2.0)

    # ── Rear garden ──
    for i, (ring, fill) in enumerate(REAR_PARTERRES):
        parts += parterre(ring, FLOOR, fill, 400 + 17 * i)
    rx, rz = FOUNTAIN_R
    parts += fountain(rx, rz, FLOOR, 2.2, finial=False)
    parts.append(cyl(0.25, 0.15, 0.9, granite(1.05), 10).T(rx, FLOOR, rz))
    # steps at both ends of the walk, up to the rear garden beside the end wings
    for xa, xb in ((-44.9, -41.5), (42.3, 45.9)):
        parts += stairs_z(xa, xb, ZR - 1.4, ZR + 2.2, FLOOR, 0.0)
    # garden walls (OSM 231916286 south, 231916324 north), 1.4 m over the garden
    parts += wall_line([(-45.8, 5.5), (-44.6, -55.0), (-44.6, -122.0)], 1.4)
    parts += wall_line([(46.5, 5.5), (45.3, -89.4), (44.4, -92.5), (41.1, -93.8), (19.6, -93.9)], 1.4)
    # broderie terrace at the end of the axis: its two parterres and the cross paths
    for i, ring in enumerate(BRODERIE):
        parts += broderie(ring, 600 + i)
    # ── Trees ──
    for i, (x, z, kind, h, crown, girth) in enumerate(TREES):
        y = level(x, z)
        if kind is giant_sequoia:
            parts += kind(x, z, y, h, crown, 700 + 37 * i, girth)
        else:
            parts += kind(x, z, y, h, crown, 700 + 37 * i)
    return merge(parts)


def main():
    out = ROOT / "models" / "Q3848635"
    out.mkdir(parents=True, exist_ok=True)
    m = merge([house(), garden()])
    tris, size = write_glb(out / "model.glb", [("casita_del_principe", m)], roughness=0.9)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
