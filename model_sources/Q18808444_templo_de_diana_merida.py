"""
Templo de Diana, Mérida (Q18808444) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: a Roman temple of the imperial cult in the municipal forum of Augusta Emerita (time of Tiberius, early
1st century AD, granite), wrongly called «of Diana» since the 17th century; in the 16th century the palace of the
Conde de los Corbos was built inside its cella, which saved part of the colonnade. Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 202213495: the podium, 32.4 × 18.1 m, long axis at 143.1°; the front (south-east, where the
  base of the stair was found) faces 143.1°. Wikipedia: podium 3.23 m high, plan 32 × 18.5 m, peripteral and
  hexastyle with 11 columns on each long side, Corinthian columns of granite about 8 m high with Attic bases and
  fluted shafts; a relieving arch over the pediment, now rebuilt.
- PNOA orthophoto (IGN, CC BY 4.0) resampled in the model's axes: the palace roof (x −6.3…+8, z −9.6…+7), the empty
  rear of the podium with the stumps of the columns, the two Roman pools (OSM ways 1206491734 / 1206491736) on the
  long sides and the excavation of the stair in front (OSM way 1206491735), the paved precinct around.
- Photos on Wikimedia Commons (near-frontal and three-quarter views) for which columns still stand and the heights
  of the architrave, the attic with the arch and the palace.

What stands (simplified): the six front columns with their architrave, the attic and the rebuilt arch; a second
row one bay behind (pronaos) with beams to the front; the south-west flank (−X) with nine columns, the architrave
over the first seven; the north-east flank (+X) with three free columns at the front, three more half-sunk in the
palace wall and one at the rear corner of the palace; stumps elsewhere. The palace: two floors, cream render, a
two-storey Renaissance loggia on the front, Mudéjar twin windows in brick frames on the north-east side, a hipped
tile roof.

Axes (docs/MODEL_SPEC.md): +Y up, the temple front faces +Z (south-east), +X to the north-east,
origin on the plaza in the middle of the podium.

    python model_sources/Q18808444_templo_de_diana_merida.py   -> models/Q18808444/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, triangulate, extrude, cap, walls, write_glb  # noqa: E402

rng = np.random.default_rng(1912)

GRANITE = np.array((0.70, 0.645, 0.55))      # weathered granite of the columns
PODIUM = np.array((0.60, 0.555, 0.47))       # the ashlar of the podium
RENDER = (0.90, 0.85, 0.70)                  # the palace walls
RENDER_SH = (0.80, 0.75, 0.62)               # the walls inside the loggia
TILE = (0.58, 0.40, 0.31)                    # old Arab tiles
BRICK = (0.66, 0.38, 0.27)
DARK = (0.13, 0.12, 0.11)                    # openings
SAND = (0.80, 0.72, 0.58)                    # the plaza
PAVE = (0.70, 0.67, 0.60)
EARTH = (0.58, 0.49, 0.38)
IRON = (0.22, 0.22, 0.23)
FOUND = -3.0

PH = 3.23                                    # podium height
PW, PL = 18.12, 32.4                         # podium width (x) and length (z)
CX = 7.96                                    # column axes at x = ±CX
CZ = 15.1                                    # front / rear column axes at z = ±CZ
FX = [-CX + i * 2 * CX / 5 for i in range(6)]          # the six columns across
SZ = [CZ - i * 2 * CZ / 10 for i in range(11)]         # the eleven along each flank (0 = front)
COL_H = 8.0
ARCH_H = 0.85                                # the architrave
TOP = PH + COL_H                             # top of the capitals


def gran(k=1.0, base=GRANITE):
    return tuple(np.clip(base * k * (1 + rng.normal(0, 0.035)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def rect(x0, z0, x1, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def prism_xy(poly, z0, z1, c):
    """A prism from an outline drawn in the (x, y) plane (a façade outline), from z0 to z1."""
    pts, tris = triangulate([(x, y) for x, y in poly])
    n = len(pts)
    ring = [tuple(p) for p in pts]
    area = sum(ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1] for i in range(n))
    side = []
    for i in range(n):
        j = (i + 1) % n
        (xa, ya), (xb, yb) = ring[i], ring[j]
        q = [(xa, ya, z0), (xb, yb, z0), (xb, yb, z1), (xa, ya, z1)]
        f = [(0, 1, 2), (0, 2, 3)] if area > 0 else [(0, 2, 1), (0, 3, 2)]
        side.append(Mesh(q, f, c))
    cap0 = Mesh([(x, y, z0) for x, y in pts], tris if area > 0 else tris[:, ::-1], c)
    cap1 = Mesh([(x, y, z1) for x, y in pts], tris[:, ::-1] if area > 0 else tris, c)
    return merge(side + [cap0, cap1])


def arch_block(x0, x1, ys, ytop, z0, z1, c, seg=12):
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
        a = 2 * i
        F += [(a, a + 3, a + 1), (a, a + 2, a + 3)]
    out.append(Mesh(V, F, c))
    return merge(out)


def arcade(x0, x1, n, y0, ys, ytop, z0, z1, c, pier_c=None):
    """A wall from x0 to x1 (in the x/y plane, z0…z1 thick) opened by n equal semicircular arches springing at ys."""
    pier = 0.42
    w = (x1 - x0 - (n + 1) * pier) / n
    out = []
    for i in range(n + 1):
        a = x0 + i * (w + pier)
        out.append(B(a, a + pier, y0, ys, z0, z1, pier_c or c))
    for i in range(n):
        a = x0 + pier + i * (w + pier)
        out.append(arch_block(a, a + w, ys, ytop, z0, z1, c))
    return out


def rot_y(m, ang):
    return m.R(0, ang, 0)


# ── The order ────────────────────────────────────────────────────────────────
def fluted(r0, r1, h, c, flutes=12):
    """A fluted drum: a star prism with 2 × flutes sides (no caps)."""
    n = 2 * flutes
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    k = np.where(np.arange(n) % 2 == 0, 1.0, 0.92)
    V = np.vstack([np.stack([np.cos(a) * r0 * k, np.zeros(n), np.sin(a) * r0 * k], 1),
                   np.stack([np.cos(a) * r1 * k, np.full(n, h), np.sin(a) * r1 * k], 1)])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [(i, n + i, n + j), (i, n + j, j)]
    return Mesh(V, F, c)


def column(x, z, y0=PH, stump=None):
    """Attic base, fluted shaft in drums, Corinthian capital (8 m in all). stump = height of a broken shaft."""
    out = [B(x - 0.62, x + 0.62, y0, y0 + 0.16, z - 0.62, z + 0.62, gran(0.97))]
    base = lathe([(0.58, 0.0), (0.60, 0.07), (0.56, 0.14), (0.50, 0.17), (0.50, 0.21), (0.53, 0.25), (0.50, 0.29),
                  (0.47, 0.29)], gran(), 14).T(x, y0 + 0.16, z)
    out.append(base)
    yb = y0 + 0.45
    shaft_h = 6.55
    r0, r1 = 0.47, 0.41
    if stump is not None:
        shaft_h = stump
    nd = max(1, int(round(shaft_h / 0.95)))
    for i in range(nd):
        t0, t1 = i / nd, (i + 1) / nd
        ra, rb = r0 + (r1 - r0) * t0 * shaft_h / 6.55, r0 + (r1 - r0) * t1 * shaft_h / 6.55
        d = fluted(ra, rb, shaft_h / nd, gran(1.0 + rng.normal(0, 0.04)))
        out.append(d.T(x + rng.normal(0, 0.012), yb + t0 * shaft_h, z + rng.normal(0, 0.012)))
    if stump is not None:
        out.append(cyl(r0 * 0.95, r0 * 0.75, 0.12, gran(0.92), 10).T(x, yb + shaft_h, z))
        return out
    yc = yb + shaft_h                                   # 7.0 above the podium: the capital, 1.0 m
    out.append(lathe([(0.43, 0.0), (0.45, 0.06), (0.42, 0.10), (0.44, 0.45), (0.50, 0.78), (0.53, 0.80)],
                     gran(1.04), 12).T(x, yc, z))
    for row, (hh, rr, rot) in enumerate(((0.38, 0.47, 0.0), (0.62, 0.50, math.pi / 8))):
        for k in range(8):                              # two rows of acanthus leaves
            a = rot + k * math.pi / 4
            leaf = B(-0.13, 0.13, 0.0, hh, -0.06, 0.06, gran(1.06)).R(0.22, 0, 0)
            leaf.R(0, -a + math.pi / 2, 0).T(x + math.cos(a) * rr * 0.86, yc + 0.05 + 0.1 * row, z + math.sin(a) * rr * 0.86)
            out.append(leaf)
    for k in range(4):                                  # corner volutes under the abacus
        a = math.pi / 4 + k * math.pi / 2
        out.append(B(-0.12, 0.12, 0, 0.22, -0.12, 0.12, gran(1.05)).T(x + math.cos(a) * 0.56, yc + 0.66, z + math.sin(a) * 0.56))
    out.append(B(x - 0.66, x + 0.66, yc + 0.80, yc + 1.0, z - 0.66, z + 0.66, gran(1.03)))
    return out


def beam(x0, x1, z0, z1, c=None):
    """Architrave blocks between two column axes (along x or along z), 0.85 m high, 0.9 m wide."""
    out = []
    if abs(x1 - x0) > abs(z1 - z0):
        out.append(B(x0, x1, TOP, TOP + ARCH_H, z0 - 0.45, z0 + 0.45, c or gran(1.02)))
        out.append(B(x0, x1, TOP + ARCH_H - 0.18, TOP + ARCH_H, z0 - 0.52, z0 + 0.52, c or gran(1.05)))
    else:
        out.append(B(x0 - 0.45, x0 + 0.45, TOP, TOP + ARCH_H, z0, z1, c or gran(1.02)))
        out.append(B(x0 - 0.52, x0 + 0.52, TOP + ARCH_H - 0.18, TOP + ARCH_H, z0, z1, c or gran(1.05)))
    return out


def temple():
    parts = []
    # podium: plinth, ashlar body, moulded cornice, foundation
    hx, hz = PW / 2, PL / 2
    parts.append(extrude(rect(-hx, -hz, hx, hz), FOUND, PH - 0.32, tuple(PODIUM)))
    parts.append(extrude(rect(-hx - 0.25, -hz - 0.25, hx + 0.25, hz + 0.25), FOUND, 0.35, tuple(PODIUM * 0.95)))
    parts.append(extrude(rect(-hx - 0.12, -hz - 0.12, hx + 0.12, hz + 0.12), PH - 0.32, PH - 0.12, tuple(PODIUM * 1.08), top=False))
    parts.append(extrude(rect(-hx - 0.28, -hz - 0.28, hx + 0.28, hz + 0.28), PH - 0.12, PH, tuple(PODIUM * 1.12),
                         top_c=(0.62, 0.58, 0.50)))
    # the ashlar of the four faces, laid in courses, each block a little lighter or darker; rubble where it is lost
    V, F, C = [], [], []
    for face in range(4):
        L = PW if face < 2 else PL
        s = 1 if face % 2 == 0 else -1
        for k, y in enumerate(np.arange(0.35, PH - 0.33, 0.55)):
            y1 = min(y + 0.55, PH - 0.32)
            u = -L / 2 - (0.6 if k % 2 else 0.0)
            while u < L / 2:
                u1 = min(u + rng.uniform(0.9, 1.5), L / 2)
                a = max(u, -L / 2)
                lost = face < 2 and rng.random() < (0.35 if face == 0 else 0.25)
                col = (np.array((0.57, 0.47, 0.37)) if lost else PODIUM) * (1 + rng.normal(0, 0.06))
                d = 0.012 if not lost else -0.0
                if face < 2:
                    zf = s * (hz + d)
                    q = [(a, y + 0.02, zf), (u1 - 0.03, y + 0.02, zf), (u1 - 0.03, y1, zf), (a, y1, zf)]
                    if s < 0:
                        q = [(-x, yy, zz) for x, yy, zz in q]
                else:
                    xf = s * (hx + d)
                    q = [(xf, y + 0.02, -a), (xf, y + 0.02, -(u1 - 0.03)), (xf, y1, -(u1 - 0.03)), (xf, y1, -a)]
                    if s < 0:
                        q = [(xx, yy, -zz) for xx, yy, zz in q]
                o = len(V)
                V += q
                F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
                C += [np.clip(col, 0, 1)] * 4
                u = u1
    parts.append(Mesh(V, F, np.array(C)))
    # the columns that stand (index along the flanks: 0 = front corner)
    stand = set()
    for x in FX:
        stand.add((round(x, 3), SZ[0]))
    for x in FX[1:5]:                                   # the second row of the pronaos
        stand.add((round(x, 3), SZ[1]))
    for i in range(0, 9):                               # south-west flank (−X)
        stand.add((round(-CX, 3), SZ[i]))
    for i in (0, 1, 2, 3, 4, 5, 8):                     # north-east flank (+X): free, sunk in the palace, the corner
        stand.add((round(CX, 3), SZ[i]))
    for x, z in sorted(stand):
        parts += column(x, z)
    # stumps of the lost columns
    stumps = [(-CX, SZ[9], 1.1), (-CX, SZ[10], 0.6), (CX, SZ[9], 0.45), (CX, SZ[10], 1.4)]
    stumps += [(FX[1], -CZ, 0.5), (FX[2], -CZ, 0.9), (FX[3], -CZ, 0.35), (FX[4], -CZ, 0.7)]
    for x, z, h in stumps:
        parts += column(x, z, stump=h)
    for (x, z) in ((-3.5, -13.0), (1.2, -12.6), (4.6, -13.4), (-1.0, -11.2)):   # loose drums on the podium
        parts.append(cyl(0.45, 0.45, 0.7, gran(0.95), 12).R(0, 0, math.pi / 2).T(x, PH + 0.45, z))

    # entablature
    parts += beam(-CX - 0.55, CX + 0.55, CZ, CZ)                       # front
    parts += beam(-CX - 0.45, CX + 0.45, SZ[1], SZ[1])                  # second row
    for x in FX[1:5]:                                                  # beams from the front to the second row
        parts += beam(x, x, SZ[1], CZ)
    parts += beam(-CX, -CX, SZ[6] - 0.55, CZ + 0.45)                   # south-west flank, over the first seven
    parts += beam(CX, CX, SZ[2] - 0.55, CZ + 0.45)                     # north-east flank, over the first three
    # the attic over the middle of the front and the rebuilt relieving arch
    y = TOP + ARCH_H
    parts.append(B(-4.4, 4.4, y, y + 0.75, CZ - 0.75, CZ + 0.62, gran(1.06)))
    parts.append(B(-4.6, 4.6, y + 0.55, y + 0.75, CZ - 0.8, CZ + 0.78, gran(1.1)))
    parts.append(B(-4.05, 4.05, y + 0.75, y + 1.45, CZ - 0.6, CZ + 0.5, gran(1.04)))
    parts.append(B(-4.25, 4.25, y + 1.3, y + 1.5, CZ - 0.65, CZ + 0.66, gran(1.1)))
    ya = y + 1.5
    parts.append(B(-2.6, -1.25, ya, ya + 0.45, CZ - 0.45, CZ + 0.4, gran(1.05)))
    parts.append(B(1.25, 2.6, ya, ya + 0.45, CZ - 0.45, CZ + 0.4, gran(1.05)))
    ro, ri = 2.2, 1.3
    ring = [(ro * math.cos(a), ya + 0.45 + ro * math.sin(a)) for a in np.linspace(0, math.pi, 13)]
    ring += [(ri * math.cos(a), ya + 0.45 + ri * math.sin(a)) for a in np.linspace(math.pi, 0, 13)]
    parts.append(prism_xy(ring, CZ - 0.42, CZ + 0.38, gran(1.08)))
    for a in np.linspace(0.12, math.pi - 0.12, 9):                    # voussoir joints
        x0, y0 = ri * math.cos(a), ya + 0.45 + ri * math.sin(a)
        x1, y1 = ro * math.cos(a), ya + 0.45 + ro * math.sin(a)
        for s in (-1, 1):
            zz = CZ + (0.385 if s > 0 else -0.425)
            d = np.array([x1 - x0, y1 - y0]); L = np.linalg.norm(d)
            j = B(-0.025, 0.025, 0, L, -0.01, 0.01, gran(0.75)).R(0, 0, a - math.pi / 2).T(x0, y0, zz)
            parts.append(j)
    return parts


# ── The palace of the Conde de los Corbos (16th century) ─────────────────────
PX0, PX1 = -6.3, CX
PZ0, PZ1, PZL = -9.6, 4.5, 7.0          # rear wall, back of the loggia, front of the loggia
EY = PH + 7.3                           # eaves
FL2 = PH + 3.9                          # upper floor


def arched_window(cx, cy, w, h, face, pos, c=DARK, frame=None):
    """A dark arched opening (w wide, h to the crown) on a wall facing face ('-x', '+x', '+z') at pos."""
    r = w / 2
    poly = [(-r, 0), (r, 0)] + [(r * math.cos(a), h - r + r * math.sin(a)) for a in np.linspace(0, math.pi, 9)][1:]
    out = []
    if frame:
        fr = [(-r - 0.18, -0.1), (r + 0.18, -0.1)] + [((r + 0.18) * math.cos(a), h - r + (r + 0.18) * math.sin(a))
                                                     for a in np.linspace(0, math.pi, 9)][1:]
        out.append(prism_xy(fr, 0.0, 0.05, frame))
    out.append(prism_xy(poly, 0.0, 0.08, c))
    m = merge(out)
    if face == "+z":
        return m.T(cx, cy, pos)
    if face == "-x":
        return m.R(0, -math.pi / 2, 0).T(pos, cy, cx)
    return m.R(0, math.pi / 2, 0).T(pos, cy, cx)


def hip_roof(x0, x1, z0, z1, y0, h, c):
    """A hipped roof on a rectangle, ridge along the longer side."""
    xm, zm = (x0 + x1) / 2, (z0 + z1) / 2
    if (z1 - z0) >= (x1 - x0):
        d = (x1 - x0) / 2
        r0, r1 = (xm, y0 + h, z0 + d), (xm, y0 + h, z1 - d)
    else:
        d = (z1 - z0) / 2
        r0, r1 = (x0 + d, y0 + h, zm), (x1 - d, y0 + h, zm)
    A, Bp, C, D = (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)
    V = [A, Bp, C, D, r0, r1]
    if (z1 - z0) >= (x1 - x0):
        F = [(0, 4, 1), (1, 4, 5), (1, 5, 2), (2, 5, 3), (3, 5, 4), (3, 4, 0)]
    else:
        F = [(0, 4, 5), (0, 5, 1), (1, 5, 2), (2, 5, 4), (2, 4, 3), (3, 4, 0)]
    m = Mesh(V, F, c)
    # make every face point up
    out = []
    for f in m.f:
        a, b, cc = m.v[f[0]], m.v[f[1]], m.v[f[2]]
        n = np.cross(b - a, cc - a)
        out.append(Mesh([a, b, cc], [(0, 1, 2)] if n[1] > 0 else [(0, 2, 1)], c))
    under = Mesh([A, Bp, C, D], [(0, 1, 2), (0, 2, 3)], (0.45, 0.33, 0.26))
    return merge(out + [under])


def palace():
    parts = []
    parts.append(extrude(rect(PX0, PZ0, PX1, PZ1), PH, EY, RENDER))
    parts.append(B(PX0, PX1, PH, EY, PZ1 - 0.02, PZ1 + 0.01, RENDER_SH))          # back wall of the loggia
    parts.append(B(PX0 - 0.05, PX1 + 0.05, PH, PH + 0.5, PZ0 - 0.05, PZ1, (0.66, 0.6, 0.5)))  # granite socle
    # the loggia: side walls, the floor between the galleries, the two arcades
    parts.append(B(PX0, PX0 + 0.45, PH, EY, PZ1, PZL, RENDER))
    parts.append(B(PX1 - 0.45, PX1, PH, EY, PZ1, PZL, RENDER))
    parts.append(B(PX0, PX1, FL2 - 0.25, FL2, PZ1, PZL, (0.78, 0.72, 0.6)))
    z0, z1 = PZL - 0.4, PZL
    parts += arcade(PX0 + 0.45, PX1 - 0.45, 5, PH, PH + 2.35, FL2, z0, z1, RENDER, (0.80, 0.77, 0.68))
    parts.append(B(PX0, PX1, FL2, FL2 + 0.9, z0 + 0.1, z1 - 0.05, RENDER))      # parapet of the upper gallery
    parts += arcade(PX0 + 0.45, PX1 - 0.45, 8, FL2 + 0.9, FL2 + 2.45, EY, z0, z1, RENDER, (0.80, 0.77, 0.68))
    parts.append(B(PX0 - 0.1, PX1 + 0.1, EY - 0.25, EY, PZ0 - 0.1, PZL + 0.1, (0.84, 0.79, 0.66)))   # eaves band
    # doors and windows inside the loggia
    for x in (-3.4, 0.8, 5.0):
        parts.append(arched_window(x, PH, 1.3, 2.6, "+z", PZ1 + 0.02))
        parts.append(arched_window(x, FL2 + 0.3, 1.0, 2.0, "+z", PZ1 + 0.02))
    # south-west side (−X): arched windows on two floors
    for z in (-7.4, -4.2, -1.0, 2.2):
        parts.append(arched_window(z, PH + 0.9, 1.1, 2.2, "-x", PX0 - 0.02, frame=(0.82, 0.78, 0.66)))
        parts.append(arched_window(z, FL2 + 0.7, 1.0, 2.0, "-x", PX0 - 0.02, frame=(0.82, 0.78, 0.66)))
    # north-east side (+X): two arched doors, two Mudéjar twin windows in brick frames, a window at the front
    for z in (-6.2, -2.6):
        parts.append(arched_window(z, PH + 0.4, 1.1, 2.3, "+x", PX1 + 0.02, frame=(0.82, 0.78, 0.66)))
    for z in (-7.4, -4.0):
        parts.append(B(PX1, PX1 + 0.06, FL2 + 0.4, FL2 + 2.6, z - 0.85, z + 0.85, BRICK))
        for dz in (-0.33, 0.33):
            parts.append(arched_window(z + dz, FL2 + 0.75, 0.5, 1.5, "+x", PX1 + 0.05))
    parts.append(B(PX1, PX1 + 0.05, FL2 + 0.6, FL2 + 2.2, 1.0, 2.0, DARK))
    parts.append(B(PX1, PX1 + 0.05, PH + 1.2, PH + 2.6, 1.0, 2.0, DARK))
    # rear (−Z): windows
    for x in (-3.0, 1.5, 5.5):
        parts.append(B(x - 0.5, x + 0.5, FL2 + 0.7, FL2 + 2.2, PZ0 - 0.05, PZ0, DARK))
    parts.append(hip_roof(PX0 - 0.35, PX1 + 0.35, PZ0 - 0.35, PZL + 0.35, EY, 2.6, TILE))
    return parts


# ── The precinct: plaza, pools, the excavation of the stair ──────────────────
def basin(x0, z0, x1, z1, depth, rim_c, floor_c, inner_walls=()):
    out = [B(x0, x1, -depth - 0.3, -depth, z0, z1, floor_c)]
    out.append(cap(rect(x0, z0, x1, z1), -depth, floor_c))
    t = 0.45
    for (a0, a1, b0, b1) in ((x0 - t, x1 + t, z0 - t, z0), (x0 - t, x1 + t, z1, z1 + t),
                             (x0 - t, x0, z0, z1), (x1, x1 + t, z0, z1)):
        out.append(B(a0, a1, -depth - 0.3, 0.12, b0, b1, rim_c))
    for (a0, a1, b0, b1, h) in inner_walls:
        out.append(B(a0, a1, -depth, -depth + h, b0, b1, (0.6, 0.55, 0.47)))
    return out


def railing(pts, h=1.0):
    out = []
    for (xa, za), (xb, zb) in zip(pts[:-1], pts[1:]):
        L = math.hypot(xb - xa, zb - za)
        n = max(1, int(L / 2.0))
        for i in range(n + 1):
            x, z = xa + (xb - xa) * i / n, za + (zb - za) * i / n
            out.append(B(x - 0.03, x + 0.03, 0, h, z - 0.03, z + 0.03, IRON))
        ang = math.atan2(zb - za, xb - xa)
        for y in (h - 0.05, h * 0.5):
            out.append(B(0, L, y, y + 0.05, -0.025, 0.025, IRON).R(0, -ang, 0).T(xa, 0, za))
    return out


PLAZA = [(-25.5, -21.5), (25.5, -21.5), (25.5, 14.5), (12.5, 14.5), (12.5, 31.0), (-25.5, 31.0)]
SW_POOL = (-21.3, -6.9, -17.6, 4.5)
NE_POOL = (17.3, -9.2, 23.8, 10.0)
PIT = (-8.6, 16.6, 9.0, 30.0)                    # the excavated stair and the area in front of it


def precinct():
    parts = []
    holes = [rect(SW_POOL[0], SW_POOL[1], SW_POOL[2], SW_POOL[3]), rect(NE_POOL[0], NE_POOL[1], NE_POOL[2], NE_POOL[3]),
             rect(PIT[0], PIT[1], PIT[2], PIT[3]), rect(-PW / 2, -PL / 2, PW / 2, PL / 2)]
    parts.append(walls(PLAZA, FOUND + 0.5, 0.0, SAND))
    parts.append(cap(PLAZA, 0.0, SAND, holes=holes))
    parts.append(cap(PLAZA, FOUND + 0.5, SAND, up=False))
    # a granite pavement band round the podium
    hx, hz = PW / 2, PL / 2
    for (a0, a1, b0, b1) in ((-hx - 2.2, hx + 2.2, -hz - 2.2, -hz), (-hx - 2.2, -hx, -hz, hz + 0.4), (hx, hx + 2.2, -hz, hz + 0.4)):
        parts.append(B(a0, a1, -0.1, 0.03, b0, b1, PAVE))
    # the two Roman pools on the long sides (dry, excavated), with their railings
    parts += basin(*SW_POOL, 1.2, (0.66, 0.62, 0.54), EARTH)
    parts += basin(*NE_POOL, 1.6, (0.66, 0.62, 0.54), EARTH,
                   inner_walls=[(17.3, 23.8, -2.0, -1.4, 0.9), (17.3, 23.8, 4.0, 4.6, 1.1), (19.8, 20.4, -9.2, 10.0, 0.7)])
    x0, z0, x1, z1 = NE_POOL
    parts += railing([(x0 - 1.0, z0 - 1.0), (x0 - 1.0, z1 + 1.0), (x1 + 0.7, z1 + 1.0)])
    x0, z0, x1, z1 = SW_POOL
    parts += railing([(x1 + 0.9, z0 - 0.9), (x1 + 0.9, z1 + 0.9)])
    # the excavation in front: the foundations of the stair and of the forum, under a walkway
    x0, z0, x1, z1 = PIT
    parts += basin(x0, z0, x1, z1, 1.8, (0.64, 0.6, 0.52), EARTH)
    for k in range(5):                                                # stepped foundation of the stair
        zz = PL / 2 + 0.2 + k * 1.0
        parts.append(B(-6.0 + k * 0.4, 6.0 - k * 0.4, -1.8, -0.3 - k * 0.3, PL / 2, zz + 1.0, gran(0.9, PODIUM)))
    for (a0, a1, b0, b1, h) in ((-8.0, 8.0, 25.5, 26.2, 1.2), (-2.0, -1.3, 23.0, 31.0, 1.0), (3.5, 4.2, 26.0, 31.0, 0.9),
                                (-7.5, -4.0, 28.5, 29.2, 0.8)):
        parts.append(B(a0, a1, -1.8, -1.8 + h, b0, b1, (0.6, 0.54, 0.45)))
    deck = (-6.8, 17.2, 7.2, 24.6)                                    # the steel walkway over the stair
    parts.append(B(deck[0], deck[2], 0.05, 0.2, deck[1], deck[3], (0.36, 0.36, 0.37)))
    for x in np.linspace(deck[0] + 0.3, deck[2] - 0.3, 6):
        for z in np.linspace(deck[1] + 0.3, deck[3] - 0.3, 3):
            parts.append(B(x - 0.06, x + 0.06, -1.8, 0.05, z - 0.06, z + 0.06, IRON))
    parts += [m.T(0, 0.2, 0) for m in railing([(deck[0], deck[1]), (deck[0], deck[3]), (deck[2], deck[3]), (deck[2], deck[1])])]
    return parts


def main():
    out = ROOT / "models" / "Q18808444"
    out.mkdir(parents=True, exist_ok=True)
    t = merge(temple())
    p = merge(palace())
    g = merge(precinct())
    tris, size = write_glb(out / "model.glb", [("temple", t), ("palace", p), ("precinct", g)], roughness=0.9)
    allv = np.vstack([t.v, p.v, g.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
