"""
Teatro romano de Mérida — Roman Theatre of Augusta Emerita, Mérida (Q2700544) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: promoted by Marcus Vipsanius Agrippa, inaugurated 16–15 BC; the present scaenae frons is a rebuilding
of the late 1st century AD (Flavian / Trajanic), reformed again in 333–337 AD (architects unknown). Excavated from
1910 (José Ramón Mélida); the frons, part of the seats and of the peristyle colonnade were rebuilt from 1962 under
José Menéndez-Pidal. The model shows the theatre as it stands today, with the restored frons. Public domain.

Measurements (see meta.json for the sources):
- Plan from OpenStreetMap relation 1316310 (outer way 37858767, inner way 89067888): a circle fitted to the outer
  arc of the cavea gives a radius of 42.8 m (diameter 85.6 m; published «about 86 m»), centre 38.915363, −6.338658
  (the origin). The line of the scaenae frons runs at 46°, so the frons faces 136° (south-east), towards the cavea.
  The orchestra (radius 13.5 m to the steps of the proedria) shares the centre. Scaena building 63 m long (OSM
  63.3 m; published 63 m), 10.5 m deep behind the frons; the pulpitum 60 × 7.5 m (published 60 × 7 m) with its
  front of alternating rectangular and semicircular niches.
- Cavea (Wikipedia es/en): three marble seats of honour (proedria) behind a marble parapet; the ima cavea with
  22 rows in five cunei divided by six stairways (rows 0.65 m deep, 0.36 m high: 31.7 m at the top, where OSM
  way 261488525 draws the praecinctio at r = 31.0 m); a balteus with the vomitoria; the media cavea with five rows;
  a second balteus; the summa cavea, ruined: seven masses of Roman concrete split by the vomitoria, the
  «Siete Sillas», behind which the outer wall stands only a few metres above the street (the hillside of San Albín
  carries more than half of the cavea). Heights checked against the IGN MDT05: the orchestra is the lowest point,
  the ground round the top of the cavea is 10–12 m above it.
- Scaenae frons, measured on near-frontal photos scaled by the 60 m of the pulpitum (±5 %): pulpitum 1.2 m above
  the orchestra; podium 2.0 m; lower Corinthian order 6.4 m; entablature 1.4 m; pedestal 1.0 m; upper order 4.9 m;
  entablature 1.3 m: 17 m above the stage (published 17.5 m). Shafts of blue-grey veined marble, white bases,
  capitals and entablatures; statues (Ceres, Pluto, Proserpina, togati and thoracati) between the lower columns.
  The central exedra (semicircular) with the valva regia, two rectangular recesses with the valvae hospitales.
  Today the upper order is complete only in parts: on the north-east half the masonry of the frons stands behind
  it, with a broken pediment; on the south-west end the upper columns stand free, without entablature.
- Behind the scaena: the porticus post scaenam, a garden (layout of beds, hedges and the round basin from OSM
  ways 260417182–260417187 and 260417193 and the PNOA orthophoto) with the stumps of its double colonnade of
  stuccoed granite columns, the stretch rebuilt with a roof on the north-east side, and the small hall of the
  imperial cult on the axis at the far end.
Colours: granite and grey-brown stone of the seats and walls, Roman concrete of the ruined summa cavea, white and
blue-veined Macael marble of the frons, white and bluish marble slabs of the orchestra.

Axes (docs/MODEL_SPEC.md): +Y up, the scaenae frons faces +Z (heading 136°, towards the cavea), +X to the
north-east (46°). Origin at the centre of the orchestra, y = 0 on its marble floor. Walls go down to −3 m.

    python model_sources/Q2700544_teatro_romano_de_merida.py   -> models/Q2700544/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, extrude, triangulate, crumple, noise3, fbm, write_glb  # noqa: E402
from meshkit import _orient  # noqa: E402

rng = np.random.default_rng(1615)

# ── Colours ──────────────────────────────────────────────────────────────────
SEAT = (0.79, 0.75, 0.67)          # granite seats (many restored)
SEAT_D = (0.70, 0.66, 0.58)
STAIR = (0.84, 0.81, 0.74)
STONE = (0.62, 0.55, 0.45)         # grey-brown masonry of the walls (diorite, granite, opus incertum)
STONE_D = (0.52, 0.46, 0.38)
CONC = (0.60, 0.53, 0.43)          # opus caementicium of the summa cavea
MARBLE = (0.93, 0.92, 0.88)        # white marble: bases, capitals, entablature
VEIN = (0.66, 0.70, 0.74)          # blue-grey veined marble of the shafts
STATUE = (0.88, 0.86, 0.80)
ORCH_A = (0.90, 0.89, 0.86)
ORCH_B = (0.66, 0.69, 0.72)
STAGE = (0.71, 0.60, 0.47)         # stage floor (earth over the pulpitum)
SAND = (0.78, 0.69, 0.55)          # paths and open ground
GRASS = (0.42, 0.52, 0.28)
HEDGE = (0.22, 0.36, 0.18)
LEAF = (0.23, 0.36, 0.18)
WOOD = (0.40, 0.31, 0.22)
BRICK = (0.70, 0.45, 0.33)
DARK = (0.16, 0.14, 0.12)          # openings seen in shadow
WATER = (0.22, 0.30, 0.30)
ROOF_G = (0.40, 0.48, 0.27)        # planted roof of the rebuilt portico

FOUND = -3.0
R_ORCH = 13.5
Z_PAR = 5.5                        # the cavea ends at z = 5.5 m; the parodoi run between it and the pulpitum
Z_PULP = 1.0                       # front of the pulpitum
Z_FRONS = -6.5                     # face of the frons wall
Z_POD = -4.2                       # front of the frons podium
Z_BACK = -17.0                     # back wall of the scaena building
X_SCAENA = 31.6
Y_STAGE = 1.2


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def mirror_x(m):
    n = m.copy(); n.v[:, 0] *= -1; n.f = n.f[:, ::-1]; return n


def both_x(ms):
    ms = [m for m in ms if m is not None]
    return ms + [mirror_x(m) for m in ms]


def tint(m, amp=0.08, freq=0.35, seed=0):
    """Vary the colour a little (stone is never flat)."""
    k = 1 + amp * (2 * fbm(m.v, freq, seed, 2) - 1)
    m.c = np.clip(m.c * k[:, None], 0, 1)
    return m


def sphere(r, c, seg=10, rings=6):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def prism_xy(poly, z0, z1, c):
    """A polygon in the (x, y) plane extruded from z0 to z1."""
    pts, tris = triangulate(poly)
    n = len(pts)
    V = [(p[0], p[1], z1) for p in pts] + [(p[0], p[1], z0) for p in pts]
    F = [tuple(t) for t in tris] + [(n + t[0], n + t[2], n + t[1]) for t in tris]
    ms = [Mesh(V, F, c)]
    R = _orient(poly, True)
    sv, sf = [], []
    for i in range(len(R)):
        (u0, y0), (u1, y1) = R[i], R[(i + 1) % len(R)]
        o = len(sv)
        sv += [(u0, y0, z0), (u1, y1, z0), (u1, y1, z1), (u0, y0, z1)]
        sf += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    ms.append(Mesh(sv, sf, c))
    return merge(ms)


# ── Polar pieces of the cavea ───────────────────────────────────────────────
def a_end(r):
    """Angle where the circle of radius r meets the end line z = Z_PAR."""
    return math.asin(min(Z_PAR / r, 0.99))


def ang(r, u):
    a0 = a_end(r)
    return a0 + u * (math.pi - 2 * a0)


def P(r, u, y):
    a = ang(r, u)
    return (r * math.cos(a), y, r * math.sin(a))


def strip(r0, r1, y, c, n=96, u0=0.0, u1=1.0):
    """Horizontal annulus between r0 and r1 at height y (u along the cavea, 0 = +x end, 1 = −x end)."""
    us = np.linspace(u0, u1, n + 1)
    V = [P(r0, u, y) for u in us] + [P(r1, u, y) for u in us]
    F = []
    for i in range(n):
        F += [(i, i + 1, n + 2 + i), (i, n + 2 + i, n + 1 + i)]
    return Mesh(V, F, c)


def riser(r, y0, y1, c, n=96, u0=0.0, u1=1.0):
    """Vertical cylindrical face at radius r (facing the orchestra)."""
    us = np.linspace(u0, u1, n + 1)
    V = [P(r, u, y0) for u in us] + [P(r, u, y1) for u in us]
    F = []
    for i in range(n):
        F += [(i, n + 2 + i, i + 1), (i, n + 1 + i, n + 2 + i)]
    return Mesh(V, F, c)


def block(r0, r1, y0, y1, c, n=96, u0=0.0, u1=1.0):
    """A closed annular block (walls, parapets)."""
    ms = [strip(r0, r1, y1, c, n, u0, u1), riser(r0, y0, y1, c, n, u0, u1),
          riser(r1, y0, y1, c, n, u0, u1)]
    for u in (u0, u1):
        p = [P(r0, u, y0), P(r1, u, y0), P(r1, u, y1), P(r0, u, y1)]
        ms.append(Mesh(p, [(0, 1, 2), (0, 2, 3)], c))
    return merge(ms)


def seg_block(r0, r1, a0, a1, y0, y1, c, n=6):
    """An annular block between two plain angles (radians), for ruins and vomitoria."""
    def p(r, a, y):
        return (r * math.cos(a), y, r * math.sin(a))
    As = np.linspace(a0, a1, n + 1)
    V, F = [], []
    top = [p(r0, a, y1) for a in As] + [p(r1, a, y1) for a in As]
    o = 0
    V += top
    for i in range(n):
        F += [(i, i + 1, n + 2 + i), (i, n + 2 + i, n + 1 + i)]
    for r in (r0, r1):
        o = len(V)
        V += [p(r, a, y0) for a in As] + [p(r, a, y1) for a in As]
        for i in range(n):
            F += [(o + i, o + n + 2 + i, o + i + 1), (o + i, o + n + 1 + i, o + n + 2 + i)]
    for a in (a0, a1):
        o = len(V)
        V += [p(r0, a, y0), p(r1, a, y0), p(r1, a, y1), p(r0, a, y1)]
        F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    return Mesh(V, F, c)


# ── Cavea ────────────────────────────────────────────────────────────────────
IMA_R0, IMA_DR, IMA_DY, IMA_N, IMA_Y0 = 17.4, 0.65, 0.36, 22, 0.75
IMA_TOP = IMA_Y0 + IMA_N * IMA_DY                     # 8.67
R_PRAEC = IMA_R0 + IMA_N * IMA_DR                     # 31.7
R_BALT, Y_BALT = 33.0, 10.0
MED_R0, MED_DR, MED_DY, MED_N = 33.4, 0.72, 0.42, 5
MED_TOP = Y_BALT + MED_N * MED_DY                     # 12.1
R_BALT2, Y_BALT2 = MED_R0 + MED_N * MED_DR, 13.1      # 37.0
R_SUM0, R_SUM1, R_OUT = 37.4, 40.6, 42.8
STAIRS_IMA = [0.07, 0.25, 0.42, 0.58, 0.75, 0.93]       # six stairways: five cunei
VOMIT = [k / 7 for k in range(1, 7)]                   # six vomitoria: the seven «sillas»


def cavea():
    seats, light, stone, conc, dark = [], [], [], [], []
    # proedria: three wide marble steps, the parapet and the walkway
    for k in range(3):
        r0 = R_ORCH + 0.8 * k
        y = 0.25 * (k + 1)
        light.append(riser(r0, y - 0.25, y, MARBLE))
        light.append(strip(r0, r0 + 0.8, y, MARBLE))
    light.append(block(15.9, 16.2, 0.0, 1.75, MARBLE))
    seats.append(strip(16.2, IMA_R0, IMA_Y0, SEAT_D))
    seats.append(riser(16.2, FOUND, IMA_Y0, SEAT_D))
    # ima cavea: 22 rows
    y = IMA_Y0
    for k in range(IMA_N):
        r0 = IMA_R0 + k * IMA_DR
        c = SEAT if k % 2 == 0 else tuple(np.array(SEAT) * 0.97)
        seats.append(riser(r0, y, y + IMA_DY, SEAT_D))
        y += IMA_DY
        seats.append(strip(r0, r0 + IMA_DR, y, c))
        for u in STAIRS_IMA:                      # half steps of the stairways
            a = ang(r0, u); w = 0.55 / r0
            seg = seg_block(r0, r0 + IMA_DR / 2, a - w, a + w, y - IMA_DY, y - IMA_DY / 2 + 0.02, STAIR, n=1)
            light.append(seg)
    # praecinctio, balteus with the vomitoria of the ima cavea
    seats.append(strip(R_PRAEC, R_BALT, IMA_TOP, SEAT_D))
    stone.append(riser(R_BALT, IMA_TOP, Y_BALT, STONE))
    for u in (0.1, 0.3, 0.5, 0.7, 0.9):
        a = ang(R_BALT, u); w = 1.1 / R_BALT
        dark.append(seg_block(R_BALT - 0.05, R_BALT + 0.3, a - w, a + w, IMA_TOP, Y_BALT - 0.05, DARK, n=2))
    # media cavea: 5 rows
    y = Y_BALT
    seats.append(strip(R_BALT, MED_R0, Y_BALT, SEAT_D))
    for k in range(MED_N):
        r0 = MED_R0 + k * MED_DR
        seats.append(riser(r0, y, y + MED_DY, SEAT_D))
        y += MED_DY
        seats.append(strip(r0, r0 + MED_DR, y, SEAT))
    stone.append(riser(R_BALT2, MED_TOP, Y_BALT2, STONE))
    seats.append(strip(R_BALT2, R_SUM0, Y_BALT2, SEAT_D))
    # summa cavea: seven ruined masses of concrete between the vomitoria (the «Siete Sillas»)
    edges = [0.0] + VOMIT + [1.0]
    for i in range(7):
        u0 = edges[i] + (0.012 if i else 0.0)
        u1 = edges[i + 1] - (0.012 if i < 6 else 0.0)
        nn = 10
        us = np.linspace(u0, u1, nn + 1)
        for j in range(nn):
            a0, a1 = ang(R_SUM0, us[j]), ang(R_SUM0, us[j + 1])
            top = 13.6 + 1.4 * fbm(np.array([[us[j] * 40, 0.0, i * 3.0]]), 0.5, 7, 2)[0] + rng.uniform(-0.3, 0.3)
            mid = Y_BALT2 + 0.4 + (top - Y_BALT2) * 0.55
            conc.append(seg_block(R_SUM0, R_SUM0 + 1.4, a0, a1, FOUND, mid, CONC, n=1))
            conc.append(seg_block(R_SUM0 + 1.4, R_SUM1, a0, a1, FOUND, top, CONC, n=1))
    for u in VOMIT:                                   # the passages of the vomitoria: floor and dark mouths
        a = ang(R_SUM0, u); w = 1.0 / R_SUM0
        seats.append(seg_block(R_SUM0, R_SUM1 + 0.3, a - w, a + w, FOUND, Y_BALT2, SEAT_D, n=1))
        dark.append(seg_block(R_BALT2 - 0.05, R_BALT2 + 0.05, a - w, a + w, MED_TOP, Y_BALT2 - 0.05, DARK, n=1))
    # outer wall, standing a few metres above the street, broken top
    nn = 60
    us = np.linspace(0, 1, nn + 1)
    for j in range(nn):
        um = (us[j] + us[j + 1]) / 2
        if any(abs(um - v) < 0.012 for v in VOMIT):
            top = Y_BALT2 + 3.2                        # the lintel over the outer door of a vomitorium
            dark.append(seg_block(R_OUT - 0.02, R_OUT + 0.02, ang(R_OUT, us[j]), ang(R_OUT, us[j + 1]),
                                  Y_BALT2, Y_BALT2 + 2.4, DARK, n=1))
        else:
            top = 14.2 + 1.2 * math.sin(um * 23.0) * 0.5 + rng.uniform(-0.6, 0.6)
        stone.append(seg_block(R_SUM1, R_OUT, ang(R_OUT, us[j]), ang(R_OUT, us[j + 1]), FOUND, top, STONE, n=1))
    # the ends of the cavea (analemmata) on the line z = Z_PAR, from the orchestra to the outer wall
    prof = [(math.sqrt(R_ORCH ** 2 - Z_PAR ** 2), 0.0)]
    steps = [(R_ORCH + 0.8 * k, 0.25 * (k + 1)) for k in range(3)] + [(16.2, IMA_Y0)]
    steps += [(IMA_R0 + k * IMA_DR, IMA_Y0 + (k + 1) * IMA_DY) for k in range(IMA_N)]
    steps += [(R_BALT, Y_BALT)] + [(MED_R0 + k * MED_DR, Y_BALT + (k + 1) * MED_DY) for k in range(MED_N)]
    steps += [(R_BALT2, Y_BALT2), (R_SUM0, Y_BALT2 + 1.0), (R_SUM1, 14.0)]
    yprev = 0.0
    for r, yv in steps:
        x = math.sqrt(r * r - Z_PAR ** 2)
        prof += [(x, yprev), (x, yv)]
        yprev = yv
    xo = math.sqrt(R_OUT ** 2 - Z_PAR ** 2)
    prof += [(xo, yprev), (xo, FOUND), (prof[0][0], FOUND)]
    end = prism_xy(prof, Z_PAR - 1.2, Z_PAR, STONE)
    stone += both_x([end])
    # parodoi: the floors between the pulpitum and the cavea ends, their outer gates and the inscribed lintel doors
    for sx in (1, -1):
        x0, x1 = sorted((sx * 13.0, sx * (xo + 1.0)))
        seats.append(B(x0, x1, FOUND, 0.0, Z_PULP, Z_PAR - 1.2, SEAT_D))
        for k in range(6):                     # steps up from the gate to the street, raised in late Roman times
            xa = sx * (xo + 0.2 + 0.6 * k)
            x0, x1 = sorted((xa, xa + sx * 0.6))
            seats.append(B(x0, x1, FOUND, 0.4 * (k + 1), Z_PULP - 0.3, Z_PAR - 0.9, SEAT_D))
        xg = sx * (xo - 0.4)
        arch = [(-2.9, 0.0), (-1.6, 0.0), (-1.6, 3.4)] + \
            [(1.6 * math.cos(t), 3.4 + 1.6 * math.sin(t)) for t in np.linspace(math.pi, 0, 9)[1:-1]] + \
            [(1.6, 3.4), (1.6, 0.0), (2.9, 0.0), (2.9, 6.6), (-2.9, 6.6)]
        g = prism_xy(arch, -0.6, 0.6, STONE).R(0, math.pi / 2, 0).T(xg, 0, (Z_PULP + Z_PAR) / 2 - 0.3)
        stone.append(g)
        stone.append(B(min(xg, xg + sx * 0.1), max(xg, xg + sx * 0.1), 6.6, 7.4, Z_PULP - 0.5, Z_PAR, STONE_D))
    ms = [tint(merge(seats), 0.06, 0.4, 1), tint(merge(light), 0.04, 0.4, 2), tint(merge(stone), 0.1, 0.3, 3),
          tint(merge(conc), 0.12, 0.25, 4), merge(dark)]
    return merge(ms)


# ── Orchestra ────────────────────────────────────────────────────────────────
def orchestra():
    ms = []
    # the solid floor below the paving
    a0 = math.asin(Z_PULP / R_ORCH)
    ring = [(R_ORCH * math.cos(a), R_ORCH * math.sin(a)) for a in np.linspace(a0, math.pi - a0, 40)]
    ms.append(extrude(ring, FOUND, -0.02, SAND))
    # marble slabs, white and bluish, laid in rows parallel to the stage
    tw, td = 1.5, 1.0
    for iz in range(int(R_ORCH / td) + 1):
        z0 = Z_PULP + iz * td
        off = (iz % 2) * tw / 2
        for ix in range(-12, 12):
            x0 = ix * tw + off
            cx, cz = x0 + tw / 2, z0 + td / 2
            if math.hypot(cx, cz) > R_ORCH - 0.4 or cz < Z_PULP:
                continue
            c = ORCH_A if rng.random() < 0.62 else ORCH_B
            c = tuple(np.clip(np.array(c) * (0.95 + 0.08 * rng.random()), 0, 1))
            ms.append(B(x0 + 0.03, x0 + tw - 0.03, -0.02, 0.0, z0 + 0.03, z0 + td - 0.03, c))
    return merge(ms)


# ── Pulpitum and scaenae frons ───────────────────────────────────────────────
def pulpitum():
    ms = []
    # plan of the front with its niches (centre part), as an outline in (x, z)
    front = [(-30.0, Z_PULP)]
    xs = [-13.0, -9.6, -6.4, -3.2, 3.2, 6.4, 9.6, 13.0]
    kinds = ["rect", "semi", "rect", "semi", "rect", "semi", "rect"]
    for (xa, xb), kd in zip(zip(xs[:-1], xs[1:]), kinds):
        w = xb - xa
        m = (xa + xb) / 2
        if kd == "rect":
            hw = min(1.2, w / 2 - 0.4)
            front += [(m - hw, Z_PULP), (m - hw, Z_PULP - 0.8), (m + hw, Z_PULP - 0.8), (m + hw, Z_PULP)]
        else:
            r = min(1.1, w / 2 - 0.35)
            front += [(m - r * math.cos(t), Z_PULP - r * math.sin(t)) for t in np.linspace(0, math.pi, 9)]
    front += [(30.0, Z_PULP), (30.0, Z_FRONS), (-30.0, Z_FRONS)]
    outline = front
    ms.append(tint(extrude(outline, FOUND, Y_STAGE, STONE, top_c=STAGE), 0.08, 0.3, 5))
    ms.append(B(-30.0, 30.0, Y_STAGE - 0.25, Y_STAGE, Z_PULP - 0.02, Z_PULP + 0.15, MARBLE))     # marble cornice
    # small stairs at both ends from the orchestra to the stage
    for sx in (1, -1):
        for k in range(4):
            x0 = sx * 13.4
            ms.append(B(min(x0, x0 + sx * 1.2), max(x0, x0 + sx * 1.2), 0.0, Y_STAGE * (k + 1) / 4,
                        Z_PULP + 0.9 - k * 0.3, Z_PULP + 1.2 - k * 0.3, SEAT))
    return merge(ms)


def column(x, z, y0, h, r, cap=0.7, base=0.4):
    """Corinthian column: white base, blue-veined shaft, white capital with its abacus."""
    out = [cyl(r * 1.35, r * 1.25, base, MARBLE, seg=10).T(x, y0, z)]
    out.append(cyl(r, r * 0.86, h - base - cap, VEIN, seg=10).T(x, y0 + base, z))
    out.append(cyl(r * 0.88, r * 1.35, cap - 0.14, MARBLE, seg=10).T(x, y0 + h - cap, z))
    out.append(B(x - r * 1.45, x + r * 1.45, y0 + h - 0.14, y0 + h, z - r * 1.45, z + r * 1.45, MARBLE))
    return out


def statue(x, z, y, s=1.0, seed=0):
    rr = np.random.default_rng(seed)
    prof = [(0.0, 0.0), (0.30, 0.0), (0.33, 0.25), (0.27, 0.9), (0.30, 1.30), (0.24, 1.50), (0.10, 1.56),
            (0.0, 1.57)]
    m = [B(x - 0.4, x + 0.4, y, y + 0.55, z - 0.35, z + 0.35, MARBLE)]
    body = lathe(prof, STATUE, 10).S(s, s, s * 0.75).T(x, y + 0.55, z)
    m.append(body)
    if rr.random() < 0.7:                                   # some are headless
        m.append(sphere(0.13 * s, STATUE).T(x, y + 0.55 + 1.70 * s, z))
    return m


LOW_X = [2.4, 4.6, 7.0, 9.4, 11.8, 14.2, 16.0, 19.0, 21.4, 23.8, 26.2, 28.6]
Y_POD = Y_STAGE + 2.0
Y_LOWTOP = Y_POD + 6.4
Y_ENT1 = Y_LOWTOP + 1.4
Y_PED = Y_ENT1 + 1.0
Y_UPTOP = Y_PED + 4.9
Y_ENT2 = Y_UPTOP + 1.3


def frons():
    wall, white, cols, dark, stat = [], [], [], [], []
    # the frons wall, in pieces between the doors; the central exedra and the two rectangular recesses
    for sx in (1, -1):
        def X(a, b):
            return sorted((sx * a, sx * b))
        # ruined top of the masonry: high on the north-east half (+x), low on the south-west end
        hi = Y_ENT2 + 0.6 if sx > 0 else Y_ENT1 + 1.2
        for xa, xb, top in ((4.4, 16.2, Y_ENT2 + 0.9 if sx > 0 else Y_ENT1 + 2.5),
                            (18.8, 31.6, hi)):
            n = max(2, int((xb - xa) / 2.0))
            for k in range(n):
                a, b = xa + (xb - xa) * k / n, xa + (xb - xa) * (k + 1) / n
                t = top + rng.uniform(-0.8, 0.5)
                x0, x1 = X(a, b)
                wall.append(B(x0, x1, FOUND, t, Z_FRONS - 2.2, Z_FRONS, STONE))
        # recess of the valva hospitalis: deeper wall, the door, the lintel
        x0, x1 = X(16.2, 18.8)
        wall.append(B(x0, x1, FOUND, Y_ENT1, Z_FRONS - 3.2, Z_FRONS - 2.2, STONE))
        dark.append(B(x0 + 0.6, x1 - 0.6, Y_STAGE, Y_STAGE + 4.2, Z_FRONS - 2.25, Z_FRONS - 2.15, DARK))
        wall.append(B(x0, x1, Y_STAGE + 4.2, hi - 1.0, Z_FRONS - 2.2, Z_FRONS - 1.4, STONE))
        # the exedra: curved wall from x = ±4.4 back to the valva regia
    for k in range(12):
        a0, a1 = math.pi * k / 12, math.pi * (k + 1) / 12
        if 5 <= k <= 6:
            continue                                         # the valva regia
        top = Y_ENT2 + rng.uniform(-0.5, 0.6) if k < 6 else Y_ENT1 + 3.5 + rng.uniform(-0.6, 0.4)
        seg = seg_block(4.4, 6.6, a0 + math.pi, a1 + math.pi, FOUND, top, STONE, n=1)
        seg.v[:, 2] = seg.v[:, 2] * 0.75 + Z_FRONS
        wall.append(seg)
    dark.append(B(-1.5, 1.5, Y_STAGE, Y_STAGE + 5.4, Z_FRONS - 3.4, Z_FRONS - 3.3, DARK))
    wall.append(B(-1.9, 1.9, Y_STAGE + 5.4, Y_ENT1 + 1.5, Z_FRONS - 5.0, Z_FRONS - 3.3, STONE))
    # podium of the columns (with marble base and cornice), cut at the three doors
    for sx in (1, -1):
        for xa, xb in ((2.0, 15.6), (19.4, 30.0)):
            x0, x1 = sorted((sx * xa, sx * xb))
            wall.append(B(x0, x1, Y_STAGE, Y_POD - 0.25, Z_FRONS, Z_POD, STONE_D))
            white.append(B(x0, x1, Y_STAGE, Y_STAGE + 0.3, Z_FRONS, Z_POD + 0.1, MARBLE))
            white.append(B(x0, x1, Y_POD - 0.25, Y_POD, Z_FRONS, Z_POD + 0.12, MARBLE))
        # steps up to the hospitalia doors
        for k in range(3):
            x0, x1 = sorted((sx * 15.8, sx * 19.2))
            wall.append(B(x0, x1, Y_STAGE, Y_STAGE + 0.25 * (k + 1), Z_FRONS - 2.2 + 0.0,
                          Z_POD - 0.6 * k, STONE_D))
    for k in range(3):                                        # steps up to the valva regia
        wall.append(B(-2.0, 2.0, Y_STAGE, Y_STAGE + 0.25 * (k + 1), Z_FRONS - 3.3, Z_POD - 0.6 * k, STONE_D))
    # lower order: front row on the podium and pilasters against the wall
    for sx in (1, -1):
        for x in LOW_X:
            cols += column(sx * x, Z_POD - 0.6, Y_POD, 6.4, 0.38)
            white.append(B(sx * x - 0.35, sx * x + 0.35, Y_POD, Y_LOWTOP, Z_FRONS - 0.02, Z_FRONS + 0.25, MARBLE))
        # entablature of the lower order, over the columns and back to the wall
        for xa, xb in ((1.9, 15.7), (15.7, 19.3), (19.3, 29.2)):
            x0, x1 = sorted((sx * xa, sx * xb))
            zf = Z_POD - 0.1 if xa != 15.7 else Z_FRONS + 0.2
            white.append(B(x0, x1, Y_LOWTOP, Y_ENT1 - 0.45, Z_FRONS, zf, MARBLE))
            white.append(B(x0, x1, Y_ENT1 - 0.45, Y_ENT1, Z_FRONS, zf + 0.35, MARBLE))
        # statues between the lower columns
        for x in (3.5, 8.2, 13.0, 20.2, 25.0):
            stat += statue(sx * x, Z_POD - 0.9, Y_POD, 1.15, seed=int(500 + sx * x * 3))
    white.append(B(-1.9, 1.9, Y_LOWTOP, Y_ENT1, Z_FRONS - 3.3, Z_FRONS, MARBLE))      # lintel over the regia
    # upper order
    for sx in (1, -1):
        for x in LOW_X:
            if x < 4.0:
                continue
            free = sx < 0 and x > 16.5                       # south-west end: shafts without entablature
            yb = Y_PED
            white.append(B(sx * x - 0.45, sx * x + 0.45, Y_ENT1, Y_PED, Z_POD - 1.05, Z_POD - 0.15, MARBLE))
            cols += column(sx * x, Z_POD - 0.6, yb, 4.9 if not free else 4.9 - (0.0 if x % 2 < 1 else 0.9), 0.31,
                           cap=0.55, base=0.3)
        # pedestal course
        for xa, xb in ((4.0, 15.7), (16.2, 29.2)):
            x0, x1 = sorted((sx * xa, sx * xb))
            white.append(B(x0, x1, Y_ENT1, Y_PED - 0.2, Z_FRONS, Z_POD - 1.1, MARBLE))
        spans = [(3.9, 15.7)] if sx < 0 else [(3.9, 15.7), (15.7, 21.6), (21.6, 29.0)]
        for xa, xb in spans:
            x0, x1 = sorted((sx * xa, sx * xb))
            yt = Y_ENT2 if xa != 21.6 else Y_ENT2 - 0.2
            white.append(B(x0, x1, Y_UPTOP, yt - 0.4, Z_FRONS, Z_POD - 0.2, MARBLE))
            white.append(B(x0, x1, yt - 0.4, yt, Z_FRONS, Z_POD + 0.1, MARBLE))
        if sx > 0:                                           # the broken pediment over the north-east bay
            for side in (-1, 1):
                xm = 18.65
                p = [(0.5, 0.0), (2.9, 0.0), (0.9, 1.3), (0.5, 1.3)]
                pm = prism_xy([(xm + side * a, Y_ENT2 + b) for a, b in p] if side > 0 else
                              [(xm - a, Y_ENT2 + b) for a, b in p[::-1]], Z_POD - 1.0, Z_POD + 0.1, MARBLE)
                white.append(pm)
    # balcony rail over the valva regia (the stage lights rail of the festival is left out)
    white.append(B(-3.9, 3.9, Y_ENT1, Y_ENT1 + 0.25, Z_FRONS - 1.0, Z_POD - 0.2, MARBLE))
    return merge([tint(merge(wall), 0.12, 0.3, 11), tint(merge(white), 0.03, 0.5, 12),
                  tint(merge(cols), 0.04, 0.6, 13), merge(dark), merge(stat)])


def scaena_building():
    ms = []
    # floor of the postscaenium and the passage to the garden
    ms.append(B(-X_SCAENA, X_SCAENA, FOUND, 0.6, Z_BACK, Z_FRONS - 2.2, SAND))
    # back wall with the central door, end walls and the cross walls of the rooms, ruined to different heights
    for sx in (1, -1):
        for xa, xb in ((2.2, 10.0), (10.0, 20.0), (20.0, X_SCAENA)):
            x0, x1 = sorted((sx * xa, sx * xb))
            ms.append(B(x0, x1, FOUND, 0.6 + rng.uniform(4.0, 7.5), Z_BACK, Z_BACK + 1.4, STONE))
        x0, x1 = sorted((sx * (X_SCAENA - 1.4), sx * X_SCAENA))
        ms.append(B(x0, x1, FOUND, 9.0, Z_BACK, Z_FRONS - 2.2, STONE))
        for x in (7.0, 13.0, 24.0):
            x0, x1 = sorted((sx * x - 0.45, sx * x + 0.45))
            ms.append(B(x0, x1, 0.6, 0.6 + rng.uniform(2.5, 5.0), Z_BACK + 1.4, Z_FRONS - 2.2, STONE_D))
        # the side wings reach the cavea ends (versurae)
        x0, x1 = sorted((sx * 30.0, sx * X_SCAENA))
        ms.append(B(x0, x1, FOUND, Y_STAGE + 3.0, Z_FRONS - 2.2, Z_PULP - 1.0, STONE))
    ms.append(B(-2.2, 2.2, FOUND, 0.6, Z_BACK, Z_BACK + 1.4, SAND))
    ms.append(B(-2.2, 2.2, 5.0, 7.2, Z_BACK, Z_BACK + 1.4, STONE))
    return tint(merge(ms), 0.1, 0.3, 21)


# ── Porticus post scaenam ────────────────────────────────────────────────────
Y_G = 0.6
GARDEN = {   # OSM ways, in model metres (x, z)
    "bed_w_out": [(-3.6, -58.9), (-3.4, -56.5), (-24.5, -56.2), (-24.6, -26.3), (-28.1, -26.4), (-27.8, -59.0)],
    "bed_e_out": [(26.5, -27.7), (25.8, -59.1), (3.6, -58.8), (3.3, -56.5), (23.0, -56.5), (24.0, -27.6)],
    "bed_w_in": [(-4.0, -52.8), (-3.3, -52.1), (-3.3, -51.2), (-3.9, -50.6), (-18.3, -50.5), (-18.6, -28.4),
                 (-12.8, -28.0), (-12.3, -27.1), (-12.8, -26.3), (-20.9, -26.4), (-20.4, -52.9)],
    "bed_e_in": [(3.0, -53.2), (20.2, -53.4), (20.9, -26.8), (12.5, -26.6), (12.0, -27.5), (12.6, -28.3),
                 (13.6, -28.4), (18.8, -28.7), (18.0, -51.3), (3.0, -51.6), (2.6, -52.2), (2.7, -52.9)],
    "lawn_e": [(4.0, -49.8), (17.0, -50.0), (17.5, -29.7), (14.8, -29.8), (14.7, -30.7), (14.4, -31.4),
               (13.5, -31.7), (12.4, -31.5), (11.9, -30.7), (11.8, -30.0), (3.4, -29.5), (3.7, -35.3),
               (5.8, -36.2), (7.4, -37.8), (7.8, -40.0), (6.7, -42.2), (5.2, -43.7), (4.0, -43.8)],
    "lawn_w": [(-3.3, -49.5), (-17.1, -49.5), (-17.4, -29.5), (-11.9, -29.7), (-11.4, -31.0), (-10.9, -31.8),
               (-9.9, -32.2), (-8.7, -31.5), (-8.1, -30.6), (-8.0, -29.4), (-3.0, -29.8), (-3.0, -35.4),
               (-5.7, -35.7), (-7.2, -36.6), (-8.1, -38.6), (-7.9, -40.5), (-7.2, -42.3), (-6.3, -43.2),
               (-4.7, -43.4), (-3.7, -43.4)],
}
GX0, GX1, GZ0, GZ1 = -37.0, 37.0, -69.4, Z_BACK       # the open ground of the porticus


def tree(x, z, y, h, crown, seed):
    rr = np.random.default_rng(seed)
    out = [tube([(x, y - 0.5, z), (x + rr.uniform(-0.3, 0.3), y + h * 0.5, z + rr.uniform(-0.3, 0.3))],
                0.22, WOOD, 6)]
    for k in range(4):
        cx = x + rr.uniform(-0.35, 0.35) * crown
        cz = z + rr.uniform(-0.35, 0.35) * crown
        cy = y + h * rr.uniform(0.55, 0.8)
        m = sphere(crown * rr.uniform(0.45, 0.6), LEAF, 9, 5).S(1, 0.85, 1)
        crumple(m, crown * 0.12, 1.2 / crown, seed + k)
        m.T(cx, cy, cz)
        out.append(tint(m, 0.15, 0.6, seed + k))
    return out


def porticus():
    ground, green, stone, white, other = [], [], [], [], []
    ground.append(B(GX0, GX1, FOUND, Y_G, GZ0, GZ1, SAND))
    for k in ("bed_w_out", "bed_e_out", "bed_w_in", "bed_e_in"):
        green.append(extrude(GARDEN[k], Y_G, Y_G + 0.9, HEDGE))
    for k in ("lawn_e", "lawn_w"):
        green.append(extrude(GARDEN[k], Y_G, Y_G + 0.08, GRASS))
    # the round basin in the middle (OSM way 260417193)
    other.append(cyl(3.05, 3.05, 0.55, SEAT, seg=24).T(0, Y_G, -40.0))
    other.append(cyl(2.7, 2.7, 0.02, WATER, seg=24, caps=True).T(0, Y_G + 0.55, -40.0))
    # the double colonnade: stumps of stuccoed granite columns (some keep their brick-coloured stucco)
    def stump(x, z, h):
        c = BRICK if rng.random() < 0.35 else SEAT
        return [B(x - 0.45, x + 0.45, Y_G, Y_G + 0.3, z - 0.45, z + 0.45, SEAT_D),
                cyl(0.3, 0.28, h, c, seg=8).T(x, Y_G + 0.3, z)]
    rows = []
    for d in (0.0, 4.0):
        xe, ze = 30.5 + d, -63.0 - d
        for z in np.arange(-26.0, ze, -3.6):
            rows.append((-xe, z))
        for x in np.arange(-xe, 27.5, 3.6):
            if d == 0 or abs(x) > 6.0:
                rows.append((x, ze))
    for x, z in rows:
        h = float(rng.choice([0.5, 0.9, 1.4, 2.0, 2.6, 3.6], p=[0.2, 0.25, 0.2, 0.15, 0.12, 0.08]))
        stone += stump(x, z, h)
    # the rebuilt stretch on the north-east side, with its planted roof
    for z in np.arange(-27.5, -62.0, -3.6):
        for x in (29.0, 33.0):
            white.append(B(x - 0.45, x + 0.45, Y_G, Y_G + 0.3, z - 0.45, z + 0.45, SEAT_D))
            stone.append(cyl(0.3, 0.27, 3.6, SEAT, seg=8).T(x, Y_G + 0.3, z))
    other.append(B(28.2, 33.8, Y_G + 3.9, Y_G + 4.3, -62.6, -26.6, STONE_D))
    other.append(B(28.4, 33.6, Y_G + 4.3, Y_G + 4.45, -62.4, -26.8, ROOF_G))
    # the back wall of the precinct, retaining the higher ground behind the excavation, and the hall of the
    # imperial cult on the axis
    for xa, xb in ((-37.0, -5.5), (5.5, 37.0)):
        stone.append(B(xa, xb, FOUND, Y_G + rng.uniform(3.2, 3.8), -70.2, -69.4, STONE))
    for x0, x1, z0, z1, h in ((-5.5, -4.6, -76.0, -69.4, 3.8), (4.6, 5.5, -76.0, -69.4, 3.6),
                              (-5.5, 5.5, -76.0, -75.1, 4.0)):
        stone.append(B(x0, x1, FOUND, Y_G + h, z0, z1, STONE))
    ground.append(B(-4.6, 4.6, FOUND, Y_G + 0.15, -75.1, -69.4, SEAT_D))
    # trees in the garden (the ones seen in the orthophoto)
    for i, (x, z, h, c) in enumerate(((-9.0, -46.0, 7.0, 4.5), (-14.0, -33.0, 6.5, 4.0), (10.0, -47.0, 7.5, 4.5),
                                      (13.5, -36.0, 6.0, 3.8), (-1.0, -55.0, 5.5, 3.2), (6.0, -55.5, 5.5, 3.0),
                                      (-22.5, -46.0, 6.0, 3.5), (22.0, -40.0, 6.0, 3.5))):
        other += tree(x, z, Y_G, h, c, 300 + 17 * i)
    return merge([tint(merge(ground), 0.06, 0.2, 31), tint(merge(green), 0.12, 0.6, 32),
                  tint(merge(stone), 0.1, 0.5, 33), merge(white), merge(other)])


def main():
    out = ROOT / "models" / "Q2700544"
    out.mkdir(parents=True, exist_ok=True)
    m = merge([cavea(), orchestra(), pulpitum(), frons(), scaena_building(), porticus()])
    tris, size = write_glb(out / "model.glb", [("teatro_romano_de_merida", m)], roughness=0.9)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
