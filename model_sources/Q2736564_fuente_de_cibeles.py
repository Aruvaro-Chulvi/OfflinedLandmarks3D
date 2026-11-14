"""
Fuente de Cibeles (Q2736564), Plaza de Cibeles, Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: designed by Ventura Rodríguez (1717–1785), 1777–1782; goddess and chariot by Francisco Gutiérrez
(died 1782), lions by Roberto Michel (died 1786); the two putti at the back were added when the fountain was
moved to the centre of the square in 1895 (Miguel Ángel Trilles, Antonio Parera). Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap relation 6345116 (the fountain, Q2736564) and its two inner rings (relations 12982518,
  18462208), all circles round the same centre: outer edge r 14.82 m, the lower pool to r 11.43, a stepped
  ring to r 9.87, the upper basin inside. The rock with the group is way 1346140195 (height 8 m in OSM),
  13.5 × 8.9 m, long axis at 82.9°/262.9°.
- The goddess looks down the first stretch of the calle de Alcalá, towards the Puerta del Sol (es.wikipedia),
  i.e. west: heading 262.9° (the long axis of the rock). Checked on photos from the Palacio de Cibeles
  mirador (Commons «008495 - Madrid (9416183413).jpg») and from the plaza: the lions point west.
- The group: 5.5 m high, 4.7 m wide, 12.5 m long (es.wikipedia). Goddess seated on a chariot with large
  wheels, turreted crown, sceptre (right hand) and key (left hand); two lions abreast pulling it; a mask
  spouting water under the lions; two putti at the back (one with an amphora, one with a shell).
- Grey Montesclaros marble for the goddess and the lions, Redueña stone for the rest.
Sculpture is stylised: volumes and silhouettes, as seen from the air or across the square.

Axes (docs/MODEL_SPEC.md): +Y up, the goddess faces +Z, origin on the ground at the centroid of the fountain.

    python model_sources/Q2736564_fuente_de_cibeles.py   -> models/Q2736564/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import GLASS, Mesh, merge, box, cyl, lathe, tube, extrude, offset, write_glb, fbm  # noqa: E402

rng = np.random.default_rng(1782)

STONE = np.array((0.78, 0.76, 0.71))      # Redueña stone: curbs, steps, rims
STONE_D = np.array((0.64, 0.62, 0.57))
ROCK = np.array((0.60, 0.575, 0.52))      # the rock under the group
MARBLE = np.array((0.70, 0.665, 0.62))    # Montesclaros marble, weathered grey
MARBLE_D = np.array((0.58, 0.55, 0.51))
WATER = (0.42, 0.55, 0.60)
FOUND = -3.0

# Rings (model metres, from OSM): centre, outer edge, lower pool edge, upper basin edge
CX, CZ = -0.11, 0.31
R_OUT, R_POOL, R_UP = 14.82, 11.43, 9.87

# way 1346140195 (the rock), in model x/z with heading 262.9 and the same origin
ROCK_XZ = [(-1.199, 6.658), (-2.31, 5.785), (-3.419, 4.269), (-4.274, 1.341), (-4.274, -0.994), (-4.242, -2.778),
           (-3.422, -4.865), (-1.715, -6.175), (-0.64, -6.451), (0.878, -6.331), (2.154, -5.881), (3.201, -4.947),
           (3.981, -3.481), (4.486, -1.879), (4.663, 0.564), (4.449, 2.555), (4.235, 4.0), (3.631, 5.259),
           (2.84, 6.127), (1.514, 6.979), (0.581, 7.017)]


def jit(c, s=0.02):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, s)), 0, 1))


def sphere(r, c, seg=14, rings=8):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def ell(rx, ry, rz, c, seg=14, rings=8):
    return sphere(1.0, c, seg, rings).S(rx, ry, rz)


def limb(p0, p1, r0, r1, c, seg=8):
    """A tapered cylinder between two points (legs, arms)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    m = cyl(r0, r1, L, c, seg)
    y = np.array([0.0, 1.0, 0.0]); t = d / L
    v = np.cross(y, t); s = np.linalg.norm(v); cth = float(np.dot(y, t))
    if s > 1e-9:
        k = v / s; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rm = np.eye(3) + math.sin(math.atan2(s, cth)) * K + (1 - cth) * K @ K
        m.v = m.v @ Rm.T
    elif cth < 0:
        m.v[:, 1] *= -1; m.f = m.f[:, ::-1]
    return m.T(*p0)


def revolve(prof, c, seg=72):
    """A ring of revolution round the fountain centre, one lathe per profile segment so the edges stay sharp.
    prof runs up the outside, inwards across the tops and down the inside (normals face out of the stone)."""
    out = []
    for a, b in zip(prof[:-1], prof[1:]):
        out.append(lathe([a, b], jit(c, 0.01), seg))
    return merge(out).T(CX, 0, CZ)


def disc(r, y, c, seg=72, r_in=0.0):
    return lathe([(r, y), (r_in, y)], c, seg).T(CX, 0, CZ)


def mirror_x(m):
    n = m.copy(); n.v[:, 0] *= -1; n.f = n.f[:, ::-1]; return n


# ── Basin ────────────────────────────────────────────────────────────────────
def basin():
    parts = []
    # outer curb, with its moulded top
    parts.append(revolve([(R_OUT, FOUND), (R_OUT, 0.32), (R_OUT - 0.08, 0.32), (R_OUT - 0.08, 0.46),
                          (R_OUT - 0.62, 0.46), (R_OUT - 0.62, 0.0)], STONE))
    # lower pool: water just below the curb
    parts.append(disc(R_OUT - 0.62, 0.26, WATER, r_in=R_POOL))
    # the stepped ring up to the upper basin (three steps and the rim)
    steps = [(R_POOL, -0.2), (R_POOL, 0.55), (R_POOL - 0.42, 0.55), (R_POOL - 0.42, 0.92), (R_POOL - 0.82, 0.92),
             (R_POOL - 0.82, 1.28), (R_UP + 0.62, 1.28), (R_UP + 0.62, 1.58), (R_UP, 1.58), (R_UP, 1.0)]
    parts.append(revolve(steps, STONE_D * 1.1))
    # a lighter band on the rim top
    parts.append(revolve([(R_UP + 0.66, 1.42), (R_UP + 0.66, 1.5)], STONE * 1.05))
    # upper basin water
    parts.append(disc(R_UP, 1.36, WATER))
    return parts


# ── The rock ─────────────────────────────────────────────────────────────────
def rock():
    out = []
    base = extrude(ROCK_XZ, 0.9, 1.5, jit(ROCK))
    out.append(base)
    # rough lumps along the outline and in the middle, rising to the plinths of the lions and the chariot
    P = np.array(ROCK_XZ)
    for i in range(0, len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        for t in (0.25, 0.75):
            p = a + (b - a) * t
            q = p * 0.82
            r = 0.75 + rng.uniform(0, 0.35)
            out.append(ell(r, 0.55 + rng.uniform(0, 0.3), r * 1.1, jit(ROCK, 0.05), 10, 6)
                       .R(0, rng.uniform(0, 3), 0).T(q[0], 1.55, q[1]))
    for x, y, z, rx, ry, rz in ((0, 1.9, 0.0, 2.6, 0.9, 4.6), (0, 2.0, -3.6, 2.2, 0.75, 2.2),
                                 (0, 1.95, 3.6, 2.5, 0.6, 2.4), (-1.5, 2.0, 1.5, 1.4, 0.7, 2.0),
                                 (1.5, 2.0, 1.5, 1.4, 0.7, 2.0)):
        out.append(ell(rx, ry, rz, jit(ROCK, 0.04), 16, 8).T(x, y, z))
    m = merge(out)
    # stone colour mottled with noise
    k = 0.88 + 0.22 * fbm(m.v, 0.9, 7, 3)
    m.c = np.clip(m.c * k[:, None], 0, 1)
    return [m]


def mask(x, y, z, c):
    """The water-spouting mask under the lions: a face with cheeks and an open mouth."""
    return [ell(0.42, 0.5, 0.22, c).T(x, y, z), ell(0.16, 0.12, 0.1, (0.25, 0.24, 0.22)).T(x, y - 0.18, z + 0.18),
            ell(0.55, 0.22, 0.18, c).T(x, y + 0.42, z - 0.04)]


# ── Lions ────────────────────────────────────────────────────────────────────
def lion(x, y, z, side, c, k=1.22):
    """A lion walking forward (+Z), head turned a little outwards; about 3.9 m long, 3.0 m to the top of the mane."""
    out = []
    cd = tuple(np.asarray(c) * 0.86)
    out.append(ell(0.66, 0.66, 1.4, c).R(-0.12, 0, 0).T(0, 1.25, -0.05))           # body
    out.append(ell(0.74, 0.8, 0.8, c).T(0, 1.45, 0.75))                              # chest
    out.append(ell(0.84, 1.0, 0.6, cd, 16, 8).R(0.5, 0, 0).T(0, 1.75, 1.1))          # mane, falling on the chest
    out.append(ell(0.8, 0.55, 0.55, cd, 14, 6).T(0, 2.25, 1.05))                     # its crest
    out.append(ell(0.42, 0.46, 0.5, c).T(0, 2.0, 1.6))                               # head
    out.append(ell(0.32, 0.28, 0.4, c).T(0, 1.84, 2.12))                             # muzzle
    out.append(ell(0.22, 0.08, 0.24, (0.3, 0.28, 0.26)).T(0, 1.66, 2.22))             # open jaw
    for s in (-1, 1):
        out.append(ell(0.1, 0.12, 0.08, c).T(s * 0.28, 2.42, 1.48))                  # ears
    # legs: the outer foreleg raised in the stride
    out.append(limb((side * 0.34, 1.3, 1.0), (side * 0.38, 0.0, 1.35), 0.3, 0.22, c))
    out.append(limb((-side * 0.34, 1.3, 1.0), (-side * 0.36, 0.75, 1.55), 0.3, 0.22, c))
    out.append(limb((-side * 0.36, 0.75, 1.55), (-side * 0.36, 0.35, 1.7), 0.22, 0.2, c))
    for s in (-1, 1):
        out.append(ell(0.4, 0.6, 0.55, c).T(s * 0.38, 1.05, -0.95))                  # haunches
        out.append(limb((s * 0.38, 0.8, -1.05), (s * 0.36, 0.0, -1.25), 0.27, 0.2, c))
        out.append(ell(0.2, 0.1, 0.3, c).T(s * 0.36, 0.06, -1.15))                   # paws
    out.append(ell(0.2, 0.1, 0.3, c).T(side * 0.38, 0.06, 1.45))
    out.append(tube([(0, 1.35, -1.35), (side * 0.25, 1.0, -1.75), (side * 0.45, 0.6, -1.9), (side * 0.5, 0.75, -2.15)],
                    0.07, c, 6))                                                     # tail
    m = merge(out).R(0, side * 0.08, 0).S(k)                                       # turned a little outwards
    return [m.T(x, y, z)]


# ── The chariot and the goddess ──────────────────────────────────────────────
def wheel(x, y, z, r, c):
    out = [tube([(x, y + r * math.sin(a), z + r * math.cos(a)) for a in np.linspace(0, 2 * math.pi, 33)], 0.16, c, 6),
           cyl(0.32, 0.26, 0.55, c, 12).R(0, 0, math.pi / 2).T(x + 0.28, y, z)]
    for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):
        out.append(tube([(x, y, z), (x, y + r * math.sin(a), z + r * math.cos(a))], 0.07, c, 5))
    return out


def chariot(y0, c):
    out = []
    cd = tuple(np.asarray(c) * 0.9)
    zc = -1.4                                                       # axle
    for s in (-1, 1):
        out += wheel(s * 1.85, y0 + 1.35, zc, 1.35, cd)
    out.append(box(3.4, 0.3, 0.34, cd, bottom=False).T(0, y0 + 1.35, zc))           # axle
    # the car: a scroll-ended shell, rising at the front
    body = [box(2.8, 1.5, 3.0, c).T(0, y0 + 0.75, zc + 0.1)]
    body.append(ell(1.45, 1.0, 0.8, c, 16, 8).T(0, y0 + 1.9, zc + 1.45))            # rounded front (the shell)
    body.append(cyl(0.5, 0.5, 3.04, cd, 12).R(0, 0, math.pi / 2).T(1.52, y0 + 2.35, zc - 1.3))  # back scroll
    body.append(box(2.9, 0.2, 3.2, cd).T(0, y0 + 2.25, zc + 0.05))                  # rim of the car
    out += body
    # draught pole to the lions
    out.append(tube([(0, y0 + 1.3, zc + 2.0), (0, y0 + 1.5, zc + 3.8), (0, y0 + 1.9, zc + 4.6)], 0.13, cd, 6))
    return out


def goddess(y0, c):
    """Cybele seated on her throne on the car: robe, mantle, turreted crown, sceptre and key."""
    out = []
    zc = -1.5
    y0 = y0 - 0.25
    cd = tuple(np.asarray(c) * 0.9)
    out.append(box(1.9, 1.5, 1.4, cd).T(0, y0 + 1.95, zc - 0.25))                  # throne
    out.append(box(1.8, 1.0, 0.3, cd).T(0, y0 + 3.4, zc - 0.85))                   # its back
    # seated figure: lap and knees forward, draped legs down to the footstool
    out.append(ell(0.95, 0.55, 1.0, c).T(0, y0 + 3.55, zc + 0.15))                 # lap and thighs
    out.append(box(1.4, 1.5, 0.75, c).T(0, y0 + 1.95, zc + 0.95))                  # draped shins
    out.append(lathe([(0.9, 0.0), (0.88, 0.35), (0.72, 0.9), (0.74, 1.3), (0.5, 1.55), (0.2, 1.66), (0, 1.67)], c, 14)
               .S(1, 1, 0.72).T(0, y0 + 3.6, zc - 0.25))                            # torso and mantle
    out.append(ell(0.27, 0.33, 0.3, c).T(0, y0 + 5.52, zc - 0.12))                 # head
    out.append(cyl(0.32, 0.36, 0.48, cd, 10).T(0, y0 + 5.72, zc - 0.12))           # turreted crown
    for a in np.linspace(0, 2 * math.pi, 6, endpoint=False):
        out.append(box(0.15, 0.16, 0.15, cd).T(0.34 * math.cos(a), y0 + 6.2, zc - 0.12 + 0.34 * math.sin(a)))
    # right arm (her right is −X) holding the sceptre, left arm holding the key out
    out.append(limb((-0.62, y0 + 4.9, zc - 0.2), (-0.85, y0 + 4.2, zc + 0.25), 0.15, 0.12, c))
    out.append(limb((-0.85, y0 + 4.2, zc + 0.25), (-0.85, y0 + 4.55, zc + 0.75), 0.12, 0.1, c))
    out.append(limb((-0.85, y0 + 3.4, zc + 0.75), (-0.85, y0 + 6.55, zc + 0.75), 0.05, 0.04, cd))   # sceptre
    out.append(ell(0.1, 0.14, 0.1, cd).T(-0.85, y0 + 6.62, zc + 0.75))
    out.append(limb((0.62, y0 + 4.9, zc - 0.2), (0.85, y0 + 4.25, zc + 0.3), 0.15, 0.12, c))
    out.append(limb((0.85, y0 + 4.25, zc + 0.3), (0.82, y0 + 4.2, zc + 0.95), 0.12, 0.1, c))
    out.append(box(0.08, 0.08, 0.6, cd).T(0.82, y0 + 4.16, zc + 1.25))                              # key
    out.append(box(0.1, 0.24, 0.06, cd).T(0.82, y0 + 4.05, zc + 1.52))
    return out


def putto(x, y, z, c, holds):
    """A child (about 1.4 m before scaling), seated on the rock at the back, with an amphora or a shell."""
    out = [ell(0.32, 0.42, 0.3, c).T(x, y + 0.55, z), ell(0.2, 0.22, 0.2, c).T(x, y + 1.15, z),
           ell(0.34, 0.2, 0.42, c).T(x, y + 0.2, z - 0.1)]
    if holds == "amphora":
        out.append(lathe([(0.0, 0.0), (0.18, 0.05), (0.26, 0.3), (0.2, 0.55), (0.09, 0.65), (0.12, 0.75), (0.0, 0.76)], c, 10)
                   .R(0.0, 0.0, 1.1).T(x + 0.15, y + 0.75, z - 0.35))
    else:
        out.append(cyl(0.42, 0.42, 0.08, c, 12).R(1.2, 0, 0).T(x, y + 0.9, z - 0.25))
    return out


def group():
    parts = []
    YR = 1.95                                                      # top of the rock where the group stands
    parts += lion(-1.08, YR, 3.15, -1, jit(MARBLE))
    parts += lion(1.08, YR, 3.15, 1, jit(MARBLE))
    parts += [box(4.2, 0.45, 5.4, jit(ROCK)).T(0, YR - 0.4, 3.0)]     # the lions' ledge
    parts += chariot(YR, jit(MARBLE * 0.97))
    parts += goddess(YR, jit(MARBLE))
    parts.append(merge(putto(0, 0, 0, jit(MARBLE), "amphora")).S(1.35).R(0, math.pi, 0).T(-0.95, YR - 0.2, -4.3))
    parts.append(merge(putto(0, 0, 0, jit(MARBLE), "shell")).S(1.35).R(0, math.pi, 0).T(0.95, YR - 0.2, -4.3))
    parts += mask(0.0, 1.85, 6.15, jit(MARBLE_D))
    return parts


def main():
    out = ROOT / "models" / "Q2736564"
    out.mkdir(parents=True, exist_ok=True)
    b = basin()
    stone = merge([b[0], b[2], b[3]] + rock() + group())
    water = merge([b[1], b[4]])
    tris, size = write_glb(out / "model.glb", [("fuente_de_cibeles", stone), ("water", water, {"material": "glass"})],
                           roughness=0.9, materials={"glass": GLASS})
    allv = np.vstack([stone.v, water.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
