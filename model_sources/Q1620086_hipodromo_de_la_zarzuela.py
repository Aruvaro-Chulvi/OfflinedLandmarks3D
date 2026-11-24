"""
Hipódromo de la Zarzuela (Q1620086), Madrid — the grandstands with Torroja's cantilevered shells, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: architects Carlos Arniches Moltó (died 1958) and Martín Domínguez Esteban (died 1970), engineer
Eduardo Torroja Miret (died 1961); works 1935–1936, finished and opened 1941; rehabilitated 2004–2016.
Bien de Interés Cultural (2009). Protected in Spain until the end of 2050 (80 years after 1970, the rule for authors who died before 1987) -> "protected".

What is modelled:
- The three grandstands (Tribuna Norte, Central and Sur) with their roofs: thin reinforced-concrete shells, each
  module a piece of a hyperboloid that vaults upwards between two transverse frames, so the front edge is a row
  of arches. In section every frame is a «V»: a long cantilever over the stands rising towards the track, a rear
  arm over the upper gallery held down by steel ties, and a single column under the low point. The shell is
  thick at the column and very thin at the edges.
- Under each roof: the stepped concrete terraces, the upper gallery at the top, and the base building at the
  back (bars, betting booths) with its arcade facing the paddock; the round stair towers at the ends of each
  grandstand; low white blocks in the gaps between the grandstands and at the south end (as far as the OSM
  building goes). The white rail along the near edge of the turf course in front, so the place reads as a
  racecourse (not the track itself: the ground comes from the map).
- Not modelled: the paddock galleries, the restaurants and the other buildings of the complex, the gardens,
  the pelouse, the furniture, signs.

Measurements (see meta.json for the sources):
- OpenStreetMap way 24542681 (building=yes): the whole row of grandstands, 231.7 × 29.3 m, long side facing
  87.7° (the track). Its centroid is the origin.
- PNOA orthophoto (IGN, 0.2 m/px, looked at only): the three shells measure 59 × 26.5 m (north, 12 modules),
  30 × 28 m (central, 6 modules) and 60 × 29.6 m (south, 12 modules); their front edges at z = 11.0, 7.6 and
  12.8 m (the central one is set back), the round stair towers at their ends, and the rail of the turf course
  32–41 m in front.
- Frames every 5 m (CSIC, Informes de la Construcción); shell 6 cm at the edges and 65 cm at the columns, the
  cantilever up to 13 m (UPM/CSIC, Anales de Edificación 2018); the cantilever is usually given as 12.8 m.
  Heights measured on Wikimedia Commons photos («Hipódromo de la Zarzuela-1.JPG», from below,
  «Hipódromo de la Zarzuela-2.JPG», the profile of a shell from the end): underside at the column ≈ 11 m, front
  edge ≈ 13.6 m, rear edge ≈ 12.6 m, top of the terraces ≈ 7.5 m (±1 m).

Axes (docs/MODEL_SPEC.md): +Y up, the grandstands face the track: +Z points to 87.7° (east), +X to 357.7°
(north). Origin = centroid of OSM way 24542681, y = 0 = the pelouse / paddock level.

    python model_sources/Q1620086_hipodromo_de_la_zarzuela.py   -> models/Q1620086/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, triangulate, offset, write_glb  # noqa: E402

# OSM way 24542681 in model x/z (heading 87.7°, origin at its centroid 40.4665972, -3.7586329)
OUTLINE = [(-114.1, -6.3), (-111.2, 14.3), (-83.7, 11.2), (-83.5, 14.4), (-23.4, 10.0), (-23.5, 7.8), (1.1, 6.9), (0.8, 9.4),
           (31.9, 9.4), (31.7, 7.2), (56.6, 7.5), (56.4, 10.5), (116.7, 14.4), (117.5, -10.4), (59.8, -14.1), (59.1, -1.3),
           (56.2, 3.6), (32.0, 3.5), (32.0, -8.6), (30.3, -9.2), (30.2, -14.7), (-0.9, -14.9), (-0.7, -6.0), (-22.0, -6.0),
           (-26.2, -14.2), (-82.1, -10.8)]

# the three grandstands: (name, x of the south end, number of 5 m modules, z of the rear edge, z of the front edge)
STANDS = [("north", 56.9, 12, -15.5, 11.0),
          ("central", 0.2, 6, -20.4, 7.6),
          ("south", -85.0, 12, -16.8, 12.8)]
MOD = 5.0
CANT = 12.8                  # front cantilever, from the column to the front edge
Y_COL, Y_FRONT, Y_REAR = 11.0, 13.6, 12.6      # underside of the shell at the column, front and rear edges
T_EDGE, T_COL = 0.06, 0.65
Y_TERR = 7.5                 # top of the terraces / floor of the upper gallery
FOUND = -2.5

WHITE = (0.93, 0.92, 0.89)
SHELL_TOP = (0.90, 0.90, 0.88)
SHELL_UNDER = (0.80, 0.78, 0.73)
CONC = (0.74, 0.73, 0.70)
STEP = (0.80, 0.79, 0.76)
RISER = (0.62, 0.61, 0.58)
SEATS = (0.24, 0.42, 0.30)
DARK = (0.22, 0.21, 0.20)
GLASS = (0.20, 0.23, 0.26)
STEEL = (0.35, 0.36, 0.38)
RAIL = (0.96, 0.96, 0.95)


def prism_x(prof, x0, x1, cols):
    """Extrude a cross-section polygon prof = [(z, y), …] from x0 to x1. cols: one colour, or one colour per
    edge (edge i runs from prof[i] to prof[i + 1]); the end caps take cols[0]."""
    P = [(float(z), float(y)) for z, y in prof]
    n = len(P)
    cl = cols if isinstance(cols, list) else [cols] * n
    area = 0.5 * sum(P[i][0] * P[(i + 1) % n][1] - P[(i + 1) % n][0] * P[i][1] for i in range(n))
    sgn = 1.0 if area > 0 else -1.0
    v, f, c = [], [], []
    for i in range(n):
        (z0, y0), (z1, y1) = P[i], P[(i + 1) % n]
        o = len(v)
        v += [(x0, y0, z0), (x0, y1, z1), (x1, y1, z1), (x1, y0, z0)]
        c += [cl[i]] * 4
        want = np.array([0.0, -(z1 - z0), (y1 - y0)]) * sgn          # outward normal of the edge
        q = np.array(v[o:o + 4])
        nn = np.cross(q[1] - q[0], q[2] - q[0])
        f += [(o, o + 1, o + 2), (o, o + 2, o + 3)] if nn @ want >= 0 else [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    pts, tris = triangulate(P)
    for xx, sx in ((x0, -1.0), (x1, 1.0)):
        o = len(v)
        v += [(xx, float(y), float(z)) for z, y in pts]
        c += [cl[0]] * len(pts)
        for tri in tris:
            q = np.array([v[o + t] for t in tri])
            nn = np.cross(q[1] - q[0], q[2] - q[0])
            f.append(tuple(o + t for t in (tri if nn[0] * sx >= 0 else tri[::-1])))
    return Mesh(np.array(v), np.array(f), np.array(c, float))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def plate_z(poly_xy, z, c, toward):
    """A flat shape drawn on a wall facing ±z (toward = +1 or -1): poly_xy = [(x, y), …]."""
    pts, tris = triangulate(poly_xy)
    v = np.stack([pts[:, 0], pts[:, 1], np.full(len(pts), z + 0.04 * toward)], 1)
    m = Mesh(v, tris, c)
    nn = np.cross(v[m.f[:, 1]] - v[m.f[:, 0]], v[m.f[:, 2]] - v[m.f[:, 0]])
    flip = nn[:, 2] * toward < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def arch_poly(xc, w, ys, n=10):
    """Outline of a round arch, width w, springing at ys, from the ground."""
    r = w / 2
    pts = [(xc - r, 0.0), (xc + r, 0.0)]
    for k in range(n + 1):
        a = math.pi * k / n
        pts.append((xc + r * math.cos(a), ys + r * math.sin(a)))
    return pts


# ── The shells ──────────────────────────────────────────────────────────────────────────────────────
def shell(x0, nmod, zr, zf):
    """The roof of one grandstand: nmod hyperboloid lobes between nmod + 1 frames."""
    zc = zf - CANT
    NU, NV_F, NV_R = 8, 10, 8
    zs = list(np.linspace(zr, zc, NV_R + 1)) + list(np.linspace(zc, zf, NV_F + 1))[1:]
    zs = np.array(zs)

    def under(z):
        if z >= zc:
            t = (z - zc) / (zf - zc)
            return Y_COL + (Y_FRONT - Y_COL) * t ** 1.6, t
        t = (zc - z) / (zc - zr)
        return Y_COL + (Y_REAR - Y_COL) * t ** 1.6, t

    def thick(t):
        return T_EDGE + (T_COL - T_EDGE) * (1 - t) ** 2.2

    def rise(t):
        return 0.55 + 0.75 * t

    ms = []
    for m in range(nmod):
        xa = x0 + m * MOD
        us = np.linspace(0, 1, NU + 1)
        top = np.zeros((len(zs), NU + 1, 3))
        bot = np.zeros((len(zs), NU + 1, 3))
        for i, z in enumerate(zs):
            yb, t = under(z)
            th = thick(t)
            for j, u in enumerate(us):
                lob = rise(t) * 4 * u * (1 - u)
                x = xa + u * MOD
                bot[i, j] = (x, yb + lob, z)
                top[i, j] = (x, yb + lob + th + 0.25 * (1 - t) * (1 - 4 * u * (1 - u)), z)
        for grid_, col, up in ((top, SHELL_TOP, True), (bot, SHELL_UNDER, False)):
            v, f = [], []
            for i in range(len(zs) - 1):
                for j in range(NU):
                    q = [grid_[i, j], grid_[i, j + 1], grid_[i + 1, j + 1], grid_[i + 1, j]]
                    o = len(v)
                    v += q
                    f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
            sm = Mesh(np.array(v), np.array(f), col)
            nn = np.cross(sm.v[sm.f[:, 1]] - sm.v[sm.f[:, 0]], sm.v[sm.f[:, 2]] - sm.v[sm.f[:, 0]])
            flip = (nn[:, 1] < 0) if up else (nn[:, 1] > 0)
            sm.f[flip] = sm.f[flip][:, ::-1]
            ms.append(sm)
        # edges: front and rear arches, and the two sides of the end lobes
        for i in (0, len(zs) - 1):
            v, f = [], []
            for j in range(NU):
                q = [bot[i, j], bot[i, j + 1], top[i, j + 1], top[i, j]]
                o = len(v)
                v += q
                f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
            ms.append(Mesh(np.array(v), np.array(f), WHITE))
        for j in ((0,) if m == 0 else ()) + ((NU,) if m == nmod - 1 else ()):
            v, f = [], []
            for i in range(len(zs) - 1):
                q = [bot[i, j], bot[i + 1, j], top[i + 1, j], top[i, j]]
                o = len(v)
                v += q
                f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
            ms.append(Mesh(np.array(v), np.array(f), WHITE))
    # the frames: a column under the low point of every frame, a steel tie behind it
    for k in range(nmod + 1):
        x = x0 + k * MOD
        x = min(max(x, x0 + 0.3), x0 + nmod * MOD - 0.3)
        ms.append(cyl(0.22, 0.45, Y_COL - Y_TERR + 0.1, CONC, seg=8, sx=1.0).T(x, Y_TERR, zc))
        zt = zc - 6.5
        yt, _ = under(zt)
        ms.append(B(x - 0.06, x + 0.06, Y_TERR, yt + 0.05, zt - 0.06, zt + 0.06, STEEL))
    return merge(ms)


# ── The grandstand under it ─────────────────────────────────────────────────────────────────────────
def grandstand(x0, nmod, zr, zf):
    zc = zf - CANT
    x1 = x0 + nmod * MOD
    zb = zr + 2.5                                   # back wall (the arcade to the paddock)
    rows = 16
    rise = (Y_TERR - 1.0) / rows
    depth = (zf + 1.5 - (zc + 1.0)) / rows
    pc = [((zb, FOUND), WHITE), ((zf + 4.5, FOUND), WHITE), ((zf + 4.5, 1.2), WHITE), ((zf + 4.2, 1.2), WHITE),
          ((zf + 4.2, 0.9), STEP)]
    z, y = zf + 1.5, 0.9
    for k in range(rows):
        pc.append(((z, y), RISER))
        y += rise
        pc.append(((z, y), SEATS if k < 4 else STEP))
        z -= depth
    # the upper gallery: a low wall at the back of the terraces, then the floor to the back wall
    pc += [((z, y), STEP), ((zc + 0.3, Y_TERR), WHITE), ((zc + 0.3, Y_TERR + 1.1), WHITE), ((zc - 0.3, Y_TERR + 1.1), WHITE),
           ((zc - 0.3, Y_TERR), STEP), ((zb + 0.4, Y_TERR), WHITE), ((zb + 0.4, Y_TERR + 1.0), WHITE),
           ((zb, Y_TERR + 1.0), WHITE)]
    prof = [p for p, _ in pc]
    cols = [c for _, c in pc]
    m = [prism_x(prof, x0 + 0.2, x1 - 0.2, cols)]
    # the arcade on the back wall: one arch per module, dark openings
    for k in range(nmod):
        xc = x0 + (k + 0.5) * MOD
        m.append(plate_z(arch_poly(xc, 3.4, 3.3), zb, DARK, -1))
    # glazed booths under the rear arm of the shell (upper gallery)
    m.append(B(x0 + 0.6, x1 - 0.6, Y_TERR, Y_TERR + 2.6, zb + 2.0, zb + 2.3, GLASS))
    # round stair towers at both ends
    for xe in (x0 - 2.2, x1 + 2.2):
        m.append(cyl(3.2, 3.2, 9.5 - FOUND, WHITE, seg=20).T(xe, FOUND, zc - 5.0))
    return merge(m)


def fillers():
    """The low white blocks of the OSM building between the grandstands and at the south end."""
    return merge([B(32.0, 56.6, FOUND, 4.0, 3.5, 7.5, WHITE),
                  B(-23.0, -0.2, FOUND, 4.0, -6.0, 8.0, WHITE),
                  B(-113.0, -88.0, FOUND, 3.5, -10.0, 12.5, WHITE)])


def rail():
    """The white rail along the near edge of the turf course (PNOA: z = 31.8 at x = -55, 41 at x = 86)."""
    ms = []
    xa, xb = -125.0, 125.0

    def zr(x):
        return 31.8 + (x + 55.0) * (41.0 - 31.8) / 141.0

    n = 50
    for k in range(n):
        x0 = xa + (xb - xa) * k / n
        x1 = xa + (xb - xa) * (k + 1) / n
        z0, z1 = zr(x0), zr(x1)
        L = math.hypot(x1 - x0, z1 - z0)
        ang = math.atan2(z1 - z0, x1 - x0)
        for yy in (0.62, 1.05):
            ms.append(box(L, 0.1, 0.12, RAIL).T(0, yy, 0).R(0, -ang, 0).T((x0 + x1) / 2, 0, (z0 + z1) / 2))
    for k in range(n + 1):
        x = xa + (xb - xa) * k / n
        ms.append(B(x - 0.06, x + 0.06, -0.5, 1.15, zr(x) - 0.06, zr(x) + 0.06, RAIL))
    return merge(ms)


def main():
    out = ROOT / "models" / "Q1620086"
    out.mkdir(parents=True, exist_ok=True)
    shells = merge([shell(x0, n, zr, zf) for _, x0, n, zr, zf in STANDS])
    stands = merge([grandstand(x0, n, zr, zf) for _, x0, n, zr, zf in STANDS] + [fillers()])
    r = rail()
    parts = [("shells", shells), ("grandstands", stands), ("rail", r)]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.9)
    allv = np.vstack([shells.v, stands.v, r.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    if "--footprint" in sys.argv:
        print(json.dumps([[round(x, 2), round(z, 2)] for x, z in offset(OUTLINE, 1.0)]))


if __name__ == "__main__":
    main()
