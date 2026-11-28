"""
Depósito elevado de Chamberí (Q5803317), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: the first elevated water tank of the Canal de Isabel II, Calle de Santa Engracia 125, built
1907–1911 (in service 1912–1952) to a design by Luis Moya Idígoras (died 1953), the pumping plant by Ramón de
Aguinaga; turned into an exhibition hall in 1986 by Antonio Lopera and Javier Alau. Protected (20th century).

Measurements (see meta.json for the sources):
- OpenStreetMap way 48238313 (building=water_tower): a circle 24 m across (area 446 m²) — the twelve-sided
  plinth at the foot of the tower; the 1911 vertical section of the Canal (Wikimedia Commons, «Depósito elevado,
  sección vertical, Madrid.jpg») scaled by those 24 m: the plinth up to 7.1 m, the battered brick shaft from
  about 10.9 m (radius) at its foot to 9.9 m at 20 m, the brick gallery up to 22.9 m, the steel tank (radius
  9.2 m) with its shoulder at 29.9 m and the top of its roof at 36.6 m (Wikipedia: water raised to 36 m,
  1500 m³).
- Photos on Commons (see meta.json): a twelve-sided brick shaft with a pier at every corner; tall recessed bays
  of windows between them (pale panels, brick mullions, curved transoms), a hood over each bay, a band of slit
  openings with brick crosses under a corbelled cornice; the piers rise above it with small caps. The tank is
  clad in zinc (diamond scales, plain straps) with two round windows per side, a heavy cornice, a ribbed zinc
  dome with a low ring on top and a finial. Round the foot, the plinth with slate lean-to roofs and twelve small
  brick pavilions with tiled gables (one in front of each pier), on a granite socle; the door in the bay facing
  the square.
- Orthophoto PNOA: the twelve pavilions at bearings about 26° + 30°·k, so the sides face 11° + 30°·k; the side
  with the door is taken as the main façade, facing 191°.
Red brick, grey granite socle, grey slate, red tiles, zinc.

Axes (docs/MODEL_SPEC.md): +Y up, the side with the door faces +Z (191°), origin on the ground at the centre of
the OSM outline.

    python model_sources/Q5803317_deposito_elevado_de_chamberi.py   -> models/Q5803317/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, fbm, noise3, write_glb  # noqa: E402

OUT = ROOT / "models" / "Q5803317"
rng = np.random.default_rng(1911)

BRICK = np.array((0.58, 0.31, 0.21))
BRICK_D = np.array((0.47, 0.22, 0.15))
PANEL = (0.78, 0.72, 0.60)          # pale panels of the windows
DARK = (0.10, 0.09, 0.09)
GRANITE = np.array((0.60, 0.59, 0.56))
SLATE = np.array((0.30, 0.32, 0.35))
TILE = np.array((0.62, 0.29, 0.20))
ZINC = np.array((0.52, 0.56, 0.58))
ZINC_D = np.array((0.42, 0.46, 0.48))
FOUND = -3.0

N = 12
STEP = 2 * math.pi / N
TAN = math.tan(STEP / 2)
COS = math.cos(STEP / 2)


def apo(y):
    """Apothem of the brick faces of the shaft at height y (battered: 10.45 m at 7 m, 9.6 m at 20 m)."""
    return 10.45 - 0.065 * (np.asarray(y, float) - 7.0)


A_REF = 10.0                          # details are built at this apothem and then warped onto the batter


def batter(m):
    """Scale x and z about the axis so that a piece built at apothem A_REF follows the batter of the shaft."""
    s = apo(np.clip(m.v[:, 1], 0.0, 23.0)) / A_REF
    m.v[:, 0] *= s; m.v[:, 2] *= s
    return m


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def around(ms, offset=0.0):
    """Copies of a piece built on the +Z side, turned to all twelve sides."""
    ms = [m for m in ms if m is not None]
    one = merge(ms)
    return merge([one.copy().R(0, offset + k * STEP, 0) for k in range(N)])


def brick(base=BRICK, seed=3):
    base = np.asarray(base, float)

    def fn(v):
        n1 = fbm(v, 0.25, seed, 3) - 0.5
        n2 = noise3(v, 1.3, seed + 7) - 0.5
        k = 1 + 0.16 * n1 + 0.08 * n2
        return np.clip(np.outer(k, base), 0, 1)
    return fn


def tone(base, amp=0.08, seed=5, freq=0.4):
    base = np.asarray(base, float)
    return lambda v: np.clip(np.outer(1 + amp * (fbm(v, freq, seed, 2) - 0.5) * 2, base), 0, 1)


def ngon_band(a0, a1, y0, y1, c, n=N, rot=0.0, top=False):
    """A band of an n-sided prism (apothem a0 at y0, a1 at y1), each side its own quad; sides face 0 + k·360/n."""
    st = 2 * math.pi / n
    t = math.tan(st / 2)
    V, F = [], []
    for k in range(n):
        p = rot + k * st
        nx, nz = math.sin(p), math.cos(p)
        tx, tz = math.cos(p), -math.sin(p)
        q = [(a0, -1, y0), (a0, 1, y0), (a1, 1, y1), (a1, -1, y1)]
        o = len(V)
        for a, s, y in q:
            w = a * t * s
            V.append((nx * a + tx * w, y, nz * a + tz * w))
        F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    m = Mesh(V, F, c)
    _out(m)
    if top:
        ring = [(a1 / math.cos(st / 2) * math.sin(rot + (k + 0.5) * st), a1 / math.cos(st / 2) * math.cos(rot + (k + 0.5) * st)) for k in range(n)]
        Vc = [(x, y1, z) for x, z in ring] + [(0, y1, 0)]
        Fc = [(n, (k + 1) % n, k) for k in range(n)]
        mc = Mesh(Vc, Fc, c)
        _up(mc)
        m = merge([m, mc])
    return m


def _out(m):
    """Make every face point away from the axis."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    nrm = np.cross(b - a, c - a)
    ctr = (a + b + c) / 3
    flip = (nrm[:, 0] * ctr[:, 0] + nrm[:, 2] * ctr[:, 2]) < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def _up(m):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    flip = np.cross(b - a, c - a)[:, 1] < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def sphere(r, c, seg=12, rings=7):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


# ── The plinth: granite socle, brick wall, slate lean-to roofs, twelve pavilions ──────────────────────────
A_PL = 11.75          # apothem of the plinth (circumradius 12.2 m: the 446 m² of the OSM outline)
Y_EAVE, Y_PL = 4.6, 7.1


def plinth():
    out = []
    out.append(ngon_band(A_PL + 0.15, A_PL + 0.15, FOUND, 0.0, GRANITE * 0.85))
    out.append(ngon_band(A_PL + 0.15, A_PL + 0.05, 0.0, 1.0, GRANITE).paint(tone(GRANITE, 0.12, 4, 1.2)))
    out.append(ngon_band(A_PL + 0.05, A_PL + 0.05, 1.0, 1.15, GRANITE * 1.05, top=False))
    out.append(ngon_band(A_PL, A_PL, 1.0, Y_EAVE - 0.3, BRICK).paint(brick(BRICK, 11)))
    out.append(ngon_band(A_PL + 0.25, A_PL + 0.25, Y_EAVE - 0.3, Y_EAVE, BRICK_D).paint(brick(BRICK_D, 12)))
    # the lean-to roof up to the shaft
    out.append(ngon_band(A_PL + 0.45, float(apo(Y_PL)) - 0.05, Y_EAVE, Y_PL, SLATE).paint(tone(SLATE, 0.1, 13, 0.8)))
    return out


def pavilion():
    """The small brick pavilion in front of a pier (built on the corner at +Z, turned by 15° later)."""
    a = A_PL / COS + 0.15       # its front, a little out of the plinth
    w = 2.5
    out = [B(-w / 2, w / 2, FOUND, 7.5, a - 3.2, a, BRICK_D).paint(brick(BRICK_D, 21)),
           B(-w / 2 - 0.12, w / 2 + 0.12, 7.5, 7.75, a - 3.2, a + 0.12, BRICK).paint(brick(BRICK, 22)),
           B(-0.35, 0.35, 0.0, 7.5, a, a + 0.18, BRICK_D)]
    # the tiled gable roof, ridge pointing outwards
    hw, y0, y1, z0, z1 = w / 2 + 0.25, 7.75, 8.75, a - 3.2, a + 0.3
    V = [(-hw, y0, z0), (0, y1, z0), (hw, y0, z0), (-hw, y0, z1), (0, y1, z1), (hw, y0, z1)]
    F = [(0, 4, 1), (0, 3, 4), (1, 4, 5), (1, 5, 2), (3, 0, 2), (3, 2, 5)]
    roof = Mesh(V, F, TILE)
    out.append(roof.paint(tone(TILE, 0.1, 23, 1.0)))
    gab = Mesh([(-w / 2, 7.75, a + 0.02), (w / 2, 7.75, a + 0.02), (0, 8.6, a + 0.02)], [(0, 1, 2)], BRICK)
    out.append(gab)
    return out


def door():
    """The arched door in the plinth on the +Z side."""
    z = A_PL + 0.03
    out = [B(-1.5, 1.5, 0.0, 3.0, z - 0.2, z, DARK)]
    arc = [(1.5 * math.cos(t), 3.0 + 0.9 * math.sin(t)) for t in np.linspace(0, math.pi, 10)]
    V = [(x, y, z) for x, y in arc] + [(0, 3.0, z)]
    F = [(len(arc), i, i + 1) for i in range(len(arc) - 1)]
    out.append(Mesh(V, F, DARK))
    out.append(B(-1.85, 1.85, 3.9, 4.15, z, z + 0.15, GRANITE))
    return out


# ── The shaft ─────────────────────────────────────────────────────────────────────────────────────────
Y_BAY, Y_HOOD, Y_GAL, Y_TOP, Y_PIER = 8.9, 19.0, 19.7, 23.0, 24.3


def shaft_core():
    out = []
    # full faces below the bays and around the gallery; in the zone of the bays the core is set back behind
    # the window panels (the piers and the jambs close the sides)
    for y0, y1, d in ((FOUND, 0.0, 0), (0.0, Y_BAY, 0), (Y_BAY, Y_HOOD, 0.62), (Y_HOOD, 19.7, 0), (19.7, Y_TOP, 0)):
        out.append(ngon_band(float(apo(y0)) - d, float(apo(y1)) - d, y0, y1, BRICK,
                             top=(y1 == Y_TOP)).paint(brick(BRICK, 31)))
    return out


def bay():
    """The recessed window bay of a side, built at apothem A_REF on +Z (then battered)."""
    a = A_REF
    w = 2 * a * TAN - 2.6            # between the piers
    hw = w / 2
    out = [B(-hw - 0.1, hw + 0.1, Y_BAY - 0.3, Y_BAY, a - 0.7, a + 0.1, BRICK_D)]          # the sill of the bay
    # the recess: pale panels set back 0.45 m, framed by brick
    zb = a - 0.45
    out.append(B(-hw, hw, Y_BAY, Y_HOOD, zb - 0.05, zb + 0.02, PANEL).paint(tone(PANEL, 0.05, 41, 0.5)))
    # the jambs of the recess (cover the edge of the core)
    out.append(B(-hw - 0.3, -hw + 0.12, Y_BAY, Y_HOOD, a - 0.7, a + 0.02, BRICK_D))
    out.append(B(hw - 0.12, hw + 0.3, Y_BAY, Y_HOOD, a - 0.7, a + 0.02, BRICK_D))
    # mullions
    for x in (-hw / 3, hw / 3):
        out.append(B(x - 0.2, x + 0.2, Y_BAY, Y_HOOD, zb, zb + 0.3, BRICK).paint(brick(BRICK, 42)))
    # curved transoms (brick bands, bowed out)
    for y in (Y_BAY, Y_BAY + 2.55, Y_BAY + 5.1, Y_BAY + 7.65):
        n = 8
        xs = np.linspace(-hw, hw, n + 1)
        bow = 0.35 * (1 - (xs / hw) ** 2)
        for i in range(n):
            out.append(B(xs[i], xs[i + 1], y - 0.25, y, zb, zb + 0.25 + 0.5 * (bow[i] + bow[i + 1]) / 2, BRICK))
    # the small iron grilles low in the lights of the two lower tiers
    for y in (Y_BAY + 2.3, Y_BAY + 4.85):
        for x0, x1 in ((-hw + 0.15, -hw / 3 - 0.2), (-hw / 3 + 0.2, hw / 3 - 0.2), (hw / 3 + 0.2, hw - 0.15)):
            out.append(B(x0 + 0.1, x1 - 0.1, y - 0.7, y - 0.25, zb - 0.02, zb + 0.04, (0.36, 0.33, 0.30)))
    # the hood over the bay: a bowed ledge
    n = 10
    xs = np.linspace(-hw - 0.1, hw + 0.1, n + 1)
    bow = 0.55 * (1 - (xs / (hw + 0.1)) ** 2)
    for i in range(n):
        out.append(B(xs[i], xs[i + 1], Y_HOOD, Y_GAL, a - 0.7, a + 0.25 + (bow[i] + bow[i + 1]) / 2, BRICK_D).paint(brick(BRICK_D, 43)))
    # the gallery: slit openings with brick crosses, corbels, cornice
    gw = 2 * apo(21.0) / A_REF * a * TAN - 2.6
    ghw = gw / 2
    k = 7
    xs = np.linspace(-ghw + 0.3, ghw - 0.3, k)
    for x in xs:
        out.append(B(x - 0.13, x + 0.13, 20.3, 21.9, a - 0.05, a + 0.02, DARK))
        out.append(B(x - 0.22, x + 0.22, 22.15, 22.27, a - 0.05, a + 0.03, DARK))
        out.append(B(x - 0.07, x + 0.07, 22.0, 22.45, a - 0.05, a + 0.03, DARK))
    out.append(B(-ghw, ghw, 20.0, 20.3, a, a + 0.25, BRICK))                 # the sill of the gallery
    for x in np.linspace(-ghw + 0.2, ghw - 0.2, 12):                          # corbels under the cornice
        out.append(B(x - 0.1, x + 0.1, 22.55, 22.75, a, a + 0.25, BRICK_D))
    out.append(B(-ghw - 0.1, ghw + 0.1, 22.75, Y_TOP, a - 0.1, a + 0.4, BRICK).paint(brick(BRICK, 44)))
    return out


def pier():
    """The brick pier on a corner, built on the radial line at +Z (then turned by 15° and battered)."""
    r = A_REF / COS
    out = [B(-1.2, 1.2, FOUND, Y_PIER, r - 1.3, r + 0.45, BRICK).paint(brick(BRICK, 51))]
    # two grooves up the front (the doubled pilasters)
    for x in (-0.45, 0.45):
        out.append(B(x - 0.06, x + 0.06, 7.5, 21.0, r + 0.44, r + 0.47, BRICK_D * 0.85))
    # set-backs and cap
    out.append(B(-1.3, 1.3, 7.1, 7.5, r - 0.5, r + 0.6, BRICK_D))
    out.append(B(-1.3, 1.3, Y_PIER, Y_PIER + 0.25, r - 1.3, r + 0.6, BRICK_D))
    out.append(B(-1.0, 1.0, Y_PIER + 0.25, Y_PIER + 0.55, r - 1.1, r + 0.4, BRICK).paint(brick(BRICK, 52)))
    for x in (-0.4, 0.4):
        out.append(B(x - 0.12, x + 0.12, 23.6, 23.95, r + 0.44, r + 0.5, DARK))
    return out


# ── The tank ──────────────────────────────────────────────────────────────────────────────────────────
R_T = 9.25
Y_T0, Y_SH, Y_DOME = 22.0, 28.0, 33.8


def tank():
    out = []
    seg = 96
    drum = cyl(R_T, R_T, Y_SH - Y_T0, ZINC, seg, caps=False).T(0, Y_T0, 0)

    def scales(v):
        ang = np.arctan2(v[:, 0], v[:, 2])
        d = ((ang * R_T / 0.55) + v[:, 1] / 0.55)
        e = ((ang * R_T / 0.55) - v[:, 1] / 0.55)
        k = 1 + 0.05 * np.sign(np.sin(d * math.pi) * np.sin(e * math.pi)) + 0.07 * (fbm(v, 0.3, 61, 2) - 0.5)
        return np.clip(np.outer(k, ZINC), 0, 1)
    out.append(drum.paint(scales))
    # plain straps at every side and a band at the foot of the drum
    for k in range(N):
        out.append(B(-0.35, 0.35, Y_T0, Y_SH, R_T - 0.05, R_T + 0.06, ZINC * 1.06).R(0, STEP / 2 + k * STEP, 0))
    out.append(cyl(R_T + 0.06, R_T + 0.06, 0.9, ZINC_D, seg, caps=False).T(0, 23.0, 0))
    # two round windows a side
    for k in range(N):
        for d in (-0.16, 0.16):
            p = k * STEP + d
            ring = cyl(0.5, 0.5, 0.18, ZINC_D, 14).R(math.pi / 2, 0, 0).T(0, 24.9, R_T - 0.02)
            glass = cyl(0.36, 0.36, 0.05, DARK, 12).R(math.pi / 2, 0, 0).T(0, 24.9, R_T + 0.17)
            out += [ring.R(0, p, 0), glass.R(0, p, 0)]
    # the cornice
    prof = [(R_T, Y_SH - 0.3), (R_T + 0.25, Y_SH - 0.1), (R_T + 0.25, Y_SH + 0.15), (R_T + 0.5, Y_SH + 0.35),
            (R_T + 0.55, Y_SH + 0.6), (R_T + 0.3, Y_SH + 0.7), (R_T + 0.15, Y_SH + 0.75)]
    out.append(lathe(prof, ZINC_D, seg))
    # the ribbed dome
    y0 = Y_SH + 0.75
    ts = np.linspace(0, math.pi / 2, 9)
    r_top = 4.6
    prof = [(r_top + (R_T + 0.15 - r_top) * math.cos(t), y0 + (Y_DOME - y0) * math.sin(t)) for t in ts]
    dome = lathe(prof, ZINC, seg)
    ang = np.arctan2(dome.v[:, 0], dome.v[:, 2])
    rib = (np.round(ang / (2 * math.pi / seg)).astype(int) % 2 == 0)
    rr = np.hypot(dome.v[:, 0], dome.v[:, 2])
    s = np.where(rib, (rr + 0.06) / np.maximum(rr, 1e-6), 1.0)
    dome.v[:, 0] *= s; dome.v[:, 2] *= s
    out.append(dome.paint(tone(ZINC, 0.08, 63, 0.4)))
    # the ring on top (ventilation), its roof and the finial
    out.append(cyl(r_top, r_top, 0.15, ZINC_D, 48).T(0, Y_DOME, 0))
    out.append(cyl(r_top - 0.15, r_top - 0.15, 0.55, DARK, 48, caps=False).T(0, Y_DOME + 0.15, 0))
    for k in range(24):
        out.append(B(-0.06, 0.06, Y_DOME + 0.15, Y_DOME + 0.7, r_top - 0.2, r_top - 0.05, ZINC_D).R(0, k * 2 * math.pi / 24, 0))
    out.append(lathe([(r_top + 0.2, Y_DOME + 0.7), (r_top + 0.2, Y_DOME + 0.85), (2.0, 34.95), (0.0, 35.15)], ZINC, 48))
    out.append(cyl(0.12, 0.08, 0.6, ZINC_D, 8).T(0, 35.0, 0))
    out.append(sphere(0.28, ZINC_D, 10, 6).T(0, 35.8, 0))
    out.append(cyl(0.05, 0.01, 2.2, DARK, 6).T(0, 36.0, 0))
    # the floor of the tank behind the brick (closes the drum at the bottom)
    out.append(cyl(R_T, R_T, 0.2, ZINC_D, seg).T(0, Y_T0, 0))
    return out


def tower():
    parts = []
    parts += plinth()
    parts.append(around(pavilion(), STEP / 2))
    parts += door()
    parts += shaft_core()
    parts.append(around([batter(m) for m in bay()]))
    parts.append(around([batter(m) for m in pier()], STEP / 2))
    return merge(parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    brickwork = tower()
    zinc = merge(tank())
    tris, size = write_glb(OUT / "model.glb", [("deposito_elevado", brickwork), ("deposito_cuba", zinc, {"material": "zinc"})],
                           roughness=0.9, materials={"zinc": {"metallic": 0.35, "roughness": 0.55}})
    m = merge([brickwork, zinc])
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
