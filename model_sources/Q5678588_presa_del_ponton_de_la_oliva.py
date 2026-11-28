"""
Presa del Pontón de la Oliva (Q5678588), Patones (Madrid) / Uceda (Guadalajara) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: the first dam of the Canal de Isabel II, on the Lozoya, project of the engineers Juan Rafo and Juan de
Ribera; first stone 11 August 1851, works finished 1856, inaugurated 24 June 1858; public domain. Abandoned a few
years later because the water leaked away through the limestone of the gorge (replaced by El Villar, 1882).

Measurements (see meta.json for the sources):
- OpenStreetMap way 312065377 (waterway=dam, building=yes): the outline seen from above. Its downstream side runs
  straight at 67° (its long sides 66.4°–68°), so the downstream face looks 157° (SSE, down the Lozoya); upstream
  edge at z = −16.1, downstream toe at z = +16.0, from x = −43.5 (west, Patones side) to +48.4 (east). The tower
  of the old intake (way 1533207225, man_made=tower) stands on the upstream side at the west end; the Lozoya
  runs through the west abutment in a culvert (way 1533207209) whose mouth is at the upstream foot by the tower.
- Wikipedia (es): gravity dam of large ashlar blocks with lime mortar, 27 m high, trapezoidal section 39 m wide
  at the base and 6.72 m at the crest, crest 72.44 m long between the rock walls; the upstream face is stepped
  from the base upwards («la cara interna… aparece escalonada»).
- Photos (Commons): from the dry bed upstream («Pontón de la Oliva, Madrid, España, 2021 15.jpg») and from the west
  end («Pontón de la Oliva (4133880482).jpg»): an upright upper wall of ashlar about a third of the height, then a
  long slope of dark rubble masonry, a berm and three big steps at the foot; the octagonal tower against the cliff
  with a vaulted tunnel mouth at its foot. From downstream («Presa del Pontón de la Oliva - panoramio (1)/(2)/(3)
  .jpg», «El Pontón de la Oliva aliviando.jpg»): a plain, almost upright face of light limestone ashlar with
  darker horizontal bands, a round turret at the east end, and a coping on the crest.
- PNOA orthophoto: the crest (a light band about 6.7 m wide on the downstream side of the outline), the shadow of
  the upper upstream wall and the stepped slope (about 21 m wide seen from above).
- IGN MDT05 (terrain.json and a 2.5 m sampling): the river bed below the dam at 702 m (y = 0), the silted bed of
  the old reservoir at 705–706 m, the limestone walls of the gorge rising above the crest at x < −50 and x > +45.
  The MDT05 does not carry the dam (it is cut out and filled to ~714 m), so the crest is taken 27 m above the bed
  downstream: 729 m (±2 m).
NOTE: the brief described the steps on the downstream face; the photos, the orthophoto and Wikipedia put them on
the upstream face, which is how it is modelled.

The rock of the gorge (a «rock skirt», MODEL_SPEC §3.1) is drawn around the dam following the MDT05, with the
dam's own hump taken out (upstream and downstream of the dam the ground is taken from the terrain 22–24 m away).

Axes (docs/MODEL_SPEC.md): +Y up, the downstream face faces +Z (157°), the dam runs along X (+X to the
east-north-east, the left bank, Guadalajara), origin on the dam wall at 40.88341, −3.44195 (OSM way 312065377);
y = 0 is the river bed below the dam (702 m above sea level).

    python model_sources/Q5678588_presa_del_ponton_de_la_oliva.py   -> models/Q5678588/model.glb
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, triangulate, crumple, noise3, fbm, write_glb  # noqa: E402

OUT = ROOT / "models" / "Q5678588"
rng = np.random.default_rng(1851)

HEADING = 157.0
E0 = 702.0                    # the river bed below the dam (m a.s.l.) = y 0
YC = 27.0                     # the crest (729 m)
YF = -4.0                     # foundation
X0, X1 = -52.0, 52.0          # the ends, inside the rock of the gorge
ZD_TOP = 14.6                 # downstream edge of the crest
BATTER = 0.05                 # downstream face, 1:20
ZU_TOP = ZD_TOP - 6.72        # upstream edge of the 6.72 m crest
# the cross-section (z, y), from the downstream foot over the crest to the upstream foot
PROFILE = [(ZD_TOP + BATTER * (YC - YF), YF), (ZD_TOP, YC),                     # downstream face
           (ZU_TOP, YC),                                                         # crest
           (ZU_TOP - 0.45, 18.0),                                                # upright upper upstream wall
           (-9.5, 7.5),                                                          # slope of rubble masonry
           (-11.5, 7.5),                                                         # berm
           (-11.5, 6.2), (-13.0, 6.2), (-13.0, 4.9), (-14.5, 4.9), (-14.5, 3.6), (-16.1, 3.6),   # three steps
           (-16.1, YF)]
KIND = ["down", "crest", "wall", "slope", "berm", "step", "tread", "step", "tread", "step", "tread", "found"]

TOWER = (-44.4, -6.0)         # the intake tower (OSM way 1533207225)
TUNNEL = (-37.5, -11.3)       # mouth of the culvert of the Lozoya (OSM way 1533207209)
TURRET = (44.0, 15.2)         # the round turret at the east end, downstream

ASHLAR = np.array((0.80, 0.77, 0.69))    # light limestone ashlar
UPW = np.array((0.60, 0.58, 0.54))       # upstream upper wall, weathered
RUBBLE = np.array((0.47, 0.46, 0.43))    # rubble masonry of the upstream slope
STEP = np.array((0.70, 0.68, 0.63))      # the big steps and the berm
CREST = np.array((0.74, 0.71, 0.64))
ROCK = np.array((0.74, 0.70, 0.61))      # limestone of the gorge
SOIL = np.array((0.60, 0.55, 0.43))
SCRUB = np.array((0.28, 0.32, 0.20))
LEAF = (0.24, 0.31, 0.16)
DARK = (0.10, 0.10, 0.09)


# ── Terrain (terrain.json, IGN MDT05) ────────────────────────────────────────
def load_terrain():
    t = json.loads((OUT / "terrain.json").read_text("utf-8"))
    n = t["n"]
    q = np.frombuffer(base64.b64decode(t["h"]), "<i2").reshape(n, n).astype(np.float64)
    h = np.where(q == -32768, np.nan, t["off"] + q / 10)
    th = math.pi - math.radians(HEADING)
    cs, sn = math.cos(th), math.sin(th)

    def T(x, z):
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        east = x * cs + z * sn
        north = -(-x * sn + z * cs)
        fi = (east - t["x0"]) / t["c"]
        fj = (t["y0"] - north) / t["c"]
        i0 = np.clip(np.floor(fi).astype(int), 0, n - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
        a = np.clip(fi - i0, 0, 1)
        b = np.clip(fj - j0, 0, 1)
        return (h[j0, i0] * (1 - a) * (1 - b) + h[j0, i0 + 1] * a * (1 - b)
                + h[j0 + 1, i0] * (1 - a) * b + h[j0 + 1, i0 + 1] * a * b)
    return T


T = load_terrain()
ZD_REAL, ZU_REAL = 24.0, -22.0           # beyond the fill of the dam in the MDT


def ground(x, z):
    """The ground of the gorge without the dam, in model y (the abutments keep the MDT as it is)."""
    x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
    dn = T(x, np.maximum(z, ZD_REAL))
    up = T(x, np.minimum(z, ZU_REAL))
    w = np.clip((z - ZU_REAL) / (ZD_REAL - ZU_REAL), 0, 1)
    gap = dn * w + up * (1 - w)
    side = np.clip((np.abs(x) - 40.0) / 8.0, 0, 1)                  # the rock walls at the ends
    return np.maximum(gap, gap * (1 - side) + T(x, z) * side) - E0


# ── Mesh helpers ─────────────────────────────────────────────────────────────
def orient(m, out):
    """Flip the winding of all faces if their summed normal points against out."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    if float(np.dot(np.cross(b - a, c - a).sum(0), out)) < 0:
        m.f = m.f[:, ::-1]
    return m


def clean(m):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    m.f = m.f[np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-6]
    return m


def sheet(P, out, col=(0.5, 0.5, 0.5)):
    """A grid surface from points P (nu+1, nv+1, 3), facing out."""
    nu, nv = P.shape[0] - 1, P.shape[1] - 1
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    m = Mesh(P.reshape(-1, 3), np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]), col)
    return clean(orient(m, out))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def poly_x(poly, x, out, col):
    """A flat polygon given in (z, y), lying in the plane x, facing out (±1 along x)."""
    pts, tris = triangulate(poly)
    m = Mesh(np.stack([np.full(len(pts), x), pts[:, 1], pts[:, 0]], 1), tris, col)
    return orient(m, (out, 0, 0))


def xs_between(a, b, step):
    n = max(1, int(math.ceil((b - a) / step)))
    return np.linspace(a, b, n + 1)


def loft(p, q, x0, x1, step_x=2.0, step_s=1.5):
    """The strip of the section from p to q (z, y), swept along x; facing out of the dam."""
    (za, ya), (zb, yb) = p, q
    L = math.hypot(zb - za, yb - ya)
    xs = xs_between(x0, x1, step_x)
    ts = np.linspace(0, 1, max(1, int(math.ceil(L / step_s))) + 1)
    X = np.repeat(xs[:, None], len(ts), 1)
    Z = np.repeat((za + (zb - za) * ts)[None, :], len(xs), 0)
    Y = np.repeat((ya + (yb - ya) * ts)[None, :], len(xs), 0)
    out = (0, -(zb - za), yb - ya)                        # the section runs counter-clockwise in (z, y)
    return sheet(np.stack([X, Y, Z], -1), out)


# ── Colours ──────────────────────────────────────────────────────────────────
def ashlar(base, seed, course=0.9, bands=()):
    base = np.asarray(base, float)

    def fn(v):
        n1 = fbm(v, 0.07, seed, 3) - 0.5
        n2 = noise3(v, 0.6, seed + 5) - 0.5
        st = fbm(v, (0.3, 0.035, 0.3), seed + 9, 2) - 0.5                  # rain streaks
        row = np.where(np.floor(v[:, 1] / course + 1e-6).astype(int) % 2 == 0, 1.02, 0.97)
        blk = np.where(np.floor(v[:, 0] / 1.6 + 0.5 * np.floor(v[:, 1] / course)).astype(int) % 3 == 0, 0.97, 1.0)
        dim = 1 - 0.14 * np.clip((6 - v[:, 1]) / 10, 0, 1)                # damp and darker low down
        k = (1 + 0.16 * n1 + 0.08 * n2 - 0.2 * np.clip(st, 0, 1)) * row * blk * dim
        for y0, y1 in bands:                                               # darker horizontal bands
            k = k * (1 - 0.16 * ((v[:, 1] > y0) & (v[:, 1] < y1)))
        return np.clip(np.outer(k, base), 0, 1)
    return fn


def rubble(base, seed):
    base = np.asarray(base, float)

    def fn(v):
        n1 = noise3(v, 0.9, seed) - 0.5
        n2 = fbm(v, 0.08, seed + 3, 3) - 0.5
        k = 1 + 0.34 * n1 + 0.2 * n2
        lime = np.clip((noise3(v, 0.35, seed + 7) - 0.62) * 4, 0, 1)[:, None]   # pale patches of mortar
        c = np.outer(k, base)
        return np.clip(c * (1 - lime) + STEP * lime, 0, 1)
    return fn


def earth(base, seed, green=0.0):
    base = np.asarray(base, float)

    def fn(v):
        n1 = fbm(v, 0.1, seed, 3) - 0.5
        n2 = noise3(v, 0.8, seed + 3) - 0.5
        c = np.outer(1 + 0.25 * n1 + 0.15 * n2, base)
        if green:
            g = np.clip((fbm(v, 0.06, seed + 11, 2) - 0.55) * 4, 0, 1)[:, None] * green
            c = c * (1 - g) + SCRUB * g
        return np.clip(c, 0, 1)
    return fn


# ── The dam ──────────────────────────────────────────────────────────────────
def dam_body():
    ms = []
    bands = ((5.6, 6.6), (12.4, 13.3), (19.2, 20.0))
    for k, (p, q) in enumerate(zip(PROFILE[:-1], PROFILE[1:])):
        kind = KIND[k]
        m = loft(p, q, X0, X1, step_x=1.6 if kind == "down" else 2.0, step_s=1.0 if kind == "down" else 1.5)
        if kind == "down":
            m.paint(ashlar(ASHLAR, 11, bands=bands))
        elif kind == "crest":
            m.paint(earth(CREST, 12))
        elif kind == "wall":
            m.paint(ashlar(UPW, 13, course=1.1))
        elif kind == "slope":
            m.paint(rubble(RUBBLE, 14))
        elif kind in ("berm", "tread"):
            m.paint(earth(STEP, 15 + k))
        elif kind == "step":
            m.paint(ashlar(STEP * 0.95, 20 + k, course=1.3))
        else:
            m.paint(ashlar(STEP * 0.8, 30))
        ms.append(m)
    for x, s in ((X0, -1), (X1, 1)):                     # the ends, inside the rock
        ms.append(poly_x(PROFILE, x, s, ROCK).paint(earth(ROCK, 31)))
    return ms


def crest_works():
    """Coping, parapet on the downstream side, string courses."""
    ms = []
    lite = tuple(np.clip(ASHLAR * 1.05, 0, 1))
    ms.append(B(X0, X1, YC - 0.9, YC - 0.45, ZD_TOP - 0.2, ZD_TOP + 0.3, lite))           # cornice
    ms.append(B(X0, X1, YC, YC + 0.95, ZD_TOP - 0.6, ZD_TOP + 0.05, tuple(ASHLAR * 0.98)))  # parapet
    ms.append(B(X0, X1, YC + 0.95, YC + 1.1, ZD_TOP - 0.7, ZD_TOP + 0.15, lite))           # its coping
    ms.append(B(X0, X1, YC - 0.1, YC + 0.25, ZU_TOP - 0.1, ZU_TOP + 0.45, lite))           # kerb upstream
    for y in (6.6, 13.3, 20.0):                                                             # string courses
        z = ZD_TOP + BATTER * (YC - y)
        ms.append(B(X0, X1, y - 0.3, y, z - 0.2, z + 0.12, tuple(ASHLAR * 0.9)))
    return ms


def tower():
    """The octagonal tower of the old intake, against the west cliff, with its top platform."""
    x, z = TOWER
    ms = []
    shaft = cyl(2.35, 2.15, 30.0, tuple(UPW), seg=8).T(x, 0.0, z)
    ms.append(shaft.paint(ashlar(UPW * 1.08, 41, course=0.8, bands=((0.0, 12.0),))))
    ms.append(cyl(2.6, 2.6, 0.5, tuple(np.clip(ASHLAR * 1.04, 0, 1)), seg=8).T(x, 29.6, z))    # cornice
    ms.append(cyl(2.5, 2.5, 0.9, tuple(UPW * 1.05), seg=8, caps=True).T(x, 30.1, z))
    # the curved landing between the end of the crest and the tower
    ms.append(B(x + 1.0, X0 + 12.0, YC - 0.4, YC + 0.1, ZU_TOP - 6.0, ZU_TOP + 0.5, tuple(CREST)))
    ms.append(B(x + 1.0, X0 + 12.0, YC + 0.1, YC + 1.0, ZU_TOP - 6.0, ZU_TOP - 5.7, (0.18, 0.18, 0.18)))  # railing
    # the gate house on the crest at the west end (OSM way 1533207190)
    ms.append(B(-43.5, -40.8, YC, YC + 3.2, 7.5, 11.4, tuple(ASHLAR)).paint(ashlar(ASHLAR, 43)))
    ms.append(B(-43.8, -40.5, YC + 3.2, YC + 3.6, 7.2, 11.7, tuple(np.clip(ASHLAR * 1.05, 0, 1))))
    ms.append(B(-42.6, -41.7, YC, YC + 2.2, 11.38, 11.45, DARK))
    return ms


def tunnel_mouth():
    """The vaulted mouth of the culvert of the Lozoya at the upstream foot, by the tower."""
    x, z = TUNNEL
    r, spring, deep = 1.4, 3.0, 6.0
    ms = []
    zf = -16.1 - 0.2                                      # the face of the steps there
    a = np.linspace(0, math.pi, 11)
    P = np.array([[(x + r * math.cos(t), spring + r * math.sin(t), zz) for zz in (zf - 0.3, zf + deep)] for t in a])
    ms.append(sheet(P, (0, -1, 0), DARK))
    for xx, s in ((x - r, 1), (x + r, -1)):
        ms.append(sheet(np.array([[(xx, 1.5, zf - 0.3), (xx, 1.5, zf + deep)], [(xx, spring, zf - 0.3),
                                                                                  (xx, spring, zf + deep)]]), (s, 0, 0), DARK))
    ms.append(sheet(np.array([[(x - r, 1.5, zf + deep), (x - r, spring + r, zf + deep)],
                              [(x + r, 1.5, zf + deep), (x + r, spring + r, zf + deep)]]), (0, 0, -1), (0.05, 0.05, 0.05)))
    # the masonry frame: a block standing out of the steps with the arch in it
    ms.append(B(x - r - 1.6, x - r, YF, spring + r + 1.2, zf - 1.2, zf + 0.5, tuple(ASHLAR * 0.92)))
    ms.append(B(x + r, x + r + 1.6, YF, spring + r + 1.2, zf - 1.2, zf + 0.5, tuple(ASHLAR * 0.92)))
    ms.append(B(x - r, x + r, spring + r + 0.1, spring + r + 1.2, zf - 1.2, zf + 0.5, tuple(ASHLAR * 0.92)))
    return ms


def turret():
    """The round turret at the east end of the downstream face."""
    x, z = TURRET
    ms = [cyl(3.0, 2.8, YC + 1.6 - YF, tuple(ASHLAR), seg=14).T(x, YF, z).paint(ashlar(ASHLAR, 51, bands=((5.6, 6.6),)))]
    ms.append(cyl(3.25, 3.25, 0.45, tuple(np.clip(ASHLAR * 1.05, 0, 1)), seg=14).T(x, YC + 1.6, z))
    return ms


# ── Trees in the gorge below ─────────────────────────────────────────────────
def sphere(r, c, seg=10, rings=6):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def blob(cx, cy, cz, r, flat, col, seed):
    m = sphere(r, col).S(1, flat, 1)
    crumple(m, r * 0.3, 1.3 / r, seed)
    m.T(cx, cy, cz)
    c = np.asarray(col, float)
    lo = cy - r * flat

    def tint(v):
        t = np.clip((v[:, 1] - lo) / (2 * r * flat + 1e-6), 0, 1)
        k = 0.66 + 0.44 * t + 0.16 * (noise3(v, 0.9, seed) - 0.5)
        return np.clip(np.outer(k, c), 0, 1)
    return m.paint(tint)


def tree(x, y, z, r, h, seed):
    ms = [B(x - 0.25, x + 0.25, y - 0.5, y + h * 0.55, z - 0.25, z + 0.25, (0.32, 0.27, 0.21))]
    ms.append(blob(x, y + h * 0.7, z, r, h / (2.2 * r), LEAF, seed))
    ms.append(blob(x + r * 0.4, y + h * 0.55, z - r * 0.3, r * 0.65, 0.9, LEAF, seed + 1))
    return ms


def trees():
    ms = []
    spots = [(-8, 30), (4, 36), (14, 33), (-14, 44), (8, 50), (22, 42), (-30, -34), (-18, -40), (10, -38), (26, -32)]
    for i, (x, z) in enumerate(spots):
        y = float(ground(x, z))
        ms += tree(x + rng.uniform(-1, 1), y, z + rng.uniform(-1, 1), rng.uniform(2.2, 3.4), rng.uniform(6, 10), 700 + i)
    return ms


# ── The gorge (rock skirt) ───────────────────────────────────────────────────
def gorge():
    xs = xs_between(-96, 96, 4.0)
    zs = xs_between(-60, 64, 4.0)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    G = ground(X, Z)
    P0 = np.stack([X, G, Z], -1).reshape(-1, 3)
    rough = (fbm(P0, 0.09, 101, 3) - 0.5).reshape(X.shape)
    Y = G + 0.5 + 2.0 * rough
    P = np.stack([X, Y, Z], -1)
    nu, nv = X.shape[0] - 1, X.shape[1] - 1
    idx = np.arange(X.size).reshape(X.shape)

    def wanted(i, j):                                    # only the gorge round the dam
        g = G[i:i + 2, j:j + 2].mean()
        return g < YC + 14
    faces, used, edge = [], np.zeros(X.shape, bool), np.zeros(X.shape, bool)
    for i in range(nu):
        for j in range(nv):
            if not wanted(i, j):
                edge[i:i + 2, j:j + 2] = True
                continue
            used[i:i + 2, j:j + 2] = True                # kept under the dam too: it closes the bottom
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            faces += [(a, b, c), (a, c, d)]
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    rim = used & edge
    P[..., 1] = np.where(rim, P[..., 1] - 14.0, P[..., 1])   # the rim dips under the ground (a rock skirt)
    m = Mesh(P.reshape(-1, 3), np.array(faces), ROCK)
    m = orient(m, (0, 1, 0))
    n = np.zeros_like(m.v)
    fa, fb, fc = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    fn = np.cross(fb - fa, fc - fa)
    for k in range(3):
        np.add.at(n, m.f[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

    def paint(v):
        steep = np.clip((0.9 - n[:, 1]) / 0.35, 0, 1)                    # bare limestone on the steep sides
        slab = np.clip((fbm(v, 0.05, 107, 3) - 0.45) * 3, 0, 1)
        r = np.clip(steep + 0.6 * slab, 0, 1)[:, None]
        c = SOIL * (1 - r) + ROCK * r
        c = c * (1 + 0.22 * (fbm(v, 0.2, 109, 2) - 0.5))[:, None]
        sc = np.clip((fbm(v, 0.07, 113, 3) - 0.5) * 3.5, 0, 1)[:, None] * (1 - 0.6 * r)
        c = c * (1 - sc) + SCRUB * sc
        bed = np.clip(1 - (np.abs(v[:, 0] + 2 - 0.25 * (v[:, 2] - 18)) - 4) / 5, 0, 1)
        bed = (bed * (v[:, 1] < 2.5) * (v[:, 2] > 17))[:, None]          # the stony river bed below the dam
        c = c * (1 - bed * 0.5) + np.array((0.62, 0.60, 0.55)) * bed * 0.5
        return np.clip(c, 0, 1)
    m.paint(paint)
    keep = np.unique(m.f)                                 # drop the vertices no face uses
    remap = np.full(len(m.v), -1)
    remap[keep] = np.arange(len(keep))
    return [Mesh(m.v[keep], remap[m.f], m.c[keep])]


def footprint():
    """Seen from above: the dam and its tower with 2 m round it (the gorge around keeps its trees)."""
    zd = PROFILE[0][0] + 2
    return [[-48.0, round(zd, 1)], [-48.0, -10.0], [-44.0, -17.0], [-30.0, -18.1], [30.0, -18.1], [42.0, -10.0],
            [48.0, 2.0], [50.0, round(zd, 1)]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dam = merge(dam_body() + crest_works() + tower() + tunnel_mouth() + turret())
    tr = merge(trees())
    rock = merge(gorge())
    tris, size = write_glb(OUT / "model.glb", [("presa_del_ponton_de_la_oliva", dam), ("arboles", tr), ("roca", rock)],
                           roughness=0.92)
    allv = np.vstack([dam.v, tr.v, rock.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    print("dam:", dam.tris, "trees:", tr.tris, "rock:", rock.tris)
    print("footprint:", json.dumps(footprint()))
    print("ground below the dam (0, 24):", float(ground(0.0, 24.0)) + E0)
    print("ends: ground at x=±50:", [round(float(ground(s * 50.0, z)), 1) for s in (-1, 1) for z in (-14, 0, 14)])


if __name__ == "__main__":
    main()
