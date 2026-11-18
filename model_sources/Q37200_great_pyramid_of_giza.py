"""
Great Pyramid of Giza (Q37200) — with the rest of the Giza plateau a flight would visit, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monuments: the pyramids of Khufu (architect traditionally Hemiunu, c. 2560 BC), Khafre and Menkaure, their
queens' pyramids, temples and causeway, and the Great Sphinx (4th Dynasty, c. 2600–2500 BC). Public domain.

What is modelled:
- The pyramid of Khufu: its stepped core courses (the casing is gone), worn corners, the flat top where the
  capstone is missing with the surveyors' mast that marks the original height, the original entrance on the
  north face with its gable of chevron blocks, the robbers' tunnel of al-Ma'mun below it, the remains of the
  limestone pavement around the base and the basalt pavement of its mortuary temple on the east side.
- The pyramid of Khafre: stepped core below, the smooth, lighter casing that still covers its top third with a
  ragged lower edge, the lowest courses on the bedrock platform 10 m above Khufu's base; its mortuary temple
  and its causeway down to the Valley Temple (megalithic limestone walls, the T-shaped hall with red granite
  pillars and architraves, alabaster floor) and the Sphinx Temple in front of the Sphinx.
- The pyramid of Menkaure: stepped core, the red granite casing courses that remain low on its north and east
  faces, the great vertical gash in its north face (al-Aziz Uthman, 1196), its mortuary temple.
- The queens' pyramids: G1-a, G1-b, G1-c east of Khufu (ruined, stepped cores) and G3-a (true pyramid), G3-b and
  G3-c (four-stage step pyramids) south of Menkaure.
- The Great Sphinx in its rock-cut enclosure: lion body with haunches, tail, forelegs and paws, the head with
  the nemes headdress and its lappets, the face; the softer middle layers more eroded, the restoration blocks
  along the base.
- Not modelled: the terrain itself, the mastaba fields, the boat pits, modern buildings, the Khentkawes tomb.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap ways 4420397 (Khufu), 4420396 (Khafre), 4420398 (Menkaure), 198032492 / 219797724 /
  219797726 (G1-a/b/c), 25416060 / 25416067 / 25416093 (G3-a/b/c), 540058405 (+ parts 540058406/7, 24585237,
  Great Sphinx), 1238523832 (Valley Temple of Khafre), 1238523833 (Sphinx Temple), 97577625 and 251168393
  (mortuary temples of Khafre and Menkaure). The pyramids are placed on the centres of their OSM outlines and
  drawn as true squares of their published sizes, aligned to the cardinal points (Khufu's sides are off by
  only ~3′ of arc).
- Khufu: base 230.33 m, slope 51°50′40″ (tan 1.2727), height today 138.5 m (146.6 m originally), ~200 courses,
  original entrance 17 m up and 7.3 m east of the centre line (en.wikipedia).
- Khafre: base 215.25 m, slope 53°10′, height today 136.4 m (143.5 m), casing on the top third, bedrock 10 m
  higher than Khufu's; causeway 494.6 m (en.wikipedia).
- Menkaure: base 102.2 × 104.6 m, slope 51°20′, height today 61 m (65.5 m), 16 courses of granite casing of
  which seven remain (en.wikipedia); size of the gash estimated from photographs (±30 %).
- Queens: G1-a 49.5 m base, 30.25 m originally, two thirds of it lost (en.wikipedia); G1-b 49 m and G1-c
  46.25 m bases, heights today estimated (±30 %); G3-a 44 m base (84 cubits), 25.4 m today; G3-b/G3-c from the
  OSM outlines (~31 m), four stages, heights estimated (±30 %).
- Sphinx: 73 m long, 20 m high, 19 m wide at the haunches, facing east, ~3 m between the paws (en.wikipedia);
  the shape of the head and the enclosure (rock walls 7–10 m high) estimated from photographs and plans (±15 %).
  The Valley Temple walls ~12 m high, the Sphinx Temple lower (estimated, ±20 %).
- Ground: SRTM 30 m and ASTER 30 m (via OpenTopoData), read away from the monuments because both are surface
  models. y = 0 is Khufu's base (≈ 60 m above sea level); Khafre's base +10, Menkaure's +11, G3 queens +7,
  G1 queens −1, the causeway falls from +9.5 to −38.5, the Sphinx enclosure floor and the temples −40
  (≈ 20 m a.s.l.). Every base goes down 6–8 m below its own level so no gap shows on the uneven rock.

Axes (docs/MODEL_SPEC.md): +Y up, the north face of Khufu (with the entrance) faces +Z (heading 0), +X points
west (to Khafre and Menkaure, which lie to the south-west), the Sphinx is ~330 m east and ~430 m south (x −326,
z −431). Origin on the ground at the centre of Khufu's base.

    python model_sources/Q37200_great_pyramid_of_giza.py   -> models/Q37200/model.glb (+ footprint in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, cap, walls, extrude, offset, fbm, crumple, write_glb  # noqa: E402

rng = np.random.default_rng(2560)

# ── Colours ──────────────────────────────────────────────────────────────────
LIME = (0.80, 0.65, 0.43)              # weathered golden core limestone
LIME_D = (0.71, 0.56, 0.37)            # shadowed, fresher breaks
LEDGE = (0.85, 0.73, 0.53)             # tops of the courses, with sand on them
CASING = (0.90, 0.84, 0.71)            # Tura limestone casing at Khafre's top
PAVE = (0.83, 0.77, 0.64)              # limestone pavement
GRANITE = (0.58, 0.38, 0.33)           # red Aswan granite
BASALT = (0.26, 0.26, 0.27)
ALABASTER = (0.90, 0.87, 0.80)
ROCK = (0.76, 0.63, 0.45)              # cut bedrock of the Sphinx enclosure
ROCK_D = (0.66, 0.53, 0.37)
RESTORE = (0.86, 0.78, 0.62)           # modern restoration blocks around the Sphinx
DARK = (0.07, 0.06, 0.05)
WOOD = (0.30, 0.24, 0.18)

KHUFU = dict(c=(0.0, 0.0), y0=0.0, half=230.33 / 2, tan=1.2727, h=138.5)
KHAFRE = dict(c=(335.45, -352.4), y0=10.0, half=215.25 / 2, tan=1.3365, h=136.4)
MENKAURE = dict(c=(574.35, -738.2), y0=11.0, half=(102.2 / 2, 104.6 / 2), tan=1.2495, h=61.0)

SPHINX_C = (-326.4, -430.9)            # centre of the OSM outline (paws to tail along −x/+x)
FLOOR = -40.0                          # floor of the Sphinx enclosure and of the temples


def yaw_box(w, h, d, c, x, y, z, ang):
    """box w (along its own x) × h × d standing on y, rotated by ang (radians, three.js about +y) and placed."""
    return box(w, h, d, c).R(0, ang, 0).T(x, y, z)


# ── Stepped pyramids ─────────────────────────────────────────────────────────
def courses(total, h_bottom, h_top, jitter, seed):
    r = np.random.default_rng(seed)
    ys, y = [0.0], 0.0
    while y < total - 1e-6:
        t = y / total
        hk = max(0.3, h_bottom + (h_top - h_bottom) * t + r.uniform(-jitter, jitter))
        y = min(total, y + hk)
        if total - y < 0.25:
            y = total
        ys.append(y)
    return ys


def stepped(p, seed, inset=0.0, top=None, chamfer=1.5, rough=0.25, ch=(1.2, 0.6), jitter=0.15,
            notch=None, colour=None, found=7.0, h_cap=None):
    """Stepped core of a pyramid: one ring per course (8 chamfered corners + a notch on the north side), its
    vertical face and the ledge on top; every course set in by its rise / tan(slope) (+ inset from the face).
    notch(y) -> (x_centre, width, depth) on the +z face; colour(x, y, z, side, k) -> rgb or None."""
    r = np.random.default_rng(seed)
    cx, cz = p["c"]
    hx, hz = (p["half"] if isinstance(p["half"], tuple) else (p["half"], p["half"]))
    ys = courses(p["h"] if top is None else top, ch[0], ch[1], jitter, seed)
    nc = len(ys) - 1
    rings, cols = [], []
    for k in range(nc):
        yt = ys[k + 1]
        ax, az = hx - inset - yt / p["tan"], hz - inset - yt / p["tan"]
        d = r.uniform(0, rough, 4)                      # per side: worn back a little
        e, n_, w_, s_ = ax - d[0], az - d[1], ax - d[2], az - d[3]   # +x, +z, −x, −z extents
        c = np.clip(chamfer * (0.5 + r.uniform(0, 1, 4)), 0.2, min(ax, az) * 0.45)
        nx, nw, nd = notch(yt) if notch else (0.0, 0.0, 0.0)
        nw = min(nw, 2 * ax - 3 * c.max())
        P = [(e, -s_ + c[0]), (e, n_ - c[1]), (e - c[1], n_)]
        # the notch on the +z side, running from +x to −x
        P += [(nx + nw / 2, n_), (nx + nw / 2 - nd * 0.4, n_ - nd), (nx - nw / 2 + nd * 0.4, n_ - nd), (nx - nw / 2, n_)]
        P += [(-w_ + c[2], n_), (-w_, n_ - c[2]), (-w_, -s_ + c[3]), (-w_ + c[3], -s_), (e - c[0], -s_)]
        # every edge split in three, the long ones worn back unevenly (blocks missing, broken edges)
        Q = []
        for i in range(len(P)):
            a, b = np.array(P[i]), np.array(P[(i + 1) % len(P)])
            L = float(np.hypot(*(b - a)))
            Q.append(tuple(a))
            for t in (r.uniform(0.2, 0.45), r.uniform(0.55, 0.8)):
                q = a + (b - a) * t
                if L > 6:
                    nn = np.array([(b - a)[1], -(b - a)[0]]) / L       # outward for a counter-clockwise ring
                    q = q - nn * r.uniform(0, rough * 2.5) * (r.random() < 0.6)
                Q.append(tuple(q))
        rings.append(np.array(Q))
    N = len(rings[0])
    V, Fc, C = [], [], []

    def col_at(x, y, z, side, k, base):
        if colour:
            cc = colour(x, y, z, side, k)
            if cc is not None:
                return cc
        return base

    for k in range(nc):
        yb = (p["y0"] - found) if k == 0 else p["y0"] + ys[k]
        yt = p["y0"] + ys[k + 1]
        R = rings[k]
        shade = 0.9 + 0.14 * r.random()
        for i in range(N):
            a, b = R[i], R[(i + 1) % N]
            if np.hypot(*(b - a)) < 1e-6:
                continue
            side = i
            o = len(V)
            pts = [(a[0], yb, a[1]), (b[0], yb, b[1]), (b[0], yt, b[1]), (a[0], yt, a[1])]
            V += pts
            Fc += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
            for q in pts:
                if q[1] < p["y0"] - 0.3:                   # the foundation below the base: bedrock
                    C.append(np.array(ROCK_D))
                else:
                    C.append(np.array(col_at(q[0] + cx, q[1], q[2] + cz, side, k, LIME)) * shade)
        # ledge from this ring to the next one (or the top)
        if k + 1 < nc:
            U = rings[k + 1]
            for i in range(N):
                a, b, a2, b2 = R[i], R[(i + 1) % N], U[i], U[(i + 1) % N]
                if np.hypot(*(b - a)) < 1e-6 and np.hypot(*(b2 - a2)) < 1e-6:
                    continue
                o = len(V)
                pts = [(a[0], yt, a[1]), (b[0], yt, b[1]), (b2[0], yt, b2[1]), (a2[0], yt, a2[1])]
                V += pts
                Fc += [(o, o + 2, o + 1), (o, o + 3, o + 2)]                # facing up
                for q in pts:
                    C.append(np.array(col_at(q[0] + cx, q[1], q[2] + cz, i, k, LEDGE)))
    m = Mesh(np.array(V), np.array(Fc), np.array(C))
    m.T(cx, 0, cz)
    tp = cap([tuple(q) for q in rings[-1]], p["y0"] + ys[-1], LEDGE if h_cap is None else h_cap).T(cx, 0, cz)
    return merge([m, tp]), ys


def casing(p, y_lo_corners, seed, thick=1.4, ns=18, nr=8, top=None):
    """Smooth casing on the four faces from a ragged lower edge up to the top; a lip turns in to the core below.
    y_lo_corners: the lower edge at the four corners (+x+z, −x+z, −x−z, +x−z)."""
    cx, cz = p["c"]
    H = p["h"] if top is None else top
    hh = lambda y: p["half"] - y / p["tan"]           # noqa: E731
    out = []
    corners = list(y_lo_corners)
    # faces: (+z) from corner 0 (+x+z) to 1 (−x+z); (−x) 1→2; (−z) 2→3; (+x) 3→0
    for f in range(4):
        r = np.random.default_rng(seed + f)
        ca, cb = corners[f], corners[(f + 1) % 4]
        ss = np.linspace(1, -1, ns + 1)
        ylo = ca + (cb - ca) * (1 - ss) / 2 + (1 - ss ** 2) * r.uniform(-7, 5, ns + 1) * (0.6 + 0.4 * np.sin(ss * 5 + f))
        V, Fc = [], []
        for j in range(nr + 1):
            t = j / nr
            for i, s in enumerate(ss):
                y = ylo[i] + (H - ylo[i]) * t
                V.append((s, y, 0.0))
        W = ns + 1
        for j in range(nr):
            for i in range(ns):
                a = j * W + i
                Fc += [(a, a + W + 1, a + 1), (a, a + W, a + W + 1)]      # facing out
        V = np.array(V)
        # lip: from the casing edge in to the core
        o = len(V)
        lip = [(s, ylo[i], thick) for i, s in enumerate(ss)]
        V = np.vstack([V, lip])
        for i in range(ns):
            Fc += [(o + i, i, i + 1), (o + i, i + 1, o + i + 1)]

        def place(v, f=f):
            s, y, inn = v[:, 0], v[:, 1], v[:, 2]
            h = hh(y)
            u = s * h                                   # along the face (local +x of the +z face)
            w = h - inn                                 # out from the centre
            if f == 0:
                X, Z = u, w
            elif f == 1:
                X, Z = -w, u
            elif f == 2:
                X, Z = -u, -w
            else:
                X, Z = w, -u
            return np.stack([X + cx, y + p["y0"], Z + cz], 1)
        m = Mesh(place(V), Fc, CASING)
        kk = 0.93 + 0.1 * fbm(m.v, 0.15, seed + 3, 2)
        m.c = np.clip(m.c * kk[:, None], 0, 1)
        out.append(m)
    hx = hh(H)
    out.append(box(2 * hx, 0.01, 2 * hx, CASING).T(cx, p["y0"] + H - 0.01, cz))
    return merge(out)


def ruin_pile(cx, cz, y, n, size, spread, seed, c=LIME_D):
    r = np.random.default_rng(seed)
    ms = []
    for _ in range(n):
        s = size * r.uniform(0.5, 1.2)
        ms.append(box(s * r.uniform(0.8, 1.6), s * r.uniform(0.5, 0.9), s, c)
                  .R(0, r.uniform(0, math.pi), 0).T(cx + r.normal(0, spread), y - 0.2, cz + r.normal(0, spread)))
    return ms


# ── Khufu ────────────────────────────────────────────────────────────────────
def khufu():
    def colour(x, y, z, side, k):
        # lower courses darker and patchier, the top a little lighter (sun and sand)
        return None
    core, ys = stepped(KHUFU, 1, chamfer=2.2, rough=0.35, ch=(1.45, 0.58), jitter=0.12, found=8.0)
    ms = [core]
    H, half, tan = KHUFU["h"], KHUFU["half"], KHUFU["tan"]
    face_z = lambda y: half - y / tan                  # noqa: E731
    # original entrance (17 m up, 7.3 m east of the centre line, i.e. −x) with its chevron gable
    ye, xe = 17.0, -7.3
    zf = face_z(ye + 1.5)
    ms.append(box(2.0, 2.2, 3.0, DARK).T(xe, ye, zf - 0.6))
    for sgn in (-1, 1):                                 # two pairs of great gable blocks leaning together
        for j, (lw, lh) in enumerate(((3.4, 1.1), (4.6, 1.3))):
            blk = box(lw, lh, 2.4, PAVE, bottom=False).R(0, 0, sgn * math.radians(-40))
            blk.T(xe + sgn * (1.05 + 0.55 * j), ye + 3.3 + 0.9 * j, zf + 0.3 - 0.7 * j)
            ms.append(blk)
    ms.append(box(2.6, 1.2, 2.0, PAVE).T(xe, ye - 1.0, zf + 0.6))       # sill
    # robbers' tunnel of al-Ma'mun, ~7 m up near the centre, with a ramp of rubble
    yr = 6.8
    ms.append(box(2.4, 2.4, 3.0, DARK).T(1.0, yr, face_z(yr + 1.2) - 0.4))
    ms += ruin_pile(1.0, face_z(2.0) + 3.0, 0.0, 10, 1.6, 2.2, 11)
    # surveyors' mast on the flat top marking the original height (146.6 m)
    ht = half - H / tan
    apex = (0.0, 146.6, 0.0)
    for a in (0.3, 2.4, 4.5):
        foot = (math.cos(a) * ht * 0.55, H, math.sin(a) * ht * 0.55)
        ms.append(tube([foot, apex], 0.18, WOOD, seg=5))
    # limestone pavement around the base (remains, ~10 m wide) and the basalt floor of the mortuary temple
    outer = [(half + 10, half + 10), (-half - 10, half + 10), (-half - 10, -half - 10), (half + 10, -half - 10)]
    inner = [(half, half), (-half, half), (-half, -half), (half, -half)]
    ms.append(cap(outer, 0.3, PAVE, holes=[inner]))
    ms.append(walls(outer, -8.0, 0.3, ROCK_D))
    ms.append(extrude([(-half - 10, -26), (-half - 50, -26), (-half - 50, 26), (-half - 10, 26)], -6, 0.45, BASALT))
    return merge(ms)


# ── Khafre ───────────────────────────────────────────────────────────────────
def khafre():
    core, ys = stepped(KHAFRE, 2, inset=1.3, chamfer=1.6, rough=0.3, ch=(1.3, 0.6), jitter=0.12, found=14.0)
    cs = casing(KHAFRE, (93.0, 97.0, 90.0, 95.0), 21)
    ms = [core, cs]
    # granite remains of the lowest course and the bedrock platform cut around it
    cx, cz = KHAFRE["c"]
    h = KHAFRE["half"]
    ring = [(cx + h + 6, cz + h + 6), (cx - h - 6, cz + h + 6), (cx - h - 6, cz - h - 6), (cx + h + 6, cz - h - 6)]
    inner = [(cx + h, cz + h), (cx - h, cz + h), (cx - h, cz - h), (cx + h, cz - h)]
    ms.append(cap(ring, KHAFRE["y0"] + 0.2, ROCK, holes=[inner]))
    ms.append(walls(ring, KHAFRE["y0"] - 12, KHAFRE["y0"] + 0.2, ROCK_D))
    return merge(ms)


# ── Menkaure ─────────────────────────────────────────────────────────────────
def gash(y):
    if y < 2 or y > 31:
        return (0.0, 0.0, 0.0)
    d = 5.0 * min(1.0, (y - 2) / 4) * min(1.0, (31 - y) / 7)
    w = 24.0 - 0.45 * y
    return (1.5, w, d)


def menkaure():
    cx, cz = MENKAURE["c"]
    r = np.random.default_rng(7)
    patches = r.uniform(0, 1, 64)

    def colour(x, y, z, side, k):
        lx, lz = x - cx, z - cz
        yy = y - MENKAURE["y0"]
        # inside the gash: fresher, darker stone
        if lz > 0 and abs(lx - 1.5) < 13 and 2 < yy < 31 and lz < MENKAURE["half"][1] - yy / MENKAURE["tan"] - 1.0:
            return (0.69, 0.54, 0.38)
        # the granite casing that remains on the north and east (−x) faces
        if yy < 9.0:
            north = lz > MENKAURE["half"][1] - yy / MENKAURE["tan"] - 3.5
            east = lx < -(MENKAURE["half"][0] - yy / MENKAURE["tan"] - 3.5)
            if north or east:
                u = lx if north else lz
                if patches[int((u + 60) / 3) % 64] < 0.85 - yy * 0.045:
                    return GRANITE
        return None
    core, ys = stepped(MENKAURE, 3, chamfer=1.2, rough=0.3, ch=(1.1, 0.55), jitter=0.12, notch=gash, colour=colour,
                       found=8.0)
    ms = [core]
    ms += ruin_pile(cx + 1.5, cz + MENKAURE["half"][1] + 4, MENKAURE["y0"], 18, 1.8, 4.0, 12)
    return merge(ms)


# ── Queens' pyramids ─────────────────────────────────────────────────────────
def queens():
    ms = []
    # G1 east of Khufu (OSM centres), ruined stepped cores
    for (c, half, top, seed) in (((-193.0, -33.4), 49.5 / 2, 10.5, 31),
                                 ((-193.2, -91.6), 49.0 / 2, 19.0, 32),
                                 ((-192.2, -143.1), 46.25 / 2, 22.0, 33)):
        p = dict(c=c, y0=-1.0, half=half, tan=1.27, h=top)
        core, _ = stepped(p, seed, chamfer=1.4, rough=0.6, ch=(1.2, 0.8), jitter=0.25, found=6.0, h_cap=LIME_D)
        ms.append(core)
        hw = half - top / 1.27
        ms += ruin_pile(c[0], c[1], -1.0 + top, 7, 1.5, hw * 0.35, seed)
    # G3-a, true pyramid (stepped core today)
    pa = dict(c=(570.3, -836.5), y0=7.0, half=44.0 / 2, tan=1.29, h=25.4)
    core, _ = stepped(pa, 41, chamfer=0.9, rough=0.3, ch=(1.0, 0.6), jitter=0.12, found=6.0)
    ms.append(core)
    # G3-b and G3-c: four-stage step pyramids, G3-c worse kept
    for c, half, stages, sh, seed in (((618.0, -835.6), 31.0 / 2, 4, 3.4, 42), ((663.3, -837.2), 29.0 / 2, 3, 3.3, 43)):
        p = dict(c=c, y0=7.0, half=half, tan=sh / 2.4, h=stages * sh)
        core, _ = stepped(p, seed, chamfer=0.8, rough=0.5, ch=(sh, sh), jitter=0.0, found=6.0)
        ms.append(core)
        ms += ruin_pile(c[0], c[1], 7.0 + stages * sh, 5, 1.2, 2.5, seed)
    return merge(ms)


# ── Temples and causeway ─────────────────────────────────────────────────────
def ruin_walls(ring, thick, y0, y_lo, y_hi, seed, c=LIME, seg=6.0, found=4.0):
    """Ruined walls along a ring: blocks of random height, set inside the outline."""
    r = np.random.default_rng(seed)
    P = np.array(ring, float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    ctr = P.mean(0)
    ms = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        L = float(np.hypot(*(b - a)))
        if L < 0.5:
            continue
        d = (b - a) / L
        nrm = np.array([-d[1], d[0]])
        if np.dot(nrm, ctr - (a + b) / 2) < 0:
            nrm = -nrm
        n = max(1, int(round(L / seg)))
        ang = math.atan2(-d[1], d[0])
        for j in range(n):
            m = a + d * (L * (j + 0.5) / n) + nrm * thick / 2
            h = r.uniform(y_lo, y_hi)
            ms.append(yaw_box(L / n + 0.3, h + found, thick, c, m[0], y0 - found, m[1], ang))
    return ms


def osm_ring(name):
    return [tuple(p) for p in PLAN[name]]


def valley_temple():
    ms = []
    ring = osm_ring("valley_khafre")
    ms += ruin_walls(ring, 9.0, FLOOR, 10.8, 12.4, 51, seg=9.0)
    P = np.array(ring)
    x0, x1, z0, z1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    ms.append(box(x1 - x0, 0.3, z1 - z0, ALABASTER).T((x0 + x1) / 2, FLOOR - 0.1, (z0 + z1) / 2))
    cxm, czm = (x0 + x1) / 2, (z0 + z1) / 2
    # T-shaped hall: transverse arm N–S near the east side (6 pillars), deep arm westwards (2 × 5 pillars)
    pil = []
    xt = cxm - 5.0
    for k in range(6):
        pil.append((xt, czm - 9.0 + k * 3.6))
    for k in range(5):
        for zz in (czm - 2.0, czm + 2.0):
            pil.append((xt + 4.0 + k * 3.2, zz))
    for (px, pz) in pil:
        ms.append(box(1.05, 4.1, 1.05, GRANITE).T(px, FLOOR, pz))
    ms.append(box(1.2, 0.9, 19.2, GRANITE).T(xt, FLOOR + 4.1, czm - 9.0 + 9.0))
    for zz in (czm - 2.0, czm + 2.0):
        ms.append(box(14.0, 0.9, 1.2, GRANITE).T(xt + 4.0 + 6.4, FLOOR + 4.1, zz))
    # granite cladding of the inner faces of the hall walls (a band)
    ms.append(box(1.0, 5.0, 22.0, GRANITE).T(xt - 2.0, FLOOR, czm))
    return ms


def sphinx_temple():
    ms = []
    ring = osm_ring("sphinx_temple")
    ms += ruin_walls(ring, 6.0, FLOOR, 6.0, 7.6, 52, seg=8.0)
    P = np.array(ring)
    x0, x1, z0, z1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    cxm, czm = (x0 + x1) / 2, (z0 + z1) / 2
    ms.append(box(x1 - x0 - 2, 0.3, z1 - z0 - 2, PAVE).T(cxm, FLOOR - 0.1, czm))
    # the open court with its piers (stumps)
    r = np.random.default_rng(53)
    for i in range(6):
        for zz in (czm - 11.0, czm + 11.0):
            ms.append(box(1.4, r.uniform(1.2, 4.5), 1.4, GRANITE).T(cxm - 10 + i * 4.0, FLOOR, zz))
    for i in range(4):
        for xx in (cxm - 14.0, cxm + 12.0):
            ms.append(box(1.4, r.uniform(1.2, 4.5), 1.4, GRANITE).T(xx, FLOOR, czm - 6 + i * 4.0))
    return ms


def mortuary_temples():
    ms = []
    ring = osm_ring("mortuary_khafre")
    ms += ruin_walls(ring, 7.0, KHAFRE["y0"], 2.5, 6.0, 61, seg=8.0, found=10.0)
    P = np.array(ring)
    x0, x1, z0, z1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    ms.append(box(x1 - x0 - 2, 0.3, z1 - z0 - 2, PAVE).T((x0 + x1) / 2, KHAFRE["y0"] - 0.1, (z0 + z1) / 2))
    # inner walls (rooms and the court), lower
    ms += ruin_walls([(x0 + 25, z0 + 10), (x1 - 30, z0 + 10), (x1 - 30, z1 - 10), (x0 + 25, z1 - 10)], 3.0,
                     KHAFRE["y0"], 1.0, 3.0, 62, seg=6.0, found=1.0)
    ring = osm_ring("mortuary_menkaure")
    ms += ruin_walls(ring, 3.5, MENKAURE["y0"], 1.2, 3.5, 63, c=(0.66, 0.55, 0.42), seg=6.0, found=8.0)
    P = np.array(ring)
    x0, x1, z0, z1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    ms.append(box(x1 - x0 - 2, 0.3, z1 - z0 - 2, (0.74, 0.64, 0.50)).T((x0 + x1) / 2, MENKAURE["y0"] - 0.1, (z0 + z1) / 2))
    return ms


def causeway():
    """The ramp from Khafre's mortuary temple down to the Valley Temple: top 5 m wide, sides with a batter
    going 6 m down, low broken side walls."""
    pts = np.array([(107.0, -351.0, 9.5), (60.0, -363.5, 1.0), (-50.0, -392.5, -8.0), (-150.0, -419.0, -18.5),
                    (-250.0, -445.5, -29.5), (-330.0, -464.5, -36.5), (-374.0, -471.0, -38.6)])
    # resample every ~8 m along the line
    seg = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 8))
        for j in range(n):
            seg.append(a + (b - a) * j / n)
    seg.append(pts[-1])
    S = np.array(seg)
    hw, depth, bat = 2.5, 6.0, 1.5
    V, F = [], []
    for i, p in enumerate(S):
        q = S[min(i + 1, len(S) - 1)] - S[max(i - 1, 0)]
        d = np.array([q[0], q[1]]); d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        y = p[2]
        V += [(p[0] + n[0] * (hw + bat), y - depth, p[1] + n[1] * (hw + bat)),
              (p[0] + n[0] * hw, y, p[1] + n[1] * hw),
              (p[0] - n[0] * hw, y, p[1] - n[1] * hw),
              (p[0] - n[0] * (hw + bat), y - depth, p[1] - n[1] * (hw + bat))]
    V = np.array(V)
    ms = []
    for k, c in ((0, LIME_D), (1, PAVE), (2, LIME_D)):         # each strip its own vertices: sharp edges
        VV, FF = [], []
        for i in range(len(S)):
            VV += [V[4 * i + k], V[4 * i + k + 1]]
        for i in range(len(S) - 1):
            o = 2 * i
            FF += [(o, o + 1, o + 3), (o, o + 3, o + 2)]
        ms.append(Mesh(VV, FF, c))
    r = np.random.default_rng(71)
    for i in range(len(S) - 1):
        a, b = S[i], S[i + 1]
        d = np.array([b[0] - a[0], b[1] - a[1]]); L = np.linalg.norm(d); d /= L
        n = np.array([-d[1], d[0]])
        ang = math.atan2(-d[1], d[0])
        for sgn in (-1, 1):
            if r.random() < 0.55:
                mid = (a + b) / 2
                h = r.uniform(0.4, 1.6)
                ms.append(yaw_box(L * r.uniform(0.5, 1.0), h + 0.6, 0.9, LIME, mid[0] + sgn * n[0] * (hw - 0.45),
                                  mid[2] - 0.6, mid[1] + sgn * n[1] * (hw - 0.45), ang))
    return ms


# ── Sphinx ───────────────────────────────────────────────────────────────────
def loft(sections, c):
    """sections: [(u, [(w, v), …])] all with the same number of points (a closed loop each), along +u.
    Built in the sphinx's own axes (u forward, v up, w to its left); ends closed with fans."""
    K = len(sections[0][1])
    V, F = [], []
    for u, pts in sections:
        V += [(u, v, w) for (w, v) in pts]
    for i in range(len(sections) - 1):
        for k in range(K):
            j = (k + 1) % K
            a, b, a2, b2 = i * K + k, i * K + j, (i + 1) * K + k, (i + 1) * K + j
            F += [(a, b, b2), (a, b2, a2)]
    for end, rev in ((0, False), (len(sections) - 1, True)):
        u, pts = sections[end]
        o = len(V)
        V.append((u, float(np.mean([p[1] for p in pts])), float(np.mean([p[0] for p in pts]))))
        for k in range(K):
            j = (k + 1) % K
            F.append((o, end * K + j, end * K + k) if not rev else (o, end * K + k, end * K + j))
    return Mesh(V, np.array(F)[:, ::-1], c)                # wound outwards


def sect(a, b, v0=0.0, n=3.0, K=16, wc=0.0):
    """Cross-section: flat bottom at v0, a rounded (superellipse) top of half-width a and top height b."""
    pts = []
    for i in range(K):
        t = math.pi * i / (K - 1)
        ct, st = math.cos(t), math.sin(t)
        w = a * math.copysign(abs(ct) ** (2 / n), ct)
        v = v0 + (b - v0) * abs(st) ** (2 / n)
        pts.append((wc + w, v))
    return pts


def sphinx():
    parts = []
    # body: tail end at u = −37, chest at u ≈ +9; haunches 19 m across
    body = [(-37.0, 5.0, 4.5), (-35.5, 8.0, 8.0), (-32.0, 9.4, 10.2), (-27.0, 9.5, 10.8), (-21.0, 8.3, 10.9),
            (-13.0, 7.7, 11.0), (-5.0, 7.8, 11.6), (1.0, 8.2, 12.6), (5.0, 7.9, 13.0), (8.0, 6.6, 12.4),
            (9.6, 4.6, 10.6)]
    parts.append(("body", loft([(u, sect(a, b)) for u, a, b in body], ROCK)))
    # haunches and the hind legs folded under
    for s in (-1, 1):
        parts.append(("hip", lathe([(0.01, -3.2), (2.2, -2.6), (3.0, -1.0), (3.1, 0.6), (2.4, 2.2), (0.01, 3.0)],
                                   ROCK, seg=14).S(2.1, 1.25, 1.0).R(0, 0, math.pi / 2)
                      .S(2.3, 1.0, 1.0).T(-27.5, 3.6, s * 8.4)))
        parts.append(("hindpaw", loft([(u, sect(1.6, 1.8, wc=s * 8.9)) for u in (-23.0, -14.0, -13.2)], ROCK)))
        # forelegs and paws, ~3 m between them
        wc = s * 3.6
        parts.append(("paw", loft([(4.0, sect(2.3, 3.4, wc=wc)), (12.0, sect(2.3, 3.0, wc=wc)),
                                   (34.0, sect(2.25, 2.8, wc=wc)), (36.3, sect(2.0, 2.3, wc=wc)),
                                   (37.0, sect(1.3, 1.4, wc=wc))], ROCK)))
        parts.append(("leg", loft([(2.0, sect(2.6, 9.0, wc=s * 5.2)), (6.0, sect(2.4, 8.0, wc=s * 4.6)),
                                   (10.0, sect(2.3, 4.0, wc=s * 4.0))], ROCK)))
    # tail curled along the right (south, +w here) haunch
    parts.append(("tail", tube([(-36.8, 0.8, 3.0), (-36.0, 0.7, 6.8), (-33.0, 0.6, 9.4), (-28.0, 0.6, 10.1),
                                (-24.5, 1.4, 10.2)], 0.55, ROCK, seg=6)))
    # nemes headdress with its lappets, and the neck
    def nemes(top, flap_w, flap_lo):
        return [(flap_w, flap_lo), (flap_w * 0.95, flap_lo + 2.8), (flap_w * 0.7, 15.6), (3.4, 18.6), (2.6, 19.6),
                (0.0, top), (-2.6, 19.6), (-3.4, 18.6), (-flap_w * 0.7, 15.6), (-flap_w * 0.95, flap_lo + 2.8),
                (-flap_w, flap_lo), (0.0, flap_lo - 0.3)]
    parts.append(("nemes", loft([(2.5, nemes(17.0, 3.6, 12.5)), (5.0, nemes(19.6, 4.6, 12.0)),
                                 (8.8, nemes(20.0, 4.9, 11.6)), (10.4, nemes(19.8, 4.8, 11.3)),
                                 (10.9, nemes(19.4, 4.5, 11.3))], ROCK)))
    parts.append(("neck", loft([(3.0, sect(3.4, 15.0, v0=9.0)), (9.0, sect(3.2, 14.5, v0=9.0)),
                                (10.6, sect(2.6, 13.0, v0=9.5))], ROCK)))
    # the face, set in the headdress
    face = loft([(10.2, sect(2.1, 18.3, v0=12.3, n=4)), (11.7, sect(2.05, 18.1, v0=12.4, n=4)),
                 (12.3, sect(1.6, 17.6, v0=12.8, n=4))], (0.80, 0.62, 0.47))
    parts.append(("face", face))
    feat = [box(0.25, 0.4, 1.0, DARK).T(12.3, 16.0, -1.0), box(0.25, 0.4, 1.0, DARK).T(12.3, 16.0, 1.0),
            box(0.25, 0.25, 2.0, DARK).T(12.25, 13.6, 0.0), box(0.6, 0.9, 1.0, (0.70, 0.55, 0.42)).T(12.3, 14.4, 0.0)]
    parts.append(("features", merge(feat)))
    m = merge([p[1] for p in parts])
    # erosion: the soft middle layers gnawed, the head harder
    wgt = np.clip(1.0 - np.abs(m.v[:, 1] - 7.0) / 6.0, 0.15, 1.0) * (m.v[:, 1] < 12.0)
    crumple(m, 0.35, 0.35, 99, weight=wgt)
    v = m.v[:, 1]
    band = 0.5 + 0.5 * np.sin(v * 5.2)
    base = np.where((v < 4.5)[:, None], np.array((0.74, 0.65, 0.50)), np.array((0.76, 0.61, 0.42)))
    base = base * (0.9 + 0.12 * band[:, None] * ((v > 4.5) & (v < 12))[:, None])
    head = (v >= 12.0) & (m.v[:, 0] > 2.0)
    base = np.where(head[:, None], np.array((0.82, 0.69, 0.49)), base)
    base = np.where((v < 2.4)[:, None], np.array(RESTORE) * 0.55 + base * 0.45, base)
    keep = np.all(np.abs(m.c - np.array(ROCK)) < 1e-6, axis=1)
    m.c = np.where(keep[:, None], base, m.c)
    # to the model: u forward = east = −x, w → −z (a rotation, not a mirror), floor of the enclosure
    xs, zs = SPHINX_C
    m.v = np.stack([xs - m.v[:, 0], FLOOR + m.v[:, 1], zs - m.v[:, 2]], 1)
    return m


def sphinx_enclosure():
    """The rock-cut ditch round the Sphinx: floor, and walls of cut bedrock on the north, south and west."""
    x_e, x_w, z_s, z_n = -366.0, -279.0, -457.0, -408.0
    ms = [box(x_w - x_e, 2.0, z_n - z_s, (0.80, 0.70, 0.53)).T((x_e + x_w) / 2, FLOOR - 2.05, (z_n + z_s) / 2)]
    r = np.random.default_rng(81)
    t = 5.0
    # north wall (inner face at z_n) and south wall, in pieces of varying top (the rim is ragged)
    for zc, rim in ((z_n + t / 2, -32.5), (z_s - t / 2, -33.5)):
        n = 10
        for i in range(n):
            x0 = x_e + 8 + (x_w - x_e - 8) * i / n
            x1 = x_e + 8 + (x_w - x_e - 8) * (i + 1) / n
            top = rim + r.uniform(-0.4, 0.4) + 2.0 * (x0 - x_e) / (x_w - x_e)
            ms.append(box(x1 - x0 + 0.2, top - (FLOOR - 2), t, ROCK).T((x0 + x1) / 2, FLOOR - 2, zc))
    for i in range(6):
        z0 = z_s - t + (z_n - z_s + 2 * t) * i / 6
        z1 = z_s - t + (z_n - z_s + 2 * t) * (i + 1) / 6
        top = -29.5 + r.uniform(-0.3, 0.3)
        ms.append(box(t, top - (FLOOR - 2), z1 - z0 + 0.2, ROCK).T(x_w + t / 2, FLOOR - 2, (z0 + z1) / 2))
    # layered colours on the cut faces
    m = merge(ms)
    v = m.v[:, 1]
    k = 0.92 + 0.1 * np.sin(v * 2.6) + 0.05 * fbm(m.v, 0.2, 5, 2)
    m.c = np.clip(m.c * k[:, None], 0, 1)
    return m


# ── Plan data (OSM, projected into the model axes, heading 0) ───────────────
PLAN = {   # ways 1238523832, 1238523833, 97577625, 251168393 (small kinks of < 2 m left out)
    "valley_khafre": [(-415.9, -469.0), (-415.6, -478.9), (-415.0, -506.2), (-414.8, -514.6), (-371.9, -513.6),
                      (-372.7, -478.2), (-373.0, -467.9)],
    "sphinx_temple": [(-414.6, -412.2), (-414.6, -433.2), (-414.7, -464.3), (-370.3, -461.5), (-371.2, -416.8),
                      (-375.9, -416.4), (-380.1, -412.6)],
    "mortuary_khafre": [(215.8, -330.3), (107.3, -329.6), (107.0, -374.6), (215.5, -375.3)],
    "mortuary_menkaure": [(510.1, -712.4), (456.4, -712.2), (456.3, -732.2), (424.6, -732.1), (424.6, -744.3),
                          (455.5, -744.4), (455.7, -759.1), (457.9, -762.0), (510.0, -762.3), (510.1, -744.5),
                          (526.9, -744.6), (526.8, -721.4), (510.1, -721.4)],
}


FOOTPRINT = [(-230.0, 140.0), (130.0, 140.0), (460.0, -228.0), (650.0, -662.0), (692.0, -870.0), (540.0, -872.0),
             (418.0, -772.0), (220.0, -485.0), (100.0, -392.0), (-360.0, -520.0), (-420.0, -522.0), (-422.0, -404.0),
             (-372.0, -398.0), (-276.0, -400.0), (-230.0, -190.0)]


def build():
    return [("khufu", khufu()), ("khafre", khafre()), ("menkaure", menkaure()), ("queens", queens()),
            ("temples", merge(valley_temple() + sphinx_temple() + mortuary_temples())),
            ("causeway", merge(causeway())), ("sphinx", sphinx()), ("enclosure", sphinx_enclosure())]


def main():
    parts = build()
    out_dir = ROOT / "models" / "Q37200"
    out_dir.mkdir(parents=True, exist_ok=True)
    named = []
    for name, m in parts:
        k = 0.94 + 0.08 * fbm(m.v, 0.06, 7, 2)          # a little variation in the stone
        if name in ("khufu", "khafre", "menkaure", "queens"):
            k = k * (0.88 + 0.2 * fbm(m.v, 0.025, 19, 3))   # weathered patches on the faces
        m.c = np.clip(m.c * k[:, None], 0, 1)
        named.append((name, m))
    tris, size = write_glb(out_dir / "model.glb", named, roughness=0.92)
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        mj["footprint"] = [[round(x, 1), round(z, 1)] for x, z in FOOTPRINT]
        meta.write_bytes((json.dumps(mj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"{tris} triangles, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
