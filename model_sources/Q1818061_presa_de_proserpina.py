"""
Presa de Proserpina (Embalse de Proserpina), Mérida (Q1818061) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: the dam of the Roman reservoir of Proserpina, 5 km north of Mérida, on the Arroyo de las Pardillas
(Roman in origin, raised and repaired in late-antique, medieval and 17th-century works); a masonry screen wall held
by an earth embankment on its downstream side. Public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 509714147 (waterway=dam, historic=dam): the water face of the wall, 417.5 m in four straight
  stretches with a bend of about 20° near the middle and one of 5° further north (the 160° and 175° of the USBR
  plan: 227.9 + 106.5 + 92.3 m). The water lies to the east: the water face is the main face (+Z, 96.3°), +X runs
  north along the dam.
- Wikipedia (es): 427.8 m long, 21.6 m high over the foundations; a screen wall of granite ashlar with a stepped
  batter on the water side, nine buttresses of rectangular section, also stepped, against that face; on the other
  side a large earth embankment («espaldón») that takes the push of the water; two intake towers on the downstream
  face of the wall.
- U.S. Bureau of Reclamation, «Proserpina Dam, plan and cross section» (Commons «Proserpina Dam.png»): 19 m high,
  2.30 m at the top, about 5 m at the base, the water face battered 1:10, the buttresses topped by pinnacles above
  the crest, the earthfill against the downstream face, the two intake towers with their control accesses.
- PNOA orthophoto: the crest along the water, the embankment 30–45 m wide behind it with paths and trees, the tops
  of the two towers just behind the wall (at x ≈ +156 and −48) and the tops of the buttresses about 45 m apart.
- IGN MDT05 (terrain.json): the reservoir at 244.0 m, the crest at 245.5 m, the embankment from 244.5 m at the wall
  down to 228 m at its foot 60 m downstream; the surface of the embankment in this model follows it.

Axes (docs/MODEL_SPEC.md): +Y up, y = 0 the ground at the origin (244.0 m: the top of the embankment, the point
given for the guided tour, 25 m behind the water face); +X north along the dam, +Z towards the water.

    python model_sources/Q1818061_presa_de_proserpina.py   -> models/Q1818061/model.glb
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
from meshkit import Mesh, merge, box, cyl, triangulate, fbm, noise3, write_glb  # noqa: E402

OUT = ROOT / "models" / "Q1818061"
HEADING = 96.3
E0 = 244.0                                   # ground at the origin (m a.s.l.) = y 0; also the water in MDT05
YC = 1.5                                     # the crest (245.5 m)
TOP_W = 2.3                                  # thickness at the top (USBR)
BATTER = 0.10                                # the water face, 1:10, in steps
STEP = 1.0
BACK = 45.0                                  # the embankment modelled this far behind the wall

# the water face of the wall (OSM way 509714147) in model x/z, south to north
NODES = np.array([(-205.58, -10.87), (-151.66, -0.39), (31.03, 30.66), (156.59, 3.65), (203.25, -10.87)])
TOWERS = (-47.8, 156.2)                       # intake towers (x), on the downstream face

ASHLAR = np.array((0.66, 0.63, 0.56))         # granite ashlar
DAMP = np.array((0.40, 0.40, 0.36))
EARTH = np.array((0.66, 0.58, 0.41))          # dry grass and earth of the embankment
GREEN = np.array((0.42, 0.45, 0.28))
PAVE = np.array((0.74, 0.71, 0.64))           # the walkway on the crest
DARK = (0.10, 0.10, 0.09)
ROOF = np.array((0.55, 0.36, 0.28))


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


# ── The line of the dam: chainage s, offset n (+ towards the water) ─────────
SEG = NODES[1:] - NODES[:-1]
SLEN = np.linalg.norm(SEG, axis=1)
S0 = np.concatenate([[0.0], np.cumsum(SLEN)])
L = float(S0[-1])
TAN = SEG / SLEN[:, None]
NRM = np.stack([-TAN[:, 1], TAN[:, 0]], 1)                     # (−dz, dx): towards +Z, the water


def mitres():
    M = [NRM[0]]
    for k in range(1, len(NODES) - 1):
        b = NRM[k - 1] + NRM[k]
        b = b / np.linalg.norm(b)
        M.append(b / float(np.dot(b, NRM[k])))
    M.append(NRM[-1])
    return np.array(M)


MIT = mitres()


def frame(s):
    """Point on the water face line at chainage s and its offset vector (dot with the segment normal = 1)."""
    s = float(np.clip(s, 0, L))
    k = min(int(np.searchsorted(S0, s, side="right")) - 1, len(SLEN) - 1)
    u = (s - S0[k]) / SLEN[k]
    p = NODES[k] + TAN[k] * (s - S0[k])
    o = MIT[k] * (1 - u) + MIT[k + 1] * u
    return p, o, TAN[k], NRM[k]


def at(s, n):
    p, o, _, _ = frame(s)
    q = p + o * n
    return float(q[0]), float(q[1])


def s_of_x(x):
    """Chainage of the point of the line whose x is closest to x (the line runs along x)."""
    ss = np.linspace(0, L, 2000)
    xs = np.array([frame(s)[0][0] for s in ss])
    return float(ss[np.argmin(np.abs(xs - x))])


def yfound(s):
    """Foundation: 5.5 m under the ground at the foot of the embankment, 55 m downstream (19 m of wall at the
    deepest, as on the USBR section; 21.6 m with the footing in Wikipedia)."""
    x, z = at(s, -55.0)
    return min(float(T(x, z)) - E0 - 5.5, -3.0)


# ── Colours ──────────────────────────────────────────────────────────────────
def stone(base, seed, course=0.5):
    base = np.asarray(base, float)

    def fn(v):
        n1 = fbm(v, 0.1, seed, 3) - 0.5
        n2 = noise3(v * np.array([1.0, 2.0, 1.0]), 0.8, seed + 5) - 0.5
        row = np.where(np.floor(v[:, 1] / course + 1e-6).astype(int) % 2 == 0, 1.03, 0.97)
        c = np.outer((1 + 0.18 * n1 + 0.10 * n2) * row, base)
        wet = np.clip((0.6 - v[:, 1]) / 1.5, 0, 1)[:, None] * 0.7               # below the usual water line
        return np.clip(c * (1 - wet) + DAMP * wet, 0, 1)
    return fn


def earth(v):
    n1 = fbm(v, 0.05, 71, 3) - 0.5
    g = np.clip((fbm(v, 0.03, 73, 2) - 0.45) * 3, 0, 1)[:, None] * 0.5
    c = np.outer(1 + 0.25 * n1, EARTH)
    return np.clip(c * (1 - g) + GREEN * g, 0, 1)


def orient_out(m, centre):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    nrm = np.cross(b - a, c - a)
    d = (a + b + c) / 3 - centre
    flip = (nrm * d).sum(1) < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


# ── The wall ─────────────────────────────────────────────────────────────────
def profile(yf):
    """Cross-section (n, y), counter-clockwise: the stepped water face, the foundation, the back face."""
    pts = [(-TOP_W, YC), (0.0, YC)]
    y, n = YC, 0.0
    while y - STEP > yf + 0.01:
        y -= STEP
        pts.append((n, y))
        n += BATTER * STEP
        pts.append((n, y))
    pts.append((n, yf))
    pts.append((-TOP_W, yf))
    return pts


def wall():
    """The wall lofted along the line; the foundation follows the valley (stations every ~10 m)."""
    stations = sorted(set(np.round(np.concatenate([S0, np.linspace(0, L, 42)]), 3)))
    yfs = [yfound(s) for s in stations]
    ylow = min(yfs)
    prof = profile(ylow)                                  # the same points everywhere; clamp y to the local foundation
    ms = []
    rows = []
    for s, yf in zip(stations, yfs):
        p, o, _, _ = frame(s)
        row = []
        for n, y in prof:
            yy = max(y, yf)
            nn = n if y >= yf else n
            q = p + o * nn
            row.append((q[0], yy, q[1]))
        rows.append(row)
    rows = np.array(rows)                                 # (stations, profile, 3)
    ns, npf = rows.shape[0], rows.shape[1]
    v, f = [], []
    for i in range(ns - 1):
        for j in range(npf):
            k = (j + 1) % npf
            a, b, c, d = rows[i, j], rows[i, k], rows[i + 1, k], rows[i + 1, j]
            if abs(a[1] - b[1]) < 1e-6 and abs(c[1] - d[1]) < 1e-6 and abs(a[0] - b[0]) + abs(a[2] - b[2]) < 1e-6:
                continue
            o = len(v)
            v += [a, b, c, d]
            f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    m = Mesh(v, f, ASHLAR)
    # wind outwards: each quad against the centre of its cross-section
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    nrm = np.cross(b - a, c - a)
    mid = (a + b + c) / 3
    # the centre of the section nearest to each triangle
    sec_c = rows.mean(1)                                   # (stations, 3)
    idx = np.argmin(((mid[:, None, [0, 2]] - sec_c[None, :, [0, 2]]) ** 2).sum(2), axis=1)
    cc = sec_c[idx].copy()
    cc[:, 1] = (YC + ylow) / 2
    d = mid - cc
    d[:, 1] *= 0.2
    flip = (nrm * d).sum(1) < 0
    m.f[flip] = m.f[flip][:, ::-1]
    ms.append(m.paint(stone(ASHLAR, 3)))
    # the two ends, closed
    for i in (0, ns - 1):
        poly = [(r[0], r[1], r[2]) for r in rows[i]]
        p, o, t, _ = frame(stations[i])
        uv = [(float(np.dot(np.array([r[0], r[2]]) - p, o)), r[1]) for r in poly]
        pts, tris = triangulate(uv)
        V = [(p[0] + o[0] * a_, y_, p[1] + o[1] * a_) for a_, y_ in pts]
        e = Mesh(V, tris, ASHLAR)
        e = orient_out(e, np.array([p[0] + t[0] * (1 if i == 0 else -1) * 50, (YC + ylow) / 2,
                                    p[1] + t[1] * (1 if i == 0 else -1) * 50]))
        e.f = e.f[:, ::-1]
        ms.append(e.paint(stone(ASHLAR, 4)))
    # coping on the crest and the walkway
    cs = np.linspace(0, L, 120)
    for n0, n1, y0, y1, col in ((-TOP_W - 0.05, 0.08, YC, YC + 0.25, PAVE),):
        v, f = [], []
        for s in cs:
            p, o, _, _ = frame(s)
            for n, y in ((n0, y0), (n0, y1), (n1, y1), (n1, y0)):
                q = p + o * n
                v.append((q[0], y, q[1]))
        f = []
        for i in range(len(cs) - 1):
            for j in range(3):
                a_, b_ = i * 4 + j, i * 4 + j + 1
                f += [(a_, b_, b_ + 4), (a_, b_ + 4, a_ + 4)]
        cm = Mesh(v, f, col)
        cm = orient_out(cm, np.array([0, YC - 5, -200.0]))
        ms.append(cm.paint(stone(col, 5, course=9)))
    return ms


# ── Buttresses, towers ───────────────────────────────────────────────────────
def oriented_box(s, n0, n1, y0, y1, w, col, seed):
    """A box between offsets n0..n1 (across the wall), heights y0..y1, w wide along the wall, at chainage s."""
    p, o, t, nr = frame(s)
    b = box(w, y1 - y0, n1 - n0, col).T(0, y0, (n0 + n1) / 2)
    b.R(0, math.atan2(-t[1], t[0]), 0)                        # +x onto t, +z onto the normal
    b.T(p[0], 0, p[1])
    return b.paint(stone(col, seed))


def pyramid(cx, cy, cz, w, h, col):
    v = [(cx - w / 2, cy, cz - w / 2), (cx + w / 2, cy, cz - w / 2), (cx + w / 2, cy, cz + w / 2),
         (cx - w / 2, cy, cz + w / 2), (cx, cy + h, cz)]
    f = [(0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)]
    m = Mesh(v, f, col)
    return orient_out(m, np.array([cx, cy + h / 3, cz]))


def buttresses():
    out = []
    for k in range(9):
        s = 24.0 + k * (L - 48.0) / 8
        yf = yfound(s)
        y = yf
        bands = []
        while y < YC:
            y1 = min(y + 2.0, YC + 0.6)
            if YC - y1 < 0.6:
                y1 = YC + 0.6
            bands.append((y, y1))
            y = y1
        for y0, y1 in bands:
            face = BATTER * max(YC - y1, 0)                     # the water face at the top of the band
            proj = 1.2 + 0.12 * (YC - y1)                       # the buttress steps in as it rises
            w = 3.6 + 0.04 * (YC - y1)
            out.append(oriented_box(s, -0.5, face + proj, y0, y1, w, ASHLAR * 1.02, 11 + k))
        # pinnacle over the crest (USBR): a plinth and a little pyramid
        p, o, t, nr = frame(s)
        c = p + o * 0.6
        out.append(oriented_box(s, -0.2, 1.4, YC + 0.6, YC + 1.0, 1.7, ASHLAR * 1.05, 21))
        out.append(pyramid(float(c[0]), YC + 1.0, float(c[1]), 1.2, 1.3, ASHLAR * 1.05).paint(stone(ASHLAR * 1.05, 22)))
    return out


def towers():
    out = []
    for x in TOWERS:
        s = s_of_x(x)
        yf = yfound(s)
        top = YC + 2.6
        out.append(oriented_box(s, -TOP_W - 4.6, -TOP_W + 0.1, yf, top, 4.6, ASHLAR, 31))
        out.append(oriented_box(s, -TOP_W - 4.85, -TOP_W + 0.2, top, top + 0.3, 4.9, ASHLAR * 1.05, 32))
        # a low hipped roof of tiles, a door on the downstream side
        p, o, t, nr = frame(s)
        c = p + o * (-TOP_W - 2.3)
        out.append(pyramid(float(c[0]), top + 0.3, float(c[1]), 4.6, 1.2, ROOF).paint(stone(ROOF, 33, course=9)))
        out.append(oriented_box(s, -TOP_W - 4.66, -TOP_W - 4.6, YC + 0.3, YC + 2.2, 1.0, np.array(DARK), 34))
    return out


# ── The earth embankment («espaldón») ───────────────────────────────────────
def embankment():
    """Its surface follows the MDT05 behind the wall, levelled into a walkway along the crest."""
    ss = np.linspace(0, L, int(L / 4) + 1)
    nn = np.array([-TOP_W, -3.5, -5.0, -7.0, -10.0, -14.0, -18.0, -23.0, -28.0, -34.0, -40.0, -BACK])
    P = np.zeros((len(ss), len(nn), 3))
    for i, s in enumerate(ss):
        for j, n in enumerate(nn):
            x, z = at(s, n)
            g = float(T(x, z)) - E0 + 0.12
            w = np.clip((-n - TOP_W) / 6.0, 0, 1)                # walkway 0.4 m under the crest next to the wall
            y = (YC - 0.25) * (1 - w) + min(g, YC - 0.25) * w
            P[i, j] = (x, y, z)
    nu, nv = P.shape[0] - 1, P.shape[1] - 1
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    m = Mesh(P.reshape(-1, 3), np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]), EARTH)
    nrm = np.cross(m.v[m.f[:, 1]] - m.v[m.f[:, 0]], m.v[m.f[:, 2]] - m.v[m.f[:, 0]])
    if (nrm[:, 1] < 0).sum() > len(nrm) / 2:
        m.f = m.f[:, ::-1]
    m.paint(earth)
    near = np.abs(P.reshape(-1, 3)[:, 1] - (YC - 0.25)) < 0.05
    m.c[near] = PAVE * 0.95
    # a skirt 3 m down along the far edge and the two ends, so no gap shows against the terrain
    sk = []
    for edge in (P[:, -1], P[0, :], P[-1, :]):
        top = edge
        bot = edge.copy()
        bot[:, 1] -= 3.0
        v, f = [], []
        for k in range(len(top) - 1):
            o = len(v)
            v += [top[k], top[k + 1], bot[k + 1], bot[k]]
            f += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        sk.append(Mesh(v, f, EARTH * 0.9))
    return [m] + sk


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = merge(wall() + buttresses() + towers() + embankment())
    tris, size = write_glb(OUT / "model.glb", [("presa_de_proserpina", m)], roughness=0.92)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}, length {L:.1f} m")


if __name__ == "__main__":
    main()
