"""
Floors for the courtyards a model leaves open (used by model_sources/*.py, after building the model).

A courtyard here is ground enclosed by the model, seen from above, that no part of it covers: the game would see
through it and draw its own ground (grass, trees, generated buildings are kept out by the footprint, but the colour
of the map stays). courtyard_floor() finds those holes on a grid and returns a floor for them:

    from ground_fill import courtyard_floor
    floor = courtyard_floor(meshes, "Q4889526", colour="pnoa")              # Spain: colours of the IGN orthophoto
    floor = courtyard_floor(meshes, "Q62408", colour=osm_colour_function)    # elsewhere: your own colours

- height: y = 0 for a "flatten" model (the game levels its footprint), the ground of terrain.json (relative to
  ground_at, the same ground the game uses) for a "keep" model; or height=callable(x, z) -> y.
- colour="pnoa": the IGN PNOA orthophoto (© Instituto Geográfico Nacional de España, CC BY 4.0, scne.es), fetched
  once into model_sources/<Q>_pnoa.jpg (+ .json with its bounds) at about 0.25 m/px; the shadows of the courtyard
  walls are taken out (a shaded pixel gets the mean colour of the sunlit ground around it), the game casts its own.
  colour=callable(x, z) -> (N, 3) rgb 0..1 for anything else, or an (r, g, b) tuple.

numpy + Pillow; a connection only the first time for the orthophoto.
"""
from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from meshkit import Mesh

ROOT = Path(__file__).resolve().parent.parent


def _meta(qid):
    return json.loads((ROOT / "models" / qid / "meta.json").read_text(encoding="utf-8"))


def _to_ll(meta, x, z):
    """Model x, z → lat, lon, as Offlined places the model (th = π − heading)."""
    lat0, lon0 = meta["position"]["lat"], meta["position"]["lon"]
    th = math.pi - math.radians(meta.get("heading", 0)); c, s = math.cos(th), math.sin(th)
    e = x * c + z * s; so = -x * s + z * c
    return lat0 - so / 110574.0, lon0 + e / (111320.0 * math.cos(math.radians(lat0)))


def terrain_height(qid):
    """y of the ground of models/<Q>/terrain.json relative to ground_at (what the game uses around the model)."""
    import base64
    meta = _meta(qid)
    t = json.loads((ROOT / "models" / qid / "terrain.json").read_text(encoding="utf-8"))
    n = t["n"]
    H = np.frombuffer(base64.b64decode(t["h"]), "<i2").reshape(n, n).astype(float) / 10.0 + t["off"]
    th = math.pi - math.radians(meta.get("heading", 0)); c, s = math.cos(th), math.sin(th)

    def T(x, z):
        x = np.asarray(x, float); z = np.asarray(z, float)
        e = x * c + z * s; no = -(-x * s + z * c)
        fi = np.clip((e - t["x0"]) / t["c"], 0, n - 1.0001); fj = np.clip((t["y0"] - no) / t["c"], 0, n - 1.0001)
        i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int); a = fi - i0; b = fj - j0
        return H[j0, i0] * (1 - a) * (1 - b) + H[j0, i0 + 1] * a * (1 - b) + H[j0 + 1, i0] * (1 - a) * b + H[j0 + 1, i0 + 1] * a * b
    gx, gz = meta.get("ground_at") or [0, 0]
    t0 = float(T(gx, gz))
    return lambda x, z: T(x, z) - t0


def pnoa_colour(qid, x0, z0, x1, z1, mpp=0.25):
    """A colour function (x, z) -> rgb from the IGN orthophoto over the box, without the baked shadows."""
    meta = _meta(qid)
    img_p = ROOT / "model_sources" / f"{qid}_pnoa.jpg"; box_p = img_p.with_suffix(".json")
    if not img_p.exists():
        ll = [_to_ll(meta, x, z) for x in (x0 - 10, x1 + 10) for z in (z0 - 10, z1 + 10)]
        S, N = min(a for a, b in ll), max(a for a, b in ll); W, E = min(b for a, b in ll), max(b for a, b in ll)
        w = int((E - W) * 111320 * math.cos(math.radians(S)) / mpp); h = int((N - S) * 110574 / mpp)
        url = ("https://www.ign.es/wms-inspire/pnoa-ma?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS=OI.OrthoimageCoverage"
               f"&STYLES=&CRS=EPSG:4326&BBOX={S},{W},{N},{E}&WIDTH={min(w, 4000)}&HEIGHT={min(h, 4000)}&FORMAT=image/jpeg")
        img_p.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "OfflinedLandmarks3D"}), timeout=180).read())
        box_p.write_text(json.dumps({"S": S, "W": W, "N": N, "E": E, "source": "PNOA máxima actualidad, © Instituto Geográfico "
                                     "Nacional de España, CC BY 4.0 (scne.es), WMS https://www.ign.es/wms-inspire/pnoa-ma",
                                     "url": url}, indent=1) + "\n", encoding="utf-8", newline="\n")
    box = json.loads(box_p.read_text(encoding="utf-8"))
    img = np.asarray(Image.open(img_p).convert("RGB"), float) / 255.0
    H, W_ = img.shape[:2]
    px = (box["E"] - box["W"]) * 111320 * math.cos(math.radians(box["S"])) / W_      # metres per pixel
    lum = img.mean(2); exg = 2 * img[..., 1] - img[..., 0] - img[..., 2]
    lit = (lum > 0.36) & (exg < 0.08)

    def boxsum(a, r):
        c = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge").cumsum(0).cumsum(1)
        return c[2 * r + 1:, 2 * r + 1:] - c[:-2 * r - 1, 2 * r + 1:] - c[2 * r + 1:, :-2 * r - 1] + c[:-2 * r - 1, :-2 * r - 1]
    out = img.copy()
    for rm in (3.0, 8.0):                                     # two passes: narrow shadows, then wide ones
        r = max(2, int(rm / px))
        wgt = boxsum(lit.astype(float), r)
        mean = np.stack([boxsum(img[..., k] * lit, r) for k in range(3)], 2) / np.maximum(wgt, 1)[..., None]
        shade = ((out.mean(2) < 0.33) | (out[..., 2] > out[..., 0] + 0.01)) & (exg < 0.05) & (wgt > (2 * r + 1) ** 2 * 0.08)
        out = np.where(shade[..., None], mean, out)

    def colour(x, z):
        la, lo = _to_ll(meta, np.asarray(x, float), np.asarray(z, float))
        i = np.clip(((lo - box["W"]) / (box["E"] - box["W"]) * W_).astype(int), 0, W_ - 1)
        j = np.clip(((box["N"] - la) / (box["N"] - box["S"]) * H).astype(int), 0, H - 1)
        return out[j, i]
    return colour


def _covered(meshes, x0, z0, nx, nz, cell, rng):
    cov = np.zeros((nz, nx), bool)
    for m in meshes:
        if m is None or not len(m.f):
            continue
        T = np.asarray(m.v)[np.asarray(m.f)]
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
        k = np.clip(np.ceil(area / (cell * cell * 0.25)).astype(int), 1, 400000)
        rep = np.repeat(np.arange(len(T)), k)
        u, v = rng.random(len(rep)), rng.random(len(rep))
        fl = u + v > 1; u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
        P = np.vstack([a[rep] + (b - a)[rep] * u[:, None] + (c - a)[rep] * v[:, None], T.reshape(-1, 3)])
        i = ((P[:, 0] - x0) / cell).astype(int); j = ((P[:, 2] - z0) / cell).astype(int)
        ok = (i >= 0) & (j >= 0) & (i < nx) & (j < nz)
        cov[j[ok], i[ok]] = True
    return cov


def courtyard_floor(meshes, qid, colour="pnoa", height=None, cell=1.0, min_area=40.0, dy=0.04, report=True):
    """A floor (Mesh, per-vertex colours, facing up) over the courtyards the meshes enclose and leave open."""
    rng = np.random.default_rng(7)
    V = np.vstack([np.asarray(m.v) for m in meshes if m is not None and len(m.v)])
    x0, z0 = V[:, 0].min() - 2 * cell, V[:, 2].min() - 2 * cell
    x1, z1 = V[:, 0].max() + 2 * cell, V[:, 2].max() + 2 * cell
    nx, nz = int((x1 - x0) / cell) + 1, int((z1 - z0) / cell) + 1
    cov = _covered(meshes, x0, z0, nx, nz, cell, rng)
    # open ground reachable from the border of the grid is outside; the rest that is open is a courtyard
    out = np.zeros_like(cov); st = []
    for j in range(nz):
        for i in (0, nx - 1):
            if not cov[j, i] and not out[j, i]: out[j, i] = True; st.append((j, i))
    for i in range(nx):
        for j in (0, nz - 1):
            if not cov[j, i] and not out[j, i]: out[j, i] = True; st.append((j, i))
    while st:
        j, i = st.pop()
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = j + dj, i + di
            if 0 <= a < nz and 0 <= b < nx and not cov[a, b] and not out[a, b]:
                out[a, b] = True; st.append((a, b))
    hole = ~cov & ~out
    # drop the tiny ones (gaps between walls)
    lab = np.zeros(hole.shape, int); keep = np.zeros_like(hole); n_c = 0; area_c = 0.0
    for j in range(nz):
        for i in range(nx):
            if hole[j, i] and not lab[j, i]:
                n_c += 1; s2 = [(j, i)]; lab[j, i] = n_c; cells = []
                while s2:
                    a, b = s2.pop(); cells.append((a, b))
                    for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        c, e = a + dj, b + di
                        if 0 <= c < nz and 0 <= e < nx and hole[c, e] and not lab[c, e]:
                            lab[c, e] = n_c; s2.append((c, e))
                if len(cells) * cell * cell >= min_area:
                    for a, b in cells: keep[a, b] = True
                    area_c += len(cells) * cell * cell
    # grow by one cell under the walls, so no seam shows at their foot
    g = keep.copy(); g[1:] |= keep[:-1]; g[:-1] |= keep[1:]; g[:, 1:] |= keep[:, :-1]; g[:, :-1] |= keep[:, 1:]
    keep = g
    meta = _meta(qid)
    if height is None:
        height = terrain_height(qid) if meta.get("ground") == "keep" else (lambda x, z: np.zeros(np.shape(x)))
    if isinstance(colour, str) and colour == "pnoa":
        colour = pnoa_colour(qid, x0, z0, x1, z1)
    idx = {}; Vs = []; F = []

    def vid(i, j):
        k = (i, j)
        if k not in idx:
            idx[k] = len(Vs); Vs.append((x0 + i * cell, 0.0, z0 + j * cell))
        return idx[k]
    for j, i in np.argwhere(keep):
        a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
        F += [(a, c, b), (a, d, c)]
    if not F:
        return None
    Vs = np.array(Vs, float)
    Vs[:, 1] = np.asarray(height(Vs[:, 0], Vs[:, 2]), float) + dy
    if callable(colour):
        C = np.asarray(colour(Vs[:, 0], Vs[:, 2]), float).reshape(-1, 3)
    else:
        C = np.tile(np.asarray(colour, float), (len(Vs), 1))
    m = Mesh(Vs, np.array(F), (0.6, 0.6, 0.6))
    m.c = np.clip(C, 0, 1)
    n = np.cross(Vs[m.f[:, 1]] - Vs[m.f[:, 0]], Vs[m.f[:, 2]] - Vs[m.f[:, 0]])
    flip = n[:, 1] < 0
    m.f[flip] = m.f[flip][:, ::-1]
    if report:
        print(f"courtyard floors: {area_c:.0f} m² in courtyards of {min_area:g} m² or more ({len(F)} triangles)")
    return m
