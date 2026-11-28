"""
Detailed terrain around a model -> models/<Q>/terrain.json (Offlined uses it instead of its own 90 m relief).

    python tools/terrain.py Q171517 Q676203        # these models
    python tools/terrain.py --keep                  # every model with "ground": "keep"
    python tools/terrain.py --all                   # every model
    python tools/terrain.py --keep --missing        # only those that have no terrain.json yet

Why: Offlined's relief is Copernicus GLO-90 (90 m) where the person has downloaded it and GEBCO (about 450 m)
everywhere else. A crest, a cliff or the slope of a palace's gardens come out blunt or metres off, and a model that
sits on them floats or sinks. With the real ground of the area the game only has to move the terrain a little to fit
the model (MODEL_SPEC §3).

Sources (all allow commercial use and redistribution with attribution; Copernicus GLO-30 does not, so it is not used):
  · Spain: IGN MDT05, bare earth, 5 m (CC BY 4.0, scne.es), WCS https://servicios.idee.es/wcs-inspire/mdt
  · United States: USGS 3DEP, bare earth, 1–10 m (public domain), ImageServer exportImage
  · Everywhere else: AWS Terrain Tiles (Tilezen; SRTM 30 m and others, see the attribution in the file), z14 tiles
    These are surface models: they include the building itself (the Colosseum, the Taj Mahal, a pyramid…). For a
    model on flat ground ("flatten") its footprint plus 20 m is cut out and filled from the ground around it, so the
    game does not put a hump of the building's own shape under the model. For a "keep" model only what stands above
    that fill inside the footprint is cut (the Taj Mahal off its platform, a pyramid off the plateau). A model that
    follows a hill with no building to remove (Machu Picchu) sets "terrain_cut": false in its meta.json.

The grid is square around the model's position, in metres east (x) and north (y) at the scale of Web Mercator at
that latitude (the same metres the game uses), row 0 the northernmost. Heights: decimetres above "off" (int16,
little-endian, base64); -32768 = no data. meta.json "terrain_radius" (m, default 1200) sets its half-width.

Standard library + numpy + Pillow (tools/requirements.txt). Needs a connection.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glbtool  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "OfflinedLandmarks3D-tools (https://github.com/Aruvaro-Chulvi/OfflinedLandmarks3D)"}
R_EARTH = 6378137.0
RADIUS = 1200.0

ATTR = {
    "ign": "MDT05 © Instituto Geográfico Nacional de España (CC BY 4.0, scne.es)",
    "usgs": "3DEP data courtesy of the U.S. Geological Survey",
    "tiles": ("Terrain Tiles (registry.opendata.aws/terrain-tiles): SRTM, GMTED2010 and 3DEP courtesy of the U.S. "
              "Geological Survey; ETOPO1 U.S. National Oceanic and Atmospheric Administration"),
    "tiles_eu": "; EU-DEM produced using Copernicus data and information funded by the European Union",
}
SPAIN = [(-9.6, 35.8, 4.6, 44.0), (-18.4, 27.4, -13.2, 29.6)]
USA = [(-125.0, 24.0, -66.5, 49.5), (-170.0, 51.0, -129.0, 71.5), (-160.5, 18.8, -154.5, 22.4)]
EUROPE = (-25.0, 34.0, 45.0, 72.0)


def inside(lat, lon, boxes):
    return any(w <= lon <= e and s <= lat <= n for w, s, e, n in boxes)


def get(url: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception as ex:                                     # noqa: BLE001
            if k == tries - 1:
                raise
            print("   retry", ex)
            time.sleep(3 * (k + 1))
    raise RuntimeError("unreachable")


def merc_y(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def local_to_ll(lat0, lon0, x, y):
    """Metres east/north (Web Mercator scale at lat0) -> lat, lon (numpy arrays)."""
    k = R_EARTH * math.cos(math.radians(lat0))
    lon = lon0 + np.degrees(x / k)
    my = merc_y(lat0) + y / k
    lat = np.degrees(2 * np.arctan(np.exp(my)) - math.pi / 2)
    return lat, lon


def bilinear(a, fi, fj):
    """a[rows, cols] at fractional (row fj, col fi); NaN outside or where a corner is NaN."""
    ny, nx = a.shape
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    tx, tz = fi - i0, fj - j0
    ok = (i0 >= 0) & (j0 >= 0) & (i0 < nx - 1) & (j0 < ny - 1)
    i0c, j0c = np.clip(i0, 0, nx - 2), np.clip(j0, 0, ny - 2)
    v = (a[j0c, i0c] * (1 - tx) * (1 - tz) + a[j0c, i0c + 1] * tx * (1 - tz)
         + a[j0c + 1, i0c] * (1 - tx) * tz + a[j0c + 1, i0c + 1] * tx * tz)
    return np.where(ok, v, np.nan)


# ── Fuentes: cada una devuelve una función (lat, lon) -> altura (NaN sin dato) ──
def src_ign(s, w, n, e):
    url = ("https://servicios.idee.es/wcs-inspire/mdt?service=WCS&version=2.0.1&request=GetCoverage"
           f"&coverageId=Elevacion4258_5&subset=Lat({s:.6f},{n:.6f})&subset=Long({w:.6f},{e:.6f})&format=ArcGrid")
    txt = get(url).decode("latin-1")
    if "ncols" not in txt:
        raise ValueError("IGN WCS: " + txt[:300])
    lines = txt[txt.index("ncols"):].splitlines()
    hd, k = {}, 0
    while k < len(lines) and lines[k].split() and lines[k].split()[0].lower() in (
            "ncols", "nrows", "xllcorner", "yllcorner", "dx", "dy", "cellsize", "nodata_value", "xllcenter", "yllcenter"):
        a, b = lines[k].split()[:2]; hd[a.lower()] = float(b); k += 1
    nc, nr = int(hd["ncols"]), int(hd["nrows"])
    dx = hd.get("dx", hd.get("cellsize")); dy = hd.get("dy", dx)
    vals = np.array(" ".join(l for l in lines[k:] if l and not l.startswith("--")).split()[:nc * nr], dtype=np.float32)
    a = vals.reshape(nr, nc)
    if "nodata_value" in hd:
        a[a == hd["nodata_value"]] = np.nan
    x0 = hd.get("xllcorner", hd.get("xllcenter", 0)); y0 = hd.get("yllcorner", hd.get("yllcenter", 0))
    top = y0 + nr * dy                                             # rows go north -> south
    return lambda lat, lon: bilinear(a, (lon - x0) / dx - 0.5, (top - lat) / dy - 0.5), "ign", 5.0


def src_usgs(s, w, n, e, cell):
    W = max(2, min(8000, int(round((e - w) * 111320 * math.cos(math.radians((s + n) / 2)) / cell))))
    H = max(2, min(8000, int(round((n - s) * 110574 / cell))))
    url = ("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
           f"?bbox={w:.6f},{s:.6f},{e:.6f},{n:.6f}&bboxSR=4326&imageSR=4326&size={W},{H}&format=bsq&pixelType=F32"
           "&interpolation=RSP_BilinearInterpolation&f=image")
    b = get(url)
    if len(b) < W * H * 4:
        raise ValueError(f"3DEP: {len(b)} bytes for {W}x{H}")
    a = np.frombuffer(b[:W * H * 4], dtype="<f4").reshape(H, W).astype(np.float32).copy()
    a[(a < -1000) | (a > 9000)] = np.nan
    dx, dy = (e - w) / W, (n - s) / H
    return lambda lat, lon: bilinear(a, (lon - w) / dx - 0.5, (n - lat) / dy - 0.5), "usgs", 1.0


def src_tiles(s, w, n, e, z=14):
    def tx(lon): return (lon + 180) / 360 * 2 ** z
    def ty(lat): return (1 - merc_y(lat) / math.pi) / 2 * 2 ** z
    x0, x1, y0, y1 = int(tx(w)), int(tx(e)), int(ty(n)), int(ty(s))
    a = np.full(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), np.nan, np.float32)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            im = np.asarray(Image.open(io.BytesIO(get(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png")))
                            .convert("RGB"), dtype=np.float32)
            a[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
    def f(lat, lon):
        lon = np.asarray(lon); lat = np.asarray(lat)
        px = ((lon + 180) / 360 * 2 ** z - x0) * 256 - 0.5
        py = ((1 - np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) / np.pi) / 2 * 2 ** z - y0) * 256 - 0.5
        return bilinear(a, px, py)
    return f, "tiles", 2 * math.pi * R_EARTH * math.cos(math.radians((s + n) / 2)) / 2 ** z / 256


def in_poly(x, y, P):
    """Point-in-polygon (ray casting) for arrays x, y; P: [[x, y], ...]."""
    ins = np.zeros(x.shape, bool)
    n = len(P)
    for i in range(n):
        x1, y1 = P[i]; x2, y2 = P[i - 1]
        c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1)
        ins ^= c
    return ins


def cut_building(h, X, Y, meta, d: Path, grow=20.0, only_down=False):
    """Surface model: the footprint (+ grow m) is filled from the ground around it (harmonic fill, Jacobi); with
    only_down, a cell keeps its height where that is below the fill."""
    fp = meta.get("footprint")
    if not fp:
        fp = glbtool.footprint(glbtool.read_glb(d / "model.glb"))
    th = math.pi - math.radians(meta.get("heading", 0)); cs, sn = math.cos(th), math.sin(th)
    # del modelo (x, z) al terreno (este, norte), como Offlined; y crecida desde su centro
    P = np.array([[x * cs + z * sn, -(-x * sn + z * cs)] for x, z in fp])
    c = P.mean(0)
    r = np.linalg.norm(P - c, axis=1, keepdims=True)
    P = c + (P - c) * (r + grow) / np.maximum(r, 1e-6)
    mask = in_poly(X, Y, P.tolist())
    if not mask.any():
        return h, 0
    g = np.where(mask, np.nan, h)
    ring = np.nanmean(g)
    g = np.where(np.isnan(g), ring, g)
    for _ in range(600):
        avg = (np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)) / 4
        g = np.where(mask, avg, g)
    if only_down:
        g = np.where(np.isfinite(h), np.minimum(g, h), g)
    return np.where(np.isfinite(h) | mask, g, h), int(mask.sum())


def build(qid: str) -> dict:
    meta = json.loads((ROOT / "models" / qid / "meta.json").read_text("utf-8"))
    lat0, lon0 = float(meta["position"]["lat"]), float(meta["position"]["lon"])
    R = float(meta.get("terrain_radius") or RADIUS)
    # caja en grados con margen (para la interpolación del borde)
    dlat = (R + 60) / 110574.0; dlon = (R + 60) / (111320.0 * math.cos(math.radians(lat0)))
    s, n, w, e = lat0 - dlat, lat0 + dlat, lon0 - dlon, lon0 + dlon
    if inside(lat0, lon0, SPAIN):
        f, kind, res = src_ign(s, w, n, e); cell = 8.0
    elif inside(lat0, lon0, USA):
        f, kind, res = src_usgs(s, w, n, e, 4.0); cell = 8.0
    else:
        f, kind, res = src_tiles(s, w, n, e); cell = 15.0
    nx = int(math.ceil(2 * R / cell)) + 1
    xs = -R + np.arange(nx) * cell                                  # oeste -> este
    ys = R - np.arange(nx) * cell                                   # norte -> sur
    X, Y = np.meshgrid(xs, ys)
    lat, lon = local_to_ll(lat0, lon0, X, Y)
    h = f(lat, lon).astype(np.float64)
    cut = 0
    if kind == "tiles" and meta.get("terrain_cut", True):
        keep = meta.get("ground", "flatten") == "keep"
        h, cut = cut_building(h, X, Y, meta, ROOT / "models" / qid, grow=0.0 if keep else 20.0, only_down=keep)
    good = np.isfinite(h)
    if good.mean() < 0.5:
        raise ValueError(f"{qid}: only {good.mean():.0%} of the area has data from {kind}")
    off = float(np.round(np.nanmean(h)))
    q = np.where(good, np.clip(np.round((h - off) * 10), -32767, 32767), -32768).astype("<i2")
    attr = ATTR[kind] + (ATTR["tiles_eu"] if kind == "tiles" and inside(lat0, lon0, [EUROPE]) else "")
    return {"v": 1, "lat": lat0, "lon": lon0, "c": cell, "n": nx, "x0": float(xs[0]), "y0": float(ys[0]), "off": off,
            "h": base64.b64encode(q.tobytes()).decode("ascii"), "source": kind, "source_res": round(res, 1),
            "attribution": attr, "min": round(float(np.nanmin(h)), 1), "max": round(float(np.nanmax(h)), 1),
            "building_cut": cut}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--keep", action="store_true", help='every model with "ground": "keep"')
    ap.add_argument("--all", action="store_true", help="every model")
    ap.add_argument("--missing", action="store_true", help="skip models that already have terrain.json")
    ap.add_argument("--source", help="only models whose existing terrain.json comes from this source (ign, usgs, tiles)")
    a = ap.parse_args()
    ids = list(a.ids)
    if a.keep or a.all:
        for d in sorted((ROOT / "models").glob("Q*")):
            if (d / "meta.json").exists():
                m = json.loads((d / "meta.json").read_text("utf-8"))
                if a.all or m.get("ground") == "keep":
                    ids.append(d.name)
    if not ids:
        sys.exit(__doc__)
    bad = 0
    for q in dict.fromkeys(ids):
        out = ROOT / "models" / q / "terrain.json"
        if a.missing and out.exists():
            continue
        if a.source and not (out.exists() and json.loads(out.read_text("utf-8")).get("source") == a.source):
            continue
        try:
            t = build(q)
        except Exception as ex:                                     # noqa: BLE001
            print(f"{q}: ERROR {ex}"); bad += 1; continue
        out.write_text(json.dumps(t, separators=(",", ":")) + "\n", "utf-8")
        print(f"{q}: {t['source']} ({t['source_res']} m) -> {t['n']}x{t['n']} cells of {t['c']:g} m, "
              f"{t['min']}–{t['max']} m, {out.stat().st_size / 1024:.0f} KB"
              + (f", building cut out ({t['building_cut']} cells)" if t.get("building_cut") else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
