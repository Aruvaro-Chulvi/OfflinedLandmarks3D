"""
What OpenStreetMap knows about a building, in the axes of a model (docs/MODEL_SPEC.md): its outline, its
courtyards, its building:part pieces with their heights and roofs, its centre and its long axis. Standard
library only. Answers are cached in tools/.cache/osm/ (delete it to refresh).

    python tools/osm.py find 40.4180 -3.7142 "Palacio Real"     # ids of what is called that within 400 m
    python tools/osm.py way 174805987                            # outline, size, axis, tags, parts
    python tools/osm.py way 174805987 --heading 80.6             # … and the outline in model x/z
    python tools/osm.py relation 6248452 --heading 0 --json out.json
    python tools/osm.py way 174805987 --origin 40.419986,-3.688725 --heading 80.6

Model axes: +Y up, the main façade faces +Z, and heading is where +Z points, clockwise from north. So +X points
to heading − 90° (to your right when you stand in front of the façade looking at it). The origin is the centroid
of the outline unless --origin is given.

Data © OpenStreetMap contributors, ODbL: cite the way or relation in meta.json "sources" and "position.source".
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.openstreetmap.org/api/0.6"
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter")
UA = {"User-Agent": "OfflinedLandmarks3D-tools/1.0 (https://github.com/Aruvaro-Chulvi/OfflinedLandmarks3D)"}
CACHE = Path(__file__).resolve().parent / ".cache" / "osm"
A, F = 6378137.0, 1 / 298.257223563                     # WGS84


def _fetch(url, data: bytes | None = None, timeout=120, rounds=3) -> dict:
    """GET/POST JSON; url may be a tuple of mirrors, tried in turn."""
    urls = url if isinstance(url, tuple) else (url,)
    for attempt in range(rounds * len(urls)):
        u = urls[attempt % len(urls)]
        try:
            req = urllib.request.Request(u, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:                                   # noqa: BLE001
            if attempt == rounds * len(urls) - 1:
                raise
            time.sleep(2 + (attempt // len(urls)) * 5)
    return {}


def _cached(key: str, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (key.replace("/", "_") + ".json")
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    d = fn()
    f.write_text(json.dumps(d), encoding="utf-8")
    return d


def full(kind: str, oid: int) -> dict:
    return _cached(f"{kind}_{oid}", lambda: _fetch(f"{API}/{kind}/{oid}/full.json"))


def overpass(q: str, key: str, fallback=None) -> dict:
    """An Overpass query (mirrors in turn); if every mirror fails, fallback() (the main OSM API) answers instead."""
    def run():
        try:
            return _fetch(OVERPASS, urllib.parse.urlencode({"data": q}).encode(), timeout=25, rounds=1)
        except Exception:                                   # noqa: BLE001
            if fallback is None:
                raise
            print("(Overpass unreachable: asking the OpenStreetMap API for the area instead)", file=sys.stderr)
            return fallback()
    return _cached(key, run)


def bbox_elements(s, w, n, e, keep):
    """Everything the OSM API returns in a small box, shaped like Overpass `out tags geom center`, filtered by
    keep(tags)."""
    d = _fetch(f"{API}/map.json?bbox={w:.6f},{s:.6f},{e:.6f},{n:.6f}")
    els = d.get("elements", [])
    nodes = {x["id"]: x for x in els if x["type"] == "node"}
    ways = {x["id"]: x for x in els if x["type"] == "way"}
    def geom(w_):
        return [{"lat": nodes[i]["lat"], "lon": nodes[i]["lon"]} for i in w_["nodes"] if i in nodes]
    def ctr(g):
        return {"lat": sum(p["lat"] for p in g) / len(g), "lon": sum(p["lon"] for p in g) / len(g)} if g else None
    out = []
    for x in els:
        t = x.get("tags") or {}
        if not t or not keep(t):
            continue
        if x["type"] == "node":
            out.append({"type": "node", "id": x["id"], "tags": t, "lat": x["lat"], "lon": x["lon"]})
        elif x["type"] == "way":
            g = geom(x)
            out.append({"type": "way", "id": x["id"], "tags": t, "geometry": g, "center": ctr(g)})
        else:
            mem = [{"type": m["type"], "ref": m["ref"], "role": m.get("role", ""),
                    "geometry": geom(ways[m["ref"]]) if m["type"] == "way" and m["ref"] in ways else []}
                   for m in x.get("members", [])]
            allg = [p for m in mem if m["role"] != "inner" for p in m["geometry"]]
            out.append({"type": "relation", "id": x["id"], "tags": t, "members": mem, "center": ctr(allg)})
    return {"elements": out}


# ── Geometry ─────────────────────────────────────────────────────────────────
def radii(lat: float):
    """Metres per radian north (M) and east (N cos φ) at latitude lat."""
    e2 = F * (2 - F)
    s = math.sin(math.radians(lat))
    w = math.sqrt(1 - e2 * s * s)
    return A * (1 - e2) / w ** 3, A / w * math.cos(math.radians(lat))


class Frame:
    """lat/lon <-> metres east/north of an origin <-> model x/z for a heading."""

    def __init__(self, lat0, lon0, heading=0.0):
        self.lat0, self.lon0, self.h = lat0, lon0, math.radians(heading)
        self.kn, self.ke = radii(lat0)

    def en(self, lat, lon):
        return math.radians(lon - self.lon0) * self.ke, math.radians(lat - self.lat0) * self.kn

    def latlon(self, e, n):
        return self.lat0 + math.degrees(n / self.kn), self.lon0 + math.degrees(e / self.ke)

    def xz(self, e, n):
        c, s = math.cos(self.h), math.sin(self.h)
        return -e * c + n * s, e * s + n * c

    def from_xz(self, x, z):
        c, s = math.cos(self.h), math.sin(self.h)
        return -x * c + z * s, x * s + z * c


def area(P):
    return 0.5 * sum(P[i - 1][0] * P[i][1] - P[i][0] * P[i - 1][1] for i in range(len(P)))


def centroid(P):
    a = cx = cy = 0.0
    for i in range(len(P)):
        (x0, y0), (x1, y1) = P[i - 1], P[i]
        k = x0 * y1 - x1 * y0
        a += k; cx += (x0 + x1) * k; cy += (y0 + y1) * k
    if abs(a) < 1e-9:
        return sum(p[0] for p in P) / len(P), sum(p[1] for p in P) / len(P)
    return cx / (3 * a), cy / (3 * a)


def hull(pts):
    P = sorted(set((round(x, 3), round(y, 3)) for x, y in pts))
    if len(P) < 3:
        return P
    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(hi) >= 2 and cr(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def min_rect(pts):
    """Smallest rectangle around points (east, north) -> (length, width, bearing of the long side 0..180)."""
    H = hull(pts)
    best = None
    for i in range(len(H)):
        (x0, y0), (x1, y1) = H[i - 1], H[i]
        a = math.atan2(y1 - y0, x1 - x0)
        c, s = math.cos(a), math.sin(a)
        us = [x * c + y * s for x, y in H]; vs = [-x * s + y * c for x, y in H]
        w, h = max(us) - min(us), max(vs) - min(vs)
        if best is None or w * h < best[0]:
            L, W, ang = (w, h, a) if w >= h else (h, w, a + math.pi / 2)
            best = (w * h, L, W, ang)
    if best is None:
        return 0.0, 0.0, 0.0
    _, L, W, ang = best
    bearing = (90 - math.degrees(ang)) % 180                 # math angle from east -> compass bearing
    return L, W, bearing


# ── OSM elements -> rings ────────────────────────────────────────────────────
def _join(ways):
    """Join way node lists into closed rings."""
    segs = [list(w) for w in ways if len(w) > 1]
    rings = []
    while segs:
        r = segs.pop(0)
        changed = True
        while r[0] != r[-1] and changed:
            changed = False
            for i, s in enumerate(segs):
                if s[0] == r[-1]:
                    r += s[1:]
                elif s[-1] == r[-1]:
                    r += s[::-1][1:]
                elif s[-1] == r[0]:
                    r = s + r[1:]
                elif s[0] == r[0]:
                    r = s[::-1] + r[1:]
                else:
                    continue
                segs.pop(i); changed = True
                break
        rings.append(r)
    return rings


def element(kind: str, oid: int):
    """-> {tags, outer: [[(lat, lon)]], inner: [[(lat, lon)]]}"""
    d = full(kind, oid)
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in d["elements"] if e["type"] == "node"}
    ways = {e["id"]: e for e in d["elements"] if e["type"] == "way"}
    if kind == "way":
        w = ways[oid]
        return {"tags": w.get("tags", {}), "outer": [[nodes[n] for n in w["nodes"]]], "inner": []}
    rel = next(e for e in d["elements"] if e["type"] == "relation" and e["id"] == oid)
    out, inn = [], []
    for m in rel["members"]:
        if m["type"] == "way" and m["ref"] in ways:
            (inn if m.get("role") == "inner" else out).append(ways[m["ref"]]["nodes"])
    return {"tags": rel.get("tags", {}),
            "outer": [[nodes[n] for n in r if n in nodes] for r in _join(out)],
            "inner": [[nodes[n] for n in r if n in nodes] for r in _join(inn)]}


def _open(r):
    return r[:-1] if len(r) > 1 and r[0] == r[-1] else r


def _in_poly(x, y, P):
    ins = False
    for i in range(len(P)):
        (xi, yi), (xj, yj) = P[i], P[i - 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            ins = not ins
    return ins


HEIGHT_KEYS = ("height", "min_height", "building:levels", "building:min_level", "roof:shape", "roof:height",
               "roof:levels", "roof:direction", "roof:orientation", "roof:angle", "building:colour", "roof:colour",
               "building:material", "roof:material", "building:part", "building", "name", "wikidata", "wikipedia",
               "start_date", "architect", "ele", "layer")


def describe(kind, oid, heading=None, origin=None, parts=True, radius=0.0):
    el = element(kind, oid)
    ring0 = _open(el["outer"][0])
    lat_c = sum(p[0] for p in ring0) / len(ring0); lon_c = sum(p[1] for p in ring0) / len(ring0)
    fr = Frame(lat_c, lon_c)
    outers = [[fr.en(*p) for p in _open(r)] for r in el["outer"]]
    big = max(outers, key=lambda r: abs(area(r)))
    ce, cn = centroid(big)
    lat0, lon0 = fr.latlon(ce, cn)
    if origin:
        lat0, lon0 = origin
    H = heading if heading is not None else 0.0
    fr = Frame(lat0, lon0, H)
    to_m = lambda r: [fr.en(*p) for p in _open(r)]                       # noqa: E731
    O = [to_m(r) for r in el["outer"]]; I = [to_m(r) for r in el["inner"]]
    allp = [p for r in O for p in r]
    L, W, brg = min_rect(allp)
    res = {
        "element": f"https://www.openstreetmap.org/{kind}/{oid}",
        "tags": el["tags"],
        "centroid": {"lat": round(lat0, 7), "lon": round(lon0, 7)},
        "area_m2": round(sum(abs(area(r)) for r in O) - sum(abs(area(r)) for r in I), 1),
        "min_rect": {"length": round(L, 2), "width": round(W, 2), "long_axis_bearing": round(brg, 1),
                     "facing_options": sorted({round((brg + 90) % 360, 1), round((brg + 270) % 360, 1)})},
        "heading": H,
    }
    if heading is not None:
        res["outline_xz"] = [[[round(v, 3) for v in fr.xz(e, n)] for e, n in r] for r in O]
        res["courtyards_xz"] = [[[round(v, 3) for v in fr.xz(e, n)] for e, n in r] for r in I]
        xs = [fr.xz(e, n) for e, n in allp]
        res["extent_xz"] = {"x": [round(min(p[0] for p in xs), 2), round(max(p[0] for p in xs), 2)],
                            "z": [round(min(p[1] for p in xs), 2), round(max(p[1] for p in xs), 2)]}
    else:
        res["outline_en"] = [[[round(v, 3) for v in p] for p in r] for r in O]
        res["courtyards_en"] = [[[round(v, 3) for v in p] for p in r] for r in I]
    if parts:
        lats = [p[0] for r in el["outer"] for p in r]; lons = [p[1] for r in el["outer"] for p in r]
        pad = radius / 111_000
        bb = f"{min(lats) - pad},{min(lons) - pad},{max(lats) + pad},{max(lons) + pad}"
        q = f'[out:json][timeout:90];(way["building:part"]({bb});relation["building:part"]({bb}););out tags geom;'
        d = overpass(q, f"parts_{kind}_{oid}_{int(radius)}",
                     lambda: bbox_elements(min(lats) - pad, min(lons) - pad, max(lats) + pad, max(lons) + pad,
                                           lambda t: "building:part" in t))
        lst = []
        for e in d.get("elements", []):
            geo = e.get("geometry") or []
            if e["type"] == "relation":                     # join the outer member ways into rings (any direction)
                ways = [[(g["lat"], g["lon"]) for g in (m.get("geometry") or [])]
                        for m in e.get("members", []) if m.get("role") != "inner"]
                rings = _join([w for w in ways if len(w) > 1])
                ring = max(rings, key=lambda r: len(r)) if rings else []
                geo = [{"lat": a, "lon": b} for a, b in ring]
            if len(geo) < 3:
                continue
            pts = [fr.en(g["lat"], g["lon"]) for g in geo]
            cx, cy = centroid(_open(pts))
            if radius <= 0 and not any(_in_poly(cx, cy, r) for r in O):
                continue
            item = {"id": f'{e["type"]}/{e["id"]}', "tags": {k: v for k, v in e.get("tags", {}).items() if k in HEIGHT_KEYS}}
            ring = _open(pts)
            if heading is not None:
                item["outline_xz"] = [[round(v, 3) for v in fr.xz(a, b)] for a, b in ring]
            else:
                item["outline_en"] = [[round(v, 3) for v in p] for p in ring]
            lst.append(item)
        res["parts"] = lst
    return res


def find(lat, lon, name, radius=400):
    q = (f'[out:json][timeout:60];nwr["name"~"{name}",i](around:{radius},{lat},{lon});out tags center;')
    import re
    rx = re.compile(name, re.I)
    dl, dn = radius / 111_000, radius / (111_000 * math.cos(math.radians(lat)))
    d = overpass(q, f"find_{lat}_{lon}_{sum(map(ord, name)) * 131 % 10**8}_{radius}",
                 lambda: bbox_elements(lat - dl, lon - dn, lat + dl, lon + dn, lambda t: bool(rx.search(t.get("name", "")))))
    out = []
    for e in d.get("elements", []):
        t = e.get("tags", {})
        c = e.get("center") or {"lat": e.get("lat"), "lon": e.get("lon")}
        out.append({"id": f'{e["type"]}/{e["id"]}', "name": t.get("name"), "wikidata": t.get("wikidata"),
                    "building": t.get("building") or t.get("building:part"), "height": t.get("height"),
                    "kind": next((f"{k}={t[k]}" for k in ("tourism", "historic", "amenity", "man_made", "leisure", "office") if k in t), None),
                    "center": [c.get("lat"), c.get("lon")]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("way", "relation", "find"))
    ap.add_argument("args", nargs="+")
    ap.add_argument("--heading", type=float, help="façade direction, degrees clockwise from north -> outline in model x/z")
    ap.add_argument("--origin", help="lat,lon of the model origin (default: centroid of the outline)")
    ap.add_argument("--no-parts", action="store_true", help="skip building:part pieces")
    ap.add_argument("--radius", type=float, default=0.0, help="also list building:part pieces this many metres around")
    ap.add_argument("--json", help="write the full answer to this file")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                       # noqa: BLE001
        pass
    if a.kind == "find":
        lat, lon, name = float(a.args[0]), float(a.args[1]), a.args[2]
        for r in find(lat, lon, name, int(a.args[3]) if len(a.args) > 3 else 400):
            print(json.dumps(r, ensure_ascii=False))
        return 0
    origin = tuple(map(float, a.origin.split(","))) if a.origin else None
    r = describe(a.kind, int(a.args[0]), a.heading, origin, not a.no_parts, a.radius)
    if a.json:
        Path(a.json).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    t = r["tags"]
    print(r["element"], "—", t.get("name", ""), "| wikidata", t.get("wikidata", "–"))
    print("tags:", json.dumps({k: v for k, v in t.items() if k in HEIGHT_KEYS}, ensure_ascii=False))
    print(f'centroid (origin): {r["centroid"]["lat"]}, {r["centroid"]["lon"]}   area {r["area_m2"]} m²')
    mr = r["min_rect"]
    print(f'smallest rectangle {mr["length"]} × {mr["width"]} m, long axis at {mr["long_axis_bearing"]}° '
          f'-> a façade on the long side faces {mr["facing_options"][0]}° or {mr["facing_options"][1]}°')
    if "extent_xz" in r:
        print(f'heading {r["heading"]}: extent x {r["extent_xz"]["x"]}, z {r["extent_xz"]["z"]}, '
              f'{sum(len(o) for o in r["outline_xz"])} outline points, {len(r["courtyards_xz"])} courtyards')
    for p in r.get("parts", []):
        pts = p.get("outline_xz") or p.get("outline_en")
        print(f'  part {p["id"]}: {json.dumps(p["tags"], ensure_ascii=False)} ({len(pts)} points)')
    if not a.json:
        print("(add --json file.json for the coordinates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
