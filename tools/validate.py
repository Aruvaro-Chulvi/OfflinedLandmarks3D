"""
Check every model folder against docs/MODEL_SPEC.md. Standard library only.

    python tools/validate.py                  # all models
    python tools/validate.py models/Q1140634

Exit code 1 if anything is wrong (errors); warnings do not fail.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_SIZE, MAX_TRIS, MAX_TEX = 8 * 2**20, 150_000, 2048
EXT_OK = {"KHR_mesh_quantization", "KHR_texture_transform",
          "KHR_materials_emissive_strength", "EXT_texture_webp"}
LICENSES = {"CC-BY-4.0", "CC0-1.0"}
RIGHTS = {"public_domain", "protected"}
PD_YEARS = 80                     # the longest term that applies in Spain (authors who died before 1987); elsewhere 70
QID = re.compile(r"^Q[1-9][0-9]*$")
FILES_OK = {"model.glb", "meta.json", "preview.jpg", "terrain.json"}   # terrain.json: tools/terrain.py


def image_size(b: bytes):
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", b[16:24])
    if b[:2] == b"\xff\xd8":
        i = 2
        while i < len(b) - 9:
            if b[i] != 0xFF:
                i += 1; continue
            m = b[i + 1]
            if m in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack(">HH", b[i + 5:i + 9]); return w, h
            i += 2 + struct.unpack(">H", b[i + 2:i + 4])[0]
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        if b[12:16] == b"VP8 ":
            w, h = struct.unpack("<HH", b[26:30]); return w & 0x3FFF, h & 0x3FFF
        if b[12:16] == b"VP8L":
            v = int.from_bytes(b[21:25], "little"); return (v & 0x3FFF) + 1, ((v >> 14) & 0x3FFF) + 1
        if b[12:16] == b"VP8X":
            return int.from_bytes(b[24:27], "little") + 1, int.from_bytes(b[27:30], "little") + 1
    return None


def check_glb(p: Path, err, warn):
    b = p.read_bytes()
    if len(b) > MAX_SIZE:
        err(f"{p.name}: {len(b) / 2**20:.1f} MB, more than {MAX_SIZE // 2**20} MB")
    if len(b) < 20 or b[:4] != b"glTF":
        err(f"{p.name}: not a binary glTF (.glb) file"); return None
    ver, _ = struct.unpack("<II", b[4:12])
    if ver != 2:
        err(f"{p.name}: glTF version {ver}, must be 2"); return None
    clen, ctype = struct.unpack("<II", b[12:20])
    if ctype != 0x4E4F534A:
        err(f"{p.name}: first chunk is not JSON"); return None
    g = json.loads(b[20:20 + clen])
    binc = None
    off = 20 + clen
    if off + 8 <= len(b):
        blen, btype = struct.unpack("<II", b[off:off + 8])
        if btype == 0x004E4942:
            binc = b[off + 8:off + 8 + blen]
    for ext in set(g.get("extensionsUsed", [])) | set(g.get("extensionsRequired", [])):
        if ext not in EXT_OK:
            err(f"{p.name}: extension {ext} is not allowed (see MODEL_SPEC §2)")
    for buf in g.get("buffers", []):
        if "uri" in buf:
            err(f"{p.name}: external buffer '{buf['uri'][:40]}': everything must be inside the .glb")
    acc = g.get("accessors", [])
    tris = 0
    for m in g.get("meshes", []):
        for pr in m.get("primitives", []):
            mode = pr.get("mode", 4)
            n = acc[pr["indices"]]["count"] if "indices" in pr else acc[pr["attributes"]["POSITION"]]["count"]
            tris += n // 3 if mode == 4 else max(0, n - 2) if mode in (5, 6) else 0
    if tris > MAX_TRIS:
        err(f"{p.name}: {tris:,} triangles, more than {MAX_TRIS:,}")
    for k in ("animations", "cameras", "skins"):
        if g.get(k):
            warn(f"{p.name}: has {k}; they are ignored")
    if (g.get("extensions") or {}).get("KHR_lights_punctual"):
        warn(f"{p.name}: has lights; they are ignored")
    views = g.get("bufferViews", [])
    px = 0
    for i, im in enumerate(g.get("images", [])):
        if "uri" in im:
            err(f"{p.name}: image {i} is an external file; textures must be inside the .glb"); continue
        if binc is None or "bufferView" not in im:
            continue
        v = views[im["bufferView"]]
        data = binc[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
        sz = image_size(data)
        if sz:
            px += sz[0] * sz[1]
            if max(sz) > MAX_TEX:
                err(f"{p.name}: texture {i} is {sz[0]}×{sz[1]} px, more than {MAX_TEX}")
    if px > 2 * MAX_TEX * MAX_TEX:
        err(f"{p.name}: {px / 1e6:.1f} Mpx of textures in all, more than two 2048×2048 ones (MODEL_SPEC §5)")
    # extent of the whole thing, from the POSITION bounds plus node translations (rotations ignored: a rough check)
    lo, hi = [1e18] * 3, [-1e18] * 3
    nodes = g.get("nodes", [])
    for n in nodes:
        if "mesh" not in n:
            continue
        t = n.get("translation", [0, 0, 0])
        for pr in g["meshes"][n["mesh"]].get("primitives", []):
            a = acc[pr["attributes"]["POSITION"]]
            if "min" in a and "max" in a:
                lo = [min(x, y + o) for x, y, o in zip(lo, a["min"], t)]
                hi = [max(x, y + o) for x, y, o in zip(hi, a["max"], t)]
    return {"triangles": tris, "lo": lo if hi[0] > -1e17 else None, "hi": hi if hi[0] > -1e17 else None}


def check_rights(r, err, warn):
    if not isinstance(r, dict) or r.get("status") not in RIGHTS:
        err(f"rights.status must be one of {sorted(RIGHTS)} (see NOTICE.md, 'Buildings and copyright')"); return
    arch = r.get("architects", [])
    if not isinstance(arch, list) or not all(isinstance(a, dict) and str(a.get("name", "")).strip() for a in arch):
        err("rights.architects: a list of {name, died?, wikidata?}"); return
    year = dt.date.today().year
    if r["status"] == "public_domain":
        late = [a for a in arch if isinstance(a.get("died"), int) and a["died"] > year - PD_YEARS - 1]
        alive = [a for a in arch if "died" not in a]
        for a in late:
            err(f"rights: {a['name']} died in {a['died']}, less than {PD_YEARS} years ago — use \"protected\"")
        if alive and not (isinstance(r.get("built"), int) and r["built"] < 1850):
            warn("rights: public domain, but some architects have no 'died' year — add it, or 'built' if older than 1850")
        if not arch and not (isinstance(r.get("built"), int) and r["built"] < 1850):
            err("rights: public domain needs the architects with their 'died' year, or 'built' (a year before 1850) if unknown")
    else:
        warn("rights: protected (a modern work) — accepted, flagged for the maintainer; see NOTICE.md")
        if not str(r.get("note", "")).strip():
            warn("rights: add a 'note' (who holds the rights, any known objection, why it should be fine)")


BRIDGE = re.compile(r"\b(bridge|puente|pont|ponte|brücke|brug|most|köprü|viaduct|viaducto|aqueduct|acueducto|dam|presa|pier|muelle)\b", re.I)
UNDER_CELL = 4.0     # metres: the grid of the ground check


def _triangles(p: Path):
    """The model's triangles ((x, y, z) × 3) with each node's translation, rotation and scale."""
    b = p.read_bytes()
    jl = struct.unpack_from("<I", b, 12)[0]
    j = json.loads(b[20:20 + jl])
    o = 20 + jl
    binc = b[o + 8:o + 8 + struct.unpack_from("<I", b, o)[0]]
    acc, bvs = j.get("accessors", []), j.get("bufferViews", [])
    CT = {5126: ("f", 4), 5125: ("I", 4), 5123: ("H", 2), 5121: ("B", 1)}

    def read(ai, nc):
        a = acc[ai]; bv = bvs[a["bufferView"]]; fmt, sz = CT[a["componentType"]]
        st = bv.get("byteOffset", 0) + a.get("byteOffset", 0); stride = bv.get("byteStride") or sz * nc
        return [struct.unpack_from("<" + fmt * nc, binc, st + i * stride) for i in range(a["count"])]

    def rot(q, v):
        x, y, z, w = q; tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
        return (v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx)
    nodes, out = j.get("nodes", []), []

    def walk(ni, chain):
        n = nodes[ni]; chain = chain + [n]
        for pr in (j["meshes"][n["mesh"]].get("primitives", []) if "mesh" in n else []):
            if pr.get("mode", 4) != 4 or "POSITION" not in pr.get("attributes", {}):
                continue
            pts = []
            for v in read(pr["attributes"]["POSITION"], 3):
                for nd in reversed(chain):
                    if nd.get("scale"): v = tuple(a * s for a, s in zip(v, nd["scale"]))
                    if nd.get("rotation"): v = rot(nd["rotation"], v)
                    if nd.get("translation"): v = tuple(a + t for a, t in zip(v, nd["translation"]))
                pts.append(v)
            idx = [i[0] for i in read(pr["indices"], 1)] if "indices" in pr else list(range(len(pts)))
            for i in range(0, len(idx) - 2, 3):
                out.append((pts[idx[i]], pts[idx[i + 1]], pts[idx[i + 2]]))
        for ch in n.get("children", []):
            walk(ch, chain)
    sc = (j.get("scenes") or [{"nodes": list(range(len(nodes)))}])[j.get("scene", 0)]
    for ni in sc.get("nodes", []):
        walk(ni, [])
    return out


def check_ground(p: Path, m: dict, warn):
    """How the model will sit on Offlined's terrain (MODEL_SPEC §3.1): the lowest surface of the model on a
    4 m grid seen from above — with "ground": "keep" Offlined fits the terrain to it."""
    try:
        T = _triangles(p)
    except Exception:                                       # noqa: BLE001 — check_glb already reports a bad file
        return
    c, g = UNDER_CELL, {}
    for a, b, q in T:
        d = (b[2] - q[2]) * (a[0] - q[0]) + (q[0] - b[0]) * (a[2] - q[2])
        xs, zs = (a[0], b[0], q[0]), (a[2], b[2], q[2])
        i0, i1, j0, j1 = int(min(xs) // c), int(max(xs) // c), int(min(zs) // c), int(max(zs) // c)
        for jj in range(j0, j1 + 1):
            for ii in range(i0, i1 + 1):
                if abs(d) < 1e-9 or (i0 == i1 and j0 == j1):
                    y = min(a[1], b[1], q[1])
                else:
                    px, pz = (ii + 0.5) * c, (jj + 0.5) * c
                    l1 = ((b[2] - q[2]) * (px - q[0]) + (q[0] - b[0]) * (pz - q[2])) / d
                    l2 = ((q[2] - a[2]) * (px - q[0]) + (a[0] - q[0]) * (pz - q[2])) / d
                    if l1 < -0.02 or l2 < -0.02 or 1 - l1 - l2 < -0.02:
                        continue
                    y = l1 * a[1] + l2 * b[1] + (1 - l1 - l2) * q[1]
                if y < g.get((ii, jj), 1e9):
                    g[(ii, jj)] = y
    if not g:
        return
    ys = sorted(g.values())
    low = ys[len(ys) // 20]                                  # the lowest 5 % (not one deep pile)
    mode = m.get("ground", "flatten")
    if mode == "flatten" and low < -8:
        warn(f"the base goes down to about {low:.0f} m below y = 0 over part of the model, as if it followed a slope: "
             'with the default "flatten" Offlined sets the whole model at the lowest ground and it sinks — '
             'use "ground": "keep" (MODEL_SPEC §3.1)')
    if mode == "keep" and m.get("ground_fit") is not False:
        # holes: cells with nothing between two cells with model, along x
        rows: dict = {}
        for (ii, jj) in g:
            rows.setdefault(jj, []).append(ii)
        span = sum(max(v) - min(v) + 1 for v in rows.values())
        holes = span - len(g)
        if span and holes / span > 0.3:
            warn(f"about {100 * holes / span:.0f} % of the area under the model has nothing (courtyards without a "
                 "floor, gaps between buildings): there Offlined only blends the coarse terrain towards the model — "
                 "if that ground matters (a paved courtyard, a terrace, a square), model it (MODEL_SPEC §3.1)")


def check_folder(d: Path, errors: list, warnings: list):
    rel = d.relative_to(ROOT).as_posix() if d.is_relative_to(ROOT) else str(d)
    err = lambda m: errors.append(f"{rel}: {m}")
    warn = lambda m: warnings.append(f"{rel}: {m}")
    qid = d.name
    if d.parent.name != "models":
        err("must be a folder directly inside models/ named after the Wikidata id"); return
    if not QID.match(qid):
        err("folder name must be a Wikidata id like Q1140634"); return
    for f in d.iterdir():
        if f.name not in FILES_OK:
            err(f"unexpected file {f.name} (allowed: {', '.join(sorted(FILES_OK))})")
    if not (d / "model.glb").exists():
        err("model.glb is missing")
    if not (d / "meta.json").exists():
        err("meta.json is missing"); return
    try:
        m = json.loads((d / "meta.json").read_text("utf-8"))
    except Exception as e:                                  # noqa: BLE001
        err(f"meta.json is not valid JSON: {e}"); return
    if m.get("id") != qid:
        err(f"meta.json id is {m.get('id')!r}, the folder is {qid}")
    if not isinstance(m.get("name"), str) or not m["name"].strip():
        err("meta.json needs a name")
    h = m.get("heading")
    if not isinstance(h, (int, float)) or not 0 <= h < 360:
        err("heading: where the main façade faces, in degrees clockwise from north, 0–359")
    p = m.get("position")
    if not (isinstance(p, dict) and isinstance(p.get("lat"), (int, float)) and isinstance(p.get("lon"), (int, float))
            and -90 <= p["lat"] <= 90 and -180 <= p["lon"] <= 180 and p.get("source")):
        err("position: {lat, lon, source} of the model's origin (MODEL_SPEC §3)")
    if m.get("license") not in LICENSES:
        err(f"license must be one of {sorted(LICENSES)}")
    au = m.get("authors")
    if not au or not all(isinstance(a, dict) and str(a.get("name", "")).strip() for a in au):
        err("authors: a list of {name, url?}, at least one")
    so = m.get("sources")
    if not so or not all(isinstance(s, dict) and (s.get("title") or s.get("url")) for s in so):
        err("sources: a list of {title, url}, at least one")
    if "based_on" in m:
        b = m["based_on"]
        if not isinstance(b, dict) or b.get("license") not in LICENSES or not b.get("author"):
            err("based_on needs author, url and a CC-BY-4.0 or CC0-1.0 license")
    check_rights(m.get("rights"), err, warn)
    if m.get("ground", "flatten") not in ("flatten", "keep"):
        err('ground: "flatten" (the default) or "keep" (MODEL_SPEC §3)')
    ga = m.get("ground_at")
    if ga is not None and not (isinstance(ga, list) and len(ga) == 2 and all(isinstance(v, (int, float)) for v in ga)):
        err("ground_at: [x, z] in metres in the model's axes")
    elif ga is not None and m.get("ground") != "keep":
        warn('ground_at is only used with "ground": "keep"')
    gl = m.get("ground_low")
    if gl is not None and not (isinstance(gl, (int, float)) and -200 <= gl <= 0):
        err("ground_low: the level (y) of the lowest ground around the model, between -200 and 0")
    elif gl is not None and m.get("ground") != "keep":
        warn('ground_low is only used with "ground": "keep"')
    su = m.get("surround")
    KINDS = ("rainforest", "forest", "conifer", "woodland", "palms", "garden")
    if su is not None:
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        ok = (isinstance(su, dict) and su.get("kind") in KINDS and num(su.get("radius")) and 50 <= su["radius"] <= 5000
              and (su.get("clear") is None or (num(su["clear"]) and 0 <= su["clear"] < su["radius"]))
              and (su.get("density") is None or (num(su["density"]) and 0 <= su["density"] <= 1)))
        if not ok:
            err(f'surround: {{"kind": one of {", ".join(KINDS)}, "radius": 50–5000 m, "clear"?: m, "density"?: 0–1}}')
    tr = m.get("terrain_radius")
    if tr is not None and not (isinstance(tr, (int, float)) and not isinstance(tr, bool) and 300 <= tr <= 5000):
        err("terrain_radius: half-width in metres of the detailed terrain (tools/terrain.py), between 300 and 5000")
    if m.get("terrain_cut") is not None and not isinstance(m.get("terrain_cut"), bool):
        err("terrain_cut: true or false (tools/terrain.py)")
    ge = m.get("ground_elev")
    if ge is not None and not (isinstance(ge, (int, float)) and not isinstance(ge, bool) and -450 <= ge <= 8900):
        err("ground_elev: the measured height above sea level (m) of the ground at y = 0, between -450 and 8900")
    gf = m.get("ground_fit")
    if gf is not None and not isinstance(gf, bool):
        err('ground_fit: true or false (MODEL_SPEC §3.1)')
    elif gf is not None and m.get("ground") != "keep":
        warn('ground_fit is only used with "ground": "keep"')
    if m.get("ground") == "keep" and gf is not False and BRIDGE.search(str(m.get("name", ""))):
        warn('it looks like a bridge: add "ground_fit": false, or Offlined will raise the ground up to the deck (MODEL_SPEC §3.1)')
    co = m.get("collision")
    if co is not None:
        box = lambda b: isinstance(b, list) and len(b) == 6 and all(isinstance(v, (int, float)) for v in b) and all(b[i] < b[i + 3] for i in range(3))
        if not (isinstance(co, dict) and set(co) <= {"voxel", "close", "add", "remove"}
                and (co.get("voxel") is None or (isinstance(co["voxel"], (int, float)) and 0.25 <= co["voxel"] <= 20))
                and (co.get("close") is None or (isinstance(co["close"], (int, float)) and 0 <= co["close"] <= 30))
                and all(isinstance(co.get(k, []), list) and all(box(b) for b in co.get(k, [])) for k in ("add", "remove"))):
            err('collision: {"voxel": 0.25–20 m, "close": 0–30 m, "add": [[x0, y0, z0, x1, y1, z1], ...], "remove": [...]} '
                "(boxes in model metres, MODEL_SPEC §3)")
    fp = m.get("footprint")
    if fp is not None and not (isinstance(fp, list) and len(fp) >= 3 and all(isinstance(q, list) and len(q) == 2 and all(isinstance(v, (int, float)) for v in q) for q in fp)):
        err("footprint: a list of at least 3 [x, z] points in metres (model space)")
    if (d / "model.glb").exists():
        info = check_glb(d / "model.glb", err, warn)
        if info and info["lo"]:
            lo, hi = info["lo"], info["hi"]
            w, hgt, dp = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
            if max(w, dp) < 2 or max(w, dp) > 4000:            # up to 4 km: long bridges, temple precincts with moats
                err(f"model is {w:.1f} × {dp:.1f} m seen from above: is it in metres?")
            if hi[1] < 1:
                err("nothing above y = 1 m: +Y must be up, y = 0 the ground")
            if lo[1] < -60:
                warn(f"the model goes {-lo[1]:.0f} m below the ground: foundations of a few metres are enough")
            check_ground(d / "model.glb", m, warn)
            if not (lo[0] <= 0 <= hi[0] and lo[2] <= 0 <= hi[2]):
                err("the origin (0, 0, 0) is outside the model: it must be on the ground, inside it (MODEL_SPEC §3)")
            if "triangles" not in m:
                warn(f"{info['triangles']:,} triangles (you can add \"triangles\" to meta.json)")
            elif abs(m["triangles"] - info["triangles"]) > 0.02 * info["triangles"] + 10:
                warn(f"meta.json says {m['triangles']:,} triangles, the model has {info['triangles']:,}")
    tj = d / "terrain.json"
    if tj.exists():
        try:
            t = json.loads(tj.read_text("utf-8"))
            import base64
            n = int(t["n"])
            if not (2 <= n <= 1200 and float(t["c"]) > 0 and len(base64.b64decode(t["h"])) == n * n * 2 and t.get("attribution")):
                raise ValueError
        except Exception:
            err("terrain.json: not a valid terrain grid — make it again with tools/terrain.py")
    else:
        warn("no terrain.json (run tools/terrain.py: the game fits the model to the real ground of its area)")
    pv = d / "preview.jpg"
    if pv.exists():
        sz = image_size(pv.read_bytes()[:65536])
        if not sz:
            err("preview.jpg is not a JPEG")
        elif sz[0] > 1920:
            warn(f"preview.jpg is {sz[0]}×{sz[1]}; 1280×720 is enough")
    else:
        warn("no preview.jpg (recommended)")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1:
        folders = [Path(a).resolve() for a in sys.argv[1:]]
    else:
        folders = [d for d in sorted((ROOT / "models").glob("*")) if d.is_dir()]
    errors, warnings = [], []
    for d in folders:
        check_folder(d, errors, warnings)
    for w in warnings:
        print("warning:", w)
    for e in errors:
        print("ERROR:  ", e)
    print(f"{len(folders)} model folder(s): {len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
