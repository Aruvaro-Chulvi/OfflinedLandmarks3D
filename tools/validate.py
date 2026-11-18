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
FILES_OK = {"model.glb", "meta.json", "preview.jpg"}


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
    fp = m.get("footprint")
    if fp is not None and not (isinstance(fp, list) and len(fp) >= 3 and all(isinstance(q, list) and len(q) == 2 and all(isinstance(v, (int, float)) for v in q) for q in fp)):
        err("footprint: a list of at least 3 [x, z] points in metres (model space)")
    if (d / "model.glb").exists():
        info = check_glb(d / "model.glb", err, warn)
        if info and info["lo"]:
            lo, hi = info["lo"], info["hi"]
            w, hgt, dp = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
            if max(w, dp) < 2 or max(w, dp) > 1500:
                err(f"model is {w:.1f} × {dp:.1f} m seen from above: is it in metres?")
            if hi[1] < 1:
                err("nothing above y = 1 m: +Y must be up, y = 0 the ground")
            if lo[1] < -60:
                warn(f"the model goes {-lo[1]:.0f} m below the ground: foundations of a few metres are enough")
            if not (lo[0] <= 0 <= hi[0] and lo[2] <= 0 <= hi[2]):
                err("the origin (0, 0, 0) is outside the model: it must be on the ground, inside it (MODEL_SPEC §3)")
            if "triangles" not in m:
                warn(f"{info['triangles']:,} triangles (you can add \"triangles\" to meta.json)")
            elif abs(m["triangles"] - info["triangles"]) > 0.02 * info["triangles"] + 10:
                warn(f"meta.json says {m['triangles']:,} triangles, the model has {info['triangles']:,}")
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
