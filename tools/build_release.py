"""
Prepare a release: every model as one file, its far-away version (-lod.glb, made here), and manifest.json
(what Offlined downloads).

    pip install -r tools/requirements.txt
    python tools/build_release.py landmarks-2026-11-13        -> dist/
    gh release create landmarks-2026-11-13 dist/* --title ... --notes ...

For each model the manifest carries what a program needs to put it in place without opening it: position,
heading, footprint (the area where generated buildings, trees and street furniture are cleared), height,
the Wikipedia titles in the ten Offlined languages (from Wikidata, to find the article), authors, licence,
rights status — and, at the top, every author with the number of models they have in the release.

Every .glb goes into the release gzip-compressed (Q….glb.gz, Q…-lod.glb.gz: about a fifth of the size). In the
manifest, "file", "size" and "sha256" describe the compressed file that is downloaded; "encoding": "gzip" says so
and "raw_size" is the size of the .glb inside. Offlined keeps them compressed and serves them with
Content-Encoding: gzip, so the browser unpacks them.

The detailed terrain of the model's area (models/Q…/terrain.json, made by tools/terrain.py) goes in Q…-ter.json.gz
(entry "terrain" in the manifest, with its source and attribution); Offlined uses it instead of its 90 m relief.

The collision volume goes in a file of its own, Q…-col.json.gz (entry "collision" in the manifest): the model
turned into a few hundred or thousand boxes (glbtool.collision) — what a plane can really hit, with the open air of
arches, bridges and gaps between towers left open. meta.json "collision" may tune it: {"voxel": metres, "close":
metres (gaps narrower than this are closed; default 6, no plane fits), "add": [[x0, y0, z0, x1, y1, z1], ...], "remove": [...]} (boxes in model metres).

Offlined reads https://github.com/Aruvaro-Chulvi/OfflinedLandmarks3D/releases/latest/download/manifest.json
and downloads the files it lists (checking size and SHA-256).

Each model and far-away version also gets a meshopt copy (Q….mo.glb.gz, Q…-lod.mo.glb.gz; entry "mo" inside the
model's entry and inside "lod", same fields): quantized and compressed geometry, about a third of the gzipped size
(tools/meshopt/compress.mjs; needs Node.js and, once, `npm ci` in tools/meshopt). Offlined 4.x downloads "mo";
older Offlined ignores it. The models with "ground": "keep" get it only for their far-away version: Offlined reads
their full model in Python to fit the terrain to its underside, and that reader takes plain glTF only.
Without Node the release is built as before, without "mo".
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glbtool  # noqa: E402
import wikidata  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/Aruvaro-Chulvi/OfflinedLandmarks3D"
META_KEYS = ("name", "heading", "position", "authors", "license", "rights", "based_on", "ground", "ground_at", "ground_low", "ground_fit", "ground_elev", "surround")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def packed(raw: bytes, path: Path) -> dict:
    """Write raw (a .glb) gzip-compressed to path (reproducible: no name, no date) -> its manifest entry."""
    data = gzip.compress(raw, compresslevel=9, mtime=0)
    path.write_bytes(data)
    return {"file": path.name, "size": len(data), "sha256": sha(data), "encoding": "gzip", "raw_size": len(raw)}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tag = sys.argv[1]
    out = ROOT / "dist"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    models, authors = [], {}
    tmp, mo_jobs = out / "_mo", []
    dirs =[d for d in sorted((ROOT / "models").glob("Q*")) if (d / "model.glb").exists() and (d / "meta.json").exists()]
    wd = wikidata.entities([d.name for d in dirs])
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text("utf-8"))
        scene = glbtool.read_glb(d / "model.glb")
        lo, hi = glbtool.bounds(scene)
        tris = glbtool.triangles(scene)
        m = {"id": d.name, **packed((d / "model.glb").read_bytes(), out / f"{d.name}.glb.gz"), "triangles": tris}
        m.update({k: meta[k] for k in META_KEYS if k in meta})
        # the far-away version: about 1/8 of the triangles, never more than 8,000
        lraw = out / f"{d.name}-lod.glb"
        ltris, _ = glbtool.write_lod(scene, lraw, target=max(600, min(8000, tris // 8)))
        m["lod"] = {**packed(lraw.read_bytes(), out / f"{d.name}-lod.glb.gz"), "triangles": ltris}
        # (for the meshopt copies, made all at once below)
        tmp.mkdir(exist_ok=True)
        shutil.move(str(lraw), str(tmp / f"{d.name}-lod.glb"))
        mo_jobs.append((m["lod"], tmp / f"{d.name}-lod.glb", out / f"{d.name}-lod.mo.glb.gz"))
        if meta.get("ground") != "keep":
            shutil.copy2(d / "model.glb", tmp / f"{d.name}.glb")
            mo_jobs.append((m, tmp / f"{d.name}.glb", out / f"{d.name}.mo.glb.gz"))
        cm = meta.get("collision") or {}
        col = glbtool.collision(scene, voxel=cm.get("voxel"), close=float(cm.get("close", glbtool.CLOSE_M)),
                                add=cm.get("add", ()), remove=cm.get("remove", ()))
        craw = json.dumps(col, separators=(",", ":")).encode()
        m["collision"] = {**packed(craw, out / f"{d.name}-col.json.gz"), "boxes": len(col["boxes"]) // 6, "voxel": col["voxel"]}
        # the detailed terrain of its area (tools/terrain.py), if it has one: Offlined uses it instead of its 90 m relief
        if (d / "terrain.json").exists():
            traw = (d / "terrain.json").read_bytes()
            t = json.loads(traw)
            m["terrain"] = {**packed(traw, out / f"{d.name}-ter.json.gz"), "source": t.get("source"), "cell": t.get("c"),
                            "attribution": t.get("attribution")}
        m["footprint"] = meta.get("footprint") or glbtool.footprint(scene)
        m["height"] = round(float(hi[1]), 2)
        m["base"] = round(float(lo[1]), 2)
        m["radius"] = round(glbtool.radius(scene), 2)
        info = wd.get(d.name, {})
        m["titles"] = info.get("titles", {})
        m["country"] = str(meta.get("country") or info.get("country") or "")      # ISO 3166-1 alpha-2, "" if unknown
        if info.get("coord"):
            m["wikidata_position"] = info["coord"]
        models.append(m)
        for a in meta.get("authors", []):
            e = authors.setdefault(a["name"], {"models": 0})
            e["models"] += 1
            if a.get("url"):
                e["url"] = a["url"]
        print(f"{d.name} {meta.get('name')}: {tris:,} tris ({m['raw_size'] / 2**20:.2f} MB, {m['size'] / 2**20:.2f} MB "
              f"gzipped), lod {ltris:,} tris ({m['lod']['size'] / 1024:.0f} KB gzipped), {len(m['footprint'])}-point "
              f"footprint, {m['collision']['boxes']} collision boxes ({m['collision']['voxel']} m), {len(m['titles'])} titles")
    add_meshopt(mo_jobs)
    shutil.rmtree(tmp, ignore_errors=True)
    man = {
        "name": "Offlined Landmarks 3D", "format": 2, "version": tag, "generated": dt.date.today().isoformat(),
        "repository": REPO, "base_url": f"{REPO}/releases/download/{tag}/",
        "license": "Each model: its own licence (CC-BY-4.0 or CC0-1.0), authors in 'authors'.",
        "authors": dict(sorted(authors.items(), key=lambda kv: (-kv[1]["models"], kv[0]))),
        "countries": {c: {"models": sum(1 for m in models if m["country"] == c),
                          "size": sum(m["size"] + m["lod"]["size"] + m["collision"]["size"] + m.get("terrain", {}).get("size", 0)
                                   for m in models if m["country"] == c)}
                      for c in sorted({m["country"] for m in models})},
        "models": models,
    }
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), "utf-8")
    total = sum(m["size"] + m["lod"]["size"] + m["collision"]["size"] for m in models)
    mo = sum(m.get("mo", m)["size"] + m["lod"].get("mo", m["lod"])["size"] + m["collision"]["size"] for m in models)
    print(f"{len(models)} model(s), {total / 2**20:.2f} MB ({mo / 2**20:.2f} MB with meshopt) -> {out}")


def add_meshopt(jobs):
    """Run tools/meshopt/compress.mjs on every (entry, plain .glb, .mo.glb.gz) and add "mo" to the entries."""
    tool = ROOT / "tools" / "meshopt" / "compress.mjs"
    node = shutil.which("node")
    if not jobs:
        return
    if not node or not (tool.parent / "node_modules").is_dir():
        print("meshopt: skipped (needs Node.js and `npm ci` in tools/meshopt)")
        return
    args = []
    for _, src, _ in jobs:
        args += [str(src), str(src.with_suffix(".mo.glb"))]
    for i in range(0, len(args), 80):                       # (Windows: command lines of limited length)
        subprocess.run([node, str(tool), *args[i:i + 80]], check=False)
    for e, src, dst in jobs:
        f = src.with_suffix(".mo.glb")
        if f.is_file():                                     # (the tool skips a copy that does not match)
            e["mo"] = packed(f.read_bytes(), dst)


if __name__ == "__main__":
    main()
