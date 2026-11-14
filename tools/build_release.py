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

Offlined reads https://github.com/Aruvaro-Chulvi/OfflinedLandmarks3D/releases/latest/download/manifest.json
and downloads the files it lists (checking size and SHA-256).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glbtool  # noqa: E402
import wikidata  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/Aruvaro-Chulvi/OfflinedLandmarks3D"
META_KEYS = ("name", "heading", "position", "authors", "license", "rights", "based_on")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tag = sys.argv[1]
    out = ROOT / "dist"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    models, authors = [], {}
    dirs = [d for d in sorted((ROOT / "models").glob("Q*")) if (d / "model.glb").exists() and (d / "meta.json").exists()]
    wd = wikidata.entities([d.name for d in dirs])
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text("utf-8"))
        scene = glbtool.read_glb(d / "model.glb")
        lo, hi = glbtool.bounds(scene)
        tris = glbtool.triangles(scene)
        name = f"{d.name}.glb"
        shutil.copy2(d / "model.glb", out / name)
        data = (out / name).read_bytes()
        m = {"id": d.name, "file": name, "size": len(data), "sha256": sha(data), "triangles": tris}
        m.update({k: meta[k] for k in META_KEYS if k in meta})
        # the far-away version: about 1/8 of the triangles, never more than 8,000
        lname = f"{d.name}-lod.glb"
        ltris, _ = glbtool.write_lod(scene, out / lname, target=max(600, min(8000, tris // 8)))
        ldata = (out / lname).read_bytes()
        m["lod"] = {"file": lname, "size": len(ldata), "sha256": sha(ldata), "triangles": ltris}
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
        print(f"{d.name} {meta.get('name')}: {tris:,} tris ({len(data) / 2**20:.2f} MB), lod {ltris:,} tris "
              f"({len(ldata) / 1024:.0f} KB), {len(m['footprint'])}-point footprint, {len(m['titles'])} titles")
    man = {
        "name": "Offlined Landmarks 3D", "version": tag, "generated": dt.date.today().isoformat(),
        "repository": REPO, "base_url": f"{REPO}/releases/download/{tag}/",
        "license": "Each model: its own licence (CC-BY-4.0 or CC0-1.0), authors in 'authors'.",
        "authors": dict(sorted(authors.items(), key=lambda kv: (-kv[1]["models"], kv[0]))),
        "countries": {c: {"models": sum(1 for m in models if m["country"] == c),
                          "size": sum(m["size"] + m["lod"]["size"] for m in models if m["country"] == c)}
                      for c in sorted({m["country"] for m in models})},
        "models": models,
    }
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), "utf-8")
    total = sum(m["size"] + m["lod"]["size"] for m in models)
    print(f"{len(models)} model(s), {total / 2**20:.2f} MB -> {out}")


if __name__ == "__main__":
    main()
