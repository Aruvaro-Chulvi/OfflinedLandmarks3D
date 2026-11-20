"""
Assemble the web page (GitHub Pages) in a folder:

    python tools/build_site.py _site

site/* + models/ (model.glb, meta.json, preview.jpg, collision.json — the collision volume, if numpy is installed)
+ models/index.json (every model with its authors,
Wikipedia titles and the number of models of each author). Standard library only; the Wikipedia titles
come from Wikidata (if it cannot be reached, the page links to Wikidata only).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "_site").resolve()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "site", out)
    dirs = [d for d in sorted((ROOT / "models").glob("Q*")) if (d / "meta.json").exists() and (d / "model.glb").exists()]
    try:
        import wikidata
        wd = wikidata.entities([d.name for d in dirs])
    except Exception as e:                                  # noqa: BLE001
        print("Wikidata not reachable:", e)
        wd = {}
    models, authors = [], {}
    for d in dirs:
        m = json.loads((d / "meta.json").read_text("utf-8"))
        dst = out / "models" / d.name
        dst.mkdir(parents=True)
        for f in ("model.glb", "meta.json", "preview.jpg"):
            if (d / f).exists():
                shutil.copy2(d / f, dst / f)
        has_col = False
        try:                                                # the collision volume, as in the release (needs numpy)
            import glbtool
            cm = m.get("collision") or {}
            col = glbtool.collision(glbtool.read_glb(d / "model.glb"), voxel=cm.get("voxel"), close=float(cm.get("close", glbtool.CLOSE_M)),
                                    add=cm.get("add", ()), remove=cm.get("remove", ()))
            (dst / "collision.json").write_text(json.dumps(col, separators=(",", ":")), "utf-8")
            has_col = True
        except ImportError:
            pass
        info = wd.get(d.name, {})
        models.append({k: m.get(k) for k in ("id", "name", "heading", "position", "authors", "license", "rights", "triangles", "sources", "notes", "ground", "ground_low")}
                      | {"titles": info.get("titles", {}), "preview": (d / "preview.jpg").exists(), "collision": has_col,
                         # ISO 3166-1 alpha-2, as in the release manifest (meta.json first, then Wikidata); "" if unknown
                         "country": str(m.get("country") or info.get("country") or "")})
        for a in m.get("authors", []):
            e = authors.setdefault(a["name"], {"models": 0})
            e["models"] += 1
            if a.get("url"):
                e["url"] = a["url"]
    (out / "models").mkdir(exist_ok=True)
    idx = {"models": models, "authors": dict(sorted(authors.items(), key=lambda kv: (-kv[1]["models"], kv[0])))}
    (out / "models" / "index.json").write_text(json.dumps(idx, ensure_ascii=False), "utf-8")
    (out / ".nojekyll").write_text("", "utf-8")
    print(f"site -> {out} ({len(models)} model(s), {len(authors)} author(s))")


if __name__ == "__main__":
    main()
