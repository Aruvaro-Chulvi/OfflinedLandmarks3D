"""
What the release needs from Wikidata for each model: its Wikipedia titles in the ten languages of Offlined,
its coordinates, and its architects with their dates (to help fill in "rights"). Standard library only.
Answers are cached in tools/.cache/wikidata/ (delete it to refresh).

    python tools/wikidata.py Q1140634          # prints what it found, ready to copy into meta.json
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

LANGS = ("en", "es", "fr", "de", "it", "pt", "ru", "zh", "ar", "hi")
API = "https://www.wikidata.org/w/api.php"
UA = {"User-Agent": "OfflinedLandmarks3D-tools/1.0 (https://github.com/Aruvaro-Chulvi/OfflinedLandmarks3D)"}
CACHE = Path(__file__).resolve().parent / ".cache" / "wikidata"


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception:                                   # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 + attempt * 3)
    return {}


def _raw(qids: list[str]) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for q in qids:
        f = CACHE / f"{q}.json"
        if f.exists():
            out[q] = json.loads(f.read_text("utf-8"))
        else:
            todo.append(q)
    for i in range(0, len(todo), 50):
        part = todo[i:i + 50]
        r = _get({"action": "wbgetentities", "ids": "|".join(part), "props": "claims|sitelinks|labels",
                  "sitefilter": "|".join(f"{l}wiki" for l in LANGS), "languages": "en"})
        for q, e in (r.get("entities") or {}).items():
            out[q] = e
            (CACHE / f"{q}.json").write_text(json.dumps(e, ensure_ascii=False), "utf-8")
    return out


def _claims(e, p):
    return [c["mainsnak"].get("datavalue", {}).get("value") for c in e.get("claims", {}).get(p, [])
            if c.get("mainsnak", {}).get("snaktype") == "value"]


def _year(v):
    try:
        return int(str(v["time"])[:5].replace("+", ""))
    except Exception:                                       # noqa: BLE001
        return None


def entities(qids: list[str]) -> dict:
    """{qid: {"titles": {lang: title}, "coord": {lat, lon}, "architects": [{name, wikidata, died}], "built": year}}"""
    raw = _raw(qids)
    arch_ids = sorted({a["id"] for e in raw.values() for a in _claims(e, "P84") if isinstance(a, dict)})
    people = _raw(arch_ids) if arch_ids else {}
    res = {}
    for q, e in raw.items():
        titles = {l: e["sitelinks"][f"{l}wiki"]["title"] for l in LANGS if f"{l}wiki" in e.get("sitelinks", {})}
        c = _claims(e, "P625")
        coord = {"lat": round(c[0]["latitude"], 6), "lon": round(c[0]["longitude"], 6)} if c else None
        arch = []
        for a in _claims(e, "P84"):
            p = people.get(a["id"], {})
            d = _claims(p, "P570")
            arch.append({"name": (p.get("labels", {}).get("en") or {}).get("value", a["id"]), "wikidata": a["id"],
                         **({"died": _year(d[0])} if d and _year(d[0]) else {})})
        built = [_year(v) for v in _claims(e, "P571") + _claims(e, "P580")]
        built = [b for b in built if b]
        res[q] = {"titles": titles, "coord": coord, "architects": arch, **({"built": min(built)} if built else {})}
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(entities(sys.argv[1:]), ensure_ascii=False, indent=1))
