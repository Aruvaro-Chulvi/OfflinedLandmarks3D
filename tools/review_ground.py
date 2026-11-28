"""
How well each model sits on the real ground of its area -> docs/GROUND_REVIEW.md

    python tools/review_ground.py              # every model that has a terrain.json (tools/terrain.py)
    python tools/review_ground.py Q171517      # these models

For every model with models/<Q>/terrain.json it places the model the way Offlined does (MODEL_SPEC §3: "flatten" on the
lowest ground of its footprint; "keep" with y = 0 on the ground at ground_at, or at ground_elev if given) and compares,
on a 3 m grid seen from above:
  · floating — at the model's edge, how far the real ground is BELOW the model's foot (the gap the game has to hide by
    raising the ground; with a detailed terrain it raises it 3 m at most, so a bigger gap stays visible)
  · buried — how far the real ground is ABOVE what the model shows to the sky (a courtyard, a terrace, a square) and
    above ground_low (the game cuts the ground down there; a big cut leaves a pit around it)
Only for "keep" models: a "flatten" model gets its footprint levelled by the game at the median height of its
ground, so for it the review gives the cut (high side) and the fill (low side) that this makes instead. The parts
of the edge that hang over lower parts of the model itself (a tower's platform, a balcony, the arches of the Eiffel
Tower) are not counted as floating; the sea (≤ 0.5 m) is taken as 0.
A model is flagged when it floats more than 3 m along more than 5 % of its edge, is buried more than 3 m under more
than 5 % of its open floors, or ("flatten") needs a cut or a fill of more than 6 m. Usual fixes: a wrong ground_at or
position, a missing ground_elev, "flatten" for a model on a slope (use "keep"), a model without its plinth or its
retaining walls.

numpy + Pillow (tools/requirements.txt). No connection needed.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glbtool  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CELL = 3.0
TOL = 3.0          # m
FLAT_MAX = 6.0     # m de desmonte o de terraplén en un «flatten»

# Revisados a mano (2026-11-27): marcados por los números, pero aceptados por este motivo (≈ en el informe)
ACCEPTED = {
    "Q10285": "the Colosseum stands in the valley between the Palatine, the Esquiline and the Caelian: the level square around it is real",
    "Q1369687": "a stadium on a slope; the game levels it at its median height (about 7 m of cut and 8 m of fill)",
    "Q164027": "a stadium on the slope of the Castellana; about 7 m of cut on the north side",
    "Q2981": "the footprint reaches down to the Seine quays: the fill is on the river side",
    "Q43473": "the footprint takes in the temple's terraces and the forest around (SRTM sees the trees); the game levels it",
    "Q45178": "on the platform of Bennelong Point, a few metres above the water",
    "Q10288": "the model carries the rock of the Acropolis, sharper than SRTM: the game cuts the blunt hill down to it",
    "Q79961": "the model carries the top of Corcovado, sharper than SRTM: the game cuts the blunt summit down to it",
    "Q9141": "SRTM sees the trees of the gardens: the game cuts them down to the model's paths and lawns",
    "Q12512": "ground_elev fitted (34 m): what is left is SRTM mixing in the Vatican buildings around",
}
SHARE = 0.05


def load_terrain(d: Path):
    t = json.loads((d / "terrain.json").read_text("utf-8"))
    n = t["n"]
    q = np.frombuffer(base64.b64decode(t["h"]), "<i2").reshape(n, n).astype(np.float64)
    h = np.where(q == -32768, np.nan, t["off"] + q / 10)

    def at(east, north):
        fi = (np.asarray(east) - t["x0"]) / t["c"]
        fj = (t["y0"] - np.asarray(north)) / t["c"]
        i0 = np.clip(np.floor(fi).astype(int), 0, n - 2); j0 = np.clip(np.floor(fj).astype(int), 0, n - 2)
        tx = np.clip(fi - i0, 0, 1); tz = np.clip(fj - j0, 0, 1)
        return (h[j0, i0] * (1 - tx) * (1 - tz) + h[j0, i0 + 1] * tx * (1 - tz)
                + h[j0 + 1, i0] * (1 - tx) * tz + h[j0 + 1, i0 + 1] * tx * tz)
    return t, at


def rasters(scene):
    """Model axes -> (lo, up, x0, z0, nx, nz): lowest point of the model and highest upward face per 3 m cell."""
    tris = np.vstack([p["pos"][p["idx"]] for p in scene["prims"]])          # (m, 3, 3)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    keep = ln > 1e-9
    tris, a, b, c, nrm, ln = tris[keep], a[keep], b[keep], c[keep], nrm[keep], ln[keep]
    ny = nrm[:, 1] / ln
    span = tris[:, :, 1].max(1) - tris[:, :, 1].min(1)
    wall = np.abs(ny) < 0.25
    P = tris.reshape(-1, 3)
    x0 = math.floor(P[:, 0].min() / CELL) * CELL; z0 = math.floor(P[:, 2].min() / CELL) * CELL
    nx = int(math.ceil((P[:, 0].max() - x0) / CELL)) + 1; nz = int(math.ceil((P[:, 2].max() - z0) / CELL)) + 1
    lo = np.full(nx * nz, np.inf); up = np.full(nx * nz, -np.inf)

    def put(pts, arr, fn):
        i = ((pts[:, 0] - x0) / CELL).astype(int); j = ((pts[:, 2] - z0) / CELL).astype(int)
        ok = (i >= 0) & (j >= 0) & (i < nx) & (j < nz)
        fn.at(arr, j[ok] * nx + i[ok], pts[ok, 1])

    # paredes de 2 m o más: sus aristas, cada metro
    W = tris[wall & (span >= 2.0)]
    for e0, e1 in ((0, 1), (1, 2), (2, 0)):
        A, B = W[:, e0], W[:, e1]
        L = np.linalg.norm((B - A)[:, [0, 2]], axis=1)
        k = np.maximum(1, np.ceil(L).astype(int))
        rep = np.repeat(np.arange(len(W)), k + 1)
        f = np.concatenate([np.linspace(0, 1, kk + 1) for kk in k]) if len(k) else np.zeros(0)
        pts = A[rep] + (B - A)[rep] * f[:, None]
        put(pts, lo, np.minimum)
    # lo demás: puntos repartidos por cada triángulo (uno cada ~2 m²), sus vértices y su centro
    F = ~wall
    T, nyF = tris[F], ny[F]
    area = ln[F] / 2
    k = np.clip(np.ceil(area / 2.0).astype(int), 1, 400000)   # big flat triangles (a paved court) need many points
    rng = np.random.default_rng(1)
    rep = np.repeat(np.arange(len(T)), k)
    u, v = rng.random(len(rep)), rng.random(len(rep))
    flip = u + v > 1; u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = T[rep, 0] + (T[rep, 1] - T[rep, 0]) * u[:, None] + (T[rep, 2] - T[rep, 0]) * v[:, None]
    pts = np.vstack([pts, T.reshape(-1, 3), T.mean(1)])
    upm = np.concatenate([nyF[rep] > 0, np.repeat(nyF > 0, 3), nyF > 0])
    put(pts, lo, np.minimum)
    put(pts[upm], up, np.maximum)
    return lo.reshape(nz, nx), up.reshape(nz, nx), x0, z0, nx, nz


def review(qid: str) -> dict:
    d = ROOT / "models" / qid
    meta = json.loads((d / "meta.json").read_text("utf-8"))
    t, ter0 = load_terrain(d)
    ter = lambda e, n: np.maximum(np.nan_to_num(ter0(e, n), nan=0.0), 0.0)     # el mar, a 0 (como el juego)
    scene = glbtool.read_glb(d / "model.glb")
    lo, up, x0, z0, nx, nz = rasters(scene)
    occ = np.isfinite(lo)
    # borde: celdas del modelo junto a una vacía (o al límite)
    pad = np.pad(occ, 1, constant_values=False)
    nb_empty = np.zeros_like(occ)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            nb_empty |= ~pad[1 + dj:1 + dj + nz, 1 + di:1 + di + nx]
    edge = occ & nb_empty
    lo_pad = np.pad(np.where(occ, lo, np.inf), 1, constant_values=np.inf)
    lo_nb = np.full_like(lo, np.inf)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            lo_nb = np.minimum(lo_nb, lo_pad[1 + dj:1 + dj + nz, 1 + di:1 + di + nx])
    # voladizo: el borde queda muy por encima de otra parte del modelo a menos de 15 m (el fuste de una torre)
    R5 = 5
    lo_pad5 = np.pad(np.where(occ, lo, np.inf), R5, constant_values=np.inf)
    lo_loc = np.full_like(lo, np.inf)
    for dj in range(-R5, R5 + 1):
        for di in range(-R5, R5 + 1):
            if di * di + dj * dj <= R5 * R5:
                lo_loc = np.minimum(lo_loc, lo_pad5[R5 + dj:R5 + dj + nz, R5 + di:R5 + di + nx])
    edge &= lo_nb - lo_loc <= TOL
    # del modelo al terreno (este, norte): como Offlined (th = π − heading)
    th = math.pi - math.radians(meta.get("heading", 0)); cs, sn = math.cos(th), math.sin(th)
    X, Z = np.meshgrid(x0 + (np.arange(nx) + 0.5) * CELL, z0 + (np.arange(nz) + 0.5) * CELL)
    east, south = X * cs + Z * sn, -X * sn + Z * cs
    g = ter(east, -south)
    keep = meta.get("ground") == "keep"
    if meta.get("ground_elev") is not None:
        base = float(meta["ground_elev"]); how = "ground_elev"
    elif keep:
        gx, gz = (meta.get("ground_at") or [0, 0])
        base = float(ter(gx * cs + gz * sn, -(-gx * sn + gz * cs))); how = "terrain at ground_at"
    else:
        base = float(np.nanmedian(np.where(occ, g, np.nan))); how = "median terrain of the footprint"
    gl = float(meta.get("ground_low", -1e9)) if keep else -1e9
    floating = np.where(edge, (base + lo_nb) - g, np.nan)                       # > 0: hueco bajo el borde
    open_up = occ & np.isfinite(up) & (up >= gl)
    buried = np.where(open_up, g - (base + up), np.nan)                          # > 0: tapado
    fl_v = floating[np.isfinite(floating)]; bu_v = buried[np.isfinite(buried)]
    fl_share = float((fl_v > TOL).mean()) if fl_v.size else 0.0
    bu_share = float((bu_v > TOL).mean()) if bu_v.size else 0.0
    drop = float(np.nanmax(np.where(occ, g, np.nan)) - np.nanmin(np.where(occ, g, np.nan)))
    # El mejor apoyo: la altura de y = 0 (como "keep") que deja menos borde flotando y menos suelo tapado. Para un
    # "flatten", lo que pasaría si fuese "keep" (sin ground_low: cuenta todo lo que mira al cielo)
    fl0 = (lo_nb - g)[edge & np.isfinite(lo_nb - g)]
    up_all = occ & np.isfinite(up) & ((up >= gl) if keep else True)
    bu0 = (g - up)[up_all]
    best = None
    for dlt in np.arange(-40.0, 40.01, 0.5):
        B = base + dlt
        fs_ = float((fl0 + B > TOL).mean()) if fl0.size else 0.0
        bs_ = float((bu0 - B > TOL).mean()) if bu0.size else 0.0
        cost = 3 * fs_ + bs_ + 0.002 * abs(dlt)          # flotar se ve (un hueco); lo tapado el juego lo recorta
        if best is None or cost < best[0]:
            best = (cost, round(float(B), 1), fs_, bs_)
    fit = {"fit_base": best[1], "fit_float": best[2], "fit_buried": best[3], "fit_ok": best[2] <= SHARE and best[3] <= SHARE}
    if not keep:                                    # «flatten»: el juego allana la planta a su mediana: desmonte y terraplén
        cut = float(np.nanmax(np.where(occ, g, np.nan)) - base); fill = float(base - np.nanmin(np.where(occ, g, np.nan)))
        return {"id": qid, "name": meta.get("name", ""), "ground": "flatten", "base": round(base, 1), "base_from": how,
                "source": t.get("source"), "drop": round(drop, 1), "cut": round(cut, 1), "fill": round(fill, 1),
                "float_max": None, "float_share": 0.0, "buried_max": None, "buried_share": 0.0,
                "flag": max(cut, fill) > FLAT_MAX, **fit}
    return {"id": qid, "name": meta.get("name", ""), "ground": meta.get("ground", "flatten"), "base": round(base, 1),
            "base_from": how, "source": t.get("source"), "float_max": round(float(fl_v.max()), 1) if fl_v.size else 0.0,
            "float_share": fl_share, "buried_max": round(float(bu_v.max()), 1) if bu_v.size else 0.0,
            "buried_share": bu_share, "drop": round(drop, 1), "flag": fl_share > SHARE or bu_share > SHARE, **fit}


def main():
    ids = sys.argv[1:] or [d.name for d in sorted((ROOT / "models").glob("Q*")) if (d / "terrain.json").exists()]
    rows, missing = [], [d.name for d in sorted((ROOT / "models").glob("Q*"))
                         if (d / "model.glb").exists() and not (d / "terrain.json").exists()]
    for q in ids:
        m = json.loads((ROOT / "models" / q / "meta.json").read_text("utf-8"))
        if m.get("ground_fit") is False:                                        # puentes, presas: no se apoyan en el suelo
            print(f"{q} {m.get('name', '')}: ground_fit false, skipped"); continue
        try:
            r = review(q)
        except Exception as ex:                                                 # noqa: BLE001
            print(f"{q}: ERROR {ex}"); continue
        rows.append(r)
        what = (f"drop across the footprint {r['drop']} m (cut {r['cut']} m, fill {r['fill']} m)" if r["ground"] == "flatten" else
                f"floating max {r['float_max']} m ({r['float_share']:.0%} of the edge > {TOL:g} m), "
                f"buried max {r['buried_max']} m ({r['buried_share']:.0%} > {TOL:g} m)")
        bf = (f"   best y = 0 at {r['fit_base']} m (now {r['base']}): floating {r['fit_float']:.0%}, buried {r['fit_buried']:.0%}"
              f"{' — fits as keep' if r['fit_ok'] else ''}")
        print(f"{q} {r['name']} [{r['ground']}]: {what}{'  <- REVIEW' if r['flag'] else ''}")
        if r["flag"]:
            print(bf)
    rows.sort(key=lambda r: (not r["flag"], -(r["float_share"] + r["buried_share"])))
    out = ["# Ground review", "",
           f"Made by `tools/review_ground.py` on {dt.date.today().isoformat()}: how each model sits on the real ground "
           "of its area (`terrain.json`, from `tools/terrain.py`). See the tool for what the numbers mean.", "",
           f"Flagged (⚠): floating or buried more than {TOL:g} m along more than {SHARE:.0%} of the edge / open floors.", "",
           "Best fit: the height of y = 0 (as \"keep\") that leaves the least edge floating and the least open floor buried, "
           "and how much of each would remain; ✓ = both under 5 %, so the model fits its ground as \"keep\" at that height.", "",
           "| | Model | ground | base (m) | from | terrain | drop in footprint (m) | floating max (m) | edge > 3 m | buried max (m) | floors > 3 m | best fit |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    dash = lambda v, f="{}": "–" if v is None else f.format(v)
    for r in rows:
        k = r["ground"] == "keep"
        fs = f"{r['float_share']:.0%}" if k else "–"
        bs = f"{r['buried_share']:.0%}" if k else "–"
        mark = ("≈" if r["id"] in ACCEPTED else "⚠") if r["flag"] else "✓"
        out.append(f"| {mark} | {r['id']} {r['name']} | {r['ground']} | {r['base']} | {r['base_from']} | "
                   f"{r['source']} | {r['drop']} | {dash(r['float_max'])} | {fs} | {dash(r['buried_max'])} | {bs} | "
                   f"{r['fit_base']} m: {r['fit_float']:.0%} / {r['fit_buried']:.0%} {'✓' if r['fit_ok'] else ''} |")
    acc = [r for r in rows if r["flag"] and r["id"] in ACCEPTED]
    if acc:
        out += ["", "## Accepted after review (≈)", ""] + [f"- **{r['id']} {r['name']}** — {ACCEPTED[r['id']]}." for r in acc]
    if missing:
        out += ["", f"Without terrain.json yet ({len(missing)}): " + ", ".join(missing)]
    if not sys.argv[1:]:
        (ROOT / "docs" / "GROUND_REVIEW.md").write_text("\n".join(out) + "\n", "utf-8")
        print("-> docs/GROUND_REVIEW.md")


if __name__ == "__main__":
    main()
