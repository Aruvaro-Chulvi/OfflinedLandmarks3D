"""
Look at a model without a browser: a small software renderer (numpy + Pillow, flat shading, vertex colours,
ground at y = 0, a 1.8 m figure for scale). For checking proportions while you build, and for preview.jpg.

    python tools/render.py models/Q1140634                      # contact sheet: front, sides, back, top, ¾ view
    python tools/render.py models/Q1140634 --preview            # writes models/Q1140634/preview.jpg (1280 × 720)
    python tools/render.py model.glb --az 30 --el 20 --out a.png # one view: az 0 = in front of the main façade
    python tools/render.py models/Q… --sheet --out sheet.png --zoom 2 --target 0,30,0   # closer, around a point

Azimuth is measured around the model from its main façade (+Z), positive towards +X; elevation above the
horizon. The top view has the main façade at the bottom.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glbtool import read_glb  # noqa: E402

SKY_TOP, SKY_LOW = np.array([0.62, 0.74, 0.88]), np.array([0.90, 0.93, 0.96])
GROUND = np.array([0.78, 0.76, 0.70])


def _lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def ground_low(path):
    """For a model with \"ground\": \"keep\" and ground_low in its meta.json: that level (else 0)."""
    meta = Path(path).parent / "meta.json"
    try:
        m = json.loads(meta.read_text("utf-8"))
        return float(m["ground_low"]) if m.get("ground") == "keep" and "ground_low" in m else 0.0
    except Exception:                                       # noqa: BLE001
        return 0.0


def load(path):
    sc = read_glb(str(path))
    mats = sc["gltf"].get("materials", [])
    V, C, Fs, S = [], [], [], []
    o = 0
    for p in sc["prims"]:
        pos = p["pos"]
        col = p["col"][:, :3] if p["col"] is not None else np.full((len(pos), 3), 0.6)
        m = mats[p["material"]] if p["material"] is not None and p["material"] < len(mats) else {}
        bc = np.array((m.get("pbrMetallicRoughness", {}).get("baseColorFactor") or [1, 1, 1, 1])[:3])
        rough = m.get("pbrMetallicRoughness", {}).get("roughnessFactor", 0.9)
        V.append(pos); C.append(_lin2srgb(col * bc)); Fs.append(p["idx"] + o); S.append(np.full(len(p["idx"]), 1 - rough))
        o += len(pos)
    V = np.vstack(V)
    V[:, 1] -= ground_low(path)          # the grey ground is drawn at y = 0: lift a model whose lowest ground is below
    return V, np.vstack(C), np.vstack(Fs), np.concatenate(S)


def person(x, z):
    """A 1.8 m figure (box body, box head) for scale."""
    V, F, C = [], [], []
    for (w, y0, y1, d, col) in ((0.45, 0.0, 1.45, 0.25, (0.85, 0.2, 0.15)), (0.22, 1.5, 1.8, 0.22, (0.9, 0.75, 0.6))):
        o = len(V)
        for yy in (y0, y1):
            for dx, dz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                V.append((x + dx * w / 2, yy, z + dz * d / 2))
        for q in ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
            F += [(o + q[0], o + q[2], o + q[1]), (o + q[0], o + q[3], o + q[2])]
        C += [col] * 8
    return np.array(V, float), np.array(F), np.array(C, float)


def ground(cx, cz, R, n=48):
    xs = np.linspace(cx - R, cx + R, n + 1); zs = np.linspace(cz - R, cz + R, n + 1)
    X, Z = np.meshgrid(xs, zs)
    V = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    F = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            F += [(a, a + n + 1, a + 1), (a + 1, a + n + 1, a + n + 2)]
    F = np.array(F)
    C = np.tile(GROUND, (len(V), 1))
    return V, F, C


def grid_lines(dr, ox, oy, W, H, ctr, R, step=10.0):
    """10 m grid over the top view (model axes), drawn on the final image."""
    k = (H / 2) / (R * 1.05)
    for g in np.arange(math.floor((ctr[0] - R * 2) / step) * step, ctr[0] + R * 2, step):
        x = ox + W / 2 + (g - ctr[0]) * k
        if ox <= x <= ox + W:
            dr.line([(x, oy), (x, oy + H)], fill=(170, 165, 150) if g else (90, 90, 200), width=1)
    for g in np.arange(math.floor((ctr[2] - R * 2) / step) * step, ctr[2] + R * 2, step):
        y = oy + H / 2 + (g - ctr[2]) * k
        if oy <= y <= oy + H:
            dr.line([(ox, y), (ox + W, y)], fill=(170, 165, 150) if g else (90, 90, 200), width=1)


def osm_rings(folder, ref):
    """OSM outline(s) of 'way/123' or 'relation/456' in this model's axes, from meta.json position and heading."""
    import json
    from osm import describe
    meta = json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))
    rings, parts = [], []
    for one in ref.split(","):
        kind, oid = one.strip().split("/")
        d = describe(kind, int(oid), float(meta["heading"]), (meta["position"]["lat"], meta["position"]["lon"]), parts=True)
        rings += d["outline_xz"] + d["courtyards_xz"]
        parts += [p["outline_xz"] for p in d.get("parts", [])]
    return rings, parts


def draw_rings(dr, ox, oy, W, H, ctr, R, rings, parts=()):
    k = (H / 2) / (R * 1.05)
    def px(x, z):
        return (ox + W / 2 + (x - ctr[0]) * k, oy + H / 2 + (z - ctr[2]) * k)
    for r in parts:
        dr.line([px(*p) for p in r] + [px(*r[0])], fill=(30, 120, 220), width=1)
    for r in rings:
        dr.line([px(*p) for p in r] + [px(*r[0])], fill=(230, 30, 30), width=2)


def render(V, C, F, S, W, H, eye, target, fov=35.0, ortho=None, up=(0, 1, 0), sun=None):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = V - eye
    xc, yc, zc = d @ r, d @ u, d @ f
    if ortho:
        k = (H / 2) / ortho
        sx, sy, depth = W / 2 + xc * k, H / 2 - yc * k, zc
    else:
        k = (H / 2) / math.tan(math.radians(fov) / 2)
        zz = np.maximum(zc, 1e-3)
        sx, sy, depth = W / 2 + xc / zz * k, H / 2 - yc / zz * k, zc
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a); ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1; n /= ln
    cen = (a + b + c) / 3
    toward = eye - cen if not ortho else -f[None, :].repeat(len(F), 0)
    flip = np.sum(n * toward, 1) < 0
    n[flip] *= -1                                             # double-sided
    L = np.asarray(sun if sun is not None else (-0.45, 0.8, 0.55), float); L /= np.linalg.norm(L)
    lam = np.clip(n @ L, 0, 1)
    sky = 0.5 + 0.5 * n[:, 1]
    hv = L + (toward / np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-9))
    hv /= np.maximum(np.linalg.norm(hv, axis=1, keepdims=True), 1e-9)
    spec = np.clip(np.sum(n * hv, 1), 0, 1) ** (8 + 120 * S) * S * 0.9
    fc = C[F].mean(1)
    shade = fc * (0.30 + 0.22 * sky + 0.62 * lam)[:, None] + spec[:, None]
    img = np.zeros((H, W, 3))
    yy = np.linspace(0, 1, H)[:, None]
    img[:] = (SKY_TOP * (1 - yy) + SKY_LOW * yy)[:, None, :]
    zb = np.full((H, W), np.inf)
    P = np.stack([sx, sy], 1)
    dz = depth
    order = np.argsort(-np.minimum(np.minimum(dz[F[:, 0]], dz[F[:, 1]]), dz[F[:, 2]]))
    for t in order:
        i0, i1, i2 = F[t]
        if min(dz[i0], dz[i1], dz[i2]) <= 0.05:
            continue
        (x0, y0), (x1, y1), (x2, y2) = P[i0], P[i1], P[i2]
        xmin, xmax = max(int(math.floor(min(x0, x1, x2))), 0), min(int(math.ceil(max(x0, x1, x2))), W - 1)
        ymin, ymax = max(int(math.floor(min(y0, y1, y2))), 0), min(int(math.ceil(max(y0, y1, y2))), H - 1)
        if xmin > xmax or ymin > ymax:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5)
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not m.any():
            continue
        if ortho:
            z = w0 * dz[i0] + w1 * dz[i1] + w2 * dz[i2]
        else:                                          # perspective-correct depth: 1/z is linear on screen
            z = 1.0 / (w0 / dz[i0] + w1 / dz[i1] + w2 / dz[i2])
        sub = zb[ymin:ymax + 1, xmin:xmax + 1]
        m &= z < sub
        if not m.any():
            continue
        sub[m] = z[m]
        img[ymin:ymax + 1, xmin:xmax + 1][m] = shade[t]
    return img


def view(V, C, F, S, W, H, az, el, zoom=1.0, target=None, top=False, bounds=None):
    lo, hi = (V.min(0), V.max(0)) if bounds is None else (bounds[0].copy(), bounds[1].copy())
    lo[1] = max(lo[1], 0.0)
    ctr = (lo + hi) / 2 if target is None else np.asarray(target, float)
    R = np.linalg.norm(hi - lo) / 2 / zoom
    if top:
        eye = ctr + np.array([0, R * 4, 0.001])
        return render(V, C, F, S, W, H, eye, ctr, ortho=R * 1.05, up=(0, 0, -1))
    a, e = math.radians(az), math.radians(el)
    dist = R / math.sin(math.radians(17.5)) * 0.95
    eye = ctr + dist * np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
    return render(V, C, F, S, W, H, eye, ctr)


def scene(path, with_person=True):
    V, C, F, S = load(path)
    lo, hi = V.min(0), V.max(0)
    R = max(np.linalg.norm(hi - lo), 20)
    gV, gF, gC = ground((lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2, R * 1.6)
    parts = [(V, F, C, S), (gV, gF, gC, np.zeros(len(gF)))]
    if with_person:
        pV, pF, pC = person(hi[0] + 2.0, hi[2] + 2.0)
        parts.append((pV, pF, pC, np.zeros(len(pF))))
    o, VV, FF, CC, SS = 0, [], [], [], []
    for v, f, c, s in parts:
        VV.append(v); FF.append(f + o); CC.append(c); SS.append(s); o += len(v)
    return np.vstack(VV), np.vstack(CC), np.vstack(FF), np.concatenate(SS), (V.min(0), V.max(0), len(F))


def to_img(a, ss=1):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    if ss > 1:
        im = im.resize((im.width // ss, im.height // ss), Image.LANCZOS)
    return im


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="a model folder (models/Q…) or a .glb")
    ap.add_argument("--preview", action="store_true", help="write preview.jpg (1280 × 720) into the model folder")
    ap.add_argument("--sheet", action="store_true", help="contact sheet of six views (the default without --az)")
    ap.add_argument("--az", type=float); ap.add_argument("--el", type=float, default=20.0)
    ap.add_argument("--zoom", type=float, default=1.0); ap.add_argument("--target", help="x,y,z to look at")
    ap.add_argument("--size", default="1280x720"); ap.add_argument("--out")
    ap.add_argument("--no-person", action="store_true")
    ap.add_argument("--top", action="store_true", help="one large top view (with --osm: the OSM outline over it)")
    ap.add_argument("--osm", help="way/123 or relation/456 (several: way/1,way/2): draw that OSM outline (red) and its building:parts (blue) "
                                  "over the top view, placed with meta.json position and heading")
    a = ap.parse_args(argv)
    p = Path(a.model)
    glb = p / "model.glb" if p.is_dir() else p
    V, C, F, S, (lo, hi, ntri) = scene(glb, not a.no_person)
    tgt = tuple(map(float, a.target.split(","))) if a.target else None
    if a.preview:
        img = to_img(view(V, C, F, S, 2560, 1440, -35, 18, a.zoom, tgt, bounds=(lo, hi)), 2)
        out = Path(a.out) if a.out else (p if p.is_dir() else p.parent) / "preview.jpg"
        img.save(out, quality=88); print(out); return 0
    def _frame():
        l2, h2 = lo.copy(), hi.copy(); l2[1] = max(l2[1], 0.0)
        ctr = (l2 + h2) / 2 if tgt is None else np.asarray(tgt, float)
        return ctr, np.linalg.norm(h2 - l2) / 2 / a.zoom
    rings = osm_rings(p, a.osm) if a.osm else None
    if a.top:
        W, H = map(int, a.size.split("x"))
        img = to_img(view(V, C, F, S, W * 2, H * 2, 0, 90, a.zoom, tgt, True, bounds=(lo, hi)), 2)
        dr = ImageDraw.Draw(img); ctr, Rr = _frame()
        grid_lines(dr, 0, 0, W, H, ctr, Rr)
        if rings:
            draw_rings(dr, 0, 0, W, H, ctr, Rr, *rings)
        dr.text((8, 6), "top view, main facade (+Z) at the bottom, +X to the right; grid 10 m (blue = model axes)"
                + ("; red = OSM outline, blue thin = OSM building:parts" if rings else ""), fill=(20, 20, 20))
        out = Path(a.out or "top.png"); img.save(out); print(out); return 0
    if a.az is not None and not a.sheet:
        W, H = map(int, a.size.split("x"))
        img = to_img(view(V, C, F, S, W * 2, H * 2, a.az, a.el, a.zoom, tgt, bounds=(lo, hi)), 2)
        out = Path(a.out or "render.png"); img.save(out); print(out); return 0
    tiles = [("front (+Z)", 0, 8, False), ("right side (+X)", 90, 8, False), ("back (-Z)", 180, 8, False),
             ("left side (-X)", -90, 8, False), ("top (front at the bottom)", 0, 90, True), ("3/4 view", -35, 25, False)]
    W, H = 640, 400
    sheet = Image.new("RGB", (W * 3, H * 2 + 28), (250, 250, 250))
    dr = ImageDraw.Draw(sheet)
    for i, (name, az, el, top) in enumerate(tiles):
        im = to_img(view(V, C, F, S, W * 2, H * 2, az, el, a.zoom, tgt, top, bounds=(lo, hi)), 2)
        sheet.paste(im, ((i % 3) * W, (i // 3) * H))
        if top and rings:
            ctr, Rr = _frame()
            draw_rings(dr, (i % 3) * W, (i // 3) * H, W, H, ctr, Rr, *rings)
        dr.text(((i % 3) * W + 8, (i // 3) * H + 6), name, fill=(20, 20, 20))
    dr.text((8, H * 2 + 8), f"{glb}: {ntri:,} triangles, x {lo[0]:.1f}..{hi[0]:.1f}  y {lo[1]:.1f}..{hi[1]:.1f}  "
            f"z {lo[2]:.1f}..{hi[2]:.1f} m   (red figure = 1.8 m, ground squares = 10 m)", fill=(20, 20, 20))
    out = Path(a.out or "sheet.png"); sheet.save(out); print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
