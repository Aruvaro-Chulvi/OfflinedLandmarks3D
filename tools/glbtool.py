"""
glbtool — read a .glb, measure it, and make its far-away version (MIT). Needs numpy.

    from glbtool import read_glb, footprint, simplify, write_lod
    scene = read_glb("models/Q1140634/model.glb")
    hull = footprint(scene)                  # [[x, z], ...] seen from above, in model metres
    write_lod(scene, "dist/Q1140634-lod.glb", target=6000)

Only what MODEL_SPEC.md allows is read: triangles (mode 4), POSITION, NORMAL, COLOR_0, TEXCOORD_0, indices,
KHR_mesh_quantization (normalized integers), materials with embedded images. Node transforms are applied
(the whole model ends up in model space).
"""
from __future__ import annotations

import json
import math
import struct

import numpy as np

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
NORM = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}


def _chunks(b: bytes):
    if b[:4] != b"glTF":
        raise ValueError("not a binary glTF")
    off, js, bin_ = 12, None, b""
    while off + 8 <= len(b):
        ln, ty = struct.unpack("<II", b[off:off + 8])
        if ty == 0x4E4F534A:
            js = json.loads(b[off + 8:off + 8 + ln])
        elif ty == 0x004E4942:
            bin_ = b[off + 8:off + 8 + ln]
        off += 8 + ln
    return js, bin_


def _accessor(g, binc, i):
    a = g["accessors"][i]
    bv = g["bufferViews"][a["bufferView"]]
    dt = np.dtype(COMP[a["componentType"]]).newbyteorder("<")
    n = NCOMP[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0) or dt.itemsize * n
    raw = np.frombuffer(binc, np.uint8, count=stride * (a["count"] - 1) + dt.itemsize * n, offset=start)
    if stride == dt.itemsize * n:
        arr = raw.view(dt).reshape(a["count"], n)
    else:
        arr = np.lib.stride_tricks.as_strided(raw, (a["count"], n * dt.itemsize), (stride, 1)).copy().view(dt).reshape(a["count"], n)
    arr = arr.astype(np.float64) if dt.kind == "f" else arr.astype(np.float64)
    if a.get("normalized") and a["componentType"] in NORM:
        arr = arr / NORM[a["componentType"]]
        if a["componentType"] in (5120, 5122):
            arr = np.maximum(arr, -1.0)
    return arr


def _node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    t = np.array(n.get("translation", [0, 0, 0]), float)
    x, y, z, w = n.get("rotation", [0, 0, 0, 1])
    s = np.array(n.get("scale", [1, 1, 1]), float)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R * s
    M[:3, 3] = t
    return M


def read_glb(path):
    """-> {"prims": [{pos, nor, col, uv, idx, material, node}], "gltf": json, "bin": bytes}"""
    b = open(path, "rb").read()
    g, binc = _chunks(b)
    prims = []

    def walk(ni, parent):
        n = g["nodes"][ni]
        M = parent @ _node_matrix(n)
        if "mesh" in n:
            for p in g["meshes"][n["mesh"]]["primitives"]:
                if p.get("mode", 4) != 4:
                    continue
                at = p["attributes"]
                pos = _accessor(g, binc, at["POSITION"])
                pos = (np.hstack([pos, np.ones((len(pos), 1))]) @ M.T)[:, :3]
                idx = _accessor(g, binc, p["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in p \
                    else np.arange(len(pos)).reshape(-1, 3)
                col = _accessor(g, binc, at["COLOR_0"]) if "COLOR_0" in at else None
                if col is not None and col.shape[1] == 3:
                    col = np.hstack([col, np.ones((len(col), 1))])
                uv = _accessor(g, binc, at["TEXCOORD_0"]) if "TEXCOORD_0" in at else None
                if np.linalg.det(M[:3, :3]) < 0:
                    idx = idx[:, ::-1]
                prims.append({"pos": pos, "col": col, "uv": uv, "idx": idx, "material": p.get("material"), "node": n.get("name", "")})
        for c in n.get("children", []):
            walk(c, M)

    for ni in g["scenes"][g.get("scene", 0)]["nodes"]:
        walk(ni, np.eye(4))
    return {"prims": prims, "gltf": g, "bin": binc}


def bounds(scene):
    P = np.vstack([p["pos"] for p in scene["prims"]])
    return P.min(0), P.max(0)


def triangles(scene):
    return int(sum(len(p["idx"]) for p in scene["prims"]))


def _hull(pts):
    """Convex hull (Andrew's monotone chain), counter-clockwise, of (N,2) points."""
    P = sorted(set(map(tuple, np.round(pts, 3))))
    if len(P) < 3:
        return P
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def footprint(scene, margin=1.0, above=0.3):
    """What the model covers seen from above: convex hull of everything above y = `above` (not the foundations),
    grown by `margin` metres. [[x, z], ...] in model space."""
    P = np.vstack([p["pos"] for p in scene["prims"]])
    P = P[P[:, 1] > above] if (P[:, 1] > above).sum() >= 3 else P
    h = np.array(_hull(P[:, [0, 2]]), float)
    if margin and len(h) >= 3:
        c = h.mean(0)
        out = []
        n = len(h)
        for i in range(n):                                    # offset each vertex along its two edge normals
            a, b, d = h[i - 1], h[i], h[(i + 1) % n]
            e1, e2 = b - a, d - b
            n1 = np.array([e1[1], -e1[0]]) / (np.linalg.norm(e1) or 1)
            n2 = np.array([e2[1], -e2[0]]) / (np.linalg.norm(e2) or 1)
            if np.dot(n1, b - c) < 0:
                n1, n2 = -n1, -n2
            m = n1 + n2
            m = m / (np.linalg.norm(m) or 1)
            k = margin / max(0.35, float(np.dot(m, n1)))
            out.append(b + m * k)
        h = np.array(out)
    return [[round(float(x), 2), round(float(z), 2)] for x, z in h]


def _components(pos, idx):
    """Connected pieces of a mesh (vertices welded by position first): label per vertex."""
    key = np.round(pos / 1e-3).astype(np.int64)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    lab = np.arange(weld.max() + 1)
    F = weld[idx]
    for _ in range(200):
        m = np.minimum(np.minimum(lab[F[:, 0]], lab[F[:, 1]]), lab[F[:, 2]])
        new = lab.copy()
        for k in range(3):
            np.minimum.at(new, F[:, k], m)
        new = new[new]                                        # pointer jumping
        if np.array_equal(new, lab):
            break
        lab = new
    return lab[weld]


def _drop_small(m, min_size):
    """Remove the pieces whose bounding box is smaller than min_size (letters, small ornaments)."""
    lab = _components(m["pos"], m["idx"])
    ids, inv = np.unique(lab, return_inverse=True)
    lo = np.full((len(ids), 3), np.inf); hi = np.full((len(ids), 3), -np.inf)
    np.minimum.at(lo, inv, m["pos"]); np.maximum.at(hi, inv, m["pos"])
    big = np.linalg.norm(hi - lo, axis=1) >= min_size
    keep = big[inv[m["idx"][:, 0]]]
    return dict(m, idx=m["idx"][keep])


def simplify(scene, target):
    """Far-away version. First the small pieces go (their size grows until the count fits or they would be
    more than 4 % of the model); then vertex clustering, with a real vertex of each cell (the one nearest to
    the cell's mean) as its representative so that walls stay on their planes. One primitive per material;
    write_prims gives it flat normals."""
    groups = {}
    for p in scene["prims"]:
        groups.setdefault(p["material"], []).append(p)
    merged = []
    for mat, ps in groups.items():
        o, pos, col, uv, idx = 0, [], [], [], []
        has_c = all(p["col"] is not None for p in ps)
        has_uv = all(p["uv"] is not None for p in ps)
        for p in ps:
            pos.append(p["pos"]); idx.append(p["idx"] + o); o += len(p["pos"])
            if has_c:
                col.append(p["col"])
            if has_uv:
                uv.append(p["uv"])
        merged.append({"material": mat, "pos": np.vstack(pos), "idx": np.vstack(idx),
                       "col": np.vstack(col) if has_c else None, "uv": np.vstack(uv) if has_uv else None})
    lo, hi = bounds(scene)
    size = float(np.max(hi - lo)) or 1.0
    if triangles(scene) <= target:
        return [_flat(m) for m in merged]
    for frac in (0.006, 0.01, 0.015, 0.02, 0.03, 0.04):         # 1) small pieces out
        cand = [_drop_small(m, size * frac) for m in merged]
        n = sum(len(m["idx"]) for m in cand)
        if n <= target * 2 or frac == 0.04:
            merged = cand
            break
    if n <= target:
        return [_flat(m) for m in merged if len(m["idx"])]
    cell = size / 300                                            # 2) clustering
    out = merged
    for _ in range(40):
        out, n = [], 0
        for m in merged:
            if not len(m["idx"]):                             # a material whose pieces were all dropped
                continue
            used_v = np.unique(m["idx"])
            key = np.floor((m["pos"][used_v] - lo) / cell).astype(np.int64)
            _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
            inv = inv.reshape(-1)
            k = len(cnt)
            mean = np.zeros((k, 3)); np.add.at(mean, inv, m["pos"][used_v]); mean /= cnt[:, None]
            d = np.linalg.norm(m["pos"][used_v] - mean[inv], axis=1)
            order = np.lexsort((d, inv))                       # per cell, the vertex nearest to its mean first
            first = order[np.r_[True, inv[order][1:] != inv[order][:-1]]]
            rep = used_v[first]                                 # original vertex index of each cell's representative
            vmap = -np.ones(len(m["pos"]), np.int64); vmap[used_v] = inv
            F = vmap[m["idx"]]
            F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
            if len(F):                                        # the same triangle twice: once
                ui = np.unique(np.sort(F, axis=1), axis=0, return_index=True)[1]
                F = F[np.sort(ui)]
            col = None
            if m["col"] is not None:                           # the cell's mean colour: no speckles from stripes
                col = np.zeros((k, m["col"].shape[1])); np.add.at(col, inv, m["col"][used_v]); col /= cnt[:, None]
            out.append({"material": m["material"], "pos": m["pos"][rep], "idx": F,
                        "col": col,
                        "uv": m["uv"][rep] if m["uv"] is not None else None})
            n += len(F)
        if n <= target:
            break
        cell *= 1.2
    return [_flat(m) for m in out if len(m["idx"])]


def _flat(m):
    """One vertex per triangle corner: flat shading (a far-away model looks crisp, not blobby)."""
    F = m["idx"]
    if not len(F):
        return m
    return {"material": m["material"], "pos": m["pos"][F].reshape(-1, 3), "idx": np.arange(F.size).reshape(-1, 3),
            "col": m["col"][F].reshape(-1, m["col"].shape[1]) if m["col"] is not None else None,
            "uv": m["uv"][F].reshape(-1, 2) if m["uv"] is not None else None}


def _normals(pos, idx):
    n = np.zeros_like(pos)
    a, b, c = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, idx[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return n / ln


def write_prims(path, prims, src_gltf, src_bin, generator="Offlined Landmarks 3D glbtool"):
    """Write primitives (model space, one node) keeping the source materials and their embedded images."""
    binbuf = bytearray()
    views, accs, out_prims = [], [], []

    def view(data, target=None):
        while len(binbuf) % 4:
            binbuf.append(0)
        off = len(binbuf); binbuf.extend(data)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    for p in prims:
        if not len(p["idx"]):
            continue
        pos = p["pos"].astype(np.float32)
        nor = _normals(p["pos"], p["idx"]).astype(np.float32)
        a0 = len(accs)
        accs.append({"bufferView": view(pos.tobytes(), 34962), "componentType": 5126, "count": len(pos), "type": "VEC3",
                     "min": pos.min(0).tolist(), "max": pos.max(0).tolist()})
        accs.append({"bufferView": view(nor.tobytes(), 34962), "componentType": 5126, "count": len(pos), "type": "VEC3"})
        attrs = {"POSITION": a0, "NORMAL": a0 + 1}
        if p["col"] is not None:
            col = np.clip(np.round(p["col"] * 255), 0, 255).astype(np.uint8)
            attrs["COLOR_0"] = len(accs)
            accs.append({"bufferView": view(col.tobytes(), 34962), "componentType": 5121, "normalized": True, "count": len(pos), "type": "VEC4"})
        if p["uv"] is not None:
            attrs["TEXCOORD_0"] = len(accs)
            accs.append({"bufferView": view(p["uv"].astype(np.float32).tobytes(), 34962), "componentType": 5126, "count": len(pos), "type": "VEC2"})
        idx = p["idx"].astype(np.uint32).ravel()
        accs.append({"bufferView": view(idx.tobytes(), 34963), "componentType": 5125, "count": len(idx), "type": "SCALAR"})
        pr = {"attributes": attrs, "indices": len(accs) - 1}
        if p["material"] is not None:
            pr["material"] = p["material"]
        out_prims.append(pr)
    g = {"asset": {"version": "2.0", "generator": generator}, "scene": 0, "scenes": [{"nodes": [0]}],
         "nodes": [{"name": "lod", "mesh": 0}], "meshes": [{"name": "lod", "primitives": out_prims}],
         "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": 0}]}
    # materials, textures, samplers and images (copied; images re-embedded)
    for k in ("materials", "textures", "samplers"):
        if src_gltf.get(k):
            g[k] = json.loads(json.dumps(src_gltf[k]))
    if src_gltf.get("images"):
        g["images"] = []
        for im in src_gltf["images"]:
            v = src_gltf["bufferViews"][im["bufferView"]]
            data = src_bin[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
            g["images"].append({"bufferView": view(data), "mimeType": im.get("mimeType", "image/png")})
    used_ext = [e for e in src_gltf.get("extensionsUsed", []) if e != "KHR_mesh_quantization"]
    if used_ext:
        g["extensionsUsed"] = used_ext
    while len(binbuf) % 4:
        binbuf.append(0)
    g["buffers"][0]["byteLength"] = len(binbuf)
    js = json.dumps(g, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(binbuf)
    with open(path, "wb") as fh:
        fh.write(b"glTF" + struct.pack("<II", 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(binbuf), 0x004E4942) + bytes(binbuf))
    return int(sum(len(p["idx"]) for p in prims)), total


def write_lod(scene, path, target):
    prims = simplify(scene, target)
    return write_prims(path, prims, scene["gltf"], scene["bin"])


def radius(scene):
    P = np.vstack([p["pos"] for p in scene["prims"]])
    return float(math.sqrt(float(np.max(P[:, 0] ** 2 + P[:, 2] ** 2))))
