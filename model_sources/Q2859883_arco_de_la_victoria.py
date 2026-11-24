"""
Arco de la Victoria (Q2859883), Moncloa, Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: Modesto López Otero (1885–1962) and Pascual Bravo Sanfeliú (1893–1984), 1950–1956, at the entrance
of the Ciudad Universitaria; attic reliefs by Moisés de Huerta (died 1962), keystone Victories by José Ortells
(died 1961), sculpture also by Julián Lozano (died 1992); the quadriga of Minerva on top (Ramón Arregui).
Rights: protected (free in 2065).

Measurements (see meta.json for the sources):
- OpenStreetMap way 193326001 and its parts 335507047 / 335507337 (the two piers, 6.9 × 8.8 m each) and 335507048
  (the span, 9.8 m): 23.6 × 8.8 m in all, long axis at 34.5°, so the south-east front (the inscription
  «MVNIFICENTIA REGIA CONDITA…», towards Moncloa and Calle de la Princesa) faces 124.5° = +Z. The size of the top
  checked on the PNOA orthophoto (IGN, CC BY 4.0): about 22–24 × 8–9 m.
- The elevation measured on near-frontal photos (Wikimedia Commons «Arco de la Victoria - Madrid 01.jpg», «Arco de
  la Victoria (Madrid) 05.jpg», «Madrid - Arco de la Victoria y Faro de Moncloa.jpg»), scaled by the OSM width
  (the piers and the span measured on the photo match the OSM parts within 2 %): plinth 1.3 m, the round arch
  7.8 m wide with its crown at 16.2 m, the archivolt 10 m wide, impost line 19.7 m, frieze and cornice to 23.1 m,
  the attic with the inscription between two relief panels to 27.8 m, its cornice and the top step to 29.1 m,
  the quadriga of Minerva to about 35 m.
  The published heights (40–42 m for the arch, 49 m with the quadriga) do not fit the OSM footprint and the
  orthophoto with the proportions on the photos; the measured proportions were kept (see meta.json, notes).

Axes (docs/MODEL_SPEC.md): +Y up, the south-east front faces +Z, +X points north-east, origin on the ground at the
centroid of the OSM outline (40.4359063, −3.7201974).

    python model_sources/Q2859883_arco_de_la_victoria.py   -> models/Q2859883/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import METAL, Mesh, box, cyl, lathe, merge, write_glb  # noqa: E402

rng = np.random.default_rng(1956)

GRAN = np.array((0.70, 0.685, 0.65))     # the grey granite cladding
GRAN_D = np.array((0.62, 0.61, 0.58))    # recessed panels, plinth
RELIEF = (0.76, 0.75, 0.71)
DARKV = (0.42, 0.41, 0.39)               # the coffered vault in shadow
BRONZE = (0.33, 0.55, 0.48)              # green patina of the quadriga
STAT = (0.80, 0.79, 0.75)                # the stone Victories
FOUND = -3.0

W2 = 11.8          # half width
D2 = 4.4           # half depth
SPAN2 = 4.9        # half width of the span between the piers (OSM)
OPEN2 = 3.9        # half width of the opening (intrados)
SPRING = 12.3      # springing of the arch
ARCH_O = 5.0       # outer radius of the archivolt
IMPOST = 19.7
CORN0, CORN1 = 21.9, 23.1
ATTIC = 27.8
TOP = 29.1


def jit(c, k=0.012):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, k)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def arch_body():
    """The block with the round arch cut through it: front/back faces with the arched hole, the vault, the piers."""
    ms = []
    seg = 24
    # front and back faces (z = ±D2): the outline of the block minus the arch, as a fan of quads
    for s in (1, -1):
        V, F = [], []
        ang = np.linspace(0, math.pi, seg + 1)
        arc = [(OPEN2 * math.cos(a), SPRING + OPEN2 * math.sin(a)) for a in ang]       # right to left over the top
        # the face above and around the arch, from y = FOUND to IMPOST (above that the frieze block takes over)
        # right pier face
        V += [(OPEN2, FOUND), (W2, FOUND), (W2, IMPOST), (OPEN2, SPRING)]
        F += [(0, 1, 2), (0, 2, 3)]
        # spandrel: each arc segment joined to the top edge
        o = len(V)
        for i, (x, y) in enumerate(arc):
            V.append((x, y))
            V.append((W2 - 2 * W2 * i / seg, IMPOST))
        for i in range(seg):
            a, b, c, d = o + 2 * i, o + 2 * i + 1, o + 2 * i + 3, o + 2 * i + 2
            F += [(a, b, c), (a, c, d)]
        o = len(V)
        V += [(-W2, FOUND), (-OPEN2, FOUND), (-OPEN2, SPRING), (-W2, IMPOST)]
        F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        V3 = [(x, y, s * D2) for x, y in V]
        m = Mesh(V3, F, jit(GRAN))
        if s > 0:
            m.f = m.f[:, ::-1]
        ms.append(m)
    # sides (x = ±W2) and the opening's inner faces
    ms.append(B(-W2, -W2 + 0.01, FOUND, IMPOST, -D2, D2, jit(GRAN)))
    ms.append(B(W2 - 0.01, W2, FOUND, IMPOST, -D2, D2, jit(GRAN)))
    for sx in (1, -1):
        V = [(sx * OPEN2, FOUND, -D2), (sx * OPEN2, FOUND, D2), (sx * OPEN2, SPRING, D2), (sx * OPEN2, SPRING, -D2)]
        ms.append(Mesh(V, [(0, 1, 2), (0, 2, 3)], jit(GRAN_D)))
    # the coffered barrel vault: dark soffit with coffer ribs
    ang = np.linspace(0, math.pi, seg + 1)
    V, F = [], []
    for a in ang:
        x, y = OPEN2 * math.cos(a), SPRING + OPEN2 * math.sin(a)
        V += [(x, y, -D2), (x, y, D2)]
    for i in range(seg):
        a, b = 2 * i, 2 * i + 2
        F += [(a, a + 1, b + 1), (a, b + 1, b)]
    ms.append(Mesh(V, F, DARKV))
    for k in range(-3, 4):
        z = k * D2 / 3.6
        prof = [(OPEN2 * math.cos(a), SPRING + OPEN2 * math.sin(a), z) for a in ang]
        ms.append(tube_ring(prof, 0.12, jit(GRAN_D)))
    return ms


def tube_ring(pts, r, c):
    from meshkit import tube
    return tube(pts, r, c, seg=4)


def front_details(s):
    """Pilasters, archivolt, medallions, the Victory keystone, frieze, cornice and attic of one front (s = ±1)."""
    st, rel = [], []
    zf = s * D2

    def slab(x0, x1, y0, y1, d, c):
        z0, z1 = (zf, zf + d) if s > 0 else (zf - d, zf)
        return B(x0, x1, y0, y1, z0, z1, c)

    # plinth all round
    st.append(slab(-W2 - 0.35, -OPEN2, FOUND, 1.3, 0.35, jit(GRAN_D)))
    st.append(slab(OPEN2, W2 + 0.35, FOUND, 1.3, 0.35, jit(GRAN_D)))
    # corner and inner pilasters of each pier
    for sx in (1, -1):
        for (a, b) in ((W2 - 2.8, W2), (SPAN2 + 0.0, SPAN2 + 1.0)):
            st.append(slab(sx * a if sx > 0 else -b, sx * b if sx > 0 else -a, 1.3, IMPOST, 0.3, jit(GRAN)))
        # a recessed panel between them
        st.append(slab(sx * (SPAN2 + 1.0) if sx > 0 else -(W2 - 2.8), sx * (W2 - 2.8) if sx > 0 else -(SPAN2 + 1.0),
                       1.3, IMPOST, 0.05, jit(GRAN_D)))
        # medallion with the year (ANNO …)
        med = cyl(0.95, 0.95, 0.42, jit(RELIEF), seg=16).R(math.pi / 2, 0, 0)
        st.append(med.T(sx * 4.5, 17.9, zf if s > 0 else zf - 0.42))
    # archivolt: a raised ring round the arch
    ang = np.linspace(0, math.pi, 25)
    V, F = [], []
    for a in ang:
        V += [(OPEN2 * math.cos(a), SPRING + OPEN2 * math.sin(a)), (ARCH_O * math.cos(a), SPRING + ARCH_O * math.sin(a) * 1.0)]
    d = 0.3
    V3 = [(x, y, zf + s * d) for x, y in V]
    for i in range(24):
        a = 2 * i
        F += [(a, a + 2, a + 3), (a, a + 3, a + 1)] if s > 0 else [(a, a + 3, a + 2), (a, a + 1, a + 3)]
    st.append(Mesh(V3, F, jit(GRAN)))
    # the archivolt's edges
    for r in (OPEN2, ARCH_O):
        V, F = [], []
        for a in ang:
            x, y = r * math.cos(a), SPRING + r * math.sin(a)
            V += [(x, y, zf), (x, y, zf + s * d)]
        for i in range(24):
            a = 2 * i
            F += [(a, a + 2, a + 3), (a, a + 3, a + 1)]
        st.append(Mesh(V, F, jit(GRAN)))
    # keystone Victory (winged figure) over the arch
    vic = lathe([(0, 0), (0.45, 0.0), (0.5, 1.2), (0.35, 2.0), (0.22, 2.3), (0.25, 2.6), (0, 2.8)], STAT, 8)
    st.append(vic.S(1, 1, 0.6).T(0, 17.6, zf + s * 0.55))
    for sx in (1, -1):
        st.append(B(-0.15, 0.15, 0, 1.6, -0.12, 0.12, STAT).R(0, 0, sx * 0.55).T(sx * 0.45, 19.0, zf + s * 0.5))
    # impost band, frieze, cornice
    st.append(slab(-W2 - 0.1, W2 + 0.1, IMPOST - 0.35, IMPOST, 0.4, jit(GRAN)))
    st.append(slab(-W2 - 0.6, W2 + 0.6, CORN0, CORN1, 0.75, jit(GRAN)))
    # attic: the inscription panel in the middle, relief panels (rows of figures) left and right
    st.append(slab(-6.6, 6.6, 24.0, 27.3, 0.08, jit(RELIEF)))
    for k in range(4):
        st.append(slab(-6.0, 6.0, 24.6 + k * 0.7, 24.75 + k * 0.7, 0.12, jit(GRAN_D)))     # lines of letters
    for sx in (1, -1):
        x0, x1 = sx * 7.0, sx * 11.3
        a, b = min(x0, x1), max(x0, x1)
        st.append(slab(a, b, 24.0, 27.3, 0.03, jit(GRAN_D)))
        for k in range(5):
            x = a + (b - a) * (k + 0.5) / 5
            fig = lathe([(0, 0), (0.32, 0.0), (0.3, 1.6), (0.22, 2.3), (0.2, 2.6), (0, 2.85)], jit(RELIEF), 8)
            rel.append(fig.S(1, 1, 0.45).T(x, 24.2, zf + s * 0.1))
    st.append(slab(-W2 - 0.4, W2 + 0.4, ATTIC, ATTIC + 0.6, 0.45, jit(GRAN)))
    return st, rel


def upper_block():
    ms = []
    ms.append(B(-W2, W2, IMPOST, CORN0, -D2, D2, jit(GRAN)))             # frieze
    ms.append(B(-W2 - 0.6, W2 + 0.6, CORN0, CORN1, -D2 - 0.6, D2 + 0.6, jit(GRAN)))
    ms.append(B(-W2 + 0.2, W2 - 0.2, CORN1, ATTIC, -D2 + 0.2, D2 - 0.2, jit(GRAN)))
    ms.append(B(-W2 - 0.2, W2 + 0.2, ATTIC, ATTIC + 0.6, -D2 - 0.2, D2 + 0.2, jit(GRAN)))
    ms.append(B(-W2 + 0.3, W2 - 0.3, ATTIC + 0.6, TOP, -D2 + 0.3, D2 - 0.3, jit(GRAN)))
    ms.append(B(-4.2, 4.2, TOP, TOP + 0.5, -2.4, 2.4, jit(GRAN_D)))    # pedestal of the quadriga
    return ms


def horse(c):
    """A stylised rearing horse facing +Z, about 2.6 m long and 2.4 m high at the head."""
    body = cyl(0.42, 0.42, 1.9, c, seg=8).R(math.pi / 2, 0, 0).T(0, 1.45, -0.95)
    neck = cyl(0.3, 0.22, 1.0, c, seg=6).R(-0.6, 0, 0).T(0, 1.55, 0.85)
    head = box(0.3, 0.32, 0.7, c).R(0.5, 0, 0).T(0, 2.25, 1.25)
    legs = []
    for lx in (-0.22, 0.22):
        legs.append(box(0.14, 1.2, 0.14, c).T(lx, 0.0, -0.75))              # hind legs
        legs.append(box(0.14, 0.9, 0.14, c).R(-1.0, 0, 0).T(lx, 1.15, 0.55))  # forelegs raised
    return merge([body, neck, head] + legs)


def quadriga():
    c = BRONZE
    ms = []
    y0 = TOP + 0.5
    for k, x in enumerate((-1.35, -0.45, 0.45, 1.35)):
        ms.append(horse(jit(c)).T(x, y0, 0.8 + (0.15 if k in (1, 2) else 0.0)))
    # the chariot (a round-fronted box with two wheels) and Minerva standing in it, spear raised
    ms.append(cyl(0.9, 0.9, 1.1, jit(c), seg=12).S(1.1, 1, 0.8).T(0, y0, -1.6))
    for sx in (1, -1):
        ms.append(cyl(0.75, 0.75, 0.12, jit(c), seg=12).R(0, 0, math.pi / 2).T(sx * 1.05, y0 + 0.75, -1.8))
    fig = lathe([(0, 0), (0.45, 0), (0.5, 0.9), (0.35, 1.9), (0.42, 2.5), (0.2, 2.9), (0.0, 2.95)], jit(c), 10).T(0, y0 + 1.0, -1.7)
    head = lathe([(0, 0), (0.18, 0.05), (0.2, 0.25), (0.12, 0.45), (0, 0.55)], jit(c), 8).T(0, y0 + 3.9, -1.7)
    crest = box(0.06, 0.35, 0.5, jit(c)).T(0, y0 + 4.35, -1.75)
    spear = cyl(0.05, 0.05, 3.6, jit(c), seg=4).T(0.55, y0 + 1.5, -1.6)
    shield = cyl(0.55, 0.55, 0.08, jit(c), seg=12).R(0, 0, math.pi / 2).T(-0.55, y0 + 2.2, -1.6)
    ms += [fig, head, crest, spear, shield]
    return ms


def main():
    out = ROOT / "models" / "Q2859883"
    out.mkdir(parents=True, exist_ok=True)
    stone, rel = [], []
    stone += arch_body()
    stone += upper_block()
    for s in (1, -1):
        a, b = front_details(s)
        stone += a; rel += b
    # short sides: corner pilasters and a recessed panel
    for sx in (1, -1):
        x = sx * W2
        x0, x1 = (x, x + 0.3) if sx > 0 else (x - 0.3, x)
        for z0, z1 in ((-D2, -D2 + 2.2), (D2 - 2.2, D2)):
            stone.append(B(x0, x1, 1.3, IMPOST, z0, z1, jit(GRAN)))
        stone.append(B(x0 - 0.05 * sx, x1, FOUND, 1.3, -D2 - 0.35, D2 + 0.35, jit(GRAN_D)))
        stone.append(B(x0, x1, IMPOST - 0.35, IMPOST, -D2 - 0.1, D2 + 0.1, jit(GRAN)))
    m_stone, m_rel, m_br = merge(stone), merge(rel), merge(quadriga())
    tris, size = write_glb(out / "model.glb", [("arco_de_la_victoria", m_stone), ("reliefs", m_rel),
                                               ("quadriga", m_br, {"material": "bronze"})],
                           roughness=0.85, materials={"bronze": METAL})
    allv = np.vstack([m_stone.v, m_rel.v, m_br.v])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
