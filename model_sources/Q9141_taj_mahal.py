"""
Taj Mahal, Agra (Q9141) — the mausoleum, its mosque and jawab, the charbagh garden and the great gate, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Ustad Ahmad Lahori (died 1649), for Shah Jahan; the mausoleum 1632–1643, the whole complex
finished c. 1653. Public domain.

What is modelled:
- The mausoleum: the white marble plinth (95.5 m square, 6 m high) with its blind arcade, the octagonal bastions
  at its corners and the stair block on the south side; the tomb (56.9 m square with chamfered corners) with the
  four great pishtaqs and their deep iwans (the black calligraphy band round each arch), two storeys of arched
  niches on the side bays and on the chamfered corners, the pinnacles (guldastas) on every corner and beside the
  pishtaqs; the drum, the onion dome with its lotus crown and the gilded finial with the crescent; the four
  chhatris round the dome; the four minarets with their three balconies and their top chhatris.
- The red sandstone riverfront terrace (300 m long, on its 8.7 m wall over the Yamuna) with the octagonal towers
  at its corners; the mosque (west) and its twin, the jawab (east): pishtaq with its iwan, four arches with white
  marble outlines, three white marble domes with gilded finials, corner towers with chhatris; the tank in front of
  each.
- The charbagh: the raised marble water channels on the two axes with the fountain jets of the north–south
  channel, the sunken lawns with the rows of cypresses beside the channels, the central raised tank (al-Hawd
  al-Kawthar) with its five jets, the sixteen sunken flower beds with their trees, the raised walkways (red
  sandstone), the bridges where the walks cross the east–west channel, the two pavilions at the ends of the cross
  axis (called Jal Mahal in OpenStreetMap), the crenellated red sandstone walls with the octagonal towers at the
  southern corners, and the great gate (Darwaza-i rauza) with its pishtaqs, corner towers and the rows of eleven
  chhatris over both fronts.
- Not modelled: the forecourt (jilaukhana) south of the gate with its courtyards and gates, the bazaars, the
  Mehtab Bagh across the river, the terrain, the interiors.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap way 375257537 (outline of the tomb on its plinth) and its building:parts (relation 11089173:
  body 56.4 m across, drum, dome, chhatris, minarets at about ±48 m from the centre); ways 376367510 (mosque,
  23.2 × 55.6 m) and 376367511 (jawab), 435831216 (retaining wall of the terrace, 310 × 121 m), 435831218–221,
  59267923, 59267925 (octagonal towers), 59156688 (great gate, 41 m wide), 346806298 / 435831217 (pavilions of the
  cross axis), 376367512 / 435831215 (tanks), 242854370–242854429, 1128521747, 59220397 (channels, central tank,
  lawns, beds), 27044943 / 59267920 / 59267927 (walls). The garden ways (channels, tank, beds, gate) are drawn in
  OSM about 9 m east of the axis of the tomb, mosque, jawab and terrace (two mappers, two imagery offsets); the
  layout was put back on the axis of the tomb and made symmetric, as built, with the 296.3 m square charbagh
  of Koch (from the terrace edge to the south wall, which then falls on the two southern corner towers of OSM).
- Heights (Wikipedia en, «Taj Mahal» and «Origins and architecture of the Taj Mahal», after Ebba Koch):
  plinth 6 m; pishtaqs 33 m (taken from the terrace, i.e. 27 m above the plinth, as the OSM parts give); drum about 7 m visible (12 m in all); dome 23 m; finial
  9.6 m; total 73 m above the terrace; minarets 43 m (more than 40 m, three equal parts); mosque 56.6 × 23.38 m,
  20.3 m high (taken to the top of the pishtaq); great gate 41.2 × 34 × 23.07 m; terrace 300 × 111.9 m, 8.7 m high
  over the river; charbagh 296.31 m square. Estimated (±15–30 %): the terrace 1.2 m above the garden; the tomb's
  side parapets 24 m above the plinth; the bulb 27.4 m across on a drum 23 m across; the chhatris; the mosque domes (taken from the OSM
  outlines, 18 and 14 m across) and their heights; the garden walls 7 m; the towers; the pavilions; the gate's
  corner towers and chhatris; the channels (5.6 and 6 m of water as in OSM), kerbs and lawns.
- Ground: SRTM / ASTER 30 m (via OpenTopoData) give the garden at ≈ 162 m and the river bank ≈ 148–153 m.
  y = 0 is the level of the garden walks; the terrace is at +1.2, the river face of the terrace goes down to −10.
White Makrana marble (tomb, domes, minarets, channels, tank), red Fatehpur sandstone (terrace, mosque, jawab,
walls, gate, walks), black marble inlay (calligraphy), gilded finials.

Axes (docs/MODEL_SPEC.md): +Y up, the south front of the tomb (towards the great gate) faces +Z (heading 179.9°,
from the OSM outline), +X points east (89.9°). Origin on the terrace at the centre of the tomb.

    python model_sources/Q9141_taj_mahal.py   -> models/Q9141/model.glb (+ footprint in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, cap, walls, triangulate, fbm, write_glb, METAL  # noqa: E402

rng = np.random.default_rng(1632)
HEADING = 179.9

# ── Colours ─────────────────────────────────────────────────────────────
MARBLE = (0.93, 0.92, 0.885)           # Makrana marble
MARBLE_W = (0.965, 0.955, 0.93)        # domes, the brightest
MARBLE_SH = (0.74, 0.73, 0.70)         # inside the iwans and niches
MARBLE_D = (0.50, 0.49, 0.47)          # doors and jalis seen in shadow
INLAY = (0.16, 0.15, 0.15)             # black marble calligraphy
SAND = (0.64, 0.33, 0.25)              # red sandstone
SAND_L = (0.72, 0.44, 0.35)            # terrace paving
SAND_SH = (0.36, 0.17, 0.13)           # arches in red sandstone
PAVE = (0.74, 0.56, 0.47)              # garden walks
GRASS = (0.40, 0.55, 0.26)
GRASS_D = (0.33, 0.48, 0.22)
CYPRESS = (0.15, 0.29, 0.14)
TREES = [(0.22, 0.38, 0.17), (0.27, 0.42, 0.19), (0.19, 0.34, 0.16), (0.30, 0.44, 0.21)]
WATER = (0.29, 0.43, 0.50)
JET = (0.90, 0.93, 0.95)
GOLD = (0.83, 0.66, 0.30)

T = 1.2            # terrace above the garden walks (estimate)
P = T + 6.0        # top of the marble plinth
H = 28.45          # half side of the tomb (56.9 m)
C = 8.5            # chamfer of the tomb's corners
ZC = 208.6         # centre of the charbagh (60.5 … 356.8)
ZN, ZS = 60.5, 356.8                   # terrace edge, inner face of the south wall
XW = 153.7         # centre line of the east and west walls
ZW = 358.0         # centre line of the south wall
WALL_H = 7.0
GATE_Z = 363.0



# ── Small helpers ───────────────────────────────────────────────────────
def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def place(ms, ang=0.0, x=0.0, z=0.0, y=0.0):
    """Turn local pieces (built facing +z) by ang degrees about y (+z -> +x at 90) and move them."""
    out = []
    for m in ms:
        m = m.copy().R(0, math.radians(ang), 0).T(x, y, z)
        out.append(m)
    return out


def mirror_x(ms):
    out = []
    for m in ms:
        m = m.copy(); m.v[:, 0] *= -1; m.f = m.f[:, ::-1]; out.append(m)
    return out


def poly_uy(pts, c, w=0.0):
    """A flat polygon [(u, y)] in the plane z = w, facing +z."""
    p2, tr = triangulate(pts)
    v = np.stack([p2[:, 0], p2[:, 1], np.full(len(p2), float(w))], 1)
    return Mesh(v, tr, c)


def prism_uy(outer, holes, w0, w1, c, back=True):
    """A slab drawn in (u, y) with holes (arches), from z = w0 to w1, front facing +z."""
    p2, tr = triangulate(outer, holes)
    ms = [Mesh(np.stack([p2[:, 0], p2[:, 1], np.full(len(p2), w1)], 1), tr, c)]
    if back:
        ms.append(Mesh(np.stack([p2[:, 0], p2[:, 1], np.full(len(p2), w0)], 1), tr[:, ::-1], c))
    for ring, hole in [(outer, False)] + [(h, True) for h in holes]:
        R = list(ring)
        a = sum(R[i][0] * R[(i + 1) % len(R)][1] - R[(i + 1) % len(R)][0] * R[i][1] for i in range(len(R)))
        if (a > 0) == hole:
            R = R[::-1]
        V, F = [], []
        for i in range(len(R)):
            (u0, y0), (u1, y1) = R[i], R[(i + 1) % len(R)]
            o = len(V)
            V += [(u0, y0, w0), (u1, y1, w0), (u1, y1, w1), (u0, y0, w1)]
            F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        ms.append(Mesh(V, F, c))
    return merge(ms)


def quad(p0, p1, p2, p3, c):
    return Mesh([p0, p1, p2, p3], [(0, 1, 2), (0, 2, 3)], c)


def arch(w, y0, hs, u=0.0, k=0.62, n=6):
    """A pointed (four-centred) Mughal arch, w wide, springing at hs, from the sill y0: [(u, y)] counter-clockwise."""
    R = k * w
    cx = w / 2 - R
    ta = math.acos(-cx / R)
    right = [(u + cx + R * math.cos(t), hs + R * math.sin(t)) for t in np.linspace(0, ta, n + 1)]
    left = [(-x + 2 * u, y) for x, y in right[::-1]][1:]
    return [(u - w / 2, y0), (u + w / 2, y0)] + right + left


def arch_frame(hw, y0, ytop, w, hs):
    """A wall hw on each side of the axis, from y0 to ytop, with an arch w wide cut out of its foot (one simple
    polygon, so the arch can reach the sill)."""
    arc = arch(w, y0, hs)[2:][::-1]
    return [(-hw, y0), (-w / 2, y0)] + arc + [(w / 2, y0), (hw, y0), (hw, ytop), (-hw, ytop)]


def arch_band(wo, wi, y0, hs, c, w):
    """A flat band between two arches (a marble border round an opening), in the plane z = w."""
    inner = arch(wi, y0, hs)[2:][::-1]
    outer = arch(wo, y0, hs)[2:]
    return poly_uy([(-wo / 2, y0), (-wi / 2, y0)] + inner + [(wi / 2, y0), (wo / 2, y0)] + outer, c, w)


def vault(a, y0, y1, d, c, c_under):
    """The filling over an iwan (local, wall at z = 0): its top in the roof colour, its underside in shadow."""
    y = y0 - 0.03
    return [B(-a, a, y0, y1, -d, 0, c), quad((-a, y, -d), (a, y, -d), (a, y, 0), (-a, y, 0), c_under)]


def apex(w, hs, k=0.62):
    R = k * w
    cx = w / 2 - R
    return hs + math.sqrt(R * R - cx * cx)


def sphere(r, c, seg=8, rings=5):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def niche(u, w, y0, hs, c_out, c_in, inner=0.5, wpos=0.04, frame=None, fw=0.35):
    """A blind arch painted on a wall (z = 0, facing +z): optional frame, the niche, a darker window inside."""
    ms = []
    if frame is not None:
        ms.append(poly_uy(arch(w + 2 * fw, y0, hs, u), frame, wpos))
    ms.append(poly_uy(arch(w, y0, hs, u), c_out, wpos + 0.03))
    if inner:
        wi = w * inner
        ms.append(poly_uy(arch(wi, y0 + (hs - y0) * 0.12, hs - (hs - y0) * 0.05, u), c_in, wpos + 0.06))
    return ms


def finial(x, y, z, s, crescent=False):
    """Gilded finial of stacked bulbs, s = height."""
    prof = [(0.0, 0), (0.09 * s, 0.0), (0.16 * s, 0.08 * s), (0.10 * s, 0.17 * s), (0.17 * s, 0.27 * s),
            (0.07 * s, 0.38 * s), (0.13 * s, 0.47 * s), (0.05 * s, 0.56 * s), (0.035 * s, 0.85 * s), (0.0, 0.88 * s)]
    ms = [lathe(prof, GOLD, 10).T(x, y, z)]
    if crescent:
        r = 0.09 * s
        pts = [(x + r * math.sin(t), y + 0.86 * s + r * (1 - math.cos(t)), z) for t in np.linspace(-2.4, 2.4, 9)]
        ms.append(tube(pts, 0.012 * s, GOLD, 6))
        ms.append(cyl(0.012 * s, 0.0, 0.14 * s, GOLD, 6).T(x, y + 0.86 * s, z))
    return ms


def onion(r_max, y0, h, c, seg=24, base_r=None):
    """An onion (bulbous) dome from y0, h high, widest r_max at about a third of its height."""
    rb = base_r if base_r is not None else r_max * 0.88
    s = [0.0, 0.06, 0.16, 0.28, 0.40, 0.52, 0.63, 0.73, 0.82, 0.89, 0.95, 1.0]
    k = [rb / r_max, 0.95, 0.995, 1.0, 0.965, 0.86, 0.70, 0.52, 0.34, 0.20, 0.10, 0.03]
    return lathe([(r_max * kk, y0 + h * ss) for ss, kk in zip(s, k)], c, seg)


def chhatri(x, z, y0, R, col=MARBLE, dome=MARBLE_W, seg=8, fin=0.55):
    """An open domed kiosk on 8 columns: platform, columns, sloping eave, drum, onion dome, gilded finial."""
    ms = [cyl(1.25 * R, 1.25 * R, 0.45 * R, col, seg).R(0, math.pi / seg, 0).T(x, y0, z)]
    y = y0 + 0.45 * R
    for i in range(8):
        a = 2 * math.pi * i / 8 + math.pi / 8
        ms.append(cyl(0.075 * R, 0.075 * R, 1.25 * R, col, 5).T(x + R * math.cos(a), y, z + R * math.sin(a)))
    y += 1.25 * R
    ms.append(cyl(1.5 * R, 1.08 * R, 0.22 * R, col, seg).R(0, math.pi / seg, 0).T(x, y, z))
    y += 0.22 * R
    ms.append(cyl(1.0 * R, 1.0 * R, 0.3 * R, col, seg).R(0, math.pi / seg, 0).T(x, y, z))
    y += 0.3 * R
    ms.append(onion(1.06 * R, y, 1.3 * R, dome, 12))
    ms[-1].T(x, 0, z)
    y += 1.3 * R
    ms += finial(x, y - 0.04 * R, z, fin * R)
    return ms


def guldasta(x, z, y0, y1, r, c=MARBLE):
    """A slender pinnacle (corner shaft) with its small cupola."""
    return [cyl(r, r, y1 - y0, c, 8).T(x, y0, z), lathe([(r * 1.35, 0), (r * 1.5, 0.25 * r), (r * 1.1, 0.9 * r),
                                                         (r * 0.5, 1.6 * r), (0.0, 2.6 * r)], c, 8).T(x, y1, z)]


# ── The tomb ────────────────────────────────────────────────────────────
A_W, A_HS = 11.0, P + 14.5            # iwan arch: width, springing
A_AP = apex(A_W, A_HS)
D_IW = 5.0                             # depth of the iwan
PISH = 10.0                            # half width of the pishtaq
FLANK = P + 24.0                       # side parapets
PTOP = P + 27.0                        # pishtaqs


def tomb_ring():
    a = A_W / 2
    side = [(-H + C, H), (-a, H), (-a, H - D_IW), (a, H - D_IW), (a, H), (H - C, H)]
    ring = []
    for k in range(4):
        t = math.radians(90 * k)
        c, s = math.cos(t), math.sin(t)
        ring += [(x * c + z * s, -x * s + z * c) for x, z in side]
    return ring


def tomb():
    ms = []
    ring = tomb_ring()
    ms.append(walls(ring, P - 0.2, FLANK, MARBLE))
    ms.append(cap(ring, FLANK, MARBLE))
    # cornice band of the side parapets
    face = []
    a = A_W / 2
    # one front (south, local z = 0 is the wall): built once, turned four times
    face += vault(a, A_AP, FLANK, D_IW, MARBLE, MARBLE_SH)                                # vault of the iwan
    face.append(prism_uy(arch_frame(PISH, P, PTOP, A_W, A_HS), [], 0.0, 1.0, MARBLE))
    face.append(arch_band(A_W + 1.4, A_W, P, A_HS, INLAY, 1.04))
    face.append(B(-PISH - 0.25, PISH + 0.25, PTOP - 0.7, PTOP, -0.2, 1.25, MARBLE))      # crest of the pishtaq
    # calligraphy band (black inlay) round the arch, and a white border
    for u0, u1, y0, y1 in ((-8.9, -8.2, P + 0.6, P + 25.0), (8.2, 8.9, P + 0.6, P + 25.0), (-8.9, 8.9, P + 24.3, P + 25.0)):
        face.append(poly_uy([(u0, y0), (u1, y0), (u1, y1), (u0, y1)], INLAY, 1.04))
    face.append(poly_uy([(-9.6, P + 25.6), (9.6, P + 25.6), (9.6, P + 26.1), (-9.6, P + 26.1)], INLAY, 1.04))
    # inside the iwan: shaded sides and back, the door and the window above
    for sgn in (-1, 1):
        u = sgn * (a - 0.04)
        face.append(quad((u, P, -D_IW), (u, P, 0), (u, A_AP, 0), (u, A_AP, -D_IW), MARBLE_SH) if sgn < 0 else
                    quad((u, P, 0), (u, P, -D_IW), (u, A_AP, -D_IW), (u, A_AP, 0), MARBLE_SH))
    face.append(poly_uy([(-a, P), (a, P), (a, A_AP), (-a, A_AP)], MARBLE_SH, -D_IW + 0.04))
    face += niche(0, 4.6, P + 0.2, P + 6.5, MARBLE_D, INLAY, inner=0.0, wpos=-D_IW + 0.05)
    face += niche(0, 4.6, P + 8.0, P + 11.8, MARBLE_D, INLAY, inner=0.0, wpos=-D_IW + 0.05)
    # two storeys of niches on the side bays
    for u in (-15.0, 15.0):
        for y0, hs in ((P + 1.2, P + 8.0), (P + 12.6, P + 18.4)):
            face += niche(u, 6.6, y0, hs, MARBLE_SH, MARBLE_D, inner=0.55, frame=INLAY, fw=0.3)
        face.append(poly_uy([(u - 4.4, P + 23.0), (u + 4.4, P + 23.0), (u + 4.4, P + 23.4), (u - 4.4, P + 23.4)], INLAY, 0.05))
    # pinnacles beside the pishtaq and at the corners
    for u in (-PISH, PISH):
        face += guldasta(u, 0.6, P, PTOP + 2.6, 0.6)
    face += guldasta(H - C, 0.0, P, FLANK + 2.8, 0.5) + guldasta(-(H - C), 0.0, P, FLANK + 2.8, 0.5)
    for k in range(4):
        ms += place(face, 90 * k, *rot(0, H, 90 * k))
    # chamfered corners: two niches each
    cham = []
    for y0, hs in ((P + 1.2, P + 8.0), (P + 12.6, P + 18.4)):
        cham += niche(0, 6.6, y0, hs, MARBLE_SH, MARBLE_D, inner=0.55, frame=INLAY, fw=0.3)
    cham.append(poly_uy([(-4.4, P + 23.0), (4.4, P + 23.0), (4.4, P + 23.4), (-4.4, P + 23.4)], INLAY, 0.05))
    dc = (2 * H - C) / math.sqrt(2) / 1.0 / 2 * 1.0
    dc = (H - C / 2) * math.sqrt(2)
    for k in range(4):
        ang = 45 + 90 * k
        ms += place(cham, ang, *rot(0, dc, ang))
    # drum, dome, finial
    ms.append(lathe([(11.5, FLANK - 0.2), (11.5, FLANK + 9.8), (11.95, FLANK + 9.9), (11.95, FLANK + 10.4),
                     (11.6, FLANK + 10.45)], MARBLE, 40))
    ms.append(lathe([(11.6, FLANK + 4.5), (11.62, FLANK + 4.5), (11.62, FLANK + 5.0), (11.6, FLANK + 5.0)], INLAY, 40))
    y_d = FLANK + 10.4
    dome = onion(13.7, y_d, 21.0, MARBLE_W, 40, base_r=11.6)
    ms.append(dome)
    y_c = y_d + 21.0
    ms.append(lathe([(0.45, y_c - 0.3), (2.4, y_c + 0.25), (2.6, y_c + 0.55), (1.6, y_c + 1.2), (0.6, y_c + 1.9)],
                    MARBLE_W, 24))                                                        # lotus crown
    ms += finial(0, y_c + 1.8, 0, (T + 73.0) - (y_c + 1.8), crescent=True)
    # the four chhatris round the dome
    for sx in (-1, 1):
        for sz in (-1, 1):
            ms += chhatri(16.5 * sx, 16.5 * sz, FLANK - 0.1, 3.0, MARBLE, MARBLE_W)
    return ms


def rot(x, z, ang):
    t = math.radians(ang)
    return (x * math.cos(t) + z * math.sin(t), -x * math.sin(t) + z * math.cos(t))


def plinth():
    ms = []
    S = 47.75
    sq = [(-S, -S), (S, -S), (S, S), (-S, S)]
    ms.append(walls(sq, -2.0, P, MARBLE))
    ms.append(cap(sq, P, MARBLE))
    for sx in (-1, 1):
        for sz in (-1, 1):
            ms.append(cyl(5.6, 5.6, P + 2.25, MARBLE, 8).R(0, math.pi / 8, 0).T(48.0 * sx, -2.0, 48.0 * sz))
    side = []
    us = np.arange(-41.8, 41.81, 4.4)
    for u in us:
        side += niche(float(u), 2.7, T + 0.9, T + 3.7, MARBLE_SH, MARBLE_D, inner=0.0, frame=None)
    side.append(poly_uy([(-S + 6, P - 0.6), (S - 6, P - 0.6), (S - 6, P - 0.3), (-S + 6, P - 0.3)], MARBLE_SH, 0.04))
    for k in range(4):
        s2 = [m for m in side]
        if k == 0:                                           # the stair block hides the middle of the south side
            s2 = [m for m in side if not (np.abs(m.v[:, 0]).max() < 11.0 and m.v[:, 1].max() < P - 1)]
        ms += place(s2, 90 * k, *rot(0, S, 90 * k))
    # stair block (south) with its two doorways
    ms.append(B(-9.4, 9.4, T - 0.5, P, S - 0.5, 51.8, MARBLE))
    ms += [m.T(0, 0, 51.8) for m in niche(-4.5, 2.6, T + 0.1, T + 3.0, MARBLE_SH, MARBLE_D, inner=0.6) +
           niche(4.5, 2.6, T + 0.1, T + 3.0, MARBLE_SH, MARBLE_D, inner=0.6)]
    return ms


def minaret(x, z):
    ms = []
    y = P + 0.25
    ms.append(cyl(3.7, 3.7, 1.4, MARBLE, 8).R(0, math.pi / 8, 0).T(x, y, z))
    y += 1.4
    tops = [P + 12.0, P + 22.2, P + 32.4]
    r0, r1 = 2.85, 2.2
    rr = lambda yy: r0 + (r1 - r0) * (yy - y) / (tops[-1] - y)                      # noqa: E731
    yb = y
    for yt in tops:
        ms.append(lathe([(rr(yb), yb), (rr(yt - 1.5), yt - 1.5)], MARBLE, 16).T(x, 0, z))
        ms.append(lathe([(rr(yb) + 0.02, yb + 0.6), (rr(yb) + 0.02, yb + 0.9)], INLAY, 16).T(x, 0, z))
        ms.append(lathe([(rr(yt - 1.5), yt - 1.5), (3.55, yt - 0.15)], MARBLE, 16).T(x, 0, z))        # corbels
        ms.append(cyl(3.7, 3.7, 0.35, MARBLE, 16).T(x, yt - 0.15, z))
        ms.append(cyl(3.6, 3.6, 0.95, MARBLE, 16, caps=False).T(x, yt + 0.2, z))                     # balustrade
        yb = yt + 0.2
    ms += chhatri(x, z, yb, 2.2, MARBLE, MARBLE_W, fin=0.85)
    return ms


# ── Terrace, mosque and jawab ─────────────────────────────────────────
def terrace():
    ms = []
    ring = [(-155.0, -60.0), (155.0, -60.0), (155.0, ZN), (-155.0, ZN)]
    ms.append(walls(ring, -10.0, T, SAND))
    ms.append(cap(ring, T, SAND_L))
    ms.append(B(-155.0, 155.0, T, T + 0.9, -60.0, -59.2, SAND))                             # parapet on the river
    # white bands and blind arcade along the river front
    river = []
    for u in np.arange(-145, 145.1, 5.0):
        river += niche(float(u), 3.0, -6.5, -2.5, SAND_SH, SAND_SH, inner=0.0, frame=MARBLE, fw=0.2)
    river.append(poly_uy([(-155, -0.6), (155, -0.6), (155, -0.2), (-155, -0.2)], MARBLE, 0.04))
    ms += place(river, 180, 0, -60.0)
    # steps down into the garden on the axis
    for i, y in enumerate((0.8, 0.4)):
        ms.append(B(-13.0, 13.0, -0.2, y, ZN, ZN + 0.45 * (i + 1), SAND_L))
    # the octagonal towers at the corners of the terrace
    for sx in (-1, 1):
        ms += octo_tower(146.5 * sx, -52.0, -10.0, T + 9.5)
        ms += octo_tower(146.5 * sx, 52.0, -1.0, T + 9.5)
    # the tanks in front of the mosque and the jawab
    for sx in (-1, 1):
        x0, x1 = 73.6 * sx, 82.6 * sx
        x0, x1 = min(x0, x1), max(x0, x1)
        ms += [B(x0 - 0.6, x1 + 0.6, T - 0.2, T + 0.55, -6.1, -5.5, SAND), B(x0 - 0.6, x1 + 0.6, T - 0.2, T + 0.55, 5.5, 6.1, SAND),
               B(x0 - 0.6, x0, T - 0.2, T + 0.55, -5.5, 5.5, SAND), B(x1, x1 + 0.6, T - 0.2, T + 0.55, -5.5, 5.5, SAND)]
        ms.append(cap([(x0, -5.5), (x1, -5.5), (x1, 5.5), (x0, 5.5)], T + 0.35, WATER))
    return ms


def octo_tower(x, z, y0, y1, r=7.2):
    ms = [cyl(r, r, y1 - y0, SAND, 8).R(0, math.pi / 8, 0).T(x, y0, z)]
    for yb in (y1 - 4.6, y1 - 0.5):
        ms.append(cyl(r + 0.25, r + 0.25, 0.5, MARBLE, 8).R(0, math.pi / 8, 0).T(x, yb, z))
    # arched openings on the eight sides of the upper storey
    for k in range(8):
        a = math.radians(45 * k)
        side = niche(0, 2.6, 0.0, 2.2, SAND_SH, SAND_SH, inner=0.0, frame=MARBLE, fw=0.18)
        ms += place(side, 45 * k, x + r * math.cos(math.radians(22.5)) * math.sin(a),
                    z + r * math.cos(math.radians(22.5)) * math.cos(a), y=y1 - 4.0)
    ms += chhatri(x, z, y1, 3.6, SAND, MARBLE_W, fin=0.5)
    return ms


def mosque_local():
    """The mosque in its own axes: the front (prayer hall) faces +z, centre of the hall at the origin, y from the
    terrace."""
    ms = []
    L, Dp = 28.3, 11.7
    y0 = 1.0
    ms.append(B(-L - 1.2, L + 1.2, -0.2, y0, -Dp - 0.8, Dp + 2.2, SAND))
    a, dn = 4.4, 3.5
    ring = [(-L, -Dp), (L, -Dp), (L, Dp), (a, Dp), (a, Dp - dn), (-a, Dp - dn), (-a, Dp), (-L, Dp)]
    top = y0 + 14.0
    ms.append(walls(ring, y0 - 0.2, top, SAND))
    ms.append(cap(ring, top, SAND))
    hs = y0 + 8.6
    ap = apex(8.8, hs)
    fr = []
    fr += vault(a, ap, top, dn, SAND, SAND_SH)
    fr.append(prism_uy(arch_frame(8.0, y0, y0 + 19.3, 8.8, hs), [], 0.0, 1.0, SAND))
    fr.append(poly_uy([(-a, y0), (a, y0), (a, ap), (-a, ap)], SAND_SH, -dn + 0.04))
    fr += niche(0, 3.0, y0, y0 + 4.8, SAND_SH, (0.2, 0.1, 0.08), inner=0.0, frame=MARBLE, fw=0.2, wpos=-dn + 0.05)
    for sgn in (-1, 1):
        u = sgn * (a - 0.04)
        fr.append(quad((u, y0, -dn), (u, y0, 0), (u, ap, 0), (u, ap, -dn), SAND_SH) if sgn < 0 else
                  quad((u, y0, 0), (u, y0, -dn), (u, ap, -dn), (u, ap, 0), SAND_SH))
    # white marble frame of the arch and the band of the pishtaq
    for u0, u1, yy0, yy1 in ((-6.9, -6.3, y0 + 0.3, y0 + 17.2), (6.3, 6.9, y0 + 0.3, y0 + 17.2), (-6.9, 6.9, y0 + 16.6, y0 + 17.2)):
        fr.append(poly_uy([(u0, yy0), (u1, yy0), (u1, yy1), (u0, yy1)], MARBLE, 1.04))
    fr.append(arch_band(9.6, 8.8, y0, hs, MARBLE, 1.02))
    fr.append(poly_uy([(-L, top - 0.6), (L, top - 0.6), (L, top - 0.2), (-L, top - 0.2)], MARBLE, 0.04))
    fr = [m.T(0, 0, Dp) for m in fr]
    for u in (-21.4, -13.2, 13.2, 21.4):
        for m in niche(u, 5.4, y0 + 0.8, y0 + 7.0, SAND_SH, (0.2, 0.1, 0.08), inner=0.45, frame=MARBLE, fw=0.3):
            fr.append(m.T(0, 0, Dp))
        for m in niche(u, 2.6, y0 + 10.0, y0 + 11.6, SAND_SH, SAND_SH, inner=0.0, frame=MARBLE, fw=0.2):
            fr.append(m.T(0, 0, Dp))
    ms += fr
    for u in (-8.0, 8.0):
        ms += guldasta(u, Dp + 0.6, y0 + 10.0, y0 + 20.6, 0.45, SAND)
    # back and ends: blind arches
    back = []
    for u in np.arange(-24, 24.1, 6.0):
        back += niche(float(u), 3.4, y0 + 2.0, y0 + 7.0, SAND_SH, SAND_SH, inner=0.0, frame=MARBLE, fw=0.2)
    ms += place(back, 180, 0, -Dp)
    for sgn in (-1, 1):
        ms += place(niche(0, 4.0, y0 + 2.0, y0 + 7.5, SAND_SH, SAND_SH, inner=0.0, frame=MARBLE, fw=0.25),
                    90 * sgn, sgn * L, 0)
    # domes
    for u, rd, rb, hd, ht in ((0.0, 7.4, 8.2, 3.5, 11.5), (-17.5, 5.4, 6.1, 2.5, 9.0), (17.5, 5.4, 6.1, 2.5, 9.0)):
        ms.append(cyl(rd, rd, hd + 0.2, MARBLE, 24).T(u, top - 0.2, -0.8))
        ms.append(onion(rb, top + hd, ht, MARBLE_W, 24, base_r=rd).T(u, 0, -0.8))
        ms += finial(u, top + hd + ht - 0.3, -0.8, 3.4 if u == 0 else 2.6)
    # corner towers
    for sx in (-1, 1):
        for sz in (-1, 1):
            ms.append(cyl(1.6, 1.6, y0 + 19.0, SAND, 8).R(0, math.pi / 8, 0).T(sx * L, 0, sz * Dp))
            for yb in (y0 + 6.0, y0 + 13.8):
                ms.append(cyl(1.8, 1.8, 0.4, MARBLE, 8).R(0, math.pi / 8, 0).T(sx * L, yb, sz * Dp))
            ms += chhatri(sx * L, sz * Dp, y0 + 19.0, 1.25, SAND, MARBLE_W, fin=0.6)
    return ms


def mosque_and_jawab():
    loc = mosque_local()
    return place(loc, 90, -139.2, 0.0, y=T) + place(loc, -90, 139.2, 0.0, y=T)


# ── The charbagh ─────────────────────────────────────────────────────────
def cross_ring():
    a, p, e, L, Le = 9.0, 17.8, 9.0, 141.0, 139.5
    q = [(a, -L), (a, -p), (p, -p), (p, -e), (Le, -e), (Le, e), (p, e), (p, p), (a, p), (a, L),
         (-a, L), (-a, p), (-p, p), (-p, e), (-Le, e), (-Le, -e), (-p, -e), (-p, -p), (-a, -p), (-a, -L)]
    return [(x, z + ZC) for x, z in q]


def beds():
    out = []
    for sx in (-1, 1):
        for sz in (-1, 1):
            inner = [(22, 16), (77, 16), (77, 77), (16, 77), (16, 22), (22, 22)]
            for poly in (inner, [(83, 16), (143.5, 16), (143.5, 77), (83, 77)],
                         [(16, 83), (77, 83), (77, 143.5), (16, 143.5)], [(83, 83), (143.5, 83), (143.5, 143.5), (83, 143.5)]):
                out.append([(sx * u, ZC + sz * v) for u, v in poly])
    return out


def inside(x, z, P_):
    ins = False
    for i in range(len(P_)):
        (xi, zi), (xj, zj) = P_[i], P_[i - 1]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            ins = not ins
    return ins


def paving(outer, beds_, cross):
    """The walks at y = 0: the garden cut on a grid along every edge of the beds and of the cross, each cell that
    is neither a bed nor the cross paved (more robust than one polygon with 17 holes)."""
    xs = sorted({round(p_[0], 3) for r in [outer] + beds_ + [cross] for p_ in r})
    zs = sorted({round(p_[1], 3) for r in [outer] + beds_ + [cross] for p_ in r})
    x0, x1, z0, z1 = outer[0][0], outer[1][0], outer[0][1], outer[2][1]
    xs = [x for x in xs if x0 <= x <= x1]; zs = [z for z in zs if z0 <= z <= z1]
    V, F = [], []
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            cx, cz = (xs[i] + xs[i + 1]) / 2, (zs[j] + zs[j + 1]) / 2
            if inside(cx, cz, cross) or any(inside(cx, cz, b) for b in beds_):
                continue
            o = len(V)
            V += [(xs[i], 0.0, zs[j]), (xs[i + 1], 0.0, zs[j]), (xs[i + 1], 0.0, zs[j + 1]), (xs[i], 0.0, zs[j + 1])]
            F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    return [Mesh(V, F, PAVE)]


def garden():
    ms = []
    outer = [(-XW + 1.2, ZN), (XW - 1.2, ZN), (XW - 1.2, ZS), (-XW + 1.2, ZS)]
    B_ = beds()
    X = cross_ring()
    ms += paving(outer, B_, X)
    for b in B_:
        ms.append(walls(b, -0.3, 0.0, PAVE, hole=True))
        ms.append(cap(b, -0.3, GRASS))
    ms.append(walls(X, -0.3, 0.0, PAVE, hole=True))
    # north–south axis: lawn strips, raised marble channel, jets, cypresses
    for sz in (-1, 1):
        v0, v1 = 17.8, 141.0
        z0, z1 = sorted((ZC + sz * v0, ZC + sz * v1))
        ms.append(cap([(-9, z0), (9, z0), (9, z1), (-9, z1)], -0.3, GRASS_D))
        ms += [B(-3.4, -2.8, -0.5, 0.45, z0, z1, MARBLE), B(2.8, 3.4, -0.5, 0.45, z0, z1, MARBLE)]
        ms.append(B(-2.8, 2.8, -0.5, 0.45, z1 - 0.6, z1, MARBLE) if sz > 0 else B(-2.8, 2.8, -0.5, 0.45, z0, z0 + 0.6, MARBLE))
        ms.append(cap([(-2.8, z0), (2.8, z0), (2.8, z1), (-2.8, z1)], 0.25, WATER))
        for v in np.arange(22.0, 139.0, 4.6):
            ms.append(cyl(0.1, 0.02, 1.6, JET, 4).T(0, 0.25, ZC + sz * v))
        for v in np.arange(21.0, 139.0, 6.8):
            for u in (-6.2, 6.2):
                ms += cypress(u, ZC + sz * v)
    # east–west axis
    for sx in (-1, 1):
        u0, u1 = sorted((sx * 17.8, sx * 139.5))
        ms.append(cap([(u0, ZC - 9), (u1, ZC - 9), (u1, ZC + 9), (u0, ZC + 9)], -0.3, GRASS_D))
        ms += [B(u0, u1, -0.5, 0.45, ZC - 3.6, ZC - 3.0, MARBLE), B(u0, u1, -0.5, 0.45, ZC + 3.0, ZC + 3.6, MARBLE)]
        ms.append(cap([(u0, ZC - 3.0), (u1, ZC - 3.0), (u1, ZC + 3.0), (u0, ZC + 3.0)], 0.25, WATER))
        ms.append(B(sx * 80 - 3.0, sx * 80 + 3.0, 0.45, 0.75, ZC - 9.0, ZC + 9.0, PAVE))           # bridge of the walk
        for u in np.arange(24.0, 136.0, 7.0):
            if abs(u - 80) < 5:
                continue
            for v in (-6.3, 6.3):
                ms += cypress(sx * u, ZC + v, 0.7, 5.0)
    # the central raised tank, al-Hawd al-Kawthar
    p, w1, w2, w3 = 17.8, 16.8, 11.85, 6.3
    sqr = lambda r: [(-r, ZC - r), (r, ZC - r), (r, ZC + r), (-r, ZC + r)]               # noqa: E731
    ms.append(walls(sqr(p), -0.3, 0.6, MARBLE))
    ms.append(cap(sqr(p), 0.6, MARBLE, holes=[sqr(w1)]))
    ms.append(walls(sqr(w1), 0.0, 0.6, MARBLE, hole=True))
    ms.append(cap(sqr(w1), 0.3, WATER, holes=[sqr(w2)]))
    ms.append(walls(sqr(w2), 0.0, 0.75, MARBLE))
    ms.append(cap(sqr(w2), 0.75, MARBLE, holes=[sqr(w3)]))
    ms.append(walls(sqr(w3), 0.3, 0.75, MARBLE, hole=True))
    ms.append(cap(sqr(w3), 0.45, WATER))
    for dx, dz in ((0, 0), (3, 3), (3, -3), (-3, 3), (-3, -3)):
        ms.append(cyl(0.14, 0.03, 2.4 if dx == 0 else 1.6, JET, 5).T(dx, 0.45, ZC + dz))
    # marble bench on the south side of the tank
    ms.append(B(-6.0, 6.0, 0.6, 1.25, ZC + p - 0.1, ZC + p + 1.6, MARBLE))
    # trees in the beds (along their edges, like the old planting)
    for b in B_:
        xs = [p_[0] for p_ in b]; zs = [p_[1] for p_ in b]
        x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
        for i in range(9):
            side = rng.integers(4)
            t = rng.uniform(0.08, 0.92)
            x = x0 + 4.5 if side == 0 else x1 - 4.5 if side == 1 else x0 + t * (x1 - x0)
            z = z0 + 4.5 if side == 2 else z1 - 4.5 if side == 3 else z0 + t * (z1 - z0)
            if side < 2:
                z = z0 + t * (z1 - z0)
            if abs(x) < 26 and abs(z - ZC) < 26:
                continue
            ms += tree(x, z)
    return ms


def cypress(x, z, r=0.85, h=6.2):
    return [cyl(r, 0.0, h, CYPRESS, 6).T(x, -0.3 + 0.6, z), cyl(0.18, 0.18, 0.6, (0.3, 0.22, 0.15), 4, caps=False).T(x, -0.3, z)]


def tree(x, z):
    r = rng.uniform(3.2, 5.0)
    c = TREES[rng.integers(len(TREES))]
    crown = sphere(r, c, 7, 4).S(1, rng.uniform(0.75, 0.95), 1).T(x, -0.3 + r * 0.8 + rng.uniform(2.0, 3.5), z)
    return [crown, cyl(0.3, 0.25, 3.5, (0.32, 0.24, 0.17), 5, caps=False).T(x, -0.3, z)]


# ── Walls, towers, pavilions, the great gate ─────────────────────────────
def wall_run(x0, z0, x1, z1, y0=-2.0, h=WALL_H, t=2.4):
    """A crenellated red sandstone wall along a straight line, both faces with a white coping band."""
    L = math.hypot(x1 - x0, z1 - z0)
    ang = math.degrees(math.atan2(x1 - x0, z1 - z0)) - 90             # local u along the wall
    ms = [B(-L / 2, L / 2, y0, h, -t / 2, t / 2, SAND)]
    for s in (1, -1):
        band = poly_uy([(-L / 2, h - 0.65), (L / 2, h - 0.65), (L / 2, h - 0.35), (-L / 2, h - 0.35)], MARBLE, t / 2 + 0.03)
        ms.append(band.R(0, 0 if s > 0 else math.pi, 0))
    n = int(L / 2.6)
    for i in range(n):
        u = -L / 2 + (i + 0.5) * L / n
        ms.append(B(u - 0.65, u + 0.65, h, h + 1.0, -t / 2, -t / 2 + 0.9, SAND))
        ms.append(B(u - 0.65, u + 0.65, h, h + 1.0, t / 2 - 0.9, t / 2, SAND))
    return place(ms, ang, (x0 + x1) / 2, (z0 + z1) / 2)


def pavilion(sx):
    """The two-storey pavilions at the ends of the east–west axis, against the walls (local front +z)."""
    ms = []
    w, d, h = 22.0, 13.4, 10.5
    ms.append(B(-w / 2, w / 2, -1.0, h, -d, 0, SAND))
    ms.append(B(-w / 2 - 0.3, w / 2 + 0.3, h - 0.6, h - 0.1, -d - 0.3, 0.3, MARBLE))
    for u in (-6.5, 0.0, 6.5):
        ms += niche(u, 4.4 if u else 5.2, 0.4, 3.6 if u else 4.0, SAND_SH, SAND_SH, inner=0.0, frame=MARBLE, fw=0.25)
        ms += niche(u, 3.2, 5.6, 7.6, SAND_SH, SAND_SH, inner=0.0, frame=MARBLE, fw=0.2)
    for u in (-w / 2 + 1.0, w / 2 - 1.0):
        ms.append(B(u - 0.8, u + 0.8, h, h + 0.9, -1.8, -0.2, SAND))
        ms += chhatri(u, -1.0, h + 0.9, 1.1, SAND, MARBLE_W, fin=0.6)
    return place(ms, -90 * sx, sx * (XW - 1.2 - d + 0.6), ZC)


def gate():
    """The Darwaza-i rauza: 41.2 × 34 m, 23.07 m to the parapet; local front +z faces the forecourt (south)."""
    ms = []
    Wg, Dg, Hg, cg = 20.6, 17.0, 23.07, 4.0
    ring = [(-Wg + cg, -Dg), (Wg - cg, -Dg), (Wg, -Dg + cg), (Wg, Dg - cg), (Wg - cg, Dg), (-Wg + cg, Dg), (-Wg, Dg - cg), (-Wg, -Dg + cg)]
    a, dn = 5.0, 5.5
    ring2 = []
    for x, z in ring:
        ring2.append((x, z))
    # cut the two iwans (north and south) into the plan
    ring2 = [(-Wg + cg, -Dg), (-a, -Dg), (-a, -Dg + dn), (a, -Dg + dn), (a, -Dg), (Wg - cg, -Dg), (Wg, -Dg + cg),
             (Wg, Dg - cg), (Wg - cg, Dg), (a, Dg), (a, Dg - dn), (-a, Dg - dn), (-a, Dg), (-Wg + cg, Dg),
             (-Wg, Dg - cg), (-Wg, -Dg + cg)]
    ms.append(walls(ring2, -2.0, Hg, SAND))
    ms.append(cap(ring2, Hg, SAND))
    hs = 11.0
    ap = apex(10.0, hs)
    fr = []
    fr += vault(a, ap, Hg, dn, SAND, SAND_SH)
    fr.append(prism_uy(arch_frame(10.5, -2.0, Hg + 1.2, 10.0, hs), [], 0.0, 0.9, SAND))
    fr.append(poly_uy([(-a, 0), (a, 0), (a, ap), (-a, ap)], SAND_SH, -dn + 0.04))
    fr += niche(0, 4.0, 0.0, 6.0, SAND_SH, (0.12, 0.06, 0.05), inner=0.0, frame=MARBLE, fw=0.25, wpos=-dn + 0.05)
    fr += niche(0, 3.6, 8.0, 10.6, SAND_SH, SAND_SH, inner=0.0, frame=MARBLE, fw=0.2, wpos=-dn + 0.05)
    for sgn in (-1, 1):
        u = sgn * (a - 0.04)
        fr.append(quad((u, 0, -dn), (u, 0, 0), (u, ap, 0), (u, ap, -dn), SAND_SH) if sgn < 0 else
                  quad((u, 0, 0), (u, 0, -dn), (u, ap, -dn), (u, ap, 0), SAND_SH))
    # white marble band with black calligraphy round the arch
    for u0, u1, y0, y1 in ((-8.6, -7.4, 0.3, 20.6), (7.4, 8.6, 0.3, 20.6), (-8.6, 8.6, 19.4, 20.6)):
        fr.append(poly_uy([(u0, y0), (u1, y0), (u1, y1), (u0, y1)], MARBLE, 0.94))
    for u0, u1, y0, y1 in ((-8.2, -7.8, 0.6, 20.2), (7.8, 8.2, 0.6, 20.2), (-8.2, 8.2, 19.8, 20.2)):
        fr.append(poly_uy([(u0, y0), (u1, y0), (u1, y1), (u0, y1)], INLAY, 0.97))
    fr.append(arch_band(10.9, 10.0, 0.0, hs, MARBLE, 0.92))
    for u in (-13.8, 13.8):
        for y0, hs2 in ((1.0, 6.5), (11.0, 16.0)):
            fr += niche(u, 4.6, y0, hs2, SAND_SH, (0.2, 0.1, 0.08), inner=0.45, frame=MARBLE, fw=0.3)
    for u in (-10.5, 10.5):
        fr += guldasta(u, 0.45, 12.0, Hg + 3.4, 0.5, SAND)
    # the row of eleven chhatris over the front
    for i in range(11):
        u = -15.0 + 3.0 * i
        fr.append(B(u - 1.3, u + 1.3, Hg - 0.2, Hg + 1.5, -1.6, 0.6, SAND))
        fr += chhatri(u, -0.5, Hg + 1.5, 0.82, SAND, MARBLE_W, seg=6, fin=0.5)
    for ang in (0, 180):
        ms += place(fr, ang, *rot(0, Dg, ang))
    # east and west sides
    side = niche(0, 6.0, 1.0, 10.0, SAND_SH, (0.2, 0.1, 0.08), inner=0.45, frame=MARBLE, fw=0.4)
    side += niche(-9.0, 4.0, 1.0, 7.0, SAND_SH, SAND_SH, inner=0.0, frame=MARBLE, fw=0.3)
    side += niche(9.0, 4.0, 1.0, 7.0, SAND_SH, SAND_SH, inner=0.0, frame=MARBLE, fw=0.3)
    for ang in (90, -90):
        ms += place(side, ang, *rot(0, Wg, ang))
    # corner towers with their chhatris
    for sx in (-1, 1):
        for sz in (-1, 1):
            cx, cz = sx * (Wg - cg / 2 + 0.6), sz * (Dg - cg / 2 + 0.6)
            ms.append(cyl(3.0, 3.0, Hg + 4.0, SAND, 8).R(0, math.pi / 8, 0).T(cx, -2.0, cz))
            for yb in (7.5, 15.5, Hg + 1.4):
                ms.append(cyl(3.2, 3.2, 0.45, MARBLE, 8).R(0, math.pi / 8, 0).T(cx, yb, cz))
            ms += chhatri(cx, cz, Hg + 2.0, 1.75, SAND, MARBLE_W, fin=0.6)
    return place(ms, 0, 0, GATE_Z)


def enclosure():
    ms = []
    for sx in (-1, 1):
        ms += wall_run(sx * XW, ZN, sx * XW, ZW)
        ms += wall_run(sx * XW, ZW, sx * 20.6, ZW)
        ms += octo_tower(sx * 147.5, ZW, -2.0, 9.5)
        ms += pavilion(sx)
    ms += gate()
    return ms


def build():
    m_tomb = tomb()
    for sx in (-1, 1):
        for sz in (-1, 1):
            m_tomb += minaret(48.0 * sx, 48.0 * sz)
    m_pl = plinth()
    m_ter = terrace()
    m_mos = mosque_and_jawab()
    m_gar = garden()
    m_enc = enclosure()
    return [("tomb", merge(m_tomb)), ("plinth", merge(m_pl)), ("terrace", merge(m_ter)), ("mosque_jawab", merge(m_mos)),
            ("garden", merge(m_gar)), ("walls_gate", merge(m_enc))]


FOOTPRINT = [(-157.0, -61.0), (157.0, -61.0), (157.0, 367.5), (22.0, 367.5), (22.0, GATE_Z + 20.0),
             (-22.0, GATE_Z + 20.0), (-22.0, 367.5), (-157.0, 367.5)]


def main():
    parts = build()
    out_dir = ROOT / "models" / "Q9141"
    out_dir.mkdir(parents=True, exist_ok=True)
    named, special = [], {"gilt": [], "water": []}
    for name, m in parts:
        for key, col in (("gilt", GOLD), ("water", WATER)):        # faces painted exactly that colour
            hit = np.all(np.abs(m.c[m.f] - np.array(col)).max(axis=2) < 1e-6, axis=1)
            if hit.any():
                special[key].append(Mesh(m.v, m.f[hit], m.c)); m = Mesh(m.v, m.f[~hit], m.c)
        k = 0.95 + 0.05 * fbm(m.v, 0.06, 7, 2)
        m.c = np.clip(m.c * k[:, None], 0, 1)
        used = np.unique(m.f)
        remap = np.full(len(m.v), -1); remap[used] = np.arange(len(used))
        named.append((name, Mesh(m.v[used], remap[m.f], m.c[used])))
    for key in ("gilt", "water"):
        w = merge(special[key])
        used = np.unique(w.f)
        remap = np.full(len(w.v), -1); remap[used] = np.arange(len(used))
        named.append((key, Mesh(w.v[used], remap[w.f], w.c[used]), {"material": key}))
    tris, size = write_glb(out_dir / "model.glb", named, roughness=0.85,
                           materials={"gilt": METAL, "water": {"metallic": 0.0, "roughness": 0.12}})
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        mj["footprint"] = [[round(x, 1), round(z, 1)] for x, z in FOOTPRINT]
        meta.write_bytes((json.dumps(mj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"{tris} triangles, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
