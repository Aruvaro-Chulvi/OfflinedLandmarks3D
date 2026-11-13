"""
Puerta de Alcalá (Q1140634), Madrid — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: Francesco Sabatini (1716–1797), 1769–1778, public domain.

Measurements (see meta.json for the sources):
- OpenStreetMap way 174805987 (traced from aerial imagery, ±0.5 m): footprint 43.2 × 12.3 m, side bodies about
  7 m deep at the cornice, central body projecting on both faces; long axis at 350.6°, so the exterior (east)
  façade faces 80.6°. Bays from the centre: central arch 5.1, wide pier with the columns ~6, side arch 5.1,
  pier ~2.45, rectangular opening ~3.2, end pier ~2.3.
- Heights: pediment apex 21.946 m (es.wikipedia, OSM height); the rest measured on near-orthographic photos
  (Wikimedia Commons, «Madrid Puerta de Alcala west side.jpg» and others), ±3–5 %: plinth 1.7, lintels 5.8,
  imposts 6.9, side cornice 13.4, parapet 14.45, central cornice 14.2, attic cornice 19.4, arms ~25 m.
- East (exterior) façade: ten engaged Ionic columns of grey granite on pedestals, satyr masks on the arch
  keystones, garland panels over the rectangular openings, groups of a child (a cardinal virtue) and a
  military trophy on the side parapets, and the royal arms with Fame and a Genius over a segmental pediment.
- West (interior) façade: pilasters, and only the two pairs of columns beside the central arch; lion heads on
  the keystones, crossed cornucopias on the panels, empty-armour trophies on the parapets, and a segmental
  pediment holding a triangular one with two reclining trophies.
- Both attics: «REGE CAROLO III / ANNO / MDCCLXXVIII».
Grey granite for the structure, white limestone for the ornament and the sculptures, lead on the roofs.
Sculpture is stylised (volumes and silhouettes, as seen from a few dozen metres up).

Axes (docs/MODEL_SPEC.md): +Y up, the exterior (east) façade faces +Z, the gate runs along X (+X to the
north-north-west), origin on the ground in the middle of the central arch.

    python model_sources/Q1140634_puerta_de_alcala.py   -> models/Q1140634/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, quad, tube, write_glb  # noqa: E402

rng = np.random.default_rng(1778)

GRANITE = np.array((0.71, 0.695, 0.665))
GRANITE_D = np.array((0.50, 0.49, 0.465))
STONE = (0.88, 0.865, 0.82)            # Colmenar limestone: capitals, panels, keystones, plaque
MARBLE = (0.93, 0.92, 0.885)          # sculpture
PLINTH = (0.78, 0.765, 0.73)          # the light lowest course
LEAD = (0.40, 0.43, 0.46)
INK = (0.16, 0.15, 0.14)
FOUND = -3.0                          # foundations below the ground (MODEL_SPEC §3)


def granite(k=1.0):
    return tuple(np.clip(GRANITE * k * (1 + rng.normal(0, 0.018)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    """Axis-aligned box from its extents."""
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def mirror_x(m):
    n = m.copy(); n.v[:, 0] *= -1; n.f = n.f[:, ::-1]; return n


def mirror_z(m):
    n = m.copy(); n.v[:, 2] *= -1; n.f = n.f[:, ::-1]; return n


def both_x(ms):
    ms = [m for m in ms if m is not None]
    return ms + [mirror_x(m) for m in ms]


def sphere(r, c, seg=12, rings=7):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


# ── Arches and openings ──────────────────────────────────────────────────────
def arch_block(x0, x1, ys, ytop, z0, z1, c, seg=24):
    """The wall over a semicircular arch: front and back faces (rectangle minus the half circle) and the
    soffit (intrados) through the wall."""
    r, xc = (x1 - x0) / 2, (x0 + x1) / 2
    arc = [(xc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(math.pi, 0, seg + 1)]
    out = []
    for z in (z0, z1):
        V = [(x, y, z) for x, y in arc] + [(x0, ytop, z), (x1, ytop, z)]
        TL, TR, h = seg + 1, seg + 2, seg // 2
        F = [(TL, i, i + 1) for i in range(h)] + [(TL, h, TR)] + [(TR, i, i + 1) for i in range(h, seg)]
        out.append(Mesh(V, F, c))
    V, F = [], []
    for x, y in arc:
        V += [(x, y, z0), (x, y, z1)]
    for i in range(seg):
        a = 2 * i; F += [(a, a + 1, a + 3), (a, a + 3, a + 2)]
    out.append(Mesh(V, F, c))
    return merge(out)


def archivolt(x0, x1, ys, z, c, off=0.2, rad=0.11, seg=20):
    r, xc = (x1 - x0) / 2 + off, (x0 + x1) / 2
    pts = [(xc + r * math.cos(a), ys + r * math.sin(a), z) for a in np.linspace(math.pi, 0, seg + 1)]
    return tube(pts, rad, c, 6)


def keystone(xc, y_crown, zf, side, c, face_c, lion):
    """Keystone with its carved head (a satyr mask on the east, a lion on the west)."""
    k = B(xc - 0.42, xc + 0.42, y_crown - 0.35, y_crown + 0.95, zf - 0.3 * side, zf + 0.32 * side, c)
    hd = sphere(0.36 if lion else 0.3, face_c, 10, 6).S(1, 1.15, 0.7).T(xc, y_crown + 0.25, zf + 0.36 * side)
    out = [k, hd]
    if lion:                                                  # the mane
        out.append(cyl(0.5, 0.5, 0.12, face_c, 12).R(math.pi / 2, 0, 0).T(xc, y_crown + 0.25, zf + 0.3 * side))
    return out


# ── Orders ───────────────────────────────────────────────────────────────────
R_COL = 0.48


def ionic_column(x, zf, side, y_cap_top, ped=None):
    """Engaged Ionic column on its pedestal, standing out of the face zf (side +1 east, −1 west)."""
    zc = zf + 0.22 * side
    px0, px1 = ped if isinstance(ped, tuple) else (x - 0.7, x + 0.7)
    out = [] if ped == "skip" else [
           B(px0, px1, 0, 1.55, zc - 0.7, zc + 0.7, granite()),                                # pedestal
           B(px0 - 0.08, px1 + 0.08, 1.45, 1.7, zc - 0.78, zc + 0.78, STONE),                  # its cap
           B(px0 - 0.08, px1 + 0.08, 0, 0.3, zc - 0.78, zc + 0.78, PLINTH)]
    y0, y1 = 1.7, y_cap_top - 0.75
    out.append(cyl(R_COL * 1.22, R_COL * 1.12, 0.28, STONE, 16).T(x, y0, zc))                  # base
    out.append(cyl(R_COL, R_COL * 0.86, y1 - y0 - 0.28, granite(1.02), 16, caps=False).T(x, y0 + 0.28, zc))
    out.append(cyl(R_COL * 0.95, R_COL * 1.12, 0.3, STONE, 16).T(x, y1, zc))                   # echinus
    out.append(B(x - 0.66, x + 0.66, y1 + 0.3, y1 + 0.5, zc - 0.66, zc + 0.66, STONE))         # abacus
    for sx in (-1, 1):                                                                          # volutes
        out.append(cyl(0.2, 0.2, 1.2, STONE, 10).R(math.pi / 2, 0, 0).T(x + sx * 0.56, y1 + 0.22, zc - 0.6))
    out.append(B(x - 0.75, x + 0.75, y1 + 0.5, y_cap_top, zc - 0.75, zc + 0.75, granite(0.97)))  # block under the entablature
    return out


def pilaster(x, zf, side, y_cap_top):
    out = [B(x - 0.62, x + 0.62, 0, 1.7, zf, zf + 0.3 * side, granite()),
           B(x - 0.66, x + 0.66, 0, 0.3, zf, zf + 0.34 * side, PLINTH),
           B(x - 0.5, x + 0.5, 1.7, y_cap_top - 0.6, zf, zf + 0.2 * side, granite(1.03)),
           B(x - 0.62, x + 0.62, y_cap_top - 0.6, y_cap_top, zf, zf + 0.28 * side, STONE)]
    for sx in (-1, 1):
        out.append(cyl(0.16, 0.16, 0.3, STONE, 8).R(math.pi / 2, 0, 0).T(x + sx * 0.46, y_cap_top - 0.42, zf + (0.0 if side > 0 else -0.3)))
    return out


# ── Sculpture (stylised) ─────────────────────────────────────────────────────
def figure(x, y, z, h, c, side, wings=False, arm=False, seated=False):
    """A draped figure: robe, head; wings and a raised arm for Fame; seated for the children."""
    if seated:
        prof = [(0.2 * h, 0), (0.21 * h, 0.25 * h), (0.15 * h, 0.5 * h), (0.11 * h, 0.66 * h), (0.06 * h, 0.72 * h), (0, 0.74 * h)]
    else:
        prof = [(0.15 * h, 0), (0.135 * h, 0.3 * h), (0.1 * h, 0.55 * h), (0.09 * h, 0.72 * h), (0.05 * h, 0.79 * h), (0, 0.8 * h)]
    out = [lathe(prof, c, 12).T(x, y, z), sphere(0.075 * h, c, 10, 6).T(x, y + 0.87 * h, z)]
    if arm:
        out.append(tube([(x + 0.08 * h, y + 0.72 * h, z), (x + 0.14 * h, y + 0.92 * h, z + 0.05 * side), (x + 0.17 * h, y + 1.06 * h, z + 0.08 * side)], 0.03 * h, c, 6))
    if wings:
        for s in (-1, 1):
            w = box(0.06 * h, 0.55 * h, 0.22 * h, c, bottom=False).R(0, 0, s * 0.45).R(0, s * 0.5, 0)
            out.append(w.T(x + s * 0.15 * h, y + 0.72 * h, z - 0.1 * h * side))
    return out


def trophy(x, y, z, h, c, side):
    """Military trophy: crossed flagstaffs with flags, a plumed helmet and a shield."""
    out = []
    for s in (-1, 1):
        st = cyl(0.05, 0.05, h, c, 6).R(0, 0, s * 0.3).T(x, y, z)
        out.append(st)
        tip = np.array([x - math.sin(s * 0.3) * h, y + math.cos(s * 0.3) * h, z])
        out.append(B(-0.45, 0.45, -0.32, 0.32, -0.03, 0.03, c).R(0, 0, s * 0.3 + s * 0.25).T(*(tip + np.array([s * 0.35, -0.45, 0]))))
    out.append(sphere(0.3, c, 10, 6).S(1, 1.1, 1).T(x, y + 0.62 * h, z + 0.08 * side))
    out.append(cyl(0.12, 0.0, 0.55, c, 8).R(0, 0, 0.6).T(x, y + 0.85 * h, z + 0.08 * side))
    out.append(cyl(0.42, 0.42, 0.1, c, 14).R(math.pi / 2, 0, 0).T(x, y + 0.36 * h, z + 0.18 * side))
    return out


def armour(x, y, z, h, c, side, recline=0.0):
    """Empty armour: cuirass, helmet with plume, flags behind, a shield and a bundle of rods."""
    out = [lathe([(0.34, 0), (0.38, 0.25), (0.33, 0.55), (0.22, 0.72), (0, 0.74)], c, 12).S(1.1, 1, 0.7).T(0, 0, 0),
           sphere(0.24, c, 10, 6).T(0, 0.98, 0), cyl(0.1, 0.0, 0.5, c, 6).R(0, 0, -0.5).T(0, 1.18, 0),
           cyl(0.4, 0.4, 0.09, c, 14).R(math.pi / 2, 0, 0).S(1, 1.1, 1).T(0.55, 0.25, 0.18 * side)]
    for s in (-1, 1):
        out.append(cyl(0.045, 0.045, h, c, 6).R(0, 0, s * 0.35).T(0, -0.2, -0.15 * side))
        out.append(B(-0.4, 0.4, -0.28, 0.28, -0.03, 0.03, c).R(0, 0, s * 0.55).T(-s * 0.55 * h * 0.4, h * 0.72, -0.15 * side))
    out.append(cyl(0.11, 0.11, 1.3, c, 8).R(0, 0, 1.2).T(-0.6, 0.15, 0.12 * side))
    m = merge(out).R(0, 0, recline).T(x, y, z)
    return [m]


def royal_arms(x, y, z, c):
    """The royal coat of arms with its crown and cross (top about 25 m)."""
    shield = cyl(1.0, 1.0, 0.45, c, 20, sx=0.84).R(math.pi / 2, 0, 0).S(1, 1.3, 1).T(x, y + 1.35, z)
    rim = tube([(x + 0.88 * math.cos(a), y + 1.35 + 1.3 * math.sin(a), z + 0.25) for a in np.linspace(0, 2 * math.pi, 25)], 0.09, c, 6)
    crown = [cyl(0.72, 0.82, 0.42, c, 16).T(x, y + 2.75, z), lathe([(0.82, 0), (0.6, 0.35), (0.25, 0.6), (0, 0.66)], c, 12).T(x, y + 3.17, z),
             sphere(0.14, c, 8, 5).T(x, y + 3.88, z)]
    cross = [B(x - 0.07, x + 0.07, y + 3.95, y + 4.65, z - 0.07, z + 0.07, c), B(x - 0.25, x + 0.25, y + 4.32, y + 4.44, z - 0.07, z + 0.07, c)]
    collar = [sphere(0.13, c, 8, 4).T(x + 0.95 * math.cos(a), y + 1.35 + 1.4 * math.sin(a), z + 0.1) for a in np.linspace(-2.6, -0.54, 8)]
    return [shield, rim] + crown + cross + collar


# ── Lettering: «REGE CAROLO III / ANNO / MDCCLXXVIII» ───────────────────────
FONT = {  # 5×7 block letters, rows from the top
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "V": ["10001", "10001", "10001", "10001", "01010", "01010", "00100"],
}


def inscription(lines, zf, side, y_top, px=0.075):
    """Dark block letters on the plaque, centred, reading left to right for someone facing that façade."""
    out = []
    for li, text in enumerate(lines):
        w = (len(text) * 6 - 1) * px
        col = 0
        for ch in text:
            if ch == " ":
                col += 4; continue
            for r_, row in enumerate(FONT[ch]):
                run = 0
                for c_, bit in enumerate(row + "0"):          # one quad per horizontal run of pixels
                    if bit == "1":
                        run += 1; continue
                    if run:
                        x = -w / 2 + (col + c_ - run / 2) * px
                        y = y_top - li * 10 * px - r_ * px - px / 2
                        out.append(quad(x * side, y, zf, run * px * 0.98, px * 0.98, "+z" if side > 0 else "-z", INK, eps=0.03))
                    run = 0
            col += 6
    return out


# ── The gate ─────────────────────────────────────────────────────────────────
def gate():
    E, W = +1, -1
    parts = []
    # Half the gate (x ≥ 0), then mirrored. Side body: x 6.55–21.6, 6.2 m deep at the base.
    D1, D2, D3 = 3.1, 5.1, 6.15            # half-depths: side body, central body outer step, central body
    TOP_S, TOP_C = 10.9, 11.6              # where the entablature starts (side, centre)
    half = []
    half.append(B(19.3, 21.6, FOUND, TOP_S, -D1, D1, granite()))                 # end pier
    half.append(B(16.1, 19.3, 5.8, TOP_S, -D1, D1, granite()))                    # wall over the rectangular opening
    half.append(B(16.1, 19.3, 5.55, 5.8, -D1 - 0.08, D1 + 0.08, STONE))            # its lintel
    half.append(B(13.65, 16.1, FOUND, TOP_S, -D1, D1, granite()))                 # pier
    half.append(arch_block(8.55, 13.65, 6.9, TOP_S, -D1, D1, granite()))          # side arch
    half.append(B(6.55, 8.55, FOUND, TOP_S, -D1, D1, granite()))                  # pier beside the side arch
    half.append(B(5.15, 6.55, FOUND, TOP_C, -D2, D2, granite()))                  # central body, outer step
    half.append(B(2.55, 5.15, FOUND, TOP_C, -D3, D3, granite()))                  # central body, pier of the central arch
    # light lowest course, impost bands, frames of the rectangular opening
    for x0, x1, d in ((19.3, 21.6, D1), (13.65, 16.1, D1), (6.55, 8.55, D1), (5.15, 6.55, D2), (2.55, 5.15, D3)):
        half.append(B(x0 - 0.04, x1 + 0.04, 0, 0.45, -d - 0.05, d + 0.05, PLINTH))
    for x0, x1, d, yi in ((13.65, 16.1, D1, 6.9), (6.55, 8.55, D1, 6.9), (5.15, 6.55, D2, 7.15), (2.55, 5.15, D3, 7.15)):
        half.append(B(x0 - 0.05, x1 + 0.05, yi - 0.28, yi, -d - 0.1, d + 0.1, STONE))
    for s in (E, W):
        zf = s * (D1 + 0.04)
        half += [B(15.85, 16.14, 0.45, 5.8, min(zf, zf + 0.1 * s), max(zf, zf + 0.1 * s), STONE),
                 B(19.26, 19.55, 0.45, 5.8, min(zf, zf + 0.1 * s), max(zf, zf + 0.1 * s), STONE)]
        half.append(archivolt(8.55, 13.65, 6.9, s * (D1 + 0.06), STONE))
        half += keystone(11.1, 6.9 + 2.55, s * D1, s, STONE, MARBLE, lion=(s == W))
    # the soffits of the openings (inside faces of the piers are the boxes' own sides)
    # Entablature and cornice of the side bodies (deeper on the east, where the columns stand out)
    half.append(B(6.55, 21.6, TOP_S, 12.6, -3.35, 3.85, granite(0.98)))
    half.append(B(6.55, 21.64, TOP_S - 0.03, TOP_S + 0.12, -3.4, 3.9, STONE))
    half.append(B(6.55, 21.95, 12.6, 13.4, -3.7, 4.2, granite(1.04)))
    half.append(B(6.6, 21.4, 13.4, 14.45, -3.0, 3.0, granite()))                  # parapet
    half.append(B(6.55, 21.5, 14.45, 14.6, -3.08, 3.08, STONE))
    half.append(hip(6.9, 21.1, -2.6, 2.6, 14.0, 15.0, LEAD))                      # low lead roof
    # Central body: entablature and cornice
    half.append(B(5.15, 6.55, TOP_C, 13.4, -5.2, 5.2, granite(0.98)))
    half.append(B(0.0, 5.15, TOP_C, 13.4, -6.9, 6.9, granite(0.98)))
    half.append(B(0.0, 5.19, TOP_C - 0.03, TOP_C + 0.12, -6.95, 6.95, STONE))
    half.append(B(5.15, 6.9, 13.4, 14.2, -5.55, 5.55, granite(1.04)))
    half.append(B(0.0, 5.15, 13.4, 14.2, -7.25, 7.25, granite(1.04)))
    # Orders: east columns, west pilasters (+ the two columns of each pair beside the central arch)
    for x in (20.45, 14.9, 7.55):
        half += ionic_column(x, E * D1, E, TOP_S)
        half += pilaster(x, W * D1, W, TOP_S)
    for s in (E, W):                                         # the pair shares one pedestal
        half += ionic_column(3.2, s * D3, s, TOP_C, ped=(2.6, 5.1))
        half += ionic_column(4.5, s * D3, s, TOP_C, ped="skip")
    # Panels over the rectangular openings: garlands (east), crossed cornucopias (west)
    half.append(B(17.7 - 1.67, 17.7 + 1.67, 7.5, 9.17, D1, D1 + 0.12, STONE))
    half.append(tube([(17.7 + 1.3 * math.cos(a), 8.6 - 0.4 * math.sin(a), D1 + 0.18) for a in np.linspace(math.pi, 0, 9)], 0.12, MARBLE, 6))
    half.append(B(17.7 - 1.5, 17.7 + 1.5, 7.1, 9.6, -D1 - 0.12, -D1, STONE))
    for s in (-1, 1):
        half.append(cyl(0.28, 0.04, 2.3, MARBLE, 10).R(0, 0, s * 0.85).T(17.7 + s * 0.8, 7.5, -D1 - 0.2))
    # Sculpture on the side parapets: child (a virtue) + trophy on the east; empty armours on the west
    for x in (20.45, 14.9):
        half += figure(x - 0.5, 14.45, 1.7, 1.55, MARBLE, E, seated=True)
        half += trophy(x + 0.45, 14.45, 1.4, 2.6, MARBLE, E)
        half += armour(x, 14.45, -1.7, 2.4, MARBLE, W)
    parts += both_x(half)

    # The central arch (spans both halves)
    parts.append(arch_block(-2.55, 2.55, 7.15, TOP_C, -D3, D3, granite()))
    for s in (E, W):
        parts.append(archivolt(-2.55, 2.55, 7.15, s * (D3 + 0.06), STONE, off=0.22, rad=0.13))
        parts += keystone(0.0, 7.15 + 2.55, s * D3, s, STONE, MARBLE, lion=(s == W))

    # Attic
    AX, AZ = 5.9, 4.7
    parts.append(B(-6.15, 6.15, 14.2, 14.85, -AZ - 0.2, AZ + 0.2, granite(1.03)))
    parts.append(B(-AX, AX, 14.85, 18.9, -AZ, AZ, granite()))
    parts.append(B(-6.25, 6.25, 18.9, 19.4, -AZ - 0.35, AZ + 0.35, granite(1.05)))
    for sx in (-1, 1):                                       # scroll consoles at the ends of the attic
        for s in (-1, 1):                                    # one console at each corner, against each façade
            parts.append(cyl(0.5, 0.5, 1.1, STONE, 12).R(math.pi / 2, 0, 0).T(sx * 6.3, 15.4, s * AZ - (1.1 if s > 0 else 0.0)))
            parts.append(B(sx * 6.3 - 0.4, sx * 6.3 + 0.4, 14.85, 15.4, s * AZ - (1.1 if s > 0 else 0.0), s * AZ + (0.0 if s > 0 else 1.1), granite()))
    for s in (E, W):
        zf = s * AZ
        lo, hi = min(zf, zf + 0.1 * s), max(zf, zf + 0.1 * s)
        parts.append(B(-3.9, 3.9, 15.45, 17.95, lo, hi, STONE))                 # inscription plaque
        parts += inscription(["REGE CAROLO III", "ANNO", "MDCCLXXVIII"], zf + 0.1 * s, s, 17.6)
        for x in (-5.35, -4.45, 4.45, 5.35):                                    # pilaster strips with their drops
            parts.append(B(x - 0.27, x + 0.27, 14.85, 18.9, min(zf, zf + 0.14 * s), max(zf, zf + 0.14 * s), granite(1.05)))
            for k in (-0.14, 0.0, 0.14):
                parts.append(B(x + k - 0.05, x + k + 0.05, 18.0, 18.55, min(zf, zf + 0.2 * s), max(zf, zf + 0.2 * s), STONE))
        for sx in (-1, 1):                                                      # little winged heads beside the plaque
            parts.append(sphere(0.22, MARBLE, 8, 5).T(sx * 4.0, 17.55, zf + 0.25 * s))
            parts.append(B(-0.35, 0.35, -0.1, 0.1, -0.04, 0.04, MARBLE).R(0, 0, sx * 0.4).T(sx * 4.0 + sx * 0.3, 17.6, zf + 0.2 * s))
        parts.append(tube([(1.8 * math.cos(a), 15.3 - 0.35 * math.sin(a), zf + 0.18 * s) for a in np.linspace(math.pi, 0, 9)], 0.1, MARBLE, 6))
    # Segmental pediment roof over the attic (a lead barrel running east–west), tympana and their cornice
    parts += pediment(AZ + 0.35)
    # East: the royal arms, Fame (south, left) and a Genius (north, right)
    parts += royal_arms(0.0, 20.35, AZ + 0.55, MARBLE)
    parts += figure(-3.4, ped_y(3.4) - 0.3, AZ + 0.3, 2.6, MARBLE, E, wings=True, arm=True)
    parts += figure(3.2, ped_y(3.2) - 0.3, AZ + 0.3, 1.8, MARBLE, E, wings=True)
    # West: a triangular pediment inside the segmental one, with two reclining armour trophies
    zw = -(AZ + 0.4)
    for p in ([(-4.9, 19.55, zw), (0.0, 21.45, zw)], [(0.0, 21.45, zw), (4.9, 19.55, zw)], [(-4.9, 19.55, zw), (4.9, 19.55, zw)]):
        parts.append(tube(p, 0.13, STONE, 6))
    for sx in (-1, 1):
        parts += armour(sx * 3.2, ped_y(3.2) - 0.35, -(AZ - 0.2), 2.3, MARBLE, W, recline=-sx * 0.55)
    return merge(parts)


# segmental pediment: chord ±5.6 m at 19.4, apex 21.95
PA, PS = 5.6, 2.55
PR = (PA * PA + PS * PS) / (2 * PS)
PCY = 19.4 + PS - PR


def ped_y(x):
    return PCY + math.sqrt(max(0.0, PR * PR - x * x))


def pediment(zA, seg=24):
    a0 = math.asin(PA / PR)
    angs = np.linspace(math.pi / 2 + a0, math.pi / 2 - a0, seg + 1)
    arc = [(PR * math.cos(a), PCY + PR * math.sin(a)) for a in angs]
    out = []
    for z in (-zA, zA):                                      # tympana
        V = [(0.0, 19.4, z)] + [(x, y, z) for x, y in arc]
        F = [(0, i + 1, i + 2) for i in range(seg)]
        out.append(Mesh(V, F, granite(1.02)))
    V, F = [], []                                            # the lead barrel on top
    for x, y in arc:
        V += [(x * 1.04, y + 0.12, -zA - 0.25), (x * 1.04, y + 0.12, zA + 0.25)]
    for i in range(seg):
        a = 2 * i; F += [(a, a + 1, a + 3), (a, a + 3, a + 2)]
    out.append(Mesh(V, F, LEAD))
    for z in (-zA - 0.15, zA + 0.15):                        # the raking cornice
        out.append(tube([(x * 1.03, y + 0.05, z) for x, y in arc], 0.2, granite(1.06), 6))
    return out


def hip(x0, x1, z0, z1, y0, y1, c):
    """Low hipped roof: ridge along x."""
    d = (z1 - z0) / 2
    zc = (z0 + z1) / 2
    r0, r1 = x0 + d, x1 - d
    V = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (r0, y1, zc), (r1, y1, zc)]
    F = [(0, 1, 5), (0, 5, 4), (3, 4, 5), (3, 5, 2), (0, 4, 3), (1, 2, 5)]
    return Mesh(V, F, c)


def main():
    out = ROOT / "models" / "Q1140634"
    out.mkdir(parents=True, exist_ok=True)
    m = gate()
    tris, size = write_glb(out / "model.glb", [("puerta_de_alcala", m)], roughness=0.92)
    lo, hi = m.v.min(0), m.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")


if __name__ == "__main__":
    main()
