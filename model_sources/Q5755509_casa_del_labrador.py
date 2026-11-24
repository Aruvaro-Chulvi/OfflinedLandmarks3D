"""
Casa del Labrador (Q5755509), Jardín del Príncipe, Aranjuez — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Juan de Villanueva (1791–1794, died 1811) for the future Charles IV, enlarged with the two wings
and the court of honour by Isidro González Velázquez (1798–1803, died 1840). Public domain.

What is modelled:
- The house: the main block and the two wings round the court of honour, three storeys — the ground floor in
  white stone (granite ashlar, partly replaced by false-stone cladding in 1964–1968), the two upper storeys in
  brick with stone frames, the balconies and niches with statues of the main floor, the frieze of reliefs under
  the cornice, the slate hipped roofs with dormers and chimneys, the royal arms with the two angels over the
  cornice of the court front, the plaque «Reinando Carlos IV. Año de MDCCCIII».
- The court: the two one-storey arcades along the inner sides of the wings, with their terraces, iron railings
  and the busts of Greco-Roman figures on pedestals; the stone paving; the iron railing with stone pillars topped
  by busts and the wrought-iron gate that closes the court.
- The surroundings: the sanded plaza round the house, the two rows of planted beds along its flanks with their
  box edging and trees, the hedges that frame them.
- Not modelled: the lawns and avenues of the Jardín del Príncipe further out, the interiors.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap way 389892208 (the house), ways 389892211 and 389892212 (the arcades), way 1448321444 (the
  railing), way 1448324505 (the plaza), ways 803123388–803123400 (the beds), in the axes of heading 149.9° (the
  court of honour opens to the south-south-east). Main block 48.5 × 11 m, wings 10 m wide, 30.6 m deep in all.
- Heights from photographs on Wikimedia Commons scaled on the plan (±10 %; the OSM height of 9 m is too low
  for three storeys): ground floor 5.6 m, cornice at 12 m, ridge at 15.6 m, the arms to 14.8 m; arcades 5.6 m to
  the terrace.
Ground: flat (the Jardín del Príncipe, ≈ 490 m); y = 0 is the court.

Axes (docs/MODEL_SPEC.md): +Y up, the court of honour (main front) faces +Z (heading 149.9°), +X points to
59.9° (east-north-east). Origin at the centroid of the OSM outline (in the main block).

    python model_sources/Q5755509_casa_del_labrador.py   -> models/Q5755509/model.glb (+ footprint in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, cap, walls, ring_area, offset, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(1803)

# ── Colours ──────────────────────────────────────────────────────────────────
BRICK = (0.80, 0.57, 0.40)            # brick of the upper storeys (ochre-orange in the sun)
STONE = (0.91, 0.90, 0.86)            # white stone of the ground floor, frames, cornices
STONE_D = (0.80, 0.79, 0.75)
PANEL = (0.95, 0.94, 0.91)            # stucco reliefs of the frieze
SLATE = (0.33, 0.35, 0.40)
GLASS = (0.13, 0.14, 0.16)
DARK = (0.10, 0.09, 0.09)
SHUTTER = (0.35, 0.18, 0.13)          # the red-brown shutters and doors
IRON = (0.10, 0.10, 0.11)
MARBLE = (0.93, 0.92, 0.89)
GOLD = (0.80, 0.62, 0.25)
SAND = (0.82, 0.74, 0.60)
PAVING = (0.78, 0.76, 0.71)
GRASS = (0.40, 0.54, 0.27)
BOX = (0.17, 0.31, 0.14)
LEAF = [(0.22, 0.36, 0.16), (0.26, 0.40, 0.17), (0.19, 0.32, 0.15), (0.29, 0.42, 0.19)]
TRUNK = (0.36, 0.31, 0.25)

FOUND = -2.0
G1, E = 5.6, 12.0          # top of the ground floor (string course), cornice

# ── Data (OpenStreetMap © OpenStreetMap contributors, ODbL), metres in the model axes ───────────────────
DATA = json.loads(r"""{"outline":[[-25.05,-11.76],[23.49,-11.67],[23.52,18.8],[13.57,18.8],[13.51,-0.76],[8.14,-0.75],[-9.77,-0.72],[-18.77,-0.7],[-18.75,10.46],[-14.85,10.4],[-14.97,18.79],[-24.89,18.8]],"arcades":[[[8.14,-0.75],[13.51,-0.76],[13.57,18.8],[7.98,18.79]],[[-18.77,-0.7],[-9.77,-0.72],[-9.51,18.8],[-14.97,18.79],[-14.85,10.4],[-18.75,10.46]]],"fence":[[-9.51,18.8],[0.03,18.79],[7.98,18.79]],"plaza":[[-48.78,-49.85],[-48.12,13.98],[-49.02,16.15],[-49.95,18.43],[-49.6,20.72],[-50.61,20.82],[-51.67,21.37],[-52.4,22.33],[-52.74,23.64],[-52.7,24.44],[-52.29,25.31],[-51.12,26.33],[-50.01,26.72],[-50.07,27.74],[-43.86,27.69],[-40.77,28.35],[-37.22,29.1],[-36.62,28.48],[-35.89,28.13],[-33.83,27.94],[-26.45,28.08],[-25.43,28.54],[-24.91,29.51],[-24.08,30.91],[-22.93,32.58],[-21.49,34.31],[-20.02,35.79],[-18.2,37.37],[-15.07,39.51],[-12.71,40.83],[-12.07,41.46],[-0.56,41.73],[10.14,41.99],[11.05,41.01],[14.53,39.15],[16.61,37.62],[18.33,36.21],[21.38,32.85],[23.89,29.22],[24.44,28.57],[25.5,28.21],[33.43,28.4],[34.85,28.75],[35.54,29.3],[39.31,28.81],[42.4,28.4],[49.04,28.5],[49.01,27.12],[50.36,26.64],[51.27,25.7],[51.71,24.38],[51.54,23.09],[50.89,22.03],[50.15,21.54],[49.19,21.19],[48.9,18.89],[48.71,17.3],[49.21,-49.67],[46.06,-51.15],[42.63,-52.75],[35.26,-40.07],[33.21,-36.53],[24.08,-49.74],[1.81,-50.44],[-3.96,-50.48],[-27.44,-50.58],[-35.77,-35.35],[-37.67,-38.79],[-44.7,-51.08],[-46.75,-50.46]],"beds":[[[41.93,-34.86],[41.92,-26.44],[41.64,-26.34],[41.36,-26.13],[41.13,-25.91],[40.96,-25.69],[40.79,-25.33],[37.61,-25.33],[36.98,-26.07]],[[40.75,-23.33],[40.92,-22.97],[41.1,-22.73],[41.35,-22.5],[41.61,-22.33],[41.95,-22.14],[41.94,-18.86],[41.69,-18.74],[41.46,-18.59],[41.22,-18.42],[41.01,-18.21],[40.8,-17.87],[37.45,-17.74],[37.29,-18.01],[37.12,-18.24],[36.94,-18.42],[36.71,-18.6],[36.53,-18.7],[36.3,-18.77],[36.31,-22.12],[36.57,-22.22],[36.83,-22.44],[37.05,-22.61],[37.26,-22.85],[37.46,-23.24]],[[40.84,-16.51],[41.07,-16.16],[41.26,-15.93],[41.54,-15.72],[41.9,-15.51],[41.89,-12.37],[41.62,-12.26],[41.37,-12.11],[41.13,-11.93],[40.91,-11.68],[40.75,-11.48],[40.63,-11.25],[37.54,-11.08],[37.39,-11.43],[37.23,-11.64],[36.99,-11.82],[36.63,-12.02],[36.29,-12.2],[36.33,-15.48],[36.76,-15.69],[37.02,-15.91],[37.24,-16.16],[37.44,-16.47]],[[40.6,-9.92],[40.8,-9.6],[40.97,-9.39],[41.22,-9.17],[41.48,-9.0],[41.84,-8.86],[41.81,-5.67],[41.55,-5.48],[41.22,-5.17],[41.03,-4.87],[40.92,-4.67],[40.86,-4.42],[37.59,-4.22],[37.45,-4.5],[37.26,-4.71],[36.98,-4.97],[36.67,-5.15],[36.3,-5.28],[36.33,-8.91],[36.53,-9.0],[36.76,-9.17],[36.97,-9.35],[37.14,-9.55],[37.3,-9.81]],[[40.68,-2.3],[40.84,-2.05],[41.05,-1.8],[41.25,-1.6],[41.47,-1.45],[41.82,-1.27],[41.91,2.04],[41.66,2.15],[41.44,2.3],[41.22,2.47],[40.98,2.76],[40.8,3.05],[37.42,3.07],[37.21,2.74],[37.05,2.53],[36.84,2.36],[36.59,2.2],[36.37,2.09],[36.32,-1.1],[36.58,-1.21],[36.81,-1.38],[37.02,-1.56],[37.25,-1.89],[37.42,-2.19]],[[40.69,4.3],[40.84,4.56],[40.98,4.8],[41.23,5.04],[41.47,5.21],[41.81,5.39],[41.86,8.34],[41.57,8.46],[41.32,8.62],[41.11,8.76],[40.85,9.04],[40.64,9.35],[37.27,9.63],[37.16,9.37],[37.0,9.13],[36.81,8.9],[36.58,8.73],[36.33,8.57],[36.35,5.41],[36.64,5.29],[36.87,5.11],[37.1,4.9],[37.31,4.63],[37.43,4.41]],[[40.74,10.9],[40.9,11.12],[41.04,11.3],[41.27,11.5],[41.5,11.64],[41.78,11.76],[41.86,14.87],[41.58,15.03],[41.39,15.16],[41.2,15.32],[41.03,15.51],[40.87,15.74],[40.71,16.09],[37.62,16.27],[37.43,15.97],[37.24,15.76],[37.03,15.59],[36.79,15.43],[36.57,15.32],[36.35,15.26],[36.36,12.05],[36.59,11.92],[36.88,11.74],[37.11,11.53],[37.29,11.29],[37.44,11.01]],[[-46.92,-39.32],[-40.27,-26.88],[-40.96,-25.59],[-45.01,-25.51],[-45.08,-25.78],[-45.16,-26.01],[-45.28,-26.31],[-45.51,-26.61],[-45.79,-26.88],[-46.15,-27.08],[-46.46,-27.2],[-46.79,-27.28]],[[-45.14,-23.48],[-40.5,-23.59],[-40.38,-23.2],[-40.2,-22.84],[-39.93,-22.5],[-39.46,-22.15],[-39.15,-22.01],[-38.81,-21.93],[-38.72,-16.73],[-39.03,-16.68],[-39.34,-16.55],[-39.66,-16.37],[-39.89,-16.19],[-40.08,-15.98],[-40.27,-15.73],[-40.39,-15.47],[-40.46,-15.19],[-45.0,-14.92],[-45.08,-15.28],[-45.27,-15.65],[-45.58,-16.09],[-45.95,-16.33],[-46.31,-16.5],[-46.67,-16.58],[-46.73,-21.73],[-46.4,-21.83],[-46.17,-21.94],[-45.88,-22.14],[-45.6,-22.39],[-45.39,-22.7],[-45.25,-22.98],[-45.16,-23.25]],[[-45.16,-13.01],[-40.34,-13.1],[-40.2,-12.78],[-39.98,-12.42],[-39.67,-12.11],[-39.35,-11.9],[-39.02,-11.77],[-38.7,-11.7],[-38.57,-5.96],[-38.85,-5.93],[-39.09,-5.84],[-39.39,-5.69],[-39.64,-5.55],[-39.87,-5.35],[-40.13,-4.98],[-40.37,-4.51],[-44.88,-4.38],[-45.12,-5.07],[-45.48,-5.51],[-45.84,-5.78],[-46.23,-5.97],[-46.55,-6.03],[-46.61,-11.47],[-46.36,-11.54],[-46.12,-11.64],[-45.88,-11.83],[-45.66,-12.03],[-45.43,-12.31],[-45.3,-12.58],[-45.2,-12.79]],[[-44.95,-2.47],[-40.41,-2.53],[-40.26,-2.1],[-40.05,-1.73],[-39.75,-1.43],[-39.42,-1.21],[-39.02,-1.06],[-38.6,-0.95],[-38.5,4.34],[-38.8,4.38],[-39.08,4.47],[-39.38,4.61],[-39.63,4.79],[-39.99,5.18],[-40.18,5.52],[-40.32,6.01],[-44.86,6.09],[-44.97,5.73],[-45.14,5.4],[-45.41,5.08],[-45.77,4.82],[-46.07,4.66],[-46.39,4.57],[-46.42,-0.83],[-46.2,-0.87],[-45.95,-0.98],[-45.65,-1.23],[-45.37,-1.49],[-45.17,-1.81],[-45.01,-2.14]],[[-44.95,7.85],[-40.22,7.82],[-40.11,8.22],[-39.96,8.55],[-39.69,8.87],[-39.39,9.15],[-38.91,9.37],[-38.46,9.48],[-38.44,14.53],[-38.75,14.58],[-39.06,14.67],[-39.34,14.81],[-39.57,14.99],[-39.84,15.26],[-40.0,15.47],[-40.15,15.71],[-40.27,16.12],[-44.46,16.17],[-44.57,15.82],[-44.79,15.43],[-45.13,15.05],[-45.52,14.77],[-45.91,14.61],[-46.23,14.58],[-46.35,9.71],[-46.05,9.58],[-45.77,9.43],[-45.52,9.19],[-45.34,9.01],[-45.2,8.77],[-45.07,8.47],[-45.0,8.2]]],"hedges":[[[34.64,-26.1],[37.98,-31.81],[45.0,-44.08],[44.84,16.41]],[[-46.44,-45.46],[-37.65,-30.75],[-34.67,-25.62],[-34.29,18.47],[-46.29,18.47]],[[-48.12,13.98],[-48.78,-49.85]],[[44.82,18.45],[32.97,18.66],[33.05,-23.19]],[[-35.77,-35.35],[-27.44,-50.58],[-3.96,-50.48]],[[1.81,-50.44],[24.08,-49.74],[33.21,-36.53]]],"trees":[[-42.45,-19.53],[-42.6,-8.55],[-42.63,2.24],[-42.11,12.3],[39.2,-20.03],[38.95,-14.11],[39.02,-7.28],[38.99,0.54],[39.06,7.04],[39.05,13.68]]}""")


def tint(c, k=0.015):
    return tuple(np.clip(np.array(c) * (1 + rng.normal(0, k)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sphere(r, c, seg=10, rings=6):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def ccw(P):
    P = [tuple(p) for p in P]
    if len(P) > 1 and math.dist(P[0], P[-1]) < 1e-6:
        P = P[:-1]
    return P if ring_area(P) > 0 else P[::-1]


def bbox(P):
    xs = [p[0] for p in P]; zs = [p[1] for p in P]
    return min(xs), max(xs), min(zs), max(zs)


class Face:
    """A wall plane: a = start point (x, z), t = direction along the wall, n = outward normal; local (u, y, w)."""

    def __init__(self, a, t, n):
        self.a, self.t, self.n = np.asarray(a, float), np.asarray(t, float), np.asarray(n, float)

    def P(self, u, y, w):
        p = self.a + self.t * u + self.n * w
        return (p[0], y, p[1])

    def poly(self, pts, c, w=0.05):
        V = [self.P(u, y, w) for u, y in pts]
        F = [(0, i, i + 1) for i in range(1, len(V) - 1)]
        m = Mesh(V, F, c)
        a, b, cc = np.array(V[0]), np.array(V[1]), np.array(V[2])
        nn = np.cross(b - a, cc - a)
        if np.dot([nn[0], nn[2]], self.n) < 0:
            m.f = m.f[:, ::-1]
        return m

    def slab(self, u0, u1, y0, y1, w0, w1, c):
        P = [self.P(u, y, w) for w in (w0, w1) for y in (y0, y1) for u in (u0, u1)]
        quads = [(4, 5, 7, 6), (1, 0, 2, 3), (0, 4, 6, 2), (5, 1, 3, 7), (6, 7, 3, 2), (0, 1, 5, 4)]
        V, F = [], []
        for q in quads:
            o = len(V); V += [P[k] for k in q]; F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        m = Mesh(V, F, c)
        ctr = np.mean(np.array(P), 0)
        for k in range(0, len(F), 2):
            a, b, cc = (np.array(V[i]) for i in F[k])
            if np.dot(np.cross(b - a, cc - a), (a + b + cc) / 3 - ctr) < 0:
                m.f[k] = m.f[k][::-1]; m.f[k + 1] = m.f[k + 1][::-1]
        return m


def arch_pts(u, y0, ys, half, seg=6):
    pts = [(u - half, y0), (u + half, y0)]
    pts += [(u + half * math.cos(a), ys + half * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)]
    return pts


def rect_pts(u, y, w, h):
    return [(u - w / 2, y - h / 2), (u + w / 2, y - h / 2), (u + w / 2, y + h / 2), (u - w / 2, y + h / 2)]


def face_of(x0, z0, x1, z1, n):
    a, b = np.array([x0, z0], float), np.array([x1, z1], float)
    L = float(np.linalg.norm(b - a))
    return Face(a, (b - a) / L, n), L


def hip_roof(x0, x1, z0, z1, E_, R, c, inset=0.0):
    x0, x1, z0, z1 = x0 + inset, x1 - inset, z0 + inset, z1 - inset
    L, W = x1 - x0, z1 - z0
    if L >= W:
        a, b = (x0 + W / 2, (z0 + z1) / 2), (x1 - W / 2, (z0 + z1) / 2)
    else:
        a, b = ((x0 + x1) / 2, z0 + L / 2), ((x0 + x1) / 2, z1 - L / 2)
    k = R / (min(L, W) / 2)
    Y = E_ + R
    c00, c10, c11, c01 = (x0, E_, z0), (x1, E_, z0), (x1, E_, z1), (x0, E_, z1)
    A, Bp = (a[0], Y, a[1]), (b[0], Y, b[1])
    if L >= W:
        faces = [(c00, c10, Bp, A), (c11, c01, A, Bp), (c01, c00, A), (c10, c11, Bp)]
    else:
        faces = [(c10, c11, Bp, A), (c01, c00, A, Bp), (c00, c10, A), (c11, c01, Bp)]
    V, F = [], []
    for fc in faces:
        o = len(V); V += list(fc)
        F += [(o, o + 2, o + 1)] + ([(o, o + 3, o + 2)] if len(fc) == 4 else [])
    m = Mesh(V, F, c)
    a_, b_, c_ = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    flip = np.cross(b_ - a_, c_ - a_)[:, 1] < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m, k


def bust(x, y, z, c=MARBLE):
    return [lathe([(0.0, 0.0), (0.32, 0.0), (0.32, 0.12), (0.18, 0.2), (0.42, 0.55), (0.36, 0.75), (0.14, 0.85),
                   (0.0, 0.86)], c, 8).T(x, y, z), sphere(0.17, c, 8, 5).T(x, y + 1.02, z)]


def statue(x, y, z, h=1.8, c=MARBLE):
    k = h / 4.2
    body = lathe([(0.0, 0.0), (0.62, 0.0), (0.62, 0.25), (0.55, 0.3), (0.62, 1.6), (0.55, 2.6), (0.62, 3.05),
                  (0.42, 3.35), (0.18, 3.45), (0.0, 3.46)], c, 8)
    return merge([body, sphere(0.3, c, 8, 5).T(0, 3.72, 0)]).S(k).T(x, y, z)


# ── The house ────────────────────────────────────────────────────────────────
def bays(L, step, margin=1.4):
    n = max(1, int(round((L - 2 * margin) / step)))
    s = (L - 2 * margin) / n
    return [margin + s * (k + 0.5) for k in range(n)], s


def wall_dress(f, L, step=4.2, court=False, ground=True, top=E):
    """Stone ground floor, brick above, windows, balconies, niches and the frieze of reliefs."""
    out = [f.slab(-0.05, L + 0.05, FOUND, G1, -0.05, 0.12, tint(STONE, 0.01)),        # white ground storey
           f.slab(-0.25, L + 0.25, G1 - 0.1, G1 + 0.35, -0.05, 0.35, STONE),             # string course
           f.slab(-0.4, L + 0.4, top - 0.7, top, -0.05, 0.6, STONE),                     # cornice
           f.slab(-0.1, 0.7, G1, top - 0.7, -0.05, 0.1, STONE), f.slab(L - 0.7, L + 0.1, G1, top - 0.7, -0.05, 0.1, STONE)]
    us, s = bays(L, step)
    for i, u in enumerate(us):
        if ground:
            out += [f.poly(rect_pts(u, 2.4, 1.4, 2.4), STONE_D, 0.14), f.poly(rect_pts(u, 2.4, 1.0, 2.0), GLASS, 0.16)]
        # main floor: balcony window, a niche with a statue between bays in the court
        out += [f.poly(rect_pts(u, G1 + 2.1, 1.75, 3.3), STONE, 0.03), f.poly(rect_pts(u, G1 + 2.0, 1.25, 2.9), SHUTTER, 0.06),
                f.slab(u - 0.6, u + 0.6, G1 + 3.6, G1 + 3.85, -0.02, 0.3, STONE),
                f.slab(u - 0.85, u + 0.85, G1 + 0.4, G1 + 0.55, 0.0, 0.6, STONE),
                f.slab(u - 0.75, u + 0.75, G1 + 0.55, G1 + 1.45, 0.55, 0.58, IRON)]
        out += [f.poly(rect_pts(u, 10.4, 0.9, 0.9), GLASS, 0.05)]
        if i < len(us) - 1:
            m = u + s / 2
            out.append(f.poly(rect_pts(m, 10.4, min(2.6, s * 0.55), 0.9), PANEL, 0.04))          # relief panel
            if court and s > 3.4:
                out.append(f.poly(arch_pts(m, G1 + 0.9, G1 + 3.4, 0.55), (0.80, 0.78, 0.74), 0.03))
                p = f.P(m, G1 + 0.95, 0.25)
                out.append(statue(p[0], p[1], p[2], 1.9))
    return out


def house():
    O = ccw(DATA["outline"])
    out = [merge([walls(O, FOUND, E, tint(BRICK, 0.02)), cap(O, E, SLATE)])]
    # every side of the outline
    for i in range(len(O)):
        a, b = np.array(O[i]), np.array(O[(i + 1) % len(O)])
        L = float(np.linalg.norm(b - a))
        if L < 3.0:
            continue
        t = (b - a) / L; n = np.array([t[1], -t[0]])
        f = Face(a, t, n)
        mid = (a + b) / 2
        in_court = -19.0 < mid[0] < 14.0 and mid[1] > -2.0 and abs(n[1]) < 0.5 or (abs(mid[1] + 0.73) < 0.2 and n[1] > 0.5)
        # behind the arcades the ground floor is hidden: no windows there
        ground = not (in_court and abs(n[0]) > 0.5)
        out += wall_dress(f, L, 4.2 if L > 12 else max(3.0, L / 3), court=in_court, ground=ground)
    # roofs: main block and the two wings, hipped, with dormers and chimneys
    for x0, x1, z0, z1 in ((-25.0, 23.5, -11.8, -0.7), (13.5, 23.5, -11.8, 18.8), (-24.9, -14.9, -11.8, 18.8)):
        m, k = hip_roof(x0, x1, z0, z1, E, 3.6, tint(SLATE, 0.02), inset=-0.35)
        out.append(m)
    for x, z, nrm in ((-15.0, -0.7, (0, 1)), (13.0, -0.7, (0, 1)), (-15.0, -11.8, (0, -1)), (0.0, -11.8, (0, -1)), (14.0, -11.8, (0, -1)),
                      (23.5, 4.0, (1, 0)), (23.5, 14.0, (1, 0)), (-24.9, 4.0, (-1, 0)), (-24.9, 14.0, (-1, 0))):
        out += dormer(x, z, nrm)
    for x, z in ((-8.0, -6.3), (8.0, -6.3), (18.5, 8.0), (-19.9, 8.0), (18.5, -6.3), (-19.9, -6.3)):
        out += [B(x - 0.5, x + 0.5, E + 1.5, E + 5.2, z - 0.8, z + 0.8, STONE), B(x - 0.65, x + 0.65, E + 5.2, E + 5.45, z - 0.95, z + 0.95, STONE)]
    # the royal arms held by two angels over the court front, and the plaque
    cx, zf = -0.75, -0.7
    out += [B(cx - 2.2, cx + 2.2, E - 0.2, E + 1.6, zf - 0.6, zf + 0.25, STONE),
            lathe([(1.4, 0.0), (1.4, 0.4), (1.1, 1.6), (0.5, 2.4), (0.0, 2.8)], STONE, 10).S(1.0, 1.0, 0.25).T(cx, E + 1.6, zf),
            sphere(0.45, GOLD, 8, 5).T(cx, E + 2.6, zf + 0.25)]
    for s in (-1, 1):
        out.append(statue(cx + s * 2.0, E + 1.6, zf - 0.1, 1.5, MARBLE))
    f = Face((cx - 2.0, zf), (1, 0), (0, 1))
    out.append(f.poly(rect_pts(2.0, E - 1.5, 3.6, 0.7), (0.30, 0.28, 0.26), 0.62))
    return out


def dormer(x, z, nrm):
    n = np.array(nrm, float); t = np.array([-n[1], n[0]])
    f = Face(np.array([x, z]) - n * 1.1, t, n)
    yb = E + 0.9
    out = [f.slab(-0.7, 0.7, E, yb + 1.9, -1.4, 0.0, SLATE), f.poly(rect_pts(0.0, yb + 0.9, 0.7, 1.1), GLASS, 0.03)]
    V = [f.P(-0.8, yb + 1.9, 0.05), f.P(0.8, yb + 1.9, 0.05), f.P(0.0, yb + 2.5, 0.05),
         f.P(-0.8, yb + 1.9, -1.5), f.P(0.8, yb + 1.9, -1.5), f.P(0.0, yb + 2.5, -1.5)]
    out.append(Mesh(V, [(0, 1, 2), (0, 2, 5), (0, 5, 3), (1, 4, 5), (1, 5, 2)], SLATE))
    return out


def arcades():
    """The two arcades along the wings (OSM ways 389892211, 389892212): granite arches, a terrace above with an
    iron railing and busts on pedestals."""
    out = []
    for A in DATA["arcades"]:
        P = ccw(A)
        out.append(merge([walls(P, FOUND, G1, tint(STONE, 0.01)), cap(P, G1, PAVING)]))
        x0, x1, z0, z1 = bbox(P)
        east = x0 > 0
        xf = x0 if east else x1                                   # the face on the court
        f, L = face_of(xf, z0, xf, z1, (-1, 0)) if east else face_of(xf, z1, xf, z0, (1, 0))
        us, s = bays(L, 3.9, 0.6)
        for u in us:
            out.append(f.poly(arch_pts(u, 0.0, 3.4, min(1.5, s * 0.36)), DARK, 0.04))
        out.append(f.slab(-0.1, L + 0.1, G1 - 0.45, G1, -0.05, 0.3, STONE))
        # the end on the railing (z1)
        g, Lg = face_of(x0, z1, x1, z1, (0, 1))
        out.append(g.poly(arch_pts(Lg / 2, 0.0, 3.2, min(1.8, Lg * 0.3)), DARK, 0.04))
        out.append(g.poly(rect_pts(Lg / 2, 1.6, min(3.2, Lg * 0.55), 3.0), IRON, 0.06))
        # terrace railing and busts along the court side and the end
        for ff, LL in ((f, L), (g, Lg)):
            out.append(ff.slab(0.0, LL, G1 + 0.95, G1 + 1.02, -0.25, -0.18, IRON))
            for u in np.arange(0.0, LL + 0.01, LL / max(1, round(LL / 1.2))):
                p = ff.P(u, 0, -0.22)
                out.append(B(p[0] - 0.02, p[0] + 0.02, G1, G1 + 1.0, p[2] - 0.02, p[2] + 0.02, IRON))
            for u in np.arange(0.0, LL + 0.01, LL / max(1, round(LL / 3.9))):
                p = ff.P(u, 0, -0.3)
                out += [B(p[0] - 0.3, p[0] + 0.3, G1, G1 + 1.1, p[2] - 0.3, p[2] + 0.3, STONE)] + bust(p[0], G1 + 1.1, p[2])
    return out


def court():
    out = []
    x0, x1 = -9.8, 8.1
    out.append(cap([(x0, -0.7), (x1, -0.7), (x1, 18.8), (x0, 18.8)], 0.04, PAVING))
    # the railing: stone pillars with busts, iron bars, the gate in the middle
    pts = DATA["fence"]
    xa, xb, zr = pts[0][0], pts[-1][0], pts[0][1]
    piers = [xa, xa + (0.03 - xa) / 2, 0.03 - 1.8, 0.03 + 1.8, 0.03 + (xb - 0.03) / 2, xb]
    for x in piers:
        out += [B(x - 0.4, x + 0.4, 0.0, 2.6, zr - 0.4, zr + 0.4, STONE), B(x - 0.5, x + 0.5, 2.6, 2.85, zr - 0.5, zr + 0.5, STONE)]
        out += bust(x, 2.85, zr)
    f = Face((xa, zr), (1, 0), (0, 1))
    L = xb - xa
    out += [f.slab(0.0, L, 0.0, 0.5, -0.2, 0.2, STONE), f.slab(0.0, L, 2.25, 2.35, -0.04, 0.04, IRON)]
    for u in np.arange(0.2, L, 0.25):
        out.append(f.slab(u - 0.02, u + 0.02, 0.5, 2.35, -0.02, 0.02, IRON))
    out.append(f.slab(0.03 - xa - 1.4, 0.03 - xa + 1.4, 2.35, 3.1, -0.03, 0.03, IRON))   # the crest of the gate
    return out


# ── Surroundings ─────────────────────────────────────────────────────────────
def tree(x, z, h, r):
    c = tint(LEAF[rng.integers(len(LEAF))], 0.06)
    hc = h * 0.66
    prof = [(0.0, 0.0), (r * 0.8, 0.1 * hc), (r, 0.38 * hc), (r * 0.88, 0.68 * hc), (r * 0.45, 0.93 * hc), (0.0, hc)]
    return [lathe(prof, c, 8).T(x, h - hc, z), cyl(0.35, 0.25, h - hc + 0.8, TRUNK, 6, caps=False).T(x, 0.0, z)]


def surroundings():
    out = [cap(ccw(DATA["plaza"]), 0.02, SAND)]
    for P in DATA["beds"]:
        P = ccw(P)
        inner = offset(P, -0.35)
        out.append(merge([walls(P, -0.1, 0.45, BOX), walls(inner, -0.1, 0.45, BOX, hole=True), cap(P, 0.45, BOX, holes=[inner])]))
        out.append(cap(inner, 0.1, tint(GRASS, 0.04)))
    for x, z in DATA["trees"]:
        out += tree(x, z, rng.uniform(13.0, 17.0), rng.uniform(3.6, 4.6))
    for H in DATA["hedges"]:
        for a, b in zip(H[:-1], H[1:]):
            a, b = np.array(a), np.array(b)
            L = float(np.linalg.norm(b - a))
            if L < 0.3:
                continue
            t = (b - a) / L; n = np.array([t[1], -t[0]])
            out.append(Face(a - n * 0.45, t, n).slab(0.0, L, -0.1, 1.3, 0.0, 0.9, tint(BOX, 0.05)))
    return out


def footprint():
    P = ccw(DATA["plaza"])
    return offset(P, 1.0)


def main():
    parts = [("house", merge(house() + arcades() + court())), ("surroundings", merge(surroundings()))]
    out_dir = ROOT / "models" / "Q5755509"
    out_dir.mkdir(parents=True, exist_ok=True)
    named = []
    for name, m in parts:
        k = 0.95 + 0.05 * fbm(m.v, 0.15, 7, 2)
        m.c = np.clip(m.c * k[:, None], 0, 1)
        named.append((name, m))
    tris, size = write_glb(out_dir / "model.glb", named, roughness=0.88)
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        mj["footprint"] = [[round(x, 1), round(z, 1)] for x, z in footprint()]
        meta.write_text(json.dumps(mj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{tris} triangles, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
