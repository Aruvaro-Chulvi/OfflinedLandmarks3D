"""
Colosseum — Colosseo, Amphitheatrum Flavium, Rome (Q10285) — built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: begun under Vespasian about 70–72 AD, inaugurated by Titus in 80 AD, finished under Domitian
(architects unknown). The two brick buttresses: Raffaele Stern (east end, 1806–1807, died 1820) and Giuseppe
Valadier (west end, 1823–1826, died 1839). Public domain.

The model shows the amphitheatre as it stands today, not as it was built.

Measurements (see meta.json for the sources):
- Plan: an ellipse of 188 × 156 m (Wikipedia en/it: 189 × 156 m; 187.75 × 155.6 m in the older surveys), the arena
  86 × 54 m, so the cavea is 51 m wide all round. The rings are drawn as offset curves of the outer ellipse
  (constant depth d from the outer face), which is how the building was laid out (rings of piers on the radial
  lines). Centre and axes from OpenStreetMap relation 1834818 (building:parts): an ellipse fitted to the low
  ring of the cavea (part 1834825, 12 m) gives 130.8 × 98.7 m, exactly the 188 × 156 ellipse offset by 28.6 m,
  long axis at 109.7°/289.7°. The OSM trace of the high outer wall lies 3–6 m inside that ellipse on the north
  (traced on the top of a 48 m wall that leans in the imagery), so the published size is used instead.
- Rings (depth d from the outer face, ±0.5 m, from the published plans): outer wall of travertine piers 0–2.4,
  outer ambulatory 2.4–7.7, second ring of travertine piers 7.7–10, then the inner ambulatories and the radial
  walls that carried the seats, a corridor (praecinctio) at d ≈ 28.6 m (the OSM ring), the podium at d = 51.
- 80 bays round the outer wall (bay 6.77 m on the 541.5 m perimeter), the four axial gates on the axes; arches
  4.2 m wide. The outer wall survives on the north side for the numbered arches XXIII–LIV plus the north axial
  gate (33 bays, model bays 4–36), with Valadier's buttress at the west end (bays 1–3, brick, the arcades
  repeated, stepping down) and Stern's at the east end (a sloping brick spur, the last arches walled up),
  together bays 1–39.9 as in the OSM part 1834827.
- Heights of the outer wall (published total 48–48.5 m, OSM height 48.5; the storeys measured from elevation
  drawings, ±5 %): Tuscan order to 10.6 m (arches 7.05 m high), Ionic 10.6–22.4, Corinthian 22.4–33.9 (arches
  6.45 m on a 1.6 m pedestal course), attic with Corinthian pilasters, small windows in every other bay and three
  corbels per bay for the velarium masts, 33.9–48.5. On the south side the outer wall is gone and the second
  ring stands as the outside of the building, to 26–34 m (OSM part 1834828: 36 m), with the stubs of the
  radial walls of the lost outer ambulatory in front. The north side inside: ambulatory vaults at each storey,
  the second ring to 39 m.
- Cavea: the seats are gone; the radial walls and the vaults that carried them are modelled as a ruined, ribbed
  slope (OSM parts: 24 m between the second ring and the praecinctio, 12–15 m lower down), higher on the north,
  falling to the podium (≈4.5 m above the arena). The two axial passages at the ends of the long axis open
  into the arena.
- Arena: the exposed hypogeum, its walls from OSM part 3371742 (simplified to 0.25 m) and the ring corridor
  wall. Its real depth (≈6 m) is not modelled: Offlined and the viewers draw the ground at y = 0 over the whole
  footprint, so the pit is drawn as a dark floor at ground level (its shadowed depth seen from above) with the
  wall tops standing 0.5 m. The reconstructed wooden floor (2000) over the east end, 0.6 m high, cut where the
  OSM part 19666835 cuts it (a line at x ≈ −21 m).
- Around it: the travertine paving ring, 10 m wide outside the outer wall (the Roman one was 17.5 m wide, with
  boundary stones; it survives only in part), and the paved floor where the outer ambulatory stood on the south.
Colours: travertine (warm cream, cleaned 2011–2016), brick and tufa for the ruins, buttresses and hypogeum.

Axes (docs/MODEL_SPEC.md): +Y up, the intact north side (the minor axis, north gate) faces +Z = heading 19.7°,
+X points along the long axis to the west-north-west (289.7°). Origin at the centre of the ellipse, y = 0 on the
paving and the arena level. Walls go down to −4 m.

    python model_sources/Q10285_colosseum.py   -> models/Q10285/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, cap, triangulate, fbm, sstep, write_glb  # noqa: E402
from meshkit import _orient  # noqa: E402

rng = np.random.default_rng(80)

# ── Colours ──────────────────────────────────────────────────────────────────
TRAV = (0.86, 0.80, 0.68)          # travertine
TRAV_L = (0.90, 0.85, 0.74)        # cornices, bands
TRAV_D = (0.74, 0.68, 0.57)        # inner faces, weathered
BRICK = (0.70, 0.44, 0.31)         # Valadier / Stern buttresses
BRICK_D = (0.60, 0.40, 0.30)
RUIN = (0.69, 0.56, 0.43)          # brick and tufa of the cavea
TUFA = (0.72, 0.65, 0.52)
PAVE = (0.80, 0.77, 0.70)
EARTH = (0.36, 0.30, 0.25)         # hypogeum, seen from above: its depth in shadow
WOOD = (0.58, 0.45, 0.32)          # reconstructed arena floor

# ── Plan ─────────────────────────────────────────────────────────────────────
A, B = 94.0, 78.0                  # semi-axes of the outer face
NB = 80                            # bays
D_RING2, D_RING2_IN, D_AMB = 7.7, 10.0, 2.4
D_PRAEC, D_ARENA = 28.6, 51.0
FOUND = -4.0

_T = np.linspace(0, 2 * math.pi, 40001)
_sp = np.hypot(A * np.sin(_T), B * np.cos(_T))
_S = np.concatenate([[0], np.cumsum((_sp[1:] + _sp[:-1]) / 2 * np.diff(_T))])
PER = float(_S[-1])
BAY = PER / NB


def to_world(v):
    """Flat coordinates (s along the outer face, y, w = depth inwards) -> model x, y, z."""
    v = np.asarray(v, float)
    t = np.interp(np.mod(v[:, 0], PER), _S, _T)
    nx, nz = B * np.cos(t), A * np.sin(t)
    ln = np.hypot(nx, nz)
    nx, nz = nx / ln, nz / ln
    return np.stack([A * np.cos(t) - v[:, 2] * nx, v[:, 1], B * np.sin(t) - v[:, 2] * nz], 1)


def ring_xz(d, n=160):
    s = np.linspace(0, PER, n, endpoint=False)
    p = to_world(np.stack([s, np.zeros(n), np.full(n, d)], 1))
    return [(float(a), float(b)) for a, b in p[:, [0, 2]]]


def B3(s0, s1, y0, y1, w0, w1, c):
    """Box from its flat extents (s, y, w)."""
    return box(s1 - s0, y1 - y0, w1 - w0, c).T((s0 + s1) / 2, y0, (w0 + w1) / 2)


def prism(outer, w0, w1, c, holes=()):
    """A polygon in the (s, y) plane, with holes, extruded from depth w0 to w1."""
    pts, tris = triangulate(outer, holes)
    n = len(pts)
    V = [(p[0], p[1], w1) for p in pts] + [(p[0], p[1], w0) for p in pts]
    F = [tuple(t) for t in tris] + [(n + t[0], n + t[2], n + t[1]) for t in tris]
    ms = [Mesh(V, F, c)]
    for ring, ccw in [(outer, True)] + [(h, False) for h in holes]:
        R = _orient(ring, ccw)
        sv, sf = [], []
        for i in range(len(R)):
            (u0, y0), (u1, y1) = R[i], R[(i + 1) % len(R)]
            o = len(sv)
            sv += [(u0, y0, w0), (u1, y1, w0), (u1, y1, w1), (u0, y0, w1)]
            sf += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        ms.append(Mesh(sv, sf, c))
    return merge(ms)


def arch_curve(sc, ys, r, seg=8):
    return [(sc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)]


def jag(s0, s1, y_a, y_b, amp, n=3):
    """A broken top edge from (s1, y_b) back to (s0, y_a)."""
    pts = []
    for k in range(n, -1, -1):
        f = k / n
        y = y_a + (y_b - y_a) * f + (rng.uniform(-amp, amp) if 0 < k < n else 0)
        pts.append((s0 + (s1 - s0) * f, y))
    return pts


def arch_bay(sc, half, y0, y1, yb, ys, r, w0, w1, c, top_l=None, top_r=None, amp=0.0):
    """One bay of an arcaded wall: the wall from y0 to its top, an arch r wide (springing ys) opening from yb.
    top_l/top_r give a broken top (ruin) instead of y1. If the top falls below the crown, two piers remain."""
    s0, s1 = sc - half, sc + half
    tl = y1 if top_l is None else top_l
    tr = y1 if top_r is None else top_r
    crown = ys + r
    if min(tl, tr) > crown + 0.5:
        top = [(s1, tr)] + (jag(s0, s1, tl, tr, amp)[1:-1] if amp else []) + [(s0, tl)]
        if yb <= y0 + 0.01:
            outer = [(s0, y0), (sc - r, y0), (sc - r, ys)] + arch_curve(sc, ys, r)[::-1][1:-1] + \
                [(sc + r, ys), (sc + r, y0), (s1, y0)] + top
            return prism(outer, w0, w1, c)
        hole = [(sc - r, yb), (sc + r, yb), (sc + r, ys)] + arch_curve(sc, ys, r)[1:-1] + [(sc - r, ys)]
        return prism([(s0, y0), (s1, y0)] + top, w0, w1, c, holes=[hole])
    ms = []
    for a, b, ya, yb_ in ((s0, sc - r, tl, tl - rng.uniform(0, 1.5)), (sc + r, s1, tr - rng.uniform(0, 1.5), tr)):
        if max(ya, yb_) > y0 + 0.3:
            ms.append(prism([(a, y0), (b, y0), (b, max(yb_, y0 + 0.3)), (a, max(ya, y0 + 0.3))], w0, w1, c))
    return merge(ms)


# ── Outer wall ───────────────────────────────────────────────────────────────
# (y0, y1, opening bottom, springing, entablature bottom) per storey
STOREYS = [(0.0, 10.6, 0.0, 4.95, 8.9), (10.6, 22.4, 12.2, 16.55, 20.8), (22.4, 33.9, 24.0, 28.35, 32.3)]
R_ARCH = 2.1
ATTIC = (33.9, 48.5)
COL_R = {0: 0.48, 1: 0.44, 2: 0.42}


def outer_bay(k, colour=TRAV, columns=True, top=None, filled=False):
    """Bay k of the outer wall, with its orders; top: storeys kept (Valadier's steps)."""
    sc, half = k * BAY, BAY / 2
    ms, light, dark = [], [], []
    nst = 4 if top is None else top
    for i, (y0, y1, yb, ys, ye) in enumerate(STOREYS[:min(nst, 3)]):
        ms.append(arch_bay(sc, half, y0, y1, yb, ys, R_ARCH, 0.0, D_AMB, colour))
        if filled:                                            # Stern walled up the last arches
            dark.append(prism([(sc - R_ARCH, max(y0, yb)), (sc + R_ARCH, max(y0, yb))] +
                              arch_curve(sc, ys, R_ARCH), 0.6, 1.2, BRICK))
        if i:                                                 # pedestal course under the upper arches
            light.append(B3(sc - half, sc + half, y0, yb, -0.35, 0.0, TRAV_L if columns else colour))
        light.append(B3(sc - half, sc + half, ye, y1 - 0.45, -0.55, 0.0, TRAV_L if columns else colour))
        light.append(B3(sc - half, sc + half, y1 - 0.45, y1, -0.9, 0.0, TRAV_L if columns else colour))
        for e in (sc - half, sc + half):                      # half-columns (pilasters on the buttress)
            yc0 = 0.4 if i == 0 else yb
            if columns:
                light.append(cyl(COL_R[i], COL_R[i] * 0.9, ye - yc0, TRAV_L, seg=10).T(e, yc0, 0.0))
                light.append(B3(e - 0.6, e + 0.6, ye - 0.55, ye, -0.6, 0.0, TRAV_L))     # capital
                light.append(B3(e - 0.6, e + 0.6, yc0 - 0.4, yc0, -0.62, 0.0, TRAV_L))   # base / plinth
            else:
                light.append(B3(e - 0.45, e + 0.45, (0 if i == 0 else yb), ye, -0.25, 0.0, TRAV_D))
    if nst >= 4:
        y0, y1 = ATTIC
        win = (k % 2 == 0)
        holes = [[(sc - 0.65, 39.6), (sc + 0.65, 39.6), (sc + 0.65, 41.8), (sc - 0.65, 41.8)]] if win else []
        ms.append(prism([(sc - half, y0), (sc + half, y0), (sc + half, 45.5), (sc - half, 45.5)], 0.0, 1.9,
                        colour, holes=holes))
        light.append(B3(sc - half, sc + half, y0, 35.2, -0.4, 0.0, TRAV_L if columns else colour))
        for e in (sc - half, sc + half):
            light.append(B3(e - 0.5, e + 0.5, 35.2, 45.5, -0.22, 0.0, TRAV_L if columns else TRAV_D))
        for u in (-1.65, 0.0, 1.65):                          # corbels for the velarium masts
            light.append(B3(sc + u - 0.3, sc + u + 0.3, 43.4, 44.0, -0.55, 0.0, TRAV_L))
        light.append(B3(sc - half, sc + half, 45.5, 47.5, 0.0, 1.9, colour))
        light.append(B3(sc - half, sc + half, 47.5, 48.5, -0.7, 1.9, TRAV_L if columns else colour))
    return merge(ms + light), merge(dark)


def wall_down(s0, s1, w0, w1, c):
    return B3(s0, s1, FOUND, 0.0, w0, w1, c)


trav, brick, ruin, pave, arena, wood = [], [], [], [], [], []

K_W, K_E = 4, 36                                  # travertine arcade
for k in range(K_W, K_E + 1):
    m, d = outer_bay(k, filled=(k == K_E))
    trav.append(m); brick.append(d)
    trav.append(wall_down(k * BAY - BAY / 2, k * BAY + BAY / 2, -0.9, D_AMB, TRAV_D))

# Valadier's buttress (west end, 1820s): the arcade repeated in brick, stepping down
for k, nst in ((3, 4), (2, 3), (1, 2)):
    m, _ = outer_bay(k, colour=BRICK, columns=False, top=nst)
    brick.append(m)
    brick.append(wall_down(k * BAY - BAY / 2, k * BAY + BAY / 2, -0.5, D_AMB, BRICK_D))
s_end = 0.5 * BAY                                   # sloping end wall, from the outer face to the second ring
brick.append(prism([(s_end - 2.6, 0.0), (s_end, 0.0), (s_end, 22.4), (s_end - 2.6, 18.0)], 0.0, D_RING2_IN, BRICK))

# Stern's buttress (east end, 1807): a sloping brick spur against the walled-up end of the wall
s0, s1 = (K_E + 0.5) * BAY, 39.9 * BAY
STERN = [(s0, 48.5), (s0 + 2.0, 44.0), (s1 - 3.0, 12.0), (s1, 7.0)]      # the top of the spur (s, y)
cuts = sorted(set(np.round(np.r_[np.arange(s0, s1, 2.0), [p[0] for p in STERN], s1], 4)))
for a, b in zip(cuts[:-1], cuts[1:]):                  # in 2 m slices, so that it follows the curve
    ya, yb_ = (float(np.interp(x, [p[0] for p in STERN], [p[1] for p in STERN])) for x in (a, b))
    brick.append(prism([(a, FOUND), (b, FOUND), (b, yb_), (a, ya)], -0.4, D_RING2_IN, BRICK))

# ── North: ambulatory vaults and the second ring ─────────────────────────────
for k in range(1, K_E + 1):
    sc, half = k * BAY, BAY / 2
    nst = 4 if k >= 3 else (3 if k == 2 else 2)
    levels = [(9.8, 10.6), (21.6, 22.4), (33.1, 33.9), (37.2, 38.0)][:nst - (0 if nst == 4 else 1)]
    for y0, y1 in levels:
        ruin.append(B3(sc - half, sc + half, y0, y1, D_AMB, D_RING2, TUFA))
    top2 = 39.0 if nst == 4 else (34.5 if nst == 3 else 25.0)
    for i, (y0, y1, yb, ys, ye) in enumerate(STOREYS):
        if y0 >= top2:
            break
        trav.append(arch_bay(sc, half, y0, min(y1, top2), yb, ys, 1.9, D_RING2, D_RING2_IN, TRAV_D))
    if top2 > 33.9:
        trav.append(B3(sc - half, sc + half, 33.9, top2, D_RING2, D_RING2_IN, TRAV_D))
    trav.append(wall_down(sc - half, sc + half, D_RING2, D_RING2_IN, TRAV_D))

# ── South: the second ring as the outside, ruined, with the stubs of the radial walls ─────────
south = list(range(40, 80)) + [0]
tops = {}
for k in south:
    u = ((k - 40) % NB) / 40.0                                    # 0 at the east end … 1 at the west end
    base = 30.5 + 3.5 * (abs(u - 0.5) * 2) ** 2
    tops[k] = base + rng.uniform(-3.5, 2.5)
for k in south:
    sc, half = k * BAY, BAY / 2
    tl = (tops[k] + tops.get(k - 1, tops[k])) / 2 + rng.uniform(-0.8, 0.8)
    tr = (tops[k] + tops.get((k + 1) % NB, tops[k])) / 2 + rng.uniform(-0.8, 0.8)
    if k == 40:
        tl = 36.0
    for i, (y0, y1, yb, ys, ye) in enumerate(STOREYS):
        t_l, t_r = min(tl, y1), min(tr, y1)
        if min(tl, tr) <= y0 + 0.3:
            break
        last = tl < y1 + 0.5 or tr < y1 + 0.5
        trav.append(arch_bay(sc, half, y0, y1, yb, ys, 1.9, D_RING2, D_RING2_IN, TRAV,
                             top_l=t_l if last else None, top_r=t_r if last else None, amp=0.8 if last else 0))
        if not last:
            trav.append(B3(sc - half, sc + half, y1 - 0.5, y1, D_RING2 - 0.35, D_RING2, TRAV_L))
    if max(tl, tr) > 33.9:
        trav.append(prism([(sc - half, 33.9), (sc + half, 33.9), (sc + half, max(tr, 34.0)),
                           (sc - half, max(tl, 34.0))], D_RING2, D_RING2_IN, TRAV))
    trav.append(wall_down(sc - half, sc + half, D_RING2, D_RING2_IN, TRAV_D))
    # stub of the radial wall that joined this pier to the lost outer wall
    if k not in (0, 40):
        e = sc + half
        ln = rng.uniform(1.2, 4.0)
        h_in = rng.uniform(3.0, min(16.0, min(tl, tr) - 6.0))
        h_out = max(0.8, h_in - rng.uniform(2.0, 7.0))
        pts = [(D_RING2 - ln, 0.0), (D_RING2, 0.0), (D_RING2, h_in), (D_RING2 - ln, h_out)]
        st = prism(pts, e - 0.75, e + 0.75, TUFA)
        st.v = st.v[:, [2, 1, 0]]; st.f = st.f[:, ::-1]           # built in (w, y), along s
        ruin.append(st)
        ruin.append(B3(e - 0.8, e + 0.8, 0.0, 0.45, 0.4, 1.8, TRAV_D))   # base of the lost outer pier

# ── Cavea: the ribbed ruin of the radial walls and vaults ────────────────────
NORTH_PROF = [(10.0, 33.5), (17.5, 31.0), (28.45, 21.0), (28.75, 18.0), (40.0, 10.5), (50.6, 4.5)]
SOUTH_PROF = [(10.0, 25.0), (17.5, 22.5), (28.45, 15.5), (28.75, 12.5), (40.0, 8.5), (50.6, 4.5)]


def prof(d, P):
    return np.interp(d, [p[0] for p in P], [p[1] for p in P])


s_list = []
for k in range(NB):
    p = (k + 0.5) * BAY
    s_list += [p - 0.6, p + 0.6, p + 1.0, (k + 1) * BAY, p + BAY - 1.0]
for c_ax in (0.0, PER / 2):                              # axial passages to the arena
    s_list += [c_ax - 2.45, c_ax - 2.2, c_ax + 2.2, c_ax + 2.45]
s_arr = np.unique(np.mod(np.round(np.array(s_list), 4), PER))
d_arr = np.array([10.0, 11.2, 12.6, 14.0, 15.5, 17.5, 19.3, 21.1, 22.9, 24.7, 26.5, 28.45, 28.75, 30.5, 32.5,
                  34.5, 36.5, 38.5, 40.5, 42.5, 44.5, 46.5, 48.5, 50.6, 51.0])
S_, D_ = np.meshgrid(s_arr, d_arr)                      # rows = d, cols = s
kpos = S_ / BAY
nw = sstep(0.0, 2.0, kpos) * (1 - sstep(38.0, 40.5, kpos))   # 1 on the north (bays 1–39)
Hn, Hs = prof(D_, NORTH_PROF), prof(D_, SOUTH_PROF)
H = Hs + (Hn - Hs) * nw
ph = np.mod(S_ - 0.5 * BAY, BAY)                        # distance after the pier line
rib = (ph <= 0.61) | (ph >= BAY - 0.61)
pts3 = np.stack([S_.ravel(), H.ravel(), D_.ravel()], 1)
nz = fbm(pts3 * np.array([1, 0, 1]), (0.09, 1, 0.3), 5, 3).reshape(H.shape)
nz2 = fbm(pts3 * np.array([1, 0, 1]), (0.03, 1, 0.08), 9, 2).reshape(H.shape)
kk = np.round(S_ / BAY).astype(int) % NB              # bay of each sample: every bay is ruined its own way
VD = rng.uniform(0.25, 1.4, NB)                         # how much of its vaults fell
RB = rng.uniform(-2.0, 0.8, NB)                         # how much of its radial wall is left
drop = np.where(rib, -RB[kk], (1.2 + 3.2 * nz) * VD[kk]) * (1.0 - 0.45 * nw)
H = H - drop - (nz2 - 0.5) * 6.0 * (1 - 0.6 * nw) * sstep(10.5, 14, D_) * sstep(50.6, 47, D_)
H[-1, :] = H[-2, :]
for c_ax in (0.0, PER / 2):                              # the passages
    dd = np.abs(((S_ - c_ax + PER / 2) % PER) - PER / 2)
    H = np.where((dd < 2.3) & (D_ > 10.2), 0.3, H)
H[-2:, :] = np.maximum(H[-2:, :], 0.3)
H = np.maximum(H, 0.3)
ns, nd = len(s_arr), len(d_arr)
V = np.stack([S_.ravel(), H.ravel(), D_.ravel()], 1)
F = []
for j in range(nd - 1):
    for i in range(ns):
        i2 = (i + 1) % ns
        a, b, c_, d_ = j * ns + i, j * ns + i2, (j + 1) * ns + i2, (j + 1) * ns + i
        F += [(a, c_, b), (a, d_, c_)]
col = np.array(RUIN) * (0.9 + 0.2 * nz.ravel()[:, None])
col = np.where(rib.ravel()[:, None], np.array(TUFA) * 1.03, col)
mix = sstep(0.55, 0.75, nz2.ravel())[:, None] * 0.5
col = col * (1 - mix) + np.array(BRICK) * mix
cavea = Mesh(V, F, col)
# podium: the wall of the arena, from the hypogeum floor up to the cavea
pod_v, pod_f = [], []
for i in range(ns):
    i2 = (i + 1) % ns
    for (si, hi) in ((s_arr[i], H[-1, i]), (s_arr[i2], H[-1, i2])):
        pod_v += [(si, 0.0, D_ARENA), (si, hi, D_ARENA)]
    o = len(pod_v) - 4
    pod_f += [(o, o + 1, o + 3), (o, o + 3, o + 2)]
podium = Mesh(pod_v, pod_f, BRICK_D)

# ── Paving ───────────────────────────────────────────────────────────────────
PAVE_OUT = -10.0
Y_PAVE = 0.15                                          # just above the levelled ground
for k in range(NB):
    s_a, s_b = k * BAY - BAY / 2, k * BAY + BAY / 2
    w_in = D_RING2 if 1 <= k <= 39 else D_RING2_IN
    pave.append(B3(s_a, s_b, -0.5, Y_PAVE, PAVE_OUT, w_in, PAVE))
    pave.append(B3(s_a, s_b, FOUND, -0.5, PAVE_OUT, PAVE_OUT + 0.4, PAVE))
pave_m = merge(pave)
pave_m.paint(lambda v: np.array(PAVE) * (0.94 + 0.1 * fbm(v * np.array([1, 0, 1]), 0.15, 3, 2))[:, None])

# ── Arena: hypogeum and the reconstructed floor (model x/z, from OSM) ────────
HYPO = json.loads(
    '{"outer":[[[-6.8,17.9],[-1.8,18.0],[2.8,17.5],[8.8,16.8],[13.8,15.7],[21.1,12.7],[24.7,10.2],[24.1,9.7],[19.3,12.6],'
    '[15.7,14.1],[11.5,15.2],[2.2,16.6],[-6.9,17.2]],[[-7.1,18.8],[0.9,19.1],[9.7,18.4],[13.7,17.7],[20.8,15.1],[23.5,13.7],'
    '[27.6,10.3],[28.3,10.7],[23.9,14.2],[20.9,15.8],[14.2,18.4],[3.5,20.1],[-1.4,20.3],[-6.5,20.2],[-7.3,19.9]],[[27.4,8.2],'
    '[28.7,7.0],[29.7,5.1],[28.8,4.9],[28.1,6.3],[26.8,6.2],[26.7,7.9],[27.0,8.5]],[[33.3,-1.1],[32.0,-4.4],[28.4,-8.0],'
    '[24.8,-10.7],[18.6,-14.3],[10.1,-16.9],[6.5,-17.6],[-0.5,-18.0],[-6.0,-17.7],[-13.0,-16.7],[-20.7,-14.4],[-20.8,-13.6],'
    '[-10.7,-16.4],[-2.0,-17.2],[4.5,-17.1],[11.6,-15.8],[17.6,-13.9],[22.5,-11.2],[29.9,-5.1],[31.9,-2.4],[32.4,-0.4],'
    '[36.1,-0.5],[36.0,-1.2]],[[2.4,-4.4],[2.4,-2.2],[0.9,-2.2],[0.9,-4.6],[-1.1,-4.8],[-7.5,-4.7],[-7.7,-4.2],[-8.7,-4.2],'
    '[-8.7,-4.9],[-9.6,-4.9],[-9.7,-2.6],[-11.2,-2.6],[-11.1,-4.9],[-11.7,-5.0],[-11.8,-2.6],[-13.3,-2.7],[-13.2,-5.0],'
    '[-13.9,-5.0],[-14.0,-2.7],[-15.5,-2.8],[-15.4,-5.1],[-16.1,-5.1],[-16.1,-2.8],[-17.7,-2.8],[-17.6,-5.1],[-19.4,-5.2],'
    '[-19.4,-5.8],[-17.9,-5.9],[28.5,-4.7],[30.1,-0.9],[25.9,-0.6],[25.8,-1.5],[28.6,-1.7],[27.6,-4.0],[21.9,-4.0],'
    '[21.8,-1.6],[20.4,-1.6],[20.4,-4.1],[19.7,-4.1],[19.7,-1.6],[18.2,-1.7],[18.2,-4.1],[17.6,-4.1],[17.5,-1.7],[16.0,-1.8],'
    '[16.1,-4.2],[15.4,-4.2],[15.3,-1.8],[13.8,-1.8],[13.9,-4.2],[13.2,-4.2],[13.1,-1.9],[11.6,-1.9],[11.7,-4.2],[10.9,-4.2],'
    '[10.9,-1.9],[9.4,-2.0],[9.5,-4.3],[8.8,-4.3],[8.7,-2.0],[7.2,-2.0],[7.3,-4.3],[6.6,-4.3],[6.5,-2.0],[5.0,-2.1],[5.1,-4.3]],'
    '[[16.0,-10.3],[15.9,-8.9],[19.8,-8.9],[20.2,-9.6],[23.2,-9.1],[23.7,-8.7],[22.5,-8.0],[-15.3,-8.9],[-16.6,-8.6],'
    '[-18.3,-8.7],[-18.4,-9.3],[-16.6,-9.2],[-16.5,-9.7],[-15.4,-9.8],[-15.3,-10.3],[-14.8,-9.8],[-13.8,-9.8],[-13.8,-11.2],'
    '[-15.0,-11.3],[-14.9,-10.6],[-15.5,-10.8],[-16.5,-10.7],[-16.7,-11.2],[-20.9,-11.2],[-20.9,-11.8],[-19.4,-12.6],'
    '[-13.1,-14.7],[-5.6,-15.6],[2.1,-15.6],[7.5,-14.9],[12.1,-13.6],[14.8,-12.6],[15.0,-12.3],[14.4,-11.2],[16.2,-11.6],'
    '[16.6,-11.2],[19.5,-11.0],[21.8,-9.9],[22.4,-10.1],[22.5,-9.6],[22.0,-9.3],[20.1,-9.8],[19.4,-10.4]],[[-8.4,-19.1],'
    '[-8.5,-19.9],[-4.0,-20.5],[0.4,-20.6],[5.8,-20.2],[9.9,-19.3],[10.4,-19.8],[11.4,-19.4],[11.2,-18.7],[10.4,-18.8],'
    '[9.9,-18.2],[4.6,-19.4],[-0.1,-19.8]],[[-11.0,-18.8],[-11.1,-19.5],[-9.9,-19.7],[-9.8,-19.0]],[[-13.6,-18.4],'
    '[-13.7,-19.1],[-12.4,-19.3],[-12.3,-18.6]],[[-16.0,-17.9],[-16.1,-18.7],[-14.9,-18.9],[-14.8,-18.1]],[[-18.5,-17.4],'
    '[-18.7,-18.2],[-17.5,-18.5],[-17.3,-17.7]],[[15.8,5.6],[15.9,3.4],[17.3,3.5],[17.3,5.7],[17.9,5.7],[18.0,3.5],[19.5,3.5],'
    '[19.4,5.7],[20.2,5.8],[20.3,3.5],[21.7,3.6],[21.7,5.8],[22.3,5.8],[22.1,6.9],[2.2,5.5],[-15.3,5.5],[-18.3,5.9],[-18.4,5.2],'
    '[-17.3,4.8],[-17.3,2.5],[-16.5,2.5],[-16.6,4.7],[-16.0,4.7],[-16.0,2.5],[-14.5,2.6],[-14.5,4.8],[-13.9,4.8],[-13.8,2.6],'
    '[-12.3,2.6],[-12.4,4.8],[-11.7,4.8],[-11.6,2.7],[-10.2,2.7],[-10.2,4.9],[-9.6,4.9],[-9.5,2.7],[-8.0,2.8],[-8.1,4.9],'
    '[-7.4,4.9],[-7.3,2.8],[-5.8,2.8],[-5.9,5.0],[-5.3,5.0],[-5.2,2.8],[-3.7,2.9],[-3.7,5.0],[-3.0,5.0],[2.5,4.9],[3.1,4.2],'
    '[4.2,4.1],[4.1,5.1],[4.8,5.2],[4.8,3.1],[6.3,3.2],[6.2,5.2],[7.0,5.2],[7.0,3.2],[8.5,3.2],[8.5,5.3],[9.2,5.3],[9.3,3.2],'
    '[10.8,3.3],[10.7,5.4],[11.4,5.4],[11.5,3.3],[13.0,3.3],[12.9,5.5],[13.6,5.5],[13.7,3.4],[15.2,3.4],[15.1,5.6]],'
    '[[15.4,10.7],[15.3,12.0],[18.6,11.7],[17.2,12.2],[9.9,13.1],[3.3,13.3],[-18.0,12.4],[-18.1,11.9],[-12.0,11.6],[1.2,12.2],'
    '[1.2,10.7],[-18.5,10.2],[-18.2,9.6],[-14.9,9.4],[-1.2,10.2],[3.9,9.9],[14.3,10.1],[17.1,10.3],[17.3,10.9]]],'
    '"inner":[[[21.4,-3.6],[21.4,-2.1],[20.8,-2.1],[20.8,-3.6]],[[19.2,-3.6],[19.2,-2.1],[18.6,-2.2],[18.6,-3.6]],'
    '[[17.1,-3.7],[17.1,-2.2],[16.5,-2.2],[16.5,-3.7]],[[14.9,-3.7],[14.9,-2.2],[14.3,-2.2],[14.3,-3.7]],[[12.7,-3.8],'
    '[12.7,-2.4],[12.0,-2.4],[12.1,-3.9]],[[10.5,-3.9],[10.4,-2.4],[9.8,-2.4],[9.9,-3.9]],[[8.3,-3.9],[8.3,-2.4],[7.7,-2.4],'
    '[7.7,-3.9]],[[6.1,-4.0],[6.1,-2.5],[5.5,-2.5],[5.5,-4.0]],[[1.9,-4.3],[1.9,-2.8],[1.3,-2.8],[1.3,-4.3]],[[-10.1,-4.6],'
    '[-10.1,-3.1],[-10.7,-3.1],[-10.7,-4.6]],[[-12.2,-4.5],[-12.2,-3.0],[-12.8,-3.0],[-12.8,-4.5]],[[-14.4,-4.6],[-14.4,-3.1],'
    '[-15.0,-3.1],[-15.0,-4.6]],[[-16.5,-4.6],[-16.6,-3.1],[-17.2,-3.1],[-17.1,-4.6]],[[12.7,-11.3],[10.7,-11.3],[10.7,-11.6],'
    '[10.0,-11.7],[10.0,-11.3],[8.5,-11.5],[8.5,-11.9],[7.8,-12.0],[7.7,-11.5],[1.1,-11.7],[1.1,-12.2],[0.2,-12.2],[0.2,-11.6],'
    '[-1.8,-11.7],[-1.9,-12.2],[-2.7,-12.2],[-2.8,-11.8],[-14.0,-11.9],[-14.2,-12.3],[-16.6,-11.9],[-19.0,-11.9],[-18.9,-12.3],'
    '[-16.6,-12.8],[-12.7,-14.2],[-9.8,-14.6],[-3.4,-15.1],[2.5,-15.0],[7.2,-14.4],[12.5,-12.9],[12.9,-12.5]],[[-12.2,-10.8],'
    '[-12.2,-9.9],[-13.4,-10.0],[-13.3,-10.8]],[[-11.8,-11.1],[-11.8,-9.8],[-10.8,-9.7],[-10.7,-11.1]],[[-9.2,-10.7],'
    '[-9.2,-9.8],[-10.3,-9.8],[-10.3,-10.7]],[[-7.7,-11.0],[-8.8,-11.1],[-8.8,-9.7],[-7.8,-9.6]],[[-6.2,-10.6],[-6.2,-9.7],'
    '[-7.3,-9.8],[-7.3,-10.6]],[[1.8,-10.8],[-5.7,-11.0],[-5.8,-9.6],[-5.0,-9.5],[-4.8,-9.9],[-3.0,-10.0],[-2.9,-9.5],'
    '[1.8,-9.4]],[[3.4,-10.5],[3.3,-9.6],[2.2,-9.6],[2.2,-10.5]],[[4.9,-10.8],[3.8,-10.8],[3.8,-9.4],[4.8,-9.3]],[[6.4,-10.4],'
    '[6.4,-9.5],[5.3,-9.6],[5.3,-10.4]],[[7.9,-10.7],[6.8,-10.7],[6.8,-9.3],[7.9,-9.2]],[[9.5,-10.3],[9.4,-9.4],[8.3,-9.5],'
    '[8.3,-10.3]],[[9.9,-10.6],[9.8,-9.2],[10.9,-9.1],[11.0,-10.6]],[[12.6,-10.1],[12.5,-9.2],[11.4,-9.3],[11.4,-10.2]],'
    '[[13.0,-10.5],[12.9,-9.0],[13.9,-8.9],[14.0,-10.4]],[[15.5,-9.9],[15.5,-9.1],[14.4,-9.1],[14.4,-10.0]],[[-14.9,3.0],'
    '[-15.0,4.5],[-15.6,4.5],[-15.6,3.0]],[[-12.8,3.0],[-12.8,4.5],[-13.4,4.5],[-13.4,3.0]],[[-10.6,3.0],[-10.7,4.4],'
    '[-11.3,4.4],[-11.2,2.9]],[[-8.5,3.1],[-8.5,4.6],[-9.1,4.5],[-9.1,3.1]],[[-6.3,3.2],[-6.3,4.7],[-6.9,4.7],[-6.9,3.2]],'
    '[[-4.1,3.8],[-4.1,4.7],[-4.9,4.6],[-4.9,3.8]],[[5.8,3.4],[5.8,4.9],[5.2,4.9],[5.2,3.4]],[[8.1,3.5],[8.0,4.9],[7.4,4.9],'
    '[7.5,3.4]],[[10.3,3.6],[10.3,5.0],[9.7,5.0],[9.7,3.6]],[[12.5,3.8],[12.5,5.2],[11.9,5.2],[11.9,3.7]],[[14.7,3.8],'
    '[14.7,5.2],[14.1,5.2],[14.1,3.7]],[[16.9,3.8],[16.9,5.3],[16.3,5.2],[16.3,3.8]],[[19.0,4.0],[19.0,5.4],[18.4,5.4],'
    '[18.4,3.9]],[[21.3,4.1],[21.2,5.6],[20.6,5.6],[20.7,4.1]],[[14.9,11.0],[14.9,11.9],[13.8,11.8],[13.8,11.0]],'
    '[[13.3,12.1],[13.4,10.9],[12.4,11.0],[12.4,12.1]],[[12.0,11.1],[11.9,12.0],[10.8,11.9],[10.8,11.1]],[[10.4,11.0],'
    '[9.4,10.9],[9.3,12.2],[10.4,12.2]],[[9.0,11.1],[8.9,11.9],[7.8,11.9],[7.8,11.0]],[[6.3,10.7],[6.2,12.2],[7.3,12.3],'
    '[7.4,10.8]],[[5.8,11.1],[5.8,11.9],[4.7,11.9],[4.7,11.1]],[[3.2,10.6],[3.2,12.2],[4.2,12.2],[4.3,10.7]],[[2.8,11.1],'
    '[2.7,12.0],[1.6,11.9],[1.6,11.1]]]}')


def inside(p, P):
    x, z = p; c = False
    for i in range(len(P)):
        (x0, z0), (x1, z1) = P[i - 1], P[i]
        if (z0 > z) != (z1 > z) and x < x0 + (z - z0) * (x1 - x0) / (z1 - z0):
            c = not c
    return c


Y_HF, Y_HW = 0.08, 0.5                                  # hypogeum: dark floor at ground level, top of its walls
ARENA = ring_xz(D_ARENA, 160)
arena.append(cap(ARENA, Y_HF, EARTH))
for o in HYPO["outer"]:
    hs = [h for h in HYPO["inner"] if inside(h[0], o)]
    pts, tris = triangulate(o, hs)
    top = Mesh(np.stack([pts[:, 0], np.full(len(pts), Y_HW), pts[:, 1]], 1), tris[:, ::-1], RUIN)
    sides = []
    for ring, ccw in [(o, True)] + [(h, False) for h in hs]:
        R = _orient(ring, ccw)
        for i in range(len(R)):
            (x0, z0), (x1, z1) = R[i], R[(i + 1) % len(R)]
            sides.append(Mesh([(x0, Y_HF, z0), (x0, Y_HW, z0), (x1, Y_HW, z1), (x1, Y_HF, z1)],
                              [(0, 1, 2), (0, 2, 3)], BRICK_D))
    arena.append(merge([top] + sides))
# the ring corridor wall of the hypogeum, 2.5 m inside the podium
for i in range(0, 160, 2):
    a, b = ARENA[i], ARENA[(i + 2) % 160]
    pa = np.array(a) * (1 - 2.5 / max(np.hypot(*a), 1)); pb = np.array(b) * (1 - 2.5 / max(np.hypot(*b), 1))
    if (pa[0] + pb[0]) / 2 < -21.0:
        continue
    arena.append(Mesh([(pa[0], Y_HF, pa[1]), (pa[0], Y_HW, pa[1]), (pb[0], Y_HW, pb[1]), (pb[0], Y_HF, pb[1])],
                      [(0, 1, 2), (0, 2, 3)], BRICK_D))


def clip_half(P, a, b):
    """Keep the part of polygon P on the left of the line a -> b."""
    def side(p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    out = []
    for i in range(len(P)):
        p, q = P[i - 1], P[i]
        sp, sq = side(p), side(q)
        if sq >= 0:
            if sp < 0:
                t = sp / (sp - sq); out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            out.append(q)
        elif sp >= 0:
            t = sp / (sp - sq); out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


Y_FL = 0.6                                             # top of the reconstructed deck
FLOOR = clip_half(ARENA, (-20.52, -22.26), (-22.03, 21.51))   # OSM 19666835: the cut of the floor
wood.append(cap(FLOOR, Y_FL, WOOD))
wood.append(cap(FLOOR, Y_FL - 0.45, WOOD, up=False))
for i in range(len(FLOOR)):
    (x0, z0), (x1, z1) = FLOOR[i - 1], FLOOR[i]
    wood.append(Mesh([(x0, Y_FL - 0.45, z0), (x0, Y_FL, z0), (x1, Y_FL, z1), (x1, Y_FL - 0.45, z1)], [(0, 1, 2), (0, 2, 3)], WOOD))

# ── Colour, assemble, write ──────────────────────────────────────────────────


def flat_part(ms):
    m = merge(ms)
    m.v = to_world(m.v)
    return m


def weather(m, k=0.10, seed=1):
    n = fbm(m.v, (0.05, 0.12, 0.05), seed, 3)
    low = sstep(6.0, -1.0, m.v[:, 1]) * 0.08
    m.c = np.clip(m.c * (1 - k / 2 + k * n)[:, None] * (1 - low)[:, None], 0, 1)
    return m


trav_m = weather(flat_part(trav), 0.12, 3)
brick_m = weather(flat_part(brick), 0.10, 5)
ruin_m = weather(flat_part(ruin + [cavea, podium]), 0.10, 7)
pave_m.v = to_world(pave_m.v)
arena_m = weather(merge(arena), 0.12, 11)
wood_m = merge(wood)

FOOTPRINT = [[round(x, 1), round(z, 1)] for x, z in ring_xz(PAVE_OUT - 1.0, 64)]

if __name__ == "__main__":
    out = ROOT / "models" / "Q10285"
    out.mkdir(parents=True, exist_ok=True)
    parts = [("travertine", trav_m), ("brick_buttresses", brick_m), ("cavea_ruins", ruin_m), ("paving", pave_m),
             ("hypogeum", arena_m), ("arena_floor", wood_m)]
    tris, size = write_glb(out / "model.glb", parts)
    allv = np.vstack([p[1].v for p in parts])
    print(f"{tris} triangles, {size / 1e6:.2f} MB, x {allv[:, 0].min():.1f}..{allv[:, 0].max():.1f}, "
          f"y {allv[:, 1].min():.1f}..{allv[:, 1].max():.1f}, z {allv[:, 2].min():.1f}..{allv[:, 2].max():.1f}")
    print("footprint", json.dumps(FOOTPRINT))
