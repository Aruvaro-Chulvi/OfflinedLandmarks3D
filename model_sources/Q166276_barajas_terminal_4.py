"""
Madrid-Barajas Airport, Terminal 4 (T4) — the main building (processor + pier), built by code.

Wikidata: the T4 building has its own item (Q63229912, «Aeropuerto de Madrid-Barajas T4»), but it has no
Wikipedia article in any language (only a Commons category), so the model is filed under the airport,
Q166276 (Aeropuerto Adolfo Suárez Madrid-Barajas), whose articles describe the T4.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Richard Rogers Partnership (now Rogers Stirk Harbour + Partners; Richard Rogers 1933–2021) and
Estudio Lamela (Antonio Lamela 1926–2017, Carlos Lamela), competition 1997, opened 5 February 2006
(Stirling Prize 2006) -> protected.

What is modelled (what identifies it from the air):
- The roof: one continuous field of 9 m wide steel «wings» clad in aluminium, laid side by side along the
  whole building. Across the building they run in bays of ~72 m (RSHP: grid 18 × 9 m, 72 m long beams); each
  wing rises from the light lines to a crown and, on the outer façades, ends in a rounded tip that gives the
  saw-toothed eaves. Underneath, the bamboo-strip soffit (laminated Chinese bamboo) hangs down in a pillow.
- The light: the big oval roof lights over the three «canyons» of the processor building (one per wing) and
  the two small round roof lights on every wing (counted on the PNOA orthophoto).
- The Y-shaped steel props on the outer façades, holding the deep overhangs, painted in the rainbow
  gradient of the building: blue at the north end, yellow in the middle, red at the south end (Aena).
- The recessed glass façades under the overhangs, a concrete plinth, and the fixed glass link-bridges
  («fingers», the T-shaped stubs of the OSM outline) to the aircraft stands, raised on legs.
- Not modelled: the movable jet bridges, the aircraft, the apron, the roads, the car park (its roof is a flat
  planted deck, not wavy), the low annexes at both ends of the processor (not in the OSM outline), the control
  tower, and the satellite T4S (2.5 km away, a separate building with no Wikidata item with articles).

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap way 8031397 (aeroway=terminal, building=transportation, «Terminal 4»), area 158,296 m²:
  the pier (dique) 67.1 m wide (z −94.6 … −27.5) and 1,168 m long (x −578 … +590), the processor building
  (x −178 … +194, west to z +118) and the forecourt canopy bay (x −123 … +142, to z +188.7). The building is
  aligned with true north (the pier sides are constant in east–west to a few cm), so heading 270.
- Bays and light lines measured on the PNOA orthophoto (IGN, 0.25 m/px): roof-light rows at z = +117.5,
  +45.5 and −26 (every ~72 m), wings 9 m wide, small roof lights at 37 % and 63 % of each bay, ~3 m across;
  oval roof lights ~7.5 × 9 m.
- Heights: not published; estimated from photographs against A320 tails (11.8 m): eaves ~16.5 m on the pier,
  crowns ~20.5 m (pier) and ~22.5 m (processor), light lines at ~18 m (±2 m).
- Ground: EU-DEM 25 m gives ~613–617 m on the apron east of the pier and ~620–622 m on the landside to the
  west, a fall of 5–8 m: "ground": "keep" with y = 0 at the apron by the east façade; the whole footprint is
  a flat platform (a paved slab under the overhangs and the bridges, so the terrain is not raised to them).

Axes (docs/MODEL_SPEC.md): +Y up, +Z faces west (heading 270°, the landside / forecourt façade), +X points
south. Origin at the centroid of the OSM outline (40.4913798, −3.5915745), y = 0 at apron level.

    python model_sources/Q166276_barajas_terminal_4.py   -> models/Q166276/model.glb
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cap, walls, offset, tube, write_glb, GLASS  # noqa: E402

# ── OSM way 8031397, heading 270 (x = south, z = west), origin at its centroid ──────────────────────────
OUTLINE = [(141.9, 187.7), (140.0, 119.2), (194.5, 119.2), (192.9, -27.8), (274.8, -27.5), (275.1, -8.6), (268.0, -8.6), (267.9, -5.0), (286.8, -5.1), (286.8, -8.6), (279.3, -8.6), (279.3, -27.5), (304.9, -27.5), (305.2, -8.8), (297.9, -8.8), (297.9, -5.1), (316.6, -5.2), (316.6, -8.8), (309.3, -8.9), (309.1, -27.5), (350.1, -27.5), (350.2, -9.0), (342.8, -9.1), (342.8, -5.5), (361.5, -5.4), (361.5, -9.0), (354.4, -9.0), (354.2, -27.5), (392.0, -27.5), (392.2, -9.2), (384.9, -9.1), (385.0, -5.4), (403.6, -5.6), (403.6, -9.2), (396.3, -9.2), (396.4, -27.5), (439.8, -27.5), (440.0, -9.2), (432.8, -9.2), (432.9, -5.8), (451.4, -5.9), (451.4, -9.3), (444.4, -9.2), (444.2, -27.5), (481.8, -27.5), (482.1, -9.6), (474.8, -9.5), (474.8, -5.9), (493.8, -6.0), (493.6, -9.7), (486.4, -9.7), (486.4, -27.5), (526.9, -27.5), (527.0, -9.7), (519.9, -9.6), (519.8, -6.1), (538.8, -6.3), (538.8, -9.7), (531.2, -9.8), (531.2, -27.5), (568.8, -27.5), (569.0, -9.8), (561.9, -9.8), (561.8, -6.2), (580.5, -6.3), (580.5, -9.9), (573.6, -9.8), (573.3, -27.5), (589.7, -27.5), (589.7, -94.6), (534.0, -94.6), (534.0, -115.4), (541.2, -115.3), (541.2, -118.8), (522.4, -118.8), (522.4, -115.6), (529.6, -115.5), (529.4, -94.6), (417.0, -94.6), (417.0, -115.1), (424.1, -115.1), (424.0, -118.7), (405.4, -118.7), (405.3, -115.1), (412.2, -115.2), (412.2, -94.6), (360.3, -94.6), (360.1, -114.6), (367.5, -114.8), (367.4, -118.6), (348.4, -118.6), (348.4, -114.7), (355.5, -114.6), (355.6, -94.6), (314.9, -94.6), (314.9, -114.7), (322.3, -114.7), (322.3, -118.5), (303.4, -118.5), (303.4, -114.7), (310.5, -114.7), (310.5, -94.6), (234.2, -94.6), (234.0, -114.3), (241.2, -114.4), (241.3, -117.9), (222.4, -117.7), (222.4, -114.3), (229.6, -114.3), (229.6, -94.6), (176.9, -94.6), (177.1, -114.0), (184.2, -114.1), (184.3, -117.7), (165.5, -117.5), (165.5, -113.9), (172.5, -113.9), (172.7, -94.6), (135.3, -94.6), (134.9, -113.9), (142.4, -114.0), (142.5, -117.4), (123.5, -117.2), (123.5, -113.8), (130.6, -113.8), (130.7, -94.6), (90.1, -94.6), (90.1, -113.7), (97.6, -113.8), (97.5, -117.4), (78.4, -117.2), (78.5, -113.7), (85.6, -113.6), (85.7, -94.6), (48.1, -94.6), (48.0, -113.3), (50.2, -113.5), (55.0, -113.4), (55.0, -117.0), (36.3, -116.9), (36.4, -113.3), (43.5, -113.3), (43.5, -94.6), (-38.8, -94.6), (-39.0, -112.8), (-31.7, -112.9), (-31.7, -116.3), (-50.6, -116.2), (-50.6, -112.9), (-43.3, -112.9), (-43.3, -94.6), (-75.0, -94.6), (-75.1, -112.8), (-67.6, -112.8), (-67.6, -116.2), (-86.6, -116.1), (-86.4, -112.7), (-79.4, -112.8), (-79.4, -94.6), (-116.7, -94.6), (-117.1, -112.6), (-109.9, -112.7), (-109.9, -116.1), (-128.7, -116.1), (-128.7, -112.8), (-121.4, -112.7), (-121.4, -94.6), (-161.8, -94.6), (-162.1, -112.3), (-154.7, -112.5), (-154.7, -116.0), (-173.7, -116.0), (-173.7, -112.4), (-166.4, -112.3), (-166.3, -94.6), (-204.0, -94.6), (-204.0, -112.1), (-196.7, -112.3), (-196.6, -115.9), (-215.6, -115.9), (-215.5, -112.2), (-208.3, -112.2), (-208.5, -94.6), (-276.0, -94.6), (-276.1, -111.9), (-268.7, -111.9), (-268.7, -115.7), (-287.7, -115.7), (-287.7, -111.9), (-280.4, -111.9), (-280.4, -94.6), (-312.1, -94.6), (-312.2, -111.7), (-304.7, -111.8), (-304.8, -115.2), (-323.6, -115.2), (-323.6, -111.7), (-316.3, -111.7), (-316.3, -94.6), (-363.1, -94.6), (-363.1, -111.7), (-355.8, -111.7), (-355.7, -115.0), (-374.5, -114.7), (-374.4, -111.5), (-367.3, -111.6), (-367.1, -94.6), (-423.0, -94.7), (-423.1, -111.3), (-415.6, -111.4), (-415.6, -114.8), (-434.5, -114.7), (-434.5, -111.4), (-427.5, -111.3), (-427.4, -94.7), (-489.2, -94.7), (-488.9, -110.9), (-481.9, -111.0), (-481.9, -114.5), (-500.6, -114.4), (-500.7, -111.1), (-493.4, -111.0), (-493.4, -94.7), (-528.1, -94.7), (-528.1, -110.9), (-520.4, -110.8), (-520.4, -114.6), (-539.6, -114.4), (-539.5, -110.8), (-532.4, -110.8), (-532.3, -94.7), (-578.1, -94.7), (-578.1, -84.1), (-591.7, -84.3), (-592.4, -188.2), (-599.8, -188.2), (-599.4, -76.5), (-593.6, -76.7), (-593.7, -80.2), (-578.1, -80.0), (-578.1, -35.1), (-593.5, -35.1), (-593.5, -39.1), (-599.3, -39.2), (-599.2, 12.0), (-591.3, 11.9), (-591.6, -31.1), (-578.1, -30.9), (-578.1, -23.0), (-270.3, -23.0), (-178.4, -24.3), (-178.2, 117.8), (-123.4, 117.5), (-122.9, 188.7)]

# the building without the bridges (the roof outline)
MAIN = [(-578.1, -94.7), (589.7, -94.6), (589.7, -27.5), (193.7, -27.6), (194.5, 119.2), (140.0, 119.2),
        (141.9, 187.7), (-122.9, 188.7), (-123.4, 117.5), (-178.2, 117.8), (-178.4, -24.3), (-578.1, -23.0)]
# the glazed building under the roof: the forecourt bay (z > 117.8) is an open canopy over the drop-off road
BODY = [(-578.1, -94.7), (589.7, -94.6), (589.7, -27.5), (193.7, -27.6), (194.5, 125.8),
        (-178.2, 125.8), (-178.4, -24.3), (-578.1, -23.0)]
X_N, X_S = -578.1, 589.7                  # north and south ends of the pier
Z_E = -94.6                               # east face of the pier (airside)
Z_C0, Z_C1, Z_C2 = -26.0, 45.5, 117.8     # the three light lines (canyons) of the processor
Z_W = 188.2                               # west edge of the forecourt canopy
X_P0, X_P1 = -178.3, 194.0                # processor, north and south faces
X_F0, X_F1 = -123.2, 141.0                # forecourt canopy bay

Y_OUT, Y_CAN = 16.5, 18.0                 # eaves tip, light lines
WING = 9.0                                # width of a roof wing

ALU = (0.83, 0.84, 0.85)
ALU_D = (0.66, 0.67, 0.69)
BAMBOO = (0.74, 0.52, 0.31)
GLAZE = (0.20, 0.25, 0.29)
SKY = (0.33, 0.42, 0.48)
CONC = (0.66, 0.65, 0.62)
PAVE = (0.58, 0.58, 0.56)
BRIDGE_C = (0.42, 0.50, 0.55)
STEEL = (0.45, 0.47, 0.50)

RAINBOW = [(0.00, (0.13, 0.30, 0.72)), (0.18, (0.10, 0.52, 0.78)), (0.36, (0.20, 0.62, 0.35)),
           (0.50, (0.96, 0.80, 0.12)), (0.68, (0.96, 0.55, 0.12)), (0.84, (0.90, 0.32, 0.12)),
           (1.00, (0.78, 0.13, 0.12))]


def rainbow(x):
    """Colour of the Y props at x: blue at the north end of the pier, yellow in the middle, red in the south."""
    t = float(np.clip((x - X_N) / (X_S - X_N), 0, 1))
    for (t0, c0), (t1, c1) in zip(RAINBOW, RAINBOW[1:]):
        if t <= t1:
            s = (t - t0) / (t1 - t0)
            return tuple(a + s * (b - a) for a, b in zip(c0, c1))
    return RAINBOW[-1][1]


# ── Roof ─────────────────────────────────────────────────────────────────────────────────────────────
class Bay:
    """A row of wings across the building from z0 to z1 (z1 > z0), for x0 … x1; each edge is 'out' (a façade
    with an overhang and pointed tips) or 'can' (a light line shared with the next bay)."""

    def __init__(self, x0, x1, z0, z1, e0, e1, crown, lights=(0.37, 0.63), nu=16):
        self.x0, self.x1, self.z0, self.z1, self.e0, self.e1 = x0, x1, z0, z1, e0, e1
        self.crown, self.lights, self.nu = crown, lights, nu
        n = max(1, round((x1 - x0) / WING))
        self.w = (x1 - x0) / n
        self.n = n

    def y(self, u):
        """Height of the crest line of a wing at the fraction u across the bay."""
        u = np.asarray(u, float)
        y0 = Y_OUT if self.e0 == "out" else Y_CAN
        y1 = Y_OUT if self.e1 == "out" else Y_CAN
        base = y0 + (y1 - y0) * u
        mid = self.crown - 0.5 * (y0 + y1)
        h = base + mid * np.sin(np.pi * u) ** 0.8
        if self.e0 == "out":                                   # the tips curl up a little (gull wing)
            h = h + 0.9 * np.clip(1 - u / 0.1, 0, 1) ** 2
        if self.e1 == "out":
            h = h + 0.9 * np.clip(1 - (1 - u) / 0.1, 0, 1) ** 2
        return h

    def thick(self, u):
        u = np.asarray(u, float)
        t = 0.45 + 2.6 * np.sin(np.pi * u) ** 0.6
        return t

    def top(self, x, z):
        u = (z - self.z0) / (self.z1 - self.z0)
        v = ((x - self.x0) / self.w) % 1.0
        return float(self.y(u) + bump(v))


def bump(v):
    return 0.7 * np.sin(np.pi * np.clip(v, 0, 1))


def wings(bay):
    """Top (aluminium), underside (bamboo) and tips of every wing of a bay."""
    nu = bay.nu
    us = np.linspace(0, 1, nu + 1)
    vs = np.array([0.0, 0.5, 1.0])
    ms = []
    for k in range(bay.n):
        xa = bay.x0 + k * bay.w
        X = np.zeros((nu + 1, 3)); Z = np.zeros((nu + 1, 3))
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                pinch = 0.0
                if (i == 0 and bay.e0 == "out") or (i == nu and bay.e1 == "out"):
                    pinch = 0.32                                  # pointed tip of the wing at the eaves
                elif (i == 1 and bay.e0 == "out") or (i == nu - 1 and bay.e1 == "out"):
                    pinch = 0.06
                vv = 0.5 + (v - 0.5) * (1 - 2 * pinch)
                X[i, j] = xa + vv * bay.w
                Z[i, j] = bay.z0 + u * (bay.z1 - bay.z0)
        tipk = np.ones(nu + 1)                                   # the wings thin out to a lip at the eaves
        if bay.e0 == "out":
            tipk[0], tipk[1] = 0.3, 0.75
        if bay.e1 == "out":
            tipk[nu], tipk[nu - 1] = 0.3, 0.75
        Yt = bay.y(us)[:, None] + bump(vs)[None, :] * tipk[:, None]
        Yb = (bay.y(us) - bay.thick(us))[:, None] - 0.7 * bump(vs)[None, :] * tipk[:, None]
        tone = ALU if k % 2 else (0.80, 0.81, 0.82)
        for Y, col, up in ((Yt, tone, True), (Yb, BAMBOO, False)):
            v = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
            f = []
            for i in range(nu):
                for j in range(2):
                    a = i * 3 + j
                    q = (a, a + 1, a + 4, a + 3)
                    f += [(q[0], q[1], q[2]), (q[0], q[2], q[3])] if up else [(q[0], q[2], q[1]), (q[0], q[3], q[2])]
            m = Mesh(v, f, col)
            nn = np.cross(m.v[m.f[:, 1]] - m.v[m.f[:, 0]], m.v[m.f[:, 2]] - m.v[m.f[:, 0]])[:, 1].sum()
            if (nn > 0) != up:
                m.f = m.f[:, ::-1]
            if up:                                               # darker joints between the wings
                c = np.tile(np.array(col, float), (len(v), 1))
                c[np.isclose(np.tile(vs, nu + 1), 0) | np.isclose(np.tile(vs, nu + 1), 1)] = ALU_D
                m.c = c
            ms.append(m)
        # the edge band round each wing at the eaves tips and at the two ends of a bay row
        for i, e in ((0, bay.e0), (nu, bay.e1)):
            if e != "out":
                continue
            P = [(X[i, j], Yb[i, j], Z[i, j]) for j in range(3)] + [(X[i, j], Yt[i, j], Z[i, j]) for j in (2, 1, 0)]
            v = np.array(P)
            f = [(0, 1, 4), (0, 4, 5), (1, 2, 3), (1, 3, 4)]
            ms.append(Mesh(v, f, ALU))
        for j, xs in ((0, None), (2, None)):
            if not ((k == 0 and j == 0) or (k == bay.n - 1 and j == 2)):
                continue
            v = np.vstack([np.stack([X[:, j], Yb[:, j], Z[:, j]], 1), np.stack([X[:, j], Yt[:, j], Z[:, j]], 1)])
            f = []
            for i in range(nu):
                f += [(i, i + 1, nu + 2 + i), (i, nu + 2 + i, nu + 1 + i)]
            ms.append(Mesh(v, f, ALU_D))
    return merge(ms)


def lens(bay, cx, cz, rx, rz, seg=10):
    """A glass roof light lying on the wing (rim on the surface, a low dome)."""
    a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    rim = [(cx + rx * math.cos(t), bay.top(cx + rx * math.cos(t), np.clip(cz + rz * math.sin(t), bay.z0, bay.z1)) + 0.12,
            cz + rz * math.sin(t)) for t in a]
    v = np.array(rim + [(cx, bay.top(cx, cz) + 0.35, cz)])
    f = [(seg, (i + 1) % seg, i) for i in range(seg)]
    m = Mesh(v, f, SKY)
    nn = np.cross(m.v[m.f[:, 1]] - m.v[m.f[:, 0]], m.v[m.f[:, 2]] - m.v[m.f[:, 0]])[:, 1].sum()
    if nn < 0:
        m.f = m.f[:, ::-1]
    return m


def roof_lights(bays):
    ms = []
    for b in bays:
        for k in range(b.n):
            cx = b.x0 + (k + 0.5) * b.w
            for u in b.lights:
                cz = b.z0 + u * (b.z1 - b.z0)
                ms.append(lens(b, cx, cz, 1.5, 1.5, 8))
    return merge(ms)


def canyon_lights(bays_by_line):
    """The big oval roof lights over the canyons: one per wing, on the light line, across both bays."""
    ms = []
    for zc, b_lo, b_hi, x0, x1 in bays_by_line:
        n = max(1, round((x1 - x0) / WING))
        w = (x1 - x0) / n
        for k in range(n):
            cx = x0 + (k + 0.5) * w
            a = np.linspace(0, 2 * math.pi, 12, endpoint=False)
            rim = []
            for t in a:
                px, pz = cx + 3.7 * math.cos(t), zc + 4.6 * math.sin(t)
                b = b_lo if pz < zc else b_hi
                rim.append((px, b.top(px, float(np.clip(pz, b.z0, b.z1))) + 0.15, pz))
            v = np.array(rim + [(cx, b_hi.top(cx, zc) + 0.6, zc)])
            f = [(12, (i + 1) % 12, i) for i in range(12)]
            m = Mesh(v, f, SKY)
            nn = np.cross(m.v[m.f[:, 1]] - m.v[m.f[:, 0]], m.v[m.f[:, 2]] - m.v[m.f[:, 0]])[:, 1].sum()
            if nn < 0:
                m.f = m.f[:, ::-1]
            ms.append(m)
    return merge(ms)


# ── Props ────────────────────────────────────────────────────────────────────────────────────────────
def yprop(x, z_edge, inward, bay, h_node=8.0):
    """A canted Y prop at an eaves: the trunk leans out from inside the glass line, two arms up to the
    underside of the wings on either side of the joint at x. inward = +1 if the building is towards +z."""
    c = rainbow(x)
    foot = (x, 0.0, z_edge + inward * 6.5)
    node = (x, h_node, z_edge + inward * 4.2)
    ms = [tube([(foot[0], -0.5, foot[2]), foot, node], 0.55, c, seg=6)]
    for s in (-1, 1):
        ax = x + s * 0.5 * WING * 0.85
        az = z_edge + inward * 1.6
        u = (az - bay.z0) / (bay.z1 - bay.z0)
        ay = float(bay.y(u) - bay.thick(u)) - 0.2
        mid = (x + s * 1.2, h_node + 1.6, z_edge + inward * 3.4)
        ms.append(tube([node, mid, (ax, ay, az)], 0.4, c, seg=6))
    return merge(ms)


def props(bays_out):
    ms = []
    for bay, side in bays_out:
        z_edge = bay.z0 if side == 0 else bay.z1
        inward = 1 if side == 0 else -1
        for k in range(0, bay.n + 1, 2):
            x = bay.x0 + k * bay.w
            if x - bay.x0 < 4 or bay.x1 - x < 4:
                continue
            ms.append(yprop(x, z_edge, inward, bay))
    return merge(ms)


# ── Body, plinth, slab, bridges ──────────────────────────────────────────────────────────────────────
def body():
    inner = offset(BODY, -8.0)
    ms = [walls(inner, 1.2, 16.0, GLAZE), walls(inner, -0.3, 1.2, CONC)]
    # mullions: a light line every 9 m is too fine to see; a band at the departures floor instead
    ms.append(walls(offset(BODY, -7.9), 7.4, 8.0, ALU_D))
    return merge(ms)


def slab():
    """The paved platform under the whole outline (overhangs and bridges included), so that Offlined fits
    the terrain to the floor and not to the underside of the roof."""
    return merge([cap(OUTLINE, 0.05, PAVE), cap(OUTLINE, -0.3, PAVE, up=False), walls(OUTLINE, -0.3, 0.05, PAVE)])


def stubs():
    """The fixed glass link-bridges (T-shaped in the OSM outline): a corridor from the façade and a head
    across it, raised on legs."""
    ms = []
    P = OUTLINE
    groups, cur = [], []
    for p in P + P[:1]:
        x, z = p
        side = None
        if z > -27.4 and x > 200:
            side = "w"
        elif x > X_N + 0.5 and z < -94.8:
            side = "e"
        if side:
            cur.append((x, z, side))
        elif cur:
            groups.append(cur); cur = []
    for g in groups:
        side = g[0][2]
        xs = [p[0] for p in g]; zs = [p[1] for p in g]
        face = -27.5 if side == "w" else -94.6
        far = max(zs) if side == "w" else min(zs)
        s = 1 if side == "w" else -1
        hx0, hx1 = min(xs), max(xs)
        sx0, sx1 = sorted((g[0][0], g[-1][0]))
        head_z0 = far - s * 3.6
        ms += corridor(sx0, sx1, face - s * 8.0, head_z0)            # from the glass line, under the eaves
        ms += corridor(hx0, hx1, head_z0, far)
        for lx in (hx0 + 1.5, hx1 - 1.5):
            ms.append(box(0.6, 4.2, 0.6, STEEL).T(lx, 0, 0.5 * (head_z0 + far)))
        ms.append(box(0.6, 4.2, 0.6, STEEL).T(0.5 * (sx0 + sx1), 0, 0.5 * (face + head_z0)))
    # the transverse link at the north end of the pier (two arms with their links to the pier)
    for (x0, x1, z0, z1) in ((-599.8, -591.7, -188.2, -76.6), (-599.3, -591.4, -39.1, 12.0),
                             (-593.7, X_N - 8, -80.2, -76.7), (-593.5, X_N - 8, -39.1, -35.1)):
        ms += corridor(x0, x1, z0, z1)
    for z in np.arange(-180.0, 12.0, 14.0):
        if -76 < z < -40:
            continue
        ms.append(box(0.6, 4.2, 0.6, STEEL).T(-595.6, 0, z))
    return merge(ms)


def corridor(x0, x1, z0, z1, y0=4.2, y1=8.4):
    x0, x1 = sorted((x0, x1)); z0, z1 = sorted((z0, z1))
    w, d = x1 - x0, z1 - z0
    if w < 0.3 or d < 0.3:
        return []
    return [box(w, y1 - y0, d, BRIDGE_C).T(0.5 * (x0 + x1), y0, 0.5 * (z0 + z1)),
            box(w + 0.3, 0.35, d + 0.3, ALU_D).T(0.5 * (x0 + x1), y1, 0.5 * (z0 + z1))]


def main():
    out = ROOT / "models" / "Q166276"
    out.mkdir(parents=True, exist_ok=True)
    # bays: the pier in three runs, the two inner bays of the processor and the forecourt canopy
    D_N = Bay(X_N, X_P0, Z_E, -23.0, "out", "out", 20.5, (0.40, 0.67))
    D_M = Bay(X_P0, X_P1, Z_E, Z_C0, "out", "can", 20.5, (0.40, 0.67))
    D_S = Bay(X_P1, X_S, Z_E, -27.5, "out", "out", 20.5, (0.40, 0.67))
    P1 = Bay(X_P0, X_P1, Z_C0, Z_C1, "can", "can", 22.5)
    P2 = Bay(X_P0, X_P1, Z_C1, Z_C2, "can", "can", 22.5)
    P3 = Bay(X_F0, X_F1, Z_C2, Z_W, "can", "out", 20.5)
    bays = [D_N, D_M, D_S, P1, P2, P3]
    roof = merge([wings(b) for b in bays])
    lights = merge([roof_lights(bays),
                    canyon_lights([(Z_C0, D_M, P1, X_P0, X_P1), (Z_C1, P1, P2, X_P0, X_P1), (Z_C2, P2, P3, X_F0, X_F1)])])
    yp = props([(D_N, 0), (D_M, 0), (D_S, 0), (D_N, 1), (D_S, 1), (P3, 1)])
    base = merge([body(), slab(), stubs()])
    parts = [("roof", roof, {"material": "alu"}), ("roof_lights", lights, {"material": "glass"}),
             ("props", yp, {"material": "paint"}), ("building", base)]
    tris, size = write_glb(out / "model.glb", parts, roughness=0.85,
                           materials={"alu": {"metallic": 0.35, "roughness": 0.45}, "glass": GLASS,
                                      "paint": {"metallic": 0.1, "roughness": 0.5}})
    allv = np.vstack([p[1].v for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.1f}..{hi[0]:.1f}, "
          f"y {lo[1]:.1f}..{hi[1]:.1f}, z {lo[2]:.1f}..{hi[2]:.1f}")


if __name__ == "__main__":
    main()
