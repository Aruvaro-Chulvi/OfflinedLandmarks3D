"""
Plaza Mayor de Madrid (Q1123493) — the ring of buildings around the square, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The square: Juan Gómez de Mora (1586–1648), 1617–1619, after a first plan by Juan de Herrera (1530–1597) and the
Casa de la Panadería begun by Diego Sillero (1590); rebuilt after the fires of 1631 (Gómez de Mora), 1672 (Tomás
Román) and 1790 (Juan de Villanueva, 1739–1811, who lowered the houses from five to three floors and closed the
corners with the arches; works went on until 1854 under Antonio López Aguado and Custodio Moreno). Public domain.
The present murals of the Casa de la Panadería (Carlos Franco, 1992) are NOT reproduced: the façade only gets a
warmer colour and abstract colour panels.

What is modelled: the houses around the square and, since 2026-11-28, the square itself (without its floor the
game saw through the open square and drew its own ground there, grass and trees):
- The paving of 1961, "ajedrezado" setts of three stones (porphyry, limestone and red Ávila granite, es.wikipedia
  «Historia de la plaza Mayor de Madrid»), measured on the IGN PNOA orthophoto (CC BY 4.0, scne.es; ~0.1 m/px):
  a grid of dark porphyry bands about 2.7 m wide, every 16.3 m east–west (±8.15, ±24.45, ±40.75, ±57.05 m from
  the statue) and every 11.5 m north–south (±5.75, ±17.25, ±28.75, ±40.25 m); a thinner light limestone band
  (~1.2 m) halfway between them both ways; the fields of pinkish-grey granite setts in between.
- The equestrian statue of Felipe III in the middle (Giambologna and Pietro Tacca, 1616, here since 1848, pedestal
  by Sabino de Medina; public domain) on its base of about 8.5 × 5.5 m (orthophoto), simplified: the horse with its
  left foreleg raised, the king with the baton at his waist (es.wikipedia «Estatua ecuestre de Felipe III»). The
  heights (about 4.5 m of pedestal, 4.5 m of horse and rider) are estimates and the way it faces was not checked.
- The four big lampposts on round bases at the crossings of the dark bands (±24.45, ±17.25 m; orthophoto), and
  the two dark ventilation grilles of the car park south-east of the statue (orthophoto). The lamps' design is
  generic.
- The café terraces against the porticoes ("instalan terrazas junto a los soportales", es.wikipedia): the groups
  of large square white parasols where the orthophoto shows them (they change with the season).

Measurements (see meta.json for the sources):
- OpenStreetMap: the square is relation 16657232 (outer way 4263036 = the line of the façades). Its four corners
  give a quadrilateral of 120.9 × 84.1–84.7 m (es.wikipedia: 120.9 × 93 m), turned 3.3° from east–west. The
  open ground floor (building:part with building:min_level=1 all around, 54 pieces) is 5.7–6.2 m deep: the
  portico. The towers of the Casa de la Panadería (north) and of the Casa de la Carnicería (south) are notches
  of the outline, 5.6–6.0 m wide, projecting 0.3 m into the square. The ten entrances are the pedestrian ways
  that cross the outline (Calle de Toledo, Botoneras, Escalerilla de Piedra = Arco de Cuchilleros, Zaragoza,
  Gerona, Sal, Felipe III, Arco del Triunfo, Siete de Julio, Ciudad Rodrigo), all tunnel=building_passage.
- The houses behind the portico: the 27 OSM buildings that touch the outline (ways 2541458xx, 2542899xx,
  2543155xx/6xx, 2543292xx, relations 3406930 Casa de la Carnicería and 18352546), minus the 6 m band of the
  portico range, rasterised at 0.1 m, cleaned and simplified to ±0.18 m (embedded below as DATA). Which of their
  walls face a street (windows) and which are party walls was decided against the neighbouring OSM buildings.
- Heights, measured on near-orthographic photos (Wikimedia Commons, «Plaza Mayor de Madrid - 01.jpg» of the
  Panadería, «Casa de la Carnicería (Plaza Mayor de Madrid) 01.jpg», ±5 %): portico lintel 4.4, first-floor
  balconies 6.2, second 10.1, third 13.9, cornice 17.4–18.0, slate roof ridge about 21; window bays 3.2–3.7 m.
  Panadería towers: body 22.7, slate cap 25.2, lantern to 28.6, spire to 35.6, cross 38.3 m. Carnicería
  towers: body 22.5 (OSM height 22.5), spire tip 31.8, cross 33.6 m.
- Arches: six large ones (Toledo, Gerona, Sal, Felipe III, Siete de Julio, Ciudad Rodrigo), rising through
  the first floor; Zaragoza, Botoneras and the Arco del Triunfo (under the west tower of the Panadería) are
  plain openings at the back of the portico. Arco de Cuchilleros: at the south-west corner a passage over the
  stone steps (OSM way 254329273, 18 m long) to a granite arch about 8 m high on Calle de Cuchilleros, some
  6 m below the square. The ground falls to the west (Cava de San Miguel) and south-west, so the foundations
  go down to 9–12 m there (3 m elsewhere).
Materials: red-ochre (almagre) plaster, granite porticoes, string courses and window frames, white shutters,
black iron balconies, slate roofs with dormers; the outer façades in ochres and creams.

Axes (docs/MODEL_SPEC.md): +Y up; origin on the ground in the middle of the square; the main façade is the
Casa de la Panadería, on the north side, which faces south into the square: so +Z points south (heading
176.7°, square to the north side of the square) and +X points east. The Panadería is at z ≈ −42 with its
façade facing +Z; the Casa de la Carnicería is at z ≈ +42 facing −Z. Being a ring, the "front view" from +Z
looks at the back of the south range.

    python model_sources/Q1123493_plaza_mayor_madrid.py   -> models/Q1123493/model.glb
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, quad, tube, cap, extrude, ring_area, write_glb, METAL  # noqa: E402

rng = np.random.default_rng(1619)

# ── Colours ──────────────────────────────────────────────────────────────────
RED = (0.62, 0.27, 0.20)               # almagre
RED_C = (0.60, 0.26, 0.20)             # Casa de la Carnicería
PAINT = (0.84, 0.60, 0.47)             # Casa de la Panadería (painted façade, average tone)
MURALS = [(0.86, 0.66, 0.52), (0.80, 0.55, 0.45), (0.76, 0.64, 0.58), (0.88, 0.71, 0.52), (0.74, 0.60, 0.58)]
GRANITE = (0.69, 0.68, 0.65)
GRANITE_D = (0.58, 0.575, 0.55)
FRAME = (0.84, 0.81, 0.74)             # light stone window frames
SHUTTER = (0.91, 0.90, 0.86)
GLASS_C = (0.13, 0.14, 0.16)
IRON = (0.08, 0.08, 0.09)
SLATE = (0.30, 0.32, 0.35)
SLATE_D = (0.23, 0.245, 0.27)
LEAD = (0.42, 0.44, 0.46)
TILE = [(0.64, 0.38, 0.29), (0.60, 0.36, 0.28), (0.67, 0.42, 0.31), (0.56, 0.37, 0.30)]
OCHRES = [(0.86, 0.63, 0.40), (0.88, 0.78, 0.60), (0.81, 0.53, 0.35), (0.72, 0.40, 0.29), (0.85, 0.71, 0.52),
          (0.80, 0.60, 0.42)]
SOFFIT = (0.80, 0.77, 0.70)
SHOP = (0.80, 0.76, 0.68)
DOOR = (0.24, 0.18, 0.14)
GILT = (0.86, 0.68, 0.26)
PAVE = (0.66, 0.64, 0.59)              # the fields of granite setts (orthophoto)
PAVE_D = (0.51, 0.50, 0.47)            # dark porphyry bands (orthophoto)
BRONZE = (0.17, 0.15, 0.11)
PARASOL = (0.93, 0.92, 0.89)           # the terraces' square white parasols
PAVE_L = (0.80, 0.78, 0.73)            # light limestone bands
GRILLE = (0.16, 0.16, 0.17)
TABLE = (0.35, 0.33, 0.31)
CLOCK = (0.93, 0.92, 0.88)

F = -50.0                              # sentinel for "down to the foundations" (see found())

# ── Plan (model metres; from OpenStreetMap, see the docstring) ───────────────
Q = [(-60.820, -42.146), (60.055, -42.151), (61.100, 41.897), (-59.870, 42.530)]   # NW, NE, SE, SW
D = 6.0                                 # depth of the portico range (the open ground floor of OSM is 5.7–6.2)
PD = 5.6                                # back of the portico
SIDE_NAMES = ("N", "E", "S", "W")
TOWERS = [  # side, u0, u1, kind
    (0, 38.49, 44.50, "pan"), (0, 76.19, 82.12, "pan"),
    (2, 39.72, 45.27, "carn"), (2, 75.91, 81.47, "carn")]
SPECIAL = [(0, 38.49, 82.12, "pan"), (2, 39.72, 81.47, "carn")]
ENTRANCES = [  # name, side, u (along the side), width, big arch?
    ("Calle de Toledo", 2, 88.45, 5.0, True), ("Calle Botoneras", 2, 10.1, 3.6, False),
    ("Arco de Cuchilleros", 2, 119.15, 3.0, False),
    ("Calle de Zaragoza", 1, 45.35, 4.0, False), ("Calle de Gerona", 1, 76.0, 5.0, True),
    ("Calle de la Sal", 1, 10.4, 5.0, True),
    ("Calle de Felipe III", 0, 108.45, 5.0, True), ("Arco del Triunfo", 0, 42.1, 2.4, False),
    ("Calle del Siete de Julio", 0, 14.65, 5.0, True),
    ("Calle Ciudad Rodrigo", 3, 75.3, 5.0, True)]
JW = 1.0                                # jambs of the big arches
FLOORS = [(6.25, 2.6), (10.15, 2.5), (14.0, 2.3)]   # (sill, window height) of the three floors
TOP = 17.5                              # walls of the range end here, inside the cornice (17.4–18.0)
EAVE, RIDGE = 18.0, 21.2

# The houses behind the portico range (OSM buildings minus the 6 m band; x, z in metres), which of their
# edges face a street ("1": windows), and the cleared area (footprint, outer outline).
DATA = json.loads(r"""{
"pieces": [
{"id":"way/254145870","name":"","P":[[-75.9,-41.4],[-76.5,-41.8],[-76.3,-42.6],[-73.0,-50.8],[-71.2,-56.7],[-60.1,-58.3],[-59.8,-58.0],[-58.9,-48.5],[-59.1,-48.3],[-67.0,-48.2],[-66.9,-38.5],[-67.1,-38.3]],"holes":[],"win":"000000000001"},
{"id":"way/254145873","name":"Casa de la Panadería","P":[[-17.4,-48.3],[-17.6,-48.5],[-17.6,-54.7],[-18.1,-55.0],[-18.2,-64.5],[-18.0,-64.7],[-9.7,-64.8],[-9.5,-64.6],[-9.3,-57.9],[8.9,-57.9],[8.9,-65.2],[9.1,-65.4],[20.1,-65.5],[20.5,-65.2],[20.5,-58.3],[19.8,-58.0],[19.8,-54.6],[20.2,-54.3],[20.2,-52.3],[19.5,-52.0],[19.6,-48.5],[19.4,-48.3]],"holes":[],"win":"0001000111000100000000"},
{"id":"way/254145875","name":"","P":[[28.1,-48.3],[27.2,-52.2],[22.7,-52.2],[22.3,-54.7],[20.1,-54.6],[19.9,-54.8],[19.9,-57.8],[20.6,-58.1],[20.6,-60.4],[20.8,-60.6],[23.0,-60.6],[23.2,-61.8],[26.8,-61.8],[27.1,-61.3],[27.3,-59.4],[39.4,-60.2],[39.7,-59.9],[41.0,-49.3],[40.8,-48.3]],"holes":[],"win":"0011000000000000100"},
{"id":"way/254145877","name":"","P":[[19.9,-48.3],[19.7,-48.5],[19.6,-51.8],[19.8,-52.0],[27.0,-52.1],[27.3,-51.8],[27.8,-48.8],[27.6,-48.3]],"holes":[],"win":"00000000"},
{"id":"way/254145878","name":"","P":[[-43.6,-48.3],[-43.9,-48.9],[-44.1,-54.1],[-43.9,-54.3],[-32.0,-54.6],[-31.8,-54.4],[-31.7,-50.3],[-31.9,-48.3]],"holes":[],"win":"01000000"},
{"id":"way/254145885","name":"","P":[[-31.4,-48.3],[-31.6,-48.5],[-31.7,-52.5],[-28.9,-52.8],[-28.9,-58.1],[-27.7,-58.4],[-21.2,-58.8],[-21.0,-58.6],[-20.9,-54.7],[-17.9,-54.7],[-17.7,-54.5],[-17.6,-49.3],[-17.9,-48.3]],"holes":[],"win":"0011000110000"},
{"id":"way/254145886","name":"","P":[[-58.6,-48.3],[-59.1,-51.4],[-57.3,-51.8],[-57.5,-54.6],[-56.6,-54.9],[-49.7,-55.5],[-49.4,-55.2],[-48.8,-49.8],[-49.0,-48.3]],"holes":[],"win":"001000100"},
{"id":"way/254289936","name":"","P":[[66.8,-7.2],[66.6,-7.4],[66.5,-18.3],[66.7,-18.5],[78.6,-18.6],[78.8,-18.4],[78.8,-15.4],[83.6,-15.2],[84.9,-14.5],[85.3,-14.1],[85.2,-13.7],[81.0,-7.1]],"holes":[],"win":"000000000010"},
{"id":"way/254289939","name":"","P":[[66.5,-35.6],[66.2,-36.1],[66.2,-40.6],[66.4,-40.8],[78.2,-40.9],[78.4,-41.9],[82.9,-42.0],[82.2,-54.0],[82.4,-54.9],[95.4,-54.9],[95.7,-53.0],[95.8,-47.9],[101.8,-47.3],[102.7,-47.0],[103.0,-43.0],[99.3,-37.6],[68.0,-35.6]],"holes":[],"win":"00000000000000110"},
{"id":"way/254289941","name":"","P":[[66.7,-18.7],[66.5,-18.9],[66.3,-29.1],[66.5,-29.3],[81.0,-30.0],[81.3,-28.7],[81.4,-24.9],[81.0,-24.7],[79.0,-25.1],[78.8,-19.0],[77.6,-18.8],[77.4,-19.0],[77.3,-21.6],[75.3,-21.6],[74.9,-18.8],[70.9,-18.7]],"holes":[],"win":"0001000010010100"},
{"id":"way/254289947","name":"","P":[[66.4,-41.0],[66.2,-41.2],[66.0,-48.3],[49.2,-48.3],[48.1,-54.3],[49.0,-54.6],[61.0,-55.7],[61.3,-55.2],[61.4,-53.5],[62.3,-53.4],[64.9,-53.7],[64.8,-56.1],[65.6,-56.4],[76.9,-57.4],[77.3,-55.6],[81.6,-56.3],[82.2,-54.0],[82.8,-42.3],[82.6,-42.1],[78.2,-42.1],[78.0,-41.1],[72.1,-41.0]],"holes":[],"win":"0001000001100000000000"},
{"id":"way/254315565","name":"","P":[[22.4,55.0],[21.5,54.7],[21.4,48.4],[21.6,48.2],[47.1,48.1],[47.3,48.3],[46.8,52.1],[46.5,52.4],[26.1,51.6],[26.3,55.1],[26.1,55.3]],"holes":[],"win":"00000100100"},
{"id":"way/254315570","name":"","P":[[-98.6,-7.8],[-90.0,-16.3],[-89.5,-15.9],[-82.9,-22.2],[-73.6,-27.4],[-71.9,-31.6],[-67.1,-29.9],[-66.8,-29.6],[-66.7,-15.5],[-66.9,-15.3],[-74.7,-15.2],[-75.0,-15.7],[-80.4,-15.7],[-92.4,-1.3]],"holes":[],"win":"00000100000001"},
{"id":"way/254315575","name":"","P":[[65.4,58.1],[65.6,55.9],[54.1,53.4],[53.4,53.0],[54.2,48.4],[54.5,48.0],[67.3,47.9],[67.2,37.3],[67.4,37.1],[78.7,42.9],[72.7,57.1],[72.2,57.5],[70.6,57.8],[68.7,59.0]],"holes":[],"win":"00010000100000"},
{"id":"way/254315580","name":"","P":[[67.1,30.4],[66.8,11.9],[66.8,6.9],[67.0,6.7],[79.2,9.6],[76.4,17.2],[75.9,17.4],[75.0,17.1],[74.6,18.0],[72.2,17.6],[72.2,22.3],[73.1,22.4],[73.3,22.8],[71.6,26.5],[72.0,27.1],[69.9,31.6]],"holes":[],"win":"0001000011000001"},
{"id":"way/254315582","name":"","P":[[-75.2,26.3],[-75.6,24.9],[-76.1,22.9],[-75.9,22.3],[-67.8,21.7],[-66.3,21.9],[-66.2,25.7],[-66.4,25.9],[-74.2,26.2],[-74.8,26.5]],"holes":[],"win":"0000000000"},
{"id":"way/254315584","name":"","P":[[-73.4,32.4],[-75.0,26.8],[-74.3,26.4],[-66.4,26.1],[-66.2,26.3],[-66.2,32.2],[-66.4,32.4]],"holes":[],"win":"1000000"},
{"id":"way/254315592","name":"","P":[[-76.4,21.9],[-77.9,16.9],[-80.9,12.7],[-80.5,12.3],[-66.6,11.4],[-66.4,11.6],[-66.3,14.7],[-66.3,21.3],[-67.7,21.6],[-75.3,21.9],[-75.9,22.2]],"holes":[],"win":"11000000000"},
{"id":"way/254315594","name":"","P":[[-56.6,60.0],[-56.9,59.6],[-57.1,48.8],[-56.9,48.6],[-48.2,48.6],[-48.0,48.8],[-47.8,57.2],[-48.7,57.5],[-53.9,57.9],[-53.9,59.8],[-54.1,60.0]],"holes":[],"win":"00000001000"},
{"id":"way/254315595","name":"","P":[[-73.1,40.7],[-73.3,40.5],[-73.5,32.8],[-73.3,32.6],[-66.4,32.6],[-66.2,32.8],[-66.1,39.7],[-66.3,39.9],[-68.9,39.9],[-69.2,40.7]],"holes":[],"win":"0100000000"},
{"id":"way/254315596","name":"","P":[[-81.5,12.0],[-86.5,4.8],[-77.4,-4.0],[-66.7,-4.7],[-66.5,-4.5],[-66.4,11.0],[-67.3,11.3],[-80.4,12.1],[-80.8,12.6]],"holes":[],"win":"100000000"},
{"id":"way/254315597","name":"","P":[[66.9,-0.6],[66.7,-0.8],[66.6,-6.8],[66.8,-7.0],[80.6,-6.9],[80.8,-6.5],[76.0,1.3]],"holes":[],"win":"0000011"},
{"id":"way/254315603","name":"","P":[[-41.7,58.8],[-42.0,58.1],[-44.2,57.1],[-47.5,57.3],[-47.7,57.1],[-47.9,48.8],[-46.8,48.5],[-32.2,48.5],[-31.9,49.0],[-30.6,56.9],[-31.0,57.2],[-34.1,57.8],[-41.3,58.8]],"holes":[],"win":"0110000010000"},
{"id":"way/254329219","name":"","P":[[-72.9,47.4],[-73.2,45.9],[-73.3,41.1],[-73.1,40.9],[-69.0,40.9],[-68.7,40.0],[-66.3,40.0],[-66.1,40.2],[-66.0,43.7],[-66.0,47.2],[-66.2,47.4]],"holes":[],"win":"01000000000"},
{"id":"way/254329273","name":"Arco de Cuchilleros","P":[[-60.3,60.8],[-60.5,60.6],[-61.1,49.3],[-60.9,48.6],[-57.4,48.6],[-57.2,48.8],[-57.2,57.4],[-56.9,60.2],[-57.4,60.8]],"holes":[],"win":"000000001"},
{"id":"relation/3406930","name":"Casa de la Carnicería","P":[[-2.9,83.2],[-3.2,77.3],[-5.0,62.4],[-5.7,57.6],[-7.1,51.2],[-10.2,51.5],[-10.6,50.2],[-12.1,50.2],[-21.9,50.8],[-22.8,51.4],[-23.1,51.1],[-23.4,48.6],[-23.2,48.4],[21.1,48.2],[21.4,49.0],[21.4,55.2],[21.2,55.7],[21.0,67.0],[22.2,67.7],[22.7,68.7],[23.1,82.5],[22.9,82.7],[10.7,82.7],[-1.0,83.4]],"holes":[[[15.4,74.3],[15.2,63.2],[-0.3,63.1],[0.7,74.2],[1.0,74.8]],[[2.4,55.0],[-1.1,54.9],[-1.2,58.4],[2.3,58.5]]],"win":"100000000010000000010110"},
{"id":"relation/18352546","name":"","P":[[-92.4,-1.3],[-80.6,-15.4],[-80.4,-15.6],[-75.2,-15.6],[-74.9,-15.0],[-66.9,-15.1],[-66.6,-14.3],[-66.6,-5.1],[-67.4,-4.8],[-77.3,-4.2],[-86.5,4.8]],"holes":[],"win":"00000000001"}],
"fp": [[-2.9,84.6],[23.2,83.8],[24.2,82.8],[23.8,68.7],[22.1,66.6],[22.3,57.1],[23.3,56.1],[26.4,56.5],[28.3,52.7],[46.8,53.5],[49.2,48.9],[51.8,48.9],[52.8,49.9],[52.2,53.3],[53.2,54.3],[63.0,56.4],[65.1,59.2],[68.7,60.1],[73.6,57.5],[79.9,42.8],[69.3,37.0],[68.3,36.0],[68.3,33.5],[71.0,31.8],[74.6,22.4],[73.3,20.1],[77.3,17.6],[80.4,9.4],[79.4,8.4],[68.9,6.0],[67.8,4.9],[67.7,1.9],[69.0,0.9],[76.1,2.5],[86.5,-14.2],[84.1,-16.1],[80.9,-16.4],[79.9,-17.4],[80.0,-22.6],[82.5,-24.5],[82.3,-30.2],[81.3,-31.2],[68.5,-30.4],[67.5,-31.4],[67.5,-33.6],[69.5,-34.7],[99.4,-36.5],[104.1,-43.0],[103.8,-47.3],[102.8,-48.3],[97.9,-48.7],[96.8,-49.8],[96.7,-55.0],[95.7,-56.0],[83.3,-56.0],[81.9,-57.5],[78.8,-56.9],[77.1,-58.6],[64.6,-57.4],[62.8,-55.5],[61.3,-56.9],[48.0,-55.6],[47.0,-54.6],[47.7,-50.2],[46.7,-49.2],[42.9,-49.2],[41.9,-50.2],[40.8,-59.9],[39.7,-61.4],[29.1,-60.6],[27.1,-63.0],[22.6,-62.6],[20.1,-66.6],[8.8,-66.5],[7.8,-65.5],[7.9,-60.0],[6.9,-59.0],[-7.3,-59.0],[-8.3,-60.0],[-8.4,-65.0],[-9.4,-66.0],[-18.3,-65.8],[-19.3,-64.8],[-19.3,-60.4],[-20.3,-59.4],[-20.9,-60.0],[-29.0,-59.4],[-31.0,-55.1],[-31.7,-55.8],[-44.2,-55.4],[-45.2,-54.4],[-45.0,-50.2],[-46.0,-49.2],[-47.8,-50.2],[-48.4,-55.7],[-49.4,-56.7],[-57.6,-55.9],[-59.8,-59.5],[-71.4,-57.8],[-72.4,-56.8],[-77.7,-41.6],[-68.8,-37.9],[-67.8,-36.9],[-67.7,-32.6],[-69.0,-31.6],[-72.1,-32.8],[-75.2,-27.6],[-82.9,-23.3],[-99.7,-7.8],[-87.5,4.9],[-78.8,17.3],[-74.6,32.4],[-74.2,47.6],[-73.2,48.6],[-66.9,48.5],[-65.9,49.6],[-63.1,49.5],[-62.1,50.5],[-61.6,61.0],[-60.6,62.0],[-53.8,61.1],[-51.8,58.8],[-44.2,58.2],[-42.0,60.0],[-30.5,58.2],[-29.5,57.2],[-30.6,50.8],[-29.6,49.4],[-25.3,49.3],[-23.1,52.5],[-12.1,51.3],[-10.5,52.7],[-8.7,52.4],[-6.8,57.6],[-4.0,83.6]]}""")


def found(x, z):
    """Foundation depth: the ground falls to the west (Cava de San Miguel) and the south-west (Cuchilleros)."""
    w = np.clip((-x - 35.0) / 28.0, 0, 1)
    sw = np.clip((z - 30.0) / 30.0, 0, 1) * np.clip(-x / 60.0, 0, 1)
    return -3.0 - 6.0 * w - 3.0 * sw


# ── Helpers ──────────────────────────────────────────────────────────────────
def B(x0, x1, y0, y1, z0, z1, c):
    """Axis-aligned box from its extents."""
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def jit(c, k=0.025):
    return tuple(np.clip(np.asarray(c) * (1 + rng.normal(0, k)), 0, 1))


def sphere(r, c, seg=12, rings=7):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def prism(P, y0, y1, edge_c, top_c=None, bottom_c=None, hole=False):
    """Prism over polygon P [(x, z)] with one colour per wall (edge_c(i, a, b) -> colour)."""
    P = [tuple(p) for p in P]
    if ring_area(P) < 0:
        P = P[::-1]
    v, f, c = [], [], []
    for i in range(len(P)):
        (x0, z0), (x1, z1) = P[i], P[(i + 1) % len(P)]
        if math.hypot(x1 - x0, z1 - z0) < 1e-6:
            continue
        o = len(v)
        v += [(x0, y0, z0), (x1, y0, z1), (x1, y1, z1), (x0, y1, z0)]
        f += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
        c += [edge_c(i, P[i], P[(i + 1) % len(P)])] * 4
    ms = [Mesh(v, np.array(f)[:, ::-1] if hole else f, np.array(c))]
    if top_c is not None:
        ms.append(cap(P, y1, top_c, up=True))
    if bottom_c is not None:
        ms.append(cap(P, y0, bottom_c, up=False))
    return merge(ms)


def clip(P, a, b):
    """Convex polygon in local (u, v) clipped to a ≤ u ≤ b."""
    def half(P, k, s):
        out = []
        for i in range(len(P)):
            p, q = P[i], P[(i + 1) % len(P)]
            fp, fq = s * (p[0] - k), s * (q[0] - k)
            if fp >= 0:
                out.append(p)
            if (fp >= 0) != (fq >= 0):
                t = fp / (fp - fq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        return out
    if a > -1e8:
        P = half(P, a, 1)
    if b < 1e8 and P:
        P = half(P, b, -1)
    return P


def arch_block(x0, x1, ys, ytop, z0, z1, c, seg=20):
    """The wall over a semicircular arch (front and back faces) and its soffit (as in the Puerta de Alcalá)."""
    r, xc = (x1 - x0) / 2, (x0 + x1) / 2
    arc = [(xc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(math.pi, 0, seg + 1)]
    out = []
    for z in (z0, z1):
        V = [(x, y, z) for x, y in arc] + [(x0, ytop, z), (x1, ytop, z)]
        TL, TR, h = seg + 1, seg + 2, seg // 2
        Fc = [(TL, i, i + 1) for i in range(h)] + [(TL, h, TR)] + [(TR, i, i + 1) for i in range(h, seg)]
        out.append(Mesh(V, Fc, c))
    V, Fc = [], []
    for x, y in arc:
        V += [(x, y, z0), (x, y, z1)]
    for i in range(seg):
        a = 2 * i
        Fc += [(a, a + 1, a + 3), (a, a + 3, a + 2)]
    out.append(Mesh(V, Fc, c))
    return merge(out)


def railing(x0, x1, y0, h, z, c=IRON, step=0.32):
    """Iron railing seen from the front: top and bottom bars and thin balusters (facing +z)."""
    out = [B(x0, x1, y0 + h - 0.05, y0 + h, z - 0.03, z + 0.03, c), B(x0, x1, y0, y0 + 0.05, z - 0.02, z + 0.02, c)]
    n = max(2, int((x1 - x0) / step))
    for k in range(n + 1):
        x = x0 + (x1 - x0) * k / n
        out.append(quad(x, y0 + h / 2, z, 0.035, h, "+z", c, eps=0.0))
    return out


# ── Sides of the square: local frames ────────────────────────────────────────
class Side:
    """Local frame of one side: u along the façade (to the right of someone in the square looking at it),
    y up, w towards the square (the façade line is w = 0, the range is w < 0)."""

    def __init__(self, i):
        self.i = i
        A, Bp = np.array(Q[i]), np.array(Q[(i + 1) % 4])
        self.A, self.L = A, float(np.linalg.norm(Bp - A))
        self.t = (Bp - A) / self.L
        self.m = np.array([-self.t[1], self.t[0]])            # into the square

    def local(self, p):
        q = np.asarray(p, float) - self.A
        return float(q @ self.t), float(q @ self.m)

    def world(self, mesh):
        n = mesh.copy()
        lx, lz = mesh.v[:, 0].copy(), mesh.v[:, 2].copy()
        n.v[:, 0] = self.A[0] + lx * self.t[0] + lz * self.m[0]
        n.v[:, 2] = self.A[1] + lx * self.t[1] + lz * self.m[1]
        return n


SIDES = [Side(i) for i in range(4)]


def offset_quad(d):
    """The quadrilateral of the façades moved outwards by d (inwards if d < 0), mitred corners."""
    lines = [(s.A - s.m * d, s.t) for s in SIDES]
    out = []
    for i in range(4):
        p1, t1 = lines[i - 1]
        p2, t2 = lines[i]
        k = np.linalg.solve(np.array([t1, -t2]).T, p2 - p1)
        out.append(tuple(map(float, p1 + t1 * k[0])))
    return out


def trapezoid(s, d0, d1):
    """Local (u, w) outline of the band between depths d0 and d1 of side s, mitred with its neighbours."""
    a, b = offset_quad(d0), offset_quad(d1)
    i, j = s.i, (s.i + 1) % 4
    return [s.local(a[i]), s.local(a[j]), s.local(b[j]), s.local(b[i])]


def edge_colour(inner_c, outer_c, end_c, d0, d1):
    def fn(i, p, q):
        if abs(p[1] + d0) < 0.05 and abs(q[1] + d0) < 0.05:
            return jit(inner_c, 0.01)
        if abs(p[1] + d1) < 0.05 and abs(q[1] + d1) < 0.05:
            return outer_c
        return end_c
    return fn


# ── Façade elements (local coordinates) ──────────────────────────────────────
def window(xc, sill, h, w=1.3, frame=FRAME, z=0.0, shut=None):
    shut = (rng.random() < 0.62) if shut is None else shut
    return [B(xc - w / 2 - 0.16, xc + w / 2 + 0.16, sill - 0.05, sill + h + 0.22, z, z + 0.05, frame),
            quad(xc, sill + h / 2, z + 0.05, w, h, "+z", jit(SHUTTER, 0.02) if shut else GLASS_C, eps=0.025)]


def bays(a, b, step):
    n = max(1, int(round((b - a) / step)))
    return [a + (k + 0.5) * (b - a) / n for k in range(n)], (b - a) / n


def portico_house(a, b, has_left, has_right, dy=0.0):
    """Lintelled granite portico between u = a and b: square pillars, zapatas, the lintel."""
    out = []
    cs, bay = bays(a, b, 3.5)
    xs = [a + k * bay for k in range(len(cs) + 1)]
    for k, x in enumerate(xs):
        if (k == 0 and not has_left) or (k == len(xs) - 1 and not has_right):
            continue
        out += [B(x - 0.38, x + 0.38, -1.0, 4.1 + dy, -0.76, 0.02, jit(GRANITE)),
                B(x - 0.48, x + 0.48, -0.5, 0.32 + dy, -0.86, 0.08, GRANITE_D),
                B(x - 0.78, x + 0.78, 4.1 + dy, 4.42 + dy, -0.76, 0.02, GRANITE)]
    out.append(B(a, b, 4.4 + dy, 5.95 + dy, -0.62, 0.06, jit(GRANITE, 0.01)))
    return out


def portico_arches(a, b, openings):
    """Arcade of round arches on granite piers (Panadería, Carnicería): openings = [(centre, width)]."""
    out = []
    edges = [a] + [e for c, w in openings for e in (c - w / 2, c + w / 2)] + [b]
    for k in range(0, len(edges), 2):
        x0, x1 = edges[k], edges[k + 1]
        if x1 - x0 > 0.05:
            out.append(B(x0, x1, -1.0, 5.0, -0.75, 0.05, jit(GRANITE)))
    for c, w in openings:
        out.append(arch_block(c - w / 2, c + w / 2, 3.6, 5.0, -0.75, 0.05, GRANITE, 16))
        out.append(B(c - w / 2 - 0.12, c + w / 2 + 0.12, 3.45, 3.6, -0.8, 0.09, GRANITE_D))  # impost
    out.append(B(a - 0.1, b + 0.1, 5.0, 5.95, -0.3, 0.09, jit(GRANITE, 0.01)))
    return out


def upper_facade(a, b, kind, skip_f1=(), centre_aedicule=False, step=3.5, balc1=True):
    """Windows, balconies, string courses and pilaster strips of the three floors between u = a and b."""
    out = []
    cs, bay = bays(a, b, step)
    # continuous balconies on the first and second floors, single ones on the third
    for (sill, h), cont in zip(FLOORS, (balc1, True, False)):
        if cont:
            out.append(B(a + 0.1, b - 0.1, sill - 0.2, sill, 0.0, 0.62, jit(GRANITE, 0.01)))
            out += railing(a + 0.15, b - 0.15, sill, 1.0, 0.58)
    for k, xc in enumerate(cs):
        for fi, (sill, h) in enumerate(FLOORS):
            if fi == 0 and any(lo <= xc <= hi for lo, hi in skip_f1):
                continue
            if centre_aedicule and fi < 2 and k == len(cs) // 2:
                continue
            out += window(xc, sill, h)
            if fi == 2:
                out.append(B(xc - 0.95, xc + 0.95, sill - 0.16, sill, 0.0, 0.48, GRANITE))
                out += railing(xc - 0.9, xc + 0.9, sill, 1.0, 0.45, step=0.3)
            if kind == "pan" and k < len(cs) - 1:                 # colour panels between the windows
                out.append(quad(xc + bay / 2, sill + h / 2, 0.0, bay - 2.0, h + 0.6, "+z", MURALS[int(rng.integers(len(MURALS)))], eps=0.03))
    # granite string course under the cornice and pilaster strips every three bays
    out.append(B(a, b, 16.85, 17.05, 0.0, 0.1, GRANITE))
    if kind != "pan":
        for k in range(0, len(cs) + 1, 3):
            x = a + k * bay
            out.append(B(x - 0.24, x + 0.24, 5.95, 16.85, 0.0, 0.05, jit(GRANITE, 0.015)))
    return out


def dormers(a, b, every):
    out = []
    cs, bay = bays(a, b, 3.5)
    for k, xc in enumerate(cs):
        if k % every:
            continue
        w0, w1 = -0.75, -2.7
        out.append(B(xc - 0.65, xc + 0.65, 18.4, 20.3, w1, w0, SLATE_D))
        out.append(quad(xc, 19.35, w0, 0.9, 1.25, "+z", FRAME, eps=0.02))
        out.append(quad(xc, 19.3, w0, 0.65, 1.05, "+z", GLASS_C if rng.random() < 0.5 else SHUTTER, eps=0.04))
        V = [(xc - 0.78, 20.3, w0 + 0.1), (xc + 0.78, 20.3, w0 + 0.1), (xc, 20.95, w0 + 0.1),
             (xc - 0.78, 20.3, w1), (xc + 0.78, 20.3, w1), (xc, 20.95, w1)]
        out.append(Mesh(V, [(0, 1, 2), (3, 5, 4), (0, 2, 5), (0, 5, 3), (1, 4, 5), (1, 5, 2)], SLATE))
    return out


# ── Towers ───────────────────────────────────────────────────────────────────
def tower(u0, u1, kind):
    """Tower of the Panadería or Carnicería (local frame of its side): body, quoins, slate cap, lantern, spire."""
    pan = kind == "pan"
    body_c = PAINT if pan else RED_C
    top, cap_t, lan_t, spire_t, cross_t = (22.7, 25.2, 28.6, 35.6, 38.3) if pan else (22.5, 24.4, 26.6, 31.8, 33.6)
    w0, w1 = -5.6, 0.3
    xc, zc = (u0 + u1) / 2, (w0 + w1) / 2
    hx, hz = (u1 - u0) / 2, (w1 - w0) / 2
    out = [B(u0, u1, 5.05, top, w0, w1, body_c)]
    for x in (u0, u1 - 0.55):                                           # quoins
        out.append(B(x - 0.02, x + 0.57, 5.95, top - 0.5, w1 - 0.55, w1 + 0.05, GRANITE))
        out.append(B(x - 0.02, x + 0.57, 17.55, top - 0.5, w0 - 0.05, w0 + 0.55, GRANITE))
    for x in (u0 + 0.1, u1 - 0.95):                                     # piers under the projecting tower
        out.append(B(x, x + 0.85, -1.0, 5.1, -0.7, w1 + 0.02, jit(GRANITE)))
    out.append(B(u0 - 0.35, u1 + 0.35, top - 0.55, top + 0.05, w0 - 0.35, w1 + 0.35, GRANITE))   # cornice
    # windows: the three floors of the range, one above the roof on the front, back and sides
    for sill, h in FLOORS:
        out += window(xc, sill, h, z=w1)
        out.append(B(xc - 0.95, xc + 0.95, sill - 0.18, sill, w1, w1 + 0.5, GRANITE))
        out += railing(xc - 0.9, xc + 0.9, sill, 1.0, w1 + 0.47, step=0.3)
        if pan:
            for sx in (-1, 1):
                out.append(quad(xc + sx * 1.9, sill + h / 2, w1, 1.1, h + 0.6, "+z", MURALS[int(rng.integers(len(MURALS)))], eps=0.03))
    out += window(xc, 19.0, 2.2, z=w1)
    out += railing(xc - 0.8, xc + 0.8, 19.0, 0.9, w1 + 0.4, step=0.3)
    back = window(xc, 19.0, 2.2)
    for m in back:
        m.v[:, 2] = w0 - m.v[:, 2]
        m.f = m.f[:, ::-1]
    out += back
    for sgn, ux in ((-1, u0), (1, u1)):                                  # side windows (face ±x)
        for m in window(zc, 19.0, 2.0, w=1.1):
            n = m.copy()
            v = n.v.copy()
            n.v[:, 0] = ux + sgn * v[:, 2]
            n.v[:, 2] = v[:, 0]
            if sgn > 0:
                n.f = n.f[:, ::-1]
            out.append(n)
    # slate cap (a square frustum), clock dormer, lantern, spire, ball and cross
    r0 = math.hypot(hx + 0.45, hz + 0.45)
    capm = cyl(r0, 1.25, cap_t - top, SLATE, 4).R(0, math.pi / 4, 0)
    capm.v[:, 0] *= (hx + 0.45) / (r0 / math.sqrt(2))
    capm.v[:, 2] *= (hz + 0.45) / (r0 / math.sqrt(2))
    out.append(capm.T(xc, top + 0.05, zc))
    for s in (1, -1):                                                    # clock dormers, front and back
        zf = zc + s * (hz - 0.2)
        out.append(B(xc - 0.95, xc + 0.95, top - 0.1, top + 1.9, min(zf, zf - s * 1.6), max(zf, zf - s * 1.6), SLATE_D))
        clk = cyl(0.68, 0.68, 0.08, CLOCK, 20).R(math.pi / 2, 0, 0)
        if s < 0:
            clk.R(0, math.pi, 0)
        out.append(clk.T(xc, top + 0.95, zf))
        out.append(sphere(0.1, IRON, 6, 4).T(xc, top + 0.95, zf + s * 0.09))
    out.append(cyl(1.05, 1.0, lan_t - cap_t, LEAD, 8).T(xc, cap_t, zc))
    for k in range(8):                                                    # openings of the lantern
        a = (k + 0.5) * math.pi / 4
        q = quad(0, (cap_t + lan_t) / 2, 0, 0.42, (lan_t - cap_t) * 0.5, "+z", GLASS_C, eps=0.0)
        q.T(0, 0, 1.0).R(0, -a + math.pi / 2, 0).T(xc, 0, zc)
        out.append(q)
    out.append(cyl(1.3, 1.25, 0.3, SLATE_D, 8).T(xc, lan_t, zc))
    out.append(cyl(1.0, 0.05, spire_t - lan_t - 0.3, SLATE, 8).T(xc, lan_t + 0.3, zc))
    out.append(sphere(0.32, GILT, 10, 6).T(xc, spire_t + 0.25, zc))
    out.append(B(xc - 0.05, xc + 0.05, spire_t + 0.5, cross_t, zc - 0.05, zc + 0.05, IRON))
    out.append(B(xc - 0.45, xc + 0.45, cross_t - 0.75, cross_t - 0.65, zc - 0.05, zc + 0.05, IRON))
    return out


# ── The portico range of one side ────────────────────────────────────────────
def side_range(s):
    """Everything of the 6 m deep range along side s, in its local frame."""
    L = s.L
    out_local, gilt = [], []
    ents = [e for e in ENTRANCES if e[1] == s.i]
    specials = [sp for sp in SPECIAL if sp[0] == s.i]
    big = [(u - w / 2 - JW, u + w / 2 + JW, name, w) for name, _, u, w, b in ents if b]
    # pieces along u: (a, b, kind); the end pieces run to ±inf so that the mitred corners are kept
    cuts = []
    for a, b, *_ in big:
        cuts.append((a, b, "arch"))
    for _, a, b, kind in specials:
        cuts.append((a, b, kind))
    cuts.sort()
    pieces, u = [], -1e9
    for a, b, kind in cuts:
        if a > u:
            pieces.append((u, a, "house"))
        pieces.append((a, b, kind))
        u = b
    pieces.append((u, 1e9, "house"))
    band = trapezoid(s, 0.0, D)
    dy = 0.03 * (s.i % 2)                                   # keeps the corner overlaps of two sides apart
    for pi_, (a, b, kind) in enumerate(pieces):
        P = clip(band, a, b)
        if len(P) < 3:
            continue
        fa, fb = max(a, 0.0), min(b, L)                          # the part of the façade on the square
        if kind == "arch":
            name, w = next((n, ww) for aa, bb, n, ww in big if abs(aa - a) < 1e-6)
            out_local.append(prism(P, 10.0, TOP, edge_colour(RED, jit(OCHRES[1]), RED, 0.0, D), None, SOFFIT))
            x0, x1 = a + JW, b - JW
            ys = 10.0 - 0.45 - w / 2
            out_local.append(arch_block(x0, x1, ys, 10.0, -D, 0.06, GRANITE, 20))
            out_local.append(tube([((x0 + x1) / 2 + (w / 2 + 0.25) * math.cos(t), ys + (w / 2 + 0.25) * math.sin(t), 0.1)
                                   for t in np.linspace(math.pi, 0, 17)], 0.16, GRANITE_D, 6))
            for j0, j1 in ((a, x0), (x1, b)):
                out_local.append(B(j0, j1, F, 10.0, -0.9, 0.06, jit(GRANITE)))           # jambs on the square
                out_local.append(B(j0, j1, F, 10.0, -D, -5.1, jit(GRANITE)))             # jambs on the street
                out_local.append(B(j0, j1, 4.6, 10.0, -5.1, -0.9, SHOP))                  # wall over the portico
                out_local.append(B(j0 - 0.05, j1 + 0.05, ys - 0.2, ys, -0.95, 0.12, GRANITE_D))  # imposts
            out_local += upper_facade(a, b, "house", skip_f1=[(a - 1, b + 1)], step=3.5, balc1=False)
            for z_ in (-D,):                                                             # windows on the street
                for xc in bays(a, b, 3.4)[0]:
                    for sill, h in FLOORS[1:]:
                        for m in window(xc, sill, h, frame=FRAME):
                            m.v[:, 2] = z_ - m.v[:, 2]
                            m.f = m.f[:, ::-1]
                            out_local.append(m)
            continue
        pan, carn = kind == "pan", kind == "carn"
        face_c = PAINT if pan else (RED_C if carn else RED)
        bottom = 5.0 if (pan or carn) else 4.6
        out_local.append(prism(P, bottom, TOP, edge_colour(face_c, jit(OCHRES[0]), RED, 0.0, D), None, SOFFIT))
        if pan or carn:
            ts = [(u0, u1) for si, u0, u1, k in TOWERS if si == s.i and k == kind]
            mid0, mid1 = ts[0][1], ts[1][0]
            cs, bay = bays(mid0, mid1, 3.5)
            openings = [((u0 + u1) / 2, 2.6) for u0, u1 in ts] + [(c, bay - 1.05) for c in cs]
            out_local += portico_arches(fa, fb, sorted(openings))
            out_local += upper_facade(mid0, mid1, kind, centre_aedicule=pan)
            for u0, u1 in ts:
                out_local += tower(u0 + 0.03, u1 - 0.03, kind)
            out_local += dormers(mid0, mid1, 1)
            if pan:                                   # the central aedicule with the royal arms, and the crown
                xc = (mid0 + mid1) / 2
                out_local += [B(xc - 1.45, xc + 1.45, 5.95, 13.7, 0.0, 0.16, GRANITE),
                              B(xc - 1.15, xc + 1.15, 6.4, 13.3, 0.16, 0.22, PAINT),
                              cyl(0.75, 0.75, 0.12, GRANITE_D, 16).R(math.pi / 2, 0, 0).S(1, 1.3, 1).T(xc, 9.9, 0.34),
                              B(xc - 1.6, xc + 1.6, 13.65, 13.92, 0.0, 0.3, GRANITE),
                              B(xc - 0.9, xc + 0.9, 18.0, 19.2, -0.75, -0.45, GRANITE)]
                gilt += [cyl(0.55, 0.75, 0.55, GILT, 12).T(xc, 19.2, -0.6),
                         lathe([(0.75, 0), (0.45, 0.5), (0.2, 0.8), (0, 0.85)], GILT, 12).T(xc, 19.75, -0.6),
                         sphere(0.13, GILT, 8, 5).T(xc, 20.65, -0.6)]
        else:
            has_l = pi_ == 0 or pieces[pi_ - 1][2] == "house"
            has_r = pi_ == len(pieces) - 1 or pieces[pi_ + 1][2] == "house"
            out_local += portico_house(fa, fb, has_l, has_r, dy)
            out_local += upper_facade(fa + 0.75, fb - 0.75, "house")
            out_local += dormers(fa + (3.0 if pi_ == 0 else 0.0), fb - (3.0 if pi_ == len(pieces) - 1 else 0.0), 2)
    # the back wall of the portico (shop fronts), interrupted by every passage
    strip = trapezoid(s, PD, D + 0.3)
    gaps = sorted((u - w / 2 - (JW if b else 0.0), u + w / 2 + (JW if b else 0.0)) for _, _, u, w, b in ents)
    u, segs = -1e9, []
    for a, b in gaps:
        segs.append((u, a))
        u = b
    segs.append((u, 1e9))
    for a, b in segs:
        P = clip(strip, a, b)
        if len(P) >= 3:
            out_local.append(prism(P, F, 4.75, edge_colour(SHOP, OCHRES[0], GRANITE_D, PD, D + 0.3), None))
            for xc in bays(max(a, 0.5), min(b, L - 0.5), 3.5)[0]:
                out_local.append(quad(xc, 1.5, -PD, 1.9, 3.0, "+z", DOOR, eps=0.03))
    # the iron railing along the eaves
    out_local.append(B(0.6, L - 0.6, EAVE + 0.85, EAVE + 0.9, 0.42, 0.47, IRON))
    for k in range(1, int(L / 0.9)):
        out_local.append(quad(k * 0.9, EAVE + 0.45, 0.445, 0.04, 0.9, "+z", IRON, eps=0.0))
    return s.world(merge(out_local)), s.world(merge(gilt)) if gilt else None


# ── Roofs, cornice ───────────────────────────────────────────────────────────
def roofs():
    cin, cout = offset_quad(-0.55), offset_quad(D + 0.35)
    out = []
    # cornice ring (17.4–18.0)
    out.append(prism(cout, 17.4, EAVE, lambda *a: GRANITE, None, None))
    out.append(prism(cin, 17.4, EAVE, lambda *a: GRANITE, None, None))
    from meshkit import cap as _cap
    out.append(_cap(cout, EAVE, GRANITE, holes=[cin], up=True))
    out.append(_cap(cout, 17.4, GRANITE, holes=[cin], up=False))
    e_in, ridge, e_out = offset_quad(-0.45), offset_quad(3.0), offset_quad(D + 0.3)
    for i in range(4):
        j = (i + 1) % 4
        for p0, p1, y0, y1, c in ((e_in, ridge, EAVE, RIDGE, SLATE), (ridge, e_out, RIDGE, EAVE, jit(SLATE, 0.02))):
            V = [(p0[i][0], y0, p0[i][1]), (p0[j][0], y0, p0[j][1]), (p1[j][0], y1, p1[j][1]), (p1[i][0], y1, p1[i][1])]
            m = Mesh(V, [(0, 1, 2), (0, 2, 3)], c)
            n = np.cross(np.subtract(V[1], V[0]), np.subtract(V[2], V[0]))
            if n[1] < 0:
                m.f = m.f[:, ::-1]
            out.append(m)
    # gable ends are not needed (the ring closes); chimneys on the ridge
    for s in SIDES:
        for u in np.arange(9.0, s.L - 5, 13.0):
            if any(si == s.i and u0 - 2 < u < u1 + 2 for si, u0, u1, _ in TOWERS):
                continue
            ch = B(u - 0.45, u + 0.45, RIDGE - 1.0, RIDGE + 1.0, -3.5, -2.6, jit(OCHRES[3], 0.05))
            cp = B(u - 0.55, u + 0.55, RIDGE + 1.0, RIDGE + 1.15, -3.6, -2.5, GRANITE_D)
            out.append(s.world(merge([ch, cp])))
    return merge(out)


# ── The houses behind the range ──────────────────────────────────────────────
def houses():
    out = []
    for k, pc in enumerate(DATA["pieces"]):
        P = [tuple(p) for p in pc["P"]]
        name = pc["name"]
        special = name.startswith("Casa de la")
        cuch = name == "Arco de Cuchilleros"
        h = 17.35 if special else float(rng.uniform(16.4, 17.25))
        wall = jit(RED if special else OCHRES[k % len(OCHRES)], 0.02)
        roof = SLATE if special else TILE[k % len(TILE)]
        y0 = 3.0 if cuch else F
        if ring_area(P) < 0:
            P = P[::-1]
        holes = [[tuple(p) for p in hh] for hh in pc["holes"]]
        out.append(prism(P, y0, h, lambda i, a, b: wall, None, SOFFIT if cuch else None))
        out.append(cap(P, h, roof, holes=holes, up=True))
        for hh in holes:
            out.append(prism(hh, F, h, lambda i, a, b: jit(OCHRES[1], 0.02), hole=True))
        # a low parapet/eave line
        for i in range(len(P)):
            a, b = np.array(P[i]), np.array(P[(i + 1) % len(P)])
            L = float(np.linalg.norm(b - a))
            if pc["win"][i] != "1" or L < 2.2:
                continue
            t = (b - a) / L
            n = np.array([t[1], -t[0]])                           # outward for a counter-clockwise ring
            mid = (a + b) / 2
            fy = float(found(mid[0], mid[1]))
            nwin = max(1, int(L / 3.2))
            for kk in range(nwin):
                c = a + t * (L * (kk + 0.5) / nwin)
                for sill in np.arange(0.6 - 3.4 * 3, h - 2.6, 3.4):
                    if sill < fy + 0.8 or (cuch and sill < 3.0):
                        continue
                    hh_ = 2.6 if sill < 1 and sill > 0 else 2.0
                    col = DOOR if 0 < sill < 1 else (SHUTTER if rng.random() < 0.4 else GLASS_C)
                    for w_, h2, cc, e in ((1.45, hh_ + 0.25, FRAME, 0.03), (1.1, hh_, col, 0.06)):
                        V = [c - t * w_ / 2 + n * e, c + t * w_ / 2 + n * e]
                        yb = sill - (h2 - hh_) / 2
                        quadv = [(V[0][0], yb, V[0][1]), (V[1][0], yb, V[1][1]), (V[1][0], yb + h2, V[1][1]), (V[0][0], yb + h2, V[0][1])]
                        out.append(Mesh(quadv, [(0, 2, 1), (0, 3, 2)], cc))
            # eave
            e0, e1 = a - t * 0.1 + n * 0.0, b + t * 0.1 + n * 0.0
            V = [(e0[0], h - 0.35, e0[1]), (e1[0], h - 0.35, e1[1]), (e1[0] + n[0] * 0.45, h, e1[1] + n[1] * 0.45), (e0[0] + n[0] * 0.45, h, e0[1] + n[1] * 0.45)]
            out.append(Mesh(V, [(0, 2, 1), (0, 3, 2)], GRANITE_D))
        if cuch:
            out += cuchilleros_arch()
    return merge(out)


def cuchilleros_arch():
    """The granite arch over the Escalerilla de Piedra on Calle de Cuchilleros (south end of the passage)."""
    x0, x1, z0, z1 = -60.3, -57.6, 59.95, 61.05
    ys = -0.2                                   # opening about 8 m high from the street, some 6.5 m below the square
    return [arch_block(x0, x1, ys, 3.0, z0, z1, GRANITE, 20),
            B(-61.25, x0, F, 3.0, z0, z1, jit(GRANITE)), B(x1, -56.65, F, 3.0, z0, z1, jit(GRANITE)),
            B(-61.3, -56.6, 3.0, 4.4, z0, z1 + 0.05, GRANITE),
            tube([((x0 + x1) / 2 + 1.6 * math.cos(t), ys + 1.6 * math.sin(t), z1 + 0.04) for t in np.linspace(math.pi, 0, 15)], 0.13, GRANITE_D, 6)]


# ── The square: paving, the statue of Felipe III, café terraces ─────────────
def centre():
    return sum(q[0] for q in Q) / 4, sum(q[1] for q in Q) / 4


def at(u, w):
    """A point of the square from the statue: u metres east, w metres south (along the sides of the square)."""
    cx, cz = centre()
    t = SIDES[0].t                                                # the north side, west to east
    n = np.array([-t[1], t[0]])                                   # southwards
    return float(cx + u * t[0] + w * n[0]), float(cz + u * t[1] + w * n[1])


def statue():
    """Equestrian statue of Felipe III on its pedestal, simplified (heights estimated), in the middle."""
    cx, cz = centre()
    out = [B(-4.25, 4.25, 0.0, 0.3, -2.75, 2.75, GRANITE), B(-3.3, 3.3, 0.3, 0.75, -2.0, 2.0, GRANITE_D),
           B(-2.9, 2.9, 0.75, 1.1, -1.65, 1.65, GRANITE), B(-2.5, 2.5, 1.1, 4.0, -1.3, 1.3, GRANITE),
           B(-2.75, 2.75, 4.0, 4.5, -1.55, 1.55, GRANITE_D)]
    y0 = 4.5
    horse = [sphere(1.0, BRONZE).S(1.55, 0.7, 0.62).T(0, y0 + 2.15, 0)]                # body
    for lx, lz, k in ((1.05, 0.33, 0.0), (1.05, -0.33, 0.55), (-1.05, 0.33, 0.0), (-1.05, -0.33, 0.0)):
        leg = cyl(0.13, 0.1, 1.75, BRONZE, seg=8)
        if k:                                                                       # the left foreleg, raised
            leg = cyl(0.13, 0.1, 1.2, BRONZE, seg=8).R(0, 0, k).T(0, 0.55, 0)
        horse.append(leg.T(lx, y0, lz))
    horse += [cyl(0.42, 0.3, 1.5, BRONZE, seg=10).R(0, 0, -0.75).T(1.3, y0 + 2.2, 0),   # neck
              sphere(0.5, BRONZE).S(1.4, 0.75, 0.55).T(2.25, y0 + 3.15, 0),            # head
              cyl(0.12, 0.05, 1.3, BRONZE, seg=6).R(0, 0, 2.5).T(-1.45, y0 + 2.4, 0)]  # tail
    rider = [cyl(0.42, 0.33, 1.2, BRONZE, seg=10).T(0.1, y0 + 2.75, 0),                # body
             sphere(0.23, BRONZE).T(0.15, y0 + 4.2, 0),                                 # head
             cyl(0.17, 0.12, 1.05, BRONZE, seg=8).R(0.35, 0, 0).T(0.05, y0 + 2.0, 0.42),  # legs
             cyl(0.17, 0.12, 1.05, BRONZE, seg=8).R(-0.35, 0, 0).T(0.05, y0 + 2.0, -0.42),
             cyl(0.05, 0.05, 0.8, BRONZE, seg=6).R(0, 0, -0.5).T(0.25, y0 + 3.0, 0.45)]   # baton at the waist
    return merge(out + horse + rider).T(cx, 0, cz)


def lamp(u, w):
    """One of the four big lampposts: round stone base, cast-iron column, four arms with lanterns (generic)."""
    x, z = at(u, w)
    out = [cyl(1.9, 1.9, 0.35, GRANITE_D, seg=20), cyl(1.4, 1.4, 0.55, GRANITE, seg=20),
           cyl(0.32, 0.22, 1.2, IRON, seg=10).T(0, 0.55, 0), cyl(0.16, 0.11, 4.6, IRON, seg=8).T(0, 1.75, 0)]
    for k in range(4):
        a = k * math.pi / 2 + math.pi / 4
        ex, ez = math.cos(a) * 0.95, math.sin(a) * 0.95
        out += [tube([(0, 5.4, 0), (ex * 0.6, 5.75, ez * 0.6), (ex, 5.7, ez)], 0.05, IRON, 5),
                cyl(0.17, 0.24, 0.55, (0.95, 0.92, 0.80), seg=6).T(ex, 5.15, ez),       # lantern glass
                cyl(0.27, 0.02, 0.28, IRON, seg=6).T(ex, 5.7, ez)]                      # its cap
    out += [cyl(0.2, 0.02, 0.6, IRON, seg=6).T(0, 6.35, 0)]
    return merge(out).T(x, 0, z)


# Groups of terrace parasols seen on the orthophoto: (u0, u1, w0, w1) in metres from the statue (east, south)
TERRACES = [(-60, -48, -37, -30.5), (-46, -39.5, -37, -30.5), (-39, -21.5, -37, -30.5), (36, 41.5, -42, -36.5),
            (45.5, 56, -42, -36.5),
            (-59, -46, -19.5, -8.5), (-58, -45, -5, 6), (-57.5, -45.5, 8.5, 30),
            (45.5, 58, -27, -20.5), (45.5, 58, -20.5, -9.5), (46, 58.5, -8, 2.5), (47, 59, 3, 15.5), (48, 59, 15.5, 21.5),
            (-55.5, -28.5, 27.5, 37), (-18.5, -8, 27.5, 37), (23.5, 57.5, 27.5, 37)]


def terraces():
    """Café terraces against the porticoes: large square white parasols in the groups of the orthophoto, a table
    and chairs under each (illustrative: they change with the season)."""
    out = []
    for u0, u1, w0, w1 in TERRACES:
        nu, nw = max(1, round((u1 - u0) / 5.6)), max(1, round((w1 - w0) / 5.6))
        su, sw = (u1 - u0) / nu, (w1 - w0) / nw
        for i in range(nu):
            for j in range(nw):
                u, w = u0 + (i + 0.5) * su, w0 + (j + 0.5) * sw
                x, z = at(u, w)
                half = min(su, sw) / 2 - 0.15
                out += [cyl(0.05, 0.05, 2.9, TABLE, seg=6).T(x, 0, z),
                        cyl(half * 1.414, 0.05, 0.5, PARASOL, seg=4).R(0, math.pi / 4, 0).T(x, 2.55, z),   # canopy
                        cyl(half * 1.414, half * 1.414, 0.12, PARASOL, seg=4, caps=False).R(0, math.pi / 4, 0).T(x, 2.43, z)]
                for dx, dz in ((-1.2, -1.1), (1.2, 1.1)):
                    tx, tz = x + dx, z + dz
                    out += [cyl(0.4, 0.4, 0.05, TABLE, seg=10).T(tx, 0.72, tz), cyl(0.04, 0.04, 0.72, TABLE, seg=5).T(tx, 0, tz),
                            B(tx - 0.8, tx - 0.4, 0, 0.45, tz - 0.2, tz + 0.2, TABLE), B(tx + 0.4, tx + 0.8, 0, 0.45, tz - 0.2, tz + 0.2, TABLE)]
    return merge(out)


def square():
    """The paving of 1961 (orthophoto): granite fields, light limestone bands, a grid of dark porphyry bands; the
    four lampposts and the two grilles of the car park."""
    out = [extrude(offset_quad(PD), -0.8, 0.0, PAVE_D, top_c=PAVE)]
    U, Wd = 60.2, 42.0                                             # half-size of the open square from the statue
    def band(u0, u1, w0, w1, y, c):
        P = [at(u0, w0), at(u1, w0), at(u1, w1), at(u0, w1)]
        out.append(cap(P if ring_area(P) > 0 else P[::-1], y, c))
    for k in range(-3, 4):                                         # light bands, halfway (and through the statue)
        band(k * 16.3 - 0.6, k * 16.3 + 0.6, -Wd, Wd, 0.008, PAVE_L)
        band(-U, U, k * 11.5 - 0.6, k * 11.5 + 0.6, 0.009, PAVE_L)
    for k in range(-4, 4):                                         # dark bands, between them (±8.15 … ±57.05)
        u = (k + 0.5) * 16.3
        if abs(u) < U:
            band(u - 1.35, u + 1.35, -Wd, Wd, 0.014, PAVE_D)
    for k in range(-4, 4):
        w = (k + 0.5) * 11.5
        if abs(w) < Wd:
            band(-U, U, w - 1.4, w + 1.4, 0.015, PAVE_D)
    band(1.2, 7.8, 12.05, 15.55, 0.02, GRILLE)                    # the car park's grilles (orthophoto)
    band(25.45, 31.75, 11.95, 15.45, 0.02, GRILLE)
    for u in (-24.45, 24.45):
        for w in (-17.25, 17.25):
            out.append(lamp(u, w))
    return merge(out)


# ── Footprint: the whole outline, the square included (its floor is part of the model now) ─
def footprint():
    outer = [tuple(p) for p in DATA["fp"]]
    if ring_area(outer) < 0:
        outer = outer[::-1]
    return [[round(x, 2), round(z, 2)] for x, z in outer]


def main():
    out = ROOT / "models" / "Q1123493"
    out.mkdir(parents=True, exist_ok=True)
    parts, gilt = [], []
    for s in SIDES:
        m, g = side_range(s)
        parts.append(m)
        if g is not None:
            gilt.append(g)
    parts.append(roofs())
    parts.append(houses())
    parts += [square(), statue(), terraces()]
    stone = merge(parts)
    gold = merge(gilt)
    for m in (stone, gold):
        sel = np.isclose(m.v[:, 1], F)
        if sel.any():
            m.v[sel, 1] = found(m.v[sel, 0], m.v[sel, 2])
    tris, size = write_glb(out / "model.glb", [("plaza_mayor", stone), ("gilt", gold, {"material": "gilt"})],
                           roughness=0.9, materials={"gilt": METAL})
    lo, hi = stone.v.min(0), stone.v.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    meta = out / "meta.json"
    if meta.exists():                                     # keep the footprint of meta.json in step with the plan
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["footprint"] = footprint()
        mj["triangles"] = tris
        meta.write_text(json.dumps(mj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"meta.json: footprint {len(mj['footprint'])} points, triangles {tris}")


if __name__ == "__main__":
    main()
