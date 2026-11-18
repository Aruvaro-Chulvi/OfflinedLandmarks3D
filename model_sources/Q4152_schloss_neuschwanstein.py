"""
Schloss Neuschwanstein (Q4152), Hohenschwangau, Bavaria — the castle on its rock spur, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: designed by the stage painter Christian Jank (died 1888) and the architects Eduard Riedel (died
1885), Georg von Dollmann (died 1895) and Julius Hofmann (died 1896) for King Ludwig II; built 1869–1892
(gatehouse 1869–1873, Palas 1869–1884, Kemenate finished 1892), never completed (the 90 m keep in the upper
courtyard was not built). Public domain.

What is modelled:
- The Palas (the five-storey hall building at the west end of the ridge): its two blocks with their steep slate
  roofs (the west block, of the Throne Hall, turned ~13° to follow the rock), the gables of the west front and
  of the courtyard (east) front with its two slender corner turrets, the projecting oriel of the west front,
  the Wintergarten, the north projection and the south annex, rows of round-arched windows (the tall paired
  windows of the Throne Hall and the Singers' Hall floors), cornices and gilded finials.
- The towers: the north stair tower (Treppenturm, 65 m: the broad lower shaft with its gallery, the slender
  round upper turret with its spire and the small stair turret beside it), the round south tower of the Palas
  (52 m), the Square Tower (Viereckturm, 45 m: corbelled crenellated platform, round turret and spire), the
  round turret of the Knights' House and the small tower between the Kemenate and the Palas.
- The Knights' House (Ritterhaus) and its long gallery along the north side of the upper courtyard; the
  Bower (Kemenate) on the south side, with its cross wing and bay; the Gatehouse (Torbau) at the east, in red
  brick with its light stone gate front, its central gable and its two round corner towers, and the north
  wing of the lower courtyard.
- The upper courtyard (Oberer Hof) and the lower courtyard (Unterer Hof) as paved platforms, the raised
  terrace in front of the Palas, the stair between the courtyards, the retaining wall between them, the
  curtain wall along the south side with its parapet, and the rock spur: a simple rock cliff below the castle
  (only where it is the castle's base), whose top is the ledge around the buildings.
- Not modelled: the interiors, the terrain around, the Marienbrücke, the roads.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap ways 221599090 (Palas), 223311650 (Ritterhaus), 221599087 (Kemenate), 221599096
  (Torbau), 221599098 (Viereckturm), 221600217/221600218 (lower and upper courtyards), 221599101 (the
  courtyard platform), 221599099 (castle wall), 221599104 (retaining wall), 221601969 (the castle's area on the
  rock), and ~40 building:part ways with their heights, levels and roof shapes (© OpenStreetMap contributors,
  ODbL). The castle is ~150 m long (de.wikipedia: "rund 150 Meter"); the long axis of the courtyard buildings
  runs at 66.6°/246.6° (length-weighted histogram of the edges).
- Heights: OpenStreetMap parts (Treppenturm 65 m with a 12 m spire, Viereckturm 45 m, Palas 45–46 m to the
  ridges with 12 m roofs, south tower 52 m, Knights' House 31 m, Kemenate 18 m, Torbau 22 m) agree with
  de.wikipedia (Treppenturm 65 m, Viereckturm 45 m, Palas five storeys, Kemenate and Ritterhaus three
  storeys). They are taken above the upper courtyard, except the Torbau and the north wing of the lower
  courtyard, taken above the lower courtyard. Roof pitches and storey heights estimated (±10 %).
- Ground: EU-DEM 25 m (Copernicus, via OpenTopoData) gives ≈949 m in the upper courtyard, ≈940 m at the lower
  courtyard and ≈934–936 m in front of the gatehouse; the 25 m model smooths the rock, so the levels of the
  castle are: y = 0 the upper courtyard; the lower courtyard 6 m lower (y = −6, the 6 m of the OSM courtyard
  platform 221599101; estimate ±2 m); the ledge of rock around the walls at y = −7.5; the rock cliff falls to
  y = −42, flaring out, so that no gap shows where the terrain is lower (it is ~15 m lower at the gatehouse).
- Materials (de.wikipedia): brick structure faced with white limestone (Alter Schrofen quarry), the courtyard
  façades in yellowish limestone, the gatehouse in red brick, portals and oriels in sandstone, slate roofs
  (dark blue-grey), gilded finials.

Axes (docs/MODEL_SPEC.md): +Y up, the west front of the Palas faces +Z (heading 246.6°, towards the
Alpsee/Pöllat side: the Throne Hall end of the castle), +X points to 156.6° (the south side seen from the
Marienbrücke). The famous view from the Marienbrücke (OSM way 41720896, ~290 m due south, bearing ≈184°) is
at azimuth ≈63° from the +Z front: it shows the west front and the south side. Origin on the ground (y = 0 =
upper courtyard) at the centroid of OSM way 221601969.

    python model_sources/Q4152_schloss_neuschwanstein.py   -> models/Q4152/model.glb (+ footprint in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, cap, walls, triangulate, ring_area, offset, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(1869)
HEADING = 246.6

# ── Colours ──────────────────────────────────────────────────────────────────
LIME = (0.905, 0.885, 0.835)        # white limestone (Alter Schrofen)
LIME_Y = (0.885, 0.825, 0.66)       # yellowish limestone of the courtyard façades
LIME_L = (0.95, 0.94, 0.90)         # cornices, frames, string courses
LIME_D = (0.76, 0.74, 0.69)         # plinths, foundations
SAND = (0.88, 0.80, 0.62)           # sandstone: gate front, gatehouse towers, oriels
BRICK = (0.66, 0.29, 0.21)          # red brick of the gatehouse
SLATE = (0.27, 0.30, 0.37)          # slate roofs, dark blue-grey
SLATE_L = (0.33, 0.36, 0.43)        # ridges, cone tips
GLASS = (0.09, 0.10, 0.12)
DARK = (0.06, 0.06, 0.07)           # gate passage, open arcades
DOOR = (0.30, 0.20, 0.12)
GOLD = (0.85, 0.66, 0.25)
PAVING = (0.74, 0.72, 0.67)
ROCK = (0.62, 0.60, 0.55)
ROCK_D = (0.50, 0.49, 0.45)
GRASS = (0.38, 0.46, 0.27)
LEDGE_C = (0.47, 0.50, 0.38)        # the rock top around the walls: thin grass on limestone

UP = 0.0          # upper courtyard
LOW = -6.0        # lower courtyard
LEDGE = -7.5      # rock top around the buildings
F_UP = -8.6       # foundations of the buildings on the upper level
F_LOW = -12.0     # foundations of the gatehouse and the lower courtyard

# ── Data (OpenStreetMap © OpenStreetMap contributors, ODbL), metres in the model axes ───────────────────
D = json.loads(r"""{"221599083":[[-12.39,-56.3],[-11.9,-55.53],[-11.77,-54.61],[-11.96,-53.85],[-12.41,-53.2],[-13.05,-52.75],[-13.8,-52.56],[-14.74,-52.7],[-15.52,-53.2],[-15.99,-53.9],[-16.16,-54.74],[-15.99,-55.59],[-15.51,-56.32],[-14.79,-56.79],[-13.95,-56.96],[-13.1,-56.78]],"221599084":[[23.43,-56.61],[23.82,-56.06],[24.05,-55.16],[23.89,-54.26],[23.45,-53.59],[22.87,-53.16],[22.21,-52.94],[21.51,-52.95],[20.92,-53.14],[20.42,-53.49],[20.04,-53.96],[19.8,-54.51],[19.74,-55.24],[19.89,-55.91],[20.3,-56.55],[20.9,-57.0],[21.41,-57.19],[21.94,-57.25],[22.75,-57.07]],"221599086":[[-13.84,-30.44],[-13.85,-48.37],[-10.59,-48.38],[-10.58,-32.51],[-10.59,-30.46]],"221599087":[[11.97,12.31],[10.71,12.31],[10.72,6.99],[11.99,7.0],[12.02,-0.89],[16.73,-0.89],[16.73,-2.63],[20.43,-2.63],[21.72,-2.63],[21.7,7.0],[22.81,6.98],[24.17,8.48],[24.16,11.07],[22.84,12.41],[21.68,12.32],[19.41,12.32],[19.39,18.5],[16.69,18.5],[16.68,21.22],[14.39,21.22],[11.95,21.22],[11.97,16.02]],"221599090":[[-10.34,56.96],[-10.33,52.46],[-7.02,52.45],[-7.03,38.77],[-13.08,38.79],[-13.08,34.35],[-10.88,34.34],[-10.84,25.39],[-6.99,25.38],[-6.99,24.36],[-7.04,24.37],[-7.42,24.23],[-7.7,23.94],[-7.81,23.69],[-7.81,23.38],[-7.7,23.04],[-7.38,22.72],[-6.9,22.59],[-6.51,22.7],[-6.19,23.01],[-6.06,23.33],[-6.07,23.37],[-5.22,23.38],[-3.45,23.36],[6.98,23.34],[6.98,23.23],[7.09,22.88],[7.41,22.56],[7.9,22.44],[8.29,22.56],[8.62,22.86],[8.74,23.18],[8.73,23.53],[8.57,23.86],[8.3,24.1],[8.1,24.17],[8.11,27.94],[9.96,27.94],[11.94,27.93],[11.93,38.69],[8.11,38.71],[8.11,53.08],[8.59,52.28],[9.4,51.64],[10.41,51.34],[11.44,51.42],[12.37,51.87],[13.09,52.62],[13.47,53.58],[13.49,54.62],[13.13,55.6],[12.44,56.38],[12.03,56.6],[16.25,76.56],[10.87,77.84],[11.17,79.31],[0.48,81.43],[0.19,79.96],[-5.2,81.17],[-8.68,64.46],[-11.73,65.03],[-14.34,63.23],[-14.92,60.1],[-13.11,57.48]],"221599096":[[21.48,-48.32],[10.82,-48.36],[10.81,-47.35],[7.51,-47.36],[3.9,-47.39],[-0.1,-47.4],[-3.32,-47.42],[-3.31,-48.33],[-7.07,-48.35],[-8.64,-48.36],[-10.59,-48.38],[-13.85,-48.37],[-13.8,-52.56],[-14.74,-52.7],[-15.52,-53.2],[-15.99,-53.9],[-16.16,-54.74],[-15.99,-55.59],[-15.51,-56.32],[-14.79,-56.79],[-13.95,-56.96],[-13.1,-56.78],[-12.39,-56.3],[-11.9,-55.53],[-11.77,-54.61],[-3.27,-54.57],[-3.27,-55.21],[-0.04,-55.19],[-0.04,-56.47],[3.94,-56.45],[7.57,-56.43],[7.56,-55.16],[10.86,-55.15],[10.86,-54.56],[19.8,-54.51],[19.74,-55.24],[19.89,-55.91],[20.3,-56.55],[20.9,-57.0],[21.41,-57.19],[21.94,-57.25],[22.75,-57.07],[23.43,-56.61],[23.82,-56.06],[24.05,-55.16],[23.89,-54.26],[23.45,-53.59],[22.87,-53.16],[22.21,-52.94],[21.51,-52.95],[21.5,-52.29],[21.5,-51.47]],"221599097":[[-11.94,63.91],[-12.5,63.66],[-13.0,63.31],[-13.43,62.85],[-13.74,62.32],[-13.94,61.74],[-14.0,61.13],[-13.95,60.51],[-13.76,59.92],[-13.46,59.38],[-13.05,58.91],[-12.56,58.55],[-12.0,58.29],[-11.39,58.16],[-10.78,58.15],[-10.17,58.27],[-9.61,58.51],[-9.11,58.87],[-8.69,59.33],[-8.37,59.86],[-8.18,60.44],[-8.11,61.05],[-8.16,61.67],[-8.36,62.26],[-8.65,62.8],[-9.06,63.26],[-9.56,63.63],[-10.12,63.88],[-10.72,64.02],[-11.33,64.03]],"221599098":[[-15.05,-21.43],[-15.06,-30.44],[-13.84,-30.44],[-10.59,-30.46],[-5.75,-30.46],[-5.75,-25.8],[-5.75,-21.46],[-6.95,-21.45],[-10.8,-21.44]],"221599099":[[20.43,-2.63],[20.74,-10.8],[21.8,-11.53],[21.96,-21.09],[28.15,-21.13],[28.1,-35.49],[28.08,-40.1],[31.25,-40.11],[31.26,-43.26],[28.09,-43.27],[28.17,-49.96],[27.83,-51.04],[26.85,-51.94],[25.69,-52.33],[21.5,-52.29]],"221599101":[[21.96,-21.09],[13.32,-21.1],[13.33,-25.76],[9.35,-25.77],[6.02,-29.36],[0.98,-29.29],[-2.71,-25.73],[-5.75,-25.8],[-5.75,-21.46],[-6.95,-21.45],[-6.93,5.47],[-6.99,23.38],[-6.07,23.37],[-5.22,23.38],[-3.45,23.36],[6.98,23.34],[8.1,23.33],[8.02,15.95],[9.99,15.99],[11.97,16.02],[11.97,12.31],[10.71,12.31],[10.72,6.99],[11.99,7.0],[12.02,-0.89],[16.73,-0.89],[16.73,-2.63],[20.43,-2.63],[20.74,-10.8],[21.8,-11.53]],"221599102":[[-3.45,23.36],[-3.46,5.46]],"221599100":[[8.02,15.95],[8.1,23.33]],"221599103":[[-7.07,-48.35],[-7.03,-35.65],[-7.03,-34.43]],"221599104":[[23.39,-35.46],[23.42,-25.72],[13.33,-25.76],[9.35,-25.77],[6.02,-29.36],[0.98,-29.29],[-2.71,-25.73],[-5.75,-25.8]],"221600217":[[28.1,-35.49],[28.08,-40.1],[31.25,-40.11],[31.26,-43.26],[28.09,-43.27],[28.17,-49.96],[27.83,-51.04],[26.85,-51.94],[25.69,-52.33],[21.5,-52.29],[21.5,-51.47],[21.48,-48.32],[10.82,-48.36],[10.81,-47.35],[3.89,-47.38],[-3.32,-47.42],[-3.31,-48.33],[-7.07,-48.35],[-8.64,-48.36],[-10.59,-48.38],[-10.58,-32.51],[-10.59,-30.46],[-5.75,-30.46],[-5.75,-25.8],[-2.71,-25.73],[0.98,-29.29],[6.02,-29.36],[9.35,-25.77],[13.33,-25.76],[23.42,-25.72],[23.39,-35.46],[25.75,-35.48]],"221600218":[[-5.75,-25.8],[-5.75,-21.46],[-6.95,-21.45],[-6.93,5.47],[-6.99,23.38],[-6.07,23.37],[-5.22,23.38],[-3.45,23.36],[6.98,23.34],[8.1,23.33],[8.1,24.17],[8.11,27.94],[9.96,27.94],[11.94,27.93],[11.97,16.02],[11.97,12.31],[10.71,12.31],[10.72,6.99],[11.99,7.0],[12.02,-0.89],[16.73,-0.89],[16.73,-2.63],[20.43,-2.63],[20.74,-10.8],[21.8,-11.53],[21.96,-21.09],[28.15,-21.13],[28.1,-35.49],[25.75,-35.48],[23.39,-35.46],[23.42,-25.72],[13.33,-25.76],[9.35,-25.77],[6.02,-29.36],[0.98,-29.29],[-2.71,-25.73]],"221601969":[[-10.33,85.92],[20.39,79.92],[17.93,57.9],[21.37,33.67],[31.76,11.61],[29.25,-7.93],[32.28,-30.69],[33.54,-51.3],[31.63,-58.36],[22.69,-61.6],[9.98,-62.33],[5.44,-72.39],[-11.56,-73.72],[-21.77,-65.91],[-29.4,-43.48],[-34.2,1.53],[-27.29,22.36],[-21.58,43.76],[-13.85,74.44]],"223069511":[[-18.9,-0.67],[-18.69,0.09],[-18.34,0.56],[-18.21,0.71],[-17.52,1.1],[-16.74,1.2],[-15.98,0.99],[-15.38,0.55],[-14.97,-0.18],[-14.87,-0.97],[-15.07,-1.73],[-15.38,-2.12],[-15.55,-2.35],[-16.24,-2.74],[-17.02,-2.84],[-17.79,-2.63],[-18.41,-2.14],[-18.8,-1.46]],"223069512":[[8.11,38.71],[8.11,27.94],[9.96,27.94],[11.94,27.93],[11.93,38.69]],"223069513":[[-10.84,25.39],[-10.83,20.52],[-10.83,14.31],[-10.83,10.58],[-10.81,0.55],[-10.81,-6.04],[-10.8,-21.44],[-6.95,-21.45],[-6.99,23.38],[-6.99,24.36],[-6.99,25.38]],"223069514":[[-7.97,-29.55],[-7.0,-28.59],[-6.37,-27.38],[-6.14,-26.04],[-6.34,-24.69],[-6.95,-23.46],[-7.91,-22.49],[-9.11,-21.85],[-10.45,-21.62],[-11.81,-21.82],[-13.02,-22.43],[-13.99,-23.38],[-14.62,-24.59],[-14.84,-25.94],[-14.65,-27.29],[-14.04,-28.51],[-13.08,-29.48],[-11.87,-30.12],[-10.54,-30.35],[-9.18,-30.15]],"223069515":[[-15.84,-20.74],[-15.84,-31.35],[-4.97,-31.38],[-4.97,-20.77]],"223071829":[[11.94,27.93],[11.95,21.22],[14.39,21.22],[16.68,21.22],[16.68,23.85],[16.68,27.92]],"223110818":[[-3.31,-48.33],[-7.07,-48.35],[-8.64,-48.36],[-10.59,-48.38],[-13.85,-48.37],[-13.8,-52.56],[-13.8,-54.62],[-11.77,-54.61],[-3.27,-54.57],[-0.05,-54.56],[-0.1,-48.32]],"223110819":[[-5.2,81.17],[-8.68,64.46],[-9.19,61.81],[-10.18,56.75],[8.11,56.59],[9.3,56.58],[12.02,56.52],[12.03,56.6],[16.25,76.56],[10.87,77.84],[0.19,79.96]],"223110820":[[7.56,-55.16],[10.86,-55.15],[10.86,-54.56],[10.84,-51.51],[10.82,-48.36],[10.81,-47.35],[7.51,-47.36],[3.9,-47.39],[-0.1,-47.4],[-3.32,-47.42],[-3.31,-48.33],[-3.29,-51.6],[-3.27,-54.57],[-3.27,-55.21],[-0.04,-55.19]],"223110823":[[-0.1,-48.32],[-0.05,-54.56],[-0.04,-55.19],[-0.04,-56.47],[3.94,-56.45],[7.57,-56.43],[7.56,-55.16],[7.56,-54.57],[7.51,-48.37]],"223110824":[[16.68,23.85],[16.68,21.22],[16.69,18.5],[19.39,18.5],[21.79,18.48],[21.79,23.84]],"223110825":[[10.86,-54.56],[19.8,-54.51],[21.52,-54.51],[21.51,-52.95],[21.5,-52.29],[21.5,-51.47],[21.48,-48.32],[10.82,-48.36],[7.51,-48.37],[7.56,-54.57]],"223116018":[[-9.06,63.57],[-8.97,64.0],[-9.21,64.37],[-9.64,64.46],[-10.01,64.22],[-10.09,63.79],[-9.86,63.42],[-9.43,63.33]],"223140028":[[13.09,52.62],[13.47,53.58],[13.49,54.62],[13.13,55.6],[12.44,56.38],[12.03,56.6],[11.52,56.86],[10.49,56.97],[9.49,56.71],[9.3,56.58],[8.65,56.1],[8.12,55.27],[8.09,55.22],[7.88,54.21],[8.06,53.18],[8.11,53.08],[8.59,52.28],[9.4,51.64],[10.41,51.34],[11.44,51.42],[12.37,51.87]],"223140029":[[-1.13,56.94],[-7.02,56.96],[-10.34,56.96],[-10.33,52.46],[-7.02,52.45],[-1.13,52.44]],"223140032":[[-7.02,56.96],[-7.02,52.45],[-7.03,38.77],[-7.03,34.34],[-6.99,25.38],[-6.99,24.36],[-6.99,23.38],[-6.07,23.37],[-5.22,23.38],[-3.45,23.36],[6.98,23.34],[8.1,23.33],[8.1,24.17],[8.11,27.94],[8.11,38.71],[8.11,53.08],[8.12,55.27],[8.11,56.59],[8.12,56.91],[-1.13,56.94]],"223145711":[[-6.19,23.01],[-6.06,23.33],[-6.07,23.37],[-6.08,23.68],[-6.23,24.02],[-6.5,24.26],[-6.79,24.36],[-6.99,24.36],[-7.04,24.37],[-7.42,24.23],[-7.7,23.94],[-7.81,23.69],[-7.81,23.38],[-7.7,23.04],[-7.38,22.72],[-6.9,22.59],[-6.51,22.7]],"223145712":[[8.62,22.86],[8.74,23.18],[8.73,23.53],[8.57,23.86],[8.3,24.1],[8.1,24.17],[8.01,24.21],[7.76,24.21],[7.38,24.08],[7.1,23.8],[6.99,23.54],[6.98,23.34],[6.98,23.23],[7.09,22.88],[7.41,22.56],[7.9,22.44],[8.29,22.56]],"223145713":[[0.19,79.96],[10.87,77.84],[11.17,79.31],[0.48,81.43]],"223179003":[[-10.88,34.34],[-10.84,25.39],[-6.99,25.38],[-7.03,34.34]],"223179004":[[-13.08,38.79],[-13.08,34.35],[-10.88,34.34],[-7.03,34.34],[-1.95,34.31],[-1.94,38.75],[-7.03,38.77]],"223311650":[[-19.94,10.59],[-18.56,10.58],[-18.56,14.31],[-15.87,14.31],[-15.88,20.52],[-10.83,20.52],[-10.83,14.31],[-10.83,10.58],[-10.81,0.55],[-10.81,-6.04],[-15.37,-6.03],[-15.38,-2.12],[-15.55,-2.35],[-16.24,-2.74],[-17.02,-2.84],[-17.79,-2.63],[-18.41,-2.14],[-18.8,-1.46],[-18.9,-0.67],[-18.69,0.09],[-18.34,0.56],[-19.94,0.56]],"951894999":[[-0.04,-55.19],[7.56,-55.16],[7.56,-54.57],[7.51,-48.37],[7.51,-47.36],[3.9,-47.39],[-0.1,-47.4],[-0.1,-48.32],[-0.05,-54.56]],"1573692063":[[-19.94,0.56],[-19.94,10.59],[-10.83,10.58],[-10.81,0.55],[-15.38,0.55],[-18.34,0.56]],"1573692064":[[-18.56,10.58],[-18.56,14.31],[-15.87,14.31],[-10.83,14.31],[-10.83,10.58]],"1573692065":[[-10.81,-6.04],[-15.37,-6.03],[-15.38,-2.12],[-15.38,0.55],[-10.81,0.55]],"1573767740":[[19.41,12.32],[19.39,18.5],[16.69,18.5],[16.68,21.22],[11.95,21.22],[11.97,16.02],[11.97,12.31],[11.99,7.0],[12.02,-0.89],[16.73,-0.89],[16.73,-2.63],[20.43,-2.63],[21.72,-2.63],[21.7,7.0],[19.42,7.0]],"1573767741":[[21.68,12.32],[22.84,12.41],[24.16,11.07],[24.17,8.48],[22.81,6.98],[21.7,7.0]],"1573767742":[[21.7,7.0],[19.42,7.0],[11.99,7.0],[10.72,6.99],[10.71,12.31],[11.97,12.31],[19.41,12.32],[21.68,12.32]],"1573767743":[[-8.55,-28.72],[-7.81,-27.99],[-7.32,-27.06],[-7.15,-26.02],[-7.31,-24.99],[-7.77,-24.05],[-8.5,-23.3],[-9.44,-22.81],[-10.46,-22.63],[-11.5,-22.79],[-12.43,-23.26],[-13.18,-23.99],[-13.67,-24.91],[-13.84,-25.95],[-13.68,-26.99],[-13.21,-27.92],[-12.49,-28.68],[-11.56,-29.16],[-10.53,-29.34],[-9.49,-29.18]],"1573767744":[[-15.87,14.31],[-15.88,20.52],[-10.83,20.52],[-10.83,14.31]],"1573767745":[[21.96,-21.09],[13.32,-21.1]],"1573767747":[[-8.6,64.45],[-6.8,61.82],[-7.37,58.7],[-9.99,56.9],[-10.34,56.96],[-13.11,57.48],[-14.92,60.1],[-14.34,63.23],[-11.73,65.03],[-8.68,64.46]],"1573767748":[[-15.84,-20.74],[-15.84,-31.35],[-4.97,-31.38],[-4.97,-20.77]],"221599092":[[-5.22,23.38],[-5.16,5.46]],"221599093":[[9.96,27.94],[9.99,15.99]],"697869170":[[19.52,-23.38],[23.82,-23.34]],"697869171":[[13.33,-23.43],[18.35,-23.39]],"697869174":[[25.77,-29.91],[25.75,-35.48]],"507649258":[[13.32,-21.1],[-5.75,-21.46]]}""")


def R(i):
    return [tuple(p) for p in D[str(i)]]


# ── Plan helpers ─────────────────────────────────────────────────────────────
def ccw(P):
    P = [tuple(p) for p in P]
    if len(P) > 1 and math.dist(P[0], P[-1]) < 1e-6:
        P = P[:-1]
    return P if ring_area(P) > 0 else P[::-1]


def inside(x, z, P):
    ins = False
    n = len(P)
    for i in range(n):
        (xi, zi), (xj, zj) = P[i], P[i - 1]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            ins = not ins
    return ins


def centroid(P):
    P = np.asarray(ccw(P), float)
    x, z = P[:, 0], P[:, 1]
    x1, z1 = np.roll(x, -1), np.roll(z, -1)
    cr = x * z1 - x1 * z
    a = cr.sum() / 2
    return float(((x + x1) * cr).sum() / (6 * a)), float(((z + z1) * cr).sum() / (6 * a))


def circle_of(P):
    cx, cz = centroid(P)
    r = float(np.mean([math.dist((cx, cz), p) for p in P]))
    return cx, cz, r


def to_local(P, th):
    c, s = math.cos(th), math.sin(th)
    return [(x * c + z * s, -x * s + z * c) for x, z in P]


def from_local(P, th):
    c, s = math.cos(th), math.sin(th)
    return [(u * c - v * s, u * s + v * c) for u, v in P]


def long_axis(P):
    """Direction (radians, in x/z) of the long side of the smallest rectangle around P."""
    P = ccw(P)
    best = None
    for i in range(len(P)):
        (x0, z0), (x1, z1) = P[i], P[(i + 1) % len(P)]
        if math.dist((x0, z0), (x1, z1)) < 0.5:
            continue
        th = math.atan2(z1 - z0, x1 - x0)
        L = to_local(P, th)
        us, vs = [p[0] for p in L], [p[1] for p in L]
        a = (max(us) - min(us)) * (max(vs) - min(vs))
        if best is None or a < best[0] - 1e-6:
            best = (a, th, max(us) - min(us), max(vs) - min(vs))
    _, th, lu, lv = best
    return th if lu >= lv else th + math.pi / 2


def clip(P, axis, val, keep_less):
    out = []
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        ia = (a[axis] <= val) if keep_less else (a[axis] >= val)
        ib = (b[axis] <= val) if keep_less else (b[axis] >= val)
        if ia:
            out.append(a)
        if ia != ib:
            t = (val - a[axis]) / (b[axis] - a[axis])
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def model_dir(bearing):
    """A compass bearing (degrees) as a unit vector in the model's x/z."""
    e, n = math.sin(math.radians(bearing)), math.cos(math.radians(bearing))
    c, s = math.cos(math.radians(HEADING)), math.sin(math.radians(HEADING))
    return np.array([-e * c + n * s, e * s + n * c])


def resample(P, step):
    P = ccw(P)
    out = []
    for i in range(len(P)):
        a, b = np.array(P[i]), np.array(P[(i + 1) % len(P)])
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        for k in range(n):
            out.append(tuple(a + (b - a) * k / n))
    return out


def flip_up(m):
    """Make every face of a roof look upwards."""
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    down = np.cross(b - a, c - a)[:, 1] < 0
    m.f[down] = m.f[down][:, ::-1]
    return m


# ── Colour of the walls: yellowish limestone where they face a courtyard ─────
COURTS = [ccw(R(221600218)), ccw(R(221600217))]


def faces_court(a, b, n):
    mx, mz = (a[0] + b[0]) / 2 + n[0] * 2.0, (a[1] + b[1]) / 2 + n[1] * 2.0
    return any(inside(mx, mz, C) for C in COURTS)


def wall_colour(c):
    def fn(a, b, n):
        if c == LIME and faces_court(a, b, n):
            return LIME_Y
        return c
    return fn


# ── Prisms and roofs ─────────────────────────────────────────────────────────
OCC = []          # (polygon, y0, y1): what hides a window


def prism_var(P, y0, ytop, cfn, c_top, skip=None, top=True):
    """Walls from y0 up to ytop(x, z) at each vertex (colour cfn(a, b, n) per wall), and the top cap."""
    P = ccw(P)
    ms = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        if math.dist(a, b) < 1e-4 or (skip and skip(a, b)):
            continue
        e = np.subtract(b, a) / math.dist(a, b)
        n = (e[1], -e[0])
        V = [(a[0], y0, a[1]), (b[0], y0, b[1]), (b[0], ytop(*b), b[1]), (a[0], ytop(*a), a[1])]
        ms.append(Mesh(V, [(0, 2, 1), (0, 3, 2)], cfn(a, b, n)))
    if top:
        pts, tris = triangulate(P)
        v = np.stack([pts[:, 0], [ytop(x, z) for x, z in pts], pts[:, 1]], 1)
        ms.append(Mesh(v, tris[:, ::-1], c_top))
    return merge(ms)


def part(P, y0, eave, shape="flat", rh=0.0, wall=LIME, roof=SLATE, ridge=None, across=False, direction=None,
         cornice=True):
    """A building:part: walls from y0 to the eave and its roof (gabled, hipped, pyramidal, skillion, flat)."""
    P = ccw(P)
    cfn = wall_colour(wall)
    OCC.append((P, y0, eave + rh * 0.6))
    out = []
    if shape == "gabled":
        th = ridge if ridge is not None else long_axis(P) + (math.pi / 2 if across else 0.0)
        L = to_local(P, th)
        vs = [p[1] for p in L]
        vc, w = (min(vs) + max(vs)) / 2, (max(vs) - min(vs)) / 2
        c, s = math.cos(th), math.sin(th)

        def yt(x, z):
            v = -x * s + z * c
            return eave + rh * max(0.0, 1 - abs(v - vc) / w)

        def on_ridge(a, b):
            return abs(-a[0] * s + a[1] * c - vc) < 1e-6 and abs(-b[0] * s + b[1] * c - vc) < 1e-6
        for keep in (True, False):
            H = clip(L, 1, vc, keep)
            if len(H) >= 3 and abs(ring_area(H)) > 1e-3:
                out.append(prism_var(from_local(H, th), y0, yt, cfn, roof, skip=on_ridge))
    elif shape == "skillion":
        d = model_dir(direction)
        pr = [x * d[0] + z * d[1] for x, z in P]
        p0, p1 = min(pr), max(pr)
        out.append(prism_var(P, y0, lambda x, z: eave + rh * (p1 - (x * d[0] + z * d[1])) / (p1 - p0), cfn, roof))
    elif shape in ("hipped", "pyramidal"):
        out.append(prism_var(P, y0, lambda x, z: eave, cfn, roof, top=False))
        out.append(hip_roof(P, eave, rh, roof, pyramid=(shape == "pyramidal")))
    else:
        out.append(prism_var(P, y0, lambda x, z: eave, cfn, roof))
    if cornice:
        out.append(cornice_ring(P, eave - 0.45, 0.45, 0.22))
    return out


def hip_roof(P, eave, rh, c, pyramid=False):
    """A hipped roof on the smallest rectangle around P (or a pyramid on the polygon)."""
    if pyramid:
        cx, cz = centroid(P)
        V = [(x, eave, z) for x, z in P] + [(cx, eave + rh, cz)]
        n = len(P)
        return flip_up(Mesh(V, [(i, (i + 1) % n, n) for i in range(n)], c))
    th = long_axis(P)
    L = to_local(P, th)
    us, vs = [p[0] for p in L], [p[1] for p in L]
    u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
    vc, w = (v0 + v1) / 2, (v1 - v0) / 2
    h = min(w, (u1 - u0) / 2)
    Q = from_local([(u0, v0), (u1, v0), (u1, v1), (u0, v1), (u0 + h, vc), (u1 - h, vc)], th)
    V = [(x, eave, z) for x, z in Q[:4]] + [(x, eave + rh, z) for x, z in Q[4:]]
    F = [(0, 1, 5), (0, 5, 4), (1, 2, 5), (2, 3, 4), (2, 4, 5), (3, 0, 4)]
    return flip_up(Mesh(V, F, c))


def cornice_ring(P, y, h, d, c=LIME_L):
    O = offset(P, d)

    return merge([walls(O, y, y + h, c), cap(O, y + h, c, holes=[P]), cap(O, y, c, holes=[P], up=False)])


# ── Oriented pieces on a wall: local (u along the wall, y up, w outwards) ────────────────────────────────
class Face:
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
    pts += [(u + half * math.cos(a), ys + half * math.sin(a)) for a in np.linspace(0, math.pi, seg + 1)[1:-1]]
    return pts


def arch_window(f, u, y0, w, h, pair=False, frame=LIME_L, sill=True):
    """A round-arched window (or a pair under one frame) with a light stone frame and sill."""
    out = [f.poly(arch_pts(u, y0 - 0.12, y0 + h - w / 2, w / 2 + 0.16), frame, 0.04)]
    if pair:
        hw = w * 0.42
        for du in (-w * 0.27, w * 0.27):
            out.append(f.poly(arch_pts(u + du, y0, y0 + h - w / 2, hw / 2, 5), GLASS, 0.09))
    else:
        out.append(f.poly(arch_pts(u, y0, y0 + h - w / 2, w / 2), GLASS, 0.09))
    if sill:
        out.append(f.slab(u - w / 2 - 0.22, u + w / 2 + 0.22, y0 - 0.3, y0 - 0.1, 0.0, 0.22, frame))
    return out


def hidden(x, z, y):
    return any(y0 - 0.1 < y < y1 and inside(x, z, P) for P, y0, y1 in OCC)


def wall_windows(P, rows, spacing, margin=1.0, min_len=2.4, only=None):
    """Rows of windows on every straight wall of P that is not hidden behind another part.
    rows: [(sill y, width, height, pair)]."""
    P = ccw(P)
    out = []
    for i in range(len(P)):
        a, b = np.array(P[i]), np.array(P[(i + 1) % len(P)])
        L = float(np.linalg.norm(b - a))
        if L < min_len:
            continue
        t = (b - a) / L
        n = np.array([t[1], -t[0]])
        if only is not None and not only(a, b, n):
            continue
        f = Face(a, t, n)
        k = max(1, int((L - 2 * margin) // spacing))
        for y0, w, h, pair in rows:
            if w + 0.4 > L - 0.6:
                continue
            for j in range(k):
                u = L / 2 + (j - (k - 1) / 2) * spacing
                p = a + t * u + n * 1.0
                if hidden(p[0], p[1], y0 + h / 2):
                    continue
                out += arch_window(f, u, y0, w, h, pair)
    return out


# ── Round towers, turrets, spires, finials ───────────────────────────────────
GOLDS = []


def finial(x, y, z, h=2.2, ball=0.28):
    """A gilded finial: a rod with a ball and a little vane."""
    GOLDS.append(cyl(0.07, 0.04, h, GOLD, 6).T(x, y, z))
    GOLDS.append(lathe([(0.0, -ball), (ball * 0.7, -ball * 0.7), (ball, 0), (ball * 0.7, ball * 0.7), (0.0, ball)],
                       GOLD, 8).T(x, y + h * 0.45, z))
    GOLDS.append(box(0.04, 0.45, 0.75, GOLD).T(x, y + h * 0.72, z + 0.36))


def spire(cx, cz, r, y, h, c=SLATE, seg=20, fin=True, flare=0.25):
    """A pointed slate cone with a little flared eave and a gilded finial on top."""
    out = [cyl(r + flare, r * 0.82, h * 0.12, c, seg).T(cx, y, cz),
           cyl(r * 0.82, 0.0, h * 0.88, c, seg, caps=False).T(cx, y + h * 0.12, cz)]
    if fin:
        finial(cx, y + h - 0.2, cz, max(1.2, h * 0.16))
    return out


def ring_windows(cx, cz, r, rows, n, a0=0.0):
    out = []
    for y0, w, h in rows:
        for k in range(n):
            th = a0 + 2 * math.pi * k / n
            nn = np.array([math.cos(th), math.sin(th)])
            p = np.array([cx, cz]) + nn * (r + 1.0)
            if hidden(p[0], p[1], y0 + h / 2):
                continue
            f = Face(np.array([cx, cz]) + nn * r, np.array([-nn[1], nn[0]]), nn)
            out += arch_window(f, 0.0, y0, w, h, sill=False)
    return out


def merlons_ring(cx, cz, r, y, h=0.9, c=LIME, n=None):
    n = n or max(6, int(2 * math.pi * r / 1.3))
    out = [cyl(r + 0.25, r + 0.25, 0.35, LIME_L, max(16, n * 2)).T(cx, y - 0.35, cz)]
    for k in range(n):
        th = 2 * math.pi * (k + 0.5) / n
        nn = np.array([math.cos(th), math.sin(th)])
        f = Face(np.array([cx, cz]) + nn * r, np.array([-nn[1], nn[0]]), nn)
        out.append(f.slab(-0.38, 0.38, y, y + h, -0.25, 0.25, c))
    return out


def merlons_line(a, b, y, h=0.9, w=0.5, c=LIME, step=1.6, inward=0.0, coping=True):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 0.8:
        return []
    t = (b - a) / L
    n = np.array([t[1], -t[0]])
    f = Face(a, t, n)
    out = [f.slab(0, L, y - 0.3, y, -w - inward - 0.05, -inward + 0.05, LIME_L)] if coping else []
    k = max(1, int(L / step))
    for j in range(k):
        u = (j + 0.5) * L / k
        out.append(f.slab(u - 0.38, u + 0.38, y, y + h, -w - inward + 0.02, -inward, c))
    return out


def turret(P, y0, y1, cone_h, c=LIME, corbel=0.0, rows=(), nwin=4, cone_r=None, seg=28):
    """A round tower or corbelled turret from an OSM circle: shaft, string course, slate spire."""
    cx, cz, r = circle_of(P)
    OCC.append(([(cx + r * math.cos(t), cz + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 12, endpoint=False)],
                y0, y1))
    out = [cyl(r, r, y1 - y0, c, seg, caps=False).T(cx, y0, cz),
           cyl(r + 0.18, r + 0.18, 0.4, LIME_L, seg).T(cx, y1 - 0.9, cz)]
    if corbel > 0:
        out.append(cyl(0.25, r, corbel, LIME_D, seg, caps=False).T(cx, y0 - corbel, cz))
    out += ring_windows(cx, cz, r, rows, nwin)
    out += spire(cx, cz, cone_r or r, y1, cone_h, seg=seg)
    return out


# ── The buildings ────────────────────────────────────────────────────────────
PALAS_ROWS = [(1.2, 1.1, 2.0, False), (6.6, 1.2, 2.6, False), (12.2, 1.2, 2.6, False),
              (17.6, 1.5, 3.8, True), (24.4, 1.6, 4.6, True)]


def palas():
    out = []
    # the main (east) block and the west (Throne Hall) block, turned ~13° to follow the rock
    east, west = R(223140032), R(223110819)
    out += part(east, F_UP, 33.0, "gabled", 12.0)
    th_w = math.atan2(76.56 - 56.6, 16.25 - 12.03)
    out += part(west, F_UP, 34.0, "gabled", 12.0, ridge=th_w)
    out += part(R(223140029), F_UP, 33.0, "hipped", 3.0)                     # junction to the stair tower
    out += part(R(223179004), F_UP, 33.0, "hipped", 6.0)                     # north projection
    out += part(R(223179003), F_UP, 28.0, "skillion", 1.0, direction=335)    # Wintergarten
    out += part(R(223069512), F_UP, 18.0, "skillion", 2.0, direction=160)    # south annex
    # the oriel / balcony of the Throne Hall on the west front, on a corbel
    ori = R(223145713)
    out += part(ori, 15.0, 21.0, "skillion", 2.0, wall=SAND, direction=236)
    cx, cz = centroid(ori)
    out.append(lathe([(0.3, 11.0), (2.0, 13.4), (2.2, 15.0)], SAND, 12).S(2.4, 1, 0.8).R(0, -math.radians(11), 0).T(cx, 0, cz))
    # windows
    for P, rows in ((east, PALAS_ROWS), (west, PALAS_ROWS), (R(223179004), PALAS_ROWS),
                    (R(223179003), PALAS_ROWS[:4]), (R(223069512), PALAS_ROWS[:3])):
        out += wall_windows(P, rows, 3.4)
    out += wall_windows(ori, [(16.0, 1.2, 3.2, True)], 2.6, margin=0.8)
    # small dormers on both slopes of the two Palas roofs (number and size estimated)
    for P, th, eave, rh in ((east, long_axis(east), 33.0, 12.0), (west, th_w, 34.0, 12.0)):
        out += dormers(P, th, eave, rh, 6.0)
    # string courses between the storeys and the plinth
    for P in (east, west):
        for y in (0.4, 5.9, 11.5, 17.0, 23.6):
            out.append(cornice_ring(ccw(P), y - 0.25, 0.25, 0.12))
    # gable finials of the Palas
    for P, th in ((east, long_axis(east)), (west, th_w)):
        L = to_local(ccw(P), th)
        us, vs = [p[0] for p in L], [p[1] for p in L]
        vc = (min(vs) + max(vs)) / 2
        for u in (min(us), max(us)):
            x, z = from_local([(u, vc)], th)[0]
            out.append(box(0.9, 1.6, 0.9, LIME_L).T(x, 44.6 if P is east else 45.6, z))
            finial(x, (44.6 if P is east else 45.6) + 1.6, z, 2.0)
    # the two slender turrets flanking the courtyard gable, corbelled out at 25 m
    for i in (223145711, 223145712):
        out += turret(R(i), 25.0, 34.5, 7.5, corbel=2.0, rows=[(28.0, 0.5, 1.6)], nwin=3)
    # the north stair tower (Treppenturm): broad lower shaft to 46 m with its gallery, slender turret, spire 65 m
    tp = R(1573767747)
    out += part(tp, F_UP, 45.0, "flat", wall=LIME, roof=SLATE_L, cornice=False)
    out += wall_windows(tp, [(4.0, 0.7, 1.6, False), (10.0, 0.7, 1.6, False), (16.0, 0.7, 1.6, False),
                             (22.0, 0.7, 1.6, False), (28.0, 0.7, 1.6, False), (34.0, 0.7, 1.6, False),
                             (39.5, 0.8, 2.2, False)], 3.0, margin=0.6, min_len=2.0)
    tcx, tcz, tr = circle_of(tp)
    out.append(cornice_ring(ccw(tp), 44.5, 0.5, 0.45))
    for a, b in zip(offset(ccw(tp), 0.45), offset(ccw(tp), 0.45)[1:] + offset(ccw(tp), 0.45)[:1]):
        out += merlons_line(a, b, 45.0, 1.0, 0.45, step=1.4)
    ucx, ucz, ur = circle_of(R(221599097))
    out += turret(R(221599097), 45.0, 53.0, 12.0, rows=[(47.5, 0.8, 2.6)], nwin=6, cone_r=ur + 0.15)
    out += turret(R(223116018), 52.0, 57.0, 4.0, seg=10)
    # the round south tower of the Palas (52 m)
    out += turret(R(223140028), F_UP, 42.0, 10.0, rows=[(r_, 0.75, 1.8) for r_ in (3.0, 9.0, 15.0, 21.0, 27.0, 33.0)] +
                  [(38.0, 0.8, 2.2)], nwin=5)
    return out


def dormers(P, th, eave, rh, step):
    L = to_local(ccw(P), th)
    us, vs = [p[0] for p in L], [p[1] for p in L]
    u0, u1, vc, w = min(us), max(us), (min(vs) + max(vs)) / 2, (max(vs) - min(vs)) / 2
    tdir = np.array([math.cos(th), math.sin(th)])
    ndir = np.array([-math.sin(th), math.cos(th)])
    k = max(1, int((u1 - u0 - 6.0) // step))
    out = []
    for side in (-1, 1):
        for j in range(k):
            u = (u0 + u1) / 2 + (j - (k - 1) / 2) * step
            vf = w - 1.2
            yf = eave + rh * (1 - vf / w)
            a = np.array(from_local([(u, vc + side * vf)], th)[0])
            f = Face(a, tdir * side, ndir * side)
            out.append(f.slab(-0.85, 0.85, yf - 0.4, yf + 2.0, -2.6, 0.0, LIME))
            out += arch_window(f, 0.0, yf + 0.2, 0.8, 1.5, sill=False)
            # its little gabled roof
            V = [f.P(-1.0, yf + 2.0, 0.15), f.P(1.0, yf + 2.0, 0.15), f.P(0.0, yf + 3.1, 0.15),
                 f.P(-1.0, yf + 2.0, -2.8), f.P(1.0, yf + 2.0, -2.8), f.P(0.0, yf + 3.1, -2.8)]
            out.append(flip_up(Mesh(V, [(0, 2, 5), (0, 5, 3), (1, 4, 5), (1, 5, 2)], SLATE)))
            out.append(Mesh(V[:3], [(0, 1, 2)], LIME))
    return out


def knights_house():
    out = []
    out += part(R(1573692063), F_UP, 27.0, "gabled", 4.0)
    out += part(R(1573692064), F_UP, 23.5, "hipped", 3.0)
    out += part(R(1573692065), F_UP, 23.5, "hipped", 3.0)
    out += part(R(1573767744), F_UP, 17.0, "hipped", 2.0)
    gal = R(223069513)                                                     # the long gallery along the north side
    out += part(gal, F_UP, 17.0, "gabled", 2.5)
    rows = [(1.2, 1.1, 2.2, False), (6.6, 1.2, 2.6, False), (12.0, 1.2, 2.6, False), (18.0, 1.2, 2.6, False)]
    for i in (1573692063, 1573692064, 1573692065, 1573767744):
        P = R(i)
        top = {1573692063: 27.0, 1573692064: 23.5, 1573692065: 23.5, 1573767744: 17.0}[i]
        out += wall_windows(P, [r for r in rows if r[0] + r[2] < top - 0.6], 3.0)
        for y in (0.4, 5.9, 11.3, 16.8):
            if y < top - 1.5:
                out.append(cornice_ring(ccw(P), y - 0.22, 0.22, 0.1))
    # the gallery: arcades on the courtyard side, small windows outside
    out += wall_windows(gal, [(1.0, 1.4, 2.6, False), (6.2, 1.4, 2.6, True), (11.4, 1.4, 3.0, True)], 2.8)
    out += turret(R(223069511), F_UP, 33.0, 5.0, rows=[(r_, 0.7, 1.7) for r_ in (4.0, 10.0, 16.0, 22.0, 28.0)], nwin=4)
    return out


def square_tower():
    """The Viereckturm (45 m): square shaft, corbelled crenellated platform, round turret and spire."""
    out = []
    sq = R(221599098)
    out += part(sq, LOW - 2.0, 28.0, "flat", cornice=False)
    out += wall_windows(sq, [(y, 0.8, 1.8, False) for y in (2.5, 8.5, 14.5, 20.5)], 9.0, margin=0.5)
    pl = ccw(R(223069515))

    out.append(prism_var(pl, 28.0, lambda x, z: 33.0, wall_colour(LIME), LIME_D))
    out.append(cap(pl, 28.0, LIME_D, up=False))
    OCC.append((pl, 28.0, 33.0))
    out += wall_windows(pl, [(29.2, 0.9, 2.2, True)], 3.2, margin=0.6)
    for k in range(len(pl)):                                               # corbels under the platform
        a, b = np.array(pl[k]), np.array(pl[(k + 1) % len(pl)])
        L = float(np.linalg.norm(b - a)); t = (b - a) / L; n = np.array([t[1], -t[0]])
        f = Face(a, t, n)
        for j in range(int(L / 1.2)):
            u = (j + 0.5) * L / int(L / 1.2)
            out.append(f.slab(u - 0.25, u + 0.25, 26.6, 28.0, -0.9, -0.02, LIME_D))
        out += merlons_line(a, b, 33.0, 1.1, 0.5, step=1.5)
    rt = R(1573767743)
    out += turret(rt, 33.0, 38.0, 7.0, rows=[(34.3, 0.8, 2.2)], nwin=6, cone_r=circle_of(R(223069514))[2])
    return out


def kemenate():
    out = []
    a, b = R(1573767740), R(1573767742)
    out += part(a, F_UP, 13.0, "gabled", 5.0)
    out += part(b, F_UP, 13.0, "gabled", 5.0, across=True)
    bay = R(1573767741)
    out += part(bay, F_UP, 13.0, "pyramidal", 4.0)
    rows = [(1.0, 1.1, 2.2, False), (5.2, 1.2, 2.4, True), (9.3, 1.2, 2.4, True)]
    for P in (a, b, bay):
        out += wall_windows(P, rows, 3.0, min_len=1.8)
        for y in (0.4, 4.6, 8.7):
            out.append(cornice_ring(ccw(P), y - 0.22, 0.22, 0.1))
    # stepped gables of the cross wing towards the south (± estimated)
    small = R(223071829)
    out += part(small, F_UP, 10.0, "hipped", 2.0)
    out += wall_windows(small, rows[:2], 2.8)
    tw = R(223110824)
    out += part(tw, F_UP, 17.0, "pyramidal", 5.5)
    out += wall_windows(tw, rows + [(13.4, 0.9, 2.0, False)], 3.0)
    cx, cz = centroid(tw)
    finial(cx, 22.4, cz, 1.6)
    return out


def gatehouse():
    """The Torbau: red brick, light stone gate front with its crenellated top, the central gable, corner towers."""
    out = []
    for i in (223110818, 223110825):
        P = R(i)
        out += part(P, F_LOW, LOW + 10.5, "hipped", 1.5, wall=BRICK, cornice=False)
        out.append(cornice_ring(ccw(P), LOW + 10.1, 0.4, 0.2, LIME_L))
        out += wall_windows(P, [(LOW + 1.4, 0.9, 1.6, False), (LOW + 5.6, 1.1, 2.4, True)], 2.6)
    mid = R(223110820)
    out += part(mid, F_LOW, LOW + 17.4, "gabled", 4.6, wall=BRICK, across=True, cornice=False)
    out.append(cornice_ring(ccw(mid), LOW + 17.0, 0.4, 0.2, LIME_L))
    out += wall_windows(mid, [(LOW + 13.0, 1.2, 2.6, True)], 2.8)
    # the gable finials of the central block
    th = long_axis(mid) + math.pi / 2
    L = to_local(ccw(mid), th)
    us, vs = [p[0] for p in L], [p[1] for p in L]
    for u in (min(us), max(us)):
        x, z = from_local([(u, (min(vs) + max(vs)) / 2)], th)[0]
        finial(x, LOW + 22.0, z, 1.8)
    # the light stone gate front, projecting, flat with battlements
    gf = ccw(R(223110823))
    out += part(gf, F_LOW, LOW + 12.0, "flat", wall=SAND, roof=LIME_D, cornice=False)
    for k in range(len(gf)):
        a, b = gf[k], gf[(k + 1) % len(gf)]
        if math.dist(a, b) > 1.0:
            out += merlons_line(b, a, LOW + 12.0, 1.0, 0.45, c=SAND, step=1.4, inward=-0.45)
    # the gate arch on both sides, the window over it
    for z0, sgn in ((-56.47, -1), (-47.4, 1)):
        f = Face((3.9, z0), (1.0 * -sgn, 0.0), (0.0, sgn))
        out.append(f.poly(arch_pts(0.0, LOW, LOW + 3.4, 1.9, 8), DARK, 0.06))
        out.append(f.poly(arch_pts(0.0, LOW - 0.05, LOW + 3.4, 2.35, 8), LIME_L, 0.03))
        if sgn < 0:
            out += arch_window(f, 0.0, LOW + 6.6, 1.6, 3.0, True)
    # round corner towers (sandstone), crenellated, with a low spire
    for i in (221599083, 221599084):
        cx, cz, r = circle_of(R(i))
        OCC.append((R(i), F_LOW, LOW + 18.0))
        out.append(cyl(r, r, LOW + 16.0 - F_LOW, SAND, 18, caps=False).T(cx, F_LOW, cz))
        out.append(cyl(r, r + 0.35, 1.0, LIME_D, 18, caps=False).T(cx, LOW + 15.0, cz))
        out.append(cyl(r + 0.35, r + 0.35, 1.0, SAND, 18, caps=False).T(cx, LOW + 16.0, cz))
        out += merlons_ring(cx, cz, r + 0.2, LOW + 17.0, 0.9, SAND)
        out += ring_windows(cx, cz, r, [(LOW + 3.0, 0.6, 1.4), (LOW + 7.5, 0.6, 1.4), (LOW + 12.0, 0.6, 1.4)], 3)
        out += spire(cx, cz, r - 0.25, LOW + 16.6, 4.5, flare=0.1)
    # the north wing of the lower courtyard
    nw = R(221599086)
    out += part(nw, F_LOW, LOW + 12.0, "gabled", 2.0)
    out += wall_windows(nw, [(LOW + 1.2, 1.0, 2.0, False), (LOW + 5.4, 1.1, 2.4, False)], 3.0)
    return out


# ── Courtyards, walls, stair, rock ───────────────────────────────────────────
def courtyards():
    out = []
    up = ccw(R(221599101))

    out.append(walls(up, F_UP, UP, LIME_D)); out.append(cap(up, UP, PAVING))
    low = ccw(R(221600217))
    out.append(walls(low, F_LOW, LOW, LIME_D)); out.append(cap(low, LOW, PAVING))
    # the raised terrace in front of the Palas, between its two retaining walls (height estimated)
    ter = [(-3.45, 5.46), (8.02, 5.46), (8.02, 23.3), (-3.45, 23.3)]
    out.append(walls(ter, UP - 0.2, UP + 1.2, LIME_D)); out.append(cap(ter, UP + 1.2, PAVING))
    # the stair between the courtyards (OSM steps 697869172/174), along the south wall
    n = 12
    for k in range(n):
        z1 = -21.6 - k * (13.6 / n)
        out.append(box(4.6, (UP - k * (UP - LOW) / n) - F_LOW, 13.6 / n, PAVING).T(25.75, F_LOW, z1 - 13.6 / n / 2))
    # the retaining wall between the courtyards, with its parapet
    rw = R(221599104)
    for a, b in zip(rw[:-1], rw[1:]):
        out += wall_seg(a, b, LOW - 0.5, UP + 1.0, 0.6, LIME_D, side=-1)
    # the curtain wall along the south: upper level to z = -21, then the lower courtyard
    cw = R(221599099)
    for a, b in zip(cw[:-1], cw[1:]):
        lev = UP if min(a[1], b[1]) > -21.5 else LOW
        out += wall_seg(a, b, LEDGE - 1.5, lev + 1.2, 0.9, LIME, side=1)
        if lev == LOW:                                                     # battlements over the lower courtyard
            out += merlons_line(a, b, lev + 1.38, 0.8, 0.9, step=1.5, inward=-0.45, coping=False)
    # the wall north of the lower courtyard
    w3 = R(221599103)
    for a, b in zip(w3[:-1], w3[1:]):
        out += wall_seg(a, b, LOW - 0.5, LOW + 2.2, 0.6, LIME, side=1)
    # forecourt in front of the gate
    out.append(box(9.0, LOW - LEDGE + 0.3, 6.0, PAVING).T(3.9, LEDGE - 0.3, -59.4))
    return out


def wall_seg(a, b, y0, y1, th, c, side=1):
    """A wall along a line, th thick, with a coping."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 0.05:
        return []
    t = (b - a) / L
    n = np.array([t[1], -t[0]]) * side
    f = Face(a, t, n)
    return [f.slab(-0.05, L + 0.05, y0, y1, -th / 2, th / 2, c),
            f.slab(-0.1, L + 0.1, y1, y1 + 0.18, -th / 2 - 0.1, th / 2 + 0.1, LIME_L)]


def rock():
    """The rock spur under the castle: the ledge at y = -7.5 inside the castle's area (OSM way 221601969),
    then a cliff flaring out down to y = -42."""
    top = resample(R(221601969), 1.6)
    rings = [(0.0, LEDGE), (0.6, LEDGE - 2.0), (1.4, LEDGE - 5.0), (2.4, LEDGE - 8.5), (3.6, LEDGE - 12.5),
             (5.2, LEDGE - 17.0), (7.4, LEDGE - 22.5), (9.6, LEDGE - 28.0), (12.0, -42.0)]
    V, C, F = [], [], []
    n = len(top)
    for k, (d, y) in enumerate(rings):
        O = offset(top, d) if d > 0 else top
        for i, (x, z) in enumerate(O):
            p = np.array([[x, y, z]])
            j = (fbm(p, 0.09, 11 + k, 3)[0] - 0.5) * (0.0 if k == 0 else 3.5)
            jy = (fbm(p, 0.12, 31 + k, 2)[0] - 0.5) * (0.0 if k == 0 else 3.0)
            nx, nz = 0.0, 0.0
            if k:
                a, b = np.array(O[i - 1]), np.array(O[(i + 1) % n])
                t = (b - a) / (np.linalg.norm(b - a) or 1)
                nx, nz = t[1], -t[0]
            V.append((x + nx * j, y + jy, z + nz * j))
            g = fbm(p, 0.15, 5, 2)[0]
            C.append(GRASS if (k == 1 and g > 0.5) else (ROCK if g > 0.45 else ROCK_D))
    for k in range(len(rings) - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            F += [(a, a + n, b + n), (a, b + n, b)]
    cliff = Mesh(V, F, np.array(C))
    # make the cliff faces look outwards
    ctr = np.array(centroid(top))
    tri = cliff.v[cliff.f]
    nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    mid = tri.mean(1)
    flip = (nn[:, 0] * (mid[:, 0] - ctr[0]) + nn[:, 2] * (mid[:, 2] - ctr[1])) < 0
    cliff.f[flip] = cliff.f[flip][:, ::-1]
    ledge = cap(top, LEDGE, LEDGE_C)
    return [cliff, ledge]


FOOTPRINT = offset(R(221601969), 3.0)


def build():
    pal = palas()
    kh = knights_house()
    st = square_tower()
    ke = kemenate()
    gh = gatehouse()
    ct = courtyards()
    rk = rock()
    return [("palas", merge(pal)), ("knights_house", merge(kh)), ("square_tower", merge(st)),
            ("kemenate", merge(ke)), ("gatehouse", merge(gh)), ("courtyards", merge(ct)), ("rock", merge(rk))]


def main():
    parts = build()
    out_dir = ROOT / "models" / "Q4152"
    out_dir.mkdir(parents=True, exist_ok=True)
    named = []
    for name, m in parts:
        if name != "rock":
            k = 0.95 + 0.05 * fbm(m.v, 0.07, 7, 2)
            m.c = np.clip(m.c * k[:, None], 0, 1)
        named.append((name, m))
    named.append(("gilding", merge(GOLDS), {"material": "gold"}))
    tris, size = write_glb(out_dir / "model.glb", named, roughness=0.85,
                           materials={"gold": {"metallic": 0.8, "roughness": 0.35}})
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        mj["footprint"] = [[round(x, 1), round(z, 1)] for x, z in FOOTPRINT]
        meta.write_bytes((json.dumps(mj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"{tris} triangles, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
