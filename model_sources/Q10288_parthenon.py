"""
Parthenon (Q10288) — with the whole Acropolis of Athens on its rock, as it stands today, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monuments: the Parthenon by Iktinos and Kallikrates with Pheidias (447–432 BC); the Propylaia by Mnesikles
(437–432 BC); the Temple of Athena Nike by Kallikrates (c. 427–424 BC); the Erechtheion (421–406 BC); the walls
of Themistokles and Kimon (after 479 / 465 BC); the Beulé Gate (3rd century AD); the Monument of Agrippa
(178 BC / 27 BC). Public domain.

What is modelled:
- The Parthenon as it is today: the three-step crepis on the 31 × 70 m stylobate (69.5 × 30.9 m, en.wikipedia),
  the 8 × 17 fluted Doric columns with entasis and capitals, the four broken columns in the middle of the south
  side, the roofless interior, the entablature (architrave, frieze with triglyphs, cornice) missing over the
  middle of the south side and its frieze and cornice missing over the middle of the north side, the complete
  west pediment and the two corners left of the east one, the six columns of each inner porch (three of the east
  porch broken) with their architraves, the west cross wall of the cella with its great door, the stepped ruins of
  the cella walls and the raised cella floor.
- The Erechtheion: the east porch (6 Ionic columns), the roofless main block with its dark Eleusinian frieze,
  the west front on the lower ground with its engaged half-columns and windows, the north porch (6 Ionic
  columns, 3.2 m lower) and the Porch of the Caryatids (6 maidens on a podium, flat roof); the sacred olive tree of
  the Pandroseion.
- The Propylaia: the west and east Doric hexastyle porticoes (wider central bay), the three pairs of Ionic columns
  of the central passage, the gate wall with its five doors, the ceiling beams of the west hall, the Pinakotheke
  (north wing) with its porch, the south wing; the Roman staircase down to the Beulé Gate and its two towers;
  the Monument of Agrippa; the Temple of Athena Nike (4 + 4 Ionic columns, no roof) on its bastion.
- The rock: the plateau (a height field cut to the outline of the Acropolis), the circuit of the walls of
  Kimon and Themistokles with their parapet, going down to the foot of the walls, and below them the rock face
  and slopes going down to about 58 m under the plateau, with scrub on the lower slopes.
- Low remains: the foundations of the Old Temple of Athena, the round Temple of Rome and Augustus, the
  Arrephorion, the base of the Athena Promachos, the ruins north-west and south-west of the Parthenon, the Old
  Acropolis Museum as a plain low block, and the Greek flag on the belvedere at the east end.
- Not modelled: the scaffolding and cranes of the restoration, the Odeon of Herodes Atticus and the Theatre of
  Dionysos (separate monuments on the south slope), the slopes' paths, the interiors.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap — building relation 5723216 (outline way 910010406, 286 building:parts: crepis, every
  column, architrave and frieze pieces, cella walls, pediments — as mapped of the present ruin), way 910010334
  (Erechtheion), relations 12352555 (Propylaia), 12352554 (Athena Nike), 12352551 (Old Temple of Athena), 12352553
  (walls of the Acropolis), 4423249 (outline of the Acropolis), ways 910010374 (Beulé Gate), 426887028 (Monument of
  Agrippa) and others listed below. Heading 77.09° = the axis through the 58 column centres of the Parthenon
  (least squares), the east front (entrance) facing +Z.
- Heights: Parthenon crepis 1.8 m (OSM steps), columns 10.43 m with lower diameter 1.905 m (shaft to 11.37 m,
  echinus and abacus to 12.23 m), entablature 3.29 m (architrave 1.35, frieze 1.35, cornice 0.6, so 13.72 m
  over the stylobate, en.wikipedia), pediment apex ~19.2 m (slope ~13°). OSM gives relative heights that are
  ~8 % low for the columns, so they were remapped to these figures; the heights of the broken columns and of the
  cella ruins follow OSM, scaled the same way. Propylaia: west columns 8.81 m, east 8.53 m (Ø 1.56 m), Ionic
  columns ~10.3 m; Erechtheion: east columns 6.58 m, north porch columns 7.63 m, caryatids 2.31 m on a 1.77 m
  podium; Athena Nike: stylobate 8.27 × 5.64 m, columns 4.07 m; Beulé Gate ~9 m and Monument of Agrippa ~10 m
  over their bases (estimates, ±15 %).
- Ground: EU-DEM 25 m and SRTM 30 m (via OpenTopoData) give the plateau at ≈147–148 m (smoothed; the summit is
  156 m) and the foot of the walls at ≈115–125 m. y = 0 is the plateau at the Parthenon (ground_at [20, 60], the
  open rock north-east of it, the DEM's highest point there); the rock falls 4.5 m to the east portico of the
  Propylaia and 1.8 m more to its west hall (−6.3), the stairs go down to the Beulé Gate (−17); the north porch of
  the Erechtheion and the Pandroseion lie 3.2 m lower than its east porch. The walls stand 10 m (north) to 15 m
  (south) over their foot, the rock goes down to −58. Levels of the plateau: estimates, ±2 m.
Pentelic marble, warm off-white weathered to honey; grey Hymettian marble for the Monument of Agrippa; dark
Eleusinian limestone for the frieze of the Erechtheion; the walls in poros and limestone ashlar.

Axes (docs/MODEL_SPEC.md): +Y up, the east front of the Parthenon faces +Z (heading 77.09°), +X points to
347.09° (the north side). Origin on the ground at the centre of the Parthenon.

    python model_sources/Q10288_parthenon.py   -> models/Q10288/model.glb (+ footprint in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, cap, walls, triangulate, ring_area, offset, fbm, sstep, write_glb  # noqa: E402

rng = np.random.default_rng(447)
HEADING = 77.09

# ── Colours ─────────────────────────────────────────────────────────────
MARBLE = (0.91, 0.88, 0.80)           # Pentelic marble, weathered
HONEY = (0.80, 0.69, 0.52)             # the golden patina
MARBLE_W = (0.93, 0.92, 0.89)          # restored drums, new marble
GREY_M = (0.56, 0.575, 0.60)           # Hymettian marble (Agrippa)
ELEUSIS = (0.30, 0.31, 0.33)           # dark Eleusinian limestone (Erechtheion frieze)
POROS = (0.72, 0.665, 0.565)           # foundations, poros
WALL = (0.745, 0.675, 0.555)           # the circuit walls, limestone/poros ashlar
WALL_D = (0.64, 0.575, 0.47)
ROCK = (0.635, 0.585, 0.515)           # Acropolis limestone
ROCK_R = (0.62, 0.49, 0.39)            # reddish schist / iron stains
PLATEAU = (0.75, 0.71, 0.63)          # the worn rock and gravel of the top
SCRUB = (0.35, 0.41, 0.25)
DARK = (0.10, 0.09, 0.08)              # doors, windows
OLIVE = (0.45, 0.52, 0.38)
BARK = (0.33, 0.27, 0.20)
MUSEUM = (0.74, 0.71, 0.64)
BLUE = (0.05, 0.37, 0.71)
WHITE = (0.96, 0.96, 0.96)
IRON = (0.25, 0.25, 0.26)

# ── Levels (y, metres) ──────────────────────────────────────────────────
Y_PE = -4.5        # east portico of the Propylaia
Y_PW = -6.3        # west hall of the Propylaia, Nike bastion
Y_B = -17.0        # Beulé Gate
Y_BOT = -58.0      # the rock goes down to here
PX = 51.65         # axis of the Propylaia (x)

# ── Data (OpenStreetMap © OpenStreetMap contributors, ODbL), metres in the model axes ───────────────────
DATA = json.loads(r"""{"acropolis":[[-56.73,125.25],[-61.44,125.24],[-61.45,129.24],[-56.73,129.24],[-56.73,141.91],[-54.9,142.37],[7.87,128.65],[8.84,133.12],[13.69,132.06],[12.73,127.59],[15.14,127.06],[14.33,123.29],[18.06,120.29],[20.39,113.2],[24.8,104.21],[26.02,100.55],[36.11,91.36],[43.79,88.05],[46.33,74.78],[51.51,64.63],[60.88,57.32],[62.41,55.9],[63.93,54.38],[64.55,53.44],[65.04,52.43],[65.87,50.76],[79.18,45.16],[83.19,25.03],[82.71,5.95],[94.29,-1.72],[95.36,-19.48],[95.41,-20.24],[95.46,-21.18],[96.62,-40.34],[101.9,-69.17],[94.09,-72.45],[91.44,-82.17],[90.14,-99.31],[90.29,-99.84],[89.51,-105.23],[93.25,-106.54],[92.08,-110.87],[88.7,-109.95],[87.03,-113.84],[85.11,-133.83],[79.71,-133.89],[79.76,-137.93],[78.44,-137.95],[79.43,-153.16],[67.31,-153.52],[67.35,-155.17],[65.91,-155.2],[65.84,-156.68],[67.37,-156.66],[67.4,-157.9],[65.77,-157.93],[65.67,-160.02],[67.44,-159.99],[67.46,-161.26],[65.61,-161.29],[65.5,-163.58],[67.5,-163.54],[67.6,-169.08],[65.23,-169.11],[64.98,-174.11],[56.16,-174.32],[56.04,-169.01],[52.44,-169.1],[48.89,-169.18],[49.02,-174.48],[41.75,-174.65],[41.61,-168.6],[41.05,-151.13],[31.41,-154.33],[29.46,-147.99],[25.54,-135.87],[24.11,-135.94],[23.58,-132.67],[-3.75,-88.9],[-3.43,-88.69],[-13.08,-73.24],[-13.42,-73.46],[-21.5,-60.32],[-30.23,-45.77],[-33.02,-40.26],[-34.7,-36.87],[-35.37,-36.85],[-39.92,-27.29],[-42.62,-18.71],[-41.94,-18.09],[-43.34,-6.76],[-46.37,-7.14],[-47.52,5.61],[-44.92,5.94],[-46.53,19.0],[-49.08,18.57],[-50.55,28.24],[-46.85,28.84],[-48.32,40.57],[-52.9,39.92],[-53.29,42.67],[-48.67,43.33],[-49.19,47.54],[-49.88,54.69],[-52.78,54.41],[-53.1,57.53],[-50.17,57.82],[-51.04,66.99],[-53.86,66.74],[-54.07,69.11],[-51.28,69.36],[-51.87,75.6],[-52.03,77.2],[-54.91,76.55],[-55.34,78.49],[-52.22,79.2],[-52.96,86.66],[-55.42,86.38],[-55.64,88.34],[-53.13,88.63],[-53.75,95.49],[-56.99,95.15],[-57.28,98.08],[-54.02,98.41],[-54.3,101.58],[-54.61,104.72],[-58.45,104.34],[-58.82,107.99],[-54.99,108.39],[-56.02,117.09],[-56.73,123.36]],"walls":[[[-56.73,125.25],[-61.44,125.24],[-61.45,129.24],[-56.73,129.24],[-56.73,141.91],[-54.9,142.37],[7.87,128.65],[8.84,133.12],[13.69,132.06],[12.73,127.59],[15.14,127.06],[14.33,123.29],[18.06,120.29],[20.39,113.2],[24.8,104.21],[26.02,100.55],[36.11,91.36],[43.79,88.05],[46.33,74.78],[51.51,64.63],[60.88,57.32],[62.41,55.9],[63.93,54.38],[64.55,53.44],[65.04,52.43],[65.87,50.76],[79.18,45.16],[83.19,25.03],[82.71,5.95],[94.29,-1.72],[95.36,-19.48],[95.41,-20.24],[95.46,-21.18],[96.62,-40.34],[101.9,-69.17],[94.09,-72.45],[91.44,-82.17],[90.14,-99.31],[90.29,-99.84],[89.51,-105.23],[93.25,-106.54],[92.08,-110.87],[88.7,-109.95],[87.03,-113.84],[85.11,-133.83],[79.71,-133.89],[79.71,-133.31],[84.59,-133.27],[86.46,-113.69],[88.33,-109.36],[89.7,-99.88],[89.56,-99.36],[90.87,-82.07],[93.6,-72.03],[101.26,-68.82],[98.27,-52.54],[96.05,-40.41],[95.95,-38.91],[94.88,-20.96],[94.84,-20.17],[94.78,-19.38],[94.32,-11.67],[93.93,-10.81],[93.69,-10.26],[93.6,-9.59],[93.17,-3.14],[92.77,-2.63],[90.65,-1.14],[88.17,0.42],[85.03,2.38],[81.45,4.84],[81.58,6.97],[81.41,7.09],[81.22,7.47],[81.19,7.89],[81.34,8.28],[81.65,8.59],[81.67,8.59],[81.68,8.79],[81.46,8.98],[81.26,9.35],[81.24,9.77],[81.39,10.17],[81.68,10.47],[81.78,10.5],[81.79,10.71],[81.6,10.88],[81.4,11.25],[81.39,11.67],[81.54,12.06],[81.83,12.36],[81.85,12.43],[81.57,14.22],[81.27,16.77],[81.3,17.8],[81.42,20.29],[81.59,20.92],[82.05,22.63],[82.23,23.29],[82.17,23.37],[81.84,23.64],[81.66,24.01],[81.63,24.44],[81.77,24.83],[82.08,25.13],[82.21,25.18],[81.95,25.24],[81.63,25.51],[81.44,25.89],[81.41,26.31],[81.56,26.71],[81.84,26.99],[81.6,27.06],[81.26,27.32],[81.07,27.69],[81.06,28.12],[81.2,28.52],[81.5,28.81],[81.56,28.83],[81.2,28.94],[80.88,29.2],[80.69,29.58],[80.66,30.0],[80.81,30.39],[81.1,30.69],[81.23,30.77],[80.86,30.87],[80.54,31.13],[80.35,31.51],[80.32,31.93],[80.47,32.32],[80.76,32.63],[80.83,32.71],[80.73,33.2],[78.44,44.36],[78.02,44.69],[72.13,47.04],[66.16,48.95],[64.75,49.78],[63.18,51.29],[62.29,52.6],[61.76,53.76],[61.2,55.28],[60.89,55.82],[59.91,56.65],[56.88,59.19],[50.84,64.11],[49.91,65.12],[49.2,66.34],[48.12,68.21],[45.31,74.34],[45.06,75.48],[42.7,87.48],[37.88,89.84],[35.79,90.86],[25.53,100.22],[24.27,103.99],[19.86,112.98],[17.56,119.94],[15.54,121.58],[13.7,123.07],[14.45,126.62],[12.04,127.15],[13.0,131.62],[9.28,132.44],[8.32,127.96],[-3.97,130.65],[-54.89,141.78],[-56.15,141.45],[-56.16,123.4],[-55.45,117.15],[-54.37,108.15],[-53.72,101.63],[-52.39,86.72],[-51.3,75.66],[-48.63,47.6],[-46.27,28.88],[-45.96,19.04],[-43.42,-1.45],[-42.75,-6.85],[-41.33,-18.33],[-41.96,-18.89],[-39.38,-27.08],[-35.01,-36.28],[-34.34,-36.29],[-32.5,-40.0],[-29.73,-45.49],[-21.0,-60.01],[-13.24,-72.66],[-12.9,-72.44],[-2.63,-88.88],[-2.96,-89.08],[6.72,-104.6],[24.12,-132.47],[24.6,-135.33],[25.96,-135.27],[30.0,-147.81],[31.79,-153.59],[35.37,-152.39],[40.74,-150.67],[40.32,-149.34],[40.04,-135.87],[40.01,-132.4],[40.97,-132.42],[41.05,-151.13],[31.41,-154.33],[29.46,-147.99],[25.54,-135.87],[24.11,-135.94],[23.58,-132.67],[-3.75,-88.9],[-3.43,-88.69],[-13.08,-73.24],[-13.42,-73.46],[-21.5,-60.32],[-30.23,-45.77],[-33.02,-40.26],[-34.7,-36.87],[-35.37,-36.85],[-39.92,-27.29],[-42.62,-18.71],[-41.94,-18.09],[-43.34,-6.76],[-46.37,-7.14],[-47.52,5.61],[-44.92,5.94],[-46.53,19.0],[-49.08,18.57],[-50.55,28.24],[-46.85,28.84],[-48.32,40.57],[-52.9,39.92],[-53.29,42.67],[-48.67,43.33],[-49.19,47.54],[-49.88,54.69],[-52.78,54.41],[-53.1,57.53],[-50.17,57.82],[-51.04,66.99],[-53.86,66.74],[-54.07,69.11],[-51.28,69.36],[-51.87,75.6],[-52.03,77.2],[-54.91,76.55],[-55.34,78.49],[-52.22,79.2],[-52.96,86.66],[-55.42,86.38],[-55.64,88.34],[-53.13,88.63],[-53.75,95.49],[-56.99,95.15],[-57.28,98.08],[-54.02,98.41],[-54.3,101.58],[-54.61,104.72],[-58.45,104.34],[-58.82,107.99],[-54.99,108.39],[-56.02,117.09],[-56.73,123.36]],[[78.44,-137.95],[79.43,-153.16],[67.31,-153.52],[67.35,-155.17],[65.91,-155.2],[65.84,-156.68],[67.37,-156.66],[67.4,-157.9],[65.77,-157.93],[65.67,-160.02],[67.44,-159.99],[67.46,-161.26],[65.61,-161.29],[65.5,-163.58],[67.5,-163.54],[67.6,-169.08],[65.23,-169.11],[65.28,-168.05],[64.25,-168.07],[64.49,-163.35],[65.01,-152.79],[65.02,-152.57],[78.35,-152.18],[77.42,-137.96]]],"nike":[[38.16,-142.58],[32.79,-144.3],[35.37,-152.39],[40.74,-150.67],[40.32,-149.34]],"nike_cella":[[38.15,-145.12],[34.28,-146.36],[35.41,-149.9],[39.28,-148.66]],"propylaia":[[41.54,-126.74],[41.58,-131.02],[42.18,-131.02],[43.04,-131.01],[42.93,-115.78],[49.15,-115.74],[49.17,-118.3],[48.77,-118.3],[48.77,-119.39],[49.18,-119.39],[49.18,-119.81],[49.87,-119.81],[49.87,-119.38],[51.05,-119.38],[51.94,-119.37],[52.82,-119.36],[60.91,-119.31],[61.02,-130.9],[61.88,-130.89],[62.49,-130.89],[62.46,-126.53],[64.03,-126.51],[64.07,-132.27],[64.14,-138.09],[77.42,-137.96],[78.44,-137.95],[79.76,-137.93],[79.71,-133.89],[79.71,-133.31],[79.61,-125.53],[61.84,-125.7],[61.78,-118.64],[62.19,-118.64],[62.15,-113.44],[62.98,-113.44],[62.97,-111.16],[63.18,-111.16],[63.14,-106.51],[52.73,-106.58],[51.84,-106.58],[51.24,-106.58],[51.24,-107.21],[50.95,-107.22],[41.99,-107.28],[41.39,-107.28],[41.45,-116.26],[42.07,-116.26],[42.15,-125.9],[33.01,-126.0],[33.1,-135.91],[40.04,-135.87],[40.01,-132.4],[39.96,-126.76]],"old_temple":[[58.14,-30.11],[44.67,-31.08],[42.15,3.16],[55.61,4.13]],"old_temple_in":[[[43.76,0.43],[43.66,1.85],[54.32,2.62],[54.43,1.2]],[[43.86,-0.98],[45.75,-0.84],[46.52,-11.34],[44.63,-11.48]],[[53.41,-10.85],[55.3,-10.71],[54.54,-0.2],[52.64,-0.34]],[[52.34,-10.92],[51.56,-0.42],[46.83,-0.76],[47.61,-11.27]],[[51.4,-12.41],[55.41,-12.12],[55.88,-18.43],[51.87,-18.72]],[[45.2,-19.2],[44.74,-12.89],[49.83,-12.52],[50.3,-18.84]],[[56.52,-27.24],[56.64,-28.8],[45.96,-29.57],[45.31,-20.61],[55.97,-19.83],[56.42,-25.82],[46.75,-26.53],[46.85,-27.93]]],"erechtheion":[[70.25,15.21],[70.28,15.58],[70.88,15.68],[71.47,11.82],[72.16,11.92],[73.31,4.49],[72.98,3.65],[72.79,3.63],[73.25,0.67],[77.67,1.35],[77.93,1.78],[80.04,2.1],[82.01,-10.7],[72.59,-12.14],[72.01,-8.45],[68.04,-9.05],[68.0,-8.77],[61.17,-9.81],[60.91,-9.85],[60.95,-10.09],[58.05,-10.54],[57.1,-4.3],[56.71,-4.36],[56.68,-4.19],[56.67,-3.79],[56.95,-3.58],[58.09,-3.45],[58.27,-3.59],[60.58,-3.24],[58.04,13.35]],"p_columns":[[-14.58,-33.99,0.79,1.8,11.4],[-14.58,-30.19,0.79,1.8,11.4],[-14.58,-25.89,0.79,1.8,11.4],[-14.57,-21.58,0.79,1.8,11.4],[-14.57,-12.98,0.79,1.8,11.4],[-14.56,-17.28,0.79,1.8,11.4],[-14.56,-8.66,0.79,1.8,11.4],[-14.56,-4.37,0.79,1.8,11.4],[-14.56,0.04,0.79,1.8,10.5],[-14.56,8.73,0.79,1.8,4.5],[-14.56,17.35,0.79,1.8,11.4],[-14.55,4.43,0.79,1.8,5.5],[-14.55,13.04,0.79,1.8,9.5],[-14.55,21.66,0.79,1.8,11.4],[-14.55,25.96,0.79,1.8,11.4],[-14.54,30.27,0.79,1.8,11.4],[-14.54,34.06,0.79,1.8,11.4],[-10.79,-34.0,0.79,1.8,11.4],[-10.76,34.05,0.79,1.8,11.4],[-9.95,-28.82,0.7,2.5,11.8],[-9.93,28.86,0.7,2.5,11.8],[-6.5,-34.01,0.79,1.8,11.4],[-6.46,34.05,0.79,1.8,11.4],[-6.21,-28.83,0.7,2.5,11.8],[-6.18,28.85,0.7,2.5,11.8],[-2.2,-34.01,0.79,1.8,11.4],[-2.16,34.03,0.79,1.8,11.4],[-1.98,-28.84,0.7,2.5,11.8],[-1.96,28.84,0.7,2.5,11.8],[1.95,-28.85,0.7,2.5,11.8],[1.98,28.82,0.7,2.5,10.0],[2.18,-34.03,0.79,1.8,11.4],[2.2,34.02,0.79,1.8,11.4],[6.17,-28.86,0.7,2.5,11.8],[6.2,28.81,0.7,2.5,8.0],[6.46,-34.04,0.79,1.8,11.4],[6.49,34.01,0.79,1.8,11.4],[9.93,-28.87,0.7,2.5,11.8],[9.95,28.8,0.7,2.5,6.0],[10.75,-34.05,0.79,1.8,11.4],[10.79,34.0,0.79,1.8,11.4],[14.54,-25.96,0.79,1.8,11.4],[14.55,-34.05,0.79,1.8,11.4],[14.55,-30.26,0.79,1.8,11.4],[14.55,-17.34,0.79,1.8,11.4],[14.56,-21.64,0.79,1.8,11.4],[14.56,-13.04,0.79,1.8,11.4],[14.56,-8.74,0.79,1.8,11.4],[14.56,-4.43,0.79,1.8,11.4],[14.56,-0.03,0.79,1.8,11.4],[14.56,4.37,0.79,1.8,11.4],[14.57,8.67,0.79,1.8,11.4],[14.57,12.97,0.79,1.8,11.4],[14.57,17.28,0.79,1.8,11.4],[14.57,21.59,0.79,1.8,11.4],[14.57,25.89,0.79,1.8,11.4],[14.58,30.2,0.79,1.8,11.4],[14.58,33.99,0.79,1.8,11.4]],"w144928413":[[2.66,59.23],[1.29,58.39],[-0.3,58.15],[-1.87,58.54],[-3.16,59.5],[-4.0,60.88],[-4.24,62.48],[-3.86,64.05],[-2.91,65.35],[-1.53,66.18],[0.07,66.43],[1.63,66.04],[2.93,65.08],[3.76,63.7],[4.0,62.09],[3.62,60.53]],"w144928412":[[36.75,65.18],[36.62,57.33],[28.11,45.83],[20.86,49.25],[15.41,63.43],[15.52,70.98],[32.68,76.23]],"w144928418":[[94.27,-49.54],[93.8,-39.01],[85.75,-39.35],[83.7,-39.44],[84.15,-49.97]],"w426887028":[[67.44,-147.97],[67.1,-144.31],[66.91,-142.22],[61.82,-142.68],[62.36,-148.42]],"w910010374":[[41.61,-168.6],[41.75,-174.65],[49.02,-174.48],[48.89,-169.18],[52.44,-169.1],[56.04,-169.01],[56.16,-174.32],[64.98,-174.11],[65.23,-169.11],[65.28,-168.05],[64.25,-168.07],[52.42,-168.35]],"w910010375":[[-24.29,67.53],[-24.33,73.35],[-23.82,73.35],[-23.83,74.15],[-23.2,74.15],[-23.32,93.29],[-24.45,93.29],[-24.49,99.85],[-25.05,99.85],[-25.2,121.79],[-33.61,121.76],[-33.62,123.93],[-34.94,123.92],[-34.98,129.39],[-47.8,129.33],[-47.77,125.35],[-43.94,125.36],[-43.93,123.88],[-50.83,123.85],[-50.75,109.68],[-44.81,109.71],[-44.74,99.07],[-43.82,99.07],[-43.67,74.9],[-44.6,74.89],[-44.55,67.44],[-37.97,67.47],[-37.95,64.87],[-30.86,64.89],[-30.88,67.5]],"w910010357":[[68.39,-110.7],[69.38,-110.82],[69.48,-119.92],[69.09,-119.92],[69.14,-124.86],[79.26,-124.76],[79.2,-119.83],[79.53,-119.82],[79.44,-111.95],[80.43,-112.05],[81.08,-106.15],[86.9,-106.8],[87.71,-99.33],[69.85,-97.33]],"w910010294":[[12.83,-98.6],[13.31,-98.31],[9.02,-91.36],[31.94,-77.13],[35.88,-84.47],[19.37,-94.72],[18.31,-95.37],[12.5,-98.99]],"Athena Promachos statue place":[52.65,-73.82],"Holy olive tree":[66.63,-12.71],"Pandroseion":[67.19,-16.99]}""")


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sphere(r, c, seg=10, rings=6):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def inside(x, z, P):
    ins = False
    n = len(P)
    for i in range(n):
        (xi, zi), (xj, zj) = P[i], P[i - 1]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            ins = not ins
    return ins


def seg_dist(p, a, b):
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    d = b - a
    t = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * d)))


def ring_dist(p, P):
    return min(seg_dist(p, P[i - 1], P[i]) for i in range(len(P)))


def densify(P, step):
    out = []
    for i in range(len(P)):
        a, b = np.array(P[i], float), np.array(P[(i + 1) % len(P)], float)
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out += [tuple(a + (b - a) * k / n) for k in range(n)]
    return out


def simplify(P, tol):
    """Douglas–Peucker on a closed ring."""
    P = [tuple(p) for p in P]

    def dp(pts):
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        d = [seg_dist(p, a, b) for p in pts[1:-1]]
        k = int(np.argmax(d)) + 1
        if d[k - 1] > tol:
            return dp(pts[:k + 1])[:-1] + dp(pts[k:])
        return [a, b]
    far = max(range(len(P)), key=lambda i: math.dist(P[0], P[i]))
    return dp(P[:far + 1])[:-1] + dp(P[far:] + [P[0]])[:-1]


def tint(c, k=0.02):
    return tuple(np.clip(np.array(c) * (1 + rng.normal(0, k)), 0, 1))


def marble_paint(m, seed=1, honey=0.35):
    """Pentelic marble: off-white, honey-gold where weathered (more low down and in patches), a few grey streaks."""
    p = m.v
    k = np.clip(honey * (0.55 + 0.9 * fbm(p, (0.15, 0.08, 0.15), seed, 3)) + 0.12 * sstep(6, 0, p[:, 1]), 0, 0.9)
    g = sstep(0.62, 0.8, fbm(p, (0.3, 0.05, 0.3), seed + 5, 2)) * 0.25
    c = m.c * (1 - k[:, None]) + np.array(HONEY) * k[:, None]
    m.c = c * (1 - g[:, None]) + np.array((0.58, 0.56, 0.52)) * g[:, None]
    return m


# ── Local frames (Erechtheion, Nike): build in (u, y, w), place with a proper rotation ─────────────────
def framed(m, o, U):
    """Local x = u along U, local z = w along W (W = U turned so that the frame keeps its handedness)."""
    U = np.asarray(U, float) / np.linalg.norm(U)
    W = np.array([U[1], -U[0]])
    if U[0] * W[1] - U[1] * W[0] < 0:
        W = -W
    v = m.v.copy()
    m.v = np.stack([o[0] + v[:, 0] * U[0] + v[:, 2] * W[0], v[:, 1], o[1] + v[:, 0] * U[1] + v[:, 2] * W[1]], 1)
    return m


def frame_check(U):
    U = np.asarray(U, float) / np.linalg.norm(U)
    W = np.array([U[1], -U[0]])
    if U[0] * W[1] - U[1] * W[0] < 0:
        W = -W
    return U, W


# ── Columns ─────────────────────────────────────────────────────────────
def fluted_shaft(h, rb, rt, c, flutes=20, depth=0.045, rings=5, entasis=0.018):
    """A fluted shaft with entasis, standing on y = 0, no caps (a broken top gets its own cap)."""
    n = flutes * 2
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    k = np.where(np.arange(n) % 2 == 0, 1.0, 1.0 - depth)
    V, F = [], []
    for i in range(rings + 1):
        t = i / rings
        r = rb + (rt - rb) * t + entasis * rb * math.sin(math.pi * t)
        V += [(r * k[j] * math.cos(a[j]), h * t, r * k[j] * math.sin(a[j])) for j in range(n)]
    for i in range(rings):
        for j in range(n):
            jj = (j + 1) % n
            p, q = i * n + j, i * n + jj
            F += [(p, p + n, q + n), (p, q + n, q)]
    return Mesh(V, F, c)


def disc(r, y, c, seg=12):
    return cyl(r, r, 0.001, c, seg).T(0, y, 0)


def doric(x, z, y0, h, rb, rt, c=MARBLE, top=None, flutes=20, abacus=None):
    """Doric column: shaft to h − capital, echinus and abacus; top < h: a broken column cut at y0 + top."""
    cap_h = 0.0825 * h
    if top is not None and top < h - 0.3:
        s = fluted_shaft(top, rb, rb + (rt - rb) * top / (h - cap_h), c, flutes, rings=max(2, int(top / 2.2)))
        ms = [s, cyl(rb * 0.97, rb * 0.97, 0.02, c, flutes, caps=True).T(0, top - 0.02, 0)]
    else:
        ab = abacus or rt * 1.38
        ms = [fluted_shaft(h - cap_h, rb, rt, c, flutes),
              lathe([(rt * 0.98, h - cap_h - 0.01), (rt * 1.05, h - cap_h + 0.08), (ab * 1.02, h - cap_h + 0.45 * cap_h),
                     (ab * 1.04, h - 0.34 * cap_h)], c, 16),
              box(ab * 2.08, 0.34 * cap_h, ab * 2.08, c).T(0, h - 0.34 * cap_h, 0)]
    m = merge(ms)
    m.c = np.tile(tint(c, 0.03), (len(m.v), 1))
    return m.T(x, y0, z)


def ionic(x, z, y0, h, rb, rt, c=MARBLE, face="z", flutes=12, broken=None):
    """Ionic column: moulded base, fluted shaft, capital with two volutes (axis along 'face')."""
    base_h = 0.06 * h
    cap_h = 0.07 * h
    if broken is not None:
        ms = [lathe([(rb * 1.45, 0), (rb * 1.45, base_h * 0.4), (rb * 1.2, base_h * 0.6), (rb * 1.3, base_h * 0.8), (rb, base_h)], c, 14),
              fluted_shaft(broken - base_h, rb, rb, c, flutes, 0.06, 2, 0).T(0, base_h, 0),
              disc(rb * 0.95, broken, c, 2 * flutes)]
        return merge(ms).T(x, y0, z)
    ms = [lathe([(rb * 1.45, 0), (rb * 1.45, base_h * 0.4), (rb * 1.2, base_h * 0.6), (rb * 1.3, base_h * 0.8), (rb, base_h)], c, 14),
          fluted_shaft(h - base_h - cap_h, rb, rt, c, flutes, 0.06, 4, 0.012).T(0, base_h, 0),
          lathe([(rt, h - cap_h), (rt * 1.15, h - cap_h * 0.55), (rt * 1.15, h - cap_h * 0.45)], c, 14)]
    vw, vr = rt * 2.9, cap_h * 0.42                 # volute pad: width across, radius of the scrolls
    if face == "z":
        ms.append(B(-vw / 2, vw / 2, h - cap_h * 0.75, h - cap_h * 0.18, -rt * 1.0, rt * 1.0, c))
        for s in (-1, 1):
            ms.append(cyl(vr, vr, rt * 2.1, c, 10).R(math.pi / 2, 0, 0).T(s * (vw / 2 - vr * 0.6), h - cap_h * 0.62, -rt * 1.05))
        ms.append(B(-rt * 1.35, rt * 1.35, h - cap_h * 0.18, h, -rt * 1.35, rt * 1.35, c))
    else:
        ms.append(B(-rt * 1.0, rt * 1.0, h - cap_h * 0.75, h - cap_h * 0.18, -vw / 2, vw / 2, c))
        for s in (-1, 1):
            ms.append(cyl(vr, vr, rt * 2.1, c, 10).R(0, 0, math.pi / 2).T(rt * 1.05, h - cap_h * 0.62, s * (vw / 2 - vr * 0.6)))
        ms.append(B(-rt * 1.35, rt * 1.35, h - cap_h * 0.18, h, -rt * 1.35, rt * 1.35, c))
    m = merge(ms)
    m.c = np.tile(tint(c, 0.03), (len(m.v), 1))
    return m.T(x, y0, z)


def caryatid(x, z, y0, c=MARBLE):
    """A maiden of the south porch of the Erechtheion, 2.31 m: plinth, draped body, head, polos and capital."""
    prof = [(0.27, 0.12), (0.28, 0.45), (0.25, 1.0), (0.24, 1.25), (0.27, 1.5), (0.25, 1.68), (0.17, 1.78), (0.09, 1.84)]
    ms = [box(0.62, 0.12, 0.62, c), lathe(prof, c, 12),
          sphere(0.13, c, 10, 6).T(0, 1.96, 0),
          lathe([(0.12, 1.98), (0.2, 2.12), (0.25, 2.18)], c, 12),
          box(0.62, 0.13, 0.62, c).T(0, 2.18, 0)]
    m = merge(ms)
    m.c = np.tile(tint(c, 0.03), (len(m.v), 1))
    return m.T(x, y0, z)


# ── The rock: the plateau ───────────────────────────────────────────────
def top(x, z):
    """Level of the plateau (y) at model x, z: flat around the Parthenon, falling to the Propylaia, the stairs
    down to the Beulé Gate, the lower ground west and north of the Erechtheion."""
    x = np.asarray(x, float); z = np.asarray(z, float)
    y = -4.5 * sstep(-45, -105, z) - 1.8 * sstep(-112, -116, z)
    fore = sstep(41.0, 43.0, x) * sstep(68.0, 66.0, x)
    y = y - fore * 10.7 * np.clip((-133.0 - z) / 35.0, 0, 1)
    low = (sstep(59.5, 61.5, x) * sstep(-40, -37, z) * sstep(-8.5, -10.5, z)
           + sstep(70.0, 72.0, x) * sstep(-10.5, -8.5, z) * sstep(4.5, 2.5, z))
    y = y - 3.2 * np.clip(low, 0, 1)
    return y


def clip_rect(P, x0, x1, z0, z1):
    """Sutherland–Hodgman: a (possibly concave) ring cut to an axis-aligned rectangle."""
    def cut(P, k, v, keep_ge):
        out = []
        n = len(P)
        for i in range(n):
            a, b = P[i - 1], P[i]
            ina = (a[k] >= v) if keep_ge else (a[k] <= v)
            inb = (b[k] >= v) if keep_ge else (b[k] <= v)
            if inb:
                if not ina:
                    t = (v - a[k]) / (b[k] - a[k])
                    out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
                out.append(b)
            elif ina:
                t = (v - a[k]) / (b[k] - a[k])
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        return out
    for k, v, ge in ((0, x0, True), (0, x1, False), (1, z0, True), (1, z1, False)):
        P = cut(P, k, v, ge)
        if len(P) < 3:
            return []
    return P


def plateau(P, cell=3.0):
    """The top of the rock: a height field over the outline P, cut cell by cell to it."""
    P = [tuple(p) for p in P]
    A = np.array(P)
    x0, z0 = A.min(0) - 0.1
    x1, z1 = A.max(0) + 0.1
    V, F = [], []
    nx, nz = int(math.ceil((x1 - x0) / cell)), int(math.ceil((z1 - z0) / cell))
    for i in range(nx):
        for j in range(nz):
            cx0, cz0 = x0 + i * cell, z0 + j * cell
            cx1, cz1 = cx0 + cell, cz0 + cell
            corners = [(cx0, cz0), (cx1, cz0), (cx1, cz1), (cx0, cz1)]
            crosses = np.any((A[:, 0] > cx0) & (A[:, 0] < cx1) & (A[:, 1] > cz0) & (A[:, 1] < cz1))
            ins = [inside(px, pz, P) for px, pz in corners]
            if all(ins) and not crosses and not any(ring_dist(c_, P) < 1e-6 for c_ in corners[:1]):
                # make sure no edge of P passes through the cell
                edge_hit = False
                for k in range(len(P)):
                    (ax, az), (bx, bz) = P[k - 1], P[k]
                    if max(ax, bx) < cx0 or min(ax, bx) > cx1 or max(az, bz) < cz0 or min(az, bz) > cz1:
                        continue
                    edge_hit = True
                    break
                if not edge_hit:
                    o = len(V); V += corners
                    F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
                    continue
            if not any(ins) and not crosses:
                hit = False
                for k in range(len(P)):
                    (ax, az), (bx, bz) = P[k - 1], P[k]
                    if max(ax, bx) < cx0 or min(ax, bx) > cx1 or max(az, bz) < cz0 or min(az, bz) > cz1:
                        continue
                    hit = True
                    break
                if not hit:
                    continue
            Q = clip_rect(P, cx0, cx1, cz0, cz1)
            if len(Q) < 3 or abs(ring_area(Q)) < 1e-3:
                continue
            pts, tris = triangulate(Q)
            o = len(V); V += [tuple(p) for p in pts]
            F += [tuple(t[::-1] + o) for t in tris]
    V = np.array(V, float)
    y = top(V[:, 0], V[:, 1])
    m = Mesh(np.stack([V[:, 0], y, V[:, 1]], 1), F, PLATEAU)
    return m


def plateau_paint(m):
    p = m.v
    k = sstep(0.55, 0.75, fbm(p, 0.06, 31, 3)) * 0.6
    c = np.array(PLATEAU) * (0.93 + 0.1 * fbm(p, 0.4, 32, 2))[:, None]
    m.c = c * (1 - k[:, None]) + np.array(ROCK) * k[:, None]
    return m


# ── The walls and the rock face ─────────────────────────────────────────
def wall_height(x):
    """How far the walls stand over their foot: ~15 m on the south (Kimon), ~10 m on the north (Themistokles)."""
    return 10.0 + 5.0 * sstep(30.0, -30.0, x)


def band(P, ytop, ybot, c, c_top):
    """A wall along the ring P (a band polygon drawn as one ring) with a top and a foot that vary per vertex."""
    P = [tuple(p) for p in P]
    if ring_area(P) < 0:
        P = P[::-1]
    A = np.array(P)
    yt, yb = ytop(A[:, 0], A[:, 1]), ybot(A[:, 0], A[:, 1])
    V, F = [], []
    n = len(P)
    for i in range(n):
        j = (i + 1) % n
        o = len(V)
        V += [(A[i, 0], yb[i], A[i, 1]), (A[j, 0], yb[j], A[j, 1]), (A[j, 0], yt[j], A[j, 1]), (A[i, 0], yt[i], A[i, 1])]
        F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    side = Mesh(V, F, c)
    pts, tris = triangulate(P)
    tp = Mesh(np.stack([pts[:, 0], ytop(pts[:, 0], pts[:, 1]), pts[:, 1]], 1), tris[:, ::-1], c_top)
    return merge([side, tp])


def circuit_walls():
    ms = []
    for P in DATA["walls"]:
        ms.append(band(P, lambda x, z: top(x, z) + 1.2,
                       lambda x, z: top(x, z) - wall_height(x) - 8.0, WALL, WALL_D))
    m = merge(ms)
    p = m.v
    k = fbm(p, (0.25, 0.6, 0.25), 41, 2)                         # courses and patches of ashlar
    m.c = np.clip(m.c * (0.86 + 0.24 * k)[:, None], 0, 1)
    return m


def rock_face():
    """Below the walls (and around the western forecourt) the rock goes down, flaring out, to Y_BOT."""
    P = densify(simplify(DATA["acropolis"], 0.6), 5.0)
    if ring_area(P) < 0:
        P = P[::-1]
    A = np.array(P)
    n = len(A)
    nrm = np.zeros((n, 2))
    for i in range(n):
        e = A[(i + 1) % n] - A[i - 1]
        nrm[i] = (e[1], -e[0])
    for _ in range(6):                                          # smooth the directions, so rings do not cross
        nrm = (np.roll(nrm, 1, 0) + 2 * nrm + np.roll(nrm, -1, 0)) / 4
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    if ring_area(P) > 0 and np.mean([inside(*(A[i] + nrm[i] * 2), P) for i in range(n)]) > 0.5:
        nrm = -nrm                                              # make them point out of the rock
    wl = [np.array(w) for w in DATA["walls"]]
    walled = np.array([min(ring_dist(A[i], w) for w in wl) < 3.0 for i in range(n)])
    walled = np.convolve(np.r_[walled[-2:], walled, walled[:2]].astype(float), np.ones(5) / 5, "same")[2:-2] > 0.5
    y_top = top(A[:, 0], A[:, 1])
    y0 = np.where(walled, y_top - wall_height(A[:, 0]), y_top)
    spread = 1.0 + 1.4 * sstep(-140, -175, A[:, 1]) + 0.5 * sstep(100, 140, A[:, 1])
    fr = [0.0, 0.1, 0.24, 0.42, 0.62, 0.82, 1.0]
    off = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0]
    rings = []
    for k, (f, o) in enumerate(zip(fr, off)):
        jit = (fbm(np.c_[A[:, 0], np.full(n, k * 7.0), A[:, 1]], 0.08, 51, 2) - 0.5) * 2.5 * (k > 0)
        d = np.where(walled, -0.8, 0.0) + (o * spread + jit) * (k > 0)
        Q = A + nrm * d[:, None]
        y = y0 + (Y_BOT - y0) * f
        if 0 < k < len(fr) - 1:
            y = y + (fbm(np.c_[Q[:, 0], np.full(n, 3.0), Q[:, 1]], 0.12, 52, 2) - 0.5) * 3.0
        rings.append(np.c_[Q[:, 0], y, Q[:, 1]])
    V = np.vstack(rings)
    F = []
    for k in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, c_, d = k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i
            F += [(a, c_, b), (a, d, c_)]
    m = Mesh(V, F, ROCK)
    # make the faces point outwards (towards the normal of the ring)
    nm = np.cross(V[m.f[:, 1]] - V[m.f[:, 0]], V[m.f[:, 2]] - V[m.f[:, 0]])
    ctr = V[m.f].mean(1)
    if np.mean(np.einsum("ij,ij->i", nm[:, [0, 2]], ctr[:, [0, 2]] - A.mean(0))) < 0:
        m.f = m.f[:, ::-1]
    p = m.v
    red = sstep(0.55, 0.75, fbm(p, (0.05, 0.15, 0.05), 53, 3)) * 0.7
    c = np.array(ROCK) * (0.86 + 0.24 * fbm(p, (0.2, 0.5, 0.2), 54, 2))[:, None]
    c = c * (1 - red[:, None]) + np.array(ROCK_R) * red[:, None]
    depth = np.clip((y0.min() - p[:, 1]) / 10.0, 0, 1)
    veg = sstep(0.45, 0.6, fbm(p, 0.07, 55, 3)) * sstep(-20, -32, p[:, 1]) * 0.85
    m.c = c * (1 - veg[:, None]) + np.array(SCRUB) * veg[:, None]
    del depth
    return m


# ── The Parthenon ───────────────────────────────────────────────────────
S_TOP = 1.8                      # stylobate
COL_H = 10.43
Y_ARCH, Y_FRIEZE, Y_CORN, Y_CTOP = 12.23, 13.58, 14.93, 15.55
X_OUT, X_IN = 15.35, 13.66       # outer and inner faces of the peristyle entablature
Z_OUT, Z_IN = 35.35, 33.66


def hmap(h):
    """OSM relative heights of the Parthenon -> the published ones."""
    if h <= 1.8:
        return h
    if h <= 11.4:
        return 1.8 + (h - 1.8) * COL_H / 9.6
    return Y_ARCH + (h - 11.4) * (Y_CTOP - Y_ARCH) / (15.34 - 11.4)


def entab_x(side, z0, z1, parts=("a", "f", "c")):
    """Entablature along a long side (side = +1 north, −1 south) between z0 and z1."""
    s = side
    xa, xb = sorted((s * X_IN, s * X_OUT))
    ms = []
    if "a" in parts:
        ms += [B(xa, xb, Y_ARCH, Y_FRIEZE - 0.2, z0, z1, MARBLE),
               B(*sorted((s * X_IN, s * (X_OUT + 0.06))), Y_FRIEZE - 0.2, Y_FRIEZE, z0, z1, MARBLE)]      # taenia
    if "f" in parts:
        ms.append(B(*sorted((s * X_IN, s * (X_OUT - 0.05))), Y_FRIEZE, Y_CORN, z0, z1, MARBLE))
    if "c" in parts:
        ms.append(B(*sorted((s * (X_IN + 0.3), s * (X_OUT + 0.6))), Y_CORN, Y_CTOP, z0, z1, MARBLE))
    return ms


def entab_z(side, x0, x1, parts=("a", "f", "c")):
    s = side
    za, zb = sorted((s * Z_IN, s * Z_OUT))
    ms = []
    if "a" in parts:
        ms += [B(x0, x1, Y_ARCH, Y_FRIEZE - 0.2, za, zb, MARBLE),
               B(x0, x1, Y_FRIEZE - 0.2, Y_FRIEZE, *sorted((s * Z_IN, s * (Z_OUT + 0.06))), MARBLE)]
    if "f" in parts:
        ms.append(B(x0, x1, Y_FRIEZE, Y_CORN, *sorted((s * Z_IN, s * (Z_OUT - 0.05))), MARBLE))
    if "c" in parts:
        ms.append(B(x0, x1, Y_CORN, Y_CTOP, *sorted((s * (Z_IN + 0.3), s * (Z_OUT + 0.6))), MARBLE))
    return ms


def triglyphs_x(side, zs):
    s = side
    return [B(*sorted((s * (X_OUT - 0.05), s * (X_OUT + 0.04))), Y_FRIEZE + 0.02, Y_CORN - 0.02, z - 0.42, z + 0.42, (0.80, 0.76, 0.67))
            for z in zs]


def triglyphs_z(side, xs):
    s = side
    return [B(x - 0.42, x + 0.42, Y_FRIEZE + 0.02, Y_CORN - 0.02, *sorted((s * (Z_OUT - 0.05), s * (Z_OUT + 0.04))), (0.80, 0.76, 0.67))
            for x in xs]


def vprism(poly, z0, z1, c):
    """A vertical polygon (x, y) extruded along z from z0 to z1 (pediments)."""
    ring = [(px, -py) for px, py in poly]
    from meshkit import extrude
    m = extrude(ring, z0, z1, c, bottom=True)
    return m.R(math.pi / 2, 0, 0)


def pediment(side, xlo=-X_OUT, xhi=X_OUT):
    """Tympanum and raking cornices of the west (side −1) or east (+1) front, cut to xlo..xhi (ruined fronts)."""
    half = X_OUT + 0.6
    rise = 3.65
    apex = Y_CTOP + rise
    ms = []
    xe = half * (1 - 0.55 / rise)
    tym = clip_rect([(-xe, Y_CTOP), (xe, Y_CTOP), (0.0, apex - 0.55)], xlo, xhi, -1e9, 1e9)
    if len(tym) >= 3:
        ms.append(vprism(tym, *sorted((side * (Z_OUT - 0.75), side * (Z_OUT - 1.6))), MARBLE))
    zc = sorted((side * (Z_OUT + 0.55), side * (Z_OUT - 1.6)))
    for s in (-1, 1):
        rake = [(0.0, apex - 0.6), (s * half, Y_CTOP - 0.6), (s * half, Y_CTOP), (0.0, apex)]
        P = clip_rect(rake, xlo - 0.6, xhi + 0.6, -1e9, 1e9)
        if len(P) >= 3 and abs(ring_area(P)) > 0.05:
            ms.append(vprism(P, *zc, MARBLE))
    return ms


def pediment_figures(side, xs):
    """A few figures left in a pediment (casts), as rounded blocks."""
    ms = []
    zc = side * (Z_OUT - 1.05)
    for x, h, w in xs:
        b = sphere(1.0, MARBLE_W, 8, 5).S(w, h, 0.45).T(x, Y_CTOP + h, zc)
        ms.append(b)
    return ms


def parthenon():
    ms = []
    # crepis (OSM rings) on its foundation
    ms.append(B(-16.9, 16.9, -3.0, 0.3, -36.4, 36.4, POROS))
    ms.append(B(-16.8, 16.8, 0.3, 0.8, -36.3, 36.3, MARBLE))
    ms.append(B(-16.15, 16.15, 0.8, 1.3, -35.65, 35.65, MARBLE))
    ms.append(B(-15.5, 15.5, 1.3, S_TOP, -35.0, 35.0, MARBLE))
    # the ramp of the east front (OSM 386428656 …) and of the west
    for s in (-1, 1):
        ms.append(B(-2.1, 2.1, 0.0, 1.1, *sorted((s * 35.6, s * 36.6)), MARBLE))
        ms.append(B(-2.1, 2.1, 1.1, 1.6, *sorted((s * 34.95, s * 35.6)), MARBLE))
    # cella floor (two steps)
    ms.append(B(-11.2, 11.2, S_TOP, 2.11, -30.1, 30.1, MARBLE))
    ms.append(B(-10.8, 10.8, 2.11, 2.5, -29.7, 29.7, MARBLE))
    cols = []
    for x, z, r, mh, h in DATA["p_columns"]:
        if r > 0.8:                                             # peristyle
            t = None if h >= 11.39 else hmap(h) - S_TOP
            c = MARBLE_W if rng.random() < 0.18 else MARBLE
            cols.append(doric(x, z, S_TOP, COL_H, 0.955, 0.74, c, t, abacus=1.0))
        else:                                                   # inner porches (prostyle hexastyle)
            hh = 9.1
            t = None if h >= 11.79 else (h - 2.5) * hh / 9.3
            cols.append(doric(x, z, 2.5, hh, 0.85, 0.66, MARBLE, t, abacus=0.85))
    ms += cols
    # peristyle entablature: north complete in the architrave, its frieze and cornice gone over the middle;
    # south gone over the four broken columns
    ms += entab_x(+1, -Z_IN, Z_IN, ("a",))
    ms += entab_x(+1, -Z_IN, -12.9, ("f", "c")) + entab_x(+1, 9.1, Z_IN, ("f", "c"))
    ms += entab_x(-1, -Z_IN, -5.4) + entab_x(-1, 16.3, Z_IN)
    ms += entab_z(+1, -X_OUT, X_OUT) + entab_z(-1, -X_OUT, X_OUT)
    zs_n = sorted(c[1] for c in DATA["p_columns"] if c[0] > 14)
    zs_s = sorted(c[1] for c in DATA["p_columns"] if c[0] < -14)
    tri_n = [z for z in zs_n[1:-1]] + [(a + b) / 2 for a, b in zip(zs_n, zs_n[1:])] + [-Z_OUT + 0.42, Z_OUT - 0.42]
    tri_n = [z for z in tri_n if not (-12.9 < z < 9.1)]
    tri_s = [z for z in zs_s[1:-1]] + [(a + b) / 2 for a, b in zip(zs_s, zs_s[1:])] + [-Z_OUT + 0.42, Z_OUT - 0.42]
    tri_s = [z for z in tri_s if not (-5.4 < z < 16.3)]
    ms += triglyphs_x(+1, tri_n) + triglyphs_x(-1, tri_s)
    xs_e = sorted(c[0] for c in DATA["p_columns"] if c[1] > 33.5 and c[2] > 0.8)
    tri_e = xs_e[1:-1] + [(a + b) / 2 for a, b in zip(xs_e, xs_e[1:])] + [-X_OUT + 0.42, X_OUT - 0.42]
    ms += triglyphs_z(+1, tri_e) + triglyphs_z(-1, tri_e)
    # pediments: the west complete (with a few figures at its north corner), two corners of the east
    ms += pediment(-1)
    ms += pediment_figures(-1, [(10.6, 1.0, 1.4), (12.6, 0.7, 0.9), (-12.4, 0.6, 1.0)])
    ms += pediment(+1, 8.0, X_OUT) + pediment(+1, -X_OUT, -8.0)
    # inner porches: architraves over the six west columns, over the three standing east ones
    for s, xa, xb in ((-1, -10.9, 10.9), (+1, -10.9, -1.0)):
        za, zb = sorted((s * 27.95, s * 29.75))
        ms.append(B(xa, xb, 11.6, 12.85, za, zb, MARBLE))
    # west cross wall of the cella with its great door
    for xa, xb in ((-10.7, -2.5), (2.5, 10.7)):
        ms.append(B(xa, xb, 2.5, hmap(12.54), -24.4, -21.8, MARBLE))
    ms.append(B(-10.7, 10.7, hmap(12.54), hmap(14.95), -24.4, -21.8, MARBLE))
    ms.append(B(-2.5, 2.5, 2.5, hmap(12.54), -23.2, -23.0, DARK))
    # stepped ruins of the cella walls (courses after OSM, from the west cross wall eastwards)
    south = [(2.5, 3.6, -2.5), (3.6, 4.65, -5.6), (4.65, 5.7, -6.8), (5.7, 6.75, -8.1), (6.75, 7.8, -12.8),
             (7.8, 8.85, -14.0), (8.85, 9.9, -19.4), (9.9, 10.95, -19.8), (10.95, 12.0, -20.2), (12.0, 13.05, -20.9)]
    north = [(2.5, 3.6, 25.9), (3.6, 4.65, -12.7), (4.65, 5.7, -13.3), (5.7, 6.75, -13.8), (6.75, 7.8, -14.4),
             (7.8, 8.85, -14.9), (8.85, 9.9, -15.9), (9.9, 10.95, -18.3), (10.95, 12.0, -18.9), (12.0, 13.05, -19.5)]
    for (x0, x1), rows in (((-10.7, -9.6), south), ((9.6, 10.7), north)):
        for h0, h1, zend in rows:
            ms.append(B(x0, x1, hmap(h0), hmap(h1), -21.8, zend, MARBLE))
    ms.append(B(-10.7, -9.6, 2.5, 3.69, 16.0, 26.0, MARBLE))
    m = merge(ms)
    return marble_paint(m, 3, 0.38)


# ── The Erechtheion (local: u from the east front westwards, w from the south wall northwards) ────────────
E_O = (58.04, 13.35)
E_U = (60.58 - 58.04, -3.24 - 13.35)
Y_E = 0.0
Y_W = Y_E - 3.2


def erechtheion():
    ms = []
    yb = Y_E + 0.5                                              # east stylobate
    # east porch
    ms.append(B(-0.3, 2.9, Y_E - 2.0, Y_E + 0.25, -0.3, 12.65, MARBLE))
    ms.append(B(0.0, 2.9, Y_E + 0.25, yb, 0.0, 12.35, MARBLE))
    for k in range(6):
        ms.append(ionic(0.6, 0.75 + 2.17 * k, yb, 6.58, 0.345, 0.29, MARBLE, face="x"))
    y_a = yb + 6.58
    ms.append(B(0.1, 3.2, y_a, y_a + 0.82, 0.1, 12.25, MARBLE))
    ms.append(B(0.1, 3.2, y_a + 0.82, y_a + 1.5, 0.1, 12.25, ELEUSIS))
    ms.append(B(-0.2, 3.2, y_a + 1.5, y_a + 1.92, -0.2, 12.55, MARBLE))
    # main block, roofless: walls on a ring, frieze band and cornice around the top
    out = [(2.9, 0.3), (23.3, 0.3), (23.3, 12.05), (2.9, 12.05)]
    inn = offset(out, -0.65)
    from meshkit import extrude
    ms.append(extrude(out, Y_W - 2.0, y_a + 0.82, MARBLE, holes=[inn]))
    ms.append(extrude(offset(out, 0.02), y_a + 0.82, y_a + 1.5, ELEUSIS, holes=[offset(out, -0.6)]))
    ms.append(extrude(offset(out, 0.3), y_a + 1.5, y_a + 1.92, MARBLE, holes=[offset(out, -0.6)]))
    # east door and windows, north door (inside the north porch)
    ms.append(B(2.84, 2.88, yb, yb + 4.6, 5.25, 7.1, DARK))
    for w in (2.2, 10.15):
        ms.append(B(2.84, 2.88, yb + 2.4, yb + 4.4, w - 0.6, w + 0.6, DARK))
    ms.append(B(18.6, 21.0, Y_W + 0.3, Y_W + 5.2, 12.07, 12.11, DARK))
    # west front: engaged half-columns over a high base, windows between them
    for w in (2.4, 4.8, 7.2, 9.6):
        c = ionic(0.0, 0.0, 0.0, y_a - (Y_W + 4.8), 0.33, 0.28, MARBLE, face="x")
        keep = c.v[:, 0] > -0.05
        c.f = c.f[np.all(keep[c.f], axis=1)]
        ms.append(c.T(23.3, Y_W + 4.8, w))
    for w in (3.6, 6.0, 8.4):
        ms.append(B(23.32, 23.36, Y_W + 6.2, Y_W + 8.4, w - 0.55, w + 0.55, DARK))
    ms.append(B(23.3, 23.7, Y_W + 4.6, Y_W + 4.8, 0.3, 12.05, MARBLE))
    # north porch on the lower ground
    ms.append(B(14.45, 27.4, Y_W - 2.0, Y_W + 0.3, 12.05, 20.05, MARBLE))
    yn = Y_W + 0.3
    for u in (15.5, 19.1, 22.7, 26.3):
        ms.append(ionic(u, 19.25, yn, 7.63, 0.41, 0.35, MARBLE, face="z"))
    for u in (15.5, 26.3):
        ms.append(ionic(u, 15.75, yn, 7.63, 0.41, 0.35, MARBLE, face="z"))
    yt = yn + 7.63
    ms.append(B(14.85, 27.0, yt, yt + 0.9, 12.05, 19.75, MARBLE))
    ms.append(B(14.85, 27.0, yt + 0.9, yt + 1.6, 12.05, 19.75, ELEUSIS))
    ms.append(B(14.55, 27.3, yt + 1.6, yt + 2.05, 12.05, 20.05, MARBLE))
    ms.append(B(14.75, 27.1, yt + 2.05, yt + 2.3, 12.05, 19.85, MARBLE))
    # Porch of the Caryatids on the south side
    ms.append(B(16.9, 22.7, Y_E - 1.0, Y_E + 1.62, -3.7, 0.3, MARBLE))
    ms.append(B(16.75, 22.85, Y_E + 1.62, Y_E + 1.77, -3.85, 0.3, MARBLE))
    yc = Y_E + 1.77
    for k in range(4):
        ms.append(caryatid(17.45 + 1.6 * k, -3.15, yc))
    for u in (17.45, 22.25):
        ms.append(caryatid(u, -1.55, yc))
    ms.append(B(16.85, 22.75, yc + 2.31, yc + 2.9, -3.65, 0.3, MARBLE))
    for k in range(24):                                         # dentils
        u = 16.95 + k * 0.24
        ms.append(B(u, u + 0.13, yc + 2.9, yc + 3.08, -3.78, -3.6, MARBLE))
    ms.append(B(16.6, 23.0, yc + 3.08, yc + 3.4, -3.95, 0.3, MARBLE))
    ms.append(B(16.8, 22.8, yc + 3.4, yc + 3.6, -3.75, 0.3, MARBLE))
    m = merge(ms)
    marble_paint(m, 7, 0.32)
    # keep the frieze dark
    return framed(m, E_O, E_U)


def olive_tree(x, z, y):
    ms = [cyl(0.35, 0.25, 2.2, BARK, 7).T(x, y, z)]
    for dx, dz, dy, r in ((0, 0, 3.4, 2.4), (1.3, 0.6, 3.0, 1.7), (-1.1, -0.8, 3.1, 1.8), (0.2, -1.2, 4.2, 1.5)):
        ms.append(sphere(r, OLIVE, 9, 5).S(1, 0.7, 1).T(x + dx, y + dy, z + dz))
    m = merge(ms)
    return m


# ── The Propylaia, the stairs, the Beulé Gate, Agrippa, Athena Nike ─────────────────────────────────
PROP_X = [PX - 9.9, PX - 6.35, PX - 2.85, PX + 2.85, PX + 6.35, PX + 9.9]


def propylaia():
    ms = []
    yw, ye = Y_PW + 0.3, Y_PE + 0.3
    # west stylobate (four steps, not across the central passage) and floors
    for k, (dz, dy) in enumerate(((1.6, -0.9), (1.15, -0.6), (0.7, -0.3), (0.0, 0.0))):
        for xa, xb in ((40.6 + 0.3 * (3 - k), PX - 2.0), (PX + 2.0, 62.7 - 0.3 * (3 - k))):
            ms.append(B(xa, xb, Y_PW - 3.0, yw + dy, -131.9 - dz, -129.4, MARBLE))
    ms.append(B(41.0, 62.3, Y_PW - 3.0, Y_PW + 0.05, -129.4, -114.2, POROS))
    ms.append(B(41.0, 62.3, Y_PW - 3.0, ye, -112.8, -106.4, MARBLE))
    for k in range(5):                                          # steps up to the gate wall
        y1 = Y_PW + (k + 1) * (Y_PE + 0.3 - Y_PW) / 5
        for xa, xb in ((41.9, PX - 2.1), (PX + 2.1, 61.4)):
            ms.append(B(xa, xb, Y_PW - 1.0, y1, -116.0 + k * 0.36, -114.2, MARBLE))
    # west portico: 6 Doric columns, architrave and frieze, the cornice over the outer bays
    for x in PROP_X:
        ms.append(doric(x, -130.7, yw, 8.81, 0.78, 0.62, MARBLE, abacus=0.84))
    ya = yw + 8.81
    ms.append(B(41.0, 62.3, ya, ya + 1.2, -131.45, -129.95, MARBLE))
    ms.append(B(41.0, 62.3, ya + 1.2, ya + 2.3, -131.4, -129.95, MARBLE))
    for xa, xb in ((40.8, 47.2), (56.1, 62.5)):
        ms.append(B(xa, xb, ya + 2.3, ya + 2.8, -131.95, -129.7, MARBLE))
    for x in PROP_X + [(a + b) / 2 for a, b in zip(PROP_X, PROP_X[1:])] + [41.4, 61.9]:
        ms.append(B(x - 0.33, x + 0.33, ya + 1.22, ya + 2.28, -131.48, -131.38, (0.80, 0.76, 0.67)))
    # side walls of the west hall and of the east portico, ending in antae
    for xa, xb in ((41.0, 41.9), (61.4, 62.3)):
        ms.append(B(xa, xb, Y_PW - 1.0, ya + 1.2, -131.0, -114.2, MARBLE))
        ms.append(B(xa, xb, Y_PE - 1.0, Y_PE + 0.3 + 8.53 + 1.2, -112.8, -107.8, MARBLE))
    # Ionic columns of the central passage, ceiling beams over the west hall
    for z in (-127.0, -123.2, -119.4):
        for x in (PX - 2.65, PX + 2.65):
            ms.append(ionic(x, z, Y_PW, ya + 1.2 - Y_PW, 0.52, 0.45, MARBLE, face="z"))
    for z in (-129.0, -125.1, -121.3, -117.5):
        ms.append(B(41.9, 61.4, ya + 1.2, ya + 1.85, z - 0.45, z + 0.45, MARBLE))
    # gate wall with five doors
    z0, z1 = -114.2, -112.8
    ytop = ya + 2.4
    doors = [(42.9, 44.4, ye, ye + 3.6), (45.7, 48.6, ye, ye + 5.8), (PX - 2.1, PX + 2.1, Y_PW, Y_PW + 8.4),
             (54.7, 57.6, ye, ye + 5.8), (58.9, 60.4, ye, ye + 3.6)]
    xs = [41.9] + [v for d in doors for v in d[:2]] + [61.4]
    for i in range(0, len(xs), 2):
        ms.append(B(xs[i], xs[i + 1], Y_PW - 1.0, ytop, z0, z1, MARBLE))
    for xa, xb, y0, y1 in doors:
        ms.append(B(xa, xb, y1, ytop, z0, z1, MARBLE))
        if y0 > Y_PW:
            ms.append(B(xa, xb, Y_PW - 1.0, y0, z0, z1, MARBLE))
    # east portico: 6 Doric columns, architrave and frieze
    for x in PROP_X:
        ms.append(doric(x, -107.25, ye, 8.53, 0.78, 0.62, MARBLE, abacus=0.84))
    yb = ye + 8.53
    ms.append(B(41.0, 62.3, yb, yb + 1.2, -108.0, -106.5, MARBLE))
    ms.append(B(41.0, 62.3, yb + 1.2, yb + 2.3, -107.95, -106.55, MARBLE))
    for x in PROP_X + [(a + b) / 2 for a, b in zip(PROP_X, PROP_X[1:])] + [41.4, 61.9]:
        ms.append(B(x - 0.33, x + 0.33, yb + 1.22, yb + 2.28, -106.58, -106.48, (0.80, 0.76, 0.67)))
    # Pinakotheke (north wing): room, porch with three columns in antis
    yp = Y_PW + 0.6
    ms.append(B(64.9, 79.7, Y_PW - 3.0, yp, -138.0, -125.7, MARBLE))
    room = [(69.9, -137.2), (78.9, -137.2), (78.9, -126.4), (69.9, -126.4)]
    from meshkit import extrude
    ms.append(extrude(offset(room, 0.75), yp, yp + 7.0, MARBLE, holes=[room]))
    ms.append(extrude(offset(room, 0.9), yp + 7.0, yp + 7.6, MARBLE, holes=[room]))
    for zc in (-134.0, -129.6):
        ms.append(B(69.13, 69.17, yp + 2.2, yp + 4.0, zc - 0.6, zc + 0.6, DARK))
    ms.append(B(69.13, 69.17, yp, yp + 4.4, -132.4, -131.2, DARK))
    for z in (-134.5, -131.8, -129.1):
        ms.append(doric(65.6, z, yp, 5.85, 0.52, 0.42, MARBLE, abacus=0.56))
    for za, zb in ((-137.95, -137.2), (-126.4, -125.65)):
        ms.append(B(65.0, 69.9, yp, yp + 5.85, za, zb, MARBLE))
    ms.append(B(65.0, 69.9, yp + 5.85, yp + 7.0, -137.95, -125.65, MARBLE))
    ms.append(B(64.8, 69.9, yp + 7.0, yp + 7.6, -138.1, -125.5, MARBLE))
    # south wing: wall on the south and the east, three columns towards the passage
    ms.append(B(33.0, 39.6, Y_PW - 3.0, yp, -136.0, -126.1, MARBLE))
    ms.append(B(33.2, 33.9, yp, yp + 7.0, -135.2, -126.8, MARBLE))
    ms.append(B(33.2, 39.0, yp, yp + 7.0, -127.5, -126.8, MARBLE))
    for z in (-134.4, -131.8, -129.2):
        ms.append(doric(38.4, z, yp, 5.85, 0.52, 0.42, MARBLE, abacus=0.56))
    ms.append(B(37.8, 39.0, yp + 5.85, yp + 7.0, -135.2, -126.8, MARBLE))
    m = merge(ms)
    return marble_paint(m, 11, 0.4)


def stairs():
    """The Roman marble staircase from the Beulé Gate up to the Propylaia."""
    ms = []
    z = -160.0
    while z < -133.4:
        z1 = min(z + 1.5, -133.0)
        y = float(top(PX, z1)) + 0.25
        ms.append(B(44.6, 59.6, y - 2.0, y, z, z1 + 0.05, (0.84, 0.81, 0.74)))
        z = z1
    return merge(ms)


def beule_gate():
    ms = []
    # retaining walls of the forecourt: the Pinakotheke terrace (north) and the Nike bastion (south)
    for xa, xb, zb, xr in ((66.4, 67.3, -139.0, 68.5), (42.1, 43.0, -137.0, 41.5)):
        for z in np.arange(-168.0, zb, 3.0):
            z1 = min(z + 3.0, zb)
            ms.append(B(xa, xb, Y_B - 6.0, float(top(xr, z1)) + 0.9, z, z1, WALL))
    y0 = Y_B
    ms.append(B(41.7, 49.0, y0 - 4.0, y0 + 9.0, -174.6, -168.3, WALL))
    ms.append(B(56.1, 65.1, y0 - 4.0, y0 + 9.0, -174.3, -168.1, WALL))
    for xa, xb in ((41.5, 49.2), (55.9, 65.3)):
        ms.append(B(xa, xb, y0 + 9.0, y0 + 9.5, -174.8, -168.0, MARBLE))
    ms.append(B(49.0, 51.3, y0 - 4.0, y0 + 7.5, -169.1, -168.3, MARBLE))
    ms.append(B(53.9, 56.1, y0 - 4.0, y0 + 7.5, -169.1, -168.3, MARBLE))
    ms.append(B(51.3, 53.9, y0 + 4.6, y0 + 7.5, -169.1, -168.3, MARBLE))
    ms.append(B(48.8, 56.3, y0 + 7.5, y0 + 8.0, -169.3, -168.1, MARBLE))
    m = merge(ms)
    p = m.v
    m.c = np.clip(m.c * (0.9 + 0.16 * fbm(p, 0.3, 61, 2))[:, None], 0, 1)
    return m


def agrippa():
    P = DATA["w426887028"]
    A = np.array(P)
    cx, cz = A.mean(0)
    y0 = float(top(cx, cz))
    ms = [B(cx - 3.0, cx + 3.0, y0 - 3.0, y0 + 0.5, cz - 2.9, cz + 2.9, GREY_M),
          B(cx - 2.6, cx + 2.6, y0 + 0.5, y0 + 1.3, cz - 2.5, cz + 2.5, GREY_M),
          B(cx - 1.95, cx + 1.95, y0 + 1.3, y0 + 9.4, cz - 1.55, cz + 1.55, GREY_M),
          B(cx - 2.15, cx + 2.15, y0 + 9.4, y0 + 10.1, cz - 1.75, cz + 1.75, GREY_M)]
    m = merge(ms)
    p = m.v
    band_ = (np.floor((p[:, 1] - y0) / 0.55) % 2) * 0.06            # courses of grey and bluish marble
    m.c = np.clip(m.c * (0.95 + band_)[:, None], 0, 1)
    return m


N_O = (36.765, -147.485)
N_U = (-2.58, 8.09)


def nike():
    ms = []
    y0 = Y_PW
    for k, (du, dw) in enumerate(((0.75, 0.75), (0.5, 0.5), (0.25, 0.25))):
        ms.append(B(-4.245 - du, 4.245 + du, y0 - 1.0 if k == 0 else y0 + 0.25 * k, y0 + 0.25 * (k + 1), -2.82 - dw, 2.82 + dw, MARBLE))
    ys = y0 + 0.75
    ms.append(B(-4.245, 4.245, ys - 0.25, ys, -2.82, 2.82, MARBLE))
    for s in (-1, 1):
        for w in (-2.2, -0.73, 0.73, 2.2):
            ms.append(ionic(s * 3.75, w, ys, 4.07, 0.27, 0.23, MARBLE, face="x"))
    # cella (inner ring of OSM, in local axes) with its antae
    U, W = frame_check(N_U)
    cel = [((p[0] - N_O[0]) * U[0] + (p[1] - N_O[1]) * U[1], (p[0] - N_O[0]) * W[0] + (p[1] - N_O[1]) * W[1])
           for p in DATA["nike_cella"]]
    from meshkit import extrude
    ms.append(extrude(offset(cel, 0.3), ys, ys + 4.07, MARBLE, holes=[offset(cel, -0.25)]))
    ya = ys + 4.07
    ms.append(B(-4.0, 4.0, ya, ya + 0.45, -2.55, 2.55, MARBLE))
    ms.append(B(-4.0, 4.0, ya + 0.45, ya + 0.95, -2.55, 2.55, MARBLE))
    ms.append(B(-4.2, 4.2, ya + 0.95, ya + 1.2, -2.75, 2.75, MARBLE))
    m = merge(ms)
    marble_paint(m, 13, 0.3)
    return framed(m, N_O, N_U)


# ── Low remains, the old museum, the flag ───────────────────────────────
def low(P, h, c, below=1.5):
    from meshkit import extrude
    A = np.array(P)
    y = float(np.min(top(A[:, 0], A[:, 1])))
    return extrude(P, y - below, y + h, c)


def remains():
    ms = []
    ms.append(low(DATA["old_temple"], 0.35, POROS))
    for r in DATA["old_temple_in"]:
        ms.append(low(r, 0.7, (0.66, 0.61, 0.52)))
    ms.append(low(DATA["w144928413"], 0.55, MARBLE))                 # Rome and Augustus
    ms.append(low(DATA["w144928418"], 0.6, POROS))                   # Arrephorion
    for k in ("w910010357", "w910010294"):
        P = DATA[k]
        from meshkit import extrude
        A = np.array(P)
        y = float(np.min(top(A[:, 0], A[:, 1])))
        ms.append(extrude(P, y - 1.5, y + 1.0, POROS, holes=[offset(P, -0.8)]))
    px, pz = DATA["Athena Promachos statue place"]
    y = float(top(px, pz))
    ms.append(B(px - 2.8, px + 2.8, y - 1.0, y + 0.8, pz - 2.8, pz + 2.8, POROS))
    # the Old Acropolis Museum, a plain low building sunk into the rock
    m = low(DATA["w910010375"], 2.6, MUSEUM)
    ms.append(m)
    m = merge(ms)
    p = m.v
    m.c = np.clip(m.c * (0.9 + 0.15 * fbm(p, 0.5, 71, 2))[:, None], 0, 1)
    return m


def fallen_blocks():
    """The rows of numbered marble blocks and drums laid out around the Parthenon and the Erechtheion."""
    ms = []
    r = np.random.default_rng(1687)
    fields = [(-26.0, -19.5, -30.0, 30.0), (19.5, 25.0, -28.0, 28.0), (43.0, 50.0, 8.0, 22.0), (-22.0, -18.0, 40.0, 58.0)]
    for x0, x1, z0, z1 in fields:
        for x in np.arange(x0, x1, 1.6):
            for z in np.arange(z0, z1, 1.9):
                if r.random() < 0.35:
                    continue
                y = float(top(x, z))
                if r.random() < 0.3:
                    d = cyl(0.85, 0.85, 0.55 + 0.3 * r.random(), MARBLE, 10).T(x, y - 0.05, z)
                    ms.append(d)
                else:
                    w, h, dd = 0.8 + 0.8 * r.random(), 0.45 + 0.35 * r.random(), 0.6 + 0.6 * r.random()
                    ms.append(B(x - w / 2, x + w / 2, y - 0.05, y + h, z - dd / 2, z + dd / 2, tint(MARBLE, 0.06)))
    m = merge(ms)
    return marble_paint(m, 81, 0.45)


def flag():
    A = np.array(DATA["acropolis"])
    i = int(np.argmax(A[:, 1]))
    x, z = A[i] + np.array([0.0, -6.0])
    y = float(top(x, z)) + 1.2
    ms = [B(x - 0.6, x + 0.6, y - 1.0, y + 0.4, z - 0.6, z + 0.6, MARBLE), cyl(0.07, 0.05, 11.5, IRON, 8).T(x, y, z)]
    fw, fh = 4.2, 2.8
    yt = y + 11.3
    for k in range(9):                                           # nine stripes, blue at the top
        c = BLUE if k % 2 == 0 else WHITE
        y1 = yt - k * fh / 9
        ms.append(B(x - 0.03, x + 0.03, y1 - fh / 9, y1, z - 0.08 - fw, z - 0.08, c))
    cw = fh * 5 / 9
    ms.append(B(x - 0.04, x + 0.04, yt - cw, yt, z - 0.08 - cw, z - 0.08, BLUE))
    ms.append(B(x - 0.05, x + 0.05, yt - cw * 0.6, yt - cw * 0.4, z - 0.08 - cw, z - 0.08, WHITE))
    ms.append(B(x - 0.05, x + 0.05, yt - cw, yt, z - 0.08 - cw * 0.6, z - 0.08 - cw * 0.4, WHITE))
    return merge(ms)


# ── Assembly ────────────────────────────────────────────────────────────
def build():
    rock_top = plateau_paint(plateau(DATA["acropolis"], 3.0))
    ox, oz = DATA["Holy olive tree"]
    parts = [("rock", merge([rock_top, rock_face()])),
             ("walls", circuit_walls()),
             ("parthenon", parthenon()),
             ("erechtheion", merge([erechtheion(), olive_tree(ox, oz, float(top(ox, oz)))])),
             ("propylaia", merge([propylaia(), stairs(), beule_gate(), agrippa()])),
             ("athena_nike", nike()),
             ("remains", merge([remains(), fallen_blocks(), flag()]))]
    return parts


def footprint():
    P = simplify(offset(simplify(DATA["acropolis"], 1.0), 12.0), 2.5)
    return P


def main():
    parts = build()
    out_dir = ROOT / "models" / "Q10288"
    out_dir.mkdir(parents=True, exist_ok=True)
    named = []
    for name, m in parts:
        named.append((name, m))
    tris, size = write_glb(out_dir / "model.glb", named, roughness=0.85)
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        mj["footprint"] = [[round(x, 1), round(z, 1)] for x, z in footprint()]
        meta.write_bytes((json.dumps(mj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    for name, m in parts:
        print(f"  {name:12s} {m.tris:7d}")
    print(f"{tris} triangles, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
