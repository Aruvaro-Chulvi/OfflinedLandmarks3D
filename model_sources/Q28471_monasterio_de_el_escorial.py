"""
Real Monasterio de San Lorenzo de El Escorial (Q28471) — the monastery, its gardens and fountains, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The building: Juan Bautista de Toledo (design 1562, died 1567) and Juan de Herrera (1567–1584, died 1597), for
Philip II; the Galería de Convalecientes and the gardens of the same years; the Botica by Francisco de Mora
(died 1610). Public domain.

What is modelled:
- The monastery: the 207 × 162 m rectangle with its wings, the four corner towers with their slate spires, the
  two smaller towers over the crossings of the College and the Convent, the basilica (nave, transepts, chancel,
  the dome with its drum, lantern, ball and cross, and the two bell towers), the «handle of the gridiron» to the
  east (the palace of Philip II around the Patio de Mascarones), the main portal of the west front with San
  Lorenzo in his niche and the two side portals, the façade of the basilica on the Patio de Reyes with the six
  kings of Judah, the 16 courtyards (the Patio de los Evangelistas with Herrera's templete, its four ponds and
  its box parterres, and the small patios with their round fountains), the cloister arcades, ~2,000 windows and
  the dormers of the slate roofs.
- The gardens: the terrace of the Jardín de los Frailes along the south front and the Jardines del Rey (de los
  Reyes) along the east front, on their retaining walls and parapets with granite balls, the box parterres
  (as OpenStreetMap draws them), a fountain in the middle of every group of four parterres, the sunken stairs
  into the vaults under the terrace, the walled lower garden at the north-east corner with its round fountain,
  the Galería de Convalecientes (two storeys of arcades over the garden and the orchard) with the Botica, and
  the great pond (Estanque) of the orchard with its walk and walls. The parapet of the Lonja (the esplanade
  on the north and west) is included as its granite walls with balls.
- Not modelled: the orchard (Huerta) itself, the terrain, the Casas de Oficios and the Casa de Infantes (separate
  buildings of the Lonja), the interiors.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap relation 1264135 (wikidata Q28471) with its 16 courtyards, and the ~180 building:part
  pieces of relation 12030499 drawn by OSM mappers (wings, towers, basilica, palace), in the axes of heading
  282.15° (the main front faces west-north-west; the outline is squared to 102.15°/12.15° within ~0.2 m).
  The rectangle is 207.7 × 162.6 m (Padre Sigüenza, 1605: 735 × 580 Castilian feet = 205 × 162 m), the palace
  projects 48.6 m from the east front.
- Gardens, parterres, ponds, fountains, retaining walls, sunken stairs, Galería de Convalecientes (relation
  3126701), Botica (way 890760522), Estanque (way 26136021), walls of the Lonja: OpenStreetMap ways (ids below).
- Heights: OSM gives relative heights for the parts, but they are too low for the known ones (cross of the dome
  95 m above the church floor, es.wikipedia; corner towers 55 m and bell towers 72 m, the usual figures), so they
  were mapped to: wings 20 m to the cornice and 25.5 m to the ridge, cloister galleries 14 m, basilica arms
  32 m / 37.5 m, crossing 38 m, drum to 57.5 m, dome to 70.5 m, lantern, ball and cross to 96.5 m; corner towers
  34 m + spire to 55 m; bell towers 47 m + drum and cupola to 72 m; main portal 13.5 m + upper body and pediment
  to 29.6 m (OSM 29 m). Dome Ø 24 m outside (17 m inside, es.wikipedia), drum with 8 windows. ±10 %.
- Ground: EU-DEM 25 m (Copernicus, via OpenTopoData). y = 0 is the Lonja and the floor of the monastery
  (≈ 1037 m); the terrace of the gardens lies 2 m lower (y = −2), the walled lower garden at −7, the orchard and
  the pond about 9–10 m below the Lonja. Outer walls go down to y = −14, the retaining walls of the terrace and
  the pond to −16/−18, so that no gap shows where the terrain falls away.
Grey Guadarrama granite, slate roofs (Bernardos, Segovia), the dome, cupolas and lantern in stone, lead on the
basilica roofs and needles.

Axes (docs/MODEL_SPEC.md): +Y up, the main (west) front on the Lonja faces +Z (heading 282.15°), +X points to
192.15° (the south front over the Jardín de los Frailes). Origin on the ground at the centre of the 207 × 162 m
rectangle.

    python model_sources/Q28471_monasterio_de_el_escorial.py   -> models/Q28471/model.glb (+ footprint in meta.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from meshkit import Mesh, merge, box, cyl, lathe, tube, cap, walls, triangulate, ring_area, offset, fbm, write_glb  # noqa: E402

rng = np.random.default_rng(1563)
HEADING = 282.15

# ── Colours ──────────────────────────────────────────────────────────────────
GRANITE = (0.745, 0.725, 0.685)        # Guadarrama granite, ashlar
GRANITE_L = (0.79, 0.77, 0.73)         # cornices, frames, templete
GRANITE_D = (0.62, 0.60, 0.565)        # plinths, retaining walls, foundations
SLATE = (0.37, 0.405, 0.475)           # slate roofs and spires
LEAD = (0.47, 0.48, 0.52)              # lead roofs of the basilica, needles
LEAD_D = (0.40, 0.41, 0.44)            # flat terraces
GLASS = (0.12, 0.13, 0.15)
DARK = (0.08, 0.08, 0.09)              # open arcades, doors
DOOR = (0.30, 0.20, 0.12)
BOX = (0.20, 0.33, 0.15)               # box hedges
GRAVEL = (0.80, 0.75, 0.64)            # garden paths
PAVING = (0.70, 0.68, 0.635)           # Patio de Reyes, walks
GRASS = (0.42, 0.53, 0.30)
WATER = (0.22, 0.36, 0.42)
MARBLE = (0.90, 0.89, 0.86)            # statues
GOLD = (0.80, 0.62, 0.25)
IRON = (0.12, 0.12, 0.13)

G = -2.0          # terrace of the gardens
G_LOW = -7.0      # walled lower garden (north-east)
POND = -10.2      # water of the Estanque
FOUND = -14.0     # outer walls
BASE_IN = -1.0    # courtyard walls
WIN_ROWS = [(2.6, 1.15, 1.55), (7.1, 1.3, 2.1), (11.6, 1.3, 2.1), (16.1, 1.2, 1.7)]   # centre y, w, h
BAY = 4.6         # window spacing

# ── Data (OpenStreetMap © OpenStreetMap contributors, ODbL), metres in the model axes ───────────────────
DATA = json.loads(r"""{"parts":[{"id":885736160,"r":[[-59.35,-7.44],[-59.36,-27.55],[-59.36,-30.82],[-66.38,-30.82],[-66.38,-27.55],[-66.38,-23.06],[-66.37,-7.45]],"height":"9.45","r_shape":"gabled","r_height":"1","r_colour":"#7E7A9B","building:levels":"2"},{"id":885736161,"r":[[-39.84,-34.06],[-39.84,-27.53],[-59.36,-27.55],[-66.38,-27.55],[-70.94,-27.56],[-85.71,-27.57],[-85.72,-34.1]],"height":"9.45","r_shape":"gabled","r_height":"1","r_colour":"#7E7A9B","building:levels":"2"},{"id":885736162,"r":[[-13.23,-67.31],[-13.24,-85.61],[-13.24,-95.55],[-8.56,-95.55],[-5.97,-95.54],[5.81,-95.54],[8.51,-95.54],[13.18,-95.53],[13.19,-85.59],[13.2,-67.28]],"height":"25.5","r_shape":"gabled","r_height":"3.75","r_orientation":"along","r_colour":"#7E7A9B"},{"id":885736166,"r":[[13.41,-38.87],[13.2,-67.28],[-13.23,-67.31],[-13.2,-38.86],[-0.04,-38.85]],"height":"26","r_colour":"#E3CFC6","building:levels":"8"},{"id":885736167,"r":[[13.2,-67.28],[13.41,-38.87],[26.11,-38.86],[31.84,-38.82],[31.83,-67.27]],"height":"25.5","r_shape":"gabled","r_height":"3.75","r_orientation":"across","r_levels":"2","r_colour":"#7E7A9B","building:levels":"4"},{"id":885736168,"r":[[-13.23,-67.31],[-32.84,-67.33],[-32.83,-60.07],[-32.82,-38.87],[-27.04,-38.87],[-13.2,-38.86]],"height":"25.5","r_shape":"gabled","r_height":"3.75","r_orientation":"across","r_levels":"2","r_colour":"#7E7A9B","building:levels":"4"},{"id":885736169,"r":[[-0.04,-38.85],[13.41,-38.87],[13.24,-18.98],[13.24,-9.78],[13.25,4.57],[-0.01,4.56]],"height":"25.5","r_shape":"skillion","r_height":"3.75","r_direction":"192.15","r_levels":"2","r_colour":"#7E7A9B","building:levels":"4"},{"id":885736170,"r":[[99.2,38.31],[67.64,38.28],[67.65,45.02],[99.21,45.05]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"282.15","r_colour":"#68769A"},{"id":885736171,"r":[[60.88,45.02],[60.9,78.47],[54.18,78.47],[54.15,45.01]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"12.15","r_colour":"#68769A"},{"id":885736172,"r":[[60.87,31.54],[67.63,31.55],[67.62,9.03],[60.85,9.02]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"192.15","r_colour":"#68769A"},{"id":885736173,"r":[[54.14,38.27],[22.65,38.24],[22.65,31.51],[54.15,31.54]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"102.15","r_colour":"#68769A"},{"id":885736174,"r":[[-23.31,38.19],[-23.31,44.95],[-55.13,44.91],[-55.13,38.17]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"282.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":885736175,"r":[[-61.68,-0.33],[-61.66,31.43],[-55.14,31.44],[-55.15,4.51],[-55.16,-0.33]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"192.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":885736176,"r":[[-61.65,44.92],[-61.63,78.36],[-68.14,78.35],[-68.15,71.94],[-68.16,67.3],[-68.17,49.36],[-68.17,44.91]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"12.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":885736177,"r":[[-68.17,38.16],[-99.17,38.13],[-99.16,44.88],[-94.53,44.89],[-89.92,44.89],[-68.17,44.91]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"282.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":886413020,"r":[[-94.4,67.25],[-99.02,67.25],[-103.76,67.27],[-103.81,-0.37],[-103.86,-67.4],[-99.23,-67.39],[-94.61,-67.39],[-94.61,-60.13],[-94.57,-7.47],[-94.57,-5.14],[-94.56,4.48],[-94.53,44.89],[-94.53,49.34]],"height":"17.5","r_shape":"gabled","r_height":"4.9","r_orientation":"along","r_levels":"2","r_colour":"#68769A","building:levels":"4"},{"id":886413022,"r":[[89.89,-67.22],[94.51,-67.22],[99.12,-67.22],[99.17,-9.7],[99.17,-0.19],[94.56,-0.2],[89.93,-0.2],[89.93,-9.71]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"12.15","r_colour":"#68769A"},{"id":886413023,"r":[[94.56,-0.2],[99.17,-0.19],[99.23,67.44],[94.6,67.44]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"12.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":886413024,"r":[[-61.65,44.92],[-55.13,44.91],[-55.11,78.37],[-61.63,78.36]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"192.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":886413025,"r":[[-61.68,-0.33],[-68.2,-0.34],[-68.19,4.5],[-68.18,31.43],[-61.66,31.43]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"12.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":886413026,"r":[[60.88,45.02],[67.65,45.02],[67.67,78.47],[60.9,78.47]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"192.15","r_colour":"#68769A"},{"id":886413027,"r":[[60.85,9.02],[54.13,9.03],[54.15,31.54],[60.87,31.54]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"12.15","r_colour":"#68769A"},{"id":886413030,"r":[[-0.01,4.56],[-13.17,4.55],[-13.18,-9.8],[-13.19,-19.01],[-13.2,-38.86],[-0.04,-38.85]],"height":"25.5","r_shape":"skillion","r_height":"3.75","r_direction":"12.15","r_levels":"2","r_colour":"#7E7A9B","building:levels":"4"},{"id":886413031,"r":[[22.65,38.24],[54.14,38.27],[54.15,45.01],[22.66,44.98]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"282.15","r_colour":"#68769A"},{"id":886413032,"r":[[67.64,38.28],[99.2,38.31],[99.19,31.57],[67.63,31.55]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"102.15","r_colour":"#68769A"},{"id":886413033,"r":[[-99.17,38.13],[-68.17,38.16],[-68.18,31.43],[-94.54,31.4],[-99.16,31.4]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"102.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":886413034,"r":[[-55.13,38.17],[-55.14,31.44],[-23.31,31.47],[-23.31,38.19]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"102.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":886618136,"r":[[54.11,-9.74],[54.13,9.03],[67.62,9.03],[67.61,-0.22],[67.61,-9.73]],"height":"24","r_shape":"hipped","r_height":"5.5","r_orientation":"across","r_colour":"#68769A","building:levels":"7"},{"id":886618137,"r":[[49.79,4.6],[49.78,-9.75],[54.11,-9.74],[67.61,-9.73],[71.98,-9.73],[71.99,4.62]],"height":"24","r_shape":"hipped","r_height":"5.5","r_colour":"#68769A","building:levels":"7"},{"id":886638330,"r":[[17.89,-95.53],[17.87,-110.7],[17.86,-129.88],[-17.96,-129.91],[-17.94,-110.9],[-17.92,-95.56],[-13.24,-95.55],[-8.56,-95.55],[-8.58,-119.26],[8.5,-119.24],[8.51,-95.54],[13.18,-95.53]],"height":"6.9","name":"Palacio Privado","building:levels":"2"},{"id":886638339,"r":[[36.52,-81.17],[-36.28,-81.24],[-47.2,-81.25],[-90.0,-81.29],[-90.01,-76.66],[-36.27,-76.6],[36.51,-76.54],[89.89,-76.5],[89.88,-81.13]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"102.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":886638340,"r":[[36.51,-76.54],[-36.27,-76.6],[-90.01,-76.66],[-90.01,-72.01],[-90.0,-67.37],[-85.74,-67.37],[-39.87,-67.33],[-36.26,-67.33],[-32.84,-67.33],[-13.23,-67.31],[13.2,-67.28],[31.83,-67.27],[36.52,-67.27],[89.89,-67.22],[89.89,-71.91],[89.89,-76.5]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"282.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":887459948,"r":[[-17.96,-129.91],[-17.94,-110.9],[-17.92,-95.56],[-13.24,-95.55],[-13.27,-124.58]],"height":"11.1","min_height":"6.9","r_shape":"skillion","r_height":"4.2","r_direction":"12.15"},{"id":887459949,"r":[[17.86,-129.88],[-17.96,-129.91],[-13.27,-124.58],[13.16,-124.56]],"height":"11.1","min_height":"6.9","r_shape":"skillion","r_height":"4.2","r_direction":"102.15"},{"id":887459950,"r":[[17.86,-129.88],[13.16,-124.56],[13.18,-95.53],[17.89,-95.53],[17.87,-110.7]],"height":"11.1","min_height":"6.9","r_shape":"skillion","r_height":"4.2","r_direction":"192.15"},{"id":887496172,"r":[[99.12,-67.22],[103.75,-67.21],[103.84,67.45],[99.23,67.44],[99.17,-9.7]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"192.15","r_levels":"2","r_colour":"#68769A","building:levels":"4"},{"id":887496173,"r":[[13.16,-124.56],[13.18,-95.53],[8.51,-95.54],[8.5,-119.24]],"height":"11.1","min_height":"6.9","r_shape":"skillion","r_height":"4.2","r_direction":"12.15"},{"id":887496174,"r":[[-13.27,-124.58],[13.16,-124.56],[8.5,-119.24],[-8.58,-119.26]],"height":"11.1","min_height":"6.9","r_shape":"skillion","r_height":"4.2","r_direction":"282.15"},{"id":887496175,"r":[[-13.24,-95.55],[-8.56,-95.55],[-8.58,-119.26],[-13.27,-124.58]],"height":"11.1","min_height":"6.9","r_shape":"skillion","r_height":"4.2","r_direction":"192.15"},{"id":887512414,"r":[[31.88,-0.26],[49.78,-0.23],[49.79,4.6],[31.87,4.59]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"282.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":887512415,"r":[[71.98,-0.21],[71.99,4.62],[99.18,4.64],[99.17,-0.19]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"282.15","r_colour":"#68769A"},{"id":887512416,"r":[[99.17,-0.19],[99.17,-9.7],[89.93,-9.71],[71.98,-9.73],[71.98,-0.21]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"102.15","r_colour":"#68769A"},{"id":887512417,"r":[[31.88,-0.26],[31.86,-9.76],[49.78,-9.75],[49.78,-0.23]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"102.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":887512424,"r":[[-13.2,-38.86],[-27.03,-19.02],[-13.19,-19.01]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"282.15","r_colour":"#7E7A9B","building:levels":"5"},{"id":887512425,"r":[[-27.03,-19.02],[-13.2,-38.86],[-27.04,-38.87]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"12.15","r_colour":"#7E7A9B","building:levels":"5"},{"id":887512426,"r":[[-36.26,-67.33],[-36.27,-76.6],[-36.28,-81.24],[-36.28,-85.63],[-13.23,-67.31],[-32.84,-67.33]],"height":"18.5","r_shape":"skillion","r_height":"5.9","r_direction":"12.15","r_colour":"#7E7A9B","building:levels":"5"},{"id":887512427,"r":[[-36.28,-85.63],[-21.8,-85.62],[-13.24,-85.61],[-13.23,-67.31]],"height":"18.5","r_shape":"skillion","r_height":"5.9","r_direction":"102.15","r_colour":"#7E7A9B","building:levels":"5"},{"id":887512428,"r":[[36.51,-85.56],[22.56,-85.58],[13.19,-85.59],[13.2,-67.28]],"height":"18.5","r_shape":"skillion","r_height":"5.9","r_direction":"102.15","r_colour":"#7E7A9B","building:levels":"5"},{"id":887512429,"r":[[36.52,-67.27],[36.51,-76.54],[36.52,-81.17],[36.51,-85.56],[13.2,-67.28],[31.83,-67.27]],"height":"18.5","r_shape":"skillion","r_height":"5.9","r_direction":"192.15","r_colour":"#7E7A9B","building:levels":"5"},{"id":887512431,"r":[[13.41,-38.87],[26.12,-19.01],[13.24,-18.98]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"282.15","r_colour":"#7E7A9B","building:levels":"5"},{"id":887821663,"r":[[-32.82,-38.87],[-27.04,-38.87],[-27.03,-19.02],[-27.02,-9.81],[-32.79,-9.82]],"height":"12.6","r_colour":"#7E7A9B","building:levels":"4"},{"id":887821664,"r":[[26.11,-38.86],[31.84,-38.82],[31.86,-9.76],[25.94,-9.77],[26.12,-19.01]],"height":"12.6","r_colour":"#7E7A9B","building:levels":"4"},{"id":887821668,"r":[[27.35,44.99],[31.91,44.99],[49.82,45.01],[54.15,45.01],[54.16,49.47],[49.83,49.47],[31.91,49.44],[27.35,49.45]],"height":"12.6","r_shape":"skillion","r_direction":"282.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821669,"r":[[27.37,72.02],[31.92,72.02],[49.84,72.04],[54.17,72.05],[54.17,67.41],[49.84,67.41],[31.92,67.39],[27.37,67.39]],"height":"12.6","r_shape":"skillion","r_direction":"102.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821670,"r":[[27.37,72.02],[27.37,67.39],[27.35,49.45],[27.35,44.99],[31.91,44.99],[31.91,49.44],[31.92,67.39],[31.92,72.02]],"height":"12.6","r_shape":"skillion","r_direction":"192.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821671,"r":[[49.82,45.01],[49.83,49.47],[49.84,67.41],[49.84,72.04],[54.17,72.05],[54.17,67.41],[54.16,49.47],[54.15,45.01]],"height":"12.6","r_shape":"skillion","r_direction":"12.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821672,"r":[[49.79,4.6],[49.8,9.01],[49.8,27.14],[49.81,31.54],[54.15,31.54],[54.14,27.14],[54.13,9.03],[54.12,4.6]],"height":"12.6","r_shape":"skillion","r_direction":"12.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821673,"r":[[27.32,4.58],[31.87,4.59],[49.79,4.6],[54.12,4.6],[54.13,9.03],[49.8,9.01],[31.88,9.0],[27.32,9.0]],"height":"12.6","r_shape":"skillion","r_direction":"282.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821674,"r":[[27.34,31.52],[27.33,27.12],[27.32,9.0],[27.32,4.58],[31.87,4.59],[31.88,9.0],[31.9,27.12],[31.9,31.51]],"height":"12.6","r_shape":"skillion","r_direction":"192.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821675,"r":[[27.34,31.52],[31.9,31.51],[49.81,31.54],[54.15,31.54],[54.14,27.14],[49.8,27.14],[31.9,27.12],[27.33,27.12]],"height":"12.6","r_shape":"skillion","r_direction":"102.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821677,"r":[[-55.1,71.95],[-50.68,71.95],[-32.73,71.97],[-28.17,71.97],[-28.18,67.33],[-32.74,67.33],[-50.68,67.31],[-55.11,67.31]],"height":"12.6","r_shape":"skillion","r_direction":"102.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821678,"r":[[-55.1,71.95],[-55.11,67.31],[-55.13,49.37],[-55.13,44.91],[-50.7,44.92],[-50.7,49.37],[-50.68,67.31],[-50.68,71.95]],"height":"12.6","r_shape":"skillion","r_direction":"192.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821679,"r":[[-55.13,44.91],[-50.7,44.92],[-32.76,44.94],[-28.19,44.94],[-28.19,49.39],[-32.76,49.39],[-50.7,49.37],[-55.13,49.37]],"height":"12.6","r_shape":"skillion","r_direction":"282.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821680,"r":[[-32.76,44.94],[-32.76,49.39],[-32.74,67.33],[-32.73,71.97],[-28.17,71.97],[-28.18,67.33],[-28.19,49.39],[-28.19,44.94]],"height":"12.6","r_shape":"skillion","r_direction":"12.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821681,"r":[[-55.15,4.51],[-50.73,4.52],[-32.78,4.53],[-28.21,4.53],[-28.22,8.95],[-32.78,8.95],[-50.72,8.93],[-55.15,8.93]],"height":"12.6","r_shape":"skillion","r_direction":"282.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821682,"r":[[-55.14,31.44],[-55.14,27.04],[-55.15,8.93],[-55.15,4.51],[-50.73,4.52],[-50.72,8.93],[-50.71,27.05],[-50.71,31.44]],"height":"12.6","r_shape":"skillion","r_direction":"192.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821683,"r":[[-55.14,31.44],[-50.71,31.44],[-32.77,31.46],[-28.2,31.46],[-28.2,27.07],[-32.77,27.07],[-50.71,27.05],[-55.14,27.04]],"height":"12.6","r_shape":"skillion","r_direction":"102.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821684,"r":[[-32.78,4.53],[-32.78,8.95],[-32.77,27.07],[-32.77,31.46],[-28.2,31.46],[-28.2,27.07],[-28.22,8.95],[-28.21,4.53]],"height":"12.6","r_shape":"skillion","r_direction":"12.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":887821685,"r":[[-23.29,67.34],[-28.18,67.33],[-28.19,44.94],[-28.2,31.46],[-28.21,4.53],[-23.34,4.54]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"12.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":887821686,"r":[[22.68,67.38],[18.01,67.37],[17.96,4.58],[22.63,4.58]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"12.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":887821687,"r":[[67.67,78.47],[67.67,81.34],[54.18,81.32],[54.18,78.47]],"height":"20","r_shape":"gabled","r_height":"2.5","r_orientation":"across","r_colour":"#E3CFC6","building:levels":"6"},{"id":887821688,"r":[[-55.11,78.37],[-55.11,81.23],[-68.14,81.22],[-68.14,78.35],[-61.63,78.36]],"height":"18.5","r_shape":"gabled","r_height":"2.5","r_orientation":"across","r_colour":"#E3CFC6","building:levels":"6"},{"id":887821689,"r":[[31.92,72.02],[-32.73,71.97],[-32.73,76.61],[-32.74,81.24],[-17.8,81.14],[18.01,81.39],[31.93,81.31],[31.93,76.66]],"height":"26.4","r_shape":"hipped","r_height":"7.9","r_colour":"#68769A","building:levels":"6"},{"id":887821690,"r":[[-28.18,67.33],[-23.29,67.34],[-17.82,67.35],[0.04,67.36],[18.01,67.37],[22.68,67.38],[27.37,67.39],[27.37,76.66],[-28.17,76.61]],"height":"26.4","r_shape":"skillion","r_height":"5.5","r_direction":"102.15","r_colour":"#68769A","building:levels":"5"},{"id":887821691,"r":[[-21.81,-95.56],[-17.92,-95.56],[-13.24,-95.55],[-13.24,-85.61]],"height":"12.6","min_height":"9.25","r_shape":"skillion","r_height":"2.35","r_direction":"102.15","r_colour":"#7E7A9B"},{"id":887821692,"r":[[-21.8,-85.62],[-21.81,-95.56],[-13.24,-85.61]],"height":"12.6","min_height":"10.25","r_shape":"skillion","r_height":"2.35","r_direction":"12.15","r_colour":"#7E7A9B"},{"id":887821693,"r":[[22.56,-85.58],[22.56,-95.52],[13.19,-85.59]],"height":"12.6","min_height":"9.25","r_shape":"skillion","r_height":"2.35","r_direction":"192.15","r_colour":"#7E7A9B"},{"id":887821694,"r":[[22.56,-95.52],[17.89,-95.53],[13.18,-95.53],[13.19,-85.59]],"height":"12.6","min_height":"9.25","r_shape":"skillion","r_height":"2.35","r_direction":"102.15","r_colour":"#7E7A9B"},{"id":887821695,"r":[[13.18,-95.53],[13.19,-85.59],[22.56,-85.58],[22.56,-95.52],[17.89,-95.53]],"height":"10.25","building:levels":"3"},{"id":887821696,"r":[[-21.8,-85.62],[-13.24,-85.61],[-13.24,-95.55],[-17.92,-95.56],[-21.81,-95.56]],"height":"10.25","building:levels":"3"},{"id":888162154,"r":[[5.79,-119.24],[5.79,-111.54],[5.8,-99.04],[5.81,-95.54],[8.51,-95.54],[8.51,-99.03],[8.51,-111.54],[8.5,-119.24]],"height":"6.9","r_shape":"skillion","r_height":"2.5","r_direction":"12.15","r_levels":"1","r_colour":"#7E7A9B","building:levels":"3"},{"id":888162155,"r":[[-8.56,-95.55],[-5.97,-95.54],[5.81,-95.54],[8.51,-95.54],[8.51,-99.03],[5.8,-99.04],[-5.98,-99.05],[-8.57,-99.06]],"height":"6.9","r_shape":"skillion","r_height":"2.5","r_direction":"102.15","r_levels":"1","r_colour":"#7E7A9B","building:levels":"3"},{"id":888162156,"r":[[-8.58,-119.26],[-5.99,-119.25],[5.79,-119.24],[8.5,-119.24],[8.51,-111.54],[5.79,-111.54],[-5.99,-111.55],[-8.58,-111.56]],"height":"6.9","r_shape":"skillion","r_height":"2.5","r_direction":"282.15","r_levels":"1","r_colour":"#7E7A9B","building:levels":"3"},{"id":888162157,"r":[[-8.56,-95.55],[-8.57,-99.06],[-8.58,-111.56],[-8.58,-119.26],[-5.99,-119.25],[-5.99,-111.55],[-5.98,-99.05],[-5.97,-95.54]],"height":"6.9","r_shape":"skillion","r_height":"2.5","r_direction":"192.15","r_levels":"1","r_colour":"#7E7A9B","building:levels":"3"},{"id":888162158,"r":[[-94.57,-5.14],[-94.57,-7.47],[-94.61,-60.13],[-94.61,-67.39],[-90.0,-67.37],[-85.74,-67.37],[-85.73,-60.12],[-85.72,-34.1],[-85.71,-27.57],[-85.71,-23.08],[-85.7,-7.47],[-85.69,-5.13]],"height":"12.6","r_shape":"skillion","r_height":"0.3","r_direction":"192.15","r_levels":"1","r_colour":"#7E7A9B","building:levels":"3"},{"id":888162159,"r":[[-59.36,-27.55],[-54.39,-27.54],[-39.84,-27.53],[-39.83,-23.04],[-54.39,-23.06],[-59.36,-23.06]],"height":"8.45","r_shape":"skillion","r_direction":"282.15","r_levels":"1","r_colour":"#68769A","building:levels":"2"},{"id":888162160,"r":[[-39.87,-67.33],[-39.86,-60.08],[-39.84,-34.06],[-39.83,-23.04],[-39.82,-7.43],[-39.83,-5.09],[-32.79,-5.08],[-32.8,-7.42],[-32.79,-9.82],[-32.82,-38.87],[-32.83,-60.07],[-32.84,-67.33],[-36.26,-67.33]],"height":"12.6","r_shape":"skillion","r_height":"0.3","r_direction":"12.15","r_levels":"1","r_colour":"#7E7A9B","building:levels":"3"},{"id":888162161,"r":[[-59.35,-7.44],[-59.36,-23.06],[-59.36,-27.55],[-54.39,-27.54],[-54.39,-23.06],[-54.37,-7.44]],"height":"8.45","r_shape":"skillion","r_direction":"192.15","r_levels":"1","r_colour":"#68769A","building:levels":"2"},{"id":888162162,"r":[[-70.94,-27.56],[-70.94,-23.07],[-70.93,-7.45],[-66.37,-7.45],[-66.38,-23.06],[-66.38,-27.55]],"height":"8.45","r_shape":"skillion","r_direction":"12.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162163,"r":[[-85.71,-27.57],[-70.94,-27.56],[-66.38,-27.55],[-66.38,-23.06],[-70.94,-23.07],[-85.71,-23.08]],"height":"8.45","r_shape":"skillion","r_direction":"282.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162164,"r":[[-94.57,-7.47],[-85.7,-7.47],[-70.93,-7.45],[-66.37,-7.45],[-59.35,-7.44],[-54.37,-7.44],[-39.82,-7.43],[-32.8,-7.42],[-32.79,-5.08],[-39.83,-5.09],[-85.69,-5.13],[-94.57,-5.14]],"height":"12.6","r_shape":"skillion","r_height":"0.3","r_direction":"102.15","r_levels":"1","r_colour":"#7E7A9B","building:levels":"3"},{"id":888162165,"r":[[-94.61,-67.39],[-90.0,-67.37],[-85.74,-67.37],[-39.87,-67.33],[-36.26,-67.33],[-32.84,-67.33],[-32.83,-60.07],[-39.86,-60.08],[-85.73,-60.12],[-94.61,-60.13]],"height":"12.6","r_shape":"skillion","r_height":"0.3","r_direction":"282.15","r_levels":"1","r_colour":"#7E7A9B","building:levels":"3"},{"id":888162166,"r":[[-13.17,4.55],[-13.18,-9.8],[-17.86,-9.81],[-17.85,4.55]],"height":"17.5","min_height":"6.5","r_shape":"gabled","r_height":"4.9","r_orientation":"across","r_levels":"2","r_colour":"#68769A","building:levels":"2"},{"id":888162167,"r":[[17.96,4.58],[17.95,-9.78],[13.24,-9.78],[13.25,4.57]],"height":"17.5","min_height":"6.5","r_shape":"gabled","r_height":"4.9","r_orientation":"across","r_levels":"2","r_colour":"#68769A","building:levels":"2"},{"id":888162168,"r":[[38.26,-15.65],[31.86,-9.76],[31.84,-38.82],[31.83,-67.27],[38.23,-61.12]],"height":"12.6","r_shape":"skillion","r_direction":"192.15","r_levels":"1","r_colour":"#7E7A9B","building:levels":"3"},{"id":888162169,"r":[[83.72,-61.08],[83.75,-15.61],[89.93,-9.71],[89.89,-67.22]],"height":"12.6","r_shape":"skillion","r_direction":"12.15","r_levels":"1","r_colour":"#7E7A9B","building:levels":"3"},{"id":888162170,"r":[[83.75,-15.61],[38.26,-15.65],[31.86,-9.76],[89.93,-9.71]],"height":"12.6","r_shape":"skillion","r_direction":"102.15","r_levels":"1","r_colour":"#7E7A9B","building:levels":"3"},{"id":888162171,"r":[[31.83,-67.27],[36.52,-67.27],[89.89,-67.22],[83.72,-61.08],[38.23,-61.12]],"height":"12.6","r_shape":"skillion","r_direction":"282.15","r_levels":"1","r_colour":"#7E7A9B","building:levels":"3"},{"id":888162173,"r":[[89.94,4.63],[89.95,9.06],[89.95,27.17],[89.96,31.57],[94.58,31.57],[94.58,27.18],[94.56,9.06],[94.55,4.64]],"height":"12.6","r_shape":"skillion","r_direction":"12.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162174,"r":[[67.62,4.62],[71.99,4.62],[89.94,4.63],[94.55,4.64],[94.56,9.06],[89.95,9.06],[71.99,9.03],[67.62,9.03]],"height":"12.6","r_shape":"skillion","r_direction":"282.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162175,"r":[[67.63,31.55],[72.01,31.55],[89.96,31.57],[94.58,31.57],[94.58,27.18],[89.95,27.17],[72.0,27.16],[67.63,27.15]],"height":"12.6","r_shape":"skillion","r_direction":"102.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162176,"r":[[67.63,31.55],[67.63,27.15],[67.62,9.03],[67.62,4.62],[71.99,4.62],[71.99,9.03],[72.0,27.16],[72.01,31.55]],"height":"12.6","r_shape":"skillion","r_direction":"192.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162178,"r":[[89.97,45.04],[89.98,49.5],[89.99,67.44],[89.99,72.08],[94.61,72.08],[94.6,67.44],[94.59,49.5],[94.59,45.05]],"height":"12.6","r_shape":"skillion","r_direction":"12.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162179,"r":[[67.65,45.02],[72.01,45.03],[89.97,45.04],[94.59,45.05],[94.59,49.5],[89.98,49.5],[72.02,49.49],[67.65,49.48]],"height":"12.6","r_shape":"skillion","r_direction":"282.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162180,"r":[[67.66,72.05],[67.67,67.42],[67.65,49.48],[67.65,45.02],[72.01,45.03],[72.02,49.49],[72.03,67.43],[72.04,72.06]],"height":"12.6","r_shape":"skillion","r_direction":"192.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162181,"r":[[67.66,72.05],[72.04,72.06],[89.99,72.08],[94.61,72.08],[94.6,67.44],[89.99,67.44],[72.03,67.43],[67.67,67.42]],"height":"12.6","r_shape":"skillion","r_direction":"102.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162183,"r":[[-94.53,44.89],[-89.92,44.89],[-72.54,44.9],[-68.17,44.91],[-68.17,49.36],[-72.53,49.35],[-89.91,49.34],[-94.53,49.34]],"height":"12.6","r_shape":"skillion","r_direction":"282.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162184,"r":[[-72.54,44.9],[-72.53,49.35],[-72.52,67.29],[-72.51,71.93],[-68.15,71.94],[-68.16,67.3],[-68.17,49.36],[-68.17,44.91]],"height":"12.6","r_shape":"skillion","r_direction":"12.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162185,"r":[[-94.4,71.89],[-89.77,71.89],[-72.51,71.93],[-68.15,71.94],[-68.16,67.3],[-72.52,67.29],[-89.9,67.29],[-94.4,67.25]],"height":"12.6","r_shape":"skillion","r_direction":"102.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162186,"r":[[-94.4,71.89],[-94.4,67.25],[-94.53,49.34],[-94.53,44.89],[-89.92,44.89],[-89.91,49.34],[-89.9,67.29],[-89.77,71.89]],"height":"12.6","r_shape":"skillion","r_direction":"192.15","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":888162187,"r":[[-89.77,76.53],[-89.77,71.89],[-72.51,71.93],[-68.15,71.94],[-32.73,71.97],[-32.73,76.61]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"102.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":888162188,"r":[[31.93,76.66],[31.92,72.02],[89.99,72.08],[89.99,76.71]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"102.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":888162189,"r":[[-23.34,4.54],[-17.85,4.55],[-17.82,67.35],[-23.29,67.34]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"192.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":888162190,"r":[[22.63,4.58],[27.32,4.58],[27.37,67.39],[22.68,67.38]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"192.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":888587185,"r":[[-32.73,76.61],[-32.74,81.24],[-55.11,81.23],[-68.14,81.22],[-89.89,81.2],[-89.77,76.53]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"282.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":888587186,"r":[[89.99,76.71],[90.0,81.35],[69.17,81.33],[67.67,81.34],[54.18,81.32],[31.93,81.31],[31.93,76.66]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"282.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":888587187,"r":[[-99.18,-0.36],[-99.19,-5.14],[-94.57,-5.14],[-39.83,-5.09],[-32.79,-5.08],[-32.79,-0.31],[-55.16,-0.33],[-68.2,-0.34]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"102.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":888587188,"r":[[-32.79,-0.31],[-32.78,4.53],[-50.73,4.52],[-55.15,4.51],[-68.19,4.5],[-94.56,4.48],[-99.19,4.47],[-99.18,-0.36],[-68.2,-0.34],[-55.16,-0.33]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"282.15","r_levels":"2","r_colour":"#68769A","building:levels":"5"},{"id":889516967,"r":[[-9.01,162.9],[9.16,162.99],[9.13,173.92],[9.13,174.75],[9.06,196.7],[-9.11,196.62]],"r_shape":"hipped","r_levels":"2","r_colour":"#68769A","building:levels":"3"},{"id":889516973,"r":[[9.16,162.99],[9.13,173.92],[9.13,174.75],[-9.07,174.67],[-128.86,174.12],[-147.48,174.03],[-147.44,162.27],[-128.83,162.36],[-9.01,162.9]],"r_shape":"mansard","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":890754639,"r":[[9.16,162.99],[15.01,163.09],[37.92,163.12],[42.92,163.15],[42.91,167.41],[42.89,174.07],[37.88,174.05],[9.13,173.92]],"r_shape":"gabled","r_levels":"1","r_colour":"#68769A","building:levels":"3"},{"id":1253264456,"r":[[26.12,-19.01],[13.41,-38.87],[26.11,-38.86]],"height":"17.5","r_shape":"skillion","r_height":"4.9","r_direction":"192.15","r_colour":"#7E7A9B","building:levels":"5"}],"hedges":[{"r":[[62.67,-26.36],[62.69,-18.66],[70.11,-18.66],[70.1,-26.36]],"h":0.8,"holes":[]},{"r":[[51.68,-26.36],[51.69,-18.65],[59.4,-18.66],[59.38,-26.36]],"h":0.8,"holes":[]},{"r":[[49.74,-26.36],[41.67,-26.36],[41.68,-18.65],[49.75,-18.65]],"h":0.8,"holes":[]},{"r":[[49.73,-35.58],[41.65,-35.57],[41.67,-28.14],[49.74,-28.15]],"h":0.8,"holes":[]},{"r":[[49.71,-46.9],[41.63,-46.9],[41.66,-38.35],[49.72,-38.35]],"h":0.8,"holes":[]},{"r":[[49.7,-49.16],[49.69,-58.24],[41.62,-58.24],[41.64,-49.15]],"h":0.8,"holes":[]},{"r":[[59.35,-49.15],[59.33,-58.24],[51.62,-58.24],[51.65,-49.15]],"h":0.8,"holes":[]},{"r":[[62.63,-49.15],[70.07,-49.16],[70.04,-58.24],[62.61,-58.24]],"h":0.8,"holes":[]},{"r":[[71.91,-49.16],[80.45,-49.15],[80.44,-58.24],[71.88,-58.24]],"h":0.8,"holes":[]},{"r":[[71.92,-38.35],[80.47,-38.36],[80.46,-46.9],[71.9,-46.91]],"h":0.8,"holes":[]},{"r":[[71.94,-28.15],[80.49,-28.15],[80.48,-35.57],[71.92,-35.57]],"h":0.8,"holes":[]},{"r":[[71.94,-26.36],[80.5,-26.36],[80.51,-18.66],[71.95,-18.66]],"h":0.8,"holes":[]},{"r":[[-80.0,-135.46],[-98.31,-135.37],[-98.37,-152.79],[-74.99,-152.83],[-75.2,-140.43],[-77.36,-139.39],[-79.04,-137.66]],"h":0.4,"holes":[]},{"r":[[-75.38,-127.22],[-77.31,-128.18],[-78.86,-129.71],[-79.85,-131.62],[-98.3,-131.58],[-98.24,-112.59],[-75.57,-112.64]],"h":0.4,"holes":[]},{"r":[[-66.82,-131.67],[-67.77,-129.76],[-69.26,-128.26],[-71.15,-127.27],[-71.28,-112.64],[-49.52,-112.68],[-49.58,-131.78]],"h":0.4,"holes":[]},{"r":[[-66.7,-135.54],[-67.62,-137.59],[-69.13,-139.24],[-71.1,-140.3],[-71.01,-152.84],[-49.65,-152.87],[-49.59,-135.63]],"h":0.4,"holes":[]},{"r":[[-70.4,-93.61],[-70.4,-95.11],[-68.27,-95.1],[-68.27,-93.61]],"h":0.4,"holes":[]},{"r":[[-65.9,-93.6],[-65.91,-91.57],[-68.27,-91.57],[-68.26,-85.66],[-53.93,-85.64],[-53.94,-93.59]],"h":0.4,"holes":[]},{"r":[[-68.28,-103.11],[-53.94,-103.1],[-53.93,-95.09],[-65.91,-95.1],[-65.92,-97.03],[-68.27,-97.01]],"h":0.4,"holes":[]},{"r":[[-84.26,-95.12],[-84.27,-103.13],[-70.41,-103.11],[-70.4,-96.8],[-72.63,-96.79],[-72.63,-95.11]],"h":0.4,"holes":[]},{"r":[[-84.26,-85.67],[-70.4,-85.66],[-70.4,-91.58],[-72.63,-91.55],[-72.63,-93.61],[-84.26,-93.62]],"h":0.4,"holes":[]},{"r":[[-21.96,-124.59],[-21.96,-129.92],[-29.95,-129.92],[-29.94,-124.6]],"h":0.4,"holes":[]},{"r":[[-29.93,-117.91],[-21.95,-117.9],[-21.95,-122.4],[-29.93,-122.42]],"h":0.4,"holes":[]},{"r":[[-39.92,-122.42],[-31.93,-122.42],[-31.93,-117.91],[-39.91,-117.91]],"h":0.4,"holes":[]},{"r":[[-39.91,-124.61],[-39.92,-129.93],[-31.93,-129.93],[-31.94,-124.6]],"h":0.4,"holes":[]},{"r":[[-27.47,-143.62],[-27.46,-146.12],[-29.96,-146.12],[-29.96,-151.64],[-21.98,-151.63],[-21.98,-143.61]],"h":0.4,"holes":[]},{"r":[[-29.94,-134.11],[-29.95,-139.12],[-27.45,-139.12],[-27.46,-141.62],[-21.97,-141.62],[-21.96,-134.1]],"h":0.4,"holes":[]},{"r":[[-31.94,-141.62],[-29.95,-141.62],[-29.95,-143.62],[-31.95,-143.62]],"h":0.4,"holes":[]},{"r":[[-39.93,-141.63],[-34.44,-141.62],[-34.44,-139.12],[-31.95,-139.12],[-31.94,-134.11],[-39.93,-134.12]],"h":0.4,"holes":[]},{"r":[[-39.93,-143.63],[-39.94,-151.64],[-31.96,-151.64],[-31.95,-146.13],[-34.45,-146.13],[-34.44,-143.62]],"h":0.4,"holes":[]},{"r":[[109.82,63.56],[109.83,51.37],[114.51,51.38],[114.55,54.2],[117.2,54.23],[117.18,63.58]],"h":0.7,"holes":[]},{"r":[[118.36,63.62],[118.38,54.28],[120.77,54.25],[120.82,51.39],[126.97,51.41],[126.91,63.59]],"h":0.7,"holes":[]},{"r":[[120.8,50.09],[126.97,50.11],[127.02,38.27],[118.44,38.27],[118.41,47.47],[120.81,47.49]],"h":0.7,"holes":[]},{"r":[[114.56,50.09],[109.85,50.09],[109.82,38.26],[117.25,38.28],[117.22,47.5],[114.58,47.45]],"h":0.7,"holes":[]},{"r":[[120.84,21.27],[127.01,21.28],[127.06,9.44],[118.47,9.45],[118.44,18.65],[120.85,18.67]],"h":0.7,"holes":[]},{"r":[[118.39,34.79],[118.42,25.45],[120.8,25.42],[120.86,22.56],[127.0,22.58],[126.94,34.76]],"h":0.7,"holes":[]},{"r":[[109.86,34.73],[109.87,22.54],[114.54,22.55],[114.58,25.38],[117.24,25.4],[117.21,34.76]],"h":0.7,"holes":[]},{"r":[[114.6,21.27],[109.88,21.26],[109.86,9.44],[117.29,9.46],[117.26,18.67],[114.63,18.62]],"h":0.7,"holes":[]},{"r":[[120.75,-24.78],[126.92,-24.76],[126.98,-34.99],[118.38,-35.03],[118.35,-27.38],[120.76,-27.38]],"h":0.7,"holes":[]},{"r":[[118.23,-13.37],[118.34,-20.59],[120.72,-20.62],[120.77,-23.48],[126.91,-23.45],[126.81,-13.36]],"h":0.7,"holes":[]},{"r":[[109.74,-13.37],[109.78,-23.5],[114.46,-23.48],[114.5,-20.66],[117.16,-20.64],[117.1,-13.36]],"h":0.7,"holes":[]},{"r":[[114.52,-24.78],[109.79,-24.78],[109.79,-35.07],[117.18,-35.03],[117.17,-27.37],[114.53,-27.42]],"h":0.7,"holes":[]},{"r":[[120.93,-50.97],[127.1,-50.96],[127.16,-61.2],[118.55,-61.23],[118.53,-53.59],[120.94,-53.57]],"h":0.7,"holes":[]},{"r":[[114.69,-50.98],[109.97,-50.98],[109.97,-61.27],[117.36,-61.23],[117.35,-53.57],[114.7,-53.62]],"h":0.7,"holes":[]},{"r":[[109.91,-39.57],[109.96,-49.7],[114.63,-49.69],[114.67,-46.86],[117.33,-46.84],[117.26,-39.57]],"h":0.7,"holes":[]},{"r":[[118.41,-39.56],[118.51,-46.79],[120.89,-46.82],[120.95,-49.68],[127.08,-49.66],[126.99,-39.57]],"h":0.7,"holes":[]},{"r":[[120.8,95.96],[126.96,95.98],[127.03,85.74],[118.43,85.71],[118.4,93.34],[120.81,93.36]],"h":0.7,"holes":[]},{"r":[[114.57,95.95],[109.84,95.96],[109.84,85.67],[117.23,85.71],[117.22,93.36],[114.58,93.32]],"h":0.7,"holes":[]},{"r":[[109.77,105.97],[109.83,97.23],[114.5,97.25],[114.54,100.07],[117.21,100.1],[117.12,105.99]],"h":0.7,"holes":[]},{"r":[[118.27,105.99],[118.39,100.15],[120.77,100.12],[120.82,97.25],[126.96,97.28],[126.87,106.01]],"h":0.7,"holes":[]},{"r":[[23.25,-141.62],[28.74,-141.61],[28.74,-139.11],[31.23,-139.11],[31.24,-134.1],[23.26,-134.11]],"h":0.4,"holes":[]},{"r":[[23.25,-143.62],[23.24,-151.63],[31.22,-151.63],[31.23,-146.12],[28.74,-146.12],[28.74,-143.61]],"h":0.4,"holes":[]},{"r":[[35.72,-143.61],[35.72,-146.11],[33.22,-146.11],[33.22,-151.63],[41.2,-151.62],[41.2,-143.6]],"h":0.4,"holes":[]},{"r":[[31.24,-141.61],[33.23,-141.61],[33.23,-143.61],[31.23,-143.61]],"h":0.4,"holes":[]},{"r":[[33.24,-134.1],[33.23,-139.11],[35.73,-139.11],[35.73,-141.61],[41.21,-141.61],[41.22,-134.09]],"h":0.4,"holes":[]},{"r":[[41.29,-124.37],[41.29,-129.7],[33.3,-129.7],[33.31,-124.38]],"h":0.4,"holes":[]},{"r":[[33.32,-117.69],[41.3,-117.68],[41.29,-122.18],[33.32,-122.2]],"h":0.4,"holes":[]},{"r":[[23.33,-122.2],[31.32,-122.2],[31.32,-117.69],[23.34,-117.69]],"h":0.4,"holes":[]},{"r":[[23.34,-124.39],[23.33,-129.71],[31.31,-129.71],[31.31,-124.38]],"h":0.4,"holes":[]},{"r":[[85.74,-103.62],[85.8,-94.86],[72.57,-94.97],[72.56,-96.89],[70.33,-96.89],[70.35,-103.54]],"h":0.7,"holes":[]},{"r":[[72.58,-93.79],[85.81,-93.78],[85.87,-84.79],[70.26,-84.71],[70.31,-91.79],[72.6,-91.8]],"h":0.7,"holes":[]},{"r":[[69.26,-91.78],[69.24,-84.7],[54.74,-84.62],[54.68,-93.87],[67.13,-93.81],[67.14,-91.78]],"h":0.7,"holes":[]},{"r":[[54.68,-95.12],[67.13,-95.01],[67.11,-96.86],[69.28,-96.89],[69.31,-103.53],[54.62,-103.46]],"h":0.7,"holes":[]},{"r":[[12.53,-133.75],[12.57,-142.3],[18.76,-142.31],[18.74,-133.71]],"h":0.7,"holes":[]},{"r":[[12.56,-143.31],[18.76,-143.27],[18.82,-151.76],[12.6,-151.79]],"h":0.7,"holes":[]},{"r":[[-17.12,-143.56],[-10.93,-143.52],[-10.86,-152.01],[-17.08,-152.05]],"h":0.7,"holes":[]},{"r":[[-17.16,-134.0],[-17.12,-142.55],[-10.92,-142.57],[-10.94,-133.96]],"h":0.7,"holes":[]},{"r":[[120.89,-94.48],[127.06,-94.47],[127.1,-103.79],[118.49,-103.77],[118.5,-97.09],[120.9,-97.08]],"h":0.8,"holes":[]},{"r":[[118.46,-84.78],[118.47,-90.3],[120.86,-90.33],[120.91,-93.18],[127.05,-93.16],[127.01,-84.66]],"h":0.8,"holes":[]},{"r":[[109.91,-84.85],[109.92,-93.2],[114.59,-93.19],[114.63,-90.36],[117.29,-90.35],[117.28,-84.77]],"h":0.8,"holes":[]},{"r":[[114.65,-94.48],[109.94,-94.49],[109.91,-103.74],[117.33,-103.76],[117.31,-97.08],[114.67,-97.12]],"h":0.8,"holes":[]},{"r":[[35.76,-96.07],[41.93,-96.06],[41.97,-105.38],[33.37,-105.36],[33.37,-98.68],[35.77,-98.67]],"h":0.8,"holes":[]},{"r":[[33.33,-86.37],[33.35,-91.89],[35.74,-91.92],[35.78,-94.77],[41.92,-94.75],[41.89,-86.25]],"h":0.8,"holes":[]},{"r":[[24.78,-86.44],[24.79,-94.79],[29.46,-94.78],[29.5,-91.95],[32.17,-91.93],[32.15,-86.36]],"h":0.8,"holes":[]},{"r":[[29.53,-96.07],[24.81,-96.08],[24.78,-105.33],[32.21,-105.35],[32.18,-98.67],[29.55,-98.71]],"h":0.8,"holes":[]},{"r":[[-28.79,-95.86],[-22.62,-95.85],[-22.58,-105.17],[-31.19,-105.15],[-31.18,-98.47],[-28.78,-98.46]],"h":0.8,"holes":[]},{"r":[[-31.22,-86.16],[-31.21,-91.68],[-28.82,-91.71],[-28.77,-94.57],[-22.63,-94.54],[-22.67,-86.04]],"h":0.8,"holes":[]},{"r":[[-39.77,-86.23],[-39.76,-94.58],[-35.09,-94.57],[-35.05,-91.74],[-32.39,-91.73],[-32.4,-86.16]],"h":0.8,"holes":[]},{"r":[[-35.03,-95.86],[-39.74,-95.87],[-39.77,-105.12],[-32.35,-105.14],[-32.37,-98.46],[-35.01,-98.51]],"h":0.8,"holes":[]}],"fountains":[[[52.7,-29.06],[58.27,-29.08],[58.26,-32.87],[56.43,-34.8],[52.67,-34.79]],[[63.54,-29.1],[69.48,-29.12],[69.44,-35.03],[65.57,-35.02],[63.52,-32.89]],[[65.55,-39.87],[69.42,-39.88],[69.4,-46.05],[63.34,-46.03],[63.35,-42.12]],[[52.64,-39.87],[56.4,-39.89],[58.53,-42.1],[58.51,-46.01],[52.62,-46.0]],[[-70.6,-133.85],[-70.9,-132.54],[-71.81,-131.55],[-73.09,-131.15],[-74.39,-131.45],[-75.38,-132.38],[-75.78,-133.66],[-75.48,-134.96],[-74.57,-135.95],[-73.29,-136.34],[-71.99,-136.04],[-71.01,-135.13]],[[-40.57,57.24],[-40.19,57.93],[-40.21,58.71],[-40.64,59.37],[-41.28,59.69],[-42.0,59.68],[-42.62,59.31],[-42.99,58.68],[-43.01,57.96],[-42.69,57.31],[-42.04,56.89],[-41.26,56.86]],[[41.89,16.92],[42.28,17.6],[42.25,18.4],[41.83,19.05],[41.18,19.38],[40.46,19.36],[39.85,18.99],[39.48,18.36],[39.45,17.64],[39.77,17.0],[40.43,16.57],[41.21,16.54]],[[41.87,57.33],[42.25,58.02],[42.23,58.81],[41.79,59.47],[41.16,59.79],[40.44,59.77],[39.81,59.4],[39.45,58.77],[39.42,58.05],[39.75,57.41],[40.39,56.98],[41.18,56.95]],[[82.14,17.02],[82.52,17.71],[82.5,18.5],[82.07,19.16],[81.43,19.48],[80.71,19.46],[80.09,19.09],[79.72,18.47],[79.69,17.75],[80.02,17.1],[80.67,16.67],[81.45,16.64]],[[82.15,57.41],[82.54,58.1],[82.52,58.89],[82.09,59.54],[81.46,59.87],[80.73,59.84],[80.12,59.47],[79.75,58.85],[79.72,58.13],[80.05,57.49],[80.69,57.06],[81.47,57.03]],[[-80.36,57.17],[-79.97,57.86],[-79.99,58.64],[-80.42,59.3],[-81.07,59.63],[-81.78,59.6],[-82.4,59.23],[-82.77,58.62],[-82.79,57.89],[-82.48,57.24],[-81.82,56.82],[-81.04,56.79]]],"templete":[[58.26,-32.87],[63.52,-32.89],[65.57,-35.02],[65.55,-39.87],[63.35,-42.12],[58.53,-42.1],[56.4,-39.89],[56.43,-34.8]],"retaining":[[1337321442,[[132.24,106.31],[132.18,75.17],[132.04,-1.81],[131.89,-72.78],[131.82,-107.9],[48.23,-108.39],[45.46,-108.39],[45.45,-110.67],[45.45,-154.83],[45.45,-156.72],[-46.23,-156.86],[-46.23,-111.0],[-46.23,-108.42],[-47.49,-108.43],[-103.96,-108.51]]],[1337321443,[[109.77,78.2],[125.49,78.31],[125.52,80.79],[109.53,80.73]]],[1337321444,[[109.54,69.04],[125.57,69.04],[125.51,71.59],[109.65,71.66]]],[1337321445,[[109.47,1.33],[125.5,1.32],[125.43,3.88],[109.58,3.96]]],[1337321446,[[109.85,-7.83],[125.88,-7.83],[125.81,-5.28],[109.96,-5.2]]],[1337321447,[[109.84,-69.38],[125.87,-69.38],[125.81,-66.83],[109.96,-66.75]]],[1337321448,[[109.85,-78.6],[125.88,-78.61],[125.81,-76.05],[109.97,-75.98]]],[1337321473,[[202.73,82.18],[139.99,81.72]]],[1337321474,[[176.51,75.95],[176.4,79.0],[166.18,78.9],[166.16,75.53]]],[1337321475,[[163.01,75.78],[162.96,78.88],[159.75,78.84]]],[1337321476,[[182.65,79.05],[179.53,79.02],[179.57,76.0]]],[1337321508,[[99.87,-86.67],[99.87,-101.65],[102.74,-101.66],[102.73,-86.44]]],[1337321509,[[90.57,-87.05],[90.58,-102.03],[93.43,-102.03],[93.44,-86.82]]],[1337321510,[[3.5,-135.35],[3.51,-150.34],[6.36,-150.33],[6.36,-135.12]]],[1337321511,[[-5.92,-135.28],[-5.92,-150.26],[-3.05,-150.26],[-3.06,-135.04]]],[1337321512,[[-92.66,-87.05],[-92.65,-102.04],[-89.8,-102.04],[-89.81,-86.82]]],[1337321513,[[-101.45,-87.11],[-101.44,-102.1],[-98.58,-102.1],[-98.58,-86.88]]]],"walls":[[231916311,[[131.82,-107.9],[132.44,-154.73],[131.55,-175.1],[133.23,-207.54]]],[232767370,[[132.44,-154.73],[45.45,-154.83]]],[1337321477,[[139.92,107.37],[139.95,129.67]]],[1337321480,[[202.7,75.39],[202.68,72.28],[252.5,71.77]]],[1337321481,[[202.71,105.04],[202.73,82.18],[202.72,78.79]]],[1337321482,[[201.46,130.09],[202.69,130.1],[202.71,107.75]]],[1337321483,[[181.75,129.96],[199.21,130.08]]],[1337321484,[[162.39,129.83],[179.12,129.94]]],[1337321485,[[143.09,129.7],[159.65,129.81]]],[1337321486,[[139.91,85.08],[139.93,104.71]]],[1337321497,[[45.45,-110.67],[17.87,-110.7]]],[1337321498,[[48.23,-108.39],[48.27,-106.12],[48.69,-81.25]]],[1337321499,[[-17.94,-110.9],[-43.0,-110.99],[-46.23,-111.0]]],[1337321500,[[-47.49,-108.43],[-47.2,-81.25]]],[1337321501,[[-103.88,-81.3],[-103.95,-103.95]]],[1337321502,[[-103.95,-106.46],[-103.96,-108.51],[-102.64,-156.79]]],[1337321503,[[-46.23,-156.86],[-72.85,-156.73],[-102.64,-156.79]]],[1337328790,[[-141.34,109.87],[-141.32,134.88],[-123.33,135.0]]],[1337328791,[[-119.39,135.02],[-63.76,135.41]]],[1337328792,[[-59.17,135.44],[-2.5,135.84]]],[1337328793,[[1.72,135.86],[57.55,136.25]]],[1337328794,[[101.76,136.36],[83.08,136.39],[79.15,136.4],[62.4,136.28]]],[1337328795,[[-141.34,106.05],[-141.15,44.19]]],[1337328796,[[-141.14,40.41],[-141.02,2.28]]],[1337328797,[[-141.01,-2.01],[-141.03,-41.19]]],[1337328798,[[-141.03,-44.86],[-141.05,-79.45],[-137.99,-79.51]]]],"steps":[[[59.56,136.26],[59.58,140.22]],[[123.89,79.53],[109.83,79.75]],[[110.35,-67.66],[123.3,-67.84]],[[100.99,-83.31],[101.06,-100.49]],[[109.57,3.03],[123.61,2.84]],[[-100.32,-84.36],[-100.24,-100.29]],[[109.75,-6.1],[123.54,-6.51]],[[-4.79,-132.25],[-4.81,-149.75]],[[5.01,-132.29],[4.55,-149.32]],[[110.03,70.61],[124.53,70.58]],[[110.38,-77.06],[123.51,-77.45]],[[91.46,-83.31],[91.67,-100.14]],[[-91.63,-84.32],[-91.48,-100.28]]],"pond":[[143.31,126.27],[143.63,85.7],[199.23,86.23],[198.91,126.8]],"galeria":{"outer":[[105.51,163.43],[105.55,148.85],[105.58,139.06],[105.59,136.98],[101.76,136.36],[101.75,129.54],[101.69,113.59],[101.67,106.17],[101.59,81.25],[103.85,81.36],[106.78,81.38],[109.16,81.38],[109.08,106.2],[124.82,106.28],[132.24,106.31],[132.23,113.73],[132.18,129.68],[132.16,137.1],[124.72,137.06],[111.81,137.01],[111.8,139.04],[111.77,148.9],[111.73,163.46]],"inner":[[[112.96,124.96],[113.0,113.64],[124.79,113.7],[124.76,125.02]]]},"botica":[[101.59,81.25],[103.85,81.36],[106.78,81.38],[109.16,81.38],[109.08,106.2],[109.07,113.63],[109.01,129.58],[109.0,137.0],[105.59,136.98],[101.76,136.36],[101.75,129.54],[101.69,113.59],[101.67,106.17]],"statues":[["Iosias",9.56,5.34],["Salomón",3.43,5.32],["David",-3.44,5.3],["Ezechias",-9.89,5.27],["Manasses",11.78,5.34],["Josaphat",-12.1,5.26],["San Lorenzo",0.06,83.35]],"courtyards":[["Patio Chico D4",[[72.03,67.43],[72.02,49.49],[89.98,49.5],[89.99,67.44]]],["Patio Chico D3",[[72.0,27.16],[71.99,9.03],[89.95,9.06],[89.95,27.17]]],["Patio Chico D2",[[31.88,9.0],[49.8,9.01],[49.8,27.14],[31.9,27.12]]],["Patio Chico D1",[[49.84,67.41],[31.92,67.39],[31.91,49.44],[49.83,49.47]]],["Patio de los Evangelistas",[[38.26,-15.65],[38.23,-61.12],[83.72,-61.08],[83.75,-15.61]]],["Patio de Reyes",[[-17.82,67.35],[-17.85,4.55],[-13.17,4.55],[-0.01,4.56],[13.25,4.57],[17.96,4.58],[18.01,67.37],[0.04,67.36]]],["Patio de Mascarones",[[-5.98,-99.05],[-5.99,-111.55],[5.79,-111.54],[5.8,-99.04],[-0.08,-99.05]]],["Patio Chico I4",[[-50.7,49.37],[-50.68,67.31],[-32.74,67.33],[-32.76,49.39]]],["Patio Chico I3",[[-50.72,8.93],[-50.71,27.05],[-32.77,27.07],[-32.78,8.95]]],["Patio Chico I2",[[-68.18,31.43],[-94.54,31.4],[-94.56,4.48],[-68.19,4.5]]],["Patio Chico I1",[[-89.91,49.34],[-72.53,49.35],[-72.52,67.29],[-89.9,67.29]]],["",[[-70.93,-7.45],[-85.7,-7.47],[-85.71,-23.08],[-70.94,-23.07]]],["",[[-54.37,-7.44],[-54.39,-23.06],[-39.83,-23.04],[-39.82,-7.43]]],["",[[-85.72,-34.1],[-85.73,-60.12],[-39.86,-60.08],[-39.84,-34.06]]]],"outline":[[-17.8,81.14],[-32.74,81.24],[-55.11,81.23],[-68.14,81.22],[-89.89,81.2],[-94.39,81.16],[-99.01,81.16],[-103.75,81.3],[-103.64,76.52],[-103.64,71.88],[-103.76,67.27],[-103.81,-0.37],[-103.86,-67.4],[-103.87,-72.03],[-103.87,-76.67],[-103.88,-81.3],[-99.25,-81.3],[-94.63,-81.29],[-90.0,-81.29],[-47.2,-81.25],[-36.28,-81.24],[-36.28,-85.63],[-21.8,-85.62],[-21.81,-95.56],[-17.92,-95.56],[-17.94,-110.9],[-17.96,-129.91],[17.86,-129.88],[17.87,-110.7],[17.89,-95.53],[22.56,-95.52],[22.56,-85.58],[36.51,-85.56],[36.52,-81.17],[89.88,-81.13],[94.5,-81.13],[99.12,-81.12],[103.73,-81.11],[103.74,-76.48],[103.74,-71.84],[103.75,-67.21],[103.84,67.45],[103.84,72.09],[103.85,76.73],[103.85,81.36],[101.59,81.25],[99.24,81.36],[94.61,81.35],[90.0,81.35],[69.17,81.33],[67.67,81.34],[54.18,81.32],[31.93,81.31],[18.01,81.39],[18.01,83.17],[10.58,83.21],[0.05,83.2],[-10.87,83.2],[-17.79,83.23]],"holes":[[[-85.72,-34.1],[-85.73,-60.12],[-39.86,-60.08],[-39.84,-34.06]],[[-54.37,-7.44],[-54.39,-23.06],[-39.83,-23.04],[-39.82,-7.43]],[[-70.93,-7.45],[-85.7,-7.47],[-85.71,-23.08],[-70.94,-23.07]],[[-89.91,49.34],[-72.53,49.35],[-72.52,67.29],[-89.9,67.29]],[[-68.18,31.43],[-94.54,31.4],[-94.56,4.48],[-68.19,4.5]],[[-50.72,8.93],[-50.71,27.05],[-32.77,27.07],[-32.78,8.95]],[[-50.7,49.37],[-50.68,67.31],[-32.74,67.33],[-32.76,49.39]],[[-5.98,-99.05],[-5.99,-111.55],[5.79,-111.54],[5.8,-99.04],[-0.08,-99.05]],[[-17.82,67.35],[-17.85,4.55],[-13.17,4.55],[-0.01,4.56],[13.25,4.57],[17.96,4.58],[18.01,67.37],[0.04,67.36]],[[38.26,-15.65],[38.23,-61.12],[83.72,-61.08],[83.75,-15.61]],[[49.84,67.41],[31.92,67.39],[31.91,49.44],[49.83,49.47]],[[31.88,9.0],[49.8,9.01],[49.8,27.14],[31.9,27.12]],[[72.0,27.16],[71.99,9.03],[89.95,9.06],[89.95,27.17]],[[72.03,67.43],[72.02,49.49],[89.98,49.5],[89.99,67.44]],[[-27.02,-9.81],[-27.03,-19.02],[-13.19,-19.01],[-13.18,-9.8],[-17.86,-9.81]],[[13.24,-9.78],[17.95,-9.78],[25.94,-9.77],[26.12,-19.01],[13.24,-18.98]]]}""")


def tint(c, k=0.012):
    return tuple(np.clip(np.array(c) * (1 + rng.normal(0, k)), 0, 1))


def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def sphere(r, c, seg=12, rings=7):
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


def ccw(P):
    P = [tuple(p) for p in P]
    if len(P) > 1 and math.dist(P[0], P[-1]) < 1e-6:
        P = P[:-1]
    return P if ring_area(P) > 0 else P[::-1]


def bbox(P):
    xs = [p[0] for p in P]; zs = [p[1] for p in P]
    return min(xs), max(xs), min(zs), max(zs)


def model_dir(bearing):
    """A compass bearing (degrees) as a unit vector in the model's x/z."""
    e, n = math.sin(math.radians(bearing)), math.cos(math.radians(bearing))
    c, s = math.cos(math.radians(HEADING)), math.sin(math.radians(HEADING))
    return np.array([-e * c + n * s, e * s + n * c])


# ── Oriented pieces on a wall: local (u along the wall, y up, w outwards) ────────────────────────────────
class Face:
    def __init__(self, a, t, n):
        self.a, self.t, self.n = np.asarray(a, float), np.asarray(t, float), np.asarray(n, float)

    def P(self, u, y, w):
        p = self.a + self.t * u + self.n * w
        return (p[0], y, p[1])

    def poly(self, pts, c, w=0.05):
        """A flat convex polygon [(u, y)] on the wall, facing out."""
        V = [self.P(u, y, w) for u, y in pts]
        F = [(0, i, i + 1) for i in range(1, len(V) - 1)]
        m = Mesh(V, F, c)
        a, b, cc = np.array(V[0]), np.array(V[1]), np.array(V[2])
        nn = np.cross(b - a, cc - a)
        if np.dot([nn[0], nn[2]], self.n) < 0:
            m.f = m.f[:, ::-1]
        return m

    def slab(self, u0, u1, y0, y1, w0, w1, c):
        """A box in local coordinates."""
        P = [self.P(u, y, w) for w in (w0, w1) for y in (y0, y1) for u in (u0, u1)]
        # index: w0: (y0,u0)0 (y0,u1)1 (y1,u0)2 (y1,u1)3 ; w1: +4
        quads = [(4, 5, 7, 6), (1, 0, 2, 3), (0, 4, 6, 2), (5, 1, 3, 7), (6, 7, 3, 2), (0, 1, 5, 4)]
        V, F = [], []
        for q in quads:
            o = len(V); V += [P[k] for k in q]; F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
        m = Mesh(V, F, c)
        # make every face point away from the centre of the box
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


def window(f, u, yc, w, h, sill=True):
    out = [f.poly(rect_pts(u, yc, w, h), GLASS, 0.04)]
    if sill:
        out.append(f.slab(u - w / 2 - 0.15, u + w / 2 + 0.15, yc - h / 2 - 0.16, yc - h / 2, 0.0, 0.2, GRANITE_L))
    return out


# ── Roofs and prisms from OpenStreetMap parts ─────────────────────────────────
def prism_var(P, y0, ytop, c_wall, c_top):
    """Walls from y0 up to ytop(x, z) at each vertex, and the top cap (planar where ytop is linear)."""
    P = ccw(P)
    V, F = [], []
    for i in range(len(P)):
        (x0, z0), (x1, z1) = P[i], P[(i + 1) % len(P)]
        o = len(V)
        V += [(x0, y0, z0), (x1, y0, z1), (x1, ytop(x1, z1), z1), (x0, ytop(x0, z0), z0)]
        F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    w = Mesh(V, F, c_wall)
    pts, tris = triangulate(P)
    v = np.stack([pts[:, 0], [ytop(x, z) for x, z in pts], pts[:, 1]], 1)
    return merge([w, Mesh(v, tris[:, ::-1], c_top)])


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


def roof_colour(p):
    rc = p.get("r_colour", "#68769A").upper()
    return {"#68769A": SLATE, "#7E7A9B": LEAD, "#E3CFC6": GRANITE}.get(rc, SLATE)


def part_heights(p):
    """OSM height of a part -> (wall bottom, eave, roof height) in metres (see the docstring)."""
    h = float(p.get("height", 0) or 0); mh = float(p.get("min_height", 0) or 0)
    rh = float(p.get("r_height", 0) or 0)
    i = p["id"]
    if i in (887821691, 887821692, 887821693, 887821694):
        return 12.0, 12.0, 2.5
    if h == 11.1 and mh == 6.9:
        return 10.5, 10.5, 4.0
    if h == 6.9:
        return None, (9.0, 2.0) if rh else (10.5, 0.0)
    T = {17.5: (20.0, 5.5), 9.45: (11.0, 1.5), 8.45: (10.0, 1.5), 25.5: (32.0, 5.5), 26.0: (38.0, 0.0),
         26.4: (23.0, 6.0), 24.0: (26.0, 5.5), 20.0: (22.0, 3.0), 10.25: (12.0, 0.0)}
    if h == 12.6:
        if p.get("r_shape") is None and not rh:
            return None, (14.0, 0.0)
        return None, (14.0, 0.6 if rh and rh < 1 else 2.2)
    if h == 18.5:
        return None, (22.0, 3.0) if rh < 3 else (21.5, 6.0)
    return None, T.get(h, (h * 1.25, rh * 1.2))


def is_outer(P):
    return any(abs(x) > 103.0 or z > 80.8 or z < -81.0 for x, z in P)


PARTS = []          # [{ring, base, E, top(x,z), id, shape}] for the window pass and for occlusion


def build_part(p):
    P = ccw(p["r"])
    hb = part_heights(p)
    if len(hb) == 3:                       # roof-only pieces over a lower block
        base, E, R = hb
    else:
        base = None; E, R = hb[1]
    y0 = base if base is not None else (FOUND if is_outer(P) else BASE_IN)
    shape = p.get("r_shape") or ("flat" if R == 0 else "skillion")
    rc = roof_colour(p)
    wallc = tint(GRANITE, 0.01)
    x0, x1, z0, z1 = bbox(P)
    out = []
    if R == 0 or shape == "flat":
        out.append(prism_var(P, y0, lambda x, z: E, wallc, LEAD_D if p.get("r_colour", "").upper() != "#E3CFC6" else GRANITE))
        top = lambda x, z: E                                                  # noqa: E731
    elif shape == "skillion":
        d = model_dir(float(p.get("r_direction", 0)))
        pr = [np.dot(d, q) for q in P]
        lo, hi = min(pr), max(pr)
        top = lambda x, z: E + R * (1 - (np.dot(d, (x, z)) - lo) / max(hi - lo, 1e-6))   # noqa: E731
        out.append(prism_var(P, y0, top, wallc, rc))
    elif shape in ("gabled", "hipped"):
        along_x = (x1 - x0) >= (z1 - z0)
        if p.get("r_orientation") == "across":
            along_x = not along_x
        if shape == "hipped" and len(P) <= 6:
            out.append(prism_var(P, y0, lambda x, z: E, wallc, rc))
            out.append(hip_roof(x0, x1, z0, z1, E, R, rc))
            top = lambda x, z: E                                              # noqa: E731
        else:
            if along_x:
                mid, half, ax = (z0 + z1) / 2, (z1 - z0) / 2, 1
            else:
                mid, half, ax = (x0 + x1) / 2, (x1 - x0) / 2, 0
            top = lambda x, z: E + R * (1 - abs((x, z)[ax] - mid) / half)      # noqa: E731
            for keep in (True, False):
                Q = clip(P, ax, mid, keep)
                if len(Q) >= 3:
                    out.append(prism_var(Q, y0, top, wallc, rc))
    else:
        out.append(prism_var(P, y0, lambda x, z: E, wallc, rc))
        top = lambda x, z: E                                                  # noqa: E731
    PARTS.append({"ring": P, "base": y0, "E": E, "top": top, "id": p["id"], "R": R, "shape": shape,
                  "dir": p.get("r_direction"), "levels": p.get("building:levels")})
    return out


def hip_roof(x0, x1, z0, z1, E, R, c):
    L, W = x1 - x0, z1 - z0
    if L >= W:
        a, b = (x0 + W / 2, (z0 + z1) / 2), (x1 - W / 2, (z0 + z1) / 2)
    else:
        a, b = ((x0 + x1) / 2, z0 + L / 2), ((x0 + x1) / 2, z1 - L / 2)
    Y = E + R
    c00, c10, c11, c01 = (x0, E, z0), (x1, E, z0), (x1, E, z1), (x0, E, z1)
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
    return orient_up(m)


def orient_up(m):
    a, b, c = m.v[m.f[:, 0]], m.v[m.f[:, 1]], m.v[m.f[:, 2]]
    nn = np.cross(b - a, c - a)
    flip = nn[:, 1] < 0
    m.f[flip] = m.f[flip][:, ::-1]
    return m


def frustum(cx, cz, s0, s1, y0, y1, c, rot=0.0):
    """Square frustum (sharp edges), side s0 at y0 to s1 at y1."""
    h0, h1 = s0 / 2, s1 / 2
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    V, F = [], []
    for i in range(4):
        (ax, az), (bx, bz) = corners[i], corners[(i + 1) % 4]
        q = [(ax * h0, y0, az * h0), (bx * h0, y0, bz * h0), (bx * h1, y1, bz * h1), (ax * h1, y1, az * h1)]
        o = len(V); V += q
        F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    if s1 > 1e-6:
        o = len(V); V += [(-h1, y1, -h1), (h1, y1, -h1), (h1, y1, h1), (-h1, y1, h1)]
        F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    m = Mesh(V, F, c)
    if rot:
        m.R(0, rot, 0)
    return m.T(cx, 0, cz)


def pinnacle(x, z, y, s=1.2, c=GRANITE_L):
    """Herrera's finial: a plinth, a pyramid and a ball."""
    return [B(x - s / 2, x + s / 2, y, y + 0.9 * s, z - s / 2, z + s / 2, c),
            frustum(x, z, 0.8 * s, 0.05, y + 0.9 * s, y + 2.6 * s, c),
            sphere(0.42 * s, c, 10, 6).T(x, y + 2.6 * s + 0.35 * s, z)]


def ball_on_pier(x, z, y, c=GRANITE_L, s=0.7, seg=10):
    return [B(x - s / 2, x + s / 2, y, y + 0.5, z - s / 2, z + s / 2, c),
            sphere(0.36 * s / 0.7, c, seg, seg // 2 + 1).T(x, y + 0.5 + 0.34 * s / 0.7, z)]


def statue(x, y, z, h=4.2, c=MARBLE, face=(0, 1)):
    k = h / 4.2
    body = lathe([(0.0, 0.0), (0.62, 0.0), (0.62, 0.25), (0.55, 0.3), (0.62, 1.6), (0.55, 2.6), (0.62, 3.05),
                  (0.42, 3.35), (0.18, 3.45), (0.0, 3.46)], c, 10)
    head = sphere(0.3, c, 10, 6).T(0, 3.72, 0)
    m = merge([body, head]).S(k)
    return m.T(x, y, z)


# ── Corner towers, crossing towers, bell towers ───────────────────────────────
def window_rows(f, u_list, rows):
    out = []
    for u in u_list:
        for yc, w, h in rows:
            out += window(f, u, yc, w, h)
    return out


def tower(cx, cz, S, top, spire_top, outer=(), spire_scale=1.0, base=FOUND):
    """A square Herrerian tower: granite body to `top`, cornice, pinnacles with balls, slate spire, needle, ball
    and cross. outer = directions ('+x', …) of the faces on the outside of the building (windows low down)."""
    h = S / 2
    out = [B(cx - h, cx + h, base, top, cz - h, cz + h, tint(GRANITE, 0.01)),
           B(cx - h - 0.45, cx + h + 0.45, top - 0.9, top, cz - h - 0.45, cz + h + 0.45, GRANITE_L),
           B(cx - h - 0.25, cx + h + 0.25, 20.0 - 0.7, 20.0, cz - h - 0.25, cz + h + 0.25, GRANITE_L)]
    faces = {"+x": Face((cx + h, cz + h), (0, -1), (1, 0)), "-x": Face((cx - h, cz - h), (0, 1), (-1, 0)),
             "+z": Face((cx - h, cz + h), (1, 0), (0, 1)), "-z": Face((cx + h, cz - h), (-1, 0), (0, -1))}
    us = [S / 2 - 4.6, S / 2, S / 2 + 4.6] if S > 12 else [S / 2 - 3.2, S / 2 + 3.2]
    upper = [(y, 1.3, 2.0) for y in np.arange(22.6, top - 2.5, 4.4)]
    for k, f in faces.items():
        rows = (list(WIN_ROWS) if k in outer else []) + upper
        if k in outer and ((k == "+x" and cx > 0) or (k == "-z" and cz < 0)):
            rows = [(-0.4, 1.0, 1.1)] + rows
        out += window_rows(f, us, rows)
    for sx in (-1, 1):
        for sz in (-1, 1):
            out += pinnacle(cx + sx * (h - 0.8), cz + sz * (h - 0.8), top, 1.3)
    s0 = S - 3.0
    ys = top + (spire_top - top) * np.array([0.0, 0.55, 0.78, 1.0])
    out += [frustum(cx, cz, s0, s0 * 0.42, ys[0], ys[1], SLATE),
            frustum(cx, cz, s0 * 0.42, s0 * 0.16, ys[1], ys[2], SLATE),
            cyl(0.35, 0.14, ys[3] - ys[2], LEAD, 8).T(cx, ys[2], cz),
            sphere(0.62 * spire_scale, GOLD, 12, 7).T(cx, ys[2] + (ys[3] - ys[2]) * 0.45, cz)]
    # dormers (lucarnes) on the spire
    for k, f in faces.items():
        g = Face(f.a + f.n * (-1.5) + f.t * 0.0, f.t, f.n)
        yy = top + 1.2
        out.append(g.slab(S / 2 - 0.8, S / 2 + 0.8, yy, yy + 2.0, -1.0, 0.0, SLATE))
        out.append(g.poly(rect_pts(S / 2, yy + 0.9, 0.9, 1.3), GLASS, 0.03))
    # iron cross / weather vane
    yt = ys[3]
    out += [B(cx - 0.06, cx + 0.06, yt, yt + 2.0, cz - 0.06, cz + 0.06, IRON),
            B(cx - 0.55, cx + 0.55, yt + 1.25, yt + 1.37, cz - 0.05, cz + 0.05, IRON)]
    return out


def bell_tower(cx, cz, S=14.6):
    h = S / 2
    top = 47.0
    out = [B(cx - h, cx + h, BASE_IN, top, cz - h, cz + h, tint(GRANITE, 0.01)),
           B(cx - h - 0.5, cx + h + 0.5, top - 1.0, top, cz - h - 0.5, cz + h + 0.5, GRANITE_L),
           B(cx - h - 0.3, cx + h + 0.3, 32.0 - 0.8, 32.0, cz - h - 0.3, cz + h + 0.3, GRANITE_L),
           B(cx - h - 0.3, cx + h + 0.3, 38.0 - 0.6, 38.0, cz - h - 0.3, cz + h + 0.3, GRANITE_L)]
    faces = [Face((cx + h, cz + h), (0, -1), (1, 0)), Face((cx - h, cz - h), (0, 1), (-1, 0)),
             Face((cx - h, cz + h), (1, 0), (0, 1)), Face((cx + h, cz - h), (-1, 0), (0, -1))]
    for f in faces:
        out.append(f.poly(arch_pts(S / 2, 39.0, 43.2, 1.9), DARK, 0.04))            # the belfry
        out += window(f, S / 2, 34.6, 1.4, 2.2)
        out += window(f, S / 2, 27.5, 1.4, 2.2)
    for sx in (-1, 1):
        for sz in (-1, 1):
            out += pinnacle(cx + sx * (h - 0.9), cz + sz * (h - 0.9), top, 1.4)
    # drum, cupola, lantern, ball, cross
    out += [cyl(5.2, 5.2, 8.0, GRANITE, 24).T(cx, top, cz),
            cyl(5.7, 5.7, 0.8, GRANITE_L, 24).T(cx, top + 7.6, cz)]
    for k in range(8):
        a = 2 * math.pi * (k + 0.5) / 8
        n = np.array([math.cos(a), math.sin(a)])
        t = np.array([-n[1], n[0]])
        f = Face(np.array([cx, cz]) + n * 5.2 - t * 0.8, t, n)
        out.append(f.poly(arch_pts(0.8, top + 1.2, top + 5.4, 0.8), DARK, 0.04))
    dome = lathe([(5.4, 0.0), (5.1, 1.4), (4.3, 3.0), (3.0, 4.2), (1.7, 4.8), (0.0, 4.95)], GRANITE, 24)
    out.append(dome.T(cx, top + 8.4, cz))
    yl = top + 8.4 + 4.6
    out += [cyl(1.6, 1.6, 3.4, GRANITE, 12).T(cx, yl, cz), cyl(1.9, 1.9, 0.4, GRANITE_L, 12).T(cx, yl + 3.4, cz),
            lathe([(1.7, 0.0), (1.2, 0.9), (0.4, 1.6), (0.0, 1.7)], GRANITE, 12).T(cx, yl + 3.8, cz),
            cyl(0.22, 0.1, 2.2, LEAD, 6).T(cx, yl + 5.4, cz),
            sphere(0.55, GOLD, 12, 7).T(cx, yl + 7.2, cz),
            B(cx - 0.07, cx + 0.07, yl + 7.6, yl + 9.4, cz - 0.07, cz + 0.07, IRON),
            B(cx - 0.6, cx + 0.6, yl + 8.6, yl + 8.74, cz - 0.06, cz + 0.06, IRON)]
    return out


def dome(cx, cz):
    out = [cyl(12.0, 12.0, 19.5, tint(GRANITE, 0.01), 40).T(cx, 38.0, cz),
           cyl(12.7, 12.7, 1.1, GRANITE_L, 40).T(cx, 56.4, cz),
           cyl(12.25, 12.25, 2.0, GRANITE, 40).T(cx, 57.5, cz),
           cyl(12.6, 12.6, 0.5, GRANITE_L, 40).T(cx, 59.3, cz)]
    for k in range(16):                                                   # paired pilasters and windows
        a = 2 * math.pi * k / 16
        n = np.array([math.cos(a), math.sin(a)]); t = np.array([-n[1], n[0]])
        f = Face(np.array([cx, cz]) + n * 12.0, t, n)
        if k % 2 == 0:
            out.append(f.slab(-0.55, 0.55, 39.0, 56.4, -0.2, 0.55, GRANITE_L))
        else:
            out.append(f.poly(arch_pts(0.0, 44.0, 51.5, 1.5), GLASS, 0.03))
            out.append(f.slab(-2.05, 2.05, 43.6, 44.0, -0.1, 0.3, GRANITE_L))
    prof = [(12.0, 0.0)] + [(12.0 * math.cos(t), 11.0 * math.sin(t)) for t in np.linspace(0.12, 1.27, 10)]
    out.append(lathe(prof, GRANITE, 40).T(cx, 59.8, cz))
    for k in range(16):                                                   # ribs
        a = 2 * math.pi * k / 16
        pts = [(cx + (r + 0.12) * math.cos(a), 59.8 + y + 0.1, cz + (r + 0.12) * math.sin(a))
               for r, y in prof[1:-1]]
        out.append(tube(pts, 0.32, GRANITE_L, 5))
    for k in range(8):                                                    # obelisks with balls around the drum
        a = 2 * math.pi * (k + 0.5) / 8
        out += pinnacle(cx + 13.2 * math.cos(a), cz + 13.2 * math.sin(a), 56.4, 1.1)
    yl = 59.8 + 11.0 * math.sin(1.27)
    out += [cyl(3.6, 3.6, 0.6, GRANITE_L, 20).T(cx, yl - 0.2, cz),
            cyl(3.3, 3.3, 7.6, GRANITE, 20).T(cx, yl + 0.4, cz),
            cyl(3.75, 3.75, 0.7, GRANITE_L, 20).T(cx, yl + 8.0, cz)]
    for k in range(8):
        a = 2 * math.pi * (k + 0.5) / 8
        n = np.array([math.cos(a), math.sin(a)]); t = np.array([-n[1], n[0]])
        f = Face(np.array([cx, cz]) + n * 3.3, t, n)
        out.append(f.poly(arch_pts(0.0, yl + 1.6, yl + 5.6, 0.75), GLASS, 0.03))
    yc = yl + 8.7
    out += [lathe([(3.5, 0.0), (3.0, 1.2), (2.0, 2.4), (0.9, 3.1), (0.0, 3.25)], GRANITE, 20).T(cx, yc, cz),
            lathe([(0.75, 0.0), (0.55, 2.5), (0.3, 5.2), (0.0, 5.3)], GRANITE_L, 10).T(cx, yc + 3.0, cz),
            sphere(1.0, GOLD, 14, 8).T(cx, yc + 8.8, cz)]
    yt = yc + 9.8
    out += [B(cx - 0.12, cx + 0.12, yt, yt + 96.5 - yt, cz - 0.12, cz + 0.12, IRON),
            B(cx - 1.1, cx + 1.1, yt + 2.9, yt + 3.15, cz - 0.1, cz + 0.1, IRON),
            B(cx - 0.1, cx + 0.1, yt + 2.9, yt + 3.15, cz - 1.1, cz + 1.1, IRON)]
    return out


# ── Windows, arcades, cornices and dormers on every outside wall of the parts ──────────────────────────
OCC = []            # extra volumes for the window test: (ring, top)


def cover_top(x, z, me):
    t = -99.0
    for q in PARTS:
        if q is me:
            continue
        x0, x1, z0, z1 = q["bb"]
        if x0 - 0.01 <= x <= x1 + 0.01 and z0 - 0.01 <= z <= z1 + 0.01 and inside(x, z, q["ring"]):
            t = max(t, q["E"] + (q["R"] if q["shape"] != "flat" else 0))
    for ring, top in OCC:
        if inside(x, z, ring):
            t = max(t, top)
    return t


def courtyard_of(x, z):
    for name, r in DATA["courtyards"]:
        if inside(x, z, r):
            return name or "patio"
    return None


def facades():
    out = []
    for q in PARTS:
        q["bb"] = bbox(q["ring"])
    for q in PARTS:
        P = q["ring"]; E = q["E"]
        if E < 9:
            continue
        church = q["id"] in (885736162, 885736167, 885736168, 886413030, 885736169, 885736166)
        for i in range(len(P)):
            a, b = np.array(P[i]), np.array(P[(i + 1) % len(P)])
            L = float(np.linalg.norm(b - a))
            if L < 3.0:
                continue
            t = (b - a) / L
            if min(abs(t[0]), abs(t[1])) > 0.05:
                continue
            n = np.array([t[1], -t[0]])
            f = Face(a, t, n)
            mid = (a + b) / 2 + n * 0.8
            yt = min(q["top"](*a), q["top"](*b))
            low_edge = abs(q["top"](*a) - E) < 0.05 and abs(q["top"](*b) - E) < 0.05
            court = courtyard_of(*mid)
            outer_face = (n[0] > 0.9 and a[0] > 103.0) or (n[0] < -0.9 and a[0] < -103.0) or \
                         (n[1] > 0.9 and a[1] > 80.8) or (n[1] < -0.9 and a[1] < -80.0)
            # cornice where the wall meets the eaves
            if low_edge and cover_top(*mid, q) < E - 1:
                out.append(f.slab(-0.3, L + 0.3, E - 0.85, E - 0.05, -0.05, 0.42, GRANITE_L))
            if church:
                if cover_top(*mid, q) < E - 8:
                    for u in np.arange(4.5, L - 3.0, 9.0):
                        out.append(f.poly(arch_pts(u, E - 10.5, E - 5.5, 1.4), GLASS, 0.04))
                continue
            gallery = abs(E - 14.0) < 0.01 and court is not None
            n_bays = max(1, int(round((L - 2.0) / BAY)))
            step = (L - 2.0) / n_bays
            us = [1.0 + step * (k + 0.5) for k in range(n_bays)]
            for u in us:
                p = a + t * u + n * 0.8
                ct = cover_top(*p, q)
                if gallery:
                    if ct > 1.0:
                        continue
                    out.append(f.poly(arch_pts(u, 0.0, 3.6, min(1.45, step * 0.32)), DARK, 0.04))
                    if court == "Patio de los Evangelistas":
                        out.append(f.poly(arch_pts(u, 7.0, 10.4, min(1.35, step * 0.3)), DARK, 0.04))
                    else:
                        out += window(f, u, 9.4, 1.2, 1.8)
                    continue
                rows = [r for r in WIN_ROWS if r[0] + r[2] / 2 < yt - 1.4 and r[0] - r[2] / 2 > ct + 0.3]
                if outer_face and ((n[0] > 0.9) or (n[1] < -0.9)) and ct < -1.0:
                    rows = [(-0.4, 1.0, 1.1)] + rows
                for yc, w, h in rows:
                    out += window(f, u, yc, w, h, sill=yc > 1)
            # dormers on the outward slope of the big wings
            if low_edge and E >= 20 and q["shape"] == "skillion" and not gallery and cover_top(*mid, q) < E:
                Rr = q["R"]
                for u in us[1::2] if len(us) > 2 else []:
                    p = a + t * u
                    if cover_top(*(p + n * 0.8), q) > E - 1:
                        continue
                    g = Face(p - n * 1.3 - t * u, t, n)
                    yb = E + Rr * 1.3 / 4.7
                    out.append(g.slab(u - 0.75, u + 0.75, E - 0.1, yb + 1.9, -1.6, 0.0, SLATE))
                    out.append(g.poly(rect_pts(u, yb + 0.8, 0.8, 1.2), GLASS, 0.03))
                    out.append(dormer_cap(g, u, yb + 1.9))
    return out


def dormer_cap(g, u, y):
    V = [g.P(u - 0.85, y, 0.05), g.P(u + 0.85, y, 0.05), g.P(u, y + 0.8, 0.05),
         g.P(u - 0.85, y, -1.7), g.P(u + 0.85, y, -1.7), g.P(u, y + 0.8, -1.7)]
    F = [(0, 1, 2), (0, 2, 5), (0, 5, 3), (1, 4, 5), (1, 5, 2)]
    return Mesh(V, F, SLATE)


# ── Portals and the façade of the basilica ─────────────────────────────────────
def half_col(f, u, y0, y1, r, c, w=0.0):
    p = f.P(u, 0, w + r * 0.35)
    return cyl(r, r * 0.9, y1 - y0, c, 10).T(p[0], y0, p[2])


def main_portal():
    """The west portal on the Lonja (OSM ways 888162152, 887821676; statue node 8266944838)."""
    out = []
    z0 = 81.2
    f = Face((-17.8, z0 + 2.0), (1, 0), (0, 1))                    # front plane of the projecting body, u from x = −17.8
    out.append(B(-17.8, 18.0, FOUND, 13.5, z0 - 0.5, z0 + 2.0, tint(GRANITE, 0.01)))
    out.append(B(-18.3, 18.5, 12.6, 13.5, z0 - 0.5, z0 + 2.5, GRANITE_L))
    for x in (-15.4, -13.0, -5.4, -3.0, 3.0, 5.4, 13.0, 15.4):
        out.append(half_col(f, x + 17.8, 0.6, 12.4, 0.5, GRANITE_L))
    out.append(f.poly(arch_pts(17.8, 0.0, 6.4, 1.9), DOOR, 0.05))
    for x in (-9.2, 9.2):
        out.append(f.poly(arch_pts(x + 17.8, 0.0, 4.6, 1.3), DOOR, 0.05))
        out += window(f, x + 17.8, 9.4, 1.3, 2.1)
    # upper body with the Ionic order, the niche of San Lorenzo and the pediment
    out.append(B(-10.9, 10.6, 13.5, 25.5, z0 - 0.5, z0 + 1.6, tint(GRANITE, 0.01)))
    g = Face((-10.9, z0 + 1.6), (1, 0), (0, 1))
    for x in (-8.6, -3.0, 3.0, 8.4):
        out.append(half_col(g, x + 10.9, 14.0, 24.4, 0.42, GRANITE_L))
    out.append(B(-11.3, 11.0, 24.5, 25.5, z0 - 0.5, z0 + 2.0, GRANITE_L))
    out.append(g.poly(arch_pts(10.9, 14.8, 19.6, 1.4), (0.35, 0.34, 0.32), 0.04))
    out.append(statue(0.06, 15.0, z0 + 1.15, 4.2, MARBLE))
    out += window(g, 10.9 - 5.8, 19.0, 1.3, 2.4) + window(g, 10.9 + 5.6, 19.0, 1.3, 2.4)
    ped = [(-11.3, 25.5), (11.0, 25.5), (-0.15, 29.6)]
    V = [(x, y, z0 + 2.0) for x, y in ped] + [(x, y, z0 - 0.5) for x, y in ped]
    out.append(Mesh(V, [(0, 1, 2), (3, 5, 4), (0, 3, 4), (0, 4, 1), (1, 4, 5), (1, 5, 2), (2, 5, 3), (2, 3, 0)], GRANITE_L))
    for x, y in ((-11.0, 25.5), (10.7, 25.5), (-0.15, 29.4)):
        out += ball_on_pier(x, z0 + 0.7, y, GRANITE_L, 0.8)
    for x in (-17.6, 17.8):
        out += ball_on_pier(x, z0 + 0.9, 13.5, GRANITE_L, 0.8)
    OCC.append(([(-18.0, z0 - 0.5), (18.2, z0 - 0.5), (18.2, z0 + 2.5), (-18.0, z0 + 2.5)], 13.5))
    # the two side portals (College and Convent): doors in their projecting bodies
    for xc in (-61.6, 61.0):
        h = Face((xc - 6.5, 81.38), (1, 0), (0, 1))
        out.append(h.poly(arch_pts(6.5, 0.0, 4.4, 1.6), DOOR, 0.05))
        out.append(half_col(h, 6.5 - 3.0, 0.4, 8.0, 0.38, GRANITE_L)); out.append(half_col(h, 6.5 + 3.0, 0.4, 8.0, 0.38, GRANITE_L))
        out.append(h.slab(2.8, 10.2, 8.0, 8.8, -0.05, 0.6, GRANITE_L))
    return out


def basilica_front():
    """The façade of the basilica on the Patio de Reyes, z = 4.56 (nave gable above)."""
    out = []
    z0 = 4.56
    f = Face((-13.2, z0), (1, 0), (0, 1))
    # steps across the front (the church floor is ~1.5 m above the patio)
    for k in range(7):
        out.append(B(-13.2, 13.2, 0.0, 1.5 - k * 0.21, z0, z0 + 0.6 * (k + 1), GRANITE_L))
    # lower order: six Tuscan half-columns under the six kings, three arches
    for x in (-12.1, -9.9, -3.44, 3.43, 9.56, 11.78):
        out.append(half_col(f, x + 13.2, 1.5, 12.6, 0.55, GRANITE_L))
        out.append(B(x - 0.75, x + 0.75, 13.7, 15.1, z0 - 0.3, z0 + 1.2, GRANITE_L))
    out.append(f.slab(-0.4, 26.8, 12.6, 13.7, -0.05, 1.1, GRANITE_L))
    out.append(f.poly(arch_pts(13.2, 1.5, 7.6, 2.4), DOOR, 0.06))
    for u in (13.2 - 6.6, 13.2 + 6.6):
        out.append(f.poly(arch_pts(u, 1.5, 6.6, 1.6), DOOR, 0.06))
    for name, x, z in DATA["statues"]:
        if z < 20:
            out.append(statue(x, 15.1, z0 + 0.45, 4.2, MARBLE))
            out.append(cyl(0.34, 0.4, 0.45, GOLD, 10).T(x, 15.1 + 3.95, z0 + 0.45))     # gilt crowns
    # upper body: the great window, the order and the pediment of the nave gable
    out.append(f.poly(arch_pts(13.2, 18.5, 24.5, 2.4), GLASS, 0.05))
    for x in (-9.0, -5.2, 5.2, 9.0):
        out.append(f.slab(x + 13.2 - 0.45, x + 13.2 + 0.45, 15.2, 31.4, -0.05, 0.45, GRANITE_L))
    out.append(f.slab(-0.4, 26.8, 31.2, 32.0, -0.05, 0.6, GRANITE_L))
    for u in (0.0, 26.4):
        out += ball_on_pier(u - 13.2, z0 + 0.3, 32.0, GRANITE_L, 0.9)
    out.append(tube([(-13.4, 32.1, z0 + 0.25), (0.0, 37.6, z0 + 0.25), (13.4, 32.1, z0 + 0.25)], 0.35, GRANITE_L, 5))
    return out


# ── Courtyards ───────────────────────────────────────────────────────────────
def floors():
    out = []
    for name, r in DATA["courtyards"]:
        c = PAVING
        if name.startswith("Patio Chico D") or name == "Patio Chico I1":
            c = GRASS
        elif name.startswith("Patio Chico I"):
            c = GRAVEL
        out.append(cap(r, 0.03, c))
    return out


def basin(ring, y_floor, rim_h, c_rim=GRANITE_L, water=0.35, w=0.4):
    """A fountain basin: a rim along the ring, water inside."""
    P = ccw(ring)
    inner = offset(P, -w)
    out = [merge([walls(P, y_floor - 0.3, y_floor + rim_h, c_rim), walls(inner, y_floor, y_floor + rim_h, c_rim, hole=True),
                  cap(P, y_floor + rim_h, c_rim, holes=[inner])]),
           cap(inner, y_floor + rim_h - water, WATER)]
    return out


def round_fountain(x, z, y, r=1.4):
    a = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    ring = [(x + r * math.cos(t), z + r * math.sin(t)) for t in a]
    out = basin(ring, y, 0.55, w=0.3)
    out.append(lathe([(0.0, 0.0), (0.22, 0.0), (0.16, 0.6), (0.55, 0.85), (0.6, 0.95), (0.12, 1.0), (0.1, 1.4), (0.0, 1.5)],
                     GRANITE_L, 10).T(x, y + 0.2, z))
    return out


def templete():
    """Herrera's templete in the middle of the Patio de los Evangelistas (way 663968140, ponds 663968139/41–43)."""
    P = ccw(DATA["templete"])
    cx = sum(p[0] for p in P) / len(P); cz = sum(p[1] for p in P) / len(P)
    out = [extr(offset(P, 1.0), 0.0, 0.45, GRANITE_L), extr(P, 0.45, 8.0, tint(GRANITE_L, 0.01)),
           extr(offset(P, 0.35), 8.0, 8.7, GRANITE_L)]
    for i in range(len(P)):
        a, b = np.array(P[i]), np.array(P[(i + 1) % len(P)])
        L = float(np.linalg.norm(b - a)); t = (b - a) / L; n = np.array([t[1], -t[0]])
        f = Face(a, t, n)
        out.append(f.poly(arch_pts(L / 2, 0.45, 4.6, min(1.4, L * 0.3)), DARK, 0.04))
        out.append(f.slab(-0.2, 0.2, 0.45, 8.0, -0.05, 0.4, GRANITE))
    for k in range(len(P)):
        out += ball_on_pier(P[k][0] + (P[k][0] - cx) * 0.06, P[k][1] + (P[k][1] - cz) * 0.06, 8.7, GRANITE_L, 0.55)
    out.append(lathe([(3.9, 0.0), (3.6, 1.4), (2.8, 2.7), (1.6, 3.5), (0.0, 3.8)], GRANITE, 16).T(cx, 8.7, cz))
    out += [cyl(0.7, 0.7, 1.4, GRANITE_L, 10).T(cx, 12.3, cz),
            lathe([(0.8, 0.0), (0.4, 0.5), (0.0, 0.7)], GRANITE_L, 10).T(cx, 13.7, cz),
            sphere(0.3, GRANITE_L, 8, 5).T(cx, 14.65, cz)]
    return out


def extr(P, y0, y1, c, top_c=None):
    P = ccw(P)
    return merge([walls(P, y0, y1, c), cap(P, y1, top_c or c)])


# ── Gardens ──────────────────────────────────────────────────────────────────
def chain_terrace():
    """The terrace of the Jardín de los Frailes and the Jardines del Rey: the building's south and east fronts
    on one side, the retaining walls on the other (OSM ways 232765850, 232765849, 1337321442)."""
    O = [tuple(p) for p in DATA["outline"]]
    i0 = min(range(len(O)), key=lambda i: math.dist(O[i], (-103.88, -81.3)))
    i1 = min(range(len(O)), key=lambda i: math.dist(O[i], (103.85, 81.36)))
    n = len(O)
    fwd = [O[(i0 + k) % n] for k in range((i1 - i0) % n + 1)]
    bwd = [O[(i0 - k) % n] for k in range((i0 - i1) % n + 1)]
    arc = fwd if min(p[1] for p in fwd) < min(p[1] for p in bwd) else bwd
    edge = [(109.16, 81.38), (109.08, 106.2), (132.24, 106.31), (132.18, 75.17), (132.04, -1.81), (131.89, -72.78),
            (131.82, -107.9), (45.46, -108.39), (45.45, -156.72), (-46.23, -156.86), (-46.23, -108.42), (-103.96, -108.51)]
    return arc + edge, edge[2:] + [arc[0]]


def gardens():
    out = []
    ring, wall_line = chain_terrace()
    slots = [ccw(r) for i, r in DATA["retaining"] if 1337321443 <= i <= 1337321448 or 1337321508 <= i <= 1337321513]
    out.append(cap(ring, G, GRAVEL, holes=slots))
    # retaining walls with a parapet and granite balls
    for k in range(len(wall_line) - 1):
        a, b = np.array(wall_line[k]), np.array(wall_line[k + 1])
        L = float(np.linalg.norm(b - a)); t = (b - a) / L
        n = np.array([t[1], -t[0]])
        # outward = away from the building centre
        if np.dot(n, (a + b) / 2) < 0:
            n = -n
        f = Face(a, t, n)
        out.append(f.slab(0.0, L, -16.0, G + 1.05, -0.05, 0.75, tint(GRANITE_D, 0.01)))
        out.append(f.slab(-0.15, L + 0.15, G + 1.05, G + 1.3, -0.15, 0.9, GRANITE_L))
        out.append(f.slab(0.0, L, -16.0, -3.0, 0.75, 1.25, GRANITE_D))              # batter of the base
        for u in np.arange(0.0, L + 0.01, L / max(1, round(L / 11.0))):
            p = a + t * u + n * 0.37
            out += ball_on_pier(p[0], p[1], G + 1.3, GRANITE_L, 0.75)
    # sunken stairs into the vaults under the terrace
    for P in slots:
        x0, x1, z0, z1 = bbox(P)
        out.append(merge([walls(P, G - 4.2, G, GRANITE_D, hole=True)]))
        along_x = (x1 - x0) > (z1 - z0)
        nst = 12
        for k in range(nst):
            s = k / nst
            if along_x:
                xa, xb = x0 + (x1 - x0) * s, x0 + (x1 - x0) * (s + 1 / nst)
                out.append(B(xa, xb, G - 4.2, G - 0.3 - 3.8 * s, z0, z1, GRANITE_D))
            else:
                za, zb = z0 + (z1 - z0) * s, z0 + (z1 - z0) * (s + 1 / nst)
                out.append(B(x0, x1, G - 4.2, G - 0.3 - 3.8 * s, za, zb, GRANITE_D))
        out.append(merge([walls(offset(P, 0.35), G, G + 0.9, GRANITE_L), cap(offset(P, 0.35), G + 0.9, GRANITE_L, holes=[P])]))
    # the walled lower garden (north-east): its floor, enclosure, parterres and round fountain
    low = [(-103.96, -108.51), (-46.23, -108.42), (-46.23, -156.86), (-102.64, -156.79)]
    out.append(merge([walls(low, -12.0, G_LOW, GRANITE_D), cap(low, G_LOW, GRAVEL)]))
    for i, r in DATA["walls"]:
        if i in (1337321502, 1337321503):
            out += wall_line_mesh(r, -12.0, G_LOW + 2.6, 0.7)
    out += round_fountain(-73.2, -133.7, G_LOW, 2.3)
    # parterres
    for hdg in DATA["hedges"]:
        P = ccw(hdg["r"])
        x0, x1, z0, z1 = bbox(P)
        cxz = ((x0 + x1) / 2, (z0 + z1) / 2)
        if courtyard_of(*cxz):
            y = 0.03
        elif cxz[0] < -46 and cxz[1] < -108.5:
            y = G_LOW
        elif inside(cxz[0], cxz[1], ring):
            y = G
        else:
            continue
        h = max(0.45, min(0.8, hdg["h"]))
        holes = [ccw(q) for q in hdg.get("holes", [])]
        out.append(merge([walls(P, y - 0.1, y + h, tint(BOX, 0.06))] + [walls(q, y - 0.1, y + h, BOX, hole=True) for q in holes] +
                         [cap(P, y + h, tint(BOX, 0.05), holes=holes)]))
        HEDGES.append((cxz, (x1 - x0) * (z1 - z0), y))
    # a fountain in the middle of every group of parterres on the terrace (the gaps between four of them)
    for c, y in hedge_group_centres():
        out += round_fountain(c[0], c[1], y, 1.25)
    # walls dividing the gardens (King's garden / Frailes) on the terrace
    for i, r in DATA["walls"]:
        if i in (1337321497, 1337321498, 1337321499, 1337321500):
            out += wall_line_mesh(r, G - 0.5, G + 2.6, 0.6)
    return out


HEDGES = []


def hedge_group_centres():
    """Centres of the groups of parterres (clusters of hedges ≥ 30 m² on the terrace, outside the courtyards)."""
    pts = [(c, y) for c, a, y in HEDGES if a > 30 and y != 0.03]
    used, out = set(), []
    for i, (c, y) in enumerate(pts):
        if i in used:
            continue
        grp = [j for j, (d, yy) in enumerate(pts) if math.dist(c, d) < 16 and yy == y]
        if len(grp) < 3:
            continue
        cl = [pts[j][0] for j in grp]
        cx = sum(p[0] for p in cl) / len(cl); cz = sum(p[1] for p in cl) / len(cl)
        sel = sorted(range(len(pts)), key=lambda j: math.dist(pts[j][0], (cx, cz)))[:4]
        if all(math.dist(pts[j][0], (cx, cz)) < 13 for j in sel):
            cx = sum(pts[j][0][0] for j in sel) / 4; cz = sum(pts[j][0][1] for j in sel) / 4
            if all(math.dist((cx, cz), o[0]) > 6 for o in out):
                out.append(((cx, cz), y))
            used.update(sel)
    return out


def wall_line_mesh(r, y0, y1, th, balls=False, cap_c=GRANITE_L):
    out = []
    for k in range(len(r) - 1):
        a, b = np.array(r[k]), np.array(r[k + 1])
        L = float(np.linalg.norm(b - a))
        if L < 0.3:
            continue
        t = (b - a) / L; n = np.array([t[1], -t[0]])
        f = Face(a - n * th / 2, t, n)
        out.append(f.slab(0.0, L, y0, y1, 0.0, th, tint(GRANITE_D, 0.01)))
        out.append(f.slab(-0.05, L + 0.05, y1, y1 + 0.2, -0.08, th + 0.08, cap_c))
        if balls:
            for u in np.arange(0.0, L + 0.01, L / max(1, round(L / 9.0))):
                p = a + t * u
                out += ball_on_pier(p[0], p[1], y1 + 0.2, GRANITE_L, 0.6, seg=8)
    return out


def galeria():
    """Galería de Convalecientes (relation 3126701) and the Botica (way 890760522): two storeys of arcades on
    the garden (−z) and orchard (+x) sides, slate hipped roofs."""
    out = []
    E, R = 9.5, 3.2
    blocks = [(109.2, 132.2, 106.3, 113.6, "x"), (109.2, 132.2, 125.0, 137.1, "x"), (109.2, 113.0, 113.6, 125.0, "z"),
              (124.8, 132.2, 113.6, 125.0, "z"), (105.5, 111.8, 137.0, 163.5, "z"), (101.6, 109.2, 81.3, 137.0, "z")]
    for x0, x1, z0, z1, ax in blocks:
        y0 = -12.0 if x1 > 120 or z0 > 136 else FOUND
        out.append(B(x0, x1, y0, E, z0, z1, tint(GRANITE, 0.01)))
        out.append(hip_roof(x0, x1, z0, z1, E, R, SLATE))
        OCC.append(([(x0, z0), (x1, z0), (x1, z1), (x0, z1)], E + R))
    # garden front (z = 106.3, facing −z): lower arcade on Tuscan piers, upper loggia on Ionic columns
    f = Face((132.2, 106.3), (-1, 0), (0, -1))
    for u in np.arange(1.7, 23.0, 3.3):
        out.append(f.poly(arch_pts(u, G, G + 3.6, 1.15), DARK, 0.04))
        out.append(f.poly(rect_pts(u, 6.0, 2.4, 4.2), DARK, 0.04))
        out.append(half_col(f, u + 1.65, 3.9, 8.2, 0.22, GRANITE_L))
    out.append(f.slab(-0.2, 23.2, 3.4, 3.9, -0.05, 0.5, GRANITE_L))
    out.append(f.slab(-0.2, 23.2, E - 0.7, E, -0.05, 0.45, GRANITE_L))
    # orchard front (x = 132.2, facing +x): blind base with small windows, then the two arcades
    f = Face((132.2, 106.3), (0, 1), (1, 0))
    for u in np.arange(1.7, 30.0, 3.3):
        out.append(f.poly(rect_pts(u, -6.0, 0.9, 1.2), GLASS, 0.04))
        out.append(f.poly(arch_pts(u, G, G + 3.6, 1.15), DARK, 0.04))
        out.append(f.poly(rect_pts(u, 6.0, 2.4, 4.2), DARK, 0.04))
    out.append(f.slab(-0.2, 31.0, 3.4, 3.9, -0.05, 0.5, GRANITE_L))
    out.append(f.slab(-0.2, 31.0, E - 0.7, E, -0.05, 0.45, GRANITE_L))
    out.append(f.slab(-0.2, 31.0, G - 0.6, G, -0.05, 0.4, GRANITE_L))
    # the plain front on the Lonja (x = 101.6) and the Botica's windows
    f = Face((101.6, 137.0), (0, -1), (-1, 0))
    for u in np.arange(3.0, 55.0, 4.6):
        out += window(f, u, 2.4, 1.1, 1.5) + window(f, u, 6.6, 1.2, 1.9)
    f = Face((105.5, 163.5), (0, -1), (-1, 0))
    for u in np.arange(2.6, 26.0, 4.6):
        out += window(f, u, 2.4, 1.1, 1.5) + window(f, u, 6.6, 1.2, 1.9)
    return out


def pond():
    """The Estanque of the orchard (way 26136021), its walk, walls (ways 1337321477–86) and the retaining wall
    on its west side (way 1337321473)."""
    out = []
    P = ccw(DATA["pond"])
    walk = offset(P, 3.2)
    out.append(merge([walls(walk, -16.0, POND + 0.6, GRANITE_D), cap(walk, POND + 0.6, PAVING, holes=[offset(P, 0.6)])]))
    out += basin(offset(P, 0.6), POND - 1.5, 2.6, GRANITE_L, water=0.55, w=0.6)
    for i, r in DATA["walls"]:
        if i in (1337321477, 1337321483, 1337321484, 1337321485, 1337321486):
            out += wall_line_mesh(r, POND + 0.4, POND + 1.6, 0.55, balls=True)
    for i, r in DATA["retaining"]:
        if i == 1337321473:
            out += wall_line_mesh(r, -18.0, -5.0, 1.2, balls=True)
    return out


def lonja():
    """The granite parapets of the Lonja with their balls (ways 1337328790–98)."""
    out = []
    for i, r in DATA["walls"]:
        if 1337328790 <= i <= 1337328798:
            out += wall_line_mesh(r, -2.0, 1.0, 0.75, balls=True)
    return out


# ── Assembly ─────────────────────────────────────────────────────────────────
TOWERS = {"Torre del Colegio": (-96.8, 74.25, ("-x", "+z")), "Torre de la Botica": (96.9, 74.4, ("+x", "+z")),
          "Torre del Noviciado": (96.85, -74.15, ("+x", "-z")), "Torre de Damas": (-96.95, -74.35, ("-x", "-z"))}
FOOTPRINT = [(-104.9, 84.2), (18.9, 84.2), (18.9, 84.4), (100.6, 84.4), (100.6, 138.0), (104.5, 138.0), (104.5, 164.5),
             (112.8, 164.5), (112.8, 138.1), (138.8, 138.1), (138.8, 131.4), (203.6, 131.4), (203.6, 80.8),
             (133.2, 80.8), (133.2, -109.4), (46.5, -109.4), (46.5, -157.8), (-104.9, -157.8)]


def build():
    parts_mesh = []
    for p in DATA["parts"]:
        x0, x1, z0, z1 = bbox(p["r"])
        if -105 < x0 and x1 < 105 and -131 < z0 and z1 < 84:            # the monastery only (not the Lonja's houses)
            parts_mesh += build_part(p)
    for name, (cx, cz, outer) in TOWERS.items():
        OCC.append(([(cx - 7, cz - 7), (cx + 7, cz - 7), (cx + 7, cz + 7), (cx - 7, cz + 7)], 34.0))
    for cx, cz in ((-61.65, 38.2), (60.9, 38.3)):
        OCC.append(([(cx - 6.6, cz - 6.6), (cx + 6.6, cz - 6.6), (cx + 6.6, cz + 6.6), (cx - 6.6, cz + 6.6)], 27.0))
    for cx in (-25.3, 25.3):
        OCC.append(([(cx - 7.3, -9.9), (cx + 7.3, -9.9), (cx + 7.3, 4.7), (cx - 7.3, 4.7)], 47.0))
    portal = main_portal()
    gal = galeria()
    towers = []
    for name, (cx, cz, outer) in TOWERS.items():
        towers += tower(cx, cz, 14.0, 34.0, 55.0, outer)
    for cx, cz in ((-61.65, 38.2), (60.9, 38.3)):
        towers += tower(cx, cz, 13.2, 27.0, 41.0, (), 0.8, BASE_IN)
    church = bell_tower(-25.3, -2.6) + bell_tower(25.3, -2.6) + dome(0.0, -53.1) + basilica_front()
    fac = facades()
    court = floors() + templete()
    for r in DATA["fountains"]:
        x0, x1, z0, z1 = bbox(r)
        if (x1 - x0) < 4 and courtyard_of((x0 + x1) / 2, (z0 + z1) / 2):
            court += round_fountain((x0 + x1) / 2, (z0 + z1) / 2, 0.03, (x1 - x0) / 2)
        elif courtyard_of((x0 + x1) / 2, (z0 + z1) / 2) == "Patio de los Evangelistas":
            court += basin(r, 0.03, 0.6, GRANITE_L, 0.25, 0.35)
    gard = gardens()
    return [("monastery", merge(parts_mesh)), ("facades", merge(fac)), ("portals", merge(portal)),
            ("towers", merge(towers)), ("basilica", merge(church)), ("courtyards", merge(court)),
            ("gardens", merge(gard)), ("galeria", merge(gal)), ("pond", merge(pond())), ("lonja", merge(lonja()))]


def main():
    parts = build()
    out_dir = ROOT / "models" / "Q28471"
    out_dir.mkdir(parents=True, exist_ok=True)
    named, water = [], []
    for name, m in parts:
        wet = np.all(np.abs(m.c[m.f] - np.array(WATER)).max(axis=2) < 1e-6, axis=1)
        if wet.any():
            water.append(Mesh(m.v, m.f[wet], m.c)); m = Mesh(m.v, m.f[~wet], m.c)
        # a little variation in the stone, so the big walls do not look flat
        k = 0.94 + 0.06 * fbm(m.v, 0.05, 7, 2)
        m.c = np.clip(m.c * k[:, None], 0, 1)
        named.append((name, m))
    w = merge(water)
    used = np.unique(w.f)                                      # drop the vertices no water face uses
    remap = np.full(len(w.v), -1); remap[used] = np.arange(len(used))
    named.append(("water", Mesh(w.v[used], remap[w.f], w.c[used]), {"material": "water"}))
    tris, size = write_glb(out_dir / "model.glb", named, roughness=0.88,
                           materials={"water": {"metallic": 0.0, "roughness": 0.12}})
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        mj["footprint"] = [[round(x, 1), round(z, 1)] for x, z in FOOTPRINT]
        meta.write_text(json.dumps(mj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{tris} triangles, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
