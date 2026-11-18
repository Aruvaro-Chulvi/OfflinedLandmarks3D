"""
Arc de Triomphe de l'Étoile (Q64436), Paris — the arch, its sculpture, the Tomb of the Unknown Soldier and the
paved island of the Place Charles-de-Gaulle, built by code.

Author: Álvaro C. Chulvi (with AI assistance). Licence of the resulting model: CC BY 4.0.
The monument: Jean-François Chalgrin (design 1806, died 1811), continued by Louis-Robert Goust (died 1838),
Jean-Arnaud Raymond (died 1811), Jean-Nicolas Huyot (died 1840) and finished by Guillaume Abel Blouet (died 1853)
in 1836; sculpture by François Rude (died 1855), Jean-Pierre Cortot (died 1843), Antoine Étex (died 1888),
James Pradier (died 1852) and others; the Tomb of the Unknown Soldier 1921, the flame 1923 (bronze by Edgar
Brandt, died 1960, after Henri Favier). Public domain.

What is modelled:
- The arch: four pillars of pale Château-Landon limestone, 44.82 × 22.21 m and 49.54 m high; the great arch
  (29.19 m high, 14.62 m wide) on the main axis (Champs-Élysées ↔ Avenue de la Grande-Armée) with its coffered
  vault, archivolt, keystone and the two Fames of Pradier in the spandrels of each face; the two transverse arches
  (18.68 × 8.44 m) through the pillars on the short sides, with their coffers, archivolts and spandrel figures;
  the names of the battles and generals engraved inside the great arch (as lines); the base courses, the impost
  band at the springing of the great arch.
- The four high-relief groups on their pedestals (stylised, ~11.6 m high): on the Champs-Élysées façade
  Le Départ des volontaires de 1792 («La Marseillaise», Rude) on the right (north-north-east pillar) and
  Le Triomphe de 1810 (Cortot) on the left; on the Grande-Armée façade La Résistance de 1814 (Étex, right)
  and La Paix de 1815 (Étex, left). Their positions are confirmed by OpenStreetMap nodes 3462009689–92.
- The six bas-reliefs above them and on the short sides (Aboukir, Funérailles de Marceau, Prise d'Alexandrie,
  Pont d'Arcole, Jemappes, Austerlitz), as framed panels with stylised figures.
- The entablature: architrave, the great frieze (Départ / Retour des armées) as a procession of small figures
  on all four faces, the cornice with its modillions; the attic with its 30 shields (10 on each long face, 5 on
  each short face) separated by upright swords, its cornice and the balustrade of the rooftop terrace with the
  modern safety railing; the raised central part of the roof (OpenStreetMap parts 226520145 and 226520146).
- Under the arch: the Tomb of the Unknown Soldier (granite slab from Vire), the eternal flame in its bronze
  rosette, the chained bollards around the tomb and across the arch.
- The «terre-plein»: the paved island in the middle of the Place Charles-de-Gaulle, out to the kerb (r ≈ 57.4 m),
  with the ring of bronze-chained bollards around the monument («cent plots … symbolisent les Cent-Jours»,
  fr.wikipedia) at r ≈ 40 m, the outer chained ring along the kerb at r ≈ 56 m, and the two stairways with their
  low walls down to the Passage du Souvenir (the pedestrian tunnel). The island belongs to the monument (it is
  its setting, closed to traffic and reached only through the tunnel); the roundabout road around it does not,
  and is left to Offlined's map.
- Not modelled: the roundabout and the avenues, the interiors (stairs, the museum hall in the attic), the
  occasional giant flag under the arch, the wreaths.

Measurements (see meta.json for the sources):
- Plan: OpenStreetMap way 226413508 (cadastre outline, height 50 m) and its building:parts, in the axes of
  heading 115.04° (the outline is square to it within 0.01 m). The cadastre outline is 47.3 × 24.4 m: the
  pillars are 44.82 × 22.21 m (Centre des monuments nationaux, fr.wikipedia), so the cadastre traces the
  projecting cornice (≈ 1.2 m each side), which is how it is modelled. Pedestals of the groups: OSM parts
  226413517–21 (8.2 m wide, 6 m high, projecting ~2.5 m). Transverse arches: OSM notch 8.5 m wide, centred on
  the pillars (8.44 m official).
- Heights: 49.54 m overall, great arch 29.19 × 14.62 m, small arches 18.68 × 8.44 m (Centre des monuments
  nationaux, fr.wikipedia). OSM parts: pillars to 32 m, roof terrace 47 m, central raised part 47–49 m and its
  round top to 50 m. The rest is estimated from photographs (±5 %): base courses to 2.1 m, group pedestals to
  6 m and groups to ~17.6 m, impost band 21.0–21.9 m, bas-reliefs 22.6–28.6 m, architrave 31.2–32.4, frieze
  32.4–34.6 (the frieze is «over two metres high»), cornice 34.6–37.0 (projecting 1.15 m), attic 37.0–44.4 with
  its shields at ~41 m, attic cornice 44.4–45.6, balustrade to 46.9 m, central part 45.6–48.9, top 49.54 m.
- Under the arch: the tomb enclosure (OSM way 339016618, 2.6 × 5.4 m), the flame (way 1222868263), the chains
  across the arch (ways 339016604–05).
- Island: bollards (294 OSM nodes) and chains (OSM ways 339874145–48, 699674342–51, 1356427943) placed exactly
  where OpenStreetMap has them; stair walls ways 339016611/14; kerb just outside the outer chain (the road
  centre-line of the Place, ways 184551793 …, runs at r ≈ 68 m).
- Ground: EU-DEM 25 m (OpenTopoData) gives 65–66 m within 30 m of the arch and 62.4–67 m at the kerb, a gentle
  slope of 2–3 m across the island; the default flatten is kept, the kerb skirt goes down 4 m.
Colours: Château-Landon limestone, cream-white (it does not blacken); the bas-reliefs and frieze in Chérence
limestone (a touch warmer); dark bronze bollards; Vire granite for the tomb; grey granite setts on the island.

Axes (docs/MODEL_SPEC.md): +Y up, the Champs-Élysées façade faces +Z (heading 115.04°, east-south-east), +X
points to 25.04° (north-north-east, the Avenue de Wagram side), origin on the ground at the centre of the arch.

    python model_sources/Q64436_arc_de_triomphe.py   -> models/Q64436/model.glb (+ triangles, footprint in meta.json)
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

rng = np.random.default_rng(1836)
HEADING = 115.04

# ── Colours ──────────────────────────────────────────────────────────────────
STONE = np.array((0.865, 0.835, 0.765))      # Château-Landon limestone
STONE_L = (0.90, 0.875, 0.81)                # mouldings, cornices
STONE_D = (0.74, 0.715, 0.655)               # base courses, foundations, shadowed recesses
RELIEF = (0.89, 0.855, 0.775)                # sculpture (Chérence / Château-Landon)
INCISED = (0.64, 0.61, 0.555)                # engraved names
PAVE = (0.60, 0.585, 0.56)                   # granite setts
PAVE_L = (0.67, 0.655, 0.625)                # the paving close to the arch
PAVE_J = (0.47, 0.46, 0.44)                  # joints and bands of the paving
KERB = (0.55, 0.54, 0.52)
BRONZE = (0.30, 0.25, 0.17)
CHAIN = (0.18, 0.16, 0.13)
GRANITE_V = (0.33, 0.33, 0.34)               # Vire granite
GILT = (0.62, 0.50, 0.25)
FLAME = (1.0, 0.55, 0.12)
FLAME_Y = (1.0, 0.86, 0.35)
DARK = (0.10, 0.10, 0.10)
IRON = (0.16, 0.17, 0.18)
ROOF = (0.80, 0.79, 0.76)                    # roof terrace paving
FOUND = -3.0

# ── Dimensions ───────────────────────────────────────────────────────────────
W2, D2 = 44.82 / 2, 22.21 / 2               # half length (x) and half depth (z) of the pillars
GA = 14.62 / 2                              # half span of the great arch
GA_SPRING = 29.19 - GA                      # 21.88
SA = 8.44 / 2                               # half span of the transverse arches
SA_SPRING = 18.68 - SA                      # 14.46
PC = (GA + W2) / 2                          # centre of a pillar along x: 14.865
Y_ARCH = 31.2                               # top of the arched body (bottom of the architrave)
Y_TERR = 45.6                               # roof terrace

# OpenStreetMap (© OpenStreetMap contributors, ODbL), in model metres: the 294 bollards of the island and the
# chains between them (ways 339874145-48, 699674342-51, 1356427943), in the axes of heading 115.04°.
BOLLARDS = "-1.2,-40.9;-1.3,-39.4;-1.8,-53.2;-3.5,-54.8;-3.2,-41.7;-3.2,-40.8;-5.1,-56.2;-5.5,-41.7;-7.6,-56.1;-5.5,-39.3;-9.6,-55.7;-7.6,-38.9;-11.8,-55.2;-9.7,-38.4;-13.9,-54.7;-16.1,-54.1;-11.8,-37.7;-18.2,-53.4;-13.9,-37.0;-20.1,-52.7;-22.2,-51.9;-16.0,-36.1;-24.1,-50.9;-17.9,-35.2;-26.2,-49.9;-27.9,-48.9;-19.8,-34.2;-29.8,-47.8;-21.6,-33.0;-31.6,-46.5;-23.3,-31.9;-33.4,-45.3;-35.2,-43.9;-25.0,-30.5;-36.8,-42.5;-26.6,-29.0;-38.4,-41.1;-39.9,-39.5;-28.2,-27.4;-41.4,-37.9;-29.6,-25.8;-42.9,-36.2;-44.2,-34.6;-31.1,-24.2;-45.5,-32.8;-32.3,-22.5;-46.8,-31.0;-33.5,-20.7;-48.0,-29.2;-49.1,-27.3;-34.5,-18.9;-50.1,-25.5;-35.5,-17.0;-51.0,-23.5;-52.0,-21.5;-36.4,-15.0;-52.8,-19.5;-37.3,-13.0;-53.5,-17.3;-37.8,-10.8;-54.1,-15.2;-54.7,-13.2;-38.4,-8.8;-55.2,-11.0;-38.8,-6.7;-55.5,-9.0;-55.9,-6.8;-39.2,-4.5;-41.5,-4.6;-54.8,-5.1;-53.5,-3.2;-41.6,-2.0;-40.8,-1.9;-41.6,0.9;-40.8,0.9;-53.6,2.3;-54.8,3.9;-41.6,3.7;-39.3,3.7;-56.3,5.8;-55.9,8.2;-38.9,5.8;-55.5,10.3;-38.5,7.9;-55.0,12.4;-38.1,10.1;-54.6,14.6;-54.0,16.6;-37.4,12.3;-53.2,18.7;-36.8,14.2;-52.4,20.8;-51.6,22.8;-36.0,16.2;-50.6,24.9;-35.0,18.2;-49.6,26.8;-34.0,20.0;-48.5,28.7;-47.4,30.5;-32.8,22.0;-46.1,32.4;-31.6,23.7;-44.8,34.0;-43.5,35.8;-30.3,25.4;-42.1,37.5;-28.9,27.0;-40.6,39.1;-39.0,40.6;-27.3,28.6;-37.4,42.1;-25.9,30.0;-35.8,43.4;-24.1,31.4;-34.1,44.8;-32.3,46.1;-22.3,32.7;-30.5,47.4;-20.5,33.9;-28.7,48.5;-26.7,49.6;-18.7,35.0;-24.8,50.6;-16.7,36.0;-22.9,51.5;-20.9,52.3;-14.7,36.9;-18.8,53.1;-12.7,37.6;-16.7,53.8;-10.7,38.3;-14.7,54.5;-12.6,55.1;-8.6,38.9;-10.5,55.5;-6.4,39.4;-6.5,41.8;-8.3,55.8;-6.2,56.2;-4.1,41.0;-4.1,41.8;-4.4,54.6;-3.0,53.3;-2.1,41.0;-2.1,41.9;1.9,42.0;1.9,41.2;2.6,53.3;4.3,54.8;3.9,41.9;3.8,41.1;6.0,56.3;6.1,41.9;8.2,55.9;6.2,39.4;10.3,55.5;8.4,39.0;12.4,55.0;14.4,54.5;10.4,38.6;16.5,53.8;12.4,38.0;18.7,53.1;14.4,37.2;20.7,52.4;22.7,51.5;16.3,36.3;24.6,50.6;18.2,35.4;26.6,49.6;28.5,48.6;20.2,34.4;30.3,47.4;22.0,33.3;32.1,46.2;23.9,32.1;33.9,44.9;35.6,43.6;25.5,30.7;37.2,42.1;27.0,29.3;38.9,40.6;28.6,27.8;40.3,39.1;41.9,37.5;30.1,26.2;43.4,35.9;31.5,24.6;45.9,32.4;32.8,22.9;47.2,30.6;34.0,21.0;48.3,28.9;49.4,26.9;35.2,19.0;50.3,24.9;36.0,17.2;51.3,23.0;36.8,15.2;52.1,21.0;52.9,19.0;37.7,13.2;53.6,16.9;38.4,11.2;54.2,14.9;54.8,12.9;38.9,9.0;55.2,10.9;39.3,7.0;55.7,9.0;56.1,7.3;39.5,4.8;42.1,4.8;56.2,5.6;56.3,3.8;42.1,2.6;41.2,2.2;54.8,2.8;53.3,1.5;41.2,-0.7;42.1,-0.8;53.4,-2.0;54.9,-2.9;56.4,-3.8;42.0,-3.5;39.7,-3.4;56.3,-5.7;56.1,-7.6;39.5,-5.6;55.7,-9.6;39.1,-7.7;55.3,-11.7;54.7,-13.9;38.6,-9.9;54.1,-16.0;38.0,-11.9;53.5,-18.0;37.3,-13.9;52.7,-20.0;51.9,-22.1;36.5,-16.1;51.0,-24.2;35.6,-18.0;50.0,-26.0;49.1,-28.0;34.5,-19.9;48.0,-29.8;33.3,-21.8;46.8,-31.8;32.2,-23.5;45.6,-33.6;44.3,-35.4;30.8,-25.4;42.9,-37.0;29.5,-27.0;41.4,-38.7;29.0,-27.6;39.9,-40.3;28.0,-28.6;38.4,-41.7;36.9,-43.2;25.9,-30.4;35.1,-44.5;24.7,-31.3;33.2,-45.8;23.0,-32.5;31.4,-47.0;21.1,-33.7;29.6,-48.1;27.7,-49.2;19.3,-34.7;25.7,-50.2;17.3,-35.7;23.8,-51.1;21.8,-52.0;15.2,-36.6;19.7,-52.9;13.2,-37.4;17.7,-53.6;11.2,-38.1;15.6,-54.3;13.6,-54.8;9.2,-38.7;7.0,-39.1;7.2,-41.6;7.0,-56.0;4.8,-40.7;4.8,-41.6;5.3,-54.6;3.6,-53.0;2.7,-40.8;2.6,-39.4;0.3,-39.3"
CHAINS = [
    "5.9,3.5;-5.9,3.8",  # way/339016604
    "5.9,11.6;-5.9,11.7",  # way/339016605
    "1.5,4.9;1.6,10.3;-1.0,10.3;-1.0,4.9;1.5,4.9",  # way/339016618
    "-4.1,41.8;-6.5,41.8;-6.4,39.4;-8.6,38.9;-10.7,38.3;-12.7,37.6;-14.7,36.9;-16.7,36.0;-18.7,35.0;-20.5,33.9;-22.3,32.7;-24.1,31.4;-25.9,30.0;-27.3,28.6;-28.9,27.0;-30.3,25.4;-31.6,23.7;-32.8,22.0;-34.0,20.0;-35.0,18.2;-36.0,16.2;-36.8,14.2;-37.4,12.3;-38.1,10.1;-38.5,7.9;-38.9,5.8;-39.3,3.7;-41.6,3.7;-41.6,0.9",  # way/339874145
    "-41.6,-2.0;-41.5,-4.6;-39.2,-4.5;-38.8,-6.7;-38.4,-8.8;-37.8,-10.8;-37.3,-13.0;-36.4,-15.0;-35.5,-17.0;-34.5,-18.9;-33.5,-20.7;-32.3,-22.5;-31.1,-24.2;-29.6,-25.8;-28.2,-27.4;-26.6,-29.0;-25.0,-30.5;-23.3,-31.9;-21.6,-33.0;-19.8,-34.2;-17.9,-35.2;-16.0,-36.1;-13.9,-37.0;-11.8,-37.7;-9.7,-38.4;-7.6,-38.9;-5.5,-39.3;-5.5,-41.7;-3.2,-41.7",  # way/339874146
    "4.8,-41.6;7.2,-41.6;7.0,-39.1;9.2,-38.7;11.2,-38.1;13.2,-37.4;15.2,-36.6;17.3,-35.7;19.3,-34.7;21.1,-33.7;23.0,-32.5;24.7,-31.3;25.9,-30.4;28.0,-28.6;29.0,-27.6;29.5,-27.0;30.8,-25.4;32.2,-23.5;33.3,-21.8;34.5,-19.9;35.6,-18.0;36.5,-16.1;37.3,-13.9;38.0,-11.9;38.6,-9.9;39.1,-7.7;39.5,-5.6;39.7,-3.4;42.0,-3.5;42.1,-0.8",  # way/339874147
    "39.5,4.8;39.3,7.0;38.9,9.0;38.4,11.2;37.7,13.2;36.8,15.2;36.0,17.2;35.2,19.0;34.0,21.0;32.8,22.9;31.5,24.6;30.1,26.2;28.6,27.8;27.0,29.3;25.5,30.7;23.9,32.1;22.0,33.3;20.2,34.4;18.2,35.4;16.3,36.3;14.4,37.2;12.4,38.0;10.4,38.6;8.4,39.0;6.2,39.4",  # way/339874148
    "-1.2,-40.9;-3.2,-40.8",  # way/699674342
    "2.6,-39.4;0.3,-39.3;-1.3,-39.4",  # way/699674343
    "4.8,-40.7;2.7,-40.8",  # way/699674344
    "-40.8,0.9;-40.8,-1.9",  # way/699674345
    "41.2,2.2;41.2,-0.7",  # way/699674346
    "42.1,2.6;42.1,4.8;39.5,4.8",  # way/699674347
    "-2.1,41.9;1.9,42.0",  # way/699674348
    "-4.1,41.0;-2.1,41.0",  # way/699674349
    "3.9,41.9;6.1,41.9;6.2,39.4",  # way/699674350
    "1.9,41.2;3.8,41.1",  # way/699674351
    "-52.4,20.8;-51.6,22.8;-50.6,24.9;-49.6,26.8;-48.5,28.7;-47.4,30.5;-46.1,32.4;-44.8,34.0;-43.5,35.8;-42.1,37.5;-40.6,39.1;-39.0,40.6;-37.4,42.1;-35.8,43.4;-34.1,44.8;-32.3,46.1;-30.5,47.4;-28.7,48.5;-26.7,49.6;-24.8,50.6;-22.9,51.5;-20.9,52.3;-18.8,53.1;-16.7,53.8;-14.7,54.5;-12.6,55.1;-10.5,55.5;-8.3,55.8;-6.2,56.2;-4.4,54.6;-3.0,53.3;2.6,53.3;4.3,54.8;6.0,56.3;8.2,55.9;10.3,55.5;12.4,55.0;14.4,54.5;16.5,53.8;18.7,53.1;20.7,52.4;22.7,51.5;24.6,50.6;26.6,49.6;28.5,48.6;30.3,47.4;32.1,46.2;33.9,44.9;35.6,43.6;37.2,42.1;38.9,40.6;40.3,39.1;41.9,37.5;43.4,35.9;44.7,34.1;45.9,32.4;47.2,30.6;48.3,28.9;49.4,26.9;50.3,24.9;51.3,23.0;52.1,21.0;52.9,19.0;53.6,16.9;54.2,14.9;54.8,12.9;55.2,10.9;55.7,9.0;56.1,7.3;56.2,5.6;56.3,3.8;54.8,2.8;53.3,1.5;53.4,-2.0;54.9,-2.9;56.4,-3.8;56.3,-5.7;56.1,-7.6;55.7,-9.6;55.3,-11.7;54.7,-13.9;54.1,-16.0;53.5,-18.0;52.7,-20.0;51.9,-22.1;51.0,-24.2;50.0,-26.0;49.1,-28.0;48.0,-29.8;46.8,-31.8;45.6,-33.6;44.3,-35.4;42.9,-37.0;41.4,-38.7;39.9,-40.3;38.4,-41.7;36.9,-43.2;35.1,-44.5;33.2,-45.8;31.4,-47.0;29.6,-48.1;27.7,-49.2;25.7,-50.2;23.8,-51.1;21.8,-52.0;19.7,-52.9;17.7,-53.6;15.6,-54.3;13.6,-54.8;11.3,-55.3;9.3,-55.7;7.0,-56.0;5.3,-54.6;3.6,-53.0;-1.8,-53.2;-3.5,-54.8;-5.1,-56.2;-7.6,-56.1;-9.6,-55.7;-11.8,-55.2;-13.9,-54.7;-16.1,-54.1;-18.2,-53.4;-20.1,-52.7;-22.2,-51.9;-24.1,-50.9;-26.2,-49.9;-27.9,-48.9;-29.8,-47.8;-31.6,-46.5;-33.4,-45.3;-35.2,-43.9;-36.8,-42.5;-38.4,-41.1;-39.9,-39.5;-41.4,-37.9;-42.9,-36.2;-44.2,-34.6;-45.5,-32.8;-46.8,-31.0;-48.0,-29.2;-49.1,-27.3;-50.1,-25.5;-51.0,-23.5;-52.0,-21.5;-52.8,-19.5;-53.5,-17.3;-54.1,-15.2;-54.7,-13.2;-55.2,-11.0;-55.5,-9.0;-55.9,-6.8;-54.8,-5.1;-53.5,-3.2;-53.6,2.3;-54.8,3.9;-56.3,5.8;-55.9,8.2;-55.5,10.3;-55.0,12.4;-54.6,14.6;-54.0,16.6;-53.2,18.7;-52.4,20.8",  # way/1356427943
]


def pts(s):
    return [tuple(map(float, p.split(","))) for p in s.split(";")]


# ── Small helpers ────────────────────────────────────────────────────────────
def B(x0, x1, y0, y1, z0, z1, c):
    return box(x1 - x0, y1 - y0, z1 - z0, c).T((x0 + x1) / 2, y0, (z0 + z1) / 2)


def mirror_x(m):
    n = m.copy(); n.v[:, 0] *= -1; n.f = n.f[:, ::-1]; return n


def swap_xz(m):
    n = m.copy(); n.v = n.v[:, [2, 1, 0]]; n.f = n.f[:, ::-1]; return n


def stone(k=1.0):
    return tuple(np.clip(STONE * k * (1 + rng.normal(0, 0.012)), 0, 1))


def sphere(r, c, seg=8, rings=5):
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, c, seg)


def plate(outline, za, zb, c, back=False):
    """A prism in the x-y plane between z = za and z = zb (za < zb): front face at zb, edges, optional back."""
    P = [tuple(p) for p in outline]
    a = 0.0
    for i in range(len(P)):
        a += P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1]
    if a < 0:
        P = P[::-1]
    p2, tri = triangulate(P)
    V = [(x, y, zb) for x, y in p2]
    F = [tuple(t) for t in tri]
    if back:
        o = len(V); V += [(x, y, za) for x, y in p2]; F += [(o + t[0], o + t[2], o + t[1]) for t in tri]
    for i in range(len(P)):
        (x0, y0), (x1, y1) = P[i], P[(i + 1) % len(P)]
        o = len(V)
        V += [(x0, y0, za), (x1, y1, za), (x1, y1, zb), (x0, y0, zb)]
        F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    return Mesh(V, F, c)


def arc_pts(xc, ys, r, a0, a1, n):
    return [(xc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(a0, a1, n + 1)]


def arch_band(xc, ys, r0, r1, za, zb, c, seg=24):
    """Half annulus (an archivolt) in the x-y plane, from z = za to zb."""
    return plate(arc_pts(xc, ys, r1, 0, math.pi, seg) + arc_pts(xc, ys, r0, math.pi, 0, seg), za, zb, c)


# ── The arches ───────────────────────────────────────────────────────────────
def arch_block(x0, x1, ys, ytop, z0, z1, c, seg=32):
    """The wall over a semicircular arch: front and back faces and the soffit through the wall."""
    r, xc = (x1 - x0) / 2, (x0 + x1) / 2
    arc = [(xc + r * math.cos(a), ys + r * math.sin(a)) for a in np.linspace(math.pi, 0, seg + 1)]
    out = []
    for z, flip in ((z0, True), (z1, False)):
        V = [(x, y, z) for x, y in arc] + [(x0, ytop, z), (x1, ytop, z)]
        TL, TR, h = seg + 1, seg + 2, seg // 2
        F = [(TL, i, i + 1) for i in range(h)] + [(TL, h, TR)] + [(TR, i, i + 1) for i in range(h, seg)]
        m = Mesh(V, F, c)
        if flip:
            m.f = m.f[:, ::-1]
        out.append(m)
    V, F = [], []
    for x, y in arc:
        V += [(x, y, z0), (x, y, z1)]
    for i in range(seg):
        a = 2 * i; F += [(a, a + 3, a + 1), (a, a + 2, a + 3)]
    out.append(Mesh(V, F, c))
    return merge(out)


def coffers(xc, ys, r, z0, z1, na, nz, frame=0.18):
    """Coffers of a barrel vault along z: a darker sunk panel and a rosette in each cell, just inside the soffit."""
    out = []
    da, dz = math.pi / na, (z1 - z0) / nz
    for i in range(na):
        for j in range(nz):
            a0, a1 = i * da, (i + 1) * da
            zc0, zc1 = z0 + j * dz + frame, z0 + (j + 1) * dz - frame
            aa0, aa1 = a0 + frame / r, a1 - frame / r
            rr = r - 0.03
            V, F = [], []
            for a in (aa0, (aa0 + aa1) / 2, aa1):
                V += [(xc + rr * math.cos(a), ys + rr * math.sin(a), zc0), (xc + rr * math.cos(a), ys + rr * math.sin(a), zc1)]
            F = [(0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)]
            out.append(Mesh(V, F, (0.76, 0.735, 0.67)))
            am, zm = (a0 + a1) / 2, (zc0 + zc1) / 2
            s = min(da * r, dz) * 0.18
            ro = sphere(s, STONE_L, 6, 3).S(1, 1, 1)
            ro.T(xc + (r - 0.05) * math.cos(am), ys + (r - 0.05) * math.sin(am), zm)
            out.append(ro)
    return out


def pillars():
    out = []
    half = []
    x0, x1 = GA, W2
    for z0, z1 in ((SA, D2), (-D2, -SA)):
        half.append(B(x0, x1, FOUND, Y_ARCH, z0, z1, stone()))
        # base courses
        half.append(B(x0 - 0.4, x1 + 0.4, FOUND, 0.55, z0 - 0.4 if z0 < 0 else z0 - 0.4, z1 + 0.4, STONE_D))
        half.append(B(x0 - 0.22, x1 + 0.22, 0.55, 1.75, z0 - 0.22, z1 + 0.22, stone(0.97)))
        half.append(B(x0 - 0.1, x1 + 0.1, 1.75, 2.1, z0 - 0.1, z1 + 0.1, STONE_L))
    # the transverse arch through the pillar (built along x with its span on z, then turned)
    ta = arch_block(-SA, SA, SA_SPRING, Y_ARCH, x0, x1, stone(), 24)
    half.append(swap_xz(ta))
    for m in coffers(0.0, SA_SPRING, SA, x0 + 0.3, x1 - 0.3, 7, 9, 0.14):
        half.append(swap_xz(m))
    # impost band at the springing of the great arch, round the whole pillar
    half.append(B(x0 - 0.25, x1 + 0.25, 21.0, GA_SPRING, -D2 - 0.25, D2 + 0.25, STONE_L))
    half.append(B(x0 - 0.12, x1 + 0.12, 20.75, 21.0, -D2 - 0.12, D2 + 0.12, STONE_L))
    # impost band of the transverse arch, on the short face and along the tunnel
    for s in (-1, 1):
        zz0, zz1 = (SA, D2) if s > 0 else (-D2, -SA)
        half.append(B(x1, x1 + 0.18, SA_SPRING - 0.55, SA_SPRING, zz0, zz1, STONE_L))
        zt = SA if s > 0 else -SA
        half.append(B(x0, x1, SA_SPRING - 0.55, SA_SPRING, min(zt, zt - 0.15 * s), max(zt, zt - 0.15 * s), STONE_L))
    out += half + [mirror_x(m) for m in half]
    # the great arch between the pillars
    out.append(arch_block(-GA, GA, GA_SPRING, Y_ARCH, -D2, D2, stone(), 40))
    out += coffers(0.0, GA_SPRING, GA, -D2 + 0.4, D2 - 0.4, 9, 13, 0.2)
    out.append(B(-GA - 0.25, GA + 0.25, FOUND, 0.0, -D2, D2, STONE_D))      # foundation under the passage
    return out


# ── Figures (stylised sculpture) ─────────────────────────────────────────────
def person(h, c, stride=0.5, hands=None, robe=False, wings=0.0, helmet=False, shield=False, weapon=None,
           lean=0.0, turn=0.0, seg=7):
    """A standing figure at the origin facing +z, h tall. hands: two (dx, dy, dz) in units of h."""
    out = []
    if robe:
        out.append(lathe([(0.16 * h, 0), (0.15 * h, 0.2 * h), (0.12 * h, 0.46 * h), (0.11 * h, 0.56 * h)], c, seg))
    else:
        for s in (-1, 1):
            out.append(tube([(s * 0.06 * h, 0.5 * h, 0), (s * 0.075 * h, 0.26 * h, s * stride * 0.11 * h),
                             (s * 0.08 * h, 0.03 * h, s * stride * 0.2 * h)], 0.042 * h, c, 5))
    out.append(lathe([(0.10 * h, 0.46 * h), (0.12 * h, 0.56 * h), (0.13 * h, 0.70 * h), (0.10 * h, 0.78 * h),
                      (0.045 * h, 0.82 * h), (0.0, 0.83 * h)], c, seg))
    out.append(sphere(0.062 * h, c, 6, 4).T(0, 0.89 * h, 0))
    if helmet:
        out.append(cyl(0.07 * h, 0.0, 0.11 * h, c, 6).T(0, 0.92 * h, 0))
    hands = hands or [(0.17, 0.48, 0.03), (-0.17, 0.48, 0.03)]
    tips = []
    for s, (dx, dy, dz) in zip((1, -1), hands):
        sh = np.array((s * 0.13 * h, 0.77 * h, 0.0)); hd = np.array((dx * h, dy * h, dz * h))
        el = (sh + hd) / 2 + np.array((s * 0.04 * h, -0.03 * h, -0.02 * h))
        out.append(tube([sh, el, hd], 0.032 * h, c, 5))
        tips.append(hd)
    if wings:
        for s in (-1, 1):
            w = plate([(0, 0), (0.42 * h, 0.2 * h), (0.55 * h, 0.55 * h), (0.25 * h, 0.35 * h)], -0.015 * h, 0.015 * h, c, back=True)
            w.R(0, s * (0.6 - 0.4 * wings), 0)
            if s < 0:
                w.v[:, 0] *= -1; w.f = w.f[:, ::-1]
            out.append(w.T(s * 0.05 * h, 0.7 * h, -0.08 * h))
    if shield:
        out.append(cyl(0.13 * h, 0.13 * h, 0.03 * h, c, 9).R(math.pi / 2, 0, 0).T(tips[1][0], tips[1][1], tips[1][2] + 0.04 * h))
    if weapon == "sword":
        d = tips[0] - np.array((0.13 * h, 0.77 * h, 0)); d = d / (np.linalg.norm(d) or 1)
        out.append(tube([tips[0], tips[0] + d * 0.42 * h], 0.012 * h, c, 4))
    elif weapon == "spear":
        out.append(tube([tips[0] - np.array((0, 0.45 * h, 0)), tips[0] + np.array((0, 0.6 * h, 0))], 0.012 * h, c, 4))
    elif weapon == "trumpet":
        d = tips[0] - np.array((0.13 * h, 0.77 * h, 0)); d = d / (np.linalg.norm(d) or 1)
        out.append(cyl(0.008 * h, 0.04 * h, 0.35 * h, c, 6).R(0, 0, -math.atan2(d[0], d[1])).T(*tips[0]))
    m = merge(out)
    return m.R(0, turn, 0).R(0, 0, lean)


def horse(h, c, rear=0.0):
    """A horse in profile along x (h = height at the withers)."""
    out = [sphere(0.5 * h, c, 8, 5).S(1.45, 0.62, 0.55).T(0, 0.95 * h, 0)]
    out.append(tube([(0.55 * h, 1.05 * h, 0), (0.8 * h, 1.45 * h, 0), (0.95 * h, 1.4 * h, 0)], 0.16 * h, c, 6))
    for sx in (-0.5, 0.5):
        for sz in (-0.15, 0.15):
            lift = rear if sx > 0 else 0.0
            out.append(tube([(sx * h, 0.8 * h, sz * h), (sx * h + lift * 0.3 * h, (0.35 + lift * 0.4) * h, sz * h),
                             (sx * h + lift * 0.45 * h, (0.0 + lift * 0.6) * h, sz * h)], 0.07 * h, c, 5))
    out.append(tube([(-0.7 * h, 1.0 * h, 0), (-0.95 * h, 0.55 * h, 0)], 0.08 * h, c, 5))
    m = merge(out)
    if rear:
        m.R(0, 0, rear * 0.5)
    return m


def relief_squash(ms, D, k):
    """Press free-standing figures against the wall at z = D (high relief)."""
    out = []
    for m in ms:
        m = m.copy(); m.v[:, 2] = D + (m.v[:, 2] - D) * k; out.append(m)
    return out


# ── The four high-relief groups ──────────────────────────────────────────────
def pedestal(cx, D):
    out = [B(cx - 4.1, cx + 4.1, FOUND, 5.4, D, D + 2.45, stone(0.99)),
           B(cx - 4.3, cx + 4.3, 0.0, 0.6, D, D + 2.65, STONE_D),
           B(cx - 4.2, cx + 4.2, 0.6, 1.0, D, D + 2.55, STONE_L),
           B(cx - 4.3, cx + 4.3, 5.4, 6.0, D, D + 2.65, STONE_L),
           B(cx - 2.6, cx + 2.6, 2.1, 4.5, D + 2.45, D + 2.52, stone(1.03))]          # the inscription tablet
    for k in range(5):
        out.append(B(cx - 2.0 + 0.4 * (k % 2), cx + 2.0 - 0.4 * (k % 2), 4.1 - k * 0.4, 4.2 - k * 0.4, D + 2.52, D + 2.55, INCISED))
    return out


def group(kind, cx, D):
    """Stylised high relief, ~11.6 m tall on its 6 m pedestal, against the wall z = D of a +z face."""
    y0, zf = 6.0, D + 1.2
    c = RELIEF
    figs = []
    rr = np.random.default_rng({"depart": 1, "triomphe": 2, "resistance": 3, "paix": 4}[kind])
    if kind == "depart":            # the Genius of Liberty over the volunteers
        figs.append(person(5.4, c, robe=True, wings=1.0, hands=[(0.55, 0.9, 0.2), (-0.3, 0.95, 0.0)], weapon="sword",
                           lean=0.35).T(cx - 0.3, y0 + 6.6, zf))
        figs.append(person(5.6, c, stride=0.9, hands=[(0.12, 1.02, 0.05), (-0.2, 0.5, 0.1)], helmet=True, shield=True,
                           weapon="sword").T(cx - 0.4, y0, zf + 0.3))
        figs.append(person(5.0, c, stride=0.6, hands=[(0.2, 0.55, 0.1), (-0.18, 0.6, 0.2)]).T(cx - 2.2, y0, zf + 0.5))
        figs.append(person(5.3, c, stride=0.8, helmet=True, weapon="spear").T(cx + 1.7, y0, zf - 0.2))
        figs.append(person(4.2, c, stride=0.4, hands=[(0.25, 0.3, 0.2), (-0.15, 0.45, 0.2)]).R(0, 0, 0.5).T(cx + 3.0, y0 + 0.3, zf + 0.4))
        figs.append(person(5.2, c, robe=True, shield=True, helmet=True).T(cx - 3.3, y0, zf - 0.3))
    elif kind == "triomphe":        # Napoleon crowned by Victory, Fame above, a vanquished city kneeling
        figs.append(person(5.8, c, robe=True, hands=[(0.2, 0.55, 0.1), (-0.18, 0.6, 0.1)]).T(cx + 0.3, y0, zf + 0.3))
        figs.append(person(4.6, c, robe=True, wings=0.6, hands=[(0.5, 0.65, 0.2), (-0.25, 0.8, 0.1)],
                           lean=0.25).T(cx - 1.2, y0 + 5.2, zf + 0.1))
        figs.append(person(4.4, c, robe=True, wings=1.0, hands=[(0.45, 0.95, 0.1), (-0.3, 0.7, 0.0)], weapon="trumpet",
                           lean=-0.5).T(cx + 1.8, y0 + 7.0, zf - 0.2))
        figs.append(person(3.6, c, robe=True, hands=[(0.25, 0.75, 0.25), (-0.1, 0.4, 0.2)]).R(0, 0, -0.35).T(cx - 2.8, y0, zf + 0.3))
        figs.append(person(4.8, c, stride=0.5, helmet=True, weapon="spear").T(cx + 3.1, y0, zf - 0.3))
        figs.append(person(3.2, c, robe=True).R(0, 0, 0.6).T(cx - 3.6, y0 + 0.2, zf))
    elif kind == "resistance":      # a young man defends his family; a horseman falls; the Genius of the Future
        figs.append(person(5.7, c, stride=1.0, hands=[(0.35, 0.95, 0.1), (-0.25, 0.55, 0.1)], weapon="sword").T(cx + 0.2, y0, zf + 0.3))
        figs.append(person(4.8, c, robe=True, wings=1.0, hands=[(0.4, 0.9, 0.1), (-0.4, 0.9, 0.1)], lean=0.1).T(cx - 0.5, y0 + 6.8, zf))
        figs.append(horse(2.6, c, rear=0.6).T(cx - 2.2, y0 + 0.2, zf - 0.3))
        figs.append(person(4.6, c, stride=0.6, helmet=True, lean=0.8).T(cx - 2.6, y0 + 1.2, zf + 0.2))
        figs.append(person(4.8, c, robe=True, hands=[(0.3, 0.6, 0.3), (-0.1, 0.62, 0.3)]).T(cx + 2.9, y0, zf - 0.2))
        figs.append(person(2.2, c, robe=True).T(cx + 2.4, y0, zf + 0.6))
    else:                           # "paix": the warrior sheathes his sword, the farmer, the mother; Minerva above
        figs.append(person(5.7, c, stride=0.4, hands=[(0.15, 0.48, 0.15), (-0.12, 0.5, 0.2)], helmet=True,
                           weapon="sword").T(cx - 0.4, y0, zf + 0.3))
        figs.append(person(4.8, c, robe=True, helmet=True, shield=True, weapon="spear", lean=0.05).T(cx + 0.6, y0 + 6.5, zf - 0.1))
        figs.append(horse(2.2, c).T(cx + 2.6, y0, zf - 0.4))
        figs.append(person(4.4, c, stride=0.6, hands=[(0.3, 0.4, 0.3), (-0.2, 0.45, 0.25)], lean=-0.3).T(cx + 1.4, y0, zf + 0.5))
        figs.append(person(3.4, c, robe=True).R(0, 0, 0.15).T(cx - 2.9, y0, zf + 0.2))
        figs.append(person(2.0, c, robe=True).T(cx - 2.2, y0, zf + 0.6))
    for m in figs:                                   # a little irregularity of carving
        m.v += rr.normal(0, 0.02, m.v.shape)
    figs.append(B(cx - 3.9, cx + 3.9, y0, y0 + 0.5, D, D + 2.2, RELIEF))                # the rocky ground
    figs.append(plate([(cx - 3.9, y0), (cx + 3.9, y0), (cx + 3.6, y0 + 9.0), (cx + 1.5, y0 + 11.6), (cx - 1.5, y0 + 11.6),
                       (cx - 3.6, y0 + 9.0)], D, D + 0.35, stone(1.01)))                  # the relief ground
    return relief_squash(figs, D, 0.75) + pedestal(cx, D)


# ── Bas-reliefs, spandrels, archivolts (built on a +z face at z = D, then turned) ──
def bas_relief(uc, w, y0, y1, D, seed):
    rr = np.random.default_rng(seed)
    out = [B(uc - w / 2 - 0.35, uc + w / 2 + 0.35, y0 - 0.35, y0, D, D + 0.32, STONE_L),
           B(uc - w / 2 - 0.35, uc + w / 2 + 0.35, y1, y1 + 0.35, D, D + 0.32, STONE_L),
           B(uc - w / 2 - 0.35, uc - w / 2, y0, y1, D, D + 0.32, STONE_L),
           B(uc + w / 2, uc + w / 2 + 0.35, y0, y1, D, D + 0.32, STONE_L),
           B(uc - w / 2, uc + w / 2, y0, y0 + 0.3, D, D + 0.25, RELIEF),
           quad_xy(uc, (y0 + y1) / 2, D + 0.01, w, y1 - y0, (0.80, 0.77, 0.70))]
    n = int(w / 1.15)
    figs = []
    for i in range(n):
        u = uc - w / 2 + (i + 0.5) * w / n + rr.normal(0, 0.12)
        h = (y1 - y0 - 0.4) * rr.uniform(0.62, 0.8)
        if rr.random() < 0.18:
            figs.append(horse(h * 0.42, RELIEF, rear=rr.uniform(0, 0.5)).T(u, y0 + 0.3, D + 0.35))
        else:
            figs.append(person(h, RELIEF, stride=rr.uniform(0.2, 1.0), robe=rr.random() < 0.3, helmet=rr.random() < 0.5,
                               weapon=rr.choice(["sword", "spear", None]), lean=rr.normal(0, 0.15)).T(u, y0 + 0.3, D + 0.35))
    return out + relief_squash(figs, D, 0.35)


def quad_xy(cx, cy, z, w, h, c):
    return Mesh([(cx - w / 2, cy - h / 2, z), (cx + w / 2, cy - h / 2, z), (cx + w / 2, cy + h / 2, z), (cx - w / 2, cy + h / 2, z)],
                [(0, 1, 2), (0, 2, 3)], c)


def fame(u, y, D, ang, h=5.0):
    """A flying Fame with a trumpet, lying along the extrados of the arch, feet outwards, head to the crown."""
    m = person(h, RELIEF, robe=True, hands=[(-0.04, 1.08, 0.12), (0.1, 0.72, 0.12)], weapon="trumpet")
    wings = [plate([(0.0, 0.74 * h), (0.42 * h, 0.98 * h), (0.5 * h, 0.7 * h), (0.2 * h, 0.5 * h)], -0.1, -0.04, RELIEF, back=True),
             plate([(0.0, 0.7 * h), (0.3 * h, 1.05 * h), (0.42 * h, 0.8 * h), (0.16 * h, 0.52 * h)], -0.22, -0.16, RELIEF, back=True)]
    m = merge([m] + wings)                      # both wings spread away from the arch (local +x)
    if u < 0:
        m.v[:, 0] *= -1; m.f = m.f[:, ::-1]
    m.R(0, 0, ang).T(u, y, D + 0.3)
    return relief_squash([m], D, 0.35)


def arch_face(D, span, spring, big):
    """Archivolt, keystone and spandrel figures of an arch on a +z face at z = D."""
    out = [arch_band(0.0, spring, span, span + (1.0 if big else 0.7), D, D + 0.2, STONE_L, 32 if big else 24),
           arch_band(0.0, spring, span + (1.0 if big else 0.7), span + (1.25 if big else 0.85), D, D + 0.32, STONE_L, 32 if big else 24)]
    top = spring + span
    k = 0.75 if big else 0.5
    out.append(plate([(-k, top - 0.3), (k, top - 0.3), (k * 1.35, top + (1.9 if big else 1.2)), (-k * 1.35, top + (1.9 if big else 1.2))],
                     D, D + (0.55 if big else 0.4), STONE_L))
    if big:
        out += fame(-9.1, 25.9, D, -0.86, 5.6) + fame(9.1, 25.9, D, 0.86, 5.6)          # Pradier's Fames
    else:
        out += fame(-6.9, spring + 1.0, D, -0.8, 4.4) + fame(6.9, spring + 1.0, D, 0.8, 4.4)
    return out


def engraved_names(D, u0, u1, y0, y1):
    """Columns of engraved names (thin dark lines) on a face at z = D."""
    out = []
    ncol = max(1, int((u1 - u0) / 1.7))
    cw = (u1 - u0) / ncol
    for i in range(ncol):
        y = y1 - 0.4
        while y > y0:
            L = cw * rng.uniform(0.45, 0.85)
            out.append(quad_xy(u0 + (i + 0.5) * cw, y, D + 0.012, L, 0.08, INCISED))
            y -= 0.32
        out.append(B(u0 + i * cw - 0.02, u0 + i * cw + 0.02, y0, y1, D, D + 0.02, INCISED))
    return out


def face_main(D, groups, seeds):
    """Decoration of a long face (built facing +z)."""
    out = []
    for cx, kind in groups:
        out += group(kind, cx, D)
    out += arch_face(D, GA, GA_SPRING, True)
    for (cx, _), sd in zip(groups, seeds):
        out += bas_relief(cx, 10.4, 22.75, 28.4, D, sd)
    return out


def face_short(D, seed):
    """Decoration of a short face (built facing +z, the arch centred on u = 0)."""
    out = arch_face(D, SA, SA_SPRING, False)
    out += bas_relief(0.0, 16.6, 22.75, 28.4, D, seed)
    return out


def face_inner(D):
    """Inner face of a pillar under the great arch (built facing +z at z = D): names and the small arch."""
    out = arch_face(D, SA, SA_SPRING, False)[:3]
    for s in (-1, 1):
        u0, u1 = (SA + 0.6, D2 - 0.6) if s > 0 else (-D2 + 0.6, -SA - 0.6)
        out += engraved_names(D, u0, u1, 2.6, 13.6)
    out += engraved_names(D, -D2 + 0.8, D2 - 0.8, 19.4, 20.5)
    return out


def turn(ms, ry):
    out = []
    for m in ms:
        m = m.copy().R(0, ry, 0); out.append(m)
    return out


# ── Entablature, attic, roof ─────────────────────────────────────────────────
def rect_box(y0, y1, e, c):
    return B(-W2 - e, W2 + e, y0, y1, -D2 - e, D2 + e, c)


def frieze_figures(D, half_len, seed):
    rr = np.random.default_rng(seed)
    out = []
    u = -half_len + 0.5
    while u < half_len - 0.5:
        h = rr.uniform(1.55, 1.85)
        if rr.random() < 0.12:
            m = horse(h * 0.45, RELIEF, rear=rr.uniform(0, 0.4))
            out.append(m.T(u, 32.55, D + 0.15)); u += 1.25; continue
        prof = [(0.12 * h, 0), (0.11 * h, 0.45 * h), (0.13 * h, 0.7 * h), (0.07 * h, 0.82 * h), (0.0, 0.84 * h)]
        m = merge([lathe(prof, RELIEF, 5), sphere(0.065 * h, RELIEF, 5, 3).T(0, 0.9 * h, 0)])
        m.R(0, 0, rr.normal(0, 0.12)).T(u, 32.55, D + 0.15)
        out.append(m)
        if rr.random() < 0.35:
            out.append(tube([(u + 0.1, 32.55 + 0.6 * h, D + 0.2), (u + 0.25, 32.55 + 1.15 * h, D + 0.2)], 0.025, RELIEF, 3))
        u += rr.uniform(0.6, 0.85)
    return relief_squash(out, D, 0.45)


def shield_sword(u, y, D, sword=False):
    if sword:
        return [B(u - 0.06, u + 0.06, y - 1.9, y + 1.2, D, D + 0.12, STONE_L),
                B(u - 0.38, u + 0.38, y + 1.2, y + 1.36, D, D + 0.16, STONE_L),
                B(u - 0.08, u + 0.08, y + 1.36, y + 1.9, D, D + 0.16, STONE_L),
                sphere(0.15, STONE_L, 6, 3).T(u, y + 2.0, D + 0.1)]
    out = [cyl(1.02, 1.02, 0.22, STONE_L, 16).R(math.pi / 2, 0, 0).T(u, y, D + 0.22),
           cyl(0.85, 0.85, 0.08, stone(1.02), 16).R(math.pi / 2, 0, 0).T(u, y, D + 0.3)]
    for k in (-1, 0, 1):                              # the engraved battle name
        out.append(quad_xy(u, y + k * 0.2, D + 0.31, 1.1 - 0.35 * abs(k), 0.08, INCISED))
    return out


def attic_face(D, half_len, n):
    out = []
    sp = 2 * half_len / n
    for i in range(n):
        u = -half_len + (i + 0.5) * sp
        out += shield_sword(u, 41.2, D)
        if i:
            out += shield_sword(u - sp / 2, 41.2, D, sword=True)
    out.append(B(-half_len + 0.3, half_len - 0.3, 38.55, 38.75, D, D + 0.08, STONE_L))
    return out


def modillions(D, half_len):
    out = []
    u = -half_len + 0.45
    while u < half_len - 0.3:
        out.append(B(u - 0.16, u + 0.16, 35.55, 36.0, D + 0.25, D + 1.05, STONE_L))
        u += 0.9
    return out


def balusters(ax, half_len, D, y0, y1):
    """One side of the balustrade, along x at z = D (built facing +z)."""
    out = [B(-half_len, half_len, y0, y0 + 0.2, D - 0.42, D, STONE_L),
           B(-half_len, half_len, y1 - 0.25, y1, D - 0.46, D + 0.04, STONE_L)]
    h = y1 - y0 - 0.45
    prof = [(0.12, 0), (0.08, 0.12 * h), (0.14, 0.42 * h), (0.07, 0.8 * h), (0.11, h)]
    u = -half_len + 0.3
    k = 0
    while u < half_len - 0.2:
        if k % 9 == 0:
            out.append(B(u - 0.22, u + 0.22, y0 + 0.2, y1 - 0.25, D - 0.44, D + 0.02, stone(1.02)))
        else:
            out.append(lathe(prof, STONE_L, 4).T(u, y0 + 0.2, D - 0.21))
        u += 0.42; k += 1
    return out


def roof():
    out = []
    for D, hl in ((D2, W2), (W2, D2)):
        for ry in ((0.0, math.pi) if D == D2 else (math.pi / 2, -math.pi / 2)):
            out += turn(balusters(0, hl - 0.45, D, Y_TERR, Y_TERR + 1.3), ry)
    # modern safety railing set back behind the balustrade
    rail = [(-W2 + 1.0, -D2 + 1.0), (W2 - 1.0, -D2 + 1.0), (W2 - 1.0, D2 - 1.0), (-W2 + 1.0, D2 - 1.0), (-W2 + 1.0, -D2 + 1.0)]
    for y in (Y_TERR + 1.15, Y_TERR + 2.2):
        out.append(tube([(x, y, z) for x, z in rail], 0.035, IRON, 4))
    for i in range(4):
        (xa, za), (xb, zb) = rail[i], rail[i + 1]
        n = int(math.hypot(xb - xa, zb - za) / 2.2)
        for k in range(n):
            t = k / n
            out.append(B(xa + (xb - xa) * t - 0.03, xa + (xb - xa) * t + 0.03, Y_TERR, Y_TERR + 2.2,
                         za + (zb - za) * t - 0.03, za + (zb - za) * t + 0.03, IRON))
    # the paved terrace with its joints
    out.append(cap([(-W2 + 0.3, -D2 + 0.3), (W2 - 0.3, -D2 + 0.3), (W2 - 0.3, D2 - 0.3), (-W2 + 0.3, D2 - 0.3)], Y_TERR + 0.02, ROOF))
    # the raised central part (OSM way 226520145) and its round top (way 226520146)
    H = [(11.24, -6.78), (11.25, -4.46), (8.63, -4.47), (8.63, 4.43), (11.13, 4.44), (11.13, 6.75), (-11.42, 6.69),
         (-11.42, 4.29), (-8.9, 4.3), (-8.9, -4.7), (-11.28, -4.7), (-11.28, -6.84)]
    out.append(walls(H, Y_TERR, 48.6, stone(0.98)))
    out.append(walls(H, 48.6, 48.9, STONE_L))
    out.append(cap(H, 48.9, ROOF))
    T = [(-3.31, -6.13), (3.06, -6.07), (3.97, -5.83), (4.96, -5.35), (5.8, -4.8), (6.43, -4.18), (7.09, -3.44), (7.5, -2.6),
         (7.89, -1.62), (8.13, -0.6), (8.13, 0.29), (7.97, 1.51), (7.63, 2.51), (7.04, 3.53), (6.22, 4.48), (5.56, 5.05),
         (4.76, 5.5), (4.01, 5.81), (3.1, 6.03), (-3.12, 6.12), (-4.05, 5.88), (-5.08, 5.49), (-5.83, 4.95), (-6.55, 4.31),
         (-7.28, 3.47), (-7.86, 2.44), (-8.2, 1.39), (-8.31, 0.16), (-8.19, -1.21), (-7.86, -2.15), (-7.35, -3.15),
         (-6.91, -3.94), (-6.13, -4.79), (-5.46, -5.23), (-4.79, -5.62), (-4.14, -5.91)]
    T = [(x * 0.97, z * 0.97) for x, z in T]
    V = [(x, 48.92, z) for x, z in T] + [(0.0, 49.54, 0.0)]
    n = len(T)
    a = sum(T[i][0] * T[(i + 1) % n][1] - T[(i + 1) % n][0] * T[i][1] for i in range(n))
    F = [(i, n, (i + 1) % n) if a > 0 else (i, (i + 1) % n, n) for i in range(n)]
    out.append(Mesh(V, F, (0.72, 0.73, 0.74)))
    # railing round the raised part
    out.append(tube([(x * 1.0, 49.95, z * 1.0) for x, z in H + [H[0]]], 0.03, IRON, 4))
    for x, z in H:
        out.append(B(x - 0.03, x + 0.03, 48.9, 49.95, z - 0.03, z + 0.03, IRON))
    # steps from the terrace up to the raised part, on the two long sides
    for s in (-1, 1):
        for k in range(10):
            zz = s * (6.8 + 2.9 - k * 0.29)
            out.append(B(-1.6, 1.6, Y_TERR, Y_TERR + 0.33 * (k + 1), min(zz, zz - 0.29 * s), max(zz, zz - 0.29 * s), stone(1.0)))
    return out


def upper_body():
    out = []
    out.append(rect_box(Y_ARCH, 32.4, 0.0, stone()))                         # architrave
    out.append(rect_box(31.75, 31.95, 0.1, STONE_L))
    out.append(rect_box(32.25, 32.4, 0.16, STONE_L))
    out.append(rect_box(32.4, 34.6, -0.05, stone(0.99)))                     # frieze
    out.append(rect_box(34.6, 35.0, 0.2, STONE_L))                           # cornice
    out.append(rect_box(35.0, 35.55, 0.4, STONE_L))
    out.append(rect_box(35.55, 36.0, 0.25, stone(0.96)))
    out.append(rect_box(36.0, 36.65, 1.15, STONE_L))
    out.append(rect_box(36.65, 37.0, 1.0, STONE_L))
    out.append(rect_box(37.0, 38.0, 0.05, stone(0.98)))                      # attic plinth
    out.append(rect_box(38.0, 44.4, -0.12, stone()))                         # attic
    out.append(rect_box(44.4, 44.85, 0.15, STONE_L))                         # attic cornice
    out.append(rect_box(44.85, 45.2, 0.38, STONE_L))
    out.append(rect_box(45.2, Y_TERR, 0.6, STONE_L))
    for D, hl, ry, n, sd in ((D2, W2, 0.0, 10, 11), (D2, W2, math.pi, 10, 12), (W2, D2, math.pi / 2, 5, 13), (W2, D2, -math.pi / 2, 5, 14)):
        dec = frieze_figures(D - 0.05, hl - 0.05, sd) + attic_face(D - 0.12, hl - 0.12, n) + modillions(D, hl)
        out += turn(dec, ry)
    return out


# ── Under the arch: the Tomb of the Unknown Soldier and the flame ────────────
def bollard(x, z, c=BRONZE, h=0.8):
    prof = [(0.15, 0), (0.13, 0.08), (0.1, 0.5 * h / 0.8), (0.125, 0.56 * h / 0.8), (0.09, 0.68 * h / 0.8), (0.0, h)]
    return lathe(prof, c, 5).T(x, 0.03, z)


def chain(p, q, y=0.62, sag=0.22):
    a, b = np.array((p[0], y, p[1])), np.array((q[0], y, q[1]))
    L = float(np.linalg.norm(b - a))
    if L < 0.05:
        return None
    s = min(sag, 0.12 * L)
    P = [a + (b - a) * t - np.array((0, s * 4 * t * (1 - t), 0)) for t in (0, 0.25, 0.5, 0.75, 1)]
    return tube(P, 0.028, CHAIN, 3)


def tomb():
    out = []
    out.append(B(-0.38, 0.9, 0.0, 0.08, 6.25, 9.55, GRANITE_V))                 # the slab
    for k in range(7):                                                          # ICI REPOSE UN SOLDAT FRANÇAIS …
        w = (0.9, 0.8, 1.0, 0.95, 0.5, 0.95, 0.6)[k]
        out.append(Mesh([(0.26 - w / 2, 0.085, 7.2 + k * 0.26), (0.26 + w / 2, 0.085, 7.2 + k * 0.26),
                         (0.26 + w / 2, 0.085, 7.3 + k * 0.26), (0.26 - w / 2, 0.085, 7.3 + k * 0.26)], [(0, 2, 1), (0, 3, 2)], GILT))
    # the flame: a bronze rosette (star on a sword hilt) with the flame rising from its centre
    fx, fz = 0.2, 5.62
    out.append(cyl(0.48, 0.42, 0.1, BRONZE, 14).T(fx, 0.0, fz))
    for k in range(8):
        a = k * math.pi / 4
        out.append(plate([(0, 0), (0.12, 0.62), (-0.12, 0.62)], -0.03, 0.03, BRONZE, back=True).R(math.pi / 2, 0, 0)
                   .R(0, a, 0).T(fx, 0.1, fz))
    out.append(cyl(0.18, 0.02, 0.85, FLAME, 8).T(fx, 0.1, fz))
    out.append(cyl(0.1, 0.0, 0.55, FLAME_Y, 6).T(fx + 0.04, 0.12, fz + 0.03))
    for dx, dz, hh in ((0.12, -0.05, 0.5), (-0.1, 0.07, 0.45)):
        out.append(cyl(0.08, 0.0, hh, FLAME, 6).R(0, 0, dx).T(fx + dx, 0.1, fz + dz))
    # enclosure (OSM way 339016618): bronze posts with chains
    R = [(1.55, 4.88), (1.57, 10.33), (-1.02, 10.33), (-1.03, 4.88)]
    posts = []
    for i in range(4):
        a, b = R[i], R[(i + 1) % 4]
        n = max(1, int(round(math.dist(a, b) / 1.35)))
        posts += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n)]
    for i, p in enumerate(posts):
        out.append(bollard(*p, h=0.65))
        out.append(chain(p, posts[(i + 1) % len(posts)], 0.5, 0.15))
    # chains across the arch on both sides of the tomb (ways 339016604-05)
    for z in (3.62, 11.65):
        P = [(-5.9 + 11.8 * k / 8, z) for k in range(9)]
        for i, p in enumerate(P):
            out.append(bollard(*p, h=0.75))
            if i:
                out.append(chain(P[i - 1], p, 0.58, 0.2))
    return [m for m in out if m is not None]


# ── The island (terre-plein) ─────────────────────────────────────────────────
R_ISLAND = 57.4
STAIRS = [(26.68, 29.23, 6.1, 13.3, +1), (26.83, 29.4, -4.3, -11.5, -1)]       # OSM ways 339016611 / 339016614


def island():
    out, bronze = [], []
    n = 128
    circ = [(R_ISLAND * math.cos(2 * math.pi * k / n), R_ISLAND * math.sin(2 * math.pi * k / n)) for k in range(n)]
    inner = [(44.8 * math.cos(2 * math.pi * k / n), 44.8 * math.sin(2 * math.pi * k / n)) for k in range(n)]
    holes = [[(x0, min(z0, z1)), (x1, min(z0, z1)), (x1, max(z0, z1)), (x0, max(z0, z1))] for x0, x1, z0, z1, _ in STAIRS]
    out.append(cap(inner, 0.04, PAVE_L, holes=holes))
    # the outer ring of paving (an annulus, as quads)
    V, F = [], []
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        o = len(V)
        V += [(44.8 * math.cos(a0), 0.04, 44.8 * math.sin(a0)), (R_ISLAND * math.cos(a0), 0.04, R_ISLAND * math.sin(a0)),
              (R_ISLAND * math.cos(a1), 0.04, R_ISLAND * math.sin(a1)), (44.8 * math.cos(a1), 0.04, 44.8 * math.sin(a1))]
        F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    out.append(Mesh(V, F, PAVE))
    # bands and radial joints of the paving
    for r in (44.8, 50.5):
        V, F = [], []
        for k in range(n):
            a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
            o = len(V)
            V += [((r - 0.3) * math.cos(a0), 0.05, (r - 0.3) * math.sin(a0)), ((r + 0.3) * math.cos(a0), 0.05, (r + 0.3) * math.sin(a0)),
                  ((r + 0.3) * math.cos(a1), 0.05, (r + 0.3) * math.sin(a1)), ((r - 0.3) * math.cos(a1), 0.05, (r - 0.3) * math.sin(a1))]
            F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
        out.append(Mesh(V, F, PAVE_J))
    for k in range(48):
        a = 2 * math.pi * (k + 0.5) / 48
        ca, sa = math.cos(a), math.sin(a)
        nx, nz = -sa * 0.09, ca * 0.09
        r0, r1 = 45.2, R_ISLAND - 0.5
        out.append(Mesh([(r0 * ca - nx, 0.05, r0 * sa - nz), (r1 * ca - nx, 0.05, r1 * sa - nz), (r1 * ca + nx, 0.05, r1 * sa + nz),
                         (r0 * ca + nx, 0.05, r0 * sa + nz)], [(0, 2, 1), (0, 3, 2)], PAVE_J))
    # the paved band round the foot of the arch
    out.append(cap([(-W2 - 3.0, -D2 - 4.2), (W2 + 3.0, -D2 - 4.2), (W2 + 3.0, D2 + 4.2), (-W2 - 3.0, D2 + 4.2)], 0.05,
                   (0.70, 0.685, 0.65), holes=[[(-W2 - 0.45, -D2 - 0.45), (W2 + 0.45, -D2 - 0.45), (W2 + 0.45, D2 + 0.45), (-W2 - 0.45, D2 + 0.45)]]))
    # kerb and skirt down to the foundations
    out.append(walls(circ, -4.0, 0.04, KERB))
    kerb = [(x * (R_ISLAND - 0.3) / R_ISLAND, z * (R_ISLAND - 0.3) / R_ISLAND) for x, z in circ]
    V, F = [], []
    for k in range(n):
        o = len(V)
        (xa, za), (xb, zb), (xc, zc), (xd, zd) = circ[k], circ[(k + 1) % n], kerb[(k + 1) % n], kerb[k]
        V += [(xa, 0.045, za), (xb, 0.045, zb), (xc, 0.045, zc), (xd, 0.045, zd)]
        F += [(o, o + 2, o + 1), (o, o + 3, o + 2)]
    out.append(Mesh(V, F, (0.74, 0.73, 0.70)))
    # the stairs down to the Passage du Souvenir, with their low walls
    for x0, x1, zt, zb, s in STAIRS:
        za, zz = min(zt, zb), max(zt, zb)
        out.append(walls([(x0, za), (x1, za), (x1, zz), (x0, zz)], -3.8, 0.04, (0.50, 0.49, 0.46), hole=True))
        nst = 20
        L = abs(zb - zt) - 0.6
        for k in range(nst):
            z0 = zt + s * (k * L / nst)
            z1 = zt + s * ((k + 1) * L / nst)
            out.append(B(x0 + 0.3, x1 - 0.3, -3.8, -(k + 1) * 3.4 / nst, min(z0, z1), max(z0, z1), (0.58, 0.57, 0.54)))
        out.append(B(x0 + 0.3, x1 - 0.3, -3.8, -3.45, min(zt + s * L, zb), max(zt + s * L, zb), DARK))
        for wx0, wx1, wz0, wz1 in ((x0, x0 + 0.3, za, zz), (x1 - 0.3, x1, za, zz),
                                   (x0, x1, zb - 0.3 if s > 0 else zb, zb if s > 0 else zb + 0.3)):
            out.append(B(wx0, wx1, -1.0, 0.95, wz0, wz1, stone(0.95)))
            out.append(B(wx0 - 0.05, wx1 + 0.05, 0.95, 1.08, wz0 - 0.05, wz1 + 0.05, STONE_L))
        for xr in (x0 + 0.42, x1 - 0.42):                                       # handrails
            out.append(tube([(xr, 0.9, zt), (xr, 0.9 - 3.4, zt + s * L)], 0.03, IRON, 4))
    # bollards and chains, where OpenStreetMap has them
    for x, z in pts(BOLLARDS):
        bronze.append(bollard(x, z))
    for s in CHAINS:
        P = pts(s)
        for i in range(1, len(P)):
            bronze.append(chain(P[i - 1], P[i]))
    return out, [m for m in bronze if m is not None]


# ── Assembly ─────────────────────────────────────────────────────────────────
def build():
    arch = pillars() + upper_body() + roof()
    deco = []
    # Champs-Élysées façade (+z): Le Triomphe de 1810 (left, −x), Le Départ de 1792 (right, +x)
    deco += face_main(D2, [(-PC, "triomphe"), (PC, "depart")], [101, 102])          # Aboukir, Funérailles de Marceau
    # Grande-Armée façade (−z), built facing +z and turned: La Paix (world +x) and La Résistance (world −x)
    deco += turn(face_main(D2, [(-PC, "paix"), (PC, "resistance")], [103, 104]), math.pi)   # Alexandrie, Arcole
    # short sides: Jemappes (+x), Austerlitz (−x)
    deco += turn(face_short(W2, 105), math.pi / 2)
    deco += turn(face_short(W2, 106), -math.pi / 2)
    # inner faces of the pillars under the great arch
    deco += turn(face_inner(-GA), -math.pi / 2)
    deco += turn(face_inner(-GA), math.pi / 2)
    ground, bronze = island()
    bronze += [m for m in tomb() if m is not None]
    stone_m = merge(arch + deco)
    k = 0.95 + 0.05 * fbm(stone_m.v, 0.08, 5, 2)
    stone_m.c = np.clip(stone_m.c * k[:, None], 0, 1)
    return [("arc", stone_m), ("terre_plein", merge(ground)), ("bronze", merge(bronze), {"material": "bronze"})]


FOOTPRINT = [(round((R_ISLAND + 1.0) * math.cos(2 * math.pi * k / 32), 1), round((R_ISLAND + 1.0) * math.sin(2 * math.pi * k / 32), 1))
             for k in range(32)]


def main():
    out_dir = ROOT / "models" / "Q64436"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = build()
    tris, size = write_glb(out_dir / "model.glb", parts, roughness=0.9, materials={"bronze": METAL})
    allv = np.vstack([p[1].v for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    print(f"model.glb: {tris:,} triangles, {size / 2**20:.2f} MB, extent x {lo[0]:.2f}..{hi[0]:.2f}, "
          f"y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}")
    meta = out_dir / "meta.json"
    if meta.exists():
        mj = json.loads(meta.read_text(encoding="utf-8"))
        mj["triangles"] = int(tris)
        mj["footprint"] = [list(p) for p in FOOTPRINT]
        meta.write_bytes((json.dumps(mj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
