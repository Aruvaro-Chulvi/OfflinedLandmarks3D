# Ground review

Made by `tools/review_ground.py` on 2026-11-28: how each model sits on the real ground of its area (`terrain.json`, from `tools/terrain.py`). See the tool for what the numbers mean.

Flagged (⚠): floating or buried more than 3 m along more than 5% of the edge / open floors.

Best fit: the height of y = 0 (as "keep") that leaves the least edge floating and the least open floor buried, and how much of each would remain; ✓ = both under 5 %, so the model fits its ground as "keep" at that height.

| | Model | ground | base (m) | from | terrain | drop in footprint (m) | floating max (m) | edge > 3 m | buried max (m) | floors > 3 m | best fit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ≈ | Q79961 Cristo Redentor | keep | 525.9 | terrain at ground_at | tiles | 78.5 | 0.6 | 0% | 66.6 | 64% | 527.9 m: 0% / 60%  |
| ≈ | Q9141 Taj Mahal | keep | 160.3 | terrain at ground_at | tiles | 14.9 | 1.9 | 0% | 10.1 | 47% | 163.3 m: 3% / 14%  |
| ≈ | Q12512 St Peter's Basilica | keep | 34.0 | ground_elev | tiles | 37.4 | 28.8 | 4% | 20.8 | 15% | 34.0 m: 4% / 15%  |
| ≈ | Q10288 Parthenon (with the Acropolis of Athens) | keep | 148.0 | terrain at ground_at | tiles | 38.6 | -19.8 | 0% | 25.4 | 8% | 164.0 m: 0% / 1% ✓ |
| ≈ | Q10285 Colosseo (Colosseum) | flatten | 36.0 | median terrain of the footprint | tiles | 14.3 | – | – | – | – | 36.5 m: 0% / 4% ✓ |
| ≈ | Q1369687 Estadio Metropolitano | flatten | 665.1 | median terrain of the footprint | ign | 14.8 | – | – | – | – | 665.1 m: 0% / 0% ✓ |
| ≈ | Q164027 Estadio Santiago Bernabéu | flatten | 693.1 | median terrain of the footprint | ign | 11.7 | – | – | – | – | 693.1 m: 0% / 0% ✓ |
| ⚠ | Q22970105 Caleido (Torre Caleido) | flatten | 698.0 | median terrain of the footprint | ign | 17.3 | – | – | – | – | 685.5 m: 0% / 3% ✓ |
| ≈ | Q2981 Cathédrale Notre-Dame de Paris | flatten | 38.0 | median terrain of the footprint | tiles | 11.2 | – | – | – | – | 37.0 m: 0% / 0% ✓ |
| ≈ | Q43473 Angkor Wat | flatten | 28.8 | median terrain of the footprint | tiles | 18.7 | – | – | – | – | 28.8 m: 1% / 6%  |
| ≈ | Q45178 Sydney Opera House | flatten | 3.7 | median terrain of the footprint | tiles | 8.7 | – | – | – | – | 7.2 m: 0% / 0% ✓ |
| ✓ | Q1013093 Castillo nuevo de Manzanares el Real | keep | 922.3 | terrain at ground_at | ign | 9.4 | 0.4 | 0% | 4.0 | 4% | 923.8 m: 0% / 0% ✓ |
| ✓ | Q849711 Catedral de Santa María la Real de la Almudena | keep | 633.0 | terrain at ground_at | ign | 11.8 | -5.4 | 0% | 7.7 | 4% | 638.0 m: 0% / 0% ✓ |
| ✓ | Q37200 Great Pyramid of Giza | keep | 58.3 | terrain at ground_at | tiles | 53.8 | 5.1 | 0% | 10.8 | 3% | 59.3 m: 0% / 2% ✓ |
| ✓ | Q4152 Schloss Neuschwanstein | keep | 960.0 | ground_elev | tiles | 29.4 | -4.8 | 0% | 6.5 | 1% | 960.5 m: 0% / 0% ✓ |
| ✓ | Q3050235 Murallas de Ávila | keep | 1118.9 | terrain at ground_at | ign | 62.7 | -3.4 | 0% | 5.9 | 0% | 1118.9 m: 0% / 0% ✓ |
| ✓ | Q47476 Alhambra | keep | 773.7 | terrain at ground_at | ign | 63.6 | 1.4 | 0% | 8.8 | 0% | 774.2 m: 0% / 0% ✓ |
| ✓ | Q12506 Hagia Sophia (Ayasofya) | keep | 39.3 | terrain at ground_at | tiles | 12.9 | -0.0 | 0% | 4.3 | 0% | 39.3 m: 0% / 0% ✓ |
| ✓ | Q676203 Machu Picchu | keep | 2441.0 | terrain at ground_at | tiles | 87.2 | 0.6 | 0% | 3.0 | 0% | 2441.0 m: 0% / 0% ✓ |
| ✓ | Q1063195 Torre de Madrid | flatten | 638.1 | median terrain of the footprint | ign | 4.6 | – | – | – | – | 638.1 m: 0% / 0% ✓ |
| ✓ | Q1123493 Plaza Mayor de Madrid | keep | 651.0 | terrain at ground_at | ign | 5.0 | 0.0 | 0% | 1.1 | 0% | 651.0 m: 0% / 0% ✓ |
| ✓ | Q1140249 Templo de Debod | flatten | 645.9 | median terrain of the footprint | ign | 2.0 | – | – | – | – | 645.9 m: 0% / 0% ✓ |
| ✓ | Q1140634 Puerta de Alcalá | flatten | 656.0 | median terrain of the footprint | ign | 0.1 | – | – | – | – | 656.0 m: 0% / 0% ✓ |
| ✓ | Q1141262 Puerta de Europa (Torres KIO) | flatten | 729.0 | median terrain of the footprint | ign | 1.0 | – | – | – | – | 729.0 m: 0% / 0% ✓ |
| ✓ | Q12495 Burj Khalifa | flatten | 5.4 | median terrain of the footprint | tiles | 7.8 | – | – | – | – | 5.4 m: 0% / 0% ✓ |
| ✓ | Q129846 Saint Basil's Cathedral (Cathedral of the Intercession on the Moat), with the Kremlin wall and the Spasskaya Tower | keep | 149.7 | terrain at ground_at | tiles | 7.3 | -2.5 | 0% | -4.3 | 0% | 149.7 m: 0% / 0% ✓ |
| ✓ | Q1364354 Torre Emperador Castellana (Torre Espacio) | flatten | 732.0 | median terrain of the footprint | ign | 6.4 | – | – | – | – | 732.0 m: 0% / 0% ✓ |
| ✓ | Q1364394 Torre Picasso | flatten | 684.2 | median terrain of the footprint | ign | 7.7 | – | – | – | – | 684.2 m: 0% / 0% ✓ |
| ✓ | Q1431985 Aeródromo de Cuatro Vientos | flatten | 688.0 | median terrain of the footprint | ign | 4.0 | – | – | – | – | 688.0 m: 0% / 0% ✓ |
| ✓ | Q160112 Museo del Prado | flatten | 634.8 | median terrain of the footprint | ign | 8.1 | – | – | – | – | 634.8 m: 0% / 0% ✓ |
| ✓ | Q1620086 Hipódromo de la Zarzuela | flatten | 604.7 | median terrain of the footprint | ign | 5.5 | – | – | – | – | 604.7 m: 0% / 0% ✓ |
| ✓ | Q166276 Aeropuerto Adolfo Suárez Madrid-Barajas — Terminal 4 | keep | 614.0 | ground_elev | ign | 6.0 | 2.7 | 0% | 2.9 | 0% | 614.0 m: 0% / 0% ✓ |
| ✓ | Q171517 Palacio Real de Madrid | keep | 640.0 | terrain at ground_at | ign | 10.1 | 0.0 | 0% | -0.0 | 0% | 640.0 m: 0% / 0% ✓ |
| ✓ | Q1752772 Monasterio de Santa María de El Paular | keep | 1154.2 | terrain at ground_at | ign | 7.4 | -1.9 | 0% | 0.8 | 0% | 1154.2 m: 0% / 0% ✓ |
| ✓ | Q1849031 Palacio de Cibeles | flatten | 645.0 | median terrain of the footprint | ign | 5.6 | – | – | – | – | 645.0 m: 0% / 0% ✓ |
| ✓ | Q2092717 Edificio Telefónica | flatten | 661.6 | median terrain of the footprint | ign | 4.0 | – | – | – | – | 661.6 m: 0% / 0% ✓ |
| ✓ | Q211250 Teatro Real | flatten | 638.9 | median terrain of the footprint | ign | 1.5 | – | – | – | – | 621.4 m: 0% / 4% ✓ |
| ✓ | Q23093 Estación de Atocha | flatten | 619.2 | median terrain of the footprint | ign | 2.0 | – | – | – | – | 619.2 m: 0% / 0% ✓ |
| ✓ | Q243 Torre Eiffel / Tour Eiffel | flatten | 40.0 | median terrain of the footprint | tiles | 4.3 | – | – | – | – | 40.0 m: 0% / 0% ✓ |
| ✓ | Q2736564 Fuente de Cibeles | flatten | 644.5 | median terrain of the footprint | ign | 1.1 | – | – | – | – | 644.5 m: 0% / 0% ✓ |
| ✓ | Q28471 Real Monasterio de San Lorenzo de El Escorial | keep | 1026.0 | terrain at ground_at | ign | 12.9 | -1.0 | 0% | 2.8 | 0% | 1026.0 m: 0% / 0% ✓ |
| ✓ | Q2859883 Arco de la Victoria | flatten | 654.9 | median terrain of the footprint | ign | 1.3 | – | – | – | – | 654.9 m: 0% / 0% ✓ |
| ✓ | Q2919969 Torres de Colón | flatten | 654.7 | median terrain of the footprint | ign | 2.0 | – | – | – | – | 654.7 m: 0% / 0% ✓ |
| ✓ | Q386413 Faro de Moncloa | flatten | 658.0 | median terrain of the footprint | ign | 4.2 | – | – | – | – | 658.0 m: 0% / 0% ✓ |
| ✓ | Q3891623 Palacio de Velázquez | flatten | 661.2 | median terrain of the footprint | ign | 2.1 | – | – | – | – | 661.2 m: 0% / 0% ✓ |
| ✓ | Q39054 Torre di Pisa (Leaning Tower of Pisa) and Piazza dei Miracoli | flatten | 5.9 | median terrain of the footprint | tiles | 3.1 | – | – | – | – | 5.9 m: 0% / 0% ✓ |
| ✓ | Q39671 Stonehenge | keep | 101.8 | terrain at ground_at | tiles | 2.9 | -0.1 | 0% | 0.0 | 0% | 101.8 m: 0% / 0% ✓ |
| ✓ | Q48435 Basílica de la Sagrada Família | flatten | 30.9 | median terrain of the footprint | ign | 5.8 | – | – | – | – | 30.9 m: 0% / 0% ✓ |
| ✓ | Q4889526 Banco de España | flatten | 641.0 | median terrain of the footprint | ign | 5.0 | – | – | – | – | 625.5 m: 0% / 9%  |
| ✓ | Q519568 Torre Moeve (Torre Cepsa) | flatten | 731.5 | median terrain of the footprint | ign | 1.8 | – | – | – | – | 731.5 m: 0% / 0% ✓ |
| ✓ | Q530129 Palacio Real de Aranjuez | flatten | 488.2 | median terrain of the footprint | ign | 5.8 | – | – | – | – | 488.2 m: 0% / 0% ✓ |
| ✓ | Q5755509 Casa del Labrador | flatten | 491.0 | median terrain of the footprint | ign | 0.0 | – | – | – | – | 491.0 m: 0% / 0% ✓ |
| ✓ | Q6057871 Palacio de Biblioteca y Museos Nacionales | flatten | 652.1 | median terrain of the footprint | ign | 7.1 | – | – | – | – | 652.1 m: 0% / 0% ✓ |
| ✓ | Q62408 Palace of Westminster | flatten | 1.8 | median terrain of the footprint | tiles | 5.5 | – | – | – | – | 1.8 m: 0% / 0% ✓ |
| ✓ | Q64436 Arc de Triomphe de l'Étoile | flatten | 62.3 | median terrain of the footprint | tiles | 4.2 | – | – | – | – | 62.3 m: 0% / 0% ✓ |
| ✓ | Q649426 Edificio España | flatten | 638.5 | median terrain of the footprint | ign | 5.6 | – | – | – | – | 638.5 m: 0% / 0% ✓ |
| ✓ | Q651643 Torre PwC (Torre Sacyr Vallehermoso) | flatten | 728.3 | median terrain of the footprint | ign | 7.2 | – | – | – | – | 728.3 m: 0% / 0% ✓ |
| ✓ | Q80290 Forbidden City (Palace Museum) | flatten | 51.3 | median terrain of the footprint | tiles | 7.9 | – | – | – | – | 51.3 m: 0% / 0% ✓ |
| ✓ | Q82425 Brandenburger Tor | flatten | 39.6 | median terrain of the footprint | tiles | 4.1 | – | – | – | – | 39.6 m: 0% / 0% ✓ |
| ✓ | Q9054619 Palacio de las Cortes (Congreso de los Diputados) | flatten | 645.0 | median terrain of the footprint | ign | 4.0 | – | – | – | – | 645.0 m: 0% / 0% ✓ |
| ✓ | Q9188 Empire State Building | flatten | 15.0 | median terrain of the footprint | usgs | 3.2 | – | – | – | – | 15.0 m: 0% / 0% ✓ |
| ✓ | Q9202 Statue of Liberty | flatten | 4.9 | median terrain of the footprint | usgs | 2.7 | – | – | – | – | 4.9 m: 0% / 0% ✓ |
| ✓ | Q953610 Torre de Cristal | flatten | 731.3 | median terrain of the footprint | ign | 2.9 | – | – | – | – | 731.3 m: 0% / 0% ✓ |
| ✓ | Q953919 Casita del Infante | keep | 1024.7 | terrain at ground_at | ign | 6.2 | -0.1 | 0% | 0.9 | 0% | 1024.7 m: 0% / 0% ✓ |
| ✓ | Q962107 Puerta de Hierro | flatten | 594.9 | median terrain of the footprint | ign | 0.5 | – | – | – | – | 594.9 m: 0% / 0% ✓ |
| ✓ | Q989733 Torrespaña (El Pirulí) | flatten | 658.0 | median terrain of the footprint | ign | 0.6 | – | – | – | – | 658.0 m: 0% / 0% ✓ |

## Accepted after review (≈)

- **Q79961 Cristo Redentor** — the model carries the top of Corcovado, sharper than SRTM: the game cuts the blunt summit down to it.
- **Q9141 Taj Mahal** — SRTM sees the trees of the gardens: the game cuts them down to the model's paths and lawns.
- **Q12512 St Peter's Basilica** — ground_elev fitted (34 m): what is left is SRTM mixing in the Vatican buildings around.
- **Q10288 Parthenon (with the Acropolis of Athens)** — the model carries the rock of the Acropolis, sharper than SRTM: the game cuts the blunt hill down to it.
- **Q10285 Colosseo (Colosseum)** — the Colosseum stands in the valley between the Palatine, the Esquiline and the Caelian: the level square around it is real.
- **Q1369687 Estadio Metropolitano** — a stadium on a slope; the game levels it at its median height (about 7 m of cut and 8 m of fill).
- **Q164027 Estadio Santiago Bernabéu** — a stadium on the slope of the Castellana; about 7 m of cut on the north side.
- **Q2981 Cathédrale Notre-Dame de Paris** — the footprint reaches down to the Seine quays: the fill is on the river side.
- **Q43473 Angkor Wat** — the footprint takes in the temple's terraces and the forest around (SRTM sees the trees); the game levels it.
- **Q45178 Sydney Opera House** — on the platform of Bennelong Point, a few metres above the water.
