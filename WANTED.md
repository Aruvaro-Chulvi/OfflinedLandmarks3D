# Wanted list

*[Leer en español](WANTED.es.md)*

Landmarks we would like to see in Wiki Flight, with the data to start already looked up. Pick one, **claim
it** (open a *Model request* issue with "I'd like to make it myself" ticked, or comment on the existing one), and
follow [CONTRIBUTING.md](CONTRIBUTING.md). When its model is merged, it leaves this list.

**What size?** The real one, in metres. The OpenStreetMap element below gives the outline (and often the height
and the `building:part` pieces): `python tools/osm.py way 335232793 --heading <h> --json out.json` writes it in
the model's axes. Heights come from the `height` tag, Wikipedia, heritage records or plans; measure the rest on
photos. Check the result with `python tools/render.py models/Q… --top --osm way/335232793`.

**Rights**: in Spain a building is free 80 years after the death of its last architect who died before 1987
(70 years for later deaths). "Protected" models are accepted too — see [NOTICE.md](NOTICE.md).

## Madrid

| Landmark | Wikidata | OpenStreetMap | Known | Rights | Difficulty |
|---|---|---|---|---|---|
| Edificio Metrópolis | [Q465250](https://www.wikidata.org/wiki/Q465250) | [relation 4655046](https://www.openstreetmap.org/relation/4655046) | Jules and Raymond Février, 1911; slate dome with gilt ornament and the Winged Victory on top | protected (the statue on top is a 1977 replacement by Federico Coullaut-Valera) | ★★ |
| Real Casa de Correos (Puerta del Sol) | [Q9067027](https://www.wikidata.org/wiki/Q9067027) | [relation 4098953](https://www.openstreetmap.org/relation/4098953) | Jacques Marquet, 1768; the clock tower of the New Year chimes (1866) | public domain | ★★ |
| Palacio de Cristal (Retiro) | [Q2420025](https://www.wikidata.org/wiki/Q2420025) | [way 23596451](https://www.openstreetmap.org/way/23596451) | Ricardo Velázquez Bosco, 1887; iron and glass on a brick base, by its pond | public domain | ★★ (glass: use the `glass` material) |
| Monumento a Alfonso XII (Retiro) | [Q3862645](https://www.wikidata.org/wiki/Q3862645) | [way 722360391](https://www.openstreetmap.org/way/722360391) (viewpoint; trace the colonnade on the aerial imagery) | José Grases Riera, 1922; semicircular colonnade and equestrian statue over the Estanque Grande | check: several sculptors, Mariano Benlliure died in 1947 (free from 2028) | ★★ |
| Puerta de Toledo | [Q2657591](https://www.wikidata.org/wiki/Q2657591) | [way 335232793](https://www.openstreetmap.org/way/335232793) | Antonio López Aguado, 1827; OSM height 15.24 m | public domain | ★ (a cousin of the Puerta de Alcalá) |
| Plaza de toros de Las Ventas | [Q756621](https://www.wikidata.org/wiki/Q756621) | [way 23321768](https://www.openstreetmap.org/way/23321768) | José Espelius and Manuel Muñoz Monasterio, 1931; neo-Mudéjar brick ring | protected (Muñoz Monasterio died in 1969) | ★★ |
| Real Basílica de San Francisco el Grande | [Q2473884](https://www.wikidata.org/wiki/Q2473884) | [way 28702618](https://www.openstreetmap.org/way/28702618) | Francisco Cabezas, Antonio Pló, Francesco Sabatini, 1784; one of the largest domes in Christendom | public domain | ★★ |

## Anywhere else

Any landmark with a Wikipedia article is welcome. Ask for the ones you would like with a *Model request* issue;
the most requested ones are added here.
