# Lista de deseos

*[Read in English](WANTED.md)*

Monumentos que nos gustaría ver en Wiki Flight, con los datos para empezar ya buscados. Elige uno,
**resérvalo** (abre un issue *Pedir un modelo* con «Me gustaría hacerlo yo» marcado, o comenta en el que ya hay)
y sigue [CONTRIBUTING.es.md](CONTRIBUTING.es.md). Cuando se acepta su modelo, sale de esta lista.

**¿De qué tamaño?** El real, en metros. El elemento de OpenStreetMap de la tabla da el contorno (y a menudo la
altura y las piezas `building:part`): `python tools/osm.py way 335232793 --heading <h> --json out.json` lo escribe
en los ejes del modelo. Las alturas salen de la etiqueta `height`, de Wikipedia, de fichas de patrimonio o de
planos; lo demás se mide en fotos. Comprueba el resultado con
`python tools/render.py models/Q… --top --osm way/335232793`.

**Derechos**: en España un edificio queda libre 80 años después de morir el último de sus arquitectos, si murió
antes de 1987 (70 años para los fallecidos después). Los modelos «protegidos» también se aceptan: ver
[NOTICE.md](NOTICE.md).

## Madrid

| Monumento | Wikidata | OpenStreetMap | Datos | Derechos | Dificultad |
|---|---|---|---|---|---|
| Edificio Metrópolis | [Q465250](https://www.wikidata.org/wiki/Q465250) | [relation 4655046](https://www.openstreetmap.org/relation/4655046) | Jules y Raymond Février, 1911; cúpula de pizarra con adornos dorados y la Victoria alada encima | protegido (la estatua de arriba es una sustitución de 1977, de Federico Coullaut-Valera) | ★★ |
| Real Casa de Correos (Puerta del Sol) | [Q9067027](https://www.wikidata.org/wiki/Q9067027) | [relation 4098953](https://www.openstreetmap.org/relation/4098953) | Jacques Marquet, 1768; la torre del reloj de las campanadas (1866) | dominio público | ★★ |
| Palacio de Cristal (Retiro) | [Q2420025](https://www.wikidata.org/wiki/Q2420025) | [way 23596451](https://www.openstreetmap.org/way/23596451) | Ricardo Velázquez Bosco, 1887; hierro y cristal sobre zócalo de ladrillo, junto a su estanque | dominio público | ★★ (cristal: material `glass`) |
| Monumento a Alfonso XII (Retiro) | [Q3862645](https://www.wikidata.org/wiki/Q3862645) | [way 722360391](https://www.openstreetmap.org/way/722360391) (mirador; la columnata hay que trazarla sobre la ortofoto) | José Grases Riera, 1922; columnata semicircular y estatua ecuestre sobre el Estanque Grande | revisar: varios escultores, Mariano Benlliure murió en 1947 (libre desde 2028) | ★★ |
| Puerta de Toledo | [Q2657591](https://www.wikidata.org/wiki/Q2657591) | [way 335232793](https://www.openstreetmap.org/way/335232793) | Antonio López Aguado, 1827; altura en OSM 15,24 m | dominio público | ★ (prima de la Puerta de Alcalá) |
| Plaza de toros de Las Ventas | [Q756621](https://www.wikidata.org/wiki/Q756621) | [way 23321768](https://www.openstreetmap.org/way/23321768) | José Espelius y Manuel Muñoz Monasterio, 1931; anillo neomudéjar de ladrillo | protegido (Muñoz Monasterio murió en 1969) | ★★ |
| Real Basílica de San Francisco el Grande | [Q2473884](https://www.wikidata.org/wiki/Q2473884) | [way 28702618](https://www.openstreetmap.org/way/28702618) | Francisco Cabezas, Antonio Pló, Francesco Sabatini, 1784; una de las mayores cúpulas de la cristiandad | dominio público | ★★ |

## En cualquier otro sitio

Cualquier monumento con artículo en Wikipedia es bienvenido. Pide los que te gustaría ver con un issue *Pedir un
modelo*; los más pedidos se añaden aquí.
