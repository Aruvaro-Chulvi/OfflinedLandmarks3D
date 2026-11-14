# Cómo colaborar

*[Read in English](CONTRIBUTING.md)*

¡Gracias! Cada modelo que añades aparece en *Wiki Flight* de [Offlined](https://www.offlined.org) para todas
las personas que sobrevuelan ese lugar, **con tu nombre en su ficha** y el número de modelos que has aportado.

## Sin modelar

- **Pide un monumento** que te gustaría ver: issue *Pedir un modelo*. Los planos, fotos y medidas ayudan a
  quien lo haga.
- **Corrige una colocación**: un modelo fuera de su sitio, mal girado, flotando o hundido, o que quita un
  edificio que no debería. Issue *Corregir colocación*, con una fuente (un enlace de OpenStreetMap, una ortofoto).

## Con un modelo

1. Elige un monumento: uno de la **[lista de deseos](WANTED.es.md)** (con su id de Wikidata, su elemento de
   OpenStreetMap y su altura ya buscados), u otro cualquiera con id de Wikidata (`Q…`, en
   [wikidata.org](https://www.wikidata.org)) y artículo en al menos una Wikipedia. Comprueba que no hay carpeta
   para él en [`models/`](models/) ni un pull request o issue abierto diciendo que alguien lo está haciendo.
   **Resérvalo**: abre un issue *Pedir un modelo* (o comenta en el que ya hay) diciendo que lo haces tú, para
   que nadie lo haga a la vez.
2. Haz el modelo según [docs/MODEL_SPEC.es.md](docs/MODEL_SPEC.es.md): `.glb`, metros, +Y arriba, fachada
   principal hacia +Z, origen en el suelo en el centro, cimientos unos metros por debajo de 0, dentro de los
   límites. Herramientas gratuitas: [Blender](https://www.blender.org) (Archivo › Exportar › glTF 2.0,
   formato *glTF Binary*, «+Y Up» marcado) y [gltf.report](https://gltf.report) para revisar el resultado.
   O hazlo por código, como los modelos de [`model_sources/`](model_sources/), con [`tools/meshkit.py`](tools/meshkit.py).
3. Acierta con el tamaño, la posición y la orientación. **El tamaño es el real**, en metros: el contorno de
   [openstreetmap.org](https://www.openstreetmap.org), la altura de su etiqueta `height`, de Wikipedia, de una
   ficha de patrimonio o de planos. `python tools/osm.py way 123456 --heading 80 --json out.json` te da el
   contorno, sus patios y sus piezas `building:part` con sus alturas, ya en los ejes del modelo, además del
   centro del contorno (tu `position`) y las dos direcciones a las que miran los lados largos (una suele ser tu
   `heading`). `python tools/osm.py find 40.42 -3.69 "nombre"` encuentra el id del elemento.
4. Añade `models/<Q…>/model.glb`, `meta.json` y, si puedes, `preview.jpg` (1280 × 720).
   `python tools/wikidata.py Q…` muestra los arquitectos y fechas que conoce Wikidata, para rellenar `rights`.
5. Míralo y compruébalo: `python tools/render.py models/Q… --top --osm way/123456` dibuja el modelo visto desde
   arriba con el contorno de OpenStreetMap encima: si no coinciden, está mal el tamaño, `position` o `heading`.
   `python tools/render.py models/Q…` dibuja seis vistas con una figura de 1,8 m para la escala, y `--preview`
   escribe `preview.jpg` (estas dos necesitan numpy y Pillow). Después ejecuta
   `python tools/validate.py models/Q…` (Python 3.10+, sin instalar nada) y corrige lo que diga.
6. Abre un pull request. Se hace la misma comprobación automáticamente y después lo revisa un responsable.

Al abrir el pull request confirmas lo que dice [NOTICE.md](NOTICE.md#aviso): trabajo propio (o basado solo en
obras CC BY 4.0 / CC0 que citas), no sacado de un servicio de mapas ni de un juego, con licencia **CC BY 4.0**
o **CC0 1.0**.

### ¿Por cuál empiezo?

Por los de la [lista de deseos](WANTED.es.md). Si no, por los monumentos a los que la gente vuela: los que
tienen artículo en muchos idiomas, los símbolos de una ciudad, lo que se ve desde el aire. Un modelo sencillo y correcto de un monumento famoso vale más que uno muy
detallado de un edificio que nadie conoce.

### Mejorar un modelo que ya existe

Cualquiera puede hacer uno mejor. Haz un fork del repositorio y cambia `models/<Q…>/` en tu fork:

- **Hecho por código** (hay un `model_sources/Q…_*.py`): mejora el programa, ejecútalo y sube el programa
  junto con el `.glb` que genera.
- **Hecho en Blender u otro programa**: sustituye `model.glb` (y `preview.jpg`).
- En `meta.json`, **mantén a los autores que ya hay y añádete**, y añade tus fuentes. A cada autor de un
  modelo se le cuenta entre sus aportaciones.
- En el pull request, explica qué mejora y por qué.

## Cómo se revisan las aportaciones

Nada llega a Offlined sin pasar tres filtros:

1. **La comprobación automática** (GitHub Actions, el mismo `tools/validate.py`): formato, límites de
   tamaño y triángulos, texturas dentro del archivo, ejes y origen, licencia, autores, fuentes y derechos.
2. **Lo revisa un responsable** antes de aceptarlo: es el monumento correcto; coincide con sus fuentes;
   tamaño, posición y orientación están bien; la licencia y la autoría son creíbles (no copiado de un modelo
   de pago, un juego o un servicio de mapas); `rights` es honesto; no hay nada ofensivo ni ajeno.
   Solo los responsables pueden aceptar: un pull request por sí solo no cambia nada.
3. **Una release.** Offlined solo descarga lo que está en una release publicada (su `manifest.json`, con el
   tamaño y el SHA-256 de cada archivo). Un modelo aceptado espera ahí hasta la siguiente release.

Los modelos que resulten erróneos, o que un titular de derechos pida quitar, salen en la siguiente release.

## Código

Las herramientas están en [`tools/`](tools/) (Python, MIT). `validate.py`, `wikidata.py` y `osm.py` solo
necesitan Python; `meshkit.py`, `render.py` y `build_release.py` (las versiones de lejos y el manifest)
necesitan numpy y Pillow:

```bash
pip install -r tools/requirements.txt
python tools/build_release.py landmarks-AAAA-MM-DD
```
