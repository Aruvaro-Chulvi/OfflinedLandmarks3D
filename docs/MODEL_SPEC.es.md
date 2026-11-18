# Especificación de los modelos

*[Read in English](MODEL_SPEC.md)*

Todos los modelos de este repositorio siguen las mismas reglas, para que [Offlined](https://www.offlined.org)
—o cualquier otro programa— pueda ponerlos en su sitio real, a su tamaño y con su orientación, y quitar el
edificio genérico que se dibujaba ahí, sin retoques a mano.

## 1. Dónde va cada modelo

```
models/<id de Wikidata>/     p. ej. models/Q1140634/  (Puerta de Alcalá, Madrid)
  model.glb                  obligatorio — el modelo
  meta.json                  obligatorio — autoría, licencia, fuentes, posición, orientación, derechos (§4)
  preview.jpg                recomendado — una imagen del modelo, 1280 × 720
```

El id de Wikidata (`Q…`) identifica el monumento sin depender del idioma, y es como Offlined encuentra su
artículo de Wikipedia en cada idioma. Búscalo en [wikidata.org](https://www.wikidata.org): el monumento
tiene que tener artículo en al menos una Wikipedia.

Los modelos hechos por código guardan su programa en `model_sources/<Q…>_<nombre>.py` (ver
[`tools/meshkit.py`](../tools/meshkit.py)).

## 2. Formato

- **glTF 2.0 binario** (`.glb`), un solo archivo con todo dentro (sin `.bin` ni texturas aparte).
- Extensiones permitidas: `KHR_mesh_quantization`, `KHR_texture_transform`, `KHR_materials_emissive_strength`,
  `EXT_texture_webp`. **No se admiten:** compresión Draco, meshopt ni KTX2/Basis (necesitan decodificadores
  que Offlined no lleva).
- Materiales: PBR metálico-rugosidad (el normal de glTF). Los colores por vértice (`COLOR_0`) son bienvenidos:
  no cuestan nada y encajan con el estilo ilustrado de Wiki Flight. Sin luces, cámaras ni animaciones.
- Texturas: JPEG, PNG o WebP, **2048 × 2048 px como mucho**, y no más de dos de ese tamaño en total.

## 3. Escala, ejes y origen

- **Tamaño real, en metros** (1 unidad glTF = 1 m).
- **+Y hacia arriba.** **La fachada principal mira a +Z** (el «frente» de glTF). `heading` en `meta.json`
  dice hacia dónde mira de verdad esa fachada.
- **Origen = en el suelo, en el centro del edificio** (o de su cuerpo principal). `position` en `meta.json`
  es la latitud y longitud de ese punto. `y = 0` es el nivel del suelo ahí.
- **Cimientos:** el suelo no es plano en todas partes. Prolonga los muros o un zócalo unos metros **por
  debajo de `y = 0`** (2–5 m; más en una cuesta fuerte), para que no quede un hueco donde el terreno esté
  más bajo que el origen. No modeles las calles de alrededor ni el terreno: Offlined los dibuja del mapa.
- Lo pequeño que forma parte del monumento (su verja, escalinata, una fuente delante) puede ir incluido.
- **En cuesta o en terrazas** (un monasterio con sus jardines en terrazas, un castillo en un cerro): por
  defecto Offlined allana el suelo dentro de la planta a su punto más bajo y apoya ahí el modelo, que está bien
  para un edificio en llano pero hunde uno en ladera. Pon `"ground": "keep"` en `meta.json`: el terreno se
  deja como está y `y = 0` se apoya en el suelo en `ground_at` (`[x, z]`, por defecto el origen) — elige un
  punto de suelo despejado a la altura de `y = 0` (una plaza delante de la fachada principal), no el centro de
  los tejados. Entonces cada muro, terraza y muro de contención debe bajar lo bastante para tocar el terreno real.

## 4. `meta.json`

```json
{
  "id": "Q1140634",
  "name": "Puerta de Alcalá",
  "heading": 80.6,
  "position": { "lat": 40.419986, "lon": -3.688725, "source": "https://www.openstreetmap.org/way/174805987" },
  "authors": [{ "name": "Tu Nombre", "url": "https://github.com/tu-usuario" }],
  "license": "CC-BY-4.0",
  "rights": { "status": "public_domain", "architects": [{ "name": "Francesco Sabatini", "died": 1797 }], "built": 1778 },
  "sources": [{ "title": "Alzado", "url": "https://…" }],
  "triangles": 17838,
  "notes": "Lo que convenga saber."
}
```

| Clave | Obligatoria | Qué es |
|---|---|---|
| `id` | sí | El id de Wikidata, igual que el nombre de la carpeta. |
| `name` | sí | Su nombre habitual (en cualquier idioma; los títulos de Wikipedia salen de Wikidata). |
| `heading` | sí | Hacia dónde mira la **fachada principal** (+Z en el archivo), en grados desde el norte en el sentido del reloj (0 = norte, 90 = este). |
| `position` | sí | `lat`, `lon` del origen y una fuente (`source`): un elemento de OpenStreetMap, un levantamiento, una ortofoto… |
| `authors` | sí | Todas las personas que lo han hecho, `{name, url?}`. Offlined las enseña en la ficha del modelo, con el número de modelos que cada una tiene en el proyecto. |
| `license` | sí | `CC-BY-4.0` o `CC0-1.0`. No se admite otra (ver [NOTICE](../NOTICE.md)). |
| `rights` | sí | `status`: `public_domain` o `protected`; `architects`: `[{name, died?, wikidata?}]`; `built` (año); `note`. Ver [NOTICE § Edificios y derechos de autor](../NOTICE.md#buildings-and-copyright). |
| `sources` | sí | Los planos, fotos, levantamientos o medidas que has usado. Al menos uno. |
| `footprint` | no | La zona, vista desde arriba, donde Offlined quita los edificios generados, árboles y mobiliario urbano: `[[x, z], …]` en metros en los ejes del propio modelo. Si no se da, el contorno del modelo visto desde arriba más 1 m. Conviene darla en edificios con patio o en L, donde el contorno quitaría también un vecino. |
| `ground` | no | `"flatten"` (por defecto): Offlined allana el suelo dentro de la planta a su punto más bajo y apoya ahí el modelo. `"keep"`: el terreno se deja como está, para modelos en cuesta o en terrazas (§3). |
| `ground_at` | no | Con `"ground": "keep"`: `[x, z]` en los ejes del modelo donde `y = 0` toca el suelo (por defecto `[0, 0]`, el origen). |
| `ground_low` | no | Con `"ground": "keep"`: el nivel (y, ≤ 0) del suelo más bajo alrededor del modelo, p. ej. la huerta bajo un jardín en terrazas. El visor web y la vista previa dibujan ahí su suelo plano, para que se vea lo que queda por debajo de `y = 0` (por defecto 0). |
| `based_on` | no | Si el modelo adapta el trabajo de otra persona: `{ "title", "author", "url", "license" }` — esa licencia también tiene que ser CC-BY-4.0 o CC0-1.0. |
| `country` | no | El país, como código ISO 3166-1 alfa-2 (`ES`, `CN`…). Normalmente no se pone: la release lo toma de Wikidata (el *país* del monumento, P17). Offlined permite descargar solo los monumentos de un país. |
| `triangles` | no | Lo rellena el comprobador si falta. |
| `notes` | no | Lo que convenga saber (qué está simplificado, qué es dudoso). |

## 5. Límites de tamaño

| | |
|---|---|
| Tamaño del archivo | 8 MB |
| Triángulos | 150.000 |
| Texturas | 2048 × 2048 px cada una, dos de ese tamaño en total |

Cada release lleva además una **versión de lejos** de cada modelo (más o menos 1/8 de sus triángulos, 8.000
como mucho), hecha automáticamente por [`tools/build_release.py`](../tools/build_release.py). Offlined solo
dibuja el modelo completo de cerca, y nunca más de unos pocos a la vez, así que una ciudad llena de modelos
sigue funcionando en un portátil modesto.

El comprobador [`tools/validate.py`](../tools/validate.py) se ejecuta en cada pull request y dice qué falla.
Puedes ejecutarlo tú: `python tools/validate.py`.

## 6. Buenas prácticas

- **Sigue el estilo ilustrado** de Wiki Flight: volúmenes limpios, proporciones verdaderas, colores de los
  materiales reales. Un modelo que está bien visto desde 50–300 m vale más que un detalle diminuto que nadie
  verá. Las texturas sacadas de fotos con sombras «horneadas» quedan mal con el sol del juego, que se mueve.
- Usa medidas reales y cítalas: alzados, fichas de patrimonio, alturas de OpenStreetMap, ortofotos. Toma la
  posición y la orientación de OpenStreetMap o de una ortofoto, no a ojo.
- **No copies** modelos, texturas ni mallas de otras webs, juegos o servicios de mapas (Google Earth, Apple
  Maps, Sketchfab…) salvo que su licencia sea CC-BY-4.0 o CC0-1.0 y lo cites en `based_on`. La fotogrametría
  sacada de los mapas 3D de Google o Apple no se admite nunca.
