# Notice

*Aviso en español más abajo · [Spanish below](#aviso)*

**Offlined Landmarks 3D** — a project of **[Offlined](https://www.offlined.org)** by **Álvaro C. Chulvi**.

## What is licensed under what

| Part | Files | License |
|---|---|---|
| 3D models | `models/**/model.glb`, `preview.jpg` (and the `-lod.glb` files in releases, derived from them) | **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** by default, or **[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)** if its `meta.json` says so — always stated in the model's `meta.json` |
| Model descriptions | `models/**/meta.json` | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| Code | `tools/`, `model_sources/`, `site/` | [MIT](LICENSE) |

Both model licences allow **any use, also commercial**, including inside Offlined. CC BY 4.0 only asks you to
credit the authors listed in the model's `meta.json`, link the licence and say if you changed the model.
*NonCommercial*, *NoDerivatives* or custom licences are **not accepted** in this repository.

## Contributing a model

By opening a pull request that adds or changes a model you confirm that:

1. you made it yourself, or it adapts only works under CC BY 4.0 or CC0 1.0 that are credited in `based_on`;
2. it is not taken from a map service, a game or a website (Google Earth, Apple Maps, Sketchfab…), neither
   directly nor by photogrammetry of their 3D views;
3. you license it under the licence written in its `meta.json` (CC BY 4.0 or CC0 1.0);
4. it does not include logos, advertising or textures you have no right to share.

## Buildings and copyright

A building is itself a work protected by copyright while its architect's rights last, and a 3D model of it
is, legally, a reproduction of that work — even if the model is entirely your own. Most countries allow
pictures of buildings in public places ("freedom of panorama"), but some only for photos and drawings, some
not for commercial use. So every model says, in `rights`, which kind of building it is:

- **`public_domain`** — the architects died more than 80 years ago (the longest term that applies; 70 in
  most countries), or the building is older than 1850 with no known author. **Accepted directly.** Most
  famous monuments are here: the Puerta de Alcalá, the Colosseum, Notre-Dame, the Taj Mahal… and Gaudí's
  own design of the Sagrada Família (he died in 1926).
- **`protected`** — a modern work, or one with protected parts (later towers, recent sculptures). **Also
  accepted**: models at the scale of a flight over a city show volumes and silhouettes, which carries little
  risk. The maintainer reviews them more carefully, and they are listed as such.

**Takedown:** if you hold rights over a building or a model here and want it removed, open an issue
(*Rights / takedown*) or write to the maintainer. It is removed from the next release; nothing has to be
proven first. This applies to both kinds.

This is not legal advice; it is how the project chooses to work.

## How to credit a model (CC BY 4.0)

> «Puerta de Alcalá» 3D model by *Author Name*, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) —
> from Offlined Landmarks 3D (https://github.com/Aruvaro-Chulvi/OfflinedLandmarks3D).

## Sources

- **Wikidata** — names, Wikipedia titles, positions, architects and dates. [CC0 1.0](https://www.wikidata.org/wiki/Wikidata:Licensing).
- **OpenStreetMap** — positions, outlines and heights used to place and check models. © OpenStreetMap
  contributors, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/); facts taken from it are cited in each
  `meta.json`.
- Each model lists its own sources (plans, surveys, photos) in its `meta.json`.

---

## Aviso

Los modelos se publican con **CC BY 4.0** (o **CC0 1.0** si así lo dice su `meta.json`), los `meta.json` con
**CC0 1.0** y el código con **MIT**. Ambas licencias de modelos permiten **cualquier uso, también comercial**.
No se admiten licencias *NoComercial*, *SinObraDerivada* ni propias.

**Al abrir un pull request con un modelo confirmas que** lo has hecho tú (o adapta solo obras CC BY 4.0 / CC0
citadas en `based_on`), que no sale de un servicio de mapas, un juego o una web (ni por fotogrametría de sus
vistas 3D), que lo licencias como dice su `meta.json` y que no lleva logotipos ni texturas que no puedas
compartir.

**Edificios y derechos de autor.** Un edificio es una obra protegida mientras duran los derechos de su
arquitecto, y un modelo 3D es, legalmente, una reproducción de esa obra aunque el modelo sea tuyo. Por eso
cada modelo indica en `rights` si es de **dominio público** (arquitectos fallecidos hace más de 80 años, o
anterior a 1850 sin autor conocido: se acepta directamente) o **protegido** (obra moderna o con partes
modernas: también se acepta, con una revisión más cuidadosa). **Retirada:** si tienes derechos sobre un
edificio o un modelo y quieres que se quite, abre un issue (*Derechos / retirada*) o escribe al responsable:
se quita en la siguiente release, sin tener que demostrar nada antes. Esto no es asesoramiento legal, es como
el proyecto decide funcionar.
