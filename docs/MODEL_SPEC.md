# Model specification

*[Leer en español](MODEL_SPEC.es.md)*

Every model in this repository follows the same rules, so that [Offlined](https://www.offlined.org) — or any
other program — can put it in its real place, size and orientation, and take away the generic building that
was drawn there, without manual tweaks.

## 1. Where a model goes

```
models/<Wikidata id>/        e.g. models/Q1140634/  (Puerta de Alcalá, Madrid)
  model.glb                  required — the model
  meta.json                  required — who made it, licence, sources, position, orientation, rights (§4)
  preview.jpg                recommended — a picture of the model, 1280 × 720
```

The Wikidata id (`Q…`) identifies the landmark without depending on any language, and is how Offlined finds
its Wikipedia article in every language. Look it up on [wikidata.org](https://www.wikidata.org): the
landmark must have an article in at least one Wikipedia.

Models built by code keep their script in `model_sources/<Q…>_<name>.py` (see [`tools/meshkit.py`](../tools/meshkit.py)).

## 2. Format

- **glTF 2.0 binary** (`.glb`), one file with everything inside (no external `.bin` or textures).
- Extensions allowed: `KHR_mesh_quantization`, `KHR_texture_transform`, `KHR_materials_emissive_strength`,
  `EXT_texture_webp`. **Not allowed:** Draco, meshopt and KTX2/Basis compression (they need decoders that
  Offlined does not ship).
- Materials: PBR metallic-roughness (the glTF default). Vertex colours (`COLOR_0`) are welcome: they cost
  nothing and fit the illustrated look of Wiki Flight. No lights, cameras or animations.
- Textures: JPEG, PNG or WebP, **2048 × 2048 px at most**, and no more than two of that size in all.

## 3. Scale, axes and origin

- **Real size, in metres** (1 glTF unit = 1 m).
- **+Y is up.** **The main façade faces +Z** (the glTF "front"). `heading` in `meta.json` says where that
  façade really faces.
- **Origin = on the ground, in the middle of the building** (or of its main body). `position` in
  `meta.json` is the latitude and longitude of that point. `y = 0` is the ground level there.
- **Foundations:** the ground is not flat everywhere. Extend the walls or a plinth a few metres **below
  `y = 0`** (2–5 m; more on a steep slope), so that no gap shows where the terrain is lower than the origin.
  Do not model the surrounding streets or the terrain itself: Offlined draws them from the map.
- Small things that belong to the monument (its railing, steps, a fountain in front) can be included.
- **On a slope or on terraces** (a monastery with its terraced gardens, a castle on a hill): by default Offlined
  levels the ground inside the footprint to its lowest point and sets the model there, which is right for a
  building on flat ground but sinks one on a hillside. Give `"ground": "keep"` in `meta.json`: the terrain is
  left as it is and `y = 0` is set on the ground at `ground_at` (`[x, z]`, default the origin) — pick a point of
  open ground at the level of `y = 0` (a square in front of the main façade), not the middle of the roofs. Then
  every wall, terrace and retaining wall must go down far enough to meet the real terrain.

## 4. `meta.json`

```json
{
  "id": "Q1140634",
  "name": "Puerta de Alcalá",
  "heading": 80.6,
  "position": { "lat": 40.419986, "lon": -3.688725, "source": "https://www.openstreetmap.org/way/174805987" },
  "authors": [{ "name": "Your Name", "url": "https://github.com/you" }],
  "license": "CC-BY-4.0",
  "rights": { "status": "public_domain", "architects": [{ "name": "Francesco Sabatini", "died": 1797 }], "built": 1778 },
  "sources": [{ "title": "Elevation drawing", "url": "https://…" }],
  "triangles": 17838,
  "notes": "Anything worth knowing."
}
```

| Key | Required | Meaning |
|---|---|---|
| `id` | yes | The Wikidata id, the same as the folder name. |
| `name` | yes | Its usual name (any language; the Wikipedia titles come from Wikidata). |
| `heading` | yes | Where the **main façade** (+Z in the file) faces, in degrees clockwise from north (0 = north, 90 = east). |
| `position` | yes | `lat`, `lon` of the origin, and a `source` (an OpenStreetMap element, a survey, an orthophoto…). |
| `authors` | yes | Everyone who made it, `{name, url?}`. Offlined shows them on the model's card, with the number of models each one has in the project. |
| `license` | yes | `CC-BY-4.0` or `CC0-1.0`. Nothing else is accepted (see [NOTICE](../NOTICE.md)). |
| `rights` | yes | `status`: `public_domain` or `protected`; `architects`: `[{name, died?, wikidata?}]`; `built` (year); `note`. See [NOTICE § Buildings and copyright](../NOTICE.md#buildings-and-copyright). |
| `sources` | yes | The plans, photos, surveys or measurements you used. At least one. |
| `footprint` | no | The area, seen from above, where Offlined clears the generated buildings, trees and street furniture: `[[x, z], …]` in metres in the model's own axes. By default, the outline of the model seen from above plus 1 m. Give it for buildings with courtyards or L shapes, where the default would also clear a neighbour. |
| `ground` | no | `"flatten"` (default): Offlined levels the ground inside the footprint to its lowest point and sets the model there. `"keep"`: the terrain is left as it is, for models on a slope or on terraces (§3). |
| `ground_at` | no | With `"ground": "keep"`: `[x, z]` in the model's axes where `y = 0` meets the ground (default `[0, 0]`, the origin). |
| `ground_low` | no | With `"ground": "keep"`: the level (y, ≤ 0) of the lowest ground around the model, e.g. the orchard below a terraced garden. The web viewer and the preview draw their flat ground there, so what stands below `y = 0` can be seen (default 0). |
| `collision` | no | Only to tune the collision volume, which is made by itself from the model when the release is built: the model turned into boxes (cubes of about 1.5–4 m merged), what a plane can really hit — arches, the space under a bridge deck and the gaps between towers stay open, closed insides are solid. `{"voxel": metres, "close": metres, "add": [[x0, y0, z0, x1, y1, z1], …], "remove": […]}`: cube size, gaps narrower than `close` metres that are closed (default 6: the smallest plane in Wiki Flight has a span of 8 m, so none gets through), and boxes in model metres made solid or open by hand. The web viewer shows it with «Collision». |
| `based_on` | no | If the model adapts someone else's work: `{ "title", "author", "url", "license" }` — that licence must be CC-BY-4.0 or CC0-1.0 too. |
| `country` | no | The country as an ISO 3166-1 alpha-2 code (`ES`, `CN`…). Normally left out: the release takes it from Wikidata (the landmark's *country*, P17). Offlined lets people download only the landmarks of one country. |
| `triangles` | no | Filled in by the checker if missing. |
| `notes` | no | Anything worth knowing (what is simplified, what is uncertain). |

## 5. Size limits

| | |
|---|---|
| File size | 8 MB |
| Triangles | 150,000 |
| Textures | 2048 × 2048 px each, two of that size in all |

Every release also carries a **far-away version** of each model (about 1/8 of its triangles, at most 8,000),
made automatically by [`tools/build_release.py`](../tools/build_release.py). Offlined draws the full model
only when you are close, and never more than a few at a time, so a city full of models still runs on a
modest laptop.

The checker in [`tools/validate.py`](../tools/validate.py) runs on every pull request and tells you what is
wrong. You can run it yourself: `python tools/validate.py`.

## 6. Good practice

- **Match the illustrated style** of Wiki Flight: clean volumes, true proportions, colours of the real
  materials. A model that is right from 50–300 m away is worth more than tiny detail nobody will see.
  Photo-scanned textures with baked shadows look wrong under the game's moving sun.
- Use real measurements and cite them: elevation drawings, heritage records, OpenStreetMap heights,
  orthophotos. Take the position and orientation from OpenStreetMap or an orthophoto, not by eye.
- **Do not copy** models, textures or meshes from other websites, games or map services (Google Earth,
  Apple Maps, Sketchfab…) unless their licence is CC-BY-4.0 or CC0-1.0 and you credit them in `based_on`.
  Photogrammetry from Google's or Apple's 3D maps is never allowed.
