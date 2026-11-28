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
  `EXT_texture_webp`. **Not allowed in the repository:** Draco, meshopt and KTX2/Basis compression (contributed
  models stay plain and easy to open). The release adds a meshopt copy of each model by itself
  (`tools/build_release.py` with `tools/meshopt/compress.mjs`: "mo" in the manifest), which Offlined 4.x downloads.
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

### 3.1 The terrain in Offlined, and how to make your model sit on it

**How detailed the terrain is.** Offlined draws the ground from elevation data, not from the real hill:

| Where | Data | Detail |
|---|---|---|
| Everywhere | GEBCO 2026 | one height every ~450 m |
| Where the player downloaded the detailed relief of that country | Copernicus GLO-90 | one height every ~90 m |

So anything narrower than a few hundred metres — a ridge, a crag, a cliff, a small hill, a river bluff — comes out
rounded, often tens of metres lower or higher than in reality, and slopes of a few metres do not show at all.
Count on the ground under your model being **±10 m off on any hillside and ±50 m off on a ridge or a cliff**.
Your model cannot rely on the terrain having the right shape; it has to bring the shape that matters.

**Pick one of three ways to sit (`ground` in `meta.json`):**

1. **Flat** (default, `"flatten"`) — a building on flat ground: the ground falls **less than ~5 m** across the
   footprint. Offlined levels the footprint to its lowest point (with a 20 m blend to the terrain around) and sets
   `y = 0` there. Your model: `y = 0` = the ground at the entrance, and walls or a plinth 2–5 m below it.
   *Do not use it for a model that follows a slope* (city walls going down a hill, a palace on a bluff): the
   whole model would be set at the lowest point and sink.
2. **On a slope, on terraces or on a hilltop** (`"ground": "keep"`) — the ground falls more than ~5 m across the
   model, or it stands on a crest. `y = 0` is set on the ground at `ground_at`, and **Offlined reshapes the
   terrain to fit your model**, but only as much as needed: at the model's edge the ground cannot stay below its
   foot (it would be seen floating) and nowhere above what is seen from the sky (courtyards, terraces, roofs). If
   the relief already falls between the two, it is left alone. What moves goes back to the hillside in a short,
   steep ramp of 10–90 m (the game paints it as rock). Whatever your model has below `ground_low` (foundation
   slabs, basements) is taken as buried. What this means for you:
   - **Model the ground you stand on**: squares, terraces, steps, retaining walls, and the rock of the hill where
     it shows. The lowest surface of your model *becomes* the ground — a courtyard without a floor or a terrace
     without its fill leaves a hole where the coarse terrain shows through.
   - **On a crest or a cliff, add a rock skirt** 20–50 m below the lowest ground (`ground_low`): the ramp then
     starts from rock, and the steep sides look like a mountain, not like a grassy bank.
   - Set `ground_low` to the lowest real ground around the model (the web viewer draws its floor there).
3. **Not touching the terrain** (`"ground": "keep"` and `"ground_fit": false`) — bridges, dams, piers,
   anything standing over water or over a void: the terrain is left exactly as it is (fitting it would raise
   the ground up to the deck).

**The real ground of its area.** `python tools/terrain.py Q…` saves in `terrain.json` the detailed terrain around it (IGN MDT05 in Spain, 3DEP in the USA, SRTM elsewhere) and `python tools/review_ground.py Q…` tells how much your model floats or is buried on it (report for all of them in [GROUND_REVIEW.md](GROUND_REVIEW.md)).

**Common mistakes** (`tools/validate.py` warns about them):
- A model whose base goes far below `y = 0` (it follows a slope) left on **flat**: it sinks. Use `keep`.
- A `keep` model whose base has large holes inside its outline.
- A bridge or anything over water without `"ground_fit": false`.
- `y = 0` that is not the real ground at `ground_at` (for example the floor of a raised terrace, or a roof).

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
| `ground_low` | no | With `"ground": "keep"`: the level (y, ≤ 0) of the lowest ground around the model, e.g. the orchard below a terraced garden. The web viewer and the preview draw their flat ground there, so what stands below `y = 0` can be seen (default 0). Anything below it (foundations, basements) Offlined keeps buried. |
| `ground_elev` | no | The height above sea level (m) of the ground where `y = 0` goes (at `ground_at` with `"keep"`, at the lowest point of the footprint with `"flatten"`), actually measured: a 1–5 m DTM (IGN, USGS 3DEP…) or a survey. Give it where Offlined's relief (90 m, or 450 m where there is no detailed relief) is far off: crests, cliffs, summits. Cite where it comes from in `sources`. |
| `terrain_radius` | no | Half-width (m, 300–5000; default 1200) of the detailed terrain `tools/terrain.py` makes around the model (`terrain.json`). Larger for a monument in the mountains (Machu Picchu: 2500). |
| `surround` | no | The vegetation around it even where the map does not have it: `{"kind": "rainforest" \| "forest" \| "conifer" \| "woodland" \| "palms" \| "garden", "radius": m, "clear"?: m, "density"?: 0–1}`. Offlined paints the ground in that colour and plants those trees in a circle of `radius` (full up to half of it, fading out to the edge), with a clearing of `clear` m next to the model; roads, buildings and water still take them out. Only where it really is like that (Machu Picchu in its mountain forest); not in a city. |
| `terrain_cut` | no | `false` so that `tools/terrain.py` does not cut the building out of a surface model (SRTM): only for models that follow a hill with no building to remove. Default `true`. |
| `collision` | no | Only to tune the collision volume, which is made by itself from the model when the release is built: the model turned into boxes (cubes of about 1.5–4 m merged), what a plane can really hit — arches, the space under a bridge deck and the gaps between towers stay open, closed insides are solid. `{"voxel": metres, "close": metres, "add": [[x0, y0, z0, x1, y1, z1], …], "remove": […]}`: cube size, gaps narrower than `close` metres that are closed (default 6: the smallest plane in Wiki Flight has a span of 8 m, so none gets through), and boxes in model metres made solid or open by hand. The web viewer shows it with «Collision». |
| `based_on` | no | If the model adapts someone else's work: `{ "title", "author", "url", "license" }` — that licence must be CC-BY-4.0 or CC0-1.0 too. |
| `ground_fit` | no | With `"ground": "keep"`: `false` for bridges, dams and anything over water or a void — Offlined does not reshape the terrain to the model (§3.1). Default `true`. |
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
