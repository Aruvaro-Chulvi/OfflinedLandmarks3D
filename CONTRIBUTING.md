# Contributing

*[Leer en español](CONTRIBUTING.es.md)*

Thank you! Every model you add appears in [Offlined](https://www.offlined.org)'s *Wiki Flight* for everyone
who flies over that place, **with your name on its card** and the number of models you have contributed.

## Without modelling

- **Ask for a landmark** you would like to see: issue *Model request*. Plans, photos and measurements help
  whoever makes it.
- **Fix a placement**: a model in the wrong spot, turned the wrong way, floating or sunk, or clearing a
  building it should not. Issue *Fix a placement*, with a source (an OpenStreetMap link, an orthophoto).

## With a model

1. Pick a landmark: one from the **[wanted list](WANTED.md)** (ready with its Wikidata id, OpenStreetMap
   element and height), or any other with a Wikidata id (`Q…`, on [wikidata.org](https://www.wikidata.org)) and a
   Wikipedia article in at least one language. Check there is no folder for it in [`models/`](models/) yet, and
   no open pull request or issue saying someone is on it. **Claim it**: open a *Model request* issue (or comment
   on the existing one) saying you are making it, so nobody else does it at the same time.
2. Make the model following [docs/MODEL_SPEC.md](docs/MODEL_SPEC.md): `.glb`, metres, +Y up, main façade
   to +Z, origin on the ground in the middle, foundations a few metres below 0, under the size limits.
   Free tools: [Blender](https://www.blender.org) (File › Export › glTF 2.0, format *glTF Binary*,
   "+Y Up" ticked) and [gltf.report](https://gltf.report) to inspect the result.
   Or build it by code, like the models in [`model_sources/`](model_sources/), with [`tools/meshkit.py`](tools/meshkit.py).
3. Get the size, position and orientation right. **The size is the real one**, in metres: the outline from
   [openstreetmap.org](https://www.openstreetmap.org), the height from its `height` tag, Wikipedia, a heritage
   record or plans. `python tools/osm.py way 123456 --heading 80 --json out.json` gives you the outline, its
   courtyards and its `building:part` pieces with their heights, already in the model's axes, plus the centre of
   the outline (your `position`) and the two directions the long sides face (one of them is usually your
   `heading`). `python tools/osm.py find 40.42 -3.69 "name"` finds the element's id.
4. Add `models/<Q…>/model.glb`, `meta.json` and, if you can, `preview.jpg` (1280 × 720).
   `python tools/wikidata.py Q…` prints the architects and dates Wikidata knows, to fill in `rights`.
5. Look at it and check it: `python tools/render.py models/Q… --top --osm way/123456` draws the model from
   above with the OpenStreetMap outline over it — if they do not match, the size, `position` or `heading` is
   wrong. `python tools/render.py models/Q…` draws six views with a 1.8 m figure for scale, `--preview` writes
   `preview.jpg` (these two need numpy and Pillow). Then run `python tools/validate.py models/Q…` (Python 3.10+,
   nothing to install) and fix what it says.
6. Open a pull request. The same check runs automatically; a maintainer then looks at the model.

By opening the pull request you confirm the points in [NOTICE.md](NOTICE.md#contributing-a-model): your own
work (or based only on CC BY 4.0 / CC0 works you credit), not taken from a map service or a game, under
**CC BY 4.0** or **CC0 1.0**.

### Which one first?

The ones on the [wanted list](WANTED.md). Otherwise, the landmarks people fly to: the ones with articles in
many languages, the symbols of a city, what you see from the air. A simple, correct model of a famous monument
is worth more than a detailed one of a building nobody knows.

### Improving a model that already exists

Anyone can make a better one. Fork the repository and change `models/<Q…>/` in your fork:

- **Built by code** (there is a `model_sources/Q…_*.py`): improve the script, run it, and commit the script
  together with the `.glb` it writes.
- **Built in Blender or another program**: replace `model.glb` (and `preview.jpg`).
- In `meta.json`, **keep the authors who are already there and add yourself**, and add your sources. Every
  author of a model counts it among their contributions.
- In the pull request, say what is better and why.

## How contributions are reviewed

Nothing reaches Offlined without three filters:

1. **The automatic check** (GitHub Actions, the same `tools/validate.py`): format, size and triangle limits,
   textures inside the file, axes and origin, licence, authors, sources and rights in `meta.json`.
2. **A maintainer looks at it** before merging: it is the right landmark; it matches its sources; size,
   position and orientation are right; the licence and authorship are credible (not copied from a paid model,
   a game or a map service); `rights` is honest; there is nothing offensive or unrelated.
   Only maintainers can merge — a pull request on its own changes nothing.
3. **A release.** Offlined only downloads what is in a published release (its `manifest.json`, with the size
   and SHA-256 of every file). A merged model waits there until the maintainer publishes the next release.

Models that turn out to be wrong, or that a rights holder asks to remove, are taken out in the next release.

## Code

The tools are in [`tools/`](tools/) (Python, MIT). `validate.py`, `wikidata.py` and `osm.py` need nothing but
Python; `meshkit.py`, `render.py` and `build_release.py` (the far-away versions and the manifest) need numpy and
Pillow:

```bash
pip install -r tools/requirements.txt
python tools/build_release.py landmarks-YYYY-MM-DD
```
