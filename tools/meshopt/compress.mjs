// Release step: a meshopt-compressed copy of each model (EXT_meshopt_compression + KHR_mesh_quantization).
//
//     cd tools/meshopt && npm ci          (once)
//     node tools/meshopt/compress.mjs in.glb out.glb [in2.glb out2.glb ...]
//
// The models are stored with float geometry; here it is quantized (positions 14 bits over each mesh: about 2 cm on a
// 300 m tower; normals 8 bits) and compressed: about a third of the gzipped size, and half the GPU memory. Quantizing
// moves each mesh's offset and scale into its node (the model looks the same, its numbers do not): Offlined reads the
// copy only in the browser. The models whose terrain is fitted to their underside ("ground": "keep") keep their plain
// file — Offlined reads that one in Python — and get only a compressed far-away version (build_release.py decides).
// Each copy is read back and checked: same triangles per mesh, same bounding box (within 0.2 % of its size or 5 cm).
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { meshopt, getBounds } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';
import fs from 'fs';

await MeshoptEncoder.ready; await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ 'meshopt.encoder': MeshoptEncoder, 'meshopt.decoder': MeshoptDecoder });

/** Triangles per mesh (by node path) and the scene's bounding box */
function shape(doc) {
  const tris = [];
  const walk = (node, path) => {
    const p = path + '/' + node.getName();
    const mesh = node.getMesh();
    if (mesh) for (const prim of mesh.listPrimitives()) {
      const n = prim.getIndices() ? prim.getIndices().getCount() : prim.getAttribute('POSITION').getCount();
      tris.push(p + ':' + n / 3);
    }
    for (const c of node.listChildren()) walk(c, p);
  };
  const scene = doc.getRoot().listScenes()[0];
  for (const n of scene.listChildren()) walk(n, '');
  return { tris: tris.sort().join('\n'), box: getBounds(scene) };
}

const args = process.argv.slice(2);
if (!args.length || args.length % 2) { console.error('usage: node compress.mjs in.glb out.glb [...]'); process.exit(2); }
let bad = 0;
for (let i = 0; i < args.length; i += 2) {
  const [inp, out] = [args[i], args[i + 1]];
  const doc = await io.read(inp);
  const a = shape(doc);
  await doc.transform(meshopt({ encoder: MeshoptEncoder, level: 'medium' }));
  const buf = await io.writeBinary(doc);
  const b = shape(await io.readBinary(buf));
  const size = Math.max(...[0, 1, 2].map(k => a.box.max[k] - a.box.min[k])), tol = Math.max(0.05, size * 0.002);
  const off = Math.max(...[0, 1, 2].flatMap(k => [Math.abs(a.box.min[k] - b.box.min[k]), Math.abs(a.box.max[k] - b.box.max[k])]));
  const name = inp.split(/[\\/]/).pop();
  if (a.tris !== b.tris || !(off <= tol)) { console.error(`${name}: the compressed copy does not match (skipped; box off by ${off.toFixed(3)} m)`); bad++; continue; }
  fs.writeFileSync(out, buf);
  console.log(`${name}: ${(fs.statSync(inp).size / 1e6).toFixed(2)} -> ${(buf.length / 1e6).toFixed(2)} MB (box within ${(off * 100).toFixed(1)} cm)`);
}
process.exit(bad ? 1 : 0);
