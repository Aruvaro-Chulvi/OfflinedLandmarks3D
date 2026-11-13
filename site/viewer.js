// Offlined Landmarks 3D — model viewer (MIT)
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const REPO = "https://github.com/Aruvaro-Chulvi/OfflinedLandmarks3D";

const TX = {
  en: {
    sub: "{n} landmarks in 3D, placed where they really are · {a} contributors", contrib: "How to help", search: "Search a monument…",
    empty: "Choose a landmark on the list", people: "Contributors", nModels: "{n} model(s)", by: "3D model by",
    count: "{n} model(s) contributed to the project", lic: "Licence", tris: "Triangles", height: "Height", facing: "Main façade faces",
    rights: "Rights", pd: "Public domain work", prot: "Modern work (protected) — see NOTICE", pos: "Position",
    wp: "Wikipedia", wd: "Wikidata", src: "Sources", person: "The figure beside it is a 1.8 m person. N marks north.",
    dirs: ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"],
  },
  es: {
    sub: "{n} monumentos en 3D, en su sitio real · {a} colaboradores", contrib: "Cómo ayudar", search: "Busca un monumento…",
    empty: "Elige un monumento de la lista", people: "Colaboradores", nModels: "{n} modelo(s)", by: "Modelo 3D de",
    count: "{n} modelo(s) aportado(s) al proyecto", lic: "Licencia", tris: "Triángulos", height: "Altura", facing: "Fachada principal al",
    rights: "Derechos", pd: "Obra de dominio público", prot: "Obra moderna (protegida) — ver NOTICE", pos: "Posición",
    wp: "Wikipedia", wd: "Wikidata", src: "Fuentes", person: "La figura de al lado es una persona de 1,8 m. N marca el norte.",
    dirs: ["norte", "nordeste", "este", "sudeste", "sur", "suroeste", "oeste", "noroeste"],
  },
};
let L = (() => { try { const s = localStorage.getItem("lm3d_lang"); if (s) return s; } catch { /* */ } return /^es\b/i.test(navigator.language) ? "es" : "en"; })();
const t = (k, v = {}) => String(TX[L][k] ?? TX.en[k] ?? k).replace(/\{(\w+)\}/g, (_, x) => v[x] ?? "");
const fmt = n => Number(n).toLocaleString(L === "es" ? "es-ES" : "en-US", { maximumFractionDigits: 1 });
const norm = s => String(s).normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

let IDX = { models: [], authors: {} }, cur = null;
const nameOf = m => m.titles[L] || m.titles.en || m.name;

async function load() {
  IDX = await fetch("models/index.json").then(r => r.json());
  for (const m of IDX.models) m._s = norm(m.name + " " + Object.values(m.titles).join(" ") + " " + m.id + " " + m.authors.map(a => a.name).join(" "));
  texts(); list();
  const h = location.hash.slice(1);
  show(IDX.models.find(x => x.id === h) || IDX.models[0] || null);
}

function texts() {
  document.documentElement.lang = L;
  $("sub").textContent = t("sub", { n: fmt(IDX.models.length), a: fmt(Object.keys(IDX.authors).length) });
  $("lnk-contrib").textContent = t("contrib");
  $("lnk-contrib").href = `${REPO}/blob/main/${L === "es" ? "CONTRIBUTING.es.md" : "CONTRIBUTING.md"}`;
  $("lang").textContent = L === "es" ? "English" : "Español";
  $("q").placeholder = t("search");
  $("empty").textContent = t("empty");
  $("people-h").textContent = t("people");
  $("people-l").innerHTML = Object.entries(IDX.authors).map(([n, a]) =>
    `<li>${a.url ? `<a href="${esc(a.url)}" target="_blank" rel="noopener">${esc(n)}</a>` : esc(n)} <span>· ${t("nModels", { n: a.models })}</span></li>`).join("");
  if (cur) card(cur);
}

function list() {
  const q = norm($("q").value.trim());
  const res = IDX.models.filter(m => !q || m._s.includes(q));
  $("list").innerHTML = res.map(m => `<div class="it${cur === m ? " on" : ""}" data-q="${m.id}">
    ${m.preview ? `<img src="models/${m.id}/preview.jpg" alt="" loading="lazy">` : '<span class="ph"></span>'}
    <span class="nm">${esc(nameOf(m))}<small>${esc(m.authors.map(a => a.name).join(", "))}</small></span></div>`).join("");
}

// ── 3D ───────────────────────────────────────────────────────────────────────
const canvas = $("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xcfe3f1);
scene.fog = new THREE.Fog(0xcfe3f1, 400, 1600);
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 5000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.495;
scene.add(new THREE.HemisphereLight(0xdfefff, 0x6f6a5c, 1.25));
const sun = new THREE.DirectionalLight(0xfff1dc, 2.0);
sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0006; sun.shadow.normalBias = 0.06;
scene.add(sun, sun.target);
const ground = new THREE.Mesh(new THREE.CircleGeometry(1, 96).rotateX(-Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0xb9b4a6, roughness: 1 }));
ground.receiveShadow = true; scene.add(ground);
let grid = null;
const person = new THREE.Group();
{
  const m = new THREE.MeshStandardMaterial({ color: 0xd2492a, roughness: 0.7 });
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.22, 1.05, 4, 10), m); body.position.y = 0.75;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 10), new THREE.MeshStandardMaterial({ color: 0xe0b48f })); head.position.y = 1.62;
  person.add(body, head); person.traverse(o => { o.castShadow = true; });
  scene.add(person);
}
// a north arrow on the ground
function northArrow() {
  const c = document.createElement("canvas"); c.width = c.height = 128; const g = c.getContext("2d");
  g.fillStyle = "#c0392b"; g.beginPath(); g.moveTo(64, 6); g.lineTo(92, 70); g.lineTo(64, 56); g.lineTo(36, 70); g.closePath(); g.fill();
  g.fillStyle = "#222"; g.font = "bold 44px system-ui"; g.textAlign = "center"; g.fillText("N", 64, 118);
  const tx = new THREE.CanvasTexture(c); tx.colorSpace = THREE.SRGBColorSpace;
  return new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ map: tx, transparent: true, depthWrite: false }));
}
const arrow = northArrow(); scene.add(arrow);
let model = null;
const gltf = new GLTFLoader();

function resize() {
  const r = canvas.parentElement.getBoundingClientRect(), w = r.width, h = Math.max(1, r.height);
  renderer.setSize(w, h, false); camera.aspect = w / h;
  const cw = !$("card").hidden && !$("card").classList.contains("min") && w > 760 ? $("card").offsetWidth + 16 : 0;
  if (cw) camera.setViewOffset(w, h, -cw / 2, 0, w, h); else camera.clearViewOffset();
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
function dispose(o) { o.traverse(x => { if (x.geometry) x.geometry.dispose(); if (x.material) [].concat(x.material).forEach(m => { for (const k in m) if (m[k] && m[k].isTexture) m[k].dispose(); m.dispose(); }); }); }

// The model is shown as it stands: its main façade (+Z in the file) turned to its real heading, north at −Z
async function build(m) {
  if (model) { scene.remove(model); dispose(model); model = null; }
  const g = await gltf.loadAsync(`models/${m.id}/model.glb`);
  model = new THREE.Group(); model.add(g.scene);
  g.scene.rotation.y = Math.PI - (m.heading || 0) * Math.PI / 180;
  model.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  scene.add(model);
  const box = new THREE.Box3().setFromObject(model), size = box.getSize(new THREE.Vector3()), c = box.getCenter(new THREE.Vector3());
  const R = Math.max(size.x, size.z, size.y, 4);
  ground.scale.setScalar(R * 6); ground.position.set(c.x, -0.02, c.z);
  if (grid) { scene.remove(grid); grid.geometry.dispose(); }
  const step = R > 150 ? 20 : R > 30 ? 5 : 1;
  grid = new THREE.GridHelper(Math.round(R * 4 / step) * step, Math.round(R * 4 / step), 0x8a8676, 0x8a8676);
  grid.material.transparent = true; grid.material.opacity = 0.18; grid.position.set(c.x, 0.01, c.z); scene.add(grid);
  arrow.scale.setScalar(Math.max(2, R * 0.12)); arrow.position.set(c.x - R * 0.9, 0.03, c.z - R * 0.9);
  person.position.set(box.max.x + Math.max(1.5, R * 0.04), 0, c.z + size.z * 0.3);
  scene.fog.near = R * 4; scene.fog.far = R * 14;
  sun.position.set(c.x + R * 0.7, R * 1.4, c.z + R * 0.9); sun.target.position.copy(c);
  const sc = sun.shadow.camera; sc.left = sc.bottom = -R * 1.2; sc.right = sc.top = R * 1.2; sc.near = 1; sc.far = R * 5; sc.updateProjectionMatrix();
  camera.near = Math.max(0.05, R / 500); camera.far = R * 40; camera.updateProjectionMatrix();
  // first look: from the side the main façade faces, a little to one side
  const h = (m.heading || 0) * Math.PI / 180, a = h + 0.55;
  controls.target.set(c.x, size.y * 0.4, c.z);
  camera.position.set(c.x + Math.sin(a) * R * 1.25, Math.max(R * 0.35, 3), c.z - Math.cos(a) * R * 1.25);
  controls.maxDistance = R * 6; controls.minDistance = Math.max(1, R * 0.05);
  $("scale").textContent = `${fmt(size.x)} × ${fmt(size.z)} × ${fmt(size.y)} m · ${t("person")}`;
}

function card(m) {
  const wl = m.titles[L] ? L : m.titles.en ? "en" : Object.keys(m.titles)[0];
  const wp = wl ? `https://${wl}.wikipedia.org/wiki/${encodeURIComponent(m.titles[wl].replace(/ /g, "_"))}` : null;
  const au = m.authors.map(a => {
    const n = (IDX.authors[a.name] || {}).models || 1;
    return `<div class="author"><small>${t("by")}</small><b>${a.url ? `<a href="${esc(a.url)}" target="_blank" rel="noopener">${esc(a.name)}</a>` : esc(a.name)}</b><small>${t("count", { n })}</small></div>`;
  }).join("");
  const dir = TX[L].dirs[Math.round(((m.heading || 0) % 360) / 45) % 8];
  const rows = [
    [t("lic"), m.license], [t("rights"), m.rights && m.rights.status === "public_domain" ? t("pd") : t("prot")],
    [t("facing"), `${dir} (${m.heading}°)`], m.triangles && [t("tris"), fmt(m.triangles)],
    [t("pos"), `${m.position.lat.toFixed(5)}, ${m.position.lon.toFixed(5)}`],
  ].filter(Boolean);
  $("card").innerHTML = `<h2><button class="tog" id="b-min" title="−/+">${$("card").classList.contains("min") ? "▸" : "▾"}</button>${esc(nameOf(m))}</h2>
    <div class="q">${m.id}</div>${au}
    <dl>${rows.map(([a, b]) => `<dt>${esc(a)}</dt><dd>${esc(b)}</dd>`).join("")}</dl>
    <div class="links">${wp ? `<a href="${wp}" target="_blank" rel="noopener">${t("wp")} (${wl})</a>` : ""}
      <a href="https://www.wikidata.org/wiki/${m.id}" target="_blank" rel="noopener">${t("wd")}</a>
      <a href="${REPO}/tree/main/models/${m.id}" target="_blank" rel="noopener">GitHub</a>
      <a href="models/${m.id}/model.glb" download="${m.id}.glb">⬇ .glb</a></div>`;
  $("card").hidden = false; resize();
  $("b-min").onclick = () => { $("card").classList.toggle("min"); card(m); };
}

async function show(m) {
  cur = m;
  document.querySelectorAll(".it").forEach(e => e.classList.toggle("on", !!m && e.dataset.q === m.id));
  $("empty").hidden = !!m; $("card").hidden = !m; $("scale").textContent = "";
  if (!m) return;
  if (location.hash.slice(1) !== m.id) history.replaceState(null, "", "#" + m.id);
  card(m);
  try { await build(m); } catch (e) { console.error(e); }
}

$("list").addEventListener("click", e => { const r = e.target.closest(".it"); if (r) show(IDX.models.find(x => x.id === r.dataset.q)); });
$("q").addEventListener("input", list);
$("lang").onclick = () => { L = L === "es" ? "en" : "es"; try { localStorage.setItem("lm3d_lang", L); } catch { /* */ } texts(); list(); };
addEventListener("hashchange", () => { const m = IDX.models.find(x => x.id === location.hash.slice(1)); if (m && m !== cur) show(m); });

resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
load();
window.__lm3d = { scene, camera, controls, renderer, show, model: () => model, IDX: () => IDX };
