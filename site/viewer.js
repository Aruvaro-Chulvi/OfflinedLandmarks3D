// Offlined Landmarks 3D — model viewer (MIT)
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const REPO = "https://github.com/Aruvaro-Chulvi/OfflinedLandmarks3D";

const TX = {
  en: {
    sub: "{n} landmarks in 3D, placed where they really are · {a} contributors", contrib: "How to contribute", search: "Search a monument…",
    empty: "Choose a landmark on the list", people: "Contributors", nModels: "{n} model(s)", by: "3D model by",
    count: "{n} model(s) contributed to the project", lic: "Licence", tris: "Triangles", height: "Height", facing: "Main façade faces",
    rights: "Rights", pd: "Public domain work", prot: "Modern work (protected) — see NOTICE", pos: "Position",
    wp: "Wikipedia", wd: "Wikidata", src: "Sources", person: "The figure beside it is a 1.8 m person. N marks north.",
    dirs: ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"],
    allCountries: "All countries ({n})", noCountry: "No country", cta: "Want to be on this list? See how to contribute.",
    helpTitle: "How to contribute", helpIntro: "Every model you add appears in Offlined's Wiki Flight for everyone who flies over that place, with your name on its card.",
    help1: "Without modelling", help1a: "Ask for a landmark you would like to see (issue “Request a model”).", help1b: "Fix a placement: a model out of place, rotated, floating or sunk (issue “Fix placement”).",
    help2: "With a model", help2a: "Choose a landmark from the wanted list, or any other with a Wikidata id and a Wikipedia article, and book it in an issue.",
    help2b: "Build it as a .glb in metres, +Y up, main façade towards +Z (Blender is free).", help2c: "Get the real size and position from OpenStreetMap: tools/osm.py does it for you.",
    help2d: "Add models/<Q…>/model.glb and meta.json, check it with tools/validate.py and open a pull request.",
    lGuide: "Full guide", lWanted: "Wanted list", lSpec: "Model specification", lRequest: "Request a model",
  },
  es: {
    sub: "{n} monumentos en 3D, en su sitio real · {a} colaboradores", contrib: "Cómo contribuir", search: "Busca un monumento…",
    empty: "Elige un monumento de la lista", people: "Colaboradores", nModels: "{n} modelo(s)", by: "Modelo 3D de",
    count: "{n} modelo(s) aportado(s) al proyecto", lic: "Licencia", tris: "Triángulos", height: "Altura", facing: "Fachada principal al",
    rights: "Derechos", pd: "Obra de dominio público", prot: "Obra moderna (protegida) — ver NOTICE", pos: "Posición",
    wp: "Wikipedia", wd: "Wikidata", src: "Fuentes", person: "La figura de al lado es una persona de 1,8 m. N marca el norte.",
    dirs: ["norte", "nordeste", "este", "sudeste", "sur", "suroeste", "oeste", "noroeste"],
    allCountries: "Todos los países ({n})", noCountry: "Sin país", cta: "¿Quieres aparecer en esta lista? Mira cómo contribuir.",
    helpTitle: "Cómo contribuir", helpIntro: "Cada modelo que añades aparece en Wiki Flight de Offlined para todas las personas que sobrevuelan ese lugar, con tu nombre en su ficha.",
    help1: "Sin modelar", help1a: "Pide un monumento que te gustaría ver (issue «Pedir un modelo»).", help1b: "Corrige una colocación: un modelo fuera de su sitio, girado, flotando o hundido (issue «Corregir colocación»).",
    help2: "Con un modelo", help2a: "Elige un monumento de la lista de deseos, u otro con id de Wikidata y artículo en Wikipedia, y resérvalo en un issue.",
    help2b: "Hazlo en .glb, en metros, +Y arriba y la fachada principal hacia +Z (Blender es gratis).", help2c: "Toma el tamaño y la posición reales de OpenStreetMap: tools/osm.py lo hace por ti.",
    help2d: "Añade models/<Q…>/model.glb y meta.json, compruébalo con tools/validate.py y abre un pull request.",
    lGuide: "Guía completa", lWanted: "Lista de deseos", lSpec: "Especificación del modelo", lRequest: "Pedir un modelo",
  },
};
let regionName = c => c;
function regionNames() { try { const dn = new Intl.DisplayNames([L, "en"], { type: "region" }); regionName = c => (c ? dn.of(c) || c : t("noCountry")); } catch { regionName = c => c || t("noCountry"); } }
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
  $("btn-contrib").textContent = t("contrib");
  regionNames(); countries(); help();
  $("lang").textContent = L === "es" ? "English" : "Español";
  $("q").placeholder = t("search");
  $("empty").textContent = t("empty");
  $("people-h").textContent = t("people");
  const medal = ["🥇", "🥈", "🥉"];
  $("people-l").innerHTML = Object.entries(IDX.authors).sort((x, y) => y[1].models - x[1].models || x[0].localeCompare(y[0])).map(([n, a], i) =>
    `<li${i < 3 ? ' class="top"' : ""}>${i < 3 ? medal[i] + " " : ""}<b>${a.url ? `<a href="${esc(a.url)}" target="_blank" rel="noopener">${esc(n)}</a>` : esc(n)}</b> <span>· ${t("nModels", { n: a.models })}</span></li>`).join("")
    + `<p class="cta"><a href="#" id="cta">${esc(t("cta"))}</a></p>`;
  $("cta").onclick = e => { e.preventDefault(); openHelp(); };
  if (cur) card(cur);
}

/** Selector de país: todos y cada país con sus modelos */
function countries() {
  const v = $("cty").value, n = {};
  for (const m of IDX.models) n[m.country || ""] = (n[m.country || ""] || 0) + 1;
  const cs = Object.keys(n).sort((a, b) => regionName(a).localeCompare(regionName(b), L));
  $("cty").innerHTML = `<option value="*">${esc(t("allCountries", { n: IDX.models.length }))}</option>`
    + cs.map(c => `<option value="${esc(c)}">${esc(regionName(c))} (${n[c]})</option>`).join("");
  $("cty").value = v && (v === "*" || v in n) ? v : "*";
}
/** La ventana «Cómo contribuir» */
function help() {
  const doc = f => `${REPO}/blob/main/${L === "es" ? f + ".es.md" : f + ".md"}`;
  $("help-h").textContent = t("helpTitle"); $("help-intro").textContent = t("helpIntro");
  $("help-h1").textContent = t("help1"); $("help-l1").innerHTML = ["help1a", "help1b"].map(k => `<li>${esc(t(k))}</li>`).join("");
  $("help-h2").textContent = t("help2"); $("help-l2").innerHTML = ["help2a", "help2b", "help2c", "help2d"].map(k => `<li>${esc(t(k))}</li>`).join("");
  $("help-links").innerHTML = [[t("lGuide"), doc("CONTRIBUTING")], [t("lWanted"), doc("WANTED")], [t("lSpec"), `${REPO}/blob/main/docs/${L === "es" ? "MODEL_SPEC.es.md" : "MODEL_SPEC.md"}`],
    [t("lRequest"), `${REPO}/issues/new?template=model-request.yml`]].map(([l, u]) => `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(l)}</a>`).join("");
}
function openHelp() { $("help").hidden = false; }
$("btn-contrib").onclick = openHelp;
$("help-x").onclick = () => { $("help").hidden = true; };
$("help").onclick = e => { if (e.target === $("help")) $("help").hidden = true; };
addEventListener("keydown", e => { if (e.key === "Escape") $("help").hidden = true; });
$("cty").onchange = () => list();

function list() {
  const q = norm($("q").value.trim()), c = $("cty").value;
  const res = IDX.models.filter(m => (!q || m._s.includes(q)) && (c === "*" || (m.country || "") === c));
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
const ground = new THREE.Mesh(new THREE.CircleGeometry(1, 96).rotateX(-Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0xb9b4a6, roughness: 1, transparent: true, opacity: 0.5, depthWrite: false }));
ground.receiveShadow = true; ground.renderOrder = 1; scene.add(ground);      // half transparent: what stands below y = 0 shows through
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
  // on a slope or on terraces ("ground": "keep") the flat ground goes to the lowest level around it (ground_low)
  const gy = m.ground === "keep" && typeof m.ground_low === "number" ? Math.min(0, m.ground_low) : 0;
  ground.scale.setScalar(R * 6); ground.position.set(c.x, gy - 0.02, c.z);
  if (grid) { scene.remove(grid); grid.geometry.dispose(); }
  // a grid of large squares (50 m on big landmarks, 10 m on middle ones, 2 m on small ones), half transparent like the ground
  const step = R > 150 ? 50 : R > 30 ? 10 : 2, n = Math.max(2, Math.round(R * 6 / step));
  grid = new THREE.GridHelper(n * step, n, 0x4f4b42, 0x4f4b42);
  grid.material.transparent = true; grid.material.opacity = 0.5; grid.material.depthWrite = false; grid.renderOrder = 2;
  grid.position.set(c.x, gy + 0.02, c.z); scene.add(grid);
  arrow.scale.setScalar(Math.max(2, R * 0.12)); arrow.position.set(c.x - R * 0.9, gy + 0.03, c.z - R * 0.9);
  person.position.set(box.max.x + Math.max(1.5, R * 0.04), gy, c.z + size.z * 0.3);
  scene.fog.near = R * 4; scene.fog.far = R * 14;
  sun.position.set(c.x + R * 0.7, R * 1.4, c.z + R * 0.9); sun.target.position.copy(c);
  const sc = sun.shadow.camera; sc.left = sc.bottom = -R * 1.2; sc.right = sc.top = R * 1.2; sc.near = 1; sc.far = R * 5; sc.updateProjectionMatrix();
  camera.near = Math.max(0.05, R / 500); camera.far = R * 40; camera.updateProjectionMatrix();
  // first look: from the side the main façade faces, a little to one side
  const h = (m.heading || 0) * Math.PI / 180, a = h + 0.55;
  controls.target.set(c.x, gy + (box.max.y - gy) * 0.4, c.z);
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
