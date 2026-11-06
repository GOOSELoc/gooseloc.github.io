// GOOSELoc Scene Explorer: aerial tile with LiDAR, ground truth and model output,
// the four cameras, a 3D view of the LiDAR sweep and a map of all scenes.
// Assets are written by tools/export_explorer.py.

// three.js is only loaded when the LiDAR tab is opened
let THREE, OrbitControls;

const BASE = './static/explorer/';
const FILE_HINT = 'The scene explorer needs a web server. Run "python3 -m http.server" in this folder and open http://localhost:8000.';
const TILE = 1024;
const GSD = 0.2;
const CAMERAS = ['front', 'right', 'back', 'left'];
const SPLIT_COLORS = ['#7c93b8', '#e3a33b', '#d9534f'];   // train, dev, test

const state = {
  scenes: [],
  cur: null,
  setting: 'imu_prior',
  layers: { lidar: true, posterior: true, prediction: true, gt: true },
  season: 'all',
  terrain: 'all',
  tab: 'cameras',
};
const cache = new Map();
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

// ---------------------------------------------------------------- colours

// Turbo colormap (Google, Apache-2.0), polynomial approximation.
function turbo(t) {
  t = Math.min(1, Math.max(0, t));
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [r, g, b].map(v => Math.round(255 * Math.min(1, Math.max(0, v))));
}

// Magma-like ramp for the position posterior.
const MAGMA = [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]];
function magma(t) {
  const x = Math.min(1, Math.max(0, t)) * (MAGMA.length - 1);
  const i = Math.min(MAGMA.length - 2, Math.floor(x));
  const f = x - i;
  return MAGMA[i].map((a, k) => Math.round(a + f * (MAGMA[i + 1][k] - a)));
}

// ---------------------------------------------------------------- loading

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function decodeU8(b64) {
  const s = atob(b64);
  const a = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
  return a;
}

async function loadScene(scene) {
  if (cache.has(scene.id)) return cache.get(scene.id);
  const dir = BASE + scene.id + '/';
  const [aerial, buf] = await Promise.all([
    loadImage(dir + 'aerial.webp'),
    fetch(dir + 'lidar.bin').then(r => r.arrayBuffer()),
  ]);
  const n = buf.byteLength / 7;
  const xyz = new Int16Array(buf, 0, n * 3);
  const pts = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) pts[i] = xyz[i] / 100;
  const z = Array.from({ length: n }, (_, i) => pts[3 * i + 2]).sort((a, b) => a - b);
  const zlo = z[Math.floor(0.02 * n)], zhi = z[Math.floor(0.98 * n)];
  const data = { aerial, pts, n, zlo, zhi };
  data.lidarLayer = renderLidarLayer(scene, data);
  data.heat = {};
  for (const s of ['north_up', 'imu_prior']) data.heat[s] = renderHeatmap(scene.pred[s].heatmap);
  cache.set(scene.id, data);
  return data;
}

// LiDAR projected into the north-up tile at the ground-truth pose, drawn once per scene.
function renderLidarLayer(scene, d) {
  const c = document.createElement('canvas');
  c.width = c.height = TILE;
  const ctx = c.getContext('2d');
  const th = scene.heading_deg * Math.PI / 180;
  const s = Math.sin(th), co = Math.cos(th);
  for (let i = 0; i < d.n; i++) {
    const x = d.pts[3 * i], y = d.pts[3 * i + 1], z = d.pts[3 * i + 2];
    const east = s * x - co * y, north = co * x + s * y;
    const u = scene.gt[0] + east / GSD, v = scene.gt[1] - north / GSD;
    const [r, g, b] = turbo((z - d.zlo) / (d.zhi - d.zlo));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(u - 0.9, v - 0.9, 1.8, 1.8);
  }
  return c;
}

function renderHeatmap(b64) {
  const v = decodeU8(b64);
  const g = Math.round(Math.sqrt(v.length));
  const c = document.createElement('canvas');
  c.width = c.height = g;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(g, g);
  for (let i = 0; i < v.length; i++) {
    const t = v[i] / 255;
    const [r, gg, b] = magma(t);
    img.data.set([r, gg, b, Math.round(235 * Math.min(1, t * 1.4))], 4 * i);
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

// ---------------------------------------------------------------- aerial view

function rotationRad(scene) {
  return scene.tile_rotation_deg[state.setting] * Math.PI / 180;
}

function arrow(ctx, x, y, headingDeg, len, color, width) {
  const th = headingDeg * Math.PI / 180;
  const ex = x + len * Math.sin(th), ey = y - len * Math.cos(th);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ex, ey); ctx.stroke();
  const a = Math.atan2(ey - y, ex - x), h = 4.5 * width;
  ctx.beginPath();
  ctx.moveTo(ex + Math.cos(a) * h * 0.6, ey + Math.sin(a) * h * 0.6);
  ctx.lineTo(ex + Math.cos(a + 2.5) * h, ey + Math.sin(a + 2.5) * h);
  ctx.lineTo(ex + Math.cos(a - 2.5) * h, ey + Math.sin(a - 2.5) * h);
  ctx.closePath(); ctx.fill();
}

function star(ctx, x, y, r) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const rr = i % 2 ? r * 0.45 : r, a = -Math.PI / 2 + i * Math.PI / 5;
    ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a));
  }
  ctx.closePath();
  ctx.fillStyle = '#ffc61a';
  ctx.strokeStyle = '#1b1b1b';
  ctx.lineWidth = 3;
  ctx.fill(); ctx.stroke();
}

function drawAerial() {
  const scene = state.cur, d = cache.get(scene.id);
  const canvas = $('#ex-aerial');
  const ctx = canvas.getContext('2d');
  const rot = rotationRad(scene);
  const inNorthUp = fn => {
    ctx.save();
    ctx.translate(TILE / 2, TILE / 2); ctx.rotate(-rot); ctx.translate(-TILE / 2, -TILE / 2);
    fn();
    ctx.restore();
  };
  ctx.fillStyle = '#0d0f12';
  ctx.fillRect(0, 0, TILE, TILE);
  inNorthUp(() => ctx.drawImage(d.aerial, 0, 0, TILE, TILE));

  if (state.layers.posterior) {      // already in the frame of the tile the model sees
    ctx.save();
    ctx.globalAlpha = 0.75;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(d.heat[state.setting], 0, 0, TILE, TILE);
    ctx.restore();
  }
  if (state.layers.lidar) inNorthUp(() => { ctx.globalAlpha = 0.85; ctx.drawImage(d.lidarLayer, 0, 0); });

  inNorthUp(() => {
    const p = scene.pred[state.setting];
    if (state.layers.prediction) {
      arrow(ctx, p.x, p.y, p.heading_deg, 70, '#ff4d5e', 5);
      ctx.beginPath(); ctx.arc(p.x, p.y, 15, 0, 2 * Math.PI);
      ctx.lineWidth = 6; ctx.strokeStyle = '#1b1b1b'; ctx.stroke();
      ctx.lineWidth = 4; ctx.strokeStyle = '#ff4d5e'; ctx.stroke();
    }
    if (state.layers.gt) {
      arrow(ctx, scene.gt[0], scene.gt[1], scene.heading_deg, 70, '#ffc61a', 5);
      star(ctx, scene.gt[0], scene.gt[1], 20);
    }
  });

  // north arrow and scale bar in screen space
  const nx = -Math.sin(rot), ny = -Math.cos(rot);
  ctx.save();
  ctx.translate(TILE - 78, 78);
  ctx.fillStyle = 'rgba(255,255,255,0.92)';
  ctx.strokeStyle = 'rgba(13,15,18,0.35)';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(0, 0, 54, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  const needle = (sign, color) => {      // compass needle: north half dark, south half light
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(sign * nx * 26, sign * ny * 26);
    ctx.lineTo(-ny * 8, nx * 8);
    ctx.lineTo(ny * 8, -nx * 8);
    ctx.closePath(); ctx.fill();
  };
  needle(-1, '#c3c9d4');
  needle(1, '#1d2433');
  ctx.fillStyle = '#1d2433';
  ctx.font = 'bold 22px "Google Sans", sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('N', nx * 41, ny * 41 + 1);
  ctx.restore();

  const bar = 50 / GSD;
  ctx.fillStyle = 'rgba(13,15,18,0.65)';
  ctx.fillRect(24, TILE - 72, bar + 32, 50);
  ctx.fillStyle = '#fff';
  ctx.fillRect(40, TILE - 38, bar, 8);
  ctx.font = '22px "Google Sans", sans-serif';
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillText('50 m', 40, TILE - 46);
}

// ---------------------------------------------------------------- heading plot

function drawHeading() {
  const scene = state.cur, p = scene.pred[state.setting];
  const canvas = $('#ex-heading');
  const dpr = window.devicePixelRatio || 1;
  const size = canvas.clientWidth;
  canvas.width = canvas.height = Math.round(size * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const c = size / 2, R = size / 2 - 22, r0 = R * 0.28;
  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue('--ex-ink').trim() || '#222';
  const faint = css.getPropertyValue('--ex-faint').trim() || '#ddd';

  ctx.clearRect(0, 0, size, size);
  ctx.strokeStyle = faint; ctx.lineWidth = 1;
  for (const rr of [r0, (r0 + R) / 2, R]) { ctx.beginPath(); ctx.arc(c, c, rr, 0, 2 * Math.PI); ctx.stroke(); }

  const rot = scene.tile_rotation_deg[state.setting];
  const probs = p.heading_probs, K = probs.length, pmax = Math.max(...probs);
  const w = 2 * Math.PI / K;
  for (let k = 0; k < K; k++) {
    const a = k * w - Math.PI / 2;                     // bin k at k*360/K clockwise from up
    const len = r0 + (R - r0) * probs[k] / pmax;
    ctx.beginPath();
    ctx.arc(c, c, len, a - w * 0.42, a + w * 0.42);
    ctx.arc(c, c, r0, a + w * 0.42, a - w * 0.42, true);
    ctx.closePath();
    ctx.fillStyle = '#6c7a96';
    ctx.globalAlpha = 0.35 + 0.65 * probs[k] / pmax;
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  const ray = (deg, color, width) => {
    const a = deg * Math.PI / 180;
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(c + r0 * 0.3 * Math.sin(a), c - r0 * 0.3 * Math.cos(a));
    ctx.lineTo(c + R * Math.sin(a), c - R * Math.cos(a));
    ctx.stroke();
  };
  if (state.layers.gt) ray(scene.heading_deg - rot, '#e0a800', 3.5);
  if (state.layers.prediction) ray(p.heading_deg - rot, '#ff4d5e', 2.5);
  ctx.fillStyle = ink;
  ctx.font = '11px "Noto Sans", sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('up', c, 10);
}

// ---------------------------------------------------------------- side panels

function monthsBetween(a, b) {
  const [ya, ma] = a.split('-').map(Number), [yb, mb] = b.split('-').map(Number);
  return (yb - ya) * 12 + (mb - ma);
}

function updateInfo() {
  const s = state.cur, p = s.pred[state.setting];
  $('#ex-date').textContent = s.date;
  $('#ex-season').textContent = s.season;
  $('#ex-terrain').textContent = s.terrain;
  $('#ex-dop').textContent = s.dop20_date;
  $('#ex-gap').textContent = `${monthsBetween(s.date, s.dop20_date)} months`;
  $('#ex-gnss').textContent = `${s.gnss_offset_m.toFixed(1)} m`;
  $('#ex-err-loc').textContent = `${p.loc_err_m.toFixed(1)} m`;
  $('#ex-err-ori').textContent = `${p.ori_err_deg.toFixed(1)}°`;
  $('#ex-id').textContent = s.id;
  $('#ex-setting-note').textContent = state.setting === 'north_up'
    ? 'North-up: the tile is north-aligned and the full 360° heading has to be recovered.'
    : `IMU prior: the tile is rotated by the true heading plus a fixed error of ${(
      ((s.tile_rotation_deg.imu_prior - s.heading_deg + 540) % 360) - 180).toFixed(1)}°, as the model sees it.`;
}

function updateCameras() {
  const s = state.cur;
  for (const cam of CAMERAS) {
    const img = $(`#ex-cam-${cam}`);
    img.src = `${BASE}${s.id}/${cam}.webp`;
    img.alt = `${cam} camera, scene ${s.id}`;
  }
}

// ---------------------------------------------------------------- 3D LiDAR

const three = { ready: false };

async function initThree() {
  if (three.loading) return three.loading;
  three.loading = (async () => {
    THREE = await import('three');
    ({ OrbitControls } = await import('three/addons/controls/OrbitControls.js'));
    setupThree();
  })();
  return three.loading;
}

function setupThree() {
  const el = $('#ex-lidar3d');
  three.renderer = new THREE.WebGLRenderer({ antialias: true });
  three.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  el.appendChild(three.renderer.domElement);
  three.scene = new THREE.Scene();
  three.scene.background = new THREE.Color(0x0d0f12);
  three.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 500);
  three.camera.position.set(-18, 22, 32);
  three.controls = new OrbitControls(three.camera, three.renderer.domElement);
  three.controls.enableDamping = true;
  three.controls.target.set(0, -1, 0);
  const car = new THREE.Mesh(new THREE.BoxGeometry(2.0, 1.6, 4.6),
    new THREE.MeshBasicMaterial({ color: 0xffc61a, wireframe: true }));
  car.position.set(0, -1.0, 0.6);
  three.scene.add(car);
  const grid = new THREE.PolarGridHelper(60, 12, 6, 64, 0x3a4150, 0x262b35);
  grid.position.y = -2.2;
  three.scene.add(grid);
  new ResizeObserver(resizeThree).observe(el);
  resizeThree();
  three.ready = true;
  const loop = () => {
    requestAnimationFrame(loop);
    if (state.tab !== 'lidar') return;
    three.controls.update();
    three.renderer.render(three.scene, three.camera);
  };
  loop();
}

function resizeThree() {
  const el = $('#ex-lidar3d');
  const w = el.clientWidth, h = el.clientHeight;
  if (!w || !h) return;
  three.renderer.setSize(w, h, false);
  three.camera.aspect = w / h;
  three.camera.updateProjectionMatrix();
}

function updateThree() {
  if (!three.ready) return;
  const d = cache.get(state.cur.id);
  if (three.points) { three.scene.remove(three.points); three.points.geometry.dispose(); }
  const pos = new Float32Array(d.n * 3), col = new Float32Array(d.n * 3);
  for (let i = 0; i < d.n; i++) {
    const x = d.pts[3 * i], y = d.pts[3 * i + 1], z = d.pts[3 * i + 2];
    pos[3 * i] = -y; pos[3 * i + 1] = z; pos[3 * i + 2] = -x;     // forward = -z, up = +y
    const [r, g, b] = turbo((z - d.zlo) / (d.zhi - d.zlo));
    col[3 * i] = r / 255; col[3 * i + 1] = g / 255; col[3 * i + 2] = b / 255;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  three.points = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.12, vertexColors: true }));
  three.scene.add(three.points);
}

// ---------------------------------------------------------------- map

const map = { ready: false };

async function initMap() {
  const L = window.L;
  map.leaflet = L.map('ex-map', { preferCanvas: true, scrollWheelZoom: false });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map.leaflet);
  const all = await fetch(BASE + 'map.json').then(r => r.json());
  const renderer = L.canvas({ padding: 0.3 });
  for (let i = 0; i < all.lat.length; i++) {
    L.circleMarker([all.lat[i], all.lon[i]], {
      renderer, radius: 2.2, stroke: false, fillOpacity: 0.75, fillColor: SPLIT_COLORS[all.split[i]],
      interactive: false,
    }).addTo(map.leaflet);
  }
  map.markers = new Map();
  for (const s of state.scenes) {
    const m = L.circleMarker([s.lat, s.lon], {
      radius: 7, color: '#fff', weight: 2, fillColor: '#1d2433', fillOpacity: 1,
    }).addTo(map.leaflet);
    m.bindTooltip(`${s.season}, ${s.terrain}, ${s.date}`);
    m.on('click', () => select(s.id));
    map.markers.set(s.id, m);
  }
  map.leaflet.fitBounds(L.latLngBounds(all.lat.map((la, i) => [la, all.lon[i]])), { padding: [10, 10] });
  map.ready = true;
  updateMap(false);
}

function updateMap(pan = true) {
  if (!map.ready) return;
  for (const [id, m] of map.markers) {
    const on = id === state.cur.id;
    m.setStyle({ fillColor: on ? '#ffc61a' : '#1d2433', radius: on ? 10 : 7 });
    if (on) m.bringToFront();
  }
  if (pan) map.leaflet.panTo([state.cur.lat, state.cur.lon]);
}

// ---------------------------------------------------------------- scene picker

function visibleScenes() {
  return state.scenes.filter(s => (state.season === 'all' || s.season === state.season)
    && (state.terrain === 'all' || s.terrain === state.terrain));
}

function buildPicker() {
  const strip = $('#ex-strip');
  strip.innerHTML = '';
  for (const s of state.scenes) {
    const b = document.createElement('button');
    b.className = 'ex-thumb';
    b.dataset.id = s.id;
    b.innerHTML = `<img src="${BASE}${s.id}/thumb.webp" alt="" loading="lazy">
      <span class="ex-thumb-label"><span class="ex-tag ex-${s.season}">${s.season}</span><span>${s.terrain}</span></span>`;
    b.addEventListener('click', () => select(s.id));
    strip.appendChild(b);
  }
}

function updatePicker() {
  const vis = new Set(visibleScenes().map(s => s.id));
  for (const b of $$('.ex-thumb')) {
    b.hidden = !vis.has(b.dataset.id);
    b.classList.toggle('is-active', b.dataset.id === state.cur?.id);
    b.setAttribute('aria-pressed', b.dataset.id === state.cur?.id);
  }
}

// ---------------------------------------------------------------- wiring

async function select(id) {
  const scene = state.scenes.find(s => s.id === id);
  if (!scene) return;
  $('#explorer-app').classList.add('is-loading');
  await loadScene(scene);
  state.cur = scene;
  $('#explorer-app').classList.remove('is-loading');
  updatePicker();
  updateCameras();
  render();
  updateThree();
  updateMap();
  // warm the cache for the neighbours
  const vis = visibleScenes(), i = vis.indexOf(scene);
  for (const n of [vis[i + 1], vis[i - 1]]) if (n) loadScene(n);
}

function render() {
  drawAerial();
  drawHeading();
  updateInfo();
}

function step(delta) {
  const vis = visibleScenes();
  if (!vis.length) return;
  const i = vis.indexOf(state.cur);
  select(vis[(i + delta + vis.length) % vis.length].id);
}

function setTab(tab) {
  state.tab = tab;
  for (const b of $$('[data-tab]')) {
    const on = b.dataset.tab === tab;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-selected', on);
  }
  for (const p of $$('.ex-pane')) p.hidden = p.dataset.pane !== tab;
  if (tab === 'lidar') {
    initThree().then(() => { resizeThree(); updateThree(); })
      .catch(err => console.error('could not load three.js', err));
  }
  if (tab === 'map') {
    if (!map.ready) initMap();
    else map.leaflet.invalidateSize();
  }
}

function wire() {
  for (const b of $$('[data-setting]')) {
    b.addEventListener('click', () => {
      state.setting = b.dataset.setting;
      for (const o of $$('[data-setting]')) o.classList.toggle('is-active', o === b);
      render();
    });
  }
  for (const b of $$('[data-layer]')) {
    b.addEventListener('click', () => {
      const k = b.dataset.layer;
      state.layers[k] = !state.layers[k];
      b.classList.toggle('is-active', state.layers[k]);
      b.setAttribute('aria-pressed', state.layers[k]);
      render();
    });
  }
  for (const b of $$('[data-season]')) {
    b.addEventListener('click', () => {
      state.season = b.dataset.season;
      for (const o of $$('[data-season]')) o.classList.toggle('is-active', o === b);
      updatePicker();
      const vis = visibleScenes();
      if (vis.length && !vis.includes(state.cur)) select(vis[0].id);
    });
  }
  for (const b of $$('[data-terrain]')) {
    b.addEventListener('click', () => {
      state.terrain = b.dataset.terrain;
      for (const o of $$('[data-terrain]')) o.classList.toggle('is-active', o === b);
      updatePicker();
      const vis = visibleScenes();
      if (vis.length && !vis.includes(state.cur)) select(vis[0].id);
    });
  }
  for (const b of $$('[data-tab]')) b.addEventListener('click', () => setTab(b.dataset.tab));
  $('#ex-prev').addEventListener('click', () => step(-1));
  $('#ex-next').addEventListener('click', () => step(1));
  $('#explorer-app').addEventListener('keydown', e => {
    if (e.target.closest('input, textarea')) return;
    if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
    if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
  });
  for (const img of $$('.ex-cam img')) {
    img.addEventListener('click', () => {
      $('#ex-lightbox-img').src = img.src;
      $('#ex-lightbox-img').alt = img.alt;
      $('#ex-lightbox').hidden = false;
    });
  }
  $('#ex-lightbox').addEventListener('click', () => { $('#ex-lightbox').hidden = true; });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') $('#ex-lightbox').hidden = true; });
  new ResizeObserver(() => state.cur && drawHeading()).observe($('#ex-heading'));
}

async function main() {
  state.scenes = await fetch(BASE + 'scenes.json').then(r => r.json());
  buildPicker();
  wire();
  await select(state.scenes[0].id);
}

main().catch(err => {
  console.error(err);
  const app = $('#explorer-app');
  if (location.protocol === 'file:') app.querySelector('.ex-error').textContent = FILE_HINT;
  app.classList.add('has-error');
});
