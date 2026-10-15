"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const PAGE = 50;
const TOKEN_KEY = "mvtree-token";
const STATUS_LABEL = { done: "Done", queued: "Queued", running: "Running", error: "Failed" };

const state = {
  status: null,
  records: [],
  total: 0,
  offset: 0,
  open: null,      // record shown in the drawer
  openKey: "",     // status + finished_at of that record, to re-render only on change
  uploads: [],     // browser-side upload queue
  pumping: false,
  timer: 0,
  refreshing: false,
  tableKey: "",
  listOptions: false,
};
let memToken = "";

// ---------------------------------------------------------------- helpers
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const spName = (s) => (s ? `<span class="sp">${esc(s.replace(/_/g, " "))}</span>` : "–");
const pct = (p, d = 1) => (p == null ? "–" : p > 0 && p < 0.5 * 10 ** (-d - 2) ? `<${(10 ** -d).toFixed(d)}%` : `${(p * 100).toFixed(d)}%`);
function bytes(n) {
  if (n == null) return "–";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}
function when(iso) {
  if (!iso) return "–";
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
const shorten = (s, n) => (s && s.length > n ? `${s.slice(0, n - 1)}…` : s || "");

/** "fagus sylvatica" -> "Fagus_sylvatica" when it names one of the model's species; other text is kept as typed. */
function canonicalSpecies(text) {
  const t = text.trim();
  const key = t.replace(/\s+/g, "_").toLowerCase();
  return ((state.status && state.status.species) || []).find((sp) => sp.toLowerCase() === key) || t;
}

function getToken() {
  try { return localStorage.getItem(TOKEN_KEY) || memToken; } catch { return memToken; }
}
function setToken(t) {
  memToken = t;
  try { localStorage.setItem(TOKEN_KEY, t); } catch { /* private mode: keep it in memory */ }
}
function authed(url) {
  const t = getToken();
  return t ? `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(t)}` : url;
}

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  const t = getToken();
  if (t) headers["X-Mvtree-Token"] = t;
  const res = await fetch(path, { ...opts, headers });
  if (res.status === 401) {
    showGate();
    throw new Error("access token required");
  }
  const isJson = (res.headers.get("content-type") || "").includes("json");
  const data = isJson ? await res.json() : null;
  if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  return data;
}

function toast(msg, bad = false) {
  let el = $("#toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    el.setAttribute("role", "status");
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.className = bad ? "show bad" : "show";
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = ""; }, 3500);
}

// ---------------------------------------------------------------- 3D viewer
const RAMP = [[0, [138, 90, 43]], [0.4, [107, 143, 58]], [0.7, [63, 174, 90]], [1, [200, 240, 143]]];
function rampColor(t) {
  for (let k = 1; k < RAMP.length; k++) {
    if (t <= RAMP[k][0]) {
      const [t0, c0] = RAMP[k - 1], [t1, c1] = RAMP[k];
      const u = (t - t0) / (t1 - t0);
      return c0.map((v, j) => v + (c1[j] - v) * u);
    }
  }
  return RAMP[RAMP.length - 1][1];
}
function cssColorABGR(color) {
  const ctx = document.createElement("canvas").getContext("2d");
  ctx.fillStyle = color;
  const hex = ctx.fillStyle;
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return ((0xff << 24) | (b << 16) | (g << 8) | r) >>> 0;
}

class PointViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.n = 0;
    this.raf = 0;
    this.drag = null;
    this.reset();
    canvas.addEventListener("pointerdown", (e) => {
      this.drag = { x: e.clientX, y: e.clientY };
      this.auto = false;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!this.drag) return;
      this.yaw += (e.clientX - this.drag.x) * 0.01;
      this.pitch = Math.max(-1.35, Math.min(1.35, this.pitch + (e.clientY - this.drag.y) * 0.01));
      this.drag = { x: e.clientX, y: e.clientY };
      this.dirty = true;
    });
    const end = () => { this.drag = null; };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.zoom = Math.max(0.5, Math.min(8, this.zoom * Math.exp(-e.deltaY * 0.0015)));
      this.auto = false;
      this.dirty = true;
    }, { passive: false });
    canvas.addEventListener("dblclick", () => { this.reset(); this.dirty = true; });
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  reset() { this.yaw = 0.5; this.pitch = 0.15; this.zoom = 1; this.auto = true; }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(this.canvas.clientWidth * dpr), h = Math.round(this.canvas.clientHeight * dpr);
    if (w && h && (w !== this.canvas.width || h !== this.canvas.height)) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.image = null;
      this.dirty = true;
    }
  }

  setPoints(buffer) {
    const p = new Float32Array(buffer);
    const n = Math.floor(p.length / 3);
    let zmax = 0, r = 0;
    for (let i = 0; i < n; i++) {
      zmax = Math.max(zmax, p[3 * i + 2]);
      r = Math.max(r, Math.hypot(p[3 * i], p[3 * i + 1]));
    }
    const col = new Uint8Array(n * 3);
    for (let i = 0; i < n; i++) {
      const c = rampColor(zmax ? p[3 * i + 2] / zmax : 0);
      col[3 * i] = c[0]; col[3 * i + 1] = c[1]; col[3 * i + 2] = c[2];
    }
    Object.assign(this, { p, n, col, zmax: zmax || 1, radius: r || 1 });
    this.bg = cssColorABGR(getComputedStyle(document.documentElement).getPropertyValue("--viewer-bg").trim() || "#10161a");
    this.reset();
    this.resize();
    this.dirty = true;
    if (!this.raf) this.raf = requestAnimationFrame(this.frame);
  }

  stop() {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.n = 0;
  }

  frame = () => {
    this.raf = requestAnimationFrame(this.frame);
    if (this.auto) { this.yaw += 0.006; this.dirty = true; }
    if (this.dirty && this.n) { this.draw(); this.dirty = false; }
  };

  draw() {
    const W = this.canvas.width, H = this.canvas.height;
    if (!W || !H) return;
    if (!this.image) {
      this.image = this.ctx.createImageData(W, H);
      this.buf = new Uint32Array(this.image.data.buffer);
      this.depth = new Float32Array(W * H);
    }
    const { p, col, n, buf, depth } = this;
    buf.fill(this.bg);
    depth.fill(Infinity);
    const zc = this.zmax / 2;
    const s = 0.86 * Math.min(H / this.zmax, W / (2 * this.radius)) * this.zoom;
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const R = Math.hypot(this.radius, zc) || 1;
    const size = Math.max(1, Math.round(Math.min(W, H) / 380));
    const ox = W / 2, oy = H / 2;
    for (let i = 0; i < n; i++) {
      const x = p[3 * i], y = p[3 * i + 1], z = p[3 * i + 2] - zc;
      const x1 = x * cy - y * sy, y1 = x * sy + y * cy;
      const px = (ox + x1 * s) | 0;
      const py = (oy - (z * cp - y1 * sp) * s) | 0;
      if (px < 0 || py < 0 || px + size > W || py + size > H) continue;
      const d = y1 * cp + z * sp;                     // larger = farther from the viewer
      const shade = 1 - 0.5 * ((d + R) / (2 * R));
      const c = (0xff000000 | (((col[3 * i + 2] * shade) | 0) << 16) | (((col[3 * i + 1] * shade) | 0) << 8)
        | ((col[3 * i] * shade) | 0)) >>> 0;
      for (let dy = 0; dy < size; dy++) {
        let idx = (py + dy) * W + px;
        for (let dx = 0; dx < size; dx++, idx++) {
          if (d < depth[idx]) { depth[idx] = d; buf[idx] = c; }
        }
      }
    }
    this.ctx.putImageData(this.image, 0, 0);
  }
}
const viewer = new PointViewer($("#d-viewer canvas"));

// ---------------------------------------------------------------- data refresh
function listQuery() {
  const params = new URLSearchParams({
    q: $("#q").value.trim(),
    species: $("#f-species").value,
    verified: $("#f-verified").value,
    status: $("#f-status").value,
    sort: $("#f-sort").value,
    limit: PAGE,
    offset: state.offset,
  });
  return `api/records?${params}`;
}

async function refresh() {
  if (state.refreshing) return;
  state.refreshing = true;
  clearTimeout(state.timer);
  try {
    const [status, list] = await Promise.all([api("api/status"), api(listQuery())]);
    state.status = status;
    state.records = list.records;
    state.total = list.total;
    if (state.offset && state.offset >= list.total) {
      state.offset = Math.max(0, Math.floor((list.total - 1) / PAGE) * PAGE);
    }
    renderStatus();
    renderSummary();
    renderTable();
    await trackUploads();
    if (state.open && ["queued", "running"].includes(state.open.status)) {
      const rec = await api(`api/records/${state.open.id}`).catch(() => null);
      if (rec && state.open && rec.id === state.open.id) showDrawer(rec, false);
    }
  } catch (err) {
    if (!String(err.message).includes("token")) {
      const pill = $("#model-status");
      pill.className = "pill pill-bad";
      $(".pill-text", pill).textContent = "Server unreachable";
    }
  } finally {
    state.refreshing = false;
    const s = state.status;
    const busy = !s || s.state === "loading" || s.stats.pending > 0 || state.uploads.some((u) => u.phase === "uploading");
    state.timer = setTimeout(refresh, busy ? 1500 : 15000);
  }
}

function renderStatus() {
  const s = state.status;
  const pill = $("#model-status");
  let label = "", cls = "pill-ok", title = s.model || "";
  if (s.state === "loading") { cls = "pill-wait"; label = "Loading model…"; }
  else if (s.state === "error") { cls = "pill-bad"; label = "Model unavailable"; title = s.error || ""; }
  else {
    label = `Ready · ${s.model.replace(/px x (\d+) folds?/g, " px ×$1").replace(", TTA ", " · TTA ").replace(", device ", " · ")}`;
  }
  if (s.stats.pending) label += ` · ${s.stats.pending} processing`;
  pill.className = `pill ${cls}`;
  pill.title = title;
  $(".pill-text", pill).textContent = label;

  if (!state.listOptions) {
    state.listOptions = true;
    const opts = s.species.map((sp) => `<option value="${esc(sp)}">${esc(sp.replace(/_/g, " "))}</option>`).join("");
    $("#f-species").insertAdjacentHTML("beforeend", opts);
    $("#species-list").innerHTML = s.species.map((sp) => `<option value="${esc(sp.replace(/_/g, " "))}">`).join("");
    $("#ext-hint").textContent = `${s.extensions.join(" ")} · one tree per file · up to ${s.max_upload_mb} MB`;
    $("#file-input").accept = s.extensions.join(",");
  }
}

function renderSummary() {
  const st = state.status.stats;
  $("#st-records").textContent = st.records;
  $("#st-verified").textContent = st.verified;
  $("#st-acc").textContent = st.verified ? pct(st.correct / st.verified, 0) : "–";
  $("#acc-hint").textContent = st.verified
    ? `${st.correct} of ${st.verified} verified trees match the model's top prediction.`
    : "Add the species you confirmed in the field to a record, and this tracks how often the model agrees.";
  const top = st.by_species.slice(0, 8);
  const max = top.length ? top[0].n : 1;
  $("#species-bars").innerHTML = top.length
    ? top.map((b) => `<li data-sp="${esc(b.species)}" title="Show only ${esc(b.species.replace(/_/g, " "))}">
        <span class="fill" style="width:${((b.n / max) * 100).toFixed(1)}%"></span>${spName(b.species)}<b>${b.n}</b></li>`).join("")
    : `<li class="none">No predictions yet</li>`;
}

function rowHtml(r) {
  const v = encodeURIComponent(r.finished_at || "");
  const thumb = r.status === "done"
    ? `<img class="thumb" loading="lazy" alt="" src="${esc(authed(`api/records/${r.id}/thumb.png?v=${v}`))}">`
    : `<div class="thumb ph">${r.status === "error" ? "!" : "…"}</div>`;
  const sub = [when(r.created_at), r.site].filter(Boolean).map(esc).join(" · ");
  let pred;
  if (r.status === "done") {
    const low = r.confidence < 0.6 ? " low" : "";
    pred = `${spName(r.species)}<div class="conf"><span class="meter${low}"><i style="width:${(r.confidence * 100).toFixed(1)}%"></i></span>${pct(r.confidence)}</div>`;
  } else if (r.status === "error") {
    pred = `<span class="muted">${esc(shorten(r.error, 90))}</span>`;
  } else {
    pred = `<span class="muted">${r.status === "running" ? "Predicting…" : "Waiting in queue"}</span>`;
  }
  let ver = `<span class="muted">–</span>`;
  if (r.verified_species) {
    const mark = r.match === true ? `<span class="b-match" title="Matches the prediction">✓</span> `
      : r.match === false ? `<span class="b-miss" title="Differs from the prediction">✗</span> ` : "";
    ver = `<span class="verified">${mark}${spName(r.verified_species)}</span>`;
  }
  const active = state.open && state.open.id === r.id ? " class=\"active\"" : "";
  return `<tr data-id="${r.id}"${active} tabindex="0">
    <td class="c-thumb">${thumb}</td>
    <td><div class="file">${esc(r.filename)}</div><div class="sub">${sub}</div></td>
    <td class="c-pred pred">${pred}</td>
    <td class="c-ver">${ver}</td>
    <td class="c-status"><span class="badge b-${r.status}">${STATUS_LABEL[r.status] || esc(r.status)}</span></td>
  </tr>`;
}

function renderTable() {
  const key = JSON.stringify([state.records, state.total, state.offset, state.open && state.open.id]);
  if (key === state.tableKey) return;
  state.tableKey = key;
  $("#records-table tbody").innerHTML = state.records.map(rowHtml).join("");
  const filtered = $("#q").value || $("#f-species").value || $("#f-verified").value || $("#f-status").value;
  const empty = $("#empty");
  empty.hidden = state.records.length > 0;
  empty.textContent = filtered ? "No records match these filters."
    : "No records yet. Drop a point cloud on the left to get a prediction.";
  $("#records-table").hidden = state.records.length === 0;
  const from = state.total ? state.offset + 1 : 0, to = state.offset + state.records.length;
  $("#page-info").textContent = state.total ? `${from}–${to} of ${state.total}` : "";
  $("#prev").disabled = state.offset === 0;
  $("#next").disabled = to >= state.total;
}

// ---------------------------------------------------------------- uploads
function addFiles(fileList) {
  const s = state.status;
  const exts = s ? s.extensions : [".las", ".laz", ".ply", ".pcd", ".npy", ".npz", ".xyz", ".txt", ".csv", ".pts"];
  const maxMb = s ? s.max_upload_mb : 500;
  for (const f of fileList) {
    const dot = f.name.lastIndexOf(".");
    const ext = dot > 0 ? f.name.slice(dot).toLowerCase() : "";
    const item = { key: `${Date.now()}-${Math.random()}`, file: f, name: f.name, size: f.size, phase: "waiting", progress: 0 };
    if (!exts.includes(ext)) { item.phase = "error"; item.error = `Unsupported file type. Use ${exts.join(" ")}`; }
    else if (f.size === 0) { item.phase = "error"; item.error = "The file is empty"; }
    else if (f.size > maxMb * 1048576) { item.phase = "error"; item.error = `Larger than the ${maxMb} MB limit`; }
    state.uploads.unshift(item);
  }
  state.uploads = state.uploads.slice(0, 30);
  renderUploads();
  pump();
}

async function pump() {
  if (state.pumping) return;
  state.pumping = true;
  try {
    for (;;) {
      const next = [...state.uploads].reverse().find((u) => u.phase === "waiting");
      if (!next) break;
      next.phase = "uploading";
      renderUploads();
      await uploadFile(next);
    }
  } finally {
    state.pumping = false;
  }
}

function uploadFile(item) {
  return new Promise((resolve) => {
    const site = $("#site-input").value.trim();
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `api/records?filename=${encodeURIComponent(item.name)}&site=${encodeURIComponent(site)}`);
    const t = getToken();
    if (t) xhr.setRequestHeader("X-Mvtree-Token", t);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) { item.progress = e.loaded / e.total; renderUploads(); }
    };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status === 201) {
        Object.assign(item, { phase: "sent", id: data.id, status: data.status });
      } else {
        item.phase = "error";
        item.error = data.error || `${xhr.status} ${xhr.statusText}`;
        if (xhr.status === 401) showGate();
      }
      item.file = null;
      renderUploads();
      resolve();
      refresh();
    };
    xhr.onerror = () => {
      Object.assign(item, { phase: "error", error: "Network error during upload", file: null });
      renderUploads();
      resolve();
    };
    xhr.send(item.file);
  });
}

async function trackUploads() {
  const pending = state.uploads.filter((u) => u.id && ["queued", "running"].includes(u.status));
  await Promise.all(pending.map(async (u) => {
    const rec = state.records.find((r) => r.id === u.id) || await api(`api/records/${u.id}`).catch(() => null);
    if (!rec) { u.status = "gone"; return; }
    Object.assign(u, { status: rec.status, species: rec.species, confidence: rec.confidence, error: rec.error });
  }));
  renderUploads();
}

function renderUploads() {
  $("#upload-list").innerHTML = state.uploads.slice(0, 12).map((u) => {
    let stateHtml = "", bar = "";
    if (u.phase === "waiting") stateHtml = "Waiting";
    else if (u.phase === "uploading") {
      stateHtml = `Uploading ${pct(u.progress, 0)}`;
      bar = `<div class="progress"><i style="width:${(u.progress * 100).toFixed(1)}%"></i></div>`;
    } else if (u.phase === "error") {
      stateHtml = "";
      bar = `<div class="state err">${esc(u.error)}</div>`;
    } else if (u.status === "queued" || u.status === "running") {
      stateHtml = u.status === "running" ? "Predicting…" : "Queued";
      bar = `<div class="progress indeterminate"><i></i></div>`;
    } else if (u.status === "done") {
      stateHtml = `${spName(u.species)} ${pct(u.confidence, 0)}`;
    } else if (u.status === "error") {
      stateHtml = "";
      bar = `<div class="state err">Prediction failed: ${esc(shorten(u.error, 120))}</div>`;
    } else if (u.status === "gone") stateHtml = "Deleted";
    const link = u.id && u.status !== "gone";
    const attrs = link ? ` class="upload-item clickable" data-id="${u.id}" title="Open record #${u.id}"` : ` class="upload-item"`;
    return `<li${attrs}><span class="name">${esc(u.name)}</span><span class="state">${stateHtml}</span>${bar}</li>`;
  }).join("");
}

// ---------------------------------------------------------------- drawer
async function openRecord(id) {
  try {
    const rec = await api(`api/records/${id}`);
    showDrawer(rec, true);
    history.replaceState(null, "", `#record-${id}`);
  } catch (err) {
    toast(err.message, true);
  }
}

function showDrawer(rec, fresh) {
  const key = `${rec.status}|${rec.finished_at || ""}`;
  state.open = rec;
  $("#drawer").hidden = false;
  $("#backdrop").hidden = false;
  $("#d-title").textContent = rec.filename;
  $("#d-sub").textContent = [`Record #${rec.id}`, `uploaded ${when(rec.created_at)}`, rec.site].filter(Boolean).join(" · ");
  if (fresh || key !== state.openKey) renderResult(rec);
  if (fresh) {
    const f = $("#d-form");
    f.verified_species.value = (rec.verified_species || "").replace(/_/g, " ");
    f.site.value = rec.site || "";
    f.notes.value = rec.notes || "";
    $("#d-saved").textContent = "";
    $("#d-delete").classList.remove("armed");
    $("#d-delete").textContent = "Delete";
  }
  renderMeta(rec);
  state.openKey = key;
  state.tableKey = "";
  renderTable();
}

function renderResult(rec) {
  const banner = $("#d-banner");
  if (rec.status === "queued") banner.innerHTML = `<div class="banner wait"><span class="spinner"></span>Waiting in the queue. The page updates by itself.</div>`;
  else if (rec.status === "running") banner.innerHTML = `<div class="banner wait"><span class="spinner"></span>Predicting… usually a few seconds per tree on a GPU, minutes on a CPU.</div>`;
  else if (rec.status === "error") banner.innerHTML = `<div class="banner err"><strong>Prediction failed.</strong><pre>${esc(rec.error)}</pre></div>`;
  else banner.innerHTML = "";

  const box = $("#d-viewer");
  viewer.stop();
  if (rec.status === "done") {
    box.hidden = false;
    $("#d-height").textContent = `0–${rec.tree_height.toFixed(1)} m`;
    const id = rec.id;
    fetch(authed(`api/records/${id}/points.f32?v=${encodeURIComponent(rec.finished_at || "")}`))
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error("no points"))))
      .then((buf) => { if (state.open && state.open.id === id) viewer.setPoints(buf); })
      .catch(() => { box.hidden = true; });
  } else {
    box.hidden = true;
  }

  if (rec.status !== "done") { $("#d-result").innerHTML = ""; return; }
  const top = rec.top_k || [];
  const chips = Object.entries(rec.per_resolution || {})
    .map(([res, v]) => `<span class="chip">${esc(res)} px model: ${spName(v.species)} ${pct(v.prob)}</span>`).join("");
  $("#d-result").innerHTML = `
    <div class="top-species"><span>${spName(rec.species)}</span><span class="pct">${pct(rec.confidence)}</span></div>
    <div class="genus">Genus <span class="sp">${esc(rec.genus || "")}</span> · top prediction of the ensemble</div>
    <ul class="probs">${top.map((t) => `<li>${spName(t.species)}<span class="track"><i style="width:${Math.max(t.prob * 100, 0.5).toFixed(1)}%"></i></span><span class="v">${pct(t.prob)}</span></li>`).join("")}</ul>
    ${chips ? `<div class="chips">${chips}</div>` : ""}
    <h3>Model input</h3>
    <img class="montage" alt="Seven orthographic renders of the tree" src="${esc(authed(`api/records/${rec.id}/montage.png?v=${encodeURIComponent(rec.finished_at || "")}`))}">
    <p class="caption">Four side views, top, bottom, and a trunk close-up, encoding depth, point density, and height.</p>`;
}

function renderMeta(rec) {
  const rows = [
    ["Points", rec.n_points != null ? rec.n_points.toLocaleString() : "–"],
    ["Tree height", rec.tree_height != null ? `${rec.tree_height.toFixed(2)} m` : "–"],
    ["File size", bytes(rec.file_size)],
    ["Model", rec.model || "–"],
    ["Prediction time", rec.duration_s != null ? `${rec.duration_s.toFixed(1)} s` : "–"],
    ["Uploaded", when(rec.created_at)],
    ["Finished", when(rec.finished_at)],
  ];
  $("#d-meta").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  const dl = $("#d-download");
  dl.hidden = !rec.has_scan;
  dl.href = authed(`api/records/${rec.id}/scan`);
  dl.setAttribute("download", rec.filename);
  $("#d-rerun").disabled = !rec.has_scan || ["queued", "running"].includes(rec.status);
  $("#d-rerun").title = rec.has_scan ? "" : "The original scan was not kept";
  $("#d-delete").disabled = rec.status === "running";
}

function closeDrawer() {
  viewer.stop();
  state.open = null;
  state.openKey = "";
  $("#drawer").hidden = true;
  $("#backdrop").hidden = true;
  if (location.hash) history.replaceState(null, "", location.pathname + location.search);
  state.tableKey = "";
  renderTable();
}

// ---------------------------------------------------------------- token gate
function showGate() {
  $("#gate").hidden = false;
  $("#gate-form").token.focus();
}

// ---------------------------------------------------------------- wiring
function init() {
  const url = new URL(location.href);
  if (url.searchParams.get("token")) {
    setToken(url.searchParams.get("token"));
    url.searchParams.delete("token");
    history.replaceState(null, "", url.pathname + url.search + url.hash);
  }

  const input = $("#file-input");
  const zone = $("#dropzone");
  input.addEventListener("change", () => { addFiles(input.files); input.value = ""; });
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  ["dragenter", "dragover"].forEach((ev) => window.addEventListener(ev, (e) => {
    e.preventDefault();
    zone.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((ev) => window.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === "drop" || e.relatedTarget === null) zone.classList.remove("over");
  }));
  window.addEventListener("drop", (e) => { if (e.dataTransfer && e.dataTransfer.files.length) addFiles(e.dataTransfer.files); });

  $("#upload-list").addEventListener("click", (e) => {
    const li = e.target.closest("[data-id]");
    if (li) openRecord(li.dataset.id);
  });
  $("#species-bars").addEventListener("click", (e) => {
    const li = e.target.closest("[data-sp]");
    if (!li) return;
    $("#f-species").value = li.dataset.sp;
    state.offset = 0;
    refresh();
  });

  let debounce = 0;
  $("#q").addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => { state.offset = 0; refresh(); }, 250);
  });
  ["#f-species", "#f-verified", "#f-status", "#f-sort"].forEach((sel) => $(sel).addEventListener("change", () => {
    state.offset = 0;
    refresh();
  }));
  $("#prev").addEventListener("click", () => { state.offset = Math.max(0, state.offset - PAGE); refresh(); });
  $("#next").addEventListener("click", () => { state.offset += PAGE; refresh(); });
  $("#export").addEventListener("click", (e) => { e.currentTarget.href = authed("api/export.csv"); });

  const tbody = $("#records-table tbody");
  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (tr) openRecord(tr.dataset.id);
  });
  tbody.addEventListener("keydown", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (tr && e.key === "Enter") openRecord(tr.dataset.id);
  });

  $("#d-close").addEventListener("click", closeDrawer);
  $("#backdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#drawer").hidden) closeDrawer();
  });

  $("#d-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.open) return;
    const f = e.currentTarget;
    try {
      const rec = await api(`api/records/${state.open.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verified_species: canonicalSpecies(f.verified_species.value), site: f.site.value,
          notes: f.notes.value }),
      });
      showDrawer(rec, false);
      $("#d-saved").textContent = "Saved";
      setTimeout(() => { $("#d-saved").textContent = ""; }, 2500);
      refresh();
    } catch (err) {
      toast(err.message, true);
    }
  });

  $("#d-rerun").addEventListener("click", async () => {
    if (!state.open) return;
    try {
      showDrawer(await api(`api/records/${state.open.id}/rerun`, { method: "POST" }), false);
      refresh();
    } catch (err) {
      toast(err.message, true);
    }
  });

  $("#d-delete").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    if (!state.open) return;
    if (!btn.classList.contains("armed")) {
      btn.classList.add("armed");
      btn.textContent = "Click again to delete";
      clearTimeout(btn._t);
      btn._t = setTimeout(() => { btn.classList.remove("armed"); btn.textContent = "Delete"; }, 4000);
      return;
    }
    try {
      const id = state.open.id;
      await api(`api/records/${id}`, { method: "DELETE" });
      closeDrawer();
      toast(`Record #${id} deleted`);
      refresh();
    } catch (err) {
      toast(err.message, true);
    }
  });

  $("#gate-form").addEventListener("submit", (e) => {
    e.preventDefault();
    setToken(e.currentTarget.token.value.trim());
    $("#gate").hidden = true;
    refresh().then(() => {
      if (state.status) renderUploads();
    });
  });

  window.addEventListener("hashchange", () => {
    const m = location.hash.match(/^#record-(\d+)$/);
    if (m && (!state.open || state.open.id !== Number(m[1]))) openRecord(m[1]);
  });

  refresh().then(() => {
    const m = location.hash.match(/^#record-(\d+)$/);
    if (m) openRecord(m[1]);
  });
}

init();
