"use strict";
/* UI: Kalibrier-Labor (Kalibrieren, Genauigkeitstest, Live-Test, Daten-Export) + Spiel */
const PER_ZONE = 10, EVAL_PER_ZONE = 15;
const PROFILE_KEY = "holo-maulwurf-profile-v3", EVALS_KEY = "holo-maulwurf-evals";
const $ = (s) => document.querySelector(s);
const holes = [...document.querySelectorAll(".hole")];
const center = $("#center");
let profile = store(PROFILE_KEY);
// Pretrained model shipped with the app (trained from exported data); used when there is no own calibration.
let pretrained = null;
fetch("model.json", { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).then((m) => { if (m && m.W) { pretrained = m; if (!audio.onTap) home(); } }).catch(() => {});
const model = () => profile || pretrained;

/* ---------- Neuronales Netz (PyTorch → ONNX), läuft per onnxruntime im Browser ---------- */
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const net = { session: null, meta: null, error: "" };
let useNet = store("holo-maulwurf-usenet") !== false;
const netActive = () => !!net.session && (useNet || !model());
fetch("model-meta.json", { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).then((meta) => {
  if (!meta) return;
  net.meta = meta;
  return new Promise((res, rej) => { const s = document.createElement("script"); s.src = ORT_URL + "ort.min.js"; s.onload = res; s.onerror = rej; document.head.appendChild(s); })
    .then(() => { window.ort.env.wasm.wasmPaths = ORT_URL; window.ort.env.wasm.numThreads = 1; return window.ort.InferenceSession.create("model.onnx?v=" + encodeURIComponent(meta.name || "")); })
    .then((s) => { net.session = s; if (!audio.onTap) home(); });
}).catch((e) => { net.error = "Netz konnte nicht geladen werden"; });
// Same preprocessing as training/tapdata.py
function netInput(data, m) {
  const { start, length } = net.meta, x = data[0];
  let mx = 1e-6, peak = 0;
  for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
  for (let i = start; i < start + length; i++) mx = Math.max(mx, Math.abs(x[i]));
  const wave = new Float32Array(length);
  for (let i = 0; i < length; i++) wave[i] = x[start + i] / mx;
  const has = m && m.peak > 0 ? 1 : 0, d = has ? m.dir : [0, 0, 0];
  const extra = Float32Array.from([Math.log10(peak + 1e-4), has * Math.log10((has ? m.peak : 0) + 1e-3), d[0], d[1], d[2], has]);
  return { wave, extra };
}
audio.pre = async (data, meta) => {
  if (!net.session) return;
  const { wave, extra } = netInput(data, meta.time != null ? motionAround(meta.time) : null);
  const out = await net.session.run({ wave: new ort.Tensor("float32", wave, [1, 1, wave.length]), extra: new ort.Tensor("float32", extra, [1, extra.length]) });
  const l = Array.from(out.logits.data), mx = Math.max(...l), ex = l.map((v) => Math.exp(v - mx)), s = ex.reduce((p, q) => p + q, 0);
  meta.net = ex.map((v) => v / s);
};
// Decision for one detected sound: neural net if active, otherwise calibration model.
function judge(a, meta) {
  if (netActive() && meta && meta.net) {
    const p = meta.net, k = p.indexOf(Math.max(...p));
    if (k === 4) return { zone: null, why: "Netz: kein Klopfen", margin: p[4] };
    if (p[k] < (net.meta.threshold || 0.6)) return { zone: null, why: "Netz unsicher", margin: p[k] };
    return { zone: k, margin: p[k] };
  }
  if (!model()) return null;
  if (a.quality !== "ok") return { zone: null, why: QUALITY[a.quality] };
  return classify(model(), a);
}
const usable = (a) => a.quality !== "weak";
const feats = (data, meta) => { const m = model(); return analyse(data, meta, m ? m.motion : motion.ok, m ? !!m.mono : false); };
// Raw audio of this session (memory only): calibration knocks for augmentation, plus everything for export.
const session = { calAudio: [], blocks: [], log: [] };
let nextId = Date.now();
function toB64(f32) { // float audio → 16-bit PCM → base64 (compact for export)
  const i16 = new Int16Array(f32.length); for (let i = 0; i < f32.length; i++) i16[i] = Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32767)));
  let bin = ""; const u8 = new Uint8Array(i16.buffer); for (let i = 0; i < u8.length; i += 8192) bin += String.fromCharCode.apply(null, u8.subarray(i, i + 8192));
  return btoa(bin);
}
/* ---------- Gesammelte Daten: dauerhaft im Browser (IndexedDB), exportierbar ---------- */
let dbp = null;
function idb() {
  if (!dbp) dbp = new Promise((res, rej) => {
    const r = indexedDB.open("holo-daten", 1);
    r.onupgradeneeded = () => r.result.createObjectStore("events", { keyPath: "id" });
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
  return dbp;
}
function idbDo(mode, fn) {
  return idb().then((d) => new Promise((res, rej) => {
    const tx = d.transaction("events", mode); const req = fn(tx.objectStore("events"));
    tx.oncomplete = () => res(req && req.result); tx.onerror = () => rej(tx.error);
  }));
}
const idbPut = (rec) => idbDo("readwrite", (st) => st.put(rec)).catch(() => {});
const idbAll = () => idbDo("readonly", (st) => st.getAll()).catch(() => []);
const idbCount = () => idbDo("readonly", (st) => st.count()).catch(() => 0);
const idbDel = (id) => idbDo("readwrite", (st) => st.delete(id)).catch(() => {});
const idbClear = () => idbDo("readwrite", (st) => st.clear()).catch(() => {});
let tableName = store("holo-maulwurf-table") || "";

// label: 0..3 = Ecke, -1 = kein Klopfen (Störgeräusch)
function logAudio(kind, label, data, a, extra = {}) {
  const rec = { id: nextId++, t: new Date().toISOString(), kind, label, table: tableName, sr: audio.sr, mics: audio.ch,
    quality: a.quality, snr: a.snr, motion: a.m && { peak: a.m.peak, noise: a.m.noise, dir: a.m.dir },
    bg: a.bg ? a.bg.map((c) => ({ be: c.be.map((v) => +v.toPrecision(4)), env: c.env.map((v) => +v.toPrecision(4)) })) : null,
    ch0: toB64(data[0]), ch1: audio.ch > 1 ? toB64(data[1]) : null, ...extra };
  idbPut(rec);
  return rec;
}

function setHoles(fn) { holes.forEach((h, i) => fn(h, i)); }
function clearHoles() { setHoles((h) => h.classList.remove("target", "up", "bonk")); }
function flash(i, cls) { const h = holes[i]; h.classList.add(cls); setTimeout(() => h.classList.remove(cls), 260); }
let onHole = null;
holes.forEach((h, i) => h.addEventListener("pointerdown", (e) => { e.preventDefault(); onHole?.(i); }));
async function keepAwake() { try { await navigator.wakeLock?.request("screen"); } catch (e) {} }
function stopAll() { audio.onTap = null; audio.loose = false; onHole = null; clearHoles(); }
const pct = (v) => Math.round(v * 100) + " %";
const QUALITY = { sustained: "klang nicht wie Klopfen", weak: "zu leise", clipped: "zu laut", masked: "geht im Hintergrund unter" };
function save(p) { profile = p; store(PROFILE_KEY, p); }
function busy(fn) { center.innerHTML = "<h1>Rechne…</h1>"; setTimeout(fn, 30); }

function meterLoop() {
  const el = $("#lvl"); if (!el) return;
  el.style.width = Math.min(100, Math.max(0, (10 * Math.log10(audio.level + 1e-10) + 70) * 1.6)) + "%";
  requestAnimationFrame(meterLoop);
}

/* ---------- Start ---------- */
let micError = "";
function home() {
  stopAll();
  const evals = store(EVALS_KEY) || [];
  const status = profile
    ? `Eigene Kalibrierung: ${pct(profile.loo)} · ${profile.neg.length} Störgeräusche gelernt · Vibration ${profile.gate ? "aktiv" : profile.motion ? "hilft hier nicht" : "aus"}`
    : net.session ? "" : pretrained ? `Vortrainiertes Modell ${pretrained.name || ""} aktiv (${pretrained.trainedOn || "?"} Aufnahmen). Eigene Kalibrierung optional.` : "Noch nicht kalibriert";
  center.innerHTML = `
    <h1>Kalibrier-Labor</h1>
    <p>iPhone flach auf den Tisch legen und nicht mehr bewegen. Geklopft wird <b>neben</b> das Handy in die vier Ecken.</p>
    ${net.session ? `<p class="small">Neuronales Netz vom ${net.meta.name} geladen (${net.meta.trainedOn} Aufnahmen) · ${netActive() ? "<b>aktiv</b>" : "aus"}</p>` : net.error ? `<p class="small">${net.error}</p>` : ""}
    <p class="small">${status}</p>
    ${evals.length ? `<p class="small">Letzte Tests: ${evals.slice(-4).map((e) => `${e.condition} <b>${pct(e.overall)}</b>`).join(" · ")}</p>` : ""}
    <div class="btns">
      <button class="b" id="bCollect">Daten sammeln</button>
      ${model() || net.session ? '<button class="b alt" id="bEval">Genauigkeitstest</button><button class="b alt" id="bLive">Live-Test</button>' : ""}
      ${net.session && model() ? `<button class="b alt" id="bNet">Erkennung: ${netActive() ? "Netz" : "Kalibrierung"}</button>` : ""}
      <button class="b alt" id="bCal">${profile ? "Kalibrierung" : "Kalibrieren"}</button>
      <button class="b alt" id="bSens">Empfindlichkeit: ${audio.sens}</button>
      <button class="b alt" id="bExp">Exportieren</button>
      ${model() || net.session ? '<button class="b alt" id="bGame">Spiel</button>' : ""}
      ${profile && pretrained ? '<button class="b alt" id="bUsePre">Vortrainiertes Modell nutzen</button>' : ""}
    </div>
    <p class="small" id="dcount"></p>
    <p class="status err">${micError}</p>`;
  $("#bCal").onclick = () => withMic(profile ? summary : () => calibrate());
  $("#bEval") && ($("#bEval").onclick = () => withMic(evaluation));
  $("#bLive") && ($("#bLive").onclick = () => withMic(live));
  $("#bExp").onclick = () => exportData($("#bExp"));
  $("#bCollect").onclick = collectSetup;
  $("#bNet") && ($("#bNet").onclick = () => { useNet = !netActive(); store("holo-maulwurf-usenet", useNet); home(); });
  $("#bUsePre") && ($("#bUsePre").onclick = () => { profile = null; store(PROFILE_KEY, null); home(); });
  idbCount().then((n) => { const el = $("#dcount"); if (el && n) el.innerHTML = `${n} Aufnahmen gesammelt · <button class="lnk" id="bClr">löschen</button>`;
    const c = $("#bClr"); if (c) c.onclick = () => { c.outerHTML = '<button class="lnk" id="bClr2">Wirklich alle löschen?</button>'; $("#bClr2").onclick = () => idbClear().then(home); }; });
  $("#bGame") && ($("#bGame").onclick = () => withMic(game));
  $("#bSens").onclick = () => {
    const ks = Object.keys(SENS); audio.sens = ks[(ks.indexOf(audio.sens) + 1) % ks.length];
    store("holo-maulwurf-sens", audio.sens); home();
  };
}
function withMic(next) {
  const m = startMotion(); // ask for motion permission inside the same tap
  startMic().then(() => m).then(() => { micError = ""; keepAwake(); next(); })
    .catch((e) => { micError = e && e.name === "NotAllowedError" ? "Mikrofon nicht erlaubt. Bitte in den Safari-Einstellungen erlauben." : (e.message || "Mikrofon nicht verfügbar."); home(); });
}

/* ---------- Kalibrieren ---------- */
// Record taps for the given zones; keeps the other zones' taps and the learned noises.
function calibrate(todo = [0, 1, 2, 3]) {
  stopAll();
  const partial = profile && todo.length < 4;
  const useMotion = partial ? profile.motion : motion.ok;
  const samples = partial ? profile.raw.filter((s) => !s.aug && !todo.includes(s.zone)) : [];
  const negatives = partial ? profile.rawNeg : [];
  let step = 0, paused = false;
  const zone = () => todo[step];
  const draw = (msg = "", err = false) => {
    setHoles((h, i) => h.classList.toggle("target", i === zone()));
    const got = samples.filter((s) => s.zone === zone()).length;
    center.innerHTML = `
      <h1>Klopf: ${ZONES[zone()]}</h1>
      <p>${PER_ZONE}× mit dem Fingerknöchel auf den Tisch klopfen, <b>${ZONES[zone()]}</b> neben dem Handy. Etwa gleich fest, Stelle leicht variieren.</p>
      <div class="dots">${Array.from({ length: PER_ZONE }, (_, i) => `<b class="${i < got ? "on" : ""}"></b>`).join("")}</div>
      <div class="meter"><div id="lvl"></div></div>
      <p class="status ${err ? "err" : ""}">${msg}</p>
      <div class="btns"><button class="b alt" id="bUndo">Rückgängig</button><button class="b alt" id="bBack">Abbrechen</button></div>`;
    $("#bBack").onclick = home;
    $("#bUndo").onclick = () => { const k = samples.findLastIndex((s) => s.zone === zone()); if (k >= 0) samples.splice(k, 1); draw("Letzter Klopfer entfernt"); };
    meterLoop();
  };
  audio.onTap = (data, meta) => {
    if (paused) return;
    const a = analyse(data, meta, useMotion);
    if (a.quality === "clipped") return draw("Zu fest, klopf etwas leichter.", true);
    if (a.quality === "weak") return draw("Zu leise, klopf etwas fester.", true);
    if (a.quality === "masked") return draw("Zu leise gegen den Hintergrund. Fester klopfen oder leiser machen.", true);
    if (a.quality === "sustained") return draw("Das klang nicht wie ein Klopfen. Kurz warten, dann nochmal.", true);
    const id = nextId++;
    samples.push({ zone: zone(), f: a.f, mpeak: a.mpeak, mnoise: a.mnoise, id });
    session.calAudio.push({ id, zone: zone(), data, m: a.m });
    logAudio("calibration", zone(), data, a, { background: "Kalibrierung" });
    flash(zone(), "hit");
    if (samples.filter((s) => s.zone === zone()).length < PER_ZONE) return draw("Gut!");
    if (step === todo.length - 1) { stopAll(); return busy(() => { save(train(withAugment(samples, useMotion), negatives, useMotion)); summary(); }); }
    step++; paused = true; draw("Weiter zur nächsten Ecke…");
    setTimeout(() => { paused = false; draw("Bereit"); }, 1100);
  };
  draw("Bereit");
}

// Adds background-mixed copies of this session's calibration knocks (if background was recorded).
function withAugment(samples, useMotion) {
  const real = samples.filter((s) => !s.aug);
  const ids = new Set(real.map((s) => s.id));
  const taps = session.calAudio.filter((t) => ids.has(t.id));
  return real.concat(augment(taps, session.blocks, useMotion));
}

// Calibration result with per-zone check (like Holo's consistency check)
function summary() {
  stopAll();
  if (!profile) return calibrate();
  const acc = profile.zoneAcc;
  const weakest = [0, 1, 2, 3].sort((a, b) => acc[a] - acc[b])[0];
  const vib = profile.gate ? "Vibrationssensor aktiv: Geräusche ohne Tisch-Erschütterung werden ignoriert."
    : profile.motion ? "Vibrationssensor misst bei deinen Klopfern kaum etwas, er wird nicht als Filter benutzt."
    : "Vibrationssensor nicht verfügbar (Bewegungszugriff erlaubt?).";
  center.innerHTML = `<h1>Kalibrierung ${pct(profile.loo)}</h1>
    <p class="small">${ZONES.map((z, i) => `${z}: <b>${pct(acc[i])}</b>`).join(" · ")}</p>
    <p class="small">${vib}</p>
    <p class="small">${profile.neg.length ? `${profile.neg.length} Störgeräusche gelernt, ${profile.raw.filter((s) => s.aug).length} Klopfer mit Hintergrund gemischt trainiert.` : "<b>Nächster Schritt:</b> Störgeräusche aufnehmen (Musik an, reden), damit sie ignoriert werden."}</p>
    <div class="btns">
      <button class="b" id="bNoise">${profile.neg.length ? "Mehr Störgeräusche" : "Störgeräusche aufnehmen"}</button>
      ${acc[weakest] < 0.9 ? `<button class="b alt" id="bRedo">„${ZONES[weakest]}“ wiederholen</button>` : ""}
      <button class="b alt" id="bEval">Genauigkeitstest</button>
      <button class="b alt" id="bAll">Alles neu</button>
      <button class="b alt" id="bHome">Zurück</button>
    </div>`;
  $("#bNoise").onclick = recordNoise; $("#bEval").onclick = evaluation; $("#bHome").onclick = home;
  $("#bAll").onclick = () => { profile = null; calibrate(); };
  $("#bRedo") && ($("#bRedo").onclick = () => calibrate([weakest]));
}

// Negative examples (Holo: rejection training). Loose trigger so borderline sounds get learned.
function recordNoise() {
  stopAll();
  const negs = profile.rawNeg.slice(), SECS = 20; let heard = 0;
  const t0 = performance.now();
  audio.loose = true; audio.collect = [];
  center.innerHTML = `<h1>Störgeräusche</h1>
    <p>Mach jetzt Musik an, rede, raschel, tipp aufs Handy. <b>Nicht auf den Tisch klopfen!</b></p>
    <div class="meter"><div id="lvl"></div></div>
    <p class="status" id="nstat">Noch ${SECS} s · 0 Geräusche</p>
    <div class="btns"><button class="b alt" id="bStop">Fertig</button></div>`;
  meterLoop();
  audio.onTap = (data, meta) => {
    const a = feats(data, meta); heard++;
    if (a.quality !== "clipped" && a.quality !== "weak") negs.push({ zone: -1, f: a.f, mpeak: a.mpeak, mnoise: a.mnoise });
    logAudio("noise", -1, data, a, { background: "Störgeräusche" });
  };
  const finish = () => {
    clearInterval(iv); stopAll();
    session.blocks = session.blocks.concat(audio.collect || []).slice(-80); audio.collect = null;
    busy(() => { save(train(withAugment(profile.raw, profile.motion), negs.slice(-100), profile.motion)); summary(); });
  };
  const iv = setInterval(() => {
    const left = Math.ceil(SECS - (performance.now() - t0) / 1000);
    const el = $("#nstat"); if (el) el.textContent = `Noch ${Math.max(0, left)} s · ${heard} Geräusche`;
    if (left <= 0) finish();
  }, 250);
  $("#bStop").onclick = finish;
}

/* ---------- Genauigkeitstest (like Holo's held-out 60-tap evaluation) ---------- */
function evaluation() {
  stopAll();
  center.innerHTML = `<h1>Genauigkeitstest</h1>
    <p>${EVAL_PER_ZONE} Klopfer pro Ecke, insgesamt ${EVAL_PER_ZONE * 4}. Danach 15 Sekunden <b>nicht</b> klopfen, um Fehlauslöser zu zählen. Was läuft im Hintergrund?</p>
    <div class="btns"><button class="b" data-c="ruhig">Ruhig</button><button class="b" data-c="Reden">Reden</button><button class="b" data-c="Musik">Musik</button><button class="b" data-c="Musik + Reden">Musik + Reden</button></div>
    <div class="btns"><button class="b alt" id="bBack">Zurück</button></div>`;
  $("#bBack").onclick = home;
  center.querySelectorAll("[data-c]").forEach((b) => (b.onclick = () => runEval(b.dataset.c)));
}
function runEval(condition) {
  const conf = [0, 1, 2, 3].map(() => [0, 0, 0, 0, 0]); // rows: target, cols: zones + rejected
  const events = []; let zone = 0, n = 0, armed = false;
  const draw = (msg = "") => {
    setHoles((h, i) => h.classList.toggle("target", i === zone));
    center.innerHTML = `<h1>Test: ${ZONES[zone]}</h1>
      <p class="small">${condition} · Klopfer ${n}/${EVAL_PER_ZONE}</p>
      <div class="meter"><div id="lvl"></div></div>
      <p class="status" id="res">${msg}</p>
      <div class="btns"><button class="b alt" id="bBack">Abbrechen</button></div>`;
    $("#bBack").onclick = home; meterLoop();
  };
  const arm = () => { armed = false; draw("Gleich geht's los…"); setTimeout(() => { armed = true; draw("Bereit, klopfen!"); }, 900); };
  audio.onTap = (data, meta) => {
    if (!armed) return;
    const a = feats(data, meta);
    if (a.quality === "weak") return;
    const r = judge(a, meta) || { zone: null, why: "kein Modell" };
    conf[zone][r.zone ?? 4]++; n++; logAudio("test", zone, data, a, { background: condition, predicted: r.zone });
    events.push({ target: zone, got: r.zone, why: r.why || null, quality: a.quality, f: a.f, mpeak: a.mpeak, snr: a.snr });
    if (r.zone === zone) flash(zone, "hit"); else if (r.zone !== null) flash(r.zone, "wrong");
    if (n >= EVAL_PER_ZONE) {
      if (zone === 3) return quietPhase();
      zone++; n = 0; return arm();
    }
    draw(r.zone === zone ? "Richtig" : r.zone === null ? "Abgelehnt: " + r.why : "Falsch: " + ZONES[r.zone]);
  };
  function quietPhase() {
    audio.onTap = null; clearHoles();
    let falseHits = 0, heard = 0; const t0 = performance.now(), SECS = 15;
    center.innerHTML = `<h1>Jetzt nicht klopfen</h1><p>Hintergrund (${condition}) einfach weiterlaufen lassen.</p>
      <div class="meter"><div id="lvl"></div></div><p class="status" id="qs">Noch ${SECS} s</p>`;
    meterLoop();
    audio.onTap = (data, meta) => {
      const a = feats(data, meta); heard++;
      const r = judge(a, meta) || { zone: null };
      if (r.zone !== null) { falseHits++; flash(r.zone, "wrong"); }
      logAudio("test-pause", -1, data, a, { background: condition, predicted: r.zone });
      events.push({ target: -1, got: r.zone, why: r.why || null, quality: a.quality, f: a.f, mpeak: a.mpeak, snr: a.snr });
    };
    const iv = setInterval(() => {
      const left = Math.ceil(SECS - (performance.now() - t0) / 1000);
      const el = $("#qs"); if (el) el.textContent = `Noch ${Math.max(0, left)} s · Fehlauslöser: ${falseHits}`;
      if (left <= 0) { clearInterval(iv); stopAll(); finishEval(falseHits, heard); }
    }, 250);
  }
  function finishEval(falseHits, heard) {
    const correct = conf.reduce((s, row, z) => s + row[z], 0), total = conf.flat().reduce((a, b) => a + b, 0);
    const res = {
      date: new Date().toISOString(), condition, sens: audio.sens, mics: audio.ch, motionGate: !!(model() && model().gate), recognizer: netActive() ? "Netz " + net.meta.name : profile ? "eigene Kalibrierung" : "vortrainiert",
      overall: correct / Math.max(1, total), perZone: conf.map((row, z) => row[z] / Math.max(1, row.reduce((a, b) => a + b, 0))),
      rejected: conf.reduce((s, row) => s + row[4], 0), confusion: conf, falseHits, quietEvents: heard,
      calibration: model() ? { loo: model().loo, zoneAcc: model().zoneAcc || [0, 0, 0, 0], negatives: model().neg.length } : null,
    };
    const evals = (store(EVALS_KEY) || []).concat([{ ...res }]).slice(-10); store(EVALS_KEY, evals);
    lastEvalEvents = events;
    showEval(res);
  }
  arm();
}
let lastEvalEvents = [];
function showEval(res) {
  const head = ZONES.map((z) => `<th>${z.replace(" ", "<br>")}</th>`).join("");
  const rows = res.confusion.map((row, z) => `<tr><th>${ZONES[z]}</th>${row.map((v, c) => `<td class="${c === z ? "diag" : v ? "off" : ""}">${v}</td>`).join("")}</tr>`).join("");
  center.innerHTML = `<h1>${pct(res.overall)} richtig</h1>
    <p class="small">${res.condition} · ${res.rejected} abgelehnt · ${res.falseHits} Fehlauslöser in 15 s Pause · Ziel laut Holo: ≥ 80 %</p>
    <div class="tbl"><table><tr><th>getippt ↓ / erkannt →</th>${head}<th>abge&shy;lehnt</th></tr>${rows}</table></div>
    <div class="btns"><button class="b" id="bCopy">Bericht kopieren</button><button class="b alt" id="bAgain">Nochmal testen</button><button class="b alt" id="bHome">Menü</button></div>
    <p class="status" id="cst"></p>`;
  $("#bAgain").onclick = evaluation; $("#bHome").onclick = home;
  $("#bCopy").onclick = () => copyText(JSON.stringify(res));
}
function copyText(t) {
  const done = (ok) => { const el = $("#cst"); if (el) el.textContent = ok ? "Kopiert. Füg es einfach im Chat ein." : "Kopieren ging nicht."; };
  navigator.clipboard?.writeText(t).then(() => done(true), () => done(false)) ?? done(false);
}

/* ---------- Live-Test (learns from corrections) ---------- */
function live() {
  stopAll();
  let last = null;
  center.innerHTML = `<h1>Live-Test</h1><p>Klopf irgendwo. Die erkannte Ecke leuchtet grün. Falsch erkannt? Tipp danach auf das richtige Loch, dann lernt es dazu.</p>
    <div class="meter"><div id="lvl"></div></div><p class="status" id="res">Warte auf Klopfen…</p>
    <div class="btns"><button class="b alt" id="bBack">Zurück</button></div>`;
  $("#bBack").onclick = home; meterLoop();
  audio.onTap = (data, meta) => {
    const a = feats(data, meta);
    if (a.quality === "weak" || (!netActive() && a.quality !== "ok")) { $("#res").textContent = "Ignoriert: " + QUALITY[a.quality]; return; }
    last = a;
    const r = judge(a, meta);
    if (r.zone === null) { $("#res").textContent = "Abgelehnt: " + r.why; return; }
    flash(r.zone, "hit");
    $("#res").textContent = `${ZONES[r.zone]} · Sicherheit ${Math.min(100, Math.round(r.margin * 100))} %`;
  };
  onHole = (z) => {
    if (!last || !profile) return;
    save(train(profile.raw.concat([{ zone: z, f: last.f, mpeak: last.mpeak, mnoise: last.mnoise, id: nextId++ }]), profile.rawNeg, profile.motion));
    last = null; flash(z, "hit"); $("#res").textContent = `Gelernt: das war ${ZONES[z]}`;
  };
}

/* ---------- Daten-Export (für spätere Auswertung / ML-Training) ---------- */
async function exportData(btn) {
  if (btn) btn.textContent = "Exportiere…";
  const events = await idbAll();
  const data = { app: "tisch-maulwurf", version: 3, exported: new Date().toISOString(),
    note: "label: 0 hinten links, 1 hinten rechts, 2 vorne links, 3 vorne rechts, -1 kein Klopfen. ch0/ch1: 85-ms-Fenster, 16-bit PCM, base64. Klopfbeginn bei Sample 256.",
    profile: profile && { motion: profile.motion, gate: profile.gate, loo: profile.loo, zoneAcc: profile.zoneAcc },
    evaluations: store(EVALS_KEY) || [], events };
  const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
  const file = new File([blob], `holo-daten-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.json`, { type: "application/json" });
  if (btn) btn.textContent = `Exportieren (${events.length})`;
  // iPhone: share sheet (AirDrop, Dateien sichern, Mail…); otherwise a normal download
  if (navigator.canShare?.({ files: [file] })) { try { await navigator.share({ files: [file], title: "Holo-Daten" }); return; } catch (e) { if (e.name === "AbortError") return; } }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(file); a.download = file.name;
  document.body.appendChild(a); a.click(); a.remove();
}

/* ---------- Daten sammeln: klopfen, dann beschriften ---------- */
function collectSetup() {
  stopAll();
  const bgs = ["ruhig", "Reden", "Musik", "Musik + Reden", "Sonstiges"];
  const lastBg = store("holo-maulwurf-bg") || "ruhig";
  center.innerHTML = `<h1>Daten sammeln</h1>
    <p class="small">Jeder Klopfer wird mit Rohaudio gespeichert und bleibt auch nach dem Schließen erhalten. Je mehr verschiedene Tische, Positionen und Geräusche, desto besser.</p>
    <label class="fld">Tisch / Ort<input id="iTable" value="${tableName.replace(/"/g, "&quot;")}" placeholder="z. B. Schreibtisch Holz" autocomplete="off"></label>
    <p class="small">Hintergrund gerade:</p>
    <div class="btns chips">${bgs.map((b) => `<button class="b alt${b === lastBg ? " sel" : ""}" data-bg="${b}">${b}</button>`).join("")}</div>
    <div class="btns"><button class="b" id="bFree">Frei klopfen + beschriften</button><button class="b" id="bGuided">Geführt (App sagt die Ecke)</button></div>
    <div class="btns"><button class="b alt" id="bOnlyNoise">Nur Geräusche (kein Klopfen)</button><button class="b alt" id="bBack">Zurück</button></div>`;
  let bg = lastBg;
  center.querySelectorAll("[data-bg]").forEach((b) => (b.onclick = () => {
    bg = b.dataset.bg; store("holo-maulwurf-bg", bg);
    center.querySelectorAll("[data-bg]").forEach((x) => x.classList.toggle("sel", x === b));
  }));
  const go = (mode) => { tableName = $("#iTable").value.trim() || "unbenannt"; store("holo-maulwurf-table", tableName); withMic(() => collect(mode, bg)); };
  $("#bFree").onclick = () => go("frei"); $("#bGuided").onclick = () => go("geführt"); $("#bOnlyNoise").onclick = () => go("geräusche");
  $("#bBack").onclick = home;
}
function collect(mode, bg) {
  stopAll();
  let pending = null, target = -1, saved = 0, lastRec = null;
  const counts = [0, 0, 0, 0, 0];
  const help = { frei: "Klopf in eine Ecke. Danach auf das Loch tippen, wo du geklopft hast, oder „Kein Klopfen“.",
    "geführt": "Klopf in die markierte Ecke. Falsch? Tipp aufs richtige Loch oder „Kein Klopfen“.",
    "geräusche": "Nicht klopfen. Alles, was jetzt erkannt wird, wird als Störgeräusch gespeichert." }[mode];
  if (mode === "geräusche") audio.loose = true;
  const pick = () => { let z; do { z = Math.floor(Math.random() * 4); } while (z === target); target = z; setHoles((h, i) => h.classList.toggle("target", i === z)); };
  const draw = (msg = "") => {
    center.innerHTML = `<h1>${mode === "geführt" ? "Klopf: " + ZONES[target] : mode === "frei" ? (pending ? "Was war das?" : "Klopf irgendwo") : "Geräusche sammeln"}</h1>
      <p class="small">${help}</p>
      <div class="meter"><div id="lvl"></div></div>
      <p class="status" id="res">${msg}</p>
      <p class="small">${tableName} · ${bg} · gespeichert: ${saved} (${ZONES.map((z, i) => counts[i]).join("/")} Ecken, ${counts[4]} Geräusche)</p>
      <div class="btns">${pending && pending.r ? `<button class="b" id="bOk">Stimmt: ${pending.r.zone === null ? "kein Klopfen" : ZONES[pending.r.zone]}</button>` : ""}
        ${pending || (mode !== "frei" && lastRec) ? '<button class="b alt" id="bNo">Kein Klopfen</button>' : ""}
        ${pending ? '<button class="b alt" id="bSkip">Verwerfen</button>' : ""}
        <button class="b alt" id="bEnd">Fertig</button></div>`;
    $("#bEnd").onclick = () => { stopAll(); collectSetup(); };
    $("#bNo") && ($("#bNo").onclick = () => label(-1));
    $("#bOk") && ($("#bOk").onclick = () => label(pending.r.zone === null ? -1 : pending.r.zone));
    $("#bSkip") && ($("#bSkip").onclick = () => { pending = null; draw("Verworfen"); });
    meterLoop();
  };
  const predict = (a, meta) => judge(a, meta);
  const label = (z) => {
    if (pending) { // free mode: save with the chosen label
      const { data, a, r, meta } = pending; pending = null;
      lastRec = logAudio("collect", z, data, a, { background: bg, mode, predicted: r ? r.zone : null, net: meta.net || null }); saved++; counts[z < 0 ? 4 : z]++;
      if (z >= 0) flash(z, "hit");
      return draw(`Gespeichert: ${z < 0 ? "kein Klopfen" : ZONES[z]}`);
    }
    if (lastRec && mode !== "frei") { // correction of the last auto-labelled event
      const old = lastRec.label; lastRec.label = z; lastRec.corrected = true; idbPut(lastRec);
      counts[old < 0 ? 4 : old]--; counts[z < 0 ? 4 : z]++; lastRec = null;
      return draw(`Korrigiert: ${z < 0 ? "kein Klopfen" : ZONES[z]}`);
    }
  };
  onHole = (z) => label(z);
  audio.onTap = (data, meta) => {
    const a = feats(data, meta);
    if (a.quality === "weak" && mode !== "geräusche") return;
    const r = predict(a, meta);
    const guess = r ? (r.zone === null ? "abgelehnt: " + r.why : "erkannt: " + ZONES[r.zone]) : "";
    if (mode === "frei") { if (pending) return; pending = { data, a, r, meta }; if (r && r.zone !== null) flash(r.zone, "hit"); return draw(guess ? `Modell sagt: ${guess}. Stimmt das? Sonst aufs richtige Loch tippen.` : "Erkannt. Welche Ecke? Aufs Loch tippen."); }
    const z = mode === "geführt" ? target : -1;
    lastRec = logAudio("collect", z, data, a, { background: bg, mode, predicted: r ? r.zone : null, net: meta.net || null }); saved++; counts[z < 0 ? 4 : z]++;
    if (mode === "geführt") { flash(z, r && r.zone !== null && r.zone !== z ? "wrong" : "hit"); pick(); }
    draw(`Gespeichert${guess ? " · Modell: " + guess : ""}`);
  };
  if (mode === "geführt") pick(); else clearHoles();
  draw("Bereit");
}

/* ---------- Spiel ---------- */
function game() {
  stopAll();
  const LEN = 45000; let score = 0, misses = 0, up = -1, upUntil = 0, timer = null, ended = false;
  const t0 = performance.now();
  center.innerHTML = `<div class="hud"><span><small>Punkte</small><b id="sc">0</b></span><span><small>Zeit</small><b id="tm">45</b></span><span><small>Daneben</small><b id="ms">0</b></span></div>
    <p class="small">Klopf neben das Handy, wo der Maulwurf ist.</p>
    <div class="btns"><button class="b alt" id="bStop">Beenden</button></div>`;
  $("#bStop").onclick = () => end();
  const show = () => {
    if (ended) return;
    let z; do { z = Math.floor(Math.random() * 4); } while (z === up);
    up = z; holes[z].classList.remove("bonk"); holes[z].classList.add("up");
    const stay = Math.max(800, 1700 - (performance.now() - t0) / LEN * 900);
    upUntil = performance.now() + stay;
    timer = setTimeout(() => { if (up === z) { holes[z].classList.remove("up"); up = -1; misses++; $("#ms").textContent = misses; } next(); }, stay);
  };
  const next = () => { if (!ended) timer = setTimeout(show, 300 + Math.random() * 400); };
  const hitZone = (z) => {
    if (ended) return;
    if (z === up) {
      clearTimeout(timer); holes[z].classList.add("bonk"); holes[z].classList.remove("up");
      score += 1 + (upUntil - performance.now() > 500 ? 1 : 0); up = -1;
      $("#sc").textContent = score; flash(z, "hit"); next();
    } else { misses++; $("#ms").textContent = misses; flash(z, "wrong"); }
  };
  audio.onTap = (data, meta) => {
    const a = feats(data, meta); if (!usable(a)) return;
    const r = judge(a, meta); if (r && r.zone !== null) hitZone(r.zone);
  };
  const tick = setInterval(() => {
    const left = Math.max(0, Math.ceil((LEN - (performance.now() - t0)) / 1000));
    const el = $("#tm"); if (el) el.textContent = left;
    if (left <= 0) end();
  }, 200);
  function end() {
    if (ended) return; ended = true; clearInterval(tick); clearTimeout(timer); stopAll();
    const best = Math.max(score, store("holo-maulwurf-best") || 0); store("holo-maulwurf-best", best);
    center.innerHTML = `<h1>${score} Punkte</h1><p>Daneben: ${misses} · Rekord: ${best}</p>
      <div class="btns"><button class="b" id="bAgain">Nochmal</button><button class="b alt" id="bHome">Menü</button></div>`;
    $("#bAgain").onclick = game; $("#bHome").onclick = home;
  }
  next();
}

home();
