"use strict";
/* ---------------------------------------------------------------
   Holo-Klopferkennung fürs iPhone (vereinfachte JS-Version von HoloCore).
   Mikrofon → Onset-Erkennung → ~85-ms-Fenster → Hintergrund-Abzug
   → Merkmale (Frequenzbänder, Zeitverlauf, Stereo, Bewegungssensor)
   → lineares Zonenmodell + Nächste-Beispiele-Prüfung mit Ablehnung
   --------------------------------------------------------------- */
const ZONES = ["hinten links", "hinten rechts", "vorne links", "vorne rechts"];
const WIN = 4096, PRE = 256, HOP = 128, NB = 20;

function store(k, v) {
  try {
    if (v === undefined) return JSON.parse(localStorage.getItem(k));
    localStorage.setItem(k, JSON.stringify(v));
  } catch (e) { return null; }
}

/* ---------- FFT (radix-2, in place) ---------- */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}

/* Band energies of x[start .. start+len), zero-padded to WIN for resolution. */
const FEAT_START = PRE - 64, FEAT_LEN = 2048;
function spectrum(x, sr, start = FEAT_START, len = FEAT_LEN) {
  const re = new Float64Array(WIN), im = new Float64Array(WIN);
  for (let i = 0; i < len; i++) re[i] = x[start + i] * (i < len * 0.75 ? 1 : 0.5 + 0.5 * Math.cos(Math.PI * (i - len * 0.75) / (len * 0.25)));
  fft(re, im);
  const lo = 80, hi = Math.min(16000, sr / 2 - 100), be = new Array(NB).fill(0);
  for (let b = 0; b < NB; b++) {
    const f0 = lo * (hi / lo) ** (b / NB), f1 = lo * (hi / lo) ** ((b + 1) / NB);
    const k0 = Math.max(1, Math.floor(f0 * WIN / sr)), k1 = Math.max(k0 + 1, Math.floor(f1 * WIN / sr));
    for (let k = k0; k < k1; k++) be[b] += re[k] ** 2 + im[k] ** 2;
  }
  return be;
}
/* 8-part energy envelope over the whole window. */
function envelope(x) {
  const seg = WIN / 8, env = [];
  for (let s = 0; s < 8; s++) { let e = 0; for (let i = s * seg; i < (s + 1) * seg; i++) e += x[i] * x[i]; env.push(e); }
  return env;
}
function blockStats(d, sr) { return [0, 1].map((c) => ({ be: spectrum(d[c], sr, 0, FEAT_LEN), env: envelope(d[c]) })); }

/* ---------- Audio capture, background spectrum, onset detection ---------- */
const SENS = { niedrig: 24, normal: 14, hoch: 8 };
const audio = {
  ctx: null, sr: 48000, on: false, ch: 1, floor: 1e-6, warm: 0, last: -1e9, t: 0,
  ring: null, ringPos: 0, capture: null, level: 0, onTap: null,
  hist: [0, 0, 0, 0, 0, 0], loose: false, sens: "normal",
  bgBuf: null, bgPos: 0, bg: null, // running background (music/speech) spectrum
  collect: null, // when an array: raw background blocks are kept for training augmentation
};

async function startMic() {
  if (audio.on) return true;
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Dieser Browser gibt kein Mikrofon frei (HTTPS nötig).");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: {
    echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: { ideal: 2 } } });
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  await ctx.resume();
  const src = ctx.createMediaStreamSource(stream);
  const node = ctx.createScriptProcessor(1024, 2, 1);
  const track = stream.getAudioTracks()[0];
  audio.ch = Math.min(2, track.getSettings?.().channelCount || 1);
  audio.ctx = ctx; audio.sr = ctx.sampleRate; audio.on = true;
  resetAudioState();
  node.onaudioprocess = (e) => process(e.inputBuffer);
  src.connect(node); node.connect(ctx.destination); // output stays silent
  audio.keep = { stream, src, node };
  return true;
}
function resetAudioState() {
  audio.warm = audio.sr * 0.75;
  audio.ring = [new Float32Array(PRE), new Float32Array(PRE)];
  audio.bgBuf = [new Float32Array(WIN), new Float32Array(WIN)];
  audio.bgPos = 0; audio.bg = null;
}

function process(buf) {
  const chs = [buf.getChannelData(0), buf.numberOfChannels > 1 && audio.ch > 1 ? buf.getChannelData(1) : buf.getChannelData(0)];
  const n = chs[0].length;
  for (let h = 0; h < n; h += HOP) {
    let e = 0;
    for (let i = h; i < h + HOP; i++) { const v = chs[0][i], w = chs[1][i]; e += Math.max(v * v, w * w); }
    e /= HOP; if (!isFinite(e)) e = 0;
    audio.level = audio.level * 0.8 + e * 0.2;
    const quietSince = audio.t - audio.last;
    if (audio.capture) {
      const c = audio.capture;
      for (let i = h; i < h + HOP && c.pos < WIN; i++, c.pos++) { c.data[0][c.pos] = chs[0][i]; c.data[1][c.pos] = chs[1][i]; }
      if (c.pos >= WIN) {
        audio.capture = null;
        // wait a little so the motion sensor samples around the knock have arrived
        setTimeout(() => {
          const go = () => audio.onTap?.(c.data, c.meta);
          audio.pre ? audio.pre(c.data, c.meta).then(go, go) : go(); // e.g. neural net inference
        }, 110);
      }
    } else {
      // A knock is a sudden jump: loud vs. the noise floor AND vs. the hops 3–6 back.
      // Speech and music mostly rise gradually, so they fail the second test.
      const k = audio.loose ? 5 : SENS[audio.sens] || 14;
      const prev = (audio.hist[0] + audio.hist[1] + audio.hist[2] + audio.hist[3]) / 4 + 1e-12;
      const trig = audio.warm <= 0 && e > audio.floor * k && e > 2e-6 && (audio.loose || e > prev * 6) && quietSince > audio.sr * 0.28;
      if (trig) {
        audio.last = audio.t;
        const data = [new Float32Array(WIN), new Float32Array(WIN)];
        for (let c = 0; c < 2; c++) for (let i = 0; i < PRE; i++) data[c][i] = audio.ring[c][(audio.ringPos + i) % PRE];
        const ago = ((n - h) / audio.sr) * 1000 + 25; // ms between this hop and "now" (+ output latency guess)
        audio.capture = { data, pos: PRE, meta: { time: performance.now() - ago, bg: audio.bg } };
        audio.bgPos = 0; // drop the half-filled background block, it may contain the knock
        h -= HOP; audio.t -= HOP; // re-run this hop into the capture
      } else if (quietSince > audio.sr * 0.28) {
        // adaptive noise floor: follow quiet quickly, loud slowly
        audio.floor = e < audio.floor ? audio.floor * 0.9 + e * 0.1 : audio.floor * 0.995 + e * 0.005;
        audio.floor = Math.max(audio.floor, 1e-9);
      }
      // background spectrum from blocks well away from knocks
      if (!audio.capture && quietSince > audio.sr * 0.15) {
        for (let i = h; i < h + HOP && audio.bgPos < WIN; i++, audio.bgPos++) { audio.bgBuf[0][audio.bgPos] = chs[0][i]; audio.bgBuf[1][audio.bgPos] = chs[1][i]; }
        if (audio.bgPos >= WIN) { updateBackground(); audio.bgPos = 0; }
      }
    }
    for (let i = h; i < h + HOP; i++) { audio.ring[0][audio.ringPos] = chs[0][i]; audio.ring[1][audio.ringPos] = chs[1][i]; audio.ringPos = (audio.ringPos + 1) % PRE; }
    audio.hist.shift(); audio.hist.push(e);
    audio.t += HOP; audio.warm -= HOP;
  }
}
function updateBackground() {
  if (audio.collect && audio.collect.length < 60) audio.collect.push([audio.bgBuf[0].slice(), audio.bgBuf[1].slice()]);
  const s = blockStats(audio.bgBuf, audio.sr);
  if (!audio.bg) { audio.bg = s; return; }
  // follow rising background fast (music starts), falling slower; new object so captures keep their snapshot
  const mix = (o, nv) => o.map((v, i) => (nv[i] > v ? 0.5 : 0.2) * nv[i] + (nv[i] > v ? 0.5 : 0.8) * v);
  audio.bg = s.map((sp, c) => ({ be: mix(audio.bg[c].be, sp.be), env: mix(audio.bg[c].env, sp.env) }));
}

/* ---------- Motion sensor: real knocks shake the table, music does not ---------- */
const motion = { ok: false, buf: [], last: null, asked: false };
function onMotion(e) {
  const a = e.accelerationIncludingGravity || e.acceleration;
  if (!a || a.x == null) return;
  const t = performance.now();
  if (motion.last) {
    motion.buf.push({ t, x: a.x - motion.last.x, y: a.y - motion.last.y, z: a.z - motion.last.z });
    if (motion.buf.length > 400) motion.buf.shift();
    motion.ok = true;
  }
  motion.last = { x: a.x, y: a.y, z: a.z };
}
// must be called from a tap (iOS asks for permission)
function startMotion() {
  if (motion.asked) return Promise.resolve(motion.ok);
  motion.asked = true;
  const listen = () => { window.addEventListener("devicemotion", onMotion); return true; };
  try {
    if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {
      return DeviceMotionEvent.requestPermission().then((r) => (r === "granted" ? listen() : false)).catch(() => false);
    }
    return Promise.resolve(typeof window !== "undefined" && "ondevicemotion" in window ? listen() : false);
  } catch (e) { return Promise.resolve(false); }
}
function motionAround(t) {
  let best = null, peak = 0;
  const pre = [];
  for (const d of motion.buf) {
    const m = Math.hypot(d.x, d.y, d.z);
    if (d.t >= t - 120 && d.t <= t + 200) { if (m > peak) { peak = m; best = d; } }
    else if (d.t < t - 150 && d.t > t - 1500) pre.push(m);
  }
  pre.sort((a, b) => a - b);
  const noise = pre.length ? pre[Math.floor(pre.length * 0.9)] : 0;
  return { peak, noise, dir: best ? [best.x / peak, best.y / peak, best.z / peak] : [0, 0, 0] };
}

/* ---------- Impact gate + features ---------- */
// withMotion: whether the profile uses motion features (fixes the feature length)
// mono: ignore the second mic (used by the pretrained model so it fits every iPhone/browser)
function analyse(data, meta = {}, withMotion = false, mono = false) {
  const sr = audio.sr, two = audio.ch > 1 && !mono, bg = meta.bg;
  let peak = 0; for (const c of data) for (const v of c) peak = Math.max(peak, Math.abs(v));
  const raw = [0, 1].map((c) => ({ be: spectrum(data[c], sr), env: envelope(data[c]) }));
  // background subtraction: remove what music/speech contributed before the knock
  const sub = (v, b) => Math.max(v - 1.5 * b, 0.01 * v);
  const clean = raw.map((sp, c) => !bg ? sp : { be: sp.be.map((v, i) => sub(v, bg[c].be[i])), env: sp.env.map((v, i) => sub(v, bg[c].env[i])) });
  const env = clean[0].env, tot = env.reduce((a, b) => a + b, 0) + 1e-12;
  const early = (env[0] + env[1]) / tot, late = (env[5] + env[6] + env[7]) / tot;
  // signal-to-background in the first ~20 ms, where the knock is
  const snr = bg ? (raw[0].env[0] + raw[0].env[1]) / (bg[0].env[0] + bg[0].env[1] + 1e-12) : 99;
  const quality = peak > 0.98 ? "clipped" : peak < 0.006 ? "weak" : snr < 2 ? "masked" : (late > 0.35 || early < 0.3) ? "sustained" : "ok";

  const f = [];
  const t0 = clean[0].be.reduce((a, b) => a + b, 0) + 1e-12;
  for (const b of clean[0].be) f.push(Math.log10(b / t0 + 1e-3)); // -30 dB floor: quiet bands are where noise hides
  for (let s = 0; s < 8; s++) f.push(Math.log10(env[s] / tot + 1e-3));
  if (two) {
    const t1 = clean[1].be.reduce((a, b) => a + b, 0) + 1e-12;
    for (const b of clean[1].be) f.push(Math.log10(b / t1 + 1e-3));
    for (let b = 0; b < NB; b++) f.push(Math.log10((clean[0].be[b] + 1e-12) / (clean[1].be[b] + 1e-12)));
    f.push(2 * Math.log10(t0 / t1));
    // time difference of arrival between the two mics (±1 ms)
    const L = Math.round(sr * 0.001); let best = 0, bestLag = 0;
    for (let lag = -L; lag <= L; lag++) {
      let s = 0; for (let i = PRE - 64; i < PRE + 1024; i++) { const j = i + lag; if (j >= 0 && j < WIN) s += data[0][i] * data[1][j]; }
      if (s > best) { best = s; bestLag = lag; }
    }
    f.push(bestLag / L);
  }
  const m = meta.m || (meta.time != null ? motionAround(meta.time) : { peak: 0, noise: 0, dir: [0, 0, 0] });
  if (withMotion) f.push(Math.log10(m.peak + 1e-3), ...m.dir);
  return { f, quality, peak, snr, mpeak: m.peak, mnoise: m.noise, m, bg };
}

/* Multi-condition training: mix each calibration knock with recorded background
   blocks, so the model learns how knocks look *under* music and speech. */
function augment(taps, blocks, withMotion, copies = 2) {
  const out = [];
  if (!blocks.length) return out;
  taps.forEach((t) => {
    for (let k = 0; k < copies; k++) {
      const b = blocks[Math.floor(Math.random() * blocks.length)];
      const est = blockStats(blocks[Math.floor(Math.random() * blocks.length)], audio.sr); // background estimate from another moment
      const mixed = [0, 1].map((c) => t.data[c].map((v, i) => v + b[c][i]));
      const a = analyse(mixed, { bg: est, m: t.m }, withMotion);
      if (a.quality !== "clipped") out.push({ zone: t.zone, f: a.f, mpeak: a.mpeak, mnoise: a.mnoise, aug: true, src: t.id });
    }
  });
  return out;
}

/* ---------- Classifier (like HoloCore): regularised linear zone model for the
   boundary, plus nearest-example checks for novelty and recorded background noise ---------- */
function solve(A, B) { // Gauss-Jordan: returns A⁻¹B
  const d = A.length, m = B[0].length;
  const M = A.map((r, i) => r.concat(B[i]));
  for (let c = 0; c < d; c++) {
    let p = c; for (let r = c + 1; r < d; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c] || 1e-12;
    for (let r = 0; r < d; r++) if (r !== c) {
      const f = M[r][c] / piv; if (f) for (let k = c; k < d + m; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r.slice(d).map((v) => v / (M[i][i] || 1e-12)));
}
function ridge(xs, zs, lambda) {
  const d = xs[0].length + 1;
  const A = Array.from({ length: d }, () => new Array(d).fill(0));
  const B = Array.from({ length: d }, () => new Array(4).fill(0));
  xs.forEach((x0, n) => {
    const x = x0.concat(1);
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) A[i][j] += x[i] * x[j];
      for (let z = 0; z < 4; z++) B[i][z] += x[i] * (zs[n] === z ? 1 : -1 / 3);
    }
  });
  for (let i = 0; i < d; i++) { for (let j = 0; j < i; j++) A[i][j] = A[j][i]; if (i < d - 1) A[i][i] += lambda; }
  return solve(A, B);
}
function linScores(W, x) {
  const d = x.length, s = [0, 0, 0, 0];
  for (let z = 0; z < 4; z++) { let v = W[d][z]; for (let i = 0; i < d; i++) v += x[i] * W[i][z]; s[z] = v; }
  return s;
}
function dist(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s / a.length); }
function near2(ds) { ds.sort((a, b) => a - b); return ds.length ? (ds[0] + (ds[1] ?? ds[0])) / 2 : Infinity; }
function knn(p, x, skip = null) {
  const per = [[], [], [], []];
  p.samples.forEach((s, i) => { if (!skip || !skip(i)) per[s.z].push(dist(x, s.x)); });
  const score = per.map(near2);
  const order = [0, 1, 2, 3].sort((a, b) => score[a] - score[b]);
  return { zone: order[0], dist: score[order[0]] };
}
function decide(p, x, W = p.W, skip = null) {
  const k = knn(p, x, skip);
  const s = linScores(W, x);
  const order = [0, 1, 2, 3].sort((a, b) => s[b] - s[a]);
  return { zone: order[0], margin: s[order[0]] - s[order[1]], agree: order[0] === k.zone, dist: k.dist };
}
function train(samples, negatives = [], withMotion = false) {
  const all = samples.concat(negatives), d = samples[0].f.length, n = all.length;
  const mean = new Array(d).fill(0), sd = new Array(d).fill(0);
  for (const s of all) s.f.forEach((v, i) => mean[i] += v / n);
  for (const s of all) s.f.forEach((v, i) => sd[i] += (v - mean[i]) ** 2 / n);
  for (let i = 0; i < d; i++) sd[i] = Math.sqrt(sd[i]) + 0.02;
  const norm = (f) => f.map((v, i) => (v - mean[i]) / sd[i]);
  const p = { v: 3, mean, sd, ch: audio.ch, motion: withMotion, raw: samples, rawNeg: negatives,
    samples: samples.map((s) => ({ z: s.zone, x: norm(s.f), id: s.id, src: s.src })), neg: negatives.map((s) => norm(s.f)) };
  const xs = p.samples.map((s) => s.x), zs = p.samples.map((s) => s.z);
  p.lambda = Math.max(4, samples.length * 0.5);
  p.W = ridge(xs, zs, p.lambda);
  // leave-one-out per zone (each example judged by a model trained without it)
  const ok = [0, 0, 0, 0], cnt = [0, 0, 0, 0], nn = [];
  // (augmented copies of the held-out knock are held out too; only real knocks are scored)
  p.samples.forEach((s, idx) => {
    if (s.src != null) return;
    const out = (i) => i === idx || (s.id != null && p.samples[i].src === s.id);
    const keep = (_, i) => !out(i);
    const W = ridge(xs.filter(keep), zs.filter(keep), p.lambda);
    const r = decide(p, s.x, W, out);
    cnt[s.z]++; if (r.zone === s.z) ok[s.z]++; nn.push(r.dist);
  });
  // knocks mixed with background are legit too: include their distances in the novelty threshold
  p.samples.forEach((s, idx) => {
    if (s.src == null) return;
    nn.push(knn(p, s.x, (i) => i === idx || p.samples[i].src === s.src || p.samples[i].id === s.src).dist);
  });
  nn.sort((a, b) => a - b);
  p.zoneAcc = ok.map((o, z) => (cnt[z] ? o / cnt[z] : 0));
  p.loo = ok.reduce((a, b) => a + b, 0) / Math.max(1, cnt.reduce((a, b) => a + b, 0));
  p.ood = nn[Math.floor(nn.length * 0.9)] * 1.8;
  // motion gate: only if knocks clearly shake the phone more than its resting jitter
  p.gate = 0;
  if (withMotion) {
    const peaks = samples.map((s) => s.mpeak || 0).sort((a, b) => a - b);
    const noises = samples.map((s) => s.mnoise || 0).sort((a, b) => a - b);
    const p20 = peaks[Math.floor(peaks.length * 0.2)], nz = noises[Math.floor(noises.length / 2)] + 0.003;
    p.motionPeak = p20; p.motionNoise = nz;
    if (p20 > 3 * nz) p.gate = Math.max(p20 * 0.3, nz * 2);
  }
  return p;
}
function classify(p, a) {
  if (!p || !p.W || p.mean.length !== a.f.length) return { zone: null, why: "Profil passt nicht, bitte neu kalibrieren" };
  if (p.gate && a.mpeak < p.gate) return { zone: null, why: "keine Tisch-Vibration" };
  const x = a.f.map((v, i) => (v - p.mean[i]) / p.sd[i]);
  const r = decide(p, x);
  if (r.dist > p.ood) return { ...r, zone: null, why: "unbekanntes Geräusch" };
  if (p.neg.length && near2(p.neg.map((n) => dist(x, n))) < r.dist * 0.9) return { ...r, zone: null, why: "Hintergrundgeräusch" };
  if (r.margin < 0.08 || (!r.agree && r.margin < 0.35)) return { ...r, zone: null, why: "nicht eindeutig" };
  return r;
}

/* Strip a trained profile down to what classify() needs (for model.json). */
function compactModel(p, info = {}) {
  const r4 = (v) => Math.round(v * 1e4) / 1e4;
  return { v: p.v, kind: "pretrained", mono: !!p.mono, motion: p.motion, gate: p.gate, ood: p.ood, loo: p.loo, zoneAcc: p.zoneAcc,
    mean: p.mean.map(r4), sd: p.sd.map(r4), W: p.W.map((row) => row.map(r4)),
    samples: p.samples.map((s) => ({ z: s.z, x: s.x.map((v) => Math.round(v * 100) / 100) })),
    neg: p.neg.map((x) => x.map((v) => Math.round(v * 100) / 100)), ...info };
}
if (typeof module !== "undefined") module.exports = { audio, analyse, train, classify, compactModel, ZONES, WIN };
