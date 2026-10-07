import * as ort from './vendor/ort.webgpu.bundle.min.mjs';

/* ---------------- configuration ---------------- */

const MODELS = [
  {
    id: 'ani4k',
    name: 'Ani4K',
    file: 'models/2x_Ani4Kv2_G6i2_Compact_107500_fp32.onnx',
    dtype: 'float32',
    scale: 2,
    note: ['Compact · 2.4 MB', 'speed ●●● · quality ●○○'],
  },
  {
    id: 'animesharp',
    name: 'Sharp V2',
    file: 'models/2x-AnimeSharpV2_MoSR_Sharp_fp16.onnx',
    dtype: 'float16',
    scale: 2,
    note: ['MoSR · 8.8 MB', 'speed ●●○ · quality ●●○'],
  },
  {
    id: 'animesharp4',
    name: 'Sharp V4',
    file: 'models/2x-AnimeSharpV4_Fast_RCAN_PU_fp16_opset17.onnx',
    dtype: 'float16',
    scale: 2,
    note: ['RCAN · 32 MB', 'speed ●○○ · quality ●●●'],
  },
];

const QUALITY_TIERS = [
  { id: 'small', label: 'Small' },
  { id: 'balanced', label: 'Balanced' },
  { id: 'high', label: 'High' },
  { id: 'max', label: 'Max' },
];
// Canvas toBlob quality (0–1) and WASM AVIF quality (0–100) per tier.
const QUALITY = {
  avif: { small: 0.5, balanced: 0.68, high: 0.87, max: 0.97 },
  heic: { small: 0.55, balanced: 0.72, high: 0.87, max: 0.96 },
  jpeg: { small: 0.7, balanced: 0.82, high: 0.92, max: 0.98 },
};
const AVIF_WASM_QUALITY = { small: 45, balanced: 62, high: 78, max: 92 };

const TILE = 224;
const OVERLAP = 16;
const PAD_MULTIPLE = 8;
const RING_C = 138.23; // 2πr for r=22
const SCALES = [1.5, 2, 3, 4];
const RES_PRESETS = [
  { value: 1080, label: '1080p' },
  { value: 1440, label: '1440p' },
  { value: 2160, label: '4K' },
  { value: 2880, label: '5K' },
];
const MAX_SCALE = 8;   // at most three chained 2x passes
const MAX_MP = 64;     // cap on any intermediate or final canvas

const FORMATS = [
  { id: 'avif', mime: 'image/avif' },
  { id: 'heic', mime: 'image/heic' },
  { id: 'jpeg', mime: 'image/jpeg' },
  { id: 'png', mime: 'image/png' },
];
const AVIF_WASM_OPTIONS = {
  qualityAlpha: -1, denoiseLevel: 0, tileColsLog2: 0, tileRowsLog2: 0,
  speed: 7, subsample: 1, chromaDeltaQ: false, sharpness: 0, tune: 0,
  enableSharpYUV: false, bitDepth: 8, lossless: false,
};

/* ---------------- state (single source of truth) ---------------- */

const state = {
  backend: null,          // 'webgpu' | 'wasm'
  formats: { avif: false, heic: false },
  encoders: { png: true, jpeg: true, avif: false, heic: false }, // avif: 'canvas' | 'wasm' | false
  downloadFormat: 'png',
  qualityTier: 'high',
  saving: false,
  savedNote: null,
  sizeMode: 'scale',      // 'scale' | 'res'
  scaleChoice: '2',       // one of SCALES as string, or 'custom'
  scaleCustom: 2.5,
  resChoice: '2160',      // one of RES_PRESETS values as string, or 'custom'
  resCustom: 1200,
  threads: 1,
  modelId: 'ani4k',
  sessions: new Map(),
  source: null,           // { bitmap, width, height, name }
  view: 'empty',          // 'empty' | 'preview' | 'compare'
  busy: false,
  run: null,              // { done, total, startedAt, tileMs: [], outMP }
  statusText: '',
  statusError: false,
  result: null,           // { width, height, ms, tiles, backend, modelName, scale }
  compareFrac: 0.5,
};

/* ---------------- backend + format detection ---------------- */

async function detectBackend() {
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) return 'webgpu';
    } catch { /* fall through to wasm */ }
  }
  return 'wasm';
}

async function canDecode(url) {
  try {
    const bitmap = await createImageBitmap(await (await fetch(url)).blob());
    bitmap.close();
    return true;
  } catch {
    return false;
  }
}

async function canEncode(mime) {
  try {
    const c = new OffscreenCanvas(2, 2);
    c.getContext('2d').fillRect(0, 0, 2, 2);
    const blob = await c.convertToBlob({ type: mime });
    return blob.type === mime; // unsupported types silently fall back to PNG
  } catch {
    return false;
  }
}

async function initRuntime() {
  const [backend, avif, heic, encAvif, encHeic] = await Promise.all([
    detectBackend(), canDecode('detect/tiny.avif'), canDecode('detect/tiny.heic'),
    canEncode('image/avif'), canEncode('image/heic'),
  ]);
  state.backend = backend;
  state.formats = { avif, heic };
  // AVIF is always offered: native canvas encoder when present, vendored WASM encoder otherwise.
  state.encoders = { png: true, jpeg: true, avif: encAvif ? 'canvas' : 'wasm', heic: encHeic };
  state.downloadFormat = 'avif';
  state.threads = Math.min(navigator.hardwareConcurrency || 4, 8);
  ort.env.wasm.wasmPaths = new URL('vendor/', document.baseURI).href;
  ort.env.wasm.numThreads = crossOriginIsolated ? state.threads : 1;
  render();
}

/* ---------------- session management ---------------- */

async function getSession(model) {
  const cached = state.sessions.get(model.id);
  if (cached) return cached;

  const eps = state.backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
  const session = await ort.InferenceSession.create(model.file, {
    executionProviders: eps,
    graphOptimizationLevel: 'all',
  });
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  // Probe with a tiny tile to learn the true scale factor and validate dynamic shapes.
  const probeEdge = 16;
  const probe = makeTensor(new Float32Array(3 * probeEdge * probeEdge), model.dtype, [1, 3, probeEdge, probeEdge]);
  const probeOut = (await session.run({ [inputName]: probe }))[outputName];
  const scale = probeOut.dims[2] / probeEdge;
  probeOut.dispose?.();

  const entry = { session, inputName, outputName, scale, dtype: model.dtype };
  state.sessions.set(model.id, entry);
  return entry;
}

/* ---------------- fp16 helpers ---------------- */

const hasFloat16Array = typeof Float16Array !== 'undefined';

function f32ToF16Bits(f32) {
  if (hasFloat16Array) {
    const h = new Float16Array(f32);
    return new Uint16Array(h.buffer);
  }
  const out = new Uint16Array(f32.length);
  const fb = new Float32Array(1); const ib = new Uint32Array(fb.buffer);
  for (let i = 0; i < f32.length; i++) {
    fb[0] = f32[i];
    const x = ib[0];
    const sign = (x >>> 16) & 0x8000;
    const exp = ((x >>> 23) & 0xFF) - 127 + 15;
    const mant = (x >>> 13) & 0x3FF;
    if (exp <= 0) { out[i] = sign; }
    else if (exp >= 31) { out[i] = sign | 0x7C00; }
    else { out[i] = sign | (exp << 10) | mant; }
  }
  return out;
}

function f16BitsToF32(u16) {
  if (hasFloat16Array) {
    return new Float32Array(new Float16Array(u16.buffer, u16.byteOffset, u16.length));
  }
  const out = new Float32Array(u16.length);
  const fb = new Float32Array(1); const ib = new Uint32Array(fb.buffer);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = (h & 0x8000) << 16;
    const exp = (h >>> 10) & 0x1F;
    const mant = h & 0x3FF;
    if (exp === 0) { const f = mant * 2 ** -24; out[i] = sign ? -f : f; continue; }
    if (exp === 31) { out[i] = sign ? -Infinity : Infinity; continue; }
    ib[0] = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    out[i] = fb[0];
  }
  return out;
}

function makeTensor(f32, dtype, dims) {
  if (dtype === 'float16') return new ort.Tensor('float16', f32ToF16Bits(f32), dims);
  return new ort.Tensor('float32', f32, dims);
}

function tensorToF32(tensor) {
  if (tensor.type === 'float16') {
    const d = tensor.data;
    return f16BitsToF32(d instanceof Uint16Array ? d : new Uint16Array(d.buffer ?? d));
  }
  return tensor.data;
}

/* ---------------- pixel <-> tensor ---------------- */

function rgbaToNCHW(imageData, w, h) {
  const { data } = imageData;
  const plane = w * h;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    out[i] = data[i * 4] / 255;
    out[plane + i] = data[i * 4 + 1] / 255;
    out[2 * plane + i] = data[i * 4 + 2] / 255;
  }
  return out;
}

function nchwToRGBA(f32, w, h) {
  const plane = w * h;
  const out = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    out[i * 4] = Math.max(0, Math.min(255, f32[i] * 255 + 0.5));
    out[i * 4 + 1] = Math.max(0, Math.min(255, f32[plane + i] * 255 + 0.5));
    out[i * 4 + 2] = Math.max(0, Math.min(255, f32[2 * plane + i] * 255 + 0.5));
    out[i * 4 + 3] = 255;
  }
  return new ImageData(out, w, h);
}

/* ---------------- target sizing ---------------- */

const clampNum = (v, lo, hi) => Math.min(hi, Math.max(lo, Number(v) || lo));

/* The scale factor the current sizing selection asks for; null without a source in res mode. */
function effectiveScale() {
  if (state.sizeMode === 'scale') {
    return state.scaleChoice === 'custom'
      ? clampNum(state.scaleCustom, 0.1, MAX_SCALE) : Number(state.scaleChoice);
  }
  if (!state.source) return null;
  const target = state.resChoice === 'custom'
    ? clampNum(state.resCustom, 16, 16384) : Number(state.resChoice);
  // Resolution presets name the short side: 1080p landscape is height 1080, portrait is width 1080.
  return target / Math.min(state.source.width, state.source.height);
}

function targetLabel() {
  if (state.sizeMode === 'scale') {
    const s = state.scaleChoice === 'custom' ? clampNum(state.scaleCustom, 0.1, MAX_SCALE) : state.scaleChoice;
    return `${s}×`;
  }
  const preset = RES_PRESETS.find(p => String(p.value) === state.resChoice);
  return preset ? preset.label : `${clampNum(state.resCustom, 16, 16384)}px`;
}

/* ---------------- tiled inference ---------------- */

async function upscale() {
  if (state.busy || !state.source) return;
  const model = MODELS.find(m => m.id === state.modelId);
  state.busy = true;
  state.result = null;
  setStatus('');
  render();

  try {
    const target = effectiveScale();
    if (target > MAX_SCALE) throw new Error(`${targetLabel()} means ${target.toFixed(1)}× — the limit is ${MAX_SCALE}×`);

    // Reach at least the target with chained model passes, then resample to exact size.
    // A target at or below 1 is a plain downscale: no model passes at all.
    const modelScale = model.scale;
    const passes = Math.max(0, Math.ceil(Math.log(target) / Math.log(modelScale) - 1e-9));
    const peakScale = modelScale ** passes;
    const { bitmap, width: W, height: H } = state.source;
    if (W * H * peakScale * peakScale > MAX_MP * 1e6) {
      const need = (W * H * peakScale * peakScale / 1e6).toFixed(0);
      throw new Error(`${targetLabel()} for ${W}×${H} needs a ${need} MP intermediate — the limit is ${MAX_MP} MP`);
    }

    let session, inputName, outputName, dtype;
    if (passes > 0) {
      ({ session, inputName, outputName, dtype } = await getSession(model));
      const probed = state.sessions.get(model.id).scale;
      if (probed !== modelScale) throw new Error(`model reports ${probed}× but registry says ${modelScale}×`);
    }

    const srcCanvas = new OffscreenCanvas(W, H);
    srcCanvas.getContext('2d', { willReadFrequently: true }).drawImage(bitmap, 0, 0);

    let totalTiles = 0;
    for (let p = 0, w = W, h = H; p < passes; p++, w *= modelScale, h *= modelScale) {
      totalTiles += Math.ceil(w / TILE) * Math.ceil(h / TILE);
    }
    state.run = { done: 0, total: totalTiles, startedAt: performance.now(), tileMs: [], outMP: 0 };

    let cur = srcCanvas, curW = W, curH = H;
    for (let pass = 0; pass < passes; pass++) {
      const outW = curW * modelScale, outH = curH * modelScale;
      const passOut = new OffscreenCanvas(outW, outH);
      const outCtx = passOut.getContext('2d');
      const cols = Math.ceil(curW / TILE), rows = Math.ceil(curH / TILE);

      for (let ty = 0; ty < rows; ty++) {
        for (let tx = 0; tx < cols; tx++) {
          const tileT0 = performance.now();
          const x0 = tx * TILE, y0 = ty * TILE;
          const iw = Math.min(TILE, curW - x0), ih = Math.min(TILE, curH - y0);
          const ex0 = Math.max(0, x0 - OVERLAP), ey0 = Math.max(0, y0 - OVERLAP);
          const ex1 = Math.min(curW, x0 + iw + OVERLAP), ey1 = Math.min(curH, y0 + ih + OVERLAP);
          const ew = ex1 - ex0, eh = ey1 - ey0;
          const pw = Math.ceil(ew / PAD_MULTIPLE) * PAD_MULTIPLE;
          const ph = Math.ceil(eh / PAD_MULTIPLE) * PAD_MULTIPLE;

          const tileCanvas = new OffscreenCanvas(pw, ph);
          const tctx = tileCanvas.getContext('2d', { willReadFrequently: true });
          tctx.drawImage(cur, ex0, ey0, ew, eh, 0, 0, ew, eh);
          if (pw > ew) tctx.drawImage(tileCanvas, ew - 1, 0, 1, eh, ew, 0, pw - ew, eh);
          if (ph > eh) tctx.drawImage(tileCanvas, 0, eh - 1, pw, 1, 0, eh, pw, ph - eh);

          const nchw = rgbaToNCHW(tctx.getImageData(0, 0, pw, ph), pw, ph);
          const input = makeTensor(nchw, dtype, [1, 3, ph, pw]);
          const output = (await session.run({ [inputName]: input }))[outputName];
          const outF32 = tensorToF32(output);
          const outImage = nchwToRGBA(outF32, pw * modelScale, ph * modelScale);
          output.dispose?.();

          const cropX = (x0 - ex0) * modelScale, cropY = (y0 - ey0) * modelScale;
          const tmp = new OffscreenCanvas(pw * modelScale, ph * modelScale);
          tmp.getContext('2d').putImageData(outImage, 0, 0);
          outCtx.drawImage(tmp, cropX, cropY, iw * modelScale, ih * modelScale,
            x0 * modelScale, y0 * modelScale, iw * modelScale, ih * modelScale);

          state.run.done++;
          state.run.tileMs.push(performance.now() - tileT0);
          state.run.outMP += (iw * modelScale * ih * modelScale) / 1e6;
          render();
          // Yield to the UI, but never depend on rAF alone: it stops firing in
          // background tabs and under modal dialogs, which would freeze the run.
          await new Promise(r => {
            const t = setTimeout(r, 120);
            requestAnimationFrame(() => { clearTimeout(t); r(); });
          });
        }
      }
      cur = passOut; curW = outW; curH = outH;
    }

    const finalW = Math.round(W * target), finalH = Math.round(H * target);
    const outCanvas = document.getElementById('afterCanvas');
    outCanvas.width = finalW;
    outCanvas.height = finalH;
    const fctx = outCanvas.getContext('2d');
    fctx.imageSmoothingEnabled = true;
    fctx.imageSmoothingQuality = 'high';
    fctx.drawImage(cur, 0, 0, finalW, finalH);

    applyAlpha(srcCanvas, fctx, W, H, finalW, finalH);

    const ms = performance.now() - state.run.startedAt;
    state.result = {
      width: finalW, height: finalH, ms, tiles: state.run.total,
      backend: state.backend, modelName: model.name, scale: target,
    };
    state.view = 'compare';
    drawBefore();
  } catch (err) {
    console.error(err);
    setStatus(`Failed: ${err.message}`, true);
    state.result = null;
  } finally {
    state.busy = false;
    render();
  }
}

function applyAlpha(srcCanvas, outCtx, W, H, outW, outH) {
  const srcData = srcCanvas.getContext('2d').getImageData(0, 0, W, H).data;
  let hasAlpha = false;
  for (let i = 3; i < srcData.length; i += 4) if (srcData[i] !== 255) { hasAlpha = true; break; }
  if (!hasAlpha) return;
  const up = new OffscreenCanvas(outW, outH);
  const uctx = up.getContext('2d');
  uctx.imageSmoothingEnabled = true;
  uctx.imageSmoothingQuality = 'high';
  uctx.drawImage(srcCanvas, 0, 0, outW, outH);
  const upData = uctx.getImageData(0, 0, outW, outH);
  const outData = outCtx.getImageData(0, 0, outW, outH);
  for (let i = 3; i < outData.data.length; i += 4) outData.data[i] = upData.data[i];
  outCtx.putImageData(outData, 0, 0);
}

/* ---------------- image loading ---------------- */

async function loadImage(fileOrBlob, name) {
  try {
    const bitmap = await createImageBitmap(fileOrBlob);
    state.source = { bitmap, width: bitmap.width, height: bitmap.height, name: name || 'image' };
    state.result = null;
    state.run = null;
    state.view = 'preview';
    setStatus('');
    drawPreview();
    render();
  } catch {
    const heicHint = /\.hei[cf]$/i.test(name || '') && !state.formats.heic
      ? ' — this browser cannot decode HEIC' : '';
    setStatus(`Could not decode that file as an image${heicHint}.`, true);
    render();
  }
}

function drawPreview() {
  const { bitmap, width, height } = state.source;
  const preview = document.getElementById('previewCanvas');
  const k = Math.min(1, 1024 / Math.max(width, height));
  preview.width = Math.round(width * k);
  preview.height = Math.round(height * k);
  const ctx = preview.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, preview.width, preview.height);
}

function drawBefore() {
  const { bitmap, width, height } = state.source;
  const before = document.getElementById('beforeCanvas');
  before.width = width; before.height = height;
  before.getContext('2d').drawImage(bitmap, 0, 0);
}

/* The canonical anime-upscaling demo input: waifu2x's Hatsune Miku sample,
   CC BY-NC by piapro (credited in the footer). */
async function fetchSample() {
  return (await fetch('sample/miku_small.png')).blob();
}

/* ---------------- rendering (all views derive from state) ---------------- */

function setStatus(text, isError = false) {
  state.statusText = text;
  state.statusError = isError;
}

const $ = id => document.getElementById(id);

function render() {
  // backend chip
  const chip = $('backendChip');
  chip.classList.toggle('on', !!state.backend);
  chip.innerHTML = `<span class="dot"></span>${
    !state.backend ? 'detecting…'
      : state.backend === 'webgpu' ? 'WebGPU'
      : crossOriginIsolated ? `WASM · ${ort.env.wasm.numThreads} threads` : 'WASM · single-thread'}`;

  $('formatList').textContent = ['PNG', 'JPEG', 'WebP',
    ...(state.formats.avif ? ['AVIF'] : []), ...(state.formats.heic ? ['HEIC'] : [])].join(' · ');

  // stage view
  const v = state.view;
  $('stageEmpty').classList.toggle('hidden', v !== 'empty');
  $('previewCanvas').classList.toggle('hidden', v !== 'preview');
  $('beforeCanvas').classList.toggle('hidden', v !== 'compare');
  $('afterClip').classList.toggle('hidden', v !== 'compare');
  $('divider').classList.toggle('hidden', v !== 'compare');
  $('tagL').classList.toggle('hidden', v !== 'compare');
  $('tagR').classList.toggle('hidden', v !== 'compare');
  const metaPill = $('metaPill');
  metaPill.classList.toggle('hidden', v === 'empty');
  if (state.source) {
    metaPill.textContent = v === 'compare' && state.result
      ? `${state.source.width}×${state.source.height} → ${state.result.width}×${state.result.height}`
      : `${state.source.name} · ${state.source.width}×${state.source.height}`;
  }
  if (v === 'compare') applyCompare();

  // segmented radiogroups (rebuilt; direct bindings; arrow-key rotation)
  const buildSegmented = (id, options, current, apply) => {
    const seg = $(id);
    seg.innerHTML = options.map(o => `
      <button role="radio" aria-checked="${String(o.value) === String(current)}" data-value="${o.value}">${o.label}</button>`).join('');
    seg.querySelectorAll('[role="radio"]').forEach(el => {
      el.onclick = () => { if (!state.busy) { apply(el.dataset.value); render(); } };
      el.onkeydown = e => {
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
          e.preventDefault();
          const vals = options.map(o => String(o.value));
          const i = vals.indexOf(String(current));
          apply(vals[(i + (e.key === 'ArrowRight' ? 1 : vals.length - 1)) % vals.length]);
          render();
          $(id).querySelector('[aria-checked="true"]')?.focus();
        }
      };
    });
  };
  buildSegmented('segModel', MODELS.map(m => ({ value: m.id, label: m.name })),
    state.modelId, v => { state.modelId = v; });
  buildSegmented('segMode', [{ value: 'scale', label: 'Scale' }, { value: 'res', label: 'Resolution' }],
    state.sizeMode, v => { state.sizeMode = v; });
  if (state.sizeMode === 'scale') {
    buildSegmented('segValue',
      [...SCALES.map(s => ({ value: s, label: `${s}×` })), { value: 'custom', label: 'Custom' }],
      state.scaleChoice, v => { state.scaleChoice = v; });
  } else {
    buildSegmented('segValue',
      [...RES_PRESETS, { value: 'custom', label: 'Custom' }],
      state.resChoice, v => { state.resChoice = v; });
  }

  // custom value field
  const isCustom = (state.sizeMode === 'scale' ? state.scaleChoice : state.resChoice) === 'custom';
  $('customWrap').classList.toggle('hidden', !isCustom);
  $('customUnit').textContent = state.sizeMode === 'scale' ? '×' : 'px';
  const customInput = $('customValue');
  if (isCustom && document.activeElement !== customInput) {
    customInput.value = state.sizeMode === 'scale' ? state.scaleCustom : state.resCustom;
    customInput.step = state.sizeMode === 'scale' ? '0.1' : '10';
  }

  // computed output size for the current selection
  const eff = state.source && effectiveScale();
  $('targetHint').textContent = eff
    ? `→ ${Math.round(state.source.width * eff)}×${Math.round(state.source.height * eff)}`
    : '';

  $('modelNote').innerHTML = MODELS.find(m => m.id === state.modelId)
    .note.map(line => `<span>${line}</span>`).join('<br>');

  // primary button adapts to state — never disabled, always the next right action
  const runBtn = $('runBtn');
  runBtn.setAttribute('aria-busy', state.busy);
  if (state.busy) runBtn.innerHTML = `<span class="spinner" aria-hidden="true"></span>Upscaling…`;
  else if (!state.source) runBtn.textContent = 'Choose Image…';
  else if (eff && eff <= 1) runBtn.textContent = `Resize to ${targetLabel()}`;
  else runBtn.textContent = state.sizeMode === 'res'
    ? `Upscale to ${targetLabel()}` : `Upscale ${targetLabel()}`;
  $('replaceBtn').classList.toggle('hidden', !state.source || state.busy);

  // metrics
  const showMetrics = !!state.run;
  $('metrics').classList.toggle('hidden', !showMetrics);
  if (showMetrics) {
    const r = state.run;
    const frac = r.total ? r.done / r.total : 1;
    $('ringFill').style.strokeDashoffset = RING_C * (1 - frac);
    $('ringPct').textContent = `${Math.round(frac * 100)}%`;
    $('ringWrap').setAttribute('aria-valuenow', Math.round(frac * 100));
    const elapsed = ((state.result ? state.result.ms : performance.now() - r.startedAt) / 1000);
    $('statTime').innerHTML = `${elapsed.toFixed(2)}<small> s</small>`;
    $('statMpps').innerHTML = elapsed > 0 ? `${(r.outMP / elapsed).toFixed(2)}` : '–';
    $('statTiles').textContent = `${r.done}/${r.total}`;
    $('statSize').innerHTML = state.result
      ? `${(state.result.width * state.result.height / 1e6).toFixed(1)}<small> MP</small>` : '–';
    const maxMs = Math.max(...r.tileMs, 1);
    $('spark').innerHTML = r.tileMs.slice(-48)
      .map(ms => `<i style="height:${Math.max(7, Math.round(ms / maxMs * 100))}%"></i>`).join('');
    const saveBtn = $('saveBtn');
    saveBtn.classList.toggle('hidden', !state.result);
    saveBtn.setAttribute('aria-busy', state.saving);
    if (state.saving) saveBtn.innerHTML = `<span class="spinner" aria-hidden="true"></span>Encoding…`;
    else saveBtn.textContent = state.savedNote || 'Download';
    $('segFormat').classList.toggle('hidden', !state.result);
    const lossy = state.downloadFormat !== 'png';
    $('segQuality').classList.toggle('hidden', !state.result || !lossy);
    if (state.result) {
      buildSegmented('segFormat',
        FORMATS.filter(f => state.encoders[f.id]).map(f => ({ value: f.id, label: f.id.toUpperCase() })),
        state.downloadFormat, v => { state.downloadFormat = v; });
      if (lossy) {
        buildSegmented('segQuality', QUALITY_TIERS.map(t => ({ value: t.id, label: t.label })),
          state.qualityTier, v => { state.qualityTier = v; });
      }
    }
  }

  // the status line exists for errors alone; everything informational lives on controls
  const statusEl = $('status');
  statusEl.textContent = state.statusError ? state.statusText : '';
  statusEl.classList.toggle('hidden', !state.statusError);
}

/* ---------------- comparison slider ---------------- */

function applyCompare() {
  const frac = state.compareFrac;
  $('afterClip').style.clipPath = `inset(0 0 0 ${frac * 100}%)`;
  const divider = $('divider');
  divider.style.left = `calc(${frac * 100}% - 1px)`;
  divider.setAttribute('aria-valuenow', Math.round(frac * 100));
}

function initSlider() {
  const stage = $('stage');
  const move = clientX => {
    const rect = stage.getBoundingClientRect();
    state.compareFrac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    applyCompare();
  };
  stage.addEventListener('pointerdown', e => {
    if (state.view !== 'compare') return;
    e.preventDefault();
    stage.setPointerCapture(e.pointerId);
    move(e.clientX);
  });
  stage.addEventListener('pointermove', e => {
    if (state.view === 'compare' && e.buttons) move(e.clientX);
  });
  $('divider').addEventListener('keydown', e => {
    const step = e.key === 'ArrowLeft' ? -0.02 : e.key === 'ArrowRight' ? 0.02
      : e.key === 'Home' ? -1 : e.key === 'End' ? 1 : 0;
    if (!step) return;
    e.preventDefault();
    state.compareFrac = Math.max(0, Math.min(1, Math.abs(step) === 1 ? (step > 0 ? 1 : 0) : state.compareFrac + step));
    applyCompare();
  });
}

/* ---------------- wiring ---------------- */

const stage = $('stage');
const fileInput = $('file');
const pick = () => fileInput.click();

$('stageEmpty').addEventListener('click', e => {
  if (e.target.id !== 'sampleLink' && e.target.id !== 'browseLink') pick();
});
$('browseLink').addEventListener('click', pick);
$('replaceBtn').addEventListener('click', pick);
stage.addEventListener('click', () => { if (state.view === 'preview' && !state.busy) pick(); });
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadImage(fileInput.files[0], fileInput.files[0].name);
});
['dragenter', 'dragover'].forEach(ev => stage.addEventListener(ev, e => {
  e.preventDefault(); stage.classList.add('armed');
}));
['dragleave', 'drop'].forEach(ev => stage.addEventListener(ev, e => {
  e.preventDefault(); stage.classList.remove('armed');
}));
stage.addEventListener('drop', e => {
  const f = e.dataTransfer.files[0];
  if (f) loadImage(f, f.name);
});
window.addEventListener('paste', e => {
  const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
  if (item) loadImage(item.getAsFile(), 'pasted image');
});
$('sampleLink').addEventListener('click', async e => {
  e.stopPropagation();
  loadImage(await fetchSample(), 'miku');
});
$('runBtn').addEventListener('click', () => {
  if (state.busy) return;
  if (!state.source) pick();
  else upscale();
});
let avifModulePromise = null;

async function encodeAvifWasm(canvas, quality) {
  if (!avifModulePromise) {
    avifModulePromise = import('./vendor/avif/avif_enc.js')
      .then(m => m.default({ noInitialRun: true }));
  }
  const mod = await avifModulePromise;
  const ctx = canvas.getContext('2d');
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const out = mod.encode(new Uint8Array(data.data.buffer), data.width, data.height,
    { ...AVIF_WASM_OPTIONS, quality });
  if (!out) throw new Error('AVIF encoding failed');
  return new Blob([out.buffer], { type: 'image/avif' });
}

function canvasToBlob(canvas, mime, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(b => b ? resolve(b) : reject(new Error('encoding failed')), mime, quality);
  });
}

$('saveBtn').addEventListener('click', async () => {
  if (state.saving) return;
  const fmt = FORMATS.find(f => f.id === state.downloadFormat) || FORMATS.at(-1);
  const canvas = $('afterCanvas');
  state.saving = true;
  setStatus('');
  render();
  try {
    let blob;
    if (fmt.id === 'avif' && state.encoders.avif === 'wasm') {
      blob = await encodeAvifWasm(canvas, AVIF_WASM_QUALITY[state.qualityTier]);
    } else {
      blob = await canvasToBlob(canvas, fmt.mime, QUALITY[fmt.id]?.[state.qualityTier]);
    }
    // If the encoder silently fell back, name the file for what it actually produced.
    const ext = { 'image/avif': 'avif', 'image/heic': 'heic', 'image/jpeg': 'jpg', 'image/png': 'png' }[blob.type] || 'png';
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${(state.source?.name || 'image').replace(/\.[^.]+$/, '')}_${state.result.width}x${state.result.height}.${ext}`;
    a.click();
    URL.revokeObjectURL(a.href);
    state.savedNote = `Saved ${(blob.size / 1024).toFixed(0)} KB`;
    setTimeout(() => { state.savedNote = null; render(); }, 2400);
  } catch (err) {
    setStatus(`Export failed: ${err.message}`, true);
  }
  state.saving = false;
  render();
});
$('customValue').addEventListener('input', () => {
  const v = Number($('customValue').value);
  if (!v) return;
  if (state.sizeMode === 'scale') state.scaleCustom = v;
  else state.resCustom = v;
  render();
});

/* URL presets: ?model=ani4k|animesharp|animesharp4 · ?scale=2.5 · ?res=1440 (short side px) */
{
  const q = new URLSearchParams(location.search);
  const model = q.get('model');
  if (MODELS.some(m => m.id === model)) state.modelId = model;
  const scale = Number(q.get('scale'));
  if (scale > 0) {
    state.sizeMode = 'scale';
    if (SCALES.includes(scale)) state.scaleChoice = String(scale);
    else { state.scaleChoice = 'custom'; state.scaleCustom = clampNum(scale, 0.1, MAX_SCALE); }
  }
  const res = Number(q.get('res'));
  if (res > 0) {
    state.sizeMode = 'res';
    if (RES_PRESETS.some(p => p.value === res)) state.resChoice = String(res);
    else { state.resChoice = 'custom'; state.resCustom = clampNum(res, 16, 16384); }
  }
}

initSlider();
render();
initRuntime();
