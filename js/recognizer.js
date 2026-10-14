// recognizer.js -- детектор ключових слів (ONNX) на телефоні + запит до сервера (онлайн-режим).

import * as ort from "../vendor/ort/ort.wasm.min.mjs";
import { FEAT_DIM, rowsToModelInput } from "./features.js";

const BASE = new URL("../", import.meta.url).href;
ort.env.wasm.wasmPaths = BASE + "vendor/ort/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

export const MIN_FRAMES = 15;   // коротше 0.5 с -- не жест
export const MAX_FRAMES = 1800; // 60 с при 30 кадрах/с

export class Recognizer {
  static async create() {
    const cfg = await (await fetch(BASE + "model/config.json")).json();
    const session = await ort.InferenceSession.create(BASE + "model/keywords.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new Recognizer(cfg, session);
  }

  constructor(cfg, session) {
    this.cfg = cfg;
    this.session = session;
  }

  /** Кадри 225 (30 кадрів/с) -> ймовірність кожного слова словника. */
  async predict(rows) {
    const T = rows.length;
    const feats = rowsToModelInput(rows, this.cfg);
    const out = await this.session.run({ feats: new ort.Tensor("float32", feats, [1, T, FEAT_DIM]) });
    return Array.from(out.probs.data);
  }

  summarize(probs, sure = this.cfg.threshold_sure, maybe = this.cfg.threshold_maybe) {
    return summarize(this.cfg.vocab, probs, sure, maybe);
  }
}

/** Ймовірності -> впевнені / можливі слова і top-10 для деталей. */
export function summarize(vocab, probs, sure, maybe) {
  const ranked = probs.map((p, i) => ({ word: vocab[i], p })).sort((a, b) => b.p - a.p);
  return {
    sure: ranked.filter((x) => x.p >= sure),
    maybe: ranked.filter((x) => x.p >= maybe && x.p < sure).slice(0, 8),
    top: ranked.slice(0, 10),
  };
}

// ---------------------------------------------------------------------------
// Онлайн-режим
// ---------------------------------------------------------------------------

function rowsToBase64(rows) {
  const buf = new Float32Array(rows.length * rows[0].length);
  rows.forEach((r, i) => buf.set(r, i * r.length));
  const bytes = new Uint8Array(buf.buffer);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

export function serverBase(setting) {
  const s = (setting || "").trim().replace(/\/+$/, "");
  return s || new URL(".", location.href).href.replace(/\/+$/, "");
}

export async function serverHealth(base, timeoutMs = 4000) {
  const r = await fetch(base + "/api/health", { signal: AbortSignal.timeout(timeoutMs) });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

/** Надіслати кадри (не відео!) на сервер: ансамбль моделей + складання речення. */
export async function recognizeOnline(base, rows, thresholds, timeoutMs = 90000) {
  const r = await fetch(base + "/api/recognize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rows_b64: rowsToBase64(rows), n: rows.length, dim: rows[0].length, fps: 30, ...thresholds }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!r.ok) throw new Error("Сервер відповів " + r.status + ": " + (await r.text()).slice(0, 200));
  return r.json();
}
