// detector.js -- MediaPipe Tasks (PoseLandmarker + HandLandmarker), ті самі моделі й пороги,
// що й у landmark_extractor.py. Кадр вписується в квадрат (з полями), як кліпи датасету 510x510.

import { FilesetResolver, HandLandmarker, PoseLandmarker } from "../vendor/mediapipe/vision_bundle.mjs";
import { detectionsToRow } from "./features.js";

const BASE = new URL("../", import.meta.url).href;
const SIDE = 512; // сторона квадрата для детекторів

async function createBoth(fileset, delegate) {
  const hand = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: BASE + "model/hand_landmarker.task", delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
  });
  const pose = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: BASE + "model/pose_landmarker_full.task", delegate },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.7,
    minPosePresenceConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });
  return { hand, pose };
}

export class Detector {
  static async create() {
    const fileset = await FilesetResolver.forVisionTasks(BASE + "vendor/mediapipe/wasm");
    // ?cpu у адресі -- примусово CPU (на випадок проблем із GPU телефона)
    const forceCpu = new URLSearchParams(location.search).has("cpu");
    let delegate = forceCpu ? "CPU" : "GPU", models;
    try {
      models = await createBoth(fileset, delegate);
    } catch (e) {
      console.warn("GPU-делегат недоступний, працюю на CPU:", e);
      delegate = "CPU";
      models = await createBoth(fileset, "CPU");
    }
    return new Detector(fileset, models, delegate);
  }

  constructor(fileset, models, delegate) {
    this.fileset = fileset;
    this.hand = models.hand;
    this.pose = models.pose;
    this.delegate = delegate;
    this.ts = 0;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = SIDE;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: false });
  }

  /** Вписати кадр у квадрат з чорними полями (без обрізання) і повернути canvas. */
  frameToSquare(source, w, h) {
    const ctx = this.ctx;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, SIDE, SIDE);
    const s = SIDE / Math.max(w, h);
    const dw = w * s, dh = h * s;
    ctx.drawImage(source, (SIDE - dw) / 2, (SIDE - dh) / 2, dw, dh);
    return this.canvas;
  }

  async _fallbackToCpu() {
    if (this.delegate === "CPU") return false;
    console.warn("Помилка на GPU, перемикаюсь на CPU");
    this.hand.close(); this.pose.close();
    const m = await createBoth(this.fileset, "CPU");
    this.hand = m.hand; this.pose = m.pose; this.delegate = "CPU";
    return true;
  }

  /**
   * Детекція на одному кадрі. Повертає { row (225), pose, hands } у координатах квадрата.
   * Мітки часу для режиму VIDEO мають строго зростати -- навіть між різними записами.
   */
  async detect(source, w, h) {
    const img = this.frameToSquare(source, w, h);
    this.ts = Math.max(this.ts + 1, Math.round(performance.now()));
    let pr, hr;
    try {
      pr = this.pose.detectForVideo(img, this.ts);
      hr = this.hand.detectForVideo(img, this.ts);
    } catch (e) {
      if (await this._fallbackToCpu()) return this.detect(source, w, h);
      throw e;
    }
    const pose = pr.landmarks && pr.landmarks.length ? pr.landmarks[0] : null;
    const hands = (hr.landmarks || []).map((lms, i) => ({
      landmarks: lms,
      label: hr.handedness?.[i]?.[0]?.categoryName ?? null,
    }));
    return { row: detectionsToRow(pose, hands), pose, hands };
  }
}

// ---------------------------------------------------------------------------
// Малювання скелета поверх кадру
// ---------------------------------------------------------------------------

const POSE_EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [0, 2], [0, 5]];
const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

export function drawSkeleton(ctx, side, det) {
  const line = (pts, edges, color, width) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    for (const [a, b] of edges) {
      if (!pts[a] || !pts[b]) continue;
      ctx.moveTo(pts[a].x * side, pts[a].y * side);
      ctx.lineTo(pts[b].x * side, pts[b].y * side);
    }
    ctx.stroke();
  };
  const dots = (pts, color, r) => {
    ctx.fillStyle = color;
    for (const p of pts) { ctx.beginPath(); ctx.arc(p.x * side, p.y * side, r, 0, 2 * Math.PI); ctx.fill(); }
  };
  if (det.pose) line(det.pose, POSE_EDGES, "rgba(255,255,255,0.75)", 3);
  // Колір за слотом датасету (ліва/права рука людини), а не за міткою детектора
  const leftPresent = det.row[33 * 3 + 2] > 0;
  for (const h of det.hands) {
    const isLeft = leftPresent && Math.abs(h.landmarks[0].x - det.row[33 * 3]) < 1e-6 && Math.abs(h.landmarks[0].y - det.row[33 * 3 + 1]) < 1e-6;
    const color = isLeft ? "#4cc9f0" : "#f9c74f";
    line(h.landmarks, HAND_EDGES, color, 2.5);
    dots(h.landmarks, color, 2.5);
  }
}
