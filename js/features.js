// features.js -- перенесений на JavaScript конвеєр ознак з Python-частини диплома.
//
// Кадр у форматі датасету USL-Suspilne (MediaPipe Holistic, 225 чисел):
//   тіло 33 x (x, y, visibility) | ліва рука 21 x (x, y, 1/0) | права рука 21 x (x, y, 1/0)
// -> статика кешу (174)        = usl_dataset.holistic_to_landmarks + frame_features.build_frame_vector
// -> заповнення пропусків рук  = sign_data.fill_hand_gaps
// -> ознаки моделі v2 (243)    = sign_data.to_model_features
// -> нормалізація              = sign_data.normalize_features
//
// Будь-яка зміна тут має збігатися з Python: tests/parity.mjs порівнює результат
// із sign_data.py на реальних кліпах датасету.

export const ROW_DIM = 225;
export const STATIC_DIM = 174;
export const FEAT_DIM = 243;
export const FPS = 30;

const RELEVANT_POSE_INDICES = [0, 2, 5, 11, 12, 13, 14, 15, 16];
const CROP = 510; // кадр датасету 510x510; для квадратного кадру важливе лише співвідношення сторін
const POSE_PRESENCE_CONFIDENCE = 0.5;
const FINGERTIPS = [4, 8, 12, 16, 20];
const MAX_GAP = 4;

// Розкладка статики (174): поза 9x3 | ліва 21x3 | права 21x3 | форма Л 9 | форма П 9 | прапорці Л, П, поза
const S_POSE = 0, S_LH = 27, S_RH = 90, S_SH_L = 153, S_SH_R = 162, F_L = 171, F_R = 172, F_P = 173;
const HAND_DIM = 51, STATIC2 = 123, MOTION2 = 120;

// ---------------------------------------------------------------------------
// 225 -> 174
// ---------------------------------------------------------------------------

function norm3(ax, ay, az, bx, by, bz) {
  const dx = ax - bx, dy = ay - by, dz = az - bz;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// pose: Float64Array(9*4) x,y,z,vis; hand: Float64Array(21*3)
function assessPose(pose, left, right, lp, rp) {
  const shoulderConf = Math.min(pose[3 * 4 + 3], pose[4 * 4 + 3]);
  if (!(shoulderConf > POSE_PRESENCE_CONFIDENCE)) return false;
  const m = 0.02;
  for (const i of [3, 4]) {
    const x = pose[i * 4], y = pose[i * 4 + 1];
    if (!(x >= -m && x <= 1 + m && y >= -m && y <= 1 + m)) return false;
  }
  const shW = Math.hypot((pose[12] - pose[16]) * CROP, (pose[13] - pose[17]) * CROP);
  const ratio = shW / CROP;
  if (!(ratio >= 0.08 && ratio <= 0.9)) return false;
  const wrists = [[pose[28] * CROP, pose[29] * CROP], [pose[32] * CROP, pose[33] * CROP]];
  let best = Infinity, any = false;
  for (const [hand, present] of [[left, lp], [right, rp]]) {
    if (!present) continue;
    any = true;
    const hx = hand[0] * CROP, hy = hand[1] * CROP;
    for (const [wx, wy] of wrists) best = Math.min(best, Math.hypot(hx - wx, hy - wy));
  }
  if (any && best / shW > 0.6) return false;
  return true;
}

function handShape(h) { // h: нормалізована рука 21x3
  const palm0 = norm3(h[6], h[7], h[8], h[51], h[52], h[53]); // MCP 2 і 17
  const palm = palm0 > 1e-6 ? palm0 : 1.0;
  const out = new Float64Array(9);
  for (let i = 0; i < 5; i++) {
    const t = FINGERTIPS[i] * 3;
    out[i] = norm3(h[t], h[t + 1], h[t + 2], h[0], h[1], h[2]) / palm;
  }
  for (let i = 0; i < 4; i++) {
    const a = FINGERTIPS[i] * 3, b = FINGERTIPS[i + 1] * 3;
    out[5 + i] = norm3(h[a], h[a + 1], h[a + 2], h[b], h[b + 1], h[b + 2]) / palm;
  }
  return out;
}

function standaloneHand(hand) {
  const scale0 = norm3(hand[27], hand[28], hand[29], hand[0], hand[1], hand[2]);
  const s = scale0 > 1e-6 ? scale0 : 1.0;
  const out = new Float64Array(63);
  for (let j = 0; j < 21; j++) for (let c = 0; c < 3; c++) out[j * 3 + c] = (hand[j * 3 + c] - hand[c]) / s;
  return out;
}

/** Один кадр (225) -> статичний вектор (174). */
export function rowToStatic(row) {
  const pose = new Float64Array(36);
  RELEVANT_POSE_INDICES.forEach((idx, i) => {
    pose[i * 4] = row[idx * 3];
    pose[i * 4 + 1] = row[idx * 3 + 1];
    pose[i * 4 + 3] = row[idx * 3 + 2];
  });
  const left = new Float64Array(63), right = new Float64Array(63);
  let lp = false, rp = false;
  for (let j = 0; j < 21; j++) {
    const l = (33 + j) * 3, r = (54 + j) * 3;
    left[j * 3] = row[l]; left[j * 3 + 1] = row[l + 1];
    right[j * 3] = row[r]; right[j * 3 + 1] = row[r + 1];
    if (row[l] !== 0 || row[l + 1] !== 0 || row[l + 2] !== 0) lp = true;
    if (row[r] !== 0 || row[r + 1] !== 0 || row[r + 2] !== 0) rp = true;
  }
  const poseOk = assessPose(pose, left, right, lp, rp);
  const out = new Float32Array(STATIC_DIM);
  let leftN, rightN;
  if (!poseOk) {
    leftN = standaloneHand(left);
    rightN = standaloneHand(right);
  } else {
    const cx = (pose[12] + pose[16]) / 2, cy = (pose[13] + pose[17]) / 2, cz = (pose[14] + pose[18]) / 2;
    const sc0 = norm3(pose[12], pose[13], pose[14], pose[16], pose[17], pose[18]);
    const s = sc0 > 1e-6 ? sc0 : 1.0;
    for (let i = 0; i < 9; i++) {
      out[S_POSE + i * 3] = (pose[i * 4] - cx) / s;
      out[S_POSE + i * 3 + 1] = (pose[i * 4 + 1] - cy) / s;
      out[S_POSE + i * 3 + 2] = (pose[i * 4 + 2] - cz) / s;
    }
    leftN = new Float64Array(63); rightN = new Float64Array(63);
    const c = [cx, cy, cz];
    for (let k = 0; k < 63; k++) {
      leftN[k] = (left[k] - c[k % 3]) / s;
      rightN[k] = (right[k] - c[k % 3]) / s;
    }
  }
  out.set(leftN, S_LH);
  out.set(rightN, S_RH);
  if (lp) out.set(handShape(leftN), S_SH_L);
  if (rp) out.set(handShape(rightN), S_SH_R);
  out[F_L] = lp ? 1 : 0;
  out[F_R] = rp ? 1 : 0;
  out[F_P] = poseOk ? 1 : 0;
  return out;
}

// ---------------------------------------------------------------------------
// Пропуски рук (sign_data.fill_hand_gaps)
// ---------------------------------------------------------------------------

export function fillHandGaps(stat, maxGap = MAX_GAP) {
  const T = stat.length;
  const out = stat.map((r) => Float32Array.from(r));
  const poseOk = out.map((r) => r[F_P] > 0.5);
  for (const [flag, c0, sh0] of [[F_L, S_LH, S_SH_L], [F_R, S_RH, S_SH_R]]) {
    const present = out.map((r) => r[flag] > 0.5);
    let t = 0;
    while (t < T) {
      if (present[t]) { t++; continue; }
      const start = t;
      while (t < T && !present[t]) t++;
      const a = start - 1, b = t;
      if (a < 0 || b >= T || b - start > maxGap) continue;
      let anyP = false, allP = true;
      for (let k = a; k <= b; k++) { anyP = anyP || poseOk[k]; allP = allP && poseOk[k]; }
      if (anyP && !allP) continue;
      for (let k = start; k < b; k++) {
        const w = Math.fround((k - a) / (b - a));
        for (const [o, n] of [[c0, 63], [sh0, 9]]) {
          for (let j = o; j < o + n; j++) out[k][j] = (1 - w) * out[a][j] + w * out[b][j];
        }
        out[k][flag] = 1;
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// 174 -> 243 (sign_data.to_model_features) і нормалізація
// ---------------------------------------------------------------------------

function handBlock(r, c0, sh0, flag, poseOk, feat, valid, off) {
  const present = r[flag] > 0.5;
  const wx = r[c0], wy = r[c0 + 1];
  const p0 = Math.hypot(r[c0 + 27] - wx, r[c0 + 28] - wy);
  const palm = p0 > 1e-6 ? p0 : 1.0;
  const wristOk = present && poseOk;
  feat[off] = wristOk ? wx : 0; feat[off + 1] = wristOk ? wy : 0;
  valid[off] = valid[off + 1] = wristOk;
  for (let j = 1; j < 21; j++) {
    const k = off + 2 + (j - 1) * 2;
    feat[k] = present ? (r[c0 + j * 3] - wx) / palm : 0;
    feat[k + 1] = present ? (r[c0 + j * 3 + 1] - wy) / palm : 0;
    valid[k] = valid[k + 1] = present;
  }
  for (let j = 0; j < 9; j++) {
    feat[off + 42 + j] = present ? r[sh0 + j] : 0;
    valid[off + 42 + j] = present;
  }
}

/** Статика (T x 174) -> нормалізовані ознаки моделі, Float32Array(T*243). */
export function staticToModelInput(stat, cfg) {
  const T = stat.length;
  const st = [], stv = [];
  for (let t = 0; t < T; t++) {
    const r = stat[t];
    const f = new Float32Array(STATIC2), v = new Array(STATIC2).fill(false);
    const poseOk = r[F_P] > 0.5;
    for (let i = 0; i < 9; i++) {
      f[i * 2] = r[S_POSE + i * 3]; f[i * 2 + 1] = r[S_POSE + i * 3 + 1];
      v[i * 2] = v[i * 2 + 1] = poseOk;
    }
    handBlock(r, S_LH, S_SH_L, F_L, poseOk, f, v, 18);
    handBlock(r, S_RH, S_SH_R, F_R, poseOk, f, v, 18 + HAND_DIM);
    f[120] = r[F_L]; f[121] = r[F_R]; f[122] = r[F_P];
    v[120] = v[121] = v[122] = true;
    st.push(f); stv.push(v);
  }
  const out = new Float32Array(T * FEAT_DIM);
  const { mean, std } = cfg;
  for (let t = 0; t < T; t++) {
    const o = t * FEAT_DIM;
    for (let j = 0; j < STATIC2; j++) {
      out[o + j] = stv[t][j] ? clip(Math.fround((st[t][j] - mean[j]) / std[j])) : 0;
    }
    for (let j = 0; j < MOTION2; j++) {
      const ok = t > 0 && stv[t][j] && stv[t - 1][j];
      const k = STATIC2 + j;
      out[o + k] = ok ? clip(Math.fround((Math.fround(st[t][j] - st[t - 1][j]) - mean[k]) / std[k])) : 0;
    }
  }
  return out;
}

function clip(x) { return x > 10 ? 10 : x < -10 ? -10 : x; }

/** Повний шлях: кадри 225 (30 fps) -> вхід моделі. */
export function rowsToModelInput(rows, cfg) {
  return staticToModelInput(fillHandGaps(rows.map(rowToStatic)), cfg);
}

// ---------------------------------------------------------------------------
// Живі детекції -> кадри датасету
// ---------------------------------------------------------------------------

/**
 * Результат MediaPipe Tasks (PoseLandmarker + HandLandmarker) для квадратного кадру
 * -> рядок 225 у форматі датасету.
 *
 * У датасеті "ліва рука" -- це ліва рука самої людини, її зап'ястя збігається з
 * точкою пози 15 (перевірено на кліпах датасету: 4193 проти 111). Мітки handedness
 * від HandLandmarker для недзеркального кадру ненадійні, тому руку прив'язуємо
 * до найближчого зап'ястя пози; мітку беремо лише коли пози немає.
 */
export function detectionsToRow(poseLms, hands) {
  const row = new Float32Array(ROW_DIM);
  if (poseLms) {
    for (let i = 0; i < 33; i++) {
      const lm = poseLms[i];
      const vis = lm.visibility ?? 0;
      const conf = lm.presence != null ? Math.min(vis, lm.presence) : vis;
      row[i * 3] = lm.x; row[i * 3 + 1] = lm.y; row[i * 3 + 2] = conf;
    }
  }
  const slots = assignHands(poseLms, hands);
  for (const [slot, lms] of slots) {
    const base = slot === "L" ? 33 : 54;
    for (let j = 0; j < 21; j++) {
      row[(base + j) * 3] = lms[j].x;
      row[(base + j) * 3 + 1] = lms[j].y;
      row[(base + j) * 3 + 2] = 1;
    }
  }
  return row;
}

export const LABEL_FOR_LEFT_SLOT = "Left"; // перевірено tests/tasks_vs_dataset.mjs: слот "ліва" датасету = мітка "Left" (500 проти 15)

function assignHands(poseLms, hands) {
  if (!hands || hands.length === 0) return [];
  const usable = poseLms && (poseLms[15].visibility ?? 0) > 0.3 && (poseLms[16].visibility ?? 0) > 0.3;
  if (usable) {
    const d = (h, i) => Math.hypot(h.landmarks[0].x - poseLms[i].x, h.landmarks[0].y - poseLms[i].y);
    if (hands.length === 1) {
      const h = hands[0];
      return [[d(h, 15) <= d(h, 16) ? "L" : "R", h.landmarks]];
    }
    const [a, b] = hands;
    const straight = d(a, 15) + d(b, 16), swapped = d(a, 16) + d(b, 15);
    return straight <= swapped ? [["L", a.landmarks], ["R", b.landmarks]] : [["R", a.landmarks], ["L", b.landmarks]];
  }
  const out = [], used = new Set();
  for (const h of hands) {
    const slot = h.label === LABEL_FOR_LEFT_SLOT ? "L" : "R";
    if (used.has(slot)) continue;
    used.add(slot);
    out.push([slot, h.landmarks]);
  }
  return out;
}

/**
 * Кадри з довільними моментами часу (мс) -> рівномірні 30 кадрів/с, як у датасеті.
 * Рух (різниця сусідніх кадрів) залежить від частоти кадрів, тож без цього модель
 * бачила б на телефоні "повільніші" чи "різкіші" жести, ніж у навчанні.
 * Координати інтерполюємо лінійно, якщо частина (тіло/рука) є в обох сусідніх
 * кадрах, інакше беремо найближчий кадр.
 */
export function resampleTo30fps(samples) {
  if (samples.length === 0) return [];
  const t0 = samples[0].t, t1 = samples[samples.length - 1].t;
  const n = Math.max(1, Math.round(((t1 - t0) / 1000) * FPS) + 1);
  const out = [];
  let k = 0;
  const parts = [[0, 33], [33, 21], [54, 21]];
  for (let i = 0; i < n; i++) {
    const t = t0 + (i * 1000) / FPS;
    while (k + 1 < samples.length && samples[k + 1].t <= t) k++;
    const a = samples[k], b = samples[Math.min(k + 1, samples.length - 1)];
    const span = b.t - a.t;
    const w = span > 0 ? Math.min(1, Math.max(0, (t - a.t) / span)) : 0;
    const row = new Float32Array(ROW_DIM);
    for (const [p0, cnt] of parts) {
      const pa = partPresent(a.row, p0, cnt), pb = partPresent(b.row, p0, cnt);
      const src = w < 0.5 ? a.row : b.row;
      for (let j = p0 * 3; j < (p0 + cnt) * 3; j++) {
        row[j] = pa && pb ? (1 - w) * a.row[j] + w * b.row[j] : src[j];
      }
    }
    out.push(row);
  }
  return out;
}

function partPresent(row, p0, cnt) {
  for (let j = p0 * 3; j < (p0 + cnt) * 3; j++) if (row[j] !== 0) return true;
  return false;
}
