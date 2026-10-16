// app.js -- інтерфейс: камера / відео з галереї -> точки MediaPipe -> слова.
// Два способи: "Запис речення" (записати й розпізнати) і "Наживо" (ковзне вікно 4 с щосекунди).

import { Detector, drawSkeleton } from "./detector.js";
import { resampleTo30fps, rowsToRaw, PersonAdapter } from "./features.js";
import { Recognizer, MIN_FRAMES, MAX_FRAMES, summarize, serverBase, serverHealth, recognizeOnline } from "./recognizer.js";

const $ = (id) => document.getElementById(id);
const view = $("view"), vctx = view.getContext("2d"), video = $("video");
const MAX_RECORD_S = 30;
const LIVE_WINDOW_MS = 4000, LIVE_HOP_MS = 1000, LIVE_REPEAT_MS = 6000, LIVE_MAX_WORDS = 40;

const state = {
  detector: null, recognizer: null, adapter: null,
  stream: null, cams: [], camId: null, facing: "user", looping: false,
  kind: "clip",                       // clip | live
  recording: false, recStart: 0, samples: [],
  live: null,                         // {samples, lastRun, lastAdded, emitted, words, running}
  busy: false, mode: "offline", history: [],
};

// ---------------------------------------------------------------------------
// Налаштування (браузерне сховище -- лише зручність; без нього все працює)
// ---------------------------------------------------------------------------

const store = {
  get(k, d) { try { const v = localStorage.getItem("uzhm." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("uzhm." + k, JSON.stringify(v)); } catch { /* приватний режим тощо */ } },
};

function setStatus(text, err = false) {
  $("status").textContent = text;
  $("status").classList.toggle("err", err);
}

function thresholds() {
  return { sure: parseFloat($("sureThr").value), maybe: parseFloat($("maybeThr").value) };
}

function initSettings(cfg) {
  $("sureThr").value = store.get("sure", cfg.threshold_sure);
  $("maybeThr").value = store.get("maybe", cfg.threshold_maybe);
  const show = () => {
    let { sure, maybe } = thresholds();
    if (maybe > sure) { $("maybeThr").value = sure; maybe = sure; }
    $("sureVal").textContent = sure.toFixed(2); $("maybeVal").textContent = maybe.toFixed(2);
    store.set("sure", sure); store.set("maybe", maybe);
  };
  $("sureThr").addEventListener("input", show);
  $("maybeThr").addEventListener("input", show);
  show();
  // QR-код від start.bat веде на застосунок з ?server=<адреса тунелю>: запам'ятовуємо її і прибираємо з адреси
  const fromLink = new URLSearchParams(location.search).get("server");
  if (fromLink) {
    store.set("server", fromLink.trim());
    store.set("mode", "online");  // прийшли за посиланням сервера -- одразу онлайн
    history.replaceState(null, "", location.pathname);
  }
  $("serverUrl").value = store.get("server", "");
  $("serverUrl").addEventListener("change", () => { store.set("server", $("serverUrl").value.trim()); if (state.mode === "online") checkServer(); });
  state.history = store.get("history", []);
  state.facing = store.get("facing", "user");
  renderHistory();
}

// ---------------------------------------------------------------------------
// Підлаштування під людину
// ---------------------------------------------------------------------------

function renderCalib() {
  const box = $("calib");
  box.classList.toggle("hidden", !state.recognizer?.adapts);
  if (!state.adapter) return;
  const s = state.adapter.frames / 30;
  const k = (state.recognizer?.cfg.adapt?.k ?? 300) / 30;
  $("calibText").textContent = s < 1 ? "Підлаштування під людину: ще немає даних"
    : `Підлаштування під людину: ${s.toFixed(0)} с жестів` + (s < k ? " (ще звикає)" : "");
}

$("btnNewPerson").addEventListener("click", () => {
  state.adapter?.reset();
  renderCalib();
  setStatus("Статистику людини скинуто. Перші 10–20 с жестів модель знову звикатиме.");
});

// ---------------------------------------------------------------------------
// Режим офлайн / онлайн і спосіб (запис / наживо)
// ---------------------------------------------------------------------------

async function checkServer() {
  const base = serverBase($("serverUrl").value);
  $("serverState").textContent = "Перевіряю сервер " + base + " …";
  try {
    const h = await serverHealth(base);
    $("serverState").textContent = `Сервер на зв'язку: ${h.models} модел${h.models === 1 ? "ь" : "і"} в ансамблі` +
      (h.llm ? `, речення складає ${h.llm}` : ", без складання речень (лише слова)");
    return true;
  } catch {
    $("serverState").textContent = "Сервер недоступний (" + base + "). Розпізнавання піде офлайн.";
    return false;
  }
}

function setMode(mode) {
  state.mode = mode;
  store.set("mode", mode);
  for (const [id, m] of [["modeOffline", "offline"], ["modeOnline", "online"]]) {
    $(id).classList.toggle("on", m === mode);
    $(id).setAttribute("aria-checked", String(m === mode));
  }
  if (mode === "online") checkServer();
}

function setKind(kind) {
  if (state.recording) stopRecording(false);
  if (state.live) stopLive();
  state.kind = kind;
  store.set("kind", kind);
  for (const [id, k] of [["kindClip", "clip"], ["kindLive", "live"]]) {
    $(id).classList.toggle("on", k === kind);
    $(id).setAttribute("aria-checked", String(k === kind));
  }
  $("btnRecord").textContent = kind === "live" ? "Почати наживо" : "Почати запис";
  $("liveBox").classList.toggle("hidden", kind !== "live");
}

// ---------------------------------------------------------------------------
// Камера
// ---------------------------------------------------------------------------

function camKind(track) {
  const s = track.getSettings ? track.getSettings() : {};
  const label = (track.label || "").toLowerCase();
  if (s.facingMode === "user" || /front|user|перед|facetime/.test(label)) return "front";
  if (s.facingMode === "environment" || /back|rear|environment|задн/.test(label)) return "back";
  return "other";
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const CAM_BASE = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } };

/** Спробувати відкрити камеру за списком обмежень. Android часто ще не встиг звільнити
 *  попередню камеру (NotReadableError / AbortError) -- тоді чекаємо і пробуємо ще раз. */
async function tryOpen(list) {
  let lastErr;
  for (const video of list) {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        return await navigator.mediaDevices.getUserMedia({ video, audio: false });
      } catch (e) {
        lastErr = e;
        if (e.name !== "NotReadableError" && e.name !== "AbortError" && e.name !== "TrackStartError") break;
        await sleep(400);
      }
    }
  }
  throw lastErr;
}

/** Камера за напрямком: спершу точний напрямок, потім камери з відповідною назвою, потім будь-яка. */
function constraintsFor(facing) {
  const want = facing === "environment" ? /back|rear|environment|задн/ : /front|user|перед|facetime/;
  const list = [{ ...CAM_BASE, facingMode: { exact: facing } }];
  for (const c of state.cams) if (c.deviceId && want.test((c.label || "").toLowerCase())) list.push({ ...CAM_BASE, deviceId: { exact: c.deviceId } });
  list.push({ ...CAM_BASE, facingMode: facing });
  return list;
}

async function startCamera(list = null) {
  stopCamera();
  try {
    state.stream = await tryOpen(list || [...constraintsFor(state.facing), true]);
  } catch (e) {
    setStatus("Немає доступу до камери: " + e.message + ". Камера працює лише через https або localhost.", true);
    return false;
  }
  const track = state.stream.getVideoTracks()[0];
  // Після дозволу браузер показує назви камер -- тепер можна скласти їх список.
  try {
    state.cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  } catch { state.cams = []; }
  state.camId = track.getSettings?.().deviceId || null;
  const kind = camKind(track);
  state.facing = kind === "back" ? "environment" : kind === "front" ? "user" : state.facing;
  store.set("facing", state.facing);
  video.srcObject = state.stream;
  await video.play();
  view.classList.toggle("mirror", kind === "front"); // дзеркалимо лише показ, не дані
  const name = kind === "front" ? "фронтальна" : kind === "back" ? "задня" : "камера";
  $("camInfo").textContent = `Камера: ${name}` + (state.cams.length > 1 ? " — ⟲ щоб змінити" : "");
  $("camInfo").classList.remove("hidden");
  $("placeholder").classList.add("hidden");
  $("badges").classList.remove("hidden");
  $("btnCamera").textContent = "Вимкнути камеру";
  $("btnRecord").disabled = !state.detector || !state.recognizer;
  $("btnFlip").disabled = false;
  setStatus("Камера увімкнена. Людину має бути видно від голови до пояса, обидві руки в кадрі.");
  state.looping = true;
  loop();
  return true;
}

function stopCamera() {
  state.looping = false;
  if (state.recording) stopRecording(false);
  if (state.live) stopLive();
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
  $("btnCamera").textContent = "Увімкнути камеру";
  $("btnRecord").disabled = true;
  $("btnFlip").disabled = true;
}

/** ⟲ -- фронтальна <-> задня. Якщо в цьому напрямку камеру не відкрити, -- будь-яка інша, ніж зараз. */
async function flipCamera() {
  const was = state.camId;
  const want = state.facing === "user" ? "environment" : "user";
  $("btnFlip").disabled = true;
  stopCamera();
  await sleep(300); // дати телефону звільнити поточну камеру
  const others = state.cams.filter((c) => c.deviceId && c.deviceId !== was)
    .map((c) => ({ ...CAM_BASE, deviceId: { exact: c.deviceId } }));
  const ok = await startCamera([...constraintsFor(want), ...others, ...constraintsFor(state.facing)]);
  if (ok && state.camId && state.camId === was) setStatus("Іншу камеру відкрити не вдалося -- працює та сама.", true);
}

function setBadges(row) {
  const any = (a, n) => { for (let j = a * 3; j < (a + n) * 3; j++) if (row[j] !== 0) return true; return false; };
  $("bBody").classList.toggle("on", row[11 * 3 + 2] > 0.5 && row[12 * 3 + 2] > 0.5);
  $("bLeft").classList.toggle("on", any(33, 21));
  $("bRight").classList.toggle("on", any(54, 21));
}

async function loop() {
  if (!state.looping) return;
  if (!state.busy && state.detector && video.readyState >= 2 && video.videoWidth) {
    state.busy = true;
    try {
      const det = await state.detector.detect(video, video.videoWidth, video.videoHeight);
      vctx.drawImage(state.detector.canvas, 0, 0, view.width, view.height);
      drawSkeleton(vctx, view.width, det);
      setBadges(det.row);
      const t = performance.now();
      if (state.recording) {
        state.samples.push({ t, row: det.row });
        const s = (t - state.recStart) / 1000;
        $("recTime").textContent = s.toFixed(1) + " с";
        if (s >= MAX_RECORD_S) stopRecording(true);
      }
      if (state.live) liveFrame(t, det.row);
    } catch (e) {
      console.error(e);
      setStatus("Помилка детекції: " + e.message, true);
    }
    state.busy = false;
  }
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------
// Запис речення
// ---------------------------------------------------------------------------

function startRecording() {
  state.samples = [];
  state.recording = true;
  state.recStart = performance.now();
  $("rec").classList.remove("hidden");
  $("btnRecord").textContent = "Зупинити";
  $("btnRecord").classList.add("recording");
  setStatus("Запис… покажіть речення жестами.");
}

function stopRecording(process = true) {
  state.recording = false;
  $("rec").classList.add("hidden");
  $("btnRecord").textContent = "Почати запис";
  $("btnRecord").classList.remove("recording");
  const samples = state.samples;
  state.samples = [];
  if (process) recognize(samples, "камера");
}

// ---------------------------------------------------------------------------
// Наживо: кожну секунду розпізнаємо останні 4 с
// ---------------------------------------------------------------------------

function startLive() {
  state.live = { samples: [], lastRun: performance.now(), lastAdded: null, emitted: new Map(), words: [], running: false };
  $("rec").classList.remove("hidden");
  $("btnRecord").textContent = "Зупинити";
  $("btnRecord").classList.add("recording");
  $("liveCaption").replaceChildren(emptySpan("слова з'являтимуться тут"));
  $("liveNow").replaceChildren();
  setStatus("Наживо: жестикулюйте, слова з'являтимуться під відео.");
}

function stopLive() {
  const live = state.live;
  state.live = null;
  $("rec").classList.add("hidden");
  $("btnRecord").textContent = state.kind === "live" ? "Почати наживо" : "Почати запис";
  $("btnRecord").classList.remove("recording");
  if (live && live.words.length) {
    addHistory({ sure: live.words.map((w) => ({ word: w.word, p: w.p })), sentence: null }, "наживо");
  }
  setStatus("Зупинено.");
}

function liveFrame(t, row) {
  const live = state.live;
  live.samples.push({ t, row });
  while (live.samples.length && live.samples[0].t < t - LIVE_WINDOW_MS - 500) live.samples.shift();
  $("recTime").textContent = "наживо";
  if (!live.running && t - live.lastRun >= LIVE_HOP_MS) {
    live.lastRun = t;
    live.running = true;
    liveStep(t).catch((e) => console.error(e)).finally(() => { live.running = false; });
  }
}

async function liveStep(now) {
  const live = state.live;
  if (!live) return;
  const win = live.samples.filter((s) => s.t >= now - LIVE_WINDOW_MS);
  const rows = resampleTo30fps(win);
  if (rows.length < MIN_FRAMES) return;
  const hands = rows.filter((r) => r[33 * 3 + 2] > 0 || r[54 * 3 + 2] > 0).length;
  const raw = rowsToRaw(rows);
  // до статистики людини -- лише кадри, яких ще не було в попередніх вікнах
  const fresh = live.lastAdded === null ? raw.T : Math.min(raw.T, Math.round(((now - live.lastAdded) / 1000) * 30));
  live.lastAdded = now;
  if (hands < rows.length * 0.3) { // рук майже не видно -- людина не жестикулює, не вгадуємо
    $("liveNow").replaceChildren(emptySpan("рук не видно"));
    return;
  }
  state.adapter.add(raw, raw.T - fresh);
  renderCalib();
  const probs = await state.recognizer.predictRaw(raw, state.adapter);
  if (!state.live) return;
  const thr = thresholds();
  const res = summarize(state.recognizer.cfg.vocab, probs, thr.sure, thr.maybe);
  for (const x of res.sure) {
    const last = live.emitted.get(x.word);
    if (last === undefined || now - last > LIVE_REPEAT_MS) live.words.push({ word: x.word, p: x.p, t: now });
    live.emitted.set(x.word, now);
  }
  live.words = live.words.slice(-LIVE_MAX_WORDS);
  renderLive(live, res, now);
}

function emptySpan(text) {
  const e = document.createElement("span"); e.className = "empty"; e.textContent = text; return e;
}

function renderLive(live, res, now) {
  const cap = $("liveCaption");
  cap.replaceChildren();
  if (!live.words.length) cap.append(emptySpan("поки нічого впевненого"));
  for (const w of live.words) {
    const s = document.createElement("span");
    s.className = "w " + (now - w.t < 2500 ? "new" : "old");
    s.textContent = w.word;
    cap.append(s, " "); // справжній пробіл: інакше слова зливаються і рядок не переноситься
  }
  const box = $("liveNow"); box.replaceChildren();
  const cand = [...res.sure, ...res.maybe].slice(0, 4);
  if (!cand.length) box.append(emptySpan("—"));
  for (const x of cand) box.append(chip(x, res.sure.includes(x) ? "sure" : "maybe"));
}

// ---------------------------------------------------------------------------
// Відео з галереї
// ---------------------------------------------------------------------------

function once(el, ev) { return new Promise((res, rej) => { el.addEventListener(ev, res, { once: true }); el.addEventListener("error", rej, { once: true }); }); }

async function processFile(file) {
  if (!state.detector || !state.recognizer) return;
  if (state.looping) stopCamera();
  const fv = document.createElement("video");
  fv.muted = true; fv.playsInline = true; fv.preload = "auto";
  fv.src = URL.createObjectURL(file);
  try {
    await once(fv, "loadeddata");
  } catch {
    setStatus("Не вдалося відкрити відео (формат не підтримується браузером).", true);
    return;
  }
  const dur = Math.min(fv.duration || 0, MAX_FRAMES / 30);
  const n = Math.floor(dur * 30);
  if (n < MIN_FRAMES) { setStatus("Відео закоротке.", true); return; }
  view.classList.remove("mirror");
  $("placeholder").classList.add("hidden");
  $("badges").classList.remove("hidden");
  $("progress").classList.remove("hidden");
  setStatus(`Обробка відео: ${n} кадрів…`);
  const samples = [];
  state.busy = true;
  try {
    for (let i = 0; i < n; i++) {
      fv.currentTime = Math.min(i / 30 + 1 / 60, fv.duration);
      await once(fv, "seeked");
      const det = await state.detector.detect(fv, fv.videoWidth, fv.videoHeight);
      vctx.drawImage(state.detector.canvas, 0, 0, view.width, view.height);
      drawSkeleton(vctx, view.width, det);
      setBadges(det.row);
      samples.push({ t: (i * 1000) / 30, row: det.row });
      $("progressBar").style.width = ((100 * (i + 1)) / n).toFixed(1) + "%";
    }
  } finally {
    state.busy = false;
    $("progress").classList.add("hidden");
    URL.revokeObjectURL(fv.src);
  }
  await recognize(samples, "відео: " + file.name);
}

// ---------------------------------------------------------------------------
// Розпізнавання запису
// ---------------------------------------------------------------------------

async function recognize(samples, source) {
  const rows = resampleTo30fps(samples).slice(0, MAX_FRAMES);
  if (rows.length < MIN_FRAMES) { setStatus("Запис закороткий — потрібно хоча б пів секунди.", true); return; }
  const handFrames = rows.filter((r) => r[33 * 3 + 2] > 0 || r[54 * 3 + 2] > 0).length;
  const thr = thresholds();
  const raw = rowsToRaw(rows);
  if (handFrames >= rows.length * 0.3) { state.adapter.add(raw); renderCalib(); } // поточний запис -- теж дані про людину
  let res, meta, note = "";
  const t0 = performance.now();
  if (state.mode === "online") {
    setStatus("Надсилаю точки на сервер…");
    try {
      const r = await recognizeOnline(serverBase($("serverUrl").value), rows, thr, state.adapter);
      res = { sure: r.sure, maybe: r.maybe, top: r.top, sentence: r.sentence, sentenceNote: r.sentence_note };
      meta = `онлайн · ансамбль ${r.models} · ${r.llm || "без мовної моделі"}`;
    } catch (e) {
      console.warn(e);
      note = "Сервер недоступний — показано офлайн-результат. ";
    }
  }
  if (!res) {
    setStatus("Розпізнаю на телефоні…");
    const probs = await state.recognizer.predictRaw(raw, state.adapter);
    res = summarize(state.recognizer.cfg.vocab, probs, thr.sure, thr.maybe);
    meta = "офлайн · на телефоні";
  }
  meta += ` · ${(rows.length / 30).toFixed(1)} с · ${((performance.now() - t0) / 1000).toFixed(1)} с обробки`;
  if (handFrames < rows.length * 0.3) note += "Руки було видно менш ніж на третині кадрів — результат ненадійний. ";
  render(res, meta, note);
  addHistory(res, source);
  setStatus(state.stream ? "Готово. Можна записувати наступне речення." : "Готово.");
}

function chip(x, cls) {
  const el = document.createElement("span");
  el.className = "chip " + cls;
  el.textContent = x.word;
  const s = document.createElement("small");
  s.textContent = Math.round(x.p * 100) + "%";
  el.append(s);
  return el;
}

function render(res, meta, note) {
  $("result").classList.remove("hidden");
  $("resMeta").textContent = meta;
  const fill = (id, items, cls, emptyText) => {
    const box = $(id); box.replaceChildren();
    if (!items.length) box.append(emptySpan(emptyText));
    for (const x of items) box.append(chip(x, cls));
  };
  fill("sure", res.sure, "sure", "немає впевнено розпізнаних слів");
  fill("maybe", res.maybe, "maybe", "—");
  const hasSentence = !!res.sentence;
  $("sentenceBox").classList.toggle("hidden", !hasSentence && !res.sentenceNote);
  $("sentence").textContent = hasSentence ? res.sentence : res.sentenceNote || "";
  $("resNote").textContent = note;
  const top = $("top"); top.replaceChildren();
  for (const x of res.top) {
    const li = document.createElement("li");
    const w = document.createElement("span"); w.textContent = x.word;
    const bar = document.createElement("span"); bar.className = "bar";
    const fillEl = document.createElement("i"); fillEl.style.width = (x.p * 100).toFixed(1) + "%"; bar.append(fillEl);
    const num = document.createElement("span"); num.className = "num"; num.textContent = Math.round(x.p * 100) + "%";
    li.append(w, bar, num); top.append(li);
  }
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function addHistory(res, source) {
  const words = res.sure.map((x) => x.word);
  const item = {
    when: new Date().toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit" }),
    source, text: res.sentence || (words.length ? words.join(", ") : "—"),
  };
  state.history = [item, ...state.history].slice(0, 10);
  store.set("history", state.history);
  renderHistory();
}

function renderHistory() {
  const ul = $("history"); ul.replaceChildren();
  $("historyBox").classList.toggle("hidden", state.history.length === 0);
  for (const h of state.history) {
    const li = document.createElement("li");
    const w = document.createElement("div"); w.className = "when"; w.textContent = `${h.when} · ${h.source}`;
    const t = document.createElement("div"); t.textContent = h.text;
    li.append(w, t); ul.append(li);
  }
}

// ---------------------------------------------------------------------------
// Старт
// ---------------------------------------------------------------------------

$("btnCamera").addEventListener("click", () => (state.stream ? stopCamera() : startCamera()));
$("btnFlip").addEventListener("click", () => flipCamera());
$("btnRecord").addEventListener("click", () => {
  if (state.kind === "live") return state.live ? stopLive() : startLive();
  return state.recording ? stopRecording(true) : startRecording();
});
$("fileInput").addEventListener("change", (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) processFile(f); });
$("modeOffline").addEventListener("click", () => setMode("offline"));
$("modeOnline").addEventListener("click", () => setMode("online"));
$("kindClip").addEventListener("click", () => setKind("clip"));
$("kindLive").addEventListener("click", () => setKind("live"));

if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
  navigator.serviceWorker.register("sw.js").catch((e) => console.warn("Service worker:", e));
}

(async () => {
  try {
    const [rec, det] = await Promise.all([Recognizer.create(), Detector.create()]);
    state.recognizer = rec; state.detector = det;
    state.adapter = new PersonAdapter(rec.cfg);
    initSettings(rec.cfg);
    setMode(store.get("mode", "offline"));
    setKind(store.get("kind", "clip"));
    renderCalib();
    $("buildInfo").textContent = `Модель: ${rec.cfg.model_desc}, ${rec.cfg.vocab.length} слів · детектор: ${det.delegate}`;
    setStatus("Готово. Увімкніть камеру або виберіть відео.");
    if (state.stream) $("btnRecord").disabled = false;
    window.__uzhm = { state, ready: true }; // для автотестів
  } catch (e) {
    console.error(e);
    setStatus("Не вдалося завантажити моделі: " + e.message, true);
  }
})();
