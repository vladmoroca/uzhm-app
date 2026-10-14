// app.js -- інтерфейс: камера / відео з галереї -> точки MediaPipe -> слова.

import { Detector, drawSkeleton } from "./detector.js";
import { resampleTo30fps } from "./features.js";
import { Recognizer, MIN_FRAMES, MAX_FRAMES, summarize, serverBase, serverHealth, recognizeOnline } from "./recognizer.js";

const $ = (id) => document.getElementById(id);
const view = $("view"), vctx = view.getContext("2d"), video = $("video");
const MAX_RECORD_S = 30;

const state = {
  detector: null, recognizer: null,
  stream: null, facing: "user", looping: false,
  recording: false, recStart: 0, samples: [],
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
  const sure = store.get("sure", cfg.threshold_sure), maybe = store.get("maybe", cfg.threshold_maybe);
  $("sureThr").value = sure; $("maybeThr").value = maybe;
  const show = () => {
    let { sure, maybe } = thresholds();
    if (maybe > sure) { $("maybeThr").value = sure; maybe = sure; }
    $("sureVal").textContent = sure.toFixed(2); $("maybeVal").textContent = maybe.toFixed(2);
    store.set("sure", sure); store.set("maybe", maybe);
  };
  $("sureThr").addEventListener("input", show);
  $("maybeThr").addEventListener("input", show);
  show();
  $("serverUrl").value = store.get("server", "");
  $("serverUrl").addEventListener("change", () => { store.set("server", $("serverUrl").value.trim()); if (state.mode === "online") checkServer(); });
  state.history = store.get("history", []);
  renderHistory();
}

// ---------------------------------------------------------------------------
// Режим офлайн / онлайн
// ---------------------------------------------------------------------------

async function checkServer() {
  const base = serverBase($("serverUrl").value);
  $("serverState").textContent = "Перевіряю сервер " + base + " …";
  try {
    const h = await serverHealth(base);
    $("serverState").textContent = `Сервер на зв'язку: ${h.models} модел${h.models === 1 ? "ь" : "і"} в ансамблі` +
      (h.llm ? `, речення складає ${h.llm}` : ", мовна модель недоступна (лише слова)");
    return true;
  } catch (e) {
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

// ---------------------------------------------------------------------------
// Камера
// ---------------------------------------------------------------------------

async function startCamera() {
  stopCamera();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: state.facing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false,
    });
  } catch (e) {
    setStatus("Немає доступу до камери: " + e.message + ". Камера працює лише через https або localhost.", true);
    return;
  }
  video.srcObject = state.stream;
  await video.play();
  view.classList.toggle("mirror", state.facing === "user"); // дзеркалимо лише показ, не дані
  $("placeholder").classList.add("hidden");
  $("badges").classList.remove("hidden");
  $("btnCamera").textContent = "Вимкнути камеру";
  $("btnRecord").disabled = !state.detector || !state.recognizer;
  $("btnFlip").disabled = false;
  setStatus("Камера увімкнена. Станьте так, щоб було видно тіло й обидві руки.");
  state.looping = true;
  loop();
}

function stopCamera() {
  state.looping = false;
  if (state.recording) stopRecording(false);
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
  $("btnCamera").textContent = "Увімкнути камеру";
  $("btnRecord").disabled = true;
  $("btnFlip").disabled = true;
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
      if (state.recording) {
        const t = performance.now();
        state.samples.push({ t, row: det.row });
        const s = (t - state.recStart) / 1000;
        $("recTime").textContent = s.toFixed(1) + " с";
        if (s >= MAX_RECORD_S) stopRecording(true);
      }
    } catch (e) {
      console.error(e);
      setStatus("Помилка детекції: " + e.message, true);
    }
    state.busy = false;
  }
  requestAnimationFrame(loop);
}

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
// Відео з галереї
// ---------------------------------------------------------------------------

function once(el, ev) { return new Promise((res, rej) => { el.addEventListener(ev, res, { once: true }); el.addEventListener("error", rej, { once: true }); }); }

async function processFile(file) {
  if (!state.detector || !state.recognizer) return;
  const wasLooping = state.looping;
  if (wasLooping) stopCamera();
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
// Розпізнавання
// ---------------------------------------------------------------------------

async function recognize(samples, source) {
  const rows = resampleTo30fps(samples).slice(0, MAX_FRAMES);
  if (rows.length < MIN_FRAMES) { setStatus("Запис закороткий — потрібно хоча б пів секунди.", true); return; }
  const handFrames = rows.filter((r) => r[33 * 3 + 2] > 0 || r[54 * 3 + 2] > 0).length;
  const thr = thresholds();
  let res, meta, note = "";
  const t0 = performance.now();
  if (state.mode === "online") {
    setStatus("Надсилаю точки на сервер…");
    try {
      const r = await recognizeOnline(serverBase($("serverUrl").value), rows, thr);
      res = { sure: r.sure, maybe: r.maybe, top: r.top, sentence: r.sentence, sentenceNote: r.sentence_note };
      meta = `онлайн · ансамбль ${r.models} · ${r.llm || "без мовної моделі"}`;
    } catch (e) {
      console.warn(e);
      note = "Сервер недоступний — показано офлайн-результат. ";
    }
  }
  if (!res) {
    setStatus("Розпізнаю на телефоні…");
    const probs = await state.recognizer.predict(rows);
    res = summarize(state.recognizer.cfg.vocab, probs, thr.sure, thr.maybe);
    meta = "офлайн · на телефоні";
  }
  const secs = rows.length / 30;
  meta += ` · ${secs.toFixed(1)} с · ${((performance.now() - t0) / 1000).toFixed(1)} с обробки`;
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
    if (!items.length) { const e = document.createElement("span"); e.className = "empty"; e.textContent = emptyText; box.append(e); }
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
$("btnFlip").addEventListener("click", () => { state.facing = state.facing === "user" ? "environment" : "user"; startCamera(); });
$("btnRecord").addEventListener("click", () => (state.recording ? stopRecording(true) : startRecording()));
$("fileInput").addEventListener("change", (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) processFile(f); });
$("modeOffline").addEventListener("click", () => setMode("offline"));
$("modeOnline").addEventListener("click", () => setMode("online"));

if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
  navigator.serviceWorker.register("sw.js").catch((e) => console.warn("Service worker:", e));
}

(async () => {
  try {
    const [rec, det] = await Promise.all([Recognizer.create(), Detector.create()]);
    state.recognizer = rec; state.detector = det;
    initSettings(rec.cfg);
    setMode(store.get("mode", "offline"));
    $("buildInfo").textContent = `Модель: ${rec.cfg.model_desc}, ${rec.cfg.vocab.length} слів · детектор: ${det.delegate}`;
    setStatus("Готово. Увімкніть камеру або виберіть відео.");
    if (state.stream) $("btnRecord").disabled = false;
    window.__uzhm = { state, ready: true }; // для автотестів
  } catch (e) {
    console.error(e);
    setStatus("Не вдалося завантажити моделі: " + e.message, true);
  }
})();
