// Service worker: кешує весь застосунок і моделі, щоб він працював без інтернету.
// Файл sw.js генерує build.py із sw.template.js -- редагуйте шаблон.
const VERSION = "724c81926b2c";
const CACHE = "uzhm-" + VERSION;
const ASSETS = [
  "./",
  "./css/app.css",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./index.html",
  "./js/app.js",
  "./js/detector.js",
  "./js/features.js",
  "./js/recognizer.js",
  "./manifest.webmanifest",
  "./model/config.json",
  "./model/hand_landmarker.task",
  "./model/keywords.onnx",
  "./model/pose_landmarker_full.task",
  "./vendor/mediapipe/vision_bundle.mjs",
  "./vendor/mediapipe/wasm/vision_wasm_internal.js",
  "./vendor/mediapipe/wasm/vision_wasm_internal.wasm",
  "./vendor/mediapipe/wasm/vision_wasm_nosimd_internal.js",
  "./vendor/mediapipe/wasm/vision_wasm_nosimd_internal.wasm",
  "./vendor/ort/ort-wasm-simd-threaded.mjs",
  "./vendor/ort/ort-wasm-simd-threaded.wasm",
  "./vendor/ort/ort.wasm.min.mjs"
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k.startsWith("uzhm-") && k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.pathname.includes("/api/") || url.origin !== location.origin) return;
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request)));
});
