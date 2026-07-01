/* ── State ── */
const state = {
  video: null,
  stream: null,
  activeAlgorithm: null,
  captureTimer: null,
  isCapturing: false,
  cameraReady: false,
};

/* ── DOM refs ── */
const video = document.getElementById("video");
const captureCanvas = document.getElementById("capture-canvas");
const overlayCanvas = document.getElementById("overlay");
const startBtn = document.getElementById("start-cam");
const camStatus = document.getElementById("cam-status");
const algoCards = document.querySelectorAll(".algo-card");
const resultsEmpty = document.getElementById("results-empty");
const resultsContent = document.getElementById("results-content");
const rAlgo = document.getElementById("r-algo");
const rFaces = document.getElementById("r-faces");
const rTime = document.getElementById("r-time");
const rConfidence = document.getElementById("r-confidence");
const rLandmarks = document.getElementById("r-landmarks");
const rError = document.getElementById("r-error");

/* ── Camera ── */
startBtn.addEventListener("click", async () => {
  if (state.stream) {
    stopCamera();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    state.stream = stream;
    state.cameraReady = true;
    startBtn.textContent = "⏹ Stop Camera";
    startBtn.classList.remove("primary");
    startBtn.classList.add("danger");
    camStatus.textContent = "📷 Camera on";
    camStatus.style.color = "#22c55e";

    // Sync overlay canvas size to video once metadata loads
    video.addEventListener("loadedmetadata", syncCanvasSizes, { once: true });
  } catch (err) {
    camStatus.textContent = "❌ Camera access denied";
    camStatus.style.color = "#ef4444";
    console.error("Camera error:", err);
  }
});

function syncCanvasSizes() {
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  overlayCanvas.width = video.videoWidth;
  overlayCanvas.height = video.videoHeight;
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  video.srcObject = null;
  state.cameraReady = false;
  stopCaptureLoop();
  startBtn.textContent = "📷 Start Camera";
  startBtn.classList.remove("danger");
  startBtn.classList.add("primary");
  camStatus.textContent = "Camera off";
  camStatus.style.color = "#94a3b8";
  clearOverlay();
  state.activeAlgorithm = null;
  algoCards.forEach((c) => c.classList.remove("active"));
  showResultsEmpty();
}

/* ── Overlay Canvas ── */
function clearOverlay() {
  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function drawOverlay(result) {
  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (result.error || !result.faces.length) return;

  // Draw bounding boxes
  ctx.strokeStyle = "#22c55e";
  ctx.lineWidth = 2;
  ctx.fillStyle = "rgba(34, 197, 94, 0.15)";

  result.faces.forEach((face) => {
    const x = face.x1;
    const y = face.y1;
    const w = face.x2 - face.x1;
    const h = face.y2 - face.y1;

    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);

    // Confidence label above box
    if (face.confidence != null) {
      ctx.fillStyle = "#22c55e";
      ctx.font = "12px -apple-system, sans-serif";
      ctx.fillText(`${(face.confidence * 100).toFixed(0)}%`, x, y - 4);
      ctx.fillStyle = "rgba(34, 197, 94, 0.15)";
    }
  });

  // Draw landmarks as small dots
  if (result.landmarks && result.landmarks.length > 0) {
    ctx.fillStyle = "#f59e0b";
    result.landmarks.forEach(([lx, ly]) => {
      ctx.beginPath();
      ctx.arc(lx, ly, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

/* ── Algorithm Selection ── */
algoCards.forEach((card) => {
  card.addEventListener("click", () => {
    const algo = card.dataset.algo;
    if (!state.cameraReady) {
      camStatus.textContent = "⚠️ Start camera first";
      camStatus.style.color = "#f59e0b";
      return;
    }
    if (state.activeAlgorithm === algo) {
      deactivateAlgorithm();
      return;
    }
    activateAlgorithm(algo);
  });
});

function activateAlgorithm(algo) {
  state.activeAlgorithm = algo;
  algoCards.forEach((c) => {
    c.classList.toggle("active", c.dataset.algo === algo);
    c.classList.remove("error");
  });
  showResultsEmpty(false);
  resultsContent.hidden = false;
  rAlgo.textContent = algo;
  rFaces.textContent = "…";
  rTime.textContent = "…";
  rConfidence.textContent = "…";
  rLandmarks.textContent = "…";
  rError.hidden = true;
  startCaptureLoop();
}

function deactivateAlgorithm() {
  stopCaptureLoop();
  state.activeAlgorithm = null;
  algoCards.forEach((c) => c.classList.remove("active"));
  showResultsEmpty();
  clearOverlay();
}

function showResultsEmpty(show = true) {
  resultsEmpty.hidden = !show;
  resultsContent.hidden = show;
  if (show) rError.hidden = true;
}

/* ── Capture Loop ── */
function startCaptureLoop() {
  if (state.captureTimer) return;
  state.isCapturing = true;

  setTimeout(() => {
    if (!state.isCapturing) return;
    captureAndDetect();
    state.captureTimer = setInterval(captureAndDetect, 1200);
  }, 200);
}

function stopCaptureLoop() {
  state.isCapturing = false;
  if (state.captureTimer) {
    clearInterval(state.captureTimer);
    state.captureTimer = null;
  }
  algoCards.forEach((c) => c.classList.remove("loading"));
}

async function captureAndDetect() {
  if (!state.activeAlgorithm || !state.cameraReady) return;
  if (video.readyState < 2) return;

  const card = document.querySelector(`.algo-card[data-algo="${state.activeAlgorithm}"]`);
  if (card) card.classList.add("loading");

  // Capture frame
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(video, 0, 0);
  const imageData = captureCanvas.toDataURL("image/jpeg", 0.8);

  try {
    const resp = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageData, algorithm: state.activeAlgorithm }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const result = await resp.json();
    updateResults(result);
  } catch (err) {
    if (card) card.classList.remove("loading");
    rError.hidden = false;
    rError.textContent = `❌ Error: ${err.message}`;
    console.error("Detection error:", err);
  }
}

/* ── Update Results ── */
function updateResults(result) {
  const card = document.querySelector(`.algo-card[data-algo="${result.algorithm}"]`);
  if (card) {
    card.classList.remove("loading");
    if (result.error) card.classList.add("error");
  }

  rAlgo.textContent = result.algorithm;
  rFaces.textContent = `${result.faces.length} face${result.faces.length !== 1 ? "s" : ""}`;

  if (result.error) {
    rTime.textContent = "—";
    rConfidence.textContent = "—";
    rLandmarks.textContent = "—";
    rError.hidden = false;
    rError.textContent = `⚠️ ${result.error}`;
    clearOverlay();
    return;
  }
  rError.hidden = true;

  rTime.textContent = `${result.time_ms} ms`;
  rLandmarks.textContent = result.landmarks
    ? `${result.landmarks.length} pts`
    : "✗ (not supported)";

  // Avg confidence
  const confs = result.faces.map((f) => f.confidence).filter(Boolean);
  if (confs.length > 0) {
    const avg = (confs.reduce((a, b) => a + b, 0) / confs.length * 100).toFixed(1);
    rConfidence.textContent = `${avg}%`;
  } else {
    rConfidence.textContent = "—";
  }

  // Draw on overlay canvas
  drawOverlay(result);
}

/* ── Keyboard shortcut ── */
document.addEventListener("keydown", (e) => {
  if (e.key === " ") {
    e.preventDefault();
    startBtn.click();
  }
});
