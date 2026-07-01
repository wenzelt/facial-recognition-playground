import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .schemas import DetectResponse, FaceBox


MODELS_DIR = Path(__file__).resolve().parent / "models"


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two axis-aligned boxes (x1, y1, x2, y2)."""
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DetectorManager:
    """Manages multiple on-device face detection algorithms."""

    def __init__(self):
        self._haar_cascade = None
        self._mp_face_mesh = None
        # PCA detector state (trained on-the-fly)
        self._pca_model = None
        self._pca_samples = []
        self._pca_trained = False
        self._pca_components = None
        self._pca_mean = None
        self._pca_threshold = None

    # --- Helpers ---

    def _decode_image(self, image: str) -> np.ndarray:
        """Convert a base64 data-URL JPEG into a BGR numpy array."""
        if "," in image:
            image = image.split(",", 1)[1]
        import base64

        raw = base64.b64decode(image)
        pil = Image.open(BytesIO(raw))
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _run_and_time(self, fn, img_bgr: cv2.Mat) -> tuple[float, any]:
        start = time.perf_counter()
        result = fn(img_bgr)
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, result

    # --- Individual detectors ---

    def detect_haar(self, img_bgr: cv2.Mat) -> tuple[list[FaceBox], list[list[float]] | None, str | None]:
        if self._haar_cascade is None:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(path)
            if self._haar_cascade.empty():
                return [], None, "Failed to load Haar cascade model"

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        rects = self._haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        faces = [
            FaceBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h), confidence=None)
            for x, y, w, h in rects
        ]
        return faces, None, None

    def detect_mediapipe_mesh(self, img_bgr: cv2.Mat) -> tuple[list[FaceBox], list[list[float]] | None, str | None]:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (FaceLandmarker,
                                                    FaceLandmarkerOptions)

        if self._mp_face_mesh is None:
            model_path = str(MODELS_DIR / "face_landmarker.task")
            if not Path(model_path).exists():
                return [], None, f"Model not found at {model_path}"
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                num_faces=3,
                min_face_detection_confidence=0.5,
            )
            self._mp_face_mesh = FaceLandmarker.create_from_options(options)

        import mediapipe as mp

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._mp_face_mesh.detect(mp_image)

        if not results.face_landmarks:
            return [], None, None

        h, w, _ = img_bgr.shape
        faces = []
        all_landmarks = []
        for face_landmarks in results.face_landmarks:
            xs = [lm.x for lm in face_landmarks]
            ys = [lm.y for lm in face_landmarks]
            faces.append(
                FaceBox(
                    x1=float(min(xs) * w),
                    y1=float(min(ys) * h),
                    x2=float(max(xs) * w),
                    y2=float(max(ys) * h),
                    confidence=None,
                )
            )
            if not all_landmarks:
                all_landmarks = [
                    [float(lm.x * w), float(lm.y * h)]
                    for lm in face_landmarks
                ]

        return faces, all_landmarks if all_landmarks else None, None

    def detect_hog(self, img_bgr: cv2.Mat) -> tuple[list[FaceBox], list[list[float]] | None, str | None]:
        try:
            import face_recognition_models  # noqa: F401
        except ImportError:
            return [], None, "face_recognition_models not installed"
        import face_recognition

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        faces = [
            FaceBox(x1=float(l), y1=float(t), x2=float(r), y2=float(b), confidence=None)
            for t, r, b, l in locations
        ]
        return faces, None, None

    def detect_cnn(self, img_bgr: cv2.Mat) -> tuple[list[FaceBox], list[list[float]] | None, str | None]:
        try:
            import face_recognition_models  # noqa: F401
        except ImportError:
            return [], None, "face_recognition_models not installed"
        import face_recognition

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="cnn")
        faces = [
            FaceBox(x1=float(l), y1=float(t), x2=float(r), y2=float(b), confidence=None)
            for t, r, b, l in locations
        ]
        return faces, None, None

    def detect_pca(self, img_bgr: cv2.Mat) -> tuple[list[FaceBox], list[list[float]] | None, str | None]:
        """PCA Eigenfaces detector — trains on-the-fly from Haar-detected face crops,
        then detects via sliding window reconstruction error."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        PATCH_SIZE = 48

        # --- Training phase: collect samples via Haar ---
        if not self._pca_trained:
            if self._haar_cascade is None:
                path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._haar_cascade = cv2.CascadeClassifier(path)

            rects = self._haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (x, y, fw, fh) in rects:
                face = gray[y : y + fh, x : x + fw]
                resized = cv2.resize(face, (PATCH_SIZE, PATCH_SIZE))
                self._pca_samples.append(resized.flatten().astype(np.float32) / 255.0)

            # Need enough samples before training
            if len(self._pca_samples) >= 10:
                from sklearn.decomposition import PCA

                X = np.array(self._pca_samples)
                n_comp = min(15, len(self._pca_samples))
                pca = PCA(n_components=n_comp)
                pca.fit(X)

                self._pca_mean = pca.mean_
                self._pca_components = pca.components_
                self._pca_model = pca

                # Threshold: mean training reconstruction error × 2
                train_recon = np.mean(
                    (X - pca.inverse_transform(pca.transform(X))) ** 2, axis=1
                )
                self._pca_threshold = float(np.mean(train_recon) * 2.0)
                self._pca_trained = True

            # Return Haar results during training phase
            faces = [
                FaceBox(x1=float(x), y1=float(y), x2=float(x + fw), y2=float(y + fh), confidence=None)
                for x, y, fw, fh in rects
            ]
            n_collected = len(self._pca_samples)
            status = f"Collecting: {n_collected}/10 frames" if n_collected < 10 else None
            return faces, None, status

        # --- Detection phase: sliding window PCA reconstruction error ---
        STRIDE = 12
        SCALE_FACTOR = 0.8
        detections = []
        scales = [1.0]
        scale = SCALE_FACTOR
        while min(int(h * scale), int(w * scale)) >= PATCH_SIZE:
            scales.append(scale)
            scale *= SCALE_FACTOR
        scales = sorted(set(scales))

        for s in scales:
            sh, sw = int(h * s), int(w * s)
            scaled = cv2.resize(gray, (sw, sh))
            for y in range(0, sh - PATCH_SIZE + 1, STRIDE):
                for x in range(0, sw - PATCH_SIZE + 1, STRIDE):
                    patch = scaled[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                    vec = patch.flatten().astype(np.float32) / 255.0

                    # Project → reconstruct → error
                    proj = self._pca_model.transform([vec])[0]
                    recon = self._pca_model.inverse_transform([proj])[0]
                    error = float(np.mean((vec - recon) ** 2))

                    if error < self._pca_threshold:
                        # Scale back to original image coords
                        x1 = x / s
                        y1 = y / s
                        x2 = (x + PATCH_SIZE) / s
                        y2 = (y + PATCH_SIZE) / s
                        detections.append((x1, y1, x2, y2, error))

        # Non-maximum suppression
        kept = self._nms(detections, overlap_threshold=0.3)
        faces = [
            FaceBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=max(0.0, 1.0 - err / self._pca_threshold))
            for x1, y1, x2, y2, err in kept
        ]
        return faces, None, None

    @staticmethod
    def _nms(
        detections: list[tuple[float, float, float, float, float]],
        overlap_threshold: float = 0.3,
    ) -> list[tuple[float, float, float, float, float]]:
        """Greedy non-maximum suppression by confidence (inverse error)."""
        if not detections:
            return []
        # Sort by confidence (ascending error = higher confidence)
        dets = sorted(detections, key=lambda d: d[4])
        kept = []
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [
                d
                for d in dets
                if _iou((d[0], d[1], d[2], d[3]), (best[0], best[1], best[2], best[3]))
                < overlap_threshold
            ]
        return kept

    # --- Public API ---

    ALGORITHMS = {
        "haar": "OpenCV Haar Cascade",
        "mediapipe_mesh": "MediaPipe Face Mesh",
        "hog": "face_recognition (HOG)",
        "cnn": "face_recognition (CNN)",
        "pca": "Eigenfaces (PCA)",
    }

    def list_algorithms(self) -> list[str]:
        return list(self.ALGORITHMS.keys())

    def detect(self, image: str, algorithm: str) -> DetectResponse:
        try:
            img_bgr = self._decode_image(image)
        except Exception as e:
            return DetectResponse(
                algorithm=algorithm,
                faces=[],
                time_ms=0,
                error=f"Image decode failed: {e}",
            )

        detector_map = {
            "haar": self.detect_haar,
            "mediapipe_mesh": self.detect_mediapipe_mesh,
            "hog": self.detect_hog,
            "cnn": self.detect_cnn,
            "pca": self.detect_pca,
        }

        detector = detector_map.get(algorithm)
        if detector is None:
            return DetectResponse(
                algorithm=algorithm,
                faces=[],
                time_ms=0,
                error=f"Unknown algorithm: {algorithm}",
            )

        elapsed, (faces, landmarks, error) = self._run_and_time(detector, img_bgr)
        return DetectResponse(
            algorithm=algorithm,
            faces=faces,
            landmarks=landmarks,
            time_ms=round(elapsed, 2),
            error=error,
        )
