import base64
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .schemas import DetectResponse, FaceBox


try:
    import face_recognition
    import face_recognition_models  # noqa: F401
    _FACE_REC_AVAILABLE = True
except ImportError:
    _FACE_REC_AVAILABLE = False


MODELS_DIR = Path(__file__).resolve().parent / "models"


class DetectorManager:
    """Manages multiple on-device face detection algorithms."""

    def __init__(self):
        self._haar_cascade = None
        self._mp_face_mesh = None

    # --- Helpers ---

    def _decode_image(self, image: str) -> np.ndarray:
        """Convert a base64 data-URL JPEG into an RGB numpy array."""
        if "," in image:
            image = image.split(",", 1)[1]
        raw = base64.b64decode(image)
        pil = Image.open(BytesIO(raw))
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return np.array(pil)

    def _run_and_time(self, fn, img: cv2.Mat) -> tuple[float, any]:
        start = time.perf_counter()
        result = fn(img)
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, result

    # --- Shared face_recognition helper ---

    def _detect_facerec(
        self, img_rgb: cv2.Mat, model: str
    ) -> tuple[list[FaceBox], list[list[float]] | None, str | None]:
        if not _FACE_REC_AVAILABLE:
            return [], None, "face_recognition_models not installed"
        locations = face_recognition.face_locations(img_rgb, model=model)
        faces = [
            FaceBox(x1=float(l), y1=float(t), x2=float(r), y2=float(b), confidence=None)
            for t, r, b, l in locations
        ]
        return faces, None, None

    # --- Individual detectors ---

    def detect_haar(self, img_rgb: cv2.Mat) -> tuple[list[FaceBox], list[list[float]] | None, str | None]:
        if self._haar_cascade is None:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(path)
            if self._haar_cascade.empty():
                return [], None, "Failed to load Haar cascade model"

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        rects = self._haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        faces = [
            FaceBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h), confidence=None)
            for x, y, w, h in rects
        ]
        return faces, None, None

    def detect_mediapipe_mesh(self, img_rgb: cv2.Mat) -> tuple[list[FaceBox], list[list[float]] | None, str | None]:
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

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = self._mp_face_mesh.detect(mp_image)

        if not results.face_landmarks:
            return [], None, None

        h, w, _ = img_rgb.shape
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

    def detect_hog(self, img_rgb: cv2.Mat) -> tuple[list[FaceBox], list[list[float]] | None, str | None]:
        return self._detect_facerec(img_rgb, "hog")

    def detect_cnn(self, img_rgb: cv2.Mat) -> tuple[list[FaceBox], list[list[float]] | None, str | None]:
        return self._detect_facerec(img_rgb, "cnn")

    # --- Public API ---

    ALGORITHMS = {
        "haar": "OpenCV Haar Cascade",
        "mediapipe_mesh": "MediaPipe Face Mesh",
        "hog": "face_recognition (HOG)",
        "cnn": "face_recognition (CNN)",
    }

    def list_algorithms(self) -> list[str]:
        return list(self.ALGORITHMS.keys())

    def detect(self, image: str, algorithm: str) -> DetectResponse:
        try:
            img_rgb = self._decode_image(image)
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
        }

        detector = detector_map.get(algorithm)
        if detector is None:
            return DetectResponse(
                algorithm=algorithm,
                faces=[],
                time_ms=0,
                error=f"Unknown algorithm: {algorithm}",
            )

        elapsed, (faces, landmarks, error) = self._run_and_time(detector, img_rgb)
        return DetectResponse(
            algorithm=algorithm,
            faces=faces,
            landmarks=landmarks,
            time_ms=round(elapsed, 2),
            error=error,
        )
