from pydantic import BaseModel


class DetectRequest(BaseModel):
    image: str  # base64-encoded JPEG
    algorithm: str  # "haar" | "mediapipe_face" | "mediapipe_mesh" | "hog" | "cnn"


class FaceBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None = None


class DetectResponse(BaseModel):
    algorithm: str
    faces: list[FaceBox]
    landmarks: list[list[float]] | None = None
    time_ms: float
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    algorithms: list[str]
