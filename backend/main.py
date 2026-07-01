from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .detectors import DetectorManager
from .schemas import DetectRequest, DetectResponse, HealthResponse

detector = DetectorManager()
app = FastAPI(title="Face Recognition Playground")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def index():
    path = FRONTEND / "index.html"
    if not path.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(path.read_text())


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        algorithms=detector.list_algorithms(),
    )


@app.post("/api/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    return detector.detect(req.image, req.algorithm)


# Serve static frontend assets (style.css, app.js) at /static/*
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="frontend")
