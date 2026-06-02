import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Gym Form Coach")


class RepFrames(BaseModel):
    frames: list  # each frame = 33 landmarks as [x, y, z, visibility]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(rep: RepFrames):
    if not rep.frames or len(rep.frames) < 2:
        return {"label": "unknown", "prob_good": None}
    from .model import predict as run_predict  # lazy: keeps torch out of health/CI
    label, prob_good = run_predict(rep.frames)
    return {"label": label, "prob_good": round(prob_good, 3)}


STATIC_DIR = os.environ.get("STATIC_DIR", "/app/static")
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")