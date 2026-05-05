from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scripts.predict_server import DEFAULT_MODEL, ROOT, Predictor, decode_data_url


class PredictRequest(BaseModel):
    imageData: str


predictor = Predictor(DEFAULT_MODEL)

app = FastAPI(
    title="Thai Handwriting 56-60 API",
    description="Predict handwritten Thai numbers in the 56-60 label range.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": predictor.model_path.name,
        "device": str(predictor.device),
        "labels": predictor.labels,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        image_data = decode_data_url(request.imageData)
        return predictor.predict(image_data)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=Path(ROOT)),
    name="static",
)
