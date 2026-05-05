import argparse
import base64
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "thai_handwriting_56_60_torch" / "thai_handwriting_56_60.pt"
THAI_LABELS = {
    "56": "๕๖",
    "57": "๕๗",
    "58": "๕๘",
    "59": "๕๙",
    "60": "๖๐",
}


class SmallCnn(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(96, class_count),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class Predictor:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.labels = [str(label) for label in checkpoint["labels"]]
        self.image_size = int(checkpoint.get("image_size", 96))
        self.model = SmallCnn(len(self.labels)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, image_data):
        tensor = self._preprocess(image_data).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        ranked = sorted(
            [
                {
                    "label": label,
                    "thaiLabel": THAI_LABELS.get(label, label),
                    "confidence": float(probabilities[index]),
                }
                for index, label in enumerate(self.labels)
            ],
            key=lambda item: item["confidence"],
            reverse=True,
        )
        winner = ranked[0]
        return {
            "prediction": winner["label"],
            "thaiPrediction": winner["thaiLabel"],
            "confidence": winner["confidence"],
            "confidences": ranked,
            "model": self.model_path.name,
            "device": str(self.device),
        }

    def _preprocess(self, image_data):
        image = Image.open(BytesIO(image_data)).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32)
        array = (255.0 - array) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        return tensor


def decode_data_url(value):
    if not isinstance(value, str):
        raise ValueError("imageData must be a string")
    if "," in value:
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def make_handler(predictor):
    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            self._send_empty(204)

        def do_GET(self):
            if self.path == "/health":
                self._send_json({"ok": True, "model": predictor.model_path.name, "device": str(predictor.device)})
                return

            path = self.path.split("?", 1)[0]
            if path == "/":
                path = "/index.html"
            file_path = (ROOT / unquote(path.lstrip("/"))).resolve()

            if not str(file_path).startswith(str(ROOT)) or not file_path.exists() or file_path.is_dir():
                self._send_json({"error": "not found"}, status=404)
                return

            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            data = file_path.read_bytes()
            self.send_response(200)
            self._cors_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/predict":
                self._send_json({"error": "not found"}, status=404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw_body)
                image_data = decode_data_url(payload.get("imageData"))
                result = predictor.predict(image_data)
                self._send_json(result)
            except Exception as error:
                self._send_json({"error": str(error)}, status=400)

        def _send_empty(self, status):
            self.send_response(status)
            self._cors_headers()
            self.end_headers()

        def _send_json(self, payload, status=200):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def log_message(self, format, *args):
            print(f"{self.address_string()} - {format % args}")

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve Thai handwriting web app and /predict API.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    predictor = Predictor(args.model)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(predictor))
    print(f"Model: {args.model}")
    print(f"Device: {predictor.device}")
    print(f"Open: http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
