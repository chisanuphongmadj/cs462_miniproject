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

try:
    from scripts.image_preprocess import crop_and_center_image, split_digit_images
except ModuleNotFoundError:
    from image_preprocess import crop_and_center_image, split_digit_images


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "thai_handwriting_56_60_torch" / "thai_handwriting_56_60.pt"
MIN_CONFIDENCE = 0.70
MIN_CONFIDENCE_MARGIN = 0.25
MIN_DIGIT_CONFIDENCE = 0.45
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
        self.model_type = checkpoint.get("model_type", "whole_label")
        self.digit_labels = [str(label) for label in checkpoint.get("digit_labels", [])]
        self.digit_to_index = {label: index for index, label in enumerate(self.digit_labels)}
        class_count = len(self.digit_labels) if self.model_type == "digit_pair" else len(self.labels)
        self.model = SmallCnn(class_count).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, image_data):
        if self.model_type == "digit_pair":
            return self._predict_digit_pair(image_data)

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
        runner_up = ranked[1] if len(ranked) > 1 else None
        margin = winner["confidence"] - runner_up["confidence"] if runner_up else winner["confidence"]
        is_uncertain = winner["confidence"] < MIN_CONFIDENCE or margin < MIN_CONFIDENCE_MARGIN
        return {
            "prediction": winner["label"],
            "thaiPrediction": winner["thaiLabel"],
            "confidence": winner["confidence"],
            "confidenceMargin": margin,
            "isUncertain": is_uncertain,
            "confidences": ranked,
            "model": self.model_path.name,
            "modelType": self.model_type,
            "device": str(self.device),
        }

    def _predict_digit_pair(self, image_data):
        left_tensor, right_tensor = self._preprocess_digit_pair(image_data)
        batch = torch.cat([left_tensor, right_tensor], dim=0).to(self.device)
        with torch.no_grad():
            digit_probabilities = torch.softmax(self.model(batch), dim=1).cpu().numpy()

        left_probs = digit_probabilities[0]
        right_probs = digit_probabilities[1]
        label_probs = self._valid_label_probabilities(left_probs, right_probs)
        ranked = sorted(
            [
                {
                    "label": label,
                    "thaiLabel": THAI_LABELS.get(label, label),
                    "confidence": float(label_probs[index]),
                }
                for index, label in enumerate(self.labels)
            ],
            key=lambda item: item["confidence"],
            reverse=True,
        )
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        margin = winner["confidence"] - runner_up["confidence"] if runner_up else winner["confidence"]
        selected_left = left_probs[self.digit_to_index[winner["label"][0]]]
        selected_right = right_probs[self.digit_to_index[winner["label"][1]]]
        digit_confidence = float(min(selected_left, selected_right))
        is_uncertain = (
            winner["confidence"] < MIN_CONFIDENCE
            or margin < MIN_CONFIDENCE_MARGIN
            or digit_confidence < MIN_DIGIT_CONFIDENCE
        )
        return {
            "prediction": winner["label"],
            "thaiPrediction": winner["thaiLabel"],
            "confidence": winner["confidence"],
            "confidenceMargin": margin,
            "digitConfidence": digit_confidence,
            "isUncertain": is_uncertain,
            "confidences": ranked,
            "digitConfidences": {
                "left": self._rank_digit_probabilities(left_probs),
                "right": self._rank_digit_probabilities(right_probs),
            },
            "model": self.model_path.name,
            "modelType": self.model_type,
            "device": str(self.device),
        }

    def _valid_label_probabilities(self, left_probs, right_probs):
        scores = []
        for label in self.labels:
            left_index = self.digit_to_index[label[0]]
            right_index = self.digit_to_index[label[1]]
            scores.append(float(left_probs[left_index] * right_probs[right_index]))

        scores = np.asarray(scores, dtype=np.float32)
        total = float(scores.sum())
        if total <= 0:
            return np.full(len(self.labels), 1.0 / len(self.labels), dtype=np.float32)
        return scores / total

    def _rank_digit_probabilities(self, probabilities):
        return sorted(
            [
                {"label": label, "confidence": float(probabilities[index])}
                for index, label in enumerate(self.digit_labels)
            ],
            key=lambda item: item["confidence"],
            reverse=True,
        )

    def _preprocess(self, image_data):
        image = Image.open(BytesIO(image_data)).convert("L")
        image = crop_and_center_image(image, self.image_size)
        array = np.asarray(image, dtype=np.float32)
        array = (255.0 - array) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        return tensor

    def _preprocess_digit_pair(self, image_data):
        image = Image.open(BytesIO(image_data)).convert("L")
        left_digit, right_digit = split_digit_images(image, self.image_size)
        return self._image_to_tensor(left_digit), self._image_to_tensor(right_digit)

    def _image_to_tensor(self, image):
        array = np.asarray(image, dtype=np.float32)
        array = (255.0 - array) / 255.0
        return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


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
                self._send_json(
                    {
                        "ok": True,
                        "model": predictor.model_path.name,
                        "modelType": predictor.model_type,
                        "device": str(predictor.device),
                        "labels": predictor.labels,
                        "digitLabels": predictor.digit_labels,
                    }
                )
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
