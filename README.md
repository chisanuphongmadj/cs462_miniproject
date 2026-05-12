# Thai Handwriting ML System: 56-60

Web application for collecting Thai handwritten number samples and predicting labels in the range `56-60`.
The current model is a PyTorch CNN with a digit-pair pipeline: it splits the handwritten number into left/right digit crops, predicts each digit, then combines valid labels (`56`, `57`, `58`, `59`, `60`).

## Features

- User page for drawing a Thai number on canvas and calling `/predict`.
- Dataset collection page with label selection, sample preview, and JSON export.
- Admin page for model file selection/status display.
- FastAPI backend with `/predict`, `/health`, and `/docs`.
- PyTorch training pipeline with preprocessing, augmentation, metrics, confusion matrix, and saved model artifacts.
- GUI helper for converting exported dataset JSON and launching training.

## Project Structure

```txt
CS462/
  index.html
  start_web_app.bat
  start_fastapi_app.bat
  open_training_app.bat
  install_api_dependencies.bat
  training_app.py
  requirements-api.txt
  dataset/
    thai_handwriting_56_60/
      56/
      57/
      58/
      59/
      60/
      manifest.csv
  models/
    thai_handwriting_56_60_torch/
      thai_handwriting_56_60.pt
      labels.json
      digit_labels.json
      metrics.json
      classification_report.txt
      training_summary.txt
      training_history.png
  scripts/
    extract_dataset.py
    image_preprocess.py
    train_model_torch.py
    fastapi_app.py
    predict_server.py
    summarize_training.py
```

## How to Run

Run:

```bat
start_web_app.bat
```

Open:

```txt
http://127.0.0.1:8000/
```

API docs:

```txt
http://127.0.0.1:8000/docs
```

Health check:

```txt
http://127.0.0.1:8000/health
```

Note: prediction uses the real backend model, so open the app through `start_web_app.bat`, not by opening `index.html` directly.

## Dataset

Current dataset:

```txt
56: 150 images
57: 150 images
58: 150 images
59: 150 images
60: 150 images
Total: 750 images
```

The dataset is stored as PNG files separated by class folder under `dataset/thai_handwriting_56_60/`.

## Training

Run from the project root:

```bat
.venv\Scripts\python.exe scripts\train_model_torch.py --data dataset\thai_handwriting_56_60 --out models\thai_handwriting_56_60_torch --epochs 60 --batch-size 32 --image-size 96 --seed 42
```

The training script:

- splits each two-digit sample into left/right digit crops,
- trains a digit classifier for `0, 5, 6, 7, 8, 9`,
- combines left/right probabilities into valid labels `56-60`,
- saves the model and reports under `models/thai_handwriting_56_60_torch/`.

## Current Model Result

Model type: `digit_pair`

Test split:

```txt
Train number samples: 525
Validation number samples: 110
Test number samples: 115
Train digit samples: 1050
Validation digit samples: 220
Test digit samples: 230
```

Metrics:

```txt
Number accuracy: 97.39%
Digit accuracy: 96.52%
Macro precision: 0.9746
Macro recall: 0.9739
Macro F1-score: 0.9737
Weighted precision: 0.9746
Weighted recall: 0.9739
Weighted F1-score: 0.9737
```

Number confusion matrix:

```txt
[[22  0  1  0  0]
 [ 1 21  0  0  1]
 [ 0  0 23  0  0]
 [ 0  0  0 23  0]
 [ 0  0  0  0 23]]
```

Full reports are in:

```txt
models/thai_handwriting_56_60_torch/training_summary.txt
models/thai_handwriting_56_60_torch/classification_report.txt
models/thai_handwriting_56_60_torch/metrics.json
models/thai_handwriting_56_60_torch/training_history.png
```

## API

### GET `/health`

Example:

```json
{
  "ok": true,
  "model": "thai_handwriting_56_60.pt",
  "modelType": "digit_pair",
  "device": "cuda",
  "labels": ["56", "57", "58", "59", "60"],
  "digitLabels": ["0", "5", "6", "7", "8", "9"]
}
```

### POST `/predict`

Request:

```json
{
  "imageData": "data:image/png;base64,..."
}
```

Response:

```json
{
  "prediction": "60",
  "thaiPrediction": "๖๐",
  "confidence": 0.99,
  "modelType": "digit_pair",
  "digitConfidences": {
    "left": [{"label": "6", "confidence": 0.99}],
    "right": [{"label": "0", "confidence": 0.99}]
  }
}
```

## Submission Notes

The repository includes source code, training code, dataset, model artifacts, and requirements. Do not include `.venv/` when submitting as a zip because it is large and machine-specific.
