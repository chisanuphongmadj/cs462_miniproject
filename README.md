# Thai Handwriting ML System: 56-60

โปรเจกต์นี้เป็น Web Application สำหรับทำนายลายมือเลขไทยในช่วง `56-60` โดยใช้โมเดล PyTorch CNN และเชื่อมต่อผ่าน FastAPI backend

## Features

- เก็บ dataset จาก canvas บนหน้าเว็บ
- แปลงไฟล์ dataset JSON เป็นรูปภาพแยกตาม class
- Train โมเดลด้วย PyTorch
- มีโปรแกรม GUI สำหรับ convert dataset และ train model
- Predict ลายมือจากหน้าเว็บด้วยโมเดลจริง
- มี FastAPI backend พร้อม endpoint `/predict`, `/health`, และ API docs `/docs`
- สรุปผลหลัง train ด้วย accuracy, precision, recall, f1-score และ confusion matrix

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
      metrics.json
      classification_report.txt
      training_summary.txt
      training_history.png
  scripts/
    extract_dataset.py
    train_model_torch.py
    fastapi_app.py
    predict_server.py
    summarize_training.py
```

## How to Open the Web Application

ใช้ไฟล์นี้เพื่อเปิดเว็บพร้อม FastAPI backend:

```txt
start_web_app.bat
```

หลังจากเปิดแล้ว เข้าเว็บที่:

```txt
http://127.0.0.1:8000/
```

หน้า FastAPI docs:

```txt
http://127.0.0.1:8000/docs
```

Health check:

```txt
http://127.0.0.1:8000/health
```

หมายเหตุ: ถ้าต้องการ predict ด้วยโมเดลจริง ห้ามเปิด `index.html` ด้วย `file:///...` โดยตรง ให้เปิดผ่าน `start_web_app.bat` เท่านั้น

## How to Convert Dataset JSON

เปิดโปรแกรม GUI:

```txt
open_training_app.bat
```

ขั้นตอน:

1. ไปที่แท็บ `Convert Dataset`
2. เลือกไฟล์ `thai_handwriting_56_60_dataset.json`
3. เลือก output folder เช่น `dataset/thai_handwriting_56_60`
4. กด `Convert JSON`
5. กด `Use Output for Training`

ผลลัพธ์จะเป็นรูปภาพแยกตาม class:

```txt
dataset/thai_handwriting_56_60/
  56/
  57/
  58/
  59/
  60/
```

## How to Train the Model

เปิดโปรแกรม GUI:

```txt
open_training_app.bat
```

ขั้นตอน:

1. ไปที่แท็บ `Train Model`
2. ตรวจสอบ `Dataset folder`
3. ตรวจสอบ `Output folder`
4. ตั้งค่า epochs เช่น `50`
5. กด `Start Training`
6. หลัง train เสร็จ ดูผลในช่อง `Metrics`

ผลลัพธ์โมเดลจะถูกเก็บที่:

```txt
models/thai_handwriting_56_60_torch/
```

ไฟล์สำคัญ:

```txt
thai_handwriting_56_60.pt
metrics.json
classification_report.txt
training_summary.txt
training_history.png
```

## Current Model Result

Dataset ที่ใช้ train มีทั้งหมด `250` รูป:

```txt
56: 50 images
57: 50 images
58: 50 images
59: 50 images
60: 50 images
```

Split:

```txt
Train samples: 175
Validation samples: 35
Test samples: 40
```

ผลโมเดลล่าสุด:

```txt
Accuracy: 92.50%
Macro precision: 0.9350
Macro recall: 0.9250
Macro F1-score: 0.9242
Weighted precision: 0.9350
Weighted recall: 0.9250
Weighted F1-score: 0.9242
```

Confusion matrix:

```txt
[[8 0 0 0 0]
 [0 8 0 0 0]
 [0 1 7 0 0]
 [0 1 1 6 0]
 [0 0 0 0 8]]
```

Classification report:

```txt
              precision    recall  f1-score   support

          56     1.0000    1.0000    1.0000         8
          57     0.8000    1.0000    0.8889         8
          58     0.8750    0.8750    0.8750         8
          59     1.0000    0.7500    0.8571         8
          60     1.0000    1.0000    1.0000         8

    accuracy                         0.9250        40
   macro avg     0.9350    0.9250    0.9242        40
weighted avg     0.9350    0.9250    0.9242        40
```

## API Endpoints

### GET `/health`

ตรวจสอบว่า backend โหลดโมเดลสำเร็จหรือไม่

ตัวอย่าง response:

```json
{
  "ok": true,
  "model": "thai_handwriting_56_60.pt",
  "device": "cuda",
  "labels": ["56", "57", "58", "59", "60"]
}
```

### POST `/predict`

รับภาพจาก canvas เป็น base64 แล้วส่งผลทำนายกลับ

ตัวอย่าง response:

```json
{
  "prediction": "60",
  "thaiPrediction": "๖๐",
  "confidence": 0.8476,
  "model": "thai_handwriting_56_60.pt",
  "device": "cuda"
}
```

## Installation Notes

ถ้าเปิด FastAPI ไม่ได้ ให้ติดตั้ง dependency ด้วย:

```txt
install_api_dependencies.bat
```

หรือใช้คำสั่ง:

```bat
.venv\Scripts\python.exe -m pip install -r requirements-api.txt
```

## Submission Notes

ก่อน zip ส่งงาน ไม่ควรใส่โฟลเดอร์ `.venv` เพราะมีขนาดใหญ่ ให้ส่ง source code, dataset, model, และไฟล์ requirements แทน

ควรส่งไฟล์หลักเหล่านี้:

```txt
index.html
README.md
start_web_app.bat
open_training_app.bat
training_app.py
requirements-api.txt
scripts/
dataset/
models/
```
