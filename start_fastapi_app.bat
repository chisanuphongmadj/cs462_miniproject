@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m uvicorn scripts.fastapi_app:app --host 127.0.0.1 --port 8000
) else (
  python -m uvicorn scripts.fastapi_app:app --host 127.0.0.1 --port 8000
)
