@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m pip install -r requirements-api.txt
) else (
  python -m pip install -r requirements-api.txt
)
