@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
set "APP_URL=http://127.0.0.1:8000/"

if not exist "%PYTHON_EXE%" (
  echo Creating virtual environment in .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo.
    echo ERROR: Could not create .venv. Please install Python 3.10+ and try again.
    pause
    exit /b 1
  )
)

"%PYTHON_EXE%" -c "import fastapi, uvicorn, torch, numpy, PIL" >nul 2>nul
if errorlevel 1 (
  echo Installing required packages ...
  "%PYTHON_EXE%" -m pip install --upgrade pip
  if errorlevel 1 (
    echo.
    echo ERROR: Could not upgrade pip.
    pause
    exit /b 1
  )

  "%PYTHON_EXE%" -m pip install -r requirements-api.txt
  if errorlevel 1 (
    echo.
    echo ERROR: Could not install dependencies from requirements-api.txt.
    echo Please check your internet connection, then run this file again.
    pause
    exit /b 1
  )
)

if not exist "index.html" (
  echo ERROR: index.html was not found in this folder.
  pause
  exit /b 1
)

if not exist "models\thai_handwriting_56_60_torch\thai_handwriting_56_60.pt" (
  echo ERROR: Model file was not found:
  echo models\thai_handwriting_56_60_torch\thai_handwriting_56_60.pt
  pause
  exit /b 1
)

echo Starting Thai Handwriting ML web app ...
echo Open: %APP_URL%
if /I not "%SKIP_OPEN_BROWSER%"=="1" start "" "%APP_URL%"
"%PYTHON_EXE%" -m uvicorn scripts.fastapi_app:app --host 127.0.0.1 --port 8000

if errorlevel 1 (
  echo.
  echo ERROR: The web app stopped because the server could not start.
  pause
  exit /b 1
)
