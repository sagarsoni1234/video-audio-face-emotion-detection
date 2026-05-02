@echo off
REM Windows: double-click or run from project folder (similar idea to `make install` + `make run`).
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install from https://www.python.org/downloads/ then retry.
  exit /b 1
)

if not exist "venv\" (
  echo Creating venv...
  python -m venv venv
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl"

echo.
echo Starting Streamlit. Open http://localhost:8501
python -m streamlit run app.py
