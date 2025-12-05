@echo off
REM ============================================================
REM Emotion Recognition Web UI - Run Script
REM ============================================================

echo.
echo ============================================================
echo          Emotion Recognition Web UI
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run setup.bat first to set up the project.
    echo.
    pause
    exit /b 1
)

REM Check if required folders exist
if not exist "input_files" (
    echo [INFO] Creating input_files directory...
    mkdir input_files
)

if not exist "output_files" (
    echo [INFO] Creating output_files directory...
    mkdir output_files
)

REM Run Streamlit using venv's Python directly (avoids PATH issues)
echo [INFO] Starting web interface...
echo [INFO] Opening in browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.

venv\Scripts\python.exe -m streamlit run app.py

pause
