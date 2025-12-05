@echo off
REM ============================================================
REM Video-Audio-Face Emotion Detection - Setup Script
REM ============================================================
REM This script sets up the project for Windows
REM ============================================================

echo.
echo ======================================================================
echo               Video-Audio-Face-Emotion-Detection Setup
echo ======================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.10 from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if FFmpeg is available
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] FFmpeg not found
    echo.
    echo FFmpeg is required for video processing.
    echo Install using: winget install FFmpeg.FFmpeg
    echo Or see "ffmpeg-install guide.txt" for other methods.
    echo.
    set /p CONTINUE="Continue without FFmpeg? (y/n): "
    if /i not "%CONTINUE%"=="y" (
        echo Setup cancelled.
        pause
        exit /b 1
    )
) else (
    echo [OK] FFmpeg found
)
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [STEP] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [STEP] Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [STEP] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [OK] pip upgraded
echo.

REM Install PyTorch
echo [STEP] Installing PyTorch (CPU version)...
pip install torch torchvision torchaudio >nul 2>&1
if errorlevel 1 (
    echo [WARNING] PyTorch installation may have issues
) else (
    echo [OK] PyTorch installed
)
echo.

REM Install requirements
echo [STEP] Installing Python dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed
) else (
    echo [OK] Dependencies installed
)
echo.

REM Install spaCy model
echo [STEP] Downloading spaCy language model...
python -m spacy download en_core_web_lg >nul 2>&1
if errorlevel 1 (
    echo [WARNING] spaCy model download failed
    echo You can install it manually later:
    echo   pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl
) else (
    echo [OK] spaCy model downloaded
)
echo.

REM Run setup.py
echo [STEP] Creating project folders...
python setup.py >nul 2>&1
echo [OK] Project folders ready
echo.

echo ======================================================================
echo                         SETUP COMPLETE!
echo ======================================================================
echo.
echo To run the application:
echo.
echo   1. Activate virtual environment:
echo      venv\Scripts\activate
echo.
echo   2. Run Web UI:
echo      python -m streamlit run app.py
echo.
echo   3. Or run CLI:
echo      python run.py
echo.
echo ======================================================================
echo.

pause
