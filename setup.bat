@echo off
REM Setup script for Windows
REM This is a wrapper that calls the Python setup script

echo ============================================================
echo Video-Audio-Face-Emotion-Recognition Setup
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

REM Run the Python setup script
python setup_project.py

pause

