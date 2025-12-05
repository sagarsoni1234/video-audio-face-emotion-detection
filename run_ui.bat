@echo off
REM Script to run the Streamlit UI for Emotion Recognition on Windows

echo.
echo ============================================================
echo   Emotion Recognition Web UI
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first to set up the project.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Streamlit not found, installing...
    pip install streamlit
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

REM Run the Streamlit app
echo.
echo [SUCCESS] Starting web interface...
echo [INFO] The app will open in your browser at http://localhost:8501
echo.
python -m streamlit run app.py

pause

