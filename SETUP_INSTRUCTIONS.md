# 🚀 Complete Setup Instructions

This guide will help you set up the Video-Audio-Face Emotion Detection project on any computer.

---

## ⚡ Quick Start (Recommended)

### Windows:
```batch
setup.bat
```

### macOS/Linux:
```bash
chmod +x setup.sh
./setup.sh
```

### Or use Python directly:
```bash
python setup_project.py
```

---

## 📋 Prerequisites

Before running the setup, make sure you have:

### 1. Python 3.10 (Required)
- Download from: https://www.python.org/downloads/
- ✅ Check "Add Python to PATH" during installation

Verify installation:
```bash
python --version
# Should show: Python 3.10.x
```

### 2. Git (Required)
- Download from: https://git-scm.com/downloads

Verify installation:
```bash
git --version
```

### 3. FFmpeg (Required for video processing)

**Windows:**
```powershell
winget install FFmpeg.FFmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

See `ffmpeg-install guide.txt` for detailed instructions.

---

## 📥 Step-by-Step Manual Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/sagarsoni1234/video-audio-face-emotion-detection.git
cd video-audio-face-emotion-detection
```

### Step 2: Create Virtual Environment

**Windows:**
```batch
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install PyTorch

**CPU Version (Default):**
```bash
pip install torch torchvision torchaudio
```

**GPU Version (NVIDIA CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Download spaCy Language Model

```bash
python -m spacy download en_core_web_lg
```

If the above fails, try:
```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl
```

### Step 7: Setup Project Folders

```bash
python setup.py
```

---

## ✅ Verify Installation

### Test 1: Check imports
```bash
python -c "import torch; import cv2; import mediapipe; print('All imports successful!')"
```

### Test 2: Run the Web UI
```bash
# Make sure venv is activated, then:
python -m streamlit run app.py
```

Open browser to: http://localhost:8501

### Test 3: Run CLI
```bash
python run.py
```

---

## 🎮 How to Run

### Option 1: Web UI (Recommended)

**Windows:**
```batch
venv\Scripts\python.exe -m streamlit run app.py
```

**macOS/Linux:**
```bash
./venv/bin/python -m streamlit run app.py
```

Or use the run scripts:
- Windows: `run_ui.bat`
- macOS/Linux: `./run_ui.sh`

### Option 2: Command Line Interface

```bash
python run.py
```

---

## 🔧 Troubleshooting

### Problem: NumPy version conflict
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```
**Solution:**
```bash
pip install "numpy==1.23.4" "opencv-python==4.8.1.78" --force-reinstall
```

### Problem: spaCy model not found
```
Can't find model 'en_core_web_lg'
```
**Solution:**
```bash
python -m spacy download en_core_web_lg
```

### Problem: FFmpeg not found
```
FileNotFoundError: ffmpeg
```
**Solution:** Install FFmpeg (see Prerequisites section)

### Problem: CUDA out of memory
**Solution:** Use CPU version or reduce batch size in config files

### Problem: Module not found
**Solution:** Make sure virtual environment is activated:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

### Problem: Permission denied (Linux/Mac)
**Solution:**
```bash
chmod +x setup.sh run_ui.sh
```

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB+ |
| Storage | 3 GB | 5 GB+ |
| Python | 3.9 | 3.10 |
| GPU | Not required | NVIDIA with CUDA |

---

## 📁 Project Structure After Setup

```
video-audio-face-emotion-detection/
├── venv/                    # Virtual environment (created)
├── models/                  # Pre-trained models
│   ├── audio/              # Audio emotion model
│   ├── face/               # Face emotion model
│   └── audio_face_combined/ # Combined model
├── source/                  # Source code
├── input_files/            # Put your test files here
├── output_files/           # Results saved here
├── app.py                  # Streamlit Web UI
├── run.py                  # CLI interface
├── requirements.txt        # Python dependencies
└── setup.bat / setup.sh    # Setup scripts
```

---

## 🎯 Quick Test

After setup, test with sample files:

1. **Face Emotion:** Upload `input_files/angry.png`
2. **Audio Emotion:** Upload `input_files/audio_happy.mp4`
3. **Video Emotion:** Upload `input_files/angry_alex.mp4`

---

## 📞 Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Make sure all prerequisites are installed
3. Try running setup again
4. Check console for error messages

---

**Happy Emotion Detecting! 🎭**
