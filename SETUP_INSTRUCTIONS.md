# Complete Setup Instructions

This guide will help you set up the Video-Audio-Face-Emotion-Recognition project on any operating system.

## Quick Start

### For macOS and Linux:
```bash
./setup.sh
```

### For Windows:
```batch
setup.bat
```

### Or use Python directly (all platforms):
```bash
python3 setup_project.py
```

## What the Setup Script Does

The setup script automatically:

1. ✅ **Checks Python version** (requires 3.9+)
2. ✅ **Detects your operating system**
3. ✅ **Checks for system dependencies** (FFmpeg, Git LFS)
4. ✅ **Creates virtual environment** (`venv/`)
5. ✅ **Installs PyTorch** (CPU version by default)
6. ✅ **Installs all Python dependencies** from `requirements.txt`
7. ✅ **Downloads spaCy language model** (`en_core_web_lg`)
8. ✅ **Clones pytorch_utils submodule** (v1.0.3)
9. ✅ **Creates project folders** (input_files, output_files, etc.)
10. ✅ **Checks for model files** and downloads if needed

## Manual Setup (Alternative)

If you prefer to set up manually or the script fails:

### 1. Prerequisites

#### System Dependencies:

**FFmpeg** (required for video processing):
- **macOS**: `brew install ffmpeg`
- **Linux (Ubuntu/Debian)**: `sudo apt-get install ffmpeg`
- **Linux (RHEL/CentOS)**: `sudo yum install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `choco install ffmpeg`

**Git LFS** (required for downloading combined model):
- **macOS**: `brew install git-lfs`
- **Linux**: `sudo apt-get install git-lfs`
- **Windows**: Download from [git-lfs.github.com](https://git-lfs.github.com/)

#### Python:
- Python 3.9 or higher (3.10 recommended)
- pip (usually comes with Python)

### 2. Clone Repository

```bash
git clone https://github.com/rishiswethan/Video-Audio-Face-Emotion-Recognition.git
cd Video-Audio-Face-Emotion-Recognition
```

### 3. Clone Submodule

```bash
git clone https://github.com/rishiswethan/pytorch_utils.git source/pytorch_utils
cd source/pytorch_utils
git checkout v1.0.3
cd ../..
```

### 4. Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```batch
python -m venv venv
venv\Scripts\activate
```

### 5. Install PyTorch

**CPU Version (recommended for testing):**
```bash
pip install torch torchvision torchaudio
```

**GPU Version (CUDA 11.7):**
```bash
pip uninstall torch
pip cache purge
pip install torch==1.13.1+cu117 torchvision>=0.13.1+cu117 torchaudio>=0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir
```

### 6. Install Dependencies

```bash
pip install -r requirements.txt
```

### 7. Download spaCy Model

```bash
python -m spacy download en_core_web_lg
```

### 8. Create Project Folders

```bash
python setup.py
```

### 9. Download Model Files (if using Git LFS)

```bash
git lfs install
git lfs pull models/audio_face_combined/audio_face_combined_model.pth
```

## Verification

After setup, verify everything works:

### Test Audio Model:
```bash
python run.py
# Select: 1 (Audio) -> 4 (Predict)
# Enter filename: audio_happy.mp4
```

### Test Face Model:
```bash
python run.py
# Select: 2 (Face) -> 4 (Predict)
# Enter filename: angry.png
```

### Test Combined Model:
```bash
python run.py
# Select: 3 (Combined) -> 4 (Predict)
# Enter filename: angry_alex.mp4
```

### Run Web UI:
```bash
# macOS/Linux:
./run_ui.sh

# Windows:
run_ui.bat

# Or directly:
python -m streamlit run app.py
```

## Troubleshooting

### Issue: "Command not found: ffmpeg"
**Solution**: Install FFmpeg using your system package manager (see Prerequisites above)

### Issue: "Git LFS not found"
**Solution**: Install Git LFS (see Prerequisites above), then run:
```bash
git lfs install
git lfs pull
```

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Make sure virtual environment is activated, then:
```bash
pip install torch torchvision torchaudio
```

### Issue: "Model file is too small" or "invalid load key"
**Solution**: The model file is a Git LFS pointer. Download the actual file:
```bash
git lfs install
git lfs pull models/audio_face_combined/audio_face_combined_model.pth
```

### Issue: "spaCy model not found"
**Solution**: Download the model:
```bash
python -m spacy download en_core_web_lg
```

### Issue: Virtual environment not activating
**Solution**: 
- **macOS/Linux**: Make sure you use `source venv/bin/activate`
- **Windows**: Make sure you use `venv\Scripts\activate`

## System Requirements

- **RAM**: 8GB+ recommended (4GB minimum)
- **Storage**: 5GB+ free space
- **Python**: 3.9+ (3.10 recommended)
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

## GPU Support (Optional)

For GPU acceleration, you need:
- NVIDIA GPU with CUDA support
- CUDA 11.7 or compatible version
- Install PyTorch with CUDA (see step 5 above)

## Next Steps

After successful setup:

1. **Test the models** using `run.py`
2. **Try the web UI** using `run_ui.sh` or `run_ui.bat`
3. **Upload test files** to `input_files/` directory
4. **Check results** in `output_files/` directory

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Ensure all prerequisites are installed
4. Try running the setup script again

## Notes

- The setup script is idempotent - you can run it multiple times safely
- Virtual environment will be reused if it already exists
- Model files are large (~130MB for combined model) - download may take time
- First run may be slower as models load into memory

