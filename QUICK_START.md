# 🚀 Quick Start Guide

Get up and running in 3 simple steps!

## Step 1: Run Setup Script

### macOS / Linux:
```bash
./setup.sh
```

### Windows:
```batch
setup.bat
```

### Or use Python directly (all platforms):
```bash
python3 setup_project.py
```

The setup script will:
- ✅ Check all prerequisites
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Set up project structure
- ✅ Download required models

## Step 2: Activate Virtual Environment

### macOS / Linux:
```bash
source venv/bin/activate
```

### Windows:
```batch
venv\Scripts\activate
```

## Step 3: Run the Application

### Option A: Web UI (Recommended)
```bash
# macOS/Linux:
./run_ui.sh

# Windows:
run_ui.bat

# Or directly:
python -m streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

### Option B: Command Line Interface
```bash
python run.py
```

## That's It! 🎉

You're ready to use the Emotion Recognition System!

## What You Can Do

1. **Upload audio files** → Get emotion predictions from speech
2. **Upload images** → Detect emotions from facial expressions
3. **Upload videos** → Combined analysis (audio + face + text sentiment)

## Need Help?

- See `SETUP_INSTRUCTIONS.md` for detailed setup guide
- See `README.md` for project documentation
- See `UI_README.md` for web UI documentation

## Troubleshooting

**Problem**: Setup script fails
- **Solution**: Check `SETUP_INSTRUCTIONS.md` for manual setup steps

**Problem**: "Command not found"
- **Solution**: Make sure you've installed FFmpeg and Git LFS (see SETUP_INSTRUCTIONS.md)

**Problem**: Models not working
- **Solution**: Run `git lfs pull` to download model files

