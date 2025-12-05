# 🎭 Video-Audio-Face Emotion Detection

A **multimodal deep learning system** that detects human emotions from video by analyzing:
- 🎵 **Audio tone** (speech patterns)
- 😀 **Facial expressions** (face detection + CNN)
- 📝 **Speech content** (transcription + sentiment analysis)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Face Emotion Detection** | Detects emotions from facial expressions using CNN (ResNet34) |
| **Audio Emotion Detection** | Analyzes voice tone using MFCC features + CNN (ResNet18) |
| **Video Emotion Detection** | Combines face + audio + text sentiment using LSTM |
| **Real-time Transcription** | Uses OpenAI Whisper for speech-to-text |
| **Grad-CAM Visualization** | Shows which facial regions the model focuses on |
| **Web Interface** | Beautiful Streamlit UI for easy usage |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/sagarsoni1234/video-audio-face-emotion-detection.git
cd video-audio-face-emotion-detection
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 4. Install FFmpeg
- **Windows**: `winget install FFmpeg.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### 5. Run the application
```bash
# Web UI (Recommended)
streamlit run app.py

# Or CLI version
python run.py
```

---

## 🧠 Models & Architecture

### Audio Model
- **Architecture**: MFCC Features → ResNet18 CNN → Dense Layers
- **Input**: Audio waveform
- **Accuracy**: ~80%

### Face Model
- **Architecture**: Face Detection → ResNet34 CNN → Dense Layers
- **Input**: Face images (64×64)
- **Accuracy**: ~74%
- **Features**: Grad-CAM visualization

### Combined Video Model
- **Architecture**: Audio CNN + Face CNN + LSTM + Text Sentiment
- **Input**: Video (4-second windows at 4 FPS)
- **Accuracy**: ~75%
- **Features**: Whisper transcription, sentiment analysis

---

## 📊 Emotion Classes

| Emotion | Description |
|---------|-------------|
| 😢 **Sad/Fear** | Sad or fearful expressions |
| 😐 **Neutral** | Neutral/calm expressions |
| 😊 **Happy** | Happy/joyful expressions |
| 😠 **Angry** | Angry/frustrated expressions |
| 😲 **Surprise/Disgust** | Surprised or disgusted expressions |

---

## 📁 Project Structure

```
video-audio-face-emotion-detection/
├── app.py                      # Streamlit web application
├── run.py                      # CLI interface
├── requirements.txt            # Python dependencies
│
├── models/                     # Trained model weights
│   ├── audio/                  # Audio emotion model
│   ├── face/                   # Face emotion model
│   └── audio_face_combined/    # Combined video model
│
├── source/                     # Source code
│   ├── audio_analysis_utils/   # Audio processing
│   ├── face_emotion_utils/     # Face processing
│   ├── audio_face_combined/    # Combined model
│   └── whisper/                # Speech transcription
│
├── input_files/                # Sample input files
├── output_files/               # Generated outputs
├── display_files/              # Demo images/videos
└── data/                       # Training data (if included)
```

---

## 📊 Datasets Used

### Audio Datasets
| Dataset | Description | Link |
|---------|-------------|------|
| CREMA-D | 7,442 emotional audio clips | [Link](https://github.com/CheyneyComputerScience/CREMA-D) |
| RAVDESS | 1,440 audio files | [Link](https://zenodo.org/record/1188976) |
| TESS | 2,800 audio files | [Link](https://tspace.library.utoronto.ca/handle/1807/24487) |
| SAVEE | 480 audio files | [Link](http://kahlan.eps.surrey.ac.uk/savee/) |

### Face Datasets
| Dataset | Description | Link |
|---------|-------------|------|
| FER2013 | 35,887 face images | [Link](https://www.kaggle.com/datasets/msambare/fer2013) |
| CK+ | 593 expression sequences | [Link](http://www.jeffcohn.net/Resources/) |
| RAF-DB | 29,672 real-world faces | [Link](http://www.whdeng.cn/RAF/model1.html) |

---

## 🔧 Technologies Used

| Technology | Purpose |
|------------|---------|
| **PyTorch** | Deep learning framework |
| **OpenAI Whisper** | Speech-to-text transcription |
| **MediaPipe** | Face detection & landmarks |
| **spaCy** | NLP text processing |
| **librosa** | Audio feature extraction |
| **Streamlit** | Web interface |
| **FFmpeg** | Video/audio processing |

---

## 📸 Sample Outputs

### Face Emotion Detection
| Input | Emotion | Grad-CAM |
|-------|---------|----------|
| ![](display_files/child%20smile.png) | Happy 😊 | ![](display_files/child_smile_grad_cam.jpg) |
| ![](display_files/angry.png) | Angry 😠 | ![](display_files/angry_grad_cam.jpg) |

---

## 📝 Usage Examples

### Analyze a video
```python
from source.audio_face_combined import predict
predict.predict_video("path/to/video.mp4")
```

### Analyze an image
```python
from source.face_emotion_utils import predict
predict.predict_image("path/to/image.jpg")
```

### Analyze audio
```python
from source.audio_analysis_utils import predict
predict.predict_audio("path/to/audio.wav")
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests
- Improve documentation

---

## 📄 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

- Original implementation inspired by emotion recognition research
- OpenAI Whisper for transcription
- MediaPipe for face detection
- PyTorch community

---

**Made with ❤️ by [Sagar Soni](https://github.com/sagarsoni1234)**
