# 📚 Video-Audio-Face Emotion Recognition - Complete Technical Guide

## 🎯 Overview

This is a **multimodal emotion detection system** that predicts a speaker's emotion using:
1. **Audio tone analysis** (CNN on MFCC features)
2. **Facial expression analysis** (CNN + optional face landmarks)
3. **Combined video analysis** (LSTM for temporal sequences + text sentiment from transcription)

---

## 🧠 Models Architecture

### 1️⃣ Audio Emotion Model

| Aspect | Details |
|--------|---------|
| **Architecture** | CNN (ResNet18) + Dense Layers |
| **Input** | MFCC (Mel-Frequency Cepstral Coefficients) features |
| **Input Shape** | (3, 126, 13) - 3 channels, 126 time steps, 13 MFCCs |
| **Best Accuracy** | **79.77%** |

#### How it works:
1. Audio is loaded and preprocessed
2. **MFCC features** are extracted using:
   - `N_FFT = 2048` (FFT window size)
   - `HOP_LENGTH = 512` (hop between windows)
   - `NUM_MFCC = 13` (number of coefficients)
3. MFCCs are converted to a 3-channel "image" format
4. Passed through **ResNet18** CNN backbone
5. Dense layers (512 units, 3 layers) with dropout (0.3) and batch normalization
6. Output: 5-class softmax prediction

#### Best Hyperparameters:
```json
{
  "conv_model": "resnet18",
  "dense_units": 512,
  "num_layers": 3,
  "dropout_rate": 0.3,
  "batch_size": 256,
  "layers_batch_norm": true,
  "N_FFT": 2048,
  "HOP_LENGTH": 512,
  "NUM_MFCC": 13
}
```

#### CNN Options Available:
- ResNet (18, 34, 50, 101, 152)
- ResNeXt (50_32x4d, 101_32x8d)
- Wide ResNet (50_2, 101_2)
- GoogLeNet
- MobileNet v2
- DenseNet121
- AlexNet
- VGG16
- SqueezeNet
- ShuffleNet
- MNASNet

---

### 2️⃣ Face Emotion Model

| Aspect | Details |
|--------|---------|
| **Architecture** | CNN (ResNet34) + Optional MediaPipe Landmarks + Dense Layers |
| **Input** | Face images (64×64×3) + Optional 1404 landmark coordinates |
| **Best Accuracy** | **73.85%** |

#### How it works:
1. Face is detected and cropped from image
2. Image resized to 64×64 pixels
3. (Optional) **MediaPipe Face Mesh** extracts 468 3D landmarks (1404 values)
4. Image passed through **ResNet34** (pretrained on ImageNet)
5. CNN output concatenated with landmarks (if used)
6. Dense layers process combined features
7. Output: 5-class softmax prediction

#### Best Hyperparameters:
```json
{
  "conv_model": "resnet34",
  "dense_units": 32,
  "num_layers": 3,
  "use_landmarks": false,
  "normalise": true,
  "batch_size": 256,
  "dropout_rate": 0.0,
  "layers_batch_norm": true
}
```

#### Features:
- **Grad-CAM visualization** - shows which facial regions the model focuses on
- **Data augmentation** using Albumentations:
  - Horizontal flip (50%)
  - Rotation (up to 180°)
  - Gaussian noise
  - CLAHE / Random brightness / Gamma
  - Sharpen / Blur / Motion blur
  - Contrast / Hue-Saturation

---

### 3️⃣ Combined Video Model (Multimodal)

| Aspect | Details |
|--------|---------|
| **Architecture** | Audio CNN + Face CNN + LSTM + Text Sentiment + Dense Layers |
| **Input** | Video (4 sec window, 4 FPS = 16 frames) |
| **Best Accuracy** | **75.24%** |

#### How it works:
1. Video split into **4-second windows**
2. **Audio stream:**
   - Audio extracted from video using FFmpeg
   - MFCC features computed
   - Passed through pretrained Audio Model (without softmax)
3. **Visual stream:**
   - Frames extracted at 4 FPS (16 frames per window)
   - Each frame → Face Model (without softmax)
   - Sequence of 16 frame embeddings → **LSTM layer(s)**
4. **Text stream:**
   - Audio transcribed using **OpenAI Whisper**
   - Transcript analyzed for sentiment using **spaCy + TextBlob**
   - Outputs emotion probabilities from text (9 features)
5. **Fusion:**
   - Audio features + LSTM output + Text sentiment → concatenated
   - Dense layers → Final 5-class prediction

#### Best Hyperparameters:
```json
{
  "dense_units": 128,
  "num_layers": 2,
  "sequence_model_dense_units": 128,
  "sequence_model_layers": 1,
  "batch_size": 128,
  "prev_layers_trainable": true,
  "dropout_rate": 0.0,
  "layers_batch_norm": true,
  "l1_l2_reg": "1e-06"
}
```

---

## 📊 Datasets Used

### For Audio Model

| Dataset | Description | Size | Link |
|---------|-------------|------|------|
| **CREMA-D** | Crowd-sourced Emotional Multimodal Actors Dataset. 91 actors (48 male, 43 female) between ages 20-74 from diverse ethnic backgrounds. | 7,442 clips | [https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) |
| **RAVDESS** | Ryerson Audio-Visual Database of Emotional Speech and Song. 24 professional actors (12 male, 12 female) with North American accent. | 1,440 audio files | [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976) |
| **TESS** | Toronto Emotional Speech Set. Two actresses (26 and 64 years old) speaking 200 target words in 7 emotions. | 2,800 audio files | [https://tspace.library.utoronto.ca/handle/1807/24487](https://tspace.library.utoronto.ca/handle/1807/24487) |
| **SAVEE** | Surrey Audio-Visual Expressed Emotion. 4 male actors from UK. | 480 audio files | [http://kahlan.eps.surrey.ac.uk/savee/](http://kahlan.eps.surrey.ac.uk/savee/) |

---

### For Face Model

| Dataset | Description | Size | Link |
|---------|-------------|------|------|
| **FER2013** | Facial Expression Recognition 2013. Collected via Google image search. 48×48 grayscale images. | 35,887 images | [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013) |
| **CK+** | Extended Cohn-Kanade Dataset. 123 subjects with posed expressions. Validated and labeled for discrete emotions. | 593 sequences | [http://www.jeffcohn.net/Resources/](http://www.jeffcohn.net/Resources/) |
| **RAF-DB** | Real-world Affective Faces Database. Collected from the Internet with real-world conditions (pose, illumination, occlusion). | 29,672 images | [http://www.whdeng.cn/RAF/model1.html](http://www.whdeng.cn/RAF/model1.html) |

---

### For Combined Video Model

| Dataset | Description | Size | Link |
|---------|-------------|------|------|
| **CREMA-D** | Video clips with synchronized audio. Same 91 actors as audio version. | 7,442 clips | [https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) |
| **RAVDESS** | Video clips with emotional speech and song performances. | 2,452 videos | [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976) |
| **SAVEE** | Audio-visual clips with British English speakers. | 480 clips | [http://kahlan.eps.surrey.ac.uk/savee/](http://kahlan.eps.surrey.ac.uk/savee/) |
| **OMG** | OMG-Emotion Dataset. YouTube videos with continuous emotion annotations. (Optional, disabled by default) | 567 videos | [https://github.com/knowledgetechnologyuhh/OMGEmotionChallenge](https://github.com/knowledgetechnologyuhh/OMGEmotionChallenge) |

---

## 🏷️ Emotion Classes

### Full 7 Emotions (Original Labels)
| Index | Emotion |
|-------|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

### Simplified 5 Emotions (Used in Training)
| Index | Emotion | Combined From |
|-------|---------|---------------|
| 0 | Sad/Fear | Fear + Sad |
| 1 | Neutral | Neutral |
| 2 | Happy | Happy |
| 3 | Angry | Angry |
| 4 | Surprise/Disgust | Surprise + Disgust |

> **Note:** Emotions are combined due to limited data for some classes (e.g., Surprise) and their perceptual/acoustic similarity.

---

## 🔧 Key Technologies Used

| Technology | Purpose | Version/Notes |
|------------|---------|---------------|
| **PyTorch** | Deep learning framework | Primary framework for all models |
| **torchvision** | Pretrained CNN backbones | ResNet, DenseNet, etc. |
| **OpenAI Whisper** | Audio transcription | Converts speech to text for sentiment analysis |
| **spaCy** | NLP processing | Uses `en_core_web_lg` model for text analysis |
| **TextBlob** | Sentiment analysis | Extracts polarity and subjectivity from transcripts |
| **MediaPipe** | Face detection & landmarks | 468 3D facial landmarks extraction |
| **librosa** | Audio feature extraction | MFCC computation |
| **FFmpeg** | Video/audio processing | Audio extraction, format conversion |
| **Hyperopt** | Hyperparameter optimization | Random search for best hyperparameters |
| **Albumentations** | Image augmentation | Training data augmentation |
| **OpenCV** | Image processing | Face detection, image manipulation |
| **NumPy** | Numerical computing | Data preprocessing and storage |

---

## 📈 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Test Split** | 15% |
| **Video Window** | 4.0 seconds |
| **Frame Rate** | 4 FPS |
| **Minimum Video Length** | 1.5 seconds |
| **Face Size** | 64×64 pixels |
| **GPU Used** | NVIDIA RTX 4090 (recommended) |
| **RAM Required** | ~40GB (preprocessing), ~8GB (inference) |
| **Python Version** | 3.10 |

### Training Callbacks:
- **Model Checkpoint** - Saves best model based on validation accuracy
- **Reduce LR on Plateau** - Reduces learning rate when training loss plateaus
- **Early Stopping** - Stops training when validation accuracy stops improving

---

## 🔄 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        VIDEO INPUT                               │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │  AUDIO   │        │  FRAMES  │        │TRANSCRIPT│
    │ Extract  │        │ Extract  │        │ (Whisper)│
    │ (FFmpeg) │        │ (4 FPS)  │        │          │
    └──────────┘        └──────────┘        └──────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │  MFCC    │        │  Face    │        │Sentiment │
    │ Features │        │Detection │        │ Analysis │
    │(librosa) │        │(MediaPipe│        │(spaCy +  │
    │          │        │ /OpenCV) │        │ TextBlob)│
    └──────────┘        └──────────┘        └──────────┘
          │                   │                   │
          ▼                   ▼                   │
    ┌──────────┐        ┌──────────┐              │
    │  Audio   │        │  Face    │              │
    │   CNN    │        │   CNN    │              │
    │(ResNet18)│        │(ResNet34)│              │
    └──────────┘        └──────────┘              │
          │                   │                   │
          │             (16 frames)               │
          │                   │                   │
          │                   ▼                   │
          │             ┌──────────┐              │
          │             │   LSTM   │              │
          │             │(Sequence)│              │
          │             │ 128 units│              │
          │             └──────────┘              │
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │ CONCATENATE  │
                      │Audio + LSTM +│
                      │  Sentiment   │
                      └──────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │ Dense Layers │
                      │  (128 units) │
                      │  (2 layers)  │
                      └──────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │   SOFTMAX    │
                      │  (5 Classes) │
                      └──────────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │ Sad/Fear | Neutral |    │
                │ Happy | Angry |         │
                │ Surprise/Disgust        │
                └─────────────────────────┘
```

---

## 📁 Project Structure

```
Video-Audio-Face-Emotion-Recognition/
│
├── app.py                          # Streamlit web application
├── run.py                          # Main CLI runner
├── setup.py                        # Project setup script
│
├── source/
│   ├── config.py                   # Global configuration
│   │
│   ├── audio_analysis_utils/       # Audio model code
│   │   ├── model.py                # Audio CNN architecture
│   │   ├── predict.py              # Audio inference
│   │   ├── preprocess_data.py      # MFCC extraction
│   │   ├── audio_config.py         # Audio hyperparameters
│   │   └── get_data.py             # Dataset formatting
│   │
│   ├── face_emotion_utils/         # Face model code
│   │   ├── model.py                # Face CNN architecture
│   │   ├── predict.py              # Face inference + Grad-CAM
│   │   ├── preprocess_main.py      # Face preprocessing
│   │   ├── face_config.py          # Face hyperparameters
│   │   └── get_data.py             # Dataset formatting
│   │
│   ├── audio_face_combined/        # Combined model code
│   │   ├── model.py                # Combined architecture (CNN+LSTM)
│   │   ├── predict.py              # Video inference
│   │   ├── preprocess_main.py      # Video preprocessing
│   │   └── combined_config.py      # Combined hyperparameters
│   │
│   ├── whisper/                    # OpenAI Whisper transcription
│   │
│   └── pytorch_utils/              # Training utilities
│       ├── training_utils.py       # Training loop
│       ├── callbacks.py            # Training callbacks
│       └── hyper_tuner.py          # Hyperparameter optimization
│
├── models/
│   ├── audio/
│   │   ├── audio_model.pth                    # Trained audio model
│   │   └── audio_best_hyperparameters.json    # Best hyperparameters
│   │
│   ├── face/
│   │   ├── face_model.pth                     # Trained face model
│   │   └── face_best_hyperparameters.json     # Best hyperparameters
│   │
│   └── audio_face_combined/
│       ├── audio_face_combined_model.pth      # Trained combined model
│       └── combined_best_hyperparameters.json # Best hyperparameters
│
├── data/
│   ├── preprocessed_audio_data/    # Preprocessed MFCC .npy files
│   ├── preprocessed_images_data/   # Preprocessed face .npy files
│   └── preprocessed_AV_data/       # Preprocessed video .npy files
│
├── input_files/                    # Input files for prediction
└── output_files/                   # Output predictions and visualizations
```

---

## 🚀 Usage

### Running Inference

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run the main program
python run.py

# Or run the Streamlit web app
streamlit run app.py
```

### Training Models

1. Prepare datasets in the correct format
2. Run preprocessing via `run.py` menu
3. Run training via `run.py` menu
4. Models saved to `models/` folder

---

## 📖 References

1. **MFCC**: Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences.
2. **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
3. **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
4. **Whisper**: Radford, A., et al. (2022). Robust speech recognition via large-scale weak supervision.
5. **MediaPipe**: Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines.



