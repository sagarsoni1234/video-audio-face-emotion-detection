# Emotion Recognition Web UI

A modern, user-friendly web interface for the Video-Audio-Face-Emotion-Recognition project built with Streamlit.

## Features

🎵 **Audio Emotion Detection**
- Upload audio files (WAV, MP3, MP4, M4A)
- Detect emotions from speech tone
- View confidence scores for all emotions
- Audio player for playback

😊 **Face Emotion Detection**
- Upload images with faces (PNG, JPG, JPEG)
- Real-time facial expression analysis
- Grad-CAM visualization showing attention areas
- Detailed emotion probability breakdown

🎬 **Video Emotion Detection (Combined)**
- Upload video files (MP4, AVI, MOV, MKV)
- Combines audio and facial expression analysis
- Processes video frames and audio tracks
- Comprehensive emotion prediction

## Installation

1. Install Streamlit:
```bash
pip install streamlit
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Running the UI

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

### Audio Emotion Detection
1. Go to the "🎵 Audio Emotion Detection" tab
2. Click "Choose an audio file" and select an audio file
3. Wait for processing (usually a few seconds)
4. View the predicted emotion and confidence scores
5. Listen to the audio using the built-in player

### Face Emotion Detection
1. Go to the "😊 Face Emotion Detection" tab
2. Upload an image containing a face
3. View the original image and Grad-CAM visualization
4. See the emotion prediction with probability breakdown

### Video Emotion Detection
1. Go to the "🎬 Video Emotion Detection" tab
2. Upload a video file
3. Wait for processing (may take longer for large videos)
4. View the results and processing details

## Sample Files

The UI includes quick access to sample files:
- **Audio**: Happy and Angry sample audio files
- **Face**: Happy, Angry, Disgust, and Nervous sample images

## Technical Details

- **Backend**: PyTorch models (Audio CNN, Face CNN, Combined LSTM)
- **Frontend**: Streamlit web framework
- **Visualization**: Grad-CAM for face model attention
- **File Handling**: Automatic temporary file management

## Troubleshooting

### File Upload Issues
- Ensure file formats are supported (see Usage section)
- Check file size (very large files may take longer)
- Make sure the file is not corrupted

### Processing Errors
- Verify models are present in `models/` directory
- Check that input_files and output_files folders exist
- Ensure all dependencies are installed

### Memory Issues
- Close other applications to free up RAM
- Use smaller test files first
- The application requires ~8GB RAM for inference

## Model Information

- **Audio Model**: 79.8% validation accuracy
- **Face Model**: 73.8% validation accuracy
- **Combined Model**: 75.2% validation accuracy

## Notes

- First run may take longer as models are loaded into memory
- Video processing can take several minutes depending on length
- The combined model processes both audio and video frames
- Grad-CAM visualizations show which parts of the face the model focuses on

## Browser Compatibility

Works best with:
- Chrome/Chromium
- Firefox
- Safari
- Edge

## Support

For issues or questions, refer to the main project README or create an issue on GitHub.

