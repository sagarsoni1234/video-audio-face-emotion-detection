"""
Streamlit Web UI for Video-Audio-Face-Emotion-Recognition
A modern, user-friendly interface for emotion detection
"""

import streamlit as st
import os
import tempfile
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import re

# Import project modules
import source.config as config
import source.audio_analysis_utils.predict as audio_predict
import source.audio_analysis_utils.utils as audio_utils
import source.face_emotion_utils.predict as face_predict
import source.face_emotion_utils.utils as face_utils
import source.audio_face_combined.predict as combined_predict

# Page configuration
st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .emotion-card {
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        background: #e0e0e0;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'audio_result' not in st.session_state:
    st.session_state.audio_result = None
if 'face_result' not in st.session_state:
    st.session_state.face_result = None
if 'combined_result' not in st.session_state:
    st.session_state.combined_result = None


def save_uploaded_file(uploaded_file, folder_path):
    """Save uploaded file to specified folder"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def predict_audio_streamlit(audio_file_path):
    """Wrapper for audio prediction that works with Streamlit"""
    try:
        # Ensure input folder exists
        os.makedirs(config.INPUT_FOLDER_PATH, exist_ok=True)
        os.makedirs(config.OUTPUT_FOLDER_PATH, exist_ok=True)
        
        # Copy file to input folder with original name
        import shutil
        filename = os.path.basename(audio_file_path)
        input_path = os.path.join(config.INPUT_FOLDER_PATH, filename)
        shutil.copy2(audio_file_path, input_path)
        
        # Verify file was copied
        if not os.path.exists(input_path):
            raise Exception(f"Failed to copy audio file to {input_path}")
        
        # Run prediction
        emotion, confidence, prob_string = audio_predict.predict(filename)
        
        # Parse probabilities from the string format
        probs = {}
        emotion_names = list(config.EMOTION_INDEX.values())
        
        # The prob_string format is like "Emotion: X.X%"
        lines = prob_string.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line and '%' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    emotion_name = parts[0].strip()
                    prob_str = parts[1].strip().replace('%', '').strip()
                    try:
                        probs[emotion_name] = float(prob_str) / 100.0
                    except:
                        pass
        
        # If parsing failed, create from confidence and emotion
        if not probs or len(probs) == 0:
            probs = {name: 0.0 for name in emotion_names}
            if emotion in emotion_names:
                probs[emotion] = confidence
                # Distribute remaining probability
                remaining = (1.0 - confidence) / max(1, len(emotion_names) - 1)
                for name in emotion_names:
                    if name != emotion:
                        probs[name] = remaining
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probs,
            'success': True
        }
    except Exception as e:
        import traceback
        return {
            'error': str(e) + "\n" + traceback.format_exc(),
            'success': False
        }


def predict_face_streamlit(image):
    """Wrapper for face prediction that works with Streamlit"""
    try:
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                # Create white background and paste image
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                img_array = np.array(rgb_image)
            elif image.mode != 'RGB':
                # Convert other modes to RGB
                img_array = np.array(image.convert('RGB'))
            else:
                img_array = np.array(image)
        else:
            img_array = image
        
        # Ensure RGB format (3 channels)
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:
                # RGBA to RGB conversion
                # Create white background
                rgb_array = np.ones((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8) * 255
                # Alpha composite
                alpha = img_array[:, :, 3:4] / 255.0
                rgb_array = (rgb_array * (1 - alpha) + img_array[:, :, :3] * alpha).astype(np.uint8)
                img_array = rgb_array
            elif img_array.shape[2] == 3:
                # Already RGB, keep as is
                pass
            else:
                raise ValueError(f"Invalid image format. Expected 3-channel RGB image, got shape: {img_array.shape}")
        elif len(img_array.shape) == 2:
            # Grayscale, convert to RGB
            img_array = np.stack([img_array, img_array, img_array], axis=2)
        else:
            raise ValueError(f"Invalid image format. Expected 2D or 3D array, got shape: {img_array.shape}")
        
        # Run prediction (disable verbose to avoid hanging, disable grad_cam for faster processing)
        # Note: grad_cam can be slow, so we'll make it optional
        result = face_predict.predict(img_array, verbose=False, grad_cam=False, imshow=False)
        
        if result is None:
            return {
                'error': 'No face detected in the image',
                'success': False
            }
        
        # Handle different return formats
        if len(result) == 5:
            # With grad_cam: (emotion_name, emotion_idx, probs_list, img, grad_cam_img)
            emotion_name, emotion_idx, probs_list, img_np, grad_cam_img = result
        elif len(result) == 4:
            # Without grad_cam: (emotion_name, emotion_idx, probs_list, img)
            emotion_name, emotion_idx, probs_list, img_np = result
            grad_cam_img = None
        else:
            raise ValueError(f"Unexpected result format. Expected 4 or 5 items, got {len(result)}: {result}")
        
        # Convert probabilities to dictionary
        probs = {}
        emotion_names = list(config.EMOTION_INDEX.values())
        for i, prob in enumerate(probs_list):
            if i < len(emotion_names):
                probs[emotion_names[i]] = float(prob)
        
        # Ensure images are in correct format for display
        if isinstance(img_np, np.ndarray):
            # Convert BGR to RGB if needed
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                # Check if it's BGR (OpenCV format)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) if img_np.dtype == np.uint8 else img_np
        
        result_dict = {
            'emotion': emotion_name,
            'confidence': float(max(probs_list)),
            'probabilities': probs,
            'image': img_np,
            'success': True
        }
        
        if grad_cam_img is not None:
            # Convert grad_cam to RGB if needed
            if isinstance(grad_cam_img, np.ndarray) and len(grad_cam_img.shape) == 3:
                grad_cam_img = cv2.cvtColor(grad_cam_img, cv2.COLOR_BGR2RGB) if grad_cam_img.dtype == np.uint8 else grad_cam_img
            result_dict['gradcam'] = grad_cam_img
        
        return result_dict
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return {
            'error': error_msg,
            'success': False
        }


def predict_combined_streamlit(video_file_path):
    """Wrapper for combined video prediction"""
    try:
        # Ensure the source file exists and is readable
        if not os.path.exists(video_file_path):
            raise Exception(f"Video file not found: {video_file_path}")
        
        # Check file size
        file_size = os.path.getsize(video_file_path)
        if file_size == 0:
            raise Exception("Video file is empty")
        
        # Copy file to input folder with a unique name to avoid conflicts
        import shutil
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{os.path.basename(video_file_path)}"
        input_path = os.path.join(config.INPUT_FOLDER_PATH, filename)
        
        # Ensure input folder exists
        os.makedirs(config.INPUT_FOLDER_PATH, exist_ok=True)
        
        # Copy file
        shutil.copy2(video_file_path, input_path)
        
        # Verify the copied file
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            raise Exception(f"Failed to copy video file to {input_path}")
        
        # Run prediction - this function processes the video
        # Note: predict_video prints to console, we capture stdout if needed
        import sys
        from io import StringIO
        
        # Redirect stdout to capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            combined_predict.predict_video(input_path)
            output = captured_output.getvalue()
        except Exception as e:
            # Clean up on error
            if os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except:
                    pass
            raise e
        finally:
            sys.stdout = old_stdout
        
        # Parse the output for emotion predictions
        # Extract overall predictions and segment predictions
        
        # Extract overall video prediction probabilities
        overall_predictions = {}
        overall_match = re.search(r'Overall video prediction probabilities:\s*\n(.*?)(?:\n\n|Check the)', output, re.DOTALL)
        if overall_match:
            overall_text = overall_match.group(1)
            # Parse lines like "Angry : 25.5%" or "Sad/Fear : 30.2%"
            for line in overall_text.strip().split('\n'):
                if ':' in line and '%' in line:
                    # Match emotion name and percentage
                    match = re.match(r'([\w/]+)\s*:\s*(\d+\.?\d*)%', line.strip())
                    if match:
                        emotion = match.group(1)
                        prob = float(match.group(2)) / 100.0
                        overall_predictions[emotion] = prob
        
        # Extract segment predictions
        segment_predictions = []
        # Pattern to match: "predictions of file X from NN: [0.1, 0.2, ...]"
        segment_blocks = re.split(r'_{10,}', output)  # Split by separator lines
        for block in segment_blocks:
            # Find predictions array
            pred_match = re.search(r'predictions of file.*?from NN:\s*\[(.*?)\]', block, re.DOTALL)
            if pred_match:
                probs_str = pred_match.group(1)
                # Parse probabilities array
                try:
                    probs = [float(x.strip()) for x in probs_str.split(',')]
                except:
                    continue
                
                # Find duration
                duration_match = re.search(r'Duration:\s*\n(.*?)\s+to\s+(.*?)(?:\n|$)', block, re.DOTALL)
                start_time = duration_match.group(1).strip() if duration_match else 'N/A'
                end_time = duration_match.group(2).strip() if duration_match else 'N/A'
                
                segment_predictions.append({
                    'start': start_time,
                    'end': end_time,
                    'probabilities': probs
                })
        
        # Find the dominant emotion (highest probability)
        dominant_emotion = None
        max_confidence = 0.0
        if overall_predictions:
            dominant_emotion = max(overall_predictions.items(), key=lambda x: x[1])[0]
            max_confidence = overall_predictions[dominant_emotion]
        
        # Extract transcript if available
        transcript_match = re.search(r'Transcript:\s*\n(.*?)(?:\n\n|Common words)', output, re.DOTALL)
        transcript = transcript_match.group(1).strip() if transcript_match else None
        
        return {
            'success': True,
            'message': 'Video processed successfully.',
            'emotion': dominant_emotion,
            'confidence': max_confidence,
            'probabilities': overall_predictions,
            'segments': segment_predictions,
            'transcript': transcript,
            'output': output  # Full output for debugging
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return {
            'error': f"{str(e)}\n\nFull traceback:\n{error_trace}",
            'success': False
        }


def display_emotion_results(result, title="Prediction Results"):
    """Display emotion prediction results in a nice format"""
    if not result or not result.get('success'):
        st.error(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    emotion = result['emotion']
    confidence = result['confidence']
    probabilities = result.get('probabilities', {})
    
    # Main emotion card
    st.markdown(f"""
        <div class="emotion-card">
            <h2>Predicted Emotion: {emotion}</h2>
            <h3>Confidence: {confidence*100:.1f}%</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Probability bars
    st.subheader("All Emotion Probabilities")
    
    # Sort by probability
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    for emotion_name, prob in sorted_probs:
        col1, col2, col3 = st.columns([2, 5, 1])
        with col1:
            st.write(f"**{emotion_name}**")
        with col2:
            st.progress(prob)
        with col3:
            st.write(f"{prob*100:.1f}%")


def main():
    # Header
    st.markdown('<h1 class="main-header">😊 Emotion Recognition System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        st.markdown("""
        This application provides three emotion detection models:
        
        1. **Audio Model** - Detects emotions from audio files
        2. **Face Model** - Detects emotions from facial images
        3. **Combined Model** - Detects emotions from videos (audio + face)
        
        Upload a file and get instant emotion predictions!
        """)
        
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.info("""
        **Models:**
        - Audio: 79.8% accuracy
        - Face: 73.8% accuracy  
        - Combined: 75.2% accuracy
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎵 Audio Emotion Detection", "😊 Face Emotion Detection", "🎬 Video Emotion Detection"])
    
    # Tab 1: Audio Emotion Detection
    with tab1:
        st.header("Audio Emotion Detection")
        st.markdown("Upload an audio file (WAV, MP3, MP4) to detect emotions from speech tone.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            audio_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'mp4', 'm4a'],
                key="audio_upload"
            )
        
        with col2:
            st.markdown("### Sample Files")
            st.info("Sample files are available in the `input_files/` directory. Upload them using the file uploader above.")
        
        if audio_file is not None:
            # Save uploaded file to persistent location
            with st.spinner("Processing audio file..."):
                # Use system temp directory for persistence
                import tempfile
                import uuid
                temp_dir = tempfile.gettempdir()
                unique_id = str(uuid.uuid4())[:8]
                tmp_filename = f"audio_{unique_id}_{audio_file.name}"
                tmp_path = os.path.join(temp_dir, tmp_filename)
                
                # Write file
                with open(tmp_path, 'wb') as f:
                    f.write(audio_file.read())
                
                # Verify file was written
                if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                    st.error("Failed to save audio file. Please try again.")
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    result = {'success': False, 'error': 'Failed to save audio file'}
                else:
                    # Run prediction
                    result = predict_audio_streamlit(tmp_path)
                    st.session_state.audio_result = result
                
                # Clean up temp file after processing
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
            
            # Display results
            if result.get('success'):
                st.success(f"✅ Analysis complete! Predicted emotion: **{result['emotion']}** ({result['confidence']*100:.1f}% confidence)")
                display_emotion_results(result, "Audio Emotion Prediction")
                
                # Audio player
                st.audio(audio_file, format=f"audio/{os.path.splitext(audio_file.name)[1][1:]}")
            else:
                error_msg = result.get('error', 'Unknown error')
                st.error(f"❌ Error: {error_msg}")
                with st.expander("View full error details"):
                    st.code(error_msg)
    
    # Tab 2: Face Emotion Detection
    with tab2:
        st.header("Face Emotion Detection")
        st.markdown("Upload an image with a face to detect emotions from facial expressions.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            image_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                key="face_upload"
            )
        
        with col2:
            st.markdown("### Sample Images")
            st.info("Sample images are available in the `input_files/` directory. Upload them using the file uploader above.")
        
        if image_file is not None:
            # Display uploaded image
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", width='stretch')
            
            # Run prediction
            with st.spinner("Analyzing facial expression..."):
                result = predict_face_streamlit(image)
                st.session_state.face_result = result
            
            # Display results
            if result.get('success'):
                # Show prediction immediately
                st.success(f"✅ Analysis complete! Predicted emotion: **{result['emotion']}** ({result['confidence']*100:.1f}% confidence)")
                
                # Display processed image
                st.subheader("Processed Face Image")
                if 'image' in result and result['image'] is not None:
                    try:
                        st.image(result['image'], width='stretch', channels='RGB')
                    except:
                        st.image(result['image'], width='stretch')
                
                # Note: Grad-CAM disabled for faster processing (saves 10-30 seconds)
                # To enable Grad-CAM, change grad_cam=False to grad_cam=True in predict_face_streamlit()
                
                display_emotion_results(result, "Face Emotion Prediction")
            else:
                error_msg = result.get('error', 'Unknown error')
                st.error(f"❌ Error: {error_msg}")
                # Show full error in expander for debugging
                with st.expander("View full error details"):
                    st.code(error_msg)
    
    # Tab 3: Video Emotion Detection
    with tab3:
        st.header("Video Emotion Detection (Combined Model)")
        st.markdown("""
        **🎬 Upload a video file to detect emotions using BOTH audio AND facial expressions in one go!**
        
        The combined model automatically:
        - 🎵 Extracts and analyzes **audio** (speech tone)
        - 😊 Extracts and analyzes **video frames** (facial expressions)
        - 📝 Transcribes audio for **text sentiment** analysis
        - 🧠 Combines all three for the **final emotion prediction**
        
        This multimodal approach provides **higher accuracy** than using audio or face alone!
        """)
        
        video_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video_upload"
        )
        
        if video_file is not None:
            # Save uploaded file to a persistent location
            result = None
            with st.spinner("Processing video (this may take a while)..."):
                # Save to a persistent temp file
                import tempfile
                import uuid
                temp_dir = tempfile.gettempdir()
                unique_id = str(uuid.uuid4())[:8]
                tmp_filename = f"video_{unique_id}_{video_file.name}"
                tmp_path = os.path.join(temp_dir, tmp_filename)
                
                # Write file
                with open(tmp_path, 'wb') as f:
                    f.write(video_file.read())
                
                # Verify file was written
                if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                    result = {
                        'success': False,
                        'error': 'Failed to save video file. Please try again.'
                    }
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                else:
                    # Run prediction
                    result = predict_combined_streamlit(tmp_path)
                    st.session_state.combined_result = result
                    
                    # Clean up temp file after processing
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except:
                            pass
            
            # Display results
            if result and result.get('success'):
                # Show main prediction if available
                if result.get('emotion') and result.get('confidence'):
                    st.success(f"✅ Analysis complete! Predicted emotion: **{result['emotion']}** ({result['confidence']*100:.1f}% confidence)")
                    display_emotion_results(result, "Video Emotion Prediction (Overall)")
                else:
                    st.success("✅ Video processed successfully!")
                
                st.info("""
                **What was analyzed:**
                - 🎵 Audio track → Emotion from speech tone
                - 😊 Video frames → Emotion from facial expressions  
                - 📝 Transcribed text → Sentiment analysis
                - 🧠 Combined → Final multimodal emotion prediction
                """)
                
                st.video(video_file)
                
                # Show transcript if available
                if result.get('transcript'):
                    with st.expander("📝 View Audio Transcript"):
                        st.text(result['transcript'])
                
                # Show segment predictions if available
                if result.get('segments'):
                    st.subheader("📊 Segment-by-Segment Predictions")
                    for i, segment in enumerate(result['segments']):
                        with st.expander(f"Segment {i+1}: {segment.get('start', 'N/A')} to {segment.get('end', 'N/A')}"):
                            if segment.get('probabilities'):
                                probs = segment['probabilities']
                                emotion_names = list(config.EMOTION_INDEX.values())
                                for j, (emotion, prob) in enumerate(zip(emotion_names, probs)):
                                    if j < len(probs):
                                        col1, col2, col3 = st.columns([2, 5, 1])
                                        with col1:
                                            st.write(f"**{emotion}**")
                                        with col2:
                                            st.progress(prob)
                                        with col3:
                                            st.write(f"{prob*100:.1f}%")
                
                # Show full output for debugging
                if result.get('output'):
                    with st.expander("🔍 View Full Processing Details"):
                        st.text(result['output'])
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                
                # Check if it's a Git LFS error and provide helpful instructions
                if "git lfs" in error_msg.lower() or "git-lfs" in error_msg.lower() or "lfs pointer" in error_msg.lower():
                    st.error("❌ **Model File Missing**")
                    st.warning("""
                    **The combined model file is not available.** This is because the model file is stored using Git LFS (Large File Storage) and wasn't downloaded.
                    
                    **To fix this, choose one of the following options:**
                    
                    **Option 1: Install Git LFS (Recommended)**
                    1. Install Git LFS:
                       - macOS: `brew install git-lfs`
                       - Linux: `sudo apt-get install git-lfs` or visit https://git-lfs.github.com/
                    2. Run: `git lfs install`
                    3. Run: `git lfs pull models/audio_face_combined/audio_face_combined_model.pth`
                    
                    **Option 2: Download from GitHub**
                    Visit the GitHub repository and download the model file directly from the releases or use the web interface.
                    """)
                else:
                    st.error(f"❌ Error: {error_msg}")
                
                with st.expander("View full error details"):
                    st.code(error_msg)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Emotion Recognition System | Powered by PyTorch & Streamlit</p>
        <p>Models: Audio CNN | Face CNN | Combined LSTM</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

