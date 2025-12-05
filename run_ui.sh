#!/bin/bash

# Script to run the Streamlit UI for Emotion Recognition

echo "🚀 Starting Emotion Recognition Web UI..."
echo ""

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit is not installed!"
    echo "Installing streamlit..."
    pip3 install streamlit
fi

# Check if required folders exist
if [ ! -d "input_files" ]; then
    echo "Creating input_files directory..."
    mkdir -p input_files
fi

if [ ! -d "output_files" ]; then
    echo "Creating output_files directory..."
    mkdir -p output_files
fi

# Run the Streamlit app
echo "✅ Starting web interface..."
echo "📱 The app will open in your browser at http://localhost:8501"
echo ""
python3 -m streamlit run app.py

