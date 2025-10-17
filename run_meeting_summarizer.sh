#!/bin/bash
# ================================================
# 🚀 AI-Powered Meeting Summarizer Setup Script
# Author: Dhilip Kumar (Edited Version)
# ================================================

# === CONFIGURATION ===
VENV_DIR=".venv"
PYTHON_BIN="python3"
REQUIREMENTS_FILE="requirements.txt"
WHISPER_CPP_REPO="https://github.com/ggerganov/whisper.cpp.git"
WHISPER_CPP_DIR="whisper.cpp"
WHISPER_MODEL="small"   # Change to: base / medium / large / large-v3
MAIN_SCRIPT="main.py"   # Change if your main file name is different

echo "==============================================="
echo "🧠 Setting up AI-Powered Meeting Summarizer..."
echo "==============================================="

# === Step 1: Check Python Installation ===
if ! command -v $PYTHON_BIN &>/dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.x and try again."
    exit 1
fi

# === Step 2: Create Virtual Environment ===
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment in '$VENV_DIR'..."
    $PYTHON_BIN -m venv $VENV_DIR
else
    echo "✅ Virtual environment already exists."
fi

# === Step 3: Activate Virtual Environment ===
echo "🔗 Activating virtual environment..."
source $VENV_DIR/bin/activate

# === Step 4: Upgrade pip ===
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# === Step 5: Install Requirements ===
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "📥 Installing dependencies from '$REQUIREMENTS_FILE'..."
    pip install -r $REQUIREMENTS_FILE
else
    echo "⚠️  requirements.txt not found. Installing essential packages only..."
    pip install requests gradio torch openai-whisper
fi

# === Step 6: Install FFmpeg if Missing ===
if ! command -v ffmpeg &>/dev/null; then
    echo "🎞️  Installing FFmpeg..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! command -v brew &>/dev/null; then
            echo "❌ Homebrew not found. Install it from https://brew.sh/ and rerun."
            exit 1
        fi
        brew install ffmpeg
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    else
        echo "⚠️  Unsupported OS for auto FFmpeg install. Install it manually."
    fi
else
    echo "✅ FFmpeg already installed."
fi

# === Step 7: Clone whisper.cpp if Needed ===
if [ ! -d "$WHISPER_CPP_DIR" ]; then
    echo "🌀 Cloning whisper.cpp repository..."
    git clone $WHISPER_CPP_REPO
else
    echo "✅ whisper.cpp repository already exists."
fi

# === Step 8: Build whisper.cpp ===
echo "🔧 Building whisper.cpp..."
cd $WHISPER_CPP_DIR || exit
make clean && make
if [ $? -ne 0 ]; then
    echo "❌ whisper.cpp build failed. Check your C++ compiler and retry."
    exit 1
fi
cd ..

# === Step 9: Download Whisper Model if Missing ===
MODEL_PATH="./$WHISPER_CPP_DIR/models/ggml-$WHISPER_MODEL.bin"
if [ ! -f "$MODEL_PATH" ]; then
    echo "📦 Downloading Whisper model: '$WHISPER_MODEL'..."
    bash $WHISPER_CPP_DIR/models/download-ggml-model.sh $WHISPER_MODEL
else
    echo "✅ Whisper model '$WHISPER_MODEL' already available."
fi

# === Step 10: Run the Python App ===
if [ -f "$MAIN_SCRIPT" ]; then
    echo "🚀 Running your Python application..."
    python "$MAIN_SCRIPT"
else
    echo "❌ '$MAIN_SCRIPT' not found. Please check your script name."
    deactivate
    exit 1
fi

# === Step 11: Cleanup & Exit ===
echo "🎉 Setup complete! Application is now running."
echo "To deactivate, use: deactivate"
