import os
import json
import subprocess
import requests
import gradio as gr

# ==========================
# 🔧 CONFIGURATION
# ==========================
OLLAMA_URL = "http://localhost:11434"
WHISPER_MODELS_DIR = "./whisper.cpp/models"


# ==========================
# 🔍 MODEL UTILITIES
# ==========================
def list_ollama_models() -> list[str]:
    """Get all available models from the local Ollama server."""
    try:
        res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        res.raise_for_status()
        data = res.json()
        return [m["model"] for m in data.get("models", [])]
    except Exception as e:
        print(f"⚠️ Could not retrieve Ollama models: {e}")
        return []


def list_whisper_models() -> list[str]:
    """Return Whisper model names based on files available in the models folder."""
    if not os.path.exists(WHISPER_MODELS_DIR):
        return []
    valid = ["base", "small", "medium", "large", "large-v3"]
    models = []
    for f in os.listdir(WHISPER_MODELS_DIR):
        if f.endswith(".bin"):
            name = os.path.splitext(f)[0].replace("ggml-", "")
            if any(v in name.lower() for v in valid):
                models.append(name)
    return sorted(set(models))


# ==========================
# 🎧 AUDIO PROCESSING
# ==========================
def preprocess_audio(file_path: str) -> str:
    """Convert audio to 16kHz mono WAV for Whisper."""
    output = f"{os.path.splitext(file_path)[0]}_clean.wav"
    cmd = f'ffmpeg -y -i "{file_path}" -ar 16000 -ac 1 "{output}"'
    subprocess.run(cmd, shell=True, check=True)
    return output


# ==========================
# 🗣️ TRANSCRIPTION + SUMMARY
# ==========================
def transcribe_and_summarize(audio_path, context, whisper_model, llm_model):
    """Convert speech to text (Whisper) and summarize with Ollama."""
    if not audio_path:
        return "❌ Please upload an audio file.", None

    with gr.Progress(track_tqdm=True) as progress:
        progress(0, desc="Preprocessing audio...")
        wav_file = preprocess_audio(audio_path)

        # Step 1: Transcribe using Whisper.cpp
        transcript_output = "transcript.txt"
        whisper_cmd = (
            f'./whisper.cpp/main -m ./whisper.cpp/models/ggml-{whisper_model}.bin '
            f'-f "{wav_file}" > "{transcript_output}"'
        )

        progress(0.4, desc="Running Whisper model...")
        subprocess.run(whisper_cmd, shell=True, check=True)

        with open(transcript_output, "r") as f:
            transcript = f.read()

        # Step 2: Summarize with Ollama
        progress(0.7, desc="Generating summary with Ollama...")
        summary = run_ollama_summary(llm_model, context, transcript)

        progress(1, desc="Done!")

    # Clean up
    os.remove(wav_file)
    return summary, transcript_output


# ==========================
# 🧠 OLLAMA INTERACTION
# ==========================
def run_ollama_summary(model, context, text):
    """Send the transcript to Ollama for summarization."""
    prompt = f"""
You are a helpful AI that summarizes meeting transcripts.

Context: {context or 'No additional context provided.'}

Transcript:
{text}

Write a clear and concise summary:
"""

    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt}

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True)
        resp.raise_for_status()
        summary = ""
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            summary += data.get("response", "")
            if data.get("done"):
                break
        return summary.strip()
    except Exception as e:
        return f"⚠️ Failed to generate summary: {e}"


# ==========================
# 🌐 GRADIO UI
# ==========================
def launch_ui():
    """Launch Gradio Blocks app."""
    ollama_models = list_ollama_models()
    whisper_models = list_whisper_models()

    with gr.Blocks(title="AI Meeting Summarizer") as app:
        gr.Markdown(
            """
            # 🎯 AI Meeting Summarizer  
            Upload your meeting audio and get an instant summary using **Whisper.cpp** and **Ollama**.  
            """,
        )

        with gr.Row():
            audio = gr.Audio(type="filepath", label="🎤 Upload Audio File")
            context = gr.Textbox(
                label="📝 Context (optional)",
                placeholder="Add meeting topic or background info (optional)",
            )

        with gr.Row():
            whisper_model = gr.Dropdown(
                whisper_models,
                label="🎧 Whisper Model",
                value=whisper_models[0] if whisper_models else None,
            )
            llm_model = gr.Dropdown(
                ollama_models,
                label="🧠 Summarization Model (Ollama)",
                value=ollama_models[0] if ollama_models else None,
            )

        with gr.Row():
            btn = gr.Button("🚀 Summarize Meeting")
        summary = gr.Textbox(label="📄 Summary", show_copy_button=True)
        transcript = gr.File(label="📥 Download Transcript")

        btn.click(
            transcribe_and_summarize,
            inputs=[audio, context, whisper_model, llm_model],
            outputs=[summary, transcript],
        )

    app.launch(debug=True, show_error=True)


# ==========================
# ▶️ MAIN
# ==========================
if __name__ == "__main__":
    launch_ui()
