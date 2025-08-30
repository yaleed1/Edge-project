from flask import Flask, render_template, request, send_from_directory, jsonify, redirect, url_for
from faster_whisper import WhisperModel
import os
import uuid
import json
import tempfile
import shutil
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
SESSION_FOLDER = "sessions"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)
def convert_to_wav(input_path, output_path):
    """Convert any audio format to 16kHz mono WAV using ffmpeg."""
    command = [
        "ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1",
        "-y", output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        audio = request.files.get("audio")
        model_size = request.form.get("model_size", "base")
        device = request.form.get("device", "cpu")
        compute_type = request.form.get("compute_type", "int8")

        if audio:
            # Generate session ID and paths
            session_id = str(uuid.uuid4())
            filename = audio.filename
            original_path = os.path.join(UPLOAD_FOLDER, filename)
            audio.save(original_path)

            # Convert to 16kHz mono WAV for faster-whisper compatibility
            wav_path = os.path.join(tempfile.gettempdir(), f"{session_id}.wav")
            convert_to_wav(original_path, wav_path)

            # Load model and transcribe
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            segments, _ = model.transcribe(wav_path, word_timestamps=True)

            # Build OpenAI-style verbose_json
            segment_data = []
            full_text = ""
            for i, segment in enumerate(segments):
                text = segment.text.strip()
                full_text += text + " "
                segment_data.append({
                    "id": i,
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text,
                    "words": [
                        {
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                            "word": w.word.strip()
                        } for w in segment.words or []
                    ]
                })

            result = {
                "session_id": session_id,
                "name": filename,
                "audio_url": f"/uploads/{filename}",
                "text": full_text.strip(),  # ✅ Top-level text
                "segments": segment_data
            }

            # Save session file
            with open(os.path.join(SESSION_FOLDER, f"{session_id}.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            return redirect(url_for("session_view", session_id=session_id))

    return render_template("index.html")

@app.route("/session/<session_id>")
def session_view(session_id):
    try:
        with open(os.path.join(SESSION_FOLDER, f"{session_id}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        return render_template("session.html", data=data)
    except FileNotFoundError:
        return "Session not found", 404

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    sessions = []
    for filename in os.listdir(SESSION_FOLDER):
        if filename.endswith(".json"):
            session_id = filename.replace(".json", "")
            with open(os.path.join(SESSION_FOLDER, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append({
                    "session_id": session_id,
                    "name": data.get("name", "Untitled")
                })
    return jsonify(sessions)

if __name__ == "__main__":
    app.run(debug=True)
