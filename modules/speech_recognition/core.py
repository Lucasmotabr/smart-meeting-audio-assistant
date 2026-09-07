import os
import tempfile
from pathlib import Path
import soundfile as sf
import time

model = None
model_name = None


def _load_model(requested_model=None):
    global model, model_name
    requested_model = requested_model or os.environ.get("SMAA_WHISPER_MODEL", "tiny")
    if model is not None and model_name == requested_model:
        return model

    import whisper

    download_root = os.environ.get(
        "SMAA_WHISPER_MODEL_DIR",
        str(Path(__file__).resolve().parents[2] / "models" / "whisper"),
    )
    model = whisper.load_model(requested_model, download_root=download_root)
    model_name = requested_model
    return model

def transcribe_audio(samples, sample_rate, language=None, model_name=None):
    start_time = time.time()
    active_model = _load_model(model_name)
    temp_path = None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        sf.write(temp_file.name, samples, sample_rate)

    try:
        options = {"fp16": False}
        language = language or os.environ.get("SMAA_WHISPER_LANGUAGE")
        if language and language != "auto":
            options["language"] = language
        result = active_model.transcribe(temp_path, **options)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "text": result["text"],
        "partial_text": "",
        "latency_ms": latency_ms,
    }
