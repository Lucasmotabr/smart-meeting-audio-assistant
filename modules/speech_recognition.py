import whisper
import tempfile
import soundfile as sf
import time

model = whisper.load_model("base")

def transcribe_audio(samples, sample_rate):
    start_time = time.time()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        sf.write(temp_file.name, samples, sample_rate)

        result = model.transcribe(temp_file.name)

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "text": result["text"],
        "partial_text": "",
        "latency_ms": latency_ms,
    }