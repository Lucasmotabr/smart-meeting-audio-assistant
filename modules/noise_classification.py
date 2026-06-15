"""
Member 3: Noise / Sound Environment Classification
----------------------------------------------------
Detects the current sound environment from an audio buffer using
YAMNet (pretrained audio event classifier from TensorFlow Hub).

Main entry point:
    classify_noise(samples, sample_rate) -> dict
        {
            "label": <one of the allowed labels>,
            "confidence": <float 0-1>
        }

Allowed labels:
    "speech", "typing", "clap", "silence", "background noise", "unknown"
"""

import numpy as np

# ----------------------------------------------------------------------
# Lazy-loaded globals (model is loaded once, on first use)
# ----------------------------------------------------------------------
_model = None
_class_names = None

YAMNET_HANDLE = "https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1"
YAMNET_SAMPLE_RATE = 16000  # YAMNet expects mono 16kHz audio

# ----------------------------------------------------------------------
# Mapping from YAMNet's 521 AudioSet classes -> our allowed labels
# ----------------------------------------------------------------------
_SPEECH_KEYWORDS = [
    "speech", "narration", "conversation", "monologue", "babbling",
    "child speech", "shout", "yell", "whisper", "singing", "talk",
]

_TYPING_KEYWORDS = [
    "typing", "computer keyboard", "typewriter", "keyboard",
]

_CLAP_KEYWORDS = [
    "clapping", "applause", "slap, smack",
]

_SILENCE_KEYWORDS = [
    "silence",
]


def _load_model():
    """Load YAMNet model and class names (cached after first call)."""
    global _model, _class_names

    if _model is not None:
        return _model, _class_names

    import tensorflow as tf
    import tensorflow_hub as hub
    import csv

    _model = hub.load(YAMNET_HANDLE)

    class_map_path = _model.class_map_path().numpy().decode("utf-8")
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_names.append(row[2])

    _class_names = class_names
    return _model, _class_names


def _map_to_allowed_label(yamnet_label):
    """Convert a YAMNet display-name label to one of our allowed labels."""
    label_lower = yamnet_label.lower()

    for kw in _SILENCE_KEYWORDS:
        if kw in label_lower:
            return "silence"

    for kw in _SPEECH_KEYWORDS:
        if kw in label_lower:
            return "speech"

    for kw in _TYPING_KEYWORDS:
        if kw in label_lower:
            return "typing"

    for kw in _CLAP_KEYWORDS:
        if kw in label_lower:
            return "clap"

    return "background noise"


def _resample_if_needed(samples, sample_rate, target_rate=YAMNET_SAMPLE_RATE):
    """Resample audio to target_rate (YAMNet requires 16kHz)."""
    if sample_rate == target_rate:
        return samples

    import librosa
    return librosa.resample(
        samples.astype(np.float32), orig_sr=sample_rate, target_sr=target_rate
    )


def _to_mono(samples):
    """Convert stereo/multi-channel audio to mono by averaging channels."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=-1)
    return samples


def classify_noise(samples, sample_rate):
    """
    Classify the dominant sound environment in an audio buffer.

    Returns
    -------
    dict
        {"label": <str>, "confidence": <float>}
    """
    samples = np.asarray(samples)

    if samples.size == 0:
        return {"label": "unknown", "confidence": 0.0}

    samples = _to_mono(samples)

    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / 32768.0
    else:
        samples = samples.astype(np.float32)

    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    SILENCE_RMS_THRESHOLD = 0.003

    if rms < SILENCE_RMS_THRESHOLD:
        confidence = float(np.clip(1.0 - (rms / SILENCE_RMS_THRESHOLD), 0.0, 1.0))
        confidence = max(confidence, 0.5)
        return {"label": "silence", "confidence": round(confidence, 2)}

    try:
        samples_16k = _resample_if_needed(samples, sample_rate)
    except Exception:
        return {"label": "unknown", "confidence": 0.0}

    if samples_16k.size == 0:
        return {"label": "unknown", "confidence": 0.0}

    try:
        model, class_names = _load_model()

        import tensorflow as tf
        waveform = tf.convert_to_tensor(samples_16k, dtype=tf.float32)

        scores, embeddings, spectrogram = model(waveform)
        scores_np = scores.numpy()

        mean_scores = scores_np.mean(axis=0)

        top_idx = int(np.argmax(mean_scores))
        top_score = float(mean_scores[top_idx])
        top_label = class_names[top_idx]

        mapped_label = _map_to_allowed_label(top_label)

        return {
            "label": mapped_label,
            "confidence": round(top_score, 2),
        }

    except Exception as e:
        import traceback
        print(f"[noise_classification] Error running YAMNet: {e!r}")
        traceback.print_exc()
        return {"label": "unknown", "confidence": 0.0}


if __name__ == "__main__":
    SR = 16000

    silence_samples = (np.random.randn(SR * 2) * 0.0005).astype(np.float32)
    result_silence = classify_noise(silence_samples, SR)
    print("Test 1 - Silence input:")
    print(result_silence)
    print()

    speech_like = (np.random.randn(SR * 2) * 0.3).astype(np.float32)
    result_speech = classify_noise(speech_like, SR)
    print("Test 2 - Loud noise (general sound) input:")
    print(result_speech)
