"""
Member 3: Noise / Sound Environment Classification
----------------------------------------------------
Detects the current sound environment from a live audio chunk using
YAMNet (pretrained audio event classifier from TensorFlow Hub).

Main entry point:
    classify_noise(samples, sample_rate) -> dict

Input contract:
    - samples: NumPy float32 array, mono, normalized [-1.0, 1.0]
    - sample_rate: 16000 Hz
    - chunk duration: ~1 second

Allowed labels:
    "speech", "typing", "clap", "silence", "background noise", "unknown"
"""

import numpy as np

_model = None
_class_names = None

YAMNET_HANDLE = "https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1"
YAMNET_SAMPLE_RATE = 16000

_SPEECH_KEYWORDS = [
    "speech", "narration", "conversation", "monologue", "babbling",
    "child speech", "shout", "yell", "whisper", "singing", "talk",
]
_TYPING_KEYWORDS = [
    "typing", "computer keyboard", "typewriter", "keyboard",
    "clicking", "click",
]
_CLAP_KEYWORDS = [
    "clapping", "applause", "slap, smack", "hands", "tap",
]
_SILENCE_KEYWORDS = [
    "silence",
]
_BACKGROUND_KEYWORDS = [
    "inside, small room", "outside, rural or natural", "room",
    "ambient", "noise", "hum", "buzz", "white noise",
    "air conditioning", "ventilation",
]


def _load_model():
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
    for kw in _BACKGROUND_KEYWORDS:
        if kw in label_lower:
            return "background noise"
    return "background noise"


def classify_noise(samples, sample_rate=16000):
    """
    Classify the dominant sound in a live audio chunk.

    Parameters
    ----------
    samples : np.ndarray
        Float32 mono array, normalized [-1.0, 1.0], at 16000 Hz, ~1 sec.
    sample_rate : int
        Expected 16000 Hz. Kept as parameter for interface compatibility.

    Returns
    -------
    dict: {"label": str, "confidence": float}
    """
    samples = np.asarray(samples, dtype=np.float32)

    if samples.size == 0:
        return {"label": "unknown", "confidence": 0.0}

    # --- Silence check ---
    rms = float(np.sqrt(np.mean(samples ** 2)))
    SILENCE_RMS_THRESHOLD = 0.005

    if rms < SILENCE_RMS_THRESHOLD:
        confidence = float(np.clip(1.0 - (rms / SILENCE_RMS_THRESHOLD) ** 0.3, 0.5, 1.0))
        return {"label": "silence", "confidence": round(confidence, 2)}

    # --- Run YAMNet ---
    try:
        model, class_names = _load_model()

        import tensorflow as tf
        waveform = tf.convert_to_tensor(samples, dtype=tf.float32)

        scores, _, _ = model(waveform)
        mean_scores = scores.numpy().mean(axis=0)

        top_idx = int(np.argmax(mean_scores))
        top_score = float(mean_scores[top_idx])
        mapped_label = _map_to_allowed_label(class_names[top_idx])

        return {
            "label": mapped_label,
            "confidence": round(top_score, 2),
        }

    except Exception as e:
        import traceback
        print(f"[noise_classification] Error: {e!r}")
        traceback.print_exc()
        return {"label": "unknown", "confidence": 0.0}


def classify_noise_multi(samples, sample_rate=16000, min_confidence=0.2):
    """
    Detect multiple sound events across time in an audio buffer.

    Returns
    -------
    list of dicts: [{"time_sec": float, "label": str, "confidence": float}, ...]
    """
    samples = np.asarray(samples, dtype=np.float32)

    if samples.size == 0:
        return []

    try:
        model, class_names = _load_model()

        import tensorflow as tf
        scores, _, _ = model(tf.convert_to_tensor(samples, dtype=tf.float32))
        scores_np = scores.numpy()

        FRAME_DURATION = 0.975
        events = []
        prev_label = None

        for i, frame_scores in enumerate(scores_np):
            top_idx = int(np.argmax(frame_scores))
            top_score = float(frame_scores[top_idx])
            mapped = _map_to_allowed_label(class_names[top_idx]) if top_score >= min_confidence else "unknown"

            if mapped != prev_label:
                events.append({
                    "time_sec": round(i * FRAME_DURATION, 2),
                    "label": mapped,
                    "confidence": round(top_score, 2),
                })
                prev_label = mapped

        return events

    except Exception as e:
        import traceback
        print(f"[noise_classification] Error in classify_noise_multi: {e!r}")
        traceback.print_exc()
        return []
