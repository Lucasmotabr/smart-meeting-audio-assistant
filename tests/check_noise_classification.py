"""
Manual test for the noise classification module (YAMNet).

Reuses the team's microphone recorder (modules.audio_input) — it does NOT
build its own recorder — and feeds ~1 second chunks into classify_noise().

Run from the project root inside the `audio` conda env (has tensorflow):

    conda activate audio
    python tests/check_noise_classification.py

It verifies the reviewer's checklist:
  - YAMNet downloads + loads (first call) and is reused (no reload per frame)
  - processing time per 1-second chunk
  - output stays {"label": str, "confidence": 0.0-1.0, "scores": dict}
  - real classification for: silence, speech, typing, clap, background noise
"""

import os
import sys
import time

import numpy as np

# Make `modules` importable when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.audio_input import get_audio_frame
from modules.noise_classification import classify_noise
from modules.noise_classification.core import _load_model


TEST_CASES = [
    ("silence", "Stay quiet."),
    ("speech", "Talk normally."),
    ("typing", "Type on your keyboard."),
    ("clap", "Clap your hands a few times."),
    ("background noise", "Let ambient room noise play."),
]


def _check_output_shape(result):
    assert set(result) == {"label", "confidence", "scores"}, f"Unexpected keys: {result}"
    assert isinstance(result["label"], str), "label must be a string"
    conf = result["confidence"]
    assert 0.0 <= conf <= 1.0, f"confidence out of range: {conf}"
    assert isinstance(result["scores"], dict), "scores must be a dict"
    assert all(0.0 <= value <= 1.0 for value in result["scores"].values()), "scores out of range"


def main():
    print("Loading YAMNet (first call downloads ~16 MB)...")
    t0 = time.perf_counter()
    _load_model()
    print(f"  model ready in {time.perf_counter() - t0:.2f}s\n")

    # Confirm the model is cached and not reloaded on each frame.
    from modules.noise_classification import core
    model_id = id(core._model)

    for expected, instruction in TEST_CASES:
        input(f"[{expected.upper()}] {instruction} Press Enter to record 1s...")
        frame = get_audio_frame()
        samples = np.asarray(frame["samples"], dtype=np.float32)

        t0 = time.perf_counter()
        result = classify_noise(samples, frame["sample_rate"])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        _check_output_shape(result)
        assert id(core._model) == model_id, "Model was reloaded — should load once!"

        print(
            f"  -> label={result['label']!r}  "
            f"confidence={result['confidence']:.2f}  "
            f"time={elapsed_ms:.0f}ms  "
            f"(expected ~{expected})\n"
        )

    print("All output-format and load-once checks passed.")


if __name__ == "__main__":
    main()
