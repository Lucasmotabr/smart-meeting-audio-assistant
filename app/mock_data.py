from __future__ import annotations

import math
import time

import numpy as np

try:
    from .contracts import (
        AdviceState,
        AudioFrame,
        AudioQuality,
        NoiseClassification,
        NoiseLabel,
        SystemSnapshot,
        TranscriptState,
        VisualizationFrame,
    )
except ImportError:
    from contracts import (
        AdviceState,
        AudioFrame,
        AudioQuality,
        NoiseClassification,
        NoiseLabel,
        SystemSnapshot,
        TranscriptState,
        VisualizationFrame,
    )


SAMPLE_RATE = 16_000
CHUNK_SECONDS = 1.0


def make_mock_snapshot(start_time: float, scenario: str) -> SystemSnapshot:
    elapsed = time.time() - start_time
    samples = _mock_audio_samples(elapsed, scenario)
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))

    audio = AudioFrame(
        samples=samples,
        sample_rate=SAMPLE_RATE,
        rms=rms,
        peak=peak,
        timestamp_seconds=elapsed,
        microphone_name=_microphone_name(scenario),
        microphone_type=_microphone_type(scenario),
    )

    classification = _classification_for_scenario(scenario, elapsed)
    quality = _quality_for_scenario(scenario, rms)

    return SystemSnapshot(
        status="recording",
        audio=audio,
        visualization=VisualizationFrame(
            waveform=samples,
            spectrogram=_mock_spectrogram(scenario, elapsed),
            voice_bars=_voice_bars(elapsed, rms),
        ),
        classification=classification,
        transcript=TranscriptState(
            text=_transcript_for_scenario(scenario),
            partial_text=_partial_text(elapsed),
            latency_ms=450 if quality.level == "high" else 900,
        ),
        quality=quality,
        advice=AdviceState(
            message=_advice_for_state(classification, quality),
            updated_at_seconds=elapsed,
        ),
    )


def _mock_audio_samples(elapsed: float, scenario: str) -> np.ndarray:
    t = np.linspace(0, CHUNK_SECONDS, int(SAMPLE_RATE * CHUNK_SECONDS), endpoint=False)
    speech_wave = 0.35 * np.sin(2 * math.pi * 180 * t)
    speech_wave += 0.15 * np.sin(2 * math.pi * 420 * t)
    envelope = 0.5 + 0.5 * np.sin(2 * math.pi * 1.6 * (t + elapsed))

    rng = np.random.default_rng(int(elapsed * 10) + len(scenario))
    noise = rng.normal(0, 0.015, size=t.shape)

    if scenario == "Keyboard typing noise":
        clicks = ((np.sin(2 * math.pi * 9 * (t + elapsed)) > 0.95).astype(float)) * 0.45
        return np.clip((speech_wave * envelope) + clicks + rng.normal(0, 0.04, size=t.shape), -1, 1)
    if scenario == "Far distance speech":
        return np.clip((speech_wave * envelope * 0.18) + rng.normal(0, 0.035, size=t.shape), -1, 1)
    if scenario == "Silence / idle room":
        return rng.normal(0, 0.006, size=t.shape)
    return np.clip((speech_wave * envelope) + noise, -1, 1)


def _simple_spectrogram(samples: np.ndarray) -> np.ndarray:
    window_size = 512
    hop = 256
    frames = []
    for start in range(0, len(samples) - window_size, hop):
        windowed = samples[start : start + window_size] * np.hanning(window_size)
        magnitude = np.abs(np.fft.rfft(windowed))
        frames.append(20 * np.log10(magnitude + 1e-6))
    return np.array(frames).T


def _mock_spectrogram(scenario: str, elapsed: float) -> np.ndarray:
    """Build stable, presentation-friendly dB data until the real module arrives."""

    rows = 128
    cols = 256
    rng = np.random.default_rng(int(elapsed * 4) + len(scenario) * 100)
    db = rng.normal(-104, 4.5, size=(rows, cols))

    frequencies = np.linspace(0, 1, rows)[:, None]
    timeline = np.linspace(0, 1, cols)[None, :]
    time_seconds = np.linspace(0, 30, cols)[None, :]
    speech_envelope = (
        0.25
        + 0.75 * np.square(np.sin(2 * math.pi * (timeline * 3.5 + elapsed * 0.08)))
    )

    if scenario != "Silence / idle room":
        low_shelf = np.exp(-0.5 * np.square(frequencies / 0.055))
        db += low_shelf * speech_envelope * 18

        for harmonic in range(1, 6):
            center = (120 * harmonic) / SAMPLE_RATE
            width = 0.008
            harmonic_band = np.exp(-0.5 * np.square((frequencies - center) / width))
            db += harmonic_band * speech_envelope * (40 / harmonic)

        voicing = np.clip(np.sin(2 * math.pi * time_seconds / 0.4), 0, 1)
        db += 15 * voicing * np.exp(-frequencies / 0.22)

        for center, width, strength in [
            (0.08, 0.025, 58),
            (0.16, 0.035, 48),
            (0.27, 0.045, 40),
            (0.42, 0.065, 30),
            (0.62, 0.08, 20),
        ]:
            band = np.exp(-0.5 * np.square((frequencies - center) / width))
            db += band * speech_envelope * strength

        harmonic_texture = 6 * np.sin(2 * math.pi * (frequencies * 22 + timeline * 2))
        db += harmonic_texture * speech_envelope

    if scenario == "Keyboard typing noise":
        for column in [8, 20, 33, 47, 62, 76, 88]:
            width = 1 + column % 2
            db[:, max(0, column - width) : column + width + 1] += 42
        db += rng.normal(0, 5, size=db.shape)
    elif scenario == "Far distance speech":
        db -= 18
        db += rng.normal(0, 6, size=db.shape)
    elif scenario == "Silence / idle room":
        db = rng.normal(-112, 3, size=db.shape)
    elif scenario == "Microphone comparison":
        db += 5 * np.exp(-0.5 * np.square((frequencies - 0.55) / 0.18))

    return np.clip(db, -120, -20)


def _voice_bars(elapsed: float, rms: float) -> np.ndarray:
    x = np.linspace(0, math.pi * 2, 24)
    bars = rms * (0.7 + 0.3 * np.sin(x + elapsed * 5))
    return np.clip(bars, 0, 1)


def _classification_for_scenario(scenario: str, elapsed: float) -> NoiseClassification:
    wobble = 0.04 * math.sin(elapsed * 2)
    if scenario == "Keyboard typing noise":
        return NoiseClassification(NoiseLabel.TYPING, 0.86 + wobble, 0.82)
    if scenario == "Far distance speech":
        return NoiseClassification(NoiseLabel.SPEECH, 0.62 + wobble, 0.58)
    if scenario == "Silence / idle room":
        return NoiseClassification(NoiseLabel.SILENCE, 0.94 + wobble, 0.91)
    return NoiseClassification(NoiseLabel.SPEECH, 0.91 + wobble, 0.88)


def _quality_for_scenario(scenario: str, rms: float) -> AudioQuality:
    if scenario == "Keyboard typing noise":
        return AudioQuality("medium", 68, signal_to_noise_db=14.5, issues=["typing noise detected"])
    if scenario == "Far distance speech":
        return AudioQuality("low", 42, signal_to_noise_db=7.8, issues=["low volume", "weak speech signal"])
    if scenario == "Silence / idle room":
        return AudioQuality("medium", 72, signal_to_noise_db=None, issues=["no speech detected"])
    score = min(100, max(0, 80 + rms * 40))
    return AudioQuality("high", score, signal_to_noise_db=28.0, issues=[])


def _advice_for_state(classification: NoiseClassification, quality: AudioQuality) -> str:
    if classification.label == NoiseLabel.TYPING:
        return "Keyboard typing noise detected. Reduce typing during speech or move the microphone closer to the speaker."
    if "low volume" in quality.issues:
        return "Speech volume is low. Move closer to the microphone or increase input gain."
    if classification.label == NoiseLabel.SILENCE:
        return "No speech detected. Start speaking or check that the correct microphone is selected."
    return "Audio quality is strong. This condition should produce stable speech recognition."


def _transcript_for_scenario(scenario: str) -> str:
    if scenario == "Far distance speech":
        return "The system can still hear speech, but recognition may be unstable at this distance."
    if scenario == "Keyboard typing noise":
        return "We are testing speech recognition while keyboard typing noise happens in the background."
    if scenario == "Silence / idle room":
        return ""
    return "This is a clean speech test for the smart meeting audio assistant."


def _partial_text(elapsed: float) -> str:
    dots = "." * (int(elapsed) % 4)
    return f"Listening{dots}"


def _microphone_name(scenario: str) -> str:
    if scenario == "Microphone comparison":
        return "EarPods condenser microphone"
    return "Alienware MEMS microphone"


def _microphone_type(scenario: str) -> str:
    if scenario == "Microphone comparison":
        return "Condenser"
    return "MEMS"
