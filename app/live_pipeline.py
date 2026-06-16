from __future__ import annotations

import importlib.util
import time
from typing import Any

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


def make_live_snapshot(start_time: float) -> SystemSnapshot:
    elapsed = time.time() - start_time
    audio = _get_audio_frame(elapsed)
    visualization = _build_visualization(audio)
    classification = _classify_noise(audio)
    transcript = _transcribe_audio(audio)
    quality = _estimate_quality(audio, classification, transcript)
    advice = _generate_advice(classification, quality, elapsed)

    return SystemSnapshot(
        status="recording",
        audio=audio,
        visualization=visualization,
        classification=classification,
        transcript=transcript,
        quality=quality,
        advice=advice,
    )


def _get_audio_frame(elapsed: float) -> AudioFrame:
    try:
        from modules.audio_input import get_audio_frame

        data = get_audio_frame()
    except Exception:
        samples = np.zeros(16_000, dtype=np.float32)
        data = {
            "samples": samples,
            "sample_rate": 16_000,
            "rms": 0.0,
            "peak": 0.0,
            "timestamp_seconds": elapsed,
            "microphone_name": "None (audio fallback)",
            "microphone_type": "Unavailable",
        }

    samples = np.asarray(data.get("samples", np.zeros(16_000)), dtype=np.float32).flatten()
    return AudioFrame(
        samples=samples,
        sample_rate=int(data.get("sample_rate", 16_000)),
        rms=float(data.get("rms", _rms(samples))),
        peak=float(data.get("peak", _peak(samples))),
        timestamp_seconds=float(data.get("timestamp_seconds", elapsed)),
        microphone_name=str(data.get("microphone_name", "Unknown microphone")),
        microphone_type=str(data.get("microphone_type", "Unknown")),
    )


def _build_visualization(audio: AudioFrame) -> VisualizationFrame:
    try:
        from modules.visualization import build_visualization

        data = build_visualization(audio.samples, audio.sample_rate)
        return VisualizationFrame(
            waveform=np.asarray(data["waveform"], dtype=np.float32),
            spectrogram=np.asarray(data["spectrogram"], dtype=np.float32),
            voice_bars=np.asarray(data["voice_bars"], dtype=np.float32),
        )
    except Exception:
        return VisualizationFrame(
            waveform=audio.samples,
            spectrogram=_fallback_spectrogram(audio.samples),
            voice_bars=np.zeros(24, dtype=np.float32),
        )


def _classify_noise(audio: AudioFrame) -> NoiseClassification:
    if audio.rms >= 0.005 and importlib.util.find_spec("tensorflow") is None:
        return NoiseClassification(NoiseLabel.UNKNOWN, 0.0, 0.0)

    try:
        from modules.noise_classification import classify_noise

        data = classify_noise(audio.samples, audio.sample_rate)
    except Exception:
        data = {"label": "unknown", "confidence": 0.0}

    label = _noise_label(data.get("label", "unknown"))
    confidence = _clamped_float(data.get("confidence", 0.0))
    return NoiseClassification(label, confidence, confidence)


def _transcribe_audio(audio: AudioFrame) -> TranscriptState:
    if not _has_packages("whisper", "soundfile"):
        return TranscriptState(
            text="",
            partial_text="Speech recognition dependencies are not installed.",
            latency_ms=None,
        )

    try:
        from modules.speech_recognition import transcribe_audio

        data = transcribe_audio(audio.samples, audio.sample_rate)
        return TranscriptState(
            text=str(data.get("text", "")),
            partial_text=str(data.get("partial_text", "")),
            latency_ms=data.get("latency_ms"),
        )
    except Exception:
        return TranscriptState(
            text="",
            partial_text="Speech recognition is unavailable.",
            latency_ms=None,
        )


def _estimate_quality(
    audio: AudioFrame,
    classification: NoiseClassification,
    transcript: TranscriptState,
) -> AudioQuality:
    issues: list[str] = []

    if classification.label == NoiseLabel.SILENCE:
        issues.append("no speech detected")
    if classification.label == NoiseLabel.TYPING:
        issues.append("typing noise detected")
    if audio.rms < 0.01:
        issues.append("low volume")
    if transcript.latency_ms and transcript.latency_ms > 1500:
        issues.append("high transcription latency")

    if "typing noise detected" in issues:
        return AudioQuality("medium", 68, signal_to_noise_db=None, issues=issues)
    if "low volume" in issues or "no speech detected" in issues:
        return AudioQuality("low", 42, signal_to_noise_db=None, issues=issues)
    return AudioQuality("high", min(100.0, 80.0 + audio.rms * 80.0), signal_to_noise_db=None, issues=issues)


def _generate_advice(
    classification: NoiseClassification,
    quality: AudioQuality,
    elapsed: float,
) -> AdviceState:
    if classification.label == NoiseLabel.UNKNOWN:
        message = "Live integration mode is running, but the classifier model is not available yet."
    elif classification.label == NoiseLabel.SILENCE:
        message = "No speech detected from the live audio path. Check microphone input or start speaking."
    elif classification.label == NoiseLabel.TYPING:
        message = "Typing noise detected. Move the microphone closer to the speaker or reduce keyboard noise."
    elif "low volume" in quality.issues:
        message = "Live audio level is low. Move closer to the microphone or increase input gain."
    else:
        message = "Live audio path is producing usable input."

    return AdviceState(message=message, updated_at_seconds=elapsed, source="integration fallback")


def _fallback_spectrogram(samples: np.ndarray) -> np.ndarray:
    window_size = 512
    hop = 256
    frames = []
    for start in range(0, max(0, len(samples) - window_size), hop):
        windowed = samples[start : start + window_size] * np.hanning(window_size)
        magnitude = np.abs(np.fft.rfft(windowed))
        frames.append(20 * np.log10(magnitude + 1e-6))
    if not frames:
        return np.full((257, 1), -120, dtype=np.float32)
    return np.array(frames, dtype=np.float32).T


def _noise_label(value: Any) -> NoiseLabel:
    try:
        return NoiseLabel(str(value))
    except ValueError:
        return NoiseLabel.UNKNOWN


def _has_packages(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


def _clamped_float(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0


def _peak(samples: np.ndarray) -> float:
    return float(np.max(np.abs(samples))) if samples.size else 0.0
