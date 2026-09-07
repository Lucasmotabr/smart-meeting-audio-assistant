from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from typing import Any

import numpy as np

# Streamlit は実行スクリプト（app/）のディレクトリだけを sys.path に追加するため、
# プロジェクトルートにある `modules` パッケージ（modules.vad など）を import できない。
# ここでルートを明示的に通し、VAD・可視化・音声入力モジュールを確実に読めるようにする。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_CLASSIFICATION_CACHE: dict[str, Any] = {
    "updated_at": 0.0,
    "rms": None,
    "classification": None,
}
CLASSIFICATION_INTERVAL_SECONDS = float(os.environ.get("SMAA_CLASSIFICATION_INTERVAL_SECONDS", "1.0"))
AI_MODEL_LOCK = threading.RLock()

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


def list_live_microphones() -> list[dict[str, Any]]:
    try:
        from modules.audio_input import list_microphones

        microphones = list_microphones()
        if microphones:
            return _prioritize_microphones(microphones)
    except Exception:
        pass
    return _prioritize_microphones(_list_sounddevice_microphones())


def _prioritize_microphones(microphones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def priority(mic: dict[str, Any]) -> tuple[int, str]:
        name = str(mic.get("name", "")).lower()
        if any(token in name for token in ("macbook", "built-in", "internal", "pc microphone")):
            return (0, name)
        if any(token in name for token in ("iphone", "continuity")):
            return (2, name)
        return (1, name)

    return sorted(microphones, key=priority)


def get_audio_diagnostics() -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "python": sys.executable,
        "sounddevice": _has_packages("sounddevice"),
        "microphone_count": 0,
        "default_device": None,
        "error": "",
    }
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        diagnostics["default_device"] = list(sd.default.device)
        diagnostics["microphone_count"] = sum(
            1 for device in devices if device.get("max_input_channels", 0) > 0
        )
    except Exception as exc:
        diagnostics["error"] = str(exc)
    return diagnostics


def _list_sounddevice_microphones() -> list[dict[str, Any]]:
    try:
        import sounddevice as sd

        microphones = []
        for idx, device in enumerate(sd.query_devices()):
            if device.get("max_input_channels", 0) > 0:
                microphones.append(
                    {
                        "id": idx,
                        "name": device.get("name", f"Device {idx}"),
                        "type": f"{device.get('hostapi', 'CoreAudio')} input",
                    }
                )
        return microphones
    except Exception:
        return []


def make_live_snapshot(start_time: float, microphone_id: str | None = None) -> SystemSnapshot:
    elapsed = time.time() - start_time
    audio = _get_audio_frame(elapsed, microphone_id)
    visualization = _build_visualization(audio)
    vad = _detect_voice_activity(audio)
    vad_bars = _vad_voice_bars(audio.samples, vad["probability"])
    visualization = VisualizationFrame(
        waveform=visualization.waveform,
        spectrogram=visualization.spectrogram,
        voice_bars=vad_bars,
    )
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


def _get_audio_frame(elapsed: float, microphone_id: str | None) -> AudioFrame:
    try:
        from modules.audio_input import get_audio_frame

        device_id, device_name = _microphone_selector(microphone_id)
        data = get_audio_frame(device_id=device_id, device_name=device_name)
        if _is_silent_fallback(data):
            data = _get_sounddevice_audio_frame(elapsed, microphone_id) or data
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

    return _audio_frame_from_dict(data, elapsed)


def _microphone_selector(microphone_id: str | None) -> tuple[str | None, str | None]:
    if microphone_id is None:
        return None, None
    value = str(microphone_id).strip()
    if value == "":
        return None, None
    if value.isdigit():
        return value, None
    aliases = {
        "pc": "macbook",
        "computer": "macbook",
        "builtin": "macbook",
        "built-in": "macbook",
        "internal": "macbook",
    }
    return None, aliases.get(value.lower(), value)


def _audio_frame_from_dict(data: dict[str, Any], elapsed: float) -> AudioFrame:
    samples = np.asarray(data.get("samples", np.zeros(16_000)), dtype=np.float32).flatten()
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    samples = np.clip(samples, -1.0, 1.0)
    return AudioFrame(
        samples=samples,
        sample_rate=int(data.get("sample_rate", 16_000)),
        rms=_finite_float(data.get("rms"), _rms(samples)),
        peak=_finite_float(data.get("peak"), _peak(samples)),
        timestamp_seconds=float(data.get("timestamp_seconds", elapsed)),
        microphone_name=str(data.get("microphone_name", "Unknown microphone")),
        microphone_type=str(data.get("microphone_type", "Unknown")),
    )


def _is_silent_fallback(data: dict[str, Any]) -> bool:
    name = str(data.get("microphone_name", ""))
    samples = np.asarray(data.get("samples", []), dtype=np.float32)
    return (
        "Fallback" in name
        or str(data.get("microphone_type", "")) == "Unavailable"
        or (samples.size > 0 and float(np.max(np.abs(samples))) == 0.0)
    )


def _get_sounddevice_audio_frame(elapsed: float, microphone_id: str | None) -> dict[str, Any] | None:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = [
            (idx, device)
            for idx, device in enumerate(devices)
            if device.get("max_input_channels", 0) > 0
        ]
        if not input_devices:
            return None

        selected_id = _resolve_sounddevice_id(microphone_id, input_devices)
        selected_device = devices[selected_id]
        sample_rate = 16_000
        samples = sd.rec(
            sample_rate,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=selected_id,
        )
        sd.wait()
        samples = np.asarray(samples, dtype=np.float32).flatten()

        return {
            "samples": samples,
            "sample_rate": sample_rate,
            "rms": _rms(samples),
            "peak": _peak(samples),
            "timestamp_seconds": elapsed,
            "microphone_name": selected_device.get("name", f"Device {selected_id}"),
            "microphone_type": "CoreAudio direct input",
        }
    except Exception:
        return None


def _resolve_sounddevice_id(
    microphone_id: str | None,
    input_devices: list[tuple[int, dict[str, Any]]],
) -> int:
    if microphone_id is not None:
        selector = str(microphone_id).strip().lower()
        aliases = {
            "pc": ("macbook", "built-in", "internal"),
            "computer": ("macbook", "built-in", "internal"),
            "builtin": ("macbook", "built-in", "internal"),
            "built-in": ("macbook", "built-in", "internal"),
            "internal": ("macbook", "built-in", "internal"),
        }
        if not selector.isdigit():
            tokens = aliases.get(selector, (selector,))
            for idx, device in input_devices:
                name = str(device.get("name", "")).lower()
                if any(token in name for token in tokens) and "iphone" not in name:
                    return idx
        for idx, _device in input_devices:
            if str(idx) == str(microphone_id):
                return idx
    try:
        import sounddevice as sd

        default_input = int(sd.default.device[0])
        if any(idx == default_input for idx, _device in input_devices):
            return default_input
    except Exception:
        pass
    return input_devices[0][0]


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
    now = time.time()
    cached = _CLASSIFICATION_CACHE.get("classification")
    cached_rms = _CLASSIFICATION_CACHE.get("rms")
    if (
        cached is not None
        and cached_rms is not None
        and now - float(_CLASSIFICATION_CACHE.get("updated_at", 0.0)) < CLASSIFICATION_INTERVAL_SECONDS
        and abs(float(cached_rms) - float(audio.rms)) < 0.01
    ):
        return cached

    if audio.rms >= 0.0005 and importlib.util.find_spec("tensorflow") is None:
        return NoiseClassification(NoiseLabel.UNKNOWN, 0.0, 0.0, _empty_classification_scores())

    acquired = AI_MODEL_LOCK.acquire(blocking=False)
    if not acquired:
        if cached is not None:
            return cached
        return NoiseClassification(NoiseLabel.UNKNOWN, 0.0, 0.0, _empty_classification_scores())

    try:
        try:
            from modules.noise_classification import classify_noise

            data = classify_noise(audio.samples, audio.sample_rate)
        except Exception:
            data = {"label": "unknown", "confidence": 0.0}
    finally:
        AI_MODEL_LOCK.release()

    label = _noise_label(data.get("label", "unknown"))
    confidence = _clamped_float(data.get("confidence", 0.0))
    scores = _classification_scores(data.get("scores"), label, confidence)
    classification = NoiseClassification(label, confidence, confidence, scores)
    _CLASSIFICATION_CACHE.update(
        {
            "updated_at": now,
            "rms": float(audio.rms),
            "classification": classification,
        }
    )
    return classification


def _transcribe_audio(audio: AudioFrame) -> TranscriptState:
    if not _has_packages("whisper", "soundfile"):
        return TranscriptState(
            text="",
            partial_text="Speech recognition dependencies are not installed.",
            latency_ms=None,
        )
    return TranscriptState(
        text="",
        partial_text="Speech recognition is running in the dashboard background worker.",
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


def _empty_classification_scores() -> dict[str, float]:
    return {
        NoiseLabel.TYPING.value: 0.0,
        NoiseLabel.SPEECH.value: 0.0,
        NoiseLabel.BACKGROUND_NOISE.value: 0.0,
        NoiseLabel.SILENCE.value: 0.0,
        NoiseLabel.CLAP.value: 0.0,
    }


def _classification_scores(raw_scores: Any, label: NoiseLabel, confidence: float) -> dict[str, float]:
    scores = _empty_classification_scores()
    if isinstance(raw_scores, dict):
        for key, value in raw_scores.items():
            noise_label = _noise_label(key)
            if noise_label.value in scores:
                scores[noise_label.value] = _clamped_float(value)
        total = sum(scores.values())
        if total > 0.0:
            return {key: value / total for key, value in scores.items()}
    if label.value in scores:
        scores[label.value] = confidence
    return scores


def _has_packages(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


def _clamped_float(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _finite_float(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if np.isfinite(result) else fallback


def _rms(samples: np.ndarray) -> float:
    if not samples.size:
        return 0.0
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(np.mean(samples**2)))


def _peak(samples: np.ndarray) -> float:
    if not samples.size:
        return 0.0
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.max(np.abs(samples)))


def _detect_voice_activity(audio: AudioFrame) -> dict[str, Any]:
    try:
        from modules.vad import detect_voice_activity

        # 一部マイク（AMD ACP デジタルマイク等）は一定の直流バイアスを乗せて返す。
        # ゼロ平均化してから推論しないと Silero VAD が音声を検出できない。
        # ここはVAD推論専用の前処理で、波形・スペクトログラム表示には影響しない。
        samples = audio.samples
        if samples.size:
            samples = samples - float(np.mean(samples))
        return detect_voice_activity(samples, audio.sample_rate)
    except Exception:
        return {"is_speech": False, "probability": 0.0}


def _vad_voice_bars(samples: np.ndarray, probability: float, n_bars: int = 24) -> np.ndarray:
    """VAD 確率で周波数帯域エネルギーをゲートし、動きのある voice_bars を生成する。

    対数スケールで周波数帯域を分割し、各帯域の RMS を dB 換算する。
    """
    if samples.size == 0:
        return np.zeros(n_bars, dtype=np.float32)

    # 推論側と同じく DC 除去してから帯域エネルギーを計算する（表示用波形とは別系統）。
    samples = samples - float(np.mean(samples))
    windowed = samples * np.hanning(len(samples))
    fft = np.abs(np.fft.rfft(windowed))
    n_fft = len(fft)

    # 対数スケールで帯域境界を生成（低周波〜ナイキスト）
    edges = np.logspace(np.log10(1), np.log10(n_fft - 1), n_bars + 1).astype(int)
    edges = np.clip(edges, 0, n_fft - 1)

    bars = np.zeros(n_bars, dtype=np.float32)
    for i in range(n_bars):
        lo, hi = edges[i], edges[i + 1]
        if lo >= hi:
            hi = lo + 1
        band = fft[lo:hi]
        rms = float(np.sqrt(np.mean(band ** 2)))
        bars[i] = max(0.0, 20 * np.log10(rms + 1e-9) + 120)  # dB、0〜120の範囲

    # Do not make the panel look dead just because the speech probability is below
    # Silero's decision threshold. Keep low-level movement visible, then boost it
    # as VAD confidence increases.
    gate = 0.18 + 0.82 * _clamped_float(probability)
    bars *= gate
    return bars
