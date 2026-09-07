from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np


QualityLevel = Literal["high", "medium", "low"]
RecordingStatus = Literal["idle", "recording", "error"]


class NoiseLabel(str, Enum):
    SPEECH = "speech"
    TYPING = "typing"
    CLAP = "clap"
    SILENCE = "silence"
    BACKGROUND_NOISE = "background noise"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class AudioFrame:
    """Latest audio chunk and visualization-friendly derived values."""

    samples: np.ndarray
    sample_rate: int
    rms: float
    peak: float
    timestamp_seconds: float
    microphone_name: str = "Mock microphone"
    microphone_type: str = "Unknown"


@dataclass(slots=True)
class VisualizationFrame:
    """Data consumed by the GUI visualization panels."""

    waveform: np.ndarray
    spectrogram: np.ndarray
    voice_bars: np.ndarray


@dataclass(slots=True)
class NoiseClassification:
    label: NoiseLabel
    confidence: float
    smoothed_confidence: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class TranscriptState:
    text: str
    partial_text: str = ""
    latency_ms: int | None = None


@dataclass(slots=True)
class AudioQuality:
    level: QualityLevel
    score: float
    signal_to_noise_db: float | None = None
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AdviceState:
    message: str
    updated_at_seconds: float
    source: str = "rule-based placeholder"


@dataclass(slots=True)
class SystemSnapshot:
    status: RecordingStatus
    audio: AudioFrame
    visualization: VisualizationFrame
    classification: NoiseClassification
    transcript: TranscriptState
    quality: AudioQuality
    advice: AdviceState
