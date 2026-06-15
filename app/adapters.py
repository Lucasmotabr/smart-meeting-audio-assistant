from __future__ import annotations

from typing import Protocol

try:
    from .contracts import (
        AdviceState,
        AudioFrame,
        AudioQuality,
        NoiseClassification,
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
        SystemSnapshot,
        TranscriptState,
        VisualizationFrame,
    )


class AudioInputAdapter(Protocol):
    """Member 1 should implement this interface."""

    def get_audio_frame(self) -> AudioFrame:
        ...


class VisualizationAdapter(Protocol):
    """Member 2 should implement this interface."""

    def build_visualization(self, audio: AudioFrame) -> VisualizationFrame:
        ...


class NoiseClassifierAdapter(Protocol):
    """Member 3 should implement this interface."""

    def classify_noise(self, audio: AudioFrame) -> NoiseClassification:
        ...


class SpeechRecognizerAdapter(Protocol):
    """Member 4 should implement this interface."""

    def transcribe_audio(self, audio: AudioFrame) -> TranscriptState:
        ...


class QualityEstimatorAdapter(Protocol):
    """The integrator can keep this simple for the class prototype."""

    def estimate_quality(
        self,
        audio: AudioFrame,
        classification: NoiseClassification,
        transcript: TranscriptState,
    ) -> AudioQuality:
        ...


class AdviceAdapter(Protocol):
    """GUI/LLM owner should implement this interface."""

    def generate_advice(self, snapshot: SystemSnapshot) -> AdviceState:
        ...
