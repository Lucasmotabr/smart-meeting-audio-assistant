import numpy as np

from app.contracts import AudioFrame, AudioQuality


def test_audio_frame_accepts_numpy_samples():
    frame = AudioFrame(
        samples=np.zeros(16000),
        sample_rate=16000,
        rms=0.0,
        peak=0.0,
        timestamp_seconds=0.0,
    )

    assert frame.samples.shape == (16000,)
    assert frame.sample_rate == 16000


def test_audio_quality_tracks_issues():
    quality = AudioQuality("low", 35, issues=["low volume"])

    assert quality.level == "low"
    assert "low volume" in quality.issues

