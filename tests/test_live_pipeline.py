from app.contracts import NoiseLabel, SystemSnapshot
from app.live_pipeline import make_live_snapshot


def test_live_snapshot_falls_back_without_optional_dependencies():
    snapshot = make_live_snapshot(start_time=0.0)

    assert isinstance(snapshot, SystemSnapshot)
    assert snapshot.status == "recording"
    assert snapshot.audio.sample_rate == 16000
    assert snapshot.audio.samples.shape == (16000,)
    assert snapshot.visualization.waveform.shape == (16000,)
    assert snapshot.visualization.spectrogram.ndim == 2
    assert snapshot.classification.label in NoiseLabel

