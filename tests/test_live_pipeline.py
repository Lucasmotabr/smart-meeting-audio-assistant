from app.contracts import NoiseLabel, SystemSnapshot
from app.dashboard import _integer_percentages
from app.live_pipeline import _classification_scores, make_live_snapshot


def test_live_snapshot_falls_back_without_optional_dependencies():
    snapshot = make_live_snapshot(start_time=0.0)

    assert isinstance(snapshot, SystemSnapshot)
    assert snapshot.status == "recording"
    assert snapshot.audio.sample_rate == 16000
    assert snapshot.audio.samples.shape == (16000,)
    assert snapshot.visualization.waveform.shape == (16000,)
    assert snapshot.visualization.spectrogram.ndim == 2
    assert snapshot.classification.label in NoiseLabel


def test_classification_scores_are_normalized():
    scores = _classification_scores(
        {"typing": 0.2, "speech": 0.3, "background noise": 0.5},
        NoiseLabel.SPEECH,
        0.3,
    )

    assert round(sum(scores.values()), 6) == 1.0
    assert scores["background noise"] == 0.5


def test_integer_percentages_sum_to_100():
    percentages = _integer_percentages([0.333, 0.333, 0.333, 0.0, 0.0])

    assert sum(percentages) == 100
    assert percentages[:3] == [34, 33, 33]
