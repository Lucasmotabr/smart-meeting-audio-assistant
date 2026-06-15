import time

from app.dashboard import _apply_demo_microphone, _compact_html, _dashboard_html, _spectrogram_png_data_uri
from app.mock_data import make_mock_snapshot


def test_keyboard_scenario_has_dynamic_spectrogram_range():
    snapshot = make_mock_snapshot(time.time(), "Keyboard typing noise")
    spectrogram = snapshot.visualization.spectrogram

    assert spectrogram.shape == (128, 256)
    assert float(spectrogram.max() - spectrogram.min()) >= 80
    assert float(spectrogram.max()) >= -30
    assert float(spectrogram.min()) <= -110
    assert _spectrogram_png_data_uri(spectrogram).startswith("data:image/png;base64,iVBOR")


def test_clean_speech_spectrogram_emphasizes_low_frequencies():
    spectrogram = make_mock_snapshot(time.time(), "Clean speech").visualization.spectrogram

    low_frequency_mean = float(spectrogram[:8].mean())
    high_frequency_mean = float(spectrogram[64:].mean())

    assert low_frequency_mean >= high_frequency_mean + 20


def test_microphone_links_preserve_scenario_and_selection():
    snapshot = make_mock_snapshot(time.time(), "Clean speech")
    _apply_demo_microphone(snapshot, "EarPods")
    markup = _compact_html(_dashboard_html(snapshot, "Clean speech", "EarPods"))

    assert snapshot.audio.microphone_name == "EarPods USB-C"
    assert "?scenario=Clean%20speech&mic=EarPods" in markup
    assert "&mic=EarPods" in markup
