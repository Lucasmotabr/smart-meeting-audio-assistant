import numpy as np

from modules.visualization import build_visualization


def test_build_visualization_returns_expected_arrays():
    samples = np.zeros(16000, dtype=np.float32)
    result = build_visualization(samples, 16000)

    assert result["waveform"].shape == (16000,)
    assert result["spectrogram"].ndim == 2
    assert result["voice_bars"].ndim == 1
    assert result["spectrogram"].shape[0] == 513

