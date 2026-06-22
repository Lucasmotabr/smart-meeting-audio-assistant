from __future__ import annotations

import numpy as np


def build_visualization(samples, sample_rate):
    samples = np.asarray(samples, dtype=np.float32).flatten()
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    samples = np.clip(samples, -1.0, 1.0)

    if samples.size == 0:
        raise ValueError("Audio samples cannot be empty.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    waveform = samples.copy()
    frame_size = 1024
    hop_size = 512

    if len(samples) < frame_size:
        analysis_samples = np.pad(samples, (0, frame_size - len(samples)))
    else:
        analysis_samples = samples

    spectrogram_frames = []
    voice_bar_values = []

    for start in range(0, len(analysis_samples) - frame_size + 1, hop_size):
        frame = analysis_samples[start : start + frame_size]
        windowed_frame = frame * np.hanning(frame_size)
        spectrum = np.abs(np.fft.rfft(windowed_frame))
        spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)
        spectrogram_frames.append(spectrum)

        rms = np.sqrt(np.mean(frame**2))
        voice_bar_values.append(rms)

    spectrogram = np.array(spectrogram_frames).T
    spectrogram = 20 * np.log10(spectrogram + 1e-10)
    spectrogram = np.nan_to_num(spectrogram, nan=-120.0, posinf=-20.0, neginf=-120.0)
    voice_bars = np.array(voice_bar_values)

    return {
        "waveform": waveform,
        "spectrogram": spectrogram,
        "voice_bars": voice_bars,
    }
