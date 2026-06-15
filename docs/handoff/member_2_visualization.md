# Member 2 Handoff: Visualization

## Your Goal

Turn audio samples into waveform, voice bars, and spectrogram data.

## What You Need To Build

Create a Python file:

```text
modules/visualization.py
```

Main function:

```python
def build_visualization(samples, sample_rate):
    ...
```

## Expected Output

Return:

```python
{
    "waveform": waveform_array,
    "spectrogram": spectrogram_2d_array,
    "voice_bars": voice_bar_array,
}
```

## How This Connects To The GUI

For Week 1, returning the dictionary above is enough.

For integration, this output will be converted into `VisualizationFrame` in:

```text
app/contracts.py
```

Important fields for the GUI:

- `waveform`
- `spectrogram`
- `voice_bars`

## Recommended Libraries

- `numpy`
- `matplotlib`
- `librosa` optional

## How To Prove It Works

Send:

- the Python file
- one screenshot of waveform or spectrogram
- the command to run it

Minimum success:

- given a NumPy audio array, your code creates a waveform array
- your code creates a 2D spectrogram array
