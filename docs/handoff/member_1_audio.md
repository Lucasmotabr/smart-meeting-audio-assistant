# Member 1 Handoff: Audio Input

## Your Goal

Capture real microphone audio continuously.

## What You Need To Build

Create a Python file:

```text
modules/audio_input.py
```

Main function:

```python
def get_audio_frame():
    ...
```

## Expected Output

Return:

```python
{
    "samples": samples,              # NumPy array, mono audio, values between -1.0 and 1.0
    "sample_rate": 16000,
    "rms": rms_volume,
    "peak": peak_volume,
    "microphone_name": "Your microphone name",
    "microphone_type": "MEMS or Condenser or Unknown",
}
```

## How This Connects To The GUI

For Week 1, returning the dictionary above is enough.

For integration, this output will be converted into `AudioFrame` in:

```text
app/contracts.py
```

Important fields for the GUI:

- `samples`
- `sample_rate`
- `rms`
- `peak`
- `microphone_name`
- `microphone_type`

## Recommended Libraries

- `sounddevice`
- `numpy`
- `scipy`

## How To Prove It Works

Send:

- the Python file
- the command to run it
- a screenshot or terminal output showing changing volume values

Minimum success:

- the script records from the microphone
- it prints RMS/volume every second
