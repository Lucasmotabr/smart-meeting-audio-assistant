# Member 5 Handoff: GUI And LLM Advice

## Your Goal

Build the dashboard and generate advice from system state.

## What You Need To Build

Main files:

```text
app/dashboard.py
app/mock_data.py
app/contracts.py
```

Advice function:

```python
def generate_advice(snapshot):
    ...
```

## Expected Input

Use:

- noise label
- confidence
- transcript
- volume level
- microphone type
- quality score

## Expected Output

Return:

```python
{
    "message": "Move closer to the microphone.",
    "source": "rule-based placeholder",
}
```

## How This Connects To The GUI

The shared GUI data shapes are defined in:

```text
app/contracts.py
```

The full dashboard state is called `SystemSnapshot`.

It combines:

- `AudioFrame`
- `VisualizationFrame`
- `NoiseClassification`
- `TranscriptState`
- `AudioQuality`
- `AdviceState`

For Week 1, the dashboard can use mock data. For Week 2, replace mock values with teammate outputs.

## Recommended Libraries

- `streamlit`
- `numpy`
- OpenAI, Groq, or Ollama later

## How To Prove It Works

Send:

- screenshot of the dashboard
- at least three demo scenarios
- one advice message for each scenario

Minimum success:

- dashboard runs with mock data
- voice bars, waveform, spectrogram, classification, transcript, quality, and advice panels are visible
