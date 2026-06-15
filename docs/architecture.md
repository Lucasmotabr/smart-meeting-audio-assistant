# Architecture Draft

## Final Pipeline

```text
Microphone Input
-> Audio Stream
-> Visualization
-> Noise Classification
-> Speech Recognition
-> Audio Quality Estimation
-> LLM Advice
-> GUI Dashboard
```

## Week 1 Strategy

The GUI should not wait for every backend module. It should render from mock data first, then each team member can replace one mock section with real output.

The shared interface is `app/contracts.py`.

## Data Contracts

### Audio Input Module

Owner: Member 1

Expected output:

```python
AudioFrame(
    samples=np.ndarray,          # mono float audio, range -1.0 to 1.0
    sample_rate=16000,
    rms=0.0,
    peak=0.0,
    timestamp_seconds=0.0,
    microphone_name="...",
    microphone_type="MEMS | Condenser | Unknown",
)
```

### Visualization Module

Owner: Member 2

Expected output:

```python
VisualizationFrame(
    waveform=np.ndarray,
    spectrogram=np.ndarray,
    voice_bars=np.ndarray,
)
```

### Noise Classification Module

Owner: Member 3

Expected output:

```python
NoiseClassification(
    label=NoiseLabel.SPEECH,
    confidence=0.91,
    smoothed_confidence=0.88,
)
```

Required labels for the demo:

- speech
- typing
- clap
- silence
- background noise
- unknown

### Speech Recognition Module

Owner: Member 4

Expected output:

```python
TranscriptState(
    text="final or stable transcript",
    partial_text="current partial phrase",
    latency_ms=450,
)
```

### GUI + LLM Advice Module

Owner: Member 5 / Lucas

Expected input:

- latest classification
- latest transcript
- audio level
- microphone name and type
- quality score

Expected output:

```python
AdviceState(
    message="Move closer to the microphone.",
    updated_at_seconds=12.3,
    source="openai | groq | ollama | rule-based placeholder",
)
```

## Integration Plan

1. Keep each module as a separate function or class.
2. Each module receives either raw audio samples or the latest `SystemSnapshot`.
3. Each module returns its own contract object.
4. The integrator creates one `SystemSnapshot` per GUI refresh.
5. Streamlit reads the snapshot and renders the dashboard.

## Latency Targets

- Audio frame size: 0.5 to 1.0 seconds for the prototype
- GUI refresh: 0.5 to 1.0 seconds
- Classifier smoothing window: 3 to 5 predictions
- Whisper chunk size: 3 to 5 seconds if real-time latency is difficult

