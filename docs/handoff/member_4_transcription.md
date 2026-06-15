# Member 4 Handoff: Speech Recognition

## Your Goal

Convert microphone speech into text.

## What You Need To Build

Create a Python file:

```text
modules/speech_recognition.py
```

Main function:

```python
def transcribe_audio(samples, sample_rate):
    ...
```

## Expected Output

Return:

```python
{
    "text": "Stable transcript text here",
    "partial_text": "Currently listening text",
    "latency_ms": 900,
}
```

## How This Connects To The GUI

For Week 1, returning the dictionary above is enough.

For integration, this output will be converted into `TranscriptState` in:

```text
app/contracts.py
```

Important fields for the GUI:

- `text`
- `partial_text`
- `latency_ms`

## Recommended Libraries

- `faster-whisper`
- `openai-whisper`
- `soundfile`

## How To Prove It Works

Send:

- the Python file
- transcript output from one short recording
- the command to run it
- what model size you used

Minimum success:

- transcribe a short WAV file
- print the recognized sentence

Real-time transcription can come later if it is too hard in Week 1.
