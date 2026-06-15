# Smart Meeting Audio Assistant

Real-time microphone dashboard for a class project about how microphone quality, noise, and recording conditions affect speech recognition robustness.

## Goal

Build a GUI-first integration shell that can receive outputs from the rest of the team:

- microphone audio stream
- waveform and spectrogram data
- noise classification labels
- speech transcript
- audio quality metrics
- LLM advice

The dashboard can run now with mock data, so the GUI and integration contract can be prepared before every module is finished.

## Quick Start

```bash
cd src/smart_meeting_audio_assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Run the tests with:

```bash
pytest -q
```

## Project Structure

```text
app/
  contracts.py      Shared data shapes between modules and GUI
  mock_data.py      Simulated real-time system output
  dashboard.py      Streamlit GUI
docs/
  architecture.md   How modules connect
  integration_workflow.md  How teammate code enters the project
  gui_target.md     Screenshot-based GUI target
  team_plan.md      Who should deliver what
  handoff/          One-page instructions for each teammate
```

## Integration Contract

Every module should eventually produce or update part of a `SystemSnapshot`.

```python
SystemSnapshot(
    audio=AudioFrame(...),
    classification=NoiseClassification(...),
    transcript=TranscriptState(...),
    quality=AudioQuality(...),
    advice=AdviceState(...),
)
```

For Week 1, teammates can return dictionaries or Python objects matching the fields in `app/contracts.py`. For Week 2, the integration owner can replace `mock_data.py` with real adapters.

## For Teammates

Do not start by reading every file. Open only the handoff page for your role:

- `docs/handoff/member_1_audio.md`
- `docs/handoff/member_2_visualization.md`
- `docs/handoff/member_3_classification.md`
- `docs/handoff/member_4_transcription.md`
- `docs/handoff/member_5_gui_llm.md`

Each page explains what to build, what input to use, what output to return, and how to test it.

Before accepting teammate files, follow `docs/integration_workflow.md`.
