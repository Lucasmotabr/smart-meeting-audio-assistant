# Integration Status

## Branch To Use Today

Use:

```text
feature/organized-integration
```

Do not work directly on `main`.

## Current Folder Ownership

```text
app/
  dashboard.py
  live_pipeline.py
  contracts.py
  mock_data.py

modules/
  audio_input/
    core.py
    macos.py
    ubuntu.py
    legacy_recorder.py

  visualization/
    core.py
    notebooks/

  noise_classification/
    core.py

  speech_recognition/
    core.py
    manual_test_transcription.py
```

## What Works Now

- The dashboard still works in mock mode.
- `?mode=live` tries the integrated module pipeline.
- Audio has macOS and Ubuntu implementation files.
- Visualization has a Python function converted from the notebook.
- Noise classification is called through the live pipeline.
- Speech recognition is lazy-loaded so missing Whisper does not crash the app.
- Fallback mode keeps the dashboard running when optional dependencies are missing.

## What Is Still Not Fully Proven

- Real microphone capture on macOS.
- Real microphone capture on Ubuntu.
- YAMNet classification with TensorFlow installed.
- Whisper transcription with model dependencies installed.
- End-to-end real audio path.

## Commands

Install base requirements:

```bash
pip install -r requirements.txt
```

Optional AI/model dependencies:

```bash
pip install -r requirements-ai.txt
```

Run tests:

```bash
pytest -q
```

Run mock dashboard:

```bash
streamlit run app/dashboard.py
```

Run live integration mode:

```text
http://localhost:8501/?mode=live
```

## What Each Member Should Edit

Audio:

```text
modules/audio_input/
```

Visualization:

```text
modules/visualization/
```

Noise classification:

```text
modules/noise_classification/
```

Speech recognition:

```text
modules/speech_recognition/
```

GUI/integration:

```text
app/dashboard.py
app/live_pipeline.py
app/contracts.py
```
