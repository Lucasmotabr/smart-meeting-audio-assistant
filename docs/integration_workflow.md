# Team Integration Workflow

## Source Of Truth

Use one GitHub repository as the source of truth for Python code and
documentation.

Use Google Drive only for:

- large audio recordings
- trained model files
- datasets
- temporary submissions from teammates who cannot use Git yet

Do not maintain different copies of the Python application in Drive and
GitHub. Once a submitted file is accepted, put it in GitHub and treat the
Drive copy as an archive.

## What Each Teammate Must Submit

Every submission must contain:

1. One Python file for the assigned module.
2. A list of new packages and versions.
3. The exact command used to run the code.
4. A small test or sample terminal output.
5. Any required model or data file, with its Drive link.
6. The expected input and actual output of the main function.

Files belong here:

```text
modules/audio_input.py
modules/visualization.py
modules/noise_classification.py
modules/speech_recognition.py
```

The required function names and outputs are documented in `docs/handoff/`.

## GitHub Workflow

1. Create one repository and put the current project in its `main` branch.
2. Give each teammate a separate branch:
   - `feature/audio-input`
   - `feature/visualization`
   - `feature/noise-classification`
   - `feature/speech-recognition`
3. Each teammate changes only their module, dependency notes, and tests.
4. Open a pull request instead of sending replacement copies of the full app.
5. Run the module alone before connecting it to the dashboard.
6. Merge and integrate one module at a time.

If a teammate submits through Drive, the integration owner should create the
branch, copy only the relevant file into `modules/`, test it, and then open the
pull request.

## Recommended Integration Order

Integrate in dependency order:

```text
audio input
-> visualization
-> noise classification
-> speech recognition
-> quality estimation and advice
-> dashboard
```

Do not connect all submitted files at once. For each module:

1. Import the module without errors.
2. Run its main function with a known input.
3. Validate its output against `app/contracts.py`.
4. Add or update a focused test.
5. Connect it to the pipeline.
6. Keep mock mode available as a fallback.

## First Review For An Incoming File

Before copying a teammate file into the app, check:

- Does it start recording, downloading, or loading a model during import?
- Does it use hard-coded absolute paths?
- Does it require API keys?
- Does it return the documented fields?
- Are audio samples mono NumPy arrays?
- Is the sample rate included?
- Are confidence values between `0.0` and `1.0`?
- Does it use only the allowed noise labels?
- Does it add packages that conflict with the current environment?

Heavy work should happen inside functions or classes, not automatically when
the file is imported.

## Definition Of Done For One Module

A module is integrated only when:

- the standalone function works
- its output matches the shared contract
- its dependencies are recorded
- tests pass
- the dashboard can use it
- the dashboard can still run in mock mode if the module fails

