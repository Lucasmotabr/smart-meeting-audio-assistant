# GUI Refactor Plan

Current status: the live dashboard works as a Streamlit-rendered HTML shell, but the visualization is still updated by rerunning Python and replacing rendered markup. That is acceptable for the current demo, but it will not produce the same smooth motion as the mock design.

## Why Refactor

- The current dashboard is one large HTML string in `app/dashboard.py`.
- Streamlit reruns the script to update live data.
- Waveform, voice bars, and spectrogram are recreated as full HTML/SVG/PNG output.
- Real microphone audio is noisy and asynchronous, so it needs smoothing and browser-side animation.

## Target Architecture

Python should produce data. The browser should animate visuals.

Python responsibilities:

- list available microphones
- keep the selected microphone stream open
- produce the latest audio frame
- compute waveform points
- compute spectrogram columns
- run VAD/noise/transcription/advice modules
- send compact JSON-style state to the frontend

Frontend responsibilities:

- keep the dashboard layout mounted
- animate voice bars
- draw waveform on a canvas
- append spectrogram columns to a rolling canvas
- update text cards at a slower cadence
- avoid full-page visual replacement

## Refactor Order

1. Keep the current dashboard as the baseline demo.
2. Move live audio state into a small runtime object so it is independent from Streamlit reruns.
3. Split dashboard rendering into component-level functions:
   - sidebar
   - top metrics
   - voice activity
   - waveform
   - spectrogram
   - classification
   - transcription
   - advice
4. Replace SVG/PNG visual panels with browser-side canvas panels.
5. Update canvas panels from compact data instead of replacing whole HTML.
6. Add smoothing:
   - waveform exponential smoothing
   - spectrogram rolling buffer
   - VAD probability smoothing
   - classification confidence smoothing
7. Keep text-heavy panels on slower updates than audio visuals.

## Teammate Boundaries

Audio input:

- provide continuous microphone frames for macOS and Ubuntu
- keep selected device open until the user changes microphones
- expose `list_microphones()` and `get_audio_frame(device_id=...)`

Visualization:

- provide clean waveform points and spectrogram columns
- avoid returning NaN/Inf values
- document expected input sample rate and output shapes

VAD/noise classification:

- return probabilities that can be smoothed by the GUI
- do not block the UI loop for long model calls

GUI:

- keep current Streamlit dashboard as the fallback
- build the next smooth version as a custom component or canvas-based frontend
- do not require teammates to edit the whole dashboard file for module-specific changes

## Definition Of Done

- Microphone stays open while the dashboard runs.
- Switching microphones resets only the audio stream and live buffers.
- Waveform updates smoothly without replacing the full page.
- Spectrogram scrolls left/right continuously instead of jumping each second.
- Voice bars visibly react to live input and get stronger when VAD detects speech.
- The mock/demo data is no longer required for the main presentation flow.
