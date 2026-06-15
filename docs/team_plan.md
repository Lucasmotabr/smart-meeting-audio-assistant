# Team Plan

## Your GUI Responsibilities

For next week, your job is not only visual design. Your job is to make the integration surface ready.

Deliver:

- runnable Streamlit dashboard
- mock data mode
- dashboard layout matching the final project plan
- shared data contracts
- clear instructions for what each teammate must return
- placeholder advice system

## Week 1 Checklist

- [x] Create repo skeleton
- [x] Create GUI dashboard with placeholder data
- [x] Define shared data objects
- [x] Add adapter interfaces for each teammate module
- [ ] Improve visual style with CSS
- [ ] Add real microphone selection placeholder
- [ ] Record a short demo video/GIF of mock dashboard

## What To Ask Each Teammate For

Give each person only their file from `docs/handoff/`. Avoid sending the whole repo as their first instruction. The repo can be confusing at first; the handoff files are the simple contract.

Member 1, audio:

- Function name: `get_audio_frame()`
- Must return mono NumPy samples
- Use sample rate 16000 if possible
- Include microphone name

Member 2, visualization:

- Function name: `build_visualization(samples, sample_rate)`
- Must return waveform, spectrogram, and voice bars
- Spectrogram can be a 2D NumPy array

Member 3, classification:

- Function name: `classify_noise(samples, sample_rate)`
- Must return label and confidence
- Labels should match `NoiseLabel`

Member 4, speech recognition:

- Function name: `transcribe_audio(samples, sample_rate)`
- Must return transcript text, partial text, and latency

Member 5 / GUI:

- Function name: `generate_advice(snapshot)`
- Must return one short practical recommendation
- Can start rule-based, then connect to OpenAI/Groq/Ollama later

## Demo Scenarios

Prepare buttons or sidebar choices for:

- Clean speech
- Keyboard typing noise
- Far distance speech
- Silence / idle room
- Microphone comparison

These scenarios help the final presentation even if the real-time system has bugs.

## How To Coordinate The Team

Use this message in your group chat:

```text
Everyone has one small module. Please do not worry about the whole app yet.

Your job is to make one Python function that returns the output described in your handoff file.
For Week 1, it is OK if your module runs alone in a separate script.
For Week 2, Lucas/the GUI side will connect the functions into the dashboard.

Please send:
1. your Python file
2. the command to run it
3. a screenshot or terminal output showing it works
4. any installation commands you used
```

Recommended file ownership:

- `modules/audio_input.py` for Member 1
- `modules/visualization.py` for Member 2
- `modules/noise_classification.py` for Member 3
- `modules/speech_recognition.py` for Member 4
- `app/dashboard.py`, `app/mock_data.py`, and `app/contracts.py` for GUI/integration

The main rule: everyone should return simple Python data first. Beautiful code can come later.
