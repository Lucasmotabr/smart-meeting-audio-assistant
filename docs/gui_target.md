# GUI Target From Reference Image

The target GUI is a dark dashboard with purple accents.

## Main Layout

Left sidebar:

- app name: SMAA
- navigation: Dashboard, History, Microphones, Settings, About
- microphone list
- recording status
- system status: CPU, memory, FPS

Top header:

- title: Smart Meeting Audio Assistant
- subtitle: Real-time Sound Analysis, Speech Recognition & AI Advice
- quick metrics:
  - Audio Quality
  - Signal Strength
  - Latency

Main panels:

- Voice Activity bars
- Waveform
- Spectrogram
- Sound Classification
- Transcription
- Audio Quality score
- AI Advice

Bottom status bar:

- system operational indicator
- sample rate
- chunk size
- model status
- current time/date

## Visual Style

- Background: near black
- Cards: dark gray with subtle borders
- Accent: purple
- Good states: green
- Warnings: orange or yellow
- Errors: red
- Font: clean sans-serif

## Week 1 GUI Priority

First priority:

- same layout sections
- mock data
- readable dashboard
- scenario selector

Second priority:

- nicer colors
- sidebar polish
- circular quality score
- smoother real-time updates

Third priority:

- exact screenshot match
- history/settings/about pages
- real CPU/memory metrics

## Streamlit Notes

Streamlit is good for a fast class prototype. It will not perfectly match the screenshot by default, but custom CSS can get close enough.

Recommended implementation order:

1. Build all sections with simple Streamlit components.
2. Add mock scenario data.
3. Add CSS for dark cards and purple highlights.
4. Replace mock data with real teammate modules.
5. Polish spacing, colors, and final demo scenarios.

