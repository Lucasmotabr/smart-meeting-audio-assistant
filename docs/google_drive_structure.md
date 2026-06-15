# Google Drive Structure For The Team

Use a simple numbered structure so nobody has to guess where things go.

```text
Smart Meeting Audio Assistant/
  00_READ_ME_FIRST/
  01_Project_Brief/
  02_Team_Handoffs/
  03_Member_Work/
  04_GUI_Design/
  05_Demo_Materials/
  06_Final_Presentation/
```

## 00_READ_ME_FIRST

Put only the most important files here.

Recommended files:

- `Start Here - Team Instructions`
- `Weekly Checklist`
- `How To Submit Your Part`

Suggested text for `Start Here - Team Instructions`:

```text
Please start here.

You do not need to understand the whole project yet.
Each person has one small module.

Step 1: Open your handoff file in 02_Team_Handoffs.
Step 2: Build only the function described in your handoff.
Step 3: Put your code/output in your folder inside 03_Member_Work.
Step 4: Send proof that it works. Terminal output is enough for Week 1.

Week 1 goal: every module works alone.
Week 2 goal: connect modules into the GUI.
Week 3 goal: polish and prepare final demo.
```

Suggested text for `Weekly Checklist`:

```text
WEEK 1 CHECKLIST

Goal for next week:
Each person must have their module working by itself.
It does not need to be connected to the GUI yet.

Required by next week:
1. Python file or notebook for your module
2. Command or steps to run it
3. Terminal output, screenshot, or short video proving it works
4. List of libraries you installed
5. Problems/questions if something is blocked

Do not spend too much time making things beautiful in Week 1.
The important thing is that each module produces the correct output.
```

Suggested text for `How To Submit Your Part`:

```text
HOW TO SUBMIT YOUR PART

Upload your work into your folder in 03_Member_Work.

Inside your folder, add:
1. your code file
2. a short note explaining how to run it
3. proof that it works
4. any errors or problems

File naming:
member_1_audio_input.py
member_2_visualization.py
member_3_noise_classification.py
member_4_speech_recognition.py
member_5_gui_llm.py

If your part is not finished, upload what you have and explain what is missing.
```

## 01_Project_Brief

Put the full project explanation here.

Recommended files:

- original class project description
- system pipeline image or diagram
- `architecture.md`
- final feature list

This folder is for context. Teammates should not start here if they are confused.

Suggested Google Doc title:

```text
Project Brief - Smart Meeting Audio Assistant
```

Suggested Google Doc content:

```text
PROJECT GOAL

We are building a real-time microphone-based dashboard.
The system will show sound activity, waveform, spectrogram, noise classification, speech transcription, audio quality, and AI advice.

FINAL PIPELINE

Microphone Input
-> Audio Stream
-> Voice Activity / Waveform / Spectrogram
-> Noise Classification
-> Speech Recognition
-> Audio Quality
-> AI Advice
-> GUI Dashboard

WEEK 1 GOAL

Each module works separately.
No full integration is required yet.

WEEK 2 GOAL

Connect all modules into the GUI.

WEEK 3 GOAL

Polish the demo and prepare the final presentation.
```

## 02_Team_Handoffs

Put one short file per teammate.

Recommended files:

- `Member 1 - Audio Input`
- `Member 2 - Visualization`
- `Member 3 - Noise Classification`
- `Member 4 - Speech Recognition`
- `Member 5 - GUI and LLM Advice`

Each handoff should include:

- goal
- what file/function to create
- expected input
- expected output
- recommended libraries
- how to prove it works

Suggested Google Doc titles:

```text
Member 1 - Audio Input
Member 2 - Visualization
Member 3 - Noise Classification
Member 4 - Speech Recognition
Member 5 - GUI and LLM Advice
```

Week 1 rule:

```text
Each person only needs to make their own module work alone.
They do not need to connect to the full GUI yet.
```

## 03_Member_Work

Create one folder per person.

```text
03_Member_Work/
  Member_1_Audio_Input/
    code/
    screenshots_or_videos/
    notes/
  Member_2_Visualization/
    code/
    screenshots_or_videos/
    notes/
  Member_3_Noise_Classification/
    code/
    screenshots_or_videos/
    notes/
  Member_4_Speech_Recognition/
    code/
    screenshots_or_videos/
    notes/
  Member_5_GUI_LLM/
    code/
    screenshots_or_videos/
    notes/
```

Ask everyone to upload:

- their Python file
- a `README` or note with the command to run it
- terminal output, screenshot, or video proof
- install commands they used
- any errors or problems

## 04_GUI_Design

Put visual references and dashboard planning here.

Recommended files:

- GUI screenshot reference
- `gui_target.md`
- color palette
- layout sketch
- dashboard component checklist

Suggested checklist:

- sidebar
- microphone selector
- recording status
- voice activity bars
- waveform
- spectrogram
- sound classification
- transcript
- audio quality
- AI advice
- bottom status bar

Week 1 GUI target:

```text
The GUI should run with mock data.
It does not need real microphone/classifier/transcription data yet.

Required panels:
- voice activity
- waveform
- spectrogram
- classification
- transcript
- audio quality
- AI advice

Nice if possible:
- dark theme
- purple accent color
- scenario selector
```

## 05_Demo_Materials

Use this folder for demo evidence and test scenarios.

Recommended folders:

```text
05_Demo_Materials/
  clean_speech/
  keyboard_typing_noise/
  far_distance_speech/
  microphone_comparison/
```

Each scenario can contain:

- sample audio
- terminal output or screenshots
- expected result
- notes for the final presentation

For Week 1, this folder can stay simple. Add materials only if someone already has them.

## 06_Final_Presentation

Use this only for final slides and speaking notes.

Recommended files:

- final slides
- demo script
- speaking order
- final checklist

For Week 1, this folder can contain only one planning document:

```text
Presentation Ideas - Draft
```

Suggested content:

```text
Possible demo story:
1. Show clean speech
2. Add keyboard noise
3. Move far from microphone
4. Compare two microphones
5. Show how AI advice explains the problem
```

## Simple Rules For The Team

Use these rules to keep the Drive clean:

- Do not upload random files to the main folder.
- Put code in your own `code/` folder.
- Put proof screenshots/videos in `screenshots_or_videos/`.
- Put problems/questions in `notes/`.
- Name files with your member number, for example `member_1_audio_input.py`.
- If something does not work, still upload it with notes explaining the problem.

## Weekly Checkpoint Template

Ask each person to fill this out before the weekly meeting:

```text
Name:
Module:

What works:

What does not work yet:

Command to run:

Libraries installed:

Output format:

Screenshot/video uploaded:
Yes/No

Help needed:
```

## Google Docs To Create

If you are creating one Google Doc inside each main folder, create these:

```text
00_READ_ME_FIRST/
  Start Here - Team Instructions
  Weekly Checklist
  How To Submit Your Part

01_Project_Brief/
  Project Brief - Smart Meeting Audio Assistant

02_Team_Handoffs/
  Member 1 - Audio Input
  Member 2 - Visualization
  Member 3 - Noise Classification
  Member 4 - Speech Recognition
  Member 5 - GUI and LLM Advice

03_Member_Work/
  Weekly Checkpoint Template

04_GUI_Design/
  GUI Target Layout

05_Demo_Materials/
  Demo Scenario Notes

06_Final_Presentation/
  Presentation Ideas - Draft
```

Week 1 focus:

```text
Do not overbuild the Drive.
The priority is that each person understands their task and has a place to upload their work.
```
