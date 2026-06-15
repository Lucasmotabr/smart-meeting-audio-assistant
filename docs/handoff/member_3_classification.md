# Member 3 Handoff: Noise Classification

## Your Goal

Detect the current sound environment.

## What You Need To Build

Create a Python file:

```text
modules/noise_classification.py
```

Main function:

```python
def classify_noise(samples, sample_rate):
    ...
```

## Expected Output

Return:

```python
{
    "label": "speech",
    "confidence": 0.91,
}
```

Allowed labels:

- `speech`
- `typing`
- `clap`
- `silence`
- `background noise`
- `unknown`

## How This Connects To The GUI

For Week 1, returning the dictionary above is enough.

For integration, this output will be converted into `NoiseClassification` in:

```text
app/contracts.py
```

Important fields for the GUI:

- `label`
- `confidence`

Please use only the allowed labels above. If the model gives a different label, convert it to the closest allowed label.

## Recommended Libraries

- `tensorflow`
- `tensorflow_hub`
- YAMNet
- `librosa`

## How To Prove It Works

Send:

- the Python file
- sample output for at least two sounds
- the command to run it

Minimum success:

- classify silence or speech
- print a label and confidence score
