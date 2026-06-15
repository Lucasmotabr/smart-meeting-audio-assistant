import whisper

model = whisper.load_model("base")

result = model.transcribe("sample.wav.m4a")

print("Transcript:")
print(result["text"])