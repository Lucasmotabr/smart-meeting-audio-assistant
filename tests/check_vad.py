"""話しながら実行して VAD 確率をリアルタイム確認するスクリプト。"""
import sys
sys.path.insert(0, ".")

import sounddevice as sd
import numpy as np
from modules.vad.core import detect_voice_activity, _load_model

SR = 16000
model, _ = _load_model()

print("マイクに向かって話してください。Ctrl+C で終了。\n")
try:
    while True:
        samples = sd.rec(SR, samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        samples = samples.flatten()
        rms = float(np.sqrt(np.mean(samples ** 2)))

        if model is not None:
            model.reset_states()

        result = detect_voice_activity(samples, SR)
        prob = result["probability"]
        bar = "█" * int(prob * 30)
        flag = "SPEECH" if result["is_speech"] else "silent"
        print(f"  rms={rms:.4f}  prob={prob:.3f}  {flag:6s}  |{bar:<30}|")
except KeyboardInterrupt:
    print("\n終了")
