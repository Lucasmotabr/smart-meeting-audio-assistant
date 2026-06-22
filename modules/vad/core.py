from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

_model: Any = None
_utils: Any = None

_SUPPORTED_RATES = {8000, 16000}
# v6: 16kHz=512サンプル固定、8kHz=256サンプル固定
_CHUNK_SIZES = {16000: 512, 8000: 256}


def detect_voice_activity(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """音声フレームを Silero VAD で推論し {"is_speech": bool, "probability": float} を返す。

    silero-vad 未インストール時、またはサポート外サンプルレートの場合は
    {"is_speech": False, "probability": 0.0} を返す。
    """
    if sample_rate not in _SUPPORTED_RATES:
        return {"is_speech": False, "probability": 0.0}

    model, utils = _load_model()
    if model is None:
        return {"is_speech": False, "probability": 0.0}

    try:
        import torch

        audio = torch.tensor(samples, dtype=torch.float32)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        prob = _infer_max_probability(model, audio, sample_rate)
        return {"is_speech": bool(prob >= 0.5), "probability": float(prob)}
    except Exception:
        return {"is_speech": False, "probability": 0.0}


def _infer_max_probability(model: Any, audio: Any, sample_rate: int) -> float:
    """フレームをチャンク分割して推論し、最大確率を返す。"""
    import torch

    chunk_size = _CHUNK_SIZES[sample_rate]
    n = audio.shape[-1]
    if n < chunk_size:
        return 0.0

    # Silero VAD は状態を持つ RNN。連続呼び出しで隠れ状態が累積し確率が
    # 減衰するため、各フレーム処理の冒頭で状態をリセットして呼び出しを独立させる。
    if hasattr(model, "reset_states"):
        model.reset_states()

    probs: list[float] = []
    for start in range(0, n - chunk_size + 1, chunk_size):
        chunk = audio[:, start : start + chunk_size]
        with torch.no_grad():
            p = float(model(chunk, sample_rate).item())
        probs.append(p)

    return max(probs) if probs else 0.0


def _load_model() -> tuple[Any, Any]:
    global _model, _utils

    if _model is not None:
        return _model, _utils

    if importlib.util.find_spec("silero_vad") is None:
        return None, None

    try:
        from silero_vad import load_silero_vad, get_speech_timestamps  # noqa: F401

        _model = load_silero_vad()
        _utils = get_speech_timestamps
        return _model, _utils
    except Exception:
        return None, None
