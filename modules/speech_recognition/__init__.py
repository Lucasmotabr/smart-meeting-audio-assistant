def transcribe_audio(samples, sample_rate, language=None, model_name=None):
    from .core import transcribe_audio as _transcribe_audio

    return _transcribe_audio(samples, sample_rate, language=language, model_name=model_name)

__all__ = ["transcribe_audio"]
