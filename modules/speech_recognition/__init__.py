def transcribe_audio(samples, sample_rate):
    from .core import transcribe_audio as _transcribe_audio

    return _transcribe_audio(samples, sample_rate)

__all__ = ["transcribe_audio"]
