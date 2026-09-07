import sounddevice as sd
import numpy as np
import logging
import time

logger = logging.getLogger("audio_input.macos")

# Global stream state
_stream = None
_active_device_name = None
_audio_buffer = None
_buffer_write_index = 0
_buffer_filled = False
_last_overflow_log = 0.0


SAMPLE_RATE = 16000
FRAME_SAMPLES = 16000
CALLBACK_BLOCKSIZE = 1024

def list_microphones():
    """Lists available input devices."""
    devices = sd.query_devices()
    input_devices = []
    for idx, dev in enumerate(devices):
        if dev.get('max_input_channels', 0) > 0:
            input_devices.append({
                "id": idx,
                "name": dev.get('name', f"Device {idx}"),
                "type": "CoreAudio input",
            })
    return input_devices

def initialize_microphone(device_id=None, device_name=None):
    """Initializes and starts the InputStream on macOS."""
    global _stream, _active_device_name, _audio_buffer, _buffer_write_index, _buffer_filled

    # Never overwrite a live stream. An orphaned-but-active PortAudio stream is
    # torn down later by the garbage collector, which corrupts PortAudio's
    # global state and segfaults the next call into it.
    close_stream()

    devices = list_microphones()
    selected_id = None
    selected_name = None

    if device_id is not None or device_name is not None:
        requested_id = str(device_id) if device_id is not None else None
        if device_name is not None:
            # Search by substring
            for dev in devices:
                if device_name.lower() in dev["name"].lower():
                    selected_id = dev["id"]
                    selected_name = dev["name"]
                    break

        if selected_id is None and device_id is not None:
            # Validate ID
            for dev in devices:
                if str(dev["id"]) == requested_id:
                    selected_id = dev["id"]
                    selected_name = dev["name"]
                    break
        
        if selected_id is None:
            # An explicitly requested device is never silently substituted:
            # the caller enters silent fallback and keeps retrying, so the
            # requested microphone is picked up if it appears later.
            raise RuntimeError(
                f"Requested microphone (ID={device_id}, Name={device_name}) not found."
            )
    else:
        selected_id, selected_name = _default_input_device(devices)

    logger.info(f"Opening InputStream on device index {selected_id}: {selected_name}")

    _audio_buffer = np.zeros(FRAME_SAMPLES * 3, dtype=np.float32)
    _buffer_write_index = 0
    _buffer_filled = False

    _stream = sd.InputStream(
        device=selected_id,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=CALLBACK_BLOCKSIZE,
        callback=_audio_callback,
    )
    _stream.start()
    _active_device_name = selected_name
    
    return True, _active_device_name


def _default_input_device(devices):
    default_device_idx = sd.default.device[0]
    if default_device_idx >= 0:
        for dev in devices:
            if dev["id"] == default_device_idx:
                return default_device_idx, dev["name"]
        return default_device_idx, f"Default Device ({default_device_idx})"
    if devices:
        return devices[0]["id"], devices[0]["name"]
    raise RuntimeError("No input microphone devices found on the system.")


def _audio_callback(indata, frames, time_info, status):
    """Continuously copy incoming microphone samples into a small ring buffer."""
    global _audio_buffer, _buffer_write_index, _buffer_filled, _last_overflow_log

    if status.input_overflow:
        now = time.time()
        if now - _last_overflow_log > 5.0:
            logger.warning("Audio input overflowed.")
            _last_overflow_log = now

    if _audio_buffer is None:
        return

    samples = np.asarray(indata, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return

    if samples.size >= _audio_buffer.size:
        _audio_buffer[:] = samples[-_audio_buffer.size :]
        _buffer_write_index = 0
        _buffer_filled = True
        return

    end = _buffer_write_index + samples.size
    if end <= _audio_buffer.size:
        _audio_buffer[_buffer_write_index:end] = samples
    else:
        split = _audio_buffer.size - _buffer_write_index
        _audio_buffer[_buffer_write_index:] = samples[:split]
        _audio_buffer[: end % _audio_buffer.size] = samples[split:]
        _buffer_filled = True

    _buffer_write_index = end % _audio_buffer.size
    if _buffer_write_index == 0:
        _buffer_filled = True


def read_frame():
    """Return the latest one second of microphone samples."""
    global _stream, _audio_buffer, _buffer_write_index, _buffer_filled
    if _stream is None:
        raise RuntimeError("InputStream is not initialized. Call initialize_microphone first.")

    if _audio_buffer is None:
        return np.zeros(FRAME_SAMPLES, dtype=np.float32)

    if not _buffer_filled and _buffer_write_index < FRAME_SAMPLES:
        frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        if _buffer_write_index > 0:
            frame[-_buffer_write_index:] = _audio_buffer[:_buffer_write_index]
        return frame

    start = (_buffer_write_index - FRAME_SAMPLES) % _audio_buffer.size
    if start < _buffer_write_index:
        return _audio_buffer[start:_buffer_write_index].copy()
    return np.concatenate((_audio_buffer[start:], _audio_buffer[:_buffer_write_index])).astype(np.float32)

def close_stream():
    """Close the active input stream, if one exists."""
    global _stream, _audio_buffer, _buffer_write_index, _buffer_filled
    if _stream is not None:
        try:
            _stream.stop()
            _stream.close()
        except Exception as exc:
            # Keep the reference. Dropping it here would orphan a still-live
            # native stream and hand its teardown to the garbage collector.
            # Propagate instead so the caller can enter fallback safely.
            logger.warning("Failed to close microphone stream cleanly: %s", exc)
            raise
        _stream = None
    _audio_buffer = None
    _buffer_write_index = 0
    _buffer_filled = False
