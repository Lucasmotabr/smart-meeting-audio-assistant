import sounddevice as sd
import numpy as np
import logging

logger = logging.getLogger("audio_input.macos")

# Global stream state
_stream = None
_active_device_name = None

def list_microphones():
    """Lists available input devices."""
    devices = sd.query_devices()
    input_devices = []
    for idx, dev in enumerate(devices):
        if dev.get('max_input_channels', 0) > 0:
            input_devices.append({
                "id": idx,
                "name": dev.get('name', f"Device {idx}")
            })
    return input_devices

def initialize_microphone(device_id=None, device_name=None):
    """Initializes and starts the InputStream on macOS."""
    global _stream, _active_device_name
    
    devices = list_microphones()
    selected_id = None
    selected_name = None

    if device_id is not None or device_name is not None:
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
                if dev["id"] == device_id:
                    selected_id = device_id
                    selected_name = dev["name"]
                    break
        
        if selected_id is None:
            raise RuntimeError(f"Requested microphone (ID={device_id}, Name={device_name}) not found.")
    else:
        # Fallback to sounddevice default input device
        default_device_idx = sd.default.device[0]
        if default_device_idx >= 0:
            selected_id = default_device_idx
            # Find name
            for dev in devices:
                if dev["id"] == default_device_idx:
                    selected_name = dev["name"]
                    break
            if selected_name is None:
                selected_name = f"Default Device ({default_device_idx})"
        else:
            raise RuntimeError("No default input device found on the system.")

    logger.info(f"Opening InputStream on device index {selected_id}: {selected_name}")
    
    # Open the input stream with 16000Hz, mono, float32, blocksize of 1 second (16000 samples)
    # Using blocksize=16000 encourages read() to yield chunks of exactly that size.
    _stream = sd.InputStream(
        device=selected_id,
        samplerate=16000,
        channels=1,
        dtype='float32',
        blocksize=16000
    )
    _stream.start()
    _active_device_name = selected_name
    
    return True, _active_device_name

def read_frame():
    """Reads one second of audio samples (16000 samples) from the active stream."""
    global _stream
    if _stream is None:
        raise RuntimeError("InputStream is not initialized. Call initialize_microphone first.")
    
    # Read exactly 16000 samples. This blocks until 16000 samples are available.
    data, overflowed = _stream.read(16000)
    if overflowed:
        logger.warning("Audio input overflowed.")
        
    # sounddevice returns a 2D array of shape (samples, channels). We flatten it to 1D.
    return data.flatten()

def __del__():
    """Ensure cleanup of the stream on destruction."""
    global _stream
    if _stream is not None:
        try:
            _stream.stop()
            _stream.close()
        except Exception:
            pass
