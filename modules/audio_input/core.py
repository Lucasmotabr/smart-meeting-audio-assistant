import sys
import time
import logging
import threading
import numpy as np

# Configure logger
logger = logging.getLogger("audio_input")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Global variables for state management
_impl = None
_os_impl = None
_initialized = False
_fallback_mode = False
_microphone_name = "Unknown Microphone"
_microphone_type = "Unknown"
_dependency_error = None
_selected_device_id = None
_selected_device_name = None
# The device the caller asked for, remembered even when initialization fails so a
# later retry can reacquire THAT device instead of silently taking the default.
# Both None means "no preference": the system default microphone is allowed.
_requested_device_id = None
_requested_device_name = None
_last_retry_at = 0.0
_audio_lock = threading.RLock()

PLATFORM = sys.platform


def _load_os_impl():
    global _os_impl, _dependency_error, _microphone_type, _fallback_mode

    if _os_impl is not None:
        return _os_impl

    if PLATFORM == "darwin":
        try:
            from . import macos as impl

            _os_impl = impl
            _dependency_error = None
            _microphone_type = "CoreAudio (macOS)"
            _fallback_mode = False
            return _os_impl
        except ImportError as exc:
            _dependency_error = exc
            _microphone_type = "Unavailable"
            return None

    if PLATFORM.startswith("linux"):
        try:
            from . import ubuntu as impl

            _os_impl = impl
            _dependency_error = None
            _microphone_type = "PulseAudio (Ubuntu)"
            _fallback_mode = False
            return _os_impl
        except ImportError as exc:
            _dependency_error = exc
            _microphone_type = "Unavailable"
            return None

    logger.warning(f"Unsupported platform: {PLATFORM}. Falling back to silence generator.")
    _microphone_type = "None"
    _fallback_mode = True
    return None


_load_os_impl()

def list_microphones():
    """
    Lists available microphone devices.
    Returns:
        list of dict: Each dict contains 'id' and 'name' of the device.
    """
    impl = _load_os_impl()
    if impl is None:
        return []
    try:
        return impl.list_microphones()
    except Exception as e:
        logger.error(f"Failed to list microphones: {e}")
        return []

def initialize_microphone(device_id=None, device_name=None):
    """
    Initializes the selected microphone device (only once).
    Args:
        device_id (int/str, optional): The device identifier.
        device_name (str, optional): The device name substring.
    Returns:
        bool: True if initialized successfully, False otherwise.
    """
    with _audio_lock:
        return _initialize_microphone_locked(device_id=device_id, device_name=device_name)


def _initialize_microphone_locked(device_id=None, device_name=None):
    global _impl, _initialized, _fallback_mode, _microphone_name, _selected_device_id, _selected_device_name
    global _requested_device_id, _requested_device_name

    # Record the request before attempting it, so a failed explicit request is
    # still remembered by the retry path. Calling with (None, None) is an
    # explicit "no preference" and clears any previously requested device.
    _requested_device_id = device_id
    _requested_device_name = device_name

    if _initialized:
        same_id = str(device_id) == str(_selected_device_id) if device_id is not None else _selected_device_id is None
        same_name = device_name == _selected_device_name
        if same_id and (device_name is None or same_name):
            return True
        reset_microphone()

    impl = _load_os_impl()

    if _fallback_mode or impl is None:
        if _dependency_error is not None:
            logger.warning(f"Audio dependency missing: {_dependency_error}. Initialization skipped.")
        else:
            logger.warning("Operating in fallback mode. Initialization skipped.")
        _microphone_name = "None (Fallback)"
        _initialized = True
        _fallback_mode = True
        return False

    try:
        # Ask implementation to initialize device and return selected name/id info
        success, active_name = impl.initialize_microphone(device_id, device_name)
        if success:
            _microphone_name = active_name
            _selected_device_id = device_id
            _selected_device_name = device_name
            _initialized = True
            _fallback_mode = False
            logger.info(f"Microphone initialized successfully: {active_name}")
            return True
        else:
            raise RuntimeError("OS implementation returned failure status on initialization.")
    except Exception as e:
        logger.error(f"Failed to initialize microphone: {e}. Switching to silent fallback mode.")
        _fallback_mode = True
        _microphone_name = "None (Fallback)"
        _initialized = True
        return False

def reset_microphone():
    global _initialized, _fallback_mode, _selected_device_id, _selected_device_name

    with _audio_lock:
        return _reset_microphone_locked()


def _reset_microphone_locked():
    global _initialized, _fallback_mode, _selected_device_id, _selected_device_name

    impl = _load_os_impl()
    if impl is not None and hasattr(impl, "close_stream"):
        try:
            impl.close_stream()
        except Exception as e:
            logger.warning(f"Failed to close microphone stream cleanly: {e}")

    _initialized = False
    _fallback_mode = False
    _selected_device_id = None
    _selected_device_name = None


def get_audio_frame(device_id=None, device_name=None):
    """
    Captures approximately one second of mono audio at 16000 Hz.
    Returns:
        dict: A dictionary containing the audio samples, sample rate, RMS, peak,
              timestamp, and microphone information.
    """
    with _audio_lock:
        return _get_audio_frame_locked(device_id=device_id, device_name=device_name)


def _get_audio_frame_locked(device_id=None, device_name=None):
    global _initialized, _fallback_mode, _last_retry_at

    # Auto-initialize with defaults if get_audio_frame is called without initialization
    selected_changed = (
        (
            device_id is not None
            and _selected_device_id is not None
            and str(device_id) != str(_selected_device_id)
        )
        or (
            device_name is not None
            and _selected_device_name is not None
            and device_name != _selected_device_name
        )
        or (
            device_name is not None
            and _selected_device_id is not None
            and _selected_device_name is None
        )
        or (
            device_id is not None
            and _selected_device_name is not None
            and _selected_device_id is None
        )
    )
    if selected_changed:
        reset_microphone()

    if _fallback_mode and time.time() - _last_retry_at >= 3.0:
        _last_retry_at = time.time()
        reset_microphone()

    if not _initialized:
        # An explicit call argument wins; otherwise reacquire the device that was
        # originally requested. Never silently substitute the default microphone
        # for an explicitly requested one that is missing.
        initialize_microphone(
            device_id=device_id if device_id is not None else _requested_device_id,
            device_name=device_name if device_name is not None else _requested_device_name,
        )

    timestamp = time.time()

    impl = _load_os_impl()

    if _fallback_mode or impl is None:
        return _generate_silent_frame(timestamp)

    try:
        # Call platform-specific audio reading logic
        raw_samples = impl.read_frame()
        
        # Verify shape and type
        if not isinstance(raw_samples, np.ndarray):
            raise TypeError("Expected NumPy array from audio implementation.")
        
        # Ensure it's 1D mono float32
        if raw_samples.ndim > 1:
            raw_samples = raw_samples.flatten()
        
        # Calculate audio metrics
        rms = float(np.sqrt(np.mean(raw_samples ** 2))) if len(raw_samples) > 0 else 0.0
        peak = float(np.max(np.abs(raw_samples))) if len(raw_samples) > 0 else 0.0

        return {
            "samples": raw_samples,
            "sample_rate": 16000,
            "rms": round(rms, 4),
            "peak": round(peak, 4),
            "timestamp_seconds": timestamp,
            "microphone_name": _microphone_name,
            "microphone_type": _microphone_type
        }

    except Exception as e:
        logger.error(f"Error reading audio frame: {e}. Falling back to silence.")
        _fallback_mode = True
        return _generate_silent_frame(timestamp)

def _generate_silent_frame(timestamp):
    """Generates a dummy silent frame when microphone fails or is disconnected."""
    # 1 second of silence at 16000 Hz
    silent_samples = np.zeros(16000, dtype=np.float32)
    return {
        "samples": silent_samples,
        "sample_rate": 16000,
        "rms": 0.0,
        "peak": 0.0,
        "timestamp_seconds": timestamp,
        "microphone_name": _microphone_name,
        "microphone_type": _microphone_type
    }
