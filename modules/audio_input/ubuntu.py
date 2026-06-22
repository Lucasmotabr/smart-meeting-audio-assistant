import ctypes
import subprocess
import numpy as np
import logging

logger = logging.getLogger("audio_input.ubuntu")

# Constants from PulseAudio header files
PA_STREAM_RECORD = 2
PA_SAMPLE_S16LE = 3  # 16-bit signed PCM little endian

# Define PulseAudio structures
class pa_sample_spec(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_int),       # pa_sample_format_t
        ("rate", ctypes.c_uint32),      # uint32_t
        ("channels", ctypes.c_uint8)    # uint8_t
    ]

# Try loading libpulse-simple
libpulse = None
for lib_name in ["libpulse-simple.so.0", "libpulse-simple.so", "libpulse-simple.so.0.1.1"]:
    try:
        libpulse = ctypes.CDLL(lib_name)
        break
    except OSError:
        continue

if libpulse is None:
    raise ImportError("Could not locate libpulse-simple on this system. Install with: sudo apt install libpulse0")

# Define function signatures if library is loaded
if libpulse is not None:
    libpulse.pa_simple_new.restype = ctypes.c_void_p
    libpulse.pa_simple_new.argtypes = [
        ctypes.c_char_p,                 # server
        ctypes.c_char_p,                 # name
        ctypes.c_int,                    # dir
        ctypes.c_char_p,                 # dev
        ctypes.c_char_p,                 # stream_name
        ctypes.POINTER(pa_sample_spec),  # ss
        ctypes.c_void_p,                 # map
        ctypes.c_void_p,                 # attr
        ctypes.POINTER(ctypes.c_int)     # error
    ]

    libpulse.pa_simple_read.restype = ctypes.c_int
    libpulse.pa_simple_read.argtypes = [
        ctypes.c_void_p,                 # s
        ctypes.c_void_p,                 # data
        ctypes.c_size_t,                 # bytes
        ctypes.POINTER(ctypes.c_int)     # error
    ]

    libpulse.pa_simple_free.restype = None
    libpulse.pa_simple_free.argtypes = [ctypes.c_void_p]

    libpulse.pa_strerror.restype = ctypes.c_char_p
    libpulse.pa_strerror.argtypes = [ctypes.c_int]

# Global stream state
_simple_handle = None
_active_device_name = None

def list_microphones():
    """Lists available input sources from pactl, excluding monitor (loopback) sources."""
    devices = []
    try:
        res = subprocess.run(
            ["pactl", "list", "sources"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            env={**__import__("os").environ, "LANG": "C"},
        )
        current_name = None
        current_desc = None
        for line in res.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Name:"):
                current_name = stripped.split(":", 1)[1].strip()
                current_desc = None
            elif stripped.startswith("Description:"):
                current_desc = stripped.split(":", 1)[1].strip()
                if current_name and not current_name.endswith(".monitor"):
                    devices.append({
                        "id": current_name,
                        "name": current_desc or current_name,
                    })
                current_name = None
                current_desc = None
    except Exception as e:
        logger.warning(f"Failed to list microphones via pactl: {e}")

    if not devices:
        devices.append({"id": "default", "name": "Default PulseAudio Source"})

    return devices

def initialize_microphone(device_id=None, device_name=None):
    """Initializes the PulseAudio simple API stream."""
    global _simple_handle, _active_device_name
    
    devices = list_microphones()
    selected_dev = None

    if device_id is not None or device_name is not None:
        if device_name is not None:
            # Match by substring
            for dev in devices:
                if device_name.lower() in dev["name"].lower():
                    selected_dev = dev["id"]
                    break
        
        if selected_dev is None and device_id is not None:
            # Match by exact ID
            for dev in devices:
                if dev["id"] == device_id:
                    selected_dev = device_id
                    break
        
        if selected_dev is None:
            raise RuntimeError(f"Requested microphone (ID={device_id}, Name={device_name}) not found.")
    else:
        # Default device
        selected_dev = "default"

    # Convert select_dev string to bytes, or None for default
    if selected_dev == "default":
        selected_dev = None

    dev_bytes = selected_dev.encode('utf-8') if selected_dev else None
    
    # Define sample specification (16000Hz, mono, S16LE)
    spec = pa_sample_spec()
    spec.format = PA_SAMPLE_S16LE
    spec.rate = 16000
    spec.channels = 1

    err = ctypes.c_int(0)
    logger.info(f"Opening PulseAudio stream on device: {selected_dev or 'default'}")
    
    _simple_handle = libpulse.pa_simple_new(
        None,                           # Use default server
        b"audio_input_ubuntu",          # Application name
        PA_STREAM_RECORD,               # Stream direction
        dev_bytes,                      # Source name
        b"microphone_record",           # Stream description
        ctypes.byref(spec),             # Sample specification
        None,                           # Default channel map
        None,                           # Default buffer attributes
        ctypes.byref(err)               # Error code container
    )

    if not _simple_handle:
        err_msg = libpulse.pa_strerror(err.value).decode('utf-8')
        raise RuntimeError(f"pa_simple_new failed: {err_msg}")

    _active_device_name = selected_dev if selected_dev else "Default PulseAudio Source"
    return True, _active_device_name

def read_frame():
    """Reads exactly 1 second (16000 samples) of audio from PulseAudio."""
    global _simple_handle
    if _simple_handle is None:
        raise RuntimeError("PulseAudio stream not initialized.")

    # 16000 samples of 16-bit PCM (S16LE) = 32000 bytes
    num_samples = 16000
    sample_size = 2 # 2 bytes for 16-bit
    buffer_size = num_samples * sample_size
    
    buffer = ctypes.create_string_buffer(buffer_size)
    err = ctypes.c_int(0)
    
    ret = libpulse.pa_simple_read(_simple_handle, buffer, buffer_size, ctypes.byref(err))
    if ret < 0:
        err_msg = libpulse.pa_strerror(err.value).decode('utf-8')
        raise RuntimeError(f"pa_simple_read failed: {err_msg}")

    # Convert raw bytes buffer to numpy int16 array
    samples_int16 = np.frombuffer(buffer.raw, dtype=np.int16)
    
    # Normalize to float32 between -1.0 and 1.0
    samples_float32 = samples_int16.astype(np.float32) / 32768.0
    
    return samples_float32

def close_stream():
    """Cleanup connection on destruction."""
    global _simple_handle
    if _simple_handle is not None and libpulse is not None:
        try:
            libpulse.pa_simple_free(_simple_handle)
            _simple_handle = None
        except Exception:
            pass

