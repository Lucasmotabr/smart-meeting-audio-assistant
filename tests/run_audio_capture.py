import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.audio_input as ai

def test_microphone_listing():
    print("=== Testing list_microphones ===")
    mics = ai.list_microphones()
    print(f"Found {len(mics)} microphone(s):")
    for mic in mics:
        print(f"  ID: {mic['id']} | Name: {mic['name']}")
    return mics

def test_recording_flow(device_id=None, device_name=None):
    print(f"\n=== Testing Recording Flow for Device: {device_id or 'default'} ===")
    
    # Reload modules to reset initialization state
    import importlib
    importlib.reload(ai)
    
    # Initialize microphone
    success = ai.initialize_microphone(device_id=device_id, device_name=device_name)
    print(f"Initialization success: {success}")
    if not success:
        print("Skipping recording since initialization failed (fallback mode active).")
        return
        
    print("Capturing 3 frames (approx 1 second each)...")
    for i in range(1, 4):
        start_time = time.time()
        frame = ai.get_audio_frame()
        elapsed = time.time() - start_time
        
        # Validate format
        assert isinstance(frame, dict), "Frame should be a dictionary"
        assert "samples" in frame, "Frame missing 'samples'"
        assert isinstance(frame["samples"], np.ndarray), "samples must be a NumPy array"
        assert frame["samples"].dtype == np.float32, "samples must be float32"
        assert frame["sample_rate"] == 16000, "sample rate must be 16000"
        assert len(frame["samples"]) == 16000, "samples array length must be 16000"
        assert -1.01 <= np.min(frame["samples"]) <= 1.01, "samples must be normalized between -1.0 and 1.0"
        assert -1.01 <= np.max(frame["samples"]) <= 1.01, "samples must be normalized between -1.0 and 1.0"
        
        print(f"Frame {i}:")
        print(f"  Samples length: {len(frame['samples'])} | Dtype: {frame['samples'].dtype}")
        print(f"  RMS: {frame['rms']:.6f} | Peak: {frame['peak']:.6f}")
        print(f"  Timestamp: {frame['timestamp_seconds']:.2f} | Captured in: {elapsed:.2f}s")
        print(f"  Microphone: {frame['microphone_name']} ({frame['microphone_type']})")
        
        zero_count = np.sum(frame["samples"] == 0.0)
        if zero_count == 16000:
            print("  Status: Silence (or fallback mode active)")
        else:
            print(f"  Status: Active Audio (zeros: {zero_count}/16000)")

if __name__ == "__main__":
    mics = test_microphone_listing()
    
    # 1. Test Default recording
    test_recording_flow()
    
    # 2. Test specific microphones from listing if available
    for mic in mics:
        if mic["id"] != "default":
            test_recording_flow(device_id=mic["id"])
            
    print("\nAll integration capture tests completed successfully!")
