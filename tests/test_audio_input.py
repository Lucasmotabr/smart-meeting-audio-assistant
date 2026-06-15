import unittest
import numpy as np
import sys
import os
import importlib

# Add project root to path to resolve modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAudioInputModule(unittest.TestCase):
    def setUp(self):
        # Force reload of the audio_input module to reset global state for each test
        import modules.audio_input as ai
        importlib.reload(ai)
        self.ai = ai

    def test_import_does_not_initialize(self):
        """Verify that importing the module does not set _initialized to True automatically."""
        self.assertFalse(self.ai._initialized)
        self.assertFalse(self.ai._fallback_mode)

    def test_list_microphones(self):
        """Verify list_microphones returns a list of dictionaries with id and name."""
        mics = self.ai.list_microphones()
        self.assertIsInstance(mics, list)
        for mic in mics:
            self.assertIsInstance(mic, dict)
            self.assertIn("id", mic)
            self.assertIn("name", mic)

    def test_invalid_device_initialization_falls_back(self):
        """Verify that initializing with an invalid device falls back to silent mode without crashing."""
        # Try to initialize with a nonexistent device
        success = self.ai.initialize_microphone(device_id="nonexistent_id_9999", device_name="nonexistent_microphone_name")
        
        # Initialization should return False (since the requested device is missing),
        # but the module should transition to a stable fallback state.
        self.assertFalse(success)
        self.assertTrue(self.ai._initialized)
        self.assertTrue(self.ai._fallback_mode)
        self.assertEqual(self.ai._microphone_name, "None (Fallback)")

        # Subsequent get_audio_frame calls should return silent frames cleanly
        frame = self.ai.get_audio_frame()
        self.assertIsInstance(frame, dict)
        self.assertEqual(frame["sample_rate"], 16000)
        self.assertEqual(len(frame["samples"]), 16000)
        self.assertEqual(frame["rms"], 0.0)
        self.assertEqual(frame["peak"], 0.0)
        self.assertEqual(frame["microphone_name"], "None (Fallback)")

if __name__ == "__main__":
    unittest.main()
