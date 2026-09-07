"""Regression tests for the native microphone stream lifecycle.

These guard the defect where re-initializing the microphone replaced
``modules.audio_input.macos._stream`` without closing the previous stream. The
orphaned-but-still-active stream was later destroyed by the garbage collector
while its CoreAudio callback was running, which corrupted PortAudio's global
state and segfaulted the next call into it.

The tests assert the ownership rules directly rather than reproducing the
crash: every stream they open is closed on the way out, including when an
assertion fails, so no orphaned stream is ever left for the collector.
"""

import importlib
import os
import sys
import unittest

# Add project root to path to resolve modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _macos_impl():
    """Return the macOS audio implementation, or None if unavailable here."""
    if sys.platform != "darwin":
        return None
    try:
        import modules.audio_input.macos as macos
    except ImportError:
        return None
    return macos


def _force_close(stream):
    """Close a stream we still hold a reference to, whatever state it is in.

    Used as a cleanup so a regression cannot leave a live orphan behind.
    """
    try:
        if not stream.closed:
            stream.stop()
            stream.close()
    except Exception:
        pass


class TestStreamShutdownContract(unittest.TestCase):
    """Ownership rules that need no audio hardware and never touch PortAudio."""

    def setUp(self):
        self.macos = _macos_impl()
        if self.macos is None:
            self.skipTest("macOS audio implementation unavailable")
        previous = self.macos._stream
        self.addCleanup(setattr, self.macos, "_stream", previous)

    def test_failed_shutdown_keeps_stream_reference(self):
        """A stream that cannot be closed must not be silently discarded."""

        class ExplodingStream:
            closed = False

            def stop(self):
                raise RuntimeError("stop failed")

            def close(self):
                raise RuntimeError("close failed")

        sentinel = ExplodingStream()
        self.macos._stream = sentinel

        with self.assertRaises(RuntimeError):
            self.macos.close_stream()

        # Dropping the reference here would orphan a live native stream and
        # hand its teardown to the garbage collector.
        self.assertIs(self.macos._stream, sentinel)

    def test_close_stream_is_idempotent(self):
        self.macos._stream = None

        self.macos.close_stream()
        self.macos.close_stream()

        self.assertIsNone(self.macos._stream)
        self.assertIsNone(self.macos._audio_buffer)

    def test_cleanup_does_not_rely_on_module_del(self):
        """CPython never calls a module-level ``__del__``, so it must not be
        the cleanup mechanism. Ownership is explicit via reset_microphone()."""
        self.assertNotIn("__del__", vars(self.macos))


class TestStreamReinitialization(unittest.TestCase):
    """Re-initialization must replace the stream without orphaning it.

    Requires a real input device; skipped when none is present.
    """

    def setUp(self):
        self.macos = _macos_impl()
        if self.macos is None:
            self.skipTest("macOS audio implementation unavailable")
        if not self.macos.list_microphones():
            self.skipTest("no input microphone available")

        import modules.audio_input.core as core

        try:
            core.reset_microphone()
        except Exception:
            pass
        importlib.reload(core)
        self.core = core
        self.addCleanup(self._release)

    def _release(self):
        try:
            self.core.reset_microphone()
        except Exception:
            pass

    def test_reinitialize_closes_previous_stream(self):
        self.assertTrue(self.core.initialize_microphone())

        first = self.macos._stream
        self.assertIsNotNone(first)
        self.assertFalse(first.closed)
        self.addCleanup(_force_close, first)

        self.macos.initialize_microphone()
        second = self.macos._stream

        self.assertIsNot(second, first, "a new stream object should be installed")
        self.assertTrue(first.closed, "the previous stream must be closed, not orphaned")
        self.assertFalse(second.closed)

    def test_reload_then_reinitialize_does_not_orphan(self):
        """The sequence that used to segfault.

        Reloading core desyncs it from the live native stream, so the next
        initialization used to install a second stream over the first.
        """
        self.assertTrue(self.core.initialize_microphone())

        first = self.macos._stream
        self.assertIsNotNone(first)
        self.addCleanup(_force_close, first)

        importlib.reload(self.core)

        # core has forgotten the microphone, but the native stream is still live.
        self.assertFalse(self.core._initialized)
        self.assertIs(self.macos._stream, first)

        self.assertTrue(self.core.initialize_microphone())

        self.assertIsNot(self.macos._stream, first)
        self.assertTrue(first.closed, "reload followed by re-init must not orphan the old stream")

    def test_reset_releases_the_stream(self):
        self.assertTrue(self.core.initialize_microphone())
        self.assertIsNotNone(self.macos._stream)

        self.core.reset_microphone()

        self.assertIsNone(self.macos._stream)
        self.assertIsNone(self.macos._audio_buffer)


if __name__ == "__main__":
    unittest.main()
