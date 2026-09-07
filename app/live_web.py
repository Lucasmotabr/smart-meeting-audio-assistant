from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .live_pipeline import AI_MODEL_LOCK, get_audio_diagnostics, list_live_microphones, make_live_snapshot
except ImportError:
    from live_pipeline import AI_MODEL_LOCK, get_audio_diagnostics, list_live_microphones, make_live_snapshot


START_TIME = time.time()
SNAPSHOT_LOCK = threading.Lock()


class TranscriptService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._future: concurrent.futures.Future | None = None
        self._buffer = np.array([], dtype=np.float32)
        self._text = ""
        self._status = "Listening..."
        self._latency_ms: int | None = None
        self._mic: str | None = None
        self._last_submit = 0.0

    def update(
        self,
        samples: np.ndarray,
        sample_rate: int,
        mic: str | None,
        language: str | None,
        model_name: str | None,
    ) -> dict[str, Any]:
        with self._lock:
            transcript_key = f"{mic or ''}|{language or 'auto'}|{model_name or ''}"
            if transcript_key != self._mic:
                if self._future is not None and not self._future.done():
                    return {
                        "text": self._text,
                        "status": "Finishing previous transcription before switching settings...",
                        "latency_ms": self._latency_ms,
                        "busy": True,
                    }
                self._collect_completed_locked()
                self._buffer = np.array([], dtype=np.float32)
                self._text = ""
                self._status = "Listening..."
                self._latency_ms = None
                self._future = None
                self._mic = transcript_key
                self._last_submit = 0.0

            self._collect_completed_locked()

            clean_samples = np.asarray(samples, dtype=np.float32).flatten()
            clean_samples = np.nan_to_num(clean_samples, nan=0.0, posinf=0.0, neginf=0.0)
            if clean_samples.size:
                self._buffer = np.concatenate([self._buffer, clean_samples])[-sample_rate * 12 :]

            chunk_samples = sample_rate * 5
            can_submit = (
                self._future is None
                and self._buffer.size >= chunk_samples
                and time.time() - self._last_submit >= 5.0
            )
            if can_submit:
                chunk = self._buffer[:chunk_samples].copy()
                self._buffer = self._buffer[chunk_samples:]
                if _rms(chunk) >= TRANSCRIBE_RMS_THRESHOLD:
                    self._future = self._executor.submit(
                        _transcribe_chunk,
                        chunk,
                        sample_rate,
                        language,
                        model_name,
                    )
                    self._last_submit = time.time()
                    self._status = "Transcribing recent speech..."
                else:
                    self._status = "Listening for speech..."

            return {
                "text": self._text,
                "status": self._status,
                "latency_ms": self._latency_ms,
                "busy": self._future is not None,
            }

    def _collect_completed_locked(self) -> None:
        if self._future is None or not self._future.done():
            return
        try:
            result = self._future.result()
            text = str(result.get("text", "")).strip()
            if text:
                self._text = f"{self._text}\n{text}".strip() if self._text else text
                self._status = "Transcript updated."
            else:
                self._status = "Listening..."
            latency = result.get("latency_ms")
            self._latency_ms = int(latency) if latency is not None else None
        except Exception as exc:
            self._status = f"Transcription error: {exc}"
            self._latency_ms = None
        finally:
            self._future = None


TRANSCRIPTS = TranscriptService()
TRANSCRIBE_RMS_THRESHOLD = float(os.environ.get("SMAA_TRANSCRIBE_RMS_THRESHOLD", "0.01"))


def _transcribe_chunk(
    samples: np.ndarray,
    sample_rate: int,
    language: str | None,
    model_name: str | None,
) -> dict[str, Any]:
    from modules.speech_recognition import transcribe_audio

    with AI_MODEL_LOCK:
        return transcribe_audio(samples, sample_rate, language=language, model_name=model_name)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0


def _downsample(values: np.ndarray, target: int) -> list[float]:
    values = np.asarray(values, dtype=np.float32).flatten()
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return []
    if values.size <= target:
        return [round(float(v), 5) for v in values]
    indexes = np.linspace(0, values.size - 1, target).astype(int)
    return [round(float(values[i]), 5) for i in indexes]


def _spectrogram_payload(spectrogram: np.ndarray, max_bins: int = 96) -> list[list[float]]:
    spectrogram = np.asarray(spectrogram, dtype=np.float32)
    spectrogram = np.nan_to_num(spectrogram, nan=-120.0, posinf=-20.0, neginf=-120.0)
    if spectrogram.ndim != 2 or spectrogram.size == 0:
        return []
    if spectrogram.shape[0] > max_bins:
        row_idx = np.linspace(0, spectrogram.shape[0] - 1, max_bins).astype(int)
        spectrogram = spectrogram[row_idx, :]
    spectrogram = np.clip(spectrogram, -120.0, -20.0)
    return [[round(float(v), 1) for v in row] for row in spectrogram]


def snapshot_payload(
    mic: str | None,
    language: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    with SNAPSHOT_LOCK:
        snapshot = make_live_snapshot(START_TIME, mic)
        transcript = TRANSCRIPTS.update(
            snapshot.audio.samples,
            snapshot.audio.sample_rate,
            mic,
            language,
            model_name,
        )
    voice_level = 0.0
    if snapshot.visualization.voice_bars.size:
        voice_level = float(np.percentile(snapshot.visualization.voice_bars, 80))
    return {
        "timestamp": time.time(),
        "status": snapshot.status,
        "audio": {
            "rms": round(float(snapshot.audio.rms), 6),
            "peak": round(float(snapshot.audio.peak), 6),
            "sample_rate": int(snapshot.audio.sample_rate),
            "microphone_name": snapshot.audio.microphone_name,
            "microphone_type": snapshot.audio.microphone_type,
            "signal_db": round(20 * np.log10(max(float(snapshot.audio.rms), 1e-6)), 1),
        },
        "visuals": {
            "waveform": _downsample(snapshot.visualization.waveform, 512),
            "spectrogram": _spectrogram_payload(snapshot.visualization.spectrogram),
            "voice_level": round(voice_level, 3),
        },
        "classification": {
            "label": snapshot.classification.label.value,
            "confidence": round(float(snapshot.classification.confidence), 3),
            "scores": {
                key: round(float(value), 4)
                for key, value in snapshot.classification.scores.items()
            },
        },
        "quality": {
            "level": snapshot.quality.level,
            "score": round(float(snapshot.quality.score), 1),
            "issues": snapshot.quality.issues,
        },
        "transcript": transcript,
        "advice": {
            "message": snapshot.advice.message,
        },
    }


class LiveHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/snapshot":
            query = parse_qs(parsed.query)
            mic = query.get("mic", [None])[0]
            language = _clean_query_value(query.get("language", [None])[0])
            model_name = _clean_query_value(query.get("model", [None])[0])
            self._send_json(snapshot_payload(mic, language=language, model_name=model_name))
            return
        if parsed.path == "/api/microphones":
            self._send_json({"microphones": list_live_microphones(), "diagnostics": get_audio_diagnostics()})
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        if self.path.startswith("/api/snapshot"):
            return
        super().log_message(format, *args)

    def _send_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _clean_query_value(value: str | None) -> str | None:
    if value is None or value in {"", "auto"}:
        return None
    return value


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart Meeting Audio Assistant Live</title>
  <style>
    :root {
        --bg: #050816;
        --panel: #0b1022;
        --panel2: #10162d;
        --border: rgba(168, 85, 247, 0.22);
        --border2: rgba(255,255,255,0.09);
        --purple: #a855f7;
        --purple2: #7c3aed;
        --pink: #ec4899;
        --green: #22c55e;
        --orange: #f59e0b;
        --red: #ef4444;
        --text: #f8f7ff;
        --muted: #a8a4bd;
    }

    html, body {
        margin: 0 !important;
        padding: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        overflow: hidden !important;
        background: var(--bg) !important;
        -webkit-font-smoothing: antialiased;
        text-rendering: geometricPrecision;
    }

    * { box-sizing: border-box; }

    .smaa-app {
        width: 100vw;
        height: 100dvh;
        box-sizing: border-box;
        display: grid;
        grid-template-columns: 170px minmax(0, 1fr);
        background:
            radial-gradient(circle at 18% 0%, rgba(168, 85, 247, 0.28), transparent 40rem),
            radial-gradient(circle at 100% 20%, rgba(236, 72, 153, 0.08), transparent 34rem),
            linear-gradient(135deg, #040611 0%, #070b1a 60%, #040611 100%);
        color: var(--text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .sidebar {
        min-width: 0;
        border-right: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(7,11,26,.98), rgba(4,6,17,.98));
        padding: 14px 12px;
        display: grid;
        grid-template-rows: auto auto auto auto minmax(0, 1fr);
        gap: 8px;
        overflow: hidden;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 2px 4px 12px;
        border-bottom: 1px solid var(--border2);
    }

    .brand-icon {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, #2e1065, var(--purple2));
        box-shadow: 0 0 18px rgba(168, 85, 247, .62), inset 0 1px 0 rgba(255,255,255,.22);
        font-weight: 900;
        letter-spacing: -0.05em;
    }

    .brand-name {
        color: var(--purple);
        font-size: 29px;
        font-weight: 900;
        line-height: 1;
    }

    .brand-version,
    .small-muted {
        color: var(--muted);
        font-size: 12px;
    }

    .nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 11px;
        border-radius: 11px;
        color: var(--muted);
        margin-bottom: 3px;
        font-size: 13px;
        transition: color .16s ease, background .16s ease, transform .16s ease;
    }

    .nav-item:hover {
        color: white;
        background: rgba(168,85,247,.09);
        transform: translateX(2px);
    }

    .nav-item.active {
        color: white;
        background: linear-gradient(90deg, rgba(168,85,247,.95), rgba(124,58,237,.58));
        box-shadow: 0 0 24px rgba(168,85,247,.2);
    }

    .side-card,
    .panel,
    .metric-card {
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(16,22,45,.88), rgba(8,12,28,.94));
        border-radius: 16px;
        box-shadow: 0 18px 44px rgba(0,0,0,.24);
    }

    .side-card {
        padding: 9px 10px;
    }

    .side-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .mic-row {
        display: grid;
        grid-template-columns: 16px 1fr;
        gap: 8px;
        padding: 6px 7px;
        border-radius: 10px;
        background: rgba(255,255,255,.035);
        margin: 5px 0;
        color: inherit;
        text-decoration: none;
        transition: background .16s ease, border-color .16s ease;
        border: 1px solid transparent;
    }

    .mic-row:hover {
        background: rgba(168,85,247,.08);
        border-color: rgba(168,85,247,.22);
    }

    .mic-dot {
        width: 9px;
        height: 9px;
        margin-top: 4px;
        border-radius: 50%;
        border: 1px solid var(--purple);
    }

    .mic-row.active .mic-dot {
        background: var(--purple);
        box-shadow: 0 0 14px rgba(168,85,247,.8);
    }

    .manage-button {
        display: block;
        text-align: center;
        color: #c4b5fd;
        text-decoration: none;
        border: 1px solid rgba(168,85,247,.32);
        border-radius: 9px;
        padding: 6px;
        margin-top: 7px;
        font-size: 11px;
        background: rgba(168,85,247,.06);
    }

    .select-control {
        width: 100%;
        margin: 5px 0 10px;
        border: 1px solid rgba(168,85,247,.32);
        border-radius: 10px;
        padding: 8px 9px;
        background:
            linear-gradient(180deg, rgba(16,22,45,.98), rgba(4,7,18,.92));
        color: var(--text);
        font-size: 12px;
        font-weight: 700;
        outline: none;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
        appearance: none;
        background-image:
            linear-gradient(45deg, transparent 50%, #c4b5fd 50%),
            linear-gradient(135deg, #c4b5fd 50%, transparent 50%),
            linear-gradient(180deg, rgba(16,22,45,.98), rgba(4,7,18,.92));
        background-position:
            calc(100% - 15px) 50%,
            calc(100% - 10px) 50%,
            0 0;
        background-size: 5px 5px, 5px 5px, 100% 100%;
        background-repeat: no-repeat;
    }

    .select-control:focus {
        border-color: rgba(168,85,247,.72);
        box-shadow: 0 0 0 2px rgba(168,85,247,.16), inset 0 1px 0 rgba(255,255,255,.05);
    }

    .select-help {
        color: var(--muted);
        font-size: 10px;
        line-height: 1.35;
        margin-top: -4px;
    }

    .mic-name {
        font-size: 13px;
        color: var(--text);
    }

    .mic-type {
        color: var(--muted);
        font-size: 11px;
    }

    .live-row,
    .footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }

    .red { color: var(--red); }
    .green { color: var(--green); }
    .orange { color: var(--orange); }
    .purple { color: var(--purple); }

    .monitor-row {
        display: grid;
        grid-template-columns: 58px 1fr 34px;
        gap: 8px;
        align-items: center;
        font-size: 11px;
        color: var(--muted);
        margin: 7px 0;
    }

    .bar-track {
        height: 7px;
        border-radius: 999px;
        background: rgba(255,255,255,.08);
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--purple2), var(--purple));
        transition: width .45s ease;
    }

    .main {
        min-width: 0;
        height: 100dvh;
        box-sizing: border-box;
        padding: 8px 10px 4px;
        display: grid;
        grid-template-rows: 72px minmax(0, 1fr) 24px;
        gap: 8px;
        overflow: hidden;
    }

    .header {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(480px, 600px) 34px;
        gap: 10px;
        min-height: 0;
    }

    .title {
        font-size: 31px;
        font-weight: 900;
        letter-spacing: -0.04em;
        margin-top: 1px;
    }

    .subtitle {
        color: #c3bed7;
        font-size: 15px;
        margin-top: 6px;
    }

    .metrics {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
    }

    .metric-card {
        padding: 8px 10px;
        display: grid;
        grid-template-columns: 30px 1fr;
        grid-template-rows: auto auto auto;
        column-gap: 8px;
        align-content: center;
    }

    .metric-label {
        grid-column: 2;
        font-size: 12px;
        color: var(--muted);
    }

    .metric-value {
        grid-column: 2;
        font-size: 16px;
        font-weight: 900;
        margin-top: 2px;
    }

    .metric-note {
        grid-column: 2;
        color: var(--muted);
        font-size: 11px;
        margin-top: 1px;
    }

    .metric-icon {
        grid-column: 1;
        grid-row: 1 / span 3;
        align-self: center;
        color: var(--purple);
    }

    .settings-button {
        width: 34px;
        height: 34px;
        align-self: center;
        border-radius: 10px;
        border: 1px solid var(--border);
        display: grid;
        place-items: center;
        color: #c4b5fd;
        background: rgba(16,22,45,.8);
    }

    .content {
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(0, .94fr) minmax(0, 1.06fr);
        grid-template-rows: minmax(118px, .72fr) minmax(148px, .88fr) minmax(0, 1fr);
        gap: 8px;
    }

    .panel {
        min-height: 0;
        overflow: hidden;
        padding: 11px 12px;
        box-sizing: border-box;
    }

    .panel-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 14px;
        font-weight: 850;
        margin-bottom: 8px;
    }

    .panel-heading {
        display: inline-flex;
        align-items: center;
        gap: 7px;
    }

    .live-badge {
        color: #d8b4fe;
        background: rgba(168,85,247,.16);
        border: 1px solid rgba(168,85,247,.42);
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 12px;
        font-weight: 600;
    }

    .voice-panel {
        grid-column: 1;
        grid-row: 1;
    }

    .wave-panel {
        grid-column: 1;
        grid-row: 2;
    }

    .spectrogram-panel {
        grid-column: 2;
        grid-row: 1 / span 2;
    }

    .bottom-grid {
        grid-column: 1 / span 2;
        grid-row: 3;
        min-height: 0;
        display: grid;
        grid-template-columns: 1fr 1.05fr .9fr 1.2fr;
        gap: 8px;
    }

    .voice-bars {
        height: calc(100% - 39px);
        display: flex;
        align-items: end;
        gap: 6px;
        padding: 8px 8px 0;
        background: repeating-linear-gradient(90deg, rgba(255,255,255,.025) 0, rgba(255,255,255,.025) 1px, transparent 1px, transparent 22px);
        border-radius: 12px;
    }

    .voice-bar {
        flex: 1;
        min-height: 7px;
        border-radius: 4px 4px 0 0;
        background: linear-gradient(180deg, var(--purple), var(--purple2));
        box-shadow: 0 0 18px rgba(168,85,247,.32);
        transition: height .42s ease, opacity .42s ease;
    }

    .listening {
        color: var(--purple);
        margin-top: 8px;
        font-size: 14px;
    }

    .wave-svg,
    .spectro-svg {
        width: 100%;
        height: calc(100% - 27px);
        border-radius: 12px;
        background:
            linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px),
            rgba(4,7,18,.8);
        background-size: 100% 36px, 44px 100%, auto;
    }

    .spectro-svg {
        height: calc(100% - 27px);
    }

    .axis-label {
        color: var(--muted);
        font-size: 11px;
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
    }

    .classification-label {
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: var(--purple);
        font-size: 16px;
        font-weight: 900;
        margin-bottom: 7px;
    }

    .detected-badge {
        border: 1px solid rgba(168,85,247,.38);
        background: rgba(168,85,247,.14);
        color: #c4b5fd;
        border-radius: 7px;
        padding: 3px 7px;
        font-size: 10px;
        font-weight: 700;
    }

    .classification-row,
    .quality-row {
        display: grid;
        grid-template-columns: 1fr 94px;
        gap: 10px;
        align-items: center;
        color: var(--text);
        font-size: 11px;
        margin: 7px 0;
    }

    .row-label {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .percent {
        text-align: right;
        color: var(--muted);
        font-size: 11px;
        margin-top: 2px;
    }

    .transcript-box,
    .advice-box {
        height: calc(100% - 48px);
        box-sizing: border-box;
        overflow: hidden;
        border: 1px solid var(--border2);
        border-radius: 13px;
        padding: 14px;
        background: rgba(4,7,18,.68);
        line-height: 1.55;
        font-size: 14px;
    }

    .transcript-box {
        overflow-y: auto;
    }

    .advice-box {
        height: calc(100% - 27px);
        border-color: rgba(168,85,247,.5);
        background: linear-gradient(180deg, rgba(168,85,247,.14), rgba(4,7,18,.74));
    }

    .advice-box ul {
        margin: 8px 0 0 18px;
        padding: 0;
    }

    .quality-ring {
        width: 86px;
        height: 86px;
        margin: 0 auto 6px;
        display: grid;
        place-items: center;
        border-radius: 50%;
        background:
            radial-gradient(circle at center, #10162d 0 52%, transparent 53%),
            conic-gradient(var(--purple) var(--score), rgba(255,255,255,.08) 0);
        filter: drop-shadow(0 0 10px rgba(168,85,247,.5));
    }

    .quality-score {
        font-size: 27px;
        font-weight: 950;
        line-height: 1;
        text-align: center;
    }

    .quality-word {
        color: var(--muted);
        font-size: 12px;
        text-align: center;
    }

    .footer {
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-size: 10px;
        padding-top: 5px;
    }

    .mini-bars {
        display: inline-flex;
        gap: 3px;
        height: 18px;
        align-items: end;
        margin-right: 6px;
    }

    .mini-bars span {
        width: 4px;
        border-radius: 99px;
        background: var(--green);
        display: block;
    }

    .quality-subtitle {
        color: var(--muted);
        font-size: 10px;
        margin: 2px 0 3px;
    }

    .confidence-pill {
        display: inline-block;
        border-radius: 6px;
        background: rgba(168,85,247,.16);
        color: #c4b5fd;
        padding: 3px 7px;
        font-size: 10px;
        margin-top: 6px;
    }

    .auto-scroll {
        display: flex;
        align-items: center;
        gap: 6px;
        color: var(--muted);
        font-size: 10px;
        margin-top: 6px;
    }

    .icon {
        width: 16px;
        height: 16px;
        display: inline-block;
        vertical-align: middle;
        flex: 0 0 auto;
    }

    @media (max-height: 760px) {
        .sidebar { padding-top: 8px; gap: 6px; }
        .brand-icon { width: 38px; height: 38px; }
        .brand-name { font-size: 25px; }
        .mic-row { padding: 4px 6px; margin: 3px 0; }
        .content { grid-template-rows: minmax(105px,.68fr) minmax(132px,.82fr) minmax(0,1fr); }
        .advice-box { font-size: 11px; }
    }
  </style>
</head>
<body>
<div class="smaa-app">
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-icon">~</div>
      <div>
        <div class="brand-name">SMAA</div>
        <div class="brand-version">v1.0.0</div>
      </div>
    </div>
    <div class="side-card">
      <div class="side-title"><span>Microphone</span><span>input</span></div>
      <div id="mics" class="small-muted">Loading...</div>
      <div id="diag" class="mic-type"></div>
      <a class="manage-button" href="#microphones">Manage Microphones</a>
    </div>
    <div class="side-card">
      <div class="side-title">Recording Status</div>
      <div class="live-row"><span><span class="red">●</span> LIVE</span><span id="elapsed" class="purple">00:00</span></div>
      <div id="recordingStats" class="mic-type" style="margin-top:7px;line-height:1.35;">RMS: --<br>Peak: --<br>Source: --</div>
    </div>
    <div class="side-card">
      <div class="side-title"><span>System</span><span>status</span></div>
      <div class="monitor-row"><span>CPU</span><div class="bar-track"><div class="bar-fill" style="width:28%"></div></div><span>28%</span></div>
      <div class="monitor-row"><span>Memory</span><div class="bar-track"><div class="bar-fill" style="width:41%"></div></div><span>41%</span></div>
      <div class="monitor-row"><span>FPS</span><div class="bar-track"><div class="bar-fill" style="width:96%"></div></div><span>30</span></div>
    </div>
    <div class="side-card">
      <div class="side-title">Transcription</div>
      <label class="mic-type" for="languageSelect">Language</label>
      <select id="languageSelect" class="select-control">
        <option value="auto">Auto</option>
        <option value="en">English</option>
        <option value="ja">Japanese</option>
      </select>
      <label class="mic-type" for="modelSelect">Whisper model</label>
      <select id="modelSelect" class="select-control">
        <option value="tiny">tiny - fastest</option>
        <option value="base">base - better</option>
        <option value="small">small - slower</option>
      </select>
      <div class="select-help">Changing these updates the live backend without reloading.</div>
    </div>
  </aside>
  <main class="main">
    <section class="header">
      <div><div class="title">Smart Meeting Audio Assistant</div><div class="subtitle">Real-time Sound Analysis, Speech Recognition & AI Advice</div></div>
      <div class="metrics">
        <div class="metric-card"><div class="metric-icon">~</div><div class="metric-label">Audio Quality</div><div id="quality" class="metric-value green">--</div><div class="metric-note">Overall Quality</div></div>
        <div class="metric-card"><div class="metric-icon">▮</div><div class="metric-label">Signal Strength</div><div id="signal" class="metric-value green">-- dB</div><div class="metric-note">Input Level</div></div>
        <div class="metric-card"><div class="metric-icon">○</div><div class="metric-label">Latency</div><div id="latency" class="metric-value purple">0 ms</div><div class="metric-note">Processing Delay</div></div>
      </div>
      <div class="settings-button">⚙</div>
    </section>
    <section class="content">
      <div class="panel voice-panel">
        <div class="panel-title"><span class="panel-heading">Voice Activity Timeline</span><span class="live-badge">Live</span></div>
        <div id="voice" class="voice-bars"></div>
        <div id="listening" class="listening">Listening...</div>
      </div>
      <div class="panel wave-panel">
        <div class="panel-title"><span class="panel-heading">Waveform</span><span class="live-badge">Live</span></div>
        <canvas id="wave" class="wave-svg"></canvas>
      </div>
      <div class="panel spectrogram-panel">
        <div class="panel-title"><span class="panel-heading">Spectrogram</span><span class="small-muted">Scale: Log</span></div>
        <canvas id="spec" class="spectro-svg"></canvas>
      </div>
      <div class="bottom-grid">
        <div class="panel">
          <div class="panel-title"><span class="panel-heading">Sound Classification</span></div>
          <div class="classification-label"><span id="classLabel">--</span><span id="classConf" class="detected-badge">Detected</span></div>
          <div id="classRows"></div>
          <div class="small-muted">Model: YAMNet | Normalized tracked-category mix</div>
        </div>
        <div class="panel">
          <div class="panel-title"><span class="panel-heading">Transcription</span><span class="live-badge">Live</span></div>
          <div id="transcript" class="transcript-box">No speech detected yet.</div>
          <div class="auto-scroll"><span class="purple">●</span> Auto Scroll</div>
        </div>
        <div class="panel">
          <div class="panel-title"><span class="panel-heading">Audio Quality</span></div>
          <div id="qualityRing" class="quality-ring" style="--score:50%"><div><div id="qualityScore" class="quality-score">--</div><div id="qualityWord" class="quality-word">--</div></div></div>
          <div class="quality-subtitle">Quality Factors</div>
          <div id="qualityRows"></div>
          <div id="qualityIssues" class="small-muted"></div>
        </div>
        <div class="panel">
          <div class="panel-title"><span class="panel-heading">AI Advice</span><span class="small-muted">Powered by Groq</span></div>
          <div class="advice-box"><div id="advice">Waiting for live input...</div><b>Recommendations:</b><ul id="recommendations"><li>Current setup is good</li><li>Keep microphone distance stable</li><li>Avoid sudden background noise</li></ul><div class="confidence-pill">Confidence: High</div></div>
        </div>
      </div>
    </section>
    <footer class="footer"><span id="status"><span class="green">●</span> All Systems Operational</span><span>Sample Rate: 16 kHz</span><span>Model Status: Live Integration</span><span id="clock">Time: --:--:--</span></footer>
  </main>
</div>
<script>
const qs = new URLSearchParams(location.search);
let selectedMic = qs.get("mic");
let selectedLanguage = qs.get("language") || "auto";
let selectedModel = qs.get("model") || "tiny";
let waveHistory = [];
let voiceHistory = [];
let specHistory = [];
let micDiagBase = "";
const maxWave = 16000 * 30 / 80;
const maxVoice = 30;
const maxSpecCols = 900;
const classLabels = ["Keyboard Typing","Speech","Background Noise","Silence","Clap"];
const qualityLevels = {
  high: {"Signal Strength":88,"Noise Level":18,"Clarity":92,"Stability":94},
  medium: {"Signal Strength":72,"Noise Level":65,"Clarity":80,"Stability":84},
  low: {"Signal Strength":38,"Noise Level":73,"Clarity":42,"Stability":55}
};
const startedAt = Date.now();

function esc(v){return String(v ?? "").replace(/[&<>"']/g,s=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[s]));}
function titleCase(v){return String(v ?? "").replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase());}
function displayClassLabel(label){const v=String(label ?? "").toLowerCase();if(v==="typing")return "Keyboard Typing";if(v==="speech")return "Speech";if(v==="background noise")return "Background Noise";if(v==="silence")return "Silence";if(v==="clap")return "Clap";return "Unknown";}
function classScoreKey(label){const v=String(label ?? "").toLowerCase();if(v==="keyboard typing")return "typing";if(v==="speech")return "speech";if(v==="background noise")return "background noise";if(v==="silence")return "silence";if(v==="clap")return "clap";return v;}
function integerPercents(values){const clean=values.map(v=>Math.max(0,Number(v)||0));const total=clean.reduce((a,b)=>a+b,0);if(total<=0)return clean.map(()=>0);const scaled=clean.map(v=>v/total*100);const base=scaled.map(Math.floor);let remainder=100-base.reduce((a,b)=>a+b,0);const order=scaled.map((v,i)=>[v-base[i],i]).sort((a,b)=>b[0]-a[0]);for(let n=0;n<remainder;n++)base[order[n%order.length][1]]+=1;return base;}
function canvas(id){const c=document.getElementById(id);const dpr=devicePixelRatio||1;const r=c.getBoundingClientRect();const w=Math.max(10,Math.floor(r.width*dpr));const h=Math.max(10,Math.floor(r.height*dpr));if(c.width!==w||c.height!==h){c.width=w;c.height=h;}return [c,c.getContext("2d"),w,h];}
function clear(ctx,w,h){ctx.fillStyle="#030617";ctx.fillRect(0,0,w,h);ctx.strokeStyle="rgba(255,255,255,.08)";ctx.lineWidth=1;for(let i=0;i<7;i++){const x=i*w/6;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}}
function drawWave(samples){waveHistory.push(...samples.filter(Number.isFinite));if(waveHistory.length>maxWave)waveHistory=waveHistory.slice(-maxWave);const [c,ctx,w,h]=canvas("wave");clear(ctx,w,h);ctx.strokeStyle="#a855f7";ctx.lineWidth=2*(devicePixelRatio||1);ctx.beginPath();const n=waveHistory.length||1;for(let i=0;i<n;i++){const x=i/(n-1||1)*w;const y=h/2-waveHistory[i]*(h*.42);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();}
function drawVoice(level){voiceHistory.push(level);if(voiceHistory.length>maxVoice)voiceHistory=voiceHistory.slice(-maxVoice);const box=document.getElementById("voice");const max=Math.max(.001,...voiceHistory);box.innerHTML=Array.from({length:maxVoice},(_,i)=>{const v=voiceHistory[i]||0;const h=Math.max(7,Math.min(100,(v/max)*88));return `<div class="voice-bar" style="height:${h}%"></div>`;}).join("");}
function color(v){const t=Math.max(0,Math.min(1,(v+120)/100));const stops=[[8,6,27],[60,9,101],[96,19,110],[133,33,107],[172,50,93],[209,75,65],[237,121,38],[252,181,44],[252,255,164]];const p=t*(stops.length-1),i=Math.floor(p),f=p-i,a=stops[i],b=stops[Math.min(i+1,stops.length-1)];return `rgb(${a[0]+(b[0]-a[0])*f},${a[1]+(b[1]-a[1])*f},${a[2]+(b[2]-a[2])*f})`;}
function drawSpec(matrix){if(matrix.length){const rows=matrix.length,cols=matrix[0].length;for(let c=0;c<cols;c++){let col=[];for(let r=0;r<rows;r++)col.push(matrix[r][c]);specHistory.push(col);}if(specHistory.length>maxSpecCols)specHistory=specHistory.slice(-maxSpecCols);}const [can,ctx,w,h]=canvas("spec");ctx.fillStyle="#030617";ctx.fillRect(0,0,w,h);const cols=specHistory.length,rows=specHistory[0]?.length||1;const cw=w/Math.max(1,cols);const rh=h/rows;for(let x=0;x<cols;x++){const col=specHistory[x];for(let r=0;r<rows;r++){ctx.fillStyle=color(col[r]);ctx.fillRect(x*cw,h-(r+1)*rh,Math.ceil(cw)+1,Math.ceil(rh)+1);}}}
function classificationRows(label, confidence, scores){const active=displayClassLabel(label);const values=classLabels.map(name=>scores?.[classScoreKey(name)] ?? 0);const hasScores=values.some(v=>Number(v)>0);const percentages=hasScores?integerPercents(values):classLabels.map(name=>name===active?Math.round((Number(confidence)||0)*100):0);return classLabels.map((name,i)=>{const v=Math.max(0,Math.min(100,percentages[i]||0));const isActive=name===active;const opacity=v>0||isActive?1:.48;return `<div class="classification-row" style="opacity:${opacity}"><span class="row-label">${esc(name)}</span><div><div class="bar-track"><div class="bar-fill" style="width:${v}%"></div></div><div class="percent">${v}%</div></div></div>`;}).join("");}
function qualityRows(level){const rows=qualityLevels[level]||qualityLevels.medium;return Object.entries(rows).map(([k,v])=>`<div class="quality-row"><span class="row-label">${esc(k)}</span><div><div class="bar-track"><div class="bar-fill" style="width:${v}%"></div></div></div></div>`).join("");}
function recommendations(d){if(d.quality.level==="low")return ["Move closer to the microphone or increase input gain","Keep microphone distance stable","Avoid sudden background noise"];if(d.classification.label==="silence")return ["Start speaking to test transcription","Confirm the selected microphone is correct","Keep the browser page open during capture"];return ["Current setup is good","Keep microphone distance stable","Avoid sudden background noise"];}
function micHref(id){const p=new URLSearchParams();p.set("mic",id);p.set("language",selectedLanguage);p.set("model",selectedModel);return `/?${p.toString()}`;}
async function loadMics(){const res=await fetch("/api/microphones");const data=await res.json();const box=document.getElementById("mics");const diag=document.getElementById("diag");micDiagBase=`sounddevice: ${data.diagnostics.sounddevice?"yes":"no"}<br>inputs: ${data.diagnostics.microphone_count}<br>default: ${esc(JSON.stringify(data.diagnostics.default_device))}`;diag.innerHTML=micDiagBase;if(!data.microphones.length){box.textContent="No microphones exposed to this Python process";return;}if(!selectedMic)selectedMic=String(data.microphones[0].id);box.innerHTML=data.microphones.map(m=>`<a class="mic-row ${String(m.id)===String(selectedMic)?"active":""}" href="${micHref(m.id)}"><span class="mic-dot"></span><span><span class="mic-name">${esc(m.name)}</span><br><span class="mic-type">${esc(m.type)}</span></span></a>`).join("");}
function initControls(){const lang=document.getElementById("languageSelect");const model=document.getElementById("modelSelect");lang.value=selectedLanguage;model.value=selectedModel;function apply(){selectedLanguage=lang.value;selectedModel=model.value;const next=new URLSearchParams(location.search);next.set("language",selectedLanguage);next.set("model",selectedModel);if(selectedMic)next.set("mic",selectedMic);history.replaceState(null,"",`/?${next.toString()}`);document.getElementById("transcript").textContent="Transcription settings changed. Listening...";document.getElementById("status").innerHTML='<span class="green">●</span> Transcription settings updated';}lang.addEventListener("change",apply);model.addEventListener("change",apply);}
function updateClock(){const sec=Math.floor((Date.now()-startedAt)/1000);const m=String(Math.floor(sec/60)).padStart(2,"0");const s=String(sec%60).padStart(2,"0");document.getElementById("elapsed").textContent=`${m}:${s}`;document.getElementById("clock").textContent=`Time: ${new Date().toLocaleTimeString()}`;}
async function tick(){try{updateClock();const params=new URLSearchParams();if(selectedMic)params.set("mic",selectedMic);params.set("language",selectedLanguage);params.set("model",selectedModel);const res=await fetch(`/api/snapshot?${params.toString()}`,{cache:"no-store"});const d=await res.json();const level=d.quality.level||"medium";const label=displayClassLabel(d.classification.label);document.getElementById("quality").textContent=titleCase(level);document.getElementById("signal").textContent=`${d.audio.signal_db} dB`;document.getElementById("latency").textContent=d.transcript.busy?"Transcribing":(d.transcript.latency_ms?`${d.transcript.latency_ms} ms`:"0 ms");document.getElementById("diag").innerHTML=micDiagBase;document.getElementById("recordingStats").innerHTML=`RMS: ${d.audio.rms}<br>Peak: ${d.audio.peak}<br>Source: ${esc(d.audio.microphone_type)}`;document.getElementById("classLabel").textContent=label;document.getElementById("classConf").textContent=`Detected ${(d.classification.confidence*100).toFixed(0)}%`;document.getElementById("classRows").innerHTML=classificationRows(d.classification.label,d.classification.confidence,d.classification.scores);document.getElementById("qualityScore").textContent=Math.round(d.quality.score);document.getElementById("qualityWord").textContent=titleCase(level);document.getElementById("qualityRing").style.setProperty("--score",`${Math.round(d.quality.score)}%`);document.getElementById("qualityRows").innerHTML=qualityRows(level);document.getElementById("qualityIssues").textContent=d.quality.issues.length?`Issues: ${d.quality.issues.join(", ")}`:"";document.getElementById("advice").textContent=d.advice.message;document.getElementById("recommendations").innerHTML=recommendations(d).map(x=>`<li>${esc(x)}</li>`).join("");document.getElementById("status").innerHTML=`<span class="green">●</span> ${esc(d.transcript.status||"All Systems Operational")}`;document.getElementById("listening").textContent=d.transcript.status||"Listening...";const tx=document.getElementById("transcript");tx.textContent=d.transcript.text||"No speech detected yet.";tx.scrollTop=tx.scrollHeight;drawWave(d.visuals.waveform);drawVoice(d.visuals.voice_level);drawSpec(d.visuals.spectrogram);}catch(e){document.getElementById("status").textContent=`Disconnected: ${e}`;}finally{setTimeout(tick,900);}}
addEventListener("resize",()=>{drawWave([]);drawVoice(0);drawSpec([]);});
initControls();
loadMics().then(tick);
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the no-refresh live SMAA web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=int(os.environ.get("SMAA_LIVE_WEB_PORT", "8502")), type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), LiveHandler)
    print(f"Live web UI running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping live web UI...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
