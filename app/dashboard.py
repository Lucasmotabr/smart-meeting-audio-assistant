from __future__ import annotations

import base64
import html
import time
from datetime import datetime
from io import BytesIO
from urllib.parse import quote

import numpy as np
import streamlit as st
from PIL import Image

try:
    from .contracts import NoiseLabel
    from .live_pipeline import make_live_snapshot
    from .mock_data import make_mock_snapshot
except ImportError:
    from contracts import NoiseLabel
    from live_pipeline import make_live_snapshot
    from mock_data import make_mock_snapshot


SCENARIOS = [
    "Clean speech",
    "Keyboard typing noise",
    "Far distance speech",
    "Silence / idle room",
    "Microphone comparison",
]

MICROPHONES = {
    "Alienware": ("Alienware x15", "MEMS"),
    "EarPods": ("EarPods USB-C", "Condenser"),
    "BlueYeti": ("Blue Yeti", "Condenser"),
}

QUALITY_FACTORS = {
    "high": {"Signal Strength": 88, "Noise Level": 18, "Clarity": 92, "Stability": 94},
    "medium": {"Signal Strength": 72, "Noise Level": 65, "Clarity": 80, "Stability": 84},
    "low": {"Signal Strength": 38, "Noise Level": 73, "Clarity": 42, "Stability": 55},
}

INFERNO = np.array(
    [
        [8, 6, 27],
        [60, 9, 101],
        [96, 19, 110],
        [133, 33, 107],
        [172, 50, 93],
        [209, 75, 65],
        [237, 121, 38],
        [252, 181, 44],
        [252, 255, 164],
    ],
    dtype=np.float32,
)


def main() -> None:
    st.set_page_config(
        page_title="Smart Meeting Audio Assistant",
        page_icon="SA",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _init_state()
    _inject_shell_css()

    scenario = _scenario_from_query()
    microphone = _microphone_from_query()
    mode = _mode_from_query()
    if mode == "live":
        snapshot = make_live_snapshot(st.session_state.start_time)
    else:
        snapshot = make_mock_snapshot(st.session_state.start_time, scenario)
        _apply_demo_microphone(snapshot, microphone)

    st.markdown(_compact_html(_dashboard_html(snapshot, scenario, microphone, mode)), unsafe_allow_html=True)

    time.sleep(0.8)
    st.rerun()


def _init_state() -> None:
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()


def _scenario_from_query() -> str:
    scenario = st.query_params.get("scenario", SCENARIOS[1])
    if scenario not in SCENARIOS:
        return SCENARIOS[1]
    return scenario


def _microphone_from_query() -> str:
    microphone = st.query_params.get("mic", "Alienware")
    if microphone not in MICROPHONES:
        return "Alienware"
    return microphone


def _mode_from_query() -> str:
    mode = st.query_params.get("mode", "demo")
    if mode not in {"demo", "live"}:
        return "demo"
    return mode


def _inject_shell_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #050816;
            --bg2: #070b1a;
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

        html, body, .stApp {
            margin: 0 !important;
            padding: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            overflow: hidden !important;
            background: var(--bg) !important;
            -webkit-font-smoothing: antialiased;
            text-rendering: geometricPrecision;
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stStatusWidget"],
        [data-testid="stSidebar"],
        #MainMenu,
        footer {
            display: none !important;
        }

        .block-container {
            max-width: none !important;
            width: 100vw !important;
            height: 100vh !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
        }

        [data-testid="stMarkdownContainer"] {
            width: 100%;
        }

        .smaa-app {
            width: 100vw;
            height: 100dvh;
            box-sizing: border-box;
            display: grid;
            grid-template-columns: 195px minmax(0, 1fr);
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
            grid-template-rows: auto auto auto auto auto 1fr;
            gap: 9px;
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
            padding: 10px;
        }

        .side-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 7px;
        }

        .scenario-link {
            display: block;
            color: var(--muted);
            text-decoration: none;
            font-size: 12px;
            border: 1px solid transparent;
            border-radius: 9px;
            padding: 5px 7px;
            margin: 2px 0;
        }

        .scenario-link.active {
            color: white;
            border-color: rgba(168,85,247,.55);
            background: rgba(168,85,247,.16);
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
            margin: 9px 0;
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
            .scenario-link { padding: 3px 6px; }
            .mic-row { padding: 4px 6px; margin: 3px 0; }
            .content { grid-template-rows: minmax(105px,.68fr) minmax(132px,.82fr) minmax(0,1fr); }
            .advice-box { font-size: 11px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _dashboard_html(snapshot, scenario: str, microphone: str = "Alienware", mode: str = "demo") -> str:
    elapsed = snapshot.audio.timestamp_seconds
    signal_db = _signal_strength_db(snapshot)
    quality_color = _quality_color(snapshot.quality.level)
    now = datetime.now()

    return f"""
    <div class="smaa-app">
        {_sidebar_html(snapshot, scenario, microphone, elapsed)}
        <main class="main">
            <section class="header">
                <div>
                    <div class="title">Smart Meeting Audio Assistant</div>
                    <div class="subtitle">Real-time Sound Analysis, Speech Recognition & AI Advice</div>
                </div>
                <div class="metrics">
                    {_metric_html("Audio Quality", snapshot.quality.level.title(), "Overall Quality", quality_color, "wave")}
                    {_metric_html("Signal Strength", f"{signal_db:.0f} dB", "Input Level", "green", "bars", _mini_bars(snapshot.audio.rms))}
                    {_metric_html("Latency", f"{snapshot.transcript.latency_ms or 0} ms", "Processing Delay", "purple", "clock")}
                </div>
                <div class="settings-button">{_icon("settings", 17)}</div>
            </section>

            <section class="content">
                <div class="panel voice-panel">
                    {_panel_title("Voice Activity", live=True, icon="wave")}
                    {_voice_bars_html(snapshot.visualization.voice_bars)}
                    <div class="listening">{html.escape(snapshot.transcript.partial_text)}</div>
                </div>

                <div class="panel wave-panel">
                    {_panel_title("Waveform", live=True, icon="wave")}
                    {_waveform_svg(snapshot.visualization.waveform)}
                </div>

                <div class="panel spectrogram-panel">
                    {_panel_title("Spectrogram", icon="spectrogram", right="Scale: Log")}
                    {_spectrogram_svg(snapshot.visualization.spectrogram)}
                </div>

                <div class="bottom-grid">
                    {_classification_html(snapshot)}
                    {_transcript_html(snapshot)}
                    {_quality_html(snapshot)}
                    {_advice_html(snapshot)}
                </div>
            </section>

            <footer class="footer">
                <span><span class="green">●</span> All Systems Operational</span>
                <span>Sample Rate: {snapshot.audio.sample_rate // 1000} kHz</span>
                <span>Chunk Size: {len(snapshot.audio.samples)}</span>
                <span>Model Status: {"Live Integration" if mode == "live" else "Mock Active"}</span>
                <span>Time: {now.strftime("%H:%M:%S")}</span>
                <span>Date: {now.strftime("%Y-%m-%d")}</span>
            </footer>
        </main>
    </div>
    """


def _compact_html(markup: str) -> str:
    """Avoid Markdown interpreting indented raw HTML as code blocks."""

    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def _sidebar_html(snapshot, scenario: str, microphone: str, elapsed: float) -> str:
    return f"""
    <aside class="sidebar">
        <div class="brand">
            <div class="brand-icon">{_icon("wave", 25)}</div>
            <div>
                <div class="brand-name">SMAA</div>
                <div class="brand-version">v1.0.0</div>
            </div>
        </div>

        <nav>
            <div class="nav-item active">{_icon("home")} Dashboard</div>
            <div class="nav-item">{_icon("clock")} History</div>
            <div class="nav-item">{_icon("microphone")} Microphones</div>
            <div class="nav-item">{_icon("settings")} Settings</div>
            <div class="nav-item">{_icon("info")} About</div>
        </nav>

        <div class="side-card">
            <div class="side-title">Demo Scenario</div>
            {_scenario_links_html(scenario, microphone)}
        </div>

        <div class="side-card">
            <div class="side-title"><span>Microphone</span>{_icon("microphone", 14)}</div>
            {_mic_row("Alienware", "Alienware x15", "MEMS microphone", microphone == "Alienware", scenario)}
            {_mic_row("EarPods", "EarPods USB-C", "Condenser microphone", microphone == "EarPods", scenario)}
            {_mic_row("BlueYeti", "Blue Yeti", "Condenser microphone", microphone == "BlueYeti", scenario)}
            <a class="manage-button" href="#microphones">Manage Microphones</a>
        </div>

        <div class="side-card">
            <div class="side-title">Recording Status</div>
            <div class="live-row"><span><span class="red">●</span> LIVE</span><span class="purple">{_format_duration(elapsed)}</span></div>
        </div>

        <div class="side-card">
            <div class="side-title"><span>System</span>{_icon("activity", 14)}</div>
            {_monitor_row("CPU", 28 + int(8 * np.sin(elapsed * .8)), "%")}
            {_monitor_row("Memory", 41 + int(5 * np.sin(elapsed * .5 + 1.4)), "%")}
            {_monitor_row("FPS", 96, "30")}
        </div>
    </aside>
    """


def _scenario_links_html(active_scenario: str, microphone: str) -> str:
    rows = []
    for scenario in SCENARIOS:
        active = " active" if scenario == active_scenario else ""
        rows.append(
            f'<a class="scenario-link{active}" href="?scenario={quote(scenario)}&mic={quote(microphone)}">'
            f"{html.escape(scenario)}</a>"
        )
    return "".join(rows)


def _mic_row(key: str, name: str, mic_type: str, active: bool, scenario: str) -> str:
    active_class = " active" if active else ""
    return f"""
    <a class="mic-row{active_class}" href="?scenario={quote(scenario)}&mic={quote(key)}">
        <div class="mic-dot"></div>
        <div>
            <div class="mic-name">{html.escape(name)}</div>
            <div class="mic-type">{html.escape(mic_type)}</div>
        </div>
    </a>
    """


def _monitor_row(label: str, percent: int, suffix: str) -> str:
    return f"""
    <div class="monitor-row">
        <span>{html.escape(label)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:{percent}%;"></div></div>
        <span>{percent if suffix == '%' else suffix}{suffix if suffix == '%' else ''}</span>
    </div>
    """


def _metric_html(
    label: str,
    value: str,
    note: str,
    color_class: str,
    icon: str,
    prefix: str = "",
) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{_icon(icon, 22)}</div>
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value {color_class}">{prefix}{html.escape(value)}</div>
        <div class="metric-note">{html.escape(note)}</div>
    </div>
    """


def _panel_title(
    title: str,
    live: bool = False,
    right: str | None = None,
    icon: str | None = None,
) -> str:
    badge = '<span class="live-badge">● Live</span>' if live else ""
    if right:
        badge = f'<span class="small-muted">{html.escape(right)}</span>'
    heading_icon = _icon(icon, 16) if icon else ""
    return f'<div class="panel-title"><span class="panel-heading">{heading_icon}{html.escape(title)}</span>{badge}</div>'


def _voice_bars_html(values: np.ndarray) -> str:
    normalized = values / max(float(values.max()), 0.001)
    bars = "".join(
        f'<div class="voice-bar" style="height:{max(8, int(value * 104))}px;"></div>'
        for value in normalized
    )
    return f'<div class="voice-bars">{bars}</div>'


def _waveform_svg(waveform: np.ndarray) -> str:
    reduced = waveform[:: max(1, len(waveform) // 180)]
    width = 900
    height = 170
    plot_x = 44
    plot_y = 8
    plot_width = 844
    plot_height = 132
    points = []
    for index, value in enumerate(reduced[:180]):
        x = plot_x + index / 179 * plot_width
        y = plot_y + plot_height / 2 - float(value) * (plot_height * 0.43)
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    x_grid = "".join(
        f'<line x1="{plot_x + index * plot_width / 6:.1f}" y1="{plot_y}" x2="{plot_x + index * plot_width / 6:.1f}" y2="{plot_y + plot_height}" />'
        for index in range(7)
    )
    x_labels = "".join(
        f'<text x="{plot_x + index * plot_width / 6:.1f}" y="160" text-anchor="middle">{index * 5}s</text>'
        for index in range(7)
    )
    return f"""
    <svg class="wave-svg" viewBox="0 0 {width} {height}" preserveAspectRatio="none">
        <g stroke="rgba(255,255,255,.09)" stroke-width="1">
            {x_grid}
            <line x1="{plot_x}" y1="{plot_y}" x2="{plot_x + plot_width}" y2="{plot_y}"/>
            <line x1="{plot_x}" y1="{plot_y + plot_height / 2}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height / 2}"/>
            <line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height}"/>
        </g>
        <g fill="#8f8aa5" font-size="11" font-family="Inter, sans-serif">
            <text x="35" y="{plot_y + 4}" text-anchor="end">1.0</text>
            <text x="35" y="{plot_y + plot_height / 2 + 4}" text-anchor="end">0</text>
            <text x="35" y="{plot_y + plot_height + 4}" text-anchor="end">-1.0</text>
            {x_labels}
        </g>
        <polyline points="{polyline}" fill="none" stroke="rgba(236,72,153,.42)" stroke-width="13" opacity=".44" vector-effect="non-scaling-stroke"/>
        <polyline points="{polyline}" fill="none" stroke="#a855f7" stroke-width="2.2" vector-effect="non-scaling-stroke"/>
    </svg>
    """


def _spectrogram_svg(spectrogram: np.ndarray) -> str:
    width = 940
    height = 380
    plot_x = 56
    plot_y = 12
    plot_width = 790
    plot_height = 320
    image_uri = _spectrogram_png_data_uri(spectrogram)

    frequency_labels = ["16k", "8k", "4k", "2k", "1k", "500", "250", "120"]
    y_labels = "".join(
        f'<text x="47" y="{plot_y + index * plot_height / 7 + 4:.1f}" text-anchor="end">{label}</text>'
        for index, label in enumerate(frequency_labels)
    )
    y_grid = "".join(
        f'<line x1="{plot_x}" y1="{plot_y + index * plot_height / 7:.1f}" x2="{plot_x + plot_width}" y2="{plot_y + index * plot_height / 7:.1f}"/>'
        for index in range(8)
    )
    x_labels = "".join(
        f'<text x="{plot_x + index * plot_width / 6:.1f}" y="350" text-anchor="middle">{index * 5}</text>'
        for index in range(7)
    )
    x_grid = "".join(
        f'<line x1="{plot_x + index * plot_width / 6:.1f}" y1="{plot_y}" x2="{plot_x + index * plot_width / 6:.1f}" y2="{plot_y + plot_height}"/>'
        for index in range(7)
    )
    legend_stops = "".join(
        f'<stop offset="{offset}%" stop-color="{color}"/>'
        for offset, color in [(0, "#fff7ad"), (22, "#f59e0b"), (48, "#ec4899"), (72, "#7c3aed"), (100, "#08061b")]
    )
    db_labels = "".join(
        f'<text x="900" y="{plot_y + index * plot_height / 5 + 4:.1f}">{value}</text>'
        for index, value in enumerate(["-20", "-40", "-60", "-80", "-100", "-120"])
    )
    return f"""
    <svg class="spectro-svg" viewBox="0 0 {width} {height}" preserveAspectRatio="none">
        <defs>
            <linearGradient id="dbLegend" x1="0" y1="0" x2="0" y2="1">{legend_stops}</linearGradient>
        </defs>
        <rect x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" rx="7" fill="#030617"/>
        <image href="{image_uri}" x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" preserveAspectRatio="none"/>
        <g stroke="rgba(255,255,255,.08)" stroke-width="1">
            {y_grid}
            {x_grid}
        </g>
        <rect x="870" y="{plot_y}" width="14" height="{plot_height}" rx="4" fill="url(#dbLegend)"/>
        <g fill="#8f8aa5" font-size="11" font-family="Inter, sans-serif">
            {y_labels}
            {x_labels}
            {db_labels}
            <text x="{plot_x + plot_width / 2}" y="372" text-anchor="middle">Time (s)</text>
            <text x="14" y="{plot_y + plot_height / 2}" text-anchor="middle" transform="rotate(-90 14 {plot_y + plot_height / 2})">Frequency (Hz)</text>
            <text x="877" y="10" text-anchor="middle">dB</text>
        </g>
    </svg>
    """


def _spectrogram_png_data_uri(spectrogram: np.ndarray) -> str:
    normalized = np.clip((spectrogram + 120) / 100, 0, 1)
    normalized = np.flipud(normalized)
    stops = np.linspace(0, 1, len(INFERNO))
    red = np.interp(normalized, stops, INFERNO[:, 0]).astype(np.uint8)
    green = np.interp(normalized, stops, INFERNO[:, 1]).astype(np.uint8)
    blue = np.interp(normalized, stops, INFERNO[:, 2]).astype(np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)

    image = Image.fromarray(rgb, "RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _classification_html(snapshot) -> str:
    confidence = int(snapshot.classification.confidence * 100)
    rows = "".join(
        _classification_row(label, value)
        for label, value in _classification_breakdown(snapshot.classification.label, confidence).items()
    )
    return f"""
    <div class="panel">
        {_panel_title("Sound Classification", icon="classification")}
        <div class="classification-label">
            <span>{html.escape(snapshot.classification.label.value.title())}</span>
            <span class="detected-badge">Detected</span>
        </div>
        {rows}
        <div class="small-muted" style="margin-top:8px;">Model: YAMNet | Confidence Threshold: 0.50</div>
    </div>
    """


def _classification_row(label: str, value: int) -> str:
    return f"""
    <div class="classification-row">
        <span class="row-label">{_icon(_classification_icon(label), 13)}{html.escape(label)}</span>
        <div>
            <div class="bar-track"><div class="bar-fill" style="width:{value}%;"></div></div>
            <div class="percent">{value}%</div>
        </div>
    </div>
    """


def _transcript_html(snapshot) -> str:
    text = snapshot.transcript.text or "No speech detected yet."
    return f"""
    <div class="panel">
        {_panel_title("Transcription", live=True, icon="message")}
        <div class="transcript-box">{html.escape(text)}</div>
        <div class="auto-scroll"><span class="purple">●</span> Auto Scroll</div>
    </div>
    """


def _quality_html(snapshot) -> str:
    score = int(snapshot.quality.score)
    rows = "".join(_quality_row(label, value) for label, value in QUALITY_FACTORS[snapshot.quality.level].items())
    return f"""
    <div class="panel">
        {_panel_title("Audio Quality", icon="quality")}
        <div class="quality-ring" style="--score:{score}%;">
            <div>
                <div class="quality-score">{score}</div>
                <div class="quality-word">{html.escape(snapshot.quality.level.title())}</div>
            </div>
        </div>
        <div class="quality-subtitle">Quality Factors</div>
        {rows}
    </div>
    """


def _quality_row(label: str, value: int) -> str:
    return f"""
    <div class="quality-row">
        <span class="row-label">{_icon(_quality_icon(label), 12)}{html.escape(label)}</span>
        <div>
            <div class="bar-track"><div class="bar-fill" style="width:{value}%;"></div></div>
        </div>
    </div>
    """


def _advice_html(snapshot) -> str:
    items = "".join(f"<li>{html.escape(item)}</li>" for item in _recommendations(snapshot))
    return f"""
    <div class="panel">
        {_panel_title("AI Advice", icon="sparkles", right="Powered by Groq")}
        <div class="advice-box">
            <div>{html.escape(snapshot.advice.message)}</div>
            <div style="height:8px;"></div>
            <strong>Recommendations:</strong>
            <ul>{items}</ul>
            <div class="confidence-pill">Confidence: High</div>
        </div>
    </div>
    """


def _classification_icon(label: str) -> str:
    return {
        "Keyboard Typing": "keyboard",
        "Speech": "user",
        "Background Noise": "wave",
        "Silence": "mute",
        "Clap": "activity",
    }.get(label, "activity")


def _quality_icon(label: str) -> str:
    return {
        "Signal Strength": "bars",
        "Noise Level": "wave",
        "Clarity": "sparkles",
        "Stability": "activity",
    }.get(label, "activity")


def _icon(name: str | None, size: int = 16) -> str:
    shapes = {
        "home": '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M9 21v-7h6v7"/>',
        "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
        "microphone": '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M6 11a6 6 0 0 0 12 0M12 17v4M9 21h6"/>',
        "settings": '<circle cx="12" cy="12" r="3"/><path d="M19 13.5v-3l-2-.7-.6-1.4.9-1.9-2.1-2.1-1.9.9-1.4-.6L10.5 3h-3l-.7 2-1.4.6-1.9-.9-2.1 2.1.9 1.9-.6 1.4-2 .7v3l2 .7.6 1.4-.9 1.9 2.1 2.1 1.9-.9 1.4.6.7 2h3l.7-2 1.4-.6 1.9.9 2.1-2.1-.9-1.9.6-1.4z" transform="translate(2.25 0) scale(.8)"/>',
        "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/>',
        "activity": '<path d="M3 12h4l2-6 4 12 2-6h6"/>',
        "wave": '<path d="M3 12h2M7 8v8M10 5v14M13 8v8M16 4v16M19 9v6M22 12h-1"/>',
        "bars": '<path d="M5 20v-6M10 20V9M15 20V4M20 20V12"/>',
        "spectrogram": '<path d="M4 20V8M8 20V4M12 20v-9M16 20V6M20 20v-3"/><path d="M3 20h18"/>',
        "classification": '<circle cx="7" cy="7" r="2"/><circle cx="17" cy="7" r="2"/><circle cx="7" cy="17" r="2"/><circle cx="17" cy="17" r="2"/><path d="M9 7h6M7 9v6M17 9v6M9 17h6"/>',
        "message": '<path d="M4 5h16v12H9l-5 4z"/><path d="M8 9h8M8 13h5"/>',
        "quality": '<circle cx="12" cy="12" r="9"/><path d="M7 14a5 5 0 0 1 10 0M12 12l3-3"/>',
        "sparkles": '<path d="m12 3 1.4 3.6L17 8l-3.6 1.4L12 13l-1.4-3.6L7 8l3.6-1.4zM18 14l.8 2.2L21 17l-2.2.8L18 20l-.8-2.2L15 17l2.2-.8zM5 14l.7 1.7L7.5 16l-1.8.7L5 18.5l-.7-1.8L2.5 16l1.8-.3z"/>',
        "keyboard": '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M6 10h1M10 10h1M14 10h1M18 10h1M6 14h9M17 14h2"/>',
        "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
        "mute": '<path d="M5 10v4h4l5 4V6L9 10zM18 9l4 6M22 9l-4 6"/>',
    }
    shape = shapes.get(name or "", shapes["activity"])
    return (
        f'<svg class="icon" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{shape}</svg>'
    )


def _classification_breakdown(active_label: NoiseLabel, confidence: int) -> dict[str, int]:
    rows = {"Keyboard Typing": 8, "Speech": 18, "Background Noise": 6, "Silence": 2, "Clap": 1}
    if active_label == NoiseLabel.TYPING:
        rows["Keyboard Typing"] = confidence
    elif active_label == NoiseLabel.SILENCE:
        rows["Silence"] = confidence
        rows["Speech"] = 3
    elif active_label == NoiseLabel.SPEECH:
        rows["Speech"] = confidence
        rows["Keyboard Typing"] = 4
    elif active_label == NoiseLabel.BACKGROUND_NOISE:
        rows["Background Noise"] = confidence
    elif active_label == NoiseLabel.CLAP:
        rows["Clap"] = confidence
    return rows


def _recommendations(snapshot) -> list[str]:
    if snapshot.classification.label == NoiseLabel.TYPING:
        return [
            "Reduce typing during speaking",
            "Move closer to the microphone",
            "Use a headset or close-range microphone",
            "Consider a quieter environment",
        ]
    if snapshot.quality.level == "low":
        return [
            "Move closer to the microphone",
            "Increase input gain carefully",
            "Use a less noisy room",
            "Repeat important sentences clearly",
        ]
    if snapshot.classification.label == NoiseLabel.SILENCE:
        return ["Confirm selected microphone", "Check input permission", "Start speaking near the microphone"]
    return ["Current setup is good", "Keep microphone distance stable", "Avoid sudden background noise"]


def _mini_bars(rms: float) -> str:
    strength = float(np.clip(rms * 7, 0.15, 1.0))
    bars = "".join(f'<span style="height:{height * strength:.1f}px"></span>' for height in [7, 12, 17, 22])
    return f'<span class="mini-bars">{bars}</span>'


def _quality_color(level: str) -> str:
    return {"high": "green", "medium": "orange", "low": "red"}[level]


def _signal_strength_db(snapshot) -> float:
    return 20 * np.log10(max(snapshot.audio.rms, 1e-6))


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def _apply_demo_microphone(snapshot, microphone: str) -> None:
    name, microphone_type = MICROPHONES.get(microphone, MICROPHONES["Alienware"])
    snapshot.audio.microphone_name = name
    snapshot.audio.microphone_type = microphone_type


if __name__ == "__main__":
    main()
