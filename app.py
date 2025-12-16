# =============================================================================
# Energy Power Dashboard — v9
# PART 1A — Core Engine + Session + Constants
# =============================================================================
# This file is intentionally verbose and explicit.
# It is designed to be expanded safely to 4000–6000+ lines.
# =============================================================================

# =============================================================================
# 0. PYTHON STANDARD LIBRARIES
# =============================================================================

from __future__ import annotations

import math
import sys
import json
import time
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Any,
    Callable,
    Iterable,
)

# =============================================================================
# 1. THIRD-PARTY LIBRARIES (SAFE & COMMON)
# =============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 2. APPLICATION METADATA
# =============================================================================

APP_NAME: str = "Energy Power Dashboard"
APP_VERSION: str = "9.0.0"
APP_BUILD: str = "core-engine"
APP_AUTHOR: str = "Dashboard Engine"

APP_FULL_TITLE: str = f"{APP_NAME} — v{APP_VERSION}"

# =============================================================================
# 3. STREAMLIT PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title=APP_FULL_TITLE,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 4. GLOBAL CONSTANTS (NUMERIC LIMITS, COLORS, BEHAVIOR)
# =============================================================================

# -----------------------------
# POWER LIMITS
# -----------------------------
DEFAULT_POWER_MIN: float = -150.0
DEFAULT_POWER_MAX: float = 150.0

ABSOLUTE_POWER_MIN: float = -1_000_000.0
ABSOLUTE_POWER_MAX: float = 1_000_000.0

# -----------------------------
# SOC LIMITS
# -----------------------------
DEFAULT_SOC_MIN: float = 0.0
DEFAULT_SOC_MAX: float = 100.0

ABSOLUTE_SOC_MIN: float = 0.0
ABSOLUTE_SOC_MAX: float = 100.0

# -----------------------------
# UNDO / REDO LIMITS
# -----------------------------
UNDO_STACK_LIMIT: int = 100

# -----------------------------
# DATE / TIME
# -----------------------------
DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# -----------------------------
# UI COLORS (USED LATER)
# -----------------------------
COLOR_PRIMARY: str = "#2563EB"
COLOR_SECONDARY: str = "#4F46E5"
COLOR_SUCCESS: str = "#10B981"
COLOR_WARNING: str = "#F59E0B"
COLOR_DANGER: str = "#EF4444"
COLOR_MUTED: str = "#6B7280"
COLOR_BG_LIGHT: str = "#F8FAFC"

# =============================================================================
# 5. SESSION STATE KEYS (STRICT & CENTRALIZED)
# =============================================================================

# -----------------------------
# CORE DATA
# -----------------------------
SS_DATAFRAME: str = "ss_dataframe"

# -----------------------------
# INPUT STATE
# -----------------------------
SS_INPUT_POWER: str = "ss_input_power"
SS_INPUT_SOC: str = "ss_input_soc"
SS_INPUT_STATUS: str = "ss_input_status"

# -----------------------------
# LIMIT SETTINGS
# -----------------------------
SS_POWER_MIN: str = "ss_power_min"
SS_POWER_MAX: str = "ss_power_max"
SS_SOC_MIN: str = "ss_soc_min"
SS_SOC_MAX: str = "ss_soc_max"

# -----------------------------
# HISTORY (UNDO / REDO)
# -----------------------------
SS_UNDO_STACK: str = "ss_undo_stack"
SS_REDO_STACK: str = "ss_redo_stack"

# -----------------------------
# INTERNAL FLAGS
# -----------------------------
SS_APP_INITIALIZED: str = "ss_app_initialized"
SS_LAST_ACTION_ID: str = "ss_last_action_id"

# =============================================================================
# 6. SESSION INITIALIZATION UTILITIES
# =============================================================================

def init_session_key(key: str, default_value: Any) -> None:
    """
    Initialize a Streamlit session_state key safely.
    This function guarantees idempotent initialization.
    """
    if key not in st.session_state:
        st.session_state[key] = default_value


def init_all_session_state() -> None:
    """
    Initialize ALL session state used by the application.
    This function MUST be called exactly once.
    """

    # -------------------------
    # Data storage
    # -------------------------
    init_session_key(
        SS_DATAFRAME,
        pd.DataFrame(
            columns=[
                "Date Time",
                "Power",
                "SOC (%)",
            ]
        ),
    )

    # -------------------------
    # Input buffers
    # -------------------------
    init_session_key(SS_INPUT_POWER, "")
    init_session_key(SS_INPUT_SOC, "")
    init_session_key(SS_INPUT_STATUS, None)

    # -------------------------
    # Limits
    # -------------------------
    init_session_key(SS_POWER_MIN, DEFAULT_POWER_MIN)
    init_session_key(SS_POWER_MAX, DEFAULT_POWER_MAX)
    init_session_key(SS_SOC_MIN, DEFAULT_SOC_MIN)
    init_session_key(SS_SOC_MAX, DEFAULT_SOC_MAX)

    # -------------------------
    # Undo / Redo
    # -------------------------
    init_session_key(SS_UNDO_STACK, [])
    init_session_key(SS_REDO_STACK, [])

    # -------------------------
    # Internal flags
    # -------------------------
    init_session_key(SS_APP_INITIALIZED, True)
    init_session_key(SS_LAST_ACTION_ID, None)


# =============================================================================
# 7. CALL SESSION INITIALIZATION (ONCE)
# =============================================================================

init_all_session_state()

# =============================================================================
# 8. UNDO / REDO CORE ENGINE
# =============================================================================

def _snapshot_dataframe() -> pd.DataFrame:
    """
    Create a deep copy snapshot of the current dataframe.
    """
    return st.session_state[SS_DATAFRAME].copy(deep=True)


def push_undo_snapshot(reason: Optional[str] = None) -> None:
    """
    Push the current dataframe into the undo stack.
    Clears the redo stack automatically.
    """
    undo_stack: List[pd.DataFrame] = st.session_state[SS_UNDO_STACK]
    undo_stack.insert(0, _snapshot_dataframe())

    # Enforce stack size limit
    if len(undo_stack) > UNDO_STACK_LIMIT:
        undo_stack.pop()

    st.session_state[SS_REDO_STACK] = []
    st.session_state[SS_LAST_ACTION_ID] = str(uuid.uuid4())


def undo_action() -> bool:
    """
    Perform undo operation.
    Returns True if successful.
    """
    undo_stack = st.session_state[SS_UNDO_STACK]
    redo_stack = st.session_state[SS_REDO_STACK]

    if not undo_stack:
        return False

    redo_stack.insert(0, _snapshot_dataframe())
    st.session_state[SS_DATAFRAME] = undo_stack.pop(0)
    return True


def redo_action() -> bool:
    """
    Perform redo operation.
    Returns True if successful.
    """
    redo_stack = st.session_state[SS_REDO_STACK]
    undo_stack = st.session_state[SS_UNDO_STACK]

    if not redo_stack:
        return False

    undo_stack.insert(0, _snapshot_dataframe())
    st.session_state[SS_DATAFRAME] = redo_stack.pop(0)
    return True


# =============================================================================
# 9. VALIDATION HELPERS (REUSED EVERYWHERE)
# =============================================================================

def validate_numeric(
    value: Any,
    min_value: float,
    max_value: float,
) -> Tuple[bool, Optional[str]]:
    """
    Validate numeric input within bounds.
    """
    try:
        val = float(value)
    except (TypeError, ValueError):
        return False, "Value must be numeric"

    if val < min_value or val > max_value:
        return False, f"Value must be between {min_value} and {max_value}"

    return True, None


def validate_power_input(value: Any) -> Tuple[bool, Optional[str]]:
    return validate_numeric(
        value,
        st.session_state[SS_POWER_MIN],
        st.session_state[SS_POWER_MAX],
    )


def validate_soc_input(value: Any) -> Tuple[bool, Optional[str]]:
    return validate_numeric(
        value,
        st.session_state[SS_SOC_MIN],
        st.session_state[SS_SOC_MAX],
    )


# =============================================================================
# 10. DATA INSERTION ENGINE (NO UI)
# =============================================================================

def append_power_soc_record(
    power: float,
    soc: float,
    timestamp: Optional[datetime] = None,
) -> None:
    """
    Append a Power + SOC record into the dataframe.
    This function NEVER touches UI.
    """

    if timestamp is None:
        timestamp = datetime.now()

    push_undo_snapshot(reason="append_row")

    new_row = {
        "Date Time": timestamp.strftime(DATETIME_FORMAT),
        "Power": float(power),
        "SOC (%)": float(soc),
    }

    st.session_state[SS_DATAFRAME] = pd.concat(
        [
            st.session_state[SS_DATAFRAME],
            pd.DataFrame([new_row]),
        ],
        ignore_index=True,
    )


# =============================================================================
# 11. SAFE DATAFRAME ACCESSORS
# =============================================================================

def get_dataframe() -> pd.DataFrame:
    """
    Always use this accessor instead of touching session_state directly.
    """
    return st.session_state[SS_DATAFRAME]


def set_dataframe(df: pd.DataFrame) -> None:
    """
    Replace the dataframe safely with undo support.
    """
    push_undo_snapshot(reason="set_dataframe")
    st.session_state[SS_DATAFRAME] = df.copy(deep=True)


# =============================================================================
# END OF PART 1A
# DO NOT ADD UI CODE BELOW THIS LINE
# =============================================================================


# =============================================================================
# PART 1B — GLOBAL CSS (Liquid / Modern / Dashboard)
# =============================================================================

def inject_global_css() -> None:
    """
    Inject global CSS for modern dashboard UI.
    This CSS is carefully written to be Streamlit-safe.
    """

    st.markdown(
        """
<style>

/* =============================================================================
   ROOT VARIABLES
   ============================================================================= */
:root {
    --bg-main: #f8fafc;
    --bg-card: rgba(255, 255, 255, 0.85);
    --bg-glass: rgba(255, 255, 255, 0.65);

    --border-soft: #e5e7eb;
    --border-muted: #d1d5db;

    --text-main: #0f172a;
    --text-muted: #64748b;

    --primary: #2563eb;
    --secondary: #4f46e5;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;

    --radius-lg: 18px;
    --radius-md: 12px;
    --radius-sm: 8px;

    --shadow-soft: 0 8px 24px rgba(15, 23, 42, 0.06);
    --shadow-glass: 0 8px 32px rgba(15, 23, 42, 0.08);

    --transition-fast: 120ms ease;
    --transition-normal: 220ms ease;
}

/* =============================================================================
   GLOBAL PAGE
   ============================================================================= */
html, body {
    background-color: var(--bg-main);
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}

/* =============================================================================
   TYPOGRAPHY
   ============================================================================= */
h1, h2, h3, h4 {
    color: var(--text-main);
    letter-spacing: 0.2px;
}

p, label, span {
    color: var(--text-main);
}

/* Muted text */
.caption, .stCaption {
    color: var(--text-muted) !important;
    font-size: 0.9rem;
}

/* =============================================================================
   HEADER
   ============================================================================= */
header[data-testid="stHeader"] {
    background: transparent;
}

/* =============================================================================
   SIDEBAR
   ============================================================================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(248, 250, 252, 0.95),
        rgba(241, 245, 249, 0.95)
    );
    border-right: 1px solid var(--border-soft);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 1.05rem;
}

/* =============================================================================
   CARD CONTAINERS
   ============================================================================= */
.section-card {
    background: var(--bg-card);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);

    border: 1px solid var(--border-soft);
    border-radius: var(--radius-lg);

    padding: 18px 20px;
    margin-bottom: 18px;

    box-shadow: var(--shadow-soft);
    transition: box-shadow var(--transition-normal),
                transform var(--transition-fast);
}

.section-card:hover {
    box-shadow: var(--shadow-glass);
    transform: translateY(-1px);
}

/* =============================================================================
   BUTTONS
   ============================================================================= */
.stButton > button {
    background: linear-gradient(
        180deg,
        #ffffff,
        #f1f5f9
    );
    border: 1px solid var(--border-muted);
    border-radius: var(--radius-md);
    padding: 0.45rem 0.9rem;
    font-weight: 500;
    transition: all var(--transition-fast);
}

.stButton > button:hover {
    background: linear-gradient(
        180deg,
        #ffffff,
        #e5e7eb
    );
    border-color: var(--primary);
}

.stButton > button:active {
    transform: scale(0.97);
}

/* =============================================================================
   INPUTS
   ============================================================================= */
.stTextInput input,
.stNumberInput input,
.stSelectbox select,
.stTextArea textarea {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-muted) !important;
    padding: 0.55rem 0.65rem !important;
}

.stTextInput input:focus,
.stNumberInput input:focus,
.stSelectbox select:focus,
.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.3) !important;
}

/* =============================================================================
   CHECKBOX & SLIDER
   ============================================================================= */
.stCheckbox label {
    font-weight: 500;
}

.stSlider > div {
    padding-top: 0.5rem;
}

/* =============================================================================
   DATA EDITOR / TABLE
   ============================================================================= */
[data-testid="stDataEditor"] {
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-soft);
    overflow: hidden;
    box-shadow: var(--shadow-soft);
}

[data-testid="stDataEditor"] table {
    border-collapse: separate;
    border-spacing: 0;
}

/* Header */
[data-testid="stDataEditor"] thead tr th {
    background: linear-gradient(
        180deg,
        #f8fafc,
        #f1f5f9
    );
    font-weight: 600;
}

/* Rows hover */
[data-testid="stDataEditor"] tbody tr:hover {
    background-color: rgba(37, 99, 235, 0.04);
}

/* =============================================================================
   TABS
   ============================================================================= */
button[data-baseweb="tab"] {
    border-radius: var(--radius-md);
    padding: 0.4rem 0.9rem;
    font-weight: 500;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(37, 99, 235, 0.1);
    color: var(--primary);
}

/* =============================================================================
   METRICS
   ============================================================================= */
[data-testid="stMetric"] {
    background: var(--bg-glass);
    border-radius: var(--radius-md);
    padding: 14px;
    border: 1px solid var(--border-soft);
    box-shadow: var(--shadow-soft);
}

/* =============================================================================
   TOAST / ALERTS
   ============================================================================= */
div[data-testid="stToast"] {
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-glass);
}

.stAlert {
    border-radius: var(--radius-md);
}

/* =============================================================================
   PLOTLY CHART CONTAINER
   ============================================================================= */
.js-plotly-plot {
    border-radius: var(--radius-lg);
    overflow: hidden;
}

/* =============================================================================
   FOOTER
   ============================================================================= */
footer {
    visibility: hidden;
}

</style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# INJECT CSS IMMEDIATELY
# =============================================================================

inject_global_css()

# =============================================================================
# END OF PART 1B
# =============================================================================


# =============================================================================
# PART 1C — Sidebar + Navigation + Utilities
# =============================================================================

# =============================================================================
# 1. SIDEBAR — SETTINGS & INFO
# =============================================================================

def render_sidebar() -> None:
    """
    Render the left sidebar.
    Contains global settings, limits, and app info.
    """

    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Settings")
        st.caption("Global configuration & limits")

        # -----------------------------
        # Power bounds
        # -----------------------------
        st.markdown("### ⚡ Power Limits")

        st.number_input(
            "Min Power",
            key=SS_POWER_MIN,
            step=1.0,
        )
        st.number_input(
            "Max Power",
            key=SS_POWER_MAX,
            step=1.0,
        )

        # -----------------------------
        # SOC bounds
        # -----------------------------
        st.markdown("### 🔋 SOC Limits")

        st.number_input(
            "Min SOC (%)",
            key=SS_SOC_MIN,
            min_value=ABSOLUTE_SOC_MIN,
            max_value=ABSOLUTE_SOC_MAX,
            step=1.0,
        )
        st.number_input(
            "Max SOC (%)",
            key=SS_SOC_MAX,
            min_value=ABSOLUTE_SOC_MIN,
            max_value=ABSOLUTE_SOC_MAX,
            step=1.0,
        )

        st.markdown("---")

        # -----------------------------
        # Undo / Redo status
        # -----------------------------
        st.markdown("### 🕒 History")

        undo_count = len(st.session_state[SS_UNDO_STACK])
        redo_count = len(st.session_state[SS_REDO_STACK])

        c1, c2 = st.columns(2)
        c1.metric("Undo", undo_count)
        c2.metric("Redo", redo_count)

        st.markdown("---")

        # -----------------------------
        # App info
        # -----------------------------
        st.markdown("### ℹ️ App Info")
        st.write(f"**Version:** {APP_VERSION}")
        st.write(f"**Build:** {APP_BUILD}")

        st.caption("Energy analytics dashboard\nBuilt with Streamlit")

# =============================================================================
# 2. TOP HEADER / TITLE BAR
# =============================================================================

def render_header() -> None:
    """
    Render the top header/title section.
    """

    st.markdown(
        f"""
        <div class="section-card">
            <h2>⚡ Energy Power Dashboard</h2>
            <p style="margin:0;color:var(--text-muted);">
                Version {APP_VERSION} — Power & SOC Analytics Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# 3. TAB NAVIGATION
# =============================================================================

def render_tabs() -> Dict[str, Any]:
    """
    Create and return all main tabs.
    Keep this centralized so new tabs can be added safely.
    """

    tab_labels = [
        "📥 Input & Table",
        "📈 Visualization",
        "📊 Statistics",
        "💾 Export",
        "❓ How to use",
    ]

    tabs = st.tabs(tab_labels)

    return {
        "input_table": tabs[0],
        "visualization": tabs[1],
        "statistics": tabs[2],
        "export": tabs[3],
        "help": tabs[4],
    }

# =============================================================================
# 4. GENERAL UI UTILITIES
# =============================================================================

def show_status_message() -> None:
    """
    Display status messages from input or actions.
    Automatically clears after display.
    """

    status = st.session_state.get(SS_INPUT_STATUS)
    if not status:
        return

    level, message = status

    if level == "success":
        st.success(message)
    elif level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.info(message)

    # Clear after showing once
    st.session_state[SS_INPUT_STATUS] = None


def section_start(title: str, caption: Optional[str] = None) -> None:
    """
    Start a styled section card.
    """
    html = f"<div class='section-card'><h3>{title}</h3>"
    if caption:
        html += f"<p style='color:var(--text-muted);margin-top:-6px;'>{caption}</p>"
    st.markdown(html, unsafe_allow_html=True)


def section_end() -> None:
    """
    End a styled section card.
    """
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 5. DATAFRAME UTILITIES
# =============================================================================

def get_row_count() -> int:
    """
    Return number of records in dataframe.
    """
    return len(st.session_state[SS_DATAFRAME])


def get_latest_record() -> Optional[Dict[str, Any]]:
    """
    Return the latest row as dict.
    """
    df = st.session_state[SS_DATAFRAME]
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def ensure_dataframe_columns() -> None:
    """
    Ensure required columns always exist.
    Prevents corruption when users manipulate table.
    """

    required = ["Date Time", "Power", "SOC (%)"]
    df = st.session_state[SS_DATAFRAME]

    missing = [c for c in required if c not in df.columns]
    if missing:
        push_undo_snapshot(reason="fix_columns")
        for col in missing:
            df[col] = ""

        st.session_state[SS_DATAFRAME] = df

# =============================================================================
# 6. INPUT VALIDATION UTILITIES
# =============================================================================

def normalize_bounds() -> None:
    """
    Ensure min/max bounds are valid (min <= max).
    """
    if st.session_state[SS_POWER_MIN] > st.session_state[SS_POWER_MAX]:
        st.session_state[SS_POWER_MIN], st.session_state[SS_POWER_MAX] = (
            st.session_state[SS_POWER_MAX],
            st.session_state[SS_POWER_MIN],
        )

    if st.session_state[SS_SOC_MIN] > st.session_state[SS_SOC_MAX]:
        st.session_state[SS_SOC_MIN], st.session_state[SS_SOC_MAX] = (
            st.session_state[SS_SOC_MAX],
            st.session_state[SS_SOC_MIN],
        )

# =============================================================================
# 7. APPLICATION BOOTSTRAP
# =============================================================================

def app_bootstrap() -> Dict[str, Any]:
    """
    Bootstrap the application layout.
    Must be called before rendering tab content.
    """

    normalize_bounds()
    ensure_dataframe_columns()

    render_sidebar()
    render_header()

    return render_tabs()

# =============================================================================
# END OF PART 1C
# =============================================================================

# =============================================================================
# PART 2A — 📥 Input Engine (Smooth, ENTER-based)
# =============================================================================
# This section handles:
# - Power + SOC input
# - ENTER-key submission
# - Validation & feedback
# - Undo snapshot integration
# - UX smoothing (no UI jump)
# =============================================================================

# =============================================================================
# 1. INPUT STATE RESET UTILITIES
# =============================================================================

def clear_input_buffers() -> None:
    """
    Clear input buffers after successful insert.
    """
    st.session_state[SS_INPUT_POWER] = ""
    st.session_state[SS_INPUT_SOC] = ""


def set_input_status(level: str, message: str) -> None:
    """
    Set a one-time status message for input actions.
    """
    st.session_state[SS_INPUT_STATUS] = (level, message)


# =============================================================================
# 2. INPUT SUBMISSION ENGINE (NO UI)
# =============================================================================

def submit_power_soc_from_inputs() -> None:
    """
    Core submission logic triggered by ENTER key.
    This function:
    - Reads input buffers
    - Validates Power & SOC
    - Appends record if valid
    - Sets status message
    """

    power_raw = st.session_state.get(SS_INPUT_POWER, "").strip()
    soc_raw = st.session_state.get(SS_INPUT_SOC, "").strip()

    # ---------------------------------
    # Empty check
    # ---------------------------------
    if not power_raw or not soc_raw:
        set_input_status(
            "warning",
            "Please enter BOTH Power and SOC before pressing ENTER",
        )
        return

    # ---------------------------------
    # Validate Power
    # ---------------------------------
    valid_power, err_power = validate_power_input(power_raw)
    if not valid_power:
        set_input_status("error", f"Power error: {err_power}")
        return

    # ---------------------------------
    # Validate SOC
    # ---------------------------------
    valid_soc, err_soc = validate_soc_input(soc_raw)
    if not valid_soc:
        set_input_status("error", f"SOC error: {err_soc}")
        return

    # ---------------------------------
    # Append record
    # ---------------------------------
    append_power_soc_record(
        power=float(power_raw),
        soc=float(soc_raw),
    )

    clear_input_buffers()
    set_input_status("success", "Power & SOC recorded successfully")


# =============================================================================
# 3. INPUT PANEL UI
# =============================================================================

def render_input_panel() -> None:
    """
    Render the smooth input panel UI.
    """

    section_start(
        "📥 Input Power & SOC",
        "Type values and press ENTER to record data",
    )

    # ---------------------------------
    # Input fields
    # ---------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.text_input(
            "⚡ Power",
            key=SS_INPUT_POWER,
            placeholder="e.g. 75",
            help=f"Allowed range: {st.session_state[SS_POWER_MIN]} to {st.session_state[SS_POWER_MAX]}",
            on_change=submit_power_soc_from_inputs,
        )

    with col2:
        st.text_input(
            "🔋 SOC (%)",
            key=SS_INPUT_SOC,
            placeholder="e.g. 45",
            help=f"Allowed range: {st.session_state[SS_SOC_MIN]} to {st.session_state[SS_SOC_MAX]}",
            on_change=submit_power_soc_from_inputs,
        )

    # ---------------------------------
    # Action buttons (Undo / Redo)
    # ---------------------------------

    section_end()


# =============================================================================
# 4. INPUT TAB ENTRY POINT
# =============================================================================

def render_input_tab() -> None:
    """
    Entry point for 📥 Input & Table tab.
    PART 2B+ will extend this tab.
    """

    render_input_panel()

    # NOTE:
    # Table tools + editable table will be added in:
    # - PART 2B (Undo/Redo enhancements)
    # - PART 2C (Data Table Tools)
    # - PART 2D (Editable Table UX polish)


# =============================================================================
# END OF PART 2A
# =============================================================================

# =============================================================================
# PART 2B — Undo / Redo + Validation + Feedback
# =============================================================================
# This section improves:
# - Undo/Redo user experience
# - Validation clarity
# - Feedback consistency
# - State awareness for users
# =============================================================================

# =============================================================================
# 1. UNDO / REDO STATUS HELPERS
# =============================================================================

def get_undo_count() -> int:
    """
    Return number of undo steps available.
    """
    return len(st.session_state.get(SS_UNDO_STACK, []))


def get_redo_count() -> int:
    """
    Return number of redo steps available.
    """
    return len(st.session_state.get(SS_REDO_STACK, []))


def can_undo() -> bool:
    """
    Check if undo is possible.
    """
    return get_undo_count() > 0


def can_redo() -> bool:
    """
    Check if redo is possible.
    """
    return get_redo_count() > 0


# =============================================================================
# 2. VALIDATION MESSAGE NORMALIZATION
# =============================================================================

def normalize_validation_message(raw: str) -> str:
    """
    Normalize validation messages to be user-friendly and consistent.
    """
    if raw is None:
        return ""

    msg = raw.strip()

    # Standardize wording
    replacements = {
        "Value must be numeric": "Please enter a numeric value",
        "Value must be between": "Allowed range is",
    }

    for src, dst in replacements.items():
        if src in msg:
            msg = msg.replace(src, dst)

    return msg


def set_validation_error(message: str) -> None:
    """
    Set a standardized validation error.
    """
    set_input_status("error", normalize_validation_message(message))


def set_validation_warning(message: str) -> None:
    """
    Set a standardized validation warning.
    """
    set_input_status("warning", normalize_validation_message(message))


def set_validation_success(message: str) -> None:
    """
    Set a standardized validation success message.
    """
    set_input_status("success", message)


# =============================================================================
# 3. ENHANCED SUBMISSION WRAPPER (OPTIONAL USE)
# =============================================================================

def submit_with_feedback() -> None:
    """
    Wrapper around submit_power_soc_from_inputs()
    Adds refined feedback control.
    """

    before_rows = get_row_count()

    submit_power_soc_from_inputs()

    after_rows = get_row_count()

    if after_rows > before_rows:
        set_validation_success("New record added")
    else:
        # If no row added, keep previous status
        pass


# =============================================================================
# 4. UNDO / REDO CONTROL PANEL (UI)
# =============================================================================

def render_undo_redo_panel() -> None:
    section_start(
        "🕒 History Controls",
        "Undo and redo changes safely",
    )

    u1, u2, u3, u4 = st.columns([0.18, 0.18, 0.32, 0.32])

    with u1:
        if st.button(
            "↶ Undo",
            key="btn_undo_history",   # ✅ UNIQUE KEY
            disabled=not can_undo(),
            use_container_width=True,
        ):
            if undo_action():
                set_validation_success("Undo successful")
            else:
                set_validation_warning("Nothing to undo")

    with u2:
        if st.button(
            "↷ Redo",
            key="btn_redo_history",   # ✅ UNIQUE KEY
            disabled=not can_redo(),
            use_container_width=True,
        ):
            if redo_action():
                set_validation_success("Redo successful")
            else:
                set_validation_warning("Nothing to redo")

    with u3:
        st.metric("Undo steps", get_undo_count())

    with u4:
        st.metric("Redo steps", get_redo_count())

    section_end()


# =============================================================================
# 5. INPUT FEEDBACK PANEL (DEDICATED)
# =============================================================================

def render_feedback_panel() -> None:
    """
    Dedicated feedback panel for validation & system messages.
    Prevents UI jump and repeated alerts.
    """

    status = st.session_state.get(SS_INPUT_STATUS)
    if not status:
        return

    level, message = status

    section_start("ℹ️ Status")

    if level == "success":
        st.success(message)
    elif level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.info(message)

    section_end()

    # Clear after rendering once
    st.session_state[SS_INPUT_STATUS] = None


# =============================================================================
# 6. EXTEND INPUT TAB WITH HISTORY + FEEDBACK
# =============================================================================

def render_input_tab_with_history() -> None:
    """
    Extended version of render_input_tab().
    Combines:
    - Input panel (PART 2A)
    - Undo/Redo panel
    - Feedback panel
    """

    render_input_panel()
    render_undo_redo_panel()
    render_feedback_panel()


# =============================================================================
# END OF PART 2B
# =============================================================================

# =============================================================================
# PART 2C — 🛠 Data Table Tools (Dropdown, Excel-like)
# =============================================================================
# This section provides:
# - Centralized dropdown table tools
# - Excel-like operations
# - Full undo/redo integration
# - Safe dataframe manipulation
# =============================================================================

# =============================================================================
# 1. TABLE TOOL ENUMERATION
# =============================================================================

TABLE_TOOL_OPTIONS: List[str] = [
    "None",
    "Insert row",
    "Delete row",
    "Insert column",
    "Delete column",
    "Duplicate column",
    "Rename column",
    "Move column left",
    "Move column right",
    "Merge columns",
]

# =============================================================================
# 2. LOW-LEVEL DATAFRAME OPERATIONS (NO UI)
# =============================================================================

def insert_row_at(position: Optional[int] = None) -> None:
    """
    Insert a new empty row at a specific position.
    """
    df = get_dataframe()

    push_undo_snapshot(reason="insert_row")

    new_row = {
        "Date Time": datetime.now().strftime(DATETIME_FORMAT),
        "Power": 0.0,
        "SOC (%)": 0.0,
    }

    if position is None or position >= len(df):
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        top = df.iloc[:position]
        bottom = df.iloc[position:]
        df = pd.concat([top, pd.DataFrame([new_row]), bottom], ignore_index=True)

    set_dataframe(df)


def delete_row_at(index: int) -> None:
    """
    Delete a row at a specific index.
    """
    df = get_dataframe()

    if index < 0 or index >= len(df):
        set_validation_error("Row index out of range")
        return

    push_undo_snapshot(reason="delete_row")
    df = df.drop(df.index[index]).reset_index(drop=True)
    set_dataframe(df)


def insert_column(name: str, default_value: Any = "") -> None:
    """
    Insert a new column with default value.
    """
    df = get_dataframe()

    if not name:
        set_validation_error("Column name cannot be empty")
        return

    if name in df.columns:
        set_validation_error("Column already exists")
        return

    push_undo_snapshot(reason="insert_column")
    df[name] = default_value
    set_dataframe(df)


def delete_column(name: str) -> None:
    """
    Delete an existing column.
    """
    df = get_dataframe()

    if name not in df.columns:
        set_validation_error("Column not found")
        return

    if name in ["Date Time"]:
        set_validation_warning("Core columns cannot be deleted")
        return

    push_undo_snapshot(reason="delete_column")
    df = df.drop(columns=[name])
    set_dataframe(df)


def duplicate_column(source: str, target: str) -> None:
    """
    Duplicate a column.
    """
    df = get_dataframe()

    if source not in df.columns:
        set_validation_error("Source column not found")
        return

    if not target:
        set_validation_error("New column name required")
        return

    if target in df.columns:
        set_validation_error("Target column already exists")
        return

    push_undo_snapshot(reason="duplicate_column")
    df[target] = df[source]
    set_dataframe(df)


def rename_column(old: str, new: str) -> None:
    """
    Rename a column.
    """
    df = get_dataframe()

    if old not in df.columns:
        set_validation_error("Column not found")
        return

    if not new:
        set_validation_error("New column name required")
        return

    if new in df.columns:
        set_validation_error("Target column already exists")
        return

    push_undo_snapshot(reason="rename_column")
    df = df.rename(columns={old: new})
    set_dataframe(df)


def move_column(name: str, direction: str) -> None:
    """
    Move a column left or right.
    """
    df = get_dataframe()
    cols = list(df.columns)

    if name not in cols:
        set_validation_error("Column not found")
        return

    idx = cols.index(name)

    if direction == "left" and idx > 0:
        cols[idx], cols[idx - 1] = cols[idx - 1], cols[idx]
    elif direction == "right" and idx < len(cols) - 1:
        cols[idx], cols[idx + 1] = cols[idx + 1], cols[idx]
    else:
        set_validation_warning("Cannot move column further")
        return

    push_undo_snapshot(reason="move_column")
    set_dataframe(df[cols])


def merge_columns(
    columns: List[str],
    new_name: str,
    separator: str = " ",
) -> None:
    """
    Merge multiple columns into one.
    """
    df = get_dataframe()

    if len(columns) < 2:
        set_validation_error("Select at least two columns to merge")
        return

    if not new_name:
        set_validation_error("New column name required")
        return

    for c in columns:
        if c not in df.columns:
            set_validation_error(f"Column not found: {c}")
            return

    push_undo_snapshot(reason="merge_columns")

    df[new_name] = df[columns].astype(str).agg(separator.join, axis=1)
    df = df.drop(columns=columns)

    set_dataframe(df)

# =============================================================================
# 3. TABLE TOOLS UI (DROPDOWN)
# =============================================================================

def render_table_tools() -> None:
    """
    Render Excel-like table tools using a single dropdown.
    MUST be rendered INSIDE the Input & Table tab.
    """

    section_start(
        "🛠 Data Table Tools",
        "Excel-like operations for managing table structure",
    )

    df = get_dataframe()

    tool = st.selectbox(
        "Choose an action",
        TABLE_TOOL_OPTIONS,
        key="table_tool_select",
    )

    # ---------------------------------------------------------
    # INSERT ROW
    # ---------------------------------------------------------
    if tool == "Insert row":
        pos = st.number_input(
            "Insert position (0 = bottom)",
            min_value=0,
            max_value=len(df),
            value=len(df),
        )
        if st.button("Apply"):
            insert_row_at(position=int(pos))
            set_validation_success("Row inserted")

    # ---------------------------------------------------------
    # DELETE ROW
    # ---------------------------------------------------------
    elif tool == "Delete row":
        if df.empty:
            st.info("Table is empty")
        else:
            idx = st.number_input(
                "Row index",
                min_value=0,
                max_value=len(df) - 1,
                value=0,
            )
            if st.button("Apply"):
                delete_row_at(int(idx))
                set_validation_success("Row deleted")

    # ---------------------------------------------------------
    # INSERT COLUMN
    # ---------------------------------------------------------
    elif tool == "Insert column":
        name = st.text_input("New column name")
        default = st.text_input("Default value", value="")
        if st.button("Apply"):
            insert_column(name, default)
            set_validation_success("Column inserted")

    # ---------------------------------------------------------
    # DELETE COLUMN
    # ---------------------------------------------------------
    elif tool == "Delete column":
        col = st.selectbox("Select column", df.columns)
        if st.button("Apply"):
            delete_column(col)
            set_validation_success("Column deleted")

    # ---------------------------------------------------------
    # DUPLICATE COLUMN
    # ---------------------------------------------------------
    elif tool == "Duplicate column":
        src = st.selectbox("Source column", df.columns)
        new = st.text_input("New column name")
        if st.button("Apply"):
            duplicate_column(src, new)
            set_validation_success("Column duplicated")

    # ---------------------------------------------------------
    # RENAME COLUMN
    # ---------------------------------------------------------
    elif tool == "Rename column":
        src = st.selectbox("Column to rename", df.columns)
        new = st.text_input("New column name")
        if st.button("Apply"):
            rename_column(src, new)
            set_validation_success("Column renamed")

    # ---------------------------------------------------------
    # MOVE COLUMN
    # ---------------------------------------------------------
    elif tool in ["Move column left", "Move column right"]:
        col = st.selectbox("Select column", df.columns)
        direction = "left" if "left" in tool else "right"
        if st.button("Apply"):
            move_column(col, direction)
            set_validation_success("Column moved")

    # ---------------------------------------------------------
    # MERGE COLUMNS
    # ---------------------------------------------------------
    elif tool == "Merge columns":
        cols = st.multiselect("Columns to merge", df.columns)
        new = st.text_input("New column name")
        sep = st.text_input("Separator", value=" ")
        if st.button("Apply"):
            merge_columns(cols, new, sep)
            set_validation_success("Columns merged")

    section_end()

# =============================================================================
# END OF PART 2C
# =============================================================================

# =============================================================================
# PART 2D — 🧾 Editable Table UX Polish
# =============================================================================
# This section provides:
# - Professional editable table
# - Column-level validation
# - Smooth UX (no flicker)
# - Undo-safe edits
# - Clear separation from table tools
# =============================================================================

# =============================================================================
# 1. COLUMN CONFIGURATION
# =============================================================================

def get_table_column_config() -> Dict[str, Any]:
    """
    Define column behavior for the editable table.
    """
    return {
        "Date Time": st.column_config.TextColumn(
            "Date Time",
            disabled=True,
            help="Automatically generated timestamp",
        ),
        "Power": st.column_config.NumberColumn(
            "Power",
            step=1.0,
            help=f"Allowed range: {st.session_state[SS_POWER_MIN]} to {st.session_state[SS_POWER_MAX]}",
        ),
        "SOC (%)": st.column_config.NumberColumn(
            "SOC (%)",
            min_value=st.session_state[SS_SOC_MIN],
            max_value=st.session_state[SS_SOC_MAX],
            step=1.0,
            help="Battery State of Charge (%)",
        ),
    }


# =============================================================================
# 2. DATAFRAME CHANGE DETECTION
# =============================================================================

def dataframe_changed(df_old: pd.DataFrame, df_new: pd.DataFrame) -> bool:
    """
    Compare two dataframes safely.
    """
    if df_old.shape != df_new.shape:
        return True

    try:
        return not df_old.equals(df_new)
    except Exception:
        return True


# =============================================================================
# 3. VALIDATE EDITED DATAFRAME
# =============================================================================

def validate_edited_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate edited dataframe values.
    """
    for idx, row in df.iterrows():
        # Validate Power
        valid_p, err_p = validate_power_input(row.get("Power"))
        if not valid_p:
            return False, f"Row {idx}: Power invalid ({err_p})"

        # Validate SOC
        valid_s, err_s = validate_soc_input(row.get("SOC (%)"))
        if not valid_s:
            return False, f"Row {idx}: SOC invalid ({err_s})"

    return True, None


# =============================================================================
# 4. EDITABLE TABLE RENDERER
# =============================================================================

def render_editable_table() -> None:
    """
    Render the polished editable table.
    """

    section_start(
        "🧾 Editable Data Table",
        "Edit values directly. Core columns are protected.",
    )

    df_current = get_dataframe()

    if df_current.empty:
        st.info("No data yet. Add Power & SOC using the input panel above.")
        section_end()
        return

    edited_df = st.data_editor(
        df_current,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config=get_table_column_config(),
        key="editable_data_table",
    )

    # ---------------------------------------------------------
    # Detect & handle changes
    # ---------------------------------------------------------
    if dataframe_changed(df_current, edited_df):

        valid, error_msg = validate_edited_dataframe(edited_df)

        if not valid:
            set_validation_error(error_msg)
        else:
            push_undo_snapshot(reason="edit_table")
            set_dataframe(edited_df)
            set_validation_success("Table updated successfully")

    section_end()


# =============================================================================
# 5. FINAL INPUT TAB COMPOSITION
# =============================================================================

def render_input_table_tab() -> None:
    """
    Final composition for 📥 Input & Table tab.
    This replaces earlier render_input_tab calls.
    """

    # Input panel + history + feedback
    render_input_tab_with_history()

    # Table tools (dropdown)
    render_table_tools()

    # Editable table
    render_editable_table()


# =============================================================================
# END OF PART 2D
# =============================================================================

# =============================================================================
# PART 3A — 📈 Visualization Core
# =============================================================================
# This section provides:
# - Visualization state management
# - Chart presets & themes
# - Shared layout & styling helpers
# - Focus/dim logic
# - Data preparation for charts
# =============================================================================

# =============================================================================
# 1. VISUALIZATION SESSION STATE KEYS
# =============================================================================

SS_VIZ_FOCUS = "ss_viz_focus"
SS_VIZ_PRESET = "ss_viz_preset"
SS_VIZ_THEME = "ss_viz_theme"

SS_VIZ_SHOW_GRID = "ss_viz_show_grid"
SS_VIZ_SMOOTH_LINES = "ss_viz_smooth_lines"
SS_VIZ_AUTO_SCALE = "ss_viz_auto_scale"

SS_VIZ_LINE_WIDTH = "ss_viz_line_width"
SS_VIZ_MARKER_SIZE = "ss_viz_marker_size"
SS_VIZ_AREA_OPACITY = "ss_viz_area_opacity"

# =============================================================================
# 2. INIT VISUALIZATION STATE
# =============================================================================

def init_visualization_state() -> None:
    """
    Initialize all visualization-related session state.
    """
    init_session_key(SS_VIZ_FOCUS, "None")
    init_session_key(SS_VIZ_PRESET, "Default")
    init_session_key(SS_VIZ_THEME, "Light")

    init_session_key(SS_VIZ_SHOW_GRID, True)
    init_session_key(SS_VIZ_SMOOTH_LINES, False)
    init_session_key(SS_VIZ_AUTO_SCALE, True)

    init_session_key(SS_VIZ_LINE_WIDTH, 2)
    init_session_key(SS_VIZ_MARKER_SIZE, 6)
    init_session_key(SS_VIZ_AREA_OPACITY, 0.55)


# Initialize visualization state immediately
init_visualization_state()

# =============================================================================
# 3. VISUALIZATION PRESETS
# =============================================================================

def apply_visualization_preset(preset: str) -> None:
    """
    Apply predefined visualization presets.
    """
    if preset == "Default":
        st.session_state[SS_VIZ_SHOW_GRID] = True
        st.session_state[SS_VIZ_SMOOTH_LINES] = False
        st.session_state[SS_VIZ_LINE_WIDTH] = 2
        st.session_state[SS_VIZ_MARKER_SIZE] = 6
        st.session_state[SS_VIZ_AREA_OPACITY] = 0.55

    elif preset == "Analysis":
        st.session_state[SS_VIZ_SHOW_GRID] = True
        st.session_state[SS_VIZ_SMOOTH_LINES] = False
        st.session_state[SS_VIZ_LINE_WIDTH] = 3
        st.session_state[SS_VIZ_MARKER_SIZE] = 4
        st.session_state[SS_VIZ_AREA_OPACITY] = 0.45

    elif preset == "Presentation":
        st.session_state[SS_VIZ_SHOW_GRID] = False
        st.session_state[SS_VIZ_SMOOTH_LINES] = True
        st.session_state[SS_VIZ_LINE_WIDTH] = 4
        st.session_state[SS_VIZ_MARKER_SIZE] = 0
        st.session_state[SS_VIZ_AREA_OPACITY] = 0.65


# =============================================================================
# 4. DATA PREPARATION FOR VISUALIZATION
# =============================================================================

def get_visualization_dataframe() -> pd.DataFrame:
    """
    Prepare and return dataframe for visualization.
    """
    df = get_dataframe().copy()

    if df.empty:
        return df

    df["Date Time"] = pd.to_datetime(df["Date Time"], errors="coerce")
    df["Power"] = pd.to_numeric(df["Power"], errors="coerce")
    df["SOC (%)"] = pd.to_numeric(df["SOC (%)"], errors="coerce")

    df = df.dropna(subset=["Date Time"])
    df = df.sort_values("Date Time")

    return df


# =============================================================================
# 5. FOCUS & DIMMING LOGIC
# =============================================================================

def get_focus_opacity(chart_name: str) -> float:
    """
    Return opacity based on focus selection.
    """
    focus = st.session_state[SS_VIZ_FOCUS]

    if focus == "None":
        return 1.0

    return 1.0 if focus.lower() in chart_name.lower() else 0.25


# =============================================================================
# 6. BASE CHART LAYOUT BUILDER
# =============================================================================

def apply_base_layout(
    fig: go.Figure,
    title: str,
    yaxis_title: Optional[str] = None,
    height: int = 320,
) -> None:
    """
    Apply consistent layout styling to all charts.
    """

    show_grid = st.session_state[SS_VIZ_SHOW_GRID]
    auto_scale = st.session_state[SS_VIZ_AUTO_SCALE]

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=show_grid,
            gridcolor="rgba(0,0,0,0.12)",
        ),
        yaxis=dict(
            title=yaxis_title,
            showgrid=show_grid,
            gridcolor="rgba(0,0,0,0.12)",
        ),
    )

    if not auto_scale:
        fig.update_yaxes(autorange=False)


# =============================================================================
# 7. LINE STYLE HELPERS
# =============================================================================

def get_line_shape() -> str:
    """
    Return line shape based on smooth setting.
    """
    return "spline" if st.session_state[SS_VIZ_SMOOTH_LINES] else "linear"


def get_marker_mode() -> str:
    """
    Return plotly mode string for markers.
    """
    return (
        "lines"
        if st.session_state[SS_VIZ_MARKER_SIZE] == 0
        else "lines+markers"
    )


# =============================================================================
# 8. VISUALIZATION TAB ENTRY POINT
# =============================================================================

def render_visualization_tab() -> None:
    """
    Entry point for 📈 Visualization tab.
    PART 3B / 3C / 3D will extend this.
    """

    section_start(
        "📈 Power & SOC Visualization",
        "Interactive charts with presets and focus control",
    )

    df = get_visualization_dataframe()

    if df.empty:
        st.info("No data available for visualization.")
        section_end()
        return

    # ---------------------------------------------------------
    # Controls (shared)
    # ---------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.selectbox(
            "Focus chart",
            ["None", "Line", "Step", "Area", "Bar", "SOC", "Compare"],
            key=SS_VIZ_FOCUS,
        )

    with c2:
        preset = st.selectbox(
            "View preset",
            ["Default", "Analysis", "Presentation"],
            key=SS_VIZ_PRESET,
        )
        apply_visualization_preset(preset)

    with c3:
        st.checkbox(
            "Smooth lines",
            key=SS_VIZ_SMOOTH_LINES,
        )

    with c4:
        st.checkbox(
            "Show grid",
            key=SS_VIZ_SHOW_GRID,
        )

    section_end()

    # NOTE:
    # Actual charts are added in:
    # - PART 3B → Power charts
    # - PART 3C → SOC + comparison charts
    # - PART 3D → Advanced controls & interactions


# =============================================================================
# END OF PART 3A
# =============================================================================

# =============================================================================
# PART 3B — ⚡ Power Charts (Line, Step, Area, Bar)
# =============================================================================
# This section provides:
# - Four professional Power charts
# - Focus-based dimming
# - Theme & preset integration
# - Clean 2x2 dashboard layout
# =============================================================================

# =============================================================================
# 1. POWER CHART BUILDERS
# =============================================================================

def build_power_line_chart(df: pd.DataFrame) -> go.Figure:
    """
    Power — Line chart
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date Time"],
            y=df["Power"],
            mode=get_marker_mode(),
            line=dict(
                width=st.session_state[SS_VIZ_LINE_WIDTH],
                shape=get_line_shape(),
                color=COLOR_PRIMARY,
            ),
            marker=dict(
                size=st.session_state[SS_VIZ_MARKER_SIZE],
            ),
            opacity=get_focus_opacity("line"),
        )
    )

    apply_base_layout(fig, "Power — Line", yaxis_title="Power")
    return fig


def build_power_step_chart(df: pd.DataFrame) -> go.Figure:
    """
    Power — Step chart
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date Time"],
            y=df["Power"],
            mode="lines",
            line=dict(
                width=st.session_state[SS_VIZ_LINE_WIDTH],
                shape="hv",
                color=COLOR_SECONDARY,
            ),
            opacity=get_focus_opacity("step"),
        )
    )

    apply_base_layout(fig, "Power — Step", yaxis_title="Power")
    return fig


def build_power_area_chart(df: pd.DataFrame) -> go.Figure:
    """
    Power — Area chart
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date Time"],
            y=df["Power"],
            fill="tozeroy",
            mode="lines",
            line=dict(
                width=st.session_state[SS_VIZ_LINE_WIDTH],
                color="#0EA5E9",
            ),
            opacity=st.session_state[SS_VIZ_AREA_OPACITY]
            * get_focus_opacity("area"),
        )
    )

    apply_base_layout(fig, "Power — Area", yaxis_title="Power")
    return fig


def build_power_bar_chart(df: pd.DataFrame) -> go.Figure:
    """
    Power — Bar chart
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["Date Time"],
            y=df["Power"],
            marker_color=COLOR_SUCCESS,
            opacity=get_focus_opacity("bar"),
        )
    )

    apply_base_layout(fig, "Power — Bar", yaxis_title="Power")
    return fig


# =============================================================================
# 2. POWER CHART GRID RENDERER
# =============================================================================

def render_power_charts() -> None:
    """
    Render all Power charts in a clean 2x2 grid.
    """

    df = get_visualization_dataframe()
    if df.empty:
        return

    section_start(
        "⚡ Power Analytics",
        "Four complementary views of Power over time",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            build_power_line_chart(df),
            use_container_width=True,
        )
        st.plotly_chart(
            build_power_area_chart(df),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            build_power_step_chart(df),
            use_container_width=True,
        )
        st.plotly_chart(
            build_power_bar_chart(df),
            use_container_width=True,
        )

    section_end()


# =============================================================================
# 3. EXTEND VISUALIZATION TAB
# =============================================================================

def render_visualization_tab_with_power() -> None:
    """
    Extend visualization tab with Power charts.
    """

    render_visualization_tab()
    render_power_charts()

    # NOTE:
    # SOC charts + comparison are added in PART 3C
    # Advanced controls in PART 3D


# =============================================================================
# END OF PART 3B
# =============================================================================

# =============================================================================
# PART 3C — 🔋 SOC Charts + Dual Axis + Presets
# =============================================================================
# This section provides:
# - SOC (%) time-series chart
# - SOC safety zones
# - Power vs SOC dual-axis comparison
# - Preset & focus integration
# =============================================================================

# =============================================================================
# 1. SOC CHART BUILDER
# =============================================================================

def build_soc_chart(df: pd.DataFrame) -> go.Figure:
    """
    SOC (%) over time chart with safety zones.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date Time"],
            y=df["SOC (%)"],
            mode=get_marker_mode(),
            line=dict(
                width=st.session_state[SS_VIZ_LINE_WIDTH],
                shape=get_line_shape(),
                color=COLOR_WARNING,
            ),
            marker=dict(
                size=st.session_state[SS_VIZ_MARKER_SIZE],
            ),
            opacity=get_focus_opacity("soc"),
        )
    )

    # -----------------------------
    # SOC safety zones
    # -----------------------------
    fig.add_hrect(
        y0=0,
        y1=20,
        fillcolor=COLOR_DANGER,
        opacity=0.08,
        line_width=0,
    )
    fig.add_hrect(
        y0=20,
        y1=80,
        fillcolor=COLOR_SUCCESS,
        opacity=0.05,
        line_width=0,
    )
    fig.add_hrect(
        y0=80,
        y1=100,
        fillcolor=COLOR_WARNING,
        opacity=0.08,
        line_width=0,
    )

    apply_base_layout(
        fig,
        "SOC (%) Over Time",
        yaxis_title="SOC (%)",
        height=360,
    )

    return fig


# =============================================================================
# 2. POWER VS SOC (DUAL AXIS)
# =============================================================================

def build_power_soc_dual_axis(df: pd.DataFrame) -> go.Figure:
    """
    Dual-axis comparison of Power and SOC.
    """
    fig = go.Figure()

    # Power (Left axis)
    fig.add_trace(
        go.Scatter(
            x=df["Date Time"],
            y=df["Power"],
            name="Power",
            yaxis="y1",
            mode="lines",
            line=dict(
                width=st.session_state[SS_VIZ_LINE_WIDTH],
                color=COLOR_PRIMARY,
            ),
            opacity=get_focus_opacity("compare"),
        )
    )

    # SOC (Right axis)
    fig.add_trace(
        go.Scatter(
            x=df["Date Time"],
            y=df["SOC (%)"],
            name="SOC (%)",
            yaxis="y2",
            mode="lines",
            line=dict(
                width=st.session_state[SS_VIZ_LINE_WIDTH],
                color=COLOR_WARNING,
            ),
            opacity=get_focus_opacity("compare"),
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=st.session_state[SS_VIZ_SHOW_GRID],
            gridcolor="rgba(0,0,0,0.12)",
        ),
        yaxis=dict(
            title="Power",
            showgrid=st.session_state[SS_VIZ_SHOW_GRID],
            gridcolor="rgba(0,0,0,0.12)",
        ),
        yaxis2=dict(
            title="SOC (%)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )

    return fig


# =============================================================================
# 3. SOC SECTION RENDERER
# =============================================================================

def render_soc_charts() -> None:
    """
    Render SOC analytics section.
    """

    df = get_visualization_dataframe()
    if df.empty:
        return

    section_start(
        "🔋 SOC Analytics",
        "Battery state-of-charge over time with safety zones",
    )

    st.plotly_chart(
        build_soc_chart(df),
        use_container_width=True,
    )

    section_end()


# =============================================================================
# 4. COMPARISON SECTION RENDERER
# =============================================================================

def render_power_soc_comparison() -> None:
    """
    Render Power vs SOC dual-axis comparison.
    """

    df = get_visualization_dataframe()
    if df.empty:
        return

    section_start(
        "🔀 Power vs SOC Comparison",
        "Dual-axis comparison for correlation analysis",
    )

    st.plotly_chart(
        build_power_soc_dual_axis(df),
        use_container_width=True,
    )

    section_end()


# =============================================================================
# 5. EXTEND VISUALIZATION TAB (FINAL)
# =============================================================================

def render_visualization_tab_full() -> None:
    """
    Final visualization tab composition.
    """

    render_visualization_tab()
    render_power_charts()
    render_soc_charts()
    render_power_soc_comparison()

    # NOTE:
    # PART 3D will add:
    # - Advanced controls
    # - Time range filters
    # - Rolling averages
    # - Trend overlays


# =============================================================================
# END OF PART 3C
# =============================================================================

# =============================================================================
# PART 3D — 📈 Advanced Chart Controls
# =============================================================================
# This section adds:
# - Time range filtering
# - Rolling averages
# - Trend lines
# - Overlay toggles
# - Preset-aware behavior
# =============================================================================

# =============================================================================
# 1. ADVANCED VISUALIZATION STATE
# =============================================================================

SS_VIZ_TIME_MODE = "ss_viz_time_mode"
SS_VIZ_LAST_N = "ss_viz_last_n"
SS_VIZ_START_DATE = "ss_viz_start_date"
SS_VIZ_END_DATE = "ss_viz_end_date"

SS_VIZ_SHOW_ROLLING = "ss_viz_show_rolling"
SS_VIZ_ROLLING_WINDOW = "ss_viz_rolling_window"

SS_VIZ_SHOW_TREND = "ss_viz_show_trend"

def init_advanced_viz_state() -> None:
    init_session_key(SS_VIZ_TIME_MODE, "All")
    init_session_key(SS_VIZ_LAST_N, 50)
    init_session_key(SS_VIZ_START_DATE, None)
    init_session_key(SS_VIZ_END_DATE, None)

    init_session_key(SS_VIZ_SHOW_ROLLING, False)
    init_session_key(SS_VIZ_ROLLING_WINDOW, 5)

    init_session_key(SS_VIZ_SHOW_TREND, False)

init_advanced_viz_state()

# =============================================================================
# 2. TIME RANGE FILTER
# =============================================================================

def apply_time_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe by selected time range.
    """
    if df.empty:
        return df

    mode = st.session_state[SS_VIZ_TIME_MODE]

    if mode == "Last N":
        n = st.session_state[SS_VIZ_LAST_N]
        return df.tail(n)

    if mode == "Custom":
        start = st.session_state[SS_VIZ_START_DATE]
        end = st.session_state[SS_VIZ_END_DATE]

        if start:
            df = df[df["Date Time"] >= pd.to_datetime(start)]
        if end:
            df = df[df["Date Time"] <= pd.to_datetime(end)]

        return df

    return df  # All


# =============================================================================
# 3. ROLLING AVERAGE OVERLAY
# =============================================================================

def add_rolling_average(
    fig: go.Figure,
    df: pd.DataFrame,
    column: str,
    color: str,
) -> None:
    """
    Add rolling average overlay.
    """
    window = st.session_state[SS_VIZ_ROLLING_WINDOW]
    if window <= 1:
        return

    roll = df[column].rolling(window=window).mean()

    fig.add_trace(
        go.Scatter(
            x=df["Date Time"],
            y=roll,
            mode="lines",
            line=dict(
                width=2,
                dash="dash",
                color=color,
            ),
            name=f"{column} ({window}-pt avg)",
        )
    )


# =============================================================================
# 4. TREND LINE OVERLAY
# =============================================================================

def add_trend_line(
    fig: go.Figure,
    df: pd.DataFrame,
    column: str,
    color: str,
) -> None:
    """
    Add linear regression trend line.
    """
    if len(df) < 2:
        return

    x = np.arange(len(df))
    y = df[column].values

    coef = np.polyfit(x, y, 1)
    trend = coef[0] * x + coef[1]

    fig.add_trace(
        go.Scatter(
            x=df["Date Time"],
            y=trend,
            mode="lines",
            line=dict(
                width=2,
                dash="dot",
                color=color,
            ),
            name=f"{column} trend",
        )
    )


# =============================================================================
# 5. ADVANCED CONTROL PANEL UI
# =============================================================================

def render_advanced_controls() -> None:
    """
    Render advanced chart controls panel.
    """

    section_start(
        "🎛 Advanced Chart Controls",
        "Filter time range, add rolling averages and trends",
    )

    # -----------------------------
    # Time range
    # -----------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.selectbox(
            "Time range",
            ["All", "Last N", "Custom"],
            key=SS_VIZ_TIME_MODE,
        )

    with c2:
        if st.session_state[SS_VIZ_TIME_MODE] == "Last N":
            st.number_input(
                "Last N points",
                min_value=5,
                step=5,
                key=SS_VIZ_LAST_N,
            )

    with c3:
        if st.session_state[SS_VIZ_TIME_MODE] == "Custom":
            st.date_input(
                "Start date",
                key=SS_VIZ_START_DATE,
            )
            st.date_input(
                "End date",
                key=SS_VIZ_END_DATE,
            )

    st.markdown("---")

    # -----------------------------
    # Overlays
    # -----------------------------
    o1, o2, o3 = st.columns(3)

    with o1:
        st.checkbox(
            "Show rolling average",
            key=SS_VIZ_SHOW_ROLLING,
        )

    with o2:
        if st.session_state[SS_VIZ_SHOW_ROLLING]:
            st.number_input(
                "Rolling window",
                min_value=2,
                step=1,
                key=SS_VIZ_ROLLING_WINDOW,
            )

    with o3:
        st.checkbox(
            "Show trend line",
            key=SS_VIZ_SHOW_TREND,
        )

    section_end()


# =============================================================================
# 6. INTEGRATION HELPERS
# =============================================================================

def prepare_filtered_viz_dataframe() -> pd.DataFrame:
    """
    Prepare dataframe for charts with time filtering applied.
    """
    df = get_visualization_dataframe()
    return apply_time_filter(df)


# =============================================================================
# 7. OVERRIDES FOR POWER & SOC CHARTS (NON-BREAKING)
# =============================================================================

def build_power_line_chart_advanced(df: pd.DataFrame) -> go.Figure:
    fig = build_power_line_chart(df)

    if st.session_state[SS_VIZ_SHOW_ROLLING]:
        add_rolling_average(fig, df, "Power", COLOR_PRIMARY)

    if st.session_state[SS_VIZ_SHOW_TREND]:
        add_trend_line(fig, df, "Power", COLOR_PRIMARY)

    return fig


def build_soc_chart_advanced(df: pd.DataFrame) -> go.Figure:
    fig = build_soc_chart(df)

    if st.session_state[SS_VIZ_SHOW_ROLLING]:
        add_rolling_average(fig, df, "SOC (%)", COLOR_WARNING)

    if st.session_state[SS_VIZ_SHOW_TREND]:
        add_trend_line(fig, df, "SOC (%)", COLOR_WARNING)

    return fig


# =============================================================================
# 8. FINAL VISUALIZATION TAB (WITH ADVANCED CONTROLS)
# =============================================================================

def render_visualization_tab_advanced() -> None:
    """
    Final Visualization tab with advanced controls enabled.
    """

    render_visualization_tab()
    render_advanced_controls()

    df = prepare_filtered_viz_dataframe()
    if df.empty:
        return

    # Replace charts with advanced versions
    section_start("⚡ Power Analytics (Advanced)")
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            build_power_line_chart_advanced(df),
            use_container_width=True,
        )
        st.plotly_chart(
            build_power_area_chart(df),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            build_power_step_chart(df),
            use_container_width=True,
        )
        st.plotly_chart(
            build_power_bar_chart(df),
            use_container_width=True,
        )

    section_end()

    section_start("🔋 SOC Analytics (Advanced)")
    st.plotly_chart(
        build_soc_chart_advanced(df),
        use_container_width=True,
    )
    section_end()

    section_start("🔀 Power vs SOC (Advanced)")
    st.plotly_chart(
        build_power_soc_dual_axis(df),
        use_container_width=True,
    )
    section_end()


# =============================================================================
# END OF PART 3D
# =============================================================================

# =============================================================================
# PART 4A — 📊 Statistics Core
# =============================================================================
# This section provides:
# - Statistical state
# - Numeric preparation
# - Descriptive statistics engine
# - Outlier detection helpers
# - Distribution analysis helpers
# =============================================================================

# =============================================================================
# 1. STATISTICS SESSION STATE
# =============================================================================

SS_STATS_COLUMN = "ss_stats_column"
SS_STATS_METHOD = "ss_stats_method"

def init_statistics_state() -> None:
    """
    Initialize statistics-related session state.
    """
    init_session_key(SS_STATS_COLUMN, "Power")
    init_session_key(SS_STATS_METHOD, "IQR")

init_statistics_state()

# =============================================================================
# 2. DATA PREPARATION FOR STATISTICS
# =============================================================================

def get_statistics_dataframe() -> pd.DataFrame:
    """
    Prepare dataframe for statistical analysis.
    Ensures numeric conversion and safety.
    """
    df = get_dataframe().copy()

    if df.empty:
        return df

    for col in ["Power", "SOC (%)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Power", "SOC (%)"], how="all")
    return df

# =============================================================================
# 3. DESCRIPTIVE STATISTICS ENGINE
# =============================================================================

def compute_descriptive_statistics(
    series: pd.Series,
) -> Dict[str, float]:
    """
    Compute core descriptive statistics.
    """
    return {
        "Count": int(series.count()),
        "Mean": float(series.mean()),
        "Median": float(series.median()),
        "Std Dev": float(series.std(ddof=1)),
        "Min": float(series.min()),
        "Max": float(series.max()),
        "Skewness": float(series.skew()),
        "Kurtosis": float(series.kurtosis()),
    }

# =============================================================================
# 4. PERCENTILES
# =============================================================================

def compute_percentiles(
    series: pd.Series,
    percentiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Compute percentile table.
    """
    if percentiles is None:
        percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    data = {
        f"{int(p*100)}%": series.quantile(p)
        for p in percentiles
    }

    return pd.DataFrame.from_dict(
        data,
        orient="index",
        columns=["Value"],
    )

# =============================================================================
# 5. OUTLIER DETECTION — IQR
# =============================================================================

def detect_outliers_iqr(series: pd.Series) -> pd.DataFrame:
    """
    Detect outliers using IQR method.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = series[(series < lower) | (series > upper)]

    return pd.DataFrame({
        "Index": outliers.index,
        "Value": outliers.values,
        "Method": "IQR",
    })

# =============================================================================
# 6. OUTLIER DETECTION — Z-SCORE
# =============================================================================

def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect outliers using Z-score method.
    """
    mean = series.mean()
    std = series.std(ddof=1)

    if std == 0 or math.isnan(std):
        return pd.DataFrame(columns=["Index", "Value", "Method"])

    z_scores = (series - mean) / std
    mask = z_scores.abs() > threshold

    return pd.DataFrame({
        "Index": series[mask].index,
        "Value": series[mask].values,
        "Method": "Z-Score",
    })

# =============================================================================
# 7. DISTRIBUTION SUMMARY
# =============================================================================

def compute_distribution_summary(series: pd.Series) -> Dict[str, Any]:
    """
    Compute distribution diagnostics.
    """
    return {
        "Normality (Skew ~0)": abs(series.skew()) < 0.5,
        "Heavy Tails (Kurtosis)": series.kurtosis(),
        "Variance": series.var(ddof=1),
        "IQR": series.quantile(0.75) - series.quantile(0.25),
    }

# =============================================================================
# 8. MASTER STATISTICS FUNCTION
# =============================================================================

def compute_full_statistics(
    column: str,
) -> Dict[str, Any]:
    """
    Compute full statistics bundle for selected column.
    """
    df = get_statistics_dataframe()

    if df.empty or column not in df.columns:
        return {}

    series = df[column].dropna()

    stats = {
        "descriptive": compute_descriptive_statistics(series),
        "percentiles": compute_percentiles(series),
        "distribution": compute_distribution_summary(series),
    }

    method = st.session_state[SS_STATS_METHOD]
    if method == "IQR":
        stats["outliers"] = detect_outliers_iqr(series)
    else:
        stats["outliers"] = detect_outliers_zscore(series)

    return stats

# =============================================================================
# END OF PART 4A
# =============================================================================
# =============================================================================
# PART 4B — 📊 Descriptive + Distribution + Outliers (UI) — FIXED
# =============================================================================

# =============================================================================
# 1. BOX PLOT BUILDER (NUMERIC SAFE)
# =============================================================================

def build_box_plot(series: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=series,
            name=title,
            boxpoints="outliers",
            marker_color=COLOR_PRIMARY,
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        title=title,
        showlegend=False,
    )

    return fig


# =============================================================================
# 2. DISTRIBUTION SUMMARY (SAFE)
# =============================================================================

def render_distribution_summary(dist: Dict[str, Any]) -> None:
    section_start("📐 Distribution Diagnostics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Variance", round(dist.get("Variance", 0), 3))
    c2.metric("IQR", round(dist.get("IQR", 0), 3))
    c3.metric("Skew ~ Normal", str(dist.get("Normality (Skew ~0)", False)))
    c4.metric("Kurtosis", round(dist.get("Heavy Tails (Kurtosis)", 0), 3))

    section_end()


# =============================================================================
# 3. OUTLIER TABLE
# =============================================================================

def render_outlier_table(outliers: pd.DataFrame) -> None:
    section_start("🚨 Outliers")

    if outliers is None or outliers.empty:
        st.success("No outliers detected")
    else:
        st.dataframe(outliers.reset_index(drop=True), use_container_width=True)

    section_end()


# =============================================================================
# 4. DESCRIPTIVE STATS (SAFE)
# =============================================================================

def render_descriptive_stats(desc: Dict[str, float]) -> None:
    section_start("📊 Descriptive Statistics")

    cols = st.columns(4)
    for i, (k, v) in enumerate(desc.items()):
        cols[i % 4].metric(k, round(v, 3))

    section_end()


# =============================================================================
# 5. PERCENTILE TABLE
# =============================================================================

def render_percentile_table(pct_df: pd.DataFrame) -> None:
    section_start("📈 Percentiles")
    st.table(pct_df)
    section_end()


# =============================================================================
# 6. STATISTICS TAB ENTRY POINT (FIXED)
# =============================================================================

def render_statistics_tab() -> None:
    section_start(
        "📊 Statistics Analysis",
        "Descriptive statistics, distribution & outlier detection",
    )

    df = get_statistics_dataframe()
    if df.empty:
        st.info("No data available for statistics.")
        section_end()
        return

    # ---------------------------------------------------------
    # ✅ ONLY NUMERIC COLUMNS (CRITICAL FIX)
    # ---------------------------------------------------------
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    if not numeric_cols:
        st.warning("No numeric columns available for statistics.")
        section_end()
        return

    c1, c2 = st.columns(2)

    with c1:
        column = st.selectbox(
            "Select numeric column",
            numeric_cols,
            key=SS_STATS_COLUMN,
        )

    with c2:
        st.selectbox(
            "Outlier method",
            ["IQR", "Z-Score"],
            key=SS_STATS_METHOD,
        )

    section_end()

    # ---------------------------------------------------------
    # Compute statistics (SAFE ENGINE)
    # ---------------------------------------------------------
    stats = compute_full_statistics(column)
    if not stats:
        st.warning("Unable to compute statistics.")
        return

    # ---------------------------------------------------------
    # Render sections
    # ---------------------------------------------------------
    render_descriptive_stats(stats["descriptive"])

    c3, c4 = st.columns(2)
    with c3:
        render_percentile_table(stats["percentiles"])
    with c4:
        render_distribution_summary(stats["distribution"])

    # ---------------------------------------------------------
    # Box plot + outliers (NUMERIC SAFE)
    # ---------------------------------------------------------
    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if series.empty:
        st.info("No numeric data available for box plot.")
        return

    section_start("📦 Box Plot")
    st.plotly_chart(
        build_box_plot(series, f"{column} Distribution"),
        use_container_width=True,
    )
    section_end()

    render_outlier_table(stats["outliers"])


# =============================================================================
# END OF PART 4B — FIXED
# =============================================================================


# =============================================================================
# PART 4C — 📊 Rolling / Trend / Correlation
# =============================================================================
# This section provides:
# - Rolling statistics analysis
# - Trend (linear regression) analysis
# - Power vs SOC correlation
# - Professional interpretation helpers
# =============================================================================

# =============================================================================
# 1. ROLLING STATISTICS ENGINE
# =============================================================================

def compute_rolling_statistics(
    series: pd.Series,
    window: int,
) -> pd.DataFrame:
    """
    Compute rolling mean and rolling std.
    """
    return pd.DataFrame({
        "Rolling Mean": series.rolling(window).mean(),
        "Rolling Std": series.rolling(window).std(ddof=1),
    })


# =============================================================================
# 2. TREND ANALYSIS ENGINE
# =============================================================================

def compute_linear_trend(series: pd.Series) -> Dict[str, float]:
    """
    Compute linear trend statistics.
    """
    if len(series) < 2:
        return {}

    x = np.arange(len(series))
    y = series.values

    slope, intercept = np.polyfit(x, y, 1)

    # R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "Slope": slope,
        "Intercept": intercept,
        "R-squared": r_squared,
    }


# =============================================================================
# 3. CORRELATION ENGINE
# =============================================================================

def compute_power_soc_correlation(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute correlation between Power and SOC.
    """
    if "Power" not in df.columns or "SOC (%)" not in df.columns:
        return {}

    clean = df[["Power", "SOC (%)"]].dropna()
    if len(clean) < 2:
        return {}

    corr = clean["Power"].corr(clean["SOC (%)"])

    return {
        "Correlation": corr,
        "Strength": interpret_correlation_strength(corr),
    }


def interpret_correlation_strength(value: float) -> str:
    """
    Human-readable interpretation of correlation.
    """
    v = abs(value)

    if v < 0.2:
        return "Very weak"
    elif v < 0.4:
        return "Weak"
    elif v < 0.6:
        return "Moderate"
    elif v < 0.8:
        return "Strong"
    else:
        return "Very strong"


# =============================================================================
# 4. ROLLING STATISTICS UI
# =============================================================================

def render_rolling_statistics(series: pd.Series) -> None:
    """
    Render rolling statistics UI.
    """

    section_start("🔁 Rolling Statistics")

    window = st.slider(
        "Rolling window size",
        min_value=2,
        max_value=max(5, len(series) // 2),
        value=5,
    )

    roll_df = compute_rolling_statistics(series, window)

    c1, c2 = st.columns(2)

    with c1:
        st.line_chart(roll_df["Rolling Mean"])

    with c2:
        st.line_chart(roll_df["Rolling Std"])

    section_end()


# =============================================================================
# 5. TREND ANALYSIS UI
# =============================================================================

def render_trend_analysis(series: pd.Series) -> None:
    """
    Render trend analysis UI.
    """

    trend = compute_linear_trend(series)

    section_start("📉 Trend Analysis")

    if not trend:
        st.info("Not enough data for trend analysis.")
        section_end()
        return

    c1, c2, c3 = st.columns(3)

    c1.metric("Slope", round(trend["Slope"], 6))
    c2.metric("Intercept", round(trend["Intercept"], 3))
    c3.metric("R²", round(trend["R-squared"], 4))

    # Interpretation
    if trend["Slope"] > 0:
        st.success("Overall upward trend detected")
    elif trend["Slope"] < 0:
        st.warning("Overall downward trend detected")
    else:
        st.info("No clear trend detected")

    section_end()


# =============================================================================
# 6. CORRELATION UI
# =============================================================================

def render_correlation_analysis(df: pd.DataFrame) -> None:
    """
    Render Power vs SOC correlation analysis.
    """

    corr = compute_power_soc_correlation(df)

    section_start("🔗 Power ↔ SOC Correlation")

    if not corr:
        st.info("Not enough data for correlation analysis.")
        section_end()
        return

    c1, c2 = st.columns(2)

    c1.metric("Correlation", round(corr["Correlation"], 4))
    c2.metric("Strength", corr["Strength"])

    if corr["Correlation"] > 0:
        st.success("As Power increases, SOC tends to increase")
    else:
        st.warning("As Power increases, SOC tends to decrease")

    section_end()


# =============================================================================
# 7. EXTEND STATISTICS TAB (ADVANCED)
# =============================================================================

def render_statistics_tab_advanced() -> None:
    """
    Full statistics tab with advanced analysis.
    """

    render_statistics_tab()

    df = get_statistics_dataframe()
    if df.empty:
        return

    column = st.session_state[SS_STATS_COLUMN]
    series = df[column].dropna()

    if series.empty:
        return

    render_rolling_statistics(series)
    render_trend_analysis(series)
    render_correlation_analysis(df)


# =============================================================================
# END OF PART 4C
# =============================================================================

# =============================================================================
# PART 4D — 💾 Export (CSV / Excel / Chart Images)
# =============================================================================
# This section provides:
# - CSV export
# - Excel export (data + statistics)
# - Chart image export (PNG)
# =============================================================================

# =============================================================================
# 1. DATA EXPORT HELPERS
# =============================================================================

def export_dataframe_csv(df: pd.DataFrame) -> bytes:
    """
    Export dataframe to CSV bytes.
    """
    return df.to_csv(index=False).encode("utf-8")


def export_dataframe_excel(
    df: pd.DataFrame,
    stats: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Export dataframe and optional statistics to Excel.
    """
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")

        if stats:
            # Descriptive
            desc_df = pd.DataFrame.from_dict(
                stats["descriptive"], orient="index", columns=["Value"]
            )
            desc_df.to_excel(writer, sheet_name="Descriptive Stats")

            # Percentiles
            stats["percentiles"].to_excel(
                writer, sheet_name="Percentiles"
            )

            # Outliers
            if not stats["outliers"].empty:
                stats["outliers"].to_excel(
                    writer, index=False, sheet_name="Outliers"
                )

    return buffer.getvalue()


# =============================================================================
# 2. CHART EXPORT HELPERS
# =============================================================================

def export_chart_png(fig: go.Figure) -> bytes:
    """
    Export Plotly figure as PNG bytes.
    """
    return fig.to_image(format="png", scale=2)


# =============================================================================
# 3. EXPORT UI — DATA
# =============================================================================

def render_export_data_section() -> None:
    """
    Render data export section.
    """

    section_start(
        "📄 Export Data",
        "Download raw Power & SOC data",
    )

    df = get_dataframe()
    if df.empty:
        st.info("No data to export.")
        section_end()
        return

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            "⬇️ Download CSV",
            data=export_dataframe_csv(df),
            file_name="power_soc_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        stats = compute_full_statistics(st.session_state[SS_STATS_COLUMN])
        st.download_button(
            "⬇️ Download Excel (Data + Stats)",
            data=export_dataframe_excel(df, stats),
            file_name="power_soc_data.xlsx",
            use_container_width=True,
        )

    section_end()


# =============================================================================
# 4. EXPORT UI — CHARTS
# =============================================================================

def render_export_charts_section() -> None:
    """
    Render chart export section.
    """

    section_start(
        "🖼 Export Charts",
        "Download visualization images",
    )

    df = get_visualization_dataframe()
    if df.empty:
        st.info("No charts available for export.")
        section_end()
        return

    # Build figures
    fig_power = build_power_line_chart(df)
    fig_soc = build_soc_chart(df)
    fig_compare = build_power_soc_dual_axis(df)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "⬇️ Power Chart (PNG)",
            data=export_chart_png(fig_power),
            file_name="power_chart.png",
            mime="image/png",
            use_container_width=True,
        )

    with c2:
        st.download_button(
            "⬇️ SOC Chart (PNG)",
            data=export_chart_png(fig_soc),
            file_name="soc_chart.png",
            mime="image/png",
            use_container_width=True,
        )

    with c3:
        st.download_button(
            "⬇️ Power vs SOC (PNG)",
            data=export_chart_png(fig_compare),
            file_name="power_soc_comparison.png",
            mime="image/png",
            use_container_width=True,
        )

    section_end()


# =============================================================================
# 5. EXPORT TAB ENTRY POINT
# =============================================================================

def render_export_tab() -> None:
    """
    Entry point for 💾 Export tab.
    """

    render_export_data_section()
    render_export_charts_section()


# =============================================================================
# END OF PART 4D
# =============================================================================

# =============================================================================
# PART 5 — ❓ HOW TO USE (In-App Help Tab)
# =============================================================================
# This section provides:
# - Full user guide inside the app
# - Step-by-step instructions
# - Explanation of every tab
# - Best practices & tips
# =============================================================================

def render_how_to_use_tab() -> None:
    """
    Render in-app help & user guide.
    """

    section_start(
        "❓ How to Use This Dashboard",
        "Step-by-step guide for all features",
    )

    st.markdown(
        """
### 👋 Welcome
This **Energy Power Dashboard** helps you record, analyze, visualize,  
and export **Power** and **SOC (%)** data easily and professionally.

You can use it for:
- Energy monitoring
- Battery analysis
- Testing & simulations
- Data exploration
"""
    )

    section_end()

    # -------------------------------------------------------------------------
    # INPUT & TABLE
    # -------------------------------------------------------------------------
    section_start("📥 Input & Table")

    st.markdown(
        """
### How to add data
1. Go to **📥 Input & Table**
2. Enter **Power** value
3. Enter **SOC (%)**
4. Press **ENTER**
5. Data is added automatically

> ⚠️ Both Power and SOC must be valid numbers and inside allowed ranges.

### Undo / Redo
- **↶ Undo** → revert last change  
- **↷ Redo** → reapply reverted change  
- Works for:
  - Input
  - Table edits
  - Table tools

### Editable Table
- Edit Power & SOC directly
- Date Time is auto-generated and locked
- Validation prevents invalid values
"""
    )

    section_end()

    # -------------------------------------------------------------------------
    # DATA TABLE TOOLS
    # -------------------------------------------------------------------------
    section_start("🛠 Data Table Tools")

    st.markdown(
        """
Use the **dropdown tool** to manage your table like Excel:

Available actions:
- Insert row
- Delete row
- Insert column
- Delete column
- Duplicate column
- Rename column
- Move column left / right
- Merge columns

> 💡 All actions are **Undo / Redo safe**
"""
    )

    section_end()

    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------
    section_start("📈 Visualization")

    st.markdown(
        """
### Power Charts
You can view Power data in 4 ways:
- Line chart
- Step chart
- Area chart
- Bar chart

### SOC Charts
- SOC (%) over time
- Color zones show battery health:
  - 🔴 Low SOC
  - 🟢 Normal
  - 🟠 High SOC

### Comparison
- Dual-axis chart shows **Power vs SOC**
- Helps identify correlation patterns

### Advanced Controls
- Focus on one chart (dim others)
- Time range filter:
  - All
  - Last N points
  - Custom date range
- Rolling average
- Trend line (linear regression)
"""
    )

    section_end()

    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------
    section_start("📊 Statistics")

    st.markdown(
        """
### Descriptive Statistics
- Count
- Mean
- Median
- Standard deviation
- Min / Max
- Skewness & Kurtosis

### Distribution Analysis
- Percentile table
- Box plot
- Variance & IQR

### Outlier Detection
- IQR method
- Z-Score method

### Advanced Analysis
- Rolling mean & volatility
- Trend detection (slope, R²)
- Power ↔ SOC correlation
"""
    )

    section_end()

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------
    section_start("💾 Export")

    st.markdown(
        """
You can export your data anytime:

### Data Export
- CSV (raw data)
- Excel (data + statistics)

### Chart Export
- Power chart (PNG)
- SOC chart (PNG)
- Power vs SOC chart (PNG)

> 💡 Perfect for reports and presentations
"""
    )

    section_end()

    # -------------------------------------------------------------------------
    # BEST PRACTICES
    # -------------------------------------------------------------------------
    section_start("✅ Best Practices")

    st.markdown(
        """
- Always verify Power & SOC ranges in **Settings**
- Use **Undo / Redo** confidently
- Use rolling averages for noisy data
- Use correlation to understand system behavior
- Export Excel for detailed offline analysis
"""
    )

    section_end()

    # -------------------------------------------------------------------------
    # FOOTER
    # -------------------------------------------------------------------------
    st.caption(
        "Energy Power Dashboard — Full Feature Version | Built with Streamlit"
    )


# =============================================================================
# END OF PART 5
# =============================================================================

tabs = app_bootstrap()

with tabs["input_table"]:
    render_input_table_tab()

with tabs["visualization"]:
    render_visualization_tab_advanced()

with tabs["statistics"]:
    render_statistics_tab_advanced()

with tabs["export"]:
    render_export_tab()

with tabs["help"]:
    render_how_to_use_tab()
