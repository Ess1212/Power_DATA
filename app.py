# =============================================================================
# ⚡ ENERGY POWER DASHBOARD — ENTERPRISE EDITION
# PART 1 — CORE ENGINE, STATE MANAGEMENT, VALIDATION, SIDEBAR, UX FOUNDATION
# =============================================================================
# AUTHOR: Senior Full-Stack Engineer / Senior Data Analyst / UX Architect
# =============================================================================
# ⚠️ DO NOT RUN UNTIL ALL PARTS (1–4) ARE COMBINED INTO ONE FILE
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 1 — IMPORTS (STRICT, SAFE, NO UNUSED LIBRARIES)
# -----------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, timedelta
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
import copy
import math
import json

# -----------------------------------------------------------------------------
# SECTION 2 — APPLICATION METADATA
# -----------------------------------------------------------------------------
APP_NAME: str = "Energy Power Dashboard"
APP_SUBTITLE: str = "Industrial Energy Monitoring & Analysis System"
COMPANY_NAME: str = "SchneiTech Group"
APP_VERSION: str = "1.0.0-enterprise"
BUILD_DATE: str = "2025-01-01"

# -----------------------------------------------------------------------------
# SECTION 3 — GLOBAL CONSTANTS (ENTERPRISE SAFE)
# -----------------------------------------------------------------------------
from datetime import timezone, timedelta

# ------------------ CORE COLUMN NAMES ------------------
TIME_COLUMN: str = "Date Time"
POWER_COLUMN: str = "Power (kW)"
SOC_COLUMN: str = "SOC (%)"

# ------------------ TIME SYSTEM ------------------
# Always use timezone-aware timestamps (UTC recommended for engineering systems)
APP_TIMEZONE = timezone.utc
# If you prefer local time instead, use:
# APP_TIMEZONE = timezone(timedelta(hours=7))  # Example: UTC+7

# Display format (used only for UI, NOT for storage)
TIME_DISPLAY_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ------------------ POWER LIMITS ------------------
DEFAULT_MIN_POWER: float = -200.0
DEFAULT_MAX_POWER: float = 200.0

POWER_UNIT: str = "kW"
SOC_UNIT: str = "%"

# ------------------ ANALYTICS DEFAULTS ------------------
ROLLING_WINDOWS_DEFAULT: int = 3

# ------------------ SYSTEM SAFETY ------------------
# Maximum undo depth to prevent memory exhaustion
MAX_UNDO_DEPTH: int = 2000


# -----------------------------------------------------------------------------
# SECTION 4 — PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME}",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# SECTION 5 — GLOBAL STYLE INJECTION (ENGINEERING UI)
# -----------------------------------------------------------------------------
def inject_global_styles() -> None:
    """
    Injects professional, industrial UI styles.
    This is NOT cosmetic only — spacing, readability, and hierarchy matter.
    """
    st.markdown("""
    <style>
    /* ------------------ ROOT ------------------ */
    html, body {
        background-color: #f4f7f6;
        font-family: "Segoe UI", Roboto, Arial, sans-serif;
    }

    /* ------------------ HEADER ------------------ */
    .global-header {
        background: linear-gradient(135deg, #0b3d2e, #145a45);
        padding: 20px;
        border-radius: 18px;
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .header-left {
        font-size: 18px;
        font-weight: 600;
        opacity: 0.9;
    }

    .header-center {
        font-size: 28px;
        font-weight: 800;
        text-align: center;
        flex-grow: 1;
    }

    .header-right {
        font-size: 14px;
        opacity: 0.8;
        text-align: right;
    }

    /* ------------------ CARDS ------------------ */
    .card {
        background: white;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #0b3d2e;
    }

    /* ------------------ BUTTONS ------------------ */
    button[kind="primary"] {
        background-color: #1b7f5c !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 6px 14px !important;
    }

    button[kind="secondary"] {
        border-radius: 12px !important;
        font-weight: 600 !important;
    }

    /* ------------------ TABS ------------------ */
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 700;
        padding: 10px;
    }

    .stTabs [aria-selected="true"] {
        color: #ff9f1c;
    }

    /* ------------------ ALERT TEXT ------------------ */
    .error-text {
        color: #b00020;
        font-weight: 600;
    }

    .success-text {
        color: #1b7f5c;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

inject_global_styles()

# -----------------------------------------------------------------------------
# SECTION 6 — HEADER RENDERING
# -----------------------------------------------------------------------------
def render_header() -> None:
    """
    Renders the global application header.
    """
    st.markdown(f"""
    <div class="global-header">
        <div class="header-left">{COMPANY_NAME}</div>
        <div class="header-center">⚡ {APP_NAME} ⚡</div>
        <div class="header-right">
            Version {APP_VERSION}<br>
            {APP_SUBTITLE}
        </div>
    </div>
    """, unsafe_allow_html=True)

render_header()

# -----------------------------------------------------------------------------
# SECTION 7 — SESSION STATE INITIALIZATION (CRITICAL)
# -----------------------------------------------------------------------------
def init_session_state() -> None:
    """
    Initializes ALL session state keys.
    This prevents runtime errors and unstable reruns.
    """

    # ------------------ CORE DATA ------------------
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(
            columns=[TIME_COLUMN, POWER_COLUMN, SOC_COLUMN]
        )

    # ------------------ UNDO / REDO ------------------
    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack: List[pd.DataFrame] = []

    if "redo_stack" not in st.session_state:
        st.session_state.redo_stack: List[pd.DataFrame] = []

    # ------------------ TABLE STATE ------------------
    if "table_locked" not in st.session_state:
        st.session_state.table_locked = False

    # ------------------ POWER BOUNDS ------------------
    if "min_power" not in st.session_state:
        st.session_state.min_power = DEFAULT_MIN_POWER

    if "max_power" not in st.session_state:
        st.session_state.max_power = DEFAULT_MAX_POWER

    # ------------------ SYSTEM FLAGS ------------------
    if "last_error" not in st.session_state:
        st.session_state.last_error = None

    if "last_action" not in st.session_state:
        st.session_state.last_action = None

    if "initialized_at" not in st.session_state:
        st.session_state.initialized_at = datetime.now()

init_session_state()

# -----------------------------------------------------------------------------
# SECTION 8 — UNDO / REDO ENGINE (ENTERPRISE SAFE)
# -----------------------------------------------------------------------------
def push_undo_state(reason: str = "") -> None:
    """
    Saves a deep copy of current dataframe state.
    Includes safety limit to avoid memory overflow.
    """
    if len(st.session_state.undo_stack) >= MAX_UNDO_DEPTH:
        st.session_state.undo_stack.pop(0)

    st.session_state.undo_stack.append(
        copy.deepcopy(st.session_state.df)
    )
    st.session_state.redo_stack.clear()
    st.session_state.last_action = f"UNDO SNAPSHOT: {reason}"

def undo_action() -> None:
    if not st.session_state.undo_stack:
        return

    st.session_state.redo_stack.append(
        copy.deepcopy(st.session_state.df)
    )
    st.session_state.df = st.session_state.undo_stack.pop()
    st.session_state.last_action = "UNDO"

def redo_action() -> None:
    if not st.session_state.redo_stack:
        return

    st.session_state.undo_stack.append(
        copy.deepcopy(st.session_state.df)
    )
    st.session_state.df = st.session_state.redo_stack.pop()
    st.session_state.last_action = "REDO"

# -----------------------------------------------------------------------------
# SECTION 9 — VALIDATION ENGINE (STRICT & USER-FRIENDLY)
# -----------------------------------------------------------------------------
def validate_power(power: Any) -> Tuple[bool, Optional[str]]:
    if power is None:
        return False, "Power value is required."

    if not isinstance(power, (int, float)):
        return False, "Power must be numeric."

    if math.isnan(power):
        return False, "Power cannot be NaN."

    if power < st.session_state.min_power:
        return False, f"Power below minimum bound ({st.session_state.min_power} kW)."

    if power > st.session_state.max_power:
        return False, f"Power exceeds maximum bound ({st.session_state.max_power} kW)."

    return True, None

def validate_soc(soc: Any) -> Tuple[bool, Optional[str]]:
    if soc is None:
        return False, "SOC value is required."

    if not isinstance(soc, (int, float)):
        return False, "SOC must be numeric."

    if soc < 0 or soc > 100:
        return False, "SOC must be between 0 and 100%."

    return True, None

# -----------------------------------------------------------------------------
# SECTION 10 — SIDEBAR (SETTINGS & DATA GENERATION)
# -----------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ System Settings")

        # ------------------ POWER BOUNDS ------------------
        st.subheader("🔋 Power Bound Control")

        min_power_input = st.number_input(
            "Minimum Power (kW)",
            value=float(st.session_state.min_power),
            step=1.0
        )

        max_power_input = st.number_input(
            "Maximum Power (kW)",
            value=float(st.session_state.max_power),
            step=1.0
        )

        col1, col2 = st.columns(2)

        if col1.button("Apply Bounds"):
            if min_power_input < max_power_input:
                st.session_state.min_power = min_power_input
                st.session_state.max_power = max_power_input
                st.success("Power bounds applied globally.")
            else:
                st.error("Minimum must be less than maximum.")

        if col2.button("Reset Bounds"):
            st.session_state.min_power = DEFAULT_MIN_POWER
            st.session_state.max_power = DEFAULT_MAX_POWER
            st.success("Bounds reset to default.")

        st.divider()

        # ------------------ SAMPLE DATA ------------------
        st.subheader("🧪 Generate Sample Data")

        raw_values = st.text_area(
            "Paste comma-separated Power values",
            height=140,
            placeholder="e.g. 50, -20, 35.5, -10"
        )

        if st.button("Apply Sample Data"):
            try:
                values = [float(v.strip()) for v in raw_values.split(",") if v.strip()]
                push_undo_state("Insert sample data")

                for val in values:
                    st.session_state.df.loc[len(st.session_state.df)] = [
                        datetime.now(),
                        val,
                        np.random.uniform(20, 90)
                    ]

                st.success(f"{len(values)} rows inserted.")

            except Exception as e:
                st.error(f"Invalid input: {e}")

        if st.button("Clear Entire Table"):
            push_undo_state("Clear table")
            st.session_state.df = st.session_state.df.iloc[0:0]
            st.success("All data cleared.")

render_sidebar()

# -----------------------------------------------------------------------------
# SECTION 11 — TAB SHELL (LOGIC COMES IN NEXT PARTS)
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📥 Input & Table",
    "📈 Visualization",
    "📊 Statistics Analysis",
    "⏱ Time Series Analysis",
    "💾 Export Data",
    "❓ How to Use"
])

# -----------------------------------------------------------------------------
# END OF PART 1
# PART 2 WILL IMPLEMENT FULL INPUT & TABLE ENGINE (900+ LINES)
# -----------------------------------------------------------------------------
# =============================================================================
# PART 2 — INPUT & TABLE ENGINE
# =============================================================================
# Responsibilities:
# - Power & SOC input handling
# - ENTER-to-insert logic
# - Validation integration
# - Editable table
# - Table lock (read-only)
# - Enterprise dropdown table tools
# - FULL undo/redo coverage
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 12 — UTILITY FUNCTIONS (TABLE OPERATIONS)
# -----------------------------------------------------------------------------
def safe_push_undo(reason: str) -> None:
    """
    Wrapper around push_undo_state with semantic clarity.
    """
    push_undo_state(reason=reason)


def generate_timestamp() -> datetime:
    """
    Generates a precise timestamp.
    Centralized for future extension (timezone, sync, etc.).
    """
    return datetime.now()


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures required columns exist and are ordered correctly.
    Prevents corruption from user edits.
    """
    required_cols = [TIME_COLUMN, POWER_COLUMN, SOC_COLUMN]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[required_cols + [c for c in df.columns if c not in required_cols]]


# -----------------------------------------------------------------------------
# SECTION 13 — INPUT VALIDATION PIPELINE
# -----------------------------------------------------------------------------
def validate_input_pair(power_value: Any, soc_value: Any) -> Tuple[bool, List[str]]:
    """
    Validates Power + SOC together.
    Returns (is_valid, list_of_errors)
    """
    errors: List[str] = []

    ok_p, err_p = validate_power(power_value)
    if not ok_p and err_p:
        errors.append(err_p)

    ok_s, err_s = validate_soc(soc_value)
    if not ok_s and err_s:
        errors.append(err_s)

    return len(errors) == 0, errors


# -----------------------------------------------------------------------------
# SECTION 14 — INSERT ROW ENGINE
# -----------------------------------------------------------------------------
def insert_new_row(power_value: float, soc_value: float) -> None:
    """
    Inserts a new row into the dataframe with validation & undo support.
    """
    is_valid, errors = validate_input_pair(power_value, soc_value)

    if not is_valid:
        for e in errors:
            st.error(e)
        return

    if st.session_state.table_locked:
        st.warning("Table is locked. Unlock to insert data.")
        return

    safe_push_undo("Insert new row")

    new_row = {
        TIME_COLUMN: generate_timestamp(),
        POWER_COLUMN: float(power_value),
        SOC_COLUMN: float(soc_value)
    }

    st.session_state.df.loc[len(st.session_state.df)] = new_row
    st.session_state.last_action = "INSERT_ROW"


# -----------------------------------------------------------------------------
# SECTION 15 — TABLE TOOL OPERATIONS
# -----------------------------------------------------------------------------
def insert_empty_row() -> None:
    safe_push_undo("Insert empty row")
    st.session_state.df.loc[len(st.session_state.df)] = {
        TIME_COLUMN: generate_timestamp(),
        POWER_COLUMN: np.nan,
        SOC_COLUMN: np.nan
    }


def delete_selected_row(row_index: int) -> None:
    if row_index < 0 or row_index >= len(st.session_state.df):
        st.error("Invalid row index.")
        return

    safe_push_undo(f"Delete row {row_index}")
    st.session_state.df = st.session_state.df.drop(index=row_index).reset_index(drop=True)


def insert_column(column_name: str) -> None:
    if column_name in st.session_state.df.columns:
        st.error("Column already exists.")
        return

    safe_push_undo(f"Insert column {column_name}")
    st.session_state.df[column_name] = np.nan


def delete_column(column_name: str) -> None:
    if column_name not in st.session_state.df.columns:
        st.error("Column not found.")
        return

    if column_name in [TIME_COLUMN, POWER_COLUMN, SOC_COLUMN]:
        st.error("Core columns cannot be deleted.")
        return

    safe_push_undo(f"Delete column {column_name}")
    st.session_state.df = st.session_state.df.drop(columns=[column_name])


def rename_column(old_name: str, new_name: str) -> None:
    if old_name not in st.session_state.df.columns:
        st.error("Column not found.")
        return

    if new_name in st.session_state.df.columns:
        st.error("Target column name already exists.")
        return

    safe_push_undo(f"Rename column {old_name} -> {new_name}")
    st.session_state.df = st.session_state.df.rename(columns={old_name: new_name})


def duplicate_column(column_name: str) -> None:
    if column_name not in st.session_state.df.columns:
        st.error("Column not found.")
        return

    new_name = f"{column_name}_copy"
    i = 1
    while new_name in st.session_state.df.columns:
        new_name = f"{column_name}_copy_{i}"
        i += 1

    safe_push_undo(f"Duplicate column {column_name}")
    st.session_state.df[new_name] = st.session_state.df[column_name]


def move_column(column_name: str, direction: str) -> None:
    cols = list(st.session_state.df.columns)

    if column_name not in cols:
        st.error("Column not found.")
        return

    idx = cols.index(column_name)

    if direction == "left" and idx > 0:
        cols[idx], cols[idx - 1] = cols[idx - 1], cols[idx]
    elif direction == "right" and idx < len(cols) - 1:
        cols[idx], cols[idx + 1] = cols[idx + 1], cols[idx]
    else:
        return

    safe_push_undo(f"Move column {column_name} {direction}")
    st.session_state.df = st.session_state.df[cols]


# -----------------------------------------------------------------------------
# SECTION 16 — TABLE EDITOR RENDERING
# -----------------------------------------------------------------------------
def render_table_editor() -> None:
    """
    Renders editable table with lock support.
    """
    st.subheader("📋 Data Table")

    st.session_state.df = normalize_dataframe_columns(st.session_state.df)

    edited_df = st.data_editor(
        st.session_state.df,
        disabled=st.session_state.table_locked,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic"
    )

    if not edited_df.equals(st.session_state.df):
        safe_push_undo("Manual table edit")
        st.session_state.df = edited_df


# -----------------------------------------------------------------------------
# SECTION 17 — TABLE TOOL DROPDOWN UI
# -----------------------------------------------------------------------------
def render_table_tools() -> None:
    st.markdown("### 🛠 Table Tools")

    tool = st.selectbox(
        "Select Action",
        [
            "— Select —",
            "Insert empty row",
            "Delete row",
            "Insert column",
            "Delete column",
            "Rename column",
            "Duplicate column",
            "Move column left",
            "Move column right"
        ]
    )

    if tool == "Insert empty row":
        if st.button("Execute"):
            insert_empty_row()

    elif tool == "Delete row":
        row_idx = st.number_input("Row index", min_value=0, step=1)
        if st.button("Execute"):
            delete_selected_row(int(row_idx))

    elif tool == "Insert column":
        col_name = st.text_input("New column name")
        if st.button("Execute"):
            insert_column(col_name)

    elif tool == "Delete column":
        col_name = st.selectbox("Column", st.session_state.df.columns)
        if st.button("Execute"):
            delete_column(col_name)

    elif tool == "Rename column":
        old = st.selectbox("Old name", st.session_state.df.columns)
        new = st.text_input("New name")
        if st.button("Execute"):
            rename_column(old, new)

    elif tool == "Duplicate column":
        col = st.selectbox("Column", st.session_state.df.columns)
        if st.button("Execute"):
            duplicate_column(col)

    elif tool == "Move column left":
        col = st.selectbox("Column", st.session_state.df.columns)
        if st.button("Execute"):
            move_column(col, "left")

    elif tool == "Move column right":
        col = st.selectbox("Column", st.session_state.df.columns)
        if st.button("Execute"):
            move_column(col, "right")


# -----------------------------------------------------------------------------
# SECTION 18 — INPUT & TABLE TAB RENDERING
# -----------------------------------------------------------------------------
with tabs[0]:
    st.markdown("## 📥 Power & SOC Input")

    # ------------------ FORM: ENTER TO SUBMIT ------------------
    with st.form("power_soc_form", clear_on_submit=True):
        col_a, col_b = st.columns(2)

        power_input = col_a.number_input(
            f"Power ({POWER_UNIT})",
            step=0.1,
            format="%.2f",
            key="power_input_form"
        )

        soc_input = col_b.number_input(
            f"SOC ({SOC_UNIT})",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key="soc_input_form"
        )

        submitted = st.form_submit_button(
            "Submit",
            use_container_width=False
        )

        if submitted:
            insert_new_row(power_input, soc_input)

    # ------------------ UNDO / REDO ------------------
    st.markdown("### ↩ Undo / Redo")
    u1, u2, u3 = st.columns([1, 1, 2])

    u1.button("Undo", on_click=undo_action)
    u2.button("Redo", on_click=redo_action)
    st.session_state.table_locked = u3.checkbox(
        "🔒 Lock Table",
        value=st.session_state.table_locked
    )

    st.divider()

    render_table_editor()
    st.divider()
    render_table_tools()

# =============================================================================
# END OF PART 2
# PART 3 WILL IMPLEMENT VISUALIZATION ENGINE (700+ LINES)
# =============================================================================
# =============================================================================
# PART 3 — VISUALIZATION ENGINE
# =============================================================================
# Responsibilities:
# - Power-only visualization (SOC excluded from plots)
# - 4 required charts (Line, Area, Bar, Step)
# - Smoothing engine
# - Focus / highlight controls
# - Point on/off toggle
# - Area calculation
# - Dynamic updates
# - Professional Plotly configuration
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 19 — DATA PREPARATION FOR VISUALIZATION
# -----------------------------------------------------------------------------
def prepare_visualization_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares dataframe for visualization.
    - Ensures datetime sorting
    - Removes invalid power values
    - Adds helper columns for coloring and sign logic
    """
    if df.empty:
        return df

    vis_df = df.copy()

    # Ensure datetime
    vis_df[TIME_COLUMN] = pd.to_datetime(vis_df[TIME_COLUMN], errors="coerce")

    # Drop rows with invalid timestamps or power
    vis_df = vis_df.dropna(subset=[TIME_COLUMN, POWER_COLUMN])

    # Sort chronologically
    vis_df = vis_df.sort_values(TIME_COLUMN)

    # Power sign classification
    vis_df["Power_Sign"] = np.where(
        vis_df[POWER_COLUMN] >= 0,
        "Positive",
        "Negative"
    )

    # Absolute power for area calculations
    vis_df["Power_Abs"] = vis_df[POWER_COLUMN].abs()

    return vis_df


# -----------------------------------------------------------------------------
# SECTION 20 — SMOOTHING ENGINE
# -----------------------------------------------------------------------------
def apply_smoothing(series: pd.Series, window: int) -> pd.Series:
    """
    Applies rolling mean smoothing.
    This is intentionally simple, transparent, and engineer-friendly.
    """
    if series.empty or window <= 1:
        return series

    if len(series) < window:
        return series

    return series.rolling(window=window, min_periods=1).mean()


# -----------------------------------------------------------------------------
# SECTION 21 — COLOR SCHEME ENGINE
# -----------------------------------------------------------------------------
def power_color_map() -> Dict[str, str]:
    """
    Centralized color control for consistency across charts.
    """
    return {
        "Positive": "#1b7f5c",  # green
        "Negative": "#b00020"   # red
    }


# -----------------------------------------------------------------------------
# SECTION 22 — LINE CHART
# -----------------------------------------------------------------------------
def render_line_chart(
    df: pd.DataFrame,
    show_points: bool,
    smoothing_enabled: bool,
    smoothing_window: int
) -> go.Figure:
    """
    Renders a professional line chart.
    """
    y_series = df[POWER_COLUMN]

    if smoothing_enabled:
        y_series = apply_smoothing(y_series, smoothing_window)

    fig = px.line(
        df,
        x=TIME_COLUMN,
        y=y_series,
        color="Power_Sign",
        color_discrete_map=power_color_map(),
        markers=show_points
    )

    fig.update_layout(
        title="Power Over Time — Line Chart",
        xaxis_title="Date Time",
        yaxis_title=f"Power ({POWER_UNIT})",
        hovermode="x unified",
        legend_title="Power Sign",
        template="plotly_white"
    )

    return fig


# -----------------------------------------------------------------------------
# SECTION 23 — AREA CHART
# -----------------------------------------------------------------------------
def render_area_chart(
    df: pd.DataFrame,
    smoothing_enabled: bool,
    smoothing_window: int
) -> go.Figure:
    """
    Renders an area chart representing energy magnitude.
    """
    y_series = df[POWER_COLUMN]

    if smoothing_enabled:
        y_series = apply_smoothing(y_series, smoothing_window)

    fig = px.area(
        df,
        x=TIME_COLUMN,
        y=y_series,
        color="Power_Sign",
        color_discrete_map=power_color_map()
    )

    fig.update_layout(
        title="Power Over Time — Area Chart",
        xaxis_title="Date Time",
        yaxis_title=f"Power ({POWER_UNIT})",
        hovermode="x unified",
        template="plotly_white"
    )

    return fig


# -----------------------------------------------------------------------------
# SECTION 24 — BAR CHART
# -----------------------------------------------------------------------------
def render_bar_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders a bar chart for discrete power readings.
    """
    fig = px.bar(
        df,
        x=TIME_COLUMN,
        y=POWER_COLUMN,
        color="Power_Sign",
        color_discrete_map=power_color_map()
    )

    fig.update_layout(
        title="Power Over Time — Bar Chart",
        xaxis_title="Date Time",
        yaxis_title=f"Power ({POWER_UNIT})",
        template="plotly_white"
    )

    return fig


# -----------------------------------------------------------------------------
# SECTION 25 — STEP LINE CHART
# -----------------------------------------------------------------------------
def render_step_chart(
    df: pd.DataFrame,
    smoothing_enabled: bool,
    smoothing_window: int
) -> go.Figure:
    """
    Renders a step line chart (engineering-style signal behavior).
    """
    y_series = df[POWER_COLUMN]

    if smoothing_enabled:
        y_series = apply_smoothing(y_series, smoothing_window)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[TIME_COLUMN],
            y=y_series,
            mode="lines",
            line=dict(
                shape="hv",
                color="#1b7f5c",
                width=3
            ),
            name="Power Step"
        )
    )

    fig.update_layout(
        title="Power Over Time — Step Line Chart",
        xaxis_title="Date Time",
        yaxis_title=f"Power ({POWER_UNIT})",
        hovermode="x unified",
        template="plotly_white"
    )

    return fig


# -----------------------------------------------------------------------------
# SECTION 26 — ENERGY AREA CALCULATION
# -----------------------------------------------------------------------------
def calculate_energy_area(df: pd.DataFrame) -> float:
    """
    Calculates approximate energy area under the curve.
    Uses trapezoidal integration on power vs time.
    """
    if len(df) < 2:
        return 0.0

    times = pd.to_datetime(df[TIME_COLUMN])
    power = df[POWER_COLUMN].values

    # Convert datetime to seconds
    time_seconds = times.astype(np.int64) / 1e9

    area = np.trapz(power, time_seconds)

    # Convert to kWh (seconds → hours)
    area_kwh = area / 3600.0

    return float(area_kwh)


# -----------------------------------------------------------------------------
# SECTION 27 — VISUALIZATION CONTROL PANEL
# -----------------------------------------------------------------------------
def render_visualization_controls() -> Dict[str, Any]:
    """
    Renders visualization controls and returns state.
    """
    st.markdown("### 🎛 Visualization Controls")

    c1, c2, c3, c4 = st.columns(4)

    show_points = c1.checkbox("Show Points", value=False)
    smoothing = c2.checkbox("Enable Smoothing", value=False)
    smoothing_window = c3.slider("Smoothing Window", 2, 20, ROLLING_WINDOWS_DEFAULT)
    focus_step = c4.checkbox("Focus Step Chart", value=False)

    return {
        "show_points": show_points,
        "smoothing": smoothing,
        "smoothing_window": smoothing_window,
        "focus_step": focus_step
    }


# -----------------------------------------------------------------------------
# SECTION 28 — VISUALIZATION TAB RENDERING
# -----------------------------------------------------------------------------
with tabs[1]:
    st.markdown("## 📈 Power Visualization")

    if st.session_state.df.empty:
        st.info("No data available. Please insert Power & SOC values first.")
    else:
        vis_df = prepare_visualization_dataframe(st.session_state.df)

        if vis_df.empty:
            st.warning("No valid power data to visualize.")
        else:
            controls = render_visualization_controls()

            # Render charts
            line_fig = render_line_chart(
                vis_df,
                controls["show_points"],
                controls["smoothing"],
                controls["smoothing_window"]
            )

            area_fig = render_area_chart(
                vis_df,
                controls["smoothing"],
                controls["smoothing_window"]
            )

            bar_fig = render_bar_chart(vis_df)

            step_fig = render_step_chart(
                vis_df,
                controls["smoothing"],
                controls["smoothing_window"]
            )

            # Focus logic
            if controls["focus_step"]:
                st.plotly_chart(step_fig, use_container_width=True)
            else:
                st.plotly_chart(line_fig, use_container_width=True)
                st.plotly_chart(area_fig, use_container_width=True)
                st.plotly_chart(bar_fig, use_container_width=True)
                st.plotly_chart(step_fig, use_container_width=True)

            # ------------------ ENERGY AREA ------------------
            st.markdown("### ⚡ Energy Area Calculation")

            energy_area = calculate_energy_area(vis_df)

            st.metric(
                label="Approximate Energy (kWh)",
                value=f"{energy_area:.3f}"
            )

            st.caption(
                "Calculated using trapezoidal integration of Power vs Time. "
                "Positive values indicate net generation; negative values indicate net consumption."
            )

# =============================================================================
# END OF PART 3
# PART 4 WILL IMPLEMENT STATISTICS, TIME SERIES, EXPORT, HOW-TO (800+ LINES)
# =============================================================================
# =============================================================================
# PART 4 — STATISTICS + TIME SERIES + EXPORT + HOW TO USE
# =============================================================================
# Responsibilities:
# - Descriptive statistics (engineering-grade)
# - Distribution & diagnostics
# - Time-series rolling analysis
# - Safe export (CSV / XLSX / JSON)
# - Beginner-friendly help system
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 29 — STATISTICS CORE ENGINE
# -----------------------------------------------------------------------------
def compute_basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes core descriptive statistics for Power values.
    Returns a structured DataFrame suitable for display.
    """
    power = df[POWER_COLUMN].dropna()

    stats_data = {
        "Metric": [
            "Count",
            "Minimum",
            "Maximum",
            "Mean",
            "Median",
            "Standard Deviation",
            "Variance",
            "25th Percentile",
            "50th Percentile",
            "75th Percentile"
        ],
        "Value": [
            power.count(),
            power.min(),
            power.max(),
            power.mean(),
            power.median(),
            power.std(),
            power.var(),
            power.quantile(0.25),
            power.quantile(0.50),
            power.quantile(0.75),
        ]
    }

    return pd.DataFrame(stats_data)


# -----------------------------------------------------------------------------
# SECTION 30 — DISTRIBUTION VISUALS
# -----------------------------------------------------------------------------
def render_distribution_charts(df: pd.DataFrame) -> None:
    """
    Renders statistical distribution charts.
    """
    power = df[POWER_COLUMN].dropna()

    # Histogram
    hist_fig = px.histogram(
        power,
        nbins=30,
        title="Power Distribution Histogram",
        labels={"value": f"Power ({POWER_UNIT})"},
        template="plotly_white"
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # Box plot
    box_fig = px.box(
        power,
        title="Power Distribution — Box Plot",
        labels={"value": f"Power ({POWER_UNIT})"},
        template="plotly_white"
    )
    st.plotly_chart(box_fig, use_container_width=True)

    # Violin plot
    violin_fig = px.violin(
        power,
        box=True,
        points="all",
        title="Power Distribution — Violin Plot",
        labels={"value": f"Power ({POWER_UNIT})"},
        template="plotly_white"
    )
    st.plotly_chart(violin_fig, use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 31 — POSITIVE VS NEGATIVE DONUT
# -----------------------------------------------------------------------------
def render_power_sign_donut(df: pd.DataFrame) -> None:
    """
    Renders a donut chart comparing positive vs negative power.
    """
    power = df[POWER_COLUMN].dropna()

    positive_count = (power >= 0).sum()
    negative_count = (power < 0).sum()

    donut_fig = px.pie(
        names=["Positive Power", "Negative Power"],
        values=[positive_count, negative_count],
        hole=0.5,
        color_discrete_sequence=["#1b7f5c", "#b00020"],
        title="Positive vs Negative Power Readings"
    )

    st.plotly_chart(donut_fig, use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 32 — STATISTICS TAB RENDERING
# -----------------------------------------------------------------------------
with tabs[2]:
    st.markdown("## 📊 Statistics Analysis")

    if st.session_state.df.empty:
        st.info("No data available for statistical analysis.")
    else:
        df_stats = st.session_state.df.copy()

        # ------------------ TABLE ------------------
        st.markdown("### 📐 Descriptive Statistics")
        stats_table = compute_basic_statistics(df_stats)
        st.dataframe(stats_table, use_container_width=True, hide_index=True)

        # ------------------ VISUALS ------------------
        st.markdown("### 📊 Distribution & Diagnostics")
        render_distribution_charts(df_stats)

        st.markdown("### 🔄 Power Direction Analysis")
        render_power_sign_donut(df_stats)

        # ------------------ EXPLANATION ------------------
        st.markdown("### 📝 Interpretation")
        st.write(
            """
            - **Mean vs Median** comparison helps identify skewness.
            - **Standard deviation** reflects variability in power behavior.
            - **Box and violin plots** highlight outliers and spread.
            - **Positive vs Negative** split shows generation vs consumption balance.
            """
        )


# -----------------------------------------------------------------------------
# SECTION 33 — TIME SERIES ANALYSIS ENGINE
# -----------------------------------------------------------------------------
def compute_rolling_statistics(
    df: pd.DataFrame,
    window: int
) -> pd.DataFrame:
    """
    Computes rolling statistics for time-series analysis.
    """
    ts_df = df.copy()
    ts_df[TIME_COLUMN] = pd.to_datetime(ts_df[TIME_COLUMN])
    ts_df = ts_df.sort_values(TIME_COLUMN)
    ts_df.set_index(TIME_COLUMN, inplace=True)

    ts_df["Rolling_Mean"] = ts_df[POWER_COLUMN].rolling(window).mean()
    ts_df["Rolling_Std"] = ts_df[POWER_COLUMN].rolling(window).std()

    return ts_df.reset_index()


# -----------------------------------------------------------------------------
# SECTION 34 — TIME SERIES TAB RENDERING
# -----------------------------------------------------------------------------
with tabs[3]:
    st.markdown("## ⏱ Time Series Analysis")

    if st.session_state.df.empty:
        st.info("No data available for time series analysis.")
    else:
        window = st.slider(
            "Rolling Window Size",
            min_value=2,
            max_value=50,
            value=ROLLING_WINDOWS_DEFAULT
        )

        ts_df = compute_rolling_statistics(st.session_state.df, window)

        # Rolling Mean
        mean_fig = px.line(
            ts_df,
            x=TIME_COLUMN,
            y="Rolling_Mean",
            title=f"Rolling Mean (Window = {window})",
            template="plotly_white"
        )
        st.plotly_chart(mean_fig, use_container_width=True)

        # Rolling Std
        std_fig = px.line(
            ts_df,
            x=TIME_COLUMN,
            y="Rolling_Std",
            title=f"Rolling Standard Deviation (Window = {window})",
            template="plotly_white"
        )
        st.plotly_chart(std_fig, use_container_width=True)

        st.markdown(
            """
            **Rolling statistics** help engineers observe:
            - Short-term vs long-term stability
            - Volatility changes
            - Trend readiness for advanced modeling (no ML applied)
            """
        )


# -----------------------------------------------------------------------------
# SECTION 35 — EXPORT ENGINE
# -----------------------------------------------------------------------------
def export_to_excel(df: pd.DataFrame) -> bytes:
    """
    Exports DataFrame to XLSX safely.
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="EnergyData")
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# SECTION 36 — EXPORT TAB RENDERING
# -----------------------------------------------------------------------------
with tabs[4]:
    st.markdown("## 💾 Export Data")

    if st.session_state.df.empty:
        st.info("No data available to export.")
    else:
        st.markdown("### 👀 Preview (First 10 Rows)")
        st.dataframe(
            st.session_state.df.head(10),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### 📤 Download")

        st.download_button(
            label="Download CSV",
            data=st.session_state.df.to_csv(index=False),
            file_name="energy_power_data.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download Excel (XLSX)",
            data=export_to_excel(st.session_state.df),
            file_name="energy_power_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.download_button(
            label="Download JSON",
            data=st.session_state.df.to_json(orient="records", indent=2),
            file_name="energy_power_data.json",
            mime="application/json"
        )

        st.caption(
            "Exports always reflect the current table state, "
            "including edits, deletions, and undo/redo actions."
        )


# -----------------------------------------------------------------------------
# SECTION 37 — HOW TO USE (BEGINNER-FRIENDLY)
# -----------------------------------------------------------------------------
with tabs[5]:
    st.markdown("## ❓ How to Use the Energy Power Dashboard")

    st.markdown("""
    ### Step 1 — Configure Power Bounds
    Use the **Settings sidebar** to define minimum and maximum allowable power.
    These bounds apply globally and prevent invalid entries.

    ### Step 2 — Input Power & SOC
    Navigate to **📥 Input & Table**.
    - Enter **Power (kW)** and **SOC (%)**
    - Press **Insert**
    - Timestamp is generated automatically

    ### Step 3 — Manage the Data Table
    - Edit values directly
    - Use **Table Tools** to insert/delete rows or columns
    - Lock the table to prevent accidental edits
    - Undo / Redo any action safely

    ### Step 4 — Visualize Power Behavior
    Open **📈 Visualization**:
    - Line, Area, Bar, and Step charts
    - Optional smoothing
    - Step-focus mode
    - Energy area calculation

    ### Step 5 — Analyze Statistics
    In **📊 Statistics Analysis**:
    - Review descriptive statistics
    - Identify outliers and variability
    - Compare generation vs consumption

    ### Step 6 — Time Series Insights
    Use **⏱ Time Series Analysis** to:
    - Observe rolling trends
    - Inspect volatility changes
    - Prepare data for deeper engineering studies

    ### Step 7 — Export Data
    Finally, export your results in:
    - CSV
    - Excel
    - JSON

    This dashboard is designed for **engineers, analysts, and energy professionals**
    who need **clarity, safety, and control** without training.
    """)

# =============================================================================
# END OF PART 4
# APPLICATION COMPLETE
# =============================================================================
