"""
app.py — Streamlit UI for the Synthetic Dataset Generator
==========================================================
Complete, fully-wired Streamlit application.  Walks the user through five
sequential stages rendered in the main panel, plus a persistent sidebar:

  01. UPLOAD      — sample presets OR CSV upload, data preview, column-type
                    correction editor
  02. SIDEBAR     — model & hyperparameter configuration, Train & Generate button
  03. GENERATION  — spinner-wrapped pipeline: fit → sample → evaluate → privacy
  04. RESULTS     — quality score hero, privacy risk hero, column bar chart,
                    CSV download
  05. COMPARISON  — per-column real-vs-synthetic distribution charts

State management
----------------
All persistent values live in st.session_state under keys defined in the `K`
namespace class.  Streamlit reruns this entire script on every widget
interaction; reading from session_state (rather than re-computing) keeps the
app fast and prevents stale-result bleed-through after a new file is loaded.

Directory layout expected
--------------------------
    project_root/
    ├── app.py              ← this file
    └── core/
        ├── __init__.py
        ├── synthesizer.py
        └── evaluator.py

Run:
    streamlit run app.py
"""

from __future__ import annotations

import copy
import logging
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sdv.metadata import SingleTableMetadata

# ── Core modules ───────────────────────────────────────────────────────────────
from core.synthesizer import generate_synthetic_data, get_available_models
from core.evaluator import (
    evaluate_synthetic_data,
    compute_privacy_risk,
    score_summary_text,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTS
# ==============================================================================

MAX_UPLOAD_ROWS = 100_000
MAX_CHART_COLS  = 8
DEFAULT_EPOCHS  = 300
MIN_EPOCHS      = 100
MAX_EPOCHS      = 1_000
DEFAULT_BATCH   = 500

COLOR_REAL  = "#3A86FF"   # blue   — real data
COLOR_SYNTH = "#FF6B35"   # orange — synthetic data
COLOR_GOOD  = "#2DC653"
COLOR_MID   = "#F4A261"
COLOR_BAD   = "#E63946"

SDTYPE_OPTIONS = ["categorical", "numerical", "datetime", "boolean", "id"]


# ── Session-state key namespace ────────────────────────────────────────────────
class K:
    """All st.session_state keys in one place — prevents magic-string bugs."""
    REAL_DF         = "real_df"
    METADATA_DICT   = "metadata_dict"
    OVERRIDES       = "type_overrides"
    SYNTHETIC_DF    = "synthetic_df"
    MODEL_INFO      = "model_info"
    EVAL_RESULTS    = "eval_results"
    PRIVACY_RESULTS = "privacy_results"
    GENERATION_DONE = "generation_done"
    LAST_FILE_NAME  = "last_file_name"


# ==============================================================================
# PAGE CONFIG — must be the very first Streamlit call
# ==============================================================================
st.set_page_config(
    page_title="Synthetic Dataset Generator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# CUSTOM CSS
# ==============================================================================

def _inject_css() -> None:
    """
    Inject CSS to polish the Streamlit default theme.

    Design direction: clean data-lab aesthetic — dark sidebar, crisp white
    cards, monospace accents (IBM Plex Mono), display headings (Syne).
    Deliberately avoids purple-gradient AI clichés.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@400;500&display=swap');

        html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }

        /* ── Header ── */
        .sdg-header {
            padding: 1.6rem 0 0.4rem 0;
            border-bottom: 3px solid #3A86FF;
            margin-bottom: 1.8rem;
        }
        .sdg-title {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            font-size: 2.4rem;
            color: #0d1117;
            letter-spacing: -0.5px;
            line-height: 1.1;
        }
        .sdg-subtitle {
            font-family: 'DM Sans', sans-serif;
            font-size: 1rem;
            color: #5a6478;
            margin-top: 0.3rem;
        }
        .sdg-badge {
            display: inline-block;
            background: #0d1117;
            color: #3A86FF;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            padding: 0.2rem 0.65rem;
            border-radius: 20px;
            border: 1px solid #3A86FF;
            margin-left: 0.8rem;
            vertical-align: middle;
            position: relative;
            top: -3px;
        }

        /* ── Section headings ── */
        .sdg-section {
            font-family: 'Syne', sans-serif;
            font-weight: 700;
            font-size: 1.25rem;
            color: #0d1117;
            margin: 2rem 0 0.6rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .sdg-section-num {
            background: #3A86FF;
            color: #fff;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
        }

        /* ── Metric cards ── */
        .sdg-metric-row {
            display: flex;
            gap: 1rem;
            margin: 0.8rem 0 1.2rem 0;
            flex-wrap: wrap;
        }
        .sdg-metric {
            flex: 1;
            min-width: 120px;
            background: #f8f9fc;
            border: 1px solid #e4e8f0;
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
        }
        .sdg-metric-label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            color: #8892a4;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .sdg-metric-value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.5rem;
            font-weight: 600;
            color: #0d1117;
            margin-top: 0.2rem;
        }

        /* ── Score hero cards ── */
        .sdg-score-hero {
            text-align: center;
            padding: 1.5rem 1rem;
            border-radius: 12px;
            border: 2px solid;
            margin-bottom: 1.4rem;
        }
        .sdg-score-label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.4rem;
            opacity: 0.75;
        }
        .sdg-score-number {
            font-family: 'Syne', sans-serif;
            font-size: 3.4rem;
            font-weight: 800;
            line-height: 1;
        }
        .sdg-score-verdict {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.9rem;
            margin-top: 0.5rem;
            opacity: 0.85;
        }
        .sdg-score-sublabel {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.6rem;
            opacity: 0.6;
        }

        /* ── Colour helpers ── */
        .c-good { color: #2DC653; border-color: #2DC653; background: #f0fdf4; }
        .c-mid  { color: #d97706; border-color: #F4A261; background: #fffbf0; }
        .c-bad  { color: #E63946; border-color: #E63946; background: #fff5f5; }

        /* ── Sample dataset buttons ── */
        .sdg-sample-btn button {
            background: #f0f6ff !important;
            border: 1.5px solid #3A86FF !important;
            color: #1a56db !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.8rem !important;
            border-radius: 6px !important;
            transition: all 0.15s !important;
        }
        .sdg-sample-btn button:hover {
            background: #3A86FF !important;
            color: #fff !important;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: #0d1117 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #c9d1d9 !important;
        }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label {
            color: #e6edf3 !important;
            font-family: 'IBM Plex Mono', monospace !important;
        }

        /* ── Misc ── */
        .stDataFrame { border-radius: 8px; overflow: hidden; }
        .streamlit-expanderHeader {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
        }
        div[data-testid="stDownloadButton"] button {
            background: #0d1117;
            color: #3A86FF;
            border: 1.5px solid #3A86FF;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            border-radius: 6px;
            transition: all 0.15s;
        }
        div[data-testid="stDownloadButton"] button:hover {
            background: #3A86FF;
            color: #fff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# SAMPLE DATASETS
# ==============================================================================

def _get_sample_datasets() -> dict[str, pd.DataFrame]:
    """
    Return three hardcoded sample DataFrames for demo use.

    All values are realistic but entirely fictional — no real individuals.
    Each dataset is 20 rows so demo training is fast even on CTGAN.
    """

    # ── 1. Titanic-style passenger data ───────────────────────────────────────
    titanic_df = pd.DataFrame({
        "survived":   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                       0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
        "pclass":     [1, 3, 2, 3, 1, 2, 3, 1, 2, 3,
                       3, 1, 2, 1, 3, 2, 3, 2, 1, 2],
        "sex":        ["female","male","female","male","female",
                       "male","female","male","female","male",
                       "male","female","male","female","male",
                       "female","male","male","female","male"],
        "age":        [29.0, 35.0, 26.0, 45.0, 58.0, 22.0, 31.0, 41.0,
                       19.0, 55.0, 38.0, 24.0, 47.0, 33.0, 27.0, 62.0,
                       20.0, 43.0, 36.0, 28.0],
        "fare":       [211.34, 7.92, 26.00, 8.05, 263.00, 13.00, 7.75,
                       120.00, 11.50, 16.10, 8.05, 151.55, 21.00, 211.34,
                       7.92, 57.00, 7.54, 23.45, 263.00, 30.50],
    })

    # ── 2. Energy / IoT sensor data ───────────────────────────────────────────
    energy_df = pd.DataFrame({
        "timestamp":    [f"2024-03-15 {h:02d}:00:00" for h in range(20)],
        "temperature":  [21.4, 21.1, 20.8, 20.5, 20.2, 20.0, 20.3, 21.7,
                         23.1, 24.5, 25.8, 26.4, 26.9, 27.1, 26.8, 26.0,
                         25.2, 24.1, 23.3, 22.5],
        "voltage":      [229.8, 230.1, 229.6, 230.4, 230.0, 229.9, 230.2,
                         229.7, 230.5, 230.1, 229.8, 230.3, 230.0, 229.6,
                         230.2, 229.9, 230.1, 229.7, 230.4, 230.0],
        "current":      [4.12, 3.98, 3.87, 3.75, 3.68, 3.62, 4.01, 5.23,
                         6.84, 7.92, 8.45, 8.61, 8.73, 8.80, 8.55, 8.12,
                         7.63, 6.94, 6.12, 5.44],
        "power_kw":     [0.947, 0.916, 0.890, 0.863, 0.847, 0.833, 0.922,
                         1.203, 1.573, 1.822, 1.944, 1.981, 2.008, 2.024,
                         1.967, 1.868, 1.755, 1.597, 1.408, 1.251],
    })

    # ── 3. HR / employee data ─────────────────────────────────────────────────
    hr_df = pd.DataFrame({
        "department":       ["Engineering","Sales","HR","Engineering","Finance",
                             "Sales","Engineering","Marketing","HR","Finance",
                             "Engineering","Sales","Marketing","HR","Finance",
                             "Engineering","Sales","Engineering","Marketing","HR"],
        "salary":           [92000, 58000, 51000, 105000, 74000, 62000, 88000,
                             67000, 49000, 80000, 115000, 55000, 71000, 47000,
                             83000, 97000, 60000, 109000, 65000, 53000],
        "years_experience": [6, 3, 4, 9, 5, 2, 7, 4, 3, 6,
                             11, 2, 5, 4, 7, 8, 3, 10, 4, 2],
        "performance_score":[4.1, 3.5, 3.8, 4.6, 4.0, 3.2, 4.3, 3.7, 3.5, 4.1,
                             4.8, 3.0, 3.9, 3.6, 4.2, 4.5, 3.3, 4.7, 3.8, 3.4],
        "attrition":        ["No","Yes","No","No","No","Yes","No","No","Yes","No",
                             "No","Yes","No","No","No","No","Yes","No","No","Yes"],
    })

    return {
        "🚢  Titanic (passenger data)":         titanic_df,
        "⚡  Energy Consumption (sensor data)":  energy_df,
        "👥  HR Dataset (employee data)":        hr_df,
    }


def _load_sample_into_state(label: str, df: pd.DataFrame) -> None:
    """
    Pre-fill session_state as if the user had uploaded a CSV file.

    Clears all downstream state so the UI behaves identically to a fresh
    file upload.  Sets K.LAST_FILE_NAME to a virtual filename so that
    new-upload detection continues to work correctly if the user later uploads
    a real CSV.

    Parameters
    ----------
    label : str  — the sample button label, used to derive a virtual filename
    df    : pd.DataFrame — the sample data to load
    """
    # Derive a safe virtual filename from the human-readable label
    safe = (
        label.split("(")[0]
        .strip()
        .lstrip("🚢⚡👥 ")
        .replace(" ", "_")
        .lower()
    )
    virtual_name = f"sample_{safe}.csv"

    # Clear ALL downstream state — sample swap must be as clean as a new upload
    for key in [K.REAL_DF, K.METADATA_DICT, K.OVERRIDES,
                K.SYNTHETIC_DF, K.MODEL_INFO, K.EVAL_RESULTS,
                K.PRIVACY_RESULTS, K.GENERATION_DONE]:
        st.session_state.pop(key, None)

    st.session_state[K.LAST_FILE_NAME] = virtual_name
    st.session_state[K.REAL_DF]        = df.copy()

    # Run metadata detection immediately (same flow as the real-upload path)
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    st.session_state[K.METADATA_DICT] = meta.to_dict()
    st.session_state[K.OVERRIDES]     = {}


# ==============================================================================
# SECTION 1 — HEADER
# ==============================================================================

def render_header() -> None:
    """Render the app title, GAN-POWERED badge, and one-line description."""
    st.markdown(
        """
        <div class="sdg-header">
            <div class="sdg-title">
                Synthetic Dataset Generator
                <span class="sdg-badge">⚡ GAN-POWERED</span>
            </div>
            <div class="sdg-subtitle">
                Upload a real CSV → train a generative model → download a
                privacy-safe synthetic twin that mirrors your data's statistics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 2 — UPLOAD + SAMPLE DATASETS + PREVIEW + COLUMN-TYPE EDITOR
# ==============================================================================

def render_upload_section() -> None:
    """
    Handle the full dataset ingestion workflow:

      a) "Try a sample" expander with three one-click presets
      b) File uploader (CSV only)
      c) New-file detection → clears downstream state
      d) CSV parsing + row cap
      e) SDV metadata detection (cached per file/sample)
      f) Summary metric cards (rows, columns, missing values, detected types)
      g) Data preview — first 5 rows
      h) Column-type correction selectboxes (optional, 3-column grid)
    """
    st.markdown(
        '<div class="sdg-section"><span class="sdg-section-num">01</span>'
        'Upload Dataset</div>',
        unsafe_allow_html=True,
    )

    # ── 2a. Sample datasets ───────────────────────────────────────────────────
    with st.expander("✨ Try a sample dataset — no upload needed", expanded=False):
        st.caption(
            "Click a button to pre-load a realistic dataset and explore the "
            "full generation pipeline without needing your own CSV file."
        )
        samples    = _get_sample_datasets()
        btn_cols   = st.columns(len(samples))

        for col, (label, sample_df) in zip(btn_cols, samples.items()):
            safe_key = (
                label.replace(" ", "_")
                     .replace("(", "").replace(")", "")
                     .replace(".", "")
            )
            with col:
                st.markdown('<div class="sdg-sample-btn">', unsafe_allow_html=True)
                if st.button(label, key=f"sample_{safe_key}", use_container_width=True):
                    _load_sample_into_state(label, sample_df)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        current = st.session_state.get(K.LAST_FILE_NAME, "")
        if current.startswith("sample_"):
            nice = (
                current.replace("sample_", "")
                       .replace("_", " ")
                       .replace(".csv", "")
                       .title()
            )
            st.success(f"✅ Sample loaded: **{nice}**")

    # ── 2b. File uploader ─────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Or upload your own CSV file",
        type=["csv"],
        help=f"Maximum {MAX_UPLOAD_ROWS:,} rows. Larger files will be trimmed.",
    )

    # ── 2c. New-file detection ────────────────────────────────────────────────
    if uploaded_file is not None:
        is_new_file = (
            uploaded_file.name != st.session_state.get(K.LAST_FILE_NAME)
        )
        if is_new_file:
            for key in [K.REAL_DF, K.METADATA_DICT, K.OVERRIDES,
                        K.SYNTHETIC_DF, K.MODEL_INFO, K.EVAL_RESULTS,
                        K.PRIVACY_RESULTS, K.GENERATION_DONE]:
                st.session_state.pop(key, None)
            st.session_state[K.LAST_FILE_NAME] = uploaded_file.name

        # ── 2d. Parse CSV ──────────────────────────────────────────────────────
        if K.REAL_DF not in st.session_state:
            df = _parse_csv(uploaded_file)
            if df is None:
                return
            st.session_state[K.REAL_DF] = df
            # Run metadata detection immediately after parse
            meta = SingleTableMetadata()
            meta.detect_from_dataframe(df)
            st.session_state[K.METADATA_DICT] = meta.to_dict()
            st.session_state[K.OVERRIDES]     = {}

    # Show a prompt if neither a file nor a sample has been loaded
    if K.REAL_DF not in st.session_state:
        st.info(
            "👆 Click a sample button above, or upload a CSV file to get started."
        )
        return

    real_df:       pd.DataFrame = st.session_state[K.REAL_DF]
    metadata_dict: dict         = st.session_state[K.METADATA_DICT]
    overrides:     dict         = st.session_state[K.OVERRIDES]

    # ── 2f. Summary metrics ───────────────────────────────────────────────────
    detected_cols = metadata_dict.get("columns", {})
    type_counts: dict[str, int] = {}
    for info in detected_cols.values():
        t = info.get("sdtype", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    type_summary = "  ·  ".join(
        f"{n} {t}" for t, n in sorted(type_counts.items())
    )

    st.markdown(
        f"""
        <div class="sdg-metric-row">
            <div class="sdg-metric">
                <div class="sdg-metric-label">Rows</div>
                <div class="sdg-metric-value">{len(real_df):,}</div>
            </div>
            <div class="sdg-metric">
                <div class="sdg-metric-label">Columns</div>
                <div class="sdg-metric-value">{real_df.shape[1]:,}</div>
            </div>
            <div class="sdg-metric">
                <div class="sdg-metric-label">Missing values</div>
                <div class="sdg-metric-value">{real_df.isna().sum().sum():,}</div>
            </div>
            <div class="sdg-metric">
                <div class="sdg-metric-label">Detected types</div>
                <div class="sdg-metric-value" style="font-size:0.85rem;margin-top:0.4rem">
                    {type_summary or "—"}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2g. Data preview ──────────────────────────────────────────────────────
    with st.expander("🔍 Data preview — first 5 rows", expanded=True):
        st.dataframe(real_df.head(5), use_container_width=True)

    # ── 2h. Column-type correction ────────────────────────────────────────────
    with st.expander("⚙️ Correct column types (optional)", expanded=False):
        st.caption(
            "SDV auto-detected the types below. "
            "Fix any surprises — especially ID columns or numerically-encoded "
            "categoricals — before clicking Train & Generate."
        )
        cols_list = list(detected_cols.keys())
        grid      = st.columns(3)

        for idx, col_name in enumerate(cols_list):
            detected_type = detected_cols[col_name].get("sdtype", "categorical")
            current_type  = overrides.get(col_name, detected_type)

            # Guard: if detected_type isn't in our options list, fall back safely
            if current_type not in SDTYPE_OPTIONS:
                current_type = "categorical"

            chosen = grid[idx % 3].selectbox(
                label=col_name,
                options=SDTYPE_OPTIONS,
                index=SDTYPE_OPTIONS.index(current_type),
                key=f"type_override_{col_name}",
                help=(
                    "categorical → discrete labels (gender, country…)\n"
                    "numerical   → continuous / integer numbers\n"
                    "datetime    → date or timestamp strings\n"
                    "boolean     → True / False\n"
                    "id          → unique identifier (excluded from synthesis)"
                ),
            )

            # Only store the override when it diverges from the detected default
            if chosen != detected_type:
                st.session_state[K.OVERRIDES][col_name] = chosen
            elif col_name in st.session_state[K.OVERRIDES]:
                del st.session_state[K.OVERRIDES][col_name]


# ==============================================================================
# SECTION 3 — SIDEBAR
# ==============================================================================

def render_sidebar(real_df: Optional[pd.DataFrame]) -> dict:
    """
    Render the configuration sidebar and return the user's settings.

    Returns
    -------
    dict
        model_name, epochs, batch_size, num_rows, run_clicked
    """
    st.sidebar.markdown(
        "<h2 style='font-size:1.1rem;margin-bottom:1rem'>⚙️ Configuration</h2>",
        unsafe_allow_html=True,
    )

    # ── Model selector ────────────────────────────────────────────────────────
    st.sidebar.markdown("**Model**")
    model_name = st.sidebar.selectbox(
        "Generative model",
        options=get_available_models(),   # ["CTGAN", "GaussianCopula", "TVAE"]
        index=0,                          # CTGAN is index 0 — recommended default
        label_visibility="collapsed",
        help=(
            "CTGAN — Conditional GAN; best fidelity on mixed-type data, slowest.\n"
            "TVAE  — Variational Autoencoder; fast convergence, good all-round.\n"
            "GaussianCopula — Statistical copula; trains in seconds, no GPU needed."
        ),
    )
    model_blurbs = {
        "CTGAN":          "🔥 Conditional GAN — highest fidelity, most compute",
        "TVAE":           "⚡ Variational Autoencoder — fast & reliable",
        "GaussianCopula": "📐 Statistical copula — instant, no GPU needed",
    }
    st.sidebar.caption(model_blurbs.get(model_name, ""))

    # ── Row count slider ──────────────────────────────────────────────────────
    st.sidebar.markdown("**Rows to generate**")
    if real_df is not None:
        min_rows     = 100
        max_rows     = min(len(real_df) * 10, MAX_UPLOAD_ROWS)
        default_rows = len(real_df)
    else:
        min_rows, max_rows, default_rows = 100, 10_000, 1_000

    num_rows = st.sidebar.slider(
        "Synthetic rows",
        min_value=min_rows,
        max_value=max(min_rows, max_rows),
        value=min(default_rows, max(min_rows, max_rows)),
        step=max(100, (max(min_rows, max_rows) - min_rows) // 50),
        label_visibility="collapsed",
        help="1× original size preserves statistical ratios.",
        disabled=(real_df is None),
    )

    # ── Epoch slider (DL models only) ─────────────────────────────────────────
    is_dl_model = model_name in ("CTGAN", "TVAE")
    epochs      = DEFAULT_EPOCHS
    batch_size  = DEFAULT_BATCH

    if is_dl_model:
        st.sidebar.markdown("**Training epochs**")
        epochs = st.sidebar.slider(
            "Epochs",
            min_value=MIN_EPOCHS,
            max_value=MAX_EPOCHS,
            value=DEFAULT_EPOCHS,
            step=50,
            label_visibility="collapsed",
            help="More epochs → higher fidelity, longer runtime.",
        )
        est = _estimate_time(model_name, epochs)
        st.sidebar.caption(f"Estimated time: ~{est} min (CPU; varies by dataset)")
    else:
        st.sidebar.info(
            "GaussianCopula is a closed-form statistical fit — "
            "no epoch configuration needed.",
            icon="ℹ️",
        )

    st.sidebar.markdown("---")

    # ── Train & Generate button ───────────────────────────────────────────────
    run_clicked = st.sidebar.button(
        "🚀  Train & Generate",
        type="primary",
        use_container_width=True,
        disabled=(real_df is None),
        help="Load a dataset first (upload or sample).",
    )
    if real_df is None:
        st.sidebar.caption("⬆️ Load a dataset on the right to enable training.")

    return {
        "model_name":  model_name,
        "epochs":      epochs,
        "batch_size":  batch_size,
        "num_rows":    num_rows,
        "run_clicked": run_clicked,
    }


# ==============================================================================
# GENERATION + EVALUATION PIPELINE
# ==============================================================================

def run_generation(config: dict) -> None:
    """
    Execute the full generate → quality-evaluate → privacy-check pipeline.

    Called only when the Train & Generate button is clicked.  All errors are
    caught and surfaced via st.error() / st.warning() with actionable hints —
    no raw tracebacks reach the user.

    On success, stores results in session_state and calls st.rerun() so the
    Results and Comparison sections render immediately.

    Parameters
    ----------
    config : dict  — settings dict returned by render_sidebar()
    """
    real_df:       pd.DataFrame = st.session_state[K.REAL_DF]
    metadata_dict: dict         = st.session_state[K.METADATA_DICT]
    overrides:     dict         = st.session_state.get(K.OVERRIDES, {})

    # Merge user type overrides into a working copy — never mutate the cache
    effective_metadata = _apply_overrides(metadata_dict, overrides)

    # ── Step 1: Synthesizer ───────────────────────────────────────────────────
    spinner_msg = (
        f"Training {config['model_name']} on your data"
        + (
            f" for {config['epochs']} epochs"
            if config["model_name"] in ("CTGAN", "TVAE")
            else ""
        )
        + " — this may take a minute…"
    )

    with st.spinner(spinner_msg):
        try:
            synthetic_df, model_info = generate_synthetic_data(
                real_df    = real_df,
                model_name = config["model_name"],
                num_rows   = config["num_rows"],
                epochs     = config["epochs"],
                batch_size = config["batch_size"],
                verbose    = False,
            )
        except ValueError as exc:
            st.error(
                f"**Configuration error:** {exc}\n\n"
                "Check your column types and dataset size, then try again."
            )
            logger.error("Generation ValueError: %s", exc)
            return
        except RuntimeError as exc:
            st.error(
                f"**Training failed:** {exc}\n\n"
                "Possible causes:\n"
                "- Dataset is very small (<50 rows) — try GaussianCopula instead\n"
                "- A column is entirely null — remove it in the column-type editor\n"
                "- Memory pressure — reduce the number of synthetic rows"
            )
            logger.error("Generation RuntimeError:\n%s", traceback.format_exc())
            return
        except Exception as exc:
            st.error(
                f"**Unexpected error:** `{type(exc).__name__}: {exc}`\n\n"
                "Check the terminal for a full traceback."
            )
            logger.error("Generation unexpected:\n%s", traceback.format_exc())
            return

    st.session_state[K.SYNTHETIC_DF] = synthetic_df
    st.session_state[K.MODEL_INFO]   = model_info

    # ── Step 2: Quality evaluation ────────────────────────────────────────────
    with st.spinner("Evaluating synthetic data quality with SDMetrics…"):
        try:
            eval_results = evaluate_synthetic_data(
                real_df       = real_df,
                synthetic_df  = synthetic_df,
                metadata_dict = effective_metadata,
            )
        except Exception as exc:
            st.warning(
                f"⚠️ Quality evaluation failed: `{exc}`\n\n"
                "Synthetic data was generated successfully and can still be "
                "downloaded — quality scores are unavailable."
            )
            logger.warning("Evaluation error: %s", exc)
            eval_results = None

    st.session_state[K.EVAL_RESULTS] = eval_results

    # ── Step 3: Privacy risk ──────────────────────────────────────────────────
    with st.spinner("Computing privacy risk (NewRowSynthesis)…"):
        try:
            privacy_results = compute_privacy_risk(
                real_df       = real_df,
                synthetic_df  = synthetic_df,
                metadata_dict = effective_metadata,
            )
        except Exception as exc:
            logger.warning("Privacy check error: %s", exc)
            privacy_results = {
                "privacy_risk":   None,
                "new_row_rate":   None,
                "risk_label":     "Unavailable",
                "risk_css_class": "c-mid",
                "error":          str(exc),
            }

    st.session_state[K.PRIVACY_RESULTS] = privacy_results
    st.session_state[K.GENERATION_DONE] = True

    # Force immediate re-render so Results and Comparison sections appear
    st.rerun()


# ==============================================================================
# SECTION 4 — RESULTS
# ==============================================================================

def render_results_section() -> None:
    """
    Render the Results section (visible only after a successful generation run).

    Shows:
      - Generation timing metric cards
      - Quality Score hero card  (high = green)
      - Privacy Risk hero card   (high = red — inverted colour logic)
      - Property-level sub-scores
      - Column-level quality bar chart
      - Download button for the synthetic CSV
      - Collapsed synthetic data preview
    """
    if not st.session_state.get(K.GENERATION_DONE):
        return

    st.markdown(
        '<div class="sdg-section"><span class="sdg-section-num">04</span>'
        'Results</div>',
        unsafe_allow_html=True,
    )

    synthetic_df:    pd.DataFrame   = st.session_state[K.SYNTHETIC_DF]
    model_info:      dict           = st.session_state[K.MODEL_INFO]
    eval_results:    Optional[dict] = st.session_state.get(K.EVAL_RESULTS)
    privacy_results: Optional[dict] = st.session_state.get(K.PRIVACY_RESULTS)

    # ── Generation timing ─────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="sdg-metric-row">
            <div class="sdg-metric">
                <div class="sdg-metric-label">Synthetic rows</div>
                <div class="sdg-metric-value">{len(synthetic_df):,}</div>
            </div>
            <div class="sdg-metric">
                <div class="sdg-metric-label">Model used</div>
                <div class="sdg-metric-value" style="font-size:1.05rem">
                    {model_info.get("model_name", "—")}
                </div>
            </div>
            <div class="sdg-metric">
                <div class="sdg-metric-label">Train time</div>
                <div class="sdg-metric-value">{model_info.get("train_time_s","—")}s</div>
            </div>
            <div class="sdg-metric">
                <div class="sdg-metric-label">Sample time</div>
                <div class="sdg-metric-value">{model_info.get("sample_time_s","—")}s</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Score hero cards (side by side) ──────────────────────────────────────
    hero_left, hero_right = st.columns(2)
    with hero_left:
        _render_quality_hero(eval_results)
    with hero_right:
        _render_privacy_hero(privacy_results)

    # ── Property sub-scores ───────────────────────────────────────────────────
    if eval_results and not eval_results.get("error"):
        prop_scores = eval_results.get("property_scores", {})
        if prop_scores:
            p_cols = st.columns(len(prop_scores))
            for i, (prop, val) in enumerate(prop_scores.items()):
                p_cols[i].metric(label=prop, value=f"{val:.1%}")

    # ── Column-level quality bar chart ────────────────────────────────────────
    if eval_results and not eval_results.get("error"):
        col_bd: pd.DataFrame = eval_results.get("column_breakdown", pd.DataFrame())
        if not col_bd.empty and "Quality Score" in col_bd.columns:
            _render_column_bar_chart(col_bd)

    # ── Download button ───────────────────────────────────────────────────────
    st.markdown("#### ⬇️ Download Synthetic Data")
    file_stem = st.session_state.get(K.LAST_FILE_NAME, "data.csv")
    csv_bytes = synthetic_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download  synthetic_{file_stem}",
        data=csv_bytes,
        file_name=f"synthetic_{file_stem}",
        mime="text/csv",
    )

    # ── Synthetic data preview ─────────────────────────────────────────────────
    with st.expander("🔍 Preview synthetic data — first 5 rows", expanded=False):
        st.dataframe(synthetic_df.head(5), use_container_width=True)


def _render_quality_hero(eval_results: Optional[dict]) -> None:
    """
    Render the Quality Score hero card.  High score → green (desirable).
    """
    if eval_results and not eval_results.get("error"):
        score              = eval_results["overall_score"]
        css_cls, verdict   = _quality_class_and_verdict(score)
        score_display      = f"{score:.0%}"
    else:
        css_cls       = "c-mid"
        score_display = "N/A"
        verdict       = "⚠️ Evaluation unavailable"

    st.markdown(
        f"""
        <div class="sdg-score-hero {css_cls}">
            <div class="sdg-score-label">Quality Score</div>
            <div class="sdg-score-number">{score_display}</div>
            <div class="sdg-score-verdict">{verdict}</div>
            <div class="sdg-score-sublabel">
                higher is better · SDMetrics QualityReport
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_privacy_hero(privacy_results: Optional[dict]) -> None:
    """
    Render the Privacy Risk hero card.

    Colour logic is INVERTED vs. quality:
      Low risk   → green  (c-good)  — most rows are genuinely novel
      Medium     → amber  (c-mid)
      High risk  → red    (c-bad)   — memorisation of real rows detected

    The new_row_rate (fraction of synthetic rows that are genuinely novel)
    is shown as a sublabel for transparency.
    """
    if not privacy_results or privacy_results.get("error"):
        error_msg = (
            privacy_results.get("error", "Not computed")
            if privacy_results
            else "Not computed"
        )
        st.markdown(
            f"""
            <div class="sdg-score-hero c-mid">
                <div class="sdg-score-label">Privacy Risk</div>
                <div class="sdg-score-number">N/A</div>
                <div class="sdg-score-verdict">⚠️ Unavailable</div>
                <div class="sdg-score-sublabel">{error_msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    risk        = privacy_results["privacy_risk"]
    css_cls     = privacy_results["risk_css_class"]
    risk_label  = privacy_results["risk_label"]
    new_row_pct = privacy_results.get("new_row_rate", 0.0) or 0.0

    if risk is None:
        risk_display = "N/A"
        verdict      = "—"
        sublabel     = "Computation unavailable"
    else:
        risk_display = f"{risk:.0%}"
        verdict      = _privacy_verdict(risk_label)
        sublabel     = (
            f"{risk_label} risk  ·  "
            f"{new_row_pct:.0%} of synthetic rows are novel"
        )

    st.markdown(
        f"""
        <div class="sdg-score-hero {css_cls}">
            <div class="sdg-score-label">Privacy Risk</div>
            <div class="sdg-score-number">{risk_display}</div>
            <div class="sdg-score-verdict">{verdict}</div>
            <div class="sdg-score-sublabel">{sublabel}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_column_bar_chart(col_breakdown: pd.DataFrame) -> None:
    """
    Render a Plotly horizontal bar chart of per-column quality scores.

    Bars coloured by threshold: green ≥ 0.80, amber 0.50–0.79, red < 0.50.
    Lowest-quality columns appear at the top (ascending sort from evaluator).
    """
    st.markdown("#### Column-level Quality Breakdown")

    # Cap display at 40 rows for readability; full table in collapsed expander
    display_df = col_breakdown.tail(40)

    bar_colors = [
        COLOR_GOOD if s >= 0.80 else (COLOR_MID if s >= 0.50 else COLOR_BAD)
        for s in display_df["Quality Score"]
    ]

    fig = go.Figure(go.Bar(
        x=display_df["Quality Score"],
        y=display_df["Column"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{s:.0%}" for s in display_df["Quality Score"]],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1.12], tickformat=".0%",
                   showgrid=True, gridcolor="#e8edf3"),
        yaxis=dict(autorange="reversed"),
        height=max(300, len(display_df) * 28 + 80),
        margin=dict(l=10, r=60, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,252,1)",
        font=dict(family="IBM Plex Mono, monospace", size=11),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Full column breakdown table", expanded=False):
        st.dataframe(
            col_breakdown.style
            .background_gradient(cmap="RdYlGn", subset=["Quality Score"], vmin=0, vmax=1)
            .format({"Quality Score": "{:.1%}"}),
            use_container_width=True,
        )


# ==============================================================================
# SECTION 5 — COMPARISON
# ==============================================================================

def render_comparison_section() -> None:
    """
    Render side-by-side distribution charts: real (blue) vs. synthetic (orange).

    Numeric  → overlapping probability-density histograms
    Categorical / boolean → grouped proportional bar charts

    For wide datasets (> MAX_CHART_COLS columns) a multiselect widget lets
    the user choose which columns to display.
    """
    if not st.session_state.get(K.GENERATION_DONE):
        return

    real_df:  pd.DataFrame = st.session_state[K.REAL_DF]
    synth_df: pd.DataFrame = st.session_state[K.SYNTHETIC_DF]

    shared_cols = [c for c in real_df.columns if c in synth_df.columns]
    if not shared_cols:
        return

    st.markdown(
        '<div class="sdg-section"><span class="sdg-section-num">05</span>'
        'Distribution Comparison</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "🔵 Blue = real data  ·  🟠 Orange = synthetic data  ·  "
        "Closer overlap = higher column fidelity."
    )

    # Column selector — only shown for wide datasets
    if len(shared_cols) > MAX_CHART_COLS:
        selected_cols = st.multiselect(
            f"Select columns to compare (showing first {MAX_CHART_COLS} by default):",
            options=shared_cols,
            default=shared_cols[:MAX_CHART_COLS],
            help=f"Dataset has {len(shared_cols)} columns. "
                 f"Limit to {MAX_CHART_COLS} for readability.",
        )
        if not selected_cols:
            st.info("Select at least one column above to view charts.")
            return
    else:
        selected_cols = shared_cols

    # 2-column chart grid
    left_col, right_col = st.columns(2)
    panel = {0: left_col, 1: right_col}

    for idx, col_name in enumerate(selected_cols):
        fig = _build_distribution_figure(real_df, synth_df, col_name)
        panel[idx % 2].plotly_chart(fig, use_container_width=True)


def _build_distribution_figure(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    column: str,
) -> go.Figure:
    """
    Build a Plotly distribution comparison figure for a single column.

    Numeric     → overlapping probability-density histograms
                  (histnorm="probability density" normalises for different
                  row counts — real and synthetic are directly comparable)
    Categorical → grouped proportional bar chart
                  (value_counts(normalize=True) ensures proportions sum to 1.0
                  regardless of row-count difference)

    BUG FIX: pandas 2.x `value_counts().reset_index()` names the first column
    after the original Series, not "index".  The old code silently produced
    wrong column names.  Fixed by building freq_df explicitly from aligned
    index lists instead of relying on reset_index() column naming.

    Parameters
    ----------
    real_df, synth_df : pd.DataFrame
    column : str

    Returns
    -------
    go.Figure
    """
    real_s = real_df[column].dropna()
    synth_s = synth_df[column].dropna()
    is_num  = pd.api.types.is_numeric_dtype(real_s)

    fig = go.Figure()

    if is_num:
        fig.add_trace(go.Histogram(
            x=real_s, name="Real",
            histnorm="probability density",
            opacity=0.65, marker_color=COLOR_REAL, nbinsx=35,
            hovertemplate="Value: %{x}<br>Density: %{y:.4f}<extra>Real</extra>",
        ))
        fig.add_trace(go.Histogram(
            x=synth_s, name="Synthetic",
            histnorm="probability density",
            opacity=0.65, marker_color=COLOR_SYNTH, nbinsx=35,
            hovertemplate="Value: %{x}<br>Density: %{y:.4f}<extra>Synthetic</extra>",
        ))
        fig.update_layout(barmode="overlay")

    else:
        real_freq  = real_s.value_counts(normalize=True)
        synth_freq = synth_s.value_counts(normalize=True)

        # Build freq_df explicitly — avoids pandas 2.x reset_index() naming quirk
        all_cats = sorted(set(real_freq.index) | set(synth_freq.index), key=str)
        freq_df  = pd.DataFrame({
            "Category":  [str(c) for c in all_cats],
            "Real":      [float(real_freq.get(c, 0.0))  for c in all_cats],
            "Synthetic": [float(synth_freq.get(c, 0.0)) for c in all_cats],
        })

        # Cap at top-15 categories by real proportion for readability
        if len(freq_df) > 15:
            freq_df = freq_df.nlargest(15, "Real").reset_index(drop=True)

        fig.add_trace(go.Bar(
            x=freq_df["Category"], y=freq_df["Real"],
            name="Real", marker_color=COLOR_REAL, opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            x=freq_df["Category"], y=freq_df["Synthetic"],
            name="Synthetic", marker_color=COLOR_SYNTH, opacity=0.85,
        ))
        fig.update_layout(barmode="group")

    # Shared layout for both chart types
    fig.update_layout(
        title=dict(
            text=column,
            font=dict(family="IBM Plex Mono, monospace", size=13, color="#0d1117"),
        ),
        height=280,
        margin=dict(t=42, b=36, l=36, r=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,252,1)",
        legend=dict(orientation="h", y=1.18, x=0, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor="#e8edf3"),
        yaxis=dict(showgrid=True, gridcolor="#e8edf3"),
        font=dict(family="DM Sans, sans-serif", size=11),
    )
    return fig


# ==============================================================================
# PRIVATE UTILITIES
# ==============================================================================

def _parse_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Parse the Streamlit UploadedFile into a pandas DataFrame.

    Returns None (after calling st.error) if:
      - The file is unparseable
      - The file is empty
      - The file has fewer than 2 columns
    Caps at MAX_UPLOAD_ROWS with a warning.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(
            f"**Could not parse the uploaded file as CSV.**\n\n"
            f"Error: `{exc}`\n\n"
            "Ensure the file is UTF-8, comma-delimited, and has a header row."
        )
        return None

    if df.empty:
        st.error(
            "**The uploaded CSV is empty.**  "
            "Please upload a file with at least one row of data."
        )
        return None

    if df.shape[1] < 2:
        st.error(
            "**The CSV has fewer than 2 columns.**\n\n"
            "Generative models require at least 2 columns to learn "
            "inter-column relationships."
        )
        return None

    if len(df) > MAX_UPLOAD_ROWS:
        st.warning(
            f"⚠️ Dataset has **{len(df):,} rows**; only the first "
            f"**{MAX_UPLOAD_ROWS:,}** will be used."
        )
        df = df.head(MAX_UPLOAD_ROWS)

    return df


def _apply_overrides(metadata_dict: dict, overrides: dict) -> dict:
    """
    Return a deep copy of metadata_dict with user type overrides merged in.

    Pure function — never mutates the session_state cache, so the user can
    freely adjust overrides and re-run without stale-state issues.
    """
    effective = copy.deepcopy(metadata_dict)
    for col_name, sdtype in overrides.items():
        if col_name in effective.get("columns", {}):
            effective["columns"][col_name]["sdtype"] = sdtype
    return effective


def _estimate_time(model_name: str, epochs: int) -> str:
    """
    Heuristic estimate of training time for the sidebar caption.

    Based on informal CPU benchmarks; purely indicative.
    Returns a "min–max" string, e.g. "5–10".
    """
    spe = {"CTGAN": 1.2, "TVAE": 0.6}.get(model_name, 0.3)
    lo  = spe * epochs / 60
    return f"{lo:.0f}–{lo * 2:.0f}"


def _quality_class_and_verdict(score: float) -> tuple[str, str]:
    """Map a quality score to a CSS class and verdict string (high = good)."""
    if score >= 0.80:
        return "c-good", "✅ Excellent — synthetic data closely mirrors the original"
    elif score >= 0.50:
        return "c-mid",  "⚠️ Fair — some distributions may differ from the original"
    return "c-bad", "❌ Poor — consider more epochs or a different model"


def _privacy_verdict(risk_label: str) -> str:
    """
    Convert a risk label string to a human-readable verdict with an emoji.

    Extracted from _render_privacy_hero to remove the fragile triple-ternary
    that was present in the previous version.

    Parameters
    ----------
    risk_label : str — "Low" | "Medium" | "High" | "Unavailable"
    """
    mapping = {
        "Low":    "🟢 Rows are genuinely novel — low memorisation",
        "Medium": "🟡 Some synthetic rows are close to real rows",
        "High":   "🔴 Memorisation detected — rows resemble real data",
    }
    return mapping.get(risk_label, "—")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """
    Top-level orchestrator.  Streamlit reruns this on every widget interaction.

    Rendering order:
      0. CSS injection
      1. Header
      2. Upload section      → writes K.REAL_DF, K.METADATA_DICT, K.OVERRIDES
      3. Sidebar             → reads K.REAL_DF; returns config dict
      4. Generation trigger  → only when button clicked AND data is loaded
      5. Results section     → visible only when K.GENERATION_DONE is True
      6. Comparison section  → visible only when K.GENERATION_DONE is True
    """
    _inject_css()
    render_header()
    render_upload_section()

    real_df: Optional[pd.DataFrame] = st.session_state.get(K.REAL_DF)
    config = render_sidebar(real_df)

    if config["run_clicked"] and real_df is not None:
        # Clear previous results before a new run so stale data is never shown
        for key in [K.SYNTHETIC_DF, K.MODEL_INFO, K.EVAL_RESULTS,
                    K.PRIVACY_RESULTS, K.GENERATION_DONE]:
            st.session_state.pop(key, None)
        run_generation(config)
        # run_generation calls st.rerun() on success — execution stops here

    if st.session_state.get(K.GENERATION_DONE):
        render_results_section()
        st.markdown("---")
        render_comparison_section()


if __name__ == "__main__":
    main()
