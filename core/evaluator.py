"""
core/evaluator.py — SDMetrics Quality & Privacy Evaluation
===========================================================
Measures two complementary dimensions of synthetic data quality:

  1. **Fidelity (Quality Score)**
     How statistically similar is the synthetic data to the real data?
     Wraps SDMetrics' QualityReport, which evaluates:
       - Column Shapes      : marginal distributions per column
       - Column Pair Trends : pairwise correlations / contingency tables
     The overall score is the unweighted average of both properties (0–1).

  2. **Privacy Risk**
     Could an attacker reconstruct real rows from the synthetic data?
     Wraps SDMetrics' NewRowSynthesis metric, which measures what fraction
     of synthetic rows are genuine novel rows vs. near-copies of real rows.
     A high "new row rate" means low privacy risk — and vice-versa.

Public API
----------
    evaluate_synthetic_data(real_df, synthetic_df, metadata_dict) -> dict
    compute_privacy_risk(real_df, synthetic_df, metadata_dict)    -> dict
    score_summary_text(results)                                   -> str

Both evaluation functions return plain dicts (never raise) so the Streamlit
UI layer can handle errors gracefully without try/except boilerplate.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np
import pandas as pd

from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import NewRowSynthesis
from sdv.metadata import SingleTableMetadata

logger = logging.getLogger(__name__)


# ===========================================================================
# Public API — Quality
# ===========================================================================

def evaluate_synthetic_data(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata_dict: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
    """
    Compute SDMetrics quality scores comparing real vs. synthetic data.

    Internally runs SDMetrics' QualityReport across two property groups:
      - "Column Shapes"      — how well each column's marginal distribution
                               (histogram / PMF) is reproduced.
      - "Column Pair Trends" — how well pairwise column relationships
                               (correlations, contingency tables) are preserved.

    Parameters
    ----------
    real_df : pd.DataFrame
        The original dataset uploaded by the user.
    synthetic_df : pd.DataFrame
        The dataset produced by synthesizer.generate_synthetic_data().
    metadata_dict : dict, optional
        SDV metadata dict (from model_info["metadata"]).  If None, metadata
        is auto-detected from real_df — useful for standalone evaluation calls.
    verbose : bool
        If True, SDMetrics prints a progress summary to stdout.

    Returns
    -------
    dict
        "overall_score"    : float in [0, 1]
        "property_scores"  : dict[str, float]  — per-property breakdown
        "column_breakdown" : pd.DataFrame       — per-column scores
        "report_object"    : QualityReport | None
        "error"            : str | None         — None on success
    """
    # 1. Build metadata
    try:
        metadata = _build_metadata(real_df, metadata_dict)
    except Exception as exc:
        logger.error("Metadata construction failed: %s", exc)
        return _error_result(f"Metadata error: {exc}")

    # 2. Align columns (SDMetrics requires identical column sets)
    try:
        real_df, synthetic_df = _align_columns(real_df, synthetic_df)
    except Exception as exc:
        logger.error("Column alignment failed: %s", exc)
        return _error_result(f"Column mismatch: {exc}")

    # 3. Run QualityReport
    report = QualityReport()
    try:
        report.generate(
            real_data=real_df,
            synthetic_data=synthetic_df,
            metadata=metadata.to_dict(),
            verbose=verbose,
        )
    except Exception as exc:
        logger.error("QualityReport generation failed: %s", exc)
        return _error_result(f"SDMetrics evaluation error: {exc}")

    # 4. Extract scores
    try:
        overall_score    = _safe_float(report.get_score())
        property_scores  = _extract_property_scores(report)
        column_breakdown = _extract_column_breakdown(report)
    except Exception as exc:
        logger.error("Score extraction failed: %s", exc)
        return _error_result(f"Score extraction error: {exc}")

    logger.info(
        "Quality evaluation complete | score=%.4f | columns=%d",
        overall_score, len(column_breakdown),
    )

    return {
        "overall_score":    overall_score,
        "property_scores":  property_scores,
        "column_breakdown": column_breakdown,
        "report_object":    report,
        "error":            None,
    }


# ===========================================================================
# Public API — Privacy
# ===========================================================================

def compute_privacy_risk(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata_dict: Optional[dict] = None,
) -> dict:
    """
    Estimate privacy risk using SDMetrics' NewRowSynthesis metric.

    NewRowSynthesis measures what fraction of synthetic rows are genuinely
    novel (i.e. not a near-copy of any real row).  A synthetic row is
    considered a "copy" if its nearest-neighbour distance in the real dataset
    falls below a statistical threshold derived from real–real distances.

    Privacy risk is defined here as the complement of the new-row rate:
        privacy_risk = 1 − new_row_synthesis_score

    Interpretation:
      - privacy_risk ≈ 0.0  → virtually all rows are novel → LOW risk
      - privacy_risk ≈ 0.5  → half of rows are near-copies → MEDIUM risk
      - privacy_risk ≈ 1.0  → most rows memorised real data → HIGH risk

    Parameters
    ----------
    real_df : pd.DataFrame
        Original dataset.
    synthetic_df : pd.DataFrame
        Generated synthetic dataset.
    metadata_dict : dict, optional
        SDV metadata dict.  Auto-detected from real_df if None.

    Returns
    -------
    dict
        "privacy_risk"   : float in [0, 1] | None on failure
        "new_row_rate"   : float in [0, 1] | None on failure  (= 1 − risk)
        "risk_label"     : str  — "Low" | "Medium" | "High" | "Unavailable"
        "risk_css_class" : str  — Streamlit CSS class for colour coding
                                  ("c-good", "c-mid", "c-bad")
        "error"          : str | None
    """
    # Build metadata
    try:
        metadata = _build_metadata(real_df, metadata_dict)
    except Exception as exc:
        logger.error("Privacy metadata error: %s", exc)
        return _privacy_error_result(f"Metadata error: {exc}")

    # Align columns
    try:
        real_aligned, synth_aligned = _align_columns(real_df, synthetic_df)
    except Exception as exc:
        logger.error("Privacy column alignment error: %s", exc)
        return _privacy_error_result(f"Column mismatch: {exc}")

    # Run NewRowSynthesis
    # NewRowSynthesis.compute() returns a single float in [0, 1]:
    #   1.0 → all synthetic rows are novel       (best privacy)
    #   0.0 → all synthetic rows copy real rows  (worst privacy)
    try:
        new_row_rate = NewRowSynthesis.compute(
            real_data=real_aligned,
            synthetic_data=synth_aligned,
            metadata=metadata.to_dict(),
        )
        new_row_rate = _safe_float(new_row_rate)
    except Exception as exc:
        logger.error("NewRowSynthesis failed: %s", exc)
        return _privacy_error_result(f"NewRowSynthesis error: {exc}")

    privacy_risk = _safe_float(1.0 - new_row_rate)

    # Map risk to a human-readable label and Streamlit CSS class.
    # NOTE: colour logic is INVERTED vs. quality score:
    #   Low risk   → green  (c-good)  — desirable outcome
    #   Medium risk→ amber  (c-mid)
    #   High risk  → red    (c-bad)   — memorisation detected
    label, css_class = _risk_label_and_class(privacy_risk)

    logger.info(
        "Privacy evaluation complete | new_row_rate=%.4f | risk=%.4f | label=%s",
        new_row_rate, privacy_risk, label,
    )

    return {
        "privacy_risk":   privacy_risk,
        "new_row_rate":   new_row_rate,
        "risk_label":     label,
        "risk_css_class": css_class,
        "error":          None,
    }


# ===========================================================================
# Public API — Text summary
# ===========================================================================

def score_summary_text(results: dict) -> str:
    """
    Convert quality evaluation results into a human-readable multi-line string.

    Parameters
    ----------
    results : dict
        The dict returned by evaluate_synthetic_data().

    Returns
    -------
    str
    """
    if results.get("error"):
        return f"⚠️ Evaluation failed: {results['error']}"

    score = results["overall_score"]
    emoji = _score_emoji(score)
    lines = [
        f"{emoji} Overall Quality Score: {score:.1%}",
        "",
        "Property Breakdown:",
    ]
    for prop, val in results.get("property_scores", {}).items():
        lines.append(f"  • {prop}: {val:.1%}")

    return "\n".join(lines)


# ===========================================================================
# Private helpers — shared
# ===========================================================================

def _build_metadata(
    real_df: pd.DataFrame,
    metadata_dict: Optional[dict],
) -> SingleTableMetadata:
    """
    Return an SDV SingleTableMetadata object from a dict or by auto-detection.

    Using the same metadata that the synthesizer was trained on avoids subtle
    column-type mismatches between training and evaluation.
    """
    if metadata_dict is not None:
        return SingleTableMetadata.load_from_dict(metadata_dict)
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(real_df)
    return meta


def _align_columns(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return both DataFrames restricted to their shared columns, in real_df's order.

    SDMetrics requires identical column sets.  This function takes the
    intersection, warns about any dropped columns, and raises ValueError if
    there is no overlap at all.
    """
    real_cols  = list(real_df.columns)
    synth_set  = set(synthetic_df.columns)
    shared     = [c for c in real_cols if c in synth_set]

    if not shared:
        raise ValueError(
            "real_df and synthetic_df share no common columns — cannot evaluate."
        )

    missing = set(real_cols) - synth_set
    if missing:
        logger.warning("Columns in real but not synthetic (excluded): %s", missing)

    return real_df[shared], synthetic_df[shared]


def _safe_float(value) -> float:
    """
    Convert value to float, returning 0.0 for NaN, inf, or non-numeric input.

    SDMetrics occasionally returns NaN or math.inf for columns it cannot
    evaluate (all-null, ID columns, etc.).  This helper ensures those edge
    cases never propagate into Plotly charts or Streamlit metric widgets.
    """
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return 0.0
        return result
    except (TypeError, ValueError):
        return 0.0


# ===========================================================================
# Private helpers — quality
# ===========================================================================

def _extract_property_scores(report: QualityReport) -> dict:
    """
    Extract {"Column Shapes": float, "Column Pair Trends": float} from report.

    Returns an empty dict on failure to keep the UI resilient to SDMetrics
    version differences in the internal DataFrame schema.
    """
    try:
        props_df = report.get_properties()
        return dict(
            zip(props_df["Property"], props_df["Score"].apply(_safe_float))
        )
    except Exception as exc:
        logger.warning("Could not extract property scores: %s", exc)
        return {}


def _extract_column_breakdown(report: QualityReport) -> pd.DataFrame:
    """
    Return a per-column score DataFrame from the "Column Shapes" property.

    Expected columns: ["Column", "Metric", "Quality Score"].
    Sorted ascending by Quality Score so the most problematic columns appear
    first in the Streamlit chart (lowest quality at the top).
    """
    try:
        details_df = report.get_details(property_name="Column Shapes")

        if details_df is None or details_df.empty:
            return pd.DataFrame(columns=["Column", "Metric", "Quality Score"])

        details_df = details_df.rename(columns=str.strip)
        if "Score" in details_df.columns:
            details_df = details_df.rename(columns={"Score": "Quality Score"})
        details_df["Quality Score"] = details_df["Quality Score"].apply(_safe_float)
        details_df = details_df.sort_values("Quality Score", ascending=True)
        details_df = details_df.reset_index(drop=True)
        return details_df

    except Exception as exc:
        logger.warning("Could not extract column breakdown: %s", exc)
        return pd.DataFrame(columns=["Column", "Metric", "Quality Score"])


def _score_emoji(score: float) -> str:
    """Map a quality score to an emoji indicator for the text summary."""
    if score >= 0.90:
        return "🟢"
    elif score >= 0.75:
        return "🟡"
    elif score >= 0.60:
        return "🟠"
    return "🔴"


def _error_result(message: str) -> dict:
    """
    Return a well-shaped failure dict for evaluate_synthetic_data().

    Packaging errors into the same dict shape as a successful result means
    the Streamlit app never needs try/except when consuming evaluation output.
    """
    return {
        "overall_score":    0.0,
        "property_scores":  {},
        "column_breakdown": pd.DataFrame(
            columns=["Column", "Metric", "Quality Score"]
        ),
        "report_object":    None,
        "error":            message,
    }


# ===========================================================================
# Private helpers — privacy
# ===========================================================================

def _risk_label_and_class(privacy_risk: float) -> tuple[str, str]:
    """
    Map a privacy risk score to a human label and a Streamlit CSS class.

    Colour logic is INVERTED vs. quality:
      Low risk   → green  (c-good)  — what we want
      Medium risk→ amber  (c-mid)
      High risk  → red    (c-bad)   — memorisation detected
    """
    if privacy_risk <= 0.20:
        return "Low",    "c-good"
    elif privacy_risk <= 0.50:
        return "Medium", "c-mid"
    return "High", "c-bad"


def _privacy_error_result(message: str) -> dict:
    """Return a well-shaped failure dict for compute_privacy_risk()."""
    return {
        "privacy_risk":   None,
        "new_row_rate":   None,
        "risk_label":     "Unavailable",
        "risk_css_class": "c-mid",
        "error":          message,
    }
