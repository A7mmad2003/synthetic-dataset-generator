"""
core/__init__.py — Public surface of the `core` package
=========================================================
The `core` package contains the two main logic modules for the
Synthetic Dataset Generator project:

  - synthesizer : SDV model training and synthetic data generation
  - evaluator   : SDMetrics quality + privacy evaluation

Re-exporting the primary public functions here lets app.py use a clean,
flat import style:

    from core.synthesizer import generate_synthetic_data, get_available_models
    from core.evaluator   import evaluate_synthetic_data, compute_privacy_risk

All six symbols are also importable directly from the package root:

    from core import generate_synthetic_data, compute_privacy_risk
"""

from core.synthesizer import (
    generate_synthetic_data,
    get_available_models,
    preview_metadata,
)
from core.evaluator import (
    evaluate_synthetic_data,
    compute_privacy_risk,
    score_summary_text,
)

__all__ = [
    # synthesizer
    "generate_synthetic_data",
    "get_available_models",
    "preview_metadata",
    # evaluator
    "evaluate_synthetic_data",
    "compute_privacy_risk",
    "score_summary_text",
]
