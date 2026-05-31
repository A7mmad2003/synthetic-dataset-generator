"""
synthesizer.py — SDV Model Training & Synthetic Data Generation
===============================================================
This module is the core generation engine of the Synthetic Dataset Generator.
It wraps the SDV (Synthetic Data Vault) library and exposes a single, unified
function interface (`generate_synthetic_data`) that supports all three supported
synthesizer models:

  - CTGANSynthesizer   : GAN-based model, best for complex mixed-type tabular data
  - TVAESynthesizer    : VAE-based model, faster than CTGAN, good general-purpose choice
  - GaussianCopulaSynthesizer : Statistical copula model, fastest, works well on
                                 data with clear statistical relationships

Usage (from app.py or a notebook):
    from core.synthesizer import generate_synthetic_data

    synthetic_df, model_info = generate_synthetic_data(
        real_df=my_dataframe,
        model_name="CTGAN",
        num_rows=1000,
        epochs=300,
    )
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Literal, Optional, Tuple

# SDV imports — all three synthesizer classes live under sdv.single_table
from sdv.single_table import (
    CTGANSynthesizer,
    TVAESynthesizer,
    GaussianCopulaSynthesizer,
)

# SDV metadata auto-detection: SingleTableMetadata infers column types
# (numerical, categorical, datetime, etc.) directly from a DataFrame
from sdv.metadata import SingleTableMetadata

# Configure module-level logger so the host app can capture log output
logger = logging.getLogger(__name__)

# Type alias for the three supported model names — enforces a clear API contract
ModelName = Literal["CTGAN", "TVAE", "GaussianCopula"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    real_df: pd.DataFrame,
    model_name: ModelName = "CTGAN",
    num_rows: Optional[int] = None,
    epochs: int = 300,
    batch_size: int = 500,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """
    Train an SDV synthesizer on `real_df` and return a synthetic DataFrame.

    This is the single entry-point for all generation work in the project.
    It handles metadata detection, model instantiation, training, and sampling
    in one call so that the Streamlit UI doesn't need to know anything about
    SDV internals.

    Parameters
    ----------
    real_df : pd.DataFrame
        The original (real) dataset uploaded by the user.  Must have at least
        one row and one column.  Any pre-processing (dropping nulls, casting
        types) should happen BEFORE calling this function.

    model_name : {"CTGAN", "TVAE", "GaussianCopula"}
        Which SDV synthesizer to use.  Each has different speed/quality
        trade-offs — see module docstring above.

    num_rows : int, optional
        Number of synthetic rows to generate.  Defaults to len(real_df) so
        the output matches the original dataset's size.

    epochs : int
        Number of training epochs.  Only relevant for CTGAN and TVAE (deep
        learning models).  GaussianCopula ignores this parameter entirely
        because it's a statistical fit rather than a gradient-based model.
        More epochs → better fidelity but slower runtime.

    batch_size : int
        Mini-batch size for CTGAN/TVAE training.  Ignored for GaussianCopula.
        Larger batches can speed up training on wide datasets.

    verbose : bool
        If True, SDV will print per-epoch loss information.  Useful for
        debugging but noisy inside a Streamlit app, so defaults to False.

    Returns
    -------
    synthetic_df : pd.DataFrame
        The generated synthetic dataset with the same columns as `real_df`.

    model_info : dict
        Metadata about the training run, including:
          - "model_name"   : str  — which model was used
          - "num_rows"     : int  — how many rows were generated
          - "train_time_s" : float — wall-clock seconds taken to fit the model
          - "sample_time_s": float — wall-clock seconds taken to sample
          - "real_shape"   : tuple — shape of the input DataFrame
          - "synth_shape"  : tuple — shape of the output DataFrame
          - "metadata"     : dict  — SDV column type metadata (for debugging)

    Raises
    ------
    ValueError
        If `model_name` is not one of the three supported values, or if
        `real_df` is empty.
    RuntimeError
        Wraps any SDV-level exception with a friendlier message.
    """

    # -----------------------------------------------------------------------
    # 1. Input validation
    # -----------------------------------------------------------------------
    if real_df is None or real_df.empty:
        raise ValueError("real_df must be a non-empty DataFrame.")

    supported_models = {"CTGAN", "TVAE", "GaussianCopula"}
    if model_name not in supported_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Choose from: {sorted(supported_models)}"
        )

    # Default to matching the real dataset's row count
    if num_rows is None:
        num_rows = len(real_df)

    logger.info(
        "Starting synthesis | model=%s | rows=%d | epochs=%d",
        model_name, num_rows, epochs,
    )

    # -----------------------------------------------------------------------
    # 2. Auto-detect metadata
    # -----------------------------------------------------------------------
    # SingleTableMetadata.detect_from_dataframe() inspects column dtypes and
    # sample values to assign SDV semantic types (numerical, categorical,
    # datetime, boolean, id).  This removes the need for manual schema
    # configuration and is robust enough for demo/academic projects.
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_df)

    # Serialise metadata to a plain dict so we can include it in model_info
    # without dragging the heavy SDV object into the return value.
    metadata_dict = metadata.to_dict()

    logger.debug("Detected metadata: %s", metadata_dict)

    # -----------------------------------------------------------------------
    # 3. Instantiate the requested synthesizer
    # -----------------------------------------------------------------------
    synthesizer = _build_synthesizer(
        model_name=model_name,
        metadata=metadata,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    # -----------------------------------------------------------------------
    # 4. Fit (train) the model on the real data
    # -----------------------------------------------------------------------
    train_start = time.perf_counter()
    try:
        synthesizer.fit(real_df)
    except Exception as exc:
        raise RuntimeError(
            f"SDV model fitting failed for '{model_name}': {exc}"
        ) from exc
    train_time = time.perf_counter() - train_start

    logger.info("Model trained in %.2f seconds.", train_time)

    # -----------------------------------------------------------------------
    # 5. Sample synthetic rows
    # -----------------------------------------------------------------------
    sample_start = time.perf_counter()
    try:
        synthetic_df = synthesizer.sample(num_rows=num_rows)
    except Exception as exc:
        raise RuntimeError(
            f"SDV sampling failed for '{model_name}': {exc}"
        ) from exc
    sample_time = time.perf_counter() - sample_start

    logger.info(
        "Sampling complete in %.2f seconds. Output shape: %s",
        sample_time, synthetic_df.shape,
    )

    # -----------------------------------------------------------------------
    # 6. Bundle run metadata for the UI and evaluator
    # -----------------------------------------------------------------------
    model_info = {
        "model_name": model_name,
        "num_rows": num_rows,
        "train_time_s": round(train_time, 3),
        "sample_time_s": round(sample_time, 3),
        "real_shape": real_df.shape,
        "synth_shape": synthetic_df.shape,
        "metadata": metadata_dict,
    }

    return synthetic_df, model_info


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_synthesizer(
    model_name: ModelName,
    metadata: SingleTableMetadata,
    epochs: int,
    batch_size: int,
    verbose: bool,
):
    """
    Instantiate and return the correct SDV synthesizer class.

    Kept private (_build_synthesizer) because callers should always go through
    generate_synthetic_data, which handles validation and logging.

    Why separate this function?
    Constructing synthesizers with the right kwargs varies per model:
    - CTGAN and TVAE accept epochs, batch_size, and verbose because they are
      deep learning models trained with gradient descent.
    - GaussianCopula is a pure statistical model; epochs and batch_size are
      not applicable — passing them would raise a TypeError.
    Isolating this logic avoids cluttering the main function with if/else chains.

    Parameters
    ----------
    model_name : ModelName
        One of "CTGAN", "TVAE", "GaussianCopula".
    metadata : SingleTableMetadata
        The SDV metadata object describing the table schema.
    epochs : int
        Training epochs (deep learning models only).
    batch_size : int
        Mini-batch size (deep learning models only).
    verbose : bool
        Whether SDV should print training progress.

    Returns
    -------
    An unfitted SDV synthesizer instance ready for .fit().
    """

    if model_name == "CTGAN":
        # CTGANSynthesizer uses a conditional GAN architecture.
        # It handles mixed types (numeric + categorical) natively by
        # conditioning the generator on discrete column values.
        return CTGANSynthesizer(
            metadata=metadata,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    elif model_name == "TVAE":
        # TVAESynthesizer uses a Variational Autoencoder.
        # Generally converges faster than CTGAN and uses less memory,
        # but can underfit on datasets with highly skewed distributions.
        return TVAESynthesizer(
            metadata=metadata,
            epochs=epochs,
            batch_size=batch_size,
        )

    elif model_name == "GaussianCopula":
        # GaussianCopulaSynthesizer fits a multivariate Gaussian copula.
        # No epochs / batch_size — this is a closed-form statistical fit.
        # Very fast and often good enough for datasets with near-normal columns.
        return GaussianCopulaSynthesizer(metadata=metadata)

    else:
        # This branch should never be reached because generate_synthetic_data
        # validates model_name before calling this function. Kept as a guard.
        raise ValueError(f"Unknown model: {model_name}")


def get_available_models() -> list[str]:
    """
    Return the list of model names supported by this module.

    Used by app.py to populate the model-selection dropdown without
    hard-coding the list in two places.

    Returns
    -------
    list[str]
        Sorted list of model name strings.
    """
    return ["CTGAN", "GaussianCopula", "TVAE"]


def preview_metadata(real_df: pd.DataFrame) -> dict:
    """
    Run SDV's metadata detection on `real_df` and return the result as a dict.

    This is a lightweight helper used by the Streamlit app to show the user
    which column types SDV has inferred BEFORE they kick off the (potentially
    slow) training step.  Catching type-detection surprises early saves time.

    Parameters
    ----------
    real_df : pd.DataFrame
        The dataset to inspect.

    Returns
    -------
    dict
        SDV metadata dict with keys like:
        {
            "columns": {
                "age": {"sdtype": "numerical"},
                "name": {"sdtype": "categorical"},
                ...
            }
        }
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_df)
    return metadata.to_dict()
