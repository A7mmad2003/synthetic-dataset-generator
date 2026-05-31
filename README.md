# 🧬 Synthetic Dataset Generator

> Train a generative model on any CSV file and download a privacy-safe synthetic twin that statistically mirrors your real data.

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![SDV](https://img.shields.io/badge/SDV-1.11-6B46C1)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What is this?

The **Synthetic Dataset Generator** is a browser-based tool (built with Streamlit) that lets you upload any tabular CSV dataset, choose a generative model, and produce a new dataset that preserves the statistical structure of your original data — without containing any real records. The app walks you through every step: it auto-detects column types, lets you correct any mistakes, trains the model with one click, and gives you a downloadable synthetic CSV alongside quality and privacy scores.

Synthetic data matters for three interconnected reasons. **Privacy**: real datasets often contain sensitive personal information — medical records, financial transactions, HR files — that legally and ethically cannot be shared. A synthetic twin carries the same statistical fingerprint as the original but contains no real individuals, making it safe to share with collaborators, publish alongside research, or use in public demos. **Data scarcity**: many machine learning problems suffer from too little labelled data; a generative model trained on even a small real dataset can produce an arbitrarily large augmented set to improve downstream model generalisation. **ML training augmentation**: class imbalance, rare events, and edge cases are systematically under-represented in collected data; synthetic generation can oversample these cases to build more robust classifiers and detectors.

---

## Generative Models Used

### CTGAN — Conditional Tabular GAN
CTGAN is a Generative Adversarial Network specifically designed for mixed-type tabular data (columns that are a mix of continuous numbers and discrete categories). It conditions the generator on sampled discrete values during training, which forces the model to reproduce rare categories rather than ignoring them. CTGAN is the most powerful model in the app and produces the highest-fidelity synthetic data, but it is also the slowest — expect several minutes on a CPU for large datasets. It is implemented via `CTGANSynthesizer` from the [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV) library.

### TVAE — Tabular Variational Autoencoder
TVAE uses a Variational Autoencoder architecture rather than a GAN's adversarial training loop, which means it typically converges faster and uses less memory than CTGAN. The encoder compresses real rows into a learned latent space; the decoder learns to reconstruct (and then sample novel) rows from that space. TVAE is a strong all-round choice for most datasets and is a good first option when you need results quickly. Like CTGAN, it is provided by the SDV library as `TVAESynthesizer` and accepts the same `epochs` and `batch_size` hyperparameters.

### GaussianCopula — Statistical Copula Model
GaussianCopula is not a deep learning model — it fits a multivariate Gaussian copula to capture the correlation structure between columns, then models each column's marginal distribution separately. Training is a closed-form statistical computation that finishes in seconds regardless of dataset size, with no `epochs` or `batch_size` to configure. It works best when columns have roughly Gaussian marginals and linear correlations; it is less expressive than CTGAN/TVAE on complex distributions, but it is the safest choice for very small datasets or quick exploratory runs. Implemented as `GaussianCopulaSynthesizer` in SDV.

---

## Demo


https://github.com/user-attachments/assets/81ac25b1-4943-4c3d-af8b-20db2527ff89


---

## Project Structure

```
synthetic-dataset-generator/
│
├── app.py                  # Streamlit UI — all five sections wired together
│
├── core/                   # Business logic — no Streamlit dependency
│   ├── __init__.py         # Package surface: re-exports all public functions
│   ├── synthesizer.py      # SDV model training & synthetic data generation
│   └── evaluator.py        # SDMetrics quality score + privacy risk evaluation
│
├── requirements.txt        # Pinned Python dependencies
├── .gitignore              # Ignores venvs, data files, __pycache__, secrets
└── README.md               # This file
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit application: upload, sidebar config, results display, distribution charts |
| `core/synthesizer.py` | Wraps SDV — detects metadata, trains CTGAN / TVAE / GaussianCopula, samples synthetic rows, returns timing metadata |
| `core/evaluator.py` | Wraps SDMetrics — runs `QualityReport` for fidelity scores and `NewRowSynthesis` for privacy risk |
| `core/__init__.py` | Re-exports `generate_synthetic_data`, `evaluate_synthetic_data`, `compute_privacy_risk`, `get_available_models`, `preview_metadata`, `score_summary_text` |
| `requirements.txt` | Pinned versions of streamlit, sdv, sdmetrics, torch, pandas, numpy, plotly and all transitive deps |

---

## Setup & Run

### Prerequisites

- Python **3.10** or **3.11** (SDV is not yet compatible with 3.12)
- `pip` ≥ 23.0 recommended
- ~3 GB free disk space (PyTorch + SDV are large)

### Install

```bash
# 1. Clone the repository
git clone https://github.com/A7mmad2003/synthetic-dataset-generator.git
cd synthetic-dataset-generator

# 2. Create and activate a virtual environment (strongly recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows PowerShell

# 3. Install all dependencies
pip install -r requirements.txt
```

> **Note — PyTorch CPU vs GPU**
> The `requirements.txt` pins the standard CPU build of PyTorch (`torch==2.3.0`).
> For GPU-accelerated CTGAN/TVAE training, replace it with the CUDA wheel:
> ```bash
> pip install torch==2.3.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
> ```

### Run

```bash
streamlit run app.py
```

The app opens automatically in your default browser at `http://localhost:8501`.

### Quick demo (no CSV needed)

Click any of the three **sample dataset** buttons inside the app to instantly pre-load a realistic dataset and try the full generation pipeline without uploading your own file.

---

## How It Works

```
1. Upload      →  Drop a CSV (up to 100,000 rows) or click a sample preset
                   App displays row/column counts and missing-value stats

2. Detect       →  SDV's SingleTableMetadata auto-detects column semantic types
   Schema           (numerical, categorical, datetime, boolean, id)
                   You can correct any mistakes in the column-type editor

3. Configure    →  Choose a model (CTGAN / TVAE / GaussianCopula) in the sidebar
                   Set epochs, batch size, and the number of rows to generate

4. Train        →  The selected SDV synthesizer is fitted on your real data
   Model            CTGAN/TVAE: gradient-based training for the chosen epochs
                   GaussianCopula: closed-form statistical fit (seconds)

5. Generate     →  The fitted model samples the requested number of synthetic rows
                   Output has the same column names and types as the input

6. Evaluate     →  SDMetrics QualityReport scores Column Shapes and Column Pair
                   Trends (0–1 scale); NewRowSynthesis measures privacy risk

7. Download     →  Click the download button to save synthetic_<filename>.csv
                   Side-by-side distribution charts let you visually verify fidelity
```

---

## Evaluation Metrics

### Quality Score (0 – 1, higher is better)

The Quality Score is computed by SDMetrics' `QualityReport` and is the unweighted average of two property groups:

- **Column Shapes** measures how well each column's marginal distribution (the histogram shape for numerical columns, the proportional frequency table for categorical columns) is reproduced in the synthetic data. A score of 1.0 means the distributions are indistinguishable.
- **Column Pair Trends** measures how well pairwise statistical relationships between columns are preserved — correlations for numerical pairs, contingency-table associations for categorical pairs, and cross-type mutual information elsewhere.

A combined Quality Score above **0.80** is generally considered excellent synthetic data for most research and ML use cases. Scores below 0.50 suggest the model struggled — try increasing epochs, switching models, or correcting column types.

### Privacy Risk Score (0 – 1, lower is better)

The Privacy Risk Score is derived from SDMetrics' `NewRowSynthesis` metric, which works by comparing every synthetic row to its nearest neighbour in the real dataset. A synthetic row is flagged as a "near-copy" if its distance to the closest real row is smaller than a threshold derived from real–real distances (i.e. typical natural variation in the real data).

- **New Row Rate** = fraction of synthetic rows that are genuinely novel (not near-copies)
- **Privacy Risk** = 1 − New Row Rate

A Privacy Risk score near **0.0** means virtually all synthetic rows are novel — the model has generalised rather than memorised. A score near **1.0** is a warning sign that the model has overfit and is reproducing real records almost verbatim, which defeats the privacy purpose of synthetic data.

| Privacy Risk | Interpretation |
|---|---|
| 0.00 – 0.20 | 🟢 Low — rows are genuinely novel, safe to share |
| 0.21 – 0.50 | 🟡 Medium — some near-copies present, review before sharing |
| 0.51 – 1.00 | 🔴 High — memorisation detected, do not share without further review |

---

## Authors

| Name | Class | GitHub |
|---|---|---|
| Ahmed Abdelkafi | 2DAD | [@A7mmad2003](https://github.com/A7mmad2003) |

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 [Ahmed Abdelkafi]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
