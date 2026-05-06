# Pediatric Heart Transplant Survival Analysis

Survival models for pediatric heart transplant outcomes using the Wisotzkey et al. (2023) registry dataset.

## Models

| Model | Description |
|---|---|
| IPCW Logistic Regression | Binary classifier at τ=1 year using inverse-probability-of-censoring weights |
| Random Survival Forest | Full survival function estimator evaluated at 1, 3, and 5 years |

## Results

| Metric | Value |
|---|---|
| C-index (classifier) | ~0.70 |
| Brier score (classifier) | ~0.06 |

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running

All commands must be run from the **project root** (`heart_transplant_survival/`).

### Train the IPCW classifier

```bash
python scripts/train_classifier.py
```

Logs metrics, params, and the calibration plot to MLflow. Saves `results/calibration_classifier.png`.

### Train the Random Survival Forest

```bash
python scripts/train_rsf.py
```

Logs Brier + C-index at 1 / 3 / 5 years. Saves `results/calibration_rsf.png` and `results/survival_curves.png`.

### Quick evaluation (no MLflow)

```bash
python scripts/evaluate_classifier.py
```

Prints metrics to stdout and saves `results/calibration_eval.png`.

### View MLflow experiment UI

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### Run tests

```bash
pytest
```

---

## Project Structure

```
heart_transplant_survival/
├── dataset/
│   └── tx_survival.csv        # Source data (Wisotzkey et al. 2023)
├── results/                   # Generated plots (git-ignored)
├── scripts/
│   ├── train_classifier.py    # Train IPCW logistic regression
│   ├── train_rsf.py           # Train Random Survival Forest
│   └── evaluate_classifier.py # Quick evaluation without MLflow
├── src/
│   ├── config.py              # Paths, constants (TAU, DROP_COLS)
│   ├── data/
│   │   ├── load_data.py       # Load CSV → train/test DataFrames
│   │   └── preprocess.py      # Feature selection + one-hot encoding
│   ├── models/
│   │   ├── classifier.py      # IPCW logistic regression + grid search
│   │   └── survival_forest.py # Random Survival Forest wrapper
│   ├── survival/
│   │   ├── km.py              # Kaplan-Meier estimator + step function
│   │   ├── ipcw.py            # IPCW weight computation
│   │   ├── metrics.py         # Brier score + IPCW C-index
│   │   └── cv.py              # K-fold CV for RSF
│   └── visualization/
│       └── plots.py           # Calibration, KM, survival curves, risk dist.
└── tests/
    ├── conftest.py            # Adds project root to sys.path
    ├── test_data.py           # Data loading tests
    ├── test_ipcw.py           # IPCW weight logic tests
    └── test_model.py          # Classifier fit / predict tests
```

---

## Dataset

Wisotzkey et al. (2023). Pediatric heart transplant registry.
