# Pediatric Heart Transplant Survival Analysis

Survival models for pediatric heart transplant outcomes using the Wisotzkey et al. (2023) registry dataset.

## Models

| Model | Description |
|---|---|
| IPCW Logistic Regression | Binary classifier at τ=1 year using inverse-probability-of-censoring weights |
| Random Survival Forest | Full survival function estimator evaluated at 1, 3, and 5 years |

---

## Results

### IPCW Logistic Regression (1-year horizon)
| Metric | Value |
|---|---|
| Brier Score | 0.0617 |
| C-index | 0.7088 |

### Random Survival Forest (multiple horizons)
| Horizon | Brier Score | C-index |
|---|---|---|
| 1 year | 0.0622 | 0.7162 |
| 3 years | 0.0963 | 0.6600 |
| 5 years | 0.1302 | 0.6137 |

### Interpretation
- **Both models perform nearly identically at 1 year:** RSF edges the classifier by only 0.007 on C-index
- **RSF degrades over time** and is expected; predicting 5 years out is genuinely harder with limited data
- **The classifier is simpler and just as good** for 1 year predictions, easier to interpret and deploy
- **The RSF is more valuable** if we need survival probabilities at multiple time points, not just 1 year

### Metric Guide
- **Brier Score** measures calibration (accuracy of predicted probabilities). Lower is better. 0 = perfect, 0.25 = useless.
- **C-index** measures discrimination (can the model correctly rank who dies sooner). Higher is better. 0.5 = random, 1.0 = perfect. 0.70+ is considered strong for clinical survival data.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 2. Install the project and all dependencies
pip install -e ".[dev]"
```

---

## Running

All commands must be run from the **project root**.

### 1. Run tests first
```bash
pytest
```

### 2. Train the IPCW classifier
```bash
python scripts/train_classifier.py
```
Logs metrics, params, and calibration plot to MLflow. Saves `results/calibration_classifier.png`.

### 3. Train the Random Survival Forest
```bash
python scripts/train_rsf.py
```
Logs Brier + C-index at 1 / 3 / 5 years. Saves `results/calibration_rsf.png` and `results/survival_curves.png`.

### 4. View results in MLflow
```bash
mlflow ui
```
Open **http://127.0.0.1:5000** in your browser to compare both experiments side by side.

### Quick evaluation (no MLflow)
```bash
python scripts/evaluate_classifier.py
```
Prints metrics to stdout and saves `results/calibration_eval.png`.

---

## Project Structure

```
pediatric-heart-transplant-survival/
├── dataset/
│   └── tx_survival.csv        # Source data (Wisotzkey et al. 2023)
├── results/                   # Generated plots
├── scripts/
│   ├── train_classifier.py    # Train IPCW logistic regression + log to MLflow
│   ├── train_rsf.py           # Train Random Survival Forest + log to MLflow
│   └── evaluate_classifier.py # Quick evaluation without MLflow
├── src/
│   ├── config.py              # Paths, constants
│   ├── data/
│   │   ├── load_data.py       # Load CSV
│   │   └── preprocess.py      # Feature selection + one-hot encoding
│   ├── models/
│   │   ├── classifier.py      # IPCW logistic regression + grid search
│   │   └── survival_forest.py # Random Survival Forest wrapper
│   ├── survival/
│   │   ├── km.py              # Kaplan-Meier estimator + step function G(t)
│   │   ├── ipcw.py            # IPCW weight computation (vectorized)
│   │   ├── metrics.py         # Brier score + IPCW C-index
│   │   └── cv.py              # K-fold cross-validation for RSF
│   └── visualization/
│       └── plots.py           # Calibration, KM, survival curves, risk distribution
├── tests/
│   ├── conftest.py            
│   ├── test_data.py           
│   ├── test_ipcw.py           
│   └── test_model.py          
├── pyproject.toml             # Package definition + dependencies
├── requirements.txt           # Pinned dependency ranges
└── .gitignore                 
```

---

## Dataset

Wisotzkey et al. (2023). Pediatric heart transplant registry (~5,000 patients).
