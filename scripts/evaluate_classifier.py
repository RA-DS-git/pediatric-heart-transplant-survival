"""
Evaluate the IPCW logistic regression classifier (without MLflow).

Useful for a quick sanity check or exploratory runs.

Run from the project root:
    python scripts/evaluate_classifier.py

Outputs (in results/):
    - calibration_eval.png
"""

from src.config import DROP_COLS, RESULTS_DIR, TAU
from src.data.load_data import load_data
from src.data.preprocess import encode, get_features
from src.models.classifier import tune_classifier
from src.survival.ipcw import compute_ipcw
from src.survival.km import compute_km, step_fn
from src.survival.metrics import brier, cindex
from src.visualization.plots import plot_calibration


def main() -> None:
    # ── 1. Load + IPCW weights ────────────────────────────────────────────────
    train, test = load_data()

    gt, gs = compute_km(train["event"] == 0, train["obs_time"])
    G = step_fn(gt, gs)

    train = compute_ipcw(train, G, TAU)
    test  = compute_ipcw(test,  G, TAU)

    train = train.dropna(subset=["y"])
    test  = test.dropna(subset=["y"])

    # ── 2. Feature encoding ───────────────────────────────────────────────────
    features = get_features(train, DROP_COLS)
    X_train  = encode(train, features)
    X_test   = encode(test,  features, ref_cols=X_train.columns)

    # ── 3. Train + predict ────────────────────────────────────────────────────
    model, best_params = tune_classifier(X_train, train["y"], train["ipcw"])
    print(f"Best params : {best_params}")

    risk = model.predict_proba(X_test)[:, 1]

    # ── 4. Metrics ────────────────────────────────────────────────────────────
    bs = brier(test["y"].values, risk, test["ipcw"].values)
    ci = cindex(train, test, risk, TAU)

    print(f"Brier score : {bs:.4f}")
    print(f"C-index     : {ci:.4f}")

    # ── 5. Calibration plot ───────────────────────────────────────────────────
    cal_path = RESULTS_DIR / "calibration_eval.png"
    plot_calibration(risk, test["y"].values, test["ipcw"].values, save_path=cal_path)
    print(f"Calibration plot saved → {cal_path}")


if __name__ == "__main__":
    main()
