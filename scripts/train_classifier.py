"""
Train the IPCW-weighted logistic regression classifier.

Run from the project root:
    python scripts/train_classifier.py

Outputs (in results/ and the MLflow run):
    - calibration.png
    - MLflow metrics: brier, cindex
    - MLflow params: C (best regularization)
    - MLflow artifact: fitted model
"""

import mlflow
import mlflow.sklearn

from src.config import DROP_COLS, RESULTS_DIR, TAU
from src.data.load_data import load_data
from src.data.preprocess import encode, get_features
from src.models.classifier import tune_classifier
from src.survival.ipcw import compute_ipcw
from src.survival.km import compute_km, step_fn
from src.survival.metrics import brier, cindex
from src.visualization.plots import plot_calibration


def main() -> None:
    # ── 1. Load data ──────────────────────────────────────────────────────────
    train, test = load_data()

    # ── 2. Censoring model (KM on the censoring distribution) ─────────────────
    gt, gs = compute_km(train["event"] == 0, train["obs_time"])
    G = step_fn(gt, gs)

    # ── 3. IPCW weights + binary outcome ──────────────────────────────────────
    train = compute_ipcw(train, G, TAU)
    test  = compute_ipcw(test,  G, TAU)

    # Drop subjects censored before tau (y == NaN)
    train = train.dropna(subset=["y"])
    test  = test.dropna(subset=["y"])

    # ── 4. Feature encoding ───────────────────────────────────────────────────
    features = get_features(train, DROP_COLS)
    X_train  = encode(train, features)
    X_test   = encode(test,  features, ref_cols=X_train.columns)

    # ── 5. Train + evaluate ───────────────────────────────────────────────────
    mlflow.set_experiment("classifier")

    with mlflow.start_run():
        model, best_params = tune_classifier(X_train, train["y"], train["ipcw"])

        mlflow.log_params(best_params)

        risk = model.predict_proba(X_test)[:, 1]

        bs = brier(test["y"].values, risk, test["ipcw"].values)
        ci = cindex(train, test, risk, TAU)

        mlflow.log_metric("brier",  bs)
        mlflow.log_metric("cindex", ci)

        print(f"Brier score : {bs:.4f}")
        print(f"C-index     : {ci:.4f}")

        # ── 6. Save calibration plot + log as artifact ────────────────────────
        cal_path = RESULTS_DIR / "calibration_classifier.png"
        plot_calibration(risk, test["y"].values, test["ipcw"].values, save_path=cal_path)
        mlflow.log_artifact(str(cal_path))

        # ── 7. Log model ──────────────────────────────────────────────────────
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
