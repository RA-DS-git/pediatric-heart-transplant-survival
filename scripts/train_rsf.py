"""
Train the Random Survival Forest model.

Run from the project root:
    python scripts/train_rsf.py

Outputs (in results/ and the MLflow run):
    - calibration_rsf.png
    - survival_curves.png
    - MLflow metrics: brier_{1,3,5}, cindex_{1,3,5}
    - MLflow params: n_estimators, min_samples_leaf
    - MLflow artifact: fitted model
"""

import numpy as np
import mlflow
import mlflow.sklearn
from sksurv.metrics import brier_score, concordance_index_ipcw
from sksurv.util import Surv

from src.config import DROP_COLS, RESULTS_DIR
from src.data.load_data import load_data
from src.data.preprocess import encode, get_features
from src.models.survival_forest import train_rsf
from src.visualization.plots import plot_calibration, plot_survival_curves

# Evaluation horizons (years)
EVAL_TIMES = [1, 3, 5]

# RSF hyperparameters
N_ESTIMATORS     = 300
MIN_SAMPLES_LEAF = 30


def evaluate_rsf(model, X_test, train_df, test_df) -> dict[int, dict[str, float]]:

    y_train = Surv.from_arrays(train_df["event"].astype(bool), train_df["obs_time"])
    y_test  = Surv.from_arrays(test_df["event"].astype(bool),  test_df["obs_time"])

    surv_funcs = model.predict_survival_function(X_test)
    results = {}

    for t in EVAL_TIMES:
        surv_t = np.array([fn(t) for fn in surv_funcs])
        risk_t = 1 - surv_t

        bs   = brier_score(y_train, y_test, surv_t, np.array([t]))[1][0]
        cidx = concordance_index_ipcw(y_train, y_test, risk_t, tau=t)[0]

        results[t] = {"brier": float(bs), "cindex": float(cidx)}

    return results


def main() -> None:
    # 1. Load + encode
    train, test = load_data()
    features    = get_features(train, DROP_COLS)
    X_train     = encode(train, features)
    X_test      = encode(test,  features, ref_cols=X_train.columns)

    y_train = Surv.from_arrays(train["event"].astype(bool), train["obs_time"])

    # 2. Train + evaluate
    mlflow.set_experiment("random_survival_forest")

    with mlflow.start_run():
        mlflow.log_param("n_estimators",     N_ESTIMATORS)
        mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)

        model   = train_rsf(X_train, y_train, N_ESTIMATORS, MIN_SAMPLES_LEAF)
        results = evaluate_rsf(model, X_test, train, test)

        # Log metrics
        for t, m in results.items():
            mlflow.log_metric(f"brier_{t}",  m["brier"])
            mlflow.log_metric(f"cindex_{t}", m["cindex"])

        # 3. Save plots + log as artifacts
        risk_at_1 = 1 - np.array([fn(1) for fn in model.predict_survival_function(X_test)])

        cal_path = RESULTS_DIR / "calibration_rsf.png"
        plot_calibration(
            risk_at_1,
            test["event"].values,
            np.ones(len(test)),
            save_path=cal_path,
        )
        mlflow.log_artifact(str(cal_path))

        sc_path = RESULTS_DIR / "survival_curves.png"
        plot_survival_curves(model, X_test, n=5, save_path=sc_path)
        mlflow.log_artifact(str(sc_path))

        # 4. Log model
        mlflow.sklearn.log_model(model, "rsf_model")

        # 5. Print summary
        print("\nRSF Results:")
        for t, m in results.items():
            print(f"  {t}y  Brier: {m['brier']:.4f}   C-index: {m['cindex']:.4f}")


if __name__ == "__main__":
    main()
