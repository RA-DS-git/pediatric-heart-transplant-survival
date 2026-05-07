from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator


def _save_or_show(save_path: Path | None) -> None:
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration(
    risk: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    save_path: Path | None = None,
) -> None:

    bins = np.linspace(0, 1, 6)
    b    = np.digitize(risk, bins)

    obs_rates, pred_rates = [], []
    for i in range(1, len(bins)):
        mask = b == i
        if mask.sum():
            obs_rates.append(np.average(y[mask], weights=weights[mask]))
            pred_rates.append(risk[mask].mean())

    plt.figure()
    plt.plot(pred_rates, obs_rates, "o-", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed event rate")
    plt.title("Calibration Plot")
    plt.legend()
    _save_or_show(save_path)


def plot_km(
    event,
    time,
    title: str = "Kaplan-Meier Curve",
    save_path: Path | None = None,
) -> None:

    t, s = kaplan_meier_estimator(event.astype(bool), time)
    plt.figure()
    plt.step(t, s, where="post")
    plt.xlabel("Time (years)")
    plt.ylabel("Survival probability")
    plt.title(title)
    _save_or_show(save_path)


def plot_risk_distribution(
    risk: np.ndarray,
    save_path: Path | None = None,
) -> None:
    plt.figure()
    plt.hist(risk, bins=30, edgecolor="white")
    plt.xlabel("Predicted risk")
    plt.ylabel("Count")
    plt.title("Risk Score Distribution")
    _save_or_show(save_path)


def plot_survival_curves(
    model,
    X: np.ndarray,
    n: int = 5,
    save_path: Path | None = None,
) -> None:

    funcs = model.predict_survival_function(X[:n])
    plt.figure()
    for fn in funcs:
        plt.step(fn.x, fn.y, where="post")
    plt.xlabel("Time (years)")
    plt.ylabel("Survival probability")
    plt.title("RSF Survival Curves")
    _save_or_show(save_path)
