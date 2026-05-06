import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator


def compute_km(event, time):
    """
    Fit a Kaplan-Meier estimator.

    Returns
    -------
    times, survival_probs : arrays returned by sksurv's estimator.
    """
    return kaplan_meier_estimator(event.astype(bool), time)


def step_fn(times: np.ndarray, surv: np.ndarray):
    """
    Build a step-function callable from KM outputs.

    Returns a function G(t) → survival probability at time t.
    Values before the first observed time return 1.0.
    """
    def G(t: float) -> float:
        idx = np.searchsorted(times, t, side="right") - 1
        return float(surv[idx]) if idx >= 0 else 1.0

    return G
