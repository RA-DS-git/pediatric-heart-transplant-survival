import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator


def compute_km(event, time):
    
    return kaplan_meier_estimator(event.astype(bool), time)


def step_fn(times: np.ndarray, surv: np.ndarray):

    def G(t: float) -> float:
        idx = np.searchsorted(times, t, side="right") - 1
        return float(surv[idx]) if idx >= 0 else 1.0

    return G
