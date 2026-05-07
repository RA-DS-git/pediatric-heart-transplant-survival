import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv


def brier(y: np.ndarray, risk: np.ndarray, weights: np.ndarray) -> float:

    return float(np.average((y - risk) ** 2, weights=weights))


def cindex(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    risk: np.ndarray,
    tau: float,
) -> float:

    y_train = Surv.from_arrays(train_df["event"].astype(bool), train_df["obs_time"])
    y_test  = Surv.from_arrays(test_df["event"].astype(bool),  test_df["obs_time"])
    return float(concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0])
