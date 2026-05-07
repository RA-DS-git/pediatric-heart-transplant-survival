import numpy as np
import pandas as pd


def compute_ipcw(df: pd.DataFrame, G, tau: float) -> pd.DataFrame:

    df = df.copy()

    t = df["obs_time"].values
    d = df["event"].values

    # Censored before tau → excluded (weight 0, y NaN)
    early_censored = (t < tau) & (d == 0)

    weights = np.where(early_censored, 0.0, 1.0 / np.vectorize(G)(np.minimum(t, tau)))
    y       = np.where(early_censored, np.nan, ((t <= tau) & (d == 1)).astype(float))

    df["ipcw"] = weights
    df["y"]    = y
    return df
