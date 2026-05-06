import numpy as np
import pandas as pd


def compute_ipcw(df: pd.DataFrame, G, tau: float) -> pd.DataFrame:
    """
    Attach IPCW weights and binary outcome 'y' to each row.

    Rules
    -----
    - If a subject is censored before tau  → weight = 0, y = NaN  (dropped later)
    - Otherwise                            → weight = 1 / G(min(t, tau))
                                             y = 1 if event happened at or before tau, else 0

    Parameters
    ----------
    df  : DataFrame with 'obs_time' and 'event' columns.
    G   : Callable, censoring survival function G(t).
    tau : Time horizon.
    """
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
