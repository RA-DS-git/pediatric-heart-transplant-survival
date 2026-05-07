import numpy as np
import pandas as pd

from src.survival.ipcw import compute_ipcw


def G(t: float) -> float:

    return 0.9


def test_early_censored_gets_zero_weight():

    df  = pd.DataFrame({"obs_time": [0.5, 2.0], "event": [0, 1]})
    out = compute_ipcw(df, G, tau=1.0)

    assert out["ipcw"].iloc[0] == 0.0, "Early-censored subject should have weight 0"
    assert np.isnan(out["y"].iloc[0]),  "Early-censored subject should have y=NaN"


def test_event_before_tau_gets_y_one():

    df  = pd.DataFrame({"obs_time": [0.5], "event": [1]})
    out = compute_ipcw(df, G, tau=1.0)

    assert out["y"].iloc[0] == 1.0, "Event before tau should give y=1"


def test_survivor_beyond_tau_gets_y_zero():

    df  = pd.DataFrame({"obs_time": [2.0], "event": [0]})
    out = compute_ipcw(df, G, tau=1.0)

    assert out["y"].iloc[0] == 0.0,   "Survivor past tau should give y=0"
    assert out["ipcw"].iloc[0] > 0.0, "Survivor past tau should have positive weight"
