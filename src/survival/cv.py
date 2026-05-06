import pandas as pd
from sklearn.model_selection import KFold
from sksurv.util import Surv

from src.models.survival_forest import train_rsf
from src.survival.metrics import cindex
from src.data.preprocess import encode


def cv_rsf(train_df: pd.DataFrame, features: list[str], n_splits: int = 5) -> list[float]:
    """
    K-fold cross-validation for the Random Survival Forest.

    Returns a list of per-fold IPCW C-index scores.

    Parameters
    ----------
    train_df  : Full training DataFrame.
    features  : Feature column names (pre-computed by get_features).
    n_splits  : Number of CV folds (default 5).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X  = encode(train_df, features)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_tr = X.iloc[train_idx]
        X_va = X.iloc[val_idx]

        y_tr = Surv.from_arrays(
            train_df.iloc[train_idx]["event"].astype(bool),
            train_df.iloc[train_idx]["obs_time"],
        )

        model = train_rsf(X_tr, y_tr)
        risk  = model.predict(X_va)

        score = cindex(
            train_df.iloc[train_idx],
            train_df.iloc[val_idx],
            risk,
            tau=1.0,
        )
        scores.append(score)

    return scores
