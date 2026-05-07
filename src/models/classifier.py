import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def _safe_cv(y: np.ndarray, max_splits: int = 3) -> StratifiedKFold:

    _, counts = np.unique(y, return_counts=True)
    n_splits  = max(2, min(max_splits, int(counts.min())))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def tune_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[LogisticRegression, dict]:

    cv   = _safe_cv(y, max_splits=3)
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        {"C": [0.01, 0.1, 1, 10]},
        scoring="roc_auc",
        cv=cv,
    )
    grid.fit(X, y, sample_weight=weights)
    return grid.best_estimator_, grid.best_params_
