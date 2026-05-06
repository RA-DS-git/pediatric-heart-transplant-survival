import numpy as np
from sksurv.ensemble import RandomSurvivalForest


def train_rsf(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 300,
    min_samples_leaf: int = 30,
    random_state: int = 42,
) -> RandomSurvivalForest:
    """
    Fit a Random Survival Forest.

    Parameters
    ----------
    X                : Feature matrix.
    y                : Structured survival array (sksurv format).
    n_estimators     : Number of trees (default 300).
    min_samples_leaf : Minimum leaf size (default 30).
    random_state     : Random seed.

    Returns
    -------
    Fitted RandomSurvivalForest model.
    """
    model = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model
