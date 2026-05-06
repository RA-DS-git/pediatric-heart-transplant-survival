import pandas as pd


def get_features(df: pd.DataFrame, drop_cols: list[str]) -> list[str]:
    """Return column names that are not in drop_cols."""
    return [c for c in df.columns if c not in drop_cols]


def encode(
    df: pd.DataFrame,
    features: list[str],
    ref_cols: pd.Index | None = None,
) -> pd.DataFrame:
    """
    One-hot encode categorical features (drop_first=True).

    Parameters
    ----------
    df       : Input DataFrame.
    features : Feature columns to encode.
    ref_cols : If provided, align output columns to this reference
               (used to match test set columns to training set columns).
    """
    X = pd.get_dummies(df[features], drop_first=True)
    if ref_cols is not None:
        X = X.reindex(columns=ref_cols, fill_value=0)
    return X
