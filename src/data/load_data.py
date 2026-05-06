import pandas as pd
from src.config import DATA_PATH


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the transplant dataset and split into train / test sets.

    Returns
    -------
    train_df, test_df : DataFrames filtered by the 'split' column.
    """
    df = pd.read_csv(DATA_PATH)
    return df[df["split"] == "train"].copy(), df[df["split"] == "test"].copy()
