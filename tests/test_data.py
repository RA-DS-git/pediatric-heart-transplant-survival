from src.data.load_data import load_data


def test_load_returns_nonempty_splits():
    train, test = load_data()
    assert len(train) > 0, "Training set is empty"
    assert len(test) > 0, "Test set is empty"


def test_required_columns_present():
    train, _ = load_data()
    for col in ("obs_time", "event", "split"):
        assert col in train.columns, f"Missing column: {col}"
