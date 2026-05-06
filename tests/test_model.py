import numpy as np
import pandas as pd

from src.models.classifier import tune_classifier


def test_tune_classifier_returns_fitted_model():
    X = pd.DataFrame({"a": [0, 1, 0, 1]})
    y = np.array([0, 1, 0, 1])
    w = np.ones(4)

    model, params = tune_classifier(X, y, w)

    assert model is not None, "tune_classifier should return a model"
    assert "C" in params,     "best_params should contain 'C'"


def test_classifier_predict_proba_shape():
    X = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1]})
    y = np.array([0, 1, 0, 1, 0, 1])
    w = np.ones(6)

    model, _ = tune_classifier(X, y, w)
    proba    = model.predict_proba(X)

    assert proba.shape == (6, 2), "predict_proba should return (n_samples, 2)"
