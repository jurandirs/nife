import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

matplotlib.use("Agg")


class ThresholdModel:
    """``high`` when x1 + x2 > 10, ``low`` otherwise."""

    classes_ = np.array(["high", "low"])

    def predict(self, X):
        return np.where(X["x1"] + X["x2"] > 10, "high", "low")


class SumModel:
    """Predicts x1 + x2."""

    def predict(self, X):
        return (X["x1"] + X["x2"]).to_numpy(float)


@pytest.fixture
def toy():
    """A dataset small enough to compute every metric by hand.

    Ranges: x1 and x2 are 10 wide; with the categorical `color`, the Gower distance is
    (|dx1| / 10 + |dx2| / 10 + [colors differ]) / 3.
    """
    data = pd.DataFrame(
        {
            "x1": [0, 10, 0, 10, 5, 8],
            "x2": [0, 0, 10, 10, 5, 6],
            "color": ["red", "red", "blue", "blue", "red", "blue"],
        }
    )
    instances = pd.DataFrame({"x1": [2, 9, 1], "x2": [2, 9, 1], "color": ["red", "blue", "red"]})
    counterfactuals = pd.DataFrame(
        {
            "instance": [0, 0, 0, 1],
            "x1": [8, 2, 3, 5],
            "x2": [6, 9, 3, 5],
            "color": ["blue", "red", "red", "red"],
        }
    )
    return data, instances, counterfactuals


@pytest.fixture(scope="session")
def iris():
    dataset = load_iris(as_frame=True)
    X = dataset.data
    y = dataset.target.map(dict(enumerate(dataset.target_names)))
    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    return X, model


@pytest.fixture(scope="session")
def diabetes():
    dataset = load_diabetes(as_frame=True)
    X = dataset.data
    model = RandomForestRegressor(n_estimators=50, random_state=0).fit(X, dataset.target)
    return X, model
