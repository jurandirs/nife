"""The base class with a minimal method: every example of the data is a candidate.

Whatever the method returns, the base class must keep only the candidates that the
model confirms and that keep the immutable features, rank them and select them.
"""

import numpy as np
import pandas as pd
import pytest

from nife.counterfactual import CounterfactualExplainer, CounterfactualExplanation, is_counterfactual
from tests.conftest import SumModel


class EveryExample(CounterfactualExplainer):
    """A method that proposes every example of the data, even those that are not counterfactuals."""

    def _search(self, x, prediction, desired, rng, label="data"):
        return self.data.assign(source=label), {}


@pytest.fixture(scope="module")
def explainer(iris):
    X, model = iris
    return EveryExample(model, X)


def test_base_cannot_be_instantiated(iris):
    X, model = iris
    with pytest.raises(TypeError):
        CounterfactualExplainer(model, X)


def test_only_confirmed_counterfactuals(iris, explainer):
    X, model = iris
    explanation = explainer.explain(X.iloc[0])
    assert isinstance(explanation, CounterfactualExplanation)
    cfs = explanation.counterfactuals
    assert 0 < len(cfs) < len(X)
    assert (model.predict(cfs[X.columns]) != explanation.prediction).all()
    assert cfs["distance"].is_monotonic_increasing
    assert list(cfs.columns) == [
        *X.columns, "prediction", "source", "distance", "changed_features", "counterpart", "counterpart_distance",
    ]


def test_immutable_features_are_kept(iris):
    X, model = iris
    explainer = EveryExample(model, X, immutable_features=["sepal width (cm)"])
    x = X.iloc[50]  # sepal width 3.2, a value found in the three classes
    counterfactuals = explainer.explain(x).counterfactuals
    assert len(counterfactuals) > 0
    assert (counterfactuals["sepal width (cm)"] == x["sepal width (cm)"]).all()
    assert explainer.evaluate(X.iloc[[50]], counterfactuals.assign(instance=0))["constraint_violations"] == 0


def test_desired_and_selection(iris, explainer):
    X, model = iris
    explanation = explainer.explain(X.iloc[0], desired="virginica", max_counterfactuals=3)
    assert len(explanation.counterfactuals) <= 3
    assert (explanation.counterfactuals["prediction"] == "virginica").all()


def test_method_options(iris, explainer):
    X, _ = iris
    assert (explainer.explain(X.iloc[0], label="custom").counterfactuals["source"] == "custom").all()


def test_explain_many_and_evaluate(iris, explainer):
    X, model = iris
    instances = X.sample(6, random_state=0)
    counterfactuals, explanations = explainer.explain_many(instances, nondominated=True, return_explanations=True)
    assert len(explanations) == 6
    assert set(counterfactuals["instance"]) <= set(range(6))
    reference = model.predict(instances)[counterfactuals["instance"]]
    assert is_counterfactual(model.predict(counterfactuals[X.columns]), reference, "classification").all()
    assert explainer.evaluate(instances, counterfactuals)["validity"] == 1


def test_counterpart_of_examples_is_themselves(iris, explainer):
    # every counterfactual is an example of the data: the closest example with its outcome is itself
    X, _ = iris
    cfs = explainer.explain(X.iloc[0]).counterfactuals
    assert (cfs["counterpart_distance"] == 0).all()
    pd.testing.assert_frame_equal(
        X.loc[cfs["counterpart"].astype(int)].reset_index(drop=True), cfs[X.columns], check_dtype=False
    )


class Given(CounterfactualExplainer):
    """A method that proposes the candidates it is given."""

    def __init__(self, *args, candidates=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidates = candidates

    def _search(self, x, prediction, desired, rng):
        return self.candidates.assign(source="given"), {}


def test_counterpart_is_the_closest_example_with_the_same_outcome():
    # x1 + x2; the instance (5, 5) is predicted 10
    data = pd.DataFrame({"x1": [0.0, 10, 6, 4, 9], "x2": [0.0, 10, 4, 6, 0]})
    given = pd.DataFrame({"x1": [8.0, 2.0], "x2": [5.0, 5.0]})  # predicted 13 (higher) and 7 (lower)
    explainer = Given(SumModel(), data, mode="regression", candidates=given)
    cfs = explainer.explain({"x1": 5.0, "x2": 5.0}).counterfactuals.set_index("prediction")
    # 13: the closest example predicted higher than 10 by more than 5% is (10, 10), the only one
    assert cfs.loc[13.0, "counterpart"] == 1
    # 7: the examples predicted lower are (0, 0) and (9, 0); the closest to (2, 5) is (0, 0)
    lower = data.iloc[[0, 4]]
    distances = (abs(lower["x1"] - 2) / 10 + abs(lower["x2"] - 5) / 10) / 2
    assert cfs.loc[7.0, "counterpart"] == distances.idxmin()
    assert cfs.loc[7.0, "counterpart_distance"] == pytest.approx(distances.min())


def test_no_counterpart():
    data = pd.DataFrame({"x1": [5.0, 6.0], "x2": [5.0, 4.0]})  # every example is predicted 10
    given = pd.DataFrame({"x1": [9.0], "x2": [5.0]})
    cfs = Given(SumModel(), data, mode="regression", candidates=given).explain({"x1": 5.0, "x2": 5.0}).counterfactuals
    assert len(cfs) == 1 and pd.isna(cfs.loc[0, "counterpart"]) and np.isnan(cfs.loc[0, "counterpart_distance"])


def test_data_must_be_a_dataframe(iris):
    X, model = iris
    with pytest.raises(TypeError, match="pandas DataFrame"):
        EveryExample(model, X.to_numpy())


def test_regressor_needs_regression_mode():
    data = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [0.5, 1.5, 2.5]})
    with pytest.raises(ValueError, match="mode='regression'"):
        EveryExample(SumModel(), data)  # classification is the default
    EveryExample(SumModel(), data, mode="regression")


def test_classifier_needs_classification_mode(iris):
    X, model = iris
    with pytest.raises(ValueError, match="mode='classification'"):
        EveryExample(model, X, mode="regression")
