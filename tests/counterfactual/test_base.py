"""The base class with a minimal method: every example of the data is a candidate.

Whatever the method returns, the base class must keep only the candidates that the
model confirms and that keep the immutable features, rank them and select them.
"""

import pandas as pd
import pytest

from nife.counterfactual import CounterfactualExplainer, CounterfactualExplanation, is_counterfactual


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
    assert list(cfs.columns) == [*X.columns, "prediction", "source", "distance", "changed_features"]


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
