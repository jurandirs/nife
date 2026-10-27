import numpy as np
import pandas as pd
import pytest

from nife.counterfactual import TreeCounterfactualExplainer, TreeCounterfactualExplanation, is_counterfactual
from nife.counterfactual.tree import Rule, fidelity_table


@pytest.fixture(scope="module")
def iris_explainer(iris):
    X, model = iris
    return TreeCounterfactualExplainer(model, X, random_state=1)


@pytest.fixture(scope="module")
def diabetes_explainer(diabetes):
    X, model = diabetes
    return TreeCounterfactualExplainer(
        model, X, mode="regression", categorical_features=["sex"], immutable_features=["sex"], random_state=1
    )


def test_explanation(iris, iris_explainer):
    X, model = iris
    explanation = iris_explainer.explain(X.iloc[0])
    assert isinstance(explanation, TreeCounterfactualExplanation)
    assert explanation.prediction == model.predict(X.iloc[[0]])[0]
    assert explanation.rules and all(isinstance(rule, Rule) for rule in explanation.rules)
    assert explanation.crossed_rule in explanation.rules
    cfs = explanation.counterfactuals
    assert len(cfs) > 0
    assert list(cfs.columns) == [*X.columns, "prediction", "source", "distance", "changed_features"]
    # valid by construction: the model gives them another class
    assert (model.predict(cfs[X.columns]) != explanation.prediction).all()
    assert (cfs["prediction"] == model.predict(cfs[X.columns])).all()
    assert cfs["distance"].is_monotonic_increasing
    # the counterfactuals from the data lie on the other side of the crossed rule
    assert not explanation.crossed_rule.holds(cfs[explanation.crossed_rule.feature]).any()
    assert set(explanation.feature_importances.index) <= set(X.columns)
    assert {"instance_agreement", "local_fidelity"} <= set(explanation.fidelity.index)
    # the candidates keep their index in the data
    candidates = explanation.candidates
    pd.testing.assert_frame_equal(candidates[X.columns], X.loc[candidates.index], check_dtype=False)
    assert "crossed rule" in repr(explanation)


def test_instance_formats_agree(iris, iris_explainer):
    X, _ = iris
    from_series = iris_explainer.explain(X.iloc[60]).counterfactuals
    for x in (X.iloc[[60]], X.iloc[60].to_dict(), X.iloc[60].to_numpy()):
        pd.testing.assert_frame_equal(iris_explainer.explain(x).counterfactuals, from_series)


def test_reproducible(iris, iris_explainer):
    X, _ = iris
    a = iris_explainer.explain_many(X.iloc[:5])
    b = iris_explainer.explain_many(X.iloc[:5])
    pd.testing.assert_frame_equal(a, b)


def test_desired_class(iris, iris_explainer):
    X, model = iris
    instances = X.sample(10, random_state=0)
    cfs = iris_explainer.explain_many(instances, desired="virginica")
    assert (model.predict(cfs[X.columns]) == "virginica").all()
    assert not set(cfs["instance"]) & set(np.flatnonzero(model.predict(instances) == "virginica"))
    assert iris_explainer.evaluate(instances, cfs, desired="virginica")["validity"] == 1


def test_selection(iris, iris_explainer):
    X, _ = iris
    instances = X.sample(10, random_state=0)
    everything = iris_explainer.explain_many(instances)
    closest = iris_explainer.explain_many(instances, max_counterfactuals=2)
    assert closest.groupby("instance").size().max() <= 2
    assert len(iris_explainer.explain_many(instances, nondominated=True)) <= len(everything)


def test_regression_desired_and_immutable(diabetes, diabetes_explainer):
    X, model = diabetes
    instances = X.sample(8, random_state=0)
    cfs = diabetes_explainer.explain_many(instances, desired="lower", from_rules=True)
    assert len(cfs) > 0
    reference = model.predict(instances)[cfs["instance"]]
    assert is_counterfactual(model.predict(cfs[X.columns]), reference, "regression", "lower").all()
    # the immutable feature never changes
    assert (cfs["sex"].to_numpy() == instances["sex"].to_numpy()[cfs["instance"]]).all()
    assert (cfs.loc[cfs["source"] == "rules", "changed_features"] == 1).all()
    summary = diabetes_explainer.evaluate(instances, cfs, desired="lower")
    assert summary["validity"] == 1
    assert summary["constraint_violations"] == 0


def test_fidelity_table(diabetes, diabetes_explainer):
    X, _ = diabetes
    _, explanations = diabetes_explainer.explain_many(X.iloc[:4], return_explanations=True)
    fidelity = fidelity_table(explanations)
    assert len(fidelity) == 4
    assert fidelity["local_fidelity"].between(0, 1).all()
    assert fidelity["sample_around_instance"].dtype == bool


def test_global_neighborhood(iris):
    X, model = iris
    explanation = TreeCounterfactualExplainer(model, X, adaptive=False, random_state=1).explain(X.iloc[0])
    assert not explanation.fidelity["sample_around_instance"]


def test_plot_tree(iris, iris_explainer):
    X, _ = iris
    ax = iris_explainer.explain(X.iloc[0]).plot_tree()
    colors = {tuple(t.get_bbox_patch().get_facecolor()) for t in ax.texts if t.get_bbox_patch() is not None}
    assert len(colors) >= 2  # the path and the crossed side


def test_invalid_inputs(iris):
    X, model = iris
    with pytest.raises(ValueError):
        TreeCounterfactualExplainer(model, X, mode="ranking")
    with pytest.raises(ValueError):
        TreeCounterfactualExplainer(model, X, immutable_features=["height"])
    with pytest.raises(ValueError):
        TreeCounterfactualExplainer(model, X.assign(kind="a"))
    explainer = TreeCounterfactualExplainer(model, X)
    with pytest.raises(ValueError):
        explainer.explain(X.iloc[0], desired="rose")
    with pytest.raises(ValueError):
        explainer.explain(X.iloc[:2])
