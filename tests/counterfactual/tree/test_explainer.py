import numpy as np
import pandas as pd
import pytest

from nife.counterfactual import TreeCounterfactualExplainer, TreeCounterfactualExplanation, is_counterfactual
from nife.counterfactual.tree import Rule, fidelity_table


@pytest.fixture(scope="module")
def iris_explainer(iris):
    X, model = iris
    return TreeCounterfactualExplainer(model, X, random_state=1)


class CornerModel:
    """``high`` when x1 is above 5 and x2 above 2: from (2, 1), x1 alone is not enough."""

    classes_ = np.array(["high", "low"])

    def predict(self, X):
        return np.where((X["x1"] > 5) & (X["x2"] > 2), "high", "low")


@pytest.fixture(scope="module")
def corner():
    grid = np.arange(0, 10.5, 0.5)
    data = pd.DataFrame([(a, b) for a in grid for b in grid], columns=["x1", "x2"])
    model = CornerModel()
    return data, model, TreeCounterfactualExplainer(model, data, adaptive=False, random_state=0)


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
    assert list(cfs.columns) == [
        *X.columns, "prediction", "source", "distance", "changed_features", "counterpart", "counterpart_distance",
    ]
    # valid by construction: the model gives them another class
    assert (model.predict(cfs[X.columns]) != explanation.prediction).all()
    assert (cfs["prediction"] == model.predict(cfs[X.columns])).all()
    assert cfs["distance"].is_monotonic_increasing
    # the fewest changes: a single feature across a decision of the path
    assert set(cfs["source"]) == {"rule"} and (cfs["changed_features"] == 1).all()
    assert set(explanation.feature_importances.index) <= set(X.columns)
    assert {"instance_agreement", "local_fidelity"} <= set(explanation.fidelity.index)
    assert "counterfactual rule" in repr(explanation)


def test_counterfactual_rule_is_confirmed_by_a_single_change(iris, iris_explainer):
    X, _ = iris
    for i in (0, 60, 120):
        explanation = iris_explainer.explain(X.iloc[i])
        rule = explanation.crossed_rule
        cfs = explanation.counterfactuals
        across = cfs[(cfs["source"] == "rule") & ~rule.holds(cfs[rule.feature])]
        crossing = across[(across[X.columns] != explanation.instance).sum(axis=1) == 1]
        assert explanation.crossed_rule_confirmed == (len(crossing) > 0)


def test_evidence_of_the_rule(iris, iris_explainer):
    X, model = iris
    explanation = iris_explainer.explain(X.iloc[60])
    evidence, rule = explanation.evidence, explanation.crossed_rule
    assert len(evidence) > 0
    assert list(evidence.columns) == [*X.columns, "prediction", "distance"]
    # real examples, with their index in the data
    pd.testing.assert_frame_equal(evidence[X.columns], X.loc[evidence.index], check_dtype=False)
    # on the other side of the rule, after the decisions above it, with another outcome for the model
    position = explanation.rules.index(rule)
    for above in explanation.rules[:position]:
        assert above.holds(evidence[above.feature]).all()
    assert not rule.holds(evidence[rule.feature]).any()
    assert (model.predict(evidence[X.columns]) != explanation.prediction).all()
    assert evidence["distance"].is_monotonic_increasing
    # the candidates for evidence are the examples on the other side, evidence or not
    candidates = explanation.evidence_candidates
    assert set(evidence.index) == set(candidates.index[candidates["is_evidence"]])


def test_rule_not_confirmed_counterfactuals_from_evidence(corner):
    data, model, explainer = corner
    explanation = explainer.explain({"x1": 2.0, "x2": 1.0})
    assert explanation.crossed_rule.feature == "x1"
    # moving x1 alone keeps the outcome: the rule is not confirmed by the model
    assert explanation.crossed_rule_confirmed is False
    assert "not confirmed" in repr(explanation)
    # the counterfactuals copy the fewest features from the evidence: both
    cfs = explanation.counterfactuals
    assert len(cfs) > 0 and set(cfs["source"]) == {"evidence"}
    assert (cfs["changed_features"] == 2).all()
    assert (model.predict(cfs[["x1", "x2"]]) == "high").all()
    # the evidence: examples with x1 on the other side that the model puts in the other class
    assert len(explanation.evidence) > 0
    assert (model.predict(explanation.evidence[["x1", "x2"]]) == "high").all()


def test_rule_confirmed_single_change(corner):
    _, model, explainer = corner
    explanation = explainer.explain({"x1": 2.0, "x2": 8.0})
    assert explanation.crossed_rule.feature == "x1"
    assert explanation.crossed_rule_confirmed is True
    cfs = explanation.counterfactuals
    assert set(cfs["source"]) == {"rule"} and (cfs["changed_features"] == 1).all()
    # the closest value of x1 on the other side that changes the class
    assert cfs.iloc[0][["x1", "x2"]].tolist() == [5.5, 8.0]
    assert "confirmed by the model" in repr(explanation)


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
    cfs = diabetes_explainer.explain_many(instances, desired="lower")
    assert len(cfs) > 0
    reference = model.predict(instances)[cfs["instance"]]
    assert is_counterfactual(model.predict(cfs[X.columns]), reference, "regression", "lower").all()
    # the immutable feature never changes
    assert (cfs["sex"].to_numpy() == instances["sex"].to_numpy()[cfs["instance"]]).all()
    assert set(cfs["source"]) <= {"rule", "evidence"}
    assert (cfs.loc[cfs["source"] == "rule", "changed_features"] == 1).all()
    # the counterfactuals of an instance all have its fewest changes
    assert (cfs.groupby("instance")["changed_features"].nunique() == 1).all()
    # their counterparts have the desired outcome
    with_counterpart = cfs.dropna(subset=["counterpart"])
    counterpart_predictions = model.predict(X.loc[with_counterpart["counterpart"].astype(int)])
    assert is_counterfactual(counterpart_predictions, reference[with_counterpart.index], "regression", "lower").all()
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
