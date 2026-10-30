import numpy as np
import pandas as pd

from nife.counterfactual.tree import Rule, counterfactuals_from_evidence, counterfactuals_from_rules
from nife.counterfactual.tree.search import search_existing
from tests.conftest import ThresholdModel


class BothModel:
    """``high`` only when both x1 and x2 are above 5: no single feature changes the outcome of (2, 2)."""

    classes_ = np.array(["high", "low"])

    def predict(self, X):
        return np.where((X["x1"] > 5) & (X["x2"] > 5), "high", "low")


class FirstModel:
    """``high`` when x1 is above 5, whatever the other features."""

    def predict(self, X):
        return np.where(X["x1"] > 5, "high", "low")

# A decision path of the instance (2, 2): x2 <= 7.5, then x1 <= 5
RULES = [Rule("x2", "<=", 7.5), Rule("x1", "<=", 5.0)]


def test_rule():
    rule = Rule("x1", "<=", 5.0)
    assert str(rule) == "x1 <= 5"
    assert rule.inverted() == Rule("x1", ">", 5.0)
    assert rule.holds(pd.Series([4, 5, 6])).tolist() == [True, True, False]


def test_counterfactuals_from_rules(toy):
    data = toy[0][["x1", "x2"]]
    x = pd.Series({"x1": 2.0, "x2": 2.0})
    found = counterfactuals_from_rules(x, RULES, data, ThresholdModel())
    # x1 above 5: (8, 2) is still low, (10, 2) is high; x2 above 7.5: (2, 10) is high
    assert found[["x1", "x2"]].values.tolist() == [[10, 2], [2, 10]]
    assert found["prediction"].tolist() == ["high", "high"]
    assert found["rule"].tolist() == [1, 0]  # the deepest decision first


def test_counterfactual_rule_first(toy):
    data = toy[0][["x1", "x2"]]
    x = pd.Series({"x1": 2.0, "x2": 2.0})
    found = counterfactuals_from_rules(x, RULES, data, ThresholdModel(), crossed=0)
    assert found["rule"].tolist() == [0, 1]
    assert found[["x1", "x2"]].values.tolist() == [[2, 10], [10, 2]]


def test_counterfactuals_from_rules_immutable(toy):
    data = toy[0][["x1", "x2"]]
    x = pd.Series({"x1": 2.0, "x2": 2.0})
    found = counterfactuals_from_rules(x, RULES, data, ThresholdModel(), immutable_features=["x1"])
    assert found[["x1", "x2"]].values.tolist() == [[2, 10]]


def test_search_existing(toy):
    data = toy[0][["x1", "x2"]]
    data = data.assign(prediction=ThresholdModel().predict(data))
    # keep x2 <= 7.5 and cross x1 <= 5: the examples with x2 <= 7.5 and x1 > 5
    found = search_existing(data, RULES, 1, "low", "classification")
    assert found[["x1", "x2"]].values.tolist() == [[10, 0], [8, 6]]
    assert found["is_evidence"].tolist() == [False, True]
    # cross the first rule: x2 > 7.5, the rules after it are ignored
    assert search_existing(data, RULES, 0, "low", "classification")[["x1", "x2"]].values.tolist() == [[0, 10], [10, 10]]


X0 = pd.Series({"x1": 2.0, "x2": 2.0})


def test_from_evidence_single_feature():
    evidence = pd.DataFrame({"x1": [10.0], "x2": [10.0]})
    found = counterfactuals_from_evidence(X0, evidence, ThresholdModel())
    # x1 + x2 > 10 with one of the example's values: (10, 2) and (2, 10)
    assert sorted(found[["x1", "x2"]].values.tolist()) == [[2, 10], [10, 2]]
    assert (found["prediction"] == "high").all()
    only_x2 = counterfactuals_from_evidence(X0, evidence, ThresholdModel(), immutable_features=["x1"])
    assert only_x2[["x1", "x2"]].values.tolist() == [[2, 10]]


def test_from_evidence_fewest_features():
    # no single feature is enough: both, from each example
    evidence = pd.DataFrame({"x1": [8.0, 10.0], "x2": [6.0, 10.0]})
    found = counterfactuals_from_evidence(X0, evidence, BothModel())
    assert found[["x1", "x2"]].values.tolist() == [[8, 6], [10, 10]]
    # with x1 immutable, nothing works
    assert counterfactuals_from_evidence(X0, evidence, BothModel(), immutable_features=["x1"]).empty


def test_from_evidence_examples_and_smallest_sets():
    # the first example needs both features, the second only x2: only the smaller set is kept
    evidence = pd.DataFrame({"x1": [8.0, 4.0], "x2": [6.0, 9.0]})
    found = counterfactuals_from_evidence(X0, evidence, ThresholdModel())
    assert found[["x1", "x2"]].values.tolist() == [[2, 9]]
    assert counterfactuals_from_evidence(X0, evidence, ThresholdModel(), max_examples=1)[["x1", "x2"]].values.tolist() == [[8, 6]]


def test_from_evidence_largest_changes_first_then_revert():
    x = pd.Series({"x1": 2.0, "x2": 2.0, "x3": 2.0})
    evidence = pd.DataFrame({"x1": [6.0], "x2": [50.0], "x3": [30.0]})
    # max_changes=0 skips the combinations: x2 and x3 (larger) are copied before x1, then reverted
    found = counterfactuals_from_evidence(x, evidence, FirstModel(), max_changes=0)
    assert found[["x1", "x2", "x3"]].values.tolist() == [[6, 2, 2]]
    # on the scale of each feature, the order changes, but the result does not
    scale = pd.Series({"x1": 1.0, "x2": 0.01, "x3": 0.01})
    found = counterfactuals_from_evidence(x, evidence, FirstModel(), max_changes=0, scale=scale)
    assert found[["x1", "x2", "x3"]].values.tolist() == [[6, 2, 2]]
    # both features, when they are needed
    evidence = pd.DataFrame({"x1": [8.0], "x2": [6.0]})
    found = counterfactuals_from_evidence(X0, evidence, BothModel(), max_changes=0)
    assert found[["x1", "x2"]].values.tolist() == [[8, 6]]


def test_from_evidence_nothing():
    empty = counterfactuals_from_evidence(X0, pd.DataFrame(columns=["x1", "x2"]), ThresholdModel())
    assert empty.empty and list(empty.columns) == ["x1", "x2", "prediction"]
