import pandas as pd

from nife.counterfactual.tree import Rule, counterfactuals_from_rules
from nife.counterfactual.tree.search import search_existing
from tests.conftest import ThresholdModel

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
    assert found["is_counterfactual"].tolist() == [False, True]
    # cross the first rule: x2 > 7.5, the rules after it are ignored
    assert search_existing(data, RULES, 0, "low", "classification")[["x1", "x2"]].values.tolist() == [[0, 10], [10, 10]]
