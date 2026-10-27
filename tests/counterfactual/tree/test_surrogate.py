import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from nife.counterfactual import TreeCounterfactualExplainer
from nife.counterfactual.tree import DEFAULT_MAX_DEPTH
from nife.counterfactual.tree.surrogate import climb_to_counterfactual, decision_path_rules, node_outcome


@pytest.fixture
def tree():
    """A tree whose root splits x1 at 5; on the left, x2 splits two leaves that both predict "a".

    x1 <= 5, x2 <= 5: 10 "a"            x1 > 5: 10 "b"
    x1 <= 5, x2 > 5:   6 "a", 4 "b"
    """
    X = np.array([[2, 2]] * 10 + [[2, 8]] * 10 + [[8, 5]] * 10, dtype=float)
    y = ["a"] * 10 + ["a"] * 6 + ["b"] * 4 + ["b"] * 10
    return DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)


def test_tree_shape(tree):
    x_row = np.array([[2.0, 8.0]], dtype=np.float32)
    rules = decision_path_rules(tree, x_row, ["x1", "x2"])
    assert [(r.feature, r.operator) for r in rules] == [("x1", "<="), ("x2", ">")]
    path = tree.decision_path(x_row).indices
    assert node_outcome(tree, path[-1]) == "a"


def test_climbs_when_the_sibling_has_the_same_outcome(tree):
    x_row = np.array([[2.0, 8.0]], dtype=np.float32)
    rules = decision_path_rules(tree, x_row, ["x1", "x2"])
    # the sibling leaf (x2 <= 5) also predicts "a": go up and cross the root (x1 <= 5)
    assert climb_to_counterfactual(tree, x_row, rules, "a", "classification") == 0


def test_desired_outcome_and_immutable_features(tree):
    x_row = np.array([[2.0, 8.0]], dtype=np.float32)
    rules = decision_path_rules(tree, x_row, ["x1", "x2"])
    assert climb_to_counterfactual(tree, x_row, rules, "a", "classification", desired="b") == 0
    # the root decides on x1: with x1 immutable, no decision of the path leads to "b"
    assert climb_to_counterfactual(tree, x_row, rules, "a", "classification", immutable_features=["x1"]) is None


def test_max_depth(iris):
    X, model = iris
    explainer = TreeCounterfactualExplainer(model, X, random_state=1)
    assert explainer.max_depth == DEFAULT_MAX_DEPTH == 5
    assert explainer.explain(X.iloc[0]).surrogate.get_depth() <= 5
    shallow = TreeCounterfactualExplainer(model, X, max_depth=1, random_state=1)
    assert shallow.explain(X.iloc[70]).surrogate.get_depth() <= 1
    with pytest.raises(ValueError):
        TreeCounterfactualExplainer(model, X, tree_params={"max_depth": 3})


def test_crossed_rule_leads_to_another_outcome(iris):
    X, model = iris
    explainer = TreeCounterfactualExplainer(model, X, random_state=1)
    for i in (0, 60, 120):
        explanation = explainer.explain(X.iloc[i])
        if explanation.crossed_rule is None:
            continue
        tree, x_row = explanation.surrogate, explanation.instance.to_numpy(np.float32).reshape(1, -1)
        path = tree.decision_path(x_row).indices
        parent = path[explanation.rules.index(explanation.crossed_rule)]
        left, right = tree.tree_.children_left[parent], tree.tree_.children_right[parent]
        sibling = right if left in path else left
        assert node_outcome(tree, sibling) != explanation.prediction
