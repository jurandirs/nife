"""Plots of the explanations of the surrogate decision tree method."""

import pytest
from matplotlib.axes import Axes

from nife.counterfactual import TreeCounterfactualExplainer, TreeCounterfactualExplanation
from nife.counterfactual.tree.plotting import format_number, node_labels


@pytest.fixture(scope="module")
def regression_explanation(diabetes):
    X, model = diabetes
    return TreeCounterfactualExplainer(model, X, mode="regression", random_state=0).explain(X.iloc[0])


@pytest.fixture(scope="module")
def explanation(iris):
    X, model = iris
    return TreeCounterfactualExplainer(model, X, random_state=1).explain(X.iloc[60])


def test_tree_explanations_add_their_plots():
    assert list(TreeCounterfactualExplanation.plot_kinds()) == ["changes", "importances", "tree"]
    assert TreeCounterfactualExplanation.default_plot == "changes"


def test_every_plot(explanation):
    for kind in explanation.plot_kinds():
        assert isinstance(explanation.plot(kind), Axes), kind


def test_importances_are_ordered(explanation):
    ax = explanation.plot_importances()
    labels = [t.get_text() for t in ax.get_yticklabels()][::-1]  # from the top
    assert labels == list(explanation.feature_importances.index)
    widths = [p.get_width() for p in ax.patches][: len(labels)]
    assert widths == sorted(widths, reverse=True)
    ax = explanation.plot_importances(max_features=1)
    assert [t.get_text() for t in ax.get_yticklabels()] == [explanation.feature_importances.index[0]]


def test_a_tree_without_decisions(iris):
    X, model = iris
    explainer = TreeCounterfactualExplainer(model, X, random_state=1)
    neighborhood = explainer.sample_neighborhood(X.iloc[0], around_instance=True, scale=0.01)
    single_leaf = explainer.explain(X.iloc[0], neighborhood=neighborhood)
    assert not single_leaf.rules
    with pytest.raises(ValueError, match="single leaf"):
        single_leaf.plot_importances()
    assert isinstance(single_leaf.plot_tree(), Axes)


def test_tree_shows_the_information_of_the_paper(explanation):
    tree = explanation.surrogate
    labels = node_labels(explanation)
    root = labels[0].split("\n")
    assert root[0].startswith(explanation.instance.index[tree.tree_.feature[0]] + " ≤ ")
    assert root[1].startswith("class = ")
    leaves = [n for n in range(tree.tree_.node_count) if tree.tree_.children_left[n] == -1]
    assert all(labels[n].startswith("class = ") and "\n" not in labels[n] for n in leaves)
    ax = explanation.plot_tree()
    texts = [t.get_text() for t in ax.texts]
    assert all(label in texts for label in labels.values())
    assert texts.count("True") == texts.count("False") == len(leaves) - 1


def test_tree_path_only_and_titles(regression_explanation):
    explanation = regression_explanation
    tree = explanation.surrogate
    full = explanation.plot_tree()
    small = explanation.plot_tree(path_only=True, titles=False)
    assert small.get_title(loc="left") == "" and full.get_title(loc="left") == "Surrogate tree"
    n_path = len(explanation.rules) + 1
    shown = [t for t in small.texts if t.get_text() not in ("True", "False")]
    assert len(shown) == 2 * n_path - 1  # the path and one node for the other side of each decision
    assert small.figure.get_size_inches()[0] <= full.figure.get_size_inches()[0]
    assert any("value = " in t.get_text() for t in shown)
    assert len(shown) < tree.tree_.node_count


def test_format_number():
    assert format_number(715.40401) == "715.404"
    assert format_number(3.52951234) == "3.5295"
    assert format_number(2.0) == "2"
    assert format_number(-0.00001) == "0"
