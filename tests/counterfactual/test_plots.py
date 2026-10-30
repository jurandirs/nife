"""The plot of every counterfactual explanation: what a counterfactual changes."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from nife.counterfactual import CounterfactualExplanation, TreeCounterfactualExplainer, plots
from nife.plotting import COLORS


@pytest.fixture(scope="module")
def explanation(diabetes):
    X, model = diabetes
    explainer = TreeCounterfactualExplainer(
        model, X, mode="regression", categorical_features=["sex"], immutable_features=["age"], random_state=0
    )
    return explainer.explain(X.iloc[0], desired="lower")


def texts(ax):
    return " ".join(t.get_text() for t in ax.texts) + " " + ax.get_title(loc="left")


def test_counterfactual_explanations_draw_their_changes():
    assert list(CounterfactualExplanation.plot_kinds()) == ["changes"]
    assert CounterfactualExplanation.default_plot == "changes"


def test_changes(explanation):
    closest = explanation.counterfactuals.iloc[0]
    ax = explanation.plot()
    assert isinstance(ax, Axes)
    changed = [f for f in explanation.instance.index if closest[f] != explanation.instance[f]]
    assert sorted(t.get_text() for t in ax.get_yticklabels()) == sorted(changed)
    assert f"1 of {len(explanation.counterfactuals)}" in texts(ax)
    # another counterfactual, counted from the end
    assert f"{len(explanation.counterfactuals)} of" in texts(explanation.plot_changes(-1))
    with pytest.raises(IndexError):
        explanation.plot_changes(len(explanation.counterfactuals))


def test_changes_draws_on_the_given_axes(explanation):
    figure, axes = plt.subplots(1, 2)
    assert explanation.plot_changes(ax=axes[1]) is axes[1]
    assert axes[1].figure is figure


def test_changes_shows_the_largest_changes_first():
    data = pd.DataFrame({"a": [0.0, 10.0], "b": [0.0, 100.0], "c": [0.0, 1.0]})
    instance = pd.Series({"a": 5.0, "b": 50.0, "c": 0.5})
    counterfactual = pd.Series({"a": 6.0, "b": 90.0, "c": 0.5})
    ax = plots.plot_changes(instance, counterfactual, data, max_features=1)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["b"]  # 40% of its range, a only 10%
    assert "1 more changed feature not shown" in texts(ax)
    assert "50  →  90" in texts(ax)
    with pytest.raises(ValueError, match="changes no feature"):
        plots.plot_changes(instance, instance, data)
    with pytest.raises(ValueError, match="lacks the features"):
        plots.plot_changes(instance, counterfactual, data.drop(columns="c"))


def test_categorical_features_are_placed_by_their_codes():
    scale = plots.FeatureScale(pd.DataFrame({"color": [0, 1, 2, 1], "x": [0.0, 1.0, 2.0, 4.0]}), ["color"])
    np.testing.assert_allclose(scale.position("color", [0, 1, 2]), [0, 0.5, 1])
    np.testing.assert_allclose(scale.change("color", [0, 2], [0, 1]), [0, 1])
    np.testing.assert_allclose(scale.change("x", [3.0], [1.0]), [0.5])


def test_explanations_without_counterfactuals_or_data(explanation):
    empty = CounterfactualExplanation(explanation.instance, 1.0, explanation.counterfactuals.iloc[:0], "regression")
    with pytest.raises(ValueError, match="no counterfactuals"):
        empty.plot()
    no_data = CounterfactualExplanation(explanation.instance, 1.0, explanation.counterfactuals, "regression")
    with pytest.raises(ValueError, match="reference_data"):
        no_data.plot_changes()


def test_colors_can_be_changed(explanation, monkeypatch):
    monkeypatch.setitem(COLORS, "counterfactual", "#d62728")
    ax = explanation.plot_changes()
    red = np.array(plt.matplotlib.colors.to_rgba("#d62728"))
    assert any(np.allclose(c.get_facecolor()[0], red) for c in ax.collections if len(c.get_facecolor()))
