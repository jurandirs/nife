"""Extracting the neighborhood of an explanation and explaining with a neighborhood of one's own."""

import numpy as np
import pandas as pd
import pytest

from nife.counterfactual import TreeCounterfactualExplainer
from nife.counterfactual.tree import fidelity_table
from nife.counterfactual.tree.surrogate import fit_surrogate


def same_explanation(a, b):
    assert a.rules == b.rules
    assert a.crossed_rule == b.crossed_rule
    pd.testing.assert_frame_equal(a.counterfactuals, b.counterfactuals)


def test_the_explanation_keeps_its_neighborhood(iris):
    X, model = iris
    explainer = TreeCounterfactualExplainer(model, X, random_state=1)
    explanation = explainer.explain(X.iloc[0])
    neighborhood = explanation.neighborhood
    assert list(neighborhood.columns) == [*X.columns, "prediction"]
    assert len(neighborhood) == explainer.num_samples
    assert (neighborhood["prediction"].to_numpy() == model.predict(neighborhood[X.columns])).all()
    # NIFE's sampling starts with the instance itself
    np.testing.assert_array_equal(neighborhood.iloc[0][X.columns].to_numpy(float), X.iloc[0].to_numpy())
    # the tree was trained on it
    points = neighborhood[X.columns].to_numpy(np.float32)
    refitted = fit_surrogate(points, neighborhood["prediction"], "classification")
    np.testing.assert_array_equal(refitted.predict(points), explanation.surrogate.predict(points))
    assert explanation.fidelity["neighborhood"] == "sampled"


def test_immutable_features_are_fixed_in_the_neighborhood(diabetes):
    X, model = diabetes
    explainer = TreeCounterfactualExplainer(
        model, X, mode="regression", categorical_features=["sex"], immutable_features=["age", "sex"], random_state=1
    )
    neighborhood = explainer.explain(X.iloc[3]).neighborhood
    for feature in ("age", "sex"):  # the instance's exact values
        assert (neighborhood[feature] == X.iloc[3][feature]).all()
    # so they stay within the range of the data (in float32, the maximum of sex would not)
    fixed = neighborhood[["age", "sex"]]
    assert ((fixed >= X[["age", "sex"]].min()) & (fixed <= X[["age", "sex"]].max())).all().all()
    assert neighborhood["bmi"].nunique() > 1


def test_giving_back_the_neighborhood_gives_the_same_explanation(iris):
    X, model = iris
    explainer = TreeCounterfactualExplainer(model, X, random_state=1)
    for i in (0, 60, 120):
        explanation = explainer.explain(X.iloc[i])
        again = explainer.explain(X.iloc[i], neighborhood=explanation.neighborhood)
        same_explanation(explanation, again)
        assert again.fidelity["neighborhood"] == "given"
        assert "neighborhood: given" in repr(again)


def test_sample_neighborhood_is_the_neighborhood_of_explain(iris):
    X, model = iris
    explainer = TreeCounterfactualExplainer(model, X, adaptive=False, random_state=1)
    neighborhood = explainer.sample_neighborhood(X.iloc[60])
    assert list(neighborhood.columns) == list(X.columns)
    assert len(neighborhood) == explainer.num_samples
    explanation = explainer.explain(X.iloc[60])
    pd.testing.assert_frame_equal(neighborhood, explanation.neighborhood[X.columns])
    same_explanation(explanation, explainer.explain(X.iloc[60], neighborhood=neighborhood))


def test_sample_neighborhood_options(iris):
    X, model = iris
    explainer = TreeCounterfactualExplainer(model, X)
    narrow = explainer.sample_neighborhood(X.iloc[0], around_instance=True, scale=0.1, num_samples=200, random_state=0)
    assert len(narrow) == 200
    assert (narrow.std() < X.std() * 0.2).all()
    assert (narrow.mean() - X.iloc[0]).abs().max() < 0.1
    again = explainer.sample_neighborhood(X.iloc[0], around_instance=True, scale=0.1, num_samples=200, random_state=0)
    pd.testing.assert_frame_equal(narrow, again)


def test_a_changed_neighborhood(iris):
    X, model = iris
    explainer = TreeCounterfactualExplainer(model, X, random_state=1)
    neighborhood = explainer.sample_neighborhood(X.iloc[0], around_instance=True, scale=0.5)
    neighborhood = neighborhood[neighborhood["petal length (cm)"] > 0]  # drop impossible points
    explanation = explainer.explain(X.iloc[0], neighborhood=neighborhood)
    assert len(explanation.neighborhood) == len(neighborhood)
    assert (explanation.neighborhood["petal length (cm)"] > 0).all()
    assert explanation.rules and len(explanation.counterfactuals) > 0


def test_array_neighborhood_and_extra_columns(iris):
    X, model = iris
    explainer = TreeCounterfactualExplainer(model, X, random_state=1)
    neighborhood = explainer.sample_neighborhood(X.iloc[0], around_instance=True)
    from_frame = explainer.explain(X.iloc[0], neighborhood=neighborhood)
    from_array = explainer.explain(X.iloc[0], neighborhood=neighborhood.to_numpy())
    reordered = neighborhood[X.columns[::-1]].assign(note="ignored")
    from_reordered = explainer.explain(X.iloc[0], neighborhood=reordered)
    same_explanation(from_frame, from_array)
    same_explanation(from_frame, from_reordered)


def test_the_model_labels_a_given_neighborhood(iris):
    X, model = iris
    explainer = TreeCounterfactualExplainer(model, X, random_state=1)
    neighborhood = explainer.explain(X.iloc[0]).neighborhood
    wrong = neighborhood.assign(prediction=0)  # labels given by the user are not used
    labels = explainer.explain(X.iloc[0], neighborhood=wrong).neighborhood["prediction"]
    assert (labels.to_numpy() == model.predict(neighborhood[X.columns])).all()


def test_sampler(diabetes):
    X, model = diabetes
    calls = []

    def sampler(instance, data, num_samples, random_state):
        calls.append((instance, data, num_samples, random_state))
        noise = random_state.normal(0, 0.5, size=(num_samples, data.shape[1])) * data.std().to_numpy()
        return instance.to_numpy() + noise  # a 2-D array, features in the order of the data

    explainer = TreeCounterfactualExplainer(
        model, X, mode="regression", immutable_features=["sex"], num_samples=300, sampler=sampler, random_state=1
    )
    explanation = explainer.explain(X.iloc[5], desired="lower")
    assert len(calls) == 1  # no adaptive neighborhood: the sampler is called once
    instance, data, num_samples, random_state = calls[0]
    pd.testing.assert_series_equal(instance, X.iloc[5], check_names=False)
    pd.testing.assert_frame_equal(data, X)
    assert num_samples == 300 and isinstance(random_state, np.random.RandomState)
    assert len(explanation.neighborhood) == 300
    assert (explanation.neighborhood["sex"] == X.iloc[5]["sex"]).all()
    assert explanation.fidelity["neighborhood"] == "sampler"
    assert "neighborhood: from the sampler" in repr(explanation)
    # reproducible: the sampler gets a random state seeded from the explainer's
    same_explanation(explanation, explainer.explain(X.iloc[5], desired="lower"))
    # a neighborhood given to explain takes the place of the sampler
    given = explainer.explain(X.iloc[5], desired="lower", neighborhood=explanation.neighborhood)
    assert len(calls) == 2 and given.fidelity["neighborhood"] == "given"


def test_explain_many_with_a_neighborhood(diabetes):
    X, model = diabetes
    explainer = TreeCounterfactualExplainer(model, X, mode="regression", random_state=1)
    global_sample = explainer.sample_neighborhood(X.iloc[0], around_instance=False)
    counterfactuals, explanations = explainer.explain_many(
        X.iloc[:3], neighborhood=global_sample, return_explanations=True
    )
    assert (fidelity_table(explanations)["neighborhood"] == "given").all()
    for explanation in explanations:
        pd.testing.assert_frame_equal(explanation.neighborhood[X.columns], global_sample)
    assert set(counterfactuals["instance"]) <= {0, 1, 2}


@pytest.mark.parametrize(
    "neighborhood, message",
    [
        (lambda X: X.drop(columns="sepal width (cm)"), "lacks the features"),
        (lambda X: X.to_numpy()[:, :3], "2-D array"),
        (lambda X: X.to_numpy()[0], "2-D array"),
        (lambda X: X.astype(object).assign(**{"sepal width (cm)": "wide"}), "numeric"),
        (lambda X: X.assign(**{"sepal width (cm)": np.nan}), "missing or infinite"),
        (lambda X: X.iloc[:0], "empty"),
    ],
)
def test_invalid_neighborhoods(iris, neighborhood, message):
    X, model = iris
    explainer = TreeCounterfactualExplainer(model, X)
    with pytest.raises(ValueError, match=message):
        explainer.explain(X.iloc[0], neighborhood=neighborhood(X))


def test_invalid_sampler(iris):
    X, model = iris
    with pytest.raises(TypeError):
        TreeCounterfactualExplainer(model, X, sampler="normal")
    explainer = TreeCounterfactualExplainer(model, X, sampler=lambda instance, data, n, rng: data.iloc[:, :2])
    with pytest.raises(ValueError, match="the neighborhood of the sampler lacks"):
        explainer.explain(X.iloc[0])


def test_default_neighborhood_size_is_capped_for_large_data():
    from nife.counterfactual.tree.explainer import MAX_DEFAULT_SAMPLES
    from tests.conftest import SumModel

    small = pd.DataFrame({"x1": np.arange(100.0), "x2": np.arange(100.0)})
    large = pd.DataFrame({"x1": np.arange(100_000.0), "x2": np.arange(100_000.0)})
    assert TreeCounterfactualExplainer(SumModel(), small, mode="regression").num_samples == 500
    assert TreeCounterfactualExplainer(SumModel(), large, mode="regression").num_samples == MAX_DEFAULT_SAMPLES
    assert TreeCounterfactualExplainer(SumModel(), large, mode="regression", num_samples=7).num_samples == 7
