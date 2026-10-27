import pytest

from nife.counterfactual import select_counterfactuals


@pytest.fixture
def ranked(toy):
    _, _, counterfactuals = toy
    # distances computed by hand (see conftest.toy)
    return counterfactuals.assign(distance=[20 / 30, 7 / 30, 2 / 30, 18 / 30], changed_features=[3, 1, 2, 3])


def test_ranked_by_distance(ranked):
    assert select_counterfactuals(ranked)["x1"].tolist() == [3, 2, 8, 5]


def test_nondominated(ranked):
    # (8, 6, blue) is farther and changes more features than both others of instance 0
    selected = select_counterfactuals(ranked, nondominated=True)
    assert selected[["instance", "x1"]].values.tolist() == [[0, 3], [0, 2], [1, 5]]


def test_max_counterfactuals(ranked):
    assert select_counterfactuals(ranked, max_counterfactuals=1)["x1"].tolist() == [3, 5]


def test_without_instance_column(ranked):
    assert select_counterfactuals(ranked.drop(columns="instance"), max_counterfactuals=2)["x1"].tolist() == [3, 2]


def test_requires_distance_columns(toy):
    with pytest.raises(ValueError):
        select_counterfactuals(toy[2])
