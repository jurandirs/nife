import numpy as np
import pytest

from nife.counterfactual import is_counterfactual
from nife.counterfactual.outcomes import check_desired


def test_classification_any_other_class():
    assert is_counterfactual(["a", "b", "c"], "a", "classification").tolist() == [False, True, True]


def test_classification_desired_class():
    assert is_counterfactual(["a", "b", "c"], "a", "classification", desired="c").tolist() == [False, False, True]
    # an instance that already has the desired class has no counterfactual
    assert is_counterfactual(["a", "c"], "c", "classification", desired="c").tolist() == [False, False]


@pytest.mark.parametrize(
    "desired, expected",
    [
        (None, [True, False, True]),  # more than 5% of 4 = 0.2, either way
        ("higher", [True, False, False]),
        ("lower", [False, False, True]),
        ((10, None), [True, False, False]),
        ((None, 3.5), [False, False, True]),
    ],
)
def test_regression(desired, expected):
    assert is_counterfactual([11.0, 4.1, 3.0], 4.0, "regression", desired).tolist() == expected


def test_regression_interval_already_reached():
    # the instance (4.6) is already in [4.5, inf): nothing is a counterfactual
    assert is_counterfactual([4.5, 5.0], 4.6, "regression", (4.5, None)).tolist() == [False, False]


def test_broadcasting():
    result = is_counterfactual(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0], [4.0]]), "regression")
    assert result.tolist() == [[False, True], [True, False]]


@pytest.mark.parametrize(
    "desired, mode",
    [("up", "regression"), ((5, 1), "regression"), ((None, None), "regression"), ("d", "classification")],
)
def test_invalid_desired(desired, mode):
    with pytest.raises(ValueError):
        check_desired(desired, mode, classes=["a", "b"])


def test_invalid_mode():
    with pytest.raises(ValueError):
        check_desired(None, "ranking")
