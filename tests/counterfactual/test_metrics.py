"""Every metric on the toy example of conftest, against values computed by hand."""

from fractions import Fraction as F

import numpy as np
import pandas as pd
import pytest

from nife.counterfactual import METRICS, evaluate_counterfactuals, register_metric
from tests.conftest import SumModel, ThresholdModel

OPTIONS = dict(categorical_features=["color"], immutable_features=["color"], k=2)

# Per instance (instance 0, instance 1); instance 2 has no counterfactual.
BY_HAND = {
    "coverage": F(2, 3),
    "count": F(3 + 1, 2),
    # (3, 3, red) is still low like instance 0
    "validity": (F(2, 3) + 1) / 2,
    # distances 20/30, 7/30, 2/30; 18/30
    "proximity": (F(29, 90) + F(18, 30)) / 2,
    "changed_features": F(2 + 3, 2),
    # reverting the color keeps (8, 6, blue) high and (5, 5, red) low
    "redundancy": (F(1, 2) + 1) / 2,
    # closest reference point of (2, 9, red) and (3, 3, red): (5, 5, red), at 7/30 and 4/30
    "implausibility": (F(11, 90) + 0) / 2,
    # neighbours with a counterfactual outcome: 1/2, 0, 0; 1 (the point itself is left out)
    "ynn": (F(1, 6) + 1) / 2,
    # pairs of instance 0: 19/30, 18/30, 7/30
    "diversity": F(44, 90),
    "constraint_violations": (F(1, 3) + 1) / 2,
}


def test_summary_matches_hand_computation(toy):
    data, instances, counterfactuals = toy
    summary = evaluate_counterfactuals(instances, counterfactuals, ThresholdModel(), data, **OPTIONS)
    for metric, value in BY_HAND.items():
        assert summary[metric] == pytest.approx(float(value)), metric


def test_per_instance(toy):
    data, instances, counterfactuals = toy
    table = evaluate_counterfactuals(instances, counterfactuals, ThresholdModel(), data, **OPTIONS, per_instance=True)
    assert table["counterfactuals"].tolist() == [3, 1, 0]
    assert table.loc[2].drop("counterfactuals").isna().all()
    assert np.isnan(table.loc[1, "diversity"])  # a single counterfactual


def test_desired_class(toy):
    data, instances, counterfactuals = toy
    table = evaluate_counterfactuals(
        instances, counterfactuals, ThresholdModel(), data, **OPTIONS, desired="high", per_instance=True
    )
    # instance 1 is already high: its counterfactual is not valid
    assert table["validity"].iloc[:2].tolist() == pytest.approx([2 / 3, 0])


@pytest.mark.parametrize(
    "desired, expected", [(None, 0.5), ("higher", 0.5), ("lower", 0.0), ((10, None), 0.5), ((None, 5), 0.0)]
)
def test_regression_validity(toy, desired, expected):
    data, instances, _ = toy
    # instance 0 is predicted 4: 11 is valid, 4.1 (2.5% more) is not
    counterfactuals = pd.DataFrame({"instance": [0, 0], "x1": [2, 2], "x2": [9, 2.1], "color": ["red", "red"]})
    summary = evaluate_counterfactuals(
        instances.iloc[:1], counterfactuals, SumModel(), data, mode="regression", desired=desired,
        categorical_features=["color"], k=2,
    )
    assert summary["validity"] == expected


def test_no_counterfactuals(toy):
    data, instances, counterfactuals = toy
    summary = evaluate_counterfactuals(instances, counterfactuals.iloc[:0], ThresholdModel(), data, k=2,
                                       categorical_features=["color"])
    assert summary["coverage"] == 0
    assert summary.drop("coverage").isna().all()


def test_target_column_is_ignored(toy):
    data, instances, counterfactuals = toy
    with_target = data.assign(label=ThresholdModel().predict(data))
    a = evaluate_counterfactuals(instances, counterfactuals, ThresholdModel(), data, **OPTIONS)
    b = evaluate_counterfactuals(instances, counterfactuals, ThresholdModel(), with_target, target="label", **OPTIONS)
    pd.testing.assert_series_equal(a, b)


def test_invalid_inputs(toy):
    data, instances, counterfactuals = toy
    model = ThresholdModel()
    with pytest.raises(ValueError):
        evaluate_counterfactuals(instances, counterfactuals.drop(columns="instance"), model, data, k=2)
    with pytest.raises(ValueError):
        evaluate_counterfactuals(instances, counterfactuals.assign(instance=5), model, data, k=2)
    with pytest.raises(ValueError):
        evaluate_counterfactuals(instances, counterfactuals, model, data, k=len(data))


def test_registry_order():
    assert list(METRICS)[:8] == [
        "validity", "proximity", "changed_features", "redundancy", "implausibility", "ynn", "diversity",
        "constraint_violations",
    ]
    assert METRICS["ynn"].better == "higher" and METRICS["proximity"].better == "lower"


def test_metric_subset(toy):
    data, instances, counterfactuals = toy
    summary = evaluate_counterfactuals(
        instances, counterfactuals, ThresholdModel(), data, categorical_features=["color"], k=2,
        metrics=["proximity", "diversity"],
    )
    assert list(summary.index) == ["coverage", "count", "proximity", "diversity"]
    assert summary["proximity"] == pytest.approx(float(BY_HAND["proximity"]))


def test_unknown_or_inapplicable_metric(toy):
    data, instances, counterfactuals = toy
    with pytest.raises(ValueError):
        evaluate_counterfactuals(instances, counterfactuals, ThresholdModel(), data, k=2, metrics=["accuracy"])
    with pytest.raises(ValueError):  # constraint_violations needs immutable features
        evaluate_counterfactuals(instances, counterfactuals, ThresholdModel(), data, k=2,
                                 metrics=["constraint_violations"])


def test_register_a_new_metric(toy):
    data, instances, counterfactuals = toy

    @register_metric("moved_x1", better="lower", reference="this test")
    def moved_x1(ctx):
        """How much x1 moves."""
        return (ctx.counterfactuals["x1"] - ctx.instances["x1"].to_numpy()[ctx.owner]).abs().to_numpy()

    try:
        summary = evaluate_counterfactuals(instances, counterfactuals, ThresholdModel(), data,
                                           categorical_features=["color"], k=2)
        # instance 0 moves x1 by 6, 0 and 1; instance 1 by 4
        assert summary["moved_x1"] == pytest.approx((7 / 3 + 4) / 2)
        assert METRICS["moved_x1"].description == "How much x1 moves."
    finally:
        del METRICS["moved_x1"]
