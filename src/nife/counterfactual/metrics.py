"""Evaluation metrics for counterfactual explanations, from any method.

Each instance can have any number of counterfactuals, including none. Every metric
is computed per counterfactual (or per instance), averaged per instance, and then
averaged over the instances that have at least one counterfactual; ``coverage`` and
``count`` describe how many there are. Distances are Gower distances (Gower, 1971),
so numeric and categorical features can be mixed.

The metrics live in the registry `METRICS`. A new metric is a function of an
`EvaluationContext`, registered with `register_metric`::

    @register_metric("my_metric", better="lower", reference="Me (2026)")
    def my_metric(ctx):
        "What it measures."
        return ...  # one value per counterfactual

Metrics
-------
validity
    Fraction of the counterfactuals that reach a counterfactual outcome — an
    outcome different from the instance's or, with `desired`, the desired one
    (Wachter et al., 2018; Mothilal et al., 2020). Higher is better.
proximity
    Mean distance between the instance and its counterfactuals (Mothilal et al.,
    2020). Lower is better.
changed_features
    Mean number of features in which a counterfactual differs from the instance —
    sparsity (Mothilal et al., 2020). Lower is better.
redundancy
    Mean number of changed features that can be reverted, one at a time, to the
    instance's value without losing validity (Pawelczyk et al., 2021). Only valid
    counterfactuals are considered. Lower is better.
implausibility
    Mean distance from each counterfactual to the closest point of the reference
    data (Guidotti, 2022). Lower is better; 0 for counterfactuals taken from the data.
ynn
    Fraction of the `k` nearest reference points of each counterfactual to which the
    model also gives a counterfactual outcome for the instance: whether the
    counterfactual lies in a region of such outcomes. For valid counterfactuals in
    binary classification, this is the yNN of Pawelczyk et al. (2021), who count the
    neighbours that share the counterfactual's class. The counterfactual itself is
    not counted when it belongs to the reference data. Higher is better.
diversity
    Mean distance between the pairs of counterfactuals of the same instance
    (Mothilal et al., 2020); instances with a single counterfactual are left out.
    Higher is better.
constraint_violations
    Fraction of the counterfactuals that change an immutable feature (Pawelczyk et
    al., 2021). Only computed when `immutable_features` is given. Lower is better.

Outcomes
--------
Which predictions are counterfactual outcomes is decided by
`nife.counterfactual.is_counterfactual`, the same criterion the explainers use.
Without `desired`: another class in classification; in regression, a prediction
that deviates from the instance's by more than `tolerance` times it (5% by
default). With `desired`: a given class, a higher or lower prediction, or a
prediction inside an interval.

References
----------
Gower, J. C. (1971). A general coefficient of similarity and some of its
    properties. Biometrics, 27(4), 857-871.
Guidotti, R. (2022). Counterfactual explanations and how to find them: literature
    review and benchmarking. Data Mining and Knowledge Discovery.
    https://doi.org/10.1007/s10618-022-00831-6
Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning
    classifiers through diverse counterfactual explanations. In Proceedings of the
    Conference on Fairness, Accountability, and Transparency (FAT* '20).
    https://doi.org/10.1145/3351095.3372850
Pawelczyk, M., Bielawski, S., van den Heuvel, J., Richter, T., & Kasneci, G.
    (2021). CARLA: A Python library to benchmark algorithmic recourse and
    counterfactual explanation algorithms. NeurIPS Datasets and Benchmarks Track.
Wachter, S., Mittelstadt, B., & Russell, C. (2018). Counterfactual explanations
    without opening the black box: automated decisions and the GDPR. Harvard
    Journal of Law & Technology, 31(2), 841-887.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..distance import GowerSpace, distance_blocks, paired_distances, select_points
from .outcomes import check_desired, is_counterfactual


def _predict(model, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(frame)) if len(frame) else np.array([])


class EvaluationContext:
    """Everything a metric can use, computed once and shared by the metrics.

    Attributes
    ----------
    model, mode, desired, tolerance, k
        As given to `evaluate_counterfactuals`.
    feature_names, immutable_features : list of str
    instances, counterfactuals, reference : pd.DataFrame
        The feature columns of the instances, of the counterfactuals and of the
        reference data.
    owner : np.ndarray
        For each counterfactual, the position of its instance.
    n_instances, n_features : int
    space : GowerSpace
        The Gower distance, with the ranges of the reference data; ``x_points``,
        ``cf_points`` and ``reference_points`` are the points of the three frames.

    The cached properties ``pred_x``, ``pred_cf``, ``pred_reference`` (the model's
    predictions), ``valid`` (whether each counterfactual reaches a counterfactual
    outcome), ``changed`` (which features each one changes) and ``nearest`` (distance
    to the closest reference point and indices of the `k` nearest ones) are computed
    only if a metric uses them.
    """

    def __init__(
        self,
        instances: pd.DataFrame,
        counterfactuals: pd.DataFrame,
        model,
        data: pd.DataFrame,
        *,
        target: Optional[str],
        mode: str,
        categorical_features: Sequence[str],
        immutable_features: Sequence[str],
        desired,
        k: int,
        tolerance: float,
    ):
        check_desired(desired, mode, getattr(model, "classes_", None))
        feature_names = [c for c in data.columns if c != target]
        unknown = sorted((set(categorical_features) | set(immutable_features)) - set(feature_names))
        if unknown:
            raise ValueError(f"unknown features: {unknown}")
        for name, frame in (("instances", instances), ("counterfactuals", counterfactuals)):
            missing = [f for f in feature_names if f not in frame.columns]
            if missing:
                raise ValueError(f"{name} lacks the features {missing}")
        if "instance" not in counterfactuals.columns:
            raise ValueError("counterfactuals needs an 'instance' column with the position of its instance")
        owner = counterfactuals["instance"].to_numpy(dtype=int)
        if len(owner) and (owner.min() < 0 or owner.max() >= len(instances)):
            raise ValueError(f"'instance' must be a position in instances, from 0 to {len(instances) - 1}")
        if not 1 <= k < len(data):
            raise ValueError(f"k must be between 1 and len(data) - 1 = {len(data) - 1}, got {k}")

        self.model, self.mode, self.desired, self.tolerance, self.k = model, mode, desired, tolerance, k
        self.feature_names = feature_names
        self.immutable_features = list(immutable_features)
        self.instances = instances[feature_names].reset_index(drop=True)
        self.counterfactuals = counterfactuals[feature_names].reset_index(drop=True)
        self.reference = data[feature_names].reset_index(drop=True)
        self.owner = owner
        self.n_instances, self.n_features = len(self.instances), len(feature_names)
        self.space = GowerSpace(self.reference, feature_names, categorical_features)
        self.x_points = self.space.points(self.instances)
        self.cf_points = self.space.points(self.counterfactuals)
        self.reference_points = self.space.points(self.reference)

    def reaches(self, prediction, reference_prediction) -> np.ndarray:
        """Elementwise `is_counterfactual` with the evaluation's mode, desired outcome and tolerance."""
        return is_counterfactual(prediction, reference_prediction, self.mode, self.desired, self.tolerance)

    @cached_property
    def pred_x(self) -> np.ndarray:
        return _predict(self.model, self.instances)

    @cached_property
    def pred_cf(self) -> np.ndarray:
        return _predict(self.model, self.counterfactuals)

    @cached_property
    def pred_reference(self) -> np.ndarray:
        return _predict(self.model, self.reference)

    @cached_property
    def valid(self) -> np.ndarray:
        if not len(self.counterfactuals):
            return np.zeros(0, dtype=bool)
        return self.reaches(self.pred_cf, self.pred_x[self.owner])

    @cached_property
    def changed(self) -> np.ndarray:
        return self.counterfactuals.to_numpy() != self.instances.to_numpy()[self.owner]

    @cached_property
    def nearest(self) -> Tuple[np.ndarray, np.ndarray]:
        """(distance to the closest reference point, indices of the `k` nearest reference points)."""
        k = self.k
        closest = np.empty(len(self.counterfactuals))
        neighbours = np.empty((len(self.counterfactuals), k), dtype=int)
        for rows, distances in distance_blocks(self.cf_points, self.reference_points, self.n_features):
            closest[rows] = distances.min(axis=1)
            nearest = np.argpartition(distances, k, axis=1)[:, : k + 1]
            nearest_distances = np.take_along_axis(distances, nearest, axis=1)
            nearest = np.take_along_axis(nearest, np.argsort(nearest_distances, axis=1), axis=1)
            # Leave out the counterfactual itself when it is a point of the reference data
            is_itself = np.sort(nearest_distances, axis=1)[:, :1] == 0
            neighbours[rows] = np.where(is_itself, nearest[:, 1:], nearest[:, :k])
        return closest, neighbours


@dataclass(frozen=True)
class Metric:
    """A registered metric (see `register_metric`)."""

    name: str
    function: Callable[[EvaluationContext], np.ndarray]
    per: str  # "counterfactual" or "instance"
    better: Optional[str]  # "higher", "lower" or None
    reference: str
    description: str
    applies: Callable[[EvaluationContext], bool]


#: The registered metrics, in the order they are reported.
METRICS: Dict[str, Metric] = {}


def register_metric(
    name: str,
    *,
    per: str = "counterfactual",
    better: Optional[str] = None,
    reference: str = "",
    applies: Optional[Callable[[EvaluationContext], bool]] = None,
):
    """Decorator that adds a metric to `METRICS`.

    The function takes an `EvaluationContext` and returns one value per
    counterfactual (``per="counterfactual"``), averaged per instance, or one value
    per instance (``per="instance"``), NaN where undefined. Its docstring's first
    line describes it. `applies` tells whether the metric makes sense for a given
    evaluation (all evaluations by default).
    """
    if per not in ("counterfactual", "instance"):
        raise ValueError(f"per must be 'counterfactual' or 'instance', got {per!r}")

    def decorator(function):
        description = (function.__doc__ or "").strip().split("\n")[0]
        METRICS[name] = Metric(name, function, per, better, reference, description, applies or (lambda ctx: True))
        return function

    return decorator


@register_metric("validity", better="higher", reference="Wachter et al. (2018); Mothilal et al. (2020)")
def validity(ctx: EvaluationContext) -> np.ndarray:
    """Fraction of the counterfactuals that reach a counterfactual outcome."""
    return ctx.valid.astype(float)


@register_metric("proximity", better="lower", reference="Mothilal et al. (2020)")
def proximity(ctx: EvaluationContext) -> np.ndarray:
    """Distance between the instance and its counterfactuals."""
    return paired_distances(ctx.cf_points, select_points(ctx.x_points, ctx.owner), ctx.n_features)


@register_metric("changed_features", better="lower", reference="Mothilal et al. (2020)")
def changed_features(ctx: EvaluationContext) -> np.ndarray:
    """Number of features changed (sparsity)."""
    return ctx.changed.sum(axis=1)


@register_metric("redundancy", better="lower", reference="Pawelczyk et al. (2021)")
def redundancy(ctx: EvaluationContext) -> np.ndarray:
    """Changed features that could be reverted without losing validity."""
    valid, changed, counterfactuals = ctx.valid, ctx.changed, ctx.counterfactuals
    result = np.full(len(counterfactuals), np.nan)
    result[valid] = 0
    reverted, source = [], []
    for j, feature in enumerate(ctx.feature_names):
        rows = np.flatnonzero(valid & changed[:, j])
        if len(rows):
            block = counterfactuals.iloc[rows].copy()
            # Assigning a Series keeps the column's dtype (e.g. categorical)
            block[feature] = ctx.instances[feature].iloc[ctx.owner[rows]].set_axis(block.index)
            reverted.append(block)
            source.append(rows)
    if reverted:
        source = np.concatenate(source)
        still_valid = ctx.reaches(_predict(ctx.model, pd.concat(reverted)), ctx.pred_x[ctx.owner[source]])
        np.add.at(result, source, still_valid)
    return result


@register_metric("implausibility", better="lower", reference="Guidotti (2022)")
def implausibility(ctx: EvaluationContext) -> np.ndarray:
    """Distance to the closest example of the reference data."""
    return ctx.nearest[0]


@register_metric("ynn", better="higher", reference="Pawelczyk et al. (2021)")
def ynn(ctx: EvaluationContext) -> np.ndarray:
    """Fraction of the k nearest examples that also get a counterfactual outcome."""
    neighbours = ctx.nearest[1]
    return ctx.reaches(ctx.pred_reference[neighbours], ctx.pred_x[ctx.owner][:, None]).mean(axis=1)


@register_metric("diversity", per="instance", better="higher", reference="Mothilal et al. (2020)")
def diversity(ctx: EvaluationContext) -> np.ndarray:
    """Distance between the counterfactuals of the same instance."""
    result = np.full(ctx.n_instances, np.nan)
    for i in np.unique(ctx.owner):
        members = np.flatnonzero(ctx.owner == i)
        m = len(members)
        if m > 1:
            group = select_points(ctx.cf_points, members)
            total = sum(block.sum() for _, block in distance_blocks(group, group, ctx.n_features))
            result[i] = total / (m * (m - 1))  # each pair counted twice, the diagonal is 0
    return result


@register_metric(
    "constraint_violations",
    better="lower",
    reference="Pawelczyk et al. (2021)",
    applies=lambda ctx: bool(ctx.immutable_features),
)
def constraint_violations(ctx: EvaluationContext) -> np.ndarray:
    """Fraction of the counterfactuals that change an immutable feature."""
    immutable = [ctx.feature_names.index(f) for f in ctx.immutable_features]
    return ctx.changed[:, immutable].any(axis=1).astype(float)


def evaluate_counterfactuals(
    instances: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    model,
    data: pd.DataFrame,
    *,
    target: Optional[str] = None,
    mode: str = "classification",
    categorical_features: Sequence[str] = (),
    immutable_features: Sequence[str] = (),
    desired=None,
    metrics: Optional[Sequence[str]] = None,
    k: int = 5,
    tolerance: float = 0.05,
    per_instance: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """Evaluate the counterfactuals found for a set of instances, by any method.

    See the module documentation for the definition of each metric. The explainers'
    ``evaluate`` method calls this function with their model, data and options.

    Parameters
    ----------
    instances : pd.DataFrame
        The explained instances, one per row. Columns other than the features are ignored.
    counterfactuals : pd.DataFrame
        The counterfactuals, with an ``instance`` column giving the position (from 0)
        in `instances` of the instance each one explains. Columns other than the
        features and ``instance`` are ignored; outcomes always come from `model`.
    model
        The explained model. Only ``model.predict`` is used, on DataFrames with the
        feature columns.
    data : pd.DataFrame
        Reference data, usually the data the model was trained on. Its columns, except
        `target`, are the features. It gives the ranges of the numeric features and
        the points searched for ``implausibility`` and ``ynn``.
    target : str, optional
        Column of `data` that is not a feature, such as the true target.
    mode : {"classification", "regression"}
        Task of the model.
    categorical_features : sequence of str
        Features compared by equality; all others are numeric.
    immutable_features : sequence of str
        Features a counterfactual should not change; enables ``constraint_violations``.
    desired
        The desired outcome: None (any other outcome), a class, or in regression
        "higher", "lower" or an interval ``(low, high)``. See `is_counterfactual`.
    metrics : sequence of str, optional
        Names of the metrics to compute (see `METRICS`); all that apply by default.
    k : int
        Number of neighbours for ``ynn``.
    tolerance : float
        Regression only: relative deviation above which a prediction is another
        outcome (with ``desired`` None, "higher" or "lower").
    per_instance : bool
        Return the metrics of each instance instead of their summary.

    Returns
    -------
    pd.Series
        ``coverage``, ``count`` and the mean of each metric over the instances that
        have counterfactuals (NaN when a metric is undefined for all of them).
    pd.DataFrame
        With ``per_instance=True``: one row per instance, in the order of `instances`,
        with its number of counterfactuals and its metrics (NaN where undefined).
    """
    ctx = EvaluationContext(
        instances, counterfactuals, model, data, target=target, mode=mode,
        categorical_features=categorical_features, immutable_features=immutable_features,
        desired=desired, k=k, tolerance=tolerance,
    )
    if metrics is None:
        selected = [metric for metric in METRICS.values() if metric.applies(ctx)]
    else:
        unknown = [name for name in metrics if name not in METRICS]
        if unknown:
            raise ValueError(f"unknown metrics {unknown}; available: {list(METRICS)}")
        selected = [METRICS[name] for name in metrics]
        not_applicable = [metric.name for metric in selected if not metric.applies(ctx)]
        if not_applicable:
            raise ValueError(f"the metrics {not_applicable} do not apply to this evaluation")

    per_counterfactual = pd.DataFrame(
        {"instance": ctx.owner, **{m.name: m.function(ctx) for m in selected if m.per == "counterfactual"}}
    )
    means = per_counterfactual.groupby("instance").mean().reindex(range(ctx.n_instances))
    columns = {"counterfactuals": np.bincount(ctx.owner, minlength=ctx.n_instances)}
    for metric in selected:
        columns[metric.name] = means[metric.name].to_numpy() if metric.per == "counterfactual" else metric.function(ctx)
    table = pd.DataFrame(columns, index=pd.RangeIndex(ctx.n_instances, name="instance"))
    if per_instance:
        return table

    explained = table[table["counterfactuals"] > 0]
    summary = pd.Series(
        {
            "coverage": len(explained) / len(table) if len(table) else np.nan,
            "count": explained["counterfactuals"].mean(),
        }
    )
    return pd.concat([summary, explained.drop(columns="counterfactuals").mean()])
