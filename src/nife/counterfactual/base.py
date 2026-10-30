"""Base classes of the counterfactual explainers.

A counterfactual explanation of a prediction describes the smallest change to the
feature values that changes the prediction to another outcome, or to a desired one
(Wachter et al., 2018; Molnar, 2022): a counterfactual is an instance close to the
one explained, with as few changed features as possible, that the model gives that
outcome.

A counterfactual method only has to find candidate counterfactuals for one instance
(`CounterfactualExplainer._search`). The base class does everything else, the same
way for every method: it validates the inputs, confirms with the model that each
candidate reaches the counterfactual outcome (another one, or the desired one),
drops those that change an immutable feature, measures their distance to the
instance and their number of changed features, ranks and selects them, finds their
counterparts, explains many instances at once and evaluates the counterfactuals.

The counterpart of a counterfactual is the example of the reference data closest to
it that the model gives the same outcome: a real case that shows the counterfactual
is realistic. The closer it is, the more plausible the counterfactual (the likely
feature values of Dandl et al., 2020). It does not confirm the counterfactual — the
model does — but supports it with the data.

Plots come from `nife.base`: every counterfactual explanation draws ``plot_changes``
(what a counterfactual changes), and a method adds its own by defining more
``plot_<kind>`` methods on its explanation type.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..base import Explainer, Explanation
from ..distance import GowerSpace, distance_blocks, paired_distances, select_points
from . import plots
from .metrics import evaluate_counterfactuals
from .outcomes import check_desired, check_mode, is_counterfactual
from .selection import select_counterfactuals

RandomState = Optional[Union[int, np.random.RandomState]]


def _plain(value):
    """A numpy scalar as the equivalent Python value (for display)."""
    return value.item() if isinstance(value, np.generic) else value


@dataclass(repr=False)
class CounterfactualExplanation(Explanation):
    """The counterfactual explanation of one instance.

    Attributes
    ----------
    instance : pd.Series
        The feature values of the instance.
    prediction
        The model's prediction for the instance.
    counterfactuals : pd.DataFrame
        The counterfactuals — instances close to this one, with few changed features,
        that the model gives a counterfactual outcome: the features, ``prediction``
        (the model's), ``source`` (how the method obtained them), ``distance`` (Gower
        distance to the instance), ``changed_features``, ``counterpart`` (the index, in
        the reference data, of the closest example that the model gives the same
        outcome) and ``counterpart_distance`` (its Gower distance to the
        counterfactual); ranked by distance.
    mode : str
        "classification" or "regression".
    desired
        The desired outcome, if any.
    reference_data : pd.DataFrame
        The explainer's reference data (not a copy), on whose scale the plots measure
        the changes.
    categorical_features : list of str
        The explainer's categorical features.
    tolerance : float
        The explainer's tolerance (regression).

    Plots (see `plot_kinds`): ``plot_changes``, the default of ``plot()``.
    """

    instance: pd.Series
    prediction: Any
    counterfactuals: pd.DataFrame
    mode: str
    desired: Any = None
    reference_data: Optional[pd.DataFrame] = None
    categorical_features: List[str] = field(default_factory=list)
    tolerance: float = 0.05

    default_plot = "changes"

    def plot_changes(self, counterfactual: int = 0, *, max_features: int = 10, ax=None):
        """What one counterfactual changes: each feature, from the instance's value to the counterfactual's.

        `counterfactual` is its position in `counterfactuals` (0, the closest, by
        default). Each line spans the range of the feature in the reference data. See
        `nife.counterfactual.plots.plot_changes`. Returns the matplotlib Axes.
        """
        self._check_plottable()
        if not -len(self.counterfactuals) <= counterfactual < len(self.counterfactuals):
            raise IndexError(f"there are {len(self.counterfactuals)} counterfactuals, no counterfactual {counterfactual}")
        chosen = self.counterfactuals.iloc[counterfactual]
        position = counterfactual % len(self.counterfactuals)
        subtitle = (
            f"Counterfactual {position + 1} of {len(self.counterfactuals)}: predicted "
            f"{plots.format_value(_plain(chosen['prediction']))} instead of {plots.format_value(_plain(self.prediction))}"
            f" · distance {chosen['distance']:.3f} · {chosen['changed_features']} changed "
            f"feature{'' if chosen['changed_features'] == 1 else 's'}"
        )
        if pd.notna(chosen.get("counterpart_distance", np.nan)):
            subtitle += f" · counterpart at distance {chosen['counterpart_distance']:.3f}"
        return plots.plot_changes(
            self.instance, chosen, self.reference_data, categorical_features=self.categorical_features,
            max_features=max_features, ax=ax, subtitle=subtitle,
        )

    def _check_plottable(self):
        if not len(self.counterfactuals):
            raise ValueError("this explanation has no counterfactuals to plot")
        if self.reference_data is None:
            raise ValueError("this explanation has no reference_data, on whose scale the changes are drawn")

    def _summary(self) -> List[str]:
        return [self._headline(), *self._counterfactual_lines()]

    def _headline(self) -> str:
        return (
            f"{type(self).__name__} ({self.mode}) of an instance predicted {_plain(self.prediction)!r}"
            + (f", desired {self.desired!r}" if self.desired is not None else "")
        )

    #: How the summary names the ``source`` of the counterfactuals (others are shown as they are).
    SOURCE_NAMES: ClassVar[Dict[str, str]] = {}

    def _counterfactual_lines(self) -> List[str]:
        counterfactuals = self.counterfactuals
        line = f"  counterfactuals: {len(counterfactuals)}"
        if len(counterfactuals):
            changes = sorted(set(int(c) for c in counterfactuals["changed_features"]))
            counted = f"{changes[0]}" if len(changes) == 1 else f"{changes[0]} to {changes[-1]}"
            line += f", changing {counted} feature{'' if changes[-1] == 1 else 's'}"
            line += " each" if len(counterfactuals) > 1 and len(changes) == 1 else ""
            sources = counterfactuals["source"].value_counts()
            line += " (" + ", ".join(
                f"{count} {self.SOURCE_NAMES[source]}" if source in self.SOURCE_NAMES else f"{source}: {count}"
                for source, count in sources.items()
            ) + ")"
        lines = [line]
        if len(counterfactuals):
            closest = counterfactuals.iloc[0]
            counterpart = closest.get("counterpart_distance", np.nan)
            lines.append(
                f"  closest: {self._describe_changes(closest)}; predicted {_plain(closest['prediction'])!r}"
                f" (distance {closest['distance']:.3f}"
                + (f"; counterpart at distance {counterpart:.3f})" if pd.notna(counterpart) else ")")
            )
        return lines

    def _describe_changes(self, counterfactual: pd.Series, most: int = 3) -> str:
        """``feature old -> new`` for the features that `counterfactual` changes (at most `most`)."""
        changed = [f for f in self.instance.index if counterfactual[f] != self.instance[f]]
        text = ", ".join(
            f"{f} {plots.format_value(_plain(self.instance[f]))} -> {plots.format_value(_plain(counterfactual[f]))}"
            for f in changed[:most]
        )
        return text + (f" and {len(changed) - most} more" if len(changed) > most else "")

    def __repr__(self) -> str:
        return "\n".join(self._summary())


def _check_predictions_match_mode(predictions: np.ndarray, mode: str) -> None:
    """Raise a clear error when the model's predictions do not fit `mode` (a regressor without mode="regression")."""
    if mode == "classification" and predictions.dtype.kind == "f" and not np.all(np.mod(predictions, 1) == 0):
        raise ValueError(
            "the model's predictions are continuous but mode is 'classification'; "
            "for a regression model, pass mode='regression'"
        )
    if mode == "regression" and predictions.dtype.kind not in "biuf":
        raise ValueError(
            "the model's predictions are not numbers but mode is 'regression'; "
            "for a classification model, use mode='classification' (the default)"
        )


class CounterfactualExplainer(Explainer):
    """Base class of the counterfactual explainers.

    Parameters
    ----------
    model
        The model to explain. Only ``model.predict`` is used, on DataFrames with the
        feature columns; ``model.classes_``, if present, validates desired classes.
    data : pd.DataFrame
        Reference data, usually the data the model was trained on. All features must
        be numeric (encode categorical features as numbers and list them in
        `categorical_features`).
    mode : {"classification", "regression"}
    target : str, optional
        A column of `data` that is not a feature, such as the true target.
    categorical_features : sequence of str
        Features compared by equality (and treated as categories by the method).
    immutable_features : sequence of str
        Features that cannot change: no counterfactual changes them.
    tolerance : float
        Regression only: relative change of the prediction that counts as a different
        outcome (see `is_counterfactual`).
    random_state : int or numpy.random.RandomState, optional
        Seed of the method's randomness. With an int, every call to `explain` or
        `explain_many` starts from it and is reproducible.

    Subclasses implement `_search` and set `explanation_type`.
    """

    #: The class of the explanations returned by `explain`.
    explanation_type = CounterfactualExplanation

    #: The columns of the counterfactuals, after the features.
    COLUMNS = ("prediction", "source", "distance", "changed_features", "counterpart", "counterpart_distance")

    def __init__(
        self,
        model,
        data: pd.DataFrame,
        *,
        mode: str = "classification",
        target: Optional[str] = None,
        categorical_features: Sequence[str] = (),
        immutable_features: Sequence[str] = (),
        tolerance: float = 0.05,
        random_state: RandomState = None,
    ):
        check_mode(mode)
        if not hasattr(model, "predict"):
            raise TypeError("model must have a predict method")
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas DataFrame of numeric features, got {type(data).__name__}; "
                "for an array, use pd.DataFrame(array, columns=feature_names)"
            )
        self.model = model
        self.mode = mode
        self.data = data.drop(columns=[target]) if target is not None else data
        self.feature_names = list(self.data.columns)
        not_numeric = [f for f in self.feature_names if not pd.api.types.is_numeric_dtype(self.data[f])]
        if not_numeric:
            raise ValueError(f"features must be numeric; encode {not_numeric} as numbers (see categorical_features)")
        for name, features in (("categorical", categorical_features), ("immutable", immutable_features)):
            unknown = sorted(set(features) - set(self.feature_names))
            if unknown:
                raise ValueError(f"unknown {name} features: {unknown}")
        self.categorical_features = list(categorical_features)
        self.immutable_features = list(immutable_features)
        self.tolerance = tolerance
        self.random_state = random_state
        self._space = GowerSpace(self.data, self.feature_names, self.categorical_features)
        # The model's predictions for the reference data and its points, to find the counterparts
        self._data_predictions = np.asarray(model.predict(self.data))
        _check_predictions_match_mode(self._data_predictions, mode)
        self._data_points = self._space.points(self.data)

    # --- to implement ---------------------------------------------------------------

    @abstractmethod
    def _search(
        self, x: pd.DataFrame, prediction, desired, rng: np.random.RandomState, **options
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Find candidate counterfactuals for one instance.

        Parameters
        ----------
        x : pd.DataFrame
            The instance: one row, the feature columns, as floats.
        prediction
            The model's prediction for `x`.
        desired
            The desired outcome (already validated), or None.
        rng : numpy.random.RandomState
            The source of randomness of this call.
        **options
            The method's own options, passed to `explain` and `explain_many`.

        Returns
        -------
        candidates : pd.DataFrame
            Candidate counterfactuals: the feature columns and ``source``, a label of
            how they were obtained. The base class keeps those that the model confirms
            and that keep the immutable features.
        details : dict
            Additional fields of the method's explanation type.
        """

    # --- public API -----------------------------------------------------------------

    def explain(
        self,
        x,
        *,
        desired=None,
        max_counterfactuals: Optional[int] = None,
        nondominated: bool = False,
        random_state: RandomState = None,
        **options,
    ) -> CounterfactualExplanation:
        """Explain the prediction for one instance.

        Parameters
        ----------
        x : pd.Series, dict, one-row pd.DataFrame or array-like
            The instance (array-likes follow the order of the features).
        desired
            The desired outcome: None (any other outcome), a class, or in regression
            "higher", "lower" or an interval ``(low, high)`` where either bound may be
            None. See `is_counterfactual`.
        max_counterfactuals, nondominated
            Keep only the best counterfactuals (see `select_counterfactuals`).
        random_state
            Overrides the explainer's `random_state` for this call.
        **options
            Options of the method.
        """
        check_desired(desired, self.mode, getattr(self.model, "classes_", None))
        return self._explain(self._as_frame(x), desired, max_counterfactuals, nondominated,
                             self._rng(random_state), options)

    def explain_many(
        self,
        instances: pd.DataFrame,
        *,
        desired=None,
        max_counterfactuals: Optional[int] = None,
        nondominated: bool = False,
        random_state: RandomState = None,
        return_explanations: bool = False,
        **options,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[CounterfactualExplanation]]]:
        """Explain several instances (the rows of `instances`; other columns are ignored).

        Returns the counterfactuals of all of them, in the format of
        `evaluate_counterfactuals`: an ``instance`` column (the position of the
        instance in `instances`) and the columns of the explanations'
        ``counterfactuals``. With ``return_explanations=True``, returns
        ``(counterfactuals, explanations)``, with the explanation of each instance.
        The other arguments are those of `explain`.
        """
        check_desired(desired, self.mode, getattr(self.model, "classes_", None))
        rng = self._rng(random_state)
        rows = instances[self.feature_names].reset_index(drop=True)
        explanations = [
            self._explain(rows.iloc[[i]], desired, max_counterfactuals, nondominated, rng, options)
            for i in range(len(rows))
        ]
        columns = ["instance", *self.feature_names, *self.COLUMNS]
        found = [e.counterfactuals.assign(instance=i) for i, e in enumerate(explanations) if len(e.counterfactuals)]
        counterfactuals = pd.concat(found, ignore_index=True)[columns] if found else pd.DataFrame(columns=columns)
        return (counterfactuals, explanations) if return_explanations else counterfactuals

    def evaluate(
        self,
        instances: pd.DataFrame,
        counterfactuals: pd.DataFrame,
        *,
        desired=None,
        metrics: Optional[Sequence[str]] = None,
        k: int = 5,
        per_instance: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """`evaluate_counterfactuals` with the explainer's model, data and options."""
        return evaluate_counterfactuals(
            instances,
            counterfactuals,
            self.model,
            self.data,
            mode=self.mode,
            categorical_features=self.categorical_features,
            immutable_features=self.immutable_features,
            desired=desired,
            metrics=metrics,
            k=k,
            tolerance=self.tolerance,
            per_instance=per_instance,
        )

    # --- internals ------------------------------------------------------------------

    def _rng(self, random_state) -> np.random.RandomState:
        seed = self.random_state if random_state is None else random_state
        return seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)

    def _as_frame(self, x) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            if len(x) != 1:
                raise ValueError("explain takes a single instance; use explain_many for several")
            return x[self.feature_names]
        if isinstance(x, (pd.Series, dict)):
            return pd.DataFrame([pd.Series(x)[self.feature_names]])
        values = np.asarray(x, dtype=float).reshape(-1)
        if len(values) != len(self.feature_names):
            raise ValueError(f"x has {len(values)} values, but there are {len(self.feature_names)} features")
        return pd.DataFrame([values], columns=self.feature_names)

    def _explain(self, x, desired, max_counterfactuals, nondominated, rng, options) -> CounterfactualExplanation:
        x = pd.DataFrame(x.to_numpy(dtype=float), columns=self.feature_names)
        x_values = x.to_numpy()[0]
        prediction = _plain(np.asarray(self.model.predict(x))[0])
        candidates, details = self._search(x, prediction, desired, rng, **options)
        counterfactuals = self._confirm(candidates, x_values, prediction, desired)
        counterfactuals = select_counterfactuals(counterfactuals, max_counterfactuals, nondominated)
        counterfactuals = self._add_counterparts(counterfactuals, prediction, desired)
        return self.explanation_type(
            instance=pd.Series(x_values, index=self.feature_names),
            prediction=prediction,
            counterfactuals=counterfactuals,
            mode=self.mode,
            desired=desired,
            reference_data=self.data,
            categorical_features=self.categorical_features,
            tolerance=self.tolerance,
            **details,
        )

    def _confirm(self, candidates: pd.DataFrame, x_values: np.ndarray, prediction, desired) -> pd.DataFrame:
        """Keep the candidates that reach a counterfactual outcome and keep the immutable features."""
        columns = [*self.feature_names, "prediction", "source"]
        if not len(candidates):
            counterfactuals = pd.DataFrame(columns=columns)
        else:
            counterfactuals = candidates[[*self.feature_names, "source"]].reset_index(drop=True)
            counterfactuals.insert(len(self.feature_names), "prediction", self.model.predict(counterfactuals[self.feature_names]))
            keep = is_counterfactual(counterfactuals["prediction"].to_numpy(), prediction, self.mode, desired, self.tolerance)
            if self.immutable_features:
                immutable = [self.feature_names.index(f) for f in self.immutable_features]
                keep &= (counterfactuals[self.immutable_features].to_numpy() == x_values[immutable]).all(axis=1)
            counterfactuals = counterfactuals.loc[keep].reset_index(drop=True)
        instance_points = select_points(self._space.points(pd.DataFrame([x_values], columns=self.feature_names)),
                                        np.zeros(len(counterfactuals), dtype=int))
        counterfactuals["distance"] = paired_distances(
            self._space.points(counterfactuals), instance_points, self._space.n_features
        )
        counterfactuals["changed_features"] = (counterfactuals[self.feature_names].to_numpy() != x_values).sum(axis=1)
        return counterfactuals

    def _add_counterparts(self, counterfactuals: pd.DataFrame, prediction, desired) -> pd.DataFrame:
        """For each counterfactual, its counterpart: the closest example of the reference data with the same outcome.

        The same outcome is the same class or, in regression, a counterfactual outcome
        for the instance on the same side of its prediction. ``counterpart`` is the
        example's index in the data and ``counterpart_distance`` its distance to the
        counterfactual; both are missing when no example qualifies.
        """
        n = len(counterfactuals)
        index = [None] * n
        distance = np.full(n, np.nan)
        if n:
            reaches = is_counterfactual(self._data_predictions, prediction, self.mode, desired, self.tolerance)
            outcomes = counterfactuals["prediction"].to_numpy()
            points = self._space.points(counterfactuals[self.feature_names])
            for rows, distances in distance_blocks(points, self._data_points, self._space.n_features):
                for offset, row in enumerate(distances):
                    i = rows.start + offset
                    if self.mode == "classification":
                        same = reaches & (self._data_predictions == outcomes[i])
                    else:
                        side = np.sign(np.asarray(outcomes[i], dtype=float) - float(prediction))
                        same = reaches & (np.sign(self._data_predictions.astype(float) - float(prediction)) == side)
                    if same.any():
                        j = np.flatnonzero(same)[np.argmin(row[same])]
                        index[i], distance[i] = self.data.index[j], row[j]
        counterfactuals = counterfactuals.copy()
        integer_index = pd.api.types.is_integer_dtype(self.data.index)
        counterfactuals["counterpart"] = pd.array(index, dtype="Int64") if integer_index else pd.Series(index, dtype=object)
        counterfactuals["counterpart_distance"] = distance
        return counterfactuals
