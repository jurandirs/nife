"""Base classes of the counterfactual explainers.

A counterfactual method only has to find candidate counterfactuals for one instance
(`CounterfactualExplainer._search`). The base class does everything else, the same
way for every method: it validates the inputs, confirms with the model that each
candidate reaches the counterfactual outcome (another one, or the desired one),
drops those that change an immutable feature, measures their distance to the
instance and their number of changed features, ranks and selects them, explains
many instances at once and evaluates the counterfactuals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..distance import GowerSpace, paired_distances, select_points
from .metrics import evaluate_counterfactuals
from .outcomes import check_desired, check_mode, is_counterfactual
from .selection import select_counterfactuals

RandomState = Optional[Union[int, np.random.RandomState]]


def _plain(value):
    """A numpy scalar as the equivalent Python value (for display)."""
    return value.item() if isinstance(value, np.generic) else value


@dataclass(repr=False)
class CounterfactualExplanation:
    """The counterfactual explanation of one instance.

    Attributes
    ----------
    instance : pd.Series
        The feature values of the instance.
    prediction
        The model's prediction for the instance.
    counterfactuals : pd.DataFrame
        The counterfactuals: features, ``prediction`` (the model's), ``source`` (how
        the method obtained them), ``distance`` (Gower distance to the instance) and
        ``changed_features``; ranked by distance.
    mode : str
        "classification" or "regression".
    desired
        The desired outcome, if any.
    """

    instance: pd.Series
    prediction: Any
    counterfactuals: pd.DataFrame
    mode: str
    desired: Any = None

    def _summary(self) -> List[str]:
        lines = [
            f"{type(self).__name__} ({self.mode}) of an instance predicted {_plain(self.prediction)!r}"
            + (f", desired {self.desired!r}" if self.desired is not None else "")
        ]
        sources = self.counterfactuals["source"].value_counts()
        detail = ", ".join(f"{source}: {count}" for source, count in sources.items())
        lines.append(f"  counterfactuals: {len(self.counterfactuals)}" + (f" ({detail})" if detail else ""))
        if len(self.counterfactuals):
            closest = self.counterfactuals.iloc[0]
            lines.append(
                f"  closest: distance {closest['distance']:.3f}, {closest['changed_features']} changed features, "
                f"predicted {_plain(closest['prediction'])!r}"
            )
        return lines

    def __repr__(self) -> str:
        return "\n".join(self._summary())


class CounterfactualExplainer(ABC):
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
        columns = ["instance", *self.feature_names, "prediction", "source", "distance", "changed_features"]
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
        return self.explanation_type(
            instance=pd.Series(x_values, index=self.feature_names),
            prediction=prediction,
            counterfactuals=counterfactuals,
            mode=self.mode,
            desired=desired,
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
