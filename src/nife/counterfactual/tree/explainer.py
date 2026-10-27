"""Counterfactual explanations and feature relevance with a surrogate decision tree.

The method of Silva, Aguilar and Salvini (2026):

1. sample a perturbed neighborhood of the instance and label it with the model;
2. fit a decision tree on it — the surrogate, which shows the relevant features
   (those the tree uses, with their importances) and the decision path of the
   instance (the rules);
3. cross the deepest decision of the path: the examples of the data on its other
   side whose prediction changes are counterfactuals.

NIFE adds single-feature counterfactuals built from the rules, immutable features
that the tree never splits on, an adaptive neighborhood and a measure of the
fidelity of the tree to the model; desired outcomes and the selection of the
counterfactuals come from `CounterfactualExplainer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..base import CounterfactualExplainer, CounterfactualExplanation, RandomState
from .search import counterfactuals_from_rules, search_existing
from .surrogate import Rule, SurrogateTree, decision_path_rules, fit_surrogate, sample_neighborhood, surrogate_fidelity

#: Neighborhoods tried by the adaptive search, narrowest first:
#: (sample around the instance, scale of the standard deviation).
ADAPTIVE_NEIGHBORHOODS = ((True, 0.25), (True, 0.5), (True, 1.0), (False, 1.0))


@dataclass(repr=False)
class TreeCounterfactualExplanation(CounterfactualExplanation):
    """The explanation of one instance by `TreeCounterfactualExplainer`.

    Besides the attributes of `CounterfactualExplanation` (``source`` is ``"data"``
    for examples of the data and ``"rules"`` for counterfactuals built from the
    rules):

    Attributes
    ----------
    rules : list of Rule
        The decision path of the instance in the surrogate tree.
    crossed_rule : Rule or None
        The decision whose other side holds the counterfactuals taken from the data —
        the deepest one on a mutable feature.
    candidates : pd.DataFrame
        The examples of the data on the other side of `crossed_rule`, with the model's
        ``prediction`` and an ``is_counterfactual`` column; their index is the one
        they have in the data.
    surrogate : DecisionTreeClassifier or DecisionTreeRegressor
        The surrogate tree.
    fidelity : pd.Series
        Agreement between the tree and the model: ``instance_agreement`` (same outcome
        for the instance) and ``local_fidelity`` (fraction of the nearest examples of
        the data with the same outcome), plus the neighborhood used.
    """

    rules: List[Rule] = field(default_factory=list)
    crossed_rule: Optional[Rule] = None
    candidates: Optional[pd.DataFrame] = None
    surrogate: Optional[SurrogateTree] = None
    fidelity: Optional[pd.Series] = None
    _crossed_index: Optional[int] = None

    @property
    def feature_importances(self) -> pd.Series:
        """Relevance of the features used by the surrogate tree (impurity decrease), highest first."""
        importances = pd.Series(self.surrogate.feature_importances_, index=self.instance.index)
        return importances[importances > 0].sort_values(ascending=False)

    def plot_tree(self, ax=None, **kwargs):
        """Draw the surrogate tree: the instance's path in green, the other side of the crossed rule in orange.

        Keyword arguments go to `sklearn.tree.plot_tree`. Returns the matplotlib Axes.
        """
        from .plotting import plot_tree

        return plot_tree(self, ax=ax, **kwargs)

    def _summary(self) -> List[str]:
        head, *counts = super()._summary()
        lines = [
            head,
            "  decision path: " + (" and ".join(str(rule) for rule in self.rules) or "(no decisions)"),
            "  relevant features: " + ", ".join(self.feature_importances.index),
        ]
        if self.crossed_rule is not None:
            lines.append(f"  crossed rule: {self.crossed_rule} -> {self.crossed_rule.inverted()}")
        neighborhood = (
            f"around the instance, scale {self.fidelity['neighborhood_scale']:g}"
            if self.fidelity["sample_around_instance"]
            else "global sample"
        )
        fidelity = (
            f"  fidelity: instance agreement {self.fidelity['instance_agreement']}, "
            f"local fidelity {self.fidelity['local_fidelity']:.2f} (neighborhood: {neighborhood})"
        )
        return [*lines, *counts, fidelity]


def fidelity_table(explanations: Sequence[TreeCounterfactualExplanation]) -> pd.DataFrame:
    """The `fidelity` of several explanations, one row per explanation.

    >>> counterfactuals, explanations = explainer.explain_many(instances, return_explanations=True)  # doctest: +SKIP
    >>> fidelity_table(explanations).mean(numeric_only=True)  # doctest: +SKIP
    """
    rows = [explanation.fidelity for explanation in explanations]
    return pd.DataFrame(rows, index=pd.RangeIndex(len(rows), name="instance")).infer_objects()


@dataclass
class _Attempt:
    candidates: pd.DataFrame
    from_rules: pd.DataFrame
    rules: List[Rule]
    crossed: Optional[int]
    tree: SurrogateTree
    fidelity: pd.Series

    def found(self) -> bool:
        return bool(self.rules) and (bool(self.candidates["is_counterfactual"].any()) or len(self.from_rules) > 0)


class TreeCounterfactualExplainer(CounterfactualExplainer):
    """Explain predictions with a surrogate decision tree and counterfactuals (Silva et al., 2026).

    Parameters
    ----------
    model, data, mode, target, categorical_features, tolerance, random_state
        See `CounterfactualExplainer`. Categorical features are sampled from their
        frequencies in the neighborhood; ``random_state`` seeds the neighborhood.
    immutable_features : sequence of str
        Features that cannot change: they keep the instance's values in the
        neighborhood (so the tree only splits on the other features) and in every
        counterfactual.
    adaptive : bool
        Adapt the neighborhood to each instance (default): try the neighborhoods of
        `ADAPTIVE_NEIGHBORHOODS`, from the narrowest, until the tree has rules and a
        counterfactual is found — like the expanding spheres of Growing Spheres
        (Laugel et al., 2017).
    sample_around_instance, neighborhood_scale
        With ``adaptive=False``: center the neighborhood on the instance instead of on
        the mean of the data (the original method), and scale its spread.
    num_samples : int, optional
        Size of the neighborhood; five times the size of `data` by default.
    tree_params : dict, optional
        Parameters of the surrogate tree, over
        `nife.counterfactual.tree.DEFAULT_TREE_PARAMS`.
    fidelity_neighbors : int
        Number of nearest examples of the data on which the local fidelity is measured.
    """

    explanation_type = TreeCounterfactualExplanation

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
        adaptive: bool = True,
        sample_around_instance: bool = False,
        neighborhood_scale: float = 1.0,
        num_samples: Optional[int] = None,
        tree_params: Optional[dict] = None,
        fidelity_neighbors: int = 50,
        random_state: RandomState = None,
    ):
        super().__init__(
            model,
            data,
            mode=mode,
            target=target,
            categorical_features=categorical_features,
            immutable_features=immutable_features,
            tolerance=tolerance,
            random_state=random_state,
        )
        self.adaptive = adaptive
        self.sample_around_instance = sample_around_instance
        self.neighborhood_scale = neighborhood_scale
        self.num_samples = num_samples or 5 * len(self.data)
        self.tree_params = tree_params
        self.fidelity_neighbors = fidelity_neighbors

        self._values = self.data.to_numpy(dtype=np.float64)
        self._values_f32 = self._values.astype(np.float32)
        self._predictions = np.asarray(model.predict(self.data))
        self._search_data = self.data.assign(prediction=self._predictions)
        self._points = self._space.points(self.data)
        self._categorical_idx = [self.feature_names.index(f) for f in self.categorical_features]
        self._immutable_idx = [self.feature_names.index(f) for f in self.immutable_features]

    def explain(
        self,
        x,
        *,
        desired=None,
        from_rules: bool = False,
        max_counterfactuals: Optional[int] = None,
        nondominated: bool = False,
        random_state: RandomState = None,
    ) -> TreeCounterfactualExplanation:
        """Explain the prediction for one instance.

        `from_rules=True` also builds counterfactuals that change a single feature of
        the decision path (``source="rules"``): sparse and close, but not examples of
        the data. The other arguments are those of `CounterfactualExplainer.explain`.
        """
        return super().explain(
            x,
            desired=desired,
            max_counterfactuals=max_counterfactuals,
            nondominated=nondominated,
            random_state=random_state,
            from_rules=from_rules,
        )

    def explain_many(
        self,
        instances: pd.DataFrame,
        *,
        desired=None,
        from_rules: bool = False,
        max_counterfactuals: Optional[int] = None,
        nondominated: bool = False,
        random_state: RandomState = None,
        return_explanations: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[TreeCounterfactualExplanation]]]:
        """Explain several instances; see `CounterfactualExplainer.explain_many` and `explain`."""
        return super().explain_many(
            instances,
            desired=desired,
            max_counterfactuals=max_counterfactuals,
            nondominated=nondominated,
            random_state=random_state,
            return_explanations=return_explanations,
            from_rules=from_rules,
        )

    def _search(self, x, prediction, desired, rng, from_rules: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        x_values = x.to_numpy()[0]
        x_points = self._space.points(x)
        neighborhoods = (
            ADAPTIVE_NEIGHBORHOODS if self.adaptive else ((self.sample_around_instance, self.neighborhood_scale),)
        )
        for around, scale in neighborhoods:
            attempt = self._attempt(x, x_values, x_points, prediction, around, scale, desired, from_rules, rng)
            if attempt.found():
                break

        found = [
            attempt.candidates.loc[attempt.candidates["is_counterfactual"], self.feature_names].assign(source="data"),
            attempt.from_rules[self.feature_names].assign(source="rules"),
        ]
        found = [part for part in found if len(part)]
        candidates = pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=[*self.feature_names, "source"])
        details = dict(
            rules=attempt.rules,
            crossed_rule=attempt.rules[attempt.crossed] if attempt.crossed is not None else None,
            candidates=attempt.candidates,
            surrogate=attempt.tree,
            fidelity=attempt.fidelity,
            _crossed_index=attempt.crossed,
        )
        return candidates, details

    def _attempt(self, x, x_values, x_points, prediction, around, scale, desired, from_rules, rng) -> _Attempt:
        """Explain the instance with one neighborhood."""
        x_row = x_values.astype(np.float32).reshape(1, -1)
        neighborhood = sample_neighborhood(
            x_row[0], self._values, self.num_samples, self._categorical_idx, around, scale, rng
        )
        # Immutable features keep the instance's values, so the tree only splits on the others
        neighborhood[:, self._immutable_idx] = x_row[0, self._immutable_idx]
        labels = self.model.predict(pd.DataFrame(neighborhood, columns=self.feature_names))
        tree = fit_surrogate(neighborhood, labels, self.mode, self.tree_params)
        rules = decision_path_rules(tree, x_row, self.feature_names)
        fidelity = surrogate_fidelity(
            tree, x_row, prediction, len(rules), self._values_f32, self._predictions, x_points, self._points,
            self._space.n_features, self.mode, self.tolerance, self.fidelity_neighbors, around, scale,
        )

        # Cross the deepest decision of the path that is not on an immutable feature
        mutable = [k for k, rule in enumerate(rules) if rule.feature not in self.immutable_features]
        crossed = mutable[-1] if mutable else None
        if rules and crossed is None:
            candidates = self._search_data.iloc[:0].assign(is_counterfactual=pd.Series(dtype=bool))
        else:
            candidates = search_existing(
                self._search_data, rules, crossed, prediction, self.mode, desired, self.tolerance
            )
        if self.immutable_features:
            unchanged = (
                candidates[self.immutable_features].to_numpy() == x[self.immutable_features].to_numpy()
            ).all(axis=1)
            candidates["is_counterfactual"] = candidates["is_counterfactual"].to_numpy() & unchanged

        if from_rules:
            built = counterfactuals_from_rules(
                x.iloc[0], rules, self.data, self.model, self.mode, desired, self.tolerance, self.immutable_features
            )
        else:
            built = pd.DataFrame(columns=[*self.feature_names, "prediction"])
        return _Attempt(candidates, built, rules, crossed, tree, fidelity)
