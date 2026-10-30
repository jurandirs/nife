"""Counterfactual explanations and feature relevance with a surrogate decision tree.

The method of Silva, Aguilar and Salvini (2026):

1. sample a perturbed neighborhood of the instance and label it with the model;
2. fit a decision tree on it — the surrogate, which shows the relevant features
   (those the tree uses, with their importances) and the decision path of the
   instance (the rules);
3. go up the decision path from the leaf until the other side of a decision leads
   to another outcome (Algorithm 1 of the paper): that decision, inverted, is the
   counterfactual rule — a counterfactual rule in the sense of LORE (Guidotti et
   al., 2018), which also explains with a local decision tree.

NIFE then turns the rule into counterfactuals in the sense of the literature (the
smallest change that changes the model's prediction, Wachter et al., 2018) and
supports the rule with evidence from the data:

- **counterfactuals**: the instance with a single feature moved to the other side
  of a decision of the path — the counterfactual rule first — at the closest value
  of the data that the model confirms; when no single feature is enough, the fewest
  features copied from the closest evidence examples (the idea of NICE, Brughmans
  et al., 2024). Each counterfactual also has its counterpart, the closest real
  example with its outcome (see `CounterfactualExplainer`);
- **evidence**: the examples of the data on the other side of the counterfactual
  rule that the model gives a counterfactual outcome — the existing instances with
  which the paper shows that the new outcome is consistent with the data.

NIFE also adds immutable features that the tree never splits on, an adaptive
neighborhood, neighborhoods of your own (``sampler`` and ``neighborhood``) and a
measure of the fidelity of the tree to the model; desired outcomes and the
selection of the counterfactuals come from `CounterfactualExplainer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ...distance import paired_distances, select_points
from ..base import CounterfactualExplainer, CounterfactualExplanation, RandomState
from .search import counterfactuals_from_evidence, counterfactuals_from_rules, search_existing
from .surrogate import (
    DEFAULT_TREE_PARAMS,
    Rule,
    SurrogateTree,
    climb_to_counterfactual,
    decision_path_rules,
    fit_surrogate,
    sample_neighborhood,
    surrogate_fidelity,
)

#: Largest default size of the neighborhood (five times the data, up to this many samples);
#: no dataset of the paper reaches it.
MAX_DEFAULT_SAMPLES = 50_000

#: Neighborhoods tried by the adaptive search, narrowest first:
#: (sample around the instance, scale of the standard deviation).
ADAPTIVE_NEIGHBORHOODS = ((True, 0.25), (True, 0.5), (True, 1.0), (False, 1.0))

#: A function that builds the neighborhood of an instance:
#: ``sampler(instance, data, num_samples, random_state)`` -> DataFrame or 2-D array.
Sampler = Callable[[pd.Series, pd.DataFrame, int, np.random.RandomState], Union[pd.DataFrame, np.ndarray]]

#: A neighborhood given by the user: a DataFrame with the features' columns, or a
#: 2-D array with the features in the order of the data.
Neighborhood = Union[pd.DataFrame, np.ndarray]


@dataclass(repr=False)
class TreeCounterfactualExplanation(CounterfactualExplanation):
    """The explanation of one instance by `TreeCounterfactualExplainer`.

    Besides the attributes of `CounterfactualExplanation` — whose counterfactuals have
    ``source`` ``"rule"`` (a single feature moved to the other side of a decision of
    the path) or ``"evidence"`` (features copied from an evidence example):

    Attributes
    ----------
    rules : list of Rule
        The decision path of the instance in the surrogate tree.
    crossed_rule : Rule or None
        The counterfactual rule, once inverted (``crossed_rule.inverted()``): going up
        the path from the leaf, the first decision (on a mutable feature) whose other
        side the tree gives a counterfactual outcome (Algorithm 1 of Silva et al.,
        2026). None if no decision of the path leads to another outcome.
    crossed_rule_confirmed : bool or None
        Whether the model confirms the counterfactual rule on the instance: whether
        moving only the feature of `crossed_rule` to its other side changes the
        model's prediction (to the desired outcome, if any). None without a rule.
    evidence : pd.DataFrame
        The evidence of the counterfactual rule: the examples of the data on the
        other side of `crossed_rule` (that satisfy the decisions above it) that the
        model gives a counterfactual outcome, with their ``prediction`` and
        ``distance`` to the instance, the closest first; their index is the one they
        have in the data. Without decisions in the tree, every example of the data
        with a counterfactual outcome.
    evidence_candidates : pd.DataFrame
        The candidates for evidence: all the examples of the data on the other side of
        `crossed_rule`, with the model's ``prediction`` and ``is_evidence``; their
        index is the one they have in the data.
    surrogate : DecisionTreeClassifier or DecisionTreeRegressor
        The surrogate tree.
    fidelity : pd.Series
        Agreement between the tree and the model: ``instance_agreement`` (same outcome
        for the instance) and ``local_fidelity`` (fraction of the nearest examples of
        the data with the same outcome), plus the neighborhood used: ``neighborhood``
        (``"sampled"`` by NIFE, ``"sampler"`` for your sampler, ``"given"`` for a
        neighborhood passed to `explain`) and, for a sampled one,
        ``sample_around_instance`` and ``neighborhood_scale``.
    neighborhood : pd.DataFrame
        The perturbed neighborhood on which the surrogate tree was trained, one row per
        point: the features (with the immutable features set to the instance's values)
        and the model's ``prediction``. With the adaptive
        neighborhood, it is the last one tried — the one of `surrogate`. It can be
        passed back to `TreeCounterfactualExplainer.explain` as ``neighborhood``.

    Plots (see `plot_kinds`): ``plot_changes`` (the default, as for every
    counterfactual explanation), ``plot_importances`` and ``plot_tree``.
    """

    rules: List[Rule] = field(default_factory=list)
    crossed_rule: Optional[Rule] = None
    crossed_rule_confirmed: Optional[bool] = None
    evidence: Optional[pd.DataFrame] = None
    evidence_candidates: Optional[pd.DataFrame] = None
    surrogate: Optional[SurrogateTree] = None
    fidelity: Optional[pd.Series] = None
    neighborhood: Optional[pd.DataFrame] = None
    _crossed_index: Optional[int] = None

    SOURCE_NAMES = {"rule": "crossing a decision of the path", "evidence": "copied from evidence examples"}

    @property
    def feature_importances(self) -> pd.Series:
        """Relevance of the features used by the surrogate tree (impurity decrease), highest first."""
        importances = pd.Series(self.surrogate.feature_importances_, index=self.instance.index)
        return importances[importances > 0].sort_values(ascending=False)

    def plot_importances(self, *, max_features: int = 10, ax=None):
        """The relevant features of the prediction, ordered by their importance in the surrogate tree.

        See `nife.counterfactual.tree.plotting.plot_importances`. Returns the matplotlib Axes.
        """
        from .plotting import plot_importances

        return plot_importances(self, max_features=max_features, ax=ax)

    def plot_tree(self, *, path_only: bool = False, precision: int = 4, fontsize: float = 8,
                  titles: bool = True, ax=None):
        """The surrogate tree, compact, with the information of the trees of the paper.

        Decision nodes show their decision and outcome, leaves their outcome; the
        instance's path is blue and the other side of the crossed rule orange.
        ``path_only=True`` draws only the decision path (the smallest figure);
        ``titles=False`` leaves out the titles. See
        `nife.counterfactual.tree.plotting.plot_tree`. Returns the matplotlib Axes.
        """
        from .plotting import plot_tree

        return plot_tree(self, path_only=path_only, precision=precision, fontsize=fontsize, titles=titles, ax=ax)

    def export_tree(self, format: str = "dot", path: Optional[str] = None, *, precision: int = 4) -> str:
        """The surrogate tree as text in an established format, returned and optionally written to `path`.

        ``"dot"``: Graphviz DOT, with the information and colors of `plot_tree`;
        ``"pmml"``: a PMML 4.4 ``TreeModel``, the standard for exchanging models;
        ``"text"``: the rules, as written by `sklearn.tree.export_text`. See
        `nife.counterfactual.tree.export`.
        """
        from .export import export_tree

        return export_tree(self, format, path, precision=precision)

    def _summary(self) -> List[str]:
        lines = [
            self._headline(),
            "  decision path: " + (" and ".join(str(rule) for rule in self.rules) or "(no decisions)"),
            "  relevant features: " + ", ".join(self.feature_importances.index),
        ]
        if self.crossed_rule is not None:
            rule = f"  counterfactual rule: {self.crossed_rule} -> {self.crossed_rule.inverted()}"
            if self.crossed_rule_confirmed:
                rule += " (confirmed by the model)"
            else:
                rule += f" (not confirmed: changing {self.crossed_rule.feature} alone keeps the model's prediction)"
            lines.append(rule)
        else:
            outcome = "the desired outcome" if self.desired is not None else "another outcome"
            reason = f"no decision of the path leads to {outcome}" if self.rules else "the tree has no decisions"
            lines.append(f"  counterfactual rule: none ({reason})")
        lines += self._counterfactual_lines()
        evidence = self.evidence if self.evidence is not None else pd.DataFrame()
        if self.crossed_rule is None and self.rules:
            lines.append("  evidence: none (there is no counterfactual rule)")
        else:
            where = " on the other side of the rule" if self.crossed_rule is not None else " with a counterfactual outcome"
            lines.append(
                f"  evidence: {len(evidence)} example{'' if len(evidence) == 1 else 's'} of the data{where}"
                + (f", the closest at distance {evidence['distance'].iloc[0]:.3f}" if len(evidence) else "")
            )
        source = self.fidelity.get("neighborhood", "sampled")
        if source == "given":
            neighborhood = "given"
        elif source == "sampler":
            neighborhood = "from the sampler"
        elif self.fidelity["sample_around_instance"]:
            neighborhood = f"around the instance, scale {self.fidelity['neighborhood_scale']:g}"
        else:
            neighborhood = "global sample"
        fidelity = (
            f"  fidelity: instance agreement {self.fidelity['instance_agreement']}, "
            f"local fidelity {self.fidelity['local_fidelity']:.2f} (neighborhood: {neighborhood})"
        )
        return [*lines, fidelity]


def fidelity_table(explanations: Sequence[TreeCounterfactualExplanation]) -> pd.DataFrame:
    """The `fidelity` of several explanations, one row per explanation.

    >>> counterfactuals, explanations = explainer.explain_many(instances, return_explanations=True)  # doctest: +SKIP
    >>> fidelity_table(explanations).mean(numeric_only=True)  # doctest: +SKIP
    """
    rows = [explanation.fidelity for explanation in explanations]
    return pd.DataFrame(rows, index=pd.RangeIndex(len(rows), name="instance")).infer_objects()


@dataclass
class _Attempt:
    evidence_candidates: pd.DataFrame
    rules: List[Rule]
    crossed: Optional[int]
    tree: SurrogateTree
    fidelity: pd.Series
    neighborhood: pd.DataFrame
    single: Optional[pd.DataFrame] = None  # single-feature counterfactuals, computed when needed


class TreeCounterfactualExplainer(CounterfactualExplainer):
    """Explain predictions with a surrogate decision tree, counterfactuals and evidence (Silva et al., 2026).

    The explanation of an instance has the relevant features and the decision path of
    the tree, the counterfactual rule (the decision to cross), the counterfactuals —
    the fewest changes that the model confirms — and the evidence of the rule in the
    data; see `TreeCounterfactualExplanation`.

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
        `ADAPTIVE_NEIGHBORHOODS`, from the narrowest, until the tree has a
        counterfactual rule with evidence or a single-feature counterfactual — like the
        expanding spheres of Growing Spheres (Laugel et al., 2017).
    sample_around_instance, neighborhood_scale
        With ``adaptive=False``: center the neighborhood on the instance instead of on
        the mean of the data (the original method), and scale its spread.
    num_samples : int, optional
        Size of the neighborhood; five times the size of `data` by default, at most
        `MAX_DEFAULT_SAMPLES` (so that large data stay fast and light).
    sampler : callable, optional
        Your own way of building the neighborhood, used for every instance instead of
        NIFE's sampling (and of the adaptive neighborhood):
        ``sampler(instance, data, num_samples, random_state)``, with the instance as a
        Series, `data` (features only), `num_samples` and a `numpy.random.RandomState`
        seeded from ``random_state``; it returns a DataFrame with the features' columns
        or a 2-D array with the features in the order of `data`, numeric and encoded as
        `data`. The model labels the points and immutable features are set to the
        instance's values, as with NIFE's sampling.
    max_depth : int
        Maximum depth of the surrogate tree (5, as in Silva et al., 2026).
    min_samples_split, min_samples_leaf, ccp_alpha
        Parameters of the surrogate tree (see scikit-learn's decision trees); the
        defaults come from the original research (the paper keeps scikit-learn's
        defaults: 2, 1 and 0.0).
    tree_params : dict, optional
        Any other parameter of scikit-learn's decision trees.
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
        max_depth: int = DEFAULT_TREE_PARAMS["max_depth"],
        min_samples_split: Union[int, float] = DEFAULT_TREE_PARAMS["min_samples_split"],
        min_samples_leaf: Union[int, float] = DEFAULT_TREE_PARAMS["min_samples_leaf"],
        ccp_alpha: float = DEFAULT_TREE_PARAMS["ccp_alpha"],
        tree_params: Optional[dict] = None,
        fidelity_neighbors: int = 50,
        sampler: Optional[Sampler] = None,
        random_state: RandomState = None,
    ):
        named = dict(
            max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
        )
        repeated = sorted(set(tree_params or {}) & set(named))
        if repeated:
            raise ValueError(f"set {repeated} with their own arguments, not in tree_params")
        if max_depth < 1:
            raise ValueError(f"max_depth must be at least 1, got {max_depth}")
        if sampler is not None and not callable(sampler):
            raise TypeError(f"sampler must be a function, got {type(sampler).__name__}")
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
        self.num_samples = num_samples or min(5 * len(self.data), MAX_DEFAULT_SAMPLES)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.tree_params = tree_params
        self._tree_params = {**(tree_params or {}), **named}
        self.fidelity_neighbors = fidelity_neighbors
        self.sampler = sampler

        self._values = self.data.to_numpy(dtype=np.float64)
        self._values_f32 = self._values.astype(np.float32)
        self._search_data = self.data.assign(prediction=self._data_predictions)
        # The scale of each feature, to compare changes: the inverse of its range (1 for categories)
        scale = pd.Series(1.0, index=self.feature_names)
        scale[self._space.numeric] = self._space.scale
        self._feature_scale = scale
        self._categorical_idx = [self.feature_names.index(f) for f in self.categorical_features]
        self._immutable_idx = [self.feature_names.index(f) for f in self.immutable_features]

    def explain(
        self,
        x,
        *,
        desired=None,
        neighborhood: Optional[Neighborhood] = None,
        max_counterfactuals: Optional[int] = None,
        nondominated: bool = False,
        random_state: RandomState = None,
    ) -> TreeCounterfactualExplanation:
        """Explain the prediction for one instance.

        `neighborhood` is your own neighborhood of the instance, used instead of NIFE's
        sampling and of `sampler`: a DataFrame with the features' columns (other
        columns, such as the ``prediction`` of `TreeCounterfactualExplanation.neighborhood`,
        are ignored) or a 2-D array with the features in the order of the data, numeric
        and encoded as the data. The model labels the points — NIFE explains the model —
        and immutable features are set to the instance's values; a single tree is fitted
        on it.

        The other arguments are those of `CounterfactualExplainer.explain`.
        """
        return super().explain(
            x,
            desired=desired,
            max_counterfactuals=max_counterfactuals,
            nondominated=nondominated,
            random_state=random_state,
            neighborhood=neighborhood,
        )

    def explain_many(
        self,
        instances: pd.DataFrame,
        *,
        desired=None,
        neighborhood: Optional[Neighborhood] = None,
        max_counterfactuals: Optional[int] = None,
        nondominated: bool = False,
        random_state: RandomState = None,
        return_explanations: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[TreeCounterfactualExplanation]]]:
        """Explain several instances; see `CounterfactualExplainer.explain_many` and `explain`.

        A `neighborhood` is used for every instance (a global sample, for example); for
        a neighborhood of its own for each instance, give the explainer a `sampler`.
        """
        return super().explain_many(
            instances,
            desired=desired,
            max_counterfactuals=max_counterfactuals,
            nondominated=nondominated,
            random_state=random_state,
            return_explanations=return_explanations,
            neighborhood=neighborhood,
        )

    def sample_neighborhood(
        self,
        x,
        *,
        around_instance: Optional[bool] = None,
        scale: Optional[float] = None,
        num_samples: Optional[int] = None,
        random_state: RandomState = None,
    ) -> pd.DataFrame:
        """Sample a perturbed neighborhood of `x` with NIFE's sampling, without explaining it.

        Numeric features are sampled from a normal distribution with the standard
        deviation of the data times `scale`, centered on the instance
        (``around_instance=True``) or on the mean of the data; categorical features from
        their frequencies in the data; immutable features keep the instance's values;
        the first row is the instance. `around_instance`, `scale` and `num_samples`
        default to the explainer's `sample_around_instance`, `neighborhood_scale` and
        `num_samples` — the neighborhood of ``adaptive=False``; `ADAPTIVE_NEIGHBORHOODS`
        lists those that the adaptive neighborhood tries.

        Returns the features of the points, which can be changed and given to `explain`
        as ``neighborhood``; it is the model that labels them when explaining.
        """
        x = self._as_frame(x)
        points = self._sampled(
            x,
            self.sample_around_instance if around_instance is None else around_instance,
            self.neighborhood_scale if scale is None else scale,
            self.num_samples if num_samples is None else num_samples,
            self._rng(random_state),
        )
        return pd.DataFrame(points, columns=self.feature_names)

    def _search(
        self, x, prediction, desired, rng, neighborhood: Optional[Neighborhood] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        x_values = x.to_numpy()[0]
        x_points = self._space.points(x)
        if neighborhood is not None:
            plans = [("given", None, None)]
        elif self.sampler is not None:
            plans = [("sampler", None, None)]
        elif self.adaptive:
            plans = [("sampled", around, scale) for around, scale in ADAPTIVE_NEIGHBORHOODS]
        else:
            plans = [("sampled", self.sample_around_instance, self.neighborhood_scale)]
        for source, around, scale in plans:
            if source == "given":
                points = self._checked(neighborhood, "the neighborhood", x)
            elif source == "sampler":
                built = self.sampler(x.iloc[0].copy(), self.data.copy(), self.num_samples, rng)
                points = self._checked(built, "the neighborhood of the sampler", x)
            else:
                points = self._sampled(x, around, scale, self.num_samples, rng)
            attempt = self._attempt(x_values, x_points, prediction, points, source, around, scale, desired)
            # Stop at the first tree with a counterfactual rule and evidence or a single-feature counterfactual
            if attempt.crossed is not None and (
                attempt.evidence_candidates["is_evidence"].any() or len(self._single(attempt, x, desired))
            ):
                break

        # The evidence of the counterfactual rule, the closest first
        found = attempt.evidence_candidates
        evidence = found.loc[found["is_evidence"]].drop(columns="is_evidence")
        x_rows = select_points(x_points, np.zeros(len(evidence), dtype=int))
        evidence["distance"] = paired_distances(
            self._space.points(evidence[self.feature_names]), x_rows, self._space.n_features
        )
        evidence = evidence.sort_values("distance", kind="stable")

        # Counterfactuals: a single feature across a decision of the path; else, the fewest from the evidence
        single = self._single(attempt, x, desired)
        if len(single):
            candidates = single[self.feature_names].assign(source="rule")
        else:
            built = counterfactuals_from_evidence(
                x.iloc[0], evidence, self.model, self.mode, desired, self.tolerance, self.immutable_features,
                scale=self._feature_scale,
            )
            candidates = built[self.feature_names].assign(source="evidence")
        crossed = attempt.crossed
        details = dict(
            rules=attempt.rules,
            crossed_rule=attempt.rules[crossed] if crossed is not None else None,
            crossed_rule_confirmed=bool((single["rule"] == crossed).any()) if crossed is not None else None,
            evidence=evidence,
            evidence_candidates=attempt.evidence_candidates,
            surrogate=attempt.tree,
            fidelity=attempt.fidelity,
            neighborhood=attempt.neighborhood,
            _crossed_index=crossed,
        )
        return candidates, details

    def _single(self, attempt: _Attempt, x, desired) -> pd.DataFrame:
        """The single-feature counterfactuals of an attempt (computed once)."""
        if attempt.single is None:
            attempt.single = counterfactuals_from_rules(
                x.iloc[0], attempt.rules, self.data, self.model, self.mode, desired, self.tolerance,
                self.immutable_features, crossed=attempt.crossed,
            )
        return attempt.single

    def _sampled(self, x, around, scale, num_samples, rng) -> np.ndarray:
        """NIFE's sampling of the neighborhood of `x`, with the immutable features fixed.

        The sampling works in float32, as the tree does; the instance's own values (the
        first row and the immutable features) are kept exact in the float64 result.
        """
        x_values = x.to_numpy(dtype=np.float64)[0]
        points = sample_neighborhood(
            x_values.astype(np.float32), self._values, num_samples, self._categorical_idx, around, scale, rng
        ).astype(np.float64)
        points[0] = x_values
        return self._fix_immutables(points, x_values)

    def _checked(self, neighborhood, what: str, x) -> np.ndarray:
        """A neighborhood given by the user, as the tree's numeric array, with the immutable features fixed."""
        if isinstance(neighborhood, pd.DataFrame):
            missing = [f for f in self.feature_names if f not in neighborhood.columns]
            if missing:
                raise ValueError(f"{what} lacks the features {missing}")
            neighborhood = neighborhood[self.feature_names]
        elif np.ndim(neighborhood) != 2 or np.shape(neighborhood)[1] != len(self.feature_names):
            raise ValueError(
                f"{what} must be a DataFrame with the features' columns or a 2-D array with "
                f"{len(self.feature_names)} columns (the features, in the order of the data), "
                f"got shape {np.shape(neighborhood)}"
            )
        try:
            points = np.array(neighborhood, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{what} must be numeric, encoded as the data") from error
        if len(points) == 0:
            raise ValueError(f"{what} is empty")
        if not np.isfinite(points).all():
            raise ValueError(f"{what} has missing or infinite values")
        return self._fix_immutables(points, x.to_numpy(dtype=np.float64)[0])

    def _fix_immutables(self, points: np.ndarray, x_row: np.ndarray) -> np.ndarray:
        # Immutable features keep the instance's values, so the tree only splits on the others
        points[:, self._immutable_idx] = x_row[self._immutable_idx]
        return points

    def _attempt(self, x_values, x_points, prediction, points, source, around, scale, desired) -> _Attempt:
        """Explain the instance with one neighborhood (`points`, float64, already checked)."""
        x_row = x_values.astype(np.float32).reshape(1, -1)
        # The model and the tree see the neighborhood in float32, as in the original method
        points_f32 = points.astype(np.float32)
        labels = self.model.predict(pd.DataFrame(points_f32, columns=self.feature_names))
        neighborhood = pd.DataFrame(points, columns=self.feature_names).assign(prediction=labels)
        tree = fit_surrogate(points_f32, labels, self.mode, self._tree_params)
        rules = decision_path_rules(tree, x_row, self.feature_names)
        fidelity = surrogate_fidelity(
            tree, x_row, prediction, len(rules), self._values_f32, self._data_predictions, x_points, self._data_points,
            self._space.n_features, self.mode, self.tolerance, self.fidelity_neighbors, around, scale, source,
        )

        # Go up the path until the other side of a decision leads to a counterfactual outcome
        crossed = climb_to_counterfactual(
            tree, x_row, rules, prediction, self.mode, desired, self.tolerance, self.immutable_features
        ) if rules else None
        if rules and crossed is None:
            evidence_candidates = self._search_data.iloc[:0].assign(is_evidence=pd.Series(dtype=bool))
        else:
            evidence_candidates = search_existing(
                self._search_data, rules, crossed, prediction, self.mode, desired, self.tolerance
            )
        return _Attempt(evidence_candidates, rules, crossed, tree, fidelity, neighborhood)
