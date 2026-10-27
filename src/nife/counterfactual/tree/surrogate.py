"""The surrogate decision tree: perturbed neighborhood, tree, decision path and fidelity.

The neighborhood sampling reimplements the mechanism of LIME's
``LimeTabularExplainer`` (Ribeiro et al., 2016) without depending on LIME.
"""

from __future__ import annotations

import operator
from typing import List, NamedTuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ...distance import Points, distance_blocks
from ..outcomes import is_counterfactual

SurrogateTree = Union[DecisionTreeClassifier, DecisionTreeRegressor]

OPERATORS = {"<=": operator.le, ">": operator.gt}

#: Default maximum depth of the surrogate tree (Silva et al., 2026).
DEFAULT_MAX_DEPTH = 5

#: Default parameters of the surrogate tree besides its depth (any parameter of
#: scikit-learn's trees can be passed in ``tree_params``).
DEFAULT_TREE_PARAMS = dict(min_samples_split=0.1, min_samples_leaf=2, ccp_alpha=0.02, random_state=0)


class Rule(NamedTuple):
    """A decision of the surrogate tree: ``feature operator threshold``."""

    feature: str
    operator: str  # "<=" or ">"
    threshold: float

    def __str__(self) -> str:
        return f"{self.feature} {self.operator} {self.threshold:.4g}"

    def holds(self, values):
        """Elementwise: do `values` (of the rule's feature) satisfy the rule?"""
        return OPERATORS[self.operator](values, self.threshold)

    def inverted(self) -> "Rule":
        """The other side of the decision."""
        return Rule(self.feature, ">" if self.operator == "<=" else "<=", self.threshold)


def sample_neighborhood(
    x: np.ndarray,
    data: np.ndarray,
    num_samples: int,
    categorical: Sequence[int] = (),
    around_instance: bool = False,
    scale: float = 1.0,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> np.ndarray:
    """Sample a perturbed neighborhood of `x` (1-D, numeric) from `data` (2-D, numeric).

    Numeric features are sampled as ``Normal(0, 1) * std * scale + center``, with the
    standard deviation of `data`; the center is the mean of `data` (a global sample,
    as in LIME and in the original method) or, with ``around_instance=True``, `x`
    itself. Categorical features (column indices in `categorical`) are sampled from
    their frequencies in `data`. The first row is always `x`, unperturbed.
    """
    rng = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
    data = np.asarray(data, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(data.shape[1])
    std = data.std(axis=0)
    std[std == 0] = 1.0  # avoids zero perturbation on constant features, as scikit-learn does
    center = x if around_instance else data.mean(axis=0)
    neighborhood = rng.normal(0, 1, size=(num_samples, data.shape[1])) * std * scale + center
    for column in categorical:
        values, counts = np.unique(data[:, column], return_counts=True)
        neighborhood[:, column] = rng.choice(values, size=num_samples, replace=True, p=counts / counts.sum())
    neighborhood[0] = x
    return neighborhood.astype(np.float32)


def fit_surrogate(
    X: np.ndarray, y, mode: str, max_depth: int = DEFAULT_MAX_DEPTH, tree_params: Optional[dict] = None
) -> SurrogateTree:
    """Fit the surrogate tree on the neighborhood `X`, labeled `y` by the model."""
    params = {**DEFAULT_TREE_PARAMS, **(tree_params or {}), "max_depth": max_depth}
    tree = DecisionTreeClassifier(**params) if mode == "classification" else DecisionTreeRegressor(**params)
    return tree.fit(X, y)


def decision_path_rules(tree: SurrogateTree, x_row: np.ndarray, feature_names: Sequence[str]) -> List[Rule]:
    """The decisions on the path of `x_row` (a single row, 2-D) from the root to its leaf."""
    nodes = tree.decision_path(x_row).indices
    feature, threshold = tree.tree_.feature, tree.tree_.threshold
    rules = []
    for node in nodes[:-1]:  # the last node is the leaf
        op = "<=" if x_row[0, feature[node]] <= threshold[node] else ">"
        rules.append(Rule(feature_names[feature[node]], op, float(threshold[node])))
    return rules


def node_outcome(tree: SurrogateTree, node: int):
    """The outcome the tree gives at `node`: its majority class, or its mean in regression."""
    value = tree.tree_.value[node]
    if isinstance(tree, DecisionTreeClassifier):
        return tree.classes_[int(np.argmax(value[0]))]
    return float(value[0][0])


def climb_to_counterfactual(
    tree: SurrogateTree,
    x_row: np.ndarray,
    rules: Sequence[Rule],
    prediction,
    mode: str,
    desired=None,
    tolerance: float = 0.05,
    immutable_features: Sequence[str] = (),
) -> Optional[int]:
    """The decision of the path to cross: Algorithm 1 of Silva et al. (2026).

    Starting at the leaf of the instance, go up its decision path: at each node, if
    the tree gives the other side of the parent's decision (the sibling node) an
    outcome that is a counterfactual outcome for the instance's `prediction` (see
    `is_counterfactual`, with `desired`), that decision is the one to cross;
    otherwise the parent becomes the current node. Decisions on immutable features
    are never crossed.

    Returns the position of the decision in `rules`, or None if no decision of the
    path leads to a counterfactual outcome.
    """
    path = tree.decision_path(x_row).indices  # from the root to the leaf
    left, right = tree.tree_.children_left, tree.tree_.children_right
    for depth in range(len(path) - 1, 0, -1):
        node, parent = path[depth], path[depth - 1]
        if rules[depth - 1].feature in immutable_features:
            continue
        sibling = right[parent] if left[parent] == node else left[parent]
        if is_counterfactual(node_outcome(tree, sibling), prediction, mode, desired, tolerance):
            return depth - 1
    return None


def surrogate_fidelity(
    tree: SurrogateTree,
    x_row: np.ndarray,
    prediction,
    n_rules: int,
    reference: np.ndarray,
    reference_predictions: np.ndarray,
    instance_points: Points,
    reference_points: Points,
    n_features: int,
    mode: str,
    tolerance: float,
    n_neighbors: int,
    around_instance: bool,
    scale: float,
) -> pd.Series:
    """Agreement between the tree and the model at the instance and on its nearest reference points.

    "Same outcome" is the complement of a counterfactual outcome: the same class, or
    in regression a prediction within `tolerance`.
    """
    surrogate_prediction = tree.predict(x_row)[0]
    _, distances = next(distance_blocks(instance_points, reference_points, n_features))
    nearest = np.argsort(distances[0], kind="stable")[:n_neighbors]
    same = ~is_counterfactual(tree.predict(reference[nearest]), reference_predictions[nearest], mode, None, tolerance)
    return pd.Series(
        {
            "rules": n_rules,
            "model_prediction": prediction,
            "surrogate_prediction": surrogate_prediction,
            "instance_agreement": not is_counterfactual(surrogate_prediction, prediction, mode, None, tolerance),
            "local_fidelity": float(same.mean()),
            "sample_around_instance": around_instance,
            "neighborhood_scale": scale,
        },
        dtype=object,
    )
