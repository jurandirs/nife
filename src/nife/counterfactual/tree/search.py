"""Counterfactuals and evidence from the decision path of the surrogate tree.

- `search_existing` — the examples of the data on the other side of the counterfactual
  rule: those that the model gives a counterfactual outcome are its evidence;
- `counterfactuals_from_rules` — counterfactuals that cross one decision of the
  decision path, changing a single feature;
- `counterfactuals_from_evidence` — the fewest features that, copied from the closest
  evidence examples into the instance, change the model's prediction.
"""

from __future__ import annotations

import itertools
import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..outcomes import is_counterfactual
from .surrogate import Rule

#: Evidence examples from which `counterfactuals_from_evidence` copies features.
MAX_EXAMPLES = 5
#: Largest number of features whose every combination it tries.
MAX_CHANGES = 3
#: Largest number of combinations it tries at once.
MAX_TRIALS = 20_000


def search_existing(
    data: pd.DataFrame,
    rules: Sequence[Rule],
    crossed: Optional[int],
    prediction,
    mode: str,
    desired=None,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Examples of `data` on the other side of the decision ``rules[crossed]``: the candidates for evidence.

    `data` holds the features and a ``prediction`` column (the model's predictions).
    The examples kept satisfy the rules before ``rules[crossed]`` and violate it; the
    rules after it are ignored. With ``crossed=None`` every example is kept. The
    ``is_evidence`` column flags those whose prediction is a counterfactual outcome
    for the instance's `prediction` (see `nife.counterfactual.is_counterfactual`): the
    evidence that the other side of the decision leads to another outcome. The
    examples keep their index in `data`, to look them up there.
    """
    if crossed is not None:
        mask = np.ones(len(data), dtype=bool)
        for rule in rules[:crossed]:
            mask &= rule.holds(data[rule.feature]).to_numpy()
        crossed_rule = rules[crossed]
        mask &= ~crossed_rule.holds(data[crossed_rule.feature]).to_numpy()
        data = data.loc[mask]
    data = data.copy()
    data["is_evidence"] = is_counterfactual(data["prediction"].to_numpy(), prediction, mode, desired, tolerance)
    return data


def counterfactuals_from_rules(
    x: pd.Series,
    rules: Sequence[Rule],
    data: pd.DataFrame,
    model,
    mode: str = "classification",
    desired=None,
    tolerance: float = 0.05,
    immutable_features: Sequence[str] = (),
    crossed: Optional[int] = None,
) -> pd.DataFrame:
    """Counterfactuals that cross one decision of the decision path, changing a single feature.

    The decision ``rules[crossed]`` (the counterfactual rule), if given, is tried
    first, then the other decisions of the path from the deepest up, one per feature;
    immutable features are skipped. For each decision, the value of the instance `x`
    (feature values, indexed by feature name) is replaced by each value of that feature
    observed in `data` on the other side of the decision's threshold, and the closest
    one whose prediction is a counterfactual outcome is kept.

    Returns one row per decision for which such a value exists, with the features (the
    columns of `data`), ``prediction`` and ``rule``, the position of the decision in
    `rules`.
    """
    x_df = pd.DataFrame([x[data.columns].to_numpy(dtype=float)], columns=data.columns)
    reference = model.predict(x_df)[0]
    order = list(reversed(range(len(rules))))
    if crossed is not None:
        order.remove(crossed)
        order.insert(0, crossed)
    found, seen = [], set(immutable_features)
    for position in order:
        rule = Rule(*rules[position])
        if rule.feature in seen:
            continue
        seen.add(rule.feature)
        values = np.unique(data[rule.feature].to_numpy(dtype=float))
        other_side = values[~rule.holds(values)]
        if not len(other_side):
            continue
        trials = x_df.loc[np.zeros(len(other_side), dtype=int)].reset_index(drop=True)
        trials[rule.feature] = other_side
        predictions = np.asarray(model.predict(trials))
        valid = np.flatnonzero(is_counterfactual(predictions, reference, mode, desired, tolerance))
        if len(valid):
            best = valid[np.argmin(np.abs(other_side[valid] - x_df.at[0, rule.feature]))]
            found.append(trials.iloc[[best]].assign(prediction=predictions[best], rule=position))
    columns = [*data.columns, "prediction", "rule"]
    return pd.concat(found, ignore_index=True)[columns] if found else pd.DataFrame(columns=columns)


def counterfactuals_from_evidence(
    x: pd.Series,
    evidence: pd.DataFrame,
    model,
    mode: str = "classification",
    desired=None,
    tolerance: float = 0.05,
    immutable_features: Sequence[str] = (),
    scale: Optional[pd.Series] = None,
    max_examples: int = MAX_EXAMPLES,
    max_changes: int = MAX_CHANGES,
    max_trials: int = MAX_TRIALS,
) -> pd.DataFrame:
    """The fewest features that, copied from the closest evidence examples, change the prediction.

    For each of the first `max_examples` rows of `evidence` (real examples with a
    counterfactual outcome, the closest to the instance first), every set of 1, 2, ...
    up to `max_changes` of the mutable features in which the example differs from the
    instance `x` is copied into `x`, and the smallest sets whose prediction is a
    counterfactual outcome are kept — the idea of NICE (Brughmans, Leyman and Martens,
    2024): sparse counterfactuals with the values of real examples.

    If no set that small works (or there are too many sets to try, more than
    `max_trials`), features are copied from the closest example that allows it, the
    largest changes first — measured with `scale`, the inverse of each feature's range
    (1 by default) — until the prediction changes; then the features that are not
    needed are reverted, the smallest changes first.

    Returns the counterfactuals found, without duplicates: the features (those of `x`)
    and ``prediction``.
    """
    features = list(x.index)
    columns = [*features, "prediction"]
    x_values = x.to_numpy(dtype=float)
    reference = model.predict(pd.DataFrame([x_values], columns=features))[0]
    mutable = [f for f in features if f not in set(immutable_features)]
    examples = evidence[features].head(max_examples)

    def changed(example):
        return [f for f in mutable if example[f] != x[f]]

    def predict(values: np.ndarray):
        predictions = np.asarray(model.predict(pd.DataFrame(values, columns=features)))
        return predictions, is_counterfactual(predictions, reference, mode, desired, tolerance)

    # The smallest sets of features, every combination of 1, 2, ... features
    found, smallest = [], None
    for _, example in examples.iterrows():
        differ = changed(example)
        positions = [features.index(f) for f in differ]
        values = example[differ].to_numpy(dtype=float)
        for size in range(1, min(smallest or max_changes, len(differ)) + 1):
            if math.comb(len(differ), size) > max_trials:
                break
            combos = list(itertools.combinations(range(len(differ)), size))
            trials = np.repeat(x_values[None, :], len(combos), axis=0)
            for row, combo in enumerate(combos):
                for j in combo:
                    trials[row, positions[j]] = values[j]
            predictions, valid = predict(trials)
            if valid.any():
                if smallest is None or size < smallest:
                    found, smallest = [], size
                found.append(pd.DataFrame(trials[valid], columns=features).assign(prediction=predictions[valid]))
                break
    if found:
        return pd.concat(found, ignore_index=True).drop_duplicates(subset=features).reset_index(drop=True)[columns]

    # Larger sets: copy the largest changes first, then revert what is not needed
    weights = pd.Series(1.0, index=features) if scale is None else scale.reindex(features).fillna(1.0)
    for _, example in examples.iterrows():
        differ = changed(example)
        if not differ:
            continue
        size = {f: abs(float(example[f]) - float(x[f])) * weights[f] for f in differ}
        trial, chosen = x_values.copy(), []
        for feature in sorted(differ, key=lambda f: -size[f]):
            trial[features.index(feature)] = example[feature]
            chosen.append(feature)
            prediction, valid = predict(trial[None, :])
            if valid[0]:
                break
        else:
            continue  # not even every mutable feature of this example changes the prediction
        for feature in sorted(chosen, key=lambda f: size[f]):
            if len(chosen) == 1:
                break
            reverted = trial.copy()
            reverted[features.index(feature)] = x_values[features.index(feature)]
            reverted_prediction, valid = predict(reverted[None, :])
            if valid[0]:
                trial, prediction = reverted, reverted_prediction
                chosen.remove(feature)
        return pd.DataFrame([trial], columns=features).assign(prediction=prediction[0])[columns]
    return pd.DataFrame(columns=columns)
