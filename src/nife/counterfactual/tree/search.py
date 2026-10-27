"""Counterfactuals from the decision path of the surrogate tree."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..outcomes import is_counterfactual
from .surrogate import Rule


def search_existing(
    data: pd.DataFrame,
    rules: Sequence[Rule],
    crossed: Optional[int],
    prediction,
    mode: str,
    desired=None,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Examples of `data` on the other side of the decision ``rules[crossed]``.

    `data` holds the features and a ``prediction`` column (the model's predictions).
    The examples kept satisfy the rules before ``rules[crossed]`` and violate it; the
    rules after it are ignored. With ``crossed=None`` every example is kept. The
    ``is_counterfactual`` column flags those whose prediction is a counterfactual
    outcome for the instance's `prediction` (see `nife.counterfactual.is_counterfactual`). The
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
    data["is_counterfactual"] = is_counterfactual(data["prediction"].to_numpy(), prediction, mode, desired, tolerance)
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
) -> pd.DataFrame:
    """Counterfactuals that change a single feature of the decision path.

    For each feature of `rules` that is not immutable (one per feature, the deepest
    rule first), the value of the instance `x` (feature values, indexed by feature
    name) is replaced by each value of that feature observed in `data` on the other
    side of the rule's threshold, and the closest one whose prediction is a
    counterfactual outcome is kept. Unlike counterfactuals taken from the data,
    these are not real examples: they trade plausibility for sparsity.

    Returns one row per feature for which such a value exists, with the features
    (the columns of `data`) and ``prediction``.
    """
    x_df = pd.DataFrame([x[data.columns].to_numpy(dtype=float)], columns=data.columns)
    reference = model.predict(x_df)[0]
    found, seen = [], set(immutable_features)
    for rule in reversed(rules):
        rule = Rule(*rule)
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
            found.append(trials.iloc[[best]].assign(prediction=predictions[best]))
    columns = [*data.columns, "prediction"]
    return pd.concat(found, ignore_index=True)[columns] if found else pd.DataFrame(columns=columns)
