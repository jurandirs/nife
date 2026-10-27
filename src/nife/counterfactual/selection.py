"""Ranking and selection of the counterfactuals of each instance."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def select_counterfactuals(
    counterfactuals: pd.DataFrame,
    max_counterfactuals: Optional[int] = None,
    nondominated: bool = False,
) -> pd.DataFrame:
    """Rank the counterfactuals of each instance and keep the best ones.

    Uses the ``distance`` (to the instance) and ``changed_features`` columns that
    NIFE adds to every counterfactual. Counterfactuals are ranked by distance, then
    by number of changed features. With ``nondominated=True``, only those that no
    other counterfactual of the same instance dominates are kept — at least as
    close and with at most as many changed features, and strictly better in one of
    them: the Pareto front over proximity and sparsity (Dandl et al., 2020). Then at
    most `max_counterfactuals` per instance are kept. Instances are told apart by
    the ``instance`` column, if there is one.

    Dandl, S., Molnar, C., Binder, M., & Bischl, B. (2020). Multi-objective
    counterfactual explanations. Parallel Problem Solving from Nature (PPSN XVI).
    """
    missing = sorted({"distance", "changed_features"} - set(counterfactuals.columns))
    if missing:
        raise ValueError(f"counterfactuals lacks the columns {missing}, added by the explainer")
    if max_counterfactuals is not None and max_counterfactuals < 1:
        raise ValueError(f"max_counterfactuals must be at least 1, got {max_counterfactuals}")

    by = ["instance"] if "instance" in counterfactuals.columns else []
    ranked = counterfactuals.sort_values([*by, "distance", "changed_features"], kind="stable")
    if nondominated:
        distance = ranked["distance"].to_numpy()
        changed = ranked["changed_features"].to_numpy()
        groups = ranked.groupby(by, sort=False).indices.values() if by else [np.arange(len(ranked))]
        keep = np.zeros(len(ranked), dtype=bool)
        for rows in groups:
            d, c = distance[rows], changed[rows]
            # dominated[i]: some j is no worse than i in both and better in one
            no_worse = (d[None, :] <= d[:, None]) & (c[None, :] <= c[:, None])
            better = (d[None, :] < d[:, None]) | (c[None, :] < c[:, None])
            keep[rows] = ~(no_worse & better).any(axis=1)
        ranked = ranked[keep]
    if max_counterfactuals is not None:
        ranked = ranked.groupby(by, sort=False).head(max_counterfactuals) if by else ranked.head(max_counterfactuals)
    return ranked.reset_index(drop=True)
