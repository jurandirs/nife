"""Plots of counterfactual explanations, for counterfactuals found by any method.

`plot_changes` draws what one counterfactual changes, feature by feature, on the
range of each feature in the reference data. Explanations draw it with
`CounterfactualExplanation.plot_changes`; the function takes the instance, the
counterfactual and the reference data directly, so it also draws counterfactuals
found outside NIFE.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..plotting import (
    COLORS,
    clean_axes,
    dot,
    format_value,
    legend_below,
    new_axes,
    set_titles,
    styled,
)

# The range of each feature is drawn from 0 to 1, and the values on the right
VALUES_AT, VALUES_END = 1.1, 1.6


class FeatureScale:
    """The scale of each feature in the reference data, to compare changes across features."""

    def __init__(self, data: pd.DataFrame, categorical_features: Sequence[str] = ()):
        self.features = list(data.columns)
        self.categorical = [f for f in categorical_features if f in self.features]
        self.low = data.min()
        span = data.max() - self.low
        self.span = span.where(span > 0, 1.0)
        self.q25, self.q75 = data.quantile(0.25), data.quantile(0.75)
        self.categories = {f: np.sort(data[f].unique()) for f in self.categorical}

    def is_categorical(self, feature: str) -> bool:
        return feature in self.categories

    def position(self, feature: str, values):
        """Where `values` lie in the feature's range: 0 at the minimum, 1 at the maximum.

        Categorical values are spread evenly, in the order of their codes.
        """
        values = np.asarray(values, dtype=float)
        if feature in self.categories:
            categories = self.categories[feature]
            return np.searchsorted(categories, values) / max(len(categories) - 1, 1)
        return (values - self.low[feature]) / self.span[feature]

    def change(self, feature: str, new, old):
        """Signed change, as a fraction of the range; 1 or 0 (changed or not) for a categorical feature."""
        new, old = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
        if feature in self.categories:
            return (new != old).astype(float)
        return (new - old) / self.span[feature]


def _check_features(instance: pd.Series, data: pd.DataFrame) -> list:
    features = list(instance.index)
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise ValueError(f"the reference data lacks the features {missing}")
    return features


@styled
def plot_changes(
    instance: pd.Series,
    counterfactual: pd.Series,
    data: pd.DataFrame,
    *,
    categorical_features: Sequence[str] = (),
    max_features: int = 10,
    ax=None,
    title: str = "What would change the prediction",
    subtitle: Optional[str] = None,
):
    """What one counterfactual changes: each changed feature, from the instance's value to the counterfactual's.

    Each line spans the range of the feature in `data`, with its middle 50% darker;
    the blue dot is the instance, the orange one the counterfactual. Features are
    ordered by the size of their change (as a fraction of the range); the values are
    written on the right. Returns the matplotlib Axes.
    """
    features = _check_features(instance, data)
    scale = FeatureScale(data[features], categorical_features)
    x, c = instance[features].astype(float), counterfactual[features].astype(float)
    changed = [f for f in features if x[f] != c[f]]
    if not changed:
        raise ValueError("the counterfactual changes no feature")
    changed.sort(key=lambda f: -abs(float(scale.change(f, c[f], x[f]))))
    shown, hidden = changed[:max_features], len(changed) - max_features

    ax = new_axes(ax, figsize=(7.5, 0.42 * len(shown) + 1.9))
    rows = len(shown)
    for row, feature in enumerate(shown):
        y = rows - 1 - row
        ax.plot([0, 1], [y, y], color=COLORS["neutral_wash"], linewidth=7, solid_capstyle="round", zorder=1)
        if not scale.is_categorical(feature):
            q25, q75 = scale.position(feature, [scale.q25[feature], scale.q75[feature]])
            ax.plot([q25, q75], [y, y], color=COLORS["grid"], linewidth=7, solid_capstyle="butt", zorder=1.1)
        px, pc = float(scale.position(feature, x[feature])), float(scale.position(feature, c[feature]))
        ax.annotate(
            "", xy=(pc, y), xytext=(px, y), zorder=2,
            arrowprops=dict(arrowstyle="-|>", color=COLORS["counterfactual"], linewidth=1.8,
                            shrinkA=5, shrinkB=6, mutation_scale=11),
        )
        ax.scatter([px], [y], s=70, color=COLORS["instance"], edgecolor=COLORS["surface"], linewidth=1.5, zorder=3)
        ax.scatter([pc], [y], s=70, color=COLORS["counterfactual"], edgecolor=COLORS["surface"], linewidth=1.5, zorder=3)
        ax.text(VALUES_AT, y, f"{format_value(x[feature])}  →  {format_value(c[feature])}", va="center",
                ha="left", color=COLORS["ink_secondary"], fontsize=9)
    ax.text(VALUES_AT, rows - 0.45, "instance  →  counterfactual", va="bottom", ha="left",
            color=COLORS["ink_muted"], fontsize=8.5)

    ax.set_yticks(range(rows))
    ax.set_yticklabels(shown[::-1])
    ax.set_ylim(-0.7, rows - 0.1)
    ax.set_xlim(-0.04, VALUES_END)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["min", "max"])
    clean_axes(ax, left=False, bottom=False)
    for label in ax.get_xticklabels():
        label.set_color(COLORS["ink_muted"])
    if hidden > 0:
        note = f"{hidden} more changed feature{'s' if hidden > 1 else ''} not shown"
        subtitle = f"{subtitle} · {note}" if subtitle else note
    set_titles(ax, title, subtitle)
    legend_below(ax, [dot(COLORS["instance"]), dot(COLORS["counterfactual"])], ["Instance", "Counterfactual"])
    return ax
