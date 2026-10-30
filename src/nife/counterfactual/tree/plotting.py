"""Plots of the explanations of `TreeCounterfactualExplainer` (matplotlib).

- `plot_tree` — the surrogate tree, with the instance's path and the other side of the
  counterfactual rule, with
  the information of the trees of Silva et al. (2026);
- `plot_importances` — the relevant features, ordered by their importance in the
  surrogate tree.

They take a `TreeCounterfactualExplanation` and share the look of `nife.plotting`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...plotting import COLORS, clean_axes, format_value, legend_below, new_axes, set_titles, styled, swatch


def _crossed_feature(explanation) -> Optional[str]:
    return explanation.crossed_rule.feature if explanation.crossed_rule is not None else None


def format_number(value: float, precision: int = 4) -> str:
    """`value` rounded to `precision` decimals, without trailing zeros, as scikit-learn writes trees: 715.404."""
    text = f"{round(float(value), precision):.{precision}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def node_labels(explanation, precision: int = 4) -> dict:
    """The text of each node of the surrogate tree, as in the trees of Silva et al. (2026).

    A decision node shows its decision and its outcome (``proline ≤ 715.404`` and
    ``class = 2``, or ``value = 30.792`` in regression); a leaf only its outcome. The
    outcome is the majority class, or the mean, of the neighborhood points in the node.
    """
    tree = explanation.surrogate
    names = list(explanation.instance.index)
    labels = {}
    for node in range(tree.tree_.node_count):
        value = tree.tree_.value[node][0]
        if hasattr(tree, "classes_"):
            outcome = f"class = {format_value(tree.classes_[int(np.argmax(value))])}"
        else:
            outcome = f"value = {format_number(value[0], precision)}"
        if tree.tree_.children_left[node] == -1:
            labels[node] = outcome
        else:
            decision = f"{names[tree.tree_.feature[node]]} ≤ {format_number(tree.tree_.threshold[node], precision)}"
            labels[node] = f"{decision}\n{outcome}"
    return labels


@styled
def plot_tree(explanation, *, path_only: bool = False, precision: int = 4, fontsize: float = 8,
              titles: bool = True, ax=None):
    """The surrogate tree, compact, with the information of the trees of Silva et al. (2026).

    Each decision node shows its decision (``feature ≤ threshold``) and its outcome
    (``class = …``, or ``value = …`` in regression); each leaf its outcome; the edges
    are ``True`` (left) and ``False`` (right). The instance's decision path is blue;
    the other side of the counterfactual rule, where its evidence lies, is orange.
    Numbers are rounded to `precision` decimals.

    ``path_only=True`` draws only the decision path, with the other side of each
    decision as a single node ("…" marks a subtree left out): the smallest figure.
    ``titles=False`` leaves out the title and subtitle, for a caption of your own.
    Save it as vector graphics with ``ax.figure.savefig("tree.pdf")``. Returns the
    matplotlib Axes.
    """
    tree = explanation.surrogate
    left, right = tree.tree_.children_left, tree.tree_.children_right
    x_row = explanation.instance.to_numpy(dtype=np.float32).reshape(1, -1)
    path = set(tree.decision_path(x_row).indices)
    crossed = None
    if explanation._crossed_index is not None:
        node = sorted(path, key=lambda n: _depth(tree, n))[explanation._crossed_index]
        crossed = right[node] if left[node] in path else left[node]

    labels = node_labels(explanation, precision)

    def expanded(node):
        return left[node] != -1 and (not path_only or node in path)

    for node in range(tree.tree_.node_count):
        if left[node] != -1 and not expanded(node):
            labels[node] += "\n…"  # a subtree left out

    # Tidy layout, in inches: each subtree is as wide as its widest level needs
    char, pad, gap = fontsize * 0.6 / 72, 0.16, 0.12
    def box(node):
        return max(len(line) for line in labels[node].split("\n")) * char + pad

    width, x, depth, shown = {}, {}, {}, []
    def measure(node):
        width[node] = box(node) if not expanded(node) else max(box(node), measure(left[node]) + gap + measure(right[node]))
        return width[node]
    def place(node, start, d):
        depth[node] = d
        shown.append(node)
        x[node] = start + width[node] / 2
        if expanded(node):
            children = width[left[node]] + gap + width[right[node]]
            offset = start + (width[node] - children) / 2
            place(left[node], offset, d + 1)
            place(right[node], offset + width[left[node]] + gap, d + 1)
    measure(0)
    place(0, 0.0, 0)

    levels = max(depth.values()) + 1
    level = 0.62 * fontsize / 8  # inches between levels
    ax = new_axes(ax, figsize=(max(2.0, width[0] + 0.2), level * (levels - 1) + 0.5 + (0.75 if titles else 0)))
    for node in shown:
        if not expanded(node):
            continue
        for child, answer in ((left[node], "True"), (right[node], "False")):
            color = (COLORS["instance"] if child in path else COLORS["counterfactual"] if child == crossed
                     else COLORS["axis"])
            ax.plot([x[node], x[child]], [-depth[node] * level, -depth[child] * level], color=color,
                    linewidth=0.8 if color == COLORS["axis"] else 1.6, zorder=1)
            ax.text((x[node] + x[child]) / 2, -(depth[node] + 0.5) * level, answer, ha="center", va="center",
                    fontsize=fontsize - 1, color=COLORS["ink_secondary"], zorder=2,
                    bbox=dict(boxstyle="square,pad=0.08", facecolor=COLORS["surface"], edgecolor="none"))
    for node in shown:
        on_path, is_crossed = node in path, node == crossed
        face = COLORS["instance_wash"] if on_path else COLORS["counterfactual_wash"] if is_crossed else COLORS["surface"]
        edge = COLORS["instance"] if on_path else COLORS["counterfactual"] if is_crossed else COLORS["axis"]
        ax.text(x[node], -depth[node] * level, labels[node], ha="center", va="center", fontsize=fontsize,
                color=COLORS["ink"], linespacing=1.15, zorder=3,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=face, edgecolor=edge, linewidth=0.9))
    ax.set_xlim(0, width[0])
    ax.set_ylim(-(levels - 1) * level - 0.22, 0.22)
    ax.axis("off")
    if titles:
        set_titles(ax, "Surrogate tree",
                   "Blue: the decision path of the instance · orange: the other side of the counterfactual rule")
    return ax


def _depth(tree, node) -> int:
    """Depth of `node` (the root is 0)."""
    parents = {child: parent for parent in range(tree.tree_.node_count)
               for child in (tree.tree_.children_left[parent], tree.tree_.children_right[parent]) if child != -1}
    depth = 0
    while node in parents:
        node, depth = parents[node], depth + 1
    return depth


@styled
def plot_importances(explanation, *, max_features: int = 10, ax=None):
    """The relevant features of the prediction: their importance in the surrogate tree.

    Blue: features on the instance's decision path; orange: the feature of the
    counterfactual rule; gray: other features the tree uses. Returns the matplotlib Axes.
    """
    importances = explanation.feature_importances
    if not len(importances):
        raise ValueError("the surrogate tree uses no feature (it has a single leaf)")
    shown = importances.head(max_features)
    on_path = {rule.feature for rule in explanation.rules}
    crossed = _crossed_feature(explanation)

    def role(feature):
        return "crossed" if feature == crossed else "path" if feature in on_path else "other"

    palette = {"path": COLORS["instance"], "crossed": COLORS["counterfactual"], "other": COLORS["neutral"]}
    ax = new_axes(ax, figsize=(7, 0.4 * len(shown) + 1.9))
    rows = len(shown)
    for row, (feature, value) in enumerate(shown.items()):
        y = rows - 1 - row
        ax.barh(y, value, height=0.55, color=palette[role(feature)], zorder=2)
        ax.text(value, y, f" {value:.2f}", va="center", ha="left", fontsize=8.5, color=COLORS["ink_secondary"])
    ax.set_yticks(range(rows))
    ax.set_yticklabels(list(shown.index)[::-1])
    ax.set_ylim(-0.7, rows - 0.3)
    ax.set_xlim(0, float(shown.max()) * 1.15)
    ax.set_xlabel("Importance in the surrogate tree (impurity decrease)")
    clean_axes(ax, grid="x", left=False)
    hidden = len(importances) - len(shown)
    subtitle = "The features the surrogate tree uses to reproduce the model around the instance"
    if hidden > 0:
        subtitle += f" · {hidden} more not shown"
    set_titles(ax, "Relevant features", subtitle)
    present = [r for r in ("path", "crossed", "other") if any(role(f) == r for f in shown.index)]
    names = {"path": "on the decision path", "crossed": "counterfactual rule", "other": "used elsewhere in the tree"}
    if len(present) > 1:
        legend_below(ax, [swatch(palette[r]) for r in present], [names[r] for r in present])
    return ax
