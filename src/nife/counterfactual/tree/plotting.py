"""Drawing of the surrogate tree of an explanation (matplotlib)."""

from __future__ import annotations

import numpy as np

PATH_COLOR = "#7bc96f"  # the instance's decision path
CROSSED_COLOR = "#f4a259"  # the other side of the crossed rule
OTHER_COLOR = "white"


def plot_tree(explanation, ax=None, **kwargs):
    """Draw the surrogate tree of `explanation` with `sklearn.tree.plot_tree`.

    The nodes on the instance's decision path are green; the node on the other side
    of the crossed rule — where the counterfactuals from the data lie — is orange.
    Keyword arguments go to `sklearn.tree.plot_tree`. Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree as sk_plot_tree

    tree = explanation.surrogate
    if ax is None:
        depth, leaves = tree.get_depth(), tree.get_n_leaves()
        _, ax = plt.subplots(figsize=(max(6, 2.4 * leaves), max(3, 1.6 * (depth + 1))))
    options = dict(
        feature_names=list(explanation.instance.index),
        class_names=[str(c) for c in tree.classes_] if hasattr(tree, "classes_") else None,
        impurity=False,
        rounded=True,
        precision=3,
        ax=ax,
    )
    options.update(kwargs)
    annotations = sk_plot_tree(tree, **options)
    # The edge labels of the root ("True", "False") come with the node boxes, which follow the node ids
    boxes = [a for a in annotations if a.get_text().strip() not in ("True", "False")]

    x_row = explanation.instance.to_numpy(dtype=np.float32).reshape(1, -1)
    path = tree.decision_path(x_row).indices
    colors = {node: PATH_COLOR for node in path}
    if explanation._crossed_index is not None:
        node = path[explanation._crossed_index]
        left, right = tree.tree_.children_left[node], tree.tree_.children_right[node]
        colors[right if left in colors else left] = CROSSED_COLOR
    for node, box in enumerate(boxes):
        patch = box.get_bbox_patch()
        if patch is not None:
            patch.set_facecolor(colors.get(node, OTHER_COLOR))
    return ax
