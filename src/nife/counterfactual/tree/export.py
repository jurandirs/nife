"""Export of the surrogate tree of an explanation as text, in established formats.

- ``"dot"`` — Graphviz DOT (https://graphviz.org), read by Graphviz and many other
  tools: the nodes show the same information as `plot_tree` (and as the trees of
  Silva et al., 2026), with the decision path and the other side of the counterfactual rule
  highlighted.
- ``"pmml"`` — PMML 4.4 (https://dmg.org/pmml), the XML standard for exchanging
  predictive models (KNIME, SAS, R, Java and Python evaluators...): a ``TreeModel``
  with the class distribution or the mean of each node; the decision path and the
  node on the other side of the counterfactual rule (``crossedNode``) are recorded in
  ``Extension`` elements, which other tools ignore.
- ``"text"`` — the rules of the tree as written by `sklearn.tree.export_text`.

The thresholds are written with full precision in PMML and in the text, so that
they reproduce the tree's decisions; DOT shows them rounded, as the plot does.
"""

from __future__ import annotations

from typing import List, Optional
from xml.sax.saxutils import quoteattr

import numpy as np

from ...plotting import COLORS
from .plotting import node_labels

FORMATS = ("dot", "pmml", "text")


def export_tree(explanation, format: str = "dot", path: Optional[str] = None, *, precision: int = 4) -> str:
    """The surrogate tree of `explanation` as text in `format` ("dot", "pmml" or "text").

    With `path`, the text is also written to that file. `precision` is the number
    of decimals of the numbers shown in DOT. Returns the text.
    """
    exporters = {"dot": lambda: to_dot(explanation, precision), "pmml": lambda: to_pmml(explanation),
                 "text": lambda: to_text(explanation)}
    if format not in exporters:
        raise ValueError(f"unknown format {format!r}; available: {', '.join(FORMATS)}")
    text = exporters[format]()
    if path is not None:
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
    return text


def _path_and_crossed(explanation):
    tree = explanation.surrogate
    x_row = explanation.instance.to_numpy(dtype=np.float32).reshape(1, -1)
    path = [int(n) for n in tree.decision_path(x_row).indices]
    crossed = None
    if explanation._crossed_index is not None:
        node = path[explanation._crossed_index]
        left, right = tree.tree_.children_left[node], tree.tree_.children_right[node]
        crossed = int(right if left in path else left)
    return path, crossed


def _dot_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def to_dot(explanation, precision: int = 4) -> str:
    """The surrogate tree in Graphviz DOT, with the information and the colors of `plot_tree`.

    Render it with ``dot -Tpdf tree.dot -o tree.pdf`` (or ``-Tsvg``, ``-Tpng``).
    """
    tree = explanation.surrogate
    left, right = tree.tree_.children_left, tree.tree_.children_right
    labels = node_labels(explanation, precision)
    path, crossed = _path_and_crossed(explanation)
    lines = [
        "digraph surrogate_tree {",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, '
        f'color="{COLORS["axis"]}", fillcolor="{COLORS["surface"]}"];',
        f'  edge [fontname="Helvetica", fontsize=9, color="{COLORS["axis"]}", arrowhead=none];',
    ]
    for node in range(tree.tree_.node_count):
        style = ""
        if node in path:
            style = f', color="{COLORS["instance"]}", fillcolor="{COLORS["instance_wash"]}", penwidth=1.5'
        elif node == crossed:
            style = f', color="{COLORS["counterfactual"]}", fillcolor="{COLORS["counterfactual_wash"]}", penwidth=1.5'
        lines.append(f"  {node} [label={_dot_string(labels[node])}{style}];")
    for node in range(tree.tree_.node_count):
        if left[node] == -1:
            continue
        for child, answer in ((left[node], "True"), (right[node], "False")):
            style = ""
            if child in path:
                style = f', color="{COLORS["instance"]}", penwidth=2'
            elif child == crossed:
                style = f', color="{COLORS["counterfactual"]}", penwidth=2'
            lines.append(f'  {node} -> {child} [label="{answer}"{style}];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def to_text(explanation) -> str:
    """The rules of the surrogate tree, as written by `sklearn.tree.export_text`."""
    from sklearn.tree import export_text

    tree = explanation.surrogate
    return export_text(tree, feature_names=list(explanation.instance.index), decimals=6)


def _number(value: float) -> str:
    """A float in full precision (shortest repr that reads back the same)."""
    return repr(float(value))


def to_pmml(explanation) -> str:
    """The surrogate tree as a PMML 4.4 ``TreeModel``.

    Each node has the outcome of the tree (``score``), its number of neighborhood
    points (``recordCount``) and, in classification, the distribution of the classes
    (``ScoreDistribution``). The fields are the features (continuous; categorical
    features keep their numeric codes, on which the tree splits) and the target,
    named ``prediction``: the model's prediction, which the tree learned. NIFE's
    information — the instance, its decision path and the node on the other side of the
    counterfactual rule — goes in
    ``Extension`` elements.
    """
    from ... import __version__

    tree = explanation.surrogate
    t = tree.tree_
    names = list(explanation.instance.index)
    classification = hasattr(tree, "classes_")
    path, crossed = _path_and_crossed(explanation)
    target = "prediction"

    def class_text(value) -> str:
        return str(value.item() if isinstance(value, np.generic) else value)

    if classification:
        classes = [class_text(c) for c in tree.classes_]
        numeric = all(isinstance(c, (int, np.integer)) for c in tree.classes_)
        target_field = (
            f'    <DataField name="{target}" optype="categorical" dataType="{"integer" if numeric else "string"}">\n'
            + "".join(f"      <Value value={quoteattr(c)}/>\n" for c in classes)
            + "    </DataField>\n"
        )
    else:
        target_field = f'    <DataField name="{target}" optype="continuous" dataType="double"/>\n'

    out: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        '<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4">\n',
        '  <Header description="Surrogate decision tree of a NIFE explanation">\n',
        f'    <Application name="NIFE" version={quoteattr(__version__)}/>\n',
        "  </Header>\n",
        f'  <DataDictionary numberOfFields="{len(names) + 1}">\n',
        *(f'    <DataField name={quoteattr(name)} optype="continuous" dataType="double"/>\n' for name in names),
        target_field,
        "  </DataDictionary>\n",
        f'  <TreeModel modelName="surrogate_tree" functionName="{"classification" if classification else "regression"}"'
        ' algorithmName="CART" splitCharacteristic="binarySplit" noTrueChildStrategy="returnNullPrediction">\n',
        '    <Extension extender="NIFE" name="instance" value='
        + quoteattr(" ".join(f"{name}={_number(v)}" for name, v in explanation.instance.items())) + "/>\n",
        '    <Extension extender="NIFE" name="decisionPath" value=' + quoteattr(" ".join(map(str, path))) + "/>\n",
    ]
    if crossed is not None:
        out.append(f'    <Extension extender="NIFE" name="crossedNode" value="{crossed}"/>\n')
    out += [
        "    <MiningSchema>\n",
        *(f"      <MiningField name={quoteattr(name)}/>\n" for name in names),
        f'      <MiningField name="{target}" usageType="target"/>\n',
        "    </MiningSchema>\n",
    ]

    def node(index: int, predicate: str, indent: str) -> None:
        value = t.value[index][0]
        weight = float(t.weighted_n_node_samples[index])
        if classification:
            counts = value * weight if np.isclose(value.sum(), 1.0) else value  # fractions in recent scikit-learn
            score = class_text(tree.classes_[int(np.argmax(value))])
        else:
            score = _number(value[0])
        out.append(f'{indent}<Node id="{index}" score={quoteattr(score)} recordCount="{_number(weight)}">\n')
        out.append(f"{indent}  {predicate}\n")
        if classification:
            for c, count in zip(tree.classes_, counts):
                out.append(
                    f"{indent}  <ScoreDistribution value={quoteattr(class_text(c))} recordCount=\"{_number(count)}\""
                    f' probability="{_number(count / weight)}"/>\n'
                )
        if t.children_left[index] != -1:
            field, threshold = quoteattr(names[t.feature[index]]), _number(t.threshold[index])
            node(int(t.children_left[index]),
                 f'<SimplePredicate field={field} operator="lessOrEqual" value="{threshold}"/>', indent + "  ")
            node(int(t.children_right[index]),
                 f'<SimplePredicate field={field} operator="greaterThan" value="{threshold}"/>', indent + "  ")
        out.append(f"{indent}</Node>\n")

    node(0, "<True/>", "    ")
    out += ["  </TreeModel>\n", "</PMML>\n"]
    return "".join(out)
