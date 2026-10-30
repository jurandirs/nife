"""Export of the surrogate tree as text: Graphviz DOT, PMML and scikit-learn's rules."""

import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from sklearn.tree import export_text

from nife.counterfactual import TreeCounterfactualExplainer
from nife.counterfactual.tree.plotting import node_labels

NS = {"p": "http://www.dmg.org/PMML-4_4"}


@pytest.fixture(scope="module")
def classification(iris):
    X, model = iris
    return X, TreeCounterfactualExplainer(model, X, random_state=1).explain(X.iloc[60])


@pytest.fixture(scope="module")
def regression(diabetes):
    X, model = diabetes
    explainer = TreeCounterfactualExplainer(model, X, mode="regression", categorical_features=["sex"], random_state=1)
    return X, explainer.explain(X.iloc[0])


def evaluate_pmml(text, data):
    """A minimal PMML TreeModel evaluator: follow the predicates down to a leaf."""
    root = ET.fromstring(text).find("p:TreeModel/p:Node", NS)
    operators = {"lessOrEqual": np.less_equal, "greaterThan": np.greater}
    scores = []
    for _, row in data.iterrows():
        node = root
        while True:
            children = node.findall("p:Node", NS)
            if not children:
                break
            for child in children:
                predicate = child.find("p:SimplePredicate", NS)
                value = row[predicate.get("field")]
                if operators[predicate.get("operator")](value, float(predicate.get("value"))):
                    node = child
                    break
        scores.append(node.get("score"))
    return scores


@pytest.mark.parametrize("case", ["classification", "regression"])
def test_pmml_reproduces_the_tree(case, request):
    X, explanation = request.getfixturevalue(case)
    text = explanation.export_tree("pmml")
    scores = evaluate_pmml(text, X)
    expected = explanation.surrogate.predict(X.to_numpy(np.float32))
    if case == "classification":
        assert scores == [str(c) for c in expected]
    else:
        np.testing.assert_allclose([float(s) for s in scores], expected)
    document = ET.fromstring(text)
    assert document.get("version") == "4.4"
    model = document.find("p:TreeModel", NS)
    assert model.get("functionName") == case
    tree = explanation.surrogate
    assert len(model.findall(".//p:Node", NS)) == tree.tree_.node_count
    assert float(model.find("p:Node", NS).get("recordCount")) == tree.tree_.n_node_samples[0]
    extensions = {e.get("name"): e.get("value") for e in model.findall("p:Extension", NS)}
    path = tree.decision_path(explanation.instance.to_numpy(np.float32).reshape(1, -1)).indices
    assert extensions["decisionPath"] == " ".join(map(str, path))
    fields = [f.get("name") for f in document.findall("p:DataDictionary/p:DataField", NS)]
    assert fields == [*X.columns, "prediction"]


def test_pmml_class_distribution(classification):
    _, explanation = classification
    model = ET.fromstring(explanation.export_tree("pmml")).find("p:TreeModel", NS)
    root = model.find("p:Node", NS)
    counts = [float(d.get("recordCount")) for d in root.findall("p:ScoreDistribution", NS)]
    assert sum(counts) == pytest.approx(float(root.get("recordCount")))
    assert root.get("score") == explanation.surrogate.classes_[int(np.argmax(counts))]


def test_dot(classification):
    _, explanation = classification
    text = explanation.export_tree("dot")
    tree = explanation.surrogate
    assert text.startswith("digraph surrogate_tree {") and text.rstrip().endswith("}")
    assert text.count(" -> ") == tree.tree_.node_count - 1
    assert text.count('label="True"') == text.count('label="False"') == (tree.tree_.node_count - 1) // 2
    for node, label in node_labels(explanation).items():
        assert f'{node} [label="{label}"'.replace("\n", "\\n") in text
    if shutil.which("dot"):  # Graphviz reads it
        result = subprocess.run(["dot", "-Tsvg"], input=text, capture_output=True, text=True)
        assert result.returncode == 0 and "<svg" in result.stdout


def test_text_is_scikit_learns(classification):
    X, explanation = classification
    assert explanation.export_tree("text") == export_text(explanation.surrogate, feature_names=list(X.columns),
                                                          decimals=6)


def test_export_to_a_file(classification, tmp_path):
    _, explanation = classification
    path = tmp_path / "tree.pmml"
    assert explanation.export_tree("pmml", path) == path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="unknown format"):
        explanation.export_tree("json")
