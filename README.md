# NIFE — NIFE Is a Framework for Explanations

[![PyPI](https://img.shields.io/pypi/v/nife-xai.svg)](https://pypi.org/project/nife-xai/)
[![Python](https://img.shields.io/pypi/pyversions/nife-xai.svg)](https://pypi.org/project/nife-xai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jurandirs/nife/blob/main/LICENSE)

NIFE is a Python framework for explaining the predictions of machine-learning models. Today it
provides counterfactual explanations of individual predictions, with one method and metrics to
evaluate them; more kinds of explanations, methods and metrics are planned. NIFE is organized by
**kind of explanation**, each with the **methods** that produce it, the **metrics** that evaluate
it and the **plots** that show it, so that new ones plug into the same structure. It is in alpha:
its API may change before version 1.0.

| Explanation | Methods | Evaluation |
|---|---|---|
| [Counterfactual](https://github.com/jurandirs/nife/blob/main/notebooks/counterfactual/counterfactual_explanations.ipynb) — the smallest change that would make the model predict something else, with a real counterpart from the data | [`TreeCounterfactualExplainer`](https://github.com/jurandirs/nife/blob/main/notebooks/counterfactual/TreeCounterfactualExplainer.ipynb) — surrogate decision tree, based on Silva, Aguilar and Salvini (2026) | [coverage and 8 metrics](https://github.com/jurandirs/nife/blob/main/notebooks/counterfactual/evaluation.ipynb), for counterfactuals found by any method |

NIFE is for tabular data: images, text and time series are not supported. It works with any model
that has a `predict` method, for classification and regression, and calls only `predict`: a
counterfactual changes the predicted class or value, whatever the model's confidence. Besides the
model, it takes the data, usually the one the model was trained on, as a pandas DataFrame of
numeric features: encode categorical features as numbers and list them in `categorical_features`.
Classification is the default; for a regression model, pass `mode="regression"`.

## Installation

```bash
pip install nife-xai
```

The package is `nife-xai` on PyPI, and you import it as `nife`.

NIFE requires Python 3.10 or later, with NumPy, pandas, scikit-learn and matplotlib. To install
the development version from GitHub:

```bash
pip install git+https://github.com/jurandirs/nife.git
```

## Quickstart

```python
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

from nife.counterfactual import TreeCounterfactualExplainer

X, y = load_wine(return_X_y=True, as_frame=True)
model = RandomForestClassifier(random_state=0).fit(X, y)

explainer = TreeCounterfactualExplainer(model, X, random_state=0)
explanation = explainer.explain(X.iloc[0])
print(explanation)
```

```text
TreeCounterfactualExplanation (classification) of an instance predicted 0
  decision path: color_intensity > 3.437 and proline > 651.7
  relevant features: proline, color_intensity, flavanoids
  counterfactual rule: proline > 651.7 -> proline <= 651.7 (not confirmed: changing proline alone keeps the model's prediction)
  counterfactuals: 16, changing 2 features each (16 copied from evidence examples)
  closest: malic_acid 1.71 -> 1.17, proline 1065 -> 510; predicted 1 (distance 0.039; counterpart at distance 0.137)
  evidence: 42 examples of the data on the other side of the rule, the closest at distance 0.176
  fidelity: instance agreement True, local fidelity 1.00 (neighborhood: around the instance, scale 1)
```

A decision tree trained on the model's behavior around this wine explains its class: `proline` and
`color_intensity` drive it, and in the tree `proline ≤ 651.7` would lead to class 1. The model
disagrees: changing `proline` alone keeps class 0. So NIFE looks for the fewest values that,
copied into the wine from the closest *evidence* — real wines on the other side of the rule that
the model puts in another class — change the model's prediction. Two are needed, so each
counterfactual changes two features: the closest takes `proline` and `malic_acid` from one of those
wines. Each counterfactual also has a *counterpart*, the closest real wine that the model puts in
its class (here, class 1). The
[`TreeCounterfactualExplainer` notebook](https://github.com/jurandirs/nife/blob/main/notebooks/counterfactual/TreeCounterfactualExplainer.ipynb)
explains this output line by line.

The counterfactuals are in `explanation.counterfactuals`, a DataFrame ranked from the closest to
the instance. `explain` also takes the outcome you want (`desired=`, a class or, in regression,
`"higher"` or `"lower"`) and the number of counterfactuals to keep, and the explainer takes
`immutable_features=`. `explainer.explain_many` explains many instances at once, and
`explainer.evaluate` measures the counterfactuals.

`explanation.plot()` draws what the closest counterfactual changes, from the instance's values
(blue) to the counterfactual's (orange), on the range of each feature in the data (the darker part
of each bar is the middle 50% of the data):

```python
explanation.plot()
```

![What the closest counterfactual changes](https://raw.githubusercontent.com/jurandirs/nife/main/docs/images/plot_changes.png)

## Guarantees and limits

Every counterfactual that NIFE returns was checked with the model: its prediction is another
outcome (or the desired one), and it does not change the immutable features. Beyond that:

- **A counterfactual may not exist.** NIFE can find none for an instance, mostly with many features
  and few examples; the evaluation reports the coverage.
- **No constraints beyond immutable features.** There are no allowed ranges and no direction of
  change (such as "only increase"). Plausibility is measured, with the counterpart and the
  evaluation metrics, not enforced.
- **The tree is an approximation.** `TreeCounterfactualExplainer` explains with a local surrogate
  tree whose agreement with the model (`explanation.fidelity`) depends on the model: check it
  before trusting the rules.
- **The model, not the real world.** A counterfactual is a change that makes the model predict
  something else, not advice on what would change the real outcome.
- **One-hot columns.** NIFE does not know that several columns come from one variable: a
  counterfactual can turn on two columns of the same group, or none, so it may not be a valid
  category. When you can, give each categorical feature a single numeric code and list it in
  `categorical_features`.

Explaining one instance took seconds in tests with up to 200,000 rows (20 features) and up to 1,000
features (500 rows).

## Notebooks

Each kind of explanation and each method has its own notebook in [`notebooks/`](https://github.com/jurandirs/nife/tree/main/notebooks),
organized like the package. They run on datasets bundled with scikit-learn:

- [Counterfactual explanations](https://github.com/jurandirs/nife/blob/main/notebooks/counterfactual/counterfactual_explanations.ipynb) — what
  NIFE calls a counterfactual and its counterpart, the options common to every method, how to
  explain many instances at once, and the `changes` plot:
  - [`TreeCounterfactualExplainer`](https://github.com/jurandirs/nife/blob/main/notebooks/counterfactual/TreeCounterfactualExplainer.ipynb) —
    the surrogate decision tree method: the counterfactual rule and its evidence, the
    counterfactuals, the plots of the tree and its export, its options and its fidelity;
  - [`TreeCounterfactualExplainer` neighborhood](https://github.com/jurandirs/nife/blob/main/notebooks/counterfactual/TreeCounterfactualExplainer_neighborhood.ipynb) —
    the perturbed samples that its tree is trained on: extract them, change them and pass them
    back, or explain with a sampler of your own;
  - [Evaluating counterfactuals](https://github.com/jurandirs/nife/blob/main/notebooks/counterfactual/evaluation.ipynb) — the metrics, for
    counterfactuals found by any method, and how to add new ones.
- [Plots](https://github.com/jurandirs/nife/blob/main/notebooks/plots.ipynb) — the plots of every explanation, and how to place them in figures
  of your own, save them and change their colors.
- [Extending NIFE](https://github.com/jurandirs/nife/blob/main/notebooks/extending.ipynb) — new methods, metrics, plots and kinds of
  explanation, and how to document them.

The tests run every notebook, so they keep working as NIFE changes. To run the notebooks
yourself, install their requirements: `pip install "nife-xai[notebooks]"`.

## How to cite

If you use NIFE, please cite it:

```bibtex
@software{nife,
  author  = {Silva, Jurandir},
  title   = {{NIFE}: {NIFE} Is a Framework for Explanations},
  url     = {https://github.com/jurandirs/nife},
  version = {0.1.0},
  year    = {2026}
}
```

The "Cite this repository" button on GitHub gives the same citation, from
[`CITATION.cff`](https://github.com/jurandirs/nife/blob/main/CITATION.cff).

If you use one of NIFE's methods, please also cite the work that introduced it; each method's
notebook has its BibTeX entry:

| Method | Reference |
|---|---|
| [`TreeCounterfactualExplainer`](https://github.com/jurandirs/nife/blob/main/notebooks/counterfactual/TreeCounterfactualExplainer.ipynb) | Silva, J., Aguilar, E. and Salvini, R. (2026). Unveiling Concise Counterfactual Explanations and Feature Relevance in Machine Learning through Decision Trees. In *Proceedings of the 18th International Conference on Agents and Artificial Intelligence (ICAART 2026) — Volume 4*, pages 2941–2948. SciTePress. https://doi.org/10.5220/0014338700004052 |

`TreeCounterfactualExplainer` is based on this paper and extends it, for example with an adaptive
neighborhood and counterfactuals that the model confirms, so its defaults do not give the paper's
results. The notebook has a section on
[reproducing the paper](https://github.com/jurandirs/nife/blob/main/notebooks/counterfactual/TreeCounterfactualExplainer.ipynb#Reproducing-the-paper).

The concepts and metrics that NIFE uses have their own references, listed in each notebook.

## Contributing

Bug reports, questions and pull requests are welcome — see [CONTRIBUTING.md](https://github.com/jurandirs/nife/blob/main/CONTRIBUTING.md).

## License

NIFE is released under the [MIT License](https://github.com/jurandirs/nife/blob/main/LICENSE).
