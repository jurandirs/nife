# NIFE — NIFE Is a Framework for Explanations

[![PyPI](https://img.shields.io/pypi/v/nife.svg)](https://pypi.org/project/nife/)
[![Python](https://img.shields.io/pypi/pyversions/nife.svg)](https://pypi.org/project/nife/)
[![Tests](https://github.com/jurandirs/nife/actions/workflows/tests.yml/badge.svg)](https://github.com/jurandirs/nife/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/DOI-10.5220%2F0014338700004052-informational)](https://doi.org/10.5220/0014338700004052)

NIFE is a Python framework for explaining the predictions of machine-learning models. It is
organized by **kind of explanation**, each with the **methods** that produce it and the
**metrics** that evaluate it, and is built to grow: new kinds of explanations, new methods and
new metrics plug into the same structure.

Today NIFE provides **counterfactual explanations** — what would have to change for the model to
predict something else — with:

- the **surrogate decision tree** method of
  [Silva, Aguilar and Salvini (ICAART 2026)](https://doi.org/10.5220/0014338700004052), which also
  shows the features that drive each prediction;
- **metrics** to evaluate counterfactuals found by any method.

| Explanation | Module | Methods | Evaluation |
|---|---|---|---|
| Counterfactual | `nife.counterfactual` | `TreeCounterfactualExplainer` — surrogate decision tree (Silva et al., 2026) | `evaluate_counterfactuals` — 8 metrics, for any method |

NIFE works with any model that has a `predict` method, for classification and regression.

## Installation

```bash
pip install nife
```

NIFE requires Python 3.9 or later, with NumPy, pandas, scikit-learn and matplotlib. To install
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
  crossed rule: proline > 651.7 -> proline <= 651.7
  counterfactuals: 42 (data: 42)
  closest: distance 0.176, 13 changed features, predicted 1
  fidelity: instance agreement True, local fidelity 1.00 (neighborhood: around the instance, scale 1)
```

The explanation holds the counterfactuals (`explanation.counterfactuals`, a DataFrame ranked by
distance to the instance), the decision path (`explanation.rules`), the relevance of the features
(`explanation.feature_importances`), the surrogate tree (`explanation.surrogate`) and its
fidelity to the model (`explanation.fidelity`). `explanation.plot_tree()` draws the tree with the
instance's path in green and, in orange, the other side of the crossed rule — where the
counterfactuals lie:

![The surrogate tree of the explanation](https://raw.githubusercontent.com/jurandirs/nife/main/docs/images/explanation_tree.png)

## Counterfactual explanations

### Options common to every method

Every counterfactual explainer derives from `nife.counterfactual.CounterfactualExplainer`, so
these options work the same way whatever the method:

- **Desired outcome** (`desired`): a class, or in regression `"higher"`, `"lower"` or an
  interval `(low, high)` (either bound may be `None`). Without it, any other outcome counts — in
  regression, a prediction more than `tolerance` (5% by default) away from the instance's. The
  model confirms every counterfactual, so they are always valid.
- **Immutable features** (`immutable_features`): features that cannot change, such as age or sex;
  no counterfactual changes them.
- **Selection** (`max_counterfactuals`, `nondominated`): keep the closest counterfactuals, or those
  that no other beats in both distance and number of changed features — the Pareto front of
  Dandl et al. (2020). `select_counterfactuals` does the same on any set of counterfactuals.
- **Many instances** (`explain_many`) and **evaluation** (`evaluate`), described below.

### The surrogate decision tree method

`TreeCounterfactualExplainer` implements and extends the method of Silva, Aguilar and Salvini
(2026):

1. **Neighborhood.** It samples a perturbed neighborhood of the instance, as LIME does, and
   labels it with the model's predictions.
2. **Surrogate tree.** A decision tree learns the model's behavior in that neighborhood. The
   features it uses, with their importances, are the **relevant features** of the prediction; the
   path of the instance from the root to its leaf is the **decision path**.
3. **Counterfactuals.** Crossing the deepest decision of the path leads to another outcome. The
   examples of the data on the other side of that decision are counterfactuals — plausible, since
   they are real examples.

Its own options:

- **Counterfactuals from the rules** (`from_rules=True`): for each feature of the decision path,
  the instance with only that feature moved to the closest value, seen in the data on the other
  side of the decision, that the model confirms. They change a single feature and are very close
  to the instance, but are not real examples (`source` tells them apart: `"data"` or `"rules"`).
- **Immutable features** are also fixed in the neighborhood, so the tree only splits on the others.
- **Adaptive neighborhood** (default): the neighborhood starts narrow around the instance and only
  widens until the tree has rules and a counterfactual is found, like the expanding spheres of
  Growing Spheres (Laugel et al., 2017). A narrow neighborhood makes the tree agree with the model
  around the instance; `adaptive=False` uses the global sample of the original method.
- **Fidelity** (`explanation.fidelity`): whether the tree gives the instance the same outcome as
  the model, and on what fraction of its 50 nearest examples they agree. The rules only explain
  the prediction when the fidelity is high.

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)
model = RandomForestRegressor(random_state=0).fit(X, y)

explainer = TreeCounterfactualExplainer(
    model, X, mode="regression", categorical_features=["sex"], immutable_features=["age", "sex"]
)
explanation = explainer.explain(X.iloc[0], desired="lower", from_rules=True, nondominated=True)
```

### Explaining and evaluating many instances

`explain_many` returns the counterfactuals of several instances in one DataFrame, with an
`instance` column (the position of each instance), and `evaluate` measures them:

```python
instances = X.sample(20, random_state=0)
counterfactuals = explainer.explain_many(instances, desired="lower", nondominated=True)
explainer.evaluate(instances, counterfactuals, desired="lower")
```

`nife.counterfactual.evaluate_counterfactuals` evaluates counterfactuals found by **any** method,
NIFE's or not; `metrics=[...]` picks some of the metrics. Every metric is computed per instance
and averaged over the instances with counterfactuals; distances are Gower distances (Gower, 1971),
so numeric and categorical features can be mixed.

| Metric | Measures | Better | Reference |
|---|---|---|---|
| `coverage` | fraction of the instances with at least one counterfactual | higher | Pawelczyk et al. (2021) |
| `count` | counterfactuals per instance | — | |
| `validity` | fraction of the counterfactuals that reach a counterfactual outcome | higher | Wachter et al. (2018); Mothilal et al. (2020) |
| `proximity` | distance between the instance and its counterfactuals | lower | Mothilal et al. (2020) |
| `changed_features` | number of features changed (sparsity) | lower | Mothilal et al. (2020) |
| `redundancy` | changed features that could be reverted without losing validity | lower | Pawelczyk et al. (2021) |
| `implausibility` | distance to the closest example of the data | lower | Guidotti (2022) |
| `ynn` | fraction of the k nearest examples that also get a counterfactual outcome | higher | Pawelczyk et al. (2021) |
| `diversity` | distance between the counterfactuals of the same instance | higher | Mothilal et al. (2020) |
| `constraint_violations` | fraction of the counterfactuals that change an immutable feature | lower | Pawelczyk et al. (2021) |

## Extending NIFE

**A new counterfactual method** derives from `CounterfactualExplainer` and only finds candidates;
the base class confirms them with the model, enforces the desired outcome and the immutable
features, ranks, selects and evaluates them:

```python
from nife.counterfactual import CounterfactualExplainer


class NearestExamples(CounterfactualExplainer):
    """Every example of the data is a candidate; the closest valid ones win."""

    def _search(self, x, prediction, desired, rng):
        return self.data.assign(source="data"), {}  # candidates and extra fields of the explanation
```

**A new metric** is a function of an `EvaluationContext`, registered once and reported by every
evaluation from then on:

```python
from nife.counterfactual import register_metric


@register_metric("max_change", better="lower", reference="—")
def max_change(ctx):
    """Largest scaled change of a single feature."""
    numeric_cf, numeric_x = ctx.cf_points[0], ctx.x_points[0][ctx.owner]
    return abs(numeric_cf - numeric_x).max(axis=1)
```

**A new kind of explanation** is a new subpackage next to `nife.counterfactual`, with its
methods and metrics; shared tools, such as the Gower distance (`nife.distance`), stay at the top
level. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Examples

The [`examples/`](examples) folder has notebooks that run on datasets bundled with scikit-learn:

- [`classification.ipynb`](examples/classification.ipynb) — wine cultivars: an explanation in
  detail, desired classes, selection and evaluation;
- [`regression.ipynb`](examples/regression.ipynb) — diabetes progression: desired outcomes,
  immutable features, counterfactuals from the rules and the fidelity of the tree.

To run them: `pip install "nife[examples]"`.

## How to cite

If you use NIFE in your research, please cite the paper that introduced its counterfactual
method:

> Silva, J., Aguilar, E. and Salvini, R. (2026). Unveiling Concise Counterfactual Explanations
> and Feature Relevance in Machine Learning through Decision Trees. In *Proceedings of the 18th
> International Conference on Agents and Artificial Intelligence (ICAART 2026) — Volume 4*,
> pages 2941–2948. SciTePress. https://doi.org/10.5220/0014338700004052

```bibtex
@inproceedings{silva2026unveiling,
  author    = {Silva, Jurandir and Aguilar, Eduardo and Salvini, Rogerio},
  title     = {Unveiling Concise Counterfactual Explanations and Feature Relevance in Machine Learning through Decision Trees},
  booktitle = {Proceedings of the 18th International Conference on Agents and Artificial Intelligence (ICAART 2026) - Volume 4},
  pages     = {2941--2948},
  year      = {2026},
  publisher = {SciTePress},
  doi       = {10.5220/0014338700004052},
  isbn      = {978-989-758-796-2},
  issn      = {2184-433X}
}
```

To cite the software itself, use the "Cite this repository" button on GitHub (from
[`CITATION.cff`](CITATION.cff)) or:

```bibtex
@software{nife,
  author  = {Silva, Jurandir},
  title   = {{NIFE}: {NIFE} Is a Framework for Explanations},
  url     = {https://github.com/jurandirs/nife},
  version = {0.1.0},
  year    = {2026}
}
```

## Contributing

Bug reports, questions and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

NIFE is released under the [MIT License](LICENSE).

## References

- Dandl, S., Molnar, C., Binder, M., & Bischl, B. (2020). Multi-objective counterfactual explanations. *Parallel Problem Solving from Nature (PPSN XVI)*.
- Gower, J. C. (1971). A general coefficient of similarity and some of its properties. *Biometrics*, 27(4), 857–871.
- Guidotti, R. (2022). Counterfactual explanations and how to find them: literature review and benchmarking. *Data Mining and Knowledge Discovery*. https://doi.org/10.1007/s10618-022-00831-6
- Laugel, T., Lesot, M.-J., Marsala, C., Renard, X., & Detyniecki, M. (2017). Inverse classification for comparison-based interpretability in machine learning. *arXiv:1712.08443*.
- Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. *FAT\* '20*. https://doi.org/10.1145/3351095.3372850
- Pawelczyk, M., Bielawski, S., van den Heuvel, J., Richter, T., & Kasneci, G. (2021). CARLA: A Python library to benchmark algorithmic recourse and counterfactual explanation algorithms. *NeurIPS Datasets and Benchmarks Track*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD*, 1135–1144.
- Silva, J., Aguilar, E., & Salvini, R. (2026). Unveiling concise counterfactual explanations and feature relevance in machine learning through decision trees. *ICAART 2026*, 2941–2948. https://doi.org/10.5220/0014338700004052
- Wachter, S., Mittelstadt, B., & Russell, C. (2018). Counterfactual explanations without opening the black box: automated decisions and the GDPR. *Harvard Journal of Law & Technology*, 31(2), 841–887.
