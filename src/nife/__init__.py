"""NIFE — NIFE Is a Framework for Explanations.

A framework for explaining the predictions of machine-learning models. Each kind of
explanation is a subpackage, with the methods that produce it and the metrics that
evaluate it:

- `nife.counterfactual` — counterfactual explanations: what would have to change
  for the model to predict something else. Methods: `TreeCounterfactualExplainer`
  (surrogate decision trees; Silva, Aguilar and Salvini, ICAART 2026). Metrics:
  `evaluate_counterfactuals`, for counterfactuals found by any method.

Shared tools, such as the Gower distance (`nife.distance`), live at the top level.

>>> from nife.counterfactual import TreeCounterfactualExplainer
>>> explainer = TreeCounterfactualExplainer(model, data)  # doctest: +SKIP
>>> explanation = explainer.explain(x)  # doctest: +SKIP
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nife")
except PackageNotFoundError:  # not installed
    __version__ = "unknown"

__all__ = ["__version__"]
