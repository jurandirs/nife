"""NIFE — NIFE Is a Framework for Explanations.

A framework for explaining the predictions of machine-learning models. Each kind of
explanation is a subpackage, with the methods that produce it and the metrics that
evaluate it:

- `nife.counterfactual` — counterfactual explanations: the smallest change that
  would make the model predict something else, with a real counterpart from the data.
  Methods: `TreeCounterfactualExplainer` (surrogate decision trees; Silva, Aguilar
  and Salvini, ICAART 2026). Metrics: `evaluate_counterfactuals`, for
  counterfactuals found by any method.

Every explainer derives from `nife.base.Explainer` and every explanation from
`nife.base.Explanation`, which draws its plots: ``explanation.plot()``,
``explanation.plot("<kind>")`` and ``explanation.plot_kinds()``, all in the look of
`nife.plotting`. Shared tools, such as the Gower distance (`nife.distance`), live at
the top level.

>>> from nife.counterfactual import TreeCounterfactualExplainer
>>> explainer = TreeCounterfactualExplainer(model, data)  # doctest: +SKIP
>>> explanation = explainer.explain(x)  # doctest: +SKIP
>>> explanation.plot()  # doctest: +SKIP
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nife")
except PackageNotFoundError:  # not installed
    __version__ = "unknown"

__all__ = ["__version__"]
