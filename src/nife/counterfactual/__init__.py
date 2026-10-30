"""Counterfactual explanations: the smallest change that would make the model predict something else.

A counterfactual is an instance close to the one explained, with as few changed
features as possible, that the model gives another outcome or a desired one
(Wachter et al., 2018; Molnar, 2022). NIFE gives each counterfactual its
counterpart: the closest real example of the data with the same outcome.

Methods
-------
TreeCounterfactualExplainer
    Surrogate decision trees (Silva, Aguilar and Salvini, 2026), with the
    counterfactual rule and its evidence; see `nife.counterfactual.tree`.

Every method derives from `CounterfactualExplainer`, which handles desired outcomes,
immutable features, the selection of the counterfactuals, their counterparts, many
instances at once and their evaluation; a new method only implements the search for
candidate counterfactuals.

Common tools
------------
is_counterfactual
    Which predictions are counterfactual outcomes (with a desired outcome or not).
select_counterfactuals
    The closest or the nondominated counterfactuals of each instance.
evaluate_counterfactuals, METRICS, register_metric
    Metrics for counterfactuals found by any method, and how to add new ones.
plots.plot_changes
    What a counterfactual changes, for counterfactuals found by any method; every
    explanation draws it with ``explanation.plot()``.
"""

from . import plots
from .base import CounterfactualExplainer, CounterfactualExplanation
from .metrics import METRICS, EvaluationContext, Metric, evaluate_counterfactuals, register_metric
from .outcomes import is_counterfactual
from .selection import select_counterfactuals
from .tree import TreeCounterfactualExplainer, TreeCounterfactualExplanation

__all__ = [
    "plots",
    "CounterfactualExplainer",
    "CounterfactualExplanation",
    "EvaluationContext",
    "METRICS",
    "Metric",
    "TreeCounterfactualExplainer",
    "TreeCounterfactualExplanation",
    "evaluate_counterfactuals",
    "is_counterfactual",
    "register_metric",
    "select_counterfactuals",
]
