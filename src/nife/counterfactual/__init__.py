"""Counterfactual explanations: what would have to change for the model to predict something else.

Methods
-------
TreeCounterfactualExplainer
    Surrogate decision trees (Silva, Aguilar and Salvini, 2026); see `nife.counterfactual.tree`.

Every method derives from `CounterfactualExplainer`, which handles desired outcomes,
immutable features, the selection of the counterfactuals, many instances at once and
their evaluation; a new method only implements the search for candidates.

Common tools
------------
is_counterfactual
    Which predictions are counterfactual outcomes (with a desired outcome or not).
select_counterfactuals
    The closest or the nondominated counterfactuals of each instance.
evaluate_counterfactuals, METRICS, register_metric
    Metrics for counterfactuals found by any method, and how to add new ones.
"""

from .base import CounterfactualExplainer, CounterfactualExplanation
from .metrics import METRICS, EvaluationContext, Metric, evaluate_counterfactuals, register_metric
from .outcomes import is_counterfactual
from .selection import select_counterfactuals
from .tree import TreeCounterfactualExplainer, TreeCounterfactualExplanation

__all__ = [
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
