"""Counterfactual explanations and feature relevance with a surrogate decision tree.

The method of Silva, Aguilar and Salvini (2026), "Unveiling Concise Counterfactual
Explanations and Feature Relevance in Machine Learning through Decision Trees",
ICAART 2026, https://doi.org/10.5220/0014338700004052.
"""

from .explainer import (
    ADAPTIVE_NEIGHBORHOODS,
    TreeCounterfactualExplainer,
    TreeCounterfactualExplanation,
    fidelity_table,
)
from .search import counterfactuals_from_rules
from .surrogate import DEFAULT_TREE_PARAMS, Rule

__all__ = [
    "ADAPTIVE_NEIGHBORHOODS",
    "DEFAULT_TREE_PARAMS",
    "Rule",
    "TreeCounterfactualExplainer",
    "TreeCounterfactualExplanation",
    "counterfactuals_from_rules",
    "fidelity_table",
]
