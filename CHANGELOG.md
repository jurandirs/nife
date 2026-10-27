# Changelog

All notable changes to NIFE are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First public release, with counterfactual explanations (`nife.counterfactual`).

- `CounterfactualExplainer`: base class of the counterfactual methods — desired outcomes,
  immutable features, confirmation by the model, selection, many instances and evaluation.
- `TreeCounterfactualExplainer`: counterfactual explanations and feature relevance with a
  surrogate decision tree (Silva, Aguilar and Salvini, ICAART 2026) — tree of depth 5 by default
  (`max_depth`), decision to cross chosen by going up the path (Algorithm 1) — with single-feature
  counterfactuals built from the rules, an adaptive neighborhood and the fidelity of the tree;
  `plot_tree` (matplotlib).
- `select_counterfactuals`: the closest or the nondominated counterfactuals of each instance.
- `evaluate_counterfactuals`: coverage, validity, proximity, sparsity, redundancy,
  implausibility, yNN, diversity and constraint violations, for counterfactuals found by any
  method; new metrics with `register_metric`.
