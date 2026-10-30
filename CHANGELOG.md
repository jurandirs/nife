# Changelog

All notable changes to NIFE are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-30

First public release, with counterfactual explanations (`nife.counterfactual`).

- `CounterfactualExplainer`: base class of the counterfactual methods — desired outcomes,
  immutable features, confirmation by the model, selection, many instances and evaluation.
  Counterfactuals follow the definition of the literature (the smallest change that changes the
  prediction), and each one has its counterpart: the closest example of the data that the model
  gives the same outcome (`counterpart`, `counterpart_distance`).
- `TreeCounterfactualExplainer`: counterfactual explanations and feature relevance with a
  surrogate decision tree (Silva, Aguilar and Salvini, ICAART 2026) — tree of depth 5 by default
  (`max_depth`), the counterfactual rule chosen by going up the path (Algorithm 1) and checked
  with the model (`crossed_rule_confirmed`), counterfactuals with the fewest changes (a single
  feature across a decision of the path, or the fewest features copied from the evidence, as in
  NICE), the evidence of the rule (`explanation.evidence`, among `explanation.evidence_candidates`),
  an adaptive neighborhood and the fidelity of the tree. Its neighborhood can be extracted
  (`explanation.neighborhood`, `sample_neighborhood`) or replaced by your own (`neighborhood=` in
  `explain`, `sampler=`).
- Plots: every explanation draws itself (`explanation.plot()`, `plot("<kind>")`, `plot_kinds()`),
  with a shared look (`nife.plotting`). Counterfactual explanations draw what a counterfactual
  changes (`plot_changes`), and those of the surrogate tree also the relevant features
  (`plot_importances`) and the tree (`plot_tree`, compact, with the information of the trees of the
  paper; `path_only=True` for the decision path only). The tree is exported as Graphviz DOT, PMML 4.4
  or scikit-learn's text rules (`export_tree`). `nife.base` holds the roots of every kind of
  explanation, `Explainer` and `Explanation`.
- `select_counterfactuals`: the closest or the nondominated counterfactuals of each instance.
- Notebooks (`notebooks/`): one per kind of explanation and per method, run by the tests.
- `evaluate_counterfactuals`: coverage, validity, proximity, sparsity, redundancy,
  implausibility, yNN, diversity and constraint violations, for counterfactuals found by any
  method; new metrics with `register_metric`.
