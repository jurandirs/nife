# Contributing to NIFE

Thank you for your interest in NIFE! Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/jurandirs/nife/issues).

## Development setup

```bash
git clone https://github.com/jurandirs/nife.git
cd nife
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The package lives in `src/nife`, so the tests always run against the installed package.

## Tests

```bash
pytest                       # everything: the unit tests, the README and the notebooks
pytest tests/counterfactual  # only the unit tests, faster
```

Every change should keep the tests passing and, for new behavior, add tests. The metrics are
checked against values computed by hand in `tests/counterfactual/test_metrics.py`.

`tests/test_documentation.py` runs the Python code of the README, checking that the output it shows
is the one its code prints, and runs every notebook of `notebooks/` from top to bottom (with
`nbclient` and `ipykernel`, installed by the `dev` extra).

## How NIFE is organized

```
src/nife/
├── base.py                   Explainer and Explanation: the roots of every kind (plots)
├── plotting.py               the look shared by every plot
├── distance.py               shared tools (Gower distance)
└── counterfactual/           one subpackage per kind of explanation
    ├── base.py               CounterfactualExplainer: what every method shares
    ├── outcomes.py           desired outcomes
    ├── selection.py          selection of counterfactuals
    ├── metrics.py            evaluation metrics (registry)
    ├── plots.py              the plot of every counterfactual explanation
    └── tree/                 one subpackage per method
```

- **A new method** for an existing kind of explanation goes in its own subpackage (for example
  `nife/counterfactual/<method>/`) and derives from the base class of that kind; for
  counterfactuals, it implements `_search` and, if it returns more information, an explanation
  class derived from `CounterfactualExplanation`.
- **A new metric** is a function registered with `register_metric` in the kind's `metrics.py`,
  with its reference and a test against values computed by hand.
- **A new plot** is a method `plot_<kind>(..., ax=None)` of an explanation class that returns
  the matplotlib Axes. It draws with the helpers of `nife/plotting.py` (`styled`, `new_axes`,
  `set_titles`, `clean_axes`, `legend_below`, `COLORS`), so that every plot looks the same, and it
  is tested by drawing it. Keep the plots few: one per question an explanation answers.
- **A new kind of explanation** is a new subpackage next to `counterfactual/`, with a base class,
  its methods, its metrics and its plots. Its classes derive from `nife.base.Explainer` and
  `nife.base.Explanation`. Tools that several kinds use go to the top level.

Tests mirror this layout under `tests/`, and the notebooks under `notebooks/`:

```
notebooks/
├── counterfactual/                                  one folder per kind of explanation
│   ├── counterfactual_explanations.ipynb            what every counterfactual method shares
│   ├── evaluation.ipynb                             the metrics
│   ├── TreeCounterfactualExplainer.ipynb            one notebook per method, named after its class
│   └── TreeCounterfactualExplainer_neighborhood.ipynb
├── plots.ipynb
└── extending.ipynb
docs/
└── images/                                          images of the README
```

## Notebooks

The notebooks are NIFE's documentation, and the tests run them. A new method gets a notebook
named after its class in the folder of its kind of explanation, ending with how to cite the work
that introduced it and its references, and a row in the tables of the README and of the kind's
common notebook. Each notebook must run from top to bottom on its own.

After a change, run the notebooks again so that their outputs show the current behavior:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/counterfactual/TreeCounterfactualExplainer.ipynb
```

## Pull requests

- Keep each pull request focused on one change and describe what it does and why.
- Follow the style of the surrounding code; public functions and classes have docstrings.
- Add an entry to `CHANGELOG.md` under "Unreleased".

## Releasing (maintainers)

1. Update the version in `pyproject.toml`, `CITATION.cff` and the BibTeX entry of `README.md`, and
   the date in `CHANGELOG.md`.
   PyPI shows only the README: make its links and images absolute URLs of the repository.
2. Create a GitHub release with a tag `vX.Y.Z`; the `publish` workflow builds the package and
   uploads it to PyPI (trusted publishing must be configured for the repository on PyPI).
