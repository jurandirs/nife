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
pytest
```

Every change should keep the tests passing and, for new behavior, add tests. The metrics are
checked against values computed by hand in `tests/test_metrics.py`.

## How NIFE is organized

```
src/nife/
├── distance.py               shared tools (Gower distance)
└── counterfactual/           one subpackage per kind of explanation
    ├── base.py               CounterfactualExplainer: what every method shares
    ├── outcomes.py           desired outcomes
    ├── selection.py          selection of counterfactuals
    ├── metrics.py            evaluation metrics (registry)
    └── tree/                 one subpackage per method
```

- **A new method** for an existing kind of explanation goes in its own subpackage (for example
  `nife/counterfactual/<method>/`) and derives from the base class of that kind; for
  counterfactuals, it implements `_search` and, if it returns more information, an explanation
  class derived from `CounterfactualExplanation`.
- **A new metric** is a function registered with `register_metric` in the kind's `metrics.py`,
  with its reference and a test against values computed by hand.
- **A new kind of explanation** is a new subpackage next to `counterfactual/`, with a base class,
  its methods and its metrics. Tools that several kinds use go to the top level.

Tests mirror this layout under `tests/`.

## Pull requests

- Keep each pull request focused on one change and describe what it does and why.
- Follow the style of the surrounding code; public functions and classes have docstrings.
- Add an entry to `CHANGELOG.md` under "Unreleased".

## Releasing (maintainers)

1. Update the version in `pyproject.toml` and `CITATION.cff`, and the date in `CHANGELOG.md`.
2. Create a GitHub release with a tag `vX.Y.Z`; the `publish` workflow builds the package and
   uploads it to PyPI (trusted publishing must be configured for the repository on PyPI).
