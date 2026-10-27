"""Which predictions count as counterfactual outcomes for an instance.

The same definition is used to generate counterfactuals and to evaluate them.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

MODES = ("classification", "regression")


def check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be 'classification' or 'regression', got {mode!r}")


def check_desired(desired, mode: str, classes: Optional[Sequence] = None) -> None:
    """Raise ValueError unless `desired` is a valid desired outcome for `mode`.

    Classification: None or a class (one of `classes`, when given). Regression:
    None, "higher", "lower" or an interval ``(low, high)`` with ``low <= high``,
    where either bound may be None (open).
    """
    check_mode(mode)
    if desired is None:
        return
    if mode == "classification":
        if classes is not None and desired not in list(classes):
            raise ValueError(f"desired must be one of the classes {list(classes)}, got {desired!r}")
        return
    if isinstance(desired, str):
        if desired in ("higher", "lower"):
            return
    elif isinstance(desired, (tuple, list)) and len(desired) == 2:
        low, high = desired
        if (low is not None or high is not None) and (low is None or high is None or low <= high):
            return
    raise ValueError(
        "in regression, desired must be None, 'higher', 'lower' or an interval (low, high) "
        f"with low <= high (either may be None), got {desired!r}"
    )


def is_counterfactual(prediction, reference, mode: str, desired=None, tolerance: float = 0.05) -> np.ndarray:
    """Elementwise: is `prediction` a counterfactual outcome for an instance predicted `reference`?

    Classification:

    - ``desired=None``: any class other than `reference`;
    - ``desired=c``: the class ``c``, when `reference` is not already ``c``.

    Regression (`tolerance` is relative to ``|reference|``):

    - ``desired=None``: deviates from `reference` by more than `tolerance`, either way;
    - ``desired="higher"`` / ``"lower"``: deviates by more than `tolerance` in that direction;
    - ``desired=(low, high)``: lies in the interval (bounds included; None is open),
      when `reference` does not.

    `prediction` and `reference` are broadcast against each other.

    Examples
    --------
    >>> is_counterfactual([11.0, 4.1], 4.0, "regression")
    array([ True, False])
    >>> is_counterfactual(["a", "b", "c"], "a", "classification", desired="c")
    array([False, False,  True])
    """
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if mode == "classification":
        if desired is None:
            return prediction != reference
        return (prediction == desired) & (reference != desired)

    prediction, reference = prediction.astype(float), reference.astype(float)
    margin = tolerance * np.abs(reference)
    if desired is None:
        return np.abs(prediction - reference) > margin
    if isinstance(desired, str):
        return prediction - reference > margin if desired == "higher" else reference - prediction > margin
    low, high = desired

    def inside(values: np.ndarray) -> np.ndarray:
        result = np.ones(values.shape, dtype=bool)
        if low is not None:
            result &= values >= low
        if high is not None:
            result &= values <= high
        return result

    return inside(prediction) & ~inside(reference)
