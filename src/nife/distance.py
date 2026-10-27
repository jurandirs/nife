"""Gower distance (Gower, 1971), which mixes numeric and categorical features.

Numeric differences are scaled by the feature's range in the reference data;
categorical features count 0 when equal and 1 when different; the distance is
the mean over the features.

Gower, J. C. (1971). A general coefficient of similarity and some of its
properties. Biometrics, 27(4), 857-871.
"""

from __future__ import annotations

from typing import Iterator, Sequence, Tuple

import numpy as np
import pandas as pd

# Largest number of entries of a distance matrix computed at once
_BLOCK_SIZE = 2_000_000

# Points as Gower sees them: (scaled numeric values, categorical codes)
Points = Tuple[np.ndarray, np.ndarray]


class GowerSpace:
    """Turns frames into points for the Gower distance, with ranges taken from `data`."""

    def __init__(self, data: pd.DataFrame, feature_names: Sequence[str], categorical_features: Sequence[str] = ()):
        self.numeric = [f for f in feature_names if f not in categorical_features]
        self.categorical = [f for f in feature_names if f in categorical_features]
        self.n_features = len(feature_names)
        values = data[self.numeric].to_numpy(dtype=float)
        ranges = values.max(axis=0) - values.min(axis=0)
        # Constant features contribute nothing to the distance
        self.scale = np.divide(1.0, ranges, out=np.zeros_like(ranges), where=ranges > 0)
        self.codes = {f: {} for f in self.categorical}

    def points(self, frame: pd.DataFrame) -> Points:
        numeric = frame[self.numeric].to_numpy(dtype=float) * self.scale
        categorical = np.empty((len(frame), len(self.categorical)), dtype=int)
        for j, feature in enumerate(self.categorical):
            codes = self.codes[feature]
            categorical[:, j] = [codes.setdefault(value, len(codes)) for value in frame[feature]]
        return numeric, categorical


def select_points(points: Points, rows) -> Points:
    return points[0][rows], points[1][rows]


def paired_distances(a: Points, b: Points, n_features: int) -> np.ndarray:
    """Distance between the i-th point of `a` and the i-th point of `b`."""
    return (np.abs(a[0] - b[0]).sum(axis=1) + (a[1] != b[1]).sum(axis=1)) / n_features


def distance_blocks(a: Points, b: Points, n_features: int) -> Iterator[Tuple[slice, np.ndarray]]:
    """Distances between every point of `a` and every point of `b`, a block of rows of `a` at a time."""
    step = max(1, _BLOCK_SIZE // max(1, len(b[0])))
    for start in range(0, len(a[0]), step):
        rows = slice(start, start + step)
        (a_num, a_cat), (b_num, b_cat) = select_points(a, rows), b
        distances = np.zeros((len(a_num), len(b_num)))
        for j in range(a_num.shape[1]):
            distances += np.abs(a_num[:, j, None] - b_num[None, :, j])
        for j in range(a_cat.shape[1]):
            distances += a_cat[:, j, None] != b_cat[None, :, j]
        yield rows, distances / n_features
