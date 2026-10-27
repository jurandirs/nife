import numpy as np
import pandas as pd
import pytest

from nife.distance import GowerSpace, distance_blocks, paired_distances


def test_gower_distance(toy):
    data = toy[0]
    space = GowerSpace(data, ["x1", "x2", "color"], ["color"])
    a = space.points(pd.DataFrame({"x1": [2, 8], "x2": [2, 6], "color": ["red", "blue"]}))
    b = space.points(pd.DataFrame({"x1": [8, 8], "x2": [6, 6], "color": ["blue", "blue"]}))
    # x1 and x2 are 10 wide: (6/10 + 4/10 + 1) / 3, and 0
    assert paired_distances(a, b, 3) == pytest.approx([2 / 3, 0])
    _, matrix = next(distance_blocks(a, b, 3))
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix[:, 0], [2 / 3, 0])


def test_constant_features_do_not_count():
    data = pd.DataFrame({"x": [1.0, 1.0], "y": [0.0, 10.0]})
    space = GowerSpace(data, ["x", "y"])
    a = space.points(pd.DataFrame({"x": [5.0], "y": [0.0]}))
    b = space.points(pd.DataFrame({"x": [1.0], "y": [5.0]}))
    assert paired_distances(a, b, 2) == pytest.approx([0.25])
