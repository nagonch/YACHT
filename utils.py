from numpy.typing import NDArray
from typing import Tuple
import numpy as np


def normalize_points(points: Tuple[NDArray], rescale=1):
    points_stacked = np.concatenate(points, axis=0)
    max_norm = np.linalg.norm(points_stacked, axis=1).max()
    points = [points_i / max_norm * rescale for points_i in points]
    return points


if __name__ == "__main__":
    N = 100
    M = 50
    K = 25
    ar1 = np.random.uniform(size=(N, 3)) * 100
    ar2 = np.random.uniform(size=(M, 3)) * 300
    ar3 = np.random.uniform(size=(K, 3)) * 5e3
    points = normalize_points((ar1, ar2, ar3))
    print(points)
