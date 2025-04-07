from numpy.typing import NDArray
from typing import Tuple
import numpy as np


def normalize_points(points: Tuple[NDArray], rescale=1):
    points_stacked = np.concatenate(points, axis=0)
    max_norm = np.linalg.norm(points_stacked, axis=1).max()
    points = [points_i / max_norm * rescale for points_i in points]
    return points


if __name__ == "__main__":
    pass
