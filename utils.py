from numpy.typing import NDArray
from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_points(points: Tuple[NDArray], rescale=1):
    points_stacked = np.concatenate(points, axis=0)
    max_norm = np.linalg.norm(points_stacked, axis=1).max()
    points = [points_i / max_norm * rescale for points_i in points]
    return points


def pose_pretty_string(rotation, translation):
    euler_angles = R.from_matrix(rotation).as_euler("xyz", degrees=True)
    result = f"Tra: x: {translation[0]:.3f}, y: {translation[1]:.3f}, z: {translation[2]:.3f}\n"
    result += f"Rot: x: {euler_angles[0]:.3f}, y: {euler_angles[1]:.3f}, z: {euler_angles[2]:.3f}\n"
    return result


if __name__ == "__main__":
    pass
