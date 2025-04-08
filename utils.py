from numpy.typing import NDArray
from typing import Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation as R
from structs import HandEyeCalibrationResult
import yaml

with open("config.yaml") as file:
    CONFIG = yaml.safe_load(file)


def normalize_points(points: Tuple[NDArray], rescale: float = 1.0) -> List[NDArray]:
    points_stacked = np.concatenate(points, axis=0)
    max_norm = np.linalg.norm(points_stacked, axis=1).max()
    points = [points_i / max_norm * rescale for points_i in points]
    return points


def pose_pretty_string(
    rotation: NDArray, translation: NDArray, convert_from_matrix: bool = True
) -> str:
    if convert_from_matrix:
        rotation = R.from_matrix(rotation).as_euler("xyz", degrees=True)
    result = f"Tra: x: {translation[0]:.3f} m, y: {translation[1]:.3f} m, z: {translation[2]:.3f} m\n"
    result += (
        f"Rot: x: {rotation[0]:.3f}°, y: {rotation[1]:.3f}°, z: {rotation[2]:.3f}°\n"
    )
    return result


def estimate_hand_eye_error(calib_result: HandEyeCalibrationResult) -> Tuple[NDArray]:
    target_to_base_rotations = calib_result.target_to_base_rotation
    target_to_base_translations = calib_result.target_to_base_translation
    mean_target_coord = target_to_base_translations.mean(axis=0)
    mean_target_rotation_quat = (
        R.from_matrix(target_to_base_rotations).as_quat().mean(axis=0)
    )
    mean_target_rotation = R.from_quat(mean_target_rotation_quat).as_matrix()
    mean_rotation_error_quat = (
        R.from_matrix(
            mean_target_rotation @ target_to_base_rotations.transpose(0, 2, 1)
        )
        .as_quat()
        .mean(axis=0)
    )
    mean_rotation_error = R.from_quat(mean_rotation_error_quat).as_euler(
        "xyz", degrees=True
    )
    mean_translation_error = np.linalg.norm(
        (target_to_base_translations - mean_target_coord) ** 2,
        axis=0,
    )
    return mean_rotation_error, mean_translation_error


if __name__ == "__main__":
    pass
