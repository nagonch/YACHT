from numpy.typing import NDArray
from typing import Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation as R
from structs import HandEyeCalibrationResult
import yaml
import logging

with open("config.yaml") as file:
    CONFIG = yaml.safe_load(file)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
LOGGER = logging.getLogger(__name__)


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
        f"Rot: x: {rotation[0]:.5f}°, y: {rotation[1]:.5f}°, z: {rotation[2]:.5f}°\n"
    )
    return result


def estimate_hand_eye_error(calib_result: HandEyeCalibrationResult) -> Tuple[NDArray]:
    target_to_base_rotations = calib_result.target_to_base_rotation
    target_to_base_translations = calib_result.target_to_base_translation

    R_ref = R.from_quat(
        R.from_matrix(target_to_base_rotations).as_quat().mean(axis=0)
    ).as_matrix()
    R_rel = R_ref.T @ target_to_base_rotations
    theta_min = np.arccos((np.trace(R_rel, axis1=1, axis2=2) - 1) / 2)
    theta_min = np.nan_to_num(theta_min)
    rotation_error = np.rad2deg(theta_min.mean())

    t_ref = target_to_base_translations.mean(axis=0)
    translation_error = np.linalg.norm(
        target_to_base_translations - t_ref, axis=1
    ).std()

    return rotation_error, translation_error


def average_target_pose(target_to_base_rotation, target_to_base_translation):
    avg_rotation = R.from_matrix(target_to_base_rotation).mean().as_matrix()
    avg_translation = target_to_base_translation.mean(axis=0)
    result_pose = np.eye(4)
    result_pose[:3, 3] = avg_translation
    result_pose[:3, :3] = avg_rotation
    return result_pose


if __name__ == "__main__":
    pass
