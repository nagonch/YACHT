import numpy as np
from PIL import Image
import yaml
import os
import cv2
from typing import List, Tuple
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from visualize import (
    viusalize_target_to_cam_poses_2D,
    viusalize_target_to_cam_poses_3D,
    visualize_hand_eye_poses,
)

with open("config.yaml") as file:
    CONFIG = yaml.safe_load(file)


@dataclass
class CameraParameters:
    rms_error: float
    intrinsics: NDArray  # [3, 3]
    distortion_coeffs: NDArray  # [5]
    target_to_cam_rotation: NDArray  # [N, 3, 3]
    target_to_cam_translation: NDArray  # [N, 3]
    image_w: int
    image_h: int


@dataclass
class HandEyeCalibrationResult:
    arm_to_base_rotation: NDArray
    arm_to_base_translation: NDArray
    cam_to_arm_rotation: NDArray
    cam_to_arm_translation: NDArray
    cam_to_base_rotation: NDArray
    cam_to_base_translation: NDArray
    target_to_base_rotation: NDArray
    target_to_base_translation: NDArray


def detect_corners(
    images: List[NDArray],
    checkerboard_size: float = 28.5e-3,
    checkerboard_dims: Tuple[int] = (6, 8),
) -> Tuple[List[NDArray], NDArray, List[NDArray]]:

    corners3D = np.zeros(
        (checkerboard_dims[0] * checkerboard_dims[1], 3), dtype=np.float32
    )
    corners_2D = np.meshgrid(
        np.arange(checkerboard_dims[0]), np.arange(checkerboard_dims[1])
    )
    corners_2D = np.stack(corners_2D, axis=-1).reshape(-1, 2)
    corners3D[:, :2] = corners_2D * checkerboard_size
    detected_images = []
    detected_corners = []
    for image in images:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_detected, corners = cv2.findChessboardCorners(
            grayscale_image, checkerboard_dims
        )
        if is_detected:
            corners = cv2.cornerSubPix(
                grayscale_image,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )[:, 0, :]
            detected_images.append(image)
            detected_corners.append(corners)
    return (
        detected_corners,
        corners3D,
        detected_images,
    )


def get_camera_parameters(
    detected_corners: List[NDArray],
    corners3D: List[NDArray],
    detected_images: List[NDArray],
) -> CameraParameters:
    n_images = len(detected_images)
    height, width = detected_images[0].shape[:2]
    cam_calib_params = cv2.calibrateCamera(
        np.stack(
            [
                corners3D,
            ]
            * n_images
        ),
        detected_corners,
        (width, height),
        cameraMatrix=None,
        distCoeffs=None,
    )
    cam_calib_params = CameraParameters(*cam_calib_params, width, height)
    cam_calib_params.target_to_cam_rotation = np.stack(
        cam_calib_params.target_to_cam_rotation
    )[:, :, 0]
    cam_calib_params.target_to_cam_rotation = R.from_euler(
        "xyz", cam_calib_params.target_to_cam_rotation
    ).as_matrix()
    cam_calib_params.target_to_cam_translation = np.stack(
        cam_calib_params.target_to_cam_translation
    )[:, :, 0]
    cam_calib_params.distortion_coeffs = cam_calib_params.distortion_coeffs[0]
    return cam_calib_params


def get_eye_to_hand_transformation(
    arm_to_base_rotation: NDArray,
    arm_to_base_translation: NDArray,
    camera_parameters: CameraParameters,
) -> HandEyeCalibrationResult:
    cam_to_arm_rotation, cam_to_arm_translation = cv2.calibrateHandEye(
        arm_to_base_rotation,
        arm_to_base_translation,
        camera_parameters.target_to_cam_rotation,
        camera_parameters.target_to_cam_translation,
    )
    cam_to_arm_translation = cam_to_arm_translation.reshape(-1)
    cam_to_base_rotation = arm_to_base_rotation @ cam_to_arm_rotation
    cam_to_base_translation = arm_to_base_translation + cam_to_arm_translation
    target_to_base_rotation = (
        cam_to_base_rotation @ camera_parameters.target_to_cam_rotation
    )
    target_to_base_translation = (
        cam_to_base_translation + camera_parameters.target_to_cam_translation
    )
    result = HandEyeCalibrationResult(
        arm_to_base_rotation,
        arm_to_base_translation,
        cam_to_arm_rotation,
        cam_to_arm_translation,
        cam_to_base_rotation,
        cam_to_base_translation,
        target_to_base_rotation,
        target_to_base_translation,
    )
    return result


if __name__ == "__main__":
    # Check folder structure
    data_folder = CONFIG["data-folder"]
    assert os.path.exists(data_folder), f"Data folder '{data_folder}' does not exist."
    assert os.path.exists(
        f"{data_folder}/images"
    ), f"Images folder '{data_folder}/images' does not exist."
    assert os.path.exists(
        f"{data_folder}/poses.txt"
    ), f"Poses file '{data_folder}/poses.txt' not found"

    # Load poses
    arm_poses = np.loadtxt(f"{data_folder}/poses.txt").reshape(-1, 4, 4)
    arm_to_base_translation = arm_poses[:, :3, -1] * 1e-3
    arm_to_base_rotation = arm_poses[:, :3, :3]

    # Load images
    image_filenames = sorted(os.listdir(f"{data_folder}/images"))
    images = [
        np.array(Image.open(f"{data_folder}/images/{image_fname}"))
        for image_fname in image_filenames
    ]
    # Detecting corners
    detected_corners, corners3D, detected_images = detect_corners(
        images,
        checkerboard_dims=CONFIG["checkerboard-dims"],
        checkerboard_size=CONFIG["checkerboard-size"],
    )
    # Getting camera calibration parameters
    camera_parameters = get_camera_parameters(
        detected_corners, corners3D, detected_images
    )
    # Getting hand-in-eye calibration
    hand_eye_calibration_result = get_eye_to_hand_transformation(
        arm_to_base_rotation, arm_to_base_translation, camera_parameters
    )
    if CONFIG["verbose"]:
        viusalize_target_to_cam_poses_2D(
            images, camera_parameters, detected_corners, "test"
        )
        viusalize_target_to_cam_poses_3D(images, camera_parameters)
        visualize_hand_eye_poses(images, camera_parameters, hand_eye_calibration_result)
    # TODO
    # Figure out the viser frames problem
    # Remove shit code
    # Calculate hand-eye calibration error
    # File standard for camera poses (ORDER MATTERS)
    # File standard for calibration results
    # Split code into functions
    # Readme
