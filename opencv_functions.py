import numpy as np
import cv2
from typing import List, Tuple
from numpy.typing import NDArray
from tqdm import tqdm
from structs import CameraParameters, HandEyeCalibrationResult


def detect_corners(
    images: List[NDArray],
    chessboard_size: float = 28.5e-3,
    chessboard_dims: Tuple[int] = (6, 8),
) -> Tuple[List[NDArray], NDArray, List[NDArray]]:
    chessboard_width, chessboard_height = chessboard_dims
    corners3D = np.zeros((chessboard_width * chessboard_height, 3), dtype=np.float32)
    corners_2D = np.meshgrid(np.arange(chessboard_width), np.arange(chessboard_height))
    corners_2D = np.stack(corners_2D, axis=-1).reshape(-1, 2)
    corners3D[:, :2] = corners_2D * chessboard_size
    detected_images = []
    detected_corners = []
    detected_inds = []
    for i, image in enumerate(tqdm(images)):
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_detected, corners = cv2.findChessboardCorners(
            grayscale_image,
            (chessboard_width, chessboard_height),
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
            detected_inds.append(i)
            detected_images.append(image)
            detected_corners.append(corners)
    return (
        detected_inds,
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
        [cv2.Rodrigues(rvec)[0] for rvec in cam_calib_params.target_to_cam_rotation]
    )
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
    cam_to_base_translation = arm_to_base_translation + np.matvec(
        arm_to_base_rotation, cam_to_arm_translation
    )
    target_to_base_rotation = (
        cam_to_base_rotation @ camera_parameters.target_to_cam_rotation
    )
    target_to_base_translation = cam_to_base_translation + np.matvec(
        cam_to_base_rotation, camera_parameters.target_to_cam_translation
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
    pass
