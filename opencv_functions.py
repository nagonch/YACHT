import numpy as np
import cv2
from typing import List, Tuple
from numpy.typing import NDArray
from tqdm import tqdm
from structs import CameraParameters, HandEyeCalibrationResult
from utils import LOGGER


def detect_corners(
    images: List[NDArray],
    chessboard_size: float = 28.5e-3,
    chessboard_dims: Tuple[int] = (6, 8),
    center_corners: bool = True,
) -> Tuple[List[NDArray], NDArray, List[NDArray]]:
    chessboard_width, chessboard_height = chessboard_dims
    corners3D = np.zeros((chessboard_width * chessboard_height, 3), dtype=np.float32)
    corners_2D = np.meshgrid(np.arange(chessboard_width), np.arange(chessboard_height))
    corners_2D = np.stack(corners_2D, axis=-1).reshape(-1, 2)
    if center_corners:
        center_shift = np.array(
            [(chessboard_width - 1) / 2, (chessboard_height - 1) / 2]
        )
        corners_2D = corners_2D.astype(np.float32) - center_shift
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
    images: List[NDArray],
    chessboard_size: float = 28.5e-3,
    chessboard_dims: Tuple[int] = (6, 8),
) -> CameraParameters:
    _, detected_corners, corners3D, detected_images = detect_corners(
        images,
        chessboard_dims=chessboard_dims,
        chessboard_size=chessboard_size * 1e-3,
    )
    assert (
        len(detected_corners) > 0
    ), f"No corners sized {chessboard_size:.1f} mm of a {chessboard_dims[0]} x {chessboard_dims[1]} board detected in cam cal images. "

    LOGGER.info(
        f"{len(detected_corners)}/{len(images)} cam cal images with detected corners.\n"
    )
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
    errors = []
    for i, detected_corner in enumerate(detected_corners):
        projected_corners, _ = cv2.projectPoints(
            corners3D,
            cv2.Rodrigues(cam_calib_params.target_to_cam_rotation[i])[0],
            cam_calib_params.target_to_cam_translation[i].reshape(3, -1),
            cam_calib_params.intrinsics,
            cam_calib_params.distortion_coeffs,
        )
        projected_corners = projected_corners[:, 0, :]
        error = np.sum((detected_corner - projected_corners) ** 2, axis=1)
        error = np.sqrt(error.mean())
        errors.append(error)
    cam_calib_params.rms_error = np.array(errors)
    return detected_images, detected_corners, cam_calib_params, corners3D


def get_camera_extrinsics(
    images: List[NDArray],
    camera_parameters: CameraParameters,
    chessboard_size: float = 28.5e-3,
    chessboard_dims: Tuple[int] = (6, 8),
    return_corners_3D: bool = False,
):
    detected_inds, detected_corners, corners3D, detected_images = detect_corners(
        images,
        chessboard_dims=chessboard_dims,
        chessboard_size=chessboard_size * 1e-3,
    )
    assert (
        len(detected_corners) > 0
    ), f"No corners sized {chessboard_size:.1f} mm of a {chessboard_dims[0]} x {chessboard_dims[1]} board detected in arm cal images. "

    LOGGER.info(
        f"{len(detected_corners)}/{len(images)} arm cal images with detected corners.\n"
    )
    rotations = []
    translations = []
    for corners in detected_corners:
        _, rvec, tvec = cv2.solvePnP(
            corners3D,
            corners,
            camera_parameters.intrinsics,
            camera_parameters.distortion_coeffs,
        )
        rotations.append(cv2.Rodrigues(rvec)[0])
        translations.append(tvec.reshape(-1))
    rotations = np.stack(rotations)
    translations = np.stack(translations)
    camera_parameters.target_to_cam_rotation = rotations
    camera_parameters.target_to_cam_translation = translations
    result = (camera_parameters, detected_inds, detected_images, detected_corners)
    if return_corners_3D:
        result += (corners3D,)
    return result


def undistort_images(detected_images, camera_parameters):
    result_images = []
    for image in detected_images:
        result_images.append(
            cv2.undistort(
                image, camera_parameters.intrinsics, camera_parameters.distortion_coeffs
            )
        )
    return result_images


def get_eye_to_hand_transformation(
    arm_to_base_rotation: NDArray,
    arm_to_base_translation: NDArray,
    camera_parameters: CameraParameters,
    method=cv2.CALIB_HAND_EYE_TSAI,
) -> HandEyeCalibrationResult:
    cam_to_arm_rotation, cam_to_arm_translation = cv2.calibrateHandEye(
        arm_to_base_rotation,
        arm_to_base_translation,
        camera_parameters.target_to_cam_rotation,
        camera_parameters.target_to_cam_translation,
        method=method,
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


def get_robot_world_transformation(
    arm_to_base_rotation: NDArray,
    arm_to_base_translation: NDArray,
    camera_parameters: CameraParameters,
) -> HandEyeCalibrationResult:
    target_to_cam_rotation = camera_parameters.target_to_cam_rotation
    target_to_cam_translation = camera_parameters.target_to_cam_translation
    base_to_arm_rotation = np.transpose(arm_to_base_rotation, (0, 2, 1))
    base_to_arm_translation = -np.matvec(base_to_arm_rotation, arm_to_base_translation)

    (
        base_to_target_rotation,
        base_to_target_translation,
        arm_to_cam_rotation,
        arm_to_cam_translation,
    ) = cv2.calibrateRobotWorldHandEye(
        target_to_cam_rotation,
        target_to_cam_translation,
        base_to_arm_rotation,
        base_to_arm_translation,
    )
    cam_to_arm_rotation = arm_to_cam_rotation.T
    cam_to_arm_translation = -cam_to_arm_rotation @ arm_to_cam_translation.reshape(-3)
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
