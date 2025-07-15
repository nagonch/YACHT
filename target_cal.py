from utils import CONFIG, LOGGER
import h5py
from opencv_functions import get_camera_extrinsics
import os
import numpy as np
from PIL import Image
from structs import CameraParameters
from scipy.spatial.transform import Rotation as R


def get_target_to_base(
    cam_to_arm_pose, arm_to_base_rotation, arm_to_base_translation, camera_parameters
):
    target_to_cam_rotation = camera_parameters.target_to_cam_rotation
    target_to_cam_translation = camera_parameters.target_to_cam_translation
    target_to_cam_T = np.stack(
        [np.eye(4) for _ in range(target_to_cam_translation.shape[0])]
    )
    target_to_cam_T[:, :3, :3] = target_to_cam_rotation
    target_to_cam_T[:, :3, 3] = target_to_cam_translation

    arm_to_base_T = np.stack(
        [np.eye(4) for _ in range(arm_to_base_translation.shape[0])]
    )
    arm_to_base_T[:, :3, :3] = arm_to_base_rotation
    arm_to_base_T[:, :3, 3] = arm_to_base_translation

    target_to_arm_T = cam_to_arm_pose @ target_to_cam_T
    target_to_base_T = np.matmul(arm_to_base_T, target_to_arm_T)  # [N, 4, 4]
    return target_to_base_T


def get_final_target_pose(target_to_base_T):
    target_to_base_final_T = np.eye(4)
    target_to_base_final_T[:3, 3] = np.mean(target_to_base_T[:, :3, 3], axis=0)
    mean_rotation = R.from_matrix(target_to_base_T[:, :3, :3]).as_quat().mean(axis=0)
    target_to_base_final_T[:3, :3] = R.from_quat(mean_rotation).as_matrix()
    return target_to_base_final_T


if __name__ == "__main__":
    HANDEYE_RESULT_FILE = CONFIG["target-cal"]["handeye-result"]
    DATA_FOLDER = CONFIG["handeye"]["data-folder"]
    TARGET_IMGS_FOLDER = f"{DATA_FOLDER}/images"
    POSES_FILE = f"{DATA_FOLDER}/arm_poses_result.npy"
    OUTPUT_FILE = f"{DATA_FOLDER}/object_pose.txt"
    with h5py.File(HANDEYE_RESULT_FILE, "r") as f:
        cam_to_arm_pose = f["cam_to_arm_pose"][:]
        camera_matrix = f["camera_matrix"][:]
        distortion_coeffs = f["distortion_coefficients"][:]
        camera_parameters = CameraParameters(
            rms_error=None,
            intrinsics=camera_matrix,
            distortion_coeffs=distortion_coeffs,
            target_to_cam_rotation=None,
            target_to_cam_translation=None,
            image_w=None,
            image_h=None,
        )

    assert os.path.exists(
        TARGET_IMGS_FOLDER
    ), f"Target images folder {TARGET_IMGS_FOLDER} does not exist."
    assert os.path.exists(POSES_FILE), f"Poses file {POSES_FILE} not found"

    # Load poses
    arm_poses = np.load(POSES_FILE)
    arm_to_base_rotation = arm_poses[:, :3, :3]
    arm_to_base_translation = arm_poses[:, :3, 3]

    img_filenames = sorted(os.listdir(TARGET_IMGS_FOLDER))
    img_calib_filenames = [
        np.array(Image.open(f"{TARGET_IMGS_FOLDER}/{image_fname}"))
        for image_fname in img_filenames
    ]
    LOGGER.info("Geting cam extrinsics...")
    camera_parameters, detected_inds, detected_images, detected_corners = (
        get_camera_extrinsics(
            img_calib_filenames,
            camera_parameters,
            chessboard_dims=(
                CONFIG["handeye"]["chessboard-width"],
                CONFIG["handeye"]["chessboard-height"],
            ),
            chessboard_size=CONFIG["handeye"]["chessboard-size"],
        )
    )
    LOGGER.info("done.")
    target_to_base_T = get_target_to_base(
        cam_to_arm_pose,
        arm_to_base_rotation[detected_inds],
        arm_to_base_translation[detected_inds],
        camera_parameters,
    )
    target_to_base_T_final = get_final_target_pose(target_to_base_T)
    np.savetxt(
        OUTPUT_FILE,
        target_to_base_T_final,
    )
