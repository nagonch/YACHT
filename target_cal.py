from utils import CONFIG, LOGGER
from visualize import (
    viusalize_target_to_cam_poses_2D,
    viusalize_target_to_cam_poses_3D,
    create_viser_server,
)
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
    DATA_FOLDER = CONFIG["target-cal"]["data-folder"]
    TARGET_IMGS_FOLDER = f"{DATA_FOLDER}/images"
    POSES_FILE = f"{DATA_FOLDER}/arm_poses_result.npy"
    OUTPUT_FILE = f"{DATA_FOLDER}/object_pose.txt"

    assert os.path.exists(
        TARGET_IMGS_FOLDER
    ), f"Target images folder {TARGET_IMGS_FOLDER} does not exist."
    assert os.path.exists(POSES_FILE), f"Poses file {POSES_FILE} not found"

    # Load poses
    arm_poses = np.load(POSES_FILE)
    arm_to_base_rotation = arm_poses[:, :3, :3]
    arm_to_base_translation = arm_poses[:, :3, 3]
    img_filenames = sorted(os.listdir(TARGET_IMGS_FOLDER))
    img_calib = [
        np.array(Image.open(f"{TARGET_IMGS_FOLDER}/{image_fname}"))
        for image_fname in img_filenames
    ]

    with h5py.File(HANDEYE_RESULT_FILE, "r") as f:
        cam_to_arm_pose = f["cam_to_arm_pose"][:]
        camera_matrix = f["camera_matrix"][:]
        distortion_coeffs = f["distortion_coefficients"][:]
        camera_parameters = CameraParameters(
            rms_error=[
                0.0,
            ]
            * cam_to_arm_pose.shape[0],
            intrinsics=camera_matrix,
            distortion_coeffs=distortion_coeffs,
            target_to_cam_rotation=None,
            target_to_cam_translation=None,
            image_w=img_calib[0].shape[1],
            image_h=img_calib[0].shape[0],
        )

    LOGGER.info("Geting cam extrinsics...")
    camera_parameters, detected_inds, detected_images, detected_corners, corners_3D = (
        get_camera_extrinsics(
            img_calib,
            camera_parameters,
            chessboard_dims=(
                CONFIG["target-cal"]["chessboard-width"],
                CONFIG["target-cal"]["chessboard-height"],
            ),
            chessboard_size=CONFIG["target-cal"]["chessboard-size"],
            return_corners_3D=True,
        )
    )
    LOGGER.info("Done.")

    if CONFIG["target-cal"]["visualize-2D"]:
        LOGGER.info("Projecting target poses to camera images...")
        output_folder = f"{CONFIG['target-cal']['data-folder']}/arm_cal_visualization"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        viusalize_target_to_cam_poses_2D(
            detected_images,
            corners_3D,
            camera_parameters,
            detected_corners,
            output_folder,
        )
        LOGGER.info(f"done. Images saved to folder '{output_folder}'\n")

    if CONFIG["target-cal"]["visualize-3D"]:
        LOGGER.info("Starting viser server...")
        viser_server = create_viser_server()
        LOGGER.info("done. Click the link above to open viser. \n")

        LOGGER.info(
            "Visualizing target to camera poses in viser... (press Ctrl+C for next visualization)"
        )
        viusalize_target_to_cam_poses_3D(
            viser_server, detected_images, camera_parameters, normalize=True
        )
        LOGGER.info("\n")

        LOGGER.info(
            "Visualizing target, camera and arm poses in viser... (press Ctrl+C to finish)"
        )

    LOGGER.info("Saving results")
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
    LOGGER.info(f"done.")
