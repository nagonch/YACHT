from utils import CONFIG, LOGGER
import h5py
from opencv_functions import get_camera_extrinsics
import os
import numpy as np
from PIL import Image
from structs import CameraParameters

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
    print(
        f"Camera parameters: {camera_parameters.target_to_cam_translation.shape}, {camera_parameters.target_to_cam_rotation.shape}"
    )
    LOGGER.info("done.")
