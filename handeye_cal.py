import numpy as np
from PIL import Image
import os
from visualize import (
    create_viser_server,
    viusalize_target_to_cam_poses_2D,
    viusalize_target_to_cam_poses_3D,
    visualize_hand_eye_poses,
)
from utils import (
    pose_pretty_string,
    estimate_hand_eye_error,
    CONFIG,
    LOGGER,
    average_target_pose,
)
from opencv_functions import (
    get_camera_parameters,
    get_eye_to_hand_transformation,
    get_camera_extrinsics,
    undistort_images,
)
from handeye_functions import get_cam_to_arm
import cv2
import h5py


def main() -> None:
    # Check folder structure
    DATA_FOLDER = CONFIG["handeye"]["data-folder"]
    ARM_CAL_IMAGES_FOLDER = f"{DATA_FOLDER}/images"
    POSES_FILE = f"{DATA_FOLDER}/arm_poses_result.npy"
    OUTPUT_FILE = f"{DATA_FOLDER}/result.h5"
    assert os.path.exists(
        ARM_CAL_IMAGES_FOLDER
    ), f"Arm cal images folder {ARM_CAL_IMAGES_FOLDER} does not exist."
    assert os.path.exists(POSES_FILE), f"Poses file {POSES_FILE} not found"

    # Load poses
    arm_poses = np.load(POSES_FILE)
    arm_to_base_rotation = arm_poses[:, :3, :3]
    arm_to_base_translation = arm_poses[:, :3, 3]

    arm_calib_filenames = sorted(os.listdir(ARM_CAL_IMAGES_FOLDER))
    arm_calib_images = [
        np.array(Image.open(f"{ARM_CAL_IMAGES_FOLDER}/{image_fname}"))
        for image_fname in arm_calib_filenames
    ]

    # Camera calibration
    LOGGER.info("Calibrating camera...")
    detected_images, detected_corners, camera_parameters, corners3D = (
        get_camera_parameters(
            arm_calib_images,
            chessboard_dims=(
                CONFIG["handeye"]["chessboard-width"],
                CONFIG["handeye"]["chessboard-height"],
            ),
            chessboard_size=CONFIG["handeye"]["chessboard-size"],
        )
    )

    LOGGER.info(f"done. RMS error: {camera_parameters.rms_error.mean()}\n")

    LOGGER.info("Geting cam extrinsics...")
    camera_parameters, detected_inds, detected_images, detected_corners = (
        get_camera_extrinsics(
            arm_calib_images,
            camera_parameters,
            chessboard_dims=(
                CONFIG["handeye"]["chessboard-width"],
                CONFIG["handeye"]["chessboard-height"],
            ),
            chessboard_size=CONFIG["handeye"]["chessboard-size"],
        )
    )
    detected_images = undistort_images(detected_images, camera_parameters)
    LOGGER.info("done.")

    arm_to_base_translation = arm_to_base_translation[detected_inds]
    arm_to_base_rotation = arm_to_base_rotation[detected_inds]

    # Camera to arm calibration
    LOGGER.info("Calibrating hand-eye transformation... ")
    if CONFIG["handeye"]["method"] == "my_method":
        hand_eye_calibration_result = get_cam_to_arm(
            arm_to_base_rotation, arm_to_base_translation, camera_parameters
        )
    else:
        method = (
            cv2.CALIB_HAND_EYE_TSAI
            if CONFIG["handeye"]["method"].lower() == "tsai"
            else cv2.CALIB_HAND_EYE_PARK
        )
        hand_eye_calibration_result = get_eye_to_hand_transformation(
            arm_to_base_rotation, arm_to_base_translation, camera_parameters, method
        )
    LOGGER.info("done.")
    target_pose_T = average_target_pose(
        hand_eye_calibration_result.target_to_base_rotation,
        hand_eye_calibration_result.target_to_base_translation,
    )
    LOGGER.info("Cam to arm result:")
    LOGGER.info(
        pose_pretty_string(
            hand_eye_calibration_result.cam_to_arm_rotation,
            hand_eye_calibration_result.cam_to_arm_translation,
        )
    )
    rotation_error, translation_error = estimate_hand_eye_error(
        hand_eye_calibration_result
    )
    LOGGER.info(f"Cam to arm rotation error (target std): {rotation_error:.3f}°")
    LOGGER.info(
        f"Cam to arm translation error (target std): {translation_error*1000:.3f} mm"
    )
    if CONFIG["handeye"]["visualize-2D"]:
        LOGGER.info("Projecting target poses to camera images...")
        output_folder = f"{CONFIG['handeye']['data-folder']}/arm_cal_visualization"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        viusalize_target_to_cam_poses_2D(
            detected_images,
            corners3D,
            camera_parameters,
            detected_corners,
            output_folder,
        )
        LOGGER.info(f"done. Images saved to folder '{output_folder}'\n")

    if CONFIG["handeye"]["visualize-3D"]:
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
        visualize_hand_eye_poses(
            viser_server,
            detected_images,
            camera_parameters,
            hand_eye_calibration_result,
            normalize=True,
        )
        LOGGER.info("\n")
        viser_server.stop()

    LOGGER.info(f"Saving result to {CONFIG['handeye']['data-folder']}/result.npy...")
    cam_to_arm_pose_T = np.eye(4)
    cam_to_arm_pose_T[:3, :3] = hand_eye_calibration_result.cam_to_arm_rotation
    cam_to_arm_pose_T[:3, 3] = hand_eye_calibration_result.cam_to_arm_translation
    result = {
        "cam_to_arm_pose": cam_to_arm_pose_T,
        "target_to_base_pose": target_pose_T,
        "camera_matrix": camera_parameters.intrinsics,
        "distortion_coefficients": camera_parameters.distortion_coeffs,
    }
    with h5py.File(OUTPUT_FILE, "w") as f:
        for key, array in result.items():
            f.create_dataset(key, data=array)
    LOGGER.info("finished.")


if __name__ == "__main__":
    main()
