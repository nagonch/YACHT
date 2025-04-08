import numpy as np
from PIL import Image
import os
from visualize import (
    create_viser_server,
    viusalize_target_to_cam_poses_2D,
    viusalize_target_to_cam_poses_3D,
    visualize_hand_eye_poses,
)
from utils import pose_pretty_string, estimate_hand_eye_error, CONFIG, LOGGER
from opencv_functions import (
    detect_corners,
    get_camera_parameters,
    get_eye_to_hand_transformation,
)


def main(
    data_folder: str,
    chessboard_height: int,
    chessboard_width: int,
    chessboard_size: float,
    verbose: bool,
) -> None:
    # Check folder structure
    assert os.path.exists(data_folder), f"Data folder '{data_folder}' does not exist."
    assert os.path.exists(
        f"{data_folder}/images"
    ), f"Images folder '{data_folder}/images' does not exist."
    assert os.path.exists(
        f"{data_folder}/poses.txt"
    ), f"Poses file '{data_folder}/poses.txt' not found"

    # Load poses
    arm_poses = np.loadtxt(f"{data_folder}/poses.txt").reshape(-1, 4, 4)
    arm_to_base_translation = arm_poses[:, :3, -1]
    arm_to_base_rotation = arm_poses[:, :3, :3]
    # Load images
    image_filenames = sorted(os.listdir(f"{data_folder}/images"))
    images = [
        np.array(Image.open(f"{data_folder}/images/{image_fname}"))
        for image_fname in image_filenames
    ]

    # Detecting corners
    LOGGER.info("Detecting corners...")
    detected_inds, detected_corners, corners3D, detected_images = detect_corners(
        images,
        chessboard_dims=(chessboard_height, chessboard_width),
        chessboard_size=chessboard_size * 1e-3,
    )
    LOGGER.info("done.")
    assert (
        len(detected_corners) > 0
    ), f"No corners sized {chessboard_size:.1f} mm of a {chessboard_height} x {chessboard_width} board detected in images. "

    LOGGER.info(
        f"{len(detected_corners)}/{len(images)} images with detected corners.\n"
    )

    # Camera calibration
    LOGGER.info("Calibrating camera...")
    camera_parameters = get_camera_parameters(
        detected_corners, corners3D, detected_images
    )
    LOGGER.info(f"done. RMS error: {camera_parameters.rms_error}\n")

    arm_to_base_translation = arm_to_base_translation[detected_inds]
    arm_to_base_rotation = arm_to_base_rotation[detected_inds]

    # Camera to arm calibration
    LOGGER.info("Calibrating hand-eye transformation... ")
    hand_eye_calibration_result = get_eye_to_hand_transformation(
        arm_to_base_rotation, arm_to_base_translation, camera_parameters
    )
    LOGGER.info("done.")
    LOGGER.info("Cam to arm result:")
    LOGGER.info(
        pose_pretty_string(
            hand_eye_calibration_result.cam_to_arm_rotation,
            hand_eye_calibration_result.cam_to_arm_translation,
        )
    )
    LOGGER.info("Cam to arm error (target to base uncertainty):")
    LOGGER.info(
        pose_pretty_string(
            *estimate_hand_eye_error(hand_eye_calibration_result),
            convert_from_matrix=False,
        )
    )
    if verbose:
        LOGGER.info("Projecting target poses to camera images...")
        output_folder = f"{data_folder}/visualization"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        viusalize_target_to_cam_poses_2D(
            detected_images, camera_parameters, detected_corners, output_folder
        )
        LOGGER.info(f"done. Images saved to folder '{output_folder}'\n")
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


if __name__ == "__main__":
    main(
        CONFIG["data-folder"],
        CONFIG["chessboard-height"],
        CONFIG["chessboard-width"],
        CONFIG["chessboard-size"],
        CONFIG["verbose"],
    )
