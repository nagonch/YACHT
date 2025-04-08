import numpy as np
from PIL import Image
import os
from visualize import (
    create_viser_server,
    viusalize_target_to_cam_poses_2D,
    viusalize_target_to_cam_poses_3D,
    visualize_hand_eye_poses,
)
from utils import pose_pretty_string, estimate_hand_eye_error, CONFIG
from opencv_functions import (
    detect_corners,
    get_camera_parameters,
    get_eye_to_hand_transformation,
)


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
    print("Detecting corners...")
    detected_inds, detected_corners, corners3D, detected_images = detect_corners(
        images,
        chessboard_dims=(CONFIG["chessboard-height"], CONFIG["chessboard-width"]),
        chessboard_size=CONFIG["chessboard-size"],
    )
    print("done.\n")

    if len(detected_corners) == 0:
        raise RuntimeError(
            f"No corners detected in {len(images)} images. "
            f"Ensure the chessboard dimensions ({CONFIG['chessboard-height']}x{CONFIG['chessboard-width']}) "
            f"and size ({CONFIG['chessboard-size']}m) are correct, and the images are clear."
        )

    # Camera calibration
    print("Calibrating camera...", end="")
    camera_parameters = get_camera_parameters(
        detected_corners, corners3D, detected_images
    )
    print(f"done. RMS error: {camera_parameters.rms_error}\n")

    arm_to_base_translation = arm_to_base_translation[detected_inds]
    arm_to_base_rotation = arm_to_base_rotation[detected_inds]

    # Camera to arm calibration
    print("Calibrating hand-eye transformation... ", end="")
    hand_eye_calibration_result = get_eye_to_hand_transformation(
        arm_to_base_rotation, arm_to_base_translation, camera_parameters
    )
    print("done.")
    print("Cam to arm result:")
    print(
        pose_pretty_string(
            hand_eye_calibration_result.cam_to_arm_rotation,
            hand_eye_calibration_result.cam_to_arm_translation,
        )
    )
    print("Cam to arm error (target to base uncertainty):")
    print(
        pose_pretty_string(
            *estimate_hand_eye_error(hand_eye_calibration_result),
            convert_from_matrix=False,
        )
    )
    if CONFIG["verbose"]:
        print("Projecting target poses to camera images...", end="")
        output_folder = f"{data_folder}/visualization"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        viusalize_target_to_cam_poses_2D(
            detected_images, camera_parameters, detected_corners, output_folder
        )
        print(f"done. Images saved to folder '{output_folder}'\n")
        print("Starting viser server...")
        viser_server = create_viser_server()
        print("done. Click the link above to open viser. \n")
        print(
            "Visualizing target to camera poses in viser... (press Ctrl+C for next visualization)"
        )
        viusalize_target_to_cam_poses_3D(
            viser_server, detected_images, camera_parameters, normalize=True
        )
        print("\n")
        print(
            "Visualizing target, camera and arm poses in viser... (press Ctrl+C to finish)"
        )
        visualize_hand_eye_poses(
            viser_server,
            detected_images,
            camera_parameters,
            hand_eye_calibration_result,
            normalize=True,
        )
        print("\n")
        viser_server.stop()
    # TODO
    # Add typing everywhere
    # Record test dataset
    # Add demo video
    # Remove shit code
    # File standard for camera poses (ORDER MATTERS)
    # File standard for calibration results
    # Readme
    # Replace prints with logging
