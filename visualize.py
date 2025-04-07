import viser
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import normalize_points
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def add_frame(scene, rotation, translation, name, frame_scale=0.1):
    scene.add_frame(
        axes_length=frame_scale * 2,
        origin_radius=frame_scale / 5,
        axes_radius=frame_scale / 10,
        name=name,
        wxyz=R.from_matrix(rotation).as_quat(),
        position=translation,
    )


def visualize_geometry(
    images,
    camera_parameters,
    camera_frustums,
    frames,
    frames_scale=0.1,
):
    server = viser.ViserServer()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

    if camera_frustums is not None and camera_parameters is not None:
        aspect_ratio = camera_parameters.image_w / camera_parameters.image_h
        fov = (
            np.arctan2(
                camera_parameters.image_w / 2, camera_parameters.intrinsics[0, 0]
            )
            * 2
        )
        for i, (image, (rotation, translation)) in enumerate(
            zip(images, camera_frustums)
        ):
            server.scene.add_camera_frustum(
                name=f"{i}_cam",
                aspect=aspect_ratio,
                fov=fov,
                scale=frames_scale,
                line_width=0.5,
                image=image,
                wxyz=R.from_matrix(rotation).as_quat(),
                position=translation,
            )
    if frames is not None:
        for i, (rotation, translation) in enumerate(frames):
            add_frame(server.scene, rotation, translation, f"{i}_frame", frames_scale)
    server.scene.add_frame(
        name="world",
    )
    try:
        while True:
            time.sleep(2.0)
    except KeyboardInterrupt:
        server.stop()


def viusalize_target_to_cam_poses_2D(
    images,
    camera_parameters,
    detected_corners,
    output_folder,
):
    intrinsics_matrix = camera_parameters.intrinsics
    target_to_cam_rotation = camera_parameters.target_to_cam_rotation
    target_to_cam_translation = camera_parameters.target_to_cam_translation
    for pose_i, (image, rotation, translation, detected_corners_i) in enumerate(
        zip(
            images,
            target_to_cam_rotation,
            target_to_cam_translation,
            detected_corners,
        )
    ):
        axes_3D = np.eye(3)
        rotated_axes = rotation @ axes_3D.T
        translated_axes = rotated_axes + translation.reshape(3, 1)
        projected_axes, _ = cv2.projectPoints(
            translated_axes.T,
            np.zeros(3),
            np.zeros(3),
            intrinsics_matrix,
            distCoeffs=camera_parameters.distortion_coeffs,
        )
        projected_axes = projected_axes.reshape(-1, 2).astype(int)
        projected_origin, _ = cv2.projectPoints(
            translation, np.zeros(3), np.zeros(3), intrinsics_matrix, None
        )
        projected_origin = projected_origin.reshape(-1, 2).astype(int)[0]
        lengths = np.linalg.norm(projected_axes - projected_origin, axis=1)
        projected_axes = (
            projected_origin
            + (projected_axes - projected_origin) / lengths[:, None] * 50
        )
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # X (red), Y (green), Z (blue)
        for i, color in enumerate(colors):
            cv2.line(
                image,
                tuple(projected_origin.astype(np.int32)),
                tuple(projected_axes[i].astype(np.int32)),
                color,
                2,
            )

        plt.imshow(image)
        plt.scatter(
            detected_corners_i[:, 0],
            detected_corners_i[:, 1],
            s=image.shape[0] / 100,
            color="yellow",
        )
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f"{output_folder}/{str(pose_i).zfill(4)}.png")
        plt.close()


def viusalize_target_to_cam_poses_3D(
    images,
    camera_parameters,
    frames_scale=0.1,
    scene_scale=10,
    normalize=False,
):
    if normalize:
        camera_parameters.target_to_cam_translation = normalize_points(
            camera_parameters.target_to_cam_translation, rescale=scene_scale
        )[0]

    cam_to_target_rotation = np.transpose(
        camera_parameters.target_to_cam_rotation, (0, 2, 1)
    )
    cam_to_target_translation = -np.matvec(
        cam_to_target_rotation, camera_parameters.target_to_cam_translation
    )
    visualize_geometry(
        images=images,
        camera_parameters=camera_parameters,
        camera_frustums=zip(cam_to_target_rotation, cam_to_target_translation),
        frames=None,
        frames_scale=frames_scale,
    )


def visualize_hand_eye_poses(
    images,
    camera_parameters,
    hand_eye_calibration_result,
    frames_scale=0.1,
    scene_scale=10,
    normalize=False,
):
    frame_translations = (
        hand_eye_calibration_result.arm_to_base_translation,
        hand_eye_calibration_result.target_to_base_translation,
    )
    frame_rotations = (
        hand_eye_calibration_result.arm_to_base_rotation,
        hand_eye_calibration_result.target_to_base_rotation,
    )
    if normalize:
        frame_translations = normalize_points(
            frame_translations,
            rescale=scene_scale,
        )
    frame_translations = np.concatenate(frame_translations, axis=0)
    frame_rotations = np.concatenate(frame_rotations, axis=0)
    visualize_geometry(
        images=images,
        camera_parameters=camera_parameters,
        camera_frustums=zip(
            hand_eye_calibration_result.cam_to_base_rotation,
            hand_eye_calibration_result.cam_to_base_translation,
        ),
        frames=zip(frame_rotations, frame_translations),
        frames_scale=frames_scale,
    )


if __name__ == "__main__":
    pass
