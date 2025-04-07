import viser
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import normalize_points
import time


def viusalize_target_to_cam_poses(
    images,
    camera_parameters,
    frames_scale=0.1,
    scene_scale=10,
    normalize_points=False,
):
    if normalize_points:
        camera_parameters.target_to_cam_translation = normalize_points(
            camera_parameters.target_to_cam_translation, rescale=scene_scale
        )[0]

    cam_to_target_rotation = np.transpose(
        camera_parameters.target_to_cam_rotation, (0, 2, 1)
    )
    cam_to_target_translation = -np.matvec(
        cam_to_target_rotation, camera_parameters.target_to_cam_translation
    )
    aspect_ratio = camera_parameters.image_w / camera_parameters.image_h
    fov = (
        np.arctan2(camera_parameters.image_w / 2, camera_parameters.intrinsics[0, 0])
        * 2
    )
    server = viser.ViserServer()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

    for i, (image, rotation, translation) in enumerate(
        zip(
            images,
            cam_to_target_rotation,
            cam_to_target_translation,
        )
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
    server.scene.add_frame(
        name="world",
    )
    try:
        while True:
            time.sleep(2.0)
    except KeyboardInterrupt:
        server.stop()


def visualize_hand_eye_poses(
    images,
    camera_parameters,
    hand_eye_calibration_result,
    frames_scale=0.1,
    scene_scale=10,
    normalize_points=False,
):
    if normalize_points:
        (
            hand_eye_calibration_result.cam_to_base_translation,
            hand_eye_calibration_result.target_to_base_translation,
            hand_eye_calibration_result.target_to_base_translation,
        ) = normalize_points(
            hand_eye_calibration_result.cam_to_base_translation,
            hand_eye_calibration_result.target_to_base_translation,
            hand_eye_calibration_result.target_to_base_translation,
            rescale=scene_scale,
        )
    aspect_ratio = camera_parameters.image_w / camera_parameters.image_h
    fov = (
        np.arctan2(camera_parameters.image_w / 2, camera_parameters.intrinsics[0, 0])
        * 2
    )
    server = viser.ViserServer()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

    for i, (image, rotation, translation) in enumerate(
        zip(
            images,
            hand_eye_calibration_result.cam_to_base_rotation,
            hand_eye_calibration_result.cam_to_base_translation,
        )
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
    for i, (rotation, translation) in enumerate(
        zip(
            hand_eye_calibration_result.arm_to_base_rotation,
            hand_eye_calibration_result.arm_to_base_translation,
        )
    ):
        server.scene.add_frame(
            axes_length=frames_scale * 2,
            origin_radius=frames_scale / 5,
            axes_radius=frames_scale / 10,
            name=f"{i}_arm",
            wxyz=R.from_matrix(rotation).as_quat(),
            position=translation,
        )
    for i, (rotation, translation) in enumerate(
        zip(
            hand_eye_calibration_result.target_to_base_rotation,
            hand_eye_calibration_result.target_to_base_translation,
        ),
    ):
        server.scene.add_frame(
            axes_length=frames_scale * 2,
            origin_radius=frames_scale / 5,
            axes_radius=frames_scale / 10,
            name=f"{i}_target",
            wxyz=R.from_matrix(rotation).as_quat(),
            position=translation,
        )
    server.scene.add_frame(
        name="world",
    )
    try:
        while True:
            time.sleep(2.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    pass
