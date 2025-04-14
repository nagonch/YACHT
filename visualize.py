import viser
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import normalize_points
import time
import cv2
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm
from structs import CameraParameters, HandEyeCalibrationResult


def add_frame(
    scene: viser.SceneApi,
    rotation: NDArray,
    translation: NDArray,
    name: str,
    frame_scale: float = 0.1,
) -> None:
    scene.add_frame(
        axes_length=frame_scale * 2,
        origin_radius=frame_scale / 5,
        axes_radius=frame_scale / 10,
        name=name,
        wxyz=R.from_matrix(rotation).as_quat(scalar_first=True),
        position=translation,
    )


def create_viser_server() -> viser.ViserServer:
    server = viser.ViserServer(verbose=False)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

    return server


def visualize_geometry(
    server: viser.ViserServer,
    images: List[NDArray],
    camera_parameters: CameraParameters,
    camera_frustums: NDArray,
    frames: NDArray,
    frames_scale=0.1,
) -> None:
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
                wxyz=R.from_matrix(rotation).as_quat(scalar_first=True),
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
        server.scene.reset()


def viusalize_target_to_cam_poses_2D(
    images: List[NDArray],
    camera_parameters: CameraParameters,
    detected_corners: List[NDArray],
    output_folder: str,
) -> None:

    intrinsics_matrix = camera_parameters.intrinsics
    target_to_cam_rotation = camera_parameters.target_to_cam_rotation
    target_to_cam_translation = camera_parameters.target_to_cam_translation
    for pose_i, (image, rotation, translation, detected_corners_i) in tqdm(
        enumerate(
            zip(
                images,
                target_to_cam_rotation,
                target_to_cam_translation,
                detected_corners,
            )
        ),
        total=len(images),
    ):
        origin = np.array([[0, 0, 0]]).T
        x_axis = np.array([[1, 0, 0]]).T
        y_axis = np.array([[0, 1, 0]]).T
        z_axis = np.array([[0, 0, 1]]).T

        translation = translation.reshape(3, 1)
        world_points = np.hstack((origin, x_axis, y_axis, z_axis))
        camera_points = np.dot(rotation, world_points) + translation

        image_points = np.dot(intrinsics_matrix, camera_points)
        image_points /= image_points[2, :]

        image_points_2D = image_points[:2, :].T
        cv2.circle(image, tuple(image_points_2D[0].astype(int)), 5, (0, 0, 255), -1)

        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            cv2.line(
                image,
                tuple(image_points_2D[0].astype(int)),
                tuple(image_points_2D[i + 1].astype(int)),
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
        plt.text(
            50,
            50,
            f"{camera_parameters.rms_error[pose_i]}",
            color="white",
            fontsize=12,
            weight="bold",
        )
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f"{output_folder}/{str(pose_i).zfill(4)}.png")
        plt.close()


def viusalize_target_to_cam_poses_3D(
    viser_server: viser.ViserServer,
    images: List[NDArray],
    camera_parameters: CameraParameters,
    frames_scale: float = 0.1,
    scene_scale: float = 10.0,
    normalize: bool = False,
) -> None:
    if normalize:
        camera_parameters.target_to_cam_translation = normalize_points(
            (camera_parameters.target_to_cam_translation,), rescale=scene_scale
        )[0]

    cam_to_target_rotation = np.transpose(
        camera_parameters.target_to_cam_rotation, (0, 2, 1)
    )
    cam_to_target_translation = -np.matvec(
        cam_to_target_rotation, camera_parameters.target_to_cam_translation
    )
    visualize_geometry(
        server=viser_server,
        images=images,
        camera_parameters=camera_parameters,
        camera_frustums=zip(cam_to_target_rotation, cam_to_target_translation),
        frames=None,
        frames_scale=frames_scale,
    )


def visualize_hand_eye_poses(
    viser_server: viser.ViserServer,
    images: List[NDArray],
    camera_parameters: CameraParameters,
    hand_eye_calibration_result: HandEyeCalibrationResult,
    frames_scale: float = 0.1,
    scene_scale: float = 10.0,
    normalize: bool = False,
) -> None:

    frame_translations = (
        hand_eye_calibration_result.arm_to_base_translation,
        hand_eye_calibration_result.target_to_base_translation,
        hand_eye_calibration_result.cam_to_base_translation,
    )
    frame_rotations = (
        hand_eye_calibration_result.arm_to_base_rotation,
        hand_eye_calibration_result.target_to_base_rotation,
        hand_eye_calibration_result.cam_to_base_rotation,
    )
    if normalize:
        frame_translations = normalize_points(
            frame_translations,
            rescale=scene_scale,
        )
    cam_to_base_translation = frame_translations[-1]
    cam_to_base_rotation = frame_rotations[-1]
    frame_translations = np.concatenate(frame_translations[:-1], axis=0)
    frame_rotations = np.concatenate(frame_rotations[:-1], axis=0)
    visualize_geometry(
        server=viser_server,
        images=images,
        camera_parameters=camera_parameters,
        camera_frustums=zip(cam_to_base_rotation, cam_to_base_translation),
        frames=zip(frame_rotations, frame_translations),
        frames_scale=frames_scale,
    )


if __name__ == "__main__":
    pass
