import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from structs import HandEyeCalibrationResult

# Extimate:
# Target to base transform
# Camera to arm transform


def get_poses_error(lhs_poses_T, rhs_poses_T, lambd=1.0):
    trans_error = (
        np.linalg.norm(lhs_poses_T[:, :3, 3] - rhs_poses_T[:, :3, 3], axis=1) ** 2
    )
    rot_error = (
        1
        - (
            R.from_matrix(lhs_poses_T[:, :3, :3]).as_quat()
            * R.from_matrix(rhs_poses_T[:, :3, :3]).as_quat()
        ).sum(axis=1)
        ** 2
    )
    return trans_error + lambd * rot_error


def get_loss(
    x,
    arm_to_base_rotation,
    arm_to_base_translation,
    target_to_cam_rotation,
    target_to_cam_translation,
    lambd_reg=2e-2,
):
    target_to_base_pose = x[:7]
    cam_to_arm_pose = x[7:]
    target_to_base_T = np.eye(4)
    target_to_base_T[:3, :3] = R.from_quat(target_to_base_pose[3:]).as_matrix()
    target_to_base_T[:3, 3] = target_to_base_pose[:3]
    cam_to_arm_T = np.eye(4)
    cam_to_arm_T[:3, :3] = R.from_quat(cam_to_arm_pose[3:]).as_matrix()
    cam_to_arm_T[:3, 3] = cam_to_arm_pose[:3]

    arm_to_base_T = np.stack(
        [
            np.eye(4),
        ]
        * arm_to_base_rotation.shape[0]
    )
    arm_to_base_T[:, :3, :3] = arm_to_base_rotation
    arm_to_base_T[:, :3, 3] = arm_to_base_translation

    target_to_cam_T = np.stack(
        [
            np.eye(4),
        ]
        * target_to_cam_rotation.shape[0]
    )
    target_to_cam_T[:, :3, :3] = target_to_cam_rotation
    target_to_cam_T[:, :3, -1] = target_to_cam_translation

    target_to_base_derived_T = arm_to_base_T @ cam_to_arm_T @ target_to_cam_T
    target_to_base_pred_T = np.stack(
        [
            target_to_base_T,
        ]
        * target_to_base_derived_T.shape[0]
    )
    result_loss = get_poses_error(target_to_base_derived_T, target_to_base_pred_T)
    result_loss += lambd_reg * np.linalg.norm(cam_to_arm_pose[:3])
    return result_loss


def get_cam_to_arm(arm_to_base_rotation, arm_to_base_translation, camera_params):
    target_to_base_pose = np.array([0, 0, 0, 0, 0, 0, 1])
    cam_to_arm_pose = np.array([0, 0, 0, 0, 0, 0, 1])
    x = np.concatenate((target_to_base_pose, cam_to_arm_pose))
    result = least_squares(
        get_loss,
        x,
        args=(
            arm_to_base_rotation,
            arm_to_base_translation,
            camera_params.target_to_cam_rotation,
            camera_params.target_to_cam_translation,
        ),
    )
    x_opt = result.x
    target_to_base_pose = x_opt[:7]
    cam_to_arm_pose = x_opt[7:]
    cam_to_arm_rotation = R.from_quat(cam_to_arm_pose[3:]).as_matrix()
    cam_to_arm_translation = cam_to_arm_pose[:3]

    cam_to_base_rotation = arm_to_base_rotation @ cam_to_arm_rotation
    cam_to_base_translation = arm_to_base_translation + np.matvec(
        arm_to_base_rotation, cam_to_arm_translation
    )
    target_to_base_rotation = (
        cam_to_base_rotation @ camera_params.target_to_cam_rotation
    )
    target_to_base_translation = cam_to_base_translation + np.matvec(
        cam_to_base_rotation, camera_params.target_to_cam_translation
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
    arm_to_base_rotation = np.load("arm_to_base_rotation.npy")
    arm_to_base_translation = np.load("arm_to_base_translation.npy")
    target_to_cam_rotation = np.load("target_to_cam_rotation.npy")
    target_to_cam_translation = np.load("target_to_cam_translation.npy")
    target_to_base_pose = np.array([0, 0, 0, 0, 0, 0, 1])
    cam_to_arm_pose = np.array([0, 0, 0, 0, 0, 0, 1])
    x = np.concatenate((target_to_base_pose, cam_to_arm_pose))
    result = least_squares(
        get_loss,
        x,
        args=(
            arm_to_base_rotation,
            arm_to_base_translation,
            target_to_cam_rotation,
            target_to_cam_translation,
        ),
    )
    x_opt = result.x
    target_to_base_pose = x_opt[:7]
    cam_to_arm_pose = x_opt[7:]

    target_to_base_T = np.eye(4)
    target_to_base_T[:3, :3] = R.from_quat(target_to_base_pose[3:]).as_matrix()
    target_to_base_T[:3, 3] = target_to_base_pose[:3]
    cam_to_arm_T = np.eye(4)
    cam_to_arm_T[:3, :3] = R.from_quat(cam_to_arm_pose[3:]).as_matrix()
    cam_to_arm_T[:3, 3] = cam_to_arm_pose[:3]
    print(cam_to_arm_T)
