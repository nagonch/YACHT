import numpy as np
from scipy.spatial.transform import Rotation as R

# Extimate:
# Target to base transform
# Camera to arm transform


def get_loss(
    x,
    arm_to_base_rotation,
    arm_to_base_translation,
    target_to_cam_rotation,
    target_to_cam_translation,
):
    target_to_base_pose = x[:7]
    cam_to_arm_pose = x[7:]
    target_to_base_T = np.eye(4)
    target_to_base_T[:3, :3] = R.from_quat(target_to_base_pose[3:]).as_matrix()
    target_to_base_T[:3, 3] = target_to_base_pose[:3]
    cam_to_arm_T = np.eye(4)
    cam_to_arm_T[:3, :3] = R.from_quat(cam_to_arm_pose[3:]).as_matrix()
    cam_to_arm_T[:3, 3] = cam_to_arm_pose[:3]
    print(cam_to_arm_T)
    print(target_to_base_T)


if __name__ == "__main__":
    arm_to_base_rotation = np.load("arm_to_base_rotation.npy")
    arm_to_base_translation = np.load("arm_to_base_translation.npy")
    target_to_cam_rotation = np.load("target_to_cam_rotation.npy")
    target_to_cam_translation = np.load("target_to_cam_translation.npy")
    target_to_base_pose = np.array([0, 0, 0, 0, 0, 0, 1])
    cam_to_arm_pose = np.array([0, 0, 0, 0, 0, 0, 1])
    x = np.concatenate((target_to_base_pose, cam_to_arm_pose))
    get_loss(
        x,
        arm_to_base_rotation,
        arm_to_base_translation,
        target_to_cam_rotation,
        target_to_cam_translation,
    )
