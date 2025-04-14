import numpy as np

if __name__ == "__main__":
    arm_to_base_rotation = np.load("arm_to_base_rotation.npy")
    arm_to_base_translation = np.load("arm_to_base_translation.npy")
    target_to_cam_rotation = np.load("target_to_cam_rotation.npy")
    target_to_cam_translation = np.load("target_to_cam_translation.npy")
