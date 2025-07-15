from utils import CONFIG
import h5py
from opencv_functions import get_camera_extrinsics

if __name__ == "__main__":
    HANDEYE_RESULT_FILE = CONFIG["target-cal"]["handeye-result"]
    DATA_FOLDER = CONFIG["handeye"]["data-folder"]
    ARM_CAL_IMAGES_FOLDER = f"{DATA_FOLDER}/images"
    POSES_FILE = f"{DATA_FOLDER}/arm_poses_result.npy"
    OUTPUT_FILE = f"{DATA_FOLDER}/object_pose.txt"
    with h5py.File(HANDEYE_RESULT_FILE, "r") as f:
        cam_to_arm_pose = f["cam_to_arm_pose"][:]
        camera_matrix = f["camera_matrix"][:]
        distortion_coeffs = f["distortion_coefficients"][:]
    print(cam_to_arm_pose, camera_matrix, distortion_coeffs)
