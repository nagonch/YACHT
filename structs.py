from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class CameraParameters:
    rms_error: float
    intrinsics: NDArray  # [3, 3]
    distortion_coeffs: NDArray  # [5]
    target_to_cam_rotation: NDArray  # [N, 3, 3]
    target_to_cam_translation: NDArray  # [N, 3]
    image_w: int
    image_h: int


@dataclass
class HandEyeCalibrationResult:
    arm_to_base_rotation: NDArray
    arm_to_base_translation: NDArray
    cam_to_arm_rotation: NDArray
    cam_to_arm_translation: NDArray
    cam_to_base_rotation: NDArray
    cam_to_base_translation: NDArray
    target_to_base_rotation: NDArray
    target_to_base_translation: NDArray


if __name__ == "__main__":
    pass
