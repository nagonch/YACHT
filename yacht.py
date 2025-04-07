import numpy as np
from PIL import Image
import yaml
import os
import cv2
from typing import List, Tuple
from numpy.typing import NDArray

with open("config.yaml") as file:
    CONFIG = yaml.safe_load(file)


def detect_corners(
    images: List[NDArray],
    checkerboard_size: float = 28.5e-3,
    checkerboard_dims: Tuple[int] = (6, 8),
) -> Tuple[List[NDArray], NDArray, List[NDArray]]:
    corners3D = np.zeros(
        (checkerboard_dims[0] * checkerboard_dims[1], 3), dtype=np.float32
    )
    corners_2D = np.meshgrid(
        np.arange(checkerboard_dims[0]), np.arange(checkerboard_dims[1])
    )
    corners_2D = np.stack(corners_2D, axis=-1).reshape(-1, 2)
    corners3D[:, :2] = corners_2D * checkerboard_size
    detected_images = []
    detected_corners = []
    for image in images:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_detected, corners = cv2.findChessboardCorners(
            grayscale_image, checkerboard_dims
        )
        if is_detected:
            corners = cv2.cornerSubPix(
                grayscale_image,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )[:, 0, :]
            detected_images.append(image)
            detected_corners.append(corners)
    return (
        detected_corners,
        corners3D,
        detected_images,
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

    # Get images filenames
    image_filenames = sorted(os.listdir(f"{data_folder}/images"))
    images = [
        np.array(Image.open(f"{data_folder}/images/{image_fname}"))
        for image_fname in image_filenames
    ]
    print(detect_corners(images))
