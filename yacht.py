import numpy as np
from PIL import Image
import yaml
import os

with open("config.yaml") as file:
    CONFIG = yaml.safe_load(file)

if __name__ == "__main__":
    data_folder = CONFIG["data-folder"]
    assert os.path.exists(data_folder), f"Data folder '{data_folder}' does not exist."
    assert os.path.exists(
        f"{data_folder}/images"
    ), f"Images folder '{data_folder}/images' does not exist."
    assert os.path.exists(
        f"{data_folder}/poses.txt"
    ), f"Poses file '{data_folder}/poses.txt' not found"

    print(os.listdir(data_folder))
