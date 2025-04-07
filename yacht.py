import numpy as np
from PIL import Image
import yaml

with open("config.yaml") as file:
    CONFIG = yaml.safe_load(file)

if __name__ == "__main__":
    print(CONFIG)
    pass
