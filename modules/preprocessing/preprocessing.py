import os
import numpy as np
import cv2

class Preprocessing:
    def __init__(self, image_size, temp_dir="temp_converted"):
        self.image_size = image_size
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def load_and_preprocess_mask(self, mask_path, as_rgb=False):
        if as_rgb:
            img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        img = np.where(img > 0, 1, 0).astype(np.uint8)
        return img

    def preprocess_array(self, img: np.ndarray):
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        img = np.where(img > 0, 1, 0).astype(np.uint8)
        return img