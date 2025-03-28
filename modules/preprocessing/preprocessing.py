import os
import numpy as np
import cv2
from PIL import Image

class Preprocessing:
    def __init__(self, image_size, temp_dir="temp_converted"):
        self.image_size = image_size
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def convert_tiff_to_png(self, file_path):
        try:
            if file_path.lower().endswith((".tiff", ".tif")):
                with Image.open(file_path) as img:
                    filename = os.path.splitext(os.path.basename(file_path))[0]
                    png_path = os.path.join(self.temp_dir, f"{filename}.png")
                    img.save(png_path, "PNG")
                    return png_path
            return file_path
        except Exception as e:
            print(f"[ERROR] Failed to convert TIFF to PNG: {e}")
            return file_path

    def handle_file_format(self, file_path):
        return self.convert_tiff_to_png(file_path)

    def load_and_preprocess_image(self, image_path):
        image_path = self.handle_file_format(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return img

    def load_and_preprocess_mask(self, mask_path, as_rgb=False):
        mask_path = self.handle_file_format(mask_path)

        if as_rgb:
            img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        img = np.where(img > 0, 1, 0).astype(np.uint8)
        return img