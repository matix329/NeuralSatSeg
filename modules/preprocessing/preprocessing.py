import tensorflow as tf
from PIL import Image
import numpy as np
import os

class Preprocessing:
    def __init__(self, image_size, temp_dir="temp_converted"):
        self.image_size = image_size
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def convert_tiff_to_png(self, file_path):
        try:
            with Image.open(file_path) as img:
                filename = os.path.splitext(os.path.basename(file_path))[0]
                png_path = os.path.join(self.temp_dir, f"{filename}.png")
                img.save(png_path, "PNG")
                return png_path
        except Exception as e:
            print(f"[ERROR] Failed to convert TIFF to PNG: {e}")
            raise

    def handle_file_format(self, file_path):
        if file_path.lower().endswith((".tiff", ".tif")):
            return self.convert_tiff_to_png(file_path)
        return file_path

    def load_and_preprocess_image(self, image_path):
        image_path = self.handle_file_format(image_path)
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(self.image_size)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        return img_tensor

    def load_and_preprocess_mask(self, mask_path):
        mask_path = self.handle_file_format(mask_path)
        with Image.open(mask_path) as img:
            img = img.convert("L")
            img = img.resize(self.image_size, Image.NEAREST)
            mask_array = np.array(img, dtype=np.float32)
            mask_tensor = tf.convert_to_tensor(mask_array, dtype=tf.float32)
        return mask_tensor