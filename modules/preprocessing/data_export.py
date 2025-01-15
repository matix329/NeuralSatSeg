import os
import cv2
import numpy as np

class DataExporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)

    def export_tile(self, img_tile, mask_tile, name):
        img_path = os.path.join(self.output_dir, "images", f"{name}.png")
        mask_path = os.path.join(self.output_dir, "masks", f"{name}.png")

        img_tile = np.transpose(img_tile, (1, 2, 0))

        if len(img_tile.shape) == 2:
            img_tile = np.expand_dims(img_tile, axis=-1)

        if img_tile.shape[2] != 3:
            raise ValueError(f"Obraz ma niepoprawną liczbę kanałów: {img_tile.shape[2]}, oczekiwano 3.")

        img_tile = np.clip(img_tile, 0, 1) * 255
        img_tile = img_tile.astype("uint8")
        mask_tile = mask_tile.astype("uint8")

        cv2.imwrite(img_path, img_tile)
        cv2.imwrite(mask_path, mask_tile)