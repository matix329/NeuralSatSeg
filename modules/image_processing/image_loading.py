import os
import numpy as np
import rasterio
import re
from typing import Dict

from .image_merge import ImageMerger

class ImageLoader:
    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        self.image_dict = self.group_images_by_id()

    def group_images_by_id(self) -> Dict[str, Dict[str, str]]:
        image_dict = {}
        for fname in os.listdir(self.image_dir):
            img_id = self.extract_img_id(fname)
            if img_id is None:
                continue

            if 'PS-RGB' in fname:
                img_type = 'PS-RGB'
            elif 'PS-MS' in fname:
                img_type = 'PS-MS'
            elif '_MS_' in fname:
                img_type = 'MS'
            elif '_PAN_' in fname:
                img_type = 'PAN'
            else:
                continue

            if img_id not in image_dict:
                image_dict[img_id] = {}
            image_dict[img_id][img_type] = os.path.join(self.image_dir, fname)

        return image_dict

    def extract_img_id(self, fname: str) -> str:
        match = re.search(r'img\d+', fname)
        return match.group(0) if match else None

    def load_all(self) -> Dict[str, np.ndarray]:
        merger = ImageMerger()
        loaded = {}
        for img_id, modalities in self.image_dict.items():
            if all(key in modalities for key in ['MS', 'PAN', 'PS-MS', 'PS-RGB']):
                merged = merger.merge_images_to_arrays({img_id: modalities})[img_id]
                loaded[img_id] = merged
            else:
                print(f"[WARNING] Missing modalities for {img_id}, skipping.")
        return loaded

    def merge_modalities(self, paths: Dict[str, str]) -> np.ndarray:
        arrays = []
        for key in ['MS', 'PAN', 'PS-MS', 'PS-RGB']:
            path = paths[key]
            with rasterio.open(path) as src:
                arr = src.read()
            arrays.append(arr)
        return np.concatenate(arrays, axis=0)