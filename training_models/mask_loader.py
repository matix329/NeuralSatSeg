import os
import numpy as np
import torch
from PIL import Image
from config import MASK_EXTENSIONS

class MaskLoader:
    @staticmethod
    def load_mask(mask_path):
        _, ext = os.path.splitext(mask_path)
        if ext not in MASK_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")

        loader_type = MASK_EXTENSIONS[ext]
        
        if loader_type == "numpy":
            return np.load(mask_path)
        elif loader_type == "torch":
            return torch.load(mask_path).numpy()
        elif loader_type == "image":
            return np.array(Image.open(mask_path))
        else:
            raise ValueError(f"Unknown loading type: {loader_type}")

    @staticmethod
    def get_mask_paths(data_dir, mask_type):
        mask_dir = os.path.join(data_dir, mask_type)
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")
        
        mask_files = []
        for ext in MASK_EXTENSIONS.keys():
            mask_files.extend([f for f in os.listdir(mask_dir) if f.endswith(ext)])
        
        return [os.path.join(mask_dir, f) for f in mask_files] 