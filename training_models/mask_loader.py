import os
import numpy as np
import torch
from PIL import Image
from config import MASK_EXTENSIONS, HEAD_NAMES, MASK_PATHS

class MaskLoader:
    @staticmethod
    def load_mask(mask_path, class_name=None):
        _, ext = os.path.splitext(mask_path)
        if ext not in MASK_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")

        loader_type = MASK_EXTENSIONS[ext]
        
        if loader_type == "numpy":
            mask = np.load(mask_path)
        elif loader_type == "torch":
            mask = torch.load(mask_path).numpy()
        elif loader_type == "image":
            mask = np.array(Image.open(mask_path))
        else:
            raise ValueError(f"Unknown loading type: {loader_type}")
            
        if class_name is not None:
            return mask.astype(np.float32) / 255.0
        return mask

    @staticmethod
    def load_all_masks(image_path):
        masks = {}
        for class_name in HEAD_NAMES:
            mask_path = image_path.replace('images', MASK_PATHS[class_name].split('/')[-2])
            masks[class_name] = MaskLoader.load_mask(mask_path, class_name)
        return masks

    @staticmethod
    def get_mask_paths(data_dir, mask_type):
        mask_dir = os.path.join(data_dir, mask_type)
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")
        
        mask_files = []
        for ext in MASK_EXTENSIONS.keys():
            mask_files.extend([f for f in os.listdir(mask_dir) if f.endswith(ext)])
        
        return [os.path.join(mask_dir, f) for f in mask_files] 