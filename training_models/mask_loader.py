import os
import numpy as np
import torch
from PIL import Image
from config import MASK_EXTENSIONS, HEAD_NAMES, MASK_PATHS, MASK_CONFIG
from modules.mask_processing.mask_generator import MaskConfig
from modules.mask_processing.buildings_masks import BuildingMaskGenerator
from modules.mask_processing.roads_masks import RoadBinaryMaskGenerator, RoadGraphMaskGenerator

class MaskLoader:
    @staticmethod
    def load_mask(mask_path, class_name=None):
        _, ext = os.path.splitext(mask_path)
        if ext not in MASK_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")

        loader_type = MASK_EXTENSIONS[ext]
        
        if loader_type == "torch":
            mask = torch.load(mask_path)
            if isinstance(mask, dict):
                return mask
            mask = mask.numpy()
        elif loader_type == "image":
            mask = np.array(Image.open(mask_path))
        else:
            raise ValueError(f"Unknown loading type: {loader_type}")
            
        if class_name is not None and not isinstance(mask, dict):
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

    @staticmethod
    def get_mask_generator(class_name, geojson_folder):
        config = MaskConfig(**MASK_CONFIG[class_name])
        
        if class_name == 'buildings':
            return BuildingMaskGenerator(geojson_folder, config)
        elif class_name == 'roads':
            if MASK_PATHS[class_name].endswith('masks_binary'):
                return RoadBinaryMaskGenerator(geojson_folder, config)
            else:
                return RoadGraphMaskGenerator(geojson_folder, config)
        else:
            raise ValueError(f"Unknown class name: {class_name}") 