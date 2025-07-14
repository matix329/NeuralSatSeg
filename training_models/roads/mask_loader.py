import os
import rasterio
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import json

class MaskLoader:
    def __init__(self, img_size=None, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        config_img_size = tuple(config.get("img_size", [512, 512]))
        self.img_size = tuple(img_size) if img_size is not None else config_img_size
        
    def load_mask(self, mask_path, mask_type="binary"):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Maska nie istnieje: {mask_path}")
        if mask_type == "binary":
            if mask_path.endswith('.tif'):
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                    mask = mask.astype(np.float32)
                    if mask.dtype == np.uint16:
                        mask = mask / 65535.0
                    elif mask.dtype == np.uint8:
                        mask = mask / 255.0
                    else:
                        if mask.max() > 1.0:
                            mask = mask / mask.max()
                    mask = np.clip(mask, 0.0, 1.0)
                    if np.isnan(mask).any() or np.isinf(mask).any():
                        raise ValueError(f"Maska zawiera NaN lub inf: {mask_path}")
            else:
                import tensorflow as tf
                mask = tf.io.read_file(mask_path)
                mask = tf.image.decode_png(mask, channels=1)
                mask = tf.cast(mask, tf.float32) / 255.0
            return mask
        else:
            raise ValueError(f"Nieobsługiwany typ maski: {mask_type}")

class GraphMaskDataset(Dataset):
    def __init__(self, root_dir, img_size=512, transform=None, pre_transform=None):
        super().__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.pt_files = [f for f in os.listdir(root_dir) if f.endswith('.pt')]
        self.img_size = img_size

    def len(self):
        return len(self.pt_files)

    def get(self, idx):
        pt_path = os.path.join(self.root_dir, self.pt_files[idx])
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        if hasattr(data, 'x') and data.x is not None:
            x = data.x.float()
            x0_min, x0_max = x[:,0].min(), x[:,0].max()
            x1_min, x1_max = x[:,1].min(), x[:,1].max()
            if (x0_max - x0_min) > 0 and (x1_max - x1_min) > 0:
                x[:,0] = (x[:,0] - x0_min) / (x0_max - x0_min) * (self.img_size - 1)
                x[:,1] = (x[:,1] - x1_min) / (x1_max - x1_min) * (self.img_size - 1)
            data.x = x
        if hasattr(data, 'y') and data.y is not None:
            label = data.y.float()
        else:
            label = torch.zeros(data.x.shape[0])
        return data, label