import os
import tensorflow as tf
from scripts.color_logger import ColorLogger
import rasterio
import numpy as np

class MaskLoader:
    def __init__(self):
        self.logger = ColorLogger("RoadsMaskLoader").get_logger()
        
    def load_mask(self, mask_path, mask_type="binary"):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Maska nie istnieje: {mask_path}")
        
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
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.cast(mask, tf.float32) / 255.0
        
        if mask_type == "graph":
            mask = self.convert_to_graph_mask(mask)
            
        return mask
    
    def convert_to_graph_mask(self, mask):
        return mask 