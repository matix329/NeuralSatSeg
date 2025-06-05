import os
import tensorflow as tf
from scripts.color_logger import ColorLogger

class MaskLoader:
    def __init__(self):
        self.logger = ColorLogger("BuildingsMaskLoader").get_logger()
        
    def load_mask(self, mask_path, mask_type="original"):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Maska nie istnieje: {mask_path}")
            
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.cast(mask, tf.float32) / 255.0
        
        if mask_type == "eroded":
            mask = self.erode_mask(mask)
            
        return mask
    
    def erode_mask(self, mask):
        return mask 