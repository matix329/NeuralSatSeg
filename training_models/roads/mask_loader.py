import os
import tensorflow as tf
from scripts.color_logger import ColorLogger

class MaskLoader:
    def __init__(self):
        self.logger = ColorLogger("RoadsMaskLoader").get_logger()
        
    def load_mask(self, mask_path, mask_type="binary"):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Maska nie istnieje: {mask_path}")
            
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.cast(mask, tf.float32) / 255.0
        
        if mask_type == "graph":
            mask = self.convert_to_graph_mask(mask)
            
        return mask
    
    def convert_to_graph_mask(self, mask):
        return mask 