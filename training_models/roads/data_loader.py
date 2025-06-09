import os
import tensorflow as tf
from scripts.color_logger import ColorLogger

class DataLoader:
    def __init__(self):
        self.logger = ColorLogger("DataLoader").get_logger()
        
    def load(self, split="train", mask_type="binary"):
        data_dir = f"data/processed/{split}/roads"
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
            
        image_paths = sorted([os.path.join(data_dir, "images", f) for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith(".png")])
        mask_dir = "masks_binary" if mask_type == "binary" else "masks_graph"
        mask_paths = sorted([os.path.join(data_dir, mask_dir, f) for f in os.listdir(os.path.join(data_dir, mask_dir)) if f.endswith(".png")])
        
        if not image_paths or not mask_paths:
            raise FileNotFoundError(f"No image-mask pairs found in {data_dir}")
            
        self.logger.info(f"Found {len(image_paths)} image-mask pairs")
        self.logger.info(f"Sample image path: {image_paths[0]}")
        self.logger.info(f"Sample mask path: {mask_paths[0]}")
        
        image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
        mask_paths = tf.convert_to_tensor(mask_paths, dtype=tf.string)
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(lambda img_path, mask_path: self.preprocess(img_path, mask_path, mask_type), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(4).prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    def preprocess(self, image_path, mask_path, mask_type):
        self.logger.info(f"Image path type: {type(image_path)}")
        self.logger.info(f"Mask path type: {type(mask_path)}")
        
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        
        if mask_type == "binary":
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.cast(mask, tf.float32) / 255.0
        else:  # graph
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=3)  # graph masks are RGB
            mask = tf.cast(mask, tf.float32) / 255.0
        
        return image, mask 