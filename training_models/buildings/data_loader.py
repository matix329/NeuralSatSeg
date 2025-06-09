import os
import numpy as np
import tensorflow as tf
from scripts.color_logger import ColorLogger
from typing import Optional, Tuple, List
from training_models.buildings.config import REDUCED_DATASET_SIZE

class DataLoader:
    def __init__(self):
        self.logger = ColorLogger("DataLoader").get_logger()
        self.image_size = (650, 650)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
    def load_image_mask_pairs(self, data_dir: str, mask_type: str, limit_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
        full_data_dir = os.path.join(self.project_root, data_dir)
        if not os.path.exists(full_data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {full_data_dir}")
            
        image_paths = sorted([os.path.join(full_data_dir, "images", f) for f in os.listdir(os.path.join(full_data_dir, "images")) if f.endswith(".png")])
        mask_dir = "masks_original" if mask_type == "original" else "masks_eroded"
        mask_paths = sorted([os.path.join(full_data_dir, mask_dir, f) for f in os.listdir(os.path.join(full_data_dir, mask_dir)) 
                           if f.endswith(".png") or f.endswith(".npy")])
        
        if not image_paths or not mask_paths:
            raise FileNotFoundError(f"No image-mask pairs found in {full_data_dir}")
            
        if limit_samples is not None:
            split = "train" if "train" in data_dir else "val"
            max_samples = REDUCED_DATASET_SIZE[split]
            indices = np.random.RandomState(42).permutation(len(image_paths))[:max_samples]
            image_paths = [image_paths[i] for i in indices]
            mask_paths = [mask_paths[i] for i in indices]
            
        self.logger.info(f"Found {len(image_paths)} image-mask pairs")
        if limit_samples is not None:
            split = "train" if "train" in data_dir else "val"
            self.logger.info(f"Using reduced dataset for buildings: {len(image_paths)} {split} samples")
            
        return image_paths, mask_paths
        
    def get_tf_dataset(self, split="train", mask_type="original", batch_size=16, limit_samples: Optional[int] = None):
        data_dir = f"data/processed/{split}/buildings"
        image_paths, mask_paths = self.load_image_mask_pairs(data_dir, mask_type, limit_samples)
        
        image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
        mask_paths = tf.convert_to_tensor(mask_paths, dtype=tf.string)
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        
        if split == "train":
            dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
            
        dataset = dataset.map(
            lambda img_path, mask_path: self.preprocess(img_path, mask_path, mask_type),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
        
    def preprocess(self, image_path, mask_path, mask_type):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        
        if mask_type == "original":
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.cast(mask, tf.float32) / 255.0
        else:
            mask = tf.numpy_function(
                lambda x: np.load(x.decode('utf-8')).astype(np.float32),
                [mask_path],
                tf.float32
            )
            mask = tf.reshape(mask, (*self.image_size, 1))
            mask = mask / 255.0
            
        return image, mask
        
    def load(self, split="train", limit_samples: Optional[int] = None, mask_type="original"):
        return self.get_tf_dataset(split=split, mask_type=mask_type, batch_size=4, limit_samples=limit_samples) 