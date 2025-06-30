import os
import tensorflow as tf
from scripts.color_logger import ColorLogger
import torch
import io
from torch.serialization import safe_globals
from torch_geometric.data import Data
import numpy as np
import json

class DataLoader:
    def __init__(self, config_path="training_models/roads/config.json"):
        self.logger = ColorLogger("DataLoader").get_logger()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.batch_size = self.config["batch_size"]
        self.img_size = self.config["img_size"]
        
    def load(self, split="train", mask_type="binary", city=None):
        data_dir = os.path.join(self.project_root, f"data/processed/{split}/roads")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        image_files = [f for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith(".png")]
        mask_dir = "masks_binary" if mask_type == "binary" else "masks_graph"
        mask_extension = ".png" if mask_type == "binary" else ".pt"
        mask_files = [f for f in os.listdir(os.path.join(data_dir, mask_dir)) if f.endswith(mask_extension)]

        if city is not None:
            image_files = [f for f in image_files if city in f]
            mask_files = [f for f in mask_files if city in f]

        image_paths = sorted([os.path.join(data_dir, "images", f) for f in image_files])
        mask_paths = sorted([os.path.join(data_dir, mask_dir, f) for f in mask_files])
        
        if not image_paths or not mask_paths:
            raise FileNotFoundError(f"No image-mask pairs found in {data_dir} for city: {city}")
        
        self.logger.info(f"Found {len(image_paths)} image-mask pairs for city: {city if city else 'ALL'}")
        self.logger.info(f"Sample image path: {image_paths[0]}")
        self.logger.info(f"Sample mask path: {mask_paths[0]}")
        
        image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
        mask_paths = tf.convert_to_tensor(mask_paths, dtype=tf.string)
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        
        if mask_type == "binary":
            dataset = dataset.map(lambda img_path, mask_path: self.preprocess_binary(img_path, mask_path), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(lambda img_path, mask_path: self.preprocess_graph(img_path, mask_path), num_parallel_calls=tf.data.AUTOTUNE)
        
        if split == "train":
            dataset = dataset.map(self.augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def normalize_img(self, image):
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        image = (image - mean) / std
        return image

    def augment_data(self, image, mask):
        seed = tf.random.uniform([], maxval=1000000, dtype=tf.int32)
        image = tf.image.stateless_random_flip_left_right(image, seed=[seed, 0])
        mask = tf.image.stateless_random_flip_left_right(mask, seed=[seed, 0])
        image = tf.image.stateless_random_flip_up_down(image, seed=[seed, 1])
        mask = tf.image.stateless_random_flip_up_down(mask, seed=[seed, 1])
        angle = tf.random.uniform([], -0.1, 0.1)
        image = tf.image.rot90(image, k=tf.cast(angle // 90, tf.int32))
        mask = tf.image.rot90(mask, k=tf.cast(angle // 90, tf.int32))
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_saturation(image, 0.9, 1.1)
        image = tf.image.random_hue(image, 0.05)
        return image, mask
        
    def preprocess_binary(self, image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, self.img_size)
        image = self.normalize_img(image)
        
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.cast(mask, tf.float32) / 255.0
        mask = tf.image.resize(mask, self.img_size)
        mask = tf.where(mask > 0.5, 1.0, 0.0)
        
        return image, mask
        
    def preprocess_graph(self, image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, self.img_size)
        image = self.normalize_img(image)
        
        def load_pt_mask(mask_path):
            return np.zeros((*self.img_size, 3), dtype=np.float32)
            
        mask = tf.py_function(
            load_pt_mask,
            [mask_path],
            tf.float32
        )
        mask = tf.cast(mask, tf.float32)
        mask.set_shape([self.img_size[0], self.img_size[1], 3])
        
        return image, mask

def tfa_rotate(image, angle):
    angle_deg = angle * 180.0 / np.pi
    return tf.image.rot90(image, k=tf.cast(angle_deg // 90, tf.int32)) 