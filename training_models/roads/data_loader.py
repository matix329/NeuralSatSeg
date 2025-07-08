import os
import tensorflow as tf
from scripts.color_logger import ColorLogger
import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import io
from torch.serialization import safe_globals
import numpy as np
import json
import rasterio
import cv2

class DataLoader:
    def __init__(self, config_path=None, use_imagenet_norm=False):
        self.logger = ColorLogger("DataLoader").get_logger()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.batch_size = self.config["batch_size"]
        self.img_size = self.config["img_size"]
        self.use_imagenet_norm = use_imagenet_norm
        self.own_mean = tf.constant([0.005, 0.005, 0.005], dtype=tf.float32)
        self.own_std = tf.constant([0.002, 0.002, 0.002], dtype=tf.float32)
        self.imagenet_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.imagenet_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        
    def load(self, split="train", mask_type="binary", city=None):
        data_dir = os.path.join(self.project_root, f"data/processed/{split}/roads")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        image_files = [f for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith(".tif")]
        mask_dir = "masks_binary" if mask_type == "binary" else "masks_graph"
        mask_extension = ".tif" if mask_type == "binary" else ".pt"
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
        if self.use_imagenet_norm:
            mean = self.imagenet_mean
            std = self.imagenet_std
        else:
            mean = self.own_mean
            std = self.own_std
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
        def load_tiff_image(img_path):
            img_path_str = img_path.numpy().decode('utf-8')
            with rasterio.open(img_path_str) as src:
                image = src.read()
                image = np.moveaxis(image, 0, -1)
                if image.dtype == np.uint16:
                    image = (image / 65535.0).astype(np.float32)
                else:
                    image = image.astype(np.float32) / 255.0
                
                if len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=-1)
                elif len(image.shape) == 3:
                    if image.shape[2] == 1:
                        image = np.concatenate([image, image, image], axis=-1)
                    elif image.shape[2] == 3:
                        pass
                    elif image.shape[2] > 3:
                        image = image[:, :, :3]
                    else:
                        image = np.concatenate([image, image, image], axis=-1)
                
                return image
        
        def load_tiff_mask(mask_path):
            mask_path_str = mask_path.numpy().decode('utf-8')
            with rasterio.open(mask_path_str) as src:
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
                    raise ValueError(f"Maska zawiera NaN lub inf: {mask_path_str}")
                return mask
        
        image = tf.py_function(load_tiff_image, [image_path], tf.float32)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, self.img_size)
        image = self.normalize_img(image)
        
        mask = tf.py_function(load_tiff_mask, [mask_path], tf.float32)
        mask.set_shape([None, None])
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.image.resize(mask, self.img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.where(mask > 0.5, 1.0, 0.0)
        
        return image, mask
        
    def preprocess_graph(self, image_path, mask_path):
        def load_tiff_image(img_path):
            img_path_str = img_path.numpy().decode('utf-8')
            with rasterio.open(img_path_str) as src:
                image = src.read()
                image = np.moveaxis(image, 0, -1)
                if image.dtype == np.uint16:
                    image = image.astype(np.float32) / 65535.0
                else:
                    image = image.astype(np.float32) / 255.0
                if len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=-1)
                elif len(image.shape) == 3:
                    if image.shape[2] == 1:
                        image = np.concatenate([image, image, image], axis=-1)
                    elif image.shape[2] == 3:
                        pass
                    elif image.shape[2] > 3:
                        image = image[:, :, :3]
                    else:
                        image = np.concatenate([image, image, image], axis=-1)
                return image
        
        image = tf.py_function(load_tiff_image, [image_path], tf.float32)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, self.img_size)
        image = self.normalize_img(image)
        
        def load_pt_mask(mask_path):
            try:
                torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])
                mask_path_str = mask_path.numpy().decode('utf-8')
                data = torch.load(mask_path_str, map_location='cpu', weights_only=False)
                mask = np.zeros((650, 650), dtype=np.float32)
                if hasattr(data, 'x') and hasattr(data, 'edge_index'):
                    nodes = data.x.cpu().numpy() if hasattr(data.x, 'cpu') else np.array(data.x)
                    edges = data.edge_index.cpu().numpy() if hasattr(data.edge_index, 'cpu') else np.array(data.edge_index)
                    edges = edges.T if edges.shape[0] == 2 else edges
                    for edge in edges:
                        i, j = edge
                        x1, y1 = nodes[i]
                        x2, y2 = nodes[j]
                        pt1 = (int(round(x1)), int(round(y1)))
                        pt2 = (int(round(x2)), int(round(y2)))
                        cv2.line(mask, pt1, pt2, color=1.0, thickness=1)
                else:
                    print(f"[load_pt_mask] Invalid .pt file structure: {mask_path_str}")
                return mask.astype(np.float32)
            except Exception as e:
                print(f"[load_pt_mask] Error loading {mask_path}: {e}")
                return np.zeros((650, 650), dtype=np.float32)
        
        mask = tf.py_function(
            load_pt_mask,
            [mask_path],
            tf.float32
        )
        mask.set_shape([self.img_size[0], self.img_size[1]])
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.image.resize(mask, self.img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask.set_shape([self.img_size[0], self.img_size[1], 1])
        
        return image, mask

def tfa_rotate(image, angle):
    angle_deg = angle * 180.0 / np.pi
    return tf.image.rot90(image, k=tf.cast(angle_deg // 90, tf.int32)) 