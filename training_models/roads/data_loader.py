import os
import tensorflow as tf
from scripts.color_logger import ColorLogger
import torch
import io
from torch.serialization import safe_globals
from torch_geometric.data import Data
import numpy as np

class DataLoader:
    def __init__(self):
        self.logger = ColorLogger("DataLoader").get_logger()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
    def load(self, split="train", mask_type="binary"):
        data_dir = os.path.join(self.project_root, f"data/processed/{split}/roads")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
            
        image_paths = sorted([os.path.join(data_dir, "images", f) for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith(".png")])
        mask_dir = "masks_binary" if mask_type == "binary" else "masks_graph"
        mask_extension = ".png" if mask_type == "binary" else ".pt"
        mask_paths = sorted([os.path.join(data_dir, mask_dir, f) for f in os.listdir(os.path.join(data_dir, mask_dir)) if f.endswith(mask_extension)])
        
        if not image_paths or not mask_paths:
            raise FileNotFoundError(f"No image-mask pairs found in {data_dir}")
            
        self.logger.info(f"Found {len(image_paths)} image-mask pairs")
        self.logger.info(f"Sample image path: {image_paths[0]}")
        self.logger.info(f"Sample mask path: {mask_paths[0]}")
        
        image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
        mask_paths = tf.convert_to_tensor(mask_paths, dtype=tf.string)
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        
        if mask_type == "binary":
            dataset = dataset.map(lambda img_path, mask_path: self.preprocess_binary(img_path, mask_path), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(lambda img_path, mask_path: self.preprocess_graph(img_path, mask_path), num_parallel_calls=tf.data.AUTOTUNE)
            
        dataset = dataset.batch(4).prefetch(tf.data.AUTOTUNE)
        return dataset
        
    def preprocess_binary(self, image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.cast(mask, tf.float32) / 255.0
        
        return image, mask
        
    def preprocess_graph(self, image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        
        def load_pt_mask(mask_path):
            with safe_globals([Data]):
                mask_data = torch.load(mask_path.numpy().decode('utf-8'), weights_only=False)
                node_features = np.zeros((650, 650))
                adj_matrix = np.zeros((650, 650))
                
                if hasattr(mask_data, 'x'):
                    x_data = mask_data.x.detach().cpu().numpy()
                    if x_data.ndim == 2:
                        for i in range(x_data.shape[0]):
                            x, y = x_data[i]
                            if 0 <= x < 650 and 0 <= y < 650:
                                node_features[int(x), int(y)] = 1
                
                if hasattr(mask_data, 'edge_index'):
                    edge_index = mask_data.edge_index.detach().cpu().numpy()
                    for i in range(edge_index.shape[1]):
                        src, dst = edge_index[:, i]
                        if 0 <= src < 650 and 0 <= dst < 650:
                            adj_matrix[int(src), int(dst)] = 1
                
                combined_features = np.zeros((650, 650, 3))
                combined_features[:, :, 0] = node_features
                combined_features[:, :, 1] = adj_matrix
                combined_features[:, :, 2] = node_features
                
                return combined_features
            
        mask = tf.py_function(
            load_pt_mask,
            [mask_path],
            tf.float32
        )
        mask = tf.cast(mask, tf.float32)
        mask.set_shape([650, 650, 3])
        
        return image, mask 