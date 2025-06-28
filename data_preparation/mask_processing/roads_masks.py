import geopandas as gpd
import rasterio
import numpy as np
import logging
import os
import warnings
import networkx as nx
import torch
from torch_geometric.data import Data
from rasterio.features import rasterize
from shapely.geometry import mapping
from typing import Dict, Optional
from .mask_generator import BaseMaskGenerator, MaskConfig
import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class RoadBinaryMaskGenerator(BaseMaskGenerator):
    def __init__(self, geojson_folder: str, config: Optional[MaskConfig] = None):
        super().__init__(geojson_folder, config)
        self.output_size = (650, 650)

    def prepare_mask(self, geojson_path: str, img_id: str) -> Optional[np.ndarray]:
        try:
            gdf = gpd.read_file(geojson_path)
            if gdf.empty:
                logger.warning(f"GeoJSON file {geojson_path} is empty or invalid.")
                return None
            
            transform, crs, size = self.get_tiff_parameters(img_id)
            gdf = gdf.to_crs(crs)
            
            with rasterio.open(self.find_tiff_path(img_id)) as src:
                res_x, res_y = src.res[0], src.res[1]
                
            px_size = (res_x + res_y) / 2
            buffer_m = self.config.line_width * px_size
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                gdf["geometry"] = gdf["geometry"].buffer(buffer_m / 2)
            
            shapes = [(mapping(geom), 1) for geom in gdf.geometry if geom.is_valid]
            
            if not shapes:
                logger.warning(f"No valid geometries to rasterize in {geojson_path}.")
                return None
            
            if size != self.output_size:
                scale_x = size[1] / self.output_size[1]
                scale_y = size[0] / self.output_size[0]
                scaled_transform = rasterio.Affine(
                    transform.a * scale_x, transform.b, transform.c,
                    transform.d, transform.e * scale_y, transform.f
                )
            else:
                scaled_transform = transform
                
            mask = rasterize(
                shapes,
                out_shape=self.output_size,
                transform=scaled_transform,
                fill=0,
                dtype=np.float32,
                all_touched=True
            )
            
            mask = mask.astype(np.float32)
            
            if self.config.erosion_kernel_size > 0 and self.config.erosion_iterations > 0:
                mask = self.apply_erosion_dilation(mask)
            
            if self.config.save_debug_mask:
                self.save_debug_mask(mask, geojson_path, img_id)
            
            if not self.validate_mask(mask):
                return None
                
            return mask
            
        except Exception as e:
            logger.error(f"Error generating mask for {geojson_path}: {str(e)}")
            return None

    def apply_erosion_dilation(self, mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        kernel = np.ones((self.config.erosion_kernel_size, self.config.erosion_kernel_size), np.uint8)
        
        if self.config.erosion_iterations > 0:
            eroded = cv2.erode(binary, kernel, iterations=self.config.erosion_iterations)
            mask = mask.astype(np.float32)
            mask[eroded == 0] = 0.0
            
        return mask

    def save_debug_mask(self, mask: np.ndarray, geojson_path: str, img_id: str):
        try:
            debug_dir = os.path.join(os.path.dirname(geojson_path), "debug_masks")
            os.makedirs(debug_dir, exist_ok=True)
            
            filename = os.path.basename(geojson_path).replace('.geojson', '_mask.png')
            debug_path = os.path.join(debug_dir, filename)
            
            road_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            coverage_percent = (road_pixels / total_pixels) * 100
            
            plt.figure(figsize=(10, 8))
            plt.imshow(mask, cmap='gray')
            plt.title(f'Road Mask - {img_id}\nCoverage: {coverage_percent:.2f}% ({road_pixels} pixels)')
            plt.colorbar()
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Debug mask saved to {debug_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug mask: {str(e)}")

    def find_tiff_path(self, img_id: str) -> str:
        for root, _, files in os.walk(os.path.dirname(os.path.dirname(self.geojson_folder))):
            for fname in files:
                if fname.endswith(('.tif', '.tiff')) and img_id in fname:
                    return os.path.join(root, fname)
        raise ValueError(f"TIFF file not found for {img_id}")

    def generate_mask_from_array(self, geojson_path: str, img_id: str) -> np.ndarray:
        mask = self.prepare_mask(geojson_path, img_id)
        if mask is None:
            return np.zeros(self.output_size, dtype=np.float32)
        return mask

class RoadGraphMaskGenerator(BaseMaskGenerator):
    def __init__(self, geojson_folder: str, config: Optional[MaskConfig] = None):
        super().__init__(geojson_folder, config)
        self.output_size = (650, 650)

    def prepare_mask(self, geojson_path: str, img_id: str) -> Dict:
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            raise ValueError(f"GeoJSON file {geojson_path} is empty or invalid.")
            
        transform, crs, size = self.get_tiff_parameters(img_id)
        gdf = gdf.to_crs(crs)
        
        road_graph = nx.Graph()
        node_id = 0
        node_mapping = {}
        
        for idx, row in gdf.iterrows():
            line = row.geometry
            if not line.is_valid:
                continue
                
            coords = np.array(line.coords)
            pixel_coords = []
            for x, y in coords:
                col_img, row_img = ~transform * (x, y)
                col_img = int((col_img / size[1]) * self.output_size[1])
                row_img = int((row_img / size[0]) * self.output_size[0])
                pixel_coords.append((col_img, row_img))
            
            for i in range(len(pixel_coords) - 1):
                p1, p2 = pixel_coords[i], pixel_coords[i + 1]
                
                for p in [p1, p2]:
                    if p not in node_mapping:
                        node_mapping[p] = node_id
                        road_graph.add_node(node_id, pos=p)
                        node_id += 1
                
                width = self.config.line_width
                typ = 'unknown'
                if hasattr(row, 'get'):
                    width = row.get('width', width)
                    typ = row.get('type', typ)
                elif isinstance(row, dict):
                    width = row.get('width', width)
                    typ = row.get('type', typ)
                else:
                    logger.warning(f"Unexpected record type in geojson: {type(row)} for img_id={img_id}, idx={idx}")
                
                road_graph.add_edge(node_mapping[p1], node_mapping[p2], width=width, type=typ)
        
        if len(road_graph.nodes) == 0:
            raise ValueError(f"No valid road segments found in {geojson_path}")
            
        node_features = torch.tensor([road_graph.nodes[n]['pos'] for n in road_graph.nodes], dtype=torch.float32)
        edge_index = torch.tensor(list(road_graph.edges)).t().contiguous()
        edge_attr = torch.tensor([[road_graph[u][v]['width']] for u, v in road_graph.edges])
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }

    def generate_mask(self, geojson_path: str, img_id: str, output_path: str) -> Optional[str]:
        try:
            graph_data = self.prepare_mask(geojson_path, img_id)
            data = Data(
                x=graph_data['node_features'],
                edge_index=graph_data['edge_index'],
                edge_attr=graph_data['edge_attr']
            )
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(data, output_path)
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate graph mask: {str(e)}")
            return None 