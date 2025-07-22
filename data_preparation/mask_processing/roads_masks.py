import geopandas as gpd
import rasterio
import numpy as np
import logging
import os
import warnings
import torch
from torch_geometric.data import Data
from rasterio.features import rasterize
from shapely.geometry import mapping, LineString, MultiLineString, Point
from typing import Dict, Optional
from .mask_generator import BaseMaskGenerator, MaskConfig
from scipy.spatial import cKDTree
import cv2

logger = logging.getLogger(__name__)

class RoadBinaryMaskGenerator(BaseMaskGenerator):
    def __init__(self, geojson_folder: str, config: Optional[MaskConfig] = None):
        super().__init__(geojson_folder, config)
        self.output_size = (1300, 1300)

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
            filename = os.path.basename(geojson_path).replace('.geojson', '_mask.tif')
            debug_path = os.path.join(debug_dir, filename)
            transform, crs, _ = self.get_tiff_parameters(img_id)
            with rasterio.open(
                debug_path, 'w',
                driver='GTiff',
                height=mask.shape[0],
                width=mask.shape[1],
                count=1,
                dtype=mask.dtype,
                crs=crs,
                transform=transform
            ) as dst:
                dst.write(mask, 1)
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
        node_pos = []
        all_lines = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if not geom.is_valid:
                continue
            lines = []
            if isinstance(geom, LineString):
                lines = [geom]
            elif isinstance(geom, MultiLineString):
                lines = list(geom.geoms)
            else:
                continue
            all_lines.extend(lines)
            for line in lines:
                coords = np.array(line.coords)
                for i in range(len(coords) - 1):
                    for pt in [coords[i], coords[i+1]]:
                        col_img, row_img = ~transform * (pt[0], pt[1])
                        col_img = int((col_img / size[1]) * self.output_size[1])
                        row_img = int((row_img / size[0]) * self.output_size[0])
                        node_pos.append((col_img, row_img))
        grid_step = 32
        for x in range(0, self.output_size[0], grid_step):
            for y in range(0, self.output_size[1], grid_step):
                node_pos.append((x, y))
        if len(node_pos) == 0:
            raise ValueError(f"No valid road segments found in {geojson_path}")
        node_pos = np.array(node_pos)
        tolerance = 2
        tree = cKDTree(node_pos)
        groups = tree.query_ball_tree(tree, r=tolerance)
        used = set()
        group_map = dict()
        group_centers = []
        for i, group in enumerate(groups):
            if i in used:
                continue
            members = set(group)
            used |= members
            group_pts = node_pos[list(members)]
            center = group_pts.mean(axis=0)
            group_id = len(group_centers)
            group_centers.append(center)
            for idx in members:
                group_map[tuple(node_pos[idx])] = group_id
        edges = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if not geom.is_valid:
                continue
            lines = []
            if isinstance(geom, LineString):
                lines = [geom]
            elif isinstance(geom, MultiLineString):
                lines = list(geom.geoms)
            else:
                continue
            for line in lines:
                coords = np.array(line.coords)
                for i in range(len(coords) - 1):
                    a = coords[i]
                    b = coords[i+1]
                    col_a, row_a = ~transform * (a[0], a[1])
                    col_a = int((col_a / size[1]) * self.output_size[1])
                    row_a = int((row_a / size[0]) * self.output_size[0])
                    col_b, row_b = ~transform * (b[0], b[1])
                    col_b = int((col_b / size[1]) * self.output_size[1])
                    row_b = int((row_b / size[0]) * self.output_size[0])
                    na = group_map[(col_a, row_a)]
                    nb = group_map[(col_b, row_b)]
                    if na != nb:
                        edges.append((na, nb))
        node_features = torch.tensor(np.array(group_centers), dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float32) if edge_index.numel() > 0 else torch.empty((0,1), dtype=torch.float32)
        y = []
        buffer = max(2, self.config.line_width * 1.5)
        for pos in node_features:
            pt = Point(float(pos[0]), float(pos[1]))
            found = False
            for line in all_lines:
                coords = np.array(line.coords)
                px_line = [((~transform * (x, y))[0] / size[1] * self.output_size[1],
                            (~transform * (x, y))[1] / size[0] * self.output_size[0]) for x, y in coords]
                shapely_line = LineString(px_line)
                if shapely_line.buffer(buffer).contains(pt):
                    found = True
                    break
            y.append(1.0 if found else 0.0)
        y = torch.tensor(y, dtype=torch.float32)
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'y': y
        }

    def generate_mask(self, geojson_path: str, img_id: str, output_path: str) -> Optional[str]:
        try:
            graph_data = self.prepare_mask(geojson_path, img_id)
            data = Data(
                x=graph_data['node_features'],
                edge_index=graph_data['edge_index'],
                edge_attr=graph_data['edge_attr'],
                y=graph_data['y']
            )
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(data, output_path)
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate graph mask: {str(e)}")
            return None