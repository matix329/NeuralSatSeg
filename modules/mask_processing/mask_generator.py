import geopandas as gpd
import rasterio
import numpy as np
import logging
import os
import warnings
import cv2
from rasterio.features import rasterize
from shapely.geometry import mapping
from typing import Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseMaskGenerator(ABC):
    def __init__(self, geojson_folder):
        self.geojson_folder = geojson_folder
        self.output_size = (650, 650)

    def get_tiff_parameters(self, img_id: str) -> Tuple[rasterio.transform.Affine, str, Tuple[int, int]]:
        tiff_path = None
        for root, _, files in os.walk(os.path.dirname(os.path.dirname(self.geojson_folder))):
            for fname in files:
                if fname.endswith(('.tif', '.tiff')) and img_id in fname:
                    tiff_path = os.path.join(root, fname)
                    break
            if tiff_path:
                break
        if not tiff_path:
            raise ValueError(f"TIFF file not found for {img_id}")
        with rasterio.open(tiff_path) as src:
            return src.transform, src.crs, (src.width, src.height)

    @abstractmethod
    def prepare_mask(self, geojson_path: str, img_id: str) -> np.ndarray:
        pass

    def generate_mask(self, geojson_path: str, img_id: str, output_path: str) -> str:
        mask = self.prepare_mask(geojson_path, img_id)
        transform, crs, _ = self.get_tiff_parameters(img_id)
        dtype = np.float32 if isinstance(self, BuildingMaskGenerator) else np.uint8
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(mask, 1)
        return output_path

    def generate_mask_from_array(self, geojson_path: str, img_id: str) -> np.ndarray:
        return self.prepare_mask(geojson_path, img_id)

class RoadMaskGenerator(BaseMaskGenerator):
    def __init__(self, geojson_folder, line_width=1):
        super().__init__(geojson_folder)
        self.line_width = line_width
        self.output_size = (1300, 1300)

    def prepare_mask(self, geojson_path: str, img_id: str) -> np.ndarray:
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            raise ValueError(f"GeoJSON file {geojson_path} is empty or invalid.")
        
        transform, crs, size = self.get_tiff_parameters(img_id)
        gdf = gdf.to_crs(crs)
        
        tiff_path = None
        for root, _, files in os.walk(os.path.dirname(os.path.dirname(self.geojson_folder))):
            for fname in files:
                if fname.endswith(('.tif', '.tiff')) and img_id in fname:
                    tiff_path = os.path.join(root, fname)
                    break
            if tiff_path:
                break
                
        if not tiff_path:
            raise ValueError(f"TIFF file not found for {img_id}")
            
        with rasterio.open(tiff_path) as src:
            res_x, res_y = src.res[0], src.res[1]
            
        px_size = (res_x + res_y) / 2
        buffer_m = self.line_width * px_size
        gdf_projected = gdf.to_crs(crs)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gdf_projected["geometry"] = gdf_projected["geometry"].buffer(buffer_m / 2)
        
        shapes = [(mapping(geom), 1) for geom in gdf_projected.geometry if geom.is_valid]
        
        if not shapes:
            raise ValueError(f"No valid geometries to rasterize in {geojson_path}.")
            
        mask = rasterize(
            shapes,
            out_shape=size,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        )
        
        return np.where(mask > 0, 255, 0)

class BuildingMaskGenerator(BaseMaskGenerator):
    def __init__(self, geojson_folder):
        super().__init__(geojson_folder)
        self.output_size = (650, 650)

    def prepare_mask(self, geojson_path: str, img_id: str) -> np.ndarray:
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            raise ValueError(f"GeoJSON file {geojson_path} is empty or invalid.")
            
        transform, crs, size = self.get_tiff_parameters(img_id)
        gdf = gdf.to_crs(crs)
        
        shapes = []
        for _, row in gdf.iterrows():
            value = row['partialDec'] if row['partialBuilding'] == 1 else 1.0
            shapes.append((mapping(row.geometry), value))
            
        if not shapes:
            raise ValueError(f"No valid geometries to rasterize in {geojson_path}.")
            
        mask = rasterize(
            shapes,
            out_shape=size,
            transform=transform,
            fill=0,
            dtype=np.float32,
            all_touched=True
        )
        return mask