import geopandas as gpd
import numpy as np
import cv2
import logging
from rasterio.features import rasterize
from shapely.geometry import mapping
from typing import Optional
from .mask_generator import BaseMaskGenerator, MaskConfig

logger = logging.getLogger(__name__)

class BuildingMaskGenerator(BaseMaskGenerator):
    def __init__(self, geojson_folder: str, config: Optional[MaskConfig] = None):
        super().__init__(geojson_folder, config)
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

    def generate_eroded_mask(self, mask_array: np.ndarray) -> np.ndarray:
        binary = (mask_array > 0).astype(np.uint8)
        kernel = np.ones((self.config.erosion_kernel_size, self.config.erosion_kernel_size), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=self.config.erosion_iterations)
        mask_eroded = mask_array.astype(np.int16)
        mask_eroded[(binary == 1) & (eroded == 0)] = -1
        return mask_eroded 