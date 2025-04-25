import geopandas as gpd
import rasterio
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import mapping
import logging
from typing import Literal

logger = logging.getLogger(__name__)

class MaskGenerator:
    def __init__(self, geojson_folder, category: Literal['roads', 'buildings'], line_width=1):
        self.geojson_folder = geojson_folder
        self.category = category
        self.line_width = line_width
        self.output_size = (1300, 1300) if category == 'roads' else (650, 650)

    def prepare_mask(self, geojson_path: str, apply_transform: bool = True):
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            raise ValueError(f"GeoJSON file {geojson_path} is empty or invalid.")

        gdf = gdf.to_crs("EPSG:3857")
        gdf["geometry"] = gdf["geometry"].buffer(self.line_width * 0.5)
        gdf = gdf.to_crs("EPSG:4326")

        bounds = gdf.total_bounds
        transform = rasterio.transform.from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3],
            self.output_size[1], self.output_size[0]
        )

        if self.category == 'roads':
            shapes = [(mapping(geom), 1) for geom in gdf.geometry if geom.is_valid]
        else:
            shapes = []
            for _, row in gdf.iterrows():
                value = row['partialDec'] if row['partialBuilding'] == 1 else 1.0
                shapes.append((mapping(row.geometry), value))

        if not shapes:
            raise ValueError(f"No valid geometries to rasterize in {geojson_path}.")

        mask = rasterize(
            shapes,
            out_shape=self.output_size,
            transform=transform,
            fill=0,
            dtype=np.float32 if self.category == 'buildings' else np.uint8,
            all_touched=True
        )

        if self.category == 'roads':
            mask = np.where(mask > 0, 255, 0)

        return mask, bounds, transform

    def generate_mask(self, geojson_path, output_path):
        mask, bounds, transform = self.prepare_mask(geojson_path, apply_transform=True)

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=self.output_size[0],
            width=self.output_size[1],
            count=1,
            dtype=np.float32 if self.category == 'buildings' else np.uint8,
            crs="EPSG:4326",
            transform=transform
        ) as dst:
            dst.write(mask, 1)

        return output_path, bounds, mask

    def generate_mask_from_array(self, geojson_path: str) -> np.ndarray:
        mask, _, _ = self.prepare_mask(geojson_path, apply_transform=True)
        return mask