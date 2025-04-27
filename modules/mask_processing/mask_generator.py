import geopandas as gpd
import rasterio
import numpy as np
import logging
import os
from rasterio.features import rasterize
from shapely.geometry import mapping
from typing import Literal, Tuple

logger = logging.getLogger(__name__)

class MaskGenerator:
    def __init__(self, geojson_folder, category: Literal['roads', 'buildings'], line_width=1):
        self.geojson_folder = geojson_folder
        self.category = category
        self.line_width = line_width
        self.output_size = (1300, 1300) if category == 'roads' else (650, 650)

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

    def prepare_mask(self, geojson_path: str, img_id: str) -> np.ndarray:
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            raise ValueError(f"GeoJSON file {geojson_path} is empty or invalid.")

        transform, crs, size = self.get_tiff_parameters(img_id)
        gdf = gdf.to_crs(crs)
        
        if self.category == 'roads':
            projected_crs = "EPSG:3857"
            gdf_projected = gdf.to_crs(projected_crs)
            gdf_projected["geometry"] = gdf_projected["geometry"].buffer(self.line_width * 0.5)
            gdf = gdf_projected.to_crs(crs)
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
            out_shape=size,
            transform=transform,
            fill=0,
            dtype=np.float32 if self.category == 'buildings' else np.uint8,
            all_touched=True
        )

        if self.category == 'roads':
            mask = np.where(mask > 0, 255, 0)

        return mask

    def generate_mask(self, geojson_path: str, img_id: str, output_path: str) -> str:
        mask = self.prepare_mask(geojson_path, img_id)

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=np.float32 if self.category == 'buildings' else np.uint8,
            crs="EPSG:4326",
            transform=rasterio.transform.from_origin(0, 0, 1, 1)
        ) as dst:
            dst.write(mask, 1)

        return output_path

    def generate_mask_from_array(self, geojson_path: str, img_id: str) -> np.ndarray:
        return self.prepare_mask(geojson_path, img_id)