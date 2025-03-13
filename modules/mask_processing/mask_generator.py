import os
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import mapping

class MaskGenerator:
    def __init__(self, geojson_folder, masks_folder, output_size=(1300, 1300), line_width=1):
        self.geojson_folder = geojson_folder
        self.masks_folder = masks_folder
        self.output_size = output_size
        self.line_width = line_width

    def generate_mask(self, geojson_path, output_path):
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

        shapes = [(mapping(geom), 1) for geom in gdf.geometry if geom.is_valid]

        if not shapes:
            raise ValueError(f"No valid geometries to rasterize in {geojson_path}.")

        mask = rasterize(
            shapes,
            out_shape=self.output_size,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        )

        mask = np.where(mask > 0, 1, 0)

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=self.output_size[0],
            width=self.output_size[1],
            count=1,
            dtype=np.uint8,
            crs="EPSG:4326",
            transform=transform
        ) as dst:
            dst.write(mask, 1)

        return output_path, bounds, mask

    def process_masks(self):
        geojson_files = [f for f in os.listdir(self.geojson_folder) if f.endswith(".geojson")]
        for geojson_file in geojson_files:
            geojson_path = os.path.join(self.geojson_folder, geojson_file)
            output_path = os.path.join(self.masks_folder, f"{os.path.splitext(geojson_file)[0]}.tif")
            try:
                self.generate_mask(geojson_path, output_path)
                print(f"Mask for {geojson_file} saved to {output_path}.")
            except Exception as e:
                print(f"Failed to process {geojson_file}: {str(e)}")