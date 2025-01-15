import os
import geopandas as gpd
import pandas as pd
import rasterio

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_geojson(self, subdir):
        geojson_dir = os.path.join(self.data_dir, subdir)
        geojson_files = [os.path.join(geojson_dir, f) for f in os.listdir(geojson_dir) if f.endswith(".geojson")]

        if not geojson_files:
            raise FileNotFoundError(f"No GeoJSON files found in {geojson_dir}")

        geodataframes = [gpd.read_file(file) for file in geojson_files]
        return pd.concat(geodataframes, ignore_index=True)

    def load_images(self, subdir):
        image_dir = os.path.join(self.data_dir, subdir)
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory {image_dir} does not exist.")

        images = []
        for file_name in os.listdir(image_dir):
            if file_name.endswith(".tif"):
                with rasterio.open(os.path.join(image_dir, file_name)) as src:
                    images.append((file_name, src.read(), src.transform, src.crs))
        return images