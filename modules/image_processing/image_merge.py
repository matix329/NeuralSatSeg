import os
import rasterio
import numpy as np
from rasterio.enums import Resampling
from scripts.color_logger import ColorLogger
import warnings
from rasterio.errors import NotGeoreferencedWarning
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

class ImageMerger:
    def __init__(self, output_folder, logger_name):
        self.output_folder = output_folder
        self.logger = ColorLogger(logger_name).get_logger()

    @staticmethod
    def normalize_band(band):
        min_val, max_val = np.min(band), np.max(band)
        if min_val == max_val:
            return np.zeros_like(band, dtype=np.uint8)
        band = (band - min_val) / (max_val - min_val)
        band = np.clip(band * 255, 0, 255).astype(np.uint8)
        return band

    def process_band(self, folder, file_path, reference_shape):
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1, resampling=Resampling.bilinear)
                if data.shape != reference_shape:
                    data = self.resize_image(data, reference_shape, src.transform)
                normalized_band = self.normalize_band(data)
                self.logger.info(f"Loaded and resized band from {file_path}")
                return folder, normalized_band
        except Exception as e:
            self.logger.error(f"Error loading band from {file_path}: {e}")
            return None

    def merge_images(self, images_by_index):
        for index, images in images_by_index.items():
            reference_shape = (1300, 1300)

            with ThreadPoolExecutor() as executor:
                results = executor.map(lambda item: self.process_band(item[0], item[1], reference_shape), images.items())

            bands = {folder: band for folder, band in results if folder is not None}

            if not bands:
                self.logger.error(f"No bands found for index {index}. Skipping...")
                continue

            merged_image = self.combine_bands(bands)
            if merged_image is None:
                self.logger.error(f"Failed to create a merged image for index {index}. Skipping...")
                continue

            output_path = os.path.join(self.output_folder, f"SN3_roads_img{index}.tif")
            self.save_image(output_path, merged_image)

    def resize_image(self, image, target_shape, transform):
        from rasterio.warp import reproject
        new_image = np.zeros(target_shape, dtype=image.dtype)
        reproject(
            source=image,
            destination=new_image,
            src_transform=transform,
            src_crs="EPSG:4326",
            dst_transform=rasterio.transform.from_origin(0, 0, 1, 1),
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
        )
        return new_image

    def combine_bands(self, bands):
        rgb = []
        if "RGB-PanSharpen" in bands:
            if bands["RGB-PanSharpen"].ndim == 2:
                rgb = [bands["RGB-PanSharpen"]] * 3
            else:
                rgb = [bands["RGB-PanSharpen"][:, :, i] for i in range(3)]
        elif "PAN" in bands:
            rgb = [bands["PAN"]] * 3

        multispectral = []
        if "MUL-PanSharpen" in bands:
            if bands["MUL-PanSharpen"].ndim == 2:
                multispectral = [bands["MUL-PanSharpen"]]
            else:
                multispectral = [bands["MUL-PanSharpen"][:, :, i] for i in range(8)]

        if rgb:
            merged = np.stack(rgb + multispectral, axis=-1) if multispectral else np.stack(rgb, axis=-1)
            return merged
        return None

    def save_image(self, output_path, image_data):
        try:
            height, width, bands = image_data.shape
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=bands,
                dtype=image_data.dtype
            ) as dst:
                for i in range(bands):
                    dst.write(image_data[:, :, i], i + 1)
            self.logger.info(f"Saved merged image: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving image {output_path}: {e}")