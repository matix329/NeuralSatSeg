import numpy as np
import rasterio
from rasterio.enums import Resampling
import logging

logger = logging.getLogger(__name__)

class ImageMerger:
    def __init__(self, reference_shape=(1300, 1300)):
        self.reference_shape = reference_shape

    def normalize_band(self, band):
        min_val, max_val = np.min(band), np.max(band)
        if min_val == max_val:
            return np.zeros_like(band, dtype=np.uint8)
        band = (band - min_val) / (max_val - min_val)
        return (band * 255).astype(np.uint8)

    def resize_band(self, band, original_transform, original_shape):
        from rasterio.warp import reproject
        target_shape = self.reference_shape
        destination = np.zeros(target_shape, dtype=band.dtype)
        reproject(
            source=band,
            destination=destination,
            src_transform=original_transform,
            src_crs="EPSG:4326",
            dst_transform=rasterio.transform.from_origin(0, 0, 1, 1),
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
        )
        return destination

    def merge_images_to_arrays(self, images_by_key):
        result = {}
        logger.info(f"Starting to merge {len(images_by_key)} images")

        for key, image_array in images_by_key.items():
            try:
                logger.debug(f"Processing image {key}")
                normalized_channels = []
                for band_idx in range(image_array.shape[0]):
                    band = image_array[band_idx]
                    if band.shape != self.reference_shape:
                        logger.debug(f"Resizing band {band_idx} from {band.shape} to {self.reference_shape}")
                        band = self.resize_band(band, None, band.shape)
                    normalized_band = self.normalize_band(band)
                    normalized_channels.append(normalized_band)

                merged_array = np.stack(normalized_channels, axis=0)
                result[key] = merged_array
                logger.debug(f"Successfully merged image {key} with shape {merged_array.shape}")
                
            except Exception as e:
                logger.error(f"Failed to process image {key}: {str(e)}")
                continue

        logger.info(f"Successfully merged {len(result)} images")
        return result