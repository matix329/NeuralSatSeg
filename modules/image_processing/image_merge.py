import numpy as np
import rasterio
from rasterio.enums import Resampling

class ImageMerger:
    def __init__(self, reference_shape=(1300, 1300), logger=None):
        self.reference_shape = reference_shape
        self.logger = logger

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

        for key, channel_paths in images_by_key.items():
            merged_channels = []

            for channel, path in channel_paths.items():
                try:
                    with rasterio.open(path) as src:
                        data = src.read()
                        for band_idx in range(data.shape[0]):
                            band = data[band_idx]
                            if band.shape != self.reference_shape:
                                band = self.resize_band(band, src.transform, band.shape)
                            band = self.normalize_band(band)
                            merged_channels.append(band)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to load {path}: {e}")
                    continue

            if merged_channels:
                merged = np.stack(merged_channels, axis=-1)
                result[key] = merged

        return result