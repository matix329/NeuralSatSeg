import rasterio
import numpy as np
import logging
import os
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MaskConfig:
    min_pixels: int = 100
    line_width: int = 5
    erosion_kernel_size: int = 3
    erosion_iterations: int = 1
    min_coverage_percent: float = 0.5

class BaseMaskGenerator(ABC):
    def __init__(self, geojson_folder: str, config: Optional[MaskConfig] = None):
        self.geojson_folder = geojson_folder
        self.config = config or MaskConfig()
        self.output_size = (1300, 1300)

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

    def validate_mask(self, mask: np.ndarray) -> bool:
        road_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage_percent = (road_pixels / total_pixels) * 100
        
        if road_pixels < self.config.min_pixels:
            logger.warning(f"Mask has less than {self.config.min_pixels} non-zero pixels ({road_pixels})")
            return False
            
        if coverage_percent < self.config.min_coverage_percent:
            logger.warning(f"Mask coverage {coverage_percent:.2f}% is below minimum {self.config.min_coverage_percent}%")
            return False
            
        return True

    def generate_mask(self, geojson_path: str, img_id: str, output_path: str) -> Optional[str]:
        mask = self.prepare_mask(geojson_path, img_id)
        
        if not self.validate_mask(mask):
            return None
            
        transform, crs, _ = self.get_tiff_parameters(img_id)
        dtype = np.float32 if mask.dtype == np.float32 else np.uint8
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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