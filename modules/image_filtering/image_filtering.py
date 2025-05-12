import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageFilter:
    def __init__(self, processed_dir: str, black_threshold: float = 0.1, min_content_ratio: float = 0.85):
        self.processed_dir = processed_dir
        self.black_threshold = black_threshold
        self.min_content_ratio = min_content_ratio

        self.roads_image_dir = os.path.join(processed_dir, "temp/roads/images")
        self.roads_mask_dir = os.path.join(processed_dir, "temp/roads/masks")
        self.buildings_image_dir = os.path.join(processed_dir, "temp/buildings/images")
        self.buildings_mask_dir = os.path.join(processed_dir, "temp/buildings/masks")
        self.buildings_mask_original_dir = os.path.join(processed_dir, "temp/buildings/masks_original")
        self.buildings_mask_eroded_dir = os.path.join(processed_dir, "temp/buildings/masks_eroded")

        for dir_path in [self.roads_image_dir, self.roads_mask_dir,
                        self.buildings_image_dir, self.buildings_mask_dir,
                        self.buildings_mask_original_dir, self.buildings_mask_eroded_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

    def analyze_image(self, image_path: str) -> Tuple[bool, Dict]:
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, {"error": "Failed to load image"}

            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            black_pixels = np.sum(gray == 0)
            total_pixels = gray.size
            black_ratio = black_pixels / total_pixels

            rows, cols = gray.shape
            quadrants = [
                gray[:rows // 2, :cols // 2],
                gray[:rows // 2, cols // 2:],
                gray[rows // 2:, :cols // 2],
                gray[rows // 2:, cols // 2:]
            ]

            quadrant_content = [np.sum(q > 0) / q.size for q in quadrants]
            min_quadrant_content = min(quadrant_content)

            stats = {
                "black_ratio": black_ratio,
                "min_quadrant_content": min_quadrant_content,
                "quadrant_content": quadrant_content
            }

            keep_image = (black_ratio <= self.black_threshold and
                          min_quadrant_content >= self.min_content_ratio)

            return keep_image, stats

        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {str(e)}")
            return False, {"error": str(e)}

    def filter_images(self) -> Tuple[List[str], List[str]]:
        kept_images = []
        removed_images = []

        for image_dir, mask_dir in [
            (self.roads_image_dir, self.roads_mask_dir),
            (self.buildings_image_dir, self.buildings_mask_dir)
        ]:
            if not os.path.exists(image_dir):
                continue

            for img_name in os.listdir(image_dir):
                if not img_name.endswith('.png'):
                    continue

                img_path = os.path.join(image_dir, img_name)

                keep_image, stats = self.analyze_image(img_path)

                if keep_image:
                    kept_images.append(img_path)
                    logger.info(f"Keeping {img_name} - Stats: {stats}")
                else:
                    removed_images.append(img_path)
                    logger.info(f"Removing {img_name} - Stats: {stats}")

                    try:
                        os.remove(img_path)
                        match = re.search(r'img(\d+)', img_name)
                        if match:
                            img_num = match.group(1)
                            found_mask = None
                            for mask_file in os.listdir(mask_dir):
                                if img_num in mask_file and mask_file.endswith('.png'):
                                    found_mask = mask_file
                                    break
                            if found_mask:
                                mask_path = os.path.join(mask_dir, found_mask)
                                os.remove(mask_path)
                                if "buildings" in mask_dir:
                                    mask_original_path = os.path.join(self.buildings_mask_original_dir, found_mask)
                                    mask_eroded_path = os.path.join(self.buildings_mask_eroded_dir, found_mask.replace('.png', '.npy'))
                                    if os.path.exists(mask_original_path):
                                        os.remove(mask_original_path)
                                    if os.path.exists(mask_eroded_path):
                                        os.remove(mask_eroded_path)
                            else:
                                logger.warning(f"No matching mask found for {img_name} (img_num: {img_num}) in {mask_dir}")
                        else:
                            logger.warning(f"Could not extract img_num from {img_name} to find mask in {mask_dir}")
                    except Exception as e:
                        logger.error(f"Error removing files for {img_name}: {str(e)}")

        return kept_images, removed_images

    def run(self) -> Dict:
        logger.info("Starting image filtering process")

        try:
            kept_images, removed_images = self.filter_images()
            total_images = len(kept_images) + len(removed_images)
            
            stats = {
                "total_processed": total_images,
                "kept_images": len(kept_images),
                "removed_images": len(removed_images),
                "kept_ratio": len(kept_images) / total_images if total_images > 0 else 0.0
            }

            logger.info(f"Filtering completed. Statistics: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error during filtering process: {str(e)}")
            raise 