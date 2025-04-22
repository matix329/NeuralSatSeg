import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageFilter:
    def __init__(self, processed_dir: str, black_threshold: float = 0.1, min_content_ratio: float = 0.85):
        self.processed_dir = processed_dir
        self.black_threshold = black_threshold
        self.min_content_ratio = min_content_ratio

        self.train_image_dir = os.path.join(processed_dir, "train/roads/images")
        self.train_mask_dir = os.path.join(processed_dir, "train/roads/masks")
        self.val_image_dir = os.path.join(processed_dir, "val/roads/images")
        self.val_mask_dir = os.path.join(processed_dir, "val/roads/masks")

        for dir_path in [self.train_image_dir, self.train_mask_dir,
                         self.val_image_dir, self.val_mask_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")

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
            (self.train_image_dir, self.train_mask_dir),
            (self.val_image_dir, self.val_mask_dir)
        ]:
            for img_name in os.listdir(image_dir):
                if not img_name.endswith('.png'):
                    continue

                img_path = os.path.join(image_dir, img_name)
                mask_path = os.path.join(mask_dir, img_name)

                keep_image, stats = self.analyze_image(img_path)

                if keep_image:
                    kept_images.append(img_path)
                    logger.info(f"Keeping {img_name} - Stats: {stats}")
                else:
                    removed_images.append(img_path)
                    logger.info(f"Removing {img_name} - Stats: {stats}")

                    try:
                        os.remove(img_path)
                        if os.path.exists(mask_path):
                            os.remove(mask_path)
                    except Exception as e:
                        logger.error(f"Error removing files for {img_name}: {str(e)}")

        return kept_images, removed_images

    def run(self) -> Dict:
        logger.info("Starting image filtering process")

        try:
            kept_images, removed_images = self.filter_images()

            stats = {
                "total_processed": len(kept_images) + len(removed_images),
                "kept_images": len(kept_images),
                "removed_images": len(removed_images),
                "kept_ratio": len(kept_images) / (len(kept_images) + len(removed_images))
            }

            logger.info(f"Filtering completed. Statistics: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error during filtering process: {str(e)}")
            raise 