import os
import time
import numpy as np
import cv2
import logging
import re

from image_processing.image_loading import ImageLoader
from mask_processing.mask_generator import MaskGenerator
from splitter.splitter import Splitter
from image_filtering.image_filtering import ImageFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(self, base_dir, image_size=(1300, 1300), test_size=0.2, seed=42, batch_size=10,
                 black_threshold=0.1, min_content_ratio=0.95):
        self.base_dir = base_dir
        
        self.source_folder = os.path.join(base_dir, "data/train/roads")
        if not os.path.exists(self.source_folder):
            raise FileNotFoundError(f"Source folder not found: {self.source_folder}")
            
        self.geojson_folder = os.path.join(self.source_folder, "geojson_roads")
        if not os.path.exists(self.geojson_folder):
            raise FileNotFoundError(f"GeoJSON folder not found: {self.geojson_folder}")
            
        self.output_base = os.path.join(base_dir, "data/processed")
        os.makedirs(self.output_base, exist_ok=True)
        
        self.image_size = image_size
        self.test_size = test_size
        self.seed = seed
        self.batch_size = batch_size
        self.black_threshold = black_threshold
        self.min_content_ratio = min_content_ratio

        self.train_image_dir = os.path.join(self.output_base, "train/roads/images")
        self.train_mask_dir = os.path.join(self.output_base, "train/roads/masks")
        self.val_image_dir = os.path.join(self.output_base, "val/roads/images")
        self.val_mask_dir = os.path.join(self.output_base, "val/roads/masks")

        for path in [self.train_image_dir, self.train_mask_dir, self.val_image_dir, self.val_mask_dir]:
            os.makedirs(path, exist_ok=True)

    def process_images_and_masks(self):
        logger.info("Starting image and mask processing")
        loader = ImageLoader(self.source_folder, target_size=self.image_size, batch_size=self.batch_size)
        mask_generator = MaskGenerator(
            geojson_folder=self.geojson_folder,
            output_size=self.image_size,
            line_width=1
        )

        self.data = []
        total_processed = 0
        total_errors = 0

        geojson_files = {}
        for f in os.listdir(self.geojson_folder):
            if not f.endswith(".geojson"):
                continue
            match = re.search(r'img(\d+)', f)
            if match:
                img_num = match.group(1)
                geojson_files[img_num] = f

        for batch in loader.load_all():
            for img_id, image in batch.items():
                try:
                    if np.all(image == 0):
                        continue
                    
                    match = re.search(r'img(\d+)', img_id)
                    if not match:
                        total_errors += 1
                        continue

                    img_num = match.group(1)
                    
                    if img_num not in geojson_files:
                        total_errors += 1
                        continue

                    matching_geojson = geojson_files[img_num]
                    geojson_path = os.path.join(self.geojson_folder, matching_geojson)
                    mask_array = mask_generator.generate_mask_from_array(geojson_path)
                    
                    if np.all(mask_array == 0):
                        continue
                    
                    self.data.append((f"img{img_num}", image, mask_array))
                    total_processed += 1
                    
                    if total_processed % 100 == 0:
                        logger.info(f"Processed {total_processed} images so far")
                    
                except Exception as e:
                    logger.error(f"Error processing {img_id}: {str(e)}")
                    total_errors += 1
                    continue

        logger.info(f"Processing completed. Successfully processed: {total_processed}, Errors: {total_errors}")

    def save_image(self, image: np.ndarray, path: str):
        try:
            image = image.copy()
            
            if np.all(image == 0):
                return
            
            if "masks" in path:
                if np.max(image) == 1:
                    image = (image * 255).astype(np.uint8)
                elif np.max(image) == 255:
                    image = image.astype(np.uint8)
            else:
                if image.dtype == np.float32 or image.dtype == np.float64:
                    min_val = np.min(image)
                    max_val = np.max(image)
                    if max_val > min_val:
                        image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        image = (image * 255).astype(np.uint8)
                elif image.dtype == np.uint16:
                    image = (image / 8).astype(np.uint8)
                elif image.dtype != np.uint8:
                    image = image.astype(np.uint8)
            
            if len(image.shape) == 3 and image.shape[0] > 3:
                if "masks" in path:
                    image = image[0]
                else:
                    image = image[-3:]
            
            if len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            
            if np.all(image == 0):
                return
            
            cv2.imwrite(path, image)
        except Exception as e:
            logger.error(f"Error saving image to {path}: {str(e)}")
            raise

    def split_data(self):
        logger.info("Starting data splitting")
        splitter = Splitter(self.data, test_size=self.test_size, shuffle=True, seed=self.seed)
        train_data, val_data = splitter.split()
        logger.info(f"Split data into train: {len(train_data)}, validation: {len(val_data)}")

        for subset, image_dir, mask_dir in [
            (train_data, self.train_image_dir, self.train_mask_dir),
            (val_data, self.val_image_dir, self.val_mask_dir)
        ]:
            for index, img_arr, mask_arr in subset:
                try:
                    out_img_path = os.path.join(image_dir, f"{index}.png")
                    out_mask_path = os.path.join(mask_dir, f"{index}.png")
                    
                    self.save_image(img_arr, out_img_path)
                    self.save_image(mask_arr, out_mask_path)
                    
                except Exception as e:
                    logger.error(f"Error saving {index}: {str(e)}")

    def filter_processed_data(self):
        logger.info("Starting image filtering")
        image_filter = ImageFilter(
            processed_dir=self.output_base,
            black_threshold=self.black_threshold,
            min_content_ratio=self.min_content_ratio
        )
        stats = image_filter.run()
        logger.info(f"Filtering completed. {stats['kept_images']} images kept, {stats['removed_images']} images removed")
        return stats

    def run(self, stage="all"):
        start_time = time.time()
        
        try:
            if stage == "all":
                self.process_images_and_masks()
                self.split_data()
                self.filter_processed_data()
            elif stage == "process":
                self.process_images_and_masks()
            elif stage == "split":
                self.split_data()
            elif stage == "filter":
                self.filter_processed_data()
            else:
                logger.error(f"Unknown stage: {stage}")
                return
                
            end_time = time.time()
            logger.info(f"Data preparation completed in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during data preparation: {str(e)}")
            raise

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    preparator = DataPreparator(base_dir=base_dir, batch_size=5)
    preparator.run(stage="all")