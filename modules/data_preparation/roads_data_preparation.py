import os
import time
import numpy as np
import cv2
import logging
import re
import gc

from modules.image_processing.image_loading import ImageLoader
from modules.mask_processing.mask_generator import MaskGenerator
from modules.splitter.splitter import Splitter
from modules.image_filtering.image_filtering import ImageFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RoadsDataPreparator:
    def __init__(self, base_dir, city_name, image_size=(1300, 1300), test_size=0.2, seed=42, batch_size=1,
                 black_threshold=0.0, min_content_ratio=1.0):
        self.base_dir = base_dir
        self.city_name = city_name
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
        self.temp_image_dir = os.path.join(self.output_base, "temp/roads/images")
        self.temp_mask_dir = os.path.join(self.output_base, "temp/roads/masks")
        self.train_image_dir = os.path.join(self.output_base, "train/roads/images")
        self.train_mask_dir = os.path.join(self.output_base, "train/roads/masks")
        self.val_image_dir = os.path.join(self.output_base, "val/roads/images")
        self.val_mask_dir = os.path.join(self.output_base, "val/roads/masks")
        for path in [self.temp_image_dir, self.temp_mask_dir,
                    self.train_image_dir, self.train_mask_dir,
                    self.val_image_dir, self.val_mask_dir]:
            os.makedirs(path, exist_ok=True)

    def process_images_and_masks(self):
        logger.info("Starting image and mask processing")
        loader = ImageLoader(self.base_dir, category='roads', batch_size=self.batch_size)
        mask_generator = MaskGenerator(
            geojson_folder=self.geojson_folder,
            category='roads',
            line_width=1
        )
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
                    mask_array = mask_generator.generate_mask_from_array(geojson_path, img_id)
                    if np.all(mask_array == 0):
                        continue
                    out_img_path = os.path.join(self.temp_image_dir, f"{self.city_name}_road_img{img_num}.png")
                    out_mask_path = os.path.join(self.temp_mask_dir, f"{self.city_name}_road_mask{img_num}.png")
                    self.save_image(image, out_img_path)
                    self.save_image(mask_array, out_mask_path)
                    total_processed += 1
                    if total_processed % 10 == 0:
                        logger.info(f"Processed {total_processed} images so far")
                        gc.collect()
                except Exception as e:
                    logger.error(f"Error processing {img_id}: {str(e)}")
                    total_errors += 1
                    continue
            del batch
            gc.collect()
        logger.info(f"Processing completed. Successfully processed: {total_processed}, Errors: {total_errors}")

    def save_image(self, image: np.ndarray, path: str):
        try:
            image = image.copy()
            if np.all(image == 0):
                logger.debug(f"Skipping empty image: {path}")
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
                logger.debug(f"Skipping zero image after processing: {path}")
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, image)
            logger.debug(f"Successfully saved image to: {path}")
            gc.collect()
        except Exception as e:
            logger.error(f"Error saving image to {path}: {str(e)}")
            raise

    def filter_processed_data(self):
        logger.info("Starting image filtering")
        image_filter = ImageFilter(
            processed_dir=self.output_base,
            black_threshold=self.black_threshold,
            min_content_ratio=self.min_content_ratio
        )
        stats = image_filter.run()
        logger.info(f"Filtering completed. {stats['kept_images']} images kept, {stats['removed_images']} images removed")
        gc.collect()
        return stats

    def split_data(self):
        logger.info("Starting data splitting")
        processed_images = os.listdir(self.temp_image_dir)
        processed_masks = os.listdir(self.temp_mask_dir)
        
        image_numbers = set()
        mask_numbers = set()
        
        for filename in processed_images:
            if filename.endswith('.png'):
                match = re.search(r'img(\d+)', filename)
                if match:
                    image_numbers.add(match.group(1))
                    
        for filename in processed_masks:
            if filename.endswith('.png'):
                match = re.search(r'mask(\d+)', filename)
                if match:
                    mask_numbers.add(match.group(1))
                    
        common_numbers = image_numbers & mask_numbers
        
        if not common_numbers:
            logger.error("No common numbers found between images and masks!")
            return
            
        data = []
        for number in common_numbers:
            img_file = f"{self.city_name}_road_img{number}.png"
            mask_file = f"{self.city_name}_road_mask{number}.png"
            data.append((img_file, mask_file))
            
        splitter = Splitter(data, test_size=self.test_size, shuffle=True, seed=self.seed)
        train_data, val_data = splitter.split()
        
        import shutil
        for subset, image_dir, mask_dir in [
            (train_data, self.train_image_dir, self.train_mask_dir),
            (val_data, self.val_image_dir, self.val_mask_dir)
        ]:
            for img_file, mask_file in subset:
                try:
                    src_img_path = os.path.join(self.temp_image_dir, img_file)
                    src_mask_path = os.path.join(self.temp_mask_dir, mask_file)
                    
                    if not os.path.exists(src_img_path) or not os.path.exists(src_mask_path):
                        continue
                        
                    dst_img_path = os.path.join(image_dir, img_file)
                    dst_mask_path = os.path.join(mask_dir, mask_file)
                    
                    shutil.copy2(src_img_path, dst_img_path)
                    shutil.copy2(src_mask_path, dst_mask_path)
                except Exception as e:
                    logger.error(f"Error copying {img_file} and {mask_file}: {str(e)}")
        gc.collect()

    def run(self, stage="all"):
        start_time = time.time()
        try:
            if stage == "all":
                self.process_images_and_masks()
                self.filter_processed_data()
                self.split_data()
                temp_dir = os.path.join(self.output_base, "temp")
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.info(f"Removed temporary directory: {temp_dir}")
            elif stage == "process":
                self.process_images_and_masks()
            elif stage == "filter":
                self.filter_processed_data()
            elif stage == "split":
                self.split_data()
            else:
                logger.error(f"Unknown stage: {stage}")
                return
            end_time = time.time()
            logger.info(f"Data preparation completed in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during data preparation: {str(e)}")
            raise 