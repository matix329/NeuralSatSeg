import os
import time
import numpy as np
import cv2
import logging
import re
import gc
from pathlib import Path

from modules.image_processing.image_loading import ImageLoader
from modules.mask_processing.mask_generator import MaskGenerator
from modules.splitter.splitter import Splitter
from modules.image_filtering.image_filtering import ImageFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BuildingDataPreparator:
    def __init__(self, base_dir, city_name, processed, image_size=(650, 650), test_size=0.2, seed=42, batch_size=1,
                 black_threshold=0.0, min_content_ratio=1.0):
        self.base_dir = Path(base_dir)
        self.city_name = city_name
        self.data_dir = self.base_dir / "data" / "train" / city_name
        self.output_dir = Path(processed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_folder = self.data_dir / "buildings" / "RGB-PanSharpen"
        if not self.source_folder.exists():
            raise FileNotFoundError(f"Source folder not found: {self.source_folder}")
        self.geojson_folder = self.data_dir / "buildings" / "geojson_buildings"
        if not self.geojson_folder.exists():
            raise FileNotFoundError(f"GeoJSON folder not found: {self.geojson_folder}")
        self.image_size = image_size
        self.test_size = test_size
        self.seed = seed
        self.batch_size = batch_size
        self.black_threshold = black_threshold
        self.min_content_ratio = min_content_ratio
        self.temp_image_dir = self.output_dir / "temp/buildings/images"
        self.temp_mask_dir = self.output_dir / "temp/buildings/masks"
        self.train_image_dir = self.output_dir / "train/buildings/images"
        self.train_mask_dir = self.output_dir / "train/buildings/masks"
        self.val_image_dir = self.output_dir / "val/buildings/images"
        self.val_mask_dir = self.output_dir / "val/buildings/masks"
        for path in [self.temp_image_dir, self.temp_mask_dir,
                    self.train_image_dir, self.train_mask_dir,
                    self.val_image_dir, self.val_mask_dir]:
            path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized BuildingDataPreparator for {city_name}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def process_images_and_masks(self):
        logger.info("Starting building image and mask processing")
        image_files = list(self.source_folder.glob('*.tif'))
        logger.info(f"Found {len(image_files)} image files in {self.source_folder}")
        if not image_files:
            logger.warning("No image files found to process!")
        loader = ImageLoader(self.source_folder, category='buildings', batch_size=self.batch_size)
        mask_generator = MaskGenerator(
            geojson_folder=self.geojson_folder,
            category='buildings',
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
            if not batch:
                logger.warning("Batch is empty! No images loaded in this batch.")
            for img_id, image in batch.items():
                try:
                    logger.info(f"Processing image: {img_id}")
                    if np.all(image == 0):
                        logger.warning(f"Image {img_id} is empty (all zeros), skipping.")
                        continue
                    match = re.search(r'img(\d+)', img_id)
                    if not match:
                        logger.warning(f"Image id {img_id} does not match expected pattern.")
                        total_errors += 1
                        continue
                    img_num = match.group(1)
                    if img_num not in geojson_files:
                        logger.warning(f"No matching geojson for image {img_id}")
                        total_errors += 1
                        continue
                    matching_geojson = geojson_files[img_num]
                    geojson_path = self.geojson_folder / matching_geojson
                    logger.info(f"Generating mask for image {img_id} using {matching_geojson}")
                    if os.path.getsize(geojson_path) == 0:
                        logger.warning(f"GeoJSON file {geojson_path} is empty, skipping.")
                        continue
                    mask_array = mask_generator.generate_mask_from_array(geojson_path, img_id)
                    if np.all(mask_array == 0):
                        logger.warning(f"Generated mask for {img_id} is empty (all zeros), skipping.")
                        continue
                    out_img_path = self.temp_image_dir / f"{self.city_name}_building_img{img_num}.png"
                    out_mask_path = self.temp_mask_dir / f"{self.city_name}_building_mask{img_num}.png"
                    try:
                        self.save_image(image, out_img_path)
                        logger.info(f"Saved image to {out_img_path}")
                    except Exception as e:
                        logger.error(f"Failed to save image {out_img_path}: {str(e)}")
                    try:
                        self.save_image(mask_array, out_mask_path)
                        logger.info(f"Saved mask to {out_mask_path}")
                    except Exception as e:
                        logger.error(f"Failed to save mask {out_mask_path}: {str(e)}")
                    total_processed += 1
                    if total_processed % 10 == 0:
                        logger.info(f"Processed {total_processed} building images so far")
                        gc.collect()
                except Exception as e:
                    logger.error(f"Error processing building {img_id}: {str(e)}")
                    total_errors += 1
                    continue
            del batch
            gc.collect()
        logger.info(f"Building processing completed. Successfully processed: {total_processed}, Errors: {total_errors}")

    def save_image(self, image: np.ndarray, path: Path):
        try:
            logger.info(f"Saving image. Path type: {type(path)}, value: {path}")
            image = image.copy()
            if np.all(image == 0):
                logger.debug(f"Skipping empty image: {path}")
                return
            if "masks" in str(path):
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
                if "masks" in str(path):
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
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Calling cv2.imwrite with path: {str(path)}")
            cv2.imwrite(str(path), image)
            logger.debug(f"Successfully saved image to: {path}")
            gc.collect()
        except Exception as e:
            logger.error(f"Error saving image to {path}: {str(e)}")
            raise

    def filter_processed_data(self):
        logger.info("Starting image filtering")
        image_filter = ImageFilter(
            processed_dir=self.output_dir,
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
            img_file = f"{self.city_name}_building_img{number}.png"
            mask_file = f"{self.city_name}_building_mask{number}.png"
            data.append((img_file, mask_file))
        splitter = Splitter(data, test_size=self.test_size, shuffle=True, seed=self.seed)
        train_data, val_data = splitter.split()
        import shutil
        for subset, image_dir, mask_dir in [
            (train_data, self.output_dir / "train/buildings/images", self.output_dir / "train/buildings/masks"),
            (val_data, self.output_dir / "val/buildings/images", self.output_dir / "val/buildings/masks")
        ]:
            for img_file, mask_file in subset:
                try:
                    src_img_path = self.temp_image_dir / img_file
                    src_mask_path = self.temp_mask_dir / mask_file
                    if not src_img_path.exists() or not src_mask_path.exists():
                        continue
                    dst_img_path = image_dir / img_file
                    dst_mask_path = mask_dir / mask_file
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
                temp_dir = self.output_dir / "temp"
                if temp_dir.exists():
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
            logger.info(f"Building data preparation completed in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during building data preparation: {str(e)}")
            raise 