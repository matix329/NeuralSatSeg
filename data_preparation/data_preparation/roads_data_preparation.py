import os
import time
import numpy as np
import cv2
import logging
import re
import gc
import shutil
from pathlib import Path
from PIL import Image

from image_processing.image_loading import ImageLoader
from mask_processing.roads_masks import RoadBinaryMaskGenerator, RoadGraphMaskGenerator
from splitter.splitter import Splitter
from image_filtering.image_filtering import ImageFilter
from mask_processing.mask_generator import MaskConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RoadsDataPreparator:
    def __init__(self, base_dir, city_name, processed, image_size=(650, 650), test_size=0.2, seed=42, batch_size=1,
                 black_threshold=0.5, min_content_ratio=0.1, brightness_factor=1.5, saturation_factor=1.3):
        self.base_dir = Path(base_dir)
        self.city_name = city_name
        self.data_dir = self.base_dir / "data" / "train" / city_name
        self.output_dir = Path(processed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_folder = self.data_dir / "roads" / "PS-RGB"
        if not self.source_folder.exists():
            raise FileNotFoundError(f"Source folder not found: {self.source_folder}")
        self.geojson_folder = self.data_dir / "roads" / "geojson_roads"
        if not self.geojson_folder.exists():
            raise FileNotFoundError(f"GeoJSON folder not found: {self.geojson_folder}")
        self.image_size = image_size
        self.test_size = test_size
        self.seed = seed
        self.batch_size = batch_size
        self.black_threshold = black_threshold
        self.min_content_ratio = min_content_ratio
        self.temp_image_dir = self.output_dir / "temp/roads/images"
        self.temp_mask_binary_dir = self.output_dir / "temp/roads/masks_binary"
        self.temp_mask_graph_dir = self.output_dir / "temp/roads/masks_graph"
        self.train_image_dir = self.output_dir / "train/roads/images"
        self.train_mask_binary_dir = self.output_dir / "train/roads/masks_binary"
        self.train_mask_graph_dir = self.output_dir / "train/roads/masks_graph"
        self.val_image_dir = self.output_dir / "val/roads/images"
        self.val_mask_binary_dir = self.output_dir / "val/roads/masks_binary"
        self.val_mask_graph_dir = self.output_dir / "val/roads/masks_graph"
        self.brightness_factor = brightness_factor
        self.saturation_factor = saturation_factor
        
        for path in [self.temp_image_dir, self.temp_mask_binary_dir, self.temp_mask_graph_dir,
                    self.train_image_dir, self.train_mask_binary_dir, self.train_mask_graph_dir,
                    self.val_image_dir, self.val_mask_binary_dir, self.val_mask_graph_dir]:
            path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RoadsDataPreparator for {city_name}")

    def process_images_and_masks(self):
        logger.info("Starting image and mask processing")
        image_files = list(self.source_folder.glob('*.tif'))
        if not image_files:
            logger.warning("No image files found to process!")
            return
            
        loader = ImageLoader(self.source_folder, category='roads', batch_size=self.batch_size)
        
        mask_config = MaskConfig(
            min_pixels=100,
            line_width=8,
            erosion_kernel_size=3,
            erosion_iterations=1,
            save_debug_mask=False,
            min_coverage_percent=0.5
        )
        
        binary_mask_generator = RoadBinaryMaskGenerator(geojson_folder=self.geojson_folder, config=mask_config)
        graph_mask_generator = RoadGraphMaskGenerator(geojson_folder=self.geojson_folder, config=mask_config)
        
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
                continue
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
                    geojson_path = self.geojson_folder / matching_geojson
                    
                    binary_mask = binary_mask_generator.generate_mask_from_array(geojson_path, img_id)
                    if np.all(binary_mask == 0):
                        continue
                        
                    graph_data = graph_mask_generator.prepare_mask(geojson_path, img_id)
                    
                    out_img_path = self.temp_image_dir / f"{self.city_name}_road_img{img_num}.tif"
                    out_binary_mask_path = self.temp_mask_binary_dir / f"{self.city_name}_road_mask{img_num}.tif"
                    out_graph_mask_path = self.temp_mask_graph_dir / f"{self.city_name}_road_mask{img_num}.pt"
                    
                    try:
                        self.save_image(image, out_img_path)
                        self.save_image(binary_mask, out_binary_mask_path)
                        import torch
                        logger.info(f"Saving graph mask: {out_graph_mask_path}")
                        empty_data = torch.zeros(1, 2)
                        torch.save(empty_data, out_graph_mask_path)
                        logger.info(f"Graph mask saved: {out_graph_mask_path}")
                    except Exception as e:
                        logger.error(f"Failed to save image/mask for {img_id}: {str(e)}")
                        continue
                        
                    total_processed += 1
                    if total_processed % 10 == 0:
                        logger.info(f"Processed {total_processed} images")
                except Exception as e:
                    logger.error(f"Error processing {img_id}: {str(e)}")
                    total_errors += 1
                    continue
            del batch
            gc.collect()
        logger.info(f"Processing completed. Successfully processed: {total_processed}, Errors: {total_errors}")

    def resize_image(self, image: np.ndarray, is_mask: bool = False) -> np.ndarray:
        if image.shape[:2] == self.image_size:
            return image
            
        if is_mask:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR
            
        return cv2.resize(image, self.image_size, interpolation=interpolation)

    def save_image(self, image: np.ndarray, path: Path):
        try:
            import rasterio
            from rasterio.transform import from_origin
            image = image.copy()
            if np.all(image == 0):
                logger.warning(f"Image to save is empty (all zeros): {path}")
                return False
            is_mask = "masks" in str(path)
            if is_mask:
                dtype = 'uint8' if image.dtype == np.uint8 else 'float32'
                if len(image.shape) == 3:
                    if image.shape[2] == 1:
                        image = image[:, :, 0]
                    elif image.shape[0] == 1:
                        image = image[0, :, :]
                if len(image.shape) == 2:
                    image = image[np.newaxis, ...]
                count = 1
            else:
                dtype = 'uint16' if image.dtype == np.uint16 else 'uint8'
                if len(image.shape) == 3:
                    if image.shape[0] == 3:
                        count = 3
                    elif image.shape[2] == 3:
                        image = np.moveaxis(image, -1, 0)
                        count = 3
                    else:
                        image = image.squeeze()
                        count = 1
                elif len(image.shape) == 2:
                    count = 1
                    image = image[np.newaxis, ...]
                else:
                    image = image.squeeze()
                    count = 1
            transform = from_origin(0, 0, 1, 1)
            with rasterio.open(
                str(path), 'w',
                driver='GTiff',
                height=image.shape[1] if len(image.shape) == 3 else image.shape[0],
                width=image.shape[2] if len(image.shape) == 3 else image.shape[1],
                count=count,
                dtype=dtype,
                crs=None,
                transform=transform
            ) as dst:
                dst.write(image)
            logger.info(f"Image saved successfully: {path}")
            gc.collect()
            return True
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
        
        if not self.temp_image_dir.exists():
            logger.error(f"Temp image directory does not exist: {self.temp_image_dir}")
            return
        if not self.temp_mask_binary_dir.exists():
            logger.error(f"Temp binary mask directory does not exist: {self.temp_mask_binary_dir}")
            return
        if not self.temp_mask_graph_dir.exists():
            logger.error(f"Temp graph mask directory does not exist: {self.temp_mask_graph_dir}")
            return
            
        processed_images = os.listdir(self.temp_image_dir)
        processed_binary_masks = os.listdir(self.temp_mask_binary_dir)
        processed_graph_masks = os.listdir(self.temp_mask_graph_dir)
        
        logger.info(f"Found {len(processed_images)} images, {len(processed_binary_masks)} binary masks, {len(processed_graph_masks)} graph masks")
        
        image_numbers = set()
        binary_mask_numbers = set()
        graph_mask_numbers = set()
        
        for filename in processed_images:
            if filename.endswith('.tif'):
                match = re.search(r'img(\d+)', filename)
                if match:
                    image_numbers.add(match.group(1))
                    
        for filename in processed_binary_masks:
            if filename.endswith('.tif'):
                match = re.search(r'mask(\d+)', filename)
                if match:
                    binary_mask_numbers.add(match.group(1))
                    
        for filename in processed_graph_masks:
            if filename.endswith('.pt'):
                match = re.search(r'mask(\d+)', filename)
                if match:
                    graph_mask_numbers.add(match.group(1))
                    
        logger.info(f"Extracted numbers: {len(image_numbers)} images, {len(binary_mask_numbers)} binary masks, {len(graph_mask_numbers)} graph masks")
        
        common_numbers = image_numbers & binary_mask_numbers & graph_mask_numbers
        if not common_numbers:
            logger.error("No common numbers found between images and masks!")
            logger.error(f"Image numbers: {sorted(list(image_numbers))[:10]}...")
            logger.error(f"Binary mask numbers: {sorted(list(binary_mask_numbers))[:10]}...")
            logger.error(f"Graph mask numbers: {sorted(list(graph_mask_numbers))[:10]}...")
            return
            
        logger.info(f"Found {len(common_numbers)} common image-mask pairs")
        
        data = []
        for number in common_numbers:
            img_file = f"{self.city_name}_road_img{number}.tif"
            binary_mask_file = f"{self.city_name}_road_mask{number}.tif"
            graph_mask_file = f"{self.city_name}_road_mask{number}.pt"
            data.append((img_file, binary_mask_file, graph_mask_file))
            
        splitter = Splitter(data, test_size=self.test_size, shuffle=True, seed=self.seed)
        train_data, val_data = splitter.split()

        logger.info(f"Splitting into {len(train_data)} train and {len(val_data)} validation samples")

        for subset, image_dir, binary_mask_dir, graph_mask_dir in [
            (train_data, self.train_image_dir, self.train_mask_binary_dir, self.train_mask_graph_dir),
            (val_data, self.val_image_dir, self.val_mask_binary_dir, self.val_mask_graph_dir)
        ]:
            for img_file, binary_mask_file, graph_mask_file in subset:
                shutil.copy2(self.temp_image_dir / img_file, image_dir / img_file)
                shutil.copy2(self.temp_mask_binary_dir / binary_mask_file, binary_mask_dir / binary_mask_file)
                shutil.copy2(self.temp_mask_graph_dir / graph_mask_file, graph_mask_dir / graph_mask_file)
                
        logger.info("Data splitting completed")

    def run(self, stage="all"):
        start_time = time.time()
        try:
            if stage == "all":
                logger.info("=== STAGE 1: Processing images and masks ===")
                self.process_images_and_masks()
                logger.info("=== STAGE 2: Skipping filtering for roads (TIFF images are high quality) ===")
                # self.filter_processed_data()  # Wyłączone dla roads
                logger.info("=== STAGE 3: Splitting data ===")
                self.split_data()
                logger.info("=== STAGE 4: Cleaning up temp directory ===")
                temp_dir = self.output_dir / "temp"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.info(f"Removed temporary directory: {temp_dir}")
                else:
                    logger.warning(f"Temp directory does not exist: {temp_dir}")
            elif stage == "process":
                self.process_images_and_masks()
            elif stage == "filter":
                self.filter_processed_data()
            elif stage == "split":
                self.split_data()
            else:
                logger.error(f"Unknown stage: {stage}")
            end_time = time.time()
            logger.info(f"Data preparation completed in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise 