import os
import tensorflow as tf
from image_processing.image_loading import ImageLoader
from image_processing.image_merge import ImageMerger
from mask_processing.mask_generator import MaskGenerator
from preprocessing.preprocessing import Preprocessing
from scripts.color_logger import ColorLogger
from resizer.resizer import ImageMaskResizer
import time

class DataPreparator:
    def __init__(self, base_dir, image_size=(1300, 1300), resize_size=(512, 512)):
        self.base_dir = os.path.abspath(base_dir)
        self.image_size = image_size
        self.resize_size = resize_size

        self.source_folder = os.path.join(self.base_dir, "data/train/roads")
        self.geojson_folder = os.path.join(self.source_folder, "geojson_roads")
        self.destination_folder = os.path.join(self.base_dir, "data/processed/train/roads")
        self.images_folder = os.path.join(self.destination_folder, "images")
        self.masks_folder = os.path.join(self.destination_folder, "masks")
        self.processed_images_folder = os.path.join(self.destination_folder, "processed_images")
        self.processed_masks_folder = os.path.join(self.destination_folder, "processed_masks")

        self.color_logger = ColorLogger("Training Data Preparation")
        self.logger = self.color_logger.get_logger()

        self.create_folders()

    def create_folders(self):
        for folder in [self.images_folder, self.masks_folder, self.processed_images_folder, self.processed_masks_folder]:
            os.makedirs(folder, exist_ok=True)

    def process_images_and_masks(self):
        try:
            self.logger.info("Starting image processing...")
            image_loader = ImageLoader(self.source_folder, self.images_folder, "Image Loader")
            images_by_index = image_loader.load_images()

            if not images_by_index:
                self.logger.error("No images to merge. Process aborted.")
                return

            image_merger = ImageMerger(self.images_folder, "Image Merger")
            image_merger.merge_images(images_by_index)

            self.logger.info("Starting mask processing...")
            mask_processor = MaskGenerator(self.geojson_folder, self.masks_folder)
            mask_processor.process_masks()
        except Exception as e:
            self.logger.error(f"Error during image and mask processing: {e}")

    def preprocess_images_and_masks(self):
        try:
            self.logger.info("Starting preprocessing of images and masks...")
            preprocessing = Preprocessing(image_size=self.image_size)

            for image_name in os.listdir(self.images_folder):
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif')):
                    self.logger.warning(f"Skipping non-image file: {image_name}")
                    continue

                image_path = os.path.join(self.images_folder, image_name)
                processed_image = preprocessing.load_and_preprocess_image(image_path)
                processed_image_path = os.path.join(self.processed_images_folder, f"{os.path.splitext(image_name)[0]}.png")
                tf.keras.utils.save_img(processed_image_path, processed_image)
                self.logger.info(f"Saved processed image: {processed_image_path}")

            for mask_name in os.listdir(self.masks_folder):
                if not mask_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif')):
                    self.logger.warning(f"Skipping non-mask file: {mask_name}")
                    continue

                mask_path = os.path.join(self.masks_folder, mask_name)
                processed_mask = preprocessing.load_and_preprocess_mask(mask_path)
                processed_mask = tf.expand_dims(processed_mask, axis=-1)
                processed_mask_path = os.path.join(self.processed_masks_folder, f"{os.path.splitext(mask_name)[0]}.png")
                tf.keras.utils.save_img(processed_mask_path, processed_mask, scale=False)
                self.logger.info(f"Saved processed mask: {processed_mask_path}")
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")

    def resize_images_and_masks(self):
        try:
            self.logger.info(f"Resizing images and masks to target size {self.resize_size}...")
            resizer = ImageMaskResizer(target_size=self.resize_size)
            resizer.resize_directory(self.processed_images_folder, self.processed_images_folder, is_mask=False)
            resizer.resize_directory(self.processed_masks_folder, self.processed_masks_folder, is_mask=True)
        except Exception as e:
            self.logger.error(f"Error during resizing: {e}")

    def run(self, stage="all"):
        if stage == "all":
            self.process_images_and_masks()
            self.preprocess_images_and_masks()
            self.resize_images_and_masks()
        elif stage == "process":
            self.process_images_and_masks()
        elif stage == "preprocess":
            self.preprocess_images_and_masks()
        elif stage == "resize":
            self.resize_images_and_masks()
        else:
            self.logger.error(f"Unknown stage: {stage}")

if __name__ == "__main__":
    start = time.time()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../NeuralSatSeg'))
    preparator = DataPreparator(base_dir=base_dir)
    preparator.run(stage="all")
    end = time.time()
    print(f"Data preparation took {end - start} seconds.")