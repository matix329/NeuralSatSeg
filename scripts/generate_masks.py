import os
import cv2
import numpy as np
from scripts.color_logger import ColorLogger

class MaskGenerator:
    def __init__(self, base_dir=None, class_map=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_dir = os.path.abspath(os.path.join(script_dir, ".."))
        self.image_base_dir = os.path.join(self.project_dir, "data", "processed", "images") if base_dir is None else os.path.join(base_dir, "processed", "images")
        self.mask_base_dir = os.path.join(self.image_base_dir, "..", "masks")
        self.class_map = class_map or {}
        self.logger_instance = ColorLogger(__name__)
        self.logger = self.logger_instance.get_logger()

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def process_class(self, split_image_dir, split_mask_dir, class_name, class_value):
        class_image_dir = os.path.join(split_image_dir, class_name)
        class_mask_dir = os.path.join(split_mask_dir, class_name)

        if not os.path.isdir(class_image_dir):
            self.logger.warning(f"{class_image_dir} is not a directory, skipping...")
            return

        self.ensure_directory_exists(class_mask_dir)

        for file_name in os.listdir(class_image_dir):
            if not file_name.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(class_image_dir, file_name)
            mask_path = os.path.join(class_mask_dir, file_name.replace(".jpg", ".png"))

            try:
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if image is None:
                    self.logger.warning(f"Cannot read image {img_path}, skipping...")
                    continue

                height, width, _ = image.shape
                mask = np.full((height, width), class_value, dtype=np.uint8)
                if not cv2.imwrite(mask_path, mask):
                    self.logger.error(f"Failed to write mask to {mask_path}")

                self.logger.info(f"Mask generated for {file_name} in {class_name}")

            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")

    def generate_masks(self):
        for split in ["train", "val", "test"]:
            split_image_dir = os.path.join(self.image_base_dir, split)
            split_mask_dir = os.path.join(self.mask_base_dir, split)

            if not os.path.exists(split_image_dir):
                self.logger.error(f"Directory {split_image_dir} does not exist!")
                continue

            self.ensure_directory_exists(split_mask_dir)

            for class_name, class_value in self.class_map.items():
                self.process_class(split_image_dir, split_mask_dir, class_name, class_value)

    def validate_dataset(self):
        for split in ["train", "val", "test"]:
            image_dir = os.path.join(self.image_base_dir, split)
            mask_dir = os.path.join(self.mask_base_dir, split)

            if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
                self.logger.error(f"Missing directories for {split}, skipping validation.")
                continue

            for class_name in os.listdir(image_dir):
                class_image_dir = os.path.join(image_dir, class_name)
                class_mask_dir = os.path.join(mask_dir, class_name)

                if not os.path.isdir(class_image_dir) or not os.path.isdir(class_mask_dir):
                    continue

                image_files = [f for f in os.listdir(class_image_dir) if f.lower().endswith(".jpg")]
                mask_files = [f for f in os.listdir(class_mask_dir) if f.lower().endswith(".png")]

                if len(image_files) != len(mask_files):
                    self.logger.error(f"Mismatch in {split}/{class_name}: {len(image_files)} images, {len(mask_files)} masks!")
                else:
                    self.logger.info(f"Validation passed for {split}/{class_name}")

    def report_log_summary(self):
        counters = self.logger_instance.get_counters()
        self.logger.info(f"Log summary: {counters}")

if __name__ == "__main__":
    class_mapping = {
        "AnnualCrop": 1,
        "Forest": 2,
        "HerbaceousVegetation": 3,
        "Highway": 4,
        "Industrial": 5,
        "Pasture": 6,
        "PermanentCrop": 7,
        "Residential": 8,
        "River": 9,
        "SeaLake": 10
    }

    generator = MaskGenerator(class_map=class_mapping)
    generator.generate_masks()
    generator.validate_dataset()
    generator.report_log_summary()