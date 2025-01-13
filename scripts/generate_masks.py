import os
import cv2
import numpy as np
from skimage.segmentation import slic
from scripts.color_logger import ColorLogger

class MaskGenerator:
    def __init__(self, base_dir=None, class_map=None, n_segments=100, compactness=10):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_dir = os.path.abspath(os.path.join(script_dir, ".."))
        self.image_base_dir = os.path.join(self.project_dir, "data", "processed", "images") if base_dir is None else os.path.join(base_dir, "processed", "images")
        self.mask_base_dir = os.path.join(self.image_base_dir, "..", "masks")
        self.class_map = class_map or {}
        self.logger_instance = ColorLogger(__name__)
        self.logger = self.logger_instance.get_logger()
        self.n_segments = n_segments
        self.compactness = compactness

        self.log_counters = {"info": 0, "warning": 0, "error": 0}

    def increment_log_count(self, level):
        if level in self.log_counters:
            self.log_counters[level] += 1

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def process_image_with_slic(self, image_path, mask_path, class_value):
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.warning(f"Cannot read image {image_path}, skipping...")
                self.increment_log_count("warning")
                return

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            segments = slic(image, n_segments=self.n_segments, compactness=self.compactness, start_label=1)
            mask = np.zeros_like(segments, dtype=np.uint8)
            mask[segments > 0] = class_value

            if not cv2.imwrite(mask_path, mask):
                self.logger.error(f"Failed to write mask to {mask_path}")
                self.increment_log_count("error")
            else:
                self.logger.info(f"SLIC mask generated for {image_path}")
                self.increment_log_count("info")

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            self.increment_log_count("error")

    def process_class(self, split_image_dir, split_mask_dir, class_name, class_value):
        class_image_dir = os.path.join(split_image_dir, class_name)
        class_mask_dir = os.path.join(split_mask_dir, class_name)

        if not os.path.isdir(class_image_dir):
            self.logger.warning(f"{class_image_dir} is not a directory, skipping...")
            self.increment_log_count("warning")
            return

        self.ensure_directory_exists(class_mask_dir)

        for file_name in os.listdir(class_image_dir):
            if not file_name.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(class_image_dir, file_name)
            mask_path = os.path.join(class_mask_dir, file_name.replace(".jpg", ".png"))
            self.process_image_with_slic(img_path, mask_path, class_value)

    def generate_masks(self):
        for split in ["train", "val", "test"]:
            split_image_dir = os.path.join(self.image_base_dir, split)
            split_mask_dir = os.path.join(self.mask_base_dir, split)

            if not os.path.exists(split_image_dir):
                self.logger.error(f"Directory {split_image_dir} does not exist!")
                self.increment_log_count("error")
                continue

            self.ensure_directory_exists(split_mask_dir)

            for class_name, class_value in self.class_map.items():
                self.process_class(split_image_dir, split_mask_dir, class_name, class_value)

    def get_counters(self):
        return self.log_counters

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

    generator = MaskGenerator(class_map=class_mapping, n_segments=150, compactness=20)
    generator.generate_masks()
    print(generator.get_counters())