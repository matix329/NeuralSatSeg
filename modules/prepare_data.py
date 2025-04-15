import os
import rasterio
import time
import numpy as np
import cv2

from image_processing.image_loading import ImageLoader
from image_processing.image_merge import ImageMerger
from preprocessing.preprocessing import Preprocessing
from mask_processing.mask_generator import MaskGenerator
from splitter.splitter import Splitter

class DataPreparator:
    def __init__(self, base_dir, image_size=(1300, 1300), test_size=0.2, seed=42):
        self.base_dir = base_dir
        self.source_folder = os.path.join(base_dir, "data/train/roads")
        self.geojson_folder = os.path.join(self.source_folder, "geojson_roads")
        self.output_base = os.path.join(base_dir, "data/processed")
        self.image_size = image_size
        self.test_size = test_size
        self.seed = seed

        self.train_image_dir = os.path.join(self.output_base, "train/roads/images")
        self.train_mask_dir = os.path.join(self.output_base, "train/roads/masks")
        self.val_image_dir = os.path.join(self.output_base, "val/roads/images")
        self.val_mask_dir = os.path.join(self.output_base, "val/roads/masks")

        for path in [self.train_image_dir, self.train_mask_dir, self.val_image_dir, self.val_mask_dir]:
            os.makedirs(path, exist_ok=True)

    def process_images_and_masks(self):
        loader = ImageLoader(self.source_folder)
        images_by_index = loader.load_all()

        merger = ImageMerger(reference_shape=self.image_size)
        merged_arrays = merger.merge_images_to_arrays(images_by_index)

        preprocessing = Preprocessing(image_size=self.image_size)
        mask_generator = MaskGenerator(
            geojson_folder=self.geojson_folder,
            output_size=self.image_size,
            line_width=1
        )

        self.data = []

        for key, image in merged_arrays.items():
            processed_image = preprocessing.preprocess_array(image)

            suffix = next((p for p in key.split("_") if p.startswith("img")), None)
            if not suffix:
                print(f"Could not extract imgXXXX from key: {key}")
                continue

            geojson_candidates = [
                f for f in os.listdir(self.geojson_folder)
                if f.endswith(".geojson") and suffix in f
            ]

            if not geojson_candidates:
                print(f"GeoJSON not found for key: {key}")
                continue

            geojson_path = os.path.join(self.geojson_folder, geojson_candidates[0])

            try:
                mask_array = mask_generator.generate_mask_from_array(geojson_path)
            except Exception as e:
                print(f"Mask generation failed for {key}: {e}")
                continue

            self.data.append((key, processed_image, mask_array))

    def split_data(self):
        splitter = Splitter(self.data, test_size=self.test_size, shuffle=True, seed=self.seed)
        train_data, val_data = splitter.split()

        for subset, image_dir, mask_dir in [
            (train_data, self.train_image_dir, self.train_mask_dir),
            (val_data, self.val_image_dir, self.val_mask_dir)
        ]:
            for index, img_arr, mask_path in subset:
                out_img_path = os.path.join(image_dir, f"{index}.png")
                out_mask_path = os.path.join(mask_dir, f"{index}.png")
                cv2.imwrite(out_img_path, (img_arr * 255).astype(np.uint8))
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                cv2.imwrite(out_mask_path, (mask * 255).astype(np.uint8))

    def run(self, stage="all"):
        if stage == "all":
            self.process_images_and_masks()
            self.split_data()
        elif stage == "process":
            self.process_images_and_masks()
        elif stage == "split":
            self.split_data()
        else:
            print(f"Unknown stage: {stage}")


if __name__ == "__main__":
    start = time.time()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../NeuralSatSeg'))
    preparator = DataPreparator(base_dir=base_dir)
    preparator.run(stage="all")
    end = time.time()
    print(f"Data preparation took {end - start:.2f} seconds.")