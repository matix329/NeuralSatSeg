from preprocessing.data_loader import DataLoader
from preprocessing.augmenter import Augmenter
from preprocessing.normalizer import DataNormalizer
from mask_processing.mask_generator import MaskGenerator
from mask_processing.mask_validator import MaskValidator
from tiling.tiler import Tiler
from preprocessing.data_export import DataExporter

def process_roads(data_dir, output_dir):
    loader = DataLoader(data_dir)
    images = loader.load_images("PS-RGB")
    geojson_data = loader.load_geojson("geojson_roads")

    for img_name, image, transform, crs in images:
        mask_generator = MaskGenerator(image.shape[1:], transform)
        mask = mask_generator.generate_mask(geojson_data)

        MaskValidator.validate_masks([(img_name, image)], [(img_name, mask)])

        normalized_image = DataNormalizer.normalize(image)

        augmenter = Augmenter()
        exporter = DataExporter(output_dir)
        
        augmented_tiles = augmenter.augment(normalized_image, mask)
        for aug_idx, (aug_img, aug_mask) in enumerate(augmented_tiles):
            exporter.export_tile(aug_img, aug_mask, f"{img_name}_aug{aug_idx}")

def main():
    raw_data_dir = "../data/train"
    processed_data_dir = "../data/processed/roads"

    print("Process data for train catalog...")
    process_roads(raw_data_dir, processed_data_dir)

if __name__ == "__main__":
    main()