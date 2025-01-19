import os
from image_processing.image_loading import ImageLoader
from image_processing.image_merge import ImageMerger
from scripts.color_logger import ColorLogger

def prepare_data():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../NeuralSatSeg'))
    source_folder = os.path.join(BASE_DIR, "data/train/roads")
    destination_folder = os.path.join(BASE_DIR, "data/processed/roads")

    color_logger = ColorLogger("Data Preparation")
    logger = color_logger.get_logger()
    image_loader = ImageLoader(source_folder, destination_folder, "Image Loader")
    images_by_index = image_loader.load_images()

    if not images_by_index:
        logger.error("No images to merge. Process aborted.")
        return

    image_merger = ImageMerger(image_loader.images_folder, "Image Merger")
    image_merger.merge_images(images_by_index)

    logger.info("Data preparation completed.")

    counters = color_logger.get_counters()
    print(f"\nProcess Summary:")
    print(f"INFO: {counters['INFO']}")
    print(f"WARNING: {counters['WARNING']}")
    print(f"ERROR: {counters['ERROR']}")
    print(f"CRITICAL: {counters['CRITICAL']}")

if __name__ == "__main__":
    prepare_data()