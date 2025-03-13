import os
import rasterio
from rasterio.enums import Resampling
from PIL import Image
from modules.image_processing.image_loading import ImageLoader
from modules.image_processing.image_merge import ImageMerger
from modules.preprocessing.preprocessing import Preprocessing
from modules.resizer.resizer import ImageMaskResizer
from scripts.color_logger import ColorLogger

def prepare_data(stage="all"):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../NeuralSatSeg'))
    source_folder = os.path.join(BASE_DIR, "data/test/roads")
    destination_folder = os.path.join(BASE_DIR, "data/processed/test/roads")
    images_folder = os.path.join(destination_folder, "images")
    processed_images_folder = os.path.join(destination_folder, "processed_images")

    color_logger = ColorLogger("Test Data Preparation")
    logger = color_logger.get_logger()

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(processed_images_folder, exist_ok=True)

    def process_images():
        logger.info("Starting image processing...")
        image_loader = ImageLoader(source_folder, images_folder, "Image Loader")
        image_loader.supported_folders = ["MUL", "MUL-PanSharpen", "PAN", "RGB-PanSharpen"]
        images_by_index = image_loader.load_images()

        if not images_by_index:
            logger.error("No images to merge. Process aborted.")
            return

        image_merger = ImageMerger(images_folder, "Image Merger")

        for index, images in images_by_index.items():
            bands = {}
            reference_shape = (1300, 1300)

            for folder, file_path in images.items():
                try:
                    with rasterio.open(file_path) as src:
                        data = src.read(1, resampling=Resampling.bilinear)
                        if data.shape != reference_shape:
                            data = image_merger.resize_image(data, reference_shape, src.transform)

                        bands[folder] = image_merger.normalize_band(data)  # Poprawiona literówka
                        logger.info(f"Loaded and resized band from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading band from {file_path}: {e}")

            if len(bands) == 0:
                logger.error(f"No bands found for index {index}. Skipping...")
                continue

            merged_image = image_merger.combine_bands(bands)
            if merged_image is None:
                logger.error(f"Failed to create a merged image for index {index}. Skipping...")
                continue

            output_path = os.path.join(images_folder, f"SN3_roads_test_img{index}.tif")
            image_merger.save_image(output_path, merged_image)

    def preprocess_images():
        logger.info("Starting preprocessing of images...")
        preprocessing = Preprocessing(image_size=(1300, 1300))

        for image_name in os.listdir(images_folder):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif')):
                logger.warning(f"Skipping non-image file: {image_name}")
                continue

            image_path = os.path.join(images_folder, image_name)
            processed_image = preprocessing.load_and_preprocess_image(image_path)

            processed_image_name = os.path.splitext(image_name)[0] + ".png"
            processed_image_path = os.path.join(processed_images_folder, processed_image_name)

            img = Image.fromarray((processed_image * 255).astype("uint8"))
            img.save(processed_image_path, "PNG")

            logger.info(f"Saved processed image: {processed_image_path}")

    def resize_images():
        logger.info("Resizing images to target size (512x512)...")
        resizer = ImageMaskResizer(target_size=(512, 512))
        resizer.resize_image(processed_images_folder, processed_images_folder)

    if stage == "all":
        process_images()
        preprocess_images()
        resize_images()
    elif stage == "process":
        process_images()
    elif stage == "preprocess":
        preprocess_images()
    elif stage == "resize":
        resize_images()
    else:
        logger.error(f"Unknown stage: {stage}")

if __name__ == "__main__":
    prepare_data(stage="all")