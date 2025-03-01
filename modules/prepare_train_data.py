import os
from image_processing.image_loading import ImageLoader
from image_processing.image_merge import ImageMerger
from mask_processing.mask_generator import MaskGenerator
from preprocessing.preprocessing import Preprocessing
from scripts.color_logger import ColorLogger
from resizer.resizer import ImageMaskResizer
import tensorflow as tf

def prepare_data(stage="all"):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../NeuralSatSeg'))
    source_folder = os.path.join(BASE_DIR, "data/train/roads")
    geojson_folder = os.path.join(source_folder, "geojson_roads")
    destination_folder = os.path.join(BASE_DIR, "data/processed/train/roads")
    images_folder = os.path.join(destination_folder, "images")
    masks_folder = os.path.join(destination_folder, "masks")
    processed_images_folder = os.path.join(destination_folder, "processed_images")
    processed_masks_folder = os.path.join(destination_folder, "processed_masks")

    color_logger = ColorLogger("Training Data Preparation")
    logger = color_logger.get_logger()

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(processed_images_folder, exist_ok=True)
    os.makedirs(processed_masks_folder, exist_ok=True)

    def process_images_and_masks():
        logger.info("Starting image processing...")
        image_loader = ImageLoader(source_folder, images_folder, "Image Loader")
        images_by_index = image_loader.load_images()

        if not images_by_index:
            logger.error("No images to merge. Process aborted.")
            return

        image_merger = ImageMerger(images_folder, "Image Merger")
        image_merger.merge_images(images_by_index)

        logger.info("Starting mask processing...")
        mask_processor = MaskGenerator(geojson_folder, masks_folder)
        mask_processor.process_masks()

    def preprocess_images_and_masks():
        logger.info("Starting preprocessing of images and masks...")
        preprocessing = Preprocessing(image_size=(1300, 1300))

        for image_name in os.listdir(images_folder):
            if not image_name.lower().endswith((
                '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif')):
                logger.warning(f"Skipping non-image file: {image_name}")
                continue
            image_path = os.path.join(images_folder, image_name)
            processed_image = preprocessing.load_and_preprocess_image(image_path)

            processed_image_name = os.path.splitext(image_name)[0] + ".png"
            processed_image_path = os.path.join(processed_images_folder, processed_image_name)
            tf.keras.utils.save_img(processed_image_path, processed_image.numpy())
            logger.info(f"Saved processed image: {processed_image_path}")

        for mask_name in os.listdir(masks_folder):
            if not mask_name.lower().endswith((
                '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif')):
                logger.warning(f"Skipping non-mask file: {mask_name}")
                continue
            mask_path = os.path.join(masks_folder, mask_name)
            processed_mask = preprocessing.load_and_preprocess_mask(mask_path)

            processed_mask = tf.expand_dims(processed_mask, axis=-1)
            processed_mask_name = os.path.splitext(mask_name)[0] + ".png"
            processed_mask_path = os.path.join(processed_masks_folder, processed_mask_name)
            tf.keras.utils.save_img(processed_mask_path, processed_mask.numpy(), scale=False)
            logger.info(f"Saved processed mask: {processed_mask_path}")

    def resize_images_and_masks():
        logger.info("Resizing images and masks to target size (512x512)...")
        resizer = ImageMaskResizer(target_size=(512, 512))
        resizer.resize(processed_images_folder, processed_images_folder)
        resizer.resize(processed_masks_folder, processed_masks_folder)

    if stage == "all":
        process_images_and_masks()
        preprocess_images_and_masks()
        resize_images_and_masks()
    elif stage == "process":
        process_images_and_masks()
    elif stage == "preprocess":
        preprocess_images_and_masks()
    elif stage == "resize":
        resize_images_and_masks()
    else:
        logger.error(f"Unknown stage: {stage}")

if __name__ == "__main__":
    prepare_data(stage="all")