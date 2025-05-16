import os
import tensorflow as tf
from scripts.color_logger import ColorLogger
from models_code.metrics.dice_metrics import dice_coefficient, iou_score
from models_code.metrics.classification_metrics import precision, recall, f1_score

logger = ColorLogger("ModelTester").get_logger()

def load_model(model_path):
    try:
        custom_objects = {
            'dice_coefficient': dice_coefficient,
            'iou_score': iou_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def process_image(model, image_path):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [512, 512]) / 255.0
        image = tf.expand_dims(image, 0)
        binary_map = tf.zeros_like(image[..., :1])
        prediction = model.predict([image, binary_map])
        return prediction
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def main():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "mlflow_artifacts", "models")
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return

    available_models = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    if not available_models:
        logger.error("No models found in the directory.")
        return

    print("Available models:")
    for i, model_name in enumerate(available_models, 1):
        print(f"{i}. {model_name}")

    try:
        model_index = int(input("Select a model by number: ")) - 1
        if model_index < 0 or model_index >= len(available_models):
            logger.error("Invalid model selection.")
            return
    except ValueError:
        logger.error("Please enter a valid number.")
        return

    model_path = os.path.join(models_dir, available_models[model_index])
    model = load_model(model_path)
    if model is None:
        return

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "val")
    for category in ["roads", "buildings"]:
        images_dir = os.path.join(data_dir, category, "images")
        if not os.path.exists(images_dir):
            logger.error(f"Directory not found: {images_dir}")
            continue

        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        logger.info(f"Found {len(image_files)} images in {images_dir}")

        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            prediction = process_image(model, image_path)
            if prediction is not None:
                logger.info(f"Processed {image_file}")

if __name__ == "__main__":
    main()