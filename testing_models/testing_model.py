import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from models_code.metrics.dice_metrics import dice_coefficient, iou_score
from models_code.metrics.classification_metrics import precision, recall, f1_score

matplotlib.use('Agg')

CLASS_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 0, 255],
}

def list_available_models(models_dir):
    models = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
    if not models:
        print("No .keras models found in the specified directory.")
        exit(1)
    return models

def select_model(models_dir):
    models = list_available_models(models_dir)
    print("Available models:")
    for i, model_name in enumerate(models, start=1):
        print(f"{i}. {model_name}")
    while True:
        try:
            choice = int(input("Select a model by number: "))
            if 1 <= choice <= len(models):
                return os.path.join(models_dir, models[choice - 1])
            else:
                print(f"Please select a number between 1 and {len(models)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def load_trained_model(model_path):
    try:
        custom_objects = {
            'dice_coefficient': dice_coefficient,
            'iou_score': iou_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def postprocess_prediction(prediction, threshold=0.5):
    binary_mask = (prediction >= threshold).astype(np.float32)
    return binary_mask

def mask_to_color(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in CLASS_COLORS.items():
        color_mask[mask == class_idx] = color
    return color_mask

def predict_and_save_results(model, test_images_dir, output_dir, image_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = tf.io.gfile.glob(f"{test_images_dir}/*.png")

    if not image_paths:
        print(f"No images found in directory {test_images_dir}")
        return

    print(f"Found {len(image_paths)} images in {test_images_dir}")

    for i, image_path in enumerate(image_paths):
        try:
            image = tf.image.decode_png(tf.io.read_file(image_path), channels=3)
            image = tf.image.resize(image, image_size) / 255.0
            image = tf.expand_dims(image, axis=0)

            prediction = model.predict(image, verbose=0)[0, :, :, 0]
            binary_mask = postprocess_prediction(prediction)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(image[0])
            ax[0].set_title("Input Image")
            ax[0].axis("off")

            ax[1].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
            ax[1].set_title("Predicted Mask")
            ax[1].axis("off")

            preview_path = os.path.join(output_dir, f"preview_{i}.png")
            plt.savefig(preview_path)
            plt.close(fig)
            print(f"Saved preview to {preview_path}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "output/mlflow_artifacts/models")
    test_images_dir = os.path.join(project_root, "data/processed/val/roads/images")
    output_dir = os.path.join(project_root, "data/results")

    model_path = select_model(models_dir)
    model = load_trained_model(model_path)
    predict_and_save_results(model, test_images_dir, output_dir)

if __name__ == "__main__":
    main()