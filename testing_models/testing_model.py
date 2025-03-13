import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

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
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def postprocess_prediction(prediction, top_percent=0.95):
    threshold = np.quantile(prediction, top_percent)
    binary_mask = prediction >= threshold
    inverted_mask = 1 - binary_mask.astype(np.float32)
    return inverted_mask

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
            image = tf.image.resize_image(image, image_size) / 255.0
            image = tf.expand_dims(image, axis=0)

            prediction = model.predict(image, verbose=0)[0, :, :, 0]

            processed_prediction = postprocess_prediction(prediction)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(image[0])
            ax[0].set_title("Input Image")
            ax[0].axis("off")

            ax[1].imshow(processed_prediction, cmap="gray", vmin=0, vmax=1)
            ax[1].set_title("Segmented Roads")
            ax[1].axis("off")

            preview_path = os.path.join(output_dir, f"preview_{i}.png")
            plt.savefig(preview_path)
            plt.close(fig)
            print(f"Saved preview to {preview_path}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

def main():
    models_dir = "../output/mlflow_artifacts/models"
    test_images_dir = "../data/processed/test/roads/processed_images"
    output_dir = "../data/results"

    model_path = select_model(models_dir)
    model = load_trained_model(model_path)
    predict_and_save_results(model, test_images_dir, output_dir)

if __name__ == "__main__":
    main()