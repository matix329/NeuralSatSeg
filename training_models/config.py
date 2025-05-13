BUILDING_MASK_OPTIONS = ["masks_original", "masks_eroded"]
ROAD_MASK_OPTIONS = ["masks_binary", "masks_graph"]

SUPPORTED_MODELS = ["unet", "cnn"]

DATA_DIR = "data/processed"
OUTPUT_DIR = "output/mlflow_artifacts/models"
LOG_DIR = "data/logs"

TRAINING_CONFIG = {
    "input_shape": (512, 512, 3),
    "num_classes": 1,
    "batch_size": 16,
    "epochs": 30,
    "learning_rate": 0.0001,
}

MASK_EXTENSIONS = {
    ".npy": "numpy",
    ".pt": "torch",
    ".png": "image"
}