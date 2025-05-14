BUILDING_MASK_OPTIONS = ["masks_original", "masks_eroded"]
ROAD_MASK_OPTIONS = ["masks_binary", "masks_graph"]

SELECTED_BUILDING_MASK = BUILDING_MASK_OPTIONS[0]
SELECTED_ROAD_MASK = ROAD_MASK_OPTIONS[0]

SUPPORTED_MODELS = ["unet", "cnn"]
MODEL_TYPE = 'unet'
NUM_HEADS = 2
HEAD_NAMES = ['buildings', 'roads']
LOSS_WEIGHTS = {'buildings': 1.0, 'roads': 1.0}

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

MASK_PATHS = {
    'buildings': f'buildings/{SELECTED_BUILDING_MASK}',
    'roads': f'roads/{SELECTED_ROAD_MASK}'
}