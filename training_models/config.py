BUILDING_MASK_OPTIONS = ["masks_original", "masks_eroded"]
ROAD_MASK_OPTIONS = ["masks_binary", "masks_graph"]

SELECTED_BUILDING_MASK = BUILDING_MASK_OPTIONS[0]
SELECTED_ROAD_MASK = ROAD_MASK_OPTIONS[0]

SUPPORTED_MODELS = ["unet", "cnn"]
MODEL_TYPE = 'unet'
NUM_HEADS = 2
HEAD_NAMES = ['buildings', 'roads']
LOSS_WEIGHTS = {'head_buildings': 0.5, 'head_roads': 2.0}

DATA_DIR = "data/processed"
OUTPUT_DIR = "output/mlflow_artifacts/models"
LOG_DIR = "data/logs"

TRAINING_CONFIG = {
    "input_shape": (512, 512, 3),
    "num_classes": 1,
    "batch_size": 16,
    "epochs": 20,
    "learning_rate": 0.0001,
    "dropout_rate": 0.3,
}

MASK_EXTENSIONS = {
    ".pt": "torch",
    ".png": "image"
}

MASK_PATHS = {
    'buildings': f'buildings/{SELECTED_BUILDING_MASK}',
    'roads': f'roads/{SELECTED_ROAD_MASK}'
}

MASK_CONFIG = {
    'buildings': {
        'min_pixels': 100,
        'erosion_kernel_size': 3,
        'erosion_iterations': 1
    },
    'roads': {
        'min_pixels': 100,
        'line_width': 7
    }
}