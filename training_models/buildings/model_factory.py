from models_code.buildings.unet import create_unet
from models_code.buildings.cnn import deep_cnn_body, create_cnn
from training_models.buildings.config import TRAINING_CONFIG
from tensorflow.keras import layers, Model

class ModelFactory:
    @staticmethod
    def create_model(architecture="unet", mask_type="original"):
        input_shape = TRAINING_CONFIG["input_shape"]
        if architecture == "unet":
            return create_unet(input_shape=input_shape)
        elif architecture == "cnn":
            return create_cnn(input_shape, num_classes=1)
        else:
            raise ValueError(f"Nieznana architektura: {architecture}") 