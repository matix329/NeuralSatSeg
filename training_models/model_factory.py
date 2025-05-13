from models_code.unet.unet import UNET
from models_code.cnn.cnn import CNN
from config import SUPPORTED_MODELS, TRAINING_CONFIG

class ModelFactory:
    @staticmethod
    def create_model(model_type):
        if model_type.lower() not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Available models: {', '.join(SUPPORTED_MODELS)}")

        if model_type.lower() == "unet":
            return UNET(
                input_shape=TRAINING_CONFIG["input_shape"],
                num_classes=TRAINING_CONFIG["num_classes"]
            ).build_model()
        elif model_type.lower() == "cnn":
            return CNN(
                input_shape=TRAINING_CONFIG["input_shape"],
                num_classes=TRAINING_CONFIG["num_classes"]
            ).build_model() 