from models_code.unet.unet import UNET
from models_code.cnn.cnn import CNN
from config import SUPPORTED_MODELS, TRAINING_CONFIG, HEAD_NAMES

class ModelFactory:
    @staticmethod
    def create_model(model_type):
        if model_type.lower() not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Available models: {', '.join(SUPPORTED_MODELS)}")

        if model_type.lower() == "unet":
            return UNET(
                input_shape=TRAINING_CONFIG["input_shape"],
                num_classes=1,
                multi_head=True,
                head_names=HEAD_NAMES,
                use_binary_embedding=True
            ).build_model()
        elif model_type.lower() == "cnn":
            return CNN(
                input_shape=TRAINING_CONFIG["input_shape"],
                num_classes=1,
                multi_head=True,
                head_names=HEAD_NAMES,
                use_binary_embedding=True
            ).build_model() 