from models_code.roads.unet import create_unet
from models_code.roads.cnn import create_cnn

class ModelFactory:
    @staticmethod
    def create_model(architecture="unet", mask_type="binary"):
        if architecture == "unet":
            return create_unet()
        elif architecture == "cnn":
            return create_cnn()
        else:
            raise ValueError(f"Nieznana architektura: {architecture}") 