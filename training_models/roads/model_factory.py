from models_code.roads.unet import create_unet, create_unet_graph
from models_code.roads.cnn import create_cnn

class ModelFactory:
    @staticmethod
    def create_model(architecture="unet", mask_type="binary"):
        if architecture == "unet":
            if mask_type == "graph":
                return create_unet_graph()
            return create_unet()
        elif architecture == "cnn":
            return create_cnn(callbacks=False)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")