from models_code.roads.unet import create_unet, create_unet_graph
from models_code.roads.cnn import create_cnn
from training_models.roads.config import TRAINING_CONFIG

class ModelFactory:
    @staticmethod
    def create_model(architecture="unet", mask_type="binary", callbacks=False):
        input_shape = TRAINING_CONFIG["input_shape"]
        if architecture == "unet":
            if mask_type == "graph":
                return create_unet_graph(input_shape=input_shape)
            return create_unet(input_shape=input_shape)
        elif architecture == "cnn":
            return create_cnn(input_shape=input_shape, callbacks=callbacks)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")