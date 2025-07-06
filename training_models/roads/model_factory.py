import os
import json
from models_code.roads.unet import create_unet, create_unet_graph
from models_code.roads.cnn import create_cnn
from models_code.roads.metrics import combined_loss, focal_loss

class ModelFactory:
    @staticmethod
    def create_model(architecture="unet", mask_type="binary", callbacks=False):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        input_shape = tuple(config["input_shape"])
        loss_type = config["loss"]
        use_skip_connections = config["use_skip_connections"]
        
        if loss_type == "focal":
            loss_function = focal_loss
        elif loss_type == "bce_dice":
            loss_function = combined_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        if architecture == "unet":
            if mask_type == "graph":
                return create_unet_graph(input_shape=input_shape)
            return create_unet(input_shape=input_shape)
        elif architecture == "cnn":
            return create_cnn(
                input_shape=input_shape, 
                callbacks=callbacks,
                loss_function=loss_function,
                use_skip_connections=use_skip_connections
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")