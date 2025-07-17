import os
import json
from models_code.roads.models.unet import create_unet
from models_code.roads.models.cnn import create_cnn
from models_code.roads.models.gnn import create_gnn
from models_code.roads.metrics.metrics_binary import combined_loss, focal_loss

class ModelFactory:
    @staticmethod
    def create_model(architecture="unet", mask_type="binary", callbacks=False):
        if mask_type == "graph":
            config_path = os.path.join(os.path.dirname(__file__), "config_graph.json")
        else:
            config_path = os.path.join(os.path.dirname(__file__), "config_binary.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if architecture == "unet":
            input_shape = tuple(config["input_shape"])
            loss_type = config["loss"]
            use_skip_connections = config["use_skip_connections"]
            
            if loss_type == "focal":
                loss_function = focal_loss
            elif loss_type == "bce_dice":
                loss_function = combined_loss
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            if mask_type == "graph":
                raise ValueError("create_unet_graph has been removed - use GNN instead!")
            return create_unet(input_shape=input_shape)
        elif architecture == "cnn":
            input_shape = tuple(config["input_shape"])
            loss_type = config["loss"]
            use_skip_connections = config["use_skip_connections"]
            
            if loss_type == "focal":
                loss_function = focal_loss
            elif loss_type == "bce_dice":
                loss_function = combined_loss
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            return create_cnn(
                input_shape=input_shape, 
                callbacks=callbacks,
                loss_function=loss_function,
                use_skip_connections=use_skip_connections
            )
        elif architecture == "gnn":
            return create_gnn(
                hidden_channels=config["hidden_channels"],
                heads=config["heads"],
                dropout=config["dropout"]
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")