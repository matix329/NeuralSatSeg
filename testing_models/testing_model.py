import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scripts.color_logger import ColorLogger
from models_code.metrics.dice_metrics import dice_coefficient, iou_score
from models_code.metrics.classification_metrics import precision, recall, f1_score

logger = ColorLogger("ModelTester").get_logger()

def load_model(model_path):
    try:
        custom_objects = {
            'dice_coefficient': dice_coefficient,
            'iou_score': iou_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def load_data(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [512, 512])
    image_np = image.numpy()
    
    category = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_name = image_name.replace('_img', '_mask')
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    if category == "roads":
        mask_path = os.path.join(base_dir, "data", "processed", "val", "roads", "masks_binary", f"{mask_name}.png")
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
        mask = tf.image.resize(mask, [512, 512])
        return image_np, None, mask.numpy(), image_name
    else:
        mask_path = os.path.join(base_dir, "data", "processed", "val", "buildings", "masks_original", f"{mask_name}.png")
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
        mask = tf.image.resize(mask, [512, 512])
        return image_np, mask.numpy(), None, image_name

def process_prediction(prediction):
    if isinstance(prediction, dict):
        processed = {}
        for key, value in prediction.items():
            value_np = value.numpy() if isinstance(value, tf.Tensor) else value
            value_2d = value_np[0, :, :, 0]
            processed[key] = {
                'raw': value_2d,
                'binary': (value_2d > 0.5).astype(np.uint8)
            }
        return processed
    return None

def calculate_metrics(true_mask, pred_mask):
    true_mask = true_mask.astype(np.float32)
    pred_mask = pred_mask.astype(np.float32)
    
    intersection = np.sum(true_mask * pred_mask)
    union = np.sum(true_mask) + np.sum(pred_mask) - intersection
    
    dice = 0.0
    if np.sum(true_mask) + np.sum(pred_mask) > 0:
        dice = (2. * intersection) / (np.sum(true_mask) + np.sum(pred_mask))
    
    iou = 0.0
    if union > 0:
        iou = intersection / union
    
    return {
        'dice': dice,
        'iou': iou,
        'true_positive': intersection,
        'true_pixels': np.sum(true_mask),
        'pred_pixels': np.sum(pred_mask)
    }

def visualize_predictions(image, true_mask, pred_dict, image_name, category, metrics):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Prediction Analysis - {image_name} ({category})', fontsize=16)
    
    image = image.astype(np.float32) / 255.0
    
    classes = ['buildings', 'roads']
    for idx, class_name in enumerate(classes):
        pred = pred_dict[f'head_{class_name}']
        
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title('Input Image')
        axes[idx, 0].axis('off')
        
        if true_mask is not None:
            axes[idx, 1].imshow(true_mask, cmap='gray')
            axes[idx, 1].set_title('True Mask')
            axes[idx, 1].axis('off')
        
        im = axes[idx, 2].imshow(pred['raw'], cmap='jet')
        axes[idx, 2].set_title('Prediction Heatmap')
        axes[idx, 2].axis('off')
        plt.colorbar(im, ax=axes[idx, 2])
        
        axes[idx, 3].imshow(pred['binary'], cmap='gray')
        metric_text = f'Binary Mask'
        if class_name in metrics:
            metric_text += f' (IoU: {metrics[class_name]["iou"]:.3f}, Dice: {metrics[class_name]["dice"]:.3f})'
        axes[idx, 3].set_title(metric_text)
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictions", category)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{image_name}_analysis.png"))
    plt.close()

def process_image(model, image_path):
    try:
        image, buildings_mask, roads_mask, image_name = load_data(image_path)
        category = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        
        image_norm = image / 255.0
        image_norm = np.expand_dims(image_norm, 0)
        binary_map = np.zeros_like(image_norm[..., :1])
        
        prediction = model.predict([image_norm, binary_map], verbose=0)
        processed_pred = process_prediction(prediction)
        
        metrics = {}
        if buildings_mask is not None:
            metrics['buildings'] = calculate_metrics(buildings_mask, processed_pred['head_buildings']['binary'])
        if roads_mask is not None:
            metrics['roads'] = calculate_metrics(roads_mask, processed_pred['head_roads']['binary'])
        
        visualize_predictions(image, buildings_mask if buildings_mask is not None else roads_mask, processed_pred, image_name, category, metrics)
        
        if buildings_mask is not None:
            logger.info(f"Processed {image_name} - Buildings IoU: {metrics['buildings']['iou']:.3f}, Dice: {metrics['buildings']['dice']:.3f}, True pixels: {metrics['buildings']['true_pixels']}, Pred pixels: {metrics['buildings']['pred_pixels']}")
        if roads_mask is not None:
            logger.info(f"Processed {image_name} - Roads IoU: {metrics['roads']['iou']:.3f}, Dice: {metrics['roads']['dice']:.3f}, True pixels: {metrics['roads']['true_pixels']}, Pred pixels: {metrics['roads']['pred_pixels']}")
        return prediction
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def main():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "mlflow_artifacts", "models")
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return

    available_models = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    if not available_models:
        logger.error("No models found in the directory.")
        return

    print("Available models:")
    for i, model_name in enumerate(available_models, 1):
        print(f"{i}. {model_name}")

    try:
        model_index = int(input("Select a model by number: ")) - 1
        if model_index < 0 or model_index >= len(available_models):
            logger.error("Invalid model selection.")
            return
    except ValueError:
        logger.error("Please enter a valid number.")
        return

    model_path = os.path.join(models_dir, available_models[model_index])
    model = load_model(model_path)
    if model is None:
        return

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "val")
    for category in ["roads", "buildings"]:
        images_dir = os.path.join(data_dir, category, "images")
        if not os.path.exists(images_dir):
            logger.error(f"Directory not found: {images_dir}")
            continue

        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        logger.info(f"Found {len(image_files)} images in {images_dir}")

        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            process_image(model, image_path)

if __name__ == "__main__":
    main()