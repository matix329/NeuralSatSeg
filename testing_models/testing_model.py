import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scripts.color_logger import ColorLogger
from models_code.buildings.metrics import dice_coefficient, iou_score, precision, recall, f1_score

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

def get_model_input_size(model):
    try:
        input_shape = model.input_shape
        logger.info(f"Model input shape: {input_shape}")
        
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        if input_shape is None:
            logger.warning("Model input shape is None, using default 650x650")
            return 650, 650
            
        if len(input_shape) >= 3:
            height, width = input_shape[1], input_shape[2]
            logger.info(f"Detected model input size: {width}x{height}")
            return height, width
        else:
            logger.warning(f"Unexpected input shape format: {input_shape}, using default 650x650")
            return 650, 650
    except Exception as e:
        logger.error(f"Error getting model input size: {e}")
        return 650, 650

def load_data(image_path, model):
    input_size = get_model_input_size(model)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, input_size)
    image_np = image.numpy()
    
    category = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_name = image_name.replace('_img', '_mask')
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    if category == "roads":
        mask_path = os.path.join(base_dir, "data", "processed", "val", "roads", "masks_binary", f"{mask_name}.png")
    else:
        mask_path = os.path.join(base_dir, "data", "processed", "val", "buildings", "masks_original", f"{mask_name}.png")
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, input_size)
    return image_np, mask.numpy(), image_name

def process_prediction(prediction):
    if prediction is None:
        logger.error("Prediction is None")
        return None
        
    logger.info(f"Prediction type: {type(prediction)}")
    if isinstance(prediction, dict):
        logger.info(f"Prediction keys: {prediction.keys()}")
        processed = {}
        for key, value in prediction.items():
            value_np = value.numpy() if isinstance(value, tf.Tensor) else value
            value_2d = value_np[0, :, :, 0]
            processed[key] = {
                'raw': value_2d,
                'binary': (value_2d > 0.5).astype(np.uint8)
            }
        return processed
    else:
        logger.info(f"Prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'no shape'}")
        prediction_np = prediction.numpy() if isinstance(prediction, tf.Tensor) else prediction
        prediction_2d = prediction_np[0, :, :, 0]
        return {
            'raw': prediction_2d,
            'binary': (prediction_2d > 0.5).astype(np.uint8)
        }

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
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Prediction Analysis - {image_name} ({category})', fontsize=16)
    
    image = image.astype(np.float32) / 255.0
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('True Mask')
    axes[1].axis('off')
    
    im = axes[2].imshow(pred_dict['raw'], cmap='jet')
    axes[2].set_title('Prediction Heatmap')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    axes[3].imshow(pred_dict['binary'], cmap='gray')
    metric_text = f'Binary Mask (IoU: {metrics["iou"]:.3f}, Dice: {metrics["dice"]:.3f})'
    axes[3].set_title(metric_text)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictions", category)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{image_name}_analysis.png"))
    plt.close()

def process_image(model, image_path):
    try:
        image, mask, image_name = load_data(image_path, model)
        category = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        
        image_norm = image / 255.0
        image_norm = np.expand_dims(image_norm, 0)
        logger.info(f"Input image shape: {image_norm.shape}")
        
        prediction = model.predict(image_norm, verbose=0)
        logger.info(f"Raw prediction type: {type(prediction)}")
        processed_pred = process_prediction(prediction)
        
        if processed_pred is None:
            logger.error(f"Failed to process prediction for {image_name}")
            return None
            
        metrics = calculate_metrics(mask, processed_pred['binary'])
        
        visualize_predictions(image, mask, processed_pred, image_name, category, metrics)
        
        logger.info(f"Processed {image_name} - IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}, True pixels: {metrics['true_pixels']}, Pred pixels: {metrics['pred_pixels']}")
        return prediction
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def main():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "mlflow_artifacts", "models")
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return

    print("\nSelect category:")
    print("1. Buildings")
    print("2. Roads")
    
    try:
        category_choice = int(input("\nEnter category number (1 or 2): "))
        if category_choice not in [1, 2]:
            logger.error("Invalid category selection.")
            return
    except ValueError:
        logger.error("Please enter a valid number.")
        return

    category = "buildings" if category_choice == 1 else "roads"
    category_dir = os.path.join(models_dir, category)
    
    if not os.path.exists(category_dir):
        logger.error(f"Category directory not found: {category_dir}")
        return

    available_models = [f for f in os.listdir(category_dir) if f.endswith('.keras')]
    if not available_models:
        logger.error(f"No models found in {category_dir}")
        return

    print(f"\nAvailable {category} models:")
    for i, model_name in enumerate(available_models, 1):
        print(f"{i}. {model_name}")

    try:
        model_index = int(input(f"\nSelect a {category} model by number: ")) - 1
        if model_index < 0 or model_index >= len(available_models):
            logger.error("Invalid model selection.")
            return
    except ValueError:
        logger.error("Please enter a valid number.")
        return

    model_path = os.path.join(category_dir, available_models[model_index])
    model = load_model(model_path)
    if model is None:
        return

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "val")
    images_dir = os.path.join(data_dir, category, "images")
    if not os.path.exists(images_dir):
        logger.error(f"Directory not found: {images_dir}")
        return

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    logger.info(f"Found {len(image_files)} images in {images_dir}")

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        process_image(model, image_path)

if __name__ == "__main__":
    main()