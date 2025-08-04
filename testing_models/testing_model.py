import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import rasterio
import cv2
from scripts.color_logger import ColorLogger

logger = ColorLogger("ModelTester").get_logger()
TOP_PERCENT = 0.95

def resize_lambda(tensors):
    return tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3])

def load_model(model_path):
    from models_code.buildings.metrics import dice_coefficient, iou_score, precision, recall, f1_score
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'iou_score': iou_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'resize_lambda': resize_lambda
    }
    tf.keras.config.enable_unsafe_deserialization()
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        try:
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            logger.info("Model loaded without compilation")
            return model
        except Exception as e2:
            logger.error(f"Failed to load model with compile=False: {e2}")
            return None

def get_model_input_size(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if input_shape and len(input_shape) >= 3:
        return input_shape[1], input_shape[2]
    return 650, 650

def load_data(image_path, model):
    input_size = get_model_input_size(model)
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.tif', '.tiff']:
        with rasterio.open(image_path) as src:
            image = src.read()
            if image.shape[0] == 3:
                image = np.moveaxis(image, 0, -1)
            image = image.astype(np.float32) / 2047.0
            image = cv2.resize(image, input_size[::-1])
    elif ext in ['.png', '.jpg', '.jpeg']:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, input_size).numpy()
        image = image / 255.0
    else:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, input_size).numpy()
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
    category = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_name = image_name.replace('_img', '_mask')
    base_dir = os.path.dirname(os.path.dirname(__file__))
    if category == "roads":
        mask_path = os.path.join(base_dir, "data", "processed", "val", "roads", "masks_binary", f"{mask_name}.tif")
    else:
        mask_path = os.path.join(base_dir, "data", "processed", "val", "buildings", "masks_original", f"{mask_name}.png")
    mask = None
    try:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            mask = cv2.resize(mask, input_size[::-1], interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        logger.error(f"Failed to load mask: {mask_path} ({e})")
        mask = np.zeros(input_size, dtype=np.uint8)
    return image, mask, image_name

def process_prediction(prediction):
    prediction = prediction[0, :, :, 0]
    threshold = np.quantile(prediction, TOP_PERCENT)
    return {
        'raw': prediction,
        'binary': (prediction > threshold).astype(np.uint8)
    }

def calculate_metrics(true_mask, pred_mask):
    true_mask = true_mask.astype(np.float32)
    pred_mask = pred_mask.astype(np.float32)
    intersection = np.sum(true_mask * pred_mask)
    union = np.sum(true_mask) + np.sum(pred_mask) - intersection
    dice = (2. * intersection) / (np.sum(true_mask) + np.sum(pred_mask)) if (np.sum(true_mask) + np.sum(pred_mask)) > 0 else 0.0
    iou = intersection / union if union > 0 else 0.0
    return {
        'dice': dice,
        'iou': iou,
        'true_positive': intersection,
        'true_pixels': np.sum(true_mask),
        'pred_pixels': np.sum(pred_mask)
    }

def denormalize_img(image):
    if image.ndim == 3 and image.shape[2] == 3:
        if image.min() < -1.0 or image.max() > 2.0:
            return image
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        denormalized = (image * std) + mean
        
        if denormalized.min() < -0.5 or denormalized.max() > 1.5:
            return image
        
        return denormalized
    return image

def visualize_predictions(image, true_mask, pred_dict, image_name, category, metrics):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Prediction Analysis - {image_name} ({category})', fontsize=16)
    
    if category == "roads":
        img_to_show = denormalize_img(image)
    else:
        img_to_show = image.copy()
    
    if img_to_show.min() < 0 or img_to_show.max() > 1:
        if img_to_show.min() < 0:
            img_to_show = img_to_show - img_to_show.min()
        if img_to_show.max() > 0:
            img_to_show = img_to_show / img_to_show.max()
    
    img_to_show = np.clip(img_to_show, 0, 1)
    
    img_stats = f"min={img_to_show.min():.3f}, max={img_to_show.max():.3f}, mean={img_to_show.mean():.3f}"
    
    axes[0].imshow(img_to_show)
    axes[0].set_title(f'Input Image\n{img_stats}')
    axes[0].axis('off')
    
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('True Mask')
    axes[1].axis('off')
    
    im = axes[2].imshow(pred_dict['raw'], cmap='jet')
    axes[2].set_title('Prediction Heatmap')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    axes[3].imshow(pred_dict['binary'], cmap='gray')
    axes[3].set_title(f'Binary (IoU: {metrics["iou"]:.3f}, Dice: {metrics["dice"]:.3f})')
    axes[3].axis('off')
    
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictions", category)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{image_name}_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()

def process_image(model, image_path):
    image, mask, image_name = load_data(image_path, model)
    category = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    
    logger.info(f"Original image stats: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}, std={image.std():.4f}")
    
    if category == "roads":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        logger.info(f"After ImageNet normalization: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}, std={image.std():.4f}")
    
    image_norm = np.expand_dims(image, 0)
    prediction = model.predict(image_norm, verbose=0)
    processed_pred = process_prediction(prediction)
    metrics = calculate_metrics(mask, processed_pred['binary'])
    visualize_predictions(image, mask, processed_pred, image_name, category, metrics)
    logger.info(f"{image_name} - IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}, TP: {metrics['true_positive']}, GT: {metrics['true_pixels']}, Pred: {metrics['pred_pixels']}")

def main():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "mlflow_artifacts", "models")
    print("\nSelect category:\n1. Buildings\n2. Roads")
    choice = input("\nEnter category number: ")
    category = "buildings" if choice == '1' else "roads"
    category_dir = os.path.join(models_dir, category)
    models = [f for f in os.listdir(category_dir) if f.endswith('.keras')]
    print("\nAvailable models:")
    models = sorted(models)
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    idx = int(input("\nSelect model number: ")) - 1
    model_path = os.path.join(category_dir, models[idx])
    model = load_model(model_path)
    if model is None:
        return
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "val", category, "images")
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.tif', '.tiff'))]
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        process_image(model, image_path)

if __name__ == "__main__":
    main()