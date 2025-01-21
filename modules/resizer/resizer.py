import os
from PIL import Image


class ImageMaskResizer:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def resize(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for file_name in os.listdir(input_dir):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif')):
                print(f"[DEBUG] Skipping non-image file: {file_name}")
                continue
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            if os.path.isfile(input_path):
                try:
                    print(f"[DEBUG] Processing file: {input_path}")
                    image = Image.open(input_path)
                    print(f"[DEBUG] Original size of {file_name}: {image.size}")

                    resized_image = image.resize(self.target_size)
                    print(f"[DEBUG] Resized size of {file_name}: {resized_image.size}")

                    resized_image.save(output_path)
                    print(f"[DEBUG] Resized and saved: {output_path}")
                except Exception as e:
                    print(f"[ERROR] Error processing file {input_path}: {e}")
            else:
                print(f"[DEBUG] Skipping, not a file: {input_path}")


def resize_images_and_masks(image_input_dir, image_output_dir, mask_input_dir, mask_output_dir, target_size=(512, 512)):
    print("[DEBUG] Starting resizing process...")
    resizer = ImageMaskResizer(target_size=target_size)

    print(f"[DEBUG] Resizing images from {image_input_dir} to {image_output_dir}")
    resizer.resize(image_input_dir, image_output_dir)

    print(f"[DEBUG] Resizing masks from {mask_input_dir} to {mask_output_dir}")
    resizer.resize(mask_input_dir, mask_output_dir)
    print("[DEBUG] Resizing process completed.")