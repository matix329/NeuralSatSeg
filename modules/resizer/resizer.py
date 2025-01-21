import os
from PIL import Image


class ImageMaskResizer:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def resize(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for file_name in os.listdir(input_dir):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif')):
                continue
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            if os.path.isfile(input_path):
                try:
                    image = Image.open(input_path)
                    resized_image = image.resize(self.target_size)
                    resized_image.save(output_path)
                except Exception as e:
                    print(f"[ERROR] Error processing file {input_path}: {e}")
            else:
                print(f"[ERROR] Skipping, not a file: {input_path}")


def resize_images_and_masks(image_input_dir, image_output_dir, mask_input_dir, mask_output_dir, target_size=(512, 512)):
    resizer = ImageMaskResizer(target_size=target_size)
    resizer.resize(image_input_dir, image_output_dir)
    resizer.resize(mask_input_dir, mask_output_dir)