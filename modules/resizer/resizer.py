import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

class ImageMaskResizer:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def resize_image(self, input_path, output_path, is_mask=False):
        try:
            with Image.open(input_path) as image:
                interpolation = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
                resized_image = image.resize(self.target_size, interpolation)
                resized_image.save(output_path)
        except Exception as e:
            print(f"[ERROR] Error processing file {input_path}: {e}")

    def resize_directory(self, input_dir, output_dir, is_mask=False):
        os.makedirs(output_dir, exist_ok=True)
        files = [
            (os.path.join(input_dir, f), os.path.join(output_dir, f), is_mask)
            for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif'))
        ]
        with ThreadPoolExecutor() as executor:
            executor.map(lambda args: self.resize_image(*args), files)

def resize_images_and_masks(image_input_dir, image_output_dir, mask_input_dir, mask_output_dir, target_size=(512, 512)):
    resizer = ImageMaskResizer(target_size=target_size)
    resizer.resize_directory(image_input_dir, image_output_dir, is_mask=False)
    resizer.resize_directory(mask_input_dir, mask_output_dir, is_mask=True)