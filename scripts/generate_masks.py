import os
import cv2

class MaskGenerator:
    def __init__(self, image_base_dir, mask_base_dir, threshold=127):
        self.image_base_dir = image_base_dir
        self.mask_base_dir = mask_base_dir
        self.threshold = threshold

    def generate(self):
        for split in ["train", "val", "test"]:
            split_image_dir = os.path.join(self.image_base_dir, split)
            split_mask_dir = os.path.join(self.mask_base_dir, split)

            if not os.path.exists(split_image_dir):
                print(f"[ERROR] Directory {split_image_dir} does not exist!")
                continue

            os.makedirs(split_mask_dir, exist_ok=True)

            for class_name in os.listdir(split_image_dir):
                class_image_dir = os.path.join(split_image_dir, class_name)
                class_mask_dir = os.path.join(split_mask_dir, class_name)
                os.makedirs(class_mask_dir, exist_ok=True)

                for file_name in os.listdir(class_image_dir):
                    img_path = os.path.join(class_image_dir, file_name)
                    mask_path = os.path.join(class_mask_dir, file_name.replace(".jpg", ".png"))

                    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if image is None:
                        print(f"[WARNING] Cannot read image {img_path}, skipping...")
                        continue

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)

                    cv2.imwrite(mask_path, mask)
                    print(f"[INFO] Mask generated for {file_name} in {split}/{class_name}")

if __name__ == "__main__":
    IMAGE_BASE_DIR = "/Users/matix329/PycharmProjects/NeuralSatSeg/data/processed"
    MASK_BASE_DIR = "/Users/matix329/PycharmProjects/NeuralSatSeg/data/processed/masks"

    generator = MaskGenerator(IMAGE_BASE_DIR, MASK_BASE_DIR, threshold=127)
    generator.generate()