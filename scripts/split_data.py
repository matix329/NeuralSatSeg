import os
import shutil
import random

class DataSplitter:
    def __init__(self, input_dir, output_dir, train_split=0.7, val_split=0.2, test_split=0.1, random_seed=4):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.splits = {
            "train": train_split,
            "val": val_split,
            "test": test_split
        }
        if random_seed is not None:
            random.seed(random_seed)

        # Sprawdzenie, czy ścieżki istnieją
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Input directory '{self.input_dir}' does not exist.")
        os.makedirs(self.output_dir, exist_ok=True)

    def split(self):
        categories = [folder for folder in os.listdir(self.input_dir)
                      if not folder.startswith(".") and os.path.isdir(os.path.join(self.input_dir, folder))]
        if not categories:
            raise ValueError(f"No valid categories found in input directory: {self.input_dir}")

        print(f"Found categories: {categories}")
        for split_name in self.splits:
            self.create_split_dirs(split_name, categories)

        for category in categories:
            print(f"Processing category: {category}")
            self.process_category(category)

    def create_split_dirs(self, split_name, categories):
        for category in categories:
            split_path = os.path.join(self.output_dir, split_name, category)
            os.makedirs(split_path, exist_ok=True)

    def process_category(self, category):
        category_path = os.path.join(self.input_dir, category)
        files = [file for file in os.listdir(category_path) if not file.startswith(".")]

        if not files:
            print(f"No files found in category: {category}")
            return

        random.shuffle(files)
        split_counts = self.get_split_counts(len(files))
        start = 0

        for split_name, count in split_counts.items():
            self.copy_files(
                files[start:start + count],
                os.path.join(self.input_dir, category),
                os.path.join(self.output_dir, split_name, category)
            )
            start += count

    def get_split_counts(self, total):
        counts = {split_name: int(total * split_ratio) for split_name, split_ratio in self.splits.items()}
        counts["train"] += total - sum(counts.values())
        return counts

    def copy_files(self, files, src_dir, dest_dir):
        for file in files:
            src_file = os.path.join(src_dir, file)
            dest_file = os.path.join(dest_dir, file)

            try:
                shutil.copy(src_file, dest_file)
                print(f"Copied {file} to {dest_dir}")
            except Exception as e:
                print(f"Error copying {file}: {e}")

if __name__ == "__main__":
    splitter = DataSplitter(
        input_dir="/Users/matix329/PycharmProjects/NeuralSatSeg/data/raw/2750",
        output_dir="/Users/matix329/PycharmProjects/NeuralSatSeg/data/processed",
        random_seed=42
    )
    splitter.split()