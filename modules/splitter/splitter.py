import random
from typing import List, Tuple

class Splitter:
    def __init__(self, data: List[Tuple[str, str]], test_size: float = 0.2, shuffle: bool = True, seed: int = None):
        self.data = data
        self.test_size = test_size
        self.shuffle = shuffle
        self.seed = seed

    def split(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        if self.seed is not None:
            random.seed(self.seed)

        data = list(self.data)
        if self.shuffle:
            random.shuffle(data)

        split_idx = int(len(data) * (1 - self.test_size))
        return data[:split_idx], data[split_idx:]

    @staticmethod
    def save_split(data: List[Tuple[str, str]], image_file: str, mask_file: str):
        """
        Zapisuje listę ścieżek do dwóch plików tekstowych: jeden dla obrazów, drugi dla masek.
        """
        with open(image_file, 'w') as img_f, open(mask_file, 'w') as msk_f:
            for img_path, mask_path in data:
                img_f.write(f"{img_path}\n")
                msk_f.write(f"{mask_path}\n")