import os
import numpy as np
import rasterio
import re
import logging
from rasterio.enums import Resampling
from typing import Dict, Generator
import gc

logger = logging.getLogger(__name__)

class ImageLoader:
    def __init__(self, image_dir: str, target_size=(1300, 1300), batch_size=10):
        self.image_dir = image_dir
        self.target_size = target_size
        self.batch_size = batch_size
        logger.info(f"Initializing ImageLoader with directory: {image_dir}")
        self.image_dict = self.group_images_by_id()

    def find_image_files(self) -> Dict[str, str]:
        """Znajduje wszystkie pliki obrazów w katalogu i podkatalogach"""
        image_files = {}
        for root, _, files in os.walk(self.image_dir):
            for fname in files:
                if fname.endswith(('.tif', '.tiff')):
                    full_path = os.path.join(root, fname)
                    image_files[fname] = full_path
        return image_files

    def group_images_by_id(self) -> Dict[str, Dict[str, str]]:
        image_dict = {}
        image_files = self.find_image_files()
        logger.info(f"Found {len(image_files)} image files")

        for fname, full_path in image_files.items():
            img_id = self.extract_img_id(fname)
            if img_id is None:
                logger.warning(f"Could not extract image ID from filename: {fname}")
                continue

            if 'PS-RGB' in fname:
                img_type = 'PS-RGB'
            elif 'PS-MS' in fname:
                img_type = 'PS-MS'
            elif '_MS_' in fname:
                img_type = 'MS'
            elif '_PAN_' in fname:
                img_type = 'PAN'
            else:
                logger.warning(f"Unknown image type in filename: {fname}")
                continue

            if img_id not in image_dict:
                image_dict[img_id] = {}
            image_dict[img_id][img_type] = full_path
            logger.debug(f"Added {img_type} image for {img_id}: {fname}")

        logger.info(f"Grouped images into {len(image_dict)} sets")
        return image_dict

    def extract_img_id(self, fname: str) -> str:
        match = re.search(r'img\d+', fname)
        return match.group(0) if match else None

    def resize_image(self, image: np.ndarray, original_transform) -> np.ndarray:
        """Przeskalowuje obraz do docelowego rozmiaru"""
        from rasterio.warp import reproject
        target_shape = (image.shape[0], self.target_size[0], self.target_size[1])
        destination = np.zeros(target_shape, dtype=image.dtype)
        
        for band_idx in range(image.shape[0]):
            reproject(
                source=image[band_idx],
                destination=destination[band_idx],
                src_transform=original_transform,
                src_crs="EPSG:4326",
                dst_transform=rasterio.transform.from_origin(0, 0, 1, 1),
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
        
        return destination

    def load_image(self, path: str) -> np.ndarray:
        try:
            with rasterio.open(path) as src:
                image = src.read()
                if image.shape[1:] != self.target_size:
                    logger.debug(f"Resizing image from {image.shape[1:]} to {self.target_size}")
                    image = self.resize_image(image, src.transform)
                return image
        except Exception as e:
            logger.error(f"Error loading image {path}: {str(e)}")
            raise

    def process_batch(self, batch_items: list) -> Dict[str, np.ndarray]:
        """Przetwarza partię obrazów"""
        batch_result = {}
        for img_id, modalities in batch_items:
            try:
                if all(key in modalities for key in ['MS', 'PAN', 'PS-MS', 'PS-RGB']):
                    logger.debug(f"Loading all modalities for {img_id}")
                    loaded_channels = []
                    
                    for modality in ['MS', 'PAN', 'PS-MS', 'PS-RGB']:
                        path = modalities[modality]
                        image_data = self.load_image(path)
                        loaded_channels.append(image_data)
                    
                    merged = np.concatenate(loaded_channels, axis=0)
                    batch_result[img_id] = merged
                    logger.debug(f"Successfully loaded and merged all modalities for {img_id}")
            except Exception as e:
                logger.error(f"Error processing {img_id}: {str(e)}")
                continue
        return batch_result

    def load_all(self) -> Generator[Dict[str, np.ndarray], None, None]:
        """Wczytuje obrazy w partiach i zwraca generator"""
        total_sets = len(self.image_dict)
        processed_sets = 0
        missing_modalities = 0

        # Konwertuj słownik na listę par (klucz, wartość)
        items = list(self.image_dict.items())
        
        # Przetwarzaj w partiach
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_result = self.process_batch(batch)
            
            # Zliczaj brakujące modalności
            for img_id, _ in batch:
                if img_id not in batch_result:
                    missing_modalities += 1
            
            processed_sets += len(batch)
            logger.info(f"Processed {processed_sets}/{total_sets} image sets")
            
            yield batch_result
            
            # Zwolnij pamięć
            del batch_result
            gc.collect()

        logger.info(f"Completed processing. {missing_modalities} sets skipped due to missing modalities")

    def merge_modalities(self, paths: Dict[str, str]) -> np.ndarray:
        arrays = []
        for key in ['MS', 'PAN', 'PS-MS', 'PS-RGB']:
            path = paths[key]
            try:
                with rasterio.open(path) as src:
                    arr = src.read()
                arrays.append(arr)
                logger.debug(f"Successfully loaded {key} image from {path}")
            except Exception as e:
                logger.error(f"Error loading {key} image from {path}: {str(e)}")
                raise
        return np.concatenate(arrays, axis=0)