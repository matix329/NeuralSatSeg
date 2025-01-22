import os
import glob
from scripts.color_logger import ColorLogger

class ImageLoader:
    def __init__(self, source_folder, destination_folder, logger_name, supported_folders=None):
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.logger = ColorLogger(logger_name).get_logger()
        self.supported_folders = supported_folders if supported_folders else ["MS", "PAN", "PS-MS", "PS-RGB"]

        if not os.path.exists(self.destination_folder):
            os.makedirs(self.destination_folder)
            self.logger.info(f"Created directory: {self.destination_folder}")

    def load_images(self):
        images_by_index = {}
        for folder in self.supported_folders:
            folder_path = os.path.join(self.source_folder, folder)

            if not os.path.exists(folder_path):
                self.logger.warning(f"Folder {folder_path} does not exist. Skipping...")
                continue

            for filepath in glob.glob(os.path.join(folder_path, "*.tif")):
                filename = os.path.basename(filepath)
                file_index = self.extract_index(filename)
                if file_index not in images_by_index:
                    images_by_index[file_index] = {}
                images_by_index[file_index][folder] = filepath

            self.logger.info(f"Processed folder: {folder_path}")

        if not images_by_index:
            self.logger.error("No images were loaded. Check your source folder.")
        else:
            self.logger.info(f"Loaded images for {len(images_by_index)} map fragments.")

        return images_by_index

    @staticmethod
    def extract_index(filename):
        return filename.split('_img')[1].split('.')[0]