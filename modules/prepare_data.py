import os
import logging

from data_preparation.roads_data_preparation import RoadsDataPreparator
from data_preparation.buildings_data_preparation import BuildingDataPreparator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    cities = ["Vegas", "Paris", "Shanghai"]
    data_types = ["roads", "buildings"]
    
    logger.info(f"Starting data preparation from base directory: {base_dir}")
    
    for city in cities:
        logger.info(f"\nProcessing city: {city}")
        for data_type in data_types:
            try:
                logger.info(f"Starting {data_type} processing for {city}")
                if data_type == "roads":
                    preparator = RoadsDataPreparator(base_dir, city, processed=processed_dir)
                else:
                    preparator = BuildingDataPreparator(base_dir, city, processed=processed_dir)
                logger.info(f"Initialized {data_type} preparator for {city}")
                preparator.run()
                logger.info(f"Successfully processed {data_type} for {city}")
            except Exception as e:
                logger.error(f"Error processing {data_type} for {city}: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()