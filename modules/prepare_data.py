import os
import logging

from data_preparation.roads_data_preparation import RoadsDataPreparator
from data_preparation.buildings_data_preparation import BuildingDataPreparator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cities = ["Vegas", "Paris", "Shanghai"]
    data_types = ["roads", "buildings"]
    
    for city in cities:
        for data_type in data_types:
            try:
                if data_type == "roads":
                    preparator = RoadsDataPreparator(base_dir, city)
                else:
                    preparator = BuildingDataPreparator(base_dir, city)
                preparator.run()
                logger.info(f"Successfully processed {data_type} for {city}")
            except Exception as e:
                logger.error(f"Error processing {data_type} for {city}: {str(e)}")

if __name__ == "__main__":
    main()