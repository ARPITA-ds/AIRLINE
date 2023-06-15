
import os,sys
from pathlib import Path


from airline.logger import logger
from airline.exception import AirlineException
from airline.constants import (CURRENT_TIME_STAMP,CONFIG_FILE_PATH,ROOT_DIR)
from airline.entity import DataIngestionConfig,TrainingPipelineConfig
from airline.utils import read_yaml,create_directories

class ConfigurationManager:

    def __init__(self,config_file_path:Path = CONFIG_FILE_PATH) -> None:
        try:
            self.config_info = read_yaml(path_to_yaml=Path(config_file_path))
            self.time_stamp = CURRENT_TIME_STAMP
            self.pipeline_config = self.get_training_pipeline_config()

        except Exception as e:
            raise AirlineException(e,sys) from e
        

    def get_data_ingestion_config(self)->DataIngestionConfig:
        try:
            logger.info("Getting data ingestion configuration")
            data_ingestion_info = self.config_info.data_ingestion_config
            pipeline_config = self.pipeline_config
            artifact_dir = pipeline_config.artifact_dir
            dataset_download_id = data_ingestion_info.dataset_download_id
            data_ingestion_dir_name = data_ingestion_info.ingested_dir
            raw_data_dir = data_ingestion_info.raw_data_dir
            raw_file_name = data_ingestion_info.dataset_download_file_name

            data_ingestion_dir = os.path.join(artifact_dir,data_ingestion_dir_name)
            raw_data_file_path = os.path.join(data_ingestion_dir,raw_data_dir,raw_file_name)
            ingested_dir_name = data_ingestion_info.ingested_dir
            ingested_dir_path = os.path.join(data_ingestion_dir,ingested_dir_name)

            ingested_train_file_path = os.path.join(ingested_dir_path,data_ingestion_info.ingested_train_file)
            ingested_test_file_path = os.path.join(ingested_dir_path,data_ingestion_info.ingested_test_file)

            create_directories([os.path.dirname(raw_data_file_path),os.path.dirname(ingested_train_file_path)])

            data_ingestion_config = DataIngestionConfig(dataset_download_id=dataset_download_id,
                                                        raw_data_file_path=raw_data_file_path,
                                                        ingested_train_file_path=ingested_train_file_path,
                                                        ingested_test_file_path=ingested_test_file_path)
            logger.info(f"Data ingestion config:{data_ingestion_config.dict()}")
            logger.info("Data ingestion configuration completed")

            return data_ingestion_config
        except Exception as e:
            raise AirlineException(e,sys) from e
    
    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        try:
            training_config = self.config_info.training_pipeline_config
            training_pipeline_name = training_config.pipeline_name
            training_artifacts = os.path.join(ROOT_DIR,training_config.artifact_dir)
            create_directories(path_to_directories=[training_artifacts])
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=training_artifacts,
                                                              pipeline_name=training_pipeline_name)
            logger.info(f"Training pipeline config:{training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise AirlineException(e,sys) from e
