import os,sys
import argparse
from pathlib import Path

import pandas as pd
from airline.config import ConfigurationManager
from airline.entity import DataIngestionConfig,DataIngestionArtifact
from airline.exception import AirlineException
from airline.logger import logger
from sklearn.model_selection import StratifiedShuffleSplit
from ensure import ensure_annotations
from airline.components.stage_02_data_transformation import DataTransformation
from airline.components.model_trainer import ModelTrainer

class DataIngestion:

    def __init__(self,data_ingestion_config_info:DataIngestionConfig):
        try:
            logger.info(f"{'>>' * 10}Stage 01 data ingestion started  {'<<' * 10}")
            self.data_ingestion_config = data_ingestion_config_info

        except Exception as e:
            raise AirlineException(e,sys)
        

    @ensure_annotations
    def download_data(self,dataset_download_id: str,raw_data_file_path: Path)->bool:
        try:
            logger.info(f"Downloading dataset from github")
            raw_data_frame = pd.read_csv(dataset_download_id)
            raw_data_frame.to_csv(raw_data_file_path,index=False)
            logger.info("dataset downloaded successfully")

            return True
        

        except Exception as e:
            raise AirlineException(e,sys) from e
        
    @ensure_annotations
    def spilt_data_as_train_test(self,data_file_path:Path)->DataIngestionArtifact:
        try:
            logger.info(f"{'>>' * 20}Data splitting.{'<<'*20}")
            train_file_path = self.data_ingestion_config.ingested_train_file_path
            test_file_path = self.data_ingestion_config.ingested_test_file_path

            logger.info(f"Reading csv file:[{data_file_path}]")
            raw_data_frame = pd.read_csv(data_file_path)

            logger.info("splitting data into train and test")
            strat_train_set = None
            strat_test_set = None

            split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=1961)

            for train_index,test_index in split.split(raw_data_frame,raw_data_frame["satisfaction"]):
                strat_train_set = raw_data_frame.loc[train_index]
                strat_test_set = raw_data_frame.loc[test_index]

            if strat_train_set is not None:
                logger.info(f"Exporting training dataset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path, index=False)

            if strat_test_set is not None:
                logger.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path, index=False)
                data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                                test_file_path=test_file_path)
                logger.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
                return data_ingestion_artifact

        except Exception as e:
            raise AirlineException(e,sys) from e
        

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            data_ingestion_config_info = self.data_ingestion_config
            dataset_download_id = data_ingestion_config_info.dataset_download_id
            raw_data_file_path = data_ingestion_config_info.raw_data_file_path
            self.download_data(dataset_download_id,Path(raw_data_file_path))

            data_ingestion_response_info = self.spilt_data_as_train_test(data_file_path=Path(raw_data_file_path))
            logger.info(f"{'>>' * 20}Data Ingestion artifact.{'<<' * 20}")
            logger.info(f" Data Ingestion Artifact{data_ingestion_response_info.dict()}")
            logger.info(f"{'>>' * 20}Data Ingestion completed.{'<<' * 20}")
            return data_ingestion_response_info
        except Exception as e:
            raise AirlineException(e, sys) from e
        

    def __del__(self):
        logger.info(f"{'>>' * 20}Data Ingestion log completed.{'<<' * 20} \n\n")


#if __name__=="__main__":
    #config = ConfigurationManager(config_file_path='configs\config.yaml')
   #data_ingestion_config = config.get_data_ingestion_config()
    #data_ingestion = DataIngestion(data_ingestion_config)
    #train_file_path,test_file_path = data_ingestion.initiate_data_ingestion()
    #data_transformation_config = config.get_data_transformation_config(data_ingestion_config=data_ingestion_config)

    #data_transformation = DataTransformation(data_transformation_config_info=data_transformation_config)
    #train_arr,test_arr,_= data_transformation.initiate_data_transformation()

    #model_trainer_config = config.get_model_trainer_config(data_ingestion_config=data_ingestion_config, data_transformation_config_info=data_transformation_config)
    #model_trainer = ModelTrainer(model_trainer_config=model_trainer_config)

    #print(model_trainer.initate_model_training(train_arr,test_arr))