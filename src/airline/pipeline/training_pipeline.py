import os,sys
from airline.config.configuration import ConfigurationManager
from airline.components.stage_01_data_ingestion import DataIngestion
from airline.components.stage_02_data_transformation import DataTransformation
from airline.components.model_trainer import ModelTrainer
from dataclasses import dataclass

class Train:
    def __init__(self):
        self.c = 0
        print(f"______{self.c}______")

#if __name__=="__main__":
    config = ConfigurationManager(config_file_path='configs\config.yaml')
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    train_file_path,test_file_path = data_ingestion.initiate_data_ingestion()
    data_transformation_config = config.get_data_transformation_config(data_ingestion_config=data_ingestion_config)

    data_transformation = DataTransformation(data_transformation_config_info=data_transformation_config)
    train_arr,test_arr,_= data_transformation.initiate_data_transformation()

    model_trainer_config = config.get_model_trainer_config(data_ingestion_config=data_ingestion_config, data_transformation_config_info=data_transformation_config)
    model_trainer = ModelTrainer(model_trainer_config=model_trainer_config)

    print(model_trainer.initate_model_training(train_arr,test_arr))