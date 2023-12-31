from pathlib import Path

from pydantic import BaseModel,DirectoryPath,FilePath

class DataIngestionConfig(BaseModel):
    raw_data_file_path: Path
    ingested_train_file_path: Path
    ingested_test_file_path: Path
    dataset_download_id: str

class TrainingPipelineConfig(BaseModel):
    artifact_dir: DirectoryPath
    pipeline_name: str


class DataTransformationConfig(BaseModel):
    train_data_file: FilePath
    test_data_file: FilePath
    preprocessed_object_file_path: Path
    data_transformed_train_file_path: Path
    data_transformed_test_file_path: Path
    feature_eng_obj_file_path: Path

class ModelTrainerConfig(BaseModel):
    #model_report_dir: DirectoryPath
    trained_model_file_path: Path
    