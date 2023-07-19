import os,sys
import pandas as pd 
import numpy as np
from airline.logger import logger
from airline.exception import AirlineException
import pickle
from sklearn.pipeline import Pipeline
import json
from airline.constants import *
import urllib
from airline.utils.common import load_model


PREDICTION_FOLDER='batch_Prediction'
PREDICTION_CSV='prediction_csv'
PREDICTION_FILE='prediction.csv'

FEATURE_ENG_FOLDER='feature_eng'

ROOT_DIR=os.getcwd()
FEATURE_ENG=os.path.join(ROOT_DIR,PREDICTION_FOLDER,FEATURE_ENG_FOLDER)
BATCH_PREDICTION=os.path.join(ROOT_DIR,PREDICTION_FOLDER,PREDICTION_CSV)
BATCH_COLLECTION_PATH ='batch_prediction'


class batch_prediction:
    def __init__(self,input_file_path, 
                 model_file_path, 
                 transformer_file_path, 
                 feature_engineering_file_path) -> None:
        
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path

    def start_batch_prediction(self):
        try:
            logger.info("loading saved pipeline")
            # Load the feature engineering pipeline
            with open(self.feature_engineering_file_path,'rb')as f:
            #with open(r"C:\project\AIRLINE\artifact\stage02_data_transformation\preprocessing\feature_eng.pkl", 'rb') as f:
                feature_pipeline =pickle.load(f)
            
            logger.info(f"Feature eng Object acessed :{self.feature_engineering_file_path}")

            # Load the data transformation pipeline
            with open(self.transformer_file_path, 'rb') as f:
                preprocessor = pickle.load(f)

            logger.info(f"preprocessor  Object acessed:{self.transformer_file_path}")

            # Load the model separately
            model =load_model(file_path=self.model_file_path)

            logger.info(f"Model File Path: {self.model_file_path}")

            # Create the feature engineering pipeline
            feature_engineering_pipeline = Pipeline([
                ('feature_engineering', feature_pipeline)
            ])
            logger.info(f"Feature eng Object acessed :{self.feature_engineering_file_path}")

             # Load the data transformation pipeline
            with open(self.transformer_file_path, 'rb') as f:
                preprocessor = pickle.load(f)

            logger.info(f"Preprocessor  Object acessed :{self.transformer_file_path}")
            # Load the model separately
            model =load_model(file_path=self.model_file_path)

            logger.info(f"Model File Path: {self.model_file_path}")

        
            # Create the feature engineering pipeline
            feature_engineering_pipeline = Pipeline([
                ('feature_engineering', feature_pipeline)
            ])

            # Read the input file
            df = pd.read_csv(self.input_file_path)

            df.to_csv("df_if_stisfied_input_Data.csv")

            # Applying feature engineering
            df=feature_engineering_pipeline.transform(df)
            df.to_csv("df_feature_enginnering.csv")

            #save the feature engineering data as csv file
            FEATURE_ENG_PATH = FEATURE_ENG  # Specify the desired path for saving the CSV file
            os.makedirs(FEATURE_ENG_PATH, exist_ok=True)
            file_path = os.path.join(FEATURE_ENG_PATH, 'batch_fea_eng.csv')
            df.to_csv(file_path, index=False)
            
            logger.info("Feature-engineered batch data saved as CSV.")

            # Dropping target column
            
            df=df.drop('satisfaction', axis=1)
            
            df.to_csv('dropped_satisfaction.csv')

            logger.info(f"Columns before transformation: {', '.join(f'{col}: {df[col].dtype}' for col in df.columns)}")
            # Transform the feature-engineered data using the preprocessor
            transformed_data = preprocessor.transform(df)
            logger.info(f"Transformed Data Shape: {transformed_data.shape}")
            
            logger.info(f"Loaded numpy from batch prediciton :{transformed_data}")
            file_path = os.path.join(FEATURE_ENG_PATH, 'preprocessor.csv')

            logger.info(f"Model Data Type : {type(model)}")
            
            predictions = model.predict(transformed_data)
            logger.info(f"Predictions done :{predictions}")
            
            

            # Create a DataFrame from the predictions array
            df_predictions = pd.DataFrame(predictions, columns=['prediction'])

             # Define the mapping dictionary          
            
           
            def map_prediction(value):
                if value == 0:
                    return 'neutral or dissatisfied'
                elif value == 1:
                    return 'satisfied'
                else:
                    return 'unknown'
            # Apply mapping function to the Prediction column
            df_predictions['prediction'] = df_predictions['prediction'].map(map_prediction)
            
            # Save the predictions to a CSV file
            BATCH_PREDICTION_PATH = BATCH_PREDICTION  # Specify the desired path for saving the CSV file
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH,'predictions.csv')
            df_predictions.to_csv(csv_path, index=False)
            logger.info(f"Batch predictions saved to '{csv_path}'.")

            
        except Exception as e:
            raise AirlineException(e,sys)