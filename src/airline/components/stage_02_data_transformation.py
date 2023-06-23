import os,sys
import pandas as pd
import numpy as np
from box import ConfigBox
from airline.config.configuration import ConfigurationManager
from airline.entity import DataTransformationConfig,DataIngestionConfig
from airline.exception import AirlineException
from airline.logger import logger
from airline.utils.common import read_yaml,save_object,create_directories
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataTransformation:
    def __init__(self,data_transformation_config_info:DataTransformationConfig) -> None:
        try:
            self.data_transformation_config_info = data_transformation_config_info
            self.train_df = pd.read_csv(self.data_transformation_config_info.train_data_file)
            self.test_df = pd.read_csv(self.data_transformation_config_info.test_data_file)
            logger.info(f"{'>>' * 10}Data Transformation log started.{'<<' * 10} ")
        except Exception as e:
            raise AirlineException(e,sys)
        

    def get_data_transformation_obj(self):
        try:
            logger.info("Loading data Transformation")

            numerical_features =['Age','Flight Distance','Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking',
                    'Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling',
                    'Checkin service','Inflight service','Cleanliness']
            
            categorical_column = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

            label_encoder_column = ['satisfaction']
            
            num_pipeline = Pipeline(
                steps=[("imputer",SimpleImputer(strategy="median")),
                       ("scaler",StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[('impute',SimpleImputer(strategy='most_frequent')),
                       ('onehot',OneHotEncoder(handle_unknown='ignore')),
                       ('scaler',StandardScaler(with_mean=False))]
            )

            #target_enc_pipeline = Pipeline(
             #    steps=[('impute',SimpleImputer(strategy='most_frequent')),
              #         ('Label',LabelEncoder()),
               #        ('scaler',StandardScaler(with_mean=False))]
            #)

           # preprocessor = ColumnTransformer([
            #    ("Num_pipeline",num_pipeline,numerical_features),
             #   ('categorical',categorical_pipeline,categorical_column),
              #  ("target",target_enc_pipeline,label_encoder_column)])

            preprocessor = ColumnTransformer([
               ("Num_pipeline",num_pipeline,numerical_features),
               ('categorical',categorical_pipeline,categorical_column)])
            
            return preprocessor
        
            logger.info("pipeline completed")


        except Exception as e:
            raise AirlineException(e,sys) from e
        
    
    def remove_outlier_IQR(self,col,df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            iqr = Q3-Q1

            upper_limit = Q3+1.5*iqr
            lower_limit = Q1-1.5*iqr

            df.loc[(df[col]>upper_limit),col] = upper_limit
            df.loc[(df[col]<lower_limit),col] = lower_limit

            return df

        except Exception as e:
            logger.info("Outliers handling code")
            raise AirlineException(e,sys) from e
    



    def initiate_data_transformation(self):
        try:
            logger.info("Handling Missing Values")
            train_df = self.train_df.dropna()
            test_df = self.test_df.dropna()

            logger.info("Dropping irrelevant featues")
            train_df.drop(['Departure Delay in Minutes', 'Arrival Delay in Minutes','Gate location'],axis=1,inplace=True)

            logger.info("Creating Pre=processing object")
            preprocessing_obj = self.get_data_transformation_obj()

            #target_column = ["satisfaction"]satisfaction

            logger.info("Creating train test file")
            X_train = train_df.drop(['satisfaction'],axis=1)
            y_train = train_df['satisfaction']

            
            X_test = test_df.drop(['satisfaction'],axis=1)
            y_test = test_df['satisfaction']

            logger.info("Doing transformation")
            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            #y_train = preprocessing_obj.fit_transform(y_train)
            #y_test = preprocessing_obj.transform(y_test)

            logger.info("Creating train test array")
            train_arr = np.c_[X_train,np.array(y_train)]
            test_arr = np.c_[X_test,np.array(y_test)]

            logger.info("Creating dataframe")
            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            

            #logger.info("Creating directories for train file")
            os.makedirs(os.path.dirname(self.data_transformation_config_info.data_transformed_train_file_path),exist_ok=True)
            df_train.to_csv(self.data_transformation_config_info.data_transformed_train_file_path,index=False,header=True)
            
            logger.info("Creating directories for test file")
            os.makedirs(os.path.dirname(self.data_transformation_config_info.data_transformed_test_file_path),exist_ok=True)
            df_test.to_csv(self.data_transformation_config_info.data_transformed_test_file_path,index=False,header=True)

            logger.info("Saving object")
            save_object(file_path = self.data_transformation_config_info.preprocessed_object_file_path,obj=preprocessing_obj)

            return(train_arr,test_arr,self.data_transformation_config_info.preprocessed_object_file_path)

        except Exception as e:
            raise AirlineException(e,sys) from e
        