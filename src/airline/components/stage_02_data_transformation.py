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
from sklearn.base import BaseEstimator,TransformerMixin

class Feature_Engineering(BaseEstimator,TransformerMixin):
    def __init__(self):
         logger.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")


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


    def transform_data(self,df):
        try:
            num_col = [feature for feature in df.columns if df[feature].dtype != '0']
            
            logger.info(f"numerical_columns: {num_col}")


            cat_col = [feature for feature in df.columns if df[feature].dtype == 'O']
            logger.info(f"categorical_columns: {cat_col}")

            #df.drop(columns=['Unnamed: 0','Gate location','Departure Delay in Minutes', 'Arrival Delay in Minutes'], inplace=True, axis=1)

            logger.info(f"columns in dataframe are: {df.columns}")

            numerical_columns = [ 'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                                   'Ease of Online booking', 'Food and drink', 'Online boarding', 'Seat comfort', 
                                   'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 
                                   'Checkin service', 'Inflight service', 'Cleanliness' ]


# outlier

            for col in numerical_columns:
                self.remove_outlier_IQR(col=col, df= df)
            
            logger.info(f"Outlier capped in train df")
            return df 
            
        except Exception as e:
            raise AirlineException(e,sys)
        
    
    def fit(self,X,y=None):
        return self
    
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            transformed_df=self.transform_data(X)
                
            return transformed_df
        except Exception as e:
            raise AirlineException(e,sys) from e

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
        
    
    #def remove_outlier_IQR(self,col,df):
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
        

    def dropping_missing_values(self,data):
        try:
            missing_values = data.isnull().sum()
            if missing_values.any():
                logger.info("Missing Value found")
                cleaned_data = data.dropna(inplace = True)
                return cleaned_data
            else:
                logger.info("No missing value found")
        except Exception as e:
            raise AirlineException(e,sys) from e
        
    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering())])
            return feature_engineering
        except Exception as e:
            raise AirlineException(e,sys) from e
    
    def initiate_data_transformation(self):
        try:
            data_transformation_config_info = self.data_transformation_config_info
            train_data = pd.read_csv(self.data_transformation_config_info.train_data_file)
            test_data = pd.read_csv(self.data_transformation_config_info.test_data_file)

            
            print(train_data.columns)
            print(test_data.columns)

            train_data.columns = train_data.columns.str.strip()
            test_data.columns = test_data.columns.str.strip()

            # Map values for the 'satisfaction' column
            satisfaction_mapping = {
                'satisfied': 1,
                'neutral or dissatisfied': 0
            }
            train_data['satisfaction'] = train_data['satisfaction'].map(satisfaction_mapping)
            test_data['satisfaction'] = test_data['satisfaction'].map(satisfaction_mapping)


            logger.info('Read train and test data completed')
            logger.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logger.info(f'Test Dataframe Head  : \n{test_data.head().to_string()}')


            numerical_features =['Age','Flight Distance','Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking',
                    'Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling',
                    'Checkin service','Inflight service','Cleanliness']
            
            #for col in numerical_features:
                #self.remove_outlier_IQR(col=col,df=train_data)
            #logger.info("Outliers on our data")

            #for col in numerical_features:
                #self.remove_outlier_IQR(col=col,df=test_data)
            #logger.info("Outliers on our test data")

            #logger.info("Dropping irrelevant featues")
            #train_data = train_data.drop(['Departure Delay in Minutes', 'Arrival Delay in Minutes','Gate location'],axis=1)
            #test_data = test_data.drop(['Departure Delay in Minutes', 'Arrival Delay in Minutes','Gate location'],axis=1)


            logger.info(f"{'>>' * 10}Handling missing value log started.{'<<' * 10} ")

            logger.info(f"Missing_values in train_dataframe are:{(train_data.isnull().mean())*100}")
            self.dropping_missing_values(data = train_data)
            logger.info(f"After handling Missing_values in train_dataframe are:{(train_data.isnull().mean())*100}")

            logger.info(f"Missing_values in test_dataframe are:{(test_data.isnull().mean())*100}")
            self.dropping_missing_values(data = test_data)
            logger.info(f"After handling Missing_values in test_dataframe are:{(test_data.isnull().mean())*100}")

            logger.info(f"{'>>' * 10}Handling missing value log Ended.{'<<' * 10} ")

            logger.info("Dropping irrelevant featues")
            train_data.drop(['Departure Delay in Minutes', 'Arrival Delay in Minutes','Gate location'],axis=1)
            test_data.drop(['Departure Delay in Minutes', 'Arrival Delay in Minutes','Gate location'],axis=1)

            logger.info(f"obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()

            logger.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logger.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logger.info(f"Feature Enineering - Train Data ")
            train_data = fe_obj.fit_transform(train_data)
            logger.info(">>>" * 20 + " Test data " + "<<<" * 20)
            logger.info(f"Feature Enineering - Test Data ")
            test_data = fe_obj.transform(test_data)

            if train_data is not None:
                train_data.to_csv("train_data.csv")
                logger.info(f"Saving csv to train_data.csv")

            if test_data is not None:
                test_data.to_csv("test_data.csv")
                logger.info(f"Saving csv to test_data.csv")


           #train_data.to_csv("train_data.csv")
            #test_data.to_csv("test_data.csv")
            logger.info(f"Saving csv to train_data and test_data.csv")
            

            logger.info("Creating Pre=processing object")
            preprocessing_obj = self.get_data_transformation_obj()

            target_column = ["satisfaction"]

            logger.info("Creating train test file")
            #X_train = train_df.drop(['satisfaction'],axis=1)
            X_train = train_data.drop(columns=target_column,axis=1)
            y_train = train_data['satisfaction']

            
            X_test = test_data.drop(['satisfaction'],axis=1)
            y_test = test_data['satisfaction']

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

            save_object(file_path = self.data_transformation_config_info.feature_eng_obj_file_path,obj= fe_obj)
            logger.info("Feature engineering file saved")

            return(train_arr,test_arr,self.data_transformation_config_info.preprocessed_object_file_path)

        except Exception as e:
            raise AirlineException(e,sys) from e
        