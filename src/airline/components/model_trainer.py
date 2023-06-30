import os,sys
import numpy as np
import pandas as pd

from airline.exception import AirlineException
from airline.logger import logger
from airline.entity import DataTransformationConfig,DataIngestionConfig,ModelTrainerConfig
from airline.config.configuration import ConfigurationManager
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from airline.utils import save_object

from sklearn.metrics import accuracy_score,classification_report,precision_score,recall_score,f1_score,roc_auc_score,roc_curve


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig):
        try:
            logger.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise AirlineException(e,sys) from e

    def evaluate_model(self,X_train,y_train,X_test,y_test,models,params):
        try:
            report = {}

            for i in range(len(models)):
                model = list(models.values())[i]
                para = params[list(models.keys())[i]]

                grid = GridSearchCV(model,para,cv=3,n_jobs=1,verbose=2)
                grid.fit(X_train,y_train)

                model.set_params(**grid.best_params_)
                model.fit(X_train,y_train)

                y_test_pred = model.predict(X_test)

                test_model_score = accuracy_score(y_test,y_test_pred)

                report[list(models.values())[i]] = test_model_score

            return report
        except Exception as e:
            raise AirlineException(e,sys)
        

    def initate_model_training(self,train_array,test_array):
        try:
            logger.info(f"{'>>'*10} Model Training started {'<<'*10}")
            logger.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])
            

            models={

            'Logistic':LogisticRegression(),          
            'KNN neighbour':KNeighborsClassifier()
            }


            params = {

                "Logistic":{
                    "class_weight":["balanced"],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                },

                "KNN neighbour":{
                        'n_neighbors': [3, 5, 7],
                        'weights': ['uniform', 'distance'],
                        
                    }
                
            }
            
           
            
            model_report:dict=self.evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                            models=models,params=params)
          
            

            print(f"model_report: {model_report}")

            df = pd.DataFrame(list(model_report.items()), columns=['Model', 'Accuracy'])
            logger.info(f" model report {pd.DataFrame(list(model_report.items()), columns=['Model', 'Accuracy']) }")
            print(df)
            
            print('\n======================================================================\n')

            logger.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            logger.info(f"best model score: {best_model_score}")

            best_model_name = list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # logger.info(f"{plot_confusion_matrix(best_model_name, X_test, self.y_test_pred, cmap='Blues', values_format='d')}")

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logger.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model

            )

            logger.info(f"{'>>'*10} Model Training Completed {'<<'*10}")
          

        except Exception as e:
            logger.info('Exception occured at Model Training')
            raise AirlineException(e,sys)



