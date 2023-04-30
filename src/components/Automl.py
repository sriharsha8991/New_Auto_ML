import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #use to create pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from flaml import AutoML

@dataclass
class AutomlConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','Preprocessor.pkl')

class Automl:
    def __init__(self):
        self.AutomlConfig= AutomlConfig()

    def spliting(self,data):
        try:
            y = data.iloc[:, -1]  # Last column is the target variable 
            x = data.iloc[:, :-1]
            logging.info("Splitting the dependent and independent")

            # save_object(

            #     file_path=self.automate.preprocessor_obj_file_path,
            #     obj=preprocessing_obj
            # )
            settings = self.type(y)
            return x,y,settings

        except Exception as e:
        
            raise CustomException(e,sys)

    def type(self,y):
        logging.info("Figuring the type of the problem")
        if len(y.unique()) > 10: # if more than 10 unique target values, treat as regression
            problem_type = 'regression'
            automl_settings = {
            "time_budget": 60,  # total running time in seconds
            "metric": "r2",  # primary metrics can be chosen from: ['accuracy', 'roc_auc', 'roc_auc_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'f1', 'log_loss', 'mae', 'mse', 'r2'] Check the documentation for more details (https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML#optimization-metric)
            "task": "regerssion",  # task type
            "estimator_list": ["xgboost", "catboost", "lgbm"],
            "log_file_name": "flaml.log",  # flaml log file
        }
            return automl_settings
        
        else:
            problem_type = 'classification'
            automl_settings = {
            "time_budget": 60,  # total running time in seconds
            "metric": "accuracy",  # primary metrics can be chosen from: ['accuracy', 'roc_auc', 'roc_auc_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'f1', 'log_loss', 'mae', 'mse', 'r2'] Check the documentation for more details (https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML#optimization-metric)
            "task": "classification",  # task type
            "estimator_list": ["xgboost", "catboost", "lgbm"],
            "log_file_name": "flaml.log",  # flaml log file
        }

            return automl_settings
        
    
    
        
        
#"Standard Scaler": {"with_mean": [True, False]
    
    def automate(self,data):
        try:
            # data = pd.read_csv(r"Notebook\train.csv")
            logging.info("Got the dataframe")
            x,y,automl_settings= self.spliting(data)
            x_train, x_test, y_train, y_test = train_test_split(x,y)
            logging.info("Started preprocessing")
            numerical_transformer = Pipeline(steps=[('scaler', StandardScaler(with_mean = False))])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            preprocessing = ColumnTransformer(transformers=[
            ('num', numerical_transformer, x.select_dtypes(include=['int64', 'float64']).columns),
            ('cat', categorical_transformer, x.select_dtypes(include=['object']).columns)
        ]) 
            automl = AutoML()
            automl_pipeline = Pipeline([("Preprocessing",preprocessing),("automl", automl)])
            pipeline_settings = {f"automl__{key}": value for key, value in automl_settings.items()}
            logging.info("Started Fitting")
            automl_pipeline.fit(x_train, y_train, **pipeline_settings)

            automl1 = automl_pipeline.steps[1][1]

            logging.info("Showing the estimators and configuration")
            # print('Best ML leaner:', automl1.best_estimator)
            # print('Best hyperparmeter config:', automl1.best_config)
            # print('Best accuracy on validation data: {0:.4g}'.format(1 - automl1.best_loss))
            # print('Training duration of best run: {0:.4g} s'.format(automl1.best_config_train_time))



            return automl1.best_estimator,automl1.best_config,f'{(1 - automl1.best_loss)}',f'{automl1.best_config_train_time}'


        
        except Exception as e:
            raise CustomException(e,sys)
        
    
if __name__=="__main__":
    obj = AutomlConfig()
    ob1 = Automl()
    ob1.automate()



