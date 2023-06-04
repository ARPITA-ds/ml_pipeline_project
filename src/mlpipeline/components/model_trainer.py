import pandas as pd
import numpy as np
import os,sys
from mlpipeline.exception import CustomException
from mlpipeline.logger import logging
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from mlpipeline.config.configuration import ConfigurationManager
from mlpipeline.entity.config_entity import DataTransformationConfig,DataIngestionConfig,ModelTrainerConfig
from mlpipeline.entity.artifact_entity import DataTransformationArtifact
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from mlpipeline.utils.common import save_object,evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self,model_trainer_config_info:ModelTrainerConfig):
        self.model_trainer_config_info = model_trainer_config_info


    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model = {
                "Random Forest":RandomForestClassifier(),
                "Decision Tree":DecisionTreeClassifier(),
                "Logistic":LogisticRegression()
            }

            params ={
                "Random Forest":{
                    "class_weight":["balanced"],
                    "n_estimators":[20,50,30],
                    "max_depth":[10,8,5],
                    "min_samples_split":[2,5,10]
                },
                "Decision Tree":{
                    "class_weight":["balanced"],
                    "criterion":['gini','entropy','log_loss'],
                    "splitter":['best','random'],
                    "max_depth":[3,4,5,0],
                    "min_samples_split":[2,3,4,5],
                    "min_samples_leaf":[1,2,3],
                    "max_features" :["auto" , "sqrt" , "log2"]
                },
                    "Logistic":{
                    "class_weight":["balanced"],
                    'penalty':['l1','l2'],
                    'C':[0.001,0.01,0.1,1,10,100],
                    'solver':['liblinear','saga']
                }
            }

            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test =y_test,
                                                models = model, params = params)
            
            # To gest best model from our report Dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = model[best_model_name]
            
            model_train_config_info = self.model_trainer_config
            model_report_dir = model_train_config_info.model_report_dir
 
            logging.INFO(f"best model found,Model Name is {best_model_name},accuracy Score:{best_model_score}")

            save_object(file_path = self.model_trainer_config.trained_model_file_path,
                        obj= best_model)
        except Exception as e:
            raise CustomException(e,sys) from e
        
#if __name__ == "__main__":
   # config = ConfigurationManager(config_file_path='configs\config.yaml')
    #data_ingestion_config = config.get_data_ingestion_config()
    #data_transformation_config = config.get_data_transformation_config(data_ingestion_config=data_ingestion_config)

    #model_trainer_config = config.get_model_trainer_config(data_ingestion_config=data_ingestion_config, data_transformation_config_info=data_transformation_config)

    #model_trainer = ModelTrainer(model_trainer_config_info=model_trainer_config)

   # _ = model_trainer.initiate_model_trainer()