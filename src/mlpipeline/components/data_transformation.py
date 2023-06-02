import pandas as pd
import numpy as np
import os,sys
from mlpipeline.exception import CustomException
from mlpipeline.logger import logging
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from mlpipeline.config.configuration import ConfigurationManager
from mlpipeline.entity.config_entity import DataTransformationConfig,DataIngestionConfig
from mlpipeline.entity.artifact_entity import DataTransformationArtifact
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from mlpipeline.utils.common import save_object


class DataTransformation:
    def __init__(self,data_transformation_config_info:DataTransformationConfig):
        try:
            self.data_transformation_config_info = data_transformation_config_info
            self.train_df = pd.read_csv(self.data_transformation_config_info.train_data_file)
            self.test_df = pd.read_csv(self.data_transformation_config_info.test_data_file)

            logging.info(f"{'>>' * 10}Data Transformation log started.{'<<' * 10} ")
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation Started")
            numerical_features =['age', 'workclass', 'educational-num', 'marital-status', 'occupation',
       'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
       'hours-per-week']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )
            
            preprocessor = ColumnTransformer([
                ("NUm_pipeline",num_pipeline,numerical_features)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys) from e
        
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
            logging.info("Outliers handling code")
            raise CustomException(e,sys) from e
        
    def inititate_data_transformation(self)->DataTransformationArtifact:

        try:
            data_transformation_config_info = self.data_transformation_config_info
            train_data = pd.read_csv(self.data_transformation_config_info.train_data_file)
            test_data = pd.read_csv(self.data_transformation_config_info.test_data_file)

            numerical_features =['age', 'workclass', 'educational-num', 'marital-status', 'occupation',
       'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
       'hours-per-week']
            
            for col in numerical_features:
                self.remove_outlier_IQR(col= col ,df = train_data)
            logging.info("Outliers on our train data")

            for col in numerical_features:   
                self.remove_outlier_IQR(col= col ,df = test_data)
            logging.info("Outliers on our test data")

            preprocess_obj = self.get_data_transformation_obj()

            target_columns = "income"
            drop_columns = [target_columns]

            logging.info("Splitting data into dependent and independent features")
            input_feature_train_data = train_data.drop(target_columns,axis=1)
            target_feature_train_data = train_data[target_columns]

            input_feature_test_data = test_data.drop(target_columns,axis=1)
            target_feature_test_data = test_data[target_columns]

            input_train_arr = preprocess_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocess_obj.transform(input_feature_test_data)

            train_array = np.c_[input_train_arr,np.array(target_feature_train_data)]
            test_array = np.c_[input_test_arr,np.array(target_feature_test_data)]

            preprocessing_obj_path = data_transformation_config_info.preprocessed_object_file_path


            logging.info("Saving the pickle file")
            save_object(file_path = self.data_transformation_config_info.preprocessed_object_file_path,obj=preprocess_obj)

            data_transformation_artifact = DataTransformationArtifact(preprocessed_object_path=preprocessing_obj_path)

            return(train_array,test_array,self.data_transformation_config_info.preprocessed_object_file_path)
            #return data_transformation_artifact
            
        
        except Exception as e:
            raise CustomException(e,sys) from e
        

if __name__ == "__main__":
    config = ConfigurationManager(config_file_path="configs\config.yaml")
    data_ingestion_config_info = config.get_data_ingestion_config()
    data_transformation_config = config.get_data_transformation_config(data_ingestion_config=data_ingestion_config_info)
    data_transformation = DataTransformation(data_transformation_config_info=data_transformation_config)
    data_transformation_response = data_transformation.inititate_data_transformation()