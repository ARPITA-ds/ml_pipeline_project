import os,sys
import pandas as pd
import numpy as np
from pathlib import Path
from mlpipeline.logger import logging
from mlpipeline.exception import CustomException
from dataclasses import dataclass
from mlpipeline.config.configuration import ConfigurationManager

from mlpipeline.entity.config_entity import DataIngestionConfig
from mlpipeline.entity.artifact_entity import DataIngestionArtifact

from ensure import ensure_annotations
from sklearn.model_selection import StratifiedShuffleSplit

class DataIngestion:
    def __init__(self,data_ingestion_config_info:DataIngestionConfig):
        try:
            logging.info(f"{'>>' * 10}Stage 01 data ingestion started  {'<<' * 10}")
            self.data_ingestion_config = data_ingestion_config_info
        except Exception as e:
            raise CustomException(e,sys)
        
    @ensure_annotations
    def download_data(self,dataset_download_id:str,raw_data_file_path:Path)->bool:
        try:
            logging.info(f"Downloading data from github")
            raw_data_frame = pd.read_csv(dataset_download_id)
            raw_data_frame.to_csv(raw_data_file_path,index=False)
            logging.info("Dataset downloaded successfully")

            return True
        except Exception as e:
            raise CustomException(e,sys) from e
        
    @ensure_annotations
    def split_data_as_train_test(self,data_file_path:Path)->DataIngestionArtifact:
        
        try:
            logging.info(f"{'>>' * 20}Data splitting.{'<<' * 20}")
            train_file_path = self.data_ingestion_config.ingested_train_file_path
            test_file_path = self.data_ingestion_config.ingested_test_file_path

            logging.info(f"Reading csv file: [{data_file_path}]")
            raw_data_frame = pd.read_csv(data_file_path)

            logging.info("Splitting data into train and test")
            strat_train_set = None
            strat_test_set = None
            
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
            
           
            for train_index, test_index in split.split(raw_data_frame , raw_data_frame['income']):
                strat_train_set = raw_data_frame.loc[train_index]
                strat_test_set = raw_data_frame.loc[test_index]

            if strat_train_set is not None:
                logging.info(f"Exporting training dataset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path, index=False)

            if strat_test_set is not None:
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path, index=False)
                data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                                test_file_path=test_file_path)
                logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
                return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """ initiate data ingestion"""
        try:
            data_ingestion_config_info = self.data_ingestion_config
            dataset_download_id = data_ingestion_config_info.dataset_download_id
            raw_data_file_path = data_ingestion_config_info.raw_data_file_path

            self.download_data(dataset_download_id, Path(raw_data_file_path))

            data_ingestion_response_info = self.split_data_as_train_test(data_file_path=Path(raw_data_file_path))
            logging.info(f"{'>>' * 20}Data Ingestion artifact.{'<<' * 20}")
            logging.info(f" Data Ingestion Artifact{data_ingestion_response_info.dict()}")
            logging.info(f"{'>>' * 20}Data Ingestion completed.{'<<' * 20}")
            return data_ingestion_response_info
        except Exception as e:
            raise CustomException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 20}Data Ingestion log completed.{'<<' * 20} \n\n")

    
if __name__=="__main__":
    config = ConfigurationManager(config_file_path='configs\config.yaml')
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion_response = data_ingestion.initiate_data_ingestion()

# src\mlpipeline\components\stage_01_data_ingestion.py