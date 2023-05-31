import os
import sys
from pathlib import Path

from  mlpipeline.constants import (CURRENT_TIME_STAMP, ROOT_DIR)

from  mlpipeline.entity.config_entity import (DataIngestionConfig,TrainingPipelineConfig)
                              
from  mlpipeline.exception import CustomException
from  mlpipeline.logger import logging
from  mlpipeline.utils.common import create_directories, read_yaml
from mlpipeline.constants import CONFIG_FILE_PATH



class ConfigurationManager:

    def __init__(self, config_file_path: Path = CONFIG_FILE_PATH ) -> None:
        """ Configuration manager class to read the configuration file and create the configuration objects.
        Args:
            config_file_path (Path, optional): _description_. Defaults to CONFIG_FILE_PATH.
        Raises:
            CustomException: _description_ if the configuration file is not found or if the configuration file is not
            in the correct format.
        """

        try:
            self.config_info = read_yaml(path_to_yaml=Path(config_file_path))
            self.pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = CURRENT_TIME_STAMP

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """ Get the data ingestion configuration object.
        Raises:
            CustomException: _description_
        Returns:
            DataIngestionConfig:  Pydanctic base model data ingestion configuration object. 
            dataset_download_id: str
            raw_data_file_path: Path
            ingested_train_file_path: Path
            ingested_test_data_path: Path
            random_state: int     """

        try:
            logging.info("Getting data ingestion configuration.")
            data_ingestion_info = self.config_info.data_ingestion_config
            pipeline_config = self.pipeline_config
            artifact_dir = pipeline_config.artifact_dir
            dataset_download_id = data_ingestion_info.dataset_download_id
            data_ingestion_dir_name = data_ingestion_info.ingestion_dir
            raw_data_dir = data_ingestion_info.raw_data_dir
            raw_file_name = data_ingestion_info.dataset_download_file_name

            data_ingestion_dir = os.path.join(artifact_dir, data_ingestion_dir_name)
            raw_data_file_path = os.path.join(data_ingestion_dir, raw_data_dir, raw_file_name)
            ingested_dir_name = data_ingestion_info.ingested_dir
            ingested_dir_path = os.path.join(data_ingestion_dir, ingested_dir_name)

            ingested_train_file_path = os.path.join(ingested_dir_path, data_ingestion_info.ingested_train_file)
            ingested_test_file_path = os.path.join(ingested_dir_path, data_ingestion_info.ingested_test_file)
            create_directories([os.path.dirname(raw_data_file_path), os.path.dirname(ingested_train_file_path)])

            data_ingestion_config = DataIngestionConfig(dataset_download_id=dataset_download_id,
                                                        raw_data_file_path=raw_data_file_path,
                                                        ingested_train_file_path=ingested_train_file_path,
                                                        ingested_test_file_path=ingested_test_file_path)
            
            logging.info(f"Data ingestion config: {data_ingestion_config.dict()}")
            logging.info("Data ingestion configuration completed.")

            return data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        """ Get the training pipeline configuration object.
        class TrainingPipelineConfig(BaseModel):
                artifact_dir: DirectoryPath
                training_random_state: int
                pipeline_name: str
                experiment_code: str
        
        """
        try:
            training_config = self.config_info.training_pipeline_config
            training_pipeline_name = training_config.pipeline_name
            training_experiment_code = training_config.experiment_code
            training_artifacts = os.path.join(ROOT_DIR, training_config.artifact_dir)
            create_directories(path_to_directories=[training_artifacts])
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=training_artifacts,
                                                              experiment_code=training_experiment_code,
                                                              pipeline_name=training_pipeline_name)
            logging.info(f"Training pipeline config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise CustomException(e, sys) from e

