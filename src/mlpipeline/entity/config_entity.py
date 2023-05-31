from pathlib import Path

from pydantic import BaseModel,DirectoryPath,FilePath

class DataIngestionConfig(BaseModel):
    raw_data_file_path:Path
    ingested_train_file_path:Path
    ingested_test_file_path:Path
    dataset_download_id:str


class TrainingPipelineConfig(BaseModel):
    artifact_dir: DirectoryPath
    pipeline_name: str
    