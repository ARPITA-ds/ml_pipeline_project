from pathlib import Path

from pydantic import BaseModel, DirectoryPath, FilePath


class DataIngestionArtifact(BaseModel):
    train_file_path: FilePath
    test_file_path: FilePath
