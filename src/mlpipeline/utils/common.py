import json
import os
import sys
from pathlib import Path
from typing import Any

import dill
import joblib
import numpy as np
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations


from mlpipeline.exception import CustomException
from mlpipeline.logger import logging
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_curve,f1_score,precision_score,recall_score
from sklearn.model_selection import GridSearchCV

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns
       read_yaml file always returns dict so configbox is used
    Args:
        path_to_yaml (str): path like input
    Raises:
        ValueError: if yaml file is empty
        e: empty file
    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories
    Args:
        path_to_directories (list): list of path of directories
        verbose (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")


@ensure_annotations
def save_object(file_path: Path, obj: object):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
    


def evaluate_model(X_train,y_train,X_test,y_test,models,params):
        try:
            report = {}

            for i in range(len(list(models))):
                model_name = list(models.keys())[i]
                model = models[model_name]
                para = params[model_name]

                grid = GridSearchCV(model, para, cv=2, n_jobs=-1, verbose=2)
                grid.fit(X_train, y_train)
                
                model.set_params(**grid.best_params_)
                model.fit(X_train, y_train)
                
                y_test_pred = model.predict(X_test)
                
                test_model_score = accuracy_score(y_test, y_test_pred)
                
                report[model_name] = test_model_score
        
        except Exception as e:
            raise CustomException(e,sys) from e
            
            return report