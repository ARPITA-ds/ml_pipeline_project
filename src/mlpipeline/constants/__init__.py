from pathlib import Path
import os
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y%m%d%H%M%S')}"


ROOT_DIR = os.getcwd()
CURRENT_TIME_STAMP = get_current_time_stamp()

CONFIG_DIR = os.path.join(ROOT_DIR,'configs')

CONFIG_FILE_PATH = Path("configs/config.yaml")