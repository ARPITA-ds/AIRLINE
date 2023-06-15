from pathlib import Path
import os
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y%m%m%H%M%S')}"

"""
Getting current working directory
"""
ROOT_DIR = os.getcwd()

CURRENT_TIME_STAMP = get_current_time_stamp()

CONFIG_FILE_PATH = os.path.join(ROOT_DIR,'configs\config.yaml')

