from pathlib import Path 

import platformdirs 

from llmsh .constants import APP_NAME ,CONFIG_FILENAME ,SESSIONS_DIR_NAME 


def config_path ()->Path :
    return Path (platformdirs .user_config_dir (APP_NAME ))/CONFIG_FILENAME 


def data_dir ()->Path :
    return Path (platformdirs .user_data_dir (APP_NAME ))


def sessions_dir ()->Path :
    return data_dir ()/SESSIONS_DIR_NAME 
