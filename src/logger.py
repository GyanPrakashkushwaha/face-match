import os
import sys
import logging
from datetime import datetime

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir, f"{datetime.now():%d_%m_%H_%M_%S}.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath), # this will create log file in the folders
        logging.StreamHandler(sys.stdout) # this will print the statements in the terminal
    ]
)

logger = logging.getLogger("ProjectLogger")

