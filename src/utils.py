from src.exceptions import CustomException
from src.logger import logger
import pickle
import sys

def load_pkl(file_path):
    try:
        with open(file_path, 'rb') as pkl_file:
            loaded_file = pickle.load(pkl_file)
            logger.info(f'{file_path} loaded successfully from the pickle file.')
            return loaded_file
    except Exception as e:
        logger.error(f"Error occurred while loading {file_path} from pickle file: {e}")
        raise CustomException(e, sys)

def dump_pkl(obj, file_path):
    try:
        with open(file_path, 'wb') as pkl_file:
            pickle.dump(obj, pkl_file)
            logger.info(f'{file_path} has been successfully dumped to a pickle file.')
    except Exception as e:
        logger.error(f"Error occurred while dumping {file_path} to a pickle file: {e}")
        raise CustomException(e, sys)
