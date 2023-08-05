from src.exceptions import CustomException
from src.logger import logger
import pickle
import sys


def load_pkl(file_path):
    try:
        with open(file_path,'rb') as pkl_file:
            loaded_file = pickle.load(file=pkl_file)
            logger.info(f'{file_path} in pickle file has been loaded.')

            return loaded_file
    except Exception as e:
        raise CustomException(e,sys)

    

def dump_pkl(obj,file_path):
    try:
        with open(file_path,'wb') as pkl_file:
            dumped_file = pickle.dump(obj=obj,file=pkl_file)
            logger.info(f'{file_path} in pickle file has been dumped.')
            
            return dumped_file
    
    except Exception as e:
        raise CustomException(e,sys)
