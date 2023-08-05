from src.logger import logger
from src.exceptions import CustomException
logger.info('Hello')
import sys

if __name__=="__main__":
    try:
        a = 1/0
        # raise CustomException
    except Exception as e:
        logger.info('ZERO Division ERRor')
        raise CustomException(e,sys)