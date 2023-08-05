import sys
from src.logger import logger


def msg_detail(error,error_details:sys):
    _,_,exc_tb = error_details.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = f"Error occured in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"

    return error_msg

class CustomException(Exception):
    def __init__(self, error_msg,error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = msg_detail(error=error_msg,error_details=error_detail)

    def __str__(self):
        return self.error_msg

if __name__=="__main__":
    try:
        a = 1/0
        # raise CustomException
    except Exception as e:
        logger.info('ZERO Division ERRor')
        raise CustomException(e,sys)


