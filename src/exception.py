import os
import sys
# Add project root to path to resolve src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Generate detailed error message with file name and line number
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    """
    Custom exception class to handle and log errors with detailed information
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        
        # Log the error
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message

# Test the exception
if __name__ == "__main__":
    try:
        # Simulate an error
        x = 1 / 0
    except Exception as e:
        raise CustomException(e, sys)