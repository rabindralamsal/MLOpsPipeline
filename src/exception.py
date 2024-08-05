import sys
from src.logger import logging

def error_message(error, detail: sys):
    _, _, tb = detail.exc_info()
    file = tb.tb_frame.f_code.co_filename
    message = f"Error in {file} at Line {tb.tb_lineno}. Error message: {str(error)}"
    return message


class CustomException(Exception):
    def __init__(self, error, detail: sys):
        super().__init__()
        self.error_message = error_message(error, detail)

    def __str__(self):
        return self.error_message