import sys
from src.Hotel_Booking_Prediction.logging import logging  

def error_message_detail(error, error_detail: sys):
    """
    Constructs a detailed error message including the script name, line number, and error message.

    Args:
        error: The exception/error object.
        error_detail: The sys module to extract exception details.

    Returns:
        str: A formatted error message with script name, line number, and error message.
    """
    # Extract exception traceback details
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the filename and line number where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    # Construct the error message
    error_message = (
        f"Error occurred in Python script: [{file_name}] "
        f"at line number: [{line_number}] "
        f"with error message: [{str(error)}]"
    )
    
    return error_message


class CustomException(Exception):
    """
    A custom exception class to provide detailed error messages.

    Args:
        error_message: The error message to be displayed.
        error_details: The sys module to extract exception details.
    """
    def __init__(self, error_message, error_details: sys):
        # Call the parent class (Exception) constructor
        super().__init__(error_message)
        
        # Generate a detailed error message using the helper function
        self.error_message = error_message_detail(error_message, error_details)

    def __str__(self):
        """
        Override the __str__ method to return the detailed error message.
        """
        return self.error_message