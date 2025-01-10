import logging
import os
from datetime import datetime

# Define the log file name using the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path where the log file will be stored
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the directory if it doesn't exist
os.makedirs(log_path, exist_ok=True)

# Define the full path to the log file
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  #file to write logs to
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  #  log format
    level=logging.INFO,  # Set the logging level to INFO
)
