import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# project name
project_name = "Hotel_Booking_Prediction"

# List of files and directories to create for the project
file_list = [
    # GitHub Actions workflow directory (for CI/CD)
    ".github/workflows/.gitkeep",
    
    # Source code directory structure
    f"src/{project_name}/__init__.py",  
    f"src/{project_name}/components/__init__.py",  
    f"src/{project_name}/components/data_ingestion.py",  
    f"src/{project_name}/components/data_transformation.py",  
    f"src/{project_name}/components/model_training.py",  
    f"src/{project_name}/components/model_evaluation.py",  
    f"src/{project_name}/pipelines/__init__.py",  
    f"src/{project_name}/pipelines/training_pipeline.py",  
    f"src/{project_name}/pipelines/prediction_pipeline.py",  
    f"src/{project_name}/utils/__init__.py",  
    f"src/{project_name}/utils/common.py",  
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/logging/logger.py",
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/exception/exception.py",
    f"src/{project_name}/data/raw_data.csv",
    f"src/{project_name}/data/train_data.csv",
    f"src/{project_name}/data/test_data.csv",

    
    "app.py",
    
    # Project requirements and Dockerfile
    "requirements.txt", 
    "Dockerfile",  
    
    # Project setup and documentation
    "setup.py",  # Package setup file
    "README.md",  # Project documentation
    
    # Git ignore file to exclude unnecessary files from version control
    ".gitignore",
     
]

# Loop through the file list and create directories/files
for filepath in file_list:
    # Convert the filepath to a Path object for easier manipulation
    filepath = Path(filepath)
    
    # Split the filepath into directory and filename
    filedir, filename = os.path.split(filepath)
    
    # Create the directory if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")
    
    # Create an empty file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Create an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")