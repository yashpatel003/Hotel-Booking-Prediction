import os 
import sys 
from src.Hotel_Booking_Prediction.exception.exception import CustomException
from src.Hotel_Booking_Prediction.logging.logger import logger
from src.Hotel_Booking_Prediction.utils.common import read_sql_data
import pandas as pd 
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self):
        """
        Constructor for the DataIngestion class.
        Initializes the paths for raw, train, and test data files.
        """
        # Define paths for raw, train, and test data
        self.raw_data_path = os.path.join("src", "Hotel_Booking_Prediction", "data", "raw_data.csv")
        self.train_data_path = os.path.join("src", "Hotel_Booking_Prediction", "data", "train_data.csv")
        self.test_data_path = os.path.join("src", "Hotel_Booking_Prediction", "data", "test_data.csv")
        
    def initiate_data_ingestion(self):
        """
        Main method to handle the data ingestion process.
        Steps:
        1. Reads data from MySQL.
        2. Saves the raw data to a CSV file.
        3. Splits the data into training and testing sets.
        4. Saves the training and testing sets to CSV files.
        5. Logs the progress and returns the paths to the saved files.
        """
        try:
            ## Step 1: Reading the data from MySQL
            df = read_sql_data()  # Fetch data from MySQL using the `read_sql_data` function
            logger.info("Reading completed from MySQL database")  

            ## Step 2: Save raw data to CSV
            df.to_csv(self.raw_data_path, index=False, header=True)  # Save raw data to the specified path
            logger.info(f"Raw data saved to {self.raw_data_path}")  

            ## Step 3: Split data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)  # Split data (80% train, 20% test)
            train_set.to_csv(self.train_data_path, index=False, header=True)  # Save training data to CSV
            test_set.to_csv(self.test_data_path, index=False, header=True)  # Save testing data to CSV
            logger.info(f"Train data saved to {self.train_data_path}") 
            logger.info(f"Test data saved to {self.test_data_path}") 

            ## Step 4: Log completion of data ingestion
            logger.info("Data Ingestion is completed") 

            ## Step 5: Return paths to the saved train and test data files
            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            ## Handle any exceptions that occur during the process  
            raise CustomException(e, sys)  # Raise a custom exception with the error details