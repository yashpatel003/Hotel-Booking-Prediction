import sys,os
import pandas as pd 
from src.Hotel_Booking_Prediction.logging.logger import logger
from src.Hotel_Booking_Prediction.exception.exception import CustomException
from src.Hotel_Booking_Prediction.components.data_ingestion import DataIngestion
from src.Hotel_Booking_Prediction.components.data_transformation import DataTransformation
from src.Hotel_Booking_Prediction.components.model_training import ModelTrainer

if __name__=="__main__":
    try:
        logger.info("The excution start")
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        # Initialize DataTransformation class
        data_transformation = DataTransformation()
        train_arr,test_arr,preprocess_file_path = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        ## Model Training
        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        
    except Exception as ex:
        raise CustomException(ex,sys)