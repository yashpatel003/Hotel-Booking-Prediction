import sys,os
from src.Hotel_Booking_Prediction.logging.logger import logger
from src.Hotel_Booking_Prediction.exception.exception import CustomException
from src.Hotel_Booking_Prediction.components.data_ingestion import DataIngestion

if __name__=="__main__":
    try:
        logger.info("The excution start")
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

    
    except Exception as ex:
        raise CustomException(ex,sys)