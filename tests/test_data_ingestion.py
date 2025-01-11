import os
import sys
import pandas as pd
import pytest
from src.Hotel_Booking_Prediction.components.data_ingestion import DataIngestion
from src.Hotel_Booking_Prediction.exception.exception import CustomException

# Fixture to initialize DataIngestion and clean up after tests
@pytest.fixture
def data_ingestion():
    """
    Fixture to initialize DataIngestion and clean up after tests.
    """
    data_ingestion = DataIngestion()
    yield data_ingestion
    # # Clean up: Remove created files after the test
    # if os.path.exists(data_ingestion.raw_data_path):
    #     os.remove(data_ingestion.raw_data_path)
    # if os.path.exists(data_ingestion.train_data_path):
    #     os.remove(data_ingestion.train_data_path)
    # if os.path.exists(data_ingestion.test_data_path):
    #     os.remove(data_ingestion.test_data_path)
   
def test_data_ingestion(data_ingestion):
    """
    Test the data ingestion process.
    """
    try:
        # Initiate data ingestion
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Check if the raw data file is created
        assert os.path.exists(data_ingestion.raw_data_path), "Raw data file not created"

        # Check if the train data file is created
        assert os.path.exists(data_ingestion.train_data_path), "Train data file not created"

        # Check if the test data file is created
        assert os.path.exists(data_ingestion.test_data_path), "Test data file not created"

        # Check if the train and test data files are not empty
        train_df = pd.read_csv(data_ingestion.train_data_path)
        test_df = pd.read_csv(data_ingestion.test_data_path)
        assert len(train_df) > 0, "Train data file is empty"
        assert len(test_df) > 0, "Test data file is empty"

        # Check if the train and test data are split correctly (80% train, 20% test)
        raw_df = pd.read_csv(data_ingestion.raw_data_path)
        assert len(train_df) >= int(0.8 * len(raw_df)), "Train data size is incorrect"
        assert len(test_df) >= int(0.2 * len(raw_df)), "Test data size is incorrect"

    except Exception as e:
        raise CustomException(e, sys)
