import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.Hotel_Booking_Prediction.exception.exception import CustomException
from src.Hotel_Booking_Prediction.logging.logger import logger
from src.Hotel_Booking_Prediction.utils.common import save_object

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join("src", "Hotel_Booking_Prediction", "data", "preprocessor.pkl")

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessing pipeline.

        Returns:
            ColumnTransformer: Preprocessing pipeline object.
        """
        try:
            logger.info("Creating preprocessing pipeline.")

            # Define columns to be processed
            # Numerical columns for scaling
            numerical_columns = [
                "lead_time", 'deposit_given', 'total_customer',
                'previous_cancellations', 'is_repeated_guest', "days_in_waiting_list",
                'previous_bookings_not_canceled', "total_of_special_requests", "adr",
                "total_stay", "booking_changes", "required_car_parking_spaces", "is_family"
            ]

            # Categorical columns for encoding
            categorical_columns = [
                "hotel", "distribution_channel", "reserved_room_type", "customer_type", "meal", "market_segment"
            ]

            # Numerical pipeline: Handles missing values and scales features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),  # Fills missing values with mean
                    ("scaler", StandardScaler())                
                ]
            )

            # Categorical pipeline: Handles missing values and applies one-hot encoding
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fills missing values with mode
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))    
                ]
            )

            # Combine both pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),  
                    ("cat", cat_pipeline, categorical_columns)  
                ]
            )

            logger.info("Preprocessing pipeline created successfully.")
            return preprocessor

        except Exception as ex:
            raise CustomException(ex, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies the preprocessing pipeline to the input data and saves the preprocessor object.

        Args:
            train_path (str): Path to the training dataset CSV file.
            test_path (str): Path to the test dataset CSV file.

        Returns:
            tuple: Transformed training array, testing array, and preprocessor object file path.
        """
        try:
            logger.info("Starting data transformation process.")

            # Read train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Verify the presence of the target column
            target_column = "is_canceled"
            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise CustomException(f"Target column '{target_column}' is missing in the dataset.", sys)

            # Handle missing or invalid target column
            if train_df[target_column].isna().sum() > 0:
                train_df[target_column].fillna(train_df[target_column].mode()[0], inplace=True)
            if test_df[target_column].isna().sum() > 0:
                test_df[target_column].fillna(test_df[target_column].mode()[0], inplace=True)

            # Ensure target column has at least two classes
            if train_df[target_column].nunique() <= 1 or test_df[target_column].nunique() <= 1:
                raise CustomException(f"Target column '{target_column}' must have at least two classes.", sys)

            # Map deposit types to numerical values for processing
            dict1 = {'No Deposit': 0, 'Non Refund': 1, 'Refundable': 0}

            # Feature engineering for the training dataset
            train_df["total_stay"] = train_df["stays_in_weekend_nights"] + train_df["stays_in_week_nights"]  # Total stay duration
            train_df["arrival_month_num"] = pd.to_datetime(  # Convert arrival date to month number
                train_df["arrival_date_year"].astype(str) + train_df["arrival_date_month"], 
                format='%Y%B').dt.month
            train_df["is_family"] = ((train_df["children"] > 0) | (train_df["adults"] > 1)).astype(int)  # Family flag
            train_df['total_customer'] = train_df['adults'] + train_df['babies'] + train_df['children']  # Total number of customers
            train_df['deposit_given'] = train_df['deposit_type'].map(dict1)  # Map deposit types

            # Feature engineering for the test dataset (similar to training dataset)
            test_df["total_stay"] = test_df["stays_in_weekend_nights"] + test_df["stays_in_week_nights"]
            test_df["arrival_month_num"] = pd.to_datetime(
                test_df["arrival_date_year"].astype(str) + test_df["arrival_date_month"], 
                format='%Y%B').dt.month
            test_df["is_family"] = ((test_df["children"] > 0) | (test_df["adults"] > 1)).astype(int)
            test_df['total_customer'] = test_df['adults'] + test_df['babies'] + test_df['children']
            test_df['deposit_given'] = test_df['deposit_type'].map(dict1)

            # Drop unnecessary columns from both datasets
            columns_to_drop = ['agent', 'company', 'stays_in_weekend_nights', 'stays_in_week_nights',
                               'children', 'adults', 'deposit_type', 'babies']
            train_df = train_df.drop(columns=columns_to_drop, axis=1)
            test_df = test_df.drop(columns=columns_to_drop, axis=1)

            # Get the preprocessing pipeline
            preprocessor = self.get_data_transformer_object()

            # Separate features and target for training and testing datasets
            input_features_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            # Logging shapes of input data
            logger.info(f"Shape of training data before preprocessing: {input_features_train_df.shape}")
            logger.info(f"Shape of testing data before preprocessing: {input_features_test_df.shape}")

            # Apply preprocessing
            logger.info("Applying Preprocessing on training and test data.")
            input_features_train_arr = preprocessor.fit_transform(input_features_train_df)  # Fit and transform training data
            input_features_test_arr = preprocessor.transform(input_features_test_df)  # Transform test data

            # Combine features and target into arrays
            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing pipeline object to a file
            logger.info("Saving the preprocessing object.")
            save_object(
                self.preprocessor_obj_file_path, 
                preprocessor
            )

            logger.info("Data transformation process completed successfully.")
            return (
                train_arr, 
                test_arr, 
                self.preprocessor_obj_file_path
            )

        except Exception as ex:
            raise CustomException(ex, sys)
