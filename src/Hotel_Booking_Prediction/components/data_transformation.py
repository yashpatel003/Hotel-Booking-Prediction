import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
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

            # Define numerical and categorical columns
            numerical_columns = [
                "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults",
                "children", "babies", "days_in_waiting_list", "adr", "required_car_parking_spaces",
                "total_of_special_requests", "total_stay", "arrival_month_num"
            ]

            categorical_columns = [
                "hotel", "meal", "market_segment", "distribution_channel",
                "reserved_room_type", "deposit_type", "customer_type", "reservation_status"
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ]
            )

            logger.info("Preprocessing pipeline created successfully.")
            return preprocessor

        except Exception as ex:
            raise CustomException(ex,sys)

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

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Verify target column existence
            target_column = "is_canceled"
            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise CustomException(f"Target column '{target_column}' is missing in the dataset.", sys)

            # Handle missing or invalid target column
            if train_df[target_column].isna().sum() > 0:
                train_df[target_column].fillna(train_df[target_column].mode()[0], inplace=True)
            if test_df[target_column].isna().sum() > 0:
                test_df[target_column].fillna(test_df[target_column].mode()[0], inplace=True)

            if train_df[target_column].nunique() <= 1 or test_df[target_column].nunique() <= 1:
                raise CustomException(f"Target column '{target_column}' must have at least two classes.", sys)

            # Feature engineering
            train_df["total_stay"] = train_df["stays_in_weekend_nights"] + train_df["stays_in_week_nights"]
            train_df["arrival_month_num"] = pd.to_datetime(
                train_df["arrival_date_year"].astype(str) + train_df["arrival_date_month"], 
                format='%Y%B').dt.month

            test_df["total_stay"] = test_df["stays_in_weekend_nights"] + test_df["stays_in_week_nights"]
            test_df["arrival_month_num"] = pd.to_datetime(
                test_df["arrival_date_year"].astype(str) + test_df["arrival_date_month"], 
                format='%Y%B').dt.month

            # Get the preprocessing pipeline
            preprocessor = self.get_data_transformer_object()

            # Separate features and target
            input_features_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            # Logging shapes of input data
            logger.info(f"Shape of training data before preprocessing: {input_features_train_df.shape}")
            logger.info(f"Shape of testing data before preprocessing: {input_features_test_df.shape}")

            # Apply preprocessing
            logger.info("Applying Preprocessing on training and test data.")
            input_features_train_arr = preprocessor.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor.transform(input_features_test_df)

            # Combine features and target back into arrays
            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing pipeline
            logger.info("Saving the preprocessing object.")
            save_object(
                self.preprocessor_obj_file_path, 
                preprocessor
            )

            logger.info("Data transformation process completed successfully.")
            return (
                train_arr, 
                test_arr, 
                self.preprocessor_obj_file_path )

        except Exception as ex:
            raise CustomException(ex,sys)

