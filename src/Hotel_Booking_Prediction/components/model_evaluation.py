import sys 
from src.Hotel_Booking_Prediction.exception.exception import CustomException
from src.Hotel_Booking_Prediction.logging.logger import logger
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluator:
    def __init__(self, models, param):
        """
        Initializes the ModelEvaluator with models and hyperparameter grid.

        Args:
            models (dict): A dictionary of models where keys are model names and values are model instances.
            param (dict): A dictionary containing hyperparameters for each model.
        """
        self.models = models
        self.param = param

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluates multiple machine learning models by performing hyperparameter tuning using GridSearchCV
        and calculating classification metrics on both training and testing datasets.

        Args:
            X_train (array-like): The feature set for the training dataset.
            y_train (array-like): The target labels for the training dataset.
            X_test (array-like): The feature set for the testing dataset.
            y_test (array-like): The target labels for the testing dataset.

        Returns:
            dict: A dictionary containing classification metrics (accuracy, precision, recall, f1_score) for each model.
        """
        try:
            report = {}

            for i in range(len(self.models)):
                model_name = list(self.models.keys())[i]
                model = self.models[model_name]
                para = self.param[model_name]  

                # Use GridSearchCV for hyperparameter tuning with cross-validation
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)  

                # Set the best hyperparameters to the model
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)  

                # Make predictions on both training and test datasets
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate classification metrics: accuracy, precision, recall, and f1_score
                accuracy = accuracy_score(y_test, y_test_pred)
                precision = precision_score(y_test, y_test_pred)
                recall = recall_score(y_test, y_test_pred)
                f1 = f1_score(y_test, y_test_pred)

                # Store the metrics for each model in the report dictionary
                report[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                }

            logger.info("Model evaluation completed successfully.")
            return report

        except Exception as ex:
            raise CustomException(ex,sys)