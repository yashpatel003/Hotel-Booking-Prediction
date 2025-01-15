import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.Hotel_Booking_Prediction.exception.exception import CustomException
from src.Hotel_Booking_Prediction.logging.logger import logger
from src.Hotel_Booking_Prediction.utils.common import save_object
from src.Hotel_Booking_Prediction.components.model_evaluation import ModelEvaluator
from sklearn.metrics import confusion_matrix, classification_report


class ModelTrainer:
    def __init__(self):
        self.trained_model_file_path = os.path.join("src", "Hotel_Booking_Prediction", "data", "model.pkl")

    @staticmethod
    def eval_metrics(actual, prediction):
        """
        Evaluate performance metrics for the model: accuracy, precision, recall, and f1 score.

        Args:
            actual (array) : test labels
            prediction (array): Prediction labels.

        Returns:
            tuple: accuracy, precision, recall, and f1 score.
        """
        accuracy = accuracy_score(actual, prediction)
        precision = precision_score(actual, prediction)
        recall = recall_score(actual, prediction)
        f1 = f1_score(actual, prediction)
        return accuracy, precision, recall, f1

    def initiate_model_trainer(self, train_arr, test_arr):
        """
        Train models, evaluate them, and save the best performing model.

        Args:
            train_arr (array): Training dataset (features and target combined).
            test_arr (array): Testing dataset (features and target combined).
        """
        try:
            # Split training and test data into features (X) and target (y)
            logger.info("Splitting training and test input data")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)

            # Define the models and their hyperparameters
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7]
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'saga']
                },
            }

            # Evaluate models using the ModelEvaluator
            evaluator_model = ModelEvaluator(models, params)
            model_report: dict = evaluator_model.evaluate(X_train, y_train, X_test, y_test)

            # Log the performance metrics for all models
            for model_name, metrics in model_report.items():
                logger.info(f"Metrics for {model_name}:")
                logger.info(f"  Accuracy: {metrics['accuracy']}")
                logger.info(f"  Precision: {metrics['precision']}")
                logger.info(f"  Recall: {metrics['recall']}")
                logger.info(f"  F1 Score: {metrics['f1_score']}")
                logger.info("-" * 20)

            # Find the best model based on accuracy
            best_model_name = max(model_report, key=lambda name: model_report[name]['accuracy'])
            best_model = models[best_model_name]

            logger.info(f"Best model identified: {best_model_name}")

            # Train the best model on the full training dataset
            best_model.fit(X_train, y_train)

            # Evaluate the best model on the test data
            predicted = best_model.predict(X_test)
            accuracy, precision, recall, f1 = self.eval_metrics(y_test, predicted)
            print("Prediction distribution:", np.unique(predicted, return_counts=True))
            print("Classification Report:")
            print(classification_report(y_test, predicted))

            # Log final performance metrics for the best model
            logger.info(f"Final evaluation metrics for {best_model_name}:")
            logger.info(f"  Accuracy: {accuracy}")
            logger.info(f"  Precision: {precision}")
            logger.info(f"  Recall: {recall}")
            logger.info(f"  F1 Score: {f1}")

            # Raise exception if model accuracy is below a threshold
            if accuracy < 0.6:
                raise CustomException("No best model found with sufficient accuracy.")

            # Save the best model to disk
            save_object(self.trained_model_file_path, best_model)
            logger.info(f"Best model saved at {self.trained_model_file_path}")

        except Exception as e:
            raise CustomException(e, sys)
