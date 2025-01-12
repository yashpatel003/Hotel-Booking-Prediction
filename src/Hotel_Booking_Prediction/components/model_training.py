import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from src.Hotel_Booking_Prediction.exception.exception import CustomException  
from src.Hotel_Booking_Prediction.logging.logger import logger  
from src.Hotel_Booking_Prediction.utils.common import save_object 
from src.Hotel_Booking_Prediction.components.model_evaluation import ModelEvaluator
from sklearn.metrics import confusion_matrix, classification_report

class ModelTrainer:
    def __init__(self):
        # Path where the trained model will be saved as a .pkl file
        self.trained_model_file_path = os.path.join("src", "Hotel_Booking_Prediction", "data", "model.pkl")


    def eval_metrics(self, actual, prediction):
        """
        Evaluate performance metrics for the model: accuracy, precision, recall, and f1 score

        """
        # Calculate accuracy, precision, recall, and f1 score
        accuracy = accuracy_score(actual, prediction)  # Accuracy: Correct predictions / Total predictions
        precision = precision_score(actual, prediction)  # Precision: Correct positive predictions / All positive predictions
        recall = recall_score(actual, prediction)  # Recall: Correct positive predictions / All actual positives
        f1 = f1_score(actual, prediction)  # F1 Score: Harmonic mean of precision and recall
        return accuracy, precision, recall, f1
    
   
    def initiate_model_trainer(self, train_arr, test_arr):
        """

            Train models and evaluate the best one based on their performance

        """
        try:
            logger.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],  # Features (all columns except the last) from the training data
                train_arr[:, -1],   # Target variable (last column) from the training data
                test_arr[:, :-1],   # Features (all columns except the last) from the test data
                test_arr[:, -1]     # Target variable (last column) from the test data
            )

            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
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

            # Initialize ModelEvaluator
            evaluator_model = ModelEvaluator(models, params)
            # Evaluate the models using the training and test data
            model_report: dict = evaluator_model.evaluate(X_train, y_train, X_test, y_test)

            best_model_score = max(model_report.values(), key=lambda x: x['accuracy'])  
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]


            print("This is the best model:")
            print(best_model_name)  


            # Train the best model on the full training dataset
            best_model.fit(X_train, y_train)

            # Make predictions on the test set using the trained best model
            predicted = best_model.predict(X_test)

            # Evaluate the performance of the best model on the test set
            accuracy, precision, recall, f1 = self.eval_metrics(y_test, predicted)
            print("Prediction distribution:", np.unique(predicted, return_counts=True))
            print("Classification Report:")
            print(classification_report(y_test, predicted))

            # Log the evaluation metrics for the best model
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")

            # If the model's accuracy is less than 60%, raise an exception
            if accuracy < 0.6:
                raise CustomException("No best model found with sufficient accuracy.")  # Raise a custom exception

            logger.info(f"Best model found on both training and testing datasets")  

            # Save the trained best model to the specified file path for future use
            save_object(file_path=self.trained_model_file_path, obj=best_model)

            # Return the accuracy of the best model as a performance metric
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
