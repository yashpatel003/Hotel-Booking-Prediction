import os , sys , pickle , pymysql
from src.Hotel_Booking_Prediction.exception.exception import CustomException
from src.Hotel_Booking_Prediction.logging.logger import logger
import pandas as pd 
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve database connection details from environment variables
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv('db')

def read_sql_data():
    """
    Function to read data from a SQL database and return it as a pandas DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame containing the data from the SQL query.
    
    """
    logger.info("Reading SQL database started")
    
    try:
        # Establish a connection to the MySQL database
        connection = pymysql.connect(
            host=host, 
            user=user, 
            password=password, 
            database=database, 
            cursorclass=pymysql.cursors.DictCursor
        )
        logger.info("Connection Established")
        
        # Use the connection to execute a SQL query
        with connection:
            with connection.cursor() as cursor:
                # Define the SQL query to fetch all records from the 'hotel_bookings' table
                sql_query = "SELECT * FROM hotel_bookings"
                cursor.execute(sql_query)
                
                # Fetch all rows from the executed query
                result = cursor.fetchall()
                
                # Convert the result to a pandas DataFrame
                df = pd.DataFrame(result)
                print(df.head())
        return df

    except Exception as ex:
        # Raise any exceptions if encountered during the process as a CustomException
        raise CustomException(ex, sys)
    
def save_object(file_path, obj):
    """
    Saves a Python object to a file in binary format using pickle.

    Args:
        file_path (str): The full file path where the object will be saved.
        obj (object): The Python object to be saved.
    """
    try:
        # Extract the directory path from the provided file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't already exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

