import os 
import sys 
from src.Hotel_Booking_Prediction.exception.exception import CustomException
from src.Hotel_Booking_Prediction.logging.logger import logger
import pandas as pd 
import pymysql
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
    
    Raises:
        CustomException: If any error occurs during the database connection or data retrieval.
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
        # Log the exception and raise a custom exception
        logger.error(f"Error occurred: {ex}")
        raise CustomException(ex, sys)

