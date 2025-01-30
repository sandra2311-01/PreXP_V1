import webbrowser
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import category_encoders as ce  # For Leave-One-Out and Hash encoding
from scipy.stats import kendalltau
from category_encoders import LeaveOneOutEncoder
import pandas as pd
from ydata_profiling import ProfileReport
import pickle
# from IPython.core.display import display, HTML          # deprecated
from IPython.display import display, HTML
import cohere
 # type: ignore
from reasoning_logger import * 
from data_loader import *
from data_profiler import *
from insights_extractor import *
from missing_data_handler import *
from date_handler import *
from data_encoder import *
from scaling_and_outlier_handler import *
from ask_cohere import *

 # Test File Path
test_file = "hotel_bookings.csv"

if __name__ == "__main__":

    #Step 1: Initialize ReasoningLogger
    reasoning_logger = ReasoningLogger()

    # Step 2: Test DataLoader
    print("\n--- Step 2: Data Loading ---")
    data_loader = DataLoader(file_path=test_file, logger=reasoning_logger)
    data_loader.load_dataset()
    print(data_loader.df.head())

    # Step 3: Test DataProfiler
    print("\n--- Step 3: Data Profiling ---")
    data_profiler = DataProfiler(df=data_loader.df, logger=reasoning_logger)
    data_profiler.generate_profile()  # Generate profile
    data_profiler.load_profile()      # Load the profile

    # Step 4: Test InsightsExtractor
    print("\n--- Step 4: Insights Extraction ---")
    insights_extractor = InsightsExtractor(profile=data_profiler.profile, logger=reasoning_logger)
    all_insights = insights_extractor.extract_description()
   
    # Read the content of the HTML report
    report_file = 'data_report.html'

    # Open the HTML file in the default web browser
    try:
        webbrowser.open(report_file)
        print(f"Report opened in your default web browser: {report_file}")
    except Exception as e:
        print(f"Error opening the report: {e}")

    # Step 1: Initialize the MissingDataHandler
    missing_data_handler = MissingDataHandler(
        insight_logs=reasoning_logger.get_logs(log_type="insight"),  # Pass the insights logs
        logger=reasoning_logger,  # Use the same logger for preprocessing
        low_threshold=0.05,       # Default low threshold for MCAR
        high_threshold=0.30,      # Default high threshold for MNAR
        min_rows_percentage=0.90  # Minimum rows to retain
    )

    # Step 2: Categorize Missing Data
    missing_data_categories = missing_data_handler.categorize_missing_data(df=data_loader.df)

    print("\n--- Missing Data Categories ---")
    print(missing_data_categories)

    # Step 3: Handle Missing Data
    df_cleaned = missing_data_handler.handle_missing_data(
        df=data_loader.df,
        missing_data_categories=missing_data_categories
    )

    # Step 2: Initialize the DateHandler
    date_handler = DateHandler(df=df_cleaned, insights=all_insights, logger=reasoning_logger)

    # Step 3: Detect and convert date-related columns
    df_with_dates, full_date_columns, date_component_columns = date_handler.detect_and_convert()

    # Initialize DataEncoder
    encoder = DataEncoder(df=df_with_dates, insights=all_insights, logger=reasoning_logger)

    # Call the encode_columns function
    encoded_df = encoder.encode_columns(target_column='is_canceled')
    handler = ScalingAndOutlierHandler(df=encoded_df, insights=all_insights, logger=reasoning_logger)
    scaling_decisions, outlier_decisions = handler.determine_scaling_and_outlier_handling()
    df_processed = handler.apply_scaling_and_outliers(scaling_decisions, outlier_decisions)

    print(df_processed.head())
    print(df_processed.info())

    query = "How was country encoded?"
    response = ask_cohere_about_logs(reasoning_logger, query)
    print(f"User's Query: {query}")
    print(f"LLM Response: {response}")