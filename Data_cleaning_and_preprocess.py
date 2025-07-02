import pandas as pd
import numpy as np
import os

# --- Helper Functions for Data Cleaning ---

def get_initial_dataframe_statistics(dataframe):
    """
    Calculates initial statistics of a DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing:
               - int: Number of attributes (columns).
               - int: Number of data rows.
               - pd.Index: Column names.
    """
    num_attributes = len(dataframe.columns)
    num_rows = len(dataframe)
    attribute_names = dataframe.columns
    return num_attributes, num_rows, attribute_names

def handle_missing_values(dataframe, numeric_strategy='mean', non_numeric_strategy='mode', fill_value=None):
    """
    Handles missing values in the DataFrame based on specified strategies.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        numeric_strategy (str): Strategy for numeric columns ('mean', 'median', 'value', 'drop').
        non_numeric_strategy (str): Strategy for non-numeric columns ('mode', 'value', 'drop').
        fill_value: Specific value to use for filling NaNs if strategy is 'value'.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    if numeric_strategy == 'drop' or non_numeric_strategy == 'drop':
        return dataframe.dropna()

    processed_df = dataframe.copy() # Work on a copy to avoid modifying original outside function scope

    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = processed_df.select_dtypes(exclude=[np.number]).columns

    # Handle numeric columns
    if numeric_strategy == 'mean':
        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
    elif numeric_strategy == 'median':
        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())
    elif numeric_strategy == 'value' and fill_value is not None:
        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(fill_value)

    # Handle non-numeric columns
    if non_numeric_strategy == 'mode':
        for col in non_numeric_cols:
            # Ensure mode is not empty before attempting to fill
            mode_val = processed_df[col].mode()
            if not mode_val.empty:
                processed_df[col] = processed_df[col].fillna(mode_val[0])
            # else: if mode is empty, column might be all NaN, consider other handling

    return processed_df

def find_outliers_iqr(dataframe, threshold=1.5):
    """
    Identifies outliers in numeric columns using the IQR method.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        threshold (float): Multiplier for the IQR to define outlier bounds.

    Returns:
        dict: A dictionary where keys are column names and values are DataFrames of outliers.
    """
    outliers_data = {}
    for column in dataframe.select_dtypes(include=[np.number]).columns:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers_data[column] = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
    return outliers_data

def remove_outliers_iqr(dataframe, threshold=1.5):
    """
    Removes outliers from numeric columns using the IQR method.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        threshold (float): Multiplier for the IQR to define outlier bounds.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    processed_df = dataframe.copy()
    for column in processed_df.select_dtypes(include=[np.number]).columns:
        Q1 = processed_df[column].quantile(0.25)
        Q3 = processed_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        processed_df = processed_df[(processed_df[column] >= lower_bound) & (processed_df[column] <= upper_bound)]
    return processed_df

def convert_data_types(dataframe):
    """
    Attempts to convert columns to numeric types where possible.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with corrected data types.
    """
    processed_df = dataframe.copy()
    for column in processed_df.columns:
        processed_df[column] = pd.to_numeric(processed_df[column], errors='ignore')
    return processed_df

def drop_duplicate_rows(dataframe):
    """
    Removes duplicate rows from the DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicate rows removed.
    """
    return dataframe.drop_duplicates()

# --- Main Data Cleaning Workflow ---

if __name__ == "__main__":
    # Define file paths
    # Using os.path.join for cross-platform compatibility and r'' for raw strings with backslashes
    CURRENT_DIR = os.path.dirname(__file__)
    ORIGINAL_DATA_PATH = os.path.join(CURRENT_DIR, 'Data', 'car_prices_ori.csv')
    CLEANED_DATA_OUTPUT_PATH = os.path.join(CURRENT_DIR, 'Data', 'car_prices_clean.csv')

    # Load the original dataset
    try:
        raw_car_data_df = pd.read_csv(ORIGINAL_DATA_PATH)
        print(f"Original dataset loaded from: {ORIGINAL_DATA_PATH}")
    except FileNotFoundError:
        print(f"Error: Original dataset not found at '{ORIGINAL_DATA_PATH}'. Please ensure the file exists.")
        exit()

    # Get initial statistics
    num_attrs, num_rows, attributes = get_initial_dataframe_statistics(raw_car_data_df)
    print(f"Initial Dataset Statistics:")
    print(f"  Number of attributes: {num_attrs}")
    print(f"  Number of rows: {num_rows}")
    print(f"  Attributes: {', '.join(attributes)}")
    print("-" * 30)

    # --- Apply Cleaning Steps ---
    processed_car_data_df = raw_car_data_df.copy() # Start with a working copy

    print("Handling missing data (dropping rows with any NaN)...")
    processed_car_data_df = handle_missing_values(processed_car_data_df, numeric_strategy='drop', non_numeric_strategy='drop')
    print(f"  Rows after handling missing data: {len(processed_car_data_df)}")
    print("-" * 30)

    # Example of identifying and removing outliers (currently commented out in original script)
    # Uncomment and modify if you wish to apply outlier removal
    '''
    print("Identifying and removing outliers (using IQR method)...")
    initial_rows_before_outlier_removal = len(processed_car_data_df)
    outliers_identified = find_outliers_iqr(processed_car_data_df)
    print("  Outliers identified in the dataset:")
    for column, outlier_subset in outliers_identified.items():
        print(f"    Column '{column}' has {len(outlier_subset)} outliers.")
    processed_car_data_df = remove_outliers_iqr(processed_car_data_df)
    print(f"  Rows after outlier removal: {len(processed_car_data_df)} (Removed {initial_rows_before_outlier_removal - len(processed_car_data_df)} rows)")
    print("-" * 30)
    '''

    print("Correcting data types (attempting numeric conversion)...")
    processed_car_data_df = convert_data_types(processed_car_data_df)
    print("  Data types after conversion attempt:")
    print(processed_car_data_df.dtypes[processed_car_data_df.dtypes != 'object']) # Show numeric dtypes
    print("-" * 30)

    print("Removing duplicate rows...")
    initial_rows_before_duplicates = len(processed_car_data_df)
    processed_car_data_df = drop_duplicate_rows(processed_car_data_df)
    print(f"  Rows after removing duplicates: {len(processed_car_data_df)} (Removed {initial_rows_before_duplicates - len(processed_car_data_df)} duplicates)")
    print("-" * 30)

    # Save the cleaned dataset
    try:
        processed_car_data_df.to_csv(CLEANED_DATA_OUTPUT_PATH, index=False)
        print(f'Cleaned dataset successfully saved to: {CLEANED_DATA_OUTPUT_PATH}')
    except Exception as e:
        print(f"Error saving cleaned dataset: {e}")

    print("\nData cleaning and preprocessing complete.")