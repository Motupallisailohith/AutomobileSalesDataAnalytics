import pandas as pd
import os

# --- Configuration ---
CLEANED_DATA_PATH = "Data/car_prices_clean.csv" # Path to the cleaned dataset
OUTPUT_DATA_DIR = "Data/" # Directory to save classified/filtered data

# --- Helper Functions for Data Classification ---

def load_data(file_path):
    """
    Loads the dataset from the specified file path.

    Args:
        file_path (str): The full path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure the data cleaning script has been run.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

def select_column_interactively(dataframe):
    """
    Allows the user to interactively select a column from the DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: The selected column data, or None if selection fails.
    """
    while True:
        try:
            print("\nColumns available:")
            for idx, col_name in enumerate(dataframe.columns):
                print(f"{idx}: {col_name}")

            column_index_input = input("\nEnter the number of the column you want to select: ")
            column_index = int(column_index_input)

            if 0 <= column_index < len(dataframe.columns):
                selected_column_name = dataframe.columns[column_index]
                print(f"\nYou have selected the column: {selected_column_name}")
                return dataframe[selected_column_name]
            else:
                print("Invalid column number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An unexpected error occurred during column selection: {e}")
            return None

def display_column_details(column_series):
    """
    Displays value counts for the selected column and prompts for detail selection.

    Args:
        column_series (pd.Series): The selected column data.
    """
    value_counts_dict = column_series.value_counts().to_dict()

    if not value_counts_dict:
        print(f"No unique values found in column '{column_series.name}'.")
        return

    select_detail_from_column(value_counts_dict, column_series)

def select_detail_from_column(value_counts_dict, column_series):
    """
    Allows the user to select a specific value from the column's details and save related data.

    Args:
        value_counts_dict (dict): Dictionary of value counts for the column.
        column_series (pd.Series): The selected column data.
    """
    print(f"\nDetails available for '{column_series.name}':")
    indexed_values = {i + 1: key for i, key in enumerate(value_counts_dict.keys())}

    for index, value in indexed_values.items():
        print(f"{index}: {value} (Count: {value_counts_dict[value]})")

    while True:
        try:
            selected_key_input = input("\nEnter the number of the value you want to select: ")
            selected_key_index = int(selected_key_input)
            selected_value = indexed_values[selected_key_index]
            print(f"\nYou have selected the value: {selected_value}")
            save_filtered_data_to_csv(selected_value, column_series)
            break
        except (ValueError, KeyError):
            print("Invalid input. Please enter a valid number from the list.")
        except Exception as e:
            print(f"An unexpected error occurred during detail selection: {e}")
            break

def save_filtered_data_to_csv(selected_value, column_series):
    """
    Filters the original DataFrame based on the selected column value and saves it to a new CSV.

    Args:
        selected_value: The specific value to filter by.
        column_series (pd.Series): The selected column data from the original DataFrame.
    """
    # Access the original DataFrame (assuming it's named 'car_sales_df' in the main scope)
    global car_sales_df
    if car_sales_df is None:
        print("Error: Original DataFrame not loaded. Cannot save filtered data.")
        return

    filtered_data_df = car_sales_df[car_sales_df[column_series.name] == selected_value]

    # Ensure output directory exists
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    # Sanitize selected_value for filename
    safe_file_name_part = str(selected_value).replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_file_name = f"{safe_file_name_part}.csv"
    output_file_path = os.path.join(OUTPUT_DATA_DIR, output_file_name)

    try:
        filtered_data_df.to_csv(output_file_path, index=False)
        print(f"\nFiltered data for '{column_series.name}' == '{selected_value}' saved to: {output_file_path}")
        print(f"Number of rows in filtered data: {len(filtered_data_df)}")
    except Exception as e:
        print(f"Error saving filtered data: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    # Get the directory of the current script
    current_script_dir = os.path.dirname(__file__)
    # Construct the full path to the cleaned data file
    full_data_path = os.path.join(current_script_dir, CLEANED_DATA_PATH)

    # Load the dataset
    car_sales_df = load_data(full_data_path)

    if car_sales_df is not None:
        select_column_interactively(car_sales_df)
    else:
        print("Data could not be loaded. Exiting data classification script.")