import pandas as pd
import os # Import os for path handling

# --- Configuration & Data Loading ---
# Define the path to your cleaned dataset
CLEANED_DATA_PATH = "Data/car_prices_clean.csv" # Using forward slashes for cross-OS compatibility

# Ensure the script runs only when executed directly
if __name__ == "__main__":
    # Get the directory of the current script
    current_script_dir = os.path.dirname(__file__)
    # Construct the full path to the cleaned data file
    full_data_path = os.path.join(current_script_dir, CLEANED_DATA_PATH)

    try:
        car_sales_df = pd.read_csv(full_data_path)
        print(f"Dataset loaded successfully from: {full_data_path}")
        print("\nFirst 5 rows of the dataset:")
        print(car_sales_df.head())
        print("-" * 50)
    except FileNotFoundError:
        print(f"Error: The file '{full_data_path}' was not found.")
        print("Please ensure the data cleaning script has been run and the file exists.")
        exit() # Exit if the data isn't found, as the script can't proceed

    # --- Generate Descriptive Statistics ---

    print("Generating Descriptive Statistics for Numerical Columns:")
    # Get descriptive statistics for numerical columns
    numerical_desc_stats = car_sales_df.describe()
    print(numerical_desc_stats)
    print("-" * 50)

    print("\nGenerating Descriptive Statistics with Custom Percentiles for Numerical Columns:")
    # Get descriptive statistics with custom percentiles
    numerical_desc_stats_custom_percentiles = car_sales_df.describe(percentiles=[.1, .25, .5, .75, .9])
    print(numerical_desc_stats_custom_percentiles)
    print("-" * 50)

    print("\nGenerating Descriptive Statistics for Categorical Columns:")
    # Get descriptive statistics for categorical/object columns
    categorical_desc_stats = car_sales_df.describe(include=['object'])
    print(categorical_desc_stats)
    print("-" * 50)

    print("\nDescriptive statistics generation complete.")