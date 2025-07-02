import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# --- Configuration ---
CLEANED_DATA_PATH = "Data/car_prices_clean.csv" # Path to the cleaned dataset
IMG_OUTPUT_DIR = "Img/Time_Series_Analysis_Img/" # Directory to save time series analysis plots
DECOMPOSED_DATA_OUTPUT_DIR = "Data/" # Directory to save decomposed data CSVs

# --- Helper Function for Time Series Analysis ---

def perform_seasonal_decomposition(dataframe, resample_period='M', min_observations=2):
    """
    Performs additive seasonal decomposition on car sales data.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing 'saledate' and 'sellingprice'.
        resample_period (str): The period to resample the data ('D' for daily, 'M' for monthly, 'Q' for quarterly).
        min_observations (int): Minimum number of observations required for decomposition.
    """
    print(f"\n--- Performing Time Series Decomposition for period: {resample_period} ---")

    # Ensure 'saledate' is datetime and set as index
    processed_df = dataframe.copy()
    processed_df['saledate'] = pd.to_datetime(processed_df['saledate'], errors='coerce')
    processed_df.dropna(subset=['saledate'], inplace=True)
    processed_df['saledate'] = processed_df['saledate'].apply(lambda x: x.replace(tzinfo=None)) # Remove timezone
    processed_df.set_index('saledate', inplace=True)

    # Resample sales data (sum of selling price) by the specified period
    # Ensure numerical data for summation
    resampled_sales = processed_df['sellingprice'].resample(resample_period).sum()

    # Drop any periods with NaN values that might result from resampling empty periods
    resampled_sales = resampled_sales.dropna()

    if len(resampled_sales) >= min_observations:
        # Perform additive seasonal decomposition
        # model='additive' is suitable when seasonality magnitude is constant over time
        decomposition = seasonal_decompose(resampled_sales, model='additive', period=12 if resample_period == 'M' else (7 if resample_period == 'D' else None)) # Period depends on resample

        # Plot decomposed components
        fig, (ax_obs, ax_trend, ax_seasonal, ax_resid) = plt.subplots(4, 1, figsize=(14, 10))
        decomposition.observed.plot(ax=ax_obs)
        ax_obs.set_ylabel('Observed')
        ax_obs.set_title(f'Decomposition of Car Sales ({resample_period} Resampling)')
        decomposition.trend.plot(ax=ax_trend)
        ax_trend.set_ylabel('Trend')
        decomposition.seasonal.plot(ax=ax_seasonal)
        ax_seasonal.set_ylabel('Seasonal')
        decomposition.resid.plot(ax=ax_resid)
        ax_resid.set_ylabel('Residual')
        plt.xlabel('Date') # Add x-label for clarity
        plt.tight_layout()

        # Ensure output directory exists before saving plot
        os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
        plot_filename = os.path.join(IMG_OUTPUT_DIR, f'time_series_decomposition_{resample_period}.png')
        plt.savefig(plot_filename)
        plt.show()
        print(f"Time series decomposition plot saved to: {plot_filename}")

        # Save the decomposed components to CSV files
        decomposition_df = pd.DataFrame({
            'observed': decomposition.observed,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        })
        # Ensure output directory exists before saving CSV
        os.makedirs(DECOMPOSED_DATA_OUTPUT_DIR, exist_ok=True)
        csv_filename = os.path.join(DECOMPOSED_DATA_OUTPUT_DIR, f'decomposed_car_sales_{resample_period}.csv')
        decomposition_df.to_csv(csv_filename)
        print(f"Decomposed components saved to: {csv_filename}")

        print(f"Time series decomposition for '{resample_period}' complete.")
    else:
        print(f"Not enough observations ({len(resampled_sales)}) for seasonal decomposition for resample period '{resample_period}'. At least {min_observations} are required.")

# --- Main Execution ---
if __name__ == "__main__":
    # Get the directory of the current script
    current_script_dir = os.path.dirname(__file__)
    # Construct the full path to the cleaned data file
    full_data_path = os.path.join(current_script_dir, CLEANED_DATA_PATH)

    # Load the cleaned dataset
    try:
        car_sales_df = pd.read_csv(full_data_path)
        print(f"Dataset loaded successfully from: {full_data_path}")
        print("\nFirst 5 rows of the dataset:")
        print(car_sales_df.head())
        print("-" * 50)
    except FileNotFoundError:
        print(f"Error: The file '{full_data_path}' was not found.")
        print("Please ensure the data cleaning script has been run and the file exists.")
        exit()

    # Call the function with different resample periods
    perform_seasonal_decomposition(car_sales_df, 'D', min_observations=7)  # Daily, requires at least 7 days for weekly seasonality if applicable
    perform_seasonal_decomposition(car_sales_df, 'W', min_observations=2)  # Weekly
    perform_seasonal_decomposition(car_sales_df, 'M', min_observations=2)  # Monthly
    perform_seasonal_decomposition(car_sales_df, 'Q', min_observations=2)  # Quarterly

    print("\nTime Series Analysis script finished.")