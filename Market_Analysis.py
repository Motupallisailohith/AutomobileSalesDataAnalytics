import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration & Data Loading ---
CLEANED_DATA_PATH = "Data/car_prices_clean.csv" # Path to the cleaned dataset
IMG_OUTPUT_DIR = "Img/Market_Analysis_Img/" # Directory to save market analysis plots

# Ensure the script runs only when executed directly
if __name__ == "__main__":
    # Get the directory of the current script
    current_script_dir = os.path.dirname(__file__)
    # Construct the full path to the cleaned data file
    full_data_path = os.path.join(current_script_dir, CLEANED_DATA_PATH)
    # Construct the full path for image output directory
    full_img_output_path = os.path.join(current_script_dir, IMG_OUTPUT_DIR)
    os.makedirs(full_img_output_path, exist_ok=True) # Ensure the output directory exists

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

    # Convert 'saledate' to datetime (important for potential future analysis here)
    if 'saledate' in car_sales_df.columns:
        car_sales_df['saledate'] = pd.to_datetime(car_sales_df['saledate'], errors='coerce')
        car_sales_df.dropna(subset=['saledate'], inplace=True) # Drop rows where conversion failed
        print(" 'saledate' column converted to datetime format.")
        print("-" * 50)

    # --- Market Analysis: Average Selling Price by Make ---
    print("Analyzing average selling prices by car make...")
    average_price_by_make = car_sales_df.groupby('make')['sellingprice'].mean().sort_values(ascending=False)
    print("Top 10 Car Makes by Average Selling Price:")
    print(average_price_by_make.head(10))

    # Plot average selling prices across different makes
    plt.figure(figsize=(15, 8))
    sns.barplot(x=average_price_by_make.index, y=average_price_by_make.values, palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel('Car Make')
    plt.ylabel('Average Selling Price')
    plt.title('Average Selling Price by Make')
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig(os.path.join(full_img_output_path, "Average_Selling_Price_by_Make.png"))
    plt.show()
    print("Plot 'Average_Selling_Price_by_Make.png' generated and saved.")
    print("-" * 50)

    # --- Market Analysis: Average Selling Price by Model ---
    print("Analyzing average selling prices by car model...")
    # Get top N models for plotting to avoid overcrowding
    top_models = car_sales_df['model'].value_counts().head(20).index # Adjust 20 as needed
    average_price_by_model = car_sales_df[car_sales_df['model'].isin(top_models)].groupby('model')['sellingprice'].mean().sort_values(ascending=False)
    print(f"Top {len(top_models)} Car Models by Average Selling Price:")
    print(average_price_by_model.head())

    # Plot average selling prices across different models
    plt.figure(figsize=(15, 8))
    sns.barplot(x=average_price_by_model.index, y=average_price_by_model.values, palette='magma')
    plt.xticks(rotation=90)
    plt.xlabel('Car Model')
    plt.ylabel('Average Selling Price')
    plt.title(f'Average Selling Price by Model (Top {len(top_models)} Models)')
    plt.tight_layout()
    plt.savefig(os.path.join(full_img_output_path, "Average_Selling_Price_by_Model.png"))
    plt.show()
    print("Plot 'Average_Selling_Price_by_Model.png' generated and saved.")
    print("-" * 50)

    # --- Market Analysis: Impact of Car Condition on Selling Price ---
    print("Analyzing impact of car condition on selling price...")
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='condition', y='sellingprice', data=car_sales_df, palette='cividis',
                order=sorted(car_sales_df['condition'].unique())) # Sort conditions numerically
    plt.xlabel('Condition')
    plt.ylabel('Selling Price')
    plt.title('Impact of Car Condition on Selling Price')
    plt.tight_layout()
    plt.savefig(os.path.join(full_img_output_path, "Impact_of_Car_Condition_on_Selling_Price.png"))
    plt.show()
    print("Plot 'Impact_of_Car_Condition_on_Selling_Price.png' generated and saved.")
    print("-" * 50)

    # Additional analysis: Scatter plot of selling price vs. condition with a trend line
    print("Generating scatter plot of selling price vs. condition with trend line...")
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='condition', y='sellingprice', data=car_sales_df, alpha=0.3, color='grey', label='Data Points')
    sns.regplot(x='condition', y='sellingprice', data=car_sales_df, scatter=False, color='red', line_kws={'linestyle': '--'}, label='Trend Line')
    plt.xlabel('Condition')
    plt.ylabel('Selling Price')
    plt.title('Selling Price vs. Condition with Trend Line')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(full_img_output_path, "Selling_Price_vs_Condition_Trend_Line.png"))
    plt.show()
    print("Plot 'Selling_Price_vs_Condition_Trend_Line.png' generated and saved.")
    print("-" * 50)

    print("\nMarket Analysis complete.")