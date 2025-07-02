import pandas as pd
import numpy as np # Imported but not directly used in the current snippet's logic
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration & Data Loading ---
CLEANED_DATA_PATH = "Data/car_prices_clean.csv" # Path to the cleaned dataset
IMG_OUTPUT_DIR = "Img/Seller_Analysis_Img/" # Directory to save seller analysis plots

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

    # --- Seller Analysis: Average Selling Price ---
    print("Analyzing average selling price by seller...")
    average_price_by_seller = car_sales_df.groupby('seller')['sellingprice'].mean().sort_values(ascending=False)
    # Consider top N sellers for plotting if there are many unique sellers
    top_n_sellers_price = average_price_by_seller.head(20) # Adjust N as needed
    print("Top 10 Sellers by Average Selling Price:")
    print(top_n_sellers_price.head(10))

    # Plot average selling prices by seller
    plt.figure(figsize=(15, 8))
    sns.barplot(x=top_n_sellers_price.index, y=top_n_sellers_price.values, palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel('Seller')
    plt.ylabel('Average Selling Price')
    plt.title('Average Selling Price by Seller (Top 20)')
    plt.tight_layout()
    plt.savefig(os.path.join(full_img_output_path, "Average_Selling_Price_by_Seller.png"))
    plt.show()
    print("Plot 'Average_Selling_Price_by_Seller.png' generated and saved.")
    print("-" * 50)

    # --- Seller Analysis: Sales Volume ---
    print("Analyzing sales volume by seller...")
    sales_volume_by_seller = car_sales_df['seller'].value_counts()
    # Consider top N sellers for plotting if there are many unique sellers
    top_n_sellers_volume = sales_volume_by_seller.head(20) # Adjust N as needed
    print("Top 10 Sellers by Sales Volume:")
    print(top_n_sellers_volume.head(10))

    # Plot sales volumes by seller
    plt.figure(figsize=(15, 8))
    sns.barplot(x=top_n_sellers_volume.index, y=top_n_sellers_volume.values, palette='magma')
    plt.xticks(rotation=90)
    plt.xlabel('Seller')
    plt.ylabel('Sales Volume')
    plt.title('Sales Volume by Seller (Top 20)')
    plt.tight_layout()
    plt.savefig(os.path.join(full_img_output_path, "Sales_Volume_by_Seller.png"))
    plt.show()
    print("Plot 'Sales_Volume_by_Seller.png' generated and saved.")
    print("-" * 50)

    # --- Seller Analysis: Combined Performance ---
    print("Combining average selling price and sales volume for performance analysis...")
    # Combine relevant series into a DataFrame
    # Ensure indices align for a clean merge
    seller_performance_df = pd.DataFrame({
        'Average Selling Price': average_price_by_seller,
        'Sales Volume': sales_volume_by_seller
    }).dropna().sort_values(by='Sales Volume', ascending=False) # Drop NaNs if a seller has no sales/price info

    # Get top N sellers for the combined plot (e.g., top 30 by sales volume)
    top_sellers_for_plot = seller_performance_df.head(30)

    if not top_sellers_for_plot.empty:
        # Plot combined performance of sellers using a dual-axis bar and line plot
        fig, ax1 = plt.subplots(figsize=(18, 10)) # Increased figure size for readability

        # Plot Average Selling Price as bars
        color = 'tab:blue'
        ax1.set_xlabel('Seller', fontsize=12)
        ax1.set_ylabel('Average Selling Price', color=color, fontsize=12)
        ax1.bar(top_sellers_for_plot.index, top_sellers_for_plot['Average Selling Price'], color=color, alpha=0.7, label='Average Selling Price')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', rotation=90) # Rotate x-axis labels

        # Create a second y-axis for Sales Volume
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Sales Volume', color=color, fontsize=12)
        ax2.plot(top_sellers_for_plot.index, top_sellers_for_plot['Sales Volume'], color=color, marker='o', linestyle='-', linewidth=2, label='Sales Volume')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Combined Seller Performance: Average Selling Price vs. Sales Volume (Top 30)', fontsize=14)
        fig.tight_layout() # Adjust layout
        # Add legends for both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.savefig(os.path.join(full_img_output_path, "Combined_Seller_Performance.png"))
        plt.show()
        print("Plot 'Combined_Seller_Performance.png' generated and saved.")
    else:
        print("No seller performance data to plot after combining and filtering.")
    print("-" * 50)

    print("\nSeller Analysis complete.")