import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration & Data Loading ---
CLEANED_DATA_PATH = "Data/car_prices_clean.csv" # Path to the cleaned dataset
IMG_OUTPUT_DIR = "Img/EDA_Img/" # Directory to save EDA plots

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

    # --- Initial Data Overview ---
    print("Checking for missing values:")
    missing_values = car_sales_df.isnull().sum()
    print(missing_values[missing_values > 0]) # Print only columns with missing values
    print("-" * 50)

    # Visualize missing values (Heatmap)
    plt.figure(figsize=(10, 7)) # Explicitly create a figure
    sns.heatmap(car_sales_df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.savefig(os.path.join(full_img_output_path, "missing_values_heatmap.png")) # Save plot
    plt.show()
    print("Missing values heatmap generated and saved.")
    print("-" * 50)

    # --- Numerical Data Analysis ---
    numerical_columns = car_sales_df.select_dtypes(include=[np.number]).columns

    # Plot distributions of numeric columns (Histograms)
    car_sales_df[numerical_columns].hist(bins=15, figsize=(15, 10))
    plt.suptitle("Distributions of Numeric Columns")
    plt.savefig(os.path.join(full_img_output_path, "distributions_of_numeric_columns.png")) # Save plot
    plt.show()
    print("Distributions of numeric columns generated and saved.")
    print("-" * 50)

    # Pairplot to see relationships between numerical variables
    # Note: Pairplots can be resource-intensive for many columns/rows
    if len(numerical_columns) <= 6: # Limit pairplot for performance
        sns.pairplot(car_sales_df[numerical_columns])
        plt.suptitle("Pairplot of Numeric Columns", y=1.02) # Adjust suptitle position
        plt.savefig(os.path.join(full_img_output_path, "pairplot_numeric_columns.png")) # Save plot
        plt.show()
        print("Pairplot of numeric columns generated and saved.")
    else:
        print("Skipping pairplot due to large number of numerical columns. Consider selecting a subset.")
    print("-" * 50)


    # Correlation matrix
    numeric_df_for_corr = car_sales_df[numerical_columns]
    correlation_matrix = numeric_df_for_corr.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f") # Format annotations
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(full_img_output_path, "correlation_matrix.png")) # Save plot
    plt.show()
    print("Correlation matrix generated and saved.")
    print("-" * 50)

    # --- Categorical Data Analysis ---
    categorical_columns = ['make', 'model', 'trim', 'body', 'transmission', 'state']

    print("Frequency distribution for categorical columns:")
    for col in categorical_columns:
        print(f"\n--- {col} ---")
        value_counts = car_sales_df[col].value_counts()
        print(value_counts)

        # Bar plots for categorical columns
        plt.figure(figsize=(12, min(8, len(value_counts) * 0.4))) # Dynamic figure height
        sns.countplot(y=car_sales_df[col], order=value_counts.index, palette='viridis')
        plt.title(f'Frequency Distribution of {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        # Clean column name for filename
        clean_col_name = col.replace(" ", "_").lower()
        plt.savefig(os.path.join(full_img_output_path, f"frequency_distribution_of_{clean_col_name}.png")) # Save plot
        plt.show()
        print(f"Frequency distribution plot for '{col}' generated and saved.")
    print("-" * 50)

    # --- Cross-Tabulation & Pivot Tables ---
    print("\nCross tabulation of make and model:")
    cross_tab_make_model = pd.crosstab(car_sales_df['make'], car_sales_df['model'])
    print(cross_tab_make_model.head()) # Print head for potentially large tables
    print("-" * 50)

    print("\nPivot table of average selling price by make and model:")
    pivot_table_avg_price = car_sales_df.pivot_table(values='sellingprice', index='make', columns='model', aggfunc='mean')
    print(pivot_table_avg_price.head()) # Print head for potentially large tables
    print("-" * 50)

    # --- Grouped Analysis & Visualizations ---
    print("\nAverage selling price grouped by make:")
    make_avg_selling_price = car_sales_df.groupby('make')['sellingprice'].mean().sort_values(ascending=False)
    print(make_avg_selling_price.head(10)) # Show top 10
    print("-" * 50)

    # Box plot of selling price by make
    plt.figure(figsize=(15, 7))
    sns.boxplot(x='make', y='sellingprice', data=car_sales_df, palette='muted')
    plt.xticks(rotation=90)
    plt.title('Box Plot of Selling Price by Make')
    plt.xlabel('Car Make')
    plt.ylabel('Selling Price')
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig(os.path.join(full_img_output_path, "box_plot_of_selling_price_by_make.png")) # Save plot
    plt.show()
    print("Box plot of selling price by make generated and saved.")
    print("-" * 50)

    print("\nExploratory Data Analysis (EDA) complete.")