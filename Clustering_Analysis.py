import pandas as pd
import matplotlib.pyplot as plt # Use pyplot directly, no need for pylab
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os # Import os for path handling, though it's not strictly used in the current snippet's file path

# --- Configuration & Data Loading ---
# Define the path to your cleaned dataset
CLEANED_DATA_PATH = "Data/car_prices_clean.csv" # Using forward slashes for cross-OS compatibility

try:
    car_data_df = pd.read_csv(CLEANED_DATA_PATH)
    print("Dataset loaded successfully.")
    print(car_data_df.head())
    print("\nDataset Description:")
    print(car_data_df.describe())
except FileNotFoundError:
    print(f"Error: The file '{CLEANED_DATA_PATH}' was not found. Please ensure the data cleaning script has been run.")
    exit() # Exit if the data isn't found, as the script can't proceed

# --- Feature Selection and Scaling ---
# Define numerical features relevant for clustering
clustering_features = ["year", "condition", "odometer", "mmr", "sellingprice"]

# Check if all clustering features exist in the DataFrame
missing_features = [f for f in clustering_features if f not in car_data_df.columns]
if missing_features:
    print(f"Error: Missing clustering features in the dataset: {missing_features}")
    exit()

# Initialize the scaler
feature_scaler = StandardScaler()

# Apply scaling and create new transformed columns
# Using a more explicit naming convention for transformed features
transformed_features_df = pd.DataFrame(
    feature_scaler.fit_transform(car_data_df[clustering_features]),
    columns=[f"{col}_scaled" for col in clustering_features]
)

# You might want to add these scaled features back to your original DataFrame
# car_data_df = pd.concat([car_data_df, transformed_features_df], axis=1)
# Or just work with the transformed_features_df for clustering

print("\nScaled Features (first 5 rows):")
print(transformed_features_df.head())

# --- K-Means Optimization (Elbow Method) ---
def plot_kmeans_inertia_elbow(data_for_clustering, max_k_clusters=10, save_path="Img/Cluster_Analysis_Img/optimise_k_means.png"):
    """
    Performs K-Means clustering for a range of K values and plots the inertia
    to help determine the optimal number of clusters using the Elbow Method.

    Args:
        data_for_clustering (pd.DataFrame): DataFrame containing numerical data for clustering.
        max_k_clusters (int): Maximum number of clusters (K) to test.
        save_path (str): File path to save the generated plot.
    """
    k_values = []
    cluster_inertias = []

    # Ensure the directory for saving exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for k in range(1, max_k_clusters + 1): # Iterate up to max_k_clusters inclusive
        kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42) # Added random_state for reproducibility
        kmeans_model.fit(data_for_clustering)

        k_values.append(k)
        cluster_inertias.append(kmeans_model.inertia_)

    plt.figure(figsize=(10, 6)) # Explicitly create a figure
    plt.plot(k_values, cluster_inertias, 'o-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)')
    plt.title('Elbow Method For Optimal K')
    plt.grid(True)
    plt.xticks(k_values) # Ensure x-axis ticks show all k values tested
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig(save_path)
    plt.show()

# Call the function to plot the elbow method
plot_kmeans_inertia_elbow(transformed_features_df, max_k_clusters=10)

# --- Final K-Means Clustering ---
# Based on the elbow plot, choose an optimal K value (e.g., 3, 4, or 5).
# For demonstration, let's assume optimal_k_value = 4 (You'd pick this after running the elbow method)
optimal_k_value = 4 # You would determine this from the plot

print(f"\nPerforming K-Means clustering with optimal K = {optimal_k_value}...")
final_kmeans_model = KMeans(n_clusters=optimal_k_value, init='k-means++', n_init=10, random_state=42)
car_data_df['cluster'] = final_kmeans_model.fit_predict(transformed_features_df)

print(f"\nClustering complete. First 5 rows with cluster assignments (K={optimal_k_value}):")
print(car_data_df[['sellingprice', 'cluster']].head())

# --- Analyze Cluster Characteristics ---
print(f"\nCluster Sizes for K={optimal_k_value}:")
print(car_data_df['cluster'].value_counts().sort_index())

print(f"\nMean characteristics per cluster for K={optimal_k_value}:")
# Include both original and scaled features for analysis if desired
# For simplicity, let's analyze original features
cluster_centers_original_scale = pd.DataFrame(
    feature_scaler.inverse_transform(final_kmeans_model.cluster_centers_),
    columns=clustering_features
)
cluster_centers_original_scale['cluster'] = range(optimal_k_value)
cluster_centers_original_scale.set_index('cluster', inplace=True)
print(cluster_centers_original_scale)

# --- Optional: Further visualization of clusters ---
# You might want to visualize clusters in 2D or 3D using PCA/t-SNE if high-dimensional
# For example, a scatter plot of two key features colored by cluster:
# plt.figure(figsize=(10, 7))
# sns.scatterplot(x='odometer_scaled', y='sellingprice_scaled', hue='cluster', data=car_data_df, palette='viridis', legend='full')
# plt.title(f'Clusters of Cars by Odometer and Selling Price (K={optimal_k_value})')
# plt.show()

print("\nClustering analysis script finished.")