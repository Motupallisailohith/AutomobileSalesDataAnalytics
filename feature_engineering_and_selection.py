import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error # Added for comprehensive evaluation (though not explicitly used for metric in original)
from sklearn.metrics import r2_score # Added for comprehensive evaluation (though not explicitly used for metric in original)
import matplotlib.pyplot as plt
import seaborn as sns
import os # Import os for path handling

# --- Configuration & Data Loading ---
CLEANED_DATA_PATH = "Data/car_prices_clean.csv"
FEATURE_IMPORTANCE_IMG_PATH = "Img/feature_importance_bar_plot.png" # Path for the output plot

# Ensure output directory exists
os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_IMG_PATH), exist_ok=True)

# Verify GPU availability for TensorFlow
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("-" * 50)

# Load the dataset
try:
    car_sales_df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Dataset loaded successfully from: {CLEANED_DATA_PATH}")
    print(car_sales_df.head())
    print("-" * 50)
except FileNotFoundError:
    print(f"Error: The file '{CLEANED_DATA_PATH}' was not found.")
    print("Please ensure the data cleaning script has been run and the file exists.")
    exit()

# --- Feature Engineering ---
print("Performing Feature Engineering...")
current_year = datetime.now().year
car_sales_df['car_age'] = current_year - car_sales_df['year']
# Handle potential division by zero if car_age is 0 (i.e., current year car)
car_sales_df['mileage_per_year'] = car_sales_df['odometer'] / car_sales_df['car_age'].replace(0, np.nan)
# Fill NaN from division by zero with 0 or a reasonable value if needed, or drop rows.
# For simplicity, if car_age was 0, mileage_per_year will be NaN. Let's fill with 0
car_sales_df['mileage_per_year'] = car_sales_df['mileage_per_year'].fillna(0)

# Convert 'saledate' to datetime if not already done by other scripts (good to be explicit)
car_sales_df['saledate'] = pd.to_datetime(car_sales_df['saledate'], errors='coerce')
# Drop rows where saledate conversion failed if any
car_sales_df.dropna(subset=['saledate'], inplace=True)

print("Feature Engineering complete: 'car_age' and 'mileage_per_year' added.")
print(car_sales_df[['year', 'car_age', 'odometer', 'mileage_per_year']].head())
print("-" * 50)

# --- Data Preprocessing for Model ---
# Define features and target variable.
# Exclude 'vin' and 'saledate' as they are not typically direct features for price prediction.
features_for_model = ['year', 'condition', 'odometer', 'mmr', 'car_age', 'mileage_per_year']
target_variable = 'sellingprice'

# Separate features (X) and target (y)
features_df = car_sales_df[features_for_model].copy()
target_series = car_sales_df[target_variable].copy()

# Standardize the numerical features
feature_scaler = StandardScaler()
features_df_scaled = pd.DataFrame(
    feature_scaler.fit_transform(features_df),
    columns=features_for_model
)
print("Numerical features standardized.")
print(features_df_scaled.head())
print("-" * 50)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features_df_scaled, target_series, test_size=0.2, random_state=42
)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")
print("-" * 50)

# --- TensorFlow Model Building ---
# Convert Pandas DataFrames/Series to TensorFlow datasets
def create_tf_dataset(features_dataframe, target_series, shuffle=True, batch_size=32):
    """
    Converts pandas DataFrame/Series to a TensorFlow tf.data.Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((dict(features_dataframe), target_series))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features_dataframe))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE) # Use prefetch for performance
    return dataset

batch_size = 32
train_tf_dataset = create_tf_dataset(X_train, y_train, batch_size=batch_size)
val_tf_dataset = create_tf_dataset(X_test, y_test, shuffle=False, batch_size=batch_size) # Use X_test, y_test for validation

# Define TensorFlow feature columns for numeric input
tf_feature_columns = [tf.feature_column.numeric_column(colname) for colname in features_for_model]
feature_input_layer = tf.keras.layers.DenseFeatures(tf_feature_columns)

# Build the Sequential Model (simpler for this case, or Functional API as in original)
# Using Functional API as in original, more flexible for complex models
input_tensors = {colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32) for colname in features_for_model}
processed_inputs = feature_input_layer(input_tensors)
x = tf.keras.layers.Dense(128, activation='relu')(processed_inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
output_tensor = tf.keras.layers.Dense(1)(x) # Output is a single price value

price_prediction_model = tf.keras.Model(inputs=input_tensors, outputs=output_tensor)

# Compile the model
price_prediction_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae']) # Added MAE for another metric
print("TensorFlow model built and compiled.")
price_prediction_model.summary()
print("-" * 50)

# --- Model Training ---
print("Training the TensorFlow model...")
history = price_prediction_model.fit(
    train_tf_dataset,
    validation_data=val_tf_dataset,
    epochs=5 # Increased epochs for potentially better training (original was 1)
)
print("Model training complete.")
print("-" * 50)

# --- Feature Importance Estimation ---
# Note: For simple Dense layers on scaled numerical data, direct weight inspection can give a
# rough idea, but permutation importance or SHAP values are more robust methods.
# This approach assumes the DenseFeatures layer is at index 1 and the first Dense layer is after that.
# It's fragile if model architecture changes.

def estimate_feature_importance_from_dense_weights(model_instance, feature_names):
    """
    Estimates feature importance from the weights of the first Dense layer
    after the DenseFeatures layer in a Sequential or simple Functional API model.
    Assumes numerical inputs are standardized.
    """
    for layer in model_instance.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            # This is the first dense layer after inputs/DenseFeatures
            if layer.input_shape[1] == len(feature_names): # Check if input size matches feature count
                dense_layer_weights = layer.get_weights()[0] # Weights connecting inputs to first dense layer
                # Sum absolute weights connected to each input feature node
                # Assumes 1:1 mapping from DenseFeatures output to first Dense layer input,
                # which is typical if only numeric_columns are used in DenseFeatures.
                importance_scores = np.sum(np.abs(dense_layer_weights), axis=1)
                break
    else:
        print("Warning: Could not find suitable Dense layer for feature importance estimation.")
        return {}

    feature_importances_dict = dict(zip(feature_names, importance_scores))
    return feature_importances_dict

feature_importances = estimate_feature_importance_from_dense_weights(price_prediction_model, features_for_model)


# Visualize feature importance
if feature_importances: # Only visualize if importances were successfully estimated
    feature_importance_df = pd.DataFrame({
        'Feature': list(feature_importances.keys()),
        'Importance': list(feature_importances.values())
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance (Estimated from Model Weights)')
    plt.xlabel('Importance Score (Sum of Absolute Weights)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_IMG_PATH) # Save the plot
    plt.show()

    print("\nEstimated Feature Importances:")
    print(feature_importance_df)
    print(f"Feature importance plot saved to: {FEATURE_IMPORTANCE_IMG_PATH}")
else:
    print("Feature importance visualization skipped.")

print("\nFeature Engineering and Selection script finished.")