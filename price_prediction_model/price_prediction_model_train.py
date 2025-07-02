import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# --- Configuration & Paths ---
CLEANED_DATA_PATH = "Data/car_prices_clean.csv"
MODEL_SAVE_DIR = "Price_Prediction_Model/scaler/" # Directory for saving model and scalers
MODEL_FILENAME = "car_price_prediction_model.h5"
FEATURE_SCALER_FILENAME = "feature_scaler.pkl"
TARGET_SCALER_FILENAME = "target_scaler.pkl"
TRAINING_PLOT_FILENAME = "Img/price_prediction_training_metrics.png" # Plot of training/validation loss/metrics
PREDICTION_SCATTER_PLOT_FILENAME = "Img/price_prediction_actual_vs_predicted.png" # Scatter plot for final evaluation

# Ensure model and image output directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TRAINING_PLOT_FILENAME), exist_ok=True) # Ensure Img/ exists

# --- Main Training Workflow ---
if __name__ == "__main__":
    print("--- Starting Price Prediction Model Training ---")

    # --- 1. System Setup Check ---
    # Verify GPU availability for TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"Num GPUs Available: {len(gpus)}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPUs available, training on CPU.")
    print("-" * 50)

    # --- 2. Data Loading ---
    try:
        car_sales_df = pd.read_csv(CLEANED_DATA_PATH)
        print(f"Dataset loaded successfully from: {CLEANED_DATA_PATH}")
        print("Dataset Head:\n", car_sales_df.head())
        print("-" * 50)
    except FileNotFoundError:
        print(f"Error: Cleaned data not found at '{CLEANED_DATA_PATH}'. Please run data cleaning script first.")
        exit()

    # --- 3. Feature Engineering ---
    print("Performing Feature Engineering...")
    current_year = datetime.now().year
    car_sales_df['car_age'] = current_year - car_sales_df['year']
    # Handle potential division by zero for new cars (car_age=0)
    car_sales_df['mileage_per_year'] = car_sales_df['odometer'] / car_sales_df['car_age'].replace(0, np.nan)
    car_sales_df['mileage_per_year'].fillna(0, inplace=True) # Fill NaN from division by zero with 0

    # Ensure 'saledate' is datetime for consistency, drop rows if conversion fails
    if 'saledate' in car_sales_df.columns:
        car_sales_df['saledate'] = pd.to_datetime(car_sales_df['saledate'], errors='coerce')
        car_sales_df.dropna(subset=['saledate'], inplace=True)

    print("Feature Engineering complete: 'car_age' and 'mileage_per_year' added.")
    print("-" * 50)

    # --- 4. Define Features and Target ---
    # These are the numerical features the model will use
    features_for_model = ['year', 'condition', 'odometer', 'mmr', 'car_age', 'mileage_per_year']
    target_variable = 'sellingprice'

    # Ensure all required features are in the DataFrame
    missing_cols = [col for col in features_for_model if col not in car_sales_df.columns]
    if missing_cols:
        print(f"Error: Missing required features in the dataset for model training: {missing_cols}")
        exit()

    model_features_df = car_sales_df[features_for_model].copy()
    model_target_series = car_sales_df[target_variable].copy()

    # --- 5. Data Scaling ---
    print("Scaling features and target variable...")
    feature_scaler = StandardScaler()
    scaled_features_df = pd.DataFrame(
        feature_scaler.fit_transform(model_features_df),
        columns=features_for_model
    )

    target_scaler = StandardScaler()
    # Reshape target_series for scaling (it needs a 2D array)
    scaled_target_series = target_scaler.fit_transform(model_target_series.values.reshape(-1, 1)).flatten()

    # Save scalers for later use in prediction
    joblib.dump(feature_scaler, os.path.join(MODEL_SAVE_DIR, FEATURE_SCALER_FILENAME))
    joblib.dump(target_scaler, os.path.join(MODEL_SAVE_DIR, TARGET_SCALER_FILENAME))
    print(f"Feature scaler saved to: {os.path.join(MODEL_SAVE_DIR, FEATURE_SCALER_FILENAME)}")
    print(f"Target scaler saved to: {os.path.join(MODEL_SAVE_DIR, TARGET_SCALER_FILENAME)}")
    print("Features and target scaled and scalers saved.")
    print("-" * 50)

    # --- 6. Data Splitting ---
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
        scaled_features_df, scaled_target_series, test_size=0.2, random_state=42
    )
    print(f"Data split into training ({len(X_train_scaled)} samples) and testing ({len(X_test_scaled)} samples).")
    print("-" * 50)

    # --- 7. Create TensorFlow Datasets ---
    def create_tf_dataset(features_df, target_series, shuffle=True, batch_size=32):
        """Converts pandas DataFrame/Series to a TensorFlow tf.data.Dataset."""
        dataset = tf.data.Dataset.from_tensor_slices((dict(features_df), target_series))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(features_df))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    BATCH_SIZE = 32
    train_tf_dataset = create_tf_dataset(X_train_scaled, y_train_scaled, batch_size=BATCH_SIZE)
    val_tf_dataset = create_tf_dataset(X_test_scaled, y_test_scaled, shuffle=False, batch_size=BATCH_SIZE)
    print("TensorFlow datasets created.")
    print("-" * 50)

    # --- 8. Build TensorFlow Model ---
    # Define input layers for each feature
    input_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
        for colname in features_for_model
    }

    # All inputs are numeric, no need for tf.feature_column.numeric_column within DenseFeatures
    # If there were categorical features, they'd be handled here using embedding/one-hot.
    # For now, it's just passing raw (scaled) numeric inputs.
    # If DenseFeatures is used, it should be passed the original columns and it will handle numeric_column internally
    # However, since X_train_scaled is already a numerical dataframe, we can directly feed it to the model.
    # Let's rebuild the model to directly accept the already-scaled numerical inputs.

    # Concatenate inputs to a single tensor
    concatenated_inputs = tf.keras.layers.concatenate(list(input_layers.values()))

    # Define hidden layers
    x = tf.keras.layers.Dense(128, activation='relu')(concatenated_inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(1)(x) # Single output for regression

    price_prediction_model = tf.keras.Model(inputs=input_layers, outputs=output_layer)

    # Compile the model
    price_prediction_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
    print("TensorFlow model built and compiled.")
    price_prediction_model.summary()
    print("-" * 50)

    # --- 9. Train the Model ---
    print("Training the TensorFlow model...")
    # Using ModelCheckpoint and EarlyStopping for better training management
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=os.path.join(MODEL_SAVE_DIR, 'best_model.h5'),
    #     save_best_only=True, monitor='val_loss', mode='min')
    # early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

    history = price_prediction_model.fit(
        train_tf_dataset,
        validation_data=val_tf_dataset,
        epochs=50, # Increased epochs for better training
        # callbacks=[model_checkpoint_callback, early_stopping_callback] # Uncomment to use callbacks
        verbose=1 # Show training progress
    )
    print("Model training complete.")
    print("-" * 50)

    # --- 10. Evaluate the Model ---
    print("Evaluating the model on the test set...")
    y_pred_scaled = price_prediction_model.predict(val_tf_dataset)

    # Inverse transform predictions and actual values to original scale
    y_pred_rescaled = target_scaler.inverse_transform(y_pred_scaled).flatten()
    y_test_rescaled = target_scaler.inverse_transform(y_test_scaled.values.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²: {r2:.2f}")
    print("-" * 50)

    # --- 11. Visualize Training History ---
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(TRAINING_PLOT_FILENAME)
    plt.show()
    print(f"Training loss plot saved to: {TRAINING_PLOT_FILENAME}")
    print("-" * 50)

    # --- 12. Visualize Predictions vs. Actual (Sample) ---
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=y_test_rescaled, y=y_pred_rescaled, alpha=0.6)
    plt.plot([min(y_test_rescaled), max(y_test_rescaled)], [min(y_test_rescaled), max(y_test_rescaled)], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Selling Price')
    plt.ylabel('Predicted Selling Price')
    plt.title('Actual vs. Predicted Selling Prices (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.savefig(PREDICTION_SCATTER_PLOT_FILENAME)
    plt.show()
    print(f"Actual vs. Predicted prices scatter plot saved to: {PREDICTION_SCATTER_PLOT_FILENAME}")
    print("-" * 50)


    # --- 13. Save the Trained Model ---
    model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
    price_prediction_model.save(model_save_path)
    print(f"Trained model saved to: {model_save_path}")
    print("-" * 50)

    # --- 14. Feature Importance Estimation (Simplified for this model type) ---
    # For a simple dense network *after* input handling, the weights of the first hidden layer
    # can give a heuristic for importance if inputs were standardized.
    # More rigorous methods like Permutation Importance or SHAP are generally preferred.

    print("Estimating Feature Importance from Model Weights (Heuristic)...")
    try:
        # Access the weights of the first dense layer that directly receives features
        # This assumes concatenated_inputs feeds directly into the first Dense layer.
        first_dense_layer_weights = price_prediction_model.layers[1].get_weights()[0] # layers[0] is input layer, layers[1] is concatenate, layers[2] is first Dense
        
        # Sum absolute weights connected to each input feature node
        # Sum across the output neurons of the first dense layer
        importance_scores = np.sum(np.abs(first_dense_layer_weights), axis=1)
        
        feature_importances_dict = dict(zip(features_for_model, importance_scores))
        
        feature_importance_df = pd.DataFrame({
            'Feature': list(feature_importances_dict.keys()),
            'Importance': list(feature_importances_dict.values())
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title('Feature Importance (Heuristic from First Dense Layer Weights)')
        plt.xlabel('Importance Score (Sum of Absolute Weights)')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig("Img/feature_importance_bar_plot.png") # Save the plot
        plt.show()

        print("\nEstimated Feature Importances:")
        print(feature_importance_df)
        print("Feature importance plot saved to Img/feature_importance_bar_plot.png")

    except Exception as e:
        print(f"Could not estimate feature importance from model weights: {e}")
        print("This method can be fragile; consider using more robust techniques like Permutation Importance or SHAP.")
    print("-" * 50)


    print("\n--- Price Prediction Model Training script finished ---")