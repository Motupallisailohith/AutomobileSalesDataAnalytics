import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
from datetime import datetime # Needed for feature engineering consistency

# --- Configuration & Paths ---
MODEL_LOAD_DIR = "Price_Prediction_Model/scaler/"
MODEL_FILENAME = "car_price_prediction_model.h5"
FEATURE_SCALER_FILENAME = "feature_scaler.pkl"
TARGET_SCALER_FILENAME = "target_scaler.pkl"

# --- Main Model Usage Workflow ---
if __name__ == "__main__":
    print("--- Starting Price Prediction Model Usage ---")

    # --- 1. Load Trained Model and Scalers ---
    try:
        model_path = os.path.join(MODEL_LOAD_DIR, MODEL_FILENAME)
        loaded_model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")

        feature_scaler_path = os.path.join(MODEL_LOAD_DIR, FEATURE_SCALER_FILENAME)
        target_scaler_path = os.path.join(MODEL_LOAD_DIR, TARGET_SCALER_FILENAME)

        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        print("Feature and target scalers loaded successfully.")
        print("-" * 50)

    except (OSError, FileNotFoundError) as e:
        print(f"Error loading model or scalers: {e}")
        print("Please ensure 'price_prediction_model_traning.py' has been run to train and save them.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during model/scaler loading: {e}")
        exit()

    # --- 2. Define Features ---
    # These must be the EXACT same features used during training
    features_for_prediction = ['year', 'condition', 'odometer', 'mmr', 'car_age', 'mileage_per_year']

    # --- 3. Example New Data for Prediction ---
    # IMPORTANT: This new data MUST have the same columns and be preprocessed
    # in the EXACT same way as the training data.
    # The 'car_age' and 'mileage_per_year' would typically be engineered here for new input.
    # Replace with your actual new car data for prediction
    example_new_cars_data = pd.DataFrame({
        'year': [2018, 2020, 2017, 2022],
        'make': ['Toyota', 'Honda', 'Nissan', 'Ford'], # Categorical features
        'model': ['Camry', 'Civic', 'Altima', 'Mustang'], # will be ignored by this model, but shown for context
        'condition': [4, 5, 3, 4],
        'odometer': [50000, 25000, 70000, 10000],
        'mmr': [18000, 22000, 15000, 35000],
        'sellingprice': [np.nan, np.nan, np.nan, np.nan] # Target is unknown for new data
    })
    print("Example new car data for prediction:")
    print(example_new_cars_data)
    print("-" * 50)

    # --- 4. Preprocess New Data (Crucial for consistency) ---
    print("Preprocessing new data for prediction...")

    # A. Feature Engineering (must match training script's engineering)
    current_year = datetime.now().year # Ensure current year is consistent
    example_new_cars_data['car_age'] = current_year - example_new_cars_data['year']
    example_new_cars_data['mileage_per_year'] = example_new_cars_data['odometer'] / example_new_cars_data['car_age'].replace(0, np.nan)
    example_new_cars_data['mileage_per_year'].fillna(0, inplace=True) # Fill NaN from division by zero with 0

    # B. Select only the features the model was trained on
    new_data_for_prediction_df = example_new_cars_data[features_for_prediction].copy()

    # C. Scale numerical features using the *loaded* feature_scaler
    scaled_new_data = pd.DataFrame(
        feature_scaler.transform(new_data_for_prediction_df), # Use .transform, not .fit_transform
        columns=features_for_prediction
    )
    print("New data preprocessed and scaled.")
    print(scaled_new_data.head())
    print("-" * 50)


    # --- 5. Prepare Data for TensorFlow Model ---
    # Convert scaled_new_data DataFrame to dictionary of NumPy arrays, as expected by model's inputs
    input_for_model = {colname: np.array(scaled_new_data[colname]) for colname in features_for_prediction}

    # No need for tf.data.Dataset.from_tensor_slices((dict(dataframe))) if making single predictions
    # Model.predict can take dict of numpy arrays directly if inputs are set up with Input layers per feature.

    # --- 6. Make Predictions ---
    print("Making predictions...")
    predictions_scaled = loaded_model.predict(input_for_model)

    # --- 7. Inverse Transform Predictions to Original Scale ---
    predicted_prices_rescaled = target_scaler.inverse_transform(predictions_scaled).flatten()

    # --- 8. Display Results ---
    example_new_cars_data['Predicted_SellingPrice'] = predicted_prices_rescaled
    print("\n--- Prediction Results ---")
    print(example_new_cars_data[['make', 'model', 'year', 'odometer', 'Predicted_SellingPrice']])
    print("-" * 50)

    print("\n--- Price Prediction Model Usage script finished ---")