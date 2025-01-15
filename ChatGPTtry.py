import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def preprocess_data(data):
    st.write("Initial Data Columns and Types:")
    st.write(data.dtypes)

    # Drop irrelevant or non-numeric columns if they exist
    if 'dteday' in data.columns:
        data = data.drop(['dteday'], axis=1)
        st.write("Dropped column 'dteday'.")
    
    # Convert non-numeric columns to numeric using encoding or drop them
    for col in data.select_dtypes(include=['object']).columns:
        st.write(f"Processing column: {col}")
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            st.write(f"Converted column '{col}' to numeric.")
        except Exception as e:
            data = data.drop(col, axis=1)
            st.write(f"Dropped column '{col}' due to error: {e}")

    # Drop rows with NaN values after conversion
    initial_row_count = data.shape[0]
    data = data.dropna()
    final_row_count = data.shape[0]
    st.write(f"Dropped {initial_row_count - final_row_count} rows containing NaN values.")

    # Check if the dataset is empty after preprocessing
    if data.empty:
        raise ValueError("The dataset is empty after preprocessing. Please check the data.")

    # Show skewness of numeric columns
    st.write("Skewness of Numeric Columns:")
    st.write(data.skew())

    return data

def train_model(train_data):
    # Preprocess training data
    train_data = preprocess_data(train_data)

    # Splitting the data into independent (X) and dependent (y) variables
    y = train_data['cnt']
    X = train_data.drop(['cnt', 'registered'], axis=1, errors='ignore')

    # Ensure there is enough data to train the model
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Training data has insufficient samples or features after preprocessing.")

    # Train a linear regression model
    model = LinearRegression().fit(X, y)

    # Predictions and performance metrics for training data
    y_pred = model.predict(X)
    r_squared = model.score(X, y)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return model, r_squared, rmse

def evaluate_model(model, test_data):
    # Preprocess test data
    test_data = preprocess_data(test_data)

    # Splitting the test data into independent (X_test) and dependent (y_test) variables
    y_test = test_data['cnt']
    X_test = test_data.drop(['cnt', 'registered'], axis=1, errors='ignore')

    # Ensure there is enough data to evaluate the model
    if X_test.shape[0] == 0 or X_test.shape[1] == 0:
        raise ValueError("Test data has insufficient samples or features after preprocessing.")

    # Predictions and performance metrics for test data
    y_pred_test = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return test_rmse

# Streamlit app
st.title("Bike Sharing Demand Prediction")

# Upload training data
st.header("Step 1: Upload Training Data")
train_file = st.file_uploader("Upload training dataset (CSV file):", type="csv")

if train_file:
    train_data = pd.read_csv(train_file)
    st.write("Training Data Preview:")
    st.write(train_data.head())

    try:
        # Train the model
        model, r_squared, train_rmse = train_model(train_data)

        # Display training results
        st.subheader("Training Results:")
        st.write(f"R-squared: {r_squared:.4f}")
        st.write(f"RMSE (Training): {train_rmse:.4f}")
    except ValueError as e:
        st.error(f"Error in training data: {e}")

# Upload test data
st.header("Step 2: Upload Test Data")
test_file = st.file_uploader("Upload test dataset (CSV file):", type="csv")

if test_file:
    test_data = pd.read_csv(test_file)
    st.write("Test Data Preview:")
    st.write(test_data.head())

    try:
        # Evaluate the model on test data
        test_rmse = evaluate_model(model, test_data)

        # Display test results
        st.subheader("Test Results:")
        st.write(f"RMSE (Test): {test_rmse:.4f}")
    except ValueError as e:
        st.error(f"Error in test data: {e}")