import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load and prepare data function
def prepare_data(df):
    df = df.copy()

    # Ensure date columns are parsed correctly if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df.drop('date', axis=1, inplace=True)

    # Convert categorical variables to dummy variables if needed
    df = pd.get_dummies(df, drop_first=True)

    # Ensure 'cnt' is in the data for target variable extraction
    y = df['cnt']
    X = df.drop(['cnt'], axis=1)

    return X, y

# App Title
st.title("Bike Sharing Demand Prediction")

# Upload Test Data
uploaded_file = st.file_uploader("Upload Test Data (CSV format)", type="csv")

if uploaded_file:
    # Read uploaded test data
    test_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Test Data Preview:")
    st.dataframe(test_data.head())

    # Load the prepared X, y for test data
    X_test, y_test = prepare_data(test_data)

    # Instantiate the model and retrain with sample data from provided file
    regmodel_new = LinearRegression().fit(X_test, y_test)

    # Make Predictions
    predictions = regmodel_new.predict(X_test)

    # Calculate Metrics
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Display Metrics
    st.write(f"### Model Performance on Test Data")
    st.write(f"- **R-squared:** {r2:.4f}")
    st.write(f"- **RMSE:** {rmse:.4f}")

    # Plot actual vs predicted
    st.write("### Actual vs Predicted Values")
    st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}))

else:
    st.write("Please upload a CSV file to evaluate the model performance.")
