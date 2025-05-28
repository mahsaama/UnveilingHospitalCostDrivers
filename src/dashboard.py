import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib



def load_and_preprocess_data():
    # Load data
    data = pd.read_csv('data/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2022_20250423.csv')

    # Remove commas and convert to numeric
    for col in ["Length of Stay", "Birth Weight", "Total Charges", "Total Costs"]:
        data[col] = pd.to_numeric(data[col].str.replace(',', '', regex=False), errors='coerce')

    # Handle missing numeric data
    for col in data.columns:
        if data[col].dtype != "object":
            data[col] = data[col].fillna(data[col].median())

    # Identify categorical columns 
    categorical_columns = []
    for col in data.columns:
        if data[col].dtype == "object":
            categorical_columns.append(col)

    # removing special characters from column names
    new_columns = []
    for col in data.columns:
        if "[" in col or "]" in col:
            new_columns.append(col.replace('[', '').replace("]", ""))
        elif "<" in col:
            new_columns.append(col.replace("<", "less than"))
        else:
            new_columns.append(col)
    data.columns = new_columns

    # Encode categorical columns
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    return data, data_encoded

def load_xgb_model():
    # model = joblib.load("models/xgb_model.pkl")
    model = joblib.load("model.json")
    return model

def load_lgb_model():
    model = joblib.load("models/lgb_model.pkl")
    return model

def predict_data(model, input):
    y_pred = model.predict(input)
    return y_pred


if __name__ == "__main__":
    data, data_encoded = load_and_preprocess_data()
    
    # Section 1: Input patient info
    st.sidebar.header("Patient Info")

    # Dictionary to collect new values
    new_patient = {}

    # Generate input widgets dynamically
    for col in data.columns:
        dtype = data[col].dtype
        if col != "Total Costs":
            if pd.api.types.is_numeric_dtype(data[col]):
                min_val = int(data[col].min())
                max_val = int(data[col].max())
                if min_val == max_val:
                    max_val += 1
                default_val = int(data[col].mean())
                new_patient[col] = st.sidebar.slider(f"{col}", min_val, max_val, default_val)

            elif pd.api.types.is_float_dtype(dtype):
                default_val = float(data[col].mean())
                new_patient[col] = st.number_input(f"{col}", value=default_val)

            elif pd.api.types.is_bool_dtype(data[col]):
                new_patient[col] = st.sidebar.checkbox(f"{col}", value=bool(data[col].iloc[0]))

            elif pd.api.types.is_categorical_dtype(data[col]) or data[col].nunique() < 10:
                options = data[col].unique().tolist()
                new_patient[col] = st.sidebar.selectbox(f"{col}", options)

            else:
                new_patient[col] = st.sidebar.text_input(f"{col}", value=str(data[col].iloc[0]))


    # Section 2: Predict cost
    xgb_model = load_xgb_model()

    patient_data = pd.DataFrame([new_patient])
    for col in patient_data.columns:
        if data[col].dtype != "object":
            patient_data[col] = pd.to_numeric(patient_data[col], errors='coerce')
        else:
            patient_data[col] = patient_data[col].astype("object")

    print(patient_data_encoded.columns)

    # if st.sidebar.button("Predict Cost"):
    #     # print(patient_data)
    #     pass

    # st.subheader("Predicted Inpatient Cost")
    # cost = predict_data(model, patient_data_encoded)
    # st.metric(label="Estimated Cost", value=f"${cost:,.2f}")


