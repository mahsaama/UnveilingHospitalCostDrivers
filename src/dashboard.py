import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import joblib
import json


def load_data():
    # Load data
    data = pd.read_csv(
        "data/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2022_20250423.csv"
    )
    with open("data/data_info.json", "r") as f:
        data_info = json.load(f)

    # Handle missing data and changing column type
    for col_name, info in data_info.items():
        # print(col_name)
        if info["type"] == "int64":
            if col_name == "Zip Code - 3 digits":
                data[col_name] = data[col_name].replace("OOS", 000)
                data[col_name] = data[col_name].fillna(000)
            if col_name == "Length of Stay":
                data[col_name] = data[col_name].replace("120 +", 121)
            if col_name == "Birth Weight":
                data[col_name] = data[col_name].replace("UNKN", -1)
                data[col_name] = data[col_name].fillna(-1)
            else:
                data[col_name] = data[col_name].fillna(-1)

            data[col_name] = pd.to_numeric(data[col_name]).astype("int")

        elif info["type"] == "str":
            data[col_name] = data[col_name].fillna("Unknown")
            data[col_name] = data[col_name].astype(str)

        elif info["type"] == "float64":
            data[col_name] = data[col_name].str.replace(",", "")
            data[col_name] = pd.to_numeric(data[col_name]).astype("float")

    # Identify categorical columns and convert them to categories
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].astype("category")

    return data, data_info


def load_xgb_model():
    model = joblib.load("models/xgb_model_v2.pkl")
    return model


def load_lgb_model():
    model = joblib.load("models/lgb_model_v2.pkl")
    return model


def load_shap_explainer():
    explainer = joblib.load("models/shap_explainer.pkl")
    return explainer


def predict_data(model, input):
    y_pred = model.predict(input)
    return y_pred


if __name__ == "__main__":
    data, data_info = load_data()

    # Section 1: Input patient info
    st.sidebar.header("Patient Info")

    # Dictionary to collect new values
    sample_patient_dict = {}

    # Generate input widgets dynamically
    for col in data.columns:
        if col != "Total Costs":
            if "options" in data_info[col]:
                sample_patient_dict[col] = st.sidebar.selectbox(
                    f"{col}",
                    data_info[col]["options"],
                    help=data_info[col]["definition"],
                )

            else:

                if data_info[col]["type"] == "int64" or data_info[col]["type"] == "float64":
                    sample_patient_dict[col] = st.sidebar.number_input(
                        f"{col}",
                        help=data_info[col]["definition"],
                        format="%.0f"
                    )

                elif data_info[col]["type"] == "str":
                    sample_patient_dict[col] = st.sidebar.text_input(
                        f"{col}",
                        value=(
                            data_info[col]["default"]
                            if "default" in data_info[col]
                            else "Unknown"
                        ),
                        placeholder=(
                            data_info[col]["default"]
                            if "default" in data_info[col]
                            else "Unknown"
                        ),
                        help=data_info[col]["definition"],
                    )

    sample_patient_df = pd.DataFrame(sample_patient_dict, index=[0])

    for col in data.select_dtypes(["category"]).columns:
        sample_patient_df[col] = pd.Categorical(
            sample_patient_df[col], categories=data[col].cat.categories
        )

    # Section 2: Predict cost
    # xgb_model = load_xgb_model()
    # xgb_model = load_lgb_model()

    # if st.sidebar.button("Predict Cost"):
    #     # print(patient_data)
    #     pass

    # st.subheader("Predicted Inpatient Cost")
    # cost = predict_data(model, patient_data_encoded)
    # st.metric(label="Estimated Cost", value=f"${cost:,.2f}")
