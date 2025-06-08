import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import joblib
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import os

gemini_api_key = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = '''
You are an expert in healthcare cost optimization and hospital operations strategy.

Your role is to analyze structured data from individual inpatient hospital cases. Each feature includes:
- The data type (e.g., int, float, str)
- The observed value for the case
- The SHAP value, which quantifies how much that feature contributed to the total inpatient cost
- A list of possible values the feature can take (if available)

Your goal is to:
1. Identify which features are the most significant cost drivers.
2. Propose feasible, actionable strategies to reduce their cost impact, based on the available options.
3. Ensure your recommendations do not compromise—ideally improve—the quality of care.

Respond only with specific, pragmatic suggestions tailored to the data provided.
'''

USER_PROMPT = f'''
You are given structured data for a single inpatient case.

Each feature contains:
- "type": the data type
- "value": the observed value
- "shap value": contribution to the total inpatient cost
- "other options": the possible values for that feature (if any)

Your tasks:
1. For each feature, suggest a specific, feasible strategy to reduce its cost impact, using the "other options" where applicable to that patient. If there is no strategy, return empty string (not None).
2. Conclude with a short summary (2–3 sentences) explaining the overall cost-reduction logic you applied.
3. Return your response in this JSON format only (no additional text):

{{
  "Feature Name 1": "strategy",
  "Feature Name 2": "strategy",
  ...
  "Summary": "your overall summary"
}}

Data:
###patient_info###
'''


# gemini google, load api_key from https://aistudio.google.com/apikey
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=gemini_api_key,
)


@st.cache_data
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
    return y_pred.item()

def get_patient_info(sample_patient_df):
    # shapley values for sample patient
    explainer = load_shap_explainer()
    shap_values = explainer(sample_patient_df)
    info = dict()
    for i, col in enumerate(sample_patient_df.columns):
        info[col] = {
            "type": data_info[col]["type"],
            "value": shap_values.data[0][i],
            "shap value": shap_values.values[0][i].item(),
            "other options": data_info[col]["options"] if "options" in data_info[col] else []
        }
    return info

def ai_agent(patient_info):
    # Send a test message
    try:
        response = llm(
            [
                HumanMessage(
                    content=USER_PROMPT.replace(
                        "###patient_info###", json.dumps(patient_info)
                    )
                ),
                SystemMessage(content=SYSTEM_PROMPT),
            ]
        )

        proposal = json.loads(response.content.strip("```json").strip("```").strip())
    except:
        proposal = ai_agent(patient_info)
    
    return proposal



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

    # --- Initialize state ---
    if "predict_clicked" not in st.session_state:
        st.session_state.predict_clicked = False
    if "suggestion_clicked" not in st.session_state:
        st.session_state.suggestion_clicked = False
    if "proposals" not in st.session_state:
        st.session_state.proposals = {}
    if "selected_proposal" not in st.session_state:
        st.session_state.selected_proposal = None


    if st.sidebar.button("Predict Cost"):
        st.session_state.predict_clicked = True
        st.session_state.suggestion_clicked = False  # reset

    if st.session_state.predict_clicked:
        # Section 2: Predict cost
        # xgb_model = load_xgb_model()
        model = load_lgb_model()

        sample_patient_df = pd.DataFrame(sample_patient_dict, index=[0])

        for col in data.select_dtypes(["category"]).columns:
            sample_patient_df[col] = pd.Categorical(
                sample_patient_df[col], categories=data[col].cat.categories
            )

        st.subheader("Predicted Inpatient Cost")
        cost = predict_data(model, sample_patient_df)
        st.metric(label="Estimated Cost", value=f"${cost:,.2f}")

        # ask if user wants cost reduction
        if st.button("Would you like to reduce the cost using AI suggestions?"):
            st.session_state.suggestion_clicked = True

    if st.session_state.suggestion_clicked and not st.session_state.proposals:
        # display suggestions from Agentic AI
        with st.spinner("Generating AI proposals..."):
            # Call your agentic AI module to generate proposals
            patient_info = get_patient_info(sample_patient_df)
            st.session_state.proposals = ai_agent(patient_info)


    # show proposals if available
    if st.session_state.proposals:
        st.success("AI has generated the following proposals:")

        proposed_options = [
            "-- Select an option --"
        ] + [f"{col}: {p}" for col, p in st.session_state.proposals.items() if len(p) > 25 and col != "Summary"]

        selected = st.selectbox(
            "Choose a proposal to apply:",
            proposed_options,
            key="proposal_selector"
        )

        # Run logic only if a real selection is made and it's new
        if selected != "-- Select an option --" and selected != st.session_state.selected_proposal:
            st.session_state.selected_proposal = selected

            selected_col = selected.split(": ")[0]
            selected_proposal = selected.split(": ")[1]
            info = data_info[selected_col]
            min_cost = cost
            strategy = None
            selected_option = None

            # STR-type feature
            if info["type"] == "str" and "options" in info:
                for option in info["options"]:
                    updated_patient_dict = sample_patient_dict.copy()
                    updated_patient_dict[selected_col] = option
                    updated_patient_df = pd.DataFrame(updated_patient_dict, index=[0])
                    for cat_col in data.select_dtypes(['category']).columns:
                        updated_patient_df[cat_col] = pd.Categorical(
                            updated_patient_df[cat_col],
                            categories=data[cat_col].cat.categories
                        )
                    new_cost = predict_data(model, updated_patient_df)
                    if new_cost < min_cost:
                        min_cost = new_cost
                        strategy = selected_col
                        selected_option = option


                # show result metric
                st.metric(
                    label="Estimated Cost After Intervention",
                    value=f"${min_cost:,.2f}",
                    delta=f"-${cost - min_cost:,.2f} by setting {strategy} to {selected_option}"
                )

            else:
                st.info("Please update patient info manually to evaluate this proposal.")

