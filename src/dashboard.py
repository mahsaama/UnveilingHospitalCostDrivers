from langchain_experimental.plan_and_execute import PlanAndExecute, load_chat_planner, load_agent_executor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import StructuredTool
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import json
import time
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

shared_memory = {}

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
    model = joblib.load("models/xgb_model.pkl")
    return model


def load_lgb_model():
    model = joblib.load("models/lgb_model.pkl")
    return model


def load_shap_explainer():
    explainer = joblib.load("models/shap_explainer.pkl")
    return explainer


def predict_data(model, input):
    y_pred = model.predict(input)
    return y_pred.item()


explainer = load_shap_explainer()
model = load_xgb_model()


def extract_shap_info():
    patient_df = pd.DataFrame(shared_memory["patient_dict"], index=[0])
    for col in data.select_dtypes(['category']).columns:
        patient_df[col] = pd.Categorical(patient_df[col], categories=data[col].cat.categories)
    shap_values = explainer(patient_df)
    shap_info = {}
    for i, col in enumerate(patient_df.columns):
        shap_info[col] = {
            "type": data_info[col]["type"],
            "value": shap_values.data[0][i],
            "shap value": float(shap_values.values[0][i]),
            "other options": data_info[col].get("options", [])
        }
    return json.dumps(shap_info)


def suggest_strategies(shap_info: str):
    shap_info = json.dumps(shap_info) if isinstance(shap_info, dict) else shap_info
    response = llm.invoke(
    [
        HumanMessage(
            content=USER_PROMPT.replace(
                "###patient_info###", json.dumps(shap_info)
            )
        ),
        SystemMessage(content=SYSTEM_PROMPT),
    ]
    )
    strategies = response.content.strip()
    return json.dumps(strategies)


def cost_predition(strategies: dict):
    strategies = json.loads(strategies) if isinstance(strategies, str) else strategies
    shared_memory["suggested strategies"] = strategies
    min_cost = shared_memory["current cost"]
    for col, method in strategies.items():
        if col == "Summary":
            continue
        info = data_info[col]
        if method != None and len(method) > 5:
            if info["type"] == "str" and "options" in info:
                print(col, info["options"])
                for option in info["options"]:
                    updated_patient_dict = shared_memory["patient_dict"].copy()
                    # apply changes
                    updated_patient_dict[col] = option
                    updated_patient_df = pd.DataFrame(updated_patient_dict, index=[0])
                    for col_n in data.select_dtypes(['category']).columns:
                        updated_patient_df[col_n] = pd.Categorical(updated_patient_df[col_n], categories=data[col_n].cat.categories)
                    cost = model.predict(updated_patient_df).item()
                    if cost < min_cost:
                        min_cost = cost
                        shared_memory["new cost"] = min_cost
                        shared_memory["target feature"] = col
                        shared_memory["target strategy"] = option
    # print(shared_memory)
    return json.dumps(shared_memory)


# --- Tool Collection and Agent Setup ---
tools = [
    StructuredTool.from_function(name="ExtractSHAPInfo", func=extract_shap_info, description="Compute SHAP values and return attributions"),
    StructuredTool.from_function(name="SuggestStrategies", func=suggest_strategies, description="Suggest realistic cost-reducing strategies for given features"),
    StructuredTool.from_function(name="CostPredition", func=cost_predition, description="Predict the cost after applying the strategies", return_direct=True),
]


planner = load_chat_planner(llm)
executor = load_agent_executor(llm=llm, tools=tools, verbose=False)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=False, input_key="input")


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
                    index=1,
                    help=data_info[col]["definition"],
                )

            else:

                if data_info[col]["type"] == "int64" or data_info[col]["type"] == "float64":
                    sample_patient_dict[col] = st.sidebar.number_input(
                        f"{col}",
                        value=data[col][1],
                        help=data_info[col]["definition"],
                    )

                elif data_info[col]["type"] == "str":
                    sample_patient_dict[col] = st.sidebar.text_input(
                        f"{col}",
                        value=(
                            data_info[col]["default"]
                            if "default" in data_info[col]
                            else data[col][1]
                        ),
                        placeholder=(
                            data_info[col]["default"]
                            if "default" in data_info[col]
                            else data[col][1]
                        ),
                        help=data_info[col]["definition"],
                    )

    shared_memory["patient_dict"] = sample_patient_dict

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
        sample_patient_df = pd.DataFrame(sample_patient_dict, index=[0])

        for col in data.select_dtypes(["category"]).columns:
            sample_patient_df[col] = pd.Categorical(
                sample_patient_df[col], categories=data[col].cat.categories
            )

        st.subheader("Predicted Inpatient Cost")
        cost = predict_data(model, sample_patient_df)
        shared_memory["current cost"] = cost
        st.metric(label="Estimated Cost", value=f"${cost:,.2f}")

        # ask if user wants cost reduction
        if st.button("Would you like to reduce the cost using AI suggestions?"):
            st.session_state.suggestion_clicked = True

    if st.session_state.suggestion_clicked and not st.session_state.proposals:
        # display suggestions from Agentic AI
        with st.spinner("Generating AI proposals..."):
            # Call your agentic AI module to generate proposals
            prompt = f"""
                You are an expert in healthcare cost optimization and hospital operations strategy.

                Execute these steps using the available tools:
                
                1. Run ExtractSHAPInfo with the result.
                2. Run SuggestStrategies with the result.
                3. Run CostPredition with suggested strategies.
                Return the final recommended strategies.
                """
            
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    output = agent.invoke({"input": prompt})
                    print("Success!")
                    print(output)
                    print("-"*100)
                    print(shared_memory)
                    st.success("AI has generated the following proposals:")
                    proposal_df = pd.DataFrame(columns=["Feature", "Proposal"])
                    for k, v in shared_memory["suggested strategies"].items():
                        proposal_df.loc[len(proposal_df)] = [k, v]

                    proposal_df.loc[len(proposal_df)] = ["Summary", output["output"]]

                    gb = GridOptionsBuilder.from_dataframe(proposal_df)
                    # Enable text wrapping and auto height for Description column
                    gb.configure_column(
                        "Proposal",
                        wrapText=True,
                        autoHeight=True,
                        cellStyle={'white-space': 'normal'}
                    )
                    gb.configure_pagination()
                    gb.configure_side_bar()
                    grid_options = gb.build()
                    AgGrid(proposal_df, gridOptions=grid_options, enable_enterprise_modules=True, height=300, fit_columns_on_grid_load=True)

                    if "new cost" in shared_memory:
                        st.metric(
                            label="Estimated Cost After Intervention",
                            value=f"${shared_memory['new cost']:,.2f}",
                            delta=f"-${shared_memory['current cost'] - shared_memory['new cost']:,.2f} by setting {shared_memory['target feature']} to {shared_memory['target strategy']}"
                        )
                    else:
                        st.info("Unfortunately, I couldn't find any cost optimization method. Please update patient info manually based on the proposals.")

                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(10)
            else:
                print("All retries failed.")