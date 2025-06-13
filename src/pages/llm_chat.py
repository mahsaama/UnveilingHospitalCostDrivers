# Begin Ge

import json
import pathlib
import uuid
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ast
import torch
import streamlit as st
import logging
from logging.handlers import RotatingFileHandler
from tabm.tabm_reference import Model
import os


log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
os.makedirs('logs', exist_ok=True)
log_file = 'logs/dashboard.log'

file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("Application starting up.")


CSV_PATH = '../Data/data.csv'



# Begin Co

import os

# Read your API key from the environment variable or set it manually
api_key = os.getenv("GEMINI_API_KEY")

if api_key == None:
    if 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets['GEMINI_API_KEY']
    else:
        logger.error("GEMINI_API_KEY not found in environment variables or Streamlit secrets.")
        st.error("API Key not configured. Please set the GEMINI_API_KEY environment variable or Streamlit secret.")
        st.stop()


from typing import Annotated, Dict, List,Sequence, TypedDict

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages # helper function to add messages to the state


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests


# End Co

@st.cache_data
def load_data(path):
    logger.info(f"Loading data from {path}...")
    data = pd.read_csv(path, low_memory=False)

    df_cleaned = data

    # Remove redundant columns, this information is contained in other columns
    # E.g. CCSR Diagnosis Code and CCSR Diagnosis Description
    df_cleaned = df_cleaned.drop(['Facility Name',
                          'CCSR Diagnosis Description',
                          'CCSR Procedure Description',
                          'APR DRG Description',
                          'APR Severity of Illness Description',
                          'APR MDC Description'
                          ], axis=1)

    # Convert the Total Charges/Costs columns to numeric
    # Remove commas that Pandas can not parse
    df_cleaned['Total Charges'] = pd.to_numeric(df_cleaned['Total Charges'].str.replace(',', '', regex=False))
    df_cleaned['Total Costs'] = pd.to_numeric(df_cleaned['Total Costs'].str.replace(',', '', regex=False))

    # Convert the Length of Stay column to Numeric
    # Right now we just map 120+ days to 120
    df_cleaned['Length of Stay'] = pd.to_numeric(df_cleaned['Length of Stay'].replace({'120 +': '120'}))


    # TODO: Check how NaNs and other weird stuff are dealt with
    categorical_columns = ['Hospital Service Area', 'Hospital County', 'Operating Certificate Number', 'Permanent Facility Id', 'Age Group', 'Zip Code - 3 digits', 'Gender', 'Race', 'Ethnicity', 'Type of Admission', 'Patient Disposition', 'CCSR Diagnosis Code', 'CCSR Procedure Code', 'APR DRG Code', 'APR MDC Code', 'APR Severity of Illness Code', 'APR Risk of Mortality', 'APR Medical Surgical Description', 'Payment Typology 1', 'Payment Typology 2', 'Payment Typology 3', 'Emergency Department Indicator']

    df_cleaned[categorical_columns] = df_cleaned[categorical_columns].astype('category')


    # It seems like there are 1.8 million NAs for birthweight, likely adults
    # Also check if UNKN should be separate or something
    # We can cast to Int64 because the normal int is not nullable,
    # but instead we will just map nans to 0 for now, because I'm
    # not sure how the neural network will deal with nan values
    df_cleaned['Birth Weight'] = pd.to_numeric(df_cleaned['Birth Weight'].replace({'UNKN': 0, np.nan: 0}))


    # Dropping the Total Charges column for now. We need to figure out
    # what it actually means and if using it is possible at inference
    # time
    # df_cleaned = df_cleaned.drop(['Total Charges'], axis=1)
    logger.info("Data loading and preprocessing complete.")
    return df_cleaned

df_cleaned = load_data(CSV_PATH)


@tool('get_column_names', args_schema=None, return_direct=True)
def get_column_names():
    """Retrieves the column names and limited metadata for the SPARCS dataset."""
    # TODO: Add the details from https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/82xm-y6g8/about_data
    logger.info("Executing tool: get_column_names")
    try:
        response = str(pd.DataFrame({
            'nunique': df_cleaned.nunique(),
            'dtype': df_cleaned.dtypes
        }))
    except Exception as e:
        logger.error(f"Error in get_column_names: {e}")
        response = {'error': str(e)}

    return response

class GetUniqueValuesSchema(BaseModel):
    list_cols: List[str] = Field(description="List of strings where each string is a valid column in the dataframe")


@tool('get_unique_values', args_schema=GetUniqueValuesSchema, return_direct=True)
def get_unique_values(list_cols: List[str]):
    """Retrieves the unique values from a list of columns."""
    logger.info(f"Executing tool: get_unique_values with cols: {list_cols}")
    try:
        # The str seems important, otherwise its truncated (I think)
        response = str({col: df_cleaned[col].unique() for col in list_cols})
    except Exception as e:
        logger.error(f"Error in get_unique_values: {e}")
        response = {'error': str(e)}

    return response

class GetGroupedStatsSchema(BaseModel):
    col_name: str = Field(description="Column to apply groupby on")
    agg_dict: Dict[str, List[str]|str] = Field(description="Dictionary to pass to agg")


@tool('get_grouped_stats', args_schema=GetGroupedStatsSchema, return_direct=True)
def get_grouped_stats(col_name: str, agg_dict: Dict[str, List[str]|str]):
    """Groups the data frame and calls agg on it.
    data.groupby(col_name).agg(agg_dict)
    agg_dict must be a Dict, not a string!
    """
    logger.info(f"Executing tool: get_grouped_stats on '{col_name}' with agg: {agg_dict}")
    try:
        # The str seems important, otherwise its truncated (I think)
        response = str(df_cleaned.groupby(col_name).agg(agg_dict))
    except Exception as e:
        logger.error(f"Error in get_grouped_stats: {e}")
        response = {'error': str(e)}

    return response


class SaveBarChartSchema(BaseModel):
    bar_chart_kwargs: Dict = Field(description="Arguments for plotly.graph_objects.Bar")
    layout_kwargs: Dict = Field(description="Arguments for fig.update_layout")


figures_dir = pathlib.Path('figures')
figures_dir.mkdir(parents=True, exist_ok=True)

@tool('save_bar_chart', args_schema=SaveBarChartSchema, return_direct=True)
def save_bar_chart(bar_chart_kwargs: Dict, layout_kwargs: Dict):
    """Draws the bar chart, saves it to file, and returns the filename
    fig = plotly.graph_objects.Bar(bar_chart_kwargs)
    fig.update_layout(layout_kwargs)
    """
    logger.info(f"Executing tool: save_bar_chart with kwargs: {bar_chart_kwargs}, {layout_kwargs}")
    try:
        fig = go.Figure(data=[
            go.Bar(
                bar_chart_kwargs
            )
        ])

        fig.update_layout(
            layout_kwargs
        )

        file_name = f"{uuid.uuid4()}.png"
        full_path = figures_dir / file_name

        fig.write_image(full_path)
        logger.info(f"Bar chart saved to {full_path}")

        response = file_name
    except Exception as e:
        logger.error(f"Error in save_bar_chart: {e}")
        response = {'error': str(e)}

    return response

MODELS_DIR = pathlib.Path('../models/tabm')

# Begin C
@st.cache_resource
def load_prediction_model_assets():
    logger.info("Loading prediction model assets...")
    preprocessing = joblib.load(MODELS_DIR /'quantile_transformer.joblib')

    with open(MODELS_DIR / 'target_scaler.json', 'r') as f:
        target_scaler = json.load(f)
    y_mean, y_std = target_scaler['mean'], target_scaler['std']

    n_cont_features = 4

    per_col_max = np.array([  7,  56, 166, 205,   4,  49,   2,   3,   3,   5,  18, 479, 320,
           333,  25,   4,   3,   2,   8,   8,   8,   1])
    cat_cardinalities = (per_col_max + 1).tolist()
    replace_nans_with = per_col_max + 1
    
    arch_type = 'tabm'
    bins = None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")


    tabm_model = Model(
        n_num_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=None,
        backbone={
            'type': 'MLP',
            'n_blocks': 3 if bins is None else 2,
            'd_block': 512,
            'dropout': 0.1,
        },
        bins=bins,
        num_embeddings=(
            None
            if bins is None
            else {
                'type': 'PiecewiseLinearEmbeddings',
                'd_embedding': 16,
                'activation': False,
                'version': 'B',
            }
        ),
        arch_type=arch_type,
        k=32,
        share_training_batches=True,
    ).to(device)

    ckpt_path = MODELS_DIR / 'best_model.pt'
    state_dict = torch.load(ckpt_path, map_location=device)
    tabm_model.load_state_dict(state_dict)
    tabm_model.eval()
    logger.info("Prediction model loaded successfully.")

    return preprocessing, y_mean, y_std, tabm_model, device, replace_nans_with

preprocessing, y_mean, y_std, tabm_model, device, replace_nans_with = load_prediction_model_assets()
# End C

class PredictTotalCostSchema(BaseModel):
    features_list: List[Dict] = Field(description="A list of dicts. Each dict in the list corresponds to one example. The dict contains feature name-feature value pairs. If the value of a feature is not available, it must be set to None.")


@tool('predict_total_cost', args_schema=PredictTotalCostSchema, return_direct=True)
def predict_total_cost(features_list: List[Dict]):
    """Predicts the total cost for one or more patients using a machine learning model.
    Input: Each dict in the list corresponds to one example. The dict contains feature name-feature value pairs. If the value of a feature is not available, it must be set to None.

    Only the following features may be included:

    Numeric Features: 'Length of Stay', 'Discharge Year', 'Birth Weight', 'Total Charges'

    Categorical Features: 'Hospital Service Area', 'Hospital County',
    'Operating Certificate Number', 'Permanent Facility Id', 'Age Group',
    'Zip Code - 3 digits', 'Gender', 'Race', 'Ethnicity',
    'Type of Admission', 'Patient Disposition', 'CCSR Diagnosis Code',
    'CCSR Procedure Code', 'APR DRG Code', 'APR MDC Code',
    'APR Severity of Illness Code', 'APR Risk of Mortality',
    'APR Medical Surgical Description', 'Payment Typology 1',
    'Payment Typology 2', 'Payment Typology 3',
    'Emergency Department Indicator'

    Output: An array of predicted costs, in the same order as the input. 
    """
    logger.info(f"Executing tool: predict_total_cost for {len(features_list)} examples")
    try:
        # Begin C

        empty = df_cleaned.iloc[0:0]

        new_rows = pd.DataFrame(features_list, columns=df_cleaned.columns)
        for col, orig_dtype in df_cleaned.dtypes.items():
            new_rows[col] = new_rows[col].astype(orig_dtype)
        new_df = pd.concat([empty, new_rows], ignore_index=True)
        new_df = new_df.drop(['Total Costs'], axis=1)

        # End C


        cont_cols = new_df.select_dtypes(exclude=['category']).columns
        X_cont = new_df[cont_cols].to_numpy(dtype='float32')

        cat_cols = new_df.select_dtypes(include=['category']).columns
        X_cat = new_df[cat_cols].apply(lambda s: s.cat.codes).to_numpy(dtype='int64')
        X_cat = np.where(X_cat == -1, replace_nans_with[np.newaxis, :], X_cat)
        X_cat = torch.as_tensor(X_cat, device=device)

        X_cont = preprocessing.transform(X_cont)
        X_cont = torch.as_tensor(X_cont, device=device)

        with torch.no_grad():
            model_output = tabm_model(X_cont, X_cat).squeeze(-1).float()

        y_pred = model_output.cpu().numpy()
        y_pred = y_pred * y_std + y_mean
        y_pred = y_pred.mean(1)

        response = y_pred
        logger.info(f"Prediction successful. Result: {response}")
        

    except Exception as e:
        logger.error(f"Error in predict_total_cost: {e}")
        response = {'error': str(e)}

    return response



tools = [get_column_names, get_unique_values, get_grouped_stats, save_bar_chart, predict_total_cost]
logger.info(f"Tools initialized: {[tool.name for tool in tools]}")
# Begin Co


from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

# Create LLM class
llm = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash-preview-05-20",
    temperature=1.0,
    max_retries=2,
    google_api_key=api_key,
)

# Bind tools to the model
# End Co
model = llm.bind_tools(tools)
logger.info("LLM model initialized and tools bound.")
# Begin Co

# Test the model with tools
# res=model.invoke(f"What is the weather in Berlin on {datetime.today()}?")

# print(res)


from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

tools_by_name = {tool.name: tool for tool in tools}

# Define our tool node
def call_tool(state: AgentState):
    logger.info("Entering tool node.")
    outputs = []
    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        logger.info(f"Processing tool call: {tool_call['name']} with ID: {tool_call['id']}")
        # TODO: Temporary workaround hack for dicts coming as strs
        args = {}
        for arg, val in tool_call["args"].items():
            try:
                args[arg] = ast.literal_eval(val)
                logger.info(f"Successfully parsed arg '{arg}' with ast.literal_eval.")
                # print(f"Fixed {arg}: {args[arg]}")
            except:
                args[arg] = val
                logger.warning(f"Could not parse arg '{arg}' with ast.literal_eval, using as string.")
                # print(f"Not Fixed {arg}: {args[arg]}")

        # Get the tool by name
        tool_result = tools_by_name[tool_call["name"]].invoke(args)
        outputs.append(
            ToolMessage(
                content=str(tool_result), # Ensure content is string
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    logger.info("Exiting tool node.")
    return {"messages": outputs}

def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    logger.info("Entering model node.")
    # Invoke the model with the system prompt and the messages
    response = model.invoke(state["messages"], config)
    logger.info("Exiting model node.")
    # We return a list, because this will get added to the existing messages state using the add_messages reducer
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If the last message is not a tool call, then we finish
    if not last_message.tool_calls:
        logger.info("Condition check: no tool calls, finishing.")
        return "end"
    # default to continue
    logger.info("Condition check: tool calls found, continuing.")
    return "continue"


from langgraph.graph import StateGraph, END

# Define a new graph with our state
workflow = StateGraph(AgentState)

# 1. Add our nodes 
workflow.add_node("llm", call_model)
workflow.add_node("tools",  call_tool)
# 2. Set the entrypoint as `agent`, this is the first node called
workflow.set_entry_point("llm")
# 3. Add a conditional edge after the `llm` node is called.
workflow.add_conditional_edges(
    # Edge is used after the `llm` node is called.
    "llm",
    # The function that will determine which node is called next.
    should_continue,
    # Mapping for where to go next, keys are strings from the function return, and the values are other nodes.
    # END is a special node marking that the graph is finish.
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
        "end": END,
    },
)
# 4. Add a normal edge after `tools` is called, `llm` node is called next.
workflow.add_edge("tools", "llm")

# Now we can compile and visualize our graph
graph = workflow.compile()
logger.info("LangGraph workflow compiled successfully.")

# with open('graph.png', 'wb') as f:
#     f.write(graph.get_graph().draw_mermaid_png())
# End co

st.title("🏥 SPARCS Data Chatbot")
st.caption("Ask me about the New York State hospital inpatient discharge dataset.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message['image']:
            st.image(message["image"])


if prompt := st.chat_input("What would you like to know?"):
    logger.info(f"User entered prompt: {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))
        
        inputs = {"messages": history}
        
        logger.info("Invoking graph stream.")
        image_to_display = None # Use a variable to hold the image path
        for state in graph.stream(inputs, stream_mode="values"):
            last_message = state["messages"][-1]

            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                 with st.expander("Thinking..."):
                    st.write(f"Executing tools: `{[tc['name'] for tc in last_message.tool_calls]}`")
                    st.write(f"Arguments: `{last_message.tool_calls[0]['args']}`")
                 
            elif isinstance(last_message, ToolMessage):
                 with st.expander(f"Tool `{last_message.name}` output:"):
                    st.markdown(f"```\n{last_message.content}\n```")
                 if last_message.name == 'save_bar_chart' and '.png' in last_message.content:
                     image_path = figures_dir / last_message.content
                     if image_path.isfile():
                         image_to_display = str(image_path)
            
            elif isinstance(last_message, AIMessage):
                full_response = last_message.content
                message_placeholder.markdown(full_response)
        
        if image_to_display:
            st.image(image_to_display)

        message_placeholder.markdown(full_response)
        
        final_assistant_message = {"role": "assistant", "content": full_response, "image": image_to_display}
        st.session_state.messages.append(final_assistant_message)
        logger.info("Graph stream finished. Assistant response complete.")
# End Ge
