import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Choose a model available on OpenRouter (e.g., GPT-3.5, Mistral, Claude)
llm = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct",  # You can change this
    temperature=0.7
)

# Send a test message
response = llm([HumanMessage(content="Give me three ways to reduce hospital costs without reducing care quality.")])
print(response.content)

