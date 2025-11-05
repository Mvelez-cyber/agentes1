"""Model Configuration"""
import os
from strands.models.openai import OpenAIModel
from dotenv import load_dotenv
load_dotenv() 

AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")

def get_configured_model() -> OpenAIModel:
    model = OpenAIModel(
    client_args={
            "api_key": AZURE_KEY,
            #"api_version":'2024-12-01-preview',
            "base_url":'https://pnl-maestria.openai.azure.com/openai/v1'
        },
        model_id="gpt-4.1-nano",   # <-- nombre de tu deployment en Azure
        params={"temperature": 0.2, "max_tokens": 10000},
    )
    return model


