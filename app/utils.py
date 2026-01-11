from langchain_openai import ChatOpenAI
import os
from app.config import Config

def get_llm():
    api_key = Config.OPENROUTER_API_KEY
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in Config")
    return ChatOpenAI(
        model="openai/gpt-oss-120b", 
        temperature=0,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
