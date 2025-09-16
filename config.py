"""Configuration management for API keys and environment variables."""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Azure OpenAI Deployment Names
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")

def get_azure_openai_config():
    """Returns the Azure OpenAI configuration as a dictionary."""
    return {
        "api_key": AZURE_OPENAI_API_KEY,
        "api_version": AZURE_OPENAI_API_VERSION,
        "azure_endpoint": AZURE_OPENAI_ENDPOINT,
    }

def get_azure_deployment():
    """Returns the Azure OpenAI deployment name."""
    return AZURE_OPENAI_DEPLOYMENT

def get_azure_model():
    """Returns the Azure OpenAI model name."""
    return AZURE_OPENAI_MODEL
