"""LLM client configuration for Azure OpenAI."""

from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

# Handle both relative imports (when used as module) and direct imports (when run directly)
try:
    from config import get_azure_openai_config, get_azure_deployment
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_azure_openai_config, get_azure_deployment

# Load environment variables
load_dotenv()


def create_llm_client() -> AzureChatOpenAI:
    """
    Create and configure Azure OpenAI client.
    
    Returns:
        AzureChatOpenAI: Configured LLM client
        
    Raises:
        RuntimeError: If required environment variables are missing
    """
    azure_config = get_azure_openai_config()
    deployment = get_azure_deployment()

    # Fail fast if anything is missing
    for key, val in [
        ("AZURE_OPENAI_API_KEY", azure_config["api_key"]),
        ("AZURE_OPENAI_ENDPOINT", azure_config["azure_endpoint"]),
        ("AZURE_OPENAI_API_VERSION", azure_config["api_version"]),
        ("AZURE_OPENAI_DEPLOYMENT", deployment),
    ]:
        if not val:
            raise RuntimeError(f"Missing {key} in .env")

    # Azure-specific client; lichte determinisme voor consistente samenvattingen
    return AzureChatOpenAI(
        azure_deployment=deployment,
        api_key=azure_config["api_key"],
        azure_endpoint=azure_config["azure_endpoint"],
        api_version=azure_config["api_version"],
    )
