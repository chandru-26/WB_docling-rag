import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "")  # Leave empty for local file storage
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "pdf_knowledge_base")
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_db")  # Directory path if using local storage

# General Processing Configuration
INPUT_DIR = os.getenv("INPUT_DIR", "input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))  # Target chunk size in characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # Overlap between chunks in characters
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "classic")  # "classic" or "agentic"


# Validate critical Azure config
def is_azure_configured() -> bool:
    return bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)
