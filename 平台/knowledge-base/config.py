"""Knowledge base configuration."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from project .env file
load_dotenv(Path(__file__).parent / ".env")

# Base paths
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
INDEX_DIR = DATA_DIR / "indexes"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# LLM settings: connect to the local GLM-4.5-Air server
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.10.2:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-local")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4.5-air")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))

# Local ONNX embedding model directory (downloaded BGE-M3 ONNX weights)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", str(PROJECT_ROOT / "data" / "models" / "bge-m3-onnx")
)

# Local ONNX reranker model (BGE-reranker-v2-m3)
RERANKER_DIR = os.getenv(
    "RERANKER_DIR",
    str(PROJECT_ROOT / "data" / "models" / "bge-reranker-v2-m3-onnx"),
)
RERANKER_ONNX = os.getenv(
    "RERANKER_ONNX", str(Path(RERANKER_DIR) / "onnx" / "model_quantized.onnx")
)
RERANKER_TOKENIZER_DIR = os.getenv(
    "RERANKER_TOKENIZER_DIR",
    str(PROJECT_ROOT / "data" / "models" / "bge-reranker-v2-m3-tokenizer"),
)

# Retrieval & context settings
ENABLE_HYBRID_DEFAULT = os.getenv("ENABLE_HYBRID", "true").lower() in ("1", "true", "yes")
ENABLE_RERANK_DEFAULT = os.getenv("ENABLE_RERANK", "true").lower() in ("1", "true", "yes")
ENABLE_QUERY_EXPANSION_DEFAULT = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() in ("1", "true", "yes")
CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", "3000"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "60"))
VECTOR_CANDIDATES = int(os.getenv("VECTOR_CANDIDATES", "30"))
KEYWORD_CANDIDATES = int(os.getenv("KEYWORD_CANDIDATES", "30"))

# Memory management settings
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.92"))
MEMORY_DEFAULT_WEIGHT = float(os.getenv("MEMORY_DEFAULT_WEIGHT", "1.0"))
MEMORY_TIME_DECAY_LAMBDA = float(os.getenv("MEMORY_TIME_DECAY_LAMBDA", "0.05"))
MEMORY_ACCESS_WEIGHT = float(os.getenv("MEMORY_ACCESS_WEIGHT", "0.3"))
MEMORY_CONFIDENCE_WEIGHT = float(os.getenv("MEMORY_CONFIDENCE_WEIGHT", "0.5"))
MEMORY_ARCHIVE_DAYS = int(os.getenv("MEMORY_ARCHIVE_DAYS", "90"))
ENABLE_MEMORY_VALUE_ESTIMATION = os.getenv("ENABLE_MEMORY_VALUE_ESTIMATION", "true").lower() in ("1", "true", "yes")

# Web UI settings
HOST = os.getenv("KB_HOST", "0.0.0.0")
PORT = int(os.getenv("KB_PORT", "7860"))

# OpenAI-compatible RAG API settings
API_HOST = os.getenv("KB_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("KB_API_PORT", "7861"))
API_DEFAULT_KB = os.getenv("KB_API_DEFAULT_KB", "")

# External LLM provider (e.g. Kimi / Moonshot / OpenAI)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")
EXTERNAL_LLM_BASE_URL = (os.getenv("EXTERNAL_LLM_BASE_URL", "") or "").strip()
EXTERNAL_LLM_API_KEY = (os.getenv("EXTERNAL_LLM_API_KEY", "") or "").strip()
EXTERNAL_LLM_MODEL = (os.getenv("EXTERNAL_LLM_MODEL", "") or "kimi-for-coding").strip()

