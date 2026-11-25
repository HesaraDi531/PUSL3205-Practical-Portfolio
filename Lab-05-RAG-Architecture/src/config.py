from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHROMA_DIR = ROOT / "chroma_db"
SCREENSHOT_DIR = ROOT / "screenshots"
EVALUATION_DIR = ROOT / "evaluation"

for path in (
    DATA_DIR,
    RESULTS_DIR,
    SCREENSHOT_DIR,
    EVALUATION_DIR
):
    path.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2:3b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "qwen3-embedding:0.6b"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

COLLECTION_NAME = "university_documents"

if not (0 <= CHUNK_OVERLAP < CHUNK_SIZE):
    raise ValueError(
        "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
    )