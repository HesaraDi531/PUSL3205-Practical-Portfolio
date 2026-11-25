from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

WORKSPACE = ROOT / "workspace"
RUNTIME = ROOT / "runtime"
SECRETS = ROOT / "secrets"

for folder in (WORKSPACE, RUNTIME, SECRETS):
    folder.mkdir(exist_ok=True, parents=True)

OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "qwen3-embedding:0.6b"
)

TIMEZONE = os.getenv("TIMEZONE", "Asia/Colombo")
CALENDAR_ID = os.getenv("CALENDAR_ID", "primary")

CHROMA_PATH = Path(
    os.getenv(
        "LAB5_CHROMA_DIR",
        "../Lab-05-RAG-Architecture/chroma_db"
    )
)

if not CHROMA_PATH.is_absolute():
    CHROMA_PATH = (ROOT / CHROMA_PATH).resolve()

CHROMA_COLLECTION = os.getenv(
    "LAB5_COLLECTION",
    "university_documents"
)

RAG_API_KEY = os.getenv("RAG_API_KEY", "")