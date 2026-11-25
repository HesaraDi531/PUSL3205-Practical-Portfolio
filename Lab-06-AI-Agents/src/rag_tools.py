from functools import lru_cache

from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import (
    CHROMA_PATH,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    OLLAMA_URL
)


@lru_cache(maxsize=1)
def _store():

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"Lab 05 ChromaDB not found: {CHROMA_PATH}"
        )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL
    )

    store = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )

    if not store.get(limit=1, include=[]).get("ids"):
        raise RuntimeError(
            f"Chroma collection {CHROMA_COLLECTION!r} "
            "is empty. Re-index Lab 05 documents."
        )

    return store


def retrieve_evidence(question: str) -> str:

    matches = _store().similarity_search_with_score(
        question,
        k=4
    )

    if not matches:
        return (
            "No relevant passages found "
            "in the indexed documents."
        )

    parts = []

    for i, (doc, distance) in enumerate(matches, 1):

        filename = doc.metadata.get(
            "source",
            "unknown"
        )

        page = doc.metadata.get(
            "page",
            "unknown"
        )

        parts.append(
            f"[{i}] {filename}, page {page}, "
            f"distance={distance:.3f}\n"
            f"{doc.page_content[:1000]}"
        )

    return "\n\n".join(parts)


@tool
def university_search(question: str) -> str:
    """Search the indexed university handbook and return passages with source names and pages."""
    return retrieve_evidence(question)