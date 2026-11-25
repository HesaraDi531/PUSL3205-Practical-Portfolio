import secrets

from fastapi import (
    FastAPI,
    HTTPException,
    Header
)

from pydantic import BaseModel, Field

from src.config import RAG_API_KEY
from src.rag_tools import retrieve_evidence


app = FastAPI(
    title="Local University RAG Bridge"
)


class SearchQuery(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500
    )


@app.post("/search")
def search(
    query: SearchQuery,
    x_api_key: str | None = Header(default=None)
):

    if (
        not RAG_API_KEY
        or not x_api_key
        or not secrets.compare_digest(
            x_api_key,
            RAG_API_KEY
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    try:
        return {
            "evidence": retrieve_evidence(
                query.question
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc