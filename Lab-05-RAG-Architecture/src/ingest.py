import json

from datetime import datetime, timezone

from src.config import DATA_DIR, RESULTS_DIR

from src.documents import (
    collect_paths,
    load_documents,
    split_documents
)

from src.rag_core import index_saved_documents


def main():

    paths = collect_paths(DATA_DIR)

    if not paths:

        print(
            f"No PDF/DOCX files in {DATA_DIR}"
        )

        print(
            "Place university documents there "
            "and run again."
        )

        return

    pages, problems = load_documents(paths)

    if not pages:

        print(
            "No readable text found."
        )

        for error in problems:
            print(" -", error)

        return

    chunks = split_documents(pages)

    count = index_saved_documents(chunks)

    summary = {

        "indexed_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),

        "files": [
            path.name
            for path in paths
        ],

        "text_pages_or_documents": len(pages),

        "chunks_indexed": count,

        "warnings": problems
    }

    output = (
        RESULTS_DIR / "ingestion_summary.json"
    )

    output.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    print("Indexing complete.")

    print(
        json.dumps(
            summary,
            indent=2
        )
    )


if __name__ == "__main__":
    main()