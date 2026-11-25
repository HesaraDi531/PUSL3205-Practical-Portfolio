from pathlib import Path
from typing import Iterable

import fitz
from docx import Document as DocxDocument

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def collect_paths(directory: Path) -> list[Path]:

    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_pdf(path: Path) -> list[Document]:

    docs = []

    with fitz.open(str(path)) as pdf:

        for page_index in range(len(pdf)):

            text = pdf[page_index].get_text(
                "text"
            ).strip()

            if text:

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "page": page_index + 1,
                            "type": "pdf"
                        }
                    )
                )

    return docs


def load_docx(path: Path) -> list[Document]:

    docx = DocxDocument(str(path))

    paragraphs = [
        p.text.strip()
        for p in docx.paragraphs
        if p.text.strip()
    ]

    for table in docx.tables:

        for row in table.rows:

            cells = [
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            ]

            if cells:
                paragraphs.append(" | ".join(cells))

    text = "\n".join(paragraphs)

    if not text:
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": path.name,
                "page": 0,
                "type": "docx"
            }
        )
    ]


def load_documents(
    paths: Iterable[Path]
) -> tuple[list[Document], list[str]]:

    all_docs = []
    problems = []

    for path in paths:

        try:

            if path.suffix.lower() == ".pdf":
                loaded = load_pdf(path)

            elif path.suffix.lower() == ".docx":
                loaded = load_docx(path)

            else:
                continue

            if not loaded:

                problems.append(
                    f"{path.name}: no extractable text"
                )

            all_docs.extend(loaded)

        except Exception as exc:

            problems.append(
                f"{path.name}: {exc}"
            )

    return all_docs, problems


def split_documents(
    documents: list[Document]
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_index"] = index

    return chunks