import tempfile

from pathlib import Path

import streamlit as st

from src.config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    DATA_DIR,
    TOP_K
)

from src.documents import (
    collect_paths,
    load_documents,
    split_documents
)

from src.rag_core import (
    get_saved_store,
    index_uploaded_documents,
    store_is_empty,
    answer_question
)


st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University AI Assistant")

st.caption(
    "Ask questions about university documents. "
    "Answers include supporting sources."
)


# Session state
if "upload_store" not in st.session_state:
    st.session_state.upload_store = None

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar
with st.sidebar:

    st.header("Document Sources")

    st.write(f"**LLM:** `{LLM_MODEL}`")

    st.write(
        f"**Embeddings:** `{EMBEDDING_MODEL}`"
    )

    source_mode = st.radio(
        "Search in",
        [
            "Uploaded documents",
            "Local indexed documents"
        ]
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=TOP_K
    )

    if source_mode == "Uploaded documents":

        uploaded = st.file_uploader(
            "Upload multiple PDFs or DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )

        if st.button(
            "Build index from uploads",
            type="primary"
        ):

            if not uploaded:

                st.warning(
                    "Upload at least one document."
                )

            else:

                with st.spinner(
                    "Reading and indexing documents..."
                ):

                    try:

                        with tempfile.TemporaryDirectory(
                            prefix="rag_upload_"
                        ) as temp_dir:

                            paths = []

                            for i, item in enumerate(uploaded):

                                safe_name = Path(
                                    item.name.replace("\\", "/")
                                ).name

                                path = (
                                    Path(temp_dir)
                                    / f"{i:03d}_{safe_name}"
                                )

                                path.write_bytes(
                                    item.getvalue()
                                )

                                paths.append(path)

                            docs, warnings = load_documents(
                                paths
                            )

                            # Restore readable source filenames
                            for doc in docs:

                                source = str(
                                    doc.metadata.get(
                                        "source",
                                        ""
                                    )
                                )

                                if (
                                    len(source) >= 4
                                    and source[:3].isdigit()
                                    and source[3] == "_"
                                ):

                                    doc.metadata["source"] = (
                                        source[4:]
                                    )

                            chunks = split_documents(docs)

                            st.session_state.upload_store = (
                                index_uploaded_documents(
                                    chunks
                                )
                            )

                            st.session_state.uploaded_names = [
                                file.name
                                for file in uploaded
                            ]

                            st.session_state.messages = []

                        st.success(
                            f"Indexed {len(chunks)} chunks "
                            f"from {len(uploaded)} files."
                        )

                        for warning in warnings:
                            st.warning(warning)

                    except Exception as exc:

                        st.error(
                            f"Indexing failed: {exc}"
                        )

        if st.session_state.uploaded_names:

            st.caption(
                "Indexed: "
                + ", ".join(
                    st.session_state.uploaded_names
                )
            )

        selected_store = (
            st.session_state.upload_store
        )

    else:

        st.caption(
            "Run python -m src.ingest "
            "to index local documents."
        )

        local_files = collect_paths(DATA_DIR)

        st.write(
            f"Local documents: {len(local_files)}"
        )

        try:

            selected_store = get_saved_store()

            if store_is_empty(selected_store):
                selected_store = None

        except Exception as exc:

            st.warning(
                f"Index unavailable: {exc}"
            )

            selected_store = None

    if st.button("Clear conversation"):

        st.session_state.messages = []

        st.rerun()


if selected_store is None:

    st.info(
        "Upload documents and build an index, "
        "or index the local documents first."
    )


# Display conversation history
for entry in st.session_state.messages:

    with st.chat_message(entry["role"]):

        st.markdown(entry["content"])

        for source in entry.get("sources", []):

            page_label = (
                f"page {source['page']}"
                if source["page"]
                else "document"
            )

            with st.expander(
                f"[{source['citation']}] "
                f"{source['source']} — {page_label}"
            ):

                st.text(source["snippet"])


question = st.chat_input(
    "Ask a question about your documents...",
    disabled=selected_store is None
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Retrieving evidence and "
                "generating an answer..."
            ):

                result = answer_question(
                    question,
                    selected_store,
                    k=top_k
                )

            st.markdown(result["answer"])

            for source in result["sources"]:

                page_label = (
                    f"page {source['page']}"
                    if source["page"]
                    else "document"
                )

                with st.expander(
                    f"[{source['citation']}] "
                    f"{source['source']} — {page_label}"
                ):

                    st.text(source["snippet"])

            st.session_state.messages.append({

                "role": "assistant",

                "content": result["answer"],

                "sources": result["sources"]
            })

        except Exception as exc:

            st.error(
                f"Question answering failed: {exc}. "
                f"Check that Ollama is running."
            )