import argparse

from src.rag_core import (
    get_saved_store,
    store_is_empty,
    answer_question
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Local university-document "
            "RAG question answering"
        )
    )

    parser.add_argument(
        "-q",
        "--question",
        help="Ask one question"
    )

    args = parser.parse_args()

    store = get_saved_store()

    if store_is_empty(store):

        print(
            "The vector database is empty."
        )

        print(
            "Add PDFs to data/ and run "
            "python -m src.ingest"
        )

        return

    while True:

        question = (
            args.question
            or input(
                "\nAsk a university question "
                "(or 'exit'): "
            ).strip()
        )

        if question.lower() in {
            "exit",
            "quit"
        }:
            return

        if not question:
            continue

        try:

            result = answer_question(
                question,
                store
            )

            print("\nANSWER:")
            print(result["answer"])

            print("\nRETRIEVED SOURCES:")

            for source in result["sources"]:

                page = (
                    f"page {source['page']}"
                    if source["page"]
                    else "document"
                )

                print(
                    f"[{source['citation']}] "
                    f"{source['source']} ({page})"
                )

        except Exception as exc:

            print("Error:", exc)

        if args.question:
            return


if __name__ == "__main__":
    main()