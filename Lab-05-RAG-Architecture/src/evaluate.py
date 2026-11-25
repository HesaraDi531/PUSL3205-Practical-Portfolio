import csv
import time

from src.config import (
    EVALUATION_DIR,
    RESULTS_DIR,
    TOP_K
)

from src.rag_core import (
    get_saved_store,
    store_is_empty,
    answer_question
)


QUESTIONS_CSV = (
    EVALUATION_DIR / "questions.csv"
)

OUTPUT_CSV = (
    RESULTS_DIR / "evaluation_results.csv"
)


def main():

    with QUESTIONS_CSV.open(
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        questions = [
            row
            for row in csv.DictReader(handle)
            if row.get("question", "").strip()
        ]

    if not questions:

        print(
            "No evaluation questions found."
        )

        print(
            "Add real questions to "
            "evaluation/questions.csv first."
        )

        return

    store = get_saved_store()

    if store_is_empty(store):

        print(
            "Index is empty. Run "
            "python -m src.ingest first."
        )

        return

    output = []

    for index, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"].strip()

        expected_source = item.get(
            "expected_source",
            ""
        ).strip().casefold()

        start = time.perf_counter()

        result = answer_question(
            question,
            store,
            k=TOP_K
        )

        elapsed = time.perf_counter() - start

        names = [
            source["source"]
            for source in result["sources"]
        ]

        rank = next(
            (
                position
                for position, name in enumerate(
                    names,
                    start=1
                )
                if (
                    expected_source
                    and name.casefold() == expected_source
                )
            ),
            0
        )

        output.append({

            "question": question,

            "reference_answer": item.get(
                "reference_answer",
                ""
            ),

            "expected_source": item.get(
                "expected_source",
                ""
            ),

            "retrieved_sources": " | ".join(names),

            "source_hit_at_k": (
                int(rank > 0)
                if expected_source
                else ""
            ),

            "source_reciprocal_rank": (
                round(1 / rank, 4)
                if rank
                else (
                    0 if expected_source else ""
                )
            ),

            "model_answer": result["answer"],

            "latency_seconds": round(
                elapsed,
                2
            ),

            "manual_answer_correct": "",

            "manual_notes": ""
        })

        print(
            f"Evaluated {index}/{len(questions)}"
        )

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(output[0].keys())
        )

        writer.writeheader()
        writer.writerows(output)

    print(
        "Evaluation saved:",
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()