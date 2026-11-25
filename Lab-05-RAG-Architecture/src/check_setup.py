import json

from urllib.request import urlopen

from src.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL
)


def main():

    url = (
        OLLAMA_BASE_URL.rstrip("/")
        + "/api/tags"
    )

    print(
        "Checking Ollama:",
        OLLAMA_BASE_URL
    )

    try:

        with urlopen(url, timeout=10) as response:

            payload = json.loads(
                response.read().decode("utf-8")
            )

    except Exception as exc:

        print(
            "Ollama connection failed:",
            exc
        )

        return

    models = [
        model.get("name", "")
        for model in payload.get("models", [])
    ]

    print("\nInstalled models:")

    for name in models:
        print("-", name)

    for required in (
        LLM_MODEL,
        EMBEDDING_MODEL
    ):

        if required in models:

            print("OK:", required)

        else:

            print(
                "MISSING:",
                required
            )


if __name__ == "__main__":
    main()