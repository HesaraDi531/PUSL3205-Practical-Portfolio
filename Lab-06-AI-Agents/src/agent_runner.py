from langchain_ollama import ChatOllama

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)

from src.config import OLLAMA_MODEL, OLLAMA_URL


APPROVAL_TOOLS = {
    "notepad_write",
    "create_text_file",
    "save_memory",
    "create_calendar_event"
}


def run_agent(tools, mission: str):

    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        temperature=0
    ).bind_tools(tools)

    tool_map = {
        tool.name: tool
        for tool in tools
    }

    history = [
        SystemMessage(
            content=(
                "You are a local student assistant. "
                "Use available tools for all factual computer, "
                "memory, calculator, calendar, and university-document "
                "operations. Do not invent tool results, file paths, "
                "university policies, or calendar events. "
                "For university questions, answer only from "
                "university_search output and cite filename/page. "
                "For explicit requests to remember a fact, use "
                "save_memory. For memory recall, use search_memory. "
                "If a user denies a tool call, do not retry it. "
                "For calendar creation, require a time-zone-aware "
                "ISO timestamp. "
                + mission
            )
        )
    ]

    print("Local AI agent ready. Enter exit to quit.")

    while True:
        message = input("\nYou: ").strip()

        if message.lower() in ("exit", "quit"):
            break

        if not message:
            continue

        history.append(HumanMessage(content=message))

        try:
            for _ in range(6):

                response = model.invoke(history)
                history.append(response)

                if not response.tool_calls:
                    print("\nAgent:", response.content)
                    break

                for call in response.tool_calls:

                    name = call["name"]
                    args = call["args"]

                    print(f"\nTOOL: {name} | ARGS: {args}")

                    if name not in tool_map:
                        result = "Unknown tool was requested."

                    elif (
                        name in APPROVAL_TOOLS
                        and input(
                            "Approve action? (yes/no): "
                        ).strip().lower() != "yes"
                    ):
                        result = (
                            "Denied by the user. "
                            "No changes were made."
                        )

                    else:
                        try:
                            result = tool_map[name].invoke(args)

                        except Exception as exc:
                            result = (
                                f"Tool failed: "
                                f"{type(exc).__name__}: {exc}"
                            )

                    print("RESULT:", result)

                    history.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=call["id"]
                        )
                    )

            else:
                print("Maximum number of tool rounds reached.")

        except Exception as exc:
            print("Agent error:", exc)
            print(
                "Check that Ollama is running and that "
                "the model supports tool calling."
            )

            history = history[:1]