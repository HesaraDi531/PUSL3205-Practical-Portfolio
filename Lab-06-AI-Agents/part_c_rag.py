from src.agent_runner import run_agent

from src.desktop_tools import calculator

from src.memory_tools import (
    save_memory,
    search_memory
)

from src.rag_tools import university_search


run_agent(
    [
        calculator,
        save_memory,
        search_memory,
        university_search
    ],
    "For university questions, call university_search; "
    "cite the actual PDF filename and page."
)