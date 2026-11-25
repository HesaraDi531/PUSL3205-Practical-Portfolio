from src.agent_runner import run_agent
from src.desktop_tools import calculator

from src.memory_tools import (
    save_memory,
    search_memory
)

run_agent(
    [
        calculator,
        save_memory,
        search_memory
    ],
    "Remember user facts only when explicitly requested."
)