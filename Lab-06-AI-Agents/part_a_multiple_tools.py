from src.agent_runner import run_agent

from src.desktop_tools import (
    calculator,
    date_time,
    notepad_write,
    file_search,
    create_text_file,
    system_information
)

run_agent(
    [
        calculator,
        date_time,
        notepad_write,
        file_search,
        create_text_file,
        system_information
    ],
    "This agent can use the six desktop tools. "
    "File operations are limited to workspace/."
)