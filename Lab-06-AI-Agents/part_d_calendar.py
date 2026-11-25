from src.agent_runner import run_agent

from src.calendar_tools import (
    today_calendar_events,
    create_calendar_event
)


run_agent(
    [
        today_calendar_events,
        create_calendar_event
    ],
    "Use Google Calendar tools only. "
    "Before creating, obtain exact time details "
    "and request user approval."
)