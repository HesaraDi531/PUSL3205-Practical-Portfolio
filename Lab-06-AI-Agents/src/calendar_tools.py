from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from src.config import (
    RUNTIME,
    SECRETS,
    TIMEZONE,
    CALENDAR_ID
)


SCOPES = [
    "https://www.googleapis.com/auth/calendar.events"
]

TOKEN_FILE = RUNTIME / "calendar_token.json"
CREDENTIAL_FILE = SECRETS / "credentials.json"


def calendar_service():

    from google.auth.transport.requests import Request

    from google.oauth2.credentials import Credentials

    from google_auth_oauthlib.flow import (
        InstalledAppFlow
    )

    from googleapiclient.discovery import build

    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(
            str(TOKEN_FILE),
            SCOPES
        )

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            if not CREDENTIAL_FILE.exists():
                raise FileNotFoundError(
                    "Google Desktop OAuth credentials missing: "
                    f"{CREDENTIAL_FILE}"
                )

            creds = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIAL_FILE),
                SCOPES
            ).run_local_server(port=0)

        TOKEN_FILE.write_text(
            creds.to_json(),
            encoding="utf-8"
        )

    return build(
        "calendar",
        "v3",
        credentials=creds
    )


@tool
def today_calendar_events() -> str:
    """Read the events scheduled for today from the connected user's Google Calendar."""

    now = datetime.now(ZoneInfo(TIMEZONE))

    start = now.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    end = start + timedelta(days=1)

    response = calendar_service().events().list(
        calendarId=CALENDAR_ID,
        timeMin=start.isoformat(),
        timeMax=end.isoformat(),
        singleEvents=True,
        orderBy="startTime",
        maxResults=50
    ).execute()

    events = response.get("items", [])

    if not events:
        return "No events found today."

    return "\n".join(
        f"{e.get('summary', 'Untitled')} | "
        f"{e['start'].get('dateTime', e['start'].get('date'))}"
        for e in events
    )


@tool
def create_calendar_event(
    summary: str,
    start_iso: str,
    end_iso: str
) -> str:
    """Create a Google Calendar event with title and ISO-8601 start/end including UTC offset. Requires explicit user approval."""

    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)

    if start.tzinfo is None or end.tzinfo is None:
        return (
            "Start and end must include a UTC offset, "
            "e.g. 2026-12-01T14:00:00+05:30."
        )

    if start >= end or end - start > timedelta(hours=12):
        return (
            "Invalid event duration. End must be later "
            "and within 12 hours."
        )

    if len(summary) > 150 or not summary.strip():
        return "Invalid event title."

    body = {
        "summary": summary.strip(),

        "start": {
            "dateTime": start.isoformat(),
            "timeZone": TIMEZONE
        },

        "end": {
            "dateTime": end.isoformat(),
            "timeZone": TIMEZONE
        }
    }

    event = calendar_service().events().insert(
        calendarId=CALENDAR_ID,
        body=body
    ).execute()

    return (
        "Created event: "
        f"{event.get('htmlLink', event.get('id', 'unknown'))}"
    )