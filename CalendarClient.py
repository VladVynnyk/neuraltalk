from datetime import datetime, timedelta, timezone

from google.oauth2 import service_account
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

KYIV = timezone(timedelta(hours=3), name='UTC+03:00')

class CalendarClient():
    def __init__(self, service_account_file, scopes, calendar_id):
        self.service_account_file = service_account_file
        self.SCOPES = scopes
        self.calendar_id = calendar_id
        self._service = None

    def _build_service(self):
        if self._service is None:
            creds = service_account.Credentials.from_service_account_file(
                self.service_account_file, scopes=self.SCOPES
            )
            self._service = build('calendar', 'v3', credentials=creds)
        return self._service

    def list_free_slots(self, start_iso: str, end_iso: str, duration_minutes: int, specialist="laserepilation"):
        """Повертає список вільних слотів у календарі між двома ISO-датами з вказаною тривалістю."""
        service = self._build_service()

        start = datetime.fromisoformat(start_iso).astimezone(KYIV)
        end = datetime.fromisoformat(end_iso).astimezone(KYIV)
        duration = timedelta(minutes=duration_minutes)

        body = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "timeZone": "Europe/Kyiv",
            "items": [{"id": self.calendar_id}],
        }

        freebusy = service.freebusy().query(body=body).execute()

        busy_raw = freebusy["calendars"].get(self.calendar_id, {}).get("busy", [])
        print("BUSY: ", busy_raw)

        busy_times = []
        for b in busy_raw:
            busy_start = datetime.fromisoformat(b["start"]).astimezone(KYIV)
            busy_end = datetime.fromisoformat(b["end"]).astimezone(KYIV)
            busy_times.append((busy_start, busy_end))

        busy_times.sort()

        free_ranges = []
        current = start

        for busy_start, busy_end in busy_times:
            if current < busy_start:
                free_ranges.append((current, busy_start))
            current = max(current, busy_end)

        if current < end:
            free_ranges.append((current, end))

        free_slots = []
        for free_start, free_end in free_ranges:
            slot_start = free_start
            while slot_start + duration <= free_end:
                slot_end = slot_start + duration
                free_slots.append((slot_start.isoformat(), slot_end.isoformat()))
                slot_start = slot_end

        return free_slots
    
    def create_appointment(
        self,
        specialist: str,
        start_iso: str,
        end_iso: str,
        summary: str,
        description: str,
        attendee_email: str = None,
    ):
        """Створює подію в Google Calendar на вказану дату, з описом і деталями."""
        service = self._build_service()

        freebusy_query = {
            "timeMin": start_iso,
            "timeMax": end_iso,
            "timeZone": "Europe/Kyiv",
            "items": [{"id": self.calendar_id}]
        }

        busy_slots = service.freebusy().query(body=freebusy_query).execute()

        if busy_slots["calendars"][self.calendar_id]["busy"]:
            print("❌ Конфлікт: у цей час вже є подія.")
            return {"error": "Time slot is already occupied."}

        event = {
            "summary": summary,
            "description": f"{description}\nФахівець: {specialist}",
            "start": {
                "dateTime": start_iso,
                "timeZone": "Europe/Kyiv",
            },
            "end": {
                "dateTime": end_iso,
                "timeZone": "Europe/Kyiv",
            },
            # "attendees": [{"email": attendee_email}] if attendee_email else [],
        }

        created_event = service.events().insert(calendarId=self.calendar_id, body=event).execute()
        print(f"✅ Подія створена: {created_event.get('htmlLink')}")
        return created_event
