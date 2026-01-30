import json


def parse_sse_events(raw: str) -> list[dict]:
    """Parse an SSE stream into a list of event dicts with 'event' and 'data' keys."""
    events: list[dict] = []
    current_event: str | None = None
    current_data: str = ""

    for line in raw.split("\n"):
        line = line.rstrip("\r")
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current_data = line[len("data:") :].strip()
        elif line == "":
            if current_data:
                if current_data == "[DONE]":
                    events.append({"event": "done", "data": "[DONE]"})
                else:
                    events.append(
                        {
                            "event": current_event,
                            "data": json.loads(current_data),
                        }
                    )
                current_event = None
                current_data = ""

    return events
