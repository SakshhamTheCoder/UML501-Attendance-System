# attendance.py
from datetime import datetime, timedelta
from threading import Lock
from config import Config

_lock = Lock()

status = {}  # name -> "in" / "out"
events = []  # list of events
event_id = 0  # incremental ID
last_event_time = {}  # name -> datetime of last IN/OUT event

# store latest face image per person (base64 JPEG string)
latest_images = {}


def now_str():
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def _add_event(name, action, image_b64: str = None):
    """Append an event and optionally store a snapshot image for that event.

    Each event will carry an `image` field (base64) if provided so the event
    snapshot remains stable even if `latest_images` changes later.
    """
    global event_id
    event_id += 1
    ev = {"name": name, "action": action, "time": now_str(), "id": event_id}
    if image_b64:
        ev["image"] = image_b64
    events.append(ev)


def mark_seen(name, image_b64: str = None):
    """
    Toggle with debouncer:
    - First time → IN
    - Next detection → OUT (after debounce)
    - Next detection → IN (after debounce)
    - etc.
    """

    now = datetime.now()

    with _lock:
        # Check debouncer
        last_time = last_event_time.get(name)
        if last_time is not None:
            if now - last_time < timedelta(seconds=Config.DEBOUNCE_SECONDS):
                # even if we don't create an event, update stored image if provided
                if image_b64:
                    latest_images[name] = image_b64
                return  # ignore detection, too soon

        # Perform toggle
        prev = status.get(name)

        if prev is None or prev == "out":
            status[name] = "in"
            _add_event(name, "in", image_b64)
        else:
            status[name] = "out"
            _add_event(name, "out", image_b64)

        # Update last event time
        last_event_time[name] = now
        # store latest image (if provided)
        if image_b64:
            latest_images[name] = image_b64


def get_status():
    with _lock:
        return dict(status)


def get_events():
    with _lock:
        return list(events)


def get_images():
    """Return a copy of latest_images mapping name -> base64 string."""
    with _lock:
        return dict(latest_images)


def get_event_id():
    with _lock:
        return event_id
