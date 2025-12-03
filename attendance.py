"""
attendance.py
- Manages attendance status and events
"""

from datetime import datetime, timedelta
from threading import Lock
from config import Config

_lock = Lock()
status = {}
events = []
event_id = 0
last_event_time = {}
latest_images = {}


def now_str():
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def _add_event(name, action, image_b64=None):
    global event_id
    event_id += 1
    ev = {"id": event_id, "name": name, "action": action, "time": now_str()}
    if image_b64:
        ev["image"] = image_b64
    events.append(ev)


def mark_seen(name, image_b64=None):
    now = datetime.now()

    with _lock:
        last_time = last_event_time.get(name)
        if last_time and now - last_time < timedelta(seconds=Config.DEBOUNCE_SECONDS):
            if image_b64:
                latest_images[name] = image_b64
            return

        prev = status.get(name)
        if prev is None or prev == "out":
            status[name] = "in"
            _add_event(name, "in", image_b64)
        else:
            status[name] = "out"
            _add_event(name, "out", image_b64)

        last_event_time[name] = now
        if image_b64:
            latest_images[name] = image_b64


def get_status():
    with _lock:
        return dict(status)


def get_events():
    with _lock:
        return list(events)


def get_images():
    with _lock:
        return dict(latest_images)


def get_event_id():
    with _lock:
        return event_id
