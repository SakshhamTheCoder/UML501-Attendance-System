# dashboard.py
from flask import Flask, render_template, Response, stream_with_context
import time
import json

from attendance import get_events, get_status, get_event_id, get_images
from config import Config

app = Flask(__name__)


@app.route("/")
def index():
    return render_template(
        "attendance.html",
        events=get_events(),
        status=get_status(),
    )


@app.route("/events")
def sse_stream():
    """
    SSE endpoint:
    - updates ONLY when a new attendance event is created (IN/OUT)
    - extremely fast (100ms check)
    - no unnecessary polling
    """

    def stream():
        last_id = get_event_id()

        while True:
            current_id = get_event_id()

            if current_id != last_id:
                payload = {
                    "events": get_events(),
                    "status": get_status(),
                    "images": get_images(),
                }

                last_id = current_id

                yield f"event: update\ndata: {json.dumps(payload)}\n\n"

            time.sleep(0.1)  # 100ms → near-instant response

    response = Response(stream_with_context(stream()), mimetype="text/event-stream")

    # Explicit SSE headers (prevent buffering)
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"

    return response


def run_dashboard():
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


if __name__ == "__main__":
    run_dashboard()
