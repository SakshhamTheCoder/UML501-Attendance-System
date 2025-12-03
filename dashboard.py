"""
dashboard.py
- Provides a web dashboard to monitor attendance events and status
"""

from flask import Flask, render_template, Response, stream_with_context
import time
import json

from attendance import get_events, get_status, get_images, get_event_id

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("attendance.html", events=get_events(), status=get_status())


@app.route("/events")
def events_stream():
    def gen():
        last_id = get_event_id()
        while True:
            cur = get_event_id()
            if cur != last_id:
                payload = {
                    "events": get_events(),
                    "status": get_status(),
                    "images": get_images(),
                }
                last_id = cur
                yield f"event: update\ndata: {json.dumps(payload)}\n\n"
            time.sleep(0.1)

    resp = Response(stream_with_context(gen()), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


def run_dashboard():
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


if __name__ == "__main__":
    run_dashboard()
