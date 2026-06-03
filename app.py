from fraud_detection.app import app, schedule_dashboard_open, DEFAULT_HOST, DEFAULT_PORT
import os


if __name__ == "__main__":
    host = os.environ.get("FRAUDGUARD_HOST", DEFAULT_HOST)
    port = int(os.environ.get("PORT", os.environ.get("FRAUDGUARD_PORT", DEFAULT_PORT)))
    schedule_dashboard_open(host, port)
    app.run(debug=False, host=host, port=port)
