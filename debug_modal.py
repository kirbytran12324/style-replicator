import os

import modal

app = modal.App()


def _otel_env_snapshot() -> dict:
    return {
        "OTEL_EXPORTER_OTLP_ENDPOINT": os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "OTEL_EXPORTER_OTLP_HEADERS": "set" if os.environ.get("OTEL_EXPORTER_OTLP_HEADERS") else "missing",
        "OTEL_SERVICE_NAME": os.environ.get("OTEL_SERVICE_NAME"),
        "OTEL_DASHBOARD_URL": os.environ.get("OTEL_DASHBOARD_URL"),
    }


@app.function()
def debug_otel_env():
    snapshot = _otel_env_snapshot()
    print("OpenTelemetry env:", snapshot)


if __name__ == "__main__":
    debug_otel_env.remote()
