import csv
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import OrderedDict, Optional

try:
    from opentelemetry import metrics
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    _HAS_OTEL = True
except Exception:
    metrics = None
    OTLPMetricExporter = None
    MeterProvider = None
    PeriodicExportingMetricReader = None
    Resource = None
    _HAS_OTEL = False

from toolkit.config_modules import LoggingConfig

logger = logging.getLogger(__name__)


class MetricsWriter:
    def __init__(self, metrics_dir: Optional[str], write_csv: bool = True, write_jsonl: bool = True) -> None:
        self.metrics_dir = metrics_dir
        self.write_csv = write_csv
        self.write_jsonl = write_jsonl
        self._csv_handle = None
        self._csv_writer = None
        self._jsonl_handle = None
        self._pending_flush_count = 0
        self._last_flush_time = time.monotonic()
        self._flush_every_writes = 20
        self._flush_every_seconds = 1.0

        if self.metrics_dir:
            os.makedirs(self.metrics_dir, exist_ok=True)
            self.metrics_csv_path = os.path.join(self.metrics_dir, "metrics.csv")
            self.metrics_jsonl_path = os.path.join(self.metrics_dir, "metrics.jsonl")
        else:
            self.metrics_csv_path = None
            self.metrics_jsonl_path = None

        self.csv_fields = [
            "timestamp",
            "step",
            "job_id",
            "loss",
            "learning_rate",
            "gpu_utilization_percent",
            "gpu_memory_used_mb",
            "gpu_memory_total_mb",
            "gpu_memory_allocated_mb",
            "gpu_memory_reserved_mb",
            "cpu_load_percent",
            "cpu_memory_percent",
            "cpu_memory_available_mb",
        ]

        self.resource_field_map = {
            "resource/gpu_utilization_percent": "gpu_utilization_percent",
            "resource/gpu_memory_used_mb": "gpu_memory_used_mb",
            "resource/gpu_memory_total_mb": "gpu_memory_total_mb",
            "resource/gpu_memory_allocated_mb": "gpu_memory_allocated_mb",
            "resource/gpu_memory_reserved_mb": "gpu_memory_reserved_mb",
            "resource/cpu_load_percent": "cpu_load_percent",
            "resource/cpu_memory_percent": "cpu_memory_percent",
            "resource/cpu_memory_available_mb": "cpu_memory_available_mb",
        }

    def _ensure_csv_writer(self) -> None:
        if not self.write_csv or not self.metrics_csv_path:
            return
        if self._csv_writer is not None:
            return

        self._csv_handle = open(self.metrics_csv_path, "a", encoding="utf-8", newline="")
        self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self.csv_fields)
        if self._csv_handle.tell() == 0:
            self._csv_writer.writeheader()

    def _ensure_jsonl_handle(self) -> None:
        if not self.write_jsonl or not self.metrics_jsonl_path:
            return
        if self._jsonl_handle is not None:
            return
        self._jsonl_handle = open(self.metrics_jsonl_path, "a", encoding="utf-8")

    def write(self, payload: dict, step: Optional[int], job_id: Optional[str]) -> None:
        if not self.metrics_dir:
            return

        timestamp = datetime.utcnow().isoformat() + "Z"
        record = {
            "timestamp": timestamp,
            "step": step,
            "job_id": job_id,
            "loss": payload.get("loss"),
            "learning_rate": payload.get("learning_rate"),
        }

        for source_key, target_key in self.resource_field_map.items():
            if source_key in payload:
                record[target_key] = payload.get(source_key)

        if self.write_csv and self.metrics_csv_path:
            self._ensure_csv_writer()
            if self._csv_writer:
                self._csv_writer.writerow(record)
                self._pending_flush_count += 1

        if self.write_jsonl and self.metrics_jsonl_path:
            json_payload = {"timestamp": timestamp, "step": step, "job_id": job_id, **payload}
            self._ensure_jsonl_handle()
            if self._jsonl_handle:
                self._jsonl_handle.write(json.dumps(json_payload) + "\n")
                self._pending_flush_count += 1

        self._maybe_flush()

    def _maybe_flush(self, force: bool = False) -> None:
        if force:
            if self._csv_handle:
                self._csv_handle.flush()
            if self._jsonl_handle:
                self._jsonl_handle.flush()
            self._pending_flush_count = 0
            self._last_flush_time = time.monotonic()
            return

        if self._pending_flush_count <= 0:
            return

        now = time.monotonic()
        if self._pending_flush_count >= self._flush_every_writes or (now - self._last_flush_time) >= self._flush_every_seconds:
            if self._csv_handle:
                self._csv_handle.flush()
            if self._jsonl_handle:
                self._jsonl_handle.flush()
            self._pending_flush_count = 0
            self._last_flush_time = now

    def close(self) -> None:
        self._maybe_flush(force=True)
        if self._csv_handle:
            self._csv_handle.close()
            self._csv_handle = None
            self._csv_writer = None
        if self._jsonl_handle:
            self._jsonl_handle.close()
            self._jsonl_handle = None


# Base logger class
# This class does nothing, it's just a placeholder
class EmptyLogger:
    def __init__(self, metrics_dir: Optional[str] = None, write_csv: bool = True, write_jsonl: bool = True,
                 job_id: Optional[str] = None, *args, **kwargs) -> None:
        self.job_id = job_id
        self.metrics_writer = MetricsWriter(metrics_dir, write_csv=write_csv, write_jsonl=write_jsonl)

    # start logging the training
    def start(self):
        pass

    # collect the log to send
    def log(self, data: Optional[dict] = None, *, step: Optional[int] = None,
            job_id: Optional[str] = None, **kwargs):
        payload = {}
        if isinstance(data, dict):
            payload.update(data)
        payload.update(kwargs)
        self.metrics_writer.write(payload, step=step, job_id=job_id or self.job_id)

    # send the log
    def commit(self, step: Optional[int] = None):
        pass

    # log image
    def log_image(self, *args, **kwargs):
        pass

    # finish logging
    def finish(self):
        self.metrics_writer.close()


def _parse_otel_headers(headers_raw: Optional[str]) -> dict:
    if not headers_raw:
        return {}
    headers = {}
    for item in headers_raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            headers[key] = value
    return headers


def _sanitize_metric_name(name: str) -> str:
    name = name.replace("/", ".")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _normalize_otlp_metrics_endpoint(endpoint: Optional[str]) -> Optional[str]:
    if not endpoint:
        return None
    endpoint = endpoint.strip()
    if not endpoint:
        return None
    if endpoint.rstrip("/").endswith("/v1/metrics"):
        return endpoint
    if "/v1/" in endpoint:
        return endpoint
    return endpoint.rstrip("/") + "/v1/metrics"


# OpenTelemetry logger class
# This class logs the data to OpenTelemetry
class OTelLogger(EmptyLogger):
    def __init__(
        self,
        service_name: str,
        config: OrderedDict,
        job_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metrics_dir: Optional[str] = None,
        write_csv: bool = True,
        write_jsonl: bool = True,
        exporter_endpoint: Optional[str] = None,
        exporter_headers: Optional[str] = None,
        dashboard_url: Optional[str] = None,
    ) -> None:
        super().__init__(metrics_dir=metrics_dir, write_csv=write_csv, write_jsonl=write_jsonl, job_id=job_id)
        self.service_name = service_name
        self.config = config
        self.tags = tags or []
        self.exporter_endpoint = exporter_endpoint
        self.exporter_headers = exporter_headers
        self.dashboard_url = dashboard_url
        self._use_otel = False
        self._meter_provider = None
        self._meter = None
        self._metric_cache: dict[str, object] = {}
        self._metrics_dir = metrics_dir
        self._otel_status = "not_started"
        self._otel_disabled_reason: Optional[str] = None

    def _resolve_exporter_endpoint(self) -> Optional[str]:
        metrics_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
        if metrics_endpoint:
            return metrics_endpoint
        return _normalize_otlp_metrics_endpoint(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or self.exporter_endpoint)

    def _resolve_exporter_headers(self) -> Optional[str]:
        return (
            os.environ.get("OTEL_EXPORTER_OTLP_METRICS_HEADERS")
            or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
            or self.exporter_headers
        )

    def _write_otel_metadata(self) -> None:
        if not self._metrics_dir:
            return
        payload = {
            "service_name": os.environ.get("OTEL_SERVICE_NAME") or self.service_name,
            "exporter_endpoint": self._resolve_exporter_endpoint(),
            "dashboard_url": self.dashboard_url or os.environ.get("OTEL_DASHBOARD_URL"),
            "status": self._otel_status,
            "disabled_reason": self._otel_disabled_reason,
        }
        try:
            os.makedirs(self._metrics_dir, exist_ok=True)
            otel_path = os.path.join(self._metrics_dir, "otel_run.json")
            with open(otel_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload))
        except Exception:
            return

    def start(self):
        custom_export_enabled = os.environ.get("AI_TOOLKIT_ENABLE_CUSTOM_OTEL_EXPORT", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not custom_export_enabled:
            self._use_otel = False
            self._otel_status = "disabled"
            self._otel_disabled_reason = (
                "Custom OTEL metric export is disabled by default; set "
                "AI_TOOLKIT_ENABLE_CUSTOM_OTEL_EXPORT=true to send direct OTLP metrics"
            )
            logger.info("Custom OpenTelemetry metrics disabled; local metrics files will still be written: %s",
                        self._otel_disabled_reason)
            self._write_otel_metadata()
            return

        exporter_endpoint = self._resolve_exporter_endpoint()
        if not exporter_endpoint:
            self._use_otel = False
            self._otel_status = "disabled"
            self._otel_disabled_reason = (
                "OTEL_EXPORTER_OTLP_ENDPOINT is not set; Modal native OTEL still handles platform logs and metrics "
                "when configured in the Modal workspace"
            )
            logger.info("Custom OpenTelemetry metrics disabled; local metrics files will still be written: %s",
                        self._otel_disabled_reason)
            self._write_otel_metadata()
            return

        if not _HAS_OTEL:
            self._use_otel = False
            self._otel_status = "disabled"
            self._otel_disabled_reason = "OpenTelemetry packages are not installed"
            logger.info("Custom OpenTelemetry metrics disabled; local metrics files will still be written: %s",
                        self._otel_disabled_reason)
            self._write_otel_metadata()
            return

        headers = _parse_otel_headers(self._resolve_exporter_headers())
        if "nr-data.net" in exporter_endpoint.lower() and not any(key.lower() == "api-key" for key in headers):
            self._use_otel = False
            self._otel_status = "disabled"
            self._otel_disabled_reason = (
                "New Relic OTLP export requires OTEL_EXPORTER_OTLP_HEADERS or "
                "OTEL_EXPORTER_OTLP_METRICS_HEADERS with api-key=<license key>"
            )
            logger.info("Custom OpenTelemetry metrics disabled; local metrics files will still be written: %s",
                        self._otel_disabled_reason)
            self._write_otel_metadata()
            return

        resource_attrs = {
            "service.name": os.environ.get("OTEL_SERVICE_NAME") or self.service_name,
        }
        if self.job_id:
            resource_attrs["ai_toolkit.job_id"] = self.job_id
        modal_task_id = os.environ.get("MODAL_TASK_ID")
        if modal_task_id:
            resource_attrs["ai_toolkit.modal_task_id"] = modal_task_id
        if self.tags:
            resource_attrs["ai_toolkit.tags"] = ",".join(self.tags)

        exporter_kwargs = {"timeout": 10, "endpoint": exporter_endpoint}
        if headers:
            exporter_kwargs["headers"] = headers

        try:
            exporter = OTLPMetricExporter(**exporter_kwargs)
            reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
            resource = Resource.create(resource_attrs)
            provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(provider)
            self._meter_provider = provider
            self._meter = metrics.get_meter("ai-toolkit.metrics")
            self._metric_cache["training.loss"] = self._meter.create_histogram("training.loss")
            self._metric_cache["training.learning_rate"] = self._meter.create_histogram("training.learning_rate")
            self._use_otel = True
            self._otel_status = "exporting"
            self._otel_disabled_reason = None
            self._write_otel_metadata()
        except Exception as exc:
            self._use_otel = False
            self._otel_status = "disabled"
            self._otel_disabled_reason = f"Failed to initialize OTEL exporter: {exc}"
            logger.warning("Custom OpenTelemetry metrics disabled; local metrics files will still be written: %s",
                           self._otel_disabled_reason)
            self._write_otel_metadata()

    def _record_metric(self, name: str, value: Optional[float], attributes: dict) -> None:
        if not self._use_otel or self._meter is None:
            return
        if value is None:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        metric_name = _sanitize_metric_name(name)
        instrument = self._metric_cache.get(metric_name)
        if instrument is None:
            instrument = self._meter.create_histogram(metric_name)
            self._metric_cache[metric_name] = instrument
        try:
            instrument.record(value, attributes=attributes)
        except Exception:
            return

    def log(self, data: Optional[dict] = None, *, step: Optional[int] = None,
            job_id: Optional[str] = None, **kwargs):
        payload = {}
        if isinstance(data, dict):
            payload.update(data)
        payload.update(kwargs)

        self.metrics_writer.write(payload, step=step, job_id=job_id or self.job_id)

        if not self._use_otel:
            return
        attributes = {}
        if step is not None:
            attributes["step"] = int(step)
        if job_id or self.job_id:
            attributes["job_id"] = job_id or self.job_id
        for key, value in payload.items():
            if key == "loss":
                self._record_metric("training.loss", value, attributes)
            elif key.startswith("loss/"):
                suffix = key.split("/", 1)[1]
                self._record_metric(f"training.loss.{suffix}", value, attributes)
            elif key == "learning_rate":
                self._record_metric("training.learning_rate", value, attributes)
            elif key.startswith("resource/"):
                suffix = key.split("/", 1)[1]
                self._record_metric(f"resource.{suffix}", value, attributes)

    def finish(self):
        if self._meter_provider is not None:
            try:
                self._meter_provider.shutdown()
            except Exception:
                pass
        self.metrics_writer.close()


# create logger based on the logging config
def create_logger(logging_config: LoggingConfig, all_config: OrderedDict, *, job_id: Optional[str] = None,
                  metrics_dir: Optional[str] = None, tags: Optional[list[str]] = None):
    if logging_config.use_otel:
        return OTelLogger(
            service_name=logging_config.otel_service_name,
            config=all_config,
            job_id=job_id,
            tags=tags,
            metrics_dir=metrics_dir,
            write_csv=logging_config.write_metrics_csv,
            write_jsonl=logging_config.write_metrics_jsonl,
            exporter_endpoint=logging_config.otel_exporter_endpoint,
            exporter_headers=logging_config.otel_exporter_headers,
            dashboard_url=logging_config.otel_dashboard_url,
        )
    return EmptyLogger(
        metrics_dir=metrics_dir,
        write_csv=logging_config.write_metrics_csv,
        write_jsonl=logging_config.write_metrics_jsonl,
        job_id=job_id,
    )
