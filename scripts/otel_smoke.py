import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.config_modules import LoggingConfig
from toolkit.logging_aitk import create_logger
from toolkit.report_generator import generate_reports


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise AssertionError(f"Expected JSON object in {path}")
    return data


def _count_jsonl(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _count_csv_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test local metrics/report writing and optional custom OTEL metric export."
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of synthetic training steps to log.")
    parser.add_argument(
        "--require-export",
        action="store_true",
        help="Fail unless custom OTEL metrics initialize with status=exporting.",
    )
    parser.add_argument(
        "--expect-local-only",
        action="store_true",
        help="Fail unless custom OTEL export is disabled because no OTLP endpoint is configured.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = ROOT / "test_outputs" / "otel_smoke"
    output_root.mkdir(parents=True, exist_ok=True)

    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logging_config = LoggingConfig(
        use_otel=True,
        otel_service_name=os.environ.get("OTEL_SERVICE_NAME", "ai-toolkit-smoke"),
        otel_exporter_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
        otel_exporter_headers=os.environ.get("OTEL_EXPORTER_OTLP_HEADERS"),
        otel_dashboard_url=os.environ.get("OTEL_DASHBOARD_URL"),
        write_metrics_csv=True,
        write_metrics_jsonl=True,
    )

    logger = create_logger(logging_config, {}, metrics_dir=str(run_dir), tags=["smoke:true"])
    logger.start()

    for step in range(1, args.steps + 1):
        loss_val = 1.0 / step
        lr_val = 1e-4 * (1.0 - (step - 1) / max(args.steps, 1))
        logger.log({"loss": loss_val, "learning_rate": lr_val}, step=step)
        logger.commit(step=step)

    logger.finish()

    report_paths = generate_reports(str(run_dir))
    metrics_csv = run_dir / "metrics.csv"
    metrics_jsonl = run_dir / "metrics.jsonl"
    otel_metadata_path = run_dir / "otel_run.json"
    otel_metadata = _load_json(otel_metadata_path)

    _require(metrics_csv.exists(), f"Missing {metrics_csv}")
    _require(metrics_jsonl.exists(), f"Missing {metrics_jsonl}")
    _require(_count_csv_rows(metrics_csv) == args.steps, f"Expected {args.steps} CSV rows")
    _require(_count_jsonl(metrics_jsonl) == args.steps, f"Expected {args.steps} JSONL rows")
    _require(bool(report_paths.get("html")), "Expected at least one HTML report")
    _require(bool(report_paths.get("png")), "Expected at least one PNG report")

    otel_status = otel_metadata.get("status")
    disabled_reason = otel_metadata.get("disabled_reason")
    if args.require_export:
        _require(otel_status == "exporting", f"Expected status=exporting, got {otel_status}: {disabled_reason}")
    if args.expect_local_only:
        _require(otel_status == "disabled", f"Expected status=disabled, got {otel_status}")
        _require(
            "Custom OTEL metric export is disabled by default" in str(disabled_reason)
            or "OTEL_EXPORTER_OTLP_ENDPOINT is not set" in str(disabled_reason),
            f"Expected local-only disabled reason, got: {disabled_reason}",
        )

    print("Metrics dir:", run_dir)
    print("OTEL status:", otel_status)
    if disabled_reason:
        print("OTEL disabled reason:", disabled_reason)
    print("Reports:", report_paths)


if __name__ == "__main__":
    main()

