import argparse
import os

from toolkit.report_generator import generate_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training reports from metrics logs.")
    parser.add_argument("--metrics-dir", required=True, help="Directory containing metrics.csv or metrics.jsonl")
    args = parser.parse_args()

    metrics_dir = os.fspath(args.metrics_dir)
    metrics_dir = os.path.abspath(metrics_dir)
    result = generate_reports(metrics_dir)
    html_files = result.get("html", [])
    png_files = result.get("png", [])

    print("Generated reports:")
    for path in html_files + png_files:
        print(f" - {path}")


if __name__ == "__main__":
    main()

