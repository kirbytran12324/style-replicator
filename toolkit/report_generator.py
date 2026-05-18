import csv
import html
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
from typing import Optional

import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    _PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    go = None
    pio = None
    make_subplots = None
    _PLOTLY_AVAILABLE = False


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _load_jsonl(metrics_jsonl_path: str) -> List[Dict]:
    if not os.path.exists(metrics_jsonl_path):
        return []

    records: List[Dict] = []
    with open(metrics_jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _load_csv(metrics_csv_path: str) -> List[Dict]:
    if not os.path.exists(metrics_csv_path):
        return []

    records: List[Dict] = []
    with open(metrics_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(row)
    return records


def _build_series(records: List[Dict]) -> Tuple[List[float], Dict[str, List[float]]]:
    numeric_keys = set()
    for rec in records:
        for key, value in rec.items():
            if key in ("timestamp", "job_id", "step"):
                continue
            if _is_number(value):
                numeric_keys.add(key)

    steps: List[float] = []
    series: Dict[str, List[float]] = {key: [] for key in sorted(numeric_keys)}
    sorted_records: List[Tuple[float, Dict]] = []
    for rec in records:
        step_val = rec.get("step")
        if step_val is None or step_val == "":
            continue
        try:
            step = float(step_val)
        except (TypeError, ValueError):
            continue
        sorted_records.append((step, rec))

    for step, rec in sorted(sorted_records, key=lambda item: item[0]):
        steps.append(step)
        for key in series.keys():
            value = rec.get(key)
            series[key].append(float(value) if _is_number(value) else None)

    return steps, series


def _display_label(key: str) -> str:
    label = key
    if label.startswith("resource/"):
        label = label[len("resource/"):]
    if label.startswith("loss/"):
        label = label[len("loss/"):]
    return label.replace("_", " ").title()


def _series_values(series: Dict[str, List[float]], keys: List[str]) -> List[float]:
    values: List[float] = []
    for key in keys:
        values.extend([value for value in series.get(key, []) if value is not None])
    return values


def _should_use_log_scale(series: Dict[str, List[float]], keys: List[str]) -> bool:
    values = _series_values(series, keys)
    positive_values = [value for value in values if value > 0]
    if len(positive_values) != len(values) or not positive_values:
        return False
    min_value = min(positive_values)
    max_value = max(positive_values)
    return min_value > 0 and (max_value / min_value) >= 100


def _ordered_loss_keys(keys: List[str]) -> List[str]:
    if "loss" in keys and "loss/loss" in keys:
        keys = [key for key in keys if key != "loss/loss"]
    return sorted(keys, key=lambda key: (key != "loss", key))


def _resource_groups(keys: List[str]) -> Dict[str, List[str]]:
    percent_keys = [
        key
        for key in keys
        if key.endswith("_percent") or "utilization_percent" in key or key.endswith("_load_percent")
    ]
    memory_keys = [key for key in keys if key.endswith("_mb")]
    other_keys = [key for key in keys if key not in set(percent_keys + memory_keys)]
    return {
        "percent": sorted(percent_keys),
        "memory_mb": sorted(memory_keys),
        "other": sorted(other_keys),
    }


def _resource_axis_title(key: str) -> str:
    if key.endswith("_percent") or "utilization_percent" in key or key.endswith("_load_percent"):
        return "Percent"
    if key.endswith("_mb"):
        return "Megabytes"
    return "Value"


def _is_flat_series(values: List[float], axis_title: str) -> bool:
    numeric_values = [value for value in values if value is not None]
    if len(numeric_values) < 2:
        return True
    min_value = min(numeric_values)
    max_value = max(numeric_values)
    value_range = max_value - min_value
    if value_range == 0:
        return True
    if axis_title == "Percent":
        return value_range < 2.0
    if axis_title == "Megabytes":
        return value_range < 64 or (max_value != 0 and abs(value_range / max_value) < 0.01)
    return max_value != 0 and abs(value_range / max_value) < 0.01


def _filter_variable_resource_keys(series: Dict[str, List[float]], resource_keys: List[str]) -> Tuple[List[str], List[str]]:
    kept: List[str] = []
    removed: List[str] = []
    for key in sorted(resource_keys):
        values = [value for value in series.get(key, []) if value is not None]
        if _is_flat_series(values, _resource_axis_title(key)):
            removed.append(key)
        else:
            kept.append(key)
    return kept, removed


def _add_line_traces(fig, steps: List[float], series: Dict[str, List[float]], keys: List[str], row: Optional[int] = None) -> None:
    for key in keys:
        kwargs = {
            "x": steps,
            "y": series[key],
            "mode": "lines",
            "name": _display_label(key),
            "connectgaps": False,
        }
        if row is None:
            fig.add_trace(go.Scatter(**kwargs))
        else:
            fig.add_trace(go.Scatter(**kwargs), row=row, col=1)


def _make_line_figure(
    steps: List[float],
    series: Dict[str, List[float]],
    keys: List[str],
    title: str,
    yaxis_title: str,
    *,
    yaxis_type: Optional[str] = None,
    y_tickformat: Optional[str] = None,
):
    fig = go.Figure()
    _add_line_traces(fig, steps, series, keys)
    fig.update_layout(
        title=title,
        xaxis_title="Training step",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        template="plotly_dark",
        legend_title="Metric",
        margin={"l": 72, "r": 24, "t": 64, "b": 64},
    )
    fig.update_xaxes(rangeslider={"visible": True})
    if yaxis_type:
        fig.update_yaxes(type=yaxis_type)
    if y_tickformat:
        fig.update_yaxes(tickformat=y_tickformat)
    return fig


def _make_resource_figure(steps: List[float], series: Dict[str, List[float]], resource_keys: List[str]):
    sections = [(_display_label(key), [key], _resource_axis_title(key)) for key in resource_keys]
    if not sections:
        return None

    fig = make_subplots(
        rows=len(sections),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[section[0] for section in sections],
    )
    for row_index, (_, keys, y_title) in enumerate(sections, start=1):
        _add_line_traces(fig, steps, series, keys, row=row_index)
        fig.update_yaxes(title_text=y_title, row=row_index, col=1)
    fig.update_xaxes(title_text="Training step", row=len(sections), col=1, rangeslider={"visible": True})
    fig.update_layout(
        title="Resource Utilization",
        hovermode="x unified",
        template="plotly_dark",
        legend_title="Metric",
        height=max(360, 320 * len(sections)),
        margin={"l": 72, "r": 24, "t": 88, "b": 64},
    )
    return fig


def _make_single_resource_figure(steps: List[float], series: Dict[str, List[float]], key: str):
    return _make_line_figure(
        steps,
        series,
        [key],
        _display_label(key),
        _resource_axis_title(key),
    )


def _write_plotly_html(fig, output_path: str, otel_url: Optional[str], page_title: str = "Training Report") -> None:
    if not _PLOTLY_AVAILABLE:
        return
    if not otel_url:
        pio.write_html(fig, output_path, include_plotlyjs="cdn", full_html=True, auto_open=False)
        return
    plot_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
    safe_title = html.escape(page_title)
    safe_url = html.escape(otel_url)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{safe_title}</title>\n"
            "</head>\n"
            "<body>\n"
            f"<p>OpenTelemetry dashboard: <a href=\"{safe_url}\" target=\"_blank\" rel=\"noopener\">{safe_url}</a></p>\n"
            f"{plot_html}\n"
            "</body>\n"
            "</html>\n"
        )


def _write_basic_html_report(title: str, png_filename: str, output_path: str, otel_url: Optional[str]) -> None:
    safe_title = html.escape(title)
    safe_png = html.escape(png_filename)
    otel_link = ""
    if otel_url:
        safe_url = html.escape(otel_url)
        otel_link = f"<p>OpenTelemetry dashboard: <a href=\"{safe_url}\" target=\"_blank\" rel=\"noopener\">{safe_url}</a></p>\n"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            f"<meta charset=\"utf-8\">\n<title>{safe_title}</title>\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<style>body{font-family:Arial,sans-serif;margin:24px;} img{max-width:100%;height:auto;border:1px solid #ddd;}</style>\n"
            "</head>\n"
            "<body>\n"
            f"<h1>{safe_title}</h1>\n"
            f"{otel_link}"
            "<p>Plotly is not installed in this environment, so this report uses a static PNG preview.</p>\n"
            f"<img src=\"{safe_png}\" alt=\"{safe_title} report\">\n"
            "</body>\n"
            "</html>\n"
        )


def _load_otel_metadata(metrics_dir: str) -> dict:
    otel_path = os.path.join(metrics_dir, "otel_run.json")
    if not os.path.exists(otel_path):
        return {}
    try:
        with open(otel_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_matplotlib_png(
    steps: List[float],
    series: Dict[str, List[float]],
    keys: List[str],
    title: str,
    output_path: str,
    y_label: str = "value",
    log_y: bool = False,
) -> None:
    if not steps or not keys:
        return

    plt.figure(figsize=(10, 5))
    for key in keys:
        values = series.get(key, [])
        if len(values) != len(steps):
            continue
        filtered = [(step, val) for step, val in zip(steps, values) if val is not None]
        if not filtered:
            continue
        x_vals, y_vals = zip(*filtered)
        plt.plot(x_vals, y_vals, label=key)
    if not plt.gca().lines:
        plt.close()
        return
    plt.title(title)
    plt.xlabel("Training step")
    plt.ylabel(y_label)
    if log_y:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _write_resource_png(
    steps: List[float],
    series: Dict[str, List[float]],
    resource_keys: List[str],
    output_path: str,
) -> None:
    sections = [(_display_label(key), [key], _resource_axis_title(key)) for key in resource_keys]
    if not steps or not sections:
        return

    fig, axes = plt.subplots(len(sections), 1, figsize=(11, max(4, 4.0 * len(sections))), sharex=True)
    if len(sections) == 1:
        axes = [axes]

    any_lines = False
    for axis, (title, keys, y_label) in zip(axes, sections):
        for key in keys:
            values = series.get(key, [])
            if len(values) != len(steps):
                continue
            filtered = [(step, val) for step, val in zip(steps, values) if val is not None]
            if not filtered:
                continue
            x_vals, y_vals = zip(*filtered)
            axis.plot(x_vals, y_vals, label=_display_label(key))
            any_lines = True
        axis.set_title(title)
        axis.set_ylabel(y_label)
        axis.legend()
        axis.grid(True, alpha=0.25)

    if not any_lines:
        plt.close(fig)
        return

    axes[-1].set_xlabel("Training step")
    fig.suptitle("Resource Utilization")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_dashboard_html(
    output_path: str,
    figures: List[Tuple[str, object]],
    png_paths: List[str],
    otel_url: Optional[str],
    metrics_summary: Dict[str, object],
    omitted_resources: Optional[List[str]] = None,
) -> None:
    safe_otel = ""
    if otel_url:
        safe_url = html.escape(otel_url)
        safe_otel = f'<a class="link" href="{safe_url}" target="_blank" rel="noopener">OpenTelemetry dashboard</a>'

    summary_items = []
    for label, value in metrics_summary.items():
        summary_items.append(
            f"<div class=\"summary-card\"><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></div>"
        )
    summary_html = "\n".join(summary_items)

    if _PLOTLY_AVAILABLE and figures:
        chart_sections = []
        for index, (title, fig) in enumerate(figures):
            include_plotlyjs = "cdn" if index == 0 else False
            plot_html = pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=False)
            chart_sections.append(
                f"<section class=\"chart-section\"><h2>{html.escape(title)}</h2>{plot_html}</section>"
            )
        body = "\n".join(chart_sections)
    else:
        image_sections = "\n".join(
            "<section class=\"chart-section\">"
            f"<h2>{html.escape(os.path.splitext(os.path.basename(path))[0].replace('_', ' ').title())}</h2>"
            f"<img src=\"{html.escape(os.path.basename(path))}\" alt=\"{html.escape(os.path.basename(path))}\" />"
            "</section>"
            for path in png_paths
        )
        body = (
            "<div class=\"muted\">Plotly is not installed in this environment, so this dashboard uses static PNG previews.</div>"
            f"{image_sections}"
        )

    omitted_html = ""
    if omitted_resources:
        omitted = ", ".join(_display_label(key) for key in omitted_resources)
        omitted_html = f"<div class=\"muted\">Hidden flat resource metrics: {html.escape(omitted)}</div>"

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<title>Training Metrics Dashboard</title>\n"
            "<style>\n"
            "body{margin:0;background:#0b1020;color:#e5e7eb;font-family:Arial,sans-serif;}\n"
            "main{max-width:1280px;margin:0 auto;padding:28px;}\n"
            "header{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:20px;}\n"
            "h1{font-size:28px;margin:0 0 8px;} h2{font-size:18px;margin:0 0 14px;}\n"
            ".muted{color:#9ca3af;font-size:13px;} .link{color:#93c5fd;text-decoration:none;} .link:hover{text-decoration:underline;}\n"
            ".summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0;}\n"
            ".summary-card{border:1px solid #1f2937;background:#111827;border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:4px;}\n"
            ".summary-card span{font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em;} .summary-card strong{font-size:16px;}\n"
            ".chart-section{border:1px solid #1f2937;background:#111827;border-radius:8px;padding:16px;margin:16px 0;}\n"
            "img{display:block;max-width:100%;height:auto;border:1px solid #1f2937;background:#030712;}\n"
            "</style>\n"
            "</head>\n"
            "<body><main>\n"
            "<header><div><h1>Training Metrics Dashboard</h1><div class=\"muted\">Compiled from local metrics logs.</div></div>"
            f"<div>{safe_otel}</div></header>\n"
            f"<div class=\"summary\">{summary_html}</div>\n"
            f"{omitted_html}\n"
            f"{body}\n"
            "</main></body></html>\n"
        )


def generate_reports(metrics_dir: str) -> Dict[str, List[str]]:
    """Generate HTML (plotly) and PNG (matplotlib) reports from metrics."""
    metrics_jsonl = os.path.join(metrics_dir, "metrics.jsonl")
    metrics_csv = os.path.join(metrics_dir, "metrics.csv")
    otel_metadata = _load_otel_metadata(metrics_dir)
    otel_url = otel_metadata.get("dashboard_url") if isinstance(otel_metadata, dict) else None

    records = _load_jsonl(metrics_jsonl)
    if not records:
        records = _load_csv(metrics_csv)

    if not records:
        return {"html": [], "png": []}

    steps, series = _build_series(records)
    if not steps:
        return {"html": [], "png": []}

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(metrics_dir, "reports", timestamp)
    os.makedirs(report_dir, exist_ok=True)

    html_paths: List[str] = []
    png_paths: List[str] = []
    dashboard_figures: List[Tuple[str, object]] = []

    # Loss plots
    loss_keys = _ordered_loss_keys([key for key in series.keys() if key.startswith("loss/") or key == "loss"])
    if loss_keys:
        loss_log_y = _should_use_log_scale(series, loss_keys)
        png_path = os.path.join(report_dir, "loss.png")
        _write_matplotlib_png(steps, series, loss_keys, "Loss", png_path, "Loss", log_y=loss_log_y)
        png_paths.append(png_path)

        html_path = os.path.join(report_dir, "loss.html")
        if _PLOTLY_AVAILABLE:
            fig = _make_line_figure(
                steps,
                series,
                loss_keys,
                "Loss",
                "Loss",
                yaxis_type="log" if loss_log_y else None,
                y_tickformat=".2e",
            )
            _write_plotly_html(fig, html_path, otel_url, "Loss")
            dashboard_figures.append(("Loss", fig))
        else:
            _write_basic_html_report("Loss", os.path.basename(png_path), html_path, otel_url)
        html_paths.append(html_path)

    # Learning rate
    if "learning_rate" in series:
        lr_log_y = _should_use_log_scale(series, ["learning_rate"])
        png_path = os.path.join(report_dir, "learning_rate.png")
        _write_matplotlib_png(
            steps,
            series,
            ["learning_rate"],
            "Learning Rate",
            png_path,
            "Learning rate",
            log_y=lr_log_y,
        )
        png_paths.append(png_path)

        html_path = os.path.join(report_dir, "learning_rate.html")
        if _PLOTLY_AVAILABLE:
            fig = _make_line_figure(
                steps,
                series,
                ["learning_rate"],
                "Learning Rate",
                "Learning rate",
                yaxis_type="log" if lr_log_y else None,
                y_tickformat=".2e",
            )
            _write_plotly_html(fig, html_path, otel_url, "Learning Rate")
            dashboard_figures.append(("Learning Rate", fig))
        else:
            _write_basic_html_report("Learning Rate", os.path.basename(png_path), html_path, otel_url)
        html_paths.append(html_path)

    # Resource utilization
    resource_keys = [key for key in series.keys() if key.startswith("resource/")]
    if resource_keys:
        resource_keys, removed_resource_keys = _filter_variable_resource_keys(series, resource_keys)
    else:
        removed_resource_keys = []

    if resource_keys:
        png_path = os.path.join(report_dir, "resources.png")
        _write_resource_png(steps, series, resource_keys, png_path)
        png_paths.append(png_path)

        html_path = os.path.join(report_dir, "resources.html")
        if _PLOTLY_AVAILABLE:
            fig = _make_resource_figure(steps, series, resource_keys)
            if fig is not None:
                _write_plotly_html(fig, html_path, otel_url, "Resource Utilization")
                for key in resource_keys:
                    dashboard_figures.append((_display_label(key), _make_single_resource_figure(steps, series, key)))
        else:
            _write_basic_html_report("Resource Utilization", os.path.basename(png_path), html_path, otel_url)
        html_paths.append(html_path)

    dashboard_path = os.path.join(report_dir, "dashboard.html")
    metrics_summary = {
        "Records": len(records),
        "First step": f"{min(steps):g}",
        "Last step": f"{max(steps):g}",
        "Numeric metrics": len(series),
        "Hidden flat resources": len(removed_resource_keys),
    }
    _write_dashboard_html(
        dashboard_path,
        dashboard_figures,
        png_paths,
        otel_url,
        metrics_summary,
        omitted_resources=removed_resource_keys,
    )
    html_paths.insert(0, dashboard_path)

    return {"html": html_paths, "png": png_paths}

