# AI Toolkit Architecture Guide

This repository is best understood as a **Modal-first AI training and generation system**. The code in this repo defines the source architecture, configs, and workers; the large runtime assets are often kept outside commits because they live in Modal volumes or other local-only caches.

If you're trying to understand how everything fits together, start with `run_modal.py`, then follow the config and job layers down into `toolkit/` and `jobs/`.

## High-Level Mental Model

1. A config in `config/` describes a job.
2. `run_modal.py` exposes the Modal + FastAPI runtime and launches jobs.
3. `toolkit/` parses configs and handles model/training behavior.
4. `jobs/` selects the job type (`train`, `generate`, `extension`, etc.).
5. The actual data, training folders, and generated outputs usually live in Modal-backed storage or other runtime directories.

## What Is Source vs. What Is Runtime Data

### Source / architecture files

- `run_modal.py` - main Modal app, API server, training worker, and generation worker
- `config/` - experiment definitions and model presets
- `jobs/` - job orchestration and process wiring
- `toolkit/` - config parsing, model setup, schedulers, losses, samplers, and training helpers
- `extensions_built_in/` - built-in trainer and model extension packages
- `scripts/` - conversion and maintenance utilities
- `ui/` - frontend app; see `ui/README.md`

### Runtime / often-not-committed data

- `datasets/` - local or staged dataset files; may be intentionally excluded
- `modal_output/` - runtime output and mounted workspace data
- `wheelhouse/` - cached wheels used by Modal image builds and dependency bootstrapping
- `output/` - generated artifacts, checkpoints, and evaluation results
- `test_outputs/` - scratch space for experiments and test runs

## Repository Map

| Path | What it is | Why it matters |
|---|---|---|
| `config/` | YAML job definitions | This is where the important Flux/KNHB configs live |
| `jobs/` | Job classes | Maps config `job:` values to actual runtime behavior |
| `toolkit/` | Core engine | Holds config parsing and model architecture logic |
| `extensions_built_in/` | Built-in extension modules | Includes the `sd_trainer` path used by most training configs |
| `run_modal.py` | Modal + API entrypoint | The best starting point for understanding runtime flow |
| `scripts/` | Utility scripts | Helpful for conversion, dataset repair, and model prep |
| `ui/` | Frontend | Separate user interface codebase |
| `datasets/`, `modal_output/`, `wheelhouse/` | Runtime assets | Often excluded because they are large, generated, or environment-specific |

## Important Files to Know

- `run_modal.py`
  - Defines the Modal app, API routes (`/api/train`, `/api/generate`, dataset routes), and worker functions (`main`, `remote_generate`).
  - Also contains the CLI entrypoint for local or direct invocation.
- `requirements.txt`
  - Dependency list for the Python environment.
- `selective_cleanup.py`
  - Helper for removing job metadata and corresponding Modal volume paths.
- `test_api.py`
  - Local test scaffold for the API.
- `log.txt`
  - Useful if you are capturing local debugging output.
- `config/flux2-klein-test.yaml`
  - The main Flux 2 klein training example in this workspace.

## Flux 2 Klein

The Flux 2 klein setup is centered on `config/flux2-klein-test.yaml`.

Key values in that config:

- `job: "train"`
- `config.process[0].type: "sd_trainer"`
- `config.process[0].model.arch: "flux2_klein"`
- `config.process[0].model.name_or_path: "black-forest-labs/FLUX.2-klein-base-4B"`
- `config.process[0].datasets[0].folder_path: "/root/modal_output/datasets/KNHB"`

### Why the paths look unusual

This repo is designed around Modal execution, so many configs use `/root/...` paths. That is normal here.

If you are only reading the repo to understand the architecture, treat those paths as **runtime-mounted locations**, not as a promise that the project is meant to be fully set up locally.

### Where Flux 2 klein logic lives

If you want to understand or extend Flux 2 klein support, check:

- `toolkit/config_modules.py` - model arch definitions include `flux2_klein`
- `toolkit/stable_diffusion_model.py` - architecture-specific model behavior
- `jobs/TrainJob.py` - training job wiring
- `toolkit/job.py` - maps `job:` names to job classes
- `extensions_built_in/sd_trainer/` - training implementation used by `sd_trainer`

## Workflow Summary

### Training flow

1. Pick a config in `config/`
2. `run_modal.py` loads it
3. `toolkit/config.py` normalizes it
4. `jobs/TrainJob.py` and `jobs/process/` execute the training process
5. Output lands in the runtime volume / training folder

### Generation flow

1. A trained model or LoRA is selected
2. `run_modal.py` calls `remote_generate`
3. The worker loads the base model and optional LoRA
4. Images are written to the output volume and returned as file URLs

## Wheelhouse Download (Required For Deploy)

If you are deploying this project to Modal from a fresh clone, you may need to download the prebuilt wheels first.

Download link:
- https://drive.google.com/drive/folders/1RWj8Ps7LWY8h02uQMuLLLw9zynyRP1My?usp=sharing

Important:
- Extract everything into a folder named `wheelhouse` at the repository root.
- Final expected path should be `ai-toolkit/wheelhouse/`.
- `run_modal.py` references this path during Modal image build (`add_local_dir(.../wheelhouse, remote_path="/root/wheels")`), so missing it can break deploy.


## Modal-First Usage Notes

Most of the real execution happens on Modal, so this repo is not primarily a "clone it and run everything locally" setup.

Useful entrypoints:

```cmd
modal deploy run_modal.py
```

For direct CLI-style execution of a config:

```cmd
python run_modal.py config\flux2-klein-test.yaml -n KNHB-flux2-klein
```

## What You Usually Do Not Commit

These folders are intentionally absent from commits because they are large, generated, or environment-specific:

- `datasets/`
- `modal_output/`
- `wheelhouse/`

That is normal for this project structure.

## Small Local Dev Note

If you do want to inspect the Python environment locally, the common setup pattern is:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

But the main purpose of this repository is still to document and support the Modal-backed workflow.

## Extra Notes

- There is a separate frontend guide at `ui/README.md`.
- Config paths may need adjustment depending on whether you are looking at Modal-mounted storage or local files.
- `run_modal.py` is the best first file to inspect if you want to understand the system end-to-end.

## Training Metrics & Reports

During training, metrics are written to `metrics.csv` and `metrics.jsonl` inside each training folder. Resource snapshots (CPU/GPU) are captured alongside loss and learning-rate values. HTML and PNG reports are generated at the end of training under `reports/`.

Generate reports manually:

```cmd
python scripts\generate_training_report.py --metrics-dir <training_folder>
```

## OpenTelemetry Metrics

Modal native OpenTelemetry is the primary New Relic integration for Modal
function logs and container/platform metrics. Configure it in the Modal
workspace, not in the training YAML:

- Modal OTEL push URL: `https://otlp.nr-data.net`
- Modal OpenTelemetry Secret key: `OTEL_HEADER_api-key`
- Secret value: your New Relic license key

Training loss, learning-rate, and resource snapshots are still written locally
to `metrics.csv` and `metrics.jsonl`, and reports are generated from those files.
The `logging.use_otel` setting only controls custom training metric export
attempts. If no `OTEL_EXPORTER_OTLP_ENDPOINT` is present, custom OTEL export is
disabled and local metrics continue to work.

Custom training metrics can be exported directly to New Relic from the Modal
training container. Create a Modal secret named `newrelic-otlp` with:

- `OTEL_EXPORTER_OTLP_HEADERS=api-key=<New Relic ingest license key>`
- Optional override: `OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.nr-data.net`
- Optional override: `OTEL_SERVICE_NAME=ai-toolkit`

The secret name can be overridden at deploy time with
`NEW_RELIC_OTLP_SECRET_NAME`. The training image sets the non-secret defaults
`OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.nr-data.net`,
`OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`, and
`OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta`.

Environment variables honored by the custom training metric logger and report
generator:
- `OTEL_EXPORTER_OTLP_ENDPOINT`
- `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`
- `OTEL_EXPORTER_OTLP_HEADERS`
- `OTEL_EXPORTER_OTLP_METRICS_HEADERS`
- `OTEL_SERVICE_NAME`
- `OTEL_DASHBOARD_URL`

The YAML fields `otel_exporter_endpoint` and `otel_exporter_headers` are legacy
direct-export overrides for local development or non-Modal deployments. For
Modal, prefer the native workspace integration and Modal-provided collector env
vars once custom metrics/spans are enabled for the workspace.

Quick smoke test (writes metrics + reports into `test_outputs/otel_smoke/`):

```bat
python scripts\otel_smoke.py
```
