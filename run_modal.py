#!/usr/bin/env python3
"""
run_modal.py - wheelhouse-first, hardened backend

- Uses local wheelhouse first (fallback to PyPI)
- Keeps Python 3.12 to match cp312 wheels in your wheelhouse
- Hardened: safe_commit, filename sanitation, streaming uploads, job_store per-job keys,
  generate saves images to mounted volume and returns /api/files URLs (no base64 payloads)
- Authentication REMOVED for local/public use.
"""

import os
import sys
import io
import shutil
import argparse
import traceback
import uuid
import json
import hashlib
import csv
import zipfile
from collections import deque

import oyaml as yaml
from pathlib import Path
from typing import List, Optional, Any
from datetime import datetime
import logging
import re
import threading

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_, **__):
        return False

load_dotenv()

import modal
from modal import asgi_app

# FastAPI imports
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query, Request
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer # Removed
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Modal imports this module from /root while the repository is mounted under
# /root/ai-toolkit for the ML image. Make that package root available before
# importing shared toolkit modules.
REMOTE_AITOOLKIT_PATH = "/root/ai-toolkit"
if Path(REMOTE_AITOOLKIT_PATH).is_dir() and REMOTE_AITOOLKIT_PATH not in sys.path:
    sys.path.insert(0, REMOTE_AITOOLKIT_PATH)

from toolkit.generation_model_resolver import (
    infer_generation_architecture,
    list_generation_models,
    load_and_activate_lora,
    public_generation_model,
    resolve_generation_model,
)

# ------------------------------------------------------------
# Configuration & paths
# ------------------------------------------------------------
MOUNT_DIR = "/root/modal_output"
CACHE_DIR = "/root/.cache/huggingface"
DEFAULT_GENERATION_BASE_MODEL = "black-forest-labs/FLUX.1-dev"
DIFFUSERS_VERSION = "0.38.0"
TRANSFORMERS_VERSION = "4.57.6"
TORCHAO_VERSION = "0.16.0"
LOCAL_AITOOLKIT_PATH = Path(__file__).parent
LOCAL_WHEELHOUSE_PATH = Path(__file__).parent / "wheelhouse"

# ------------------------------------------------------------
# Logging & helpers
# ------------------------------------------------------------
logger = logging.getLogger("run_modal")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

ALLOWED_SERVE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".txt", ".json", ".mp4", ".html", ".zip"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB cap per file
_filename_sanitize_re = re.compile(r"[^A-Za-z0-9._-]")


def _split_env_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_otel_env_summary() -> dict:
    return {
        "endpoint": os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "metrics_endpoint": os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"),
        "headers": "set" if (
            os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
            or os.environ.get("OTEL_EXPORTER_OTLP_METRICS_HEADERS")
        ) else "missing",
        "protocol": os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL"),
        "metrics_temporality": os.environ.get("OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE"),
        "dashboard_url": os.environ.get("OTEL_DASHBOARD_URL"),
        "service_name": os.environ.get("OTEL_SERVICE_NAME"),
    }


def safe_filename(name: str, max_len: int = 200) -> str:
    name = Path(name).name
    name = _filename_sanitize_re.sub("_", name)
    if len(name) > max_len:
        name = name[:max_len]
    return name


def iso_now() -> str:
    return datetime.now().isoformat() + "Z"


def safe_commit(volume) -> bool:
    try:
        volume.commit()
    except Exception as e:
        logger.warning("volume.commit() failed (non-fatal): %s", e)
        return False
    return True


def _normalize_generation_base_model(base_model: Optional[str]) -> str:
    if isinstance(base_model, str) and base_model.strip():
        return base_model.strip()
    return DEFAULT_GENERATION_BASE_MODEL


def _load_generation_pipeline(
        pipeline_cls: Any,
        base_model: Optional[str],
        torch_dtype: Any,
        hf_token: Optional[str] = None,
        cache_dir: str = CACHE_DIR,
        hf_volume_obj: Any = None,
):
    normalized_base_model = _normalize_generation_base_model(base_model)

    try:
        logger.info("Attempting to load %s from local cache...", normalized_base_model)
        pipe = pipeline_cls.from_pretrained(
            normalized_base_model,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch_dtype,
        )
        logger.info("Loaded successfully from cache.")
        return pipe
    except Exception:
        logger.info("Model not found in cache (or incomplete). Downloading %s...", normalized_base_model)

    try:
        pipe = pipeline_cls.from_pretrained(
            normalized_base_model,
            cache_dir=cache_dir,
            local_files_only=False,
            token=hf_token,
            torch_dtype=torch_dtype,
        )
        if hf_volume_obj is not None:
            hf_volume_obj.commit()
        logger.info("Download complete and volume committed.")
        return pipe
    except Exception as download_error:
        logger.error("Failed to download model: %s", download_error)
        raise RuntimeError(f"Could not load or download model {normalized_base_model}. Check token/internet.")


def _call_generation_pipeline(
        pipe: Any,
        prompt: str,
        generator: Any,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 20,
):
    return pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )


# ------------------------------------------------------------
# Modal images: wheelhouse-first, Python 3.12
# ------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1", "libglib2.0-0", "pkg-config", "build-essential", "gfortran",
        "cmake", "ninja-build", "libopenblas-dev", "liblapack-dev",
        "ca-certificates", "git", "curl", "wget", "libatlas-base-dev",
        "libblas-dev", "libsentencepiece-dev", "libprotobuf-dev", "protobuf-compiler",
        "python3-dev"
    )
    .add_local_dir(LOCAL_WHEELHOUSE_PATH, remote_path="/root/wheels", copy=True)
    .run_commands(
        "echo '=== Upgrade pip/setuptools/wheel/build ==='",
        "python -m pip install --upgrade pip setuptools wheel build",
        "python -m pip --version",
        "python -m pip debug --verbose | true",
        "echo 'Wheelhouse contents:'",
        "ls -lah /root/wheels | true",
    )
    .run_commands(
        "echo '=== Write constraints (binary pins) ==='",
        "echo 'numpy==1.26.4' > /root/constraints.txt",
        "echo 'scipy==1.14.1' >> /root/constraints.txt",
        "echo 'pillow==12.0.0' >> /root/constraints.txt",
        "echo 'torch==2.9.0' >> /root/constraints.txt",
    )
    .run_commands(
        "echo '=== Preinstall binary-critical packages from wheelhouse (prefer-binary) ==='",
        "python -m pip install --prefer-binary --find-links /root/wheels --constraint /root/constraints.txt "
        "numpy==1.26.4 scipy==1.14.1 pillow==12.0.0 | true",
        "python -c \"import numpy, scipy, PIL; print('preinstalled:', numpy.__version__, scipy.__version__, PIL.__version__)\" | true",
    )
    # fixed echo line here
    .run_commands(
        "echo '=== Install torch 2.9.0 from wheelhouse ==='",
        "python -m pip install --prefer-binary --find-links /root/wheels --constraint /root/constraints.txt torch==2.9.0",
        "python -c \"import torch; print('torch version:', torch.__version__)\"",
    )
    .run_commands(
        "echo '=== Install wheels from wheelhouse (no-deps) ==='",
        "python -m pip install --no-index --find-links /root/wheels --prefer-binary --no-deps /root/wheels/*.whl || true",
    )
    .run_commands(
        "echo '=== Installing application packages (wheelhouse-first; PyPI fallback) ==='",
        "python -m pip install --prefer-binary --find-links /root/wheels --constraint /root/constraints.txt "
        f"transformers=={TRANSFORMERS_VERSION} torchao=={TORCHAO_VERSION} python-dotenv accelerate ftfy safetensors albumentations lycoris-lora timm einops "
        "opencv-python-headless huggingface_hub peft lpips hf_transfer flatten_json pyyaml oyaml tensorboard "
        "toml albucore pydantic omegaconf k-diffusion controlnet_aux optimum-quanto python-slugify open_clip_torch "
        f"bitsandbytes pytorch_fid sentencepiece pytorch-wavelets matplotlib diffusers=={DIFFUSERS_VERSION} fastapi[standard] "
        "python-multipart modal psutil plotly opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp "
        "opentelemetry-semantic-conventions"
    )
    .env({
        "HUGGINGFACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
        "OTEL_EXPORTER_OTLP_ENDPOINT": os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        "OTEL_EXPORTER_OTLP_PROTOCOL": os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"),
        "OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE": os.environ.get(
            "OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE",
            "delta",
        ),
        "AI_TOOLKIT_ENABLE_CUSTOM_OTEL_EXPORT": os.environ.get("AI_TOOLKIT_ENABLE_CUSTOM_OTEL_EXPORT", "false"),
    })
    .add_local_dir(
        LOCAL_AITOOLKIT_PATH,
        remote_path="/root/ai-toolkit",
        ignore=[".git/**", "**/.idea/**", "**/__pycache__/**", "**/*.pyc", ".venv/**", "web-ui/**", "ui/**"],
    )
)

# Lightweight image for the API app (also uses wheelhouse for convenience)
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(LOCAL_WHEELHOUSE_PATH, remote_path="/root/wheels", copy=True)
    .add_local_dir(
        LOCAL_AITOOLKIT_PATH / "toolkit",
        remote_path="/root/toolkit",
        ignore=["**/__pycache__/**", "**/*.pyc"],
        copy=True,
    )
    .apt_install("ca-certificates", "git", "curl", "wget")
    .run_commands(
        "echo '=== Ensure pip + build tools for web image ==='",
        "python -m pip install --upgrade pip setuptools wheel build",
    )
    .run_commands(
        "echo '=== Install web dependencies (wheelhouse-first) ==='",
        "python -m pip install --prefer-binary --find-links /root/wheels "
        "fastapi[standard] modal python-dotenv oyaml python-multipart matplotlib plotly || true"
    )
)

# Modal app
app = modal.App(
    name="flex-lora-training",
    image=image,
    volumes={MOUNT_DIR: modal.Volume.from_name("flux-lora-models", create_if_missing=True),
             CACHE_DIR: modal.Volume.from_name("hf-cache", create_if_missing=True)},
)
APP_NAME = app.name

# Volumes & persistent stores
model_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
job_store = modal.Dict.from_name("user-job-store", create_if_missing=True)
NEW_RELIC_OTLP_SECRET_NAME = os.environ.get("NEW_RELIC_OTLP_SECRET_NAME", "newrelic-otlp")
new_relic_otlp_secret = modal.Secret.from_name(NEW_RELIC_OTLP_SECRET_NAME)


# ------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    num_samples: int = 1
    model_name: Optional[str] = None
    base_model: Optional[str] = DEFAULT_GENERATION_BASE_MODEL
    hf_token: Optional[str] = None
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    images: List[str]
    status: str
    seed: int
    model_name: Optional[str] = None
    base_model: str
    checkpoint: Optional[str] = None
    adapter_loaded: bool = False


class TrainRequest(BaseModel):
    config: dict
    recover: Optional[bool] = False
    name: Optional[str] = None
    hf_token: Optional[str] = None


class TrainResponse(BaseModel):
    job_id: str
    status: str


class DatasetCreateRequest(BaseModel):
    name: str


class DatasetDeleteRequest(BaseModel):
    name: str


class CaptionRequest(BaseModel):
    path: str
    caption: str
    caption_short: Optional[str] = None
    caption_ext: Optional[str] = None


class ProgressPayload(BaseModel):
    step: Optional[int] = None
    total: Optional[int] = None
    percent: Optional[float] = None
    phase: Optional[str] = None
    message: Optional[str] = None
    info: Optional[dict] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    config_name: Optional[str] = None
    created_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: Optional[ProgressPayload] = None
    ended_at: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: List[JobStatus]


class JobConfigResponse(BaseModel):
    config_yaml: Optional[str] = None


# ------------------------------------------------------------
# Authentication stub (Removed)
# ------------------------------------------------------------
async def get_current_user_id():
    # Always return a default user, effectively disabling auth
    return "default_user"


# ------------------------------------------------------------
# Job store helpers (per-job key storage)
# ------------------------------------------------------------
def _put_job_record(job_id: str, record: dict):
    try:
        job_store.put(job_id, record)
    except Exception as e:
        logger.exception("Failed to put job_record: %s", e)
        raise


def _get_job_record(job_id: str) -> Optional[dict]:
    try:
        return job_store.get(job_id)
    except Exception:
        return None


def _delete_job_record(job_id: str):
    try:
        job_store.pop(job_id)
    except KeyError:
        return
    except AttributeError:
        try:
            del job_store[job_id]
        except KeyError:
            return
    except Exception as e:
        logger.warning("Failed to remove job %s from job_store: %s", job_id, e)


def _user_jobs_list(user_id: str) -> List[dict]:
    try:
        keys = job_store.keys()
    except Exception:
        keys = []
    jobs = []
    for k in keys:
        rec = job_store.get(k)
        if not rec:
            continue
        if rec.get("user_id") == user_id:
            jobs.append(rec)
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs


# ------------------------------------------------------------
# Job status utility for worker
# ------------------------------------------------------------
def _current_call_id() -> Optional[str]:
    """Return the current Modal function call id inside a worker, if available."""
    try:
        call = modal.FunctionCall.current()
        return call.object_id
    except Exception:
        return None


# ------------------------------------------------------------
# FastAPI app & CORS
# ------------------------------------------------------------
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8788",
    "http://127.0.0.1:8787"
]
EXTRA_ALLOWED_ORIGINS = _split_env_list(os.environ.get("CORS_EXTRA_ORIGINS", ""))
ALLOW_ORIGIN_REGEX = os.environ.get(
    "CORS_ORIGIN_REGEX",
    r"https://.*\.(pages\.dev|workers\.dev)"
)

api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ALLOWED_ORIGINS + EXTRA_ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/api")
async def root():
    return {"status": "ok", "service": "AI Toolkit Backend"}


# ------------------------------------------------------------
# Helpers: user paths
# ------------------------------------------------------------
def get_user_dataset_path(user_id: str) -> Path:
    path = Path(MOUNT_DIR) / "datasets" / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_user_training_path(user_id: str) -> Path:
    path = Path(MOUNT_DIR) / "trainings" / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_job_training_dir(job_record: dict, user_id: str) -> Path:
    training_folder = job_record.get("training_folder")
    if training_folder:
        return Path(training_folder)
    folder_name = job_record.get("config_name") or job_record.get("job_id")
    return get_user_training_path(user_id) / folder_name


def _resolve_metrics_dir(job_record: dict, user_id: str) -> Path:
    training_dir = _get_job_training_dir(job_record, user_id)
    if (training_dir / "metrics.jsonl").exists() or (training_dir / "metrics.csv").exists():
        return training_dir
    if (training_dir / "reports").exists():
        return training_dir

    try:
        for child in training_dir.iterdir():
            if not child.is_dir() or child.name.startswith("_"):
                continue
            if (child / "metrics.jsonl").exists() or (child / "metrics.csv").exists():
                return child
            if (child / "reports").exists():
                return child
    except FileNotFoundError:
        pass

    return training_dir


def _load_metrics_jsonl(metrics_jsonl: Path, limit: int) -> List[dict]:
    records: deque[dict] = deque(maxlen=limit)
    with open(metrics_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                records.append(rec)
    return list(records)


def _load_metrics_csv(metrics_csv: Path, limit: int) -> List[dict]:
    records: deque[dict] = deque(maxlen=limit)
    with open(metrics_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(dict(row))
    return list(records)


def _load_metrics_records(metrics_dir: Path, limit: int) -> List[dict]:
    metrics_jsonl = metrics_dir / "metrics.jsonl"
    metrics_csv = metrics_dir / "metrics.csv"
    if metrics_jsonl.exists():
        return _load_metrics_jsonl(metrics_jsonl, limit)
    if metrics_csv.exists():
        return _load_metrics_csv(metrics_csv, limit)
    return []


def _extract_logging_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except Exception:
        return {}

    try:
        logging_cfg = cfg["config"]["process"][0].get("logging") or {}
    except (KeyError, IndexError, TypeError, AttributeError):
        logging_cfg = {}

    return {
        "use_otel": bool(logging_cfg.get("use_otel", True)),
        "otel_service_name": logging_cfg.get("otel_service_name"),
        "otel_exporter_endpoint": logging_cfg.get("otel_exporter_endpoint"),
        "otel_exporter_headers": logging_cfg.get("otel_exporter_headers"),
        "otel_dashboard_url": logging_cfg.get("otel_dashboard_url"),
        "track_resources": logging_cfg.get("track_resources", True),
        "log_every": logging_cfg.get("log_every"),
        "resource_log_every": logging_cfg.get("resource_log_every"),
        "resource_log_seconds": logging_cfg.get("resource_log_seconds"),
        "write_metrics_csv": logging_cfg.get("write_metrics_csv", True),
        "write_metrics_jsonl": logging_cfg.get("write_metrics_jsonl", True),
    }


def _get_latest_report_dir(metrics_dir: Path) -> Optional[Path]:
    reports_root = metrics_dir / "reports"
    if not reports_root.exists():
        return None
    dirs = [d for d in reports_root.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.name)
    return dirs[-1]


def _load_otel_metadata(metrics_dir: Path) -> dict:
    otel_path = metrics_dir / "otel_run.json"
    if not otel_path.exists():
        return {}
    try:
        with open(otel_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _relativize_paths(paths: List[Path]) -> List[str]:
    rel_paths = []
    for path in paths:
        try:
            rel_paths.append(str(path.relative_to(Path(MOUNT_DIR))))
        except Exception:
            continue
    return rel_paths


# ------------------------------------------------------------
# Helpers: dataset paths
# ------------------------------------------------------------
def _normalize_dataset_paths(cfg: dict, user_id: str) -> None:
    """Rewrite dataset folder_path entries to the correct per-user path under MOUNT_DIR."""
    base_root = Path(MOUNT_DIR) / "datasets" / user_id

    try:
        datasets = cfg["config"]["process"][0].get("datasets", [])
    except (KeyError, IndexError, TypeError):
        return

    for ds in datasets:
        if not isinstance(ds, dict):
            continue
        raw = ds.get("folder_path")
        if not raw:
            continue

        # Use only the last segment as the dataset name
        name = Path(str(raw)).name
        safe_name = safe_filename(name)

        fixed = base_root / safe_name
        ds["folder_path"] = str(fixed)


# ------------------------------------------------------------
# Jobs endpoints
# ------------------------------------------------------------
@api.get("/api/jobs", response_model=JobListResponse)
async def get_jobs(user_id: str = Depends(get_current_user_id)):
    jobs = _user_jobs_list(user_id)
    return JobListResponse(jobs=[JobStatus(**j) for j in jobs])


@api.get("/api/job-status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str, user_id: str = Depends(get_current_user_id)):
    job_record = _get_job_record(job_id)
    if not job_record:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_record.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Use the explicitly stored Modal ID if available, otherwise fallback to job_id (back-compat)
    modal_id = job_record.get("modal_id", job_id)

    # If job is started/running, try to refresh status from Modal
    if job_record.get("status") in ["started", "running"]:
        try:
            call = modal.FunctionCall.from_id(modal_id)
            try:
                result = call.get(timeout=0)
                job_record["status"] = "completed"
                job_record["result"] = result
            except modal.TimeoutError:
                job_record["status"] = "running"
            except modal.CancelledError:
                job_record["status"] = "canceled"
                job_record["error"] = "Job was canceled from Modal."
            except Exception as e:
                job_record["status"] = "failed"
                job_record["error"] = str(e)
            _put_job_record(job_id, job_record)
        except Exception as e:
            logger.warning("Could not refresh job status for %s (modal_id: %s): %s", job_id, modal_id, e)

    return JobStatus(**job_record)


# ------------------------------------------------------------
# Pausing job
# ------------------------------------------------------------
@api.post("/api/jobs/{job_id}/pause")
async def pause_job_endpoint(job_id: str, user_id: str = Depends(get_current_user_id)):
    job_record = _get_job_record(job_id)
    if not job_record:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_record.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    status = job_record.get("status")
    if status not in {"running", "started"}:
        raise HTTPException(status_code=400, detail=f"Cannot pause job in status {status}")

    # Mark as pausing to avoid concurrent actions
    job_record["status"] = "pausing"
    _put_job_record(job_id, job_record)

    modal_id = job_record.get("modal_id", job_id)

    try:
        try:
            modal.FunctionCall.from_id(modal_id).cancel()
        except Exception as cancel_err:
            logger.warning("Failed to cancel job %s (modal_id: %s) during pause: %s", job_id, modal_id, cancel_err)

        # Mark as paused and record that we can recover
        job_record["status"] = "paused"
        job_record["can_recover"] = True
        job_record["ended_at"] = iso_now()
        _put_job_record(job_id, job_record)
        return {"status": "paused", "job_id": job_id}
    except Exception as e:
        logger.error("Failed to pause job %s: %s", job_id, e)
        # Best-effort rollback
        job_record["status"] = status
        _put_job_record(job_id, job_record)
        raise HTTPException(status_code=500, detail="Failed to pause job")


# ------------------------------------------------------------
# Resuming job
# ------------------------------------------------------------
@api.post("/api/jobs/{job_id}/resume")
async def resume_job_endpoint(job_id: str, user_id: str = Depends(get_current_user_id)):
    job_record = _get_job_record(job_id)
    if not job_record:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_record.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    if job_record.get("status") != "paused":
        raise HTTPException(status_code=400, detail=f"Cannot resume job in status {job_record.get('status')}")

    config_name = job_record.get("config_name")
    if not config_name:
        raise HTTPException(status_code=400, detail="Job is missing config_name")

    # Find the config file
    config_path = get_user_training_path(user_id) / "_configs" / f"{safe_filename(config_name)}.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config file not found for this job")

    # Generate new ID
    new_job_id = str(uuid.uuid4())

    # Spawn a new training call with recover=True
    train_func = modal.Function.from_name(APP_NAME, "main")
    try:
        call = train_func.spawn(
            config_file_list_str=str(config_path),
            recover=True,
            name=config_name,
            hf_token=None,  # optionally store token in job_record and reuse here
            base_model=None,  # base_model is already encoded in the config
            job_id=new_job_id  # Explicitly pass the ID
        )
    except Exception as e:
        logger.error("Failed to spawn resumed job from %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail="Failed to resume job")

    new_job_record = {
        "job_id": new_job_id,
        "modal_id": call.object_id,
        "user_id": user_id,
        "status": "started",
        "config_name": config_name,
        "training_folder": str(get_user_training_path(user_id) / safe_filename(config_name)),
        "created_at": iso_now(),
        "result": None,
        "error": None,
        "parent_job_id": job_id,
        "progress": None,
    }
    _put_job_record(new_job_id, new_job_record)

    return {"status": "started", "job_id": new_job_id}


# ------------------------------------------------------------
# Remove job endpoint (Updated to use remote cleanup)
# ------------------------------------------------------------
@api.delete("/api/jobs/{job_id}")
async def delete_job_endpoint(job_id: str, user_id: str = Depends(get_current_user_id)):
    job_record = _get_job_record(job_id)
    if not job_record:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_record.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # 1. Stop if running/started
        if job_record.get("status") in {"running", "started"}:
            modal_id = job_record.get("modal_id", job_id)
            try:
                modal.FunctionCall.from_id(modal_id).cancel()
            except Exception as cancel_err:
                logger.warning("Failed to cancel job %s (modal_id: %s): %s", job_id, modal_id, cancel_err)

        # 2. Collect paths to delete
        paths_to_delete = []
        config_name = job_record.get("config_name")
        if config_name:
            # Training output folder (relative to MOUNT_DIR)
            folder_path = f"trainings/{user_id}/{safe_filename(config_name)}"
            # Config file
            config_path = f"trainings/{user_id}/_configs/{safe_filename(config_name)}.yaml"
            paths_to_delete.append(folder_path)
            paths_to_delete.append(config_path)

        # 3. Offload deletion to remote worker
        if paths_to_delete:
            try:
                # Call the remote cleanup function (non-blocking call, or blocking if we want to confirm)
                # We use .remote() to invoke it.
                volume_cleanup_task.remote(paths_to_delete)
            except Exception as e:
                logger.warning("Failed to trigger remote volume cleanup: %s", e)

        # 4. Remove record
        _delete_job_record(job_id)

        return {"status": "deleted", "job_id": job_id}
    except Exception as e:
        logger.error("Failed to delete job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail="Failed to delete job")


# ------------------------------------------------------------
# File serving (allowlist)
# ------------------------------------------------------------
@api.get("/api/files/{file_path:path}")
async def get_file(file_path: str, request: Request, user_id: str = Depends(get_current_user_id)):
    if file_path == "caption":
        caption_path = request.query_params.get("path")
        if not caption_path:
            raise HTTPException(status_code=400, detail="Missing caption path parameter")
        return await get_caption(path=caption_path, user_id=user_id)

    safe_root = Path(MOUNT_DIR)
    raw_path = safe_root / file_path
    resolved_for_check = raw_path.resolve()

    if not str(resolved_for_check).startswith(str(safe_root.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    ext = raw_path.suffix.lower()
    if ext not in ALLOWED_SERVE_EXTS:
        raise HTTPException(status_code=403, detail="This file type is not directly served")

    media_type = "application/octet-stream"
    if ext in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif ext == ".png":
        media_type = "image/png"
    elif ext == ".txt":
        media_type = "text/plain"
    elif ext == ".json":
        media_type = "application/json"
    elif ext == ".mp4":
        media_type = "video/mp4"
    elif ext == ".html":
        media_type = "text/html"
    elif ext == ".zip":
        media_type = "application/zip"

    return FileResponse(raw_path, media_type=media_type)


# ------------------------------------------------------------
# Logs & samples
# ------------------------------------------------------------
@api.get("/api/jobs/{job_id}/log")
async def get_job_log(job_id: str, user_id: str = Depends(get_current_user_id)):
    job = _get_job_record(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    training_dir = _get_job_training_dir(job, user_id)
    log_path = training_dir / "log.txt"
    try:
        model_volume.reload()
    except Exception as e:
        logger.debug("model_volume.reload() failed before reading log: %s", e)
    if not log_path.exists():
        return {"log": "Waiting for logs..."}
    return {"log": log_path.read_text(errors="replace")}


@api.get("/api/jobs/{job_id}/samples")
async def get_job_samples(job_id: str, user_id: str = Depends(get_current_user_id)):
    job = _get_job_record(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    training_dir = _get_job_training_dir(job, user_id)
    samples_dir = training_dir / "samples"
    if not samples_dir.exists():
        return {"samples": []}
    paths = []
    for f in samples_dir.glob("*"):
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            rel_path = f.relative_to(Path(MOUNT_DIR))
            paths.append(str(rel_path))
    return {"samples": sorted(paths)}


# ------------------------------------------------------------
# Metrics & reports
# ------------------------------------------------------------
@api.get("/api/jobs/{job_id}/metrics")
async def get_job_metrics(
    job_id: str,
    limit: int = Query(1000, ge=1, le=10000),
    user_id: str = Depends(get_current_user_id),
):
    job = _get_job_record(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        model_volume.reload()
    except Exception as e:
        logger.debug("model_volume.reload() failed before reading metrics: %s", e)

    metrics_dir = _resolve_metrics_dir(job, user_id)
    records = _load_metrics_records(metrics_dir, limit)
    fields = sorted({
        key
        for rec in records
        for key in rec.keys()
        if key not in {"timestamp", "job_id", "step"}
    })

    config_name = job.get("config_name") or job_id
    config_path = get_user_training_path(user_id) / "_configs" / f"{safe_filename(config_name)}.yaml"
    logging_cfg = _extract_logging_config(config_path)
    otel_meta = _load_otel_metadata(metrics_dir)

    return {
        "has_metrics": bool(records),
        "records": records,
        "fields": fields,
        "logging": logging_cfg,
        "otel": otel_meta,
    }


@api.get("/api/jobs/{job_id}/reports")
async def get_job_reports(job_id: str, user_id: str = Depends(get_current_user_id)):
    job = _get_job_record(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        model_volume.reload()
    except Exception as e:
        logger.debug("model_volume.reload() failed before reading reports: %s", e)

    metrics_dir = _resolve_metrics_dir(job, user_id)
    latest_dir = _get_latest_report_dir(metrics_dir)
    if not latest_dir:
        return {"reports": {"html": [], "png": []}, "latest_dir": None}

    html_paths = _relativize_paths(sorted(latest_dir.glob("*.html")))
    png_paths = _relativize_paths(sorted(latest_dir.glob("*.png")))
    return {
        "reports": {"html": html_paths, "png": png_paths},
        "latest_dir": str(latest_dir.relative_to(Path(MOUNT_DIR))),
    }


@api.post("/api/jobs/{job_id}/reports/generate")
async def generate_job_reports(job_id: str, user_id: str = Depends(get_current_user_id)):
    job = _get_job_record(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        from toolkit.report_generator import generate_reports
    except ModuleNotFoundError as e:
        logger.exception("Report generator is unavailable in the API image")
        raise HTTPException(status_code=500, detail="Report generator is unavailable") from e

    metrics_dir = _resolve_metrics_dir(job, user_id)
    report_paths = generate_reports(str(metrics_dir))
    safe_commit(model_volume)

    html_paths = _relativize_paths([Path(p) for p in report_paths.get("html", [])])
    png_paths = _relativize_paths([Path(p) for p in report_paths.get("png", [])])
    return {
        "reports": {"html": html_paths, "png": png_paths},
    }


@api.post("/api/jobs/{job_id}/reports/zip")
async def zip_job_reports(
    job_id: str,
    format: str = Query("all", pattern="^(all|png|html)$"),
    user_id: str = Depends(get_current_user_id),
):
    job = _get_job_record(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    metrics_dir = _resolve_metrics_dir(job, user_id)
    latest_dir = _get_latest_report_dir(metrics_dir)
    if not latest_dir:
        raise HTTPException(status_code=404, detail="No reports found")

    if format == "png":
        files = list(latest_dir.glob("*.png"))
    elif format == "html":
        files = list(latest_dir.glob("*.html"))
    else:
        files = list(latest_dir.glob("*.png")) + list(latest_dir.glob("*.html"))

    if not files:
        raise HTTPException(status_code=404, detail="No report files found for this format")

    export_dir = latest_dir.parent / "_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"reports_{latest_dir.name}_{format}.zip"
    zip_path = export_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
        for file_path in files:
            arcname = f"{latest_dir.name}/{file_path.name}"
            zip_handle.write(file_path, arcname=arcname)

    safe_commit(model_volume)
    rel_path = str(zip_path.relative_to(Path(MOUNT_DIR)))
    return {"zip_path": rel_path}


# ------------------------------------------------------------
# Job config fetch endpoint
# ------------------------------------------------------------
@api.get("/api/jobs/{job_id}/config", response_model=JobConfigResponse)
async def get_job_config(job_id: str, user_id: str = Depends(get_current_user_id)):
    job_record = _get_job_record(job_id)
    if not job_record:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_record.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    config_name = job_record.get("config_name")
    if not config_name:
        config_name = job_id

    config_path = get_user_training_path(user_id) / "_configs" / f"{safe_filename(config_name)}.yaml"

    if not config_path.exists():
        return JobConfigResponse(config_yaml=None)

    try:
        text = config_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read config file %s: %s", config_path, e)
        raise HTTPException(status_code=500, detail="Failed to read config file")

    return JobConfigResponse(config_yaml=text)


# ------------------------------------------------------------
# Datasets endpoints (streamed uploads)
# ------------------------------------------------------------
@api.get("/api/datasets")
async def list_datasets(user_id: str = Depends(get_current_user_id)):
    try:
        model_volume.reload()
    except Exception as e:
        logger.debug("model_volume.reload() failed before listing datasets: %s", e)
    ds_root = get_user_dataset_path(user_id)
    return [d.name for d in ds_root.iterdir() if d.is_dir()]


@api.post("/api/datasets/create")
async def create_dataset(req: DatasetCreateRequest, user_id: str = Depends(get_current_user_id)):
    ds_path = get_user_dataset_path(user_id) / safe_filename(req.name)
    ds_path.mkdir(exist_ok=True)
    safe_commit(model_volume)
    return {"name": req.name, "status": "created"}


@api.post("/api/datasets/delete")
async def delete_dataset(req: DatasetDeleteRequest, user_id: str = Depends(get_current_user_id)):
    ds_path = get_user_dataset_path(user_id) / safe_filename(req.name)
    if ds_path.exists():
        # Use remote cleanup for safety here too
        volume_cleanup_task.remote([str(ds_path.relative_to(MOUNT_DIR))])
    return {"status": "deleted"}


@api.post("/api/datasets/upload")
async def upload_dataset_files(
        name: str = Form(...),
        files: List[UploadFile] = File(...),
        user_id: str = Depends(get_current_user_id)
):
    ds_path = get_user_dataset_path(user_id) / safe_filename(name)
    ds_path.mkdir(exist_ok=True)
    stats = {"written": [], "paired": [], "skipped": [], "errors": []}

    async def _write_stream(upload_file: UploadFile, destination: Path) -> int:
        size = 0
        with open(destination, "wb") as out_f:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out_f.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File too large")
                out_f.write(chunk)
        return size

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".mp4"}
    caption_exts = {".txt", ".json"}
    staged_pairs: dict[str, dict[str, Path]] = {}

    for upload in files:
        filename = safe_filename(upload.filename)
        if not filename:
            stats["skipped"].append({"file": upload.filename, "reason": "empty filename"})
            continue
        dest = ds_path / filename
        ext = dest.suffix.lower()
        staged_pairs.setdefault(dest.stem.lower(), {})
        try:
            await _write_stream(upload, dest)
        finally:
            try:
                await upload.close()
            except Exception:
                pass
        stats["written"].append(str(dest.relative_to(Path(MOUNT_DIR))))
        if ext in image_exts:
            staged_pairs[dest.stem.lower()]["image"] = dest
        elif ext in caption_exts:
            staged_pairs[dest.stem.lower()]["caption"] = dest

    for pair in staged_pairs.values():
        img = pair.get("image")
        cap = pair.get("caption")
        if not img or not cap:
            continue
        expected_caption_path = img.with_suffix(cap.suffix.lower())
        if cap != expected_caption_path:
            shutil.move(cap, expected_caption_path)
            stats["paired"].append({
                "image": str(img.relative_to(Path(MOUNT_DIR))),
                "caption": str(expected_caption_path.relative_to(Path(MOUNT_DIR)))
            })
        else:
            stats["paired"].append({
                "image": str(img.relative_to(Path(MOUNT_DIR))),
                "caption": str(cap.relative_to(Path(MOUNT_DIR)))
            })

    safe_commit(model_volume)
    return {"status": "success", **stats}


@api.get("/api/datasets/{name}/images")
async def list_dataset_images(name: str, user_id: str = Depends(get_current_user_id)):
    ds_path = get_user_dataset_path(user_id) / safe_filename(name)
    if not ds_path.exists():
        return {"images": []}
    images = []
    # Removed .txt from this list to prevent text files from showing up as images
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".mp4"}
    for f in ds_path.iterdir():
        if f.suffix.lower() in valid_exts:
            rel = f.relative_to(Path(MOUNT_DIR))
            images.append({"img_path": str(rel)})
    return {"images": images}


# ------------------------------------------------------------
# Training endpoint (validate config, write, spawn)
# ------------------------------------------------------------
@api.post("/api/train", response_model=TrainResponse)
async def start_training(request: TrainRequest, user_id: str = Depends(get_current_user_id)):
    try:
        cfg = request.config
        _normalize_dataset_paths(cfg, user_id)
        if not isinstance(cfg, dict) or "config" not in cfg:
            raise HTTPException(status_code=400, detail="Invalid config: missing 'config' key")

        proc = cfg["config"].get("process")
        if not isinstance(proc, list) or len(proc) == 0:
            raise HTTPException(status_code=400, detail="Invalid config: 'config.process' must be a non-empty list")

        # Extract Base Model Name ---
        base_model_path = None
        try:
            # Navigate the config structure: config -> process -> [0] -> model -> name_or_path
            # This is standard for ai-toolkit configs
            base_model_path = cfg["config"]["process"][0]["model"]["name_or_path"]
        except (KeyError, IndexError, TypeError):
            logger.warning("Could not extract base_model path from config, skipping pre-cache.")
        # ------------------------------------------

        train_func = modal.Function.from_name(APP_NAME, "main")
        proposed_name = request.name or str(uuid.uuid4())
        job_name = safe_filename(proposed_name) or str(uuid.uuid4())
        user_train_root = f"{MOUNT_DIR}/trainings/{user_id}"
        job_training_folder = str(Path(user_train_root) / job_name)
        os.makedirs(job_training_folder, exist_ok=True)

        try:
            cfg["config"]["process"][0]["training_folder"] = job_training_folder
            cfg["config"]["training_folder"] = job_training_folder
            cfg["config"]["name"] = job_name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid config shape: {e}")

        config_dir = f"{user_train_root}/_configs"
        os.makedirs(config_dir, exist_ok=True)
        config_path = f"{config_dir}/{job_name}.yaml"

        try:
            with open(config_path, 'w') as f:
                yaml.dump(cfg, f)
        except Exception as e:
            logger.exception("Failed to write config file: %s", e)
            raise HTTPException(status_code=500, detail="Failed to write config file")

        safe_commit(model_volume)

        # Generate a new persistent Job ID (UUID)
        new_job_id = str(uuid.uuid4())

        # --- UPDATED SPAWN CALL ---
        call = train_func.spawn(
            config_file_list_str=config_path,
            recover=request.recover,
            name=job_name,
            hf_token=request.hf_token,  # Pass the token
            base_model=base_model_path,  # Pass the model name
            job_id=new_job_id  # Explicitly pass the ID
        )
        # --------------------------

        job_record = {
            "job_id": new_job_id,
            "modal_id": call.object_id,  # Store Modal ID for control later
            "user_id": user_id,
            "status": "started",
            "config_name": job_name,
            "training_folder": job_training_folder,
            "created_at": iso_now(),
            "result": None,
            "error": None,
            "progress": None,
        }
        _put_job_record(new_job_id, job_record)
        return TrainResponse(job_id=new_job_id, status="started")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Training failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to start training")


# ------------------------------------------------------------
# Models listing
# ------------------------------------------------------------
@api.get("/api/models")
async def list_models(user_id: str = Depends(get_current_user_id)):
    train_root = get_user_training_path(user_id)
    models = [
        public_generation_model(model)
        for model in list_generation_models(train_root)
    ]
    return {"models": models}


# ------------------------------------------------------------
# Generate endpoint: calls remote worker which writes files and returns relative paths
# ------------------------------------------------------------
@api.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, user_id: str = Depends(get_current_user_id)):
    try:
        lora_path = None
        model_name = None
        checkpoint = None
        base_model = _normalize_generation_base_model(request.base_model)
        model_architecture = infer_generation_architecture({"name_or_path": base_model})

        if request.model_name:
            model_name = request.model_name.strip()
            if not model_name or safe_filename(model_name) != model_name:
                raise HTTPException(status_code=400, detail="Invalid trained model name.")
            resolved_model = resolve_generation_model(get_user_training_path(user_id), model_name)
            if not resolved_model["selectable"]:
                raise HTTPException(
                    status_code=400,
                    detail=resolved_model["status_reason"] or "Selected trained model is unavailable.",
                )
            base_model = resolved_model["base_model"]
            model_architecture = resolved_model["architecture"]
            checkpoint = resolved_model["checkpoint"]
            lora_path = resolved_model["lora_path"]

        job_stub = uuid.uuid4().hex
        out_dir = f"{MOUNT_DIR}/generated/{user_id}/{job_stub}"

        remote_generate_func = modal.Function.from_name(APP_NAME, "remote_generate")

        base_seed = request.seed
        if base_seed is None:
            seed_input = f"{user_id}:{job_stub}:{request.prompt}"
            base_seed = int.from_bytes(hashlib.sha256(seed_input.encode("utf-8")).digest()[:4], "big")

        generation_result = await remote_generate_func.remote.aio(
            prompt=request.prompt,
            num_samples=request.num_samples,
            lora_path=lora_path,
            out_dir=out_dir,
            base_model=base_model,
            model_architecture=model_architecture,
            adapter_name=model_name,
            hf_token=request.hf_token,
            seed=base_seed,
        )

        if not isinstance(generation_result, dict) or not isinstance(generation_result.get("paths"), list):
            raise RuntimeError("Generation worker returned an invalid response.")
        saved_rel_paths = generation_result["paths"]
        adapter_loaded = bool(generation_result.get("adapter_loaded"))
        if model_name and not adapter_loaded:
            raise RuntimeError("Generation worker did not confirm that the selected LoRA was loaded.")

        safe_commit(model_volume)
        model_volume.reload()

        urls: list[str] = []
        for p in saved_rel_paths:
            rp = Path(p)
            if rp.is_absolute():
                rp = rp.relative_to(Path(MOUNT_DIR))
            if not rp.suffix:
                logger.warning("Skipping non-file path returned by remote_generate: %s", p)
                continue
            url = f"/api/files/{rp}"
            logger.info("Generated image URL: %s (path: %s)", url, rp)
            urls.append(url)

        return GenerateResponse(
            images=urls,
            status="success",
            seed=base_seed,
            model_name=model_name,
            base_model=base_model,
            checkpoint=checkpoint,
            adapter_loaded=adapter_loaded,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        detail = str(e).strip() or "Generation failed; check server logs."
        raise HTTPException(status_code=502, detail=detail)


# ------------------------------------------------------------
# Generic upload for control images
# ------------------------------------------------------------
@api.post("/api/upload")
async def upload_generic_file(file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    uploads_dir = get_user_dataset_path(user_id) / "_uploads"
    uploads_dir.mkdir(exist_ok=True)
    filename = safe_filename(file.filename)
    dest = uploads_dir / filename
    size = 0
    with open(dest, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large")
            out.write(chunk)
    safe_commit(model_volume)
    rel_path = dest.relative_to(Path(MOUNT_DIR))
    return {"path": str(rel_path)}


# ------------------------------------------------------------
# Caption endpoints
# ------------------------------------------------------------
@api.get("/api/files/caption")
async def get_caption(
        path: str = Query(...),  # e.g. "datasets/default_user/test/2.jpg"
        user_id: str = Depends(get_current_user_id),
):
    img_path = (Path(MOUNT_DIR) / path).resolve()

    # Make sure we stay inside MOUNT_DIR
    if not str(img_path).startswith(str(Path(MOUNT_DIR).resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    json_path = img_path.with_suffix(".json")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON caption file %s: %s", json_path, e)
            raise HTTPException(status_code=400, detail="Invalid JSON caption file")
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="JSON caption file must contain an object")
        return {
            "caption": data.get("caption", "") or "",
            "caption_short": data.get("caption_short", "") or "",
            "caption_ext": "json",
        }

    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        return {
            "caption": txt_path.read_text(encoding="utf-8"),
            "caption_short": "",
            "caption_ext": "txt",
        }

    # Always 200 with empty caption so the UI just treats it as "no caption yet"
    return {"caption": "", "caption_short": "", "caption_ext": "txt"}


@api.post("/api/files/caption")
async def save_caption(req: CaptionRequest, user_id: str = Depends(get_current_user_id)):
    mount_root = Path(MOUNT_DIR)
    resolved_mount_root = mount_root.resolve()

    # req.path is like "datasets/default_user/test/2.jpg"
    img_path = (mount_root / req.path).resolve()

    # Stay inside MOUNT_DIR
    if not str(img_path).startswith(str(resolved_mount_root)):
        raise HTTPException(status_code=403, detail="Access denied")

    caption_ext = (req.caption_ext or "txt").lower().lstrip(".")
    if caption_ext not in {"txt", "json"}:
        raise HTTPException(status_code=400, detail="Unsupported caption extension")

    caption_path = img_path.with_suffix(f".{caption_ext}")

    try:
        rel = caption_path.relative_to(resolved_mount_root)
    except ValueError:
        logger.warning("Caption save outside mount root rejected: %s", caption_path)
        raise HTTPException(status_code=403, detail="Access denied")

    # Simple user check: require the per-user folder segment
    if f"/{user_id}/" not in f"/{rel.as_posix()}/":
        raise HTTPException(status_code=403, detail="Access denied")

    if caption_ext == "json":
        data = {}
        if caption_path.exists():
            try:
                existing = json.loads(caption_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON caption file %s: %s", caption_path, e)
                raise HTTPException(status_code=400, detail="Invalid JSON caption file")
            if not isinstance(existing, dict):
                raise HTTPException(status_code=400, detail="JSON caption file must contain an object")
            data = existing
        data["caption"] = req.caption
        data["caption_short"] = req.caption_short or ""
        caption_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        caption_path.write_text(req.caption, encoding="utf-8")

    safe_commit(model_volume)
    return {"status": "saved"}


@api.delete("/api/files/{path:path}")
async def delete_file(path: str, user_id: str = Depends(get_current_user_id)):
    target = (Path(MOUNT_DIR) / path).resolve()

    if not str(target).startswith(str(Path(MOUNT_DIR).resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    rel = target.relative_to(Path(MOUNT_DIR))
    if f"/{user_id}/" not in f"/{rel.as_posix()}/":
        raise HTTPException(status_code=403, detail="Access denied")

    if target.exists():
        target.unlink()
        txt = target.with_suffix(".txt")
        if txt.exists():
            txt.unlink()
        safe_commit(model_volume)
        return {"status": "deleted"}

    raise HTTPException(status_code=404, detail="File not found")


# ------------------------------------------------------------
# Modal Entrypoints: API app and worker functions
# ------------------------------------------------------------
@app.function(
    image=web_image,
    timeout=600,
    min_containers=0,
    max_containers=5,
    volumes={MOUNT_DIR: model_volume}
)
@asgi_app()
def api_app():
    return api


@app.function(
    image=image,
    timeout=900,
    volumes={MOUNT_DIR: model_volume, CACHE_DIR: hf_volume},
)
def preflight_imports(require_flux2_klein: bool = True) -> str:
    """
    Import the same training modules used by a real job so dependency/image
    issues fail before launching GPU work.
    """
    import importlib
    import json as json_lib
    import sys

    if "/root/ai-toolkit" not in sys.path:
        sys.path.insert(0, "/root/ai-toolkit")

    results = {}

    transformers = importlib.import_module("transformers")
    results["transformers_version"] = getattr(transformers, "__version__", "unknown")
    results["transformers_version_matches"] = results["transformers_version"] == TRANSFORMERS_VERSION
    required_transformers = [
        "Qwen3ForCausalLM",
        "Qwen2TokenizerFast",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLProcessor",
    ]
    results["transformers_symbols"] = {
        name: hasattr(transformers, name)
        for name in required_transformers
    }

    diffusers = importlib.import_module("diffusers")
    results["diffusers_version"] = getattr(diffusers, "__version__", "unknown")
    results["diffusers_version_matches"] = results["diffusers_version"] == DIFFUSERS_VERSION
    required_diffusers = [
        "DiffusionPipeline",
        "FluxPipeline",
        "Flux2KleinPipeline",
    ]
    results["diffusers_symbols"] = {
        name: hasattr(diffusers, name)
        for name in required_diffusers
    }

    torchao = importlib.import_module("torchao")
    results["torchao_version"] = getattr(torchao, "__version__", "unknown")
    results["torchao_version_matches"] = results["torchao_version"] == TORCHAO_VERSION
    lora_loaders = importlib.import_module("diffusers.loaders")
    required_lora_loaders = [
        "FluxLoraLoaderMixin",
        "Flux2LoraLoaderMixin",
    ]
    results["diffusers_lora_loader_symbols"] = {
        name: hasattr(lora_loaders, name)
        for name in required_lora_loaders
    }

    importlib.import_module("toolkit.dataloader_mixins")
    importlib.import_module("toolkit.data_loader")
    importlib.import_module("toolkit.custom_adapter")
    importlib.import_module("toolkit.ip_adapter")
    importlib.import_module("toolkit.reference_adapter")
    importlib.import_module("toolkit.stable_diffusion_model")
    importlib.import_module("jobs.process.BaseSDTrainProcess")
    importlib.import_module("jobs")
    importlib.import_module("toolkit.job")

    missing = [
        f"transformers.{name}"
        for name, present in results["transformers_symbols"].items()
        if not present
    ]
    if not results["transformers_version_matches"]:
        missing.append(f"transformers=={TRANSFORMERS_VERSION}")
    if not results["torchao_version_matches"]:
        missing.append(f"torchao=={TORCHAO_VERSION}")
    for name, present in results["diffusers_symbols"].items():
        if name == "Flux2KleinPipeline" and not require_flux2_klein:
            continue
        if not present:
            missing.append(f"diffusers.{name}")
    for name, present in results["diffusers_lora_loader_symbols"].items():
        if name == "Flux2LoraLoaderMixin" and not require_flux2_klein:
            continue
        if not present:
            missing.append(f"diffusers.loaders.{name}")
    if not results["diffusers_version_matches"]:
        missing.append(f"diffusers=={DIFFUSERS_VERSION}")

    if missing:
        raise RuntimeError(f"Preflight missing required symbols: {missing}; versions={results}")

    results["status"] = "ok"
    return json_lib.dumps(results, indent=2, sort_keys=True)


@app.function(volumes={MOUNT_DIR: model_volume})
def volume_cleanup_task(paths: List[str]) -> int:
    """
    Dedicated worker to delete a list of relative paths (files or folders) inside the volume.
    Safe to call from async contexts.
    """
    import shutil
    root = Path(MOUNT_DIR)
    deleted_count = 0

    print(f"[Cleanup] Starting cleanup for {len(paths)} paths.")

    for p in paths:
        # Prevent trying to delete outside of root (basic sanity check)
        if ".." in p or p.startswith("/"):
            p = p.lstrip("/")

        target = root / p

        # Ensure we are still inside MOUNT_DIR
        try:
            target.resolve().relative_to(root.resolve())
        except ValueError:
            print(f"[Cleanup] Skipping unsafe path: {p}")
            continue

        if target.exists():
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
                print(f"[Cleanup] Deleted: {p}")
                deleted_count += 1
            except Exception as e:
                print(f"[Cleanup] Error deleting {p}: {e}")
        else:
            print(f"[Cleanup] Not found (already deleted?): {p}")

    if deleted_count > 0:
        print("[Cleanup] Committing changes to volume...")
        model_volume.commit()

    return deleted_count


@app.function(
    gpu="A100-80GB",
    timeout=28800,
    image=image,
    volumes={MOUNT_DIR: model_volume, CACHE_DIR: hf_volume},
    secrets=[new_relic_otlp_secret],
)
def main(
        config_file_list_str: str,
        recover: bool = False,
        name: str | None = None,
        hf_token: str | None = None,
        base_model: str | None = None,
        job_id: str | None = None,
):
    import os
    import sys
    import time
    import logging
    from pathlib import Path

    # ----------------------------------------------------
    # 0. Environment Setup
    # ----------------------------------------------------
    # Force Python to be unbuffered to help with logging lag
    os.environ["PYTHONUNBUFFERED"] = "1"

    if "/root/ai-toolkit" not in sys.path:
        sys.path.insert(0, "/root/ai-toolkit")

    # ----------------------------------------------------
    # 1. Resolve config & training folder
    # ----------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [setup] %(message)s",
        force=True,
    )
    setup_logger = logging.getLogger("setup")

    setup_logger.info("OpenTelemetry env: %s", _get_otel_env_summary())

    config_file = config_file_list_str.split(",")[0]
    cfg_path = Path(config_file)

    setup_logger.info(f"Waiting for config file: {cfg_path}")

    wait_secs = 15
    poll = 0.5
    start = time.time()

    while not cfg_path.exists() and (time.time() - start < wait_secs):
        try:
            model_volume.reload()
        except Exception:
            pass
        time.sleep(poll)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not visible: {config_file}")

    job_name = name or os.path.splitext(os.path.basename(config_file))[0]
    train_root = Path(MOUNT_DIR) / "trainings"
    train_root.mkdir(parents=True, exist_ok=True)

    from toolkit.job import get_job
    from toolkit.config import get_config
    from toolkit.progress_tracker import ProgressTracker

    try:
        cfg = get_config(config_file, name=job_name)
        found = (
            cfg.get("config", {})
            .get("process", [{}])[0]
            .get("training_folder")
        )
        training_folder = Path(found) if found else train_root
    except Exception as e:
        setup_logger.warning(f"Unable to parse training folder from config: {e}")
        training_folder = train_root

    config_user_id = None
    try:
        cfg_path_parts = cfg_path.resolve().parts
        mount_parts = Path(MOUNT_DIR).resolve().parts
        if len(cfg_path_parts) >= len(mount_parts) + 3:
            # Expect: {MOUNT_DIR}/trainings/<user_id>/_configs/<job_name>.yaml
            if cfg_path_parts[:len(mount_parts) + 1] == (*mount_parts, "trainings"):
                config_user_id = cfg_path_parts[len(mount_parts) + 1]
    except Exception as e:
        setup_logger.debug("Unable to resolve config user id: %s", e)

    if config_user_id:
        expected_training_folder = Path(MOUNT_DIR) / "trainings" / config_user_id / job_name
        needs_rewrite = False
        cfg_training_folder = cfg.get("config", {}).get("training_folder")
        if not cfg_training_folder:
            needs_rewrite = True
        else:
            try:
                cfg_training_folder_resolved = Path(cfg_training_folder).resolve()
                if not str(cfg_training_folder_resolved).startswith(str(Path(MOUNT_DIR).resolve())):
                    needs_rewrite = True
            except Exception:
                needs_rewrite = True

        if needs_rewrite:
            setup_logger.warning(
                "Config training_folder is outside MOUNT_DIR; rewriting to %s",
                expected_training_folder,
            )
            cfg.setdefault("config", {})["training_folder"] = str(expected_training_folder)
            try:
                cfg.setdefault("config", {}).setdefault("process", [{}])[0]["training_folder"] = str(expected_training_folder)
            except Exception:
                pass
            try:
                with open(cfg_path, "w", encoding="utf-8") as handle:
                    yaml.dump(cfg, handle)
                safe_commit(model_volume)
            except Exception as e:
                setup_logger.warning("Failed to rewrite training_folder in config: %s", e)
            training_folder = expected_training_folder

    training_folder.mkdir(parents=True, exist_ok=True)
    log_file_path = training_folder / "log.txt"

    setup_logger.info(f"Using training folder: {training_folder}")
    setup_logger.info(f"Will write logs to: {log_file_path}")

    # ----------------------------------------------------
    # 2. Capture stdout/stderr + Root Logger
    # ----------------------------------------------------
    # We open in line-buffered mode (buffering=1) to ensure immediate writes
    log_file = open(log_file_path, "a", encoding="utf-8", buffering=1)

    class DualOutput:
        def __init__(self, terminal, logfile):
            self.terminal = terminal
            self.logfile = logfile

        def write(self, message):
            if self.terminal:
                self.terminal.write(message)
                self.terminal.flush()
            if self.logfile:
                self.logfile.write(message)
                self.logfile.flush()  # Force flush on every write

        def flush(self):
            if self.terminal:
                self.terminal.flush()
            if self.logfile:
                self.logfile.flush()

        def isatty(self):
            return getattr(self.terminal, "isatty", lambda: False)()

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = DualOutput(original_stdout, log_file)
    sys.stderr = DualOutput(original_stderr, log_file)

    # Force toolkit.print to mirror DualOutput instead of suppressing non-main processes.
    try:
        from toolkit import print as toolkit_print

        class _ModalLogger(toolkit_print.Logger):
            def __init__(self):
                # Point to existing DualOutput streams; no new files opened.
                self.terminal = sys.stdout
                self.log = log_file

        toolkit_print.print_acc = lambda *args, **kwargs: print(*args, **kwargs, flush=True)
        toolkit_print.setup_log_to_file = lambda *_args, **_kwargs: None
        toolkit_print.Logger = _ModalLogger
    except Exception as log_patch_err:
        logger.debug("Failed to patch toolkit.print logging shim: %s", log_patch_err)

    # ALSO add a FileHandler to the root logger.
    # This catches cases where libraries use logger.info() directly, bypassing stdout.
    root_logger = logging.getLogger()
    f_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    f_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger.addHandler(f_handler)

    # ----------------------------------------------------
    # 3. Reconfigure logging after redirection
    # ----------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [worker] %(message)s",
        force=True,
        handlers=[logging.StreamHandler(sys.stderr), f_handler],
    )
    logger = logging.getLogger("run_modal.worker")
    logger.info("Redirected stdout & stderr. Logging capture activated.")

    # Sanity check line to verify DualOutput is active in log.txt
    print("[AITK-MODAL] main() logging capture is active.")

    # Periodically commit the shared volume so other containers can see new log bytes.
    stop_commit = threading.Event()

    def _volume_commit_loop():
        while not stop_commit.is_set():
            try:
                model_volume.commit()
            except Exception as commit_err:
                logger.debug("Periodic model_volume.commit() failed: %s", commit_err)
            stop_commit.wait(5)

    commit_thread = threading.Thread(target=_volume_commit_loop, daemon=True)
    commit_thread.start()

    # ----------------------------------------------------
    # 4. HF authentication
    # ----------------------------------------------------
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    # ----------------------------------------------------
    # 5. PRE-CACHE BASE MODEL (resilient version)
    # ----------------------------------------------------
    if base_model and not base_model.startswith("/") and not base_model.startswith("."):
        from huggingface_hub import snapshot_download

        logger.info(f"Checking cache for base model: {base_model}")
        try:
            snapshot_download(
                base_model,
                cache_dir=CACHE_DIR,
                local_files_only=True,
            )
            logger.info("Base model already cached.")
            os.environ["HF_HUB_OFFLINE"] = "1"
        except Exception:
            logger.info("Model not cached. Downloading...")
            try:
                snapshot_download(
                    base_model,
                    cache_dir=CACHE_DIR,
                    token=hf_token,
                )
                hf_volume.commit()
                os.environ["HF_HUB_OFFLINE"] = "1"
            except Exception as e:
                logger.warning(f"Failed to download model: {e}")

    # ----------------------------------------------------
    # 6. Run the actual job with richer status updates
    # ----------------------------------------------------

    # Fallback to current call id if job_id was not provided (for older calls)
    if not job_id:
        job_id = _current_call_id()
        if not job_id:
            job_id = os.environ.get("MODAL_TASK_ID")

    print(f"[DEBUG] Worker Job ID: {job_id}")

    progress_record_cache = None

    def _publish_progress(snapshot):
        nonlocal progress_record_cache
        if not job_id:
            print(f"[DEBUG] Progress emit FAILED: No Job ID")
            return

        try:
            rec = _get_job_record(job_id)
            if not rec:
                print(f"[DEBUG] Job record missing for {job_id}, skipping update")
                return

            merged = {
                "step": snapshot.step,
                "total": snapshot.total,
                "percent": snapshot.percent,
                "phase": snapshot.phase,
                "message": snapshot.message,
                "info": snapshot.info,
                "updated_at": datetime.now().isoformat() + "Z",
            }
            rec["progress"] = merged

            # Force status to running if we are receiving progress
            if rec["status"] == "started":
                rec["status"] = "running"

            _put_job_record(job_id, rec)
        except Exception as e:
            print(f"[DEBUG] Progress write error: {e}")

    try:
        job = get_job(config_file, name, job_id=job_id)
        job.set_progress_tracker(ProgressTracker(_publish_progress))
        logger.info(f"Starting job: {job_name}")
        logger.debug("Note: toolkit.print.setup_log_to_file is intentionally unused inside Modal workers because DualOutput already handles log capture.")

        # Best-effort status update so UI can show "running"
        try:
            rec = _get_job_record(job_id) if job_id else None
            if rec:
                rec["status"] = "running"
                _put_job_record(job_id, rec)
        except Exception as e:
            logger.warning("Failed to update job_store status to running: %s", e)

        job.run()

        # Mark as completed when done
        try:
            rec = _get_job_record(job_id) if job_id else None
            if rec:
                rec["status"] = "completed"
                rec["ended_at"] = iso_now()
                _put_job_record(job_id, rec)
        except Exception as e:
            logger.warning("Failed to update job_store status to completed: %s", e)

    except Exception as e:
        logger.exception("Training job failed.")
        # Mark as failed for UI
        try:
            rec = _get_job_record(job_id) if job_id else None
            if rec:
                rec["status"] = "failed"
                rec["error"] = str(e)
                rec["ended_at"] = iso_now()
                _put_job_record(job_id, rec)
        except Exception as inner:
            logger.warning("Failed to update job_store status to failed: %s", inner)
        raise

    finally:
        # Ensure all logs and volumes flush
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        try:
            model_volume.commit()
            hf_volume.commit()
        except Exception as e:
            logger.warning(f"Volume commit failed: {e}")

        if "job" in locals():
            try:
                job.cleanup()
            except Exception as e:
                logger.warning(f"Job cleanup failed: {e}")

        logger.info("Job completed.")

        stop_commit.set()
        if 'commit_thread' in locals() and commit_thread.is_alive():
            commit_thread.join(timeout=5)

        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


@app.function(gpu="A100", timeout=1800, image=image, volumes={MOUNT_DIR: model_volume, CACHE_DIR: hf_volume})
def remote_generate(
        prompt: str,
        num_samples: int = 1,
        lora_path: Optional[str] = None,
        out_dir: Optional[str] = None,
        base_model: Optional[str] = DEFAULT_GENERATION_BASE_MODEL,
        model_architecture: Optional[str] = None,
        adapter_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        seed: Optional[int] = None,
):
    import diffusers
    import torchao
    import transformers
    from diffusers import DiffusionPipeline
    import torch
    import os
    import uuid

    # 1. AUTHENTICATION
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    base_model = _normalize_generation_base_model(base_model)
    logger.info(
        "remote_generate called model=%s architecture=%s checkpoint=%s diffusers=%s transformers=%s torchao=%s",
        base_model,
        model_architecture or "unknown",
        Path(lora_path).name if lora_path else None,
        getattr(diffusers, "__version__", "unknown"),
        getattr(transformers, "__version__", "unknown"),
        getattr(torchao, "__version__", "unknown"),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # 2. CACHING LOGIC: Try local load, failover to download
    pipe = _load_generation_pipeline(
        DiffusionPipeline,
        base_model,
        torch_dtype,
        hf_token=hf_token,
        hf_volume_obj=hf_volume,
    )

    pipe.to(device)

    # 3. LORA LOADING
    adapter_loaded = False
    active_adapter_name = None
    if lora_path:
        resolved_adapter_name = adapter_name or "trained_lora"
        logger.info("Loading LoRA checkpoint=%s adapter=%s", Path(lora_path).name, resolved_adapter_name)
        try:
            adapter_loaded = load_and_activate_lora(
                pipe,
                lora_path,
                resolved_adapter_name,
                adapter_weight=1.0,
            )
        except Exception:
            logger.exception(
                "LoRA activation failed model=%s architecture=%s checkpoint=%s",
                base_model,
                model_architecture or "unknown",
                Path(lora_path).name,
            )
            raise
        logger.info(
            "LoRA activated model=%s architecture=%s checkpoint=%s adapter=%s weight=1.0",
            base_model,
            model_architecture or "unknown",
            Path(lora_path).name,
            resolved_adapter_name,
        )
        active_adapter_name = resolved_adapter_name

    # 4. GENERATION LOOP
    if out_dir is None:
        out_dir = f"{MOUNT_DIR}/generated/tmp_{uuid.uuid4().hex}"
    os.makedirs(out_dir, exist_ok=True)

    base_seed = int(seed) if seed is not None else 1024
    torch.manual_seed(base_seed)
    logger.info("Using base seed %d", base_seed)

    saved_rel_paths = []
    for i in range(num_samples):
        current_seed = base_seed + i
        generator = torch.Generator(device=device).manual_seed(current_seed)

        logger.info(f"Generating sample {i + 1}/{num_samples} with seed {current_seed}...")
        out = _call_generation_pipeline(
            pipe,
            prompt,
            generator,
        )
        pil_img = out.images[0]

        fname = f"{uuid.uuid4().hex}.png"
        fpath = os.path.join(out_dir, fname)
        pil_img.save(fpath)

        try:
            rel = str(Path(fpath).relative_to(Path(MOUNT_DIR)))
        except Exception:
            rel = os.path.relpath(fpath, MOUNT_DIR)
        saved_rel_paths.append(rel)

    # 5. COMMIT OUTPUTS
    try:
        model_volume.commit()
    except Exception as e:
        logger.warning("model_volume.commit() failed after generation: %s", e)

    return {
        "paths": saved_rel_paths,
        "adapter_loaded": adapter_loaded,
        "adapter_name": active_adapter_name,
    }


# ------------------------------------------------------------
# Model cache check
# ------------------------------------------------------------
def _ensure_cache_has_model(repo_id: str, cache_dir: str = CACHE_DIR):
    expected_fragment = f"models--{repo_id.replace('/', '--')}"
    p = Path(cache_dir)
    if not p.exists():
        return False
    for sub in p.rglob("*"):
        if sub.is_dir() and expected_fragment in str(sub):
            return True
    return False


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_list", nargs="+", type=str)
    parser.add_argument("-r", "--recover", action="store_true")
    parser.add_argument("-n", "--name", type=str, default=None)
    args = parser.parse_args()
    try:
        main.call(config_file_list_str=",".join(args.config_file_list), recover=args.recover, name=args.name,
                  job_id=None)
    except Exception as e:
        logger.exception("Local CLI main call failed: %s", e)
        raise
