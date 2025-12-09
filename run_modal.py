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
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer # Removed
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ------------------------------------------------------------
# Configuration & paths
# ------------------------------------------------------------
MOUNT_DIR = "/root/modal_output"
CACHE_DIR = "/root/.cache/huggingface"
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

ALLOWED_SERVE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".txt", ".json", ".mp4"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB cap per file
_filename_sanitize_re = re.compile(r"[^A-Za-z0-9._-]")


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
        "transformers python-dotenv accelerate ftfy safetensors albumentations lycoris-lora timm einops "
        "opencv-python-headless huggingface_hub peft lpips hf_transfer flatten_json pyyaml oyaml tensorboard "
        "toml albucore pydantic omegaconf k-diffusion controlnet_aux optimum-quanto python-slugify open_clip_torch "
        "bitsandbytes pytorch_fid sentencepiece pytorch-wavelets matplotlib diffusers fastapi[standard] "
        "python-multipart modal || true"
    )
    .env({
        "HUGGINGFACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(
        LOCAL_AITOOLKIT_PATH,
        remote_path="/root/ai-toolkit",
        ignore=["**/.idea/**", "**/__pycache__/**", "**/*.pyc", ".venv/**", "web-ui/**", "ui/**"],
    )
)

# Lightweight image for the API app (also uses wheelhouse for convenience)
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(LOCAL_WHEELHOUSE_PATH, remote_path="/root/wheels", copy=True)
    .apt_install("ca-certificates", "git", "curl", "wget")
    .run_commands(
        "echo '=== Ensure pip + build tools for web image ==='",
        "python -m pip install --upgrade pip setuptools wheel build",
    )
    .run_commands(
        "echo '=== Install web dependencies (wheelhouse-first) ==='",
        "python -m pip install --prefer-binary --find-links /root/wheels fastapi[standard] modal python-dotenv oyaml python-multipart || true"
    )
)

# Modal app
app = modal.App(
    name="flex-lora-training",
    image=image,
    volumes={MOUNT_DIR: modal.Volume.from_name("flux-lora-models", create_if_missing=True),
             CACHE_DIR: modal.Volume.from_name("hf-cache", create_if_missing=True)}
)

# Volumes & persistent stores
model_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
job_store = modal.Dict.from_name("user-job-store", create_if_missing=True)


# ------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    num_samples: int = 1
    model_name: Optional[str] = None
    base_model: Optional[str] = "black-forest-labs/FLUX.1-dev"
    hf_token: Optional[str] = None


class GenerateResponse(BaseModel):
    images: List[str]
    status: str


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
api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.pages.dev"],
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
    train_func = modal.Function.from_name("flex-lora-training", "main")
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
    folder_name = job.get("config_name", job_id)
    log_path = get_user_training_path(user_id) / folder_name / "log.txt"
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
    folder_name = job.get("config_name", job_id)
    samples_dir = get_user_training_path(user_id) / folder_name / "samples"
    if not samples_dir.exists():
        return {"samples": []}
    paths = []
    for f in samples_dir.glob("*"):
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            rel_path = f.relative_to(Path(MOUNT_DIR))
            paths.append(str(rel_path))
    return {"samples": sorted(paths)}


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
        name: str = Body(...),
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
    caption_ext = ".txt"
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
        elif ext == caption_ext:
            staged_pairs[dest.stem.lower()]["caption"] = dest

    for pair in staged_pairs.values():
        img = pair.get("image")
        cap = pair.get("caption")
        if not img or not cap:
            continue
        expected_caption_path = img.with_suffix(caption_ext)
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

        # --- NEW LOGIC: Extract Base Model Name ---
        base_model_path = None
        try:
            # Navigate the config structure: config -> process -> [0] -> model -> name_or_path
            # This is standard for ai-toolkit configs
            base_model_path = cfg["config"]["process"][0]["model"]["name_or_path"]
        except (KeyError, IndexError, TypeError):
            logger.warning("Could not extract base_model path from config, skipping pre-cache.")
        # ------------------------------------------

        train_func = modal.Function.from_name("flex-lora-training", "main")
        proposed_name = request.name or str(uuid.uuid4())
        job_name = safe_filename(proposed_name) or str(uuid.uuid4())
        user_train_root = f"{MOUNT_DIR}/trainings/{user_id}"
        job_training_folder = str(Path(user_train_root) / job_name)
        os.makedirs(job_training_folder, exist_ok=True)

        try:
            cfg["config"]["process"][0]["training_folder"] = job_training_folder
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
    if not train_root.exists():
        return {"models": []}

    models = []
    # Iterate over training folders
    for d in train_root.iterdir():
        if d.is_dir() and not d.name.startswith("_"):
            # Default fallback if config isn't found
            base_model = "black-forest-labs/FLUX.1-dev"

            # Construct path to the config file: trainings/{user}/_configs/{job_name}.yaml
            config_path = train_root / "_configs" / f"{d.name}.yaml"

            if config_path.exists():
                try:
                    # Read the yaml to find the model path
                    with open(config_path, 'r') as f:
                        cfg = yaml.safe_load(f)
                        # Navigate: config -> process -> [0] -> model -> name_or_path
                        base_model = cfg["config"]["process"][0]["model"]["name_or_path"]
                except Exception as e:
                    logger.warning(f"Error reading config for {d.name}: {e}")

            # Return object with metadata
            models.append({
                "name": d.name,
                "base_model": base_model
            })

    return {"models": models}


# ------------------------------------------------------------
# Generate endpoint: calls remote worker which writes files and returns relative paths
# ------------------------------------------------------------
@api.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, user_id: str = Depends(get_current_user_id)):
    try:
        # Determine LoRA path (existing logic)
        lora_path = None
        if request.model_name:
            model_folder = get_user_training_path(user_id) / safe_filename(request.model_name)
            safetensors = sorted(
                list(model_folder.glob("*.safetensors")),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if safetensors:
                lora_path = str(safetensors[0])

        job_stub = uuid.uuid4().hex
        out_dir = f"{MOUNT_DIR}/generated/{user_id}/{job_stub}"
        os.makedirs(out_dir, exist_ok=True)

        remote_generate_func = modal.Function.from_name("flex-lora-training", "remote_generate")

        # --- CALL WITH NEW ARGUMENTS ---
        saved_rel_paths = await remote_generate_func.remote.aio(
            prompt=request.prompt,
            num_samples=request.num_samples,
            lora_path=lora_path,
            out_dir=out_dir,
            base_model=request.base_model,  # Pass the base model
            hf_token=request.hf_token  # Pass the token
        )
        # -------------------------------

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

        return GenerateResponse(images=urls, status="success")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Generation failed; check server logs")


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

    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        return {"caption": txt_path.read_text(encoding="utf-8")}

    # Always 200 with empty caption so the UI just treats it as "no caption yet"
    return {"caption": ""}


@api.post("/api/files/caption")
async def save_caption(req: CaptionRequest, user_id: str = Depends(get_current_user_id)):
    mount_root = Path(MOUNT_DIR)
    resolved_mount_root = mount_root.resolve()

    # req.path is like "datasets/default_user/test/2.jpg"
    img_path = (mount_root / req.path).resolve()

    # Stay inside MOUNT_DIR
    if not str(img_path).startswith(str(resolved_mount_root)):
        raise HTTPException(status_code=403, detail="Access denied")

    txt_path = img_path.with_suffix(".txt")

    try:
        rel = txt_path.relative_to(resolved_mount_root)
    except ValueError:
        logger.warning("Caption save outside mount root rejected: %s", txt_path)
        raise HTTPException(status_code=403, detail="Access denied")

    # Simple user check: require the per-user folder segment
    if f"/{user_id}/" not in f"/{rel.as_posix()}/":
        raise HTTPException(status_code=403, detail="Access denied")

    txt_path.write_text(req.caption, encoding="utf-8")
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
    gpu="A100",
    timeout=28800,
    image=image,
    volumes={MOUNT_DIR: model_volume, CACHE_DIR: hf_volume},
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
        job = get_job(config_file, name)
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
        base_model: str = "black-forest-labs/FLUX.1-dev",  # Default to Flux if not specified
        hf_token: Optional[str] = None
):
    from diffusers import AutoPipelineForText2Image
    import torch
    import os
    import uuid
    from PIL import Image

    # 1. AUTHENTICATION
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    logger.info("remote_generate called prompt=%s model=%s", prompt, base_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # 2. CACHING LOGIC: Try local load, failover to download
    pipe = None
    try:
        logger.info(f"Attempting to load {base_model} from local cache...")
        pipe = AutoPipelineForText2Image.from_pretrained(
            base_model,
            cache_dir=CACHE_DIR,
            local_files_only=True,
            torch_dtype=torch_dtype
        )
        logger.info("Loaded successfully from cache.")
    except Exception as e:
        logger.info(f"Model not found in cache (or incomplete). Downloading {base_model}...")
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                base_model,
                cache_dir=CACHE_DIR,
                local_files_only=False,  # Allow download
                token=hf_token,  # Use token for gated models
                torch_dtype=torch_dtype
            )
            # CRITICAL: Save the downloaded model to persistent storage
            hf_volume.commit()
            logger.info("Download complete and volume committed.")
        except Exception as download_error:
            logger.error(f"Failed to download model: {download_error}")
            raise RuntimeError(f"Could not load or download model {base_model}. Check token/internet.")

    pipe.to(device)

    # 3. LORA LOADING
    if lora_path and os.path.exists(lora_path):
        logger.info("Loading LoRA: %s", lora_path)
        try:
            if hasattr(pipe, "load_lora_weights"):
                pipe.load_lora_weights(lora_path)
            elif hasattr(pipe.unet, "load_attn_procs"):
                pipe.unet.load_attn_procs(lora_path)
        except Exception as e:
            logger.warning("Error loading LoRA: %s", e)

    # 4. GENERATION LOOP
    if out_dir is None:
        out_dir = f"{MOUNT_DIR}/generated/tmp_{uuid.uuid4().hex}"
    os.makedirs(out_dir, exist_ok=True)

    saved_rel_paths = []
    for i in range(num_samples):
        # Use a fresh generator for each image for variety
        seed = 1024 + i
        generator = torch.Generator(device=device).manual_seed(seed)

        logger.info(f"Generating sample {i + 1}/{num_samples}...")
        out = pipe(
            prompt,
            guidance_scale=3.5,
            num_inference_steps=20,
            generator=generator
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

    return saved_rel_paths


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

