#!/usr/bin/env python3
"""
run_modal.py - wheelhouse-first, hardened backend

- Uses local wheelhouse first (fallback to PyPI)
- Keeps Python 3.12 to match cp312 wheels in your wheelhouse
- Hardened: safe_commit, filename sanitation, streaming uploads, job_store per-job keys,
  generate saves images to mounted volume and returns /api/files URLs (no base64 payloads)
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

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_, **__):
        return False

load_dotenv()

import modal
from modal import asgi_app

# FastAPI imports
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ------------------------------------------------------------
# Configuration & paths
# ------------------------------------------------------------
MOUNT_DIR = "/root/modal_output"
CACHE_DIR = "/root/.cache/huggingface"
LOCAL_AITOOLKIT_PATH = r"C:\Users\tranh\PycharmProjects\ai-toolkit"
LOCAL_WHEELHOUSE_PATH = r"C:\Users\tranh\PycharmProjects\ai-toolkit\wheelhouse"

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
    return datetime.utcnow().isoformat() + "Z"

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
    # System deps (kept reasonably small, but present for compiled packages)
    .apt_install(
        "libgl1", "libglib2.0-0", "pkg-config", "build-essential", "gfortran",
        "cmake", "ninja-build", "libopenblas-dev", "liblapack-dev",
        "ca-certificates", "git", "curl", "wget", "libatlas-base-dev",
        "libblas-dev", "libsentencepiece-dev", "libprotobuf-dev", "protobuf-compiler",
        "python3-dev"
    )
    # 1) Attach the local wheelhouse early (copy->remote)
    .add_local_dir(LOCAL_WHEELHOUSE_PATH, remote_path="/root/wheels", copy=True)
    # 2) Ensure pip + build tooling installed
    .run_commands(
        "echo '=== Upgrade pip/setuptools/wheel/build ==='",
        "python -m pip install --upgrade pip setuptools wheel build",
        "python -m pip --version",
        "python -m pip debug --verbose | true",
        "echo 'Wheelhouse contents:'",
        "ls -lah /root/wheels | true",
    )
    # 3) Write constraints/pins for binary-critical packages (to avoid resolver bouncing)
    .run_commands(
        "echo '=== Write constraints (binary pins) ==='",
        "echo 'numpy==1.26.4' > /root/constraints.txt",
        "echo 'scipy==1.14.1' >> /root/constraints.txt",
        "echo 'pillow==12.0.0' >> /root/constraints.txt",
    )
    # 4) Pre-install binary-critical packages using wheelhouse first (prefer-binary)
    .run_commands(
        "echo '=== Preinstall binary-critical packages from wheelhouse (prefer-binary) ==='",
        "python -m pip install --prefer-binary --find-links /root/wheels --constraint /root/constraints.txt "
        "numpy==1.26.4 scipy==1.14.1 pillow==12.0.0 | true",
        "python -c \"import numpy, scipy, PIL; print('preinstalled:', numpy.__version__, scipy.__version__, PIL.__version__)\" | true",
    )
    # 5) Install any local wheelhouse wheels as a fast path (no-deps ensures wheels present)
    .run_commands(
        "echo '=== Install wheels from wheelhouse (no-deps) ==='",
        "python -m pip install --no-index --find-links /root/wheels --prefer-binary --no-deps /root/wheels/*.whl || true",
    )
    # 6) Install application packages using wheelhouse-first (PyPI fallback allowed)
    .run_commands(
        "echo '=== Installing application packages (wheelhouse-first; PyPI fallback) ==='",
        "python -m pip install --prefer-binary --find-links /root/wheels --constraint /root/constraints.txt "
        "transformers python-dotenv accelerate ftfy safetensors albumentations lycoris-lora timm einops "
        "opencv-python-headless huggingface_hub peft lpips hf_transfer flatten_json pyyaml oyaml tensorboard "
        "toml albucore pydantic omegaconf k-diffusion controlnet_aux optimum-quanto python-slugify open_clip_torch "
        "bitsandbytes pytorch_fid sentencepiece pytorch-wavelets matplotlib diffusers || true"
    )
    # Env
    .env({
        "HUGGINGFACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
        "CUDA_VISIBLE_DEVICES": "0",
    })
    # 7) Copy local source
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

class GenerateResponse(BaseModel):
    images: List[str]
    status: str

class TrainRequest(BaseModel):
    config: dict
    recover: Optional[bool] = False
    name: Optional[str] = None

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

class JobStatus(BaseModel):
    job_id: str
    status: str
    config_name: Optional[str] = None
    created_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None

class JobListResponse(BaseModel):
    jobs: List[JobStatus]

# ------------------------------------------------------------
# Authentication stub
# ------------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user_id(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    if token.startswith("test_user_"):
        return token
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
        job_store.delete(job_id)
    except Exception:
        pass

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
    if job_record.get("status") in ["started", "running"]:
        try:
            call = modal.FunctionCall.from_id(job_id)
            try:
                result = call.get(timeout=0)
                job_record["status"] = "completed"
                job_record["result"] = result
            except modal.TimeoutError:
                job_record["status"] = "running"
            except Exception as e:
                job_record["status"] = "failed"
                job_record["error"] = str(e)
            _put_job_record(job_id, job_record)
        except Exception as e:
            logger.warning("Could not refresh job status for %s: %s", job_id, e)
    return JobStatus(**job_record)

# ------------------------------------------------------------
# File serving (allowlist)
# ------------------------------------------------------------
@api.get("/api/files/{file_path:path}")
async def get_file(file_path: str, user_id: str = Depends(get_current_user_id)):
    safe_root = Path(MOUNT_DIR).resolve()
    requested_path = (safe_root / file_path).resolve()
    if not str(requested_path).startswith(str(safe_root)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not requested_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    ext = requested_path.suffix.lower()
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
    return FileResponse(requested_path, media_type=media_type)

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
        shutil.rmtree(ds_path)
        safe_commit(model_volume)
    return {"status": "deleted"}

@api.post("/api/datasets/upload")
async def upload_dataset_files(
    name: str = Body(...),
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user_id)
):
    ds_path = get_user_dataset_path(user_id) / safe_filename(name)
    ds_path.mkdir(exist_ok=True)
    count = 0
    for upload in files:
        filename = safe_filename(upload.filename)
        dest = ds_path / filename
        size = 0
        try:
            with open(dest, "wb") as out_f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        out_f.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail="File too large")
                    out_f.write(chunk)
        finally:
            try:
                await upload.close()
            except Exception:
                pass
        count += 1
    safe_commit(model_volume)
    return {"status": "success", "count": count}

@api.get("/api/datasets/{name}/images")
async def list_dataset_images(name: str, user_id: str = Depends(get_current_user_id)):
    ds_path = get_user_dataset_path(user_id) / safe_filename(name)
    if not ds_path.exists():
        return {"images": []}
    images = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".txt", ".mp4"}
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
        if not isinstance(cfg, dict) or "config" not in cfg:
            raise HTTPException(status_code=400, detail="Invalid config: missing 'config' key")
        proc = cfg["config"].get("process")
        if not isinstance(proc, list) or len(proc) == 0:
            raise HTTPException(status_code=400, detail="Invalid config: 'config.process' must be a non-empty list")
        train_func = modal.Function.from_name("flex-lora-training", "main")
        job_name = request.name or str(uuid.uuid4())
        user_train_root = f"{MOUNT_DIR}/trainings/{user_id}"
        try:
            cfg["config"]["process"][0]["training_folder"] = user_train_root
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
        call = train_func.spawn(
            config_file_list_str=config_path,
            recover=request.recover,
            name=job_name
        )
        job_record = {
            "job_id": call.call_id,
            "user_id": user_id,
            "status": "started",
            "config_name": job_name,
            "created_at": iso_now(),
            "result": None,
            "error": None,
        }
        _put_job_record(job_record["job_id"], job_record)
        return TrainResponse(job_id=call.call_id, status="started")
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
    path = get_user_training_path(user_id)
    if not path.exists():
        return {"models": []}
    models = [d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith("_")]
    return {"models": models}

# ------------------------------------------------------------
# Generate endpoint: calls remote worker which writes files and returns relative paths
# ------------------------------------------------------------
@api.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, user_id: str = Depends(get_current_user_id)):
    try:
        lora_path = None
        if request.model_name:
            model_folder = get_user_training_path(user_id) / safe_filename(request.model_name)
            safetensors = sorted(list(model_folder.glob("*.safetensors")), key=lambda f: f.stat().st_mtime, reverse=True)
            if safetensors:
                lora_path = str(safetensors[0])
        job_stub = uuid.uuid4().hex
        out_dir = f"{MOUNT_DIR}/generated/{user_id}/{job_stub}"
        os.makedirs(out_dir, exist_ok=True)
        remote_generate_func = modal.Function.from_name("flex-lora-training", "remote_generate")
        saved_rel_paths = await remote_generate_func.remote.aio(
            prompt=request.prompt,
            num_samples=request.num_samples,
            lora_path=lora_path,
            out_dir=out_dir
        )
        safe_commit(model_volume)
        urls = []
        for p in saved_rel_paths:
            try:
                rp = Path(p)
                if rp.is_absolute():
                    rp = rp.relative_to(Path(MOUNT_DIR))
                urls.append(f"/api/files/{str(rp)}")
            except Exception:
                urls.append(f"/api/files/{p}")
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
async def get_caption(path: str = Query(...), user_id: str = Depends(get_current_user_id)):
    img_path = (Path(MOUNT_DIR) / path).resolve()
    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        return {"caption": txt_path.read_text()}
    raise HTTPException(status_code=404, detail="Caption not found")

@api.post("/api/files/caption")
async def save_caption(req: CaptionRequest, user_id: str = Depends(get_current_user_id)):
    img_path = (Path(MOUNT_DIR) / req.path).resolve()
    txt_path = img_path.with_suffix(".txt")
    if user_id not in str(txt_path):
        raise HTTPException(status_code=403, detail="Access denied")
    txt_path.write_text(req.caption)
    safe_commit(model_volume)
    return {"status": "saved"}

@api.delete("/api/files/{path:path}")
async def delete_file(path: str, user_id: str = Depends(get_current_user_id)):
    target = (Path(MOUNT_DIR) / path).resolve()
    if user_id not in str(target):
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

@app.function(gpu="A100", timeout=28800, image=image, volumes={MOUNT_DIR: model_volume, CACHE_DIR: hf_volume})
def main(config_file_list_str: str, recover: bool = False, name: str | None = None):
    if "/root/ai-toolkit" not in sys.path:
        sys.path.insert(0, "/root/ai-toolkit")
    from toolkit.job import get_job
    config_file = config_file_list_str.split(",")[0]
    logger.info("Starting training job: %s", name)
    logger.info("Using config file: %s", config_file)
    job = get_job(config_file, name)
    job.run()
    try:
        model_volume.commit()
    except Exception as e:
        logger.warning("model_volume.commit() failed in worker: %s", e)
    job.cleanup()
    logger.info("Job completed.")

@app.function(gpu="A100", timeout=1800, image=image, volumes={MOUNT_DIR: model_volume, CACHE_DIR: hf_volume})
def remote_generate(
    prompt: str,
    num_samples: int = 1,
    lora_path: Optional[str] = None,
    out_dir: Optional[str] = None
):
    from diffusers import AutoPipelineForText2Image
    import torch
    from PIL import Image
    logger.info("remote_generate called prompt=%s num_samples=%s lora=%s out_dir=%s",
                prompt, num_samples, bool(lora_path), out_dir)
    base_repo = "black-forest-labs/FLUX.1-dev"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if not _ensure_cache_has_model(base_repo, cache_dir=CACHE_DIR):
        raise RuntimeError("Model not found in cache")
    pipe = AutoPipelineForText2Image.from_pretrained(
        base_repo,
        cache_dir=CACHE_DIR,
        local_files_only=True,
        torch_dtype=torch_dtype
    )
    pipe.to(device)
    if lora_path and os.path.exists(lora_path):
        logger.info("Loading LoRA: %s", lora_path)
        try:
            if hasattr(pipe, "load_lora_weights"):
                pipe.load_lora_weights(lora_path)
            elif hasattr(pipe.unet, "load_attn_procs"):
                pipe.unet.load_attn_procs(lora_path)
        except Exception as e:
            logger.warning("Error loading LoRA: %s", e)
    if out_dir is None:
        out_dir = f"{MOUNT_DIR}/generated/tmp_{uuid.uuid4().hex}"
    os.makedirs(out_dir, exist_ok=True)
    saved_rel_paths = []
    for i in range(num_samples):
        generator = torch.Generator(device=device).manual_seed(1024 + i) if device == "cuda" else torch.Generator().manual_seed(1024 + i)
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
        main.call(config_file_list_str=",".join(args.config_file_list), recover=args.recover, name=args.name)
    except Exception as e:
        logger.exception("Local CLI main call failed: %s", e)
        raise
