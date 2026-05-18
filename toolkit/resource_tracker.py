import subprocess
import time
from typing import Dict, Optional

import torch

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


_NVIDIA_SMI_CACHE: Dict[str, Optional[float]] = {}
_NVIDIA_SMI_CACHE_TIME = 0.0
_NVIDIA_SMI_CACHE_TTL_SECONDS = 2.0


def _to_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_nvidia_smi() -> Dict[str, Optional[float]]:
    """Best-effort GPU utilization via nvidia-smi."""
    global _NVIDIA_SMI_CACHE_TIME

    now = time.monotonic()
    if _NVIDIA_SMI_CACHE and (now - _NVIDIA_SMI_CACHE_TIME) < _NVIDIA_SMI_CACHE_TTL_SECONDS:
        return dict(_NVIDIA_SMI_CACHE)

    query = "utilization.gpu,memory.used,memory.total"
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        _NVIDIA_SMI_CACHE.clear()
        _NVIDIA_SMI_CACHE_TIME = now
        return {}

    if result.returncode != 0:
        _NVIDIA_SMI_CACHE.clear()
        _NVIDIA_SMI_CACHE_TIME = now
        return {}

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line:
        _NVIDIA_SMI_CACHE.clear()
        _NVIDIA_SMI_CACHE_TIME = now
        return {}

    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        _NVIDIA_SMI_CACHE.clear()
        _NVIDIA_SMI_CACHE_TIME = now
        return {}

    snapshot = {
        "resource/gpu_utilization_percent": _to_float(parts[0]),
        "resource/gpu_memory_used_mb": _to_float(parts[1]),
        "resource/gpu_memory_total_mb": _to_float(parts[2]),
    }
    _NVIDIA_SMI_CACHE.clear()
    _NVIDIA_SMI_CACHE.update(snapshot)
    _NVIDIA_SMI_CACHE_TIME = now
    return snapshot


def get_cpu_stats() -> Dict[str, Optional[float]]:
    if psutil is None:
        return {}

    try:
        load = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
    except Exception:
        return {}

    return {
        "resource/cpu_load_percent": _to_float(load),
        "resource/cpu_memory_percent": _to_float(mem.percent),
        "resource/cpu_memory_available_mb": _to_float(mem.available / (1024 * 1024)),
    }


def get_gpu_stats() -> Dict[str, Optional[float]]:
    if not torch.cuda.is_available():
        return {}

    stats: Dict[str, Optional[float]] = {}
    try:
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        stats["resource/gpu_memory_allocated_mb"] = _to_float(allocated)
        stats["resource/gpu_memory_reserved_mb"] = _to_float(reserved)
    except Exception:
        pass

    stats.update(_run_nvidia_smi())
    return stats


def get_resource_snapshot() -> Dict[str, Optional[float]]:
    """Return a combined snapshot of CPU and GPU metrics."""
    snapshot: Dict[str, Optional[float]] = {}
    snapshot.update(get_cpu_stats())
    snapshot.update(get_gpu_stats())
    return snapshot
