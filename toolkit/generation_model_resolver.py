import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


SUPPORTED_GENERATION_ARCHITECTURES = {"flux", "flux2_klein"}
SUPPORTED_GENERATION_NETWORK_TYPES = {"lora"}


def infer_generation_architecture(model_config: dict[str, Any]) -> str:
    arch = model_config.get("arch")
    if isinstance(arch, str) and arch.strip():
        return arch.strip().lower()
    if model_config.get("is_flux2_klein"):
        return "flux2_klein"
    if model_config.get("is_flux"):
        return "flux"

    base_model = model_config.get("name_or_path")
    if isinstance(base_model, str):
        normalized = base_model.lower()
        if "flux.2-klein" in normalized or "flux2-klein" in normalized:
            return "flux2_klein"
        if "flux.1" in normalized:
            return "flux"
    return "unknown"


def resolve_generation_checkpoint(model_folder: Path, model_name: str) -> Optional[Path]:
    canonical = model_folder / f"{model_name}.safetensors"
    if canonical.is_file():
        return canonical

    step_pattern = re.compile(rf"^{re.escape(model_name)}_(\d+)\.safetensors$")
    candidates: list[tuple[int, Path]] = []
    if model_folder.is_dir():
        for candidate in model_folder.glob("*.safetensors"):
            match = step_pattern.fullmatch(candidate.name)
            if match and candidate.is_file():
                candidates.append((int(match.group(1)), candidate))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _unavailable_model(
    model_name: str,
    status: str,
    reason: str,
    *,
    base_model: Optional[str] = None,
    architecture: str = "unknown",
    network_type: str = "unknown",
    checkpoint: Optional[Path] = None,
) -> dict[str, Any]:
    return {
        "name": model_name,
        "base_model": base_model,
        "architecture": architecture,
        "network_type": network_type,
        "checkpoint": checkpoint.name if checkpoint else None,
        "selectable": False,
        "status": status,
        "status_reason": reason,
        "lora_path": str(checkpoint) if checkpoint else None,
    }


def resolve_generation_model(train_root: Path, model_name: str) -> dict[str, Any]:
    model_folder = train_root / model_name
    if not model_folder.is_dir():
        return _unavailable_model(model_name, "not_found", "Training folder was not found.")

    config_path = train_root / "_configs" / f"{model_name}.yaml"
    if not config_path.is_file():
        return _unavailable_model(model_name, "missing_config", "Training configuration is missing.")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        processes = config["config"]["process"]
        if not isinstance(processes, list) or not processes or not isinstance(processes[0], dict):
            raise ValueError("config.process must contain a process")
        process = processes[0]
        model_config = process["model"]
        network_config = process["network"]
        if not isinstance(model_config, dict) or not isinstance(network_config, dict):
            raise ValueError("model and network must be mappings")
        base_model = model_config["name_or_path"]
        if not isinstance(base_model, str) or not base_model.strip():
            raise ValueError("model.name_or_path is required")
        base_model = base_model.strip()
        network_type = network_config["type"]
        if not isinstance(network_type, str) or not network_type.strip():
            raise ValueError("network.type is required")
        network_type = network_type.strip().lower()
    except (KeyError, TypeError, ValueError, yaml.YAMLError, OSError) as exc:
        return _unavailable_model(
            model_name,
            "invalid_config",
            f"Training configuration is invalid: {exc}",
        )

    architecture = infer_generation_architecture(model_config)
    checkpoint = resolve_generation_checkpoint(model_folder, model_name)

    if network_type not in SUPPORTED_GENERATION_NETWORK_TYPES:
        return _unavailable_model(
            model_name,
            "unsupported_network",
            f"Adapter type '{network_type}' is not supported for generation.",
            base_model=base_model,
            architecture=architecture,
            network_type=network_type,
            checkpoint=checkpoint,
        )
    if architecture not in SUPPORTED_GENERATION_ARCHITECTURES:
        return _unavailable_model(
            model_name,
            "unsupported_architecture",
            f"Model architecture '{architecture}' is not supported for generation.",
            base_model=base_model,
            architecture=architecture,
            network_type=network_type,
            checkpoint=checkpoint,
        )
    if checkpoint is None:
        return _unavailable_model(
            model_name,
            "missing_checkpoint",
            "No final or numbered LoRA checkpoint is available yet.",
            base_model=base_model,
            architecture=architecture,
            network_type=network_type,
        )

    return {
        "name": model_name,
        "base_model": base_model,
        "architecture": architecture,
        "network_type": network_type,
        "checkpoint": checkpoint.name,
        "selectable": True,
        "status": "ready",
        "status_reason": None,
        "lora_path": str(checkpoint),
    }


def list_generation_models(train_root: Path) -> list[dict[str, Any]]:
    if not train_root.is_dir():
        return []
    model_names: Iterable[str] = (
        child.name
        for child in train_root.iterdir()
        if child.is_dir() and not child.name.startswith("_")
    )
    return [
        resolve_generation_model(train_root, model_name)
        for model_name in sorted(model_names, key=str.casefold)
    ]


def public_generation_model(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model.items() if key != "lora_path"}


def load_and_activate_lora(
    pipe: Any,
    lora_path: str,
    adapter_name: str,
    adapter_weight: float = 1.0,
) -> bool:
    checkpoint = Path(lora_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LoRA checkpoint '{checkpoint.name}' does not exist.")

    loader = getattr(pipe, "load_lora_weights", None)
    if not callable(loader):
        raise RuntimeError(f"Pipeline '{type(pipe).__name__}' does not support LoRA loading.")
    activator = getattr(pipe, "set_adapters", None)
    if not callable(activator):
        raise RuntimeError(f"Pipeline '{type(pipe).__name__}' does not support explicit LoRA activation.")

    try:
        loader(os.fspath(checkpoint), adapter_name=adapter_name)
        activator(adapter_name, adapter_weights=adapter_weight)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load LoRA adapter '{adapter_name}' from '{checkpoint.name}': {exc}"
        ) from exc
    return True
