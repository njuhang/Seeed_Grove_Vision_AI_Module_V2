from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

FLASH_SLOT_SIZE = 0x00200000


@dataclass(frozen=True)
class VelaSpec:
    executable: str | None = None
    executable_args: tuple[str, ...] = ()
    accelerator_config: str | None = None
    config: str | None = None
    system_config: str | None = None
    memory_mode: str | None = None
    optimise: str | None = None
    tensor_allocator: str | None = None
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str
    source: dict[str, Any]
    input_shape: list[int]
    calibration_dataset: str
    tier: int
    flash_address: int
    model_zoo_ref: str | None = None
    vela: VelaSpec | None = None


def _parse_string_list(raw: Any, *, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list of strings")
    return tuple(str(item) for item in raw)


def _parse_vela_spec(raw: Any) -> VelaSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("vela must be a mapping when provided")
    return VelaSpec(
        executable=raw.get("executable"),
        executable_args=_parse_string_list(raw.get("executable_args"), field_name="vela.executable_args"),
        accelerator_config=raw.get("accelerator_config"),
        config=raw.get("config"),
        system_config=raw.get("system_config"),
        memory_mode=raw.get("memory_mode"),
        optimise=raw.get("optimise"),
        tensor_allocator=raw.get("tensor_allocator"),
        extra_args=_parse_string_list(raw.get("extra_args"), field_name="vela.extra_args"),
    )


def _validate_model(raw: dict[str, Any]) -> ModelSpec:
    flash_address = int(raw["flash_address"])
    if flash_address % 0x1000 != 0:
        raise ValueError("flash_address must be 4KB aligned")
    if raw["input_shape"][0] != 1:
        raise ValueError("phase-1 only supports batch size 1")
    return ModelSpec(
        name=raw["name"],
        task=raw["task"],
        source=raw["source"],
        input_shape=list(raw["input_shape"]),
        calibration_dataset=raw["calibration_dataset"],
        tier=int(raw["tier"]),
        flash_address=flash_address,
        model_zoo_ref=raw.get("model_zoo_ref"),
        vela=_parse_vela_spec(raw.get("vela")),
    )


def load_models(manifest_path: str | Path) -> list[ModelSpec]:
    payload = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    models = [_validate_model(item) for item in payload["models"]]

    seen_names: set[str] = set()
    for model in models:
        if model.name in seen_names:
            raise ValueError(f"duplicate model name: {model.name}")
        seen_names.add(model.name)

    by_address = sorted(models, key=lambda model: model.flash_address)
    for previous, current in zip(by_address, by_address[1:]):
        if current.flash_address < previous.flash_address + FLASH_SLOT_SIZE:
            raise ValueError(
                f"flash slots overlap: {previous.name} at 0x{previous.flash_address:x} "
                f"and {current.name} at 0x{current.flash_address:x}"
            )

    return models
