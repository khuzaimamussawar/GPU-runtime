from __future__ import annotations

from typing import Any

from src.image_pod.adapters import Krea2Adapter


TASK_FAMILY_REGISTRY = {
    "krea2_image": Krea2Adapter,
}


_ADAPTERS: dict[str, Any] = {}


def get_adapter(task_family: str):
    family = str(task_family or "").strip()
    adapter_type = TASK_FAMILY_REGISTRY.get(family)
    if adapter_type is None:
        raise ValueError(f"Unsupported taskFamily: {family or '<missing>'}")
    if family not in _ADAPTERS:
        _ADAPTERS[family] = adapter_type()
    return _ADAPTERS[family]


def supported_task_families() -> list[str]:
    return sorted(TASK_FAMILY_REGISTRY)
