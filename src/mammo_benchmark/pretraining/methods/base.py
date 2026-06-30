from __future__ import annotations

from importlib import import_module
from types import ModuleType

METHOD_REGISTRY = {
    "dinov2": "methods.dinov2",
    "dinov3": "methods.dinov3",
    "mae": "methods.mae",
}


def list_methods() -> list[str]:
    return sorted(METHOD_REGISTRY)


def get_method_module(method_name: str) -> ModuleType:
    try:
        module_path = METHOD_REGISTRY[method_name]
    except KeyError as exc:
        available = ", ".join(list_methods())
        raise ValueError(f"Unknown method '{method_name}'. Available methods: {available}") from exc

    return import_module(module_path)
