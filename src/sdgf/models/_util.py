"""Helpers shared by the real backends: lazy SDK imports and spec-param access."""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any

from sdgf.models.base import ModelBackendError


def lazy_import(module: str, extra: str) -> ModuleType:
    """Import an optional SDK at setup time, with an install hint if it is missing."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ModelBackendError(
            f"backend needs the optional package {module!r}; install it with: pip install {extra}"
        ) from e


def api_key_from_env(params: dict[str, Any], backend: str) -> str | None:
    """Keys never live in a spec; params.api_key_env names the environment variable."""
    if "api_key" in params:
        raise ModelBackendError(
            f"{backend}: put the API key in an environment variable and name it with "
            "params.api_key_env; literal keys are not allowed in a spec"
        )
    env = params.get("api_key_env")
    if env is None:
        return None
    key = os.environ.get(env)
    if not key:
        raise ModelBackendError(f"{backend}: environment variable {env!r} is not set")
    return key


def reject_tools(backend: str, tools: list[Any] | None) -> None:
    if tools:
        raise ModelBackendError(f"backend {backend!r} does not support tool calls")
