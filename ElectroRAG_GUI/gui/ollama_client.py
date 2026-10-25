"""Minimal local Ollama status check using only the standard library. Read-only GET requests."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaStatus:
    reachable: bool
    version: str = ""
    installed_models: tuple[str, ...] = ()
    loaded_models: tuple[str, ...] = ()
    model_installed: bool = False
    model_loaded: bool = False
    error: str = ""


def _get_json(url: str, timeout: float) -> dict:
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Ollama URL must start with http:// or https://")
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (scheme validated above)
        return json.loads(response.read().decode("utf-8"))


def _has_model(target: str, names: tuple[str, ...]) -> bool:
    if target in names:
        return True
    if ":" not in target:
        return any(n.split(":")[0] == target for n in names)
    return False


def get_status(base_url: str, model: str, timeout: float = 2.0) -> OllamaStatus:
    try:
        version = str(_get_json(f"{base_url}/api/version", timeout).get("version", ""))
        tags = _get_json(f"{base_url}/api/tags", timeout)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        return OllamaStatus(False, error=f"{type(exc).__name__}: {exc}")

    installed = tuple(m.get("name", "") for m in tags.get("models", []))
    try:
        loaded = tuple(m.get("name", "") for m in _get_json(f"{base_url}/api/ps", timeout).get("models", []))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        loaded = ()

    return OllamaStatus(
        reachable=True,
        version=version,
        installed_models=installed,
        loaded_models=loaded,
        model_installed=_has_model(model, installed),
        model_loaded=_has_model(model, loaded),
    )
