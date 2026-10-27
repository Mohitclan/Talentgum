"""Loads each optional model once per process and hands it to the pipeline when a criterion needs it."""

from __future__ import annotations

import logging
import threading

from . import MODELS, is_available

log = logging.getLogger("ras.models")


class ModelHub:
    def __init__(self, device: str | None = None):
        self._device = device
        self._cache: dict[str, object] = {}
        self._lock = threading.Lock()
        self.errors: dict[str, str] = {}

    def available(self, name: str) -> bool:
        return name in MODELS and is_available(name) and name not in self.errors

    def get(self, name: str):
        """The loaded model, or None (and a remembered reason) when it cannot be used."""
        with self._lock:
            if name in self._cache:
                return self._cache[name]
            if not self.available(name):
                self.errors.setdefault(name, "not_downloaded" if name in MODELS else "unknown_model")
                return None
            try:
                if name == "hft":
                    from ras.core.transcribe.hft_adapter import HftTranscriber

                    obj = HftTranscriber(device=self._device)
                elif name == "kong":
                    from ras.core.transcribe.kong_adapter import KongTranscriber

                    obj = KongTranscriber(device=self._device)
                elif name == "dynamics":
                    from ras.core.dynamics_model import DynamicsEstimator

                    obj = DynamicsEstimator(device=self._device)
                else:
                    raise KeyError(name)
            except Exception as exc:  # missing extra, broken file: the criterion becomes "not assessed"
                log.warning("model %s unavailable: %s", name, exc)
                self.errors[name] = str(exc)
                return None
            self._cache[name] = obj
            return obj

    def downloaded(self) -> list[str]:
        return [n for n in MODELS if is_available(n)]


_default: ModelHub | None = None


def default_hub() -> ModelHub:
    global _default
    if _default is None:
        _default = ModelHub()
    return _default
