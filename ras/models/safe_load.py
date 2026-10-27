"""Load third-party checkpoints without letting the pickle run arbitrary code.

`torch.load(weights_only=True)` in PyTorch 2.2 refuses the numpy arrays and module objects some released
checkpoints contain. Instead of falling back to a full unpickle, `safe_torch_load` resolves only the
classes on an explicit allow-list (tensors, numpy arrays, plain containers, and model classes we
vendored and reviewed). Anything else raises `UnsafeCheckpoint`.
"""

from __future__ import annotations

import codecs
import collections
import copyreg
import io
import pickle
import types
from pathlib import Path

import numpy as np
import torch


class UnsafeCheckpoint(pickle.UnpicklingError):
    pass


def _load_storage_bytes(b: bytes):  # legacy whole-module pickles store each tensor as a nested torch.save
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)


_BASE = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "slice"): slice,
    ("_codecs", "encode"): codecs.encode,
    ("copyreg", "_reconstructor"): copyreg._reconstructor,
    ("builtins", "object"): object,
    ("torch", "device"): torch.device,
    ("torch", "Size"): torch.Size,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("torch.storage", "_load_from_bytes"): _load_storage_bytes,
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}
for _name in ("FloatStorage", "DoubleStorage", "HalfStorage", "LongStorage", "IntStorage", "ShortStorage",
              "CharStorage", "ByteStorage", "BoolStorage", "BFloat16Storage"):
    _BASE[("torch", _name)] = getattr(torch, _name)

# torch.nn building blocks that whole-module pickles reference; instantiated only as empty shells whose
# state then comes from tensors
_NN_MODULES = ("torch.nn.modules.activation", "torch.nn.modules.container", "torch.nn.modules.conv",
               "torch.nn.modules.dropout", "torch.nn.modules.linear", "torch.nn.modules.normalization",
               "torch.nn.modules.sparse", "torch.nn.modules.batchnorm", "torch.nn.modules.pooling")


def _pickle_module(extra: dict[tuple[str, str], object]) -> types.ModuleType:
    allowed = {**_BASE, **extra}

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in allowed:
                return allowed[(module, name)]
            if module in _NN_MODULES:
                obj = getattr(__import__(module, fromlist=[name]), name, None)
                if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                    return obj
            raise UnsafeCheckpoint(f"checkpoint references {module}.{name}, which is not on the allow-list")

    mod = types.ModuleType("ras_restricted_pickle")
    mod.Unpickler = Restricted
    mod.load = lambda f, **kw: Restricted(f, **kw).load()
    mod.__dict__["UnpicklingError"] = pickle.UnpicklingError
    return mod


def safe_torch_load(path: str | Path, extra: dict[tuple[str, str], object] | None = None):
    """torch.load through the allow-list (both the zip and the legacy format)."""
    return torch.load(str(path), map_location="cpu", weights_only=False, pickle_module=_pickle_module(extra or {}))


def safe_pickle_load(path: str | Path, extra: dict[tuple[str, str], object] | None = None):
    """Plain pickle.load through the allow-list (for checkpoints written with pickle.dump)."""
    with open(path, "rb") as f:
        return _pickle_module(extra or {}).load(f)
