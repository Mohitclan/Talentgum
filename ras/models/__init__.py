"""Optional open-source models: where each comes from, its licence, which criteria use it, and how to get it.

Weights are not in git or in the default image. `ras models download <name>` fetches them into
`RAS_MODELS_DIR` (default `./models`), checks each file's SHA-256 and converts whole-module pickles to
plain state dicts through the allow-list loader (ras.models.safe_load), so nothing downloaded is ever
unpickled freely. A model that is not downloaded makes the criteria that need it "not assessed".
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    url: str
    sha256: str
    name: str  # saved as <models_dir>/<model>/<name>


@dataclass(frozen=True)
class ModelSpec:
    name: str
    title: str
    licence: str
    source: str
    role: str
    criteria: tuple[str, ...]
    files: tuple[ModelFile, ...]
    ready_file: str  # present once downloaded and converted
    size_mb: int
    notes: str = ""
    extra_pip: tuple[str, ...] = field(default_factory=tuple)


MODELS: dict[str, ModelSpec] = {m.name: m for m in [
    ModelSpec(
        name="hft",
        title="hFT-Transformer (Sony, ISMIR 2023)",
        licence="MIT",
        source="https://github.com/sony/hFT-Transformer @ 71a2ee06",
        role="second transcriber: a 'missing' or wrong note it hears correctly becomes 'uncertain'",
        criteria=("missed_notes", "incorrect_notes", "note_duration"),
        files=(ModelFile("https://github.com/sony/hFT-Transformer/releases/download/ismir2023/checkpoint.zip",
                         "bb3c88758e4802e6b3881ccb6dc7927417fb43f76980df3739ab2bbc418d70c5", "checkpoint.zip"),),
        ready_file="hft_maestro_v3.pt",
        size_mb=21,
        notes="Trained on MAESTRO v3 (concert grand). Re-check on student keyboards before trusting its votes.",
    ),
    ModelSpec(
        name="kong",
        title="High-resolution piano transcription with pedals (ByteDance, Kong et al. 2021)",
        licence="MIT (piano_transcription_inference on PyPI); weights on Zenodo record 4034264",
        source="https://github.com/qiuqiangkong/piano_transcription_inference 0.0.6",
        role="sustain-pedal and velocity cross-check; a third transcription vote",
        criteria=("sustain_pedal", "dynamics", "missed_notes"),
        files=(ModelFile("https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth"
                         "?download=1",
                         "c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141",
                         "note_pedal_crnn.pth"),),
        ready_file="note_pedal_crnn.pth",
        size_mb=164,
        extra_pip=("piano_transcription_inference==0.0.6", "torchlibrosa==0.1.0"),
        notes="Pedal F1 0.92 on MAESTRO. Needs the 'models' extra.",
    ),
    ModelSpec(
        name="dynamics",
        title="Piano dynamics estimation (He et al., ICASSP 2026)",
        licence="MIT (encoder adapted from GuzhengTech99, no licence file upstream: see DECISIONS.md)",
        source="https://github.com/zhanh-he/piano-dynamic-estimation @ 3beac7a2",
        role="estimates pp/p/mf/f/ff from audio, so dynamics can be checked even when note velocities are flat",
        criteria=("dynamics", "dynamic_consistency"),
        files=(ModelFile("https://raw.githubusercontent.com/zhanh-he/piano-dynamic-estimation/"
                         "3beac7a20ffb0607d57d93c095b00815cd146b88/workspaces/checkpoints/"
                         "formal%20latent8%20mmoe8%205x5-sone-MultiTaskCNN-%5B%27dynamic%27%2C%20%27change_point"
                         "%27%2C%20%27beat%27%2C%20%27downbeat%27%5D-fold0/epoch_96_valf1_0.53453.pth",
                         "2094a56896b2f62b832253c17338908dfad041a72042c938e8afe9f37e43e453",
                         "dynamics_raw.pth"),),
        ready_file="dynamics_multitask.pt",
        size_mb=6,
        notes="Validation F1 0.53 on Chopin mazurkas (5 levels). Use for direction of change, not absolute level.",
    ),
]}


def models_dir() -> Path:
    return Path(os.environ.get("RAS_MODELS_DIR", "models")).expanduser()


def model_path(name: str) -> Path:
    return models_dir() / name / MODELS[name].ready_file


def is_available(name: str) -> bool:
    return model_path(name).is_file()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(f: ModelFile, dest: Path) -> None:
    if dest.is_file() and _sha256(dest) == f.sha256:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        with urllib.request.urlopen(f.url, timeout=60) as r:  # noqa: S310 (fixed https URLs above)
            shutil.copyfileobj(r, tmp)
    got = _sha256(Path(tmp.name))
    if got != f.sha256:
        os.unlink(tmp.name)
        raise RuntimeError(f"{dest.name}: SHA-256 mismatch (expected {f.sha256[:12]}, got {got[:12]})")
    os.replace(tmp.name, dest)


def _convert(name: str, d: Path) -> None:
    """Turn the downloaded file into a plain state dict (loaded later with weights_only=True)."""
    import torch

    from .safe_load import safe_pickle_load, safe_torch_load

    if name == "hft":
        from .vendor import hft_model as H

        with zipfile.ZipFile(d / "checkpoint.zip") as z:
            pkl = z.read("checkpoint/MAESTRO-V3/model_016_003.pkl")
            params = json.loads(z.read("checkpoint/MAESTRO-V3/parameter.json"))
        tmp = d / "model_016_003.pkl"
        tmp.write_bytes(pkl)
        classes = ("Model_SPEC2MIDI", "Encoder_SPEC2MIDI", "Decoder_SPEC2MIDI", "EncoderLayer", "DecoderLayer",
                   "DecoderLayer_Zero", "MultiHeadAttentionLayer", "PositionwiseFeedforwardLayer")
        model = safe_pickle_load(tmp, {("model.model_spec2midi", c): getattr(H, c) for c in classes})
        H.build(params).load_state_dict(model.state_dict(), strict=True)  # proves the vendored code matches
        torch.save({"params": params, "state_dict": model.state_dict()}, d / MODELS[name].ready_file)
        tmp.unlink()
    elif name == "dynamics":
        from .vendor.pde_model import MultiTaskCNN

        ckpt = safe_torch_load(d / "dynamics_raw.pth")
        cfg = {k: ckpt["cfg"][k] for k in ("feature", "cnn")}
        MultiTaskCNN(cfg).load_state_dict(ckpt["model"], strict=True)
        torch.save({"cfg": json.loads(json.dumps(cfg)), "state_dict": ckpt["model"]}, d / MODELS[name].ready_file)
    elif name == "kong":
        torch.load(d / MODELS[name].ready_file, map_location="cpu", weights_only=True)  # already plain tensors


def download(name: str) -> Path:
    spec = MODELS[name]
    d = models_dir() / name
    for f in spec.files:
        _fetch(f, d / f.name)
    _convert(name, d)
    (d / "MODEL.json").write_text(json.dumps({"name": name, "title": spec.title, "licence": spec.licence,
                                              "source": spec.source}, indent=2))
    return model_path(name)


def status() -> list[dict]:
    return [{"name": m.name, "title": m.title, "licence": m.licence, "available": is_available(m.name),
             "size_mb": m.size_mb, "criteria": list(m.criteria), "role": m.role} for m in MODELS.values()]
