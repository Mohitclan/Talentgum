"""Annotate one class recording: transcribe it in pieces, find the song attempts, write the record and clips."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from ras.core.audio.decode import decode_audio, normalise_level
from ras.core.score.reference import load_reference
from ras.types import PerfNote

from .find import SegmentConfig, SongModel, find_clips, song_model

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".mp4", ".webm", ".mov"}
REFERENCE_EXTS = {".mxl", ".musicxml", ".xml", ".mid", ".midi"}
SR = 44100


def student_id(folder_name: str, keep_names: bool = False) -> str:
    """A short stable id for a student folder, so files shared outside the team carry no name.

    Pseudonymous, not anonymous: anyone with the folder names can recompute it. Set RAS_ID_SALT to make it
    harder to reverse.
    """
    if keep_names:
        return folder_name
    salt = os.environ.get("RAS_ID_SALT", "")
    return "s_" + hashlib.sha256((salt + folder_name.strip().lower()).encode()).hexdigest()[:8]


def load_songs(sources: list[Path] | Path, cfg: SegmentConfig | None = None) -> list[SongModel]:
    """Song models from a folder of MXL/MIDI files (file name = song id) or from `id=path` pairs."""
    cfg = cfg or SegmentConfig()
    pairs: list[tuple[str, Path]] = []
    for src in sources if isinstance(sources, list) else [sources]:
        if isinstance(src, tuple):
            pairs.append(src)
        elif src.is_dir():
            pairs += [(p.stem.strip().replace(" ", "_"), p) for p in sorted(src.iterdir())
                      if p.suffix.lower() in REFERENCE_EXTS]
        else:
            pairs.append((src.stem.strip().replace(" ", "_"), src))
    ids = [i for i, _ in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"two reference files have the same song id: {sorted(i for i in ids if ids.count(i) > 1)}")
    return [song_model(i, load_reference(p), cfg) for i, p in pairs]


def merge_chunks(parts: list[tuple[float, float, list[PerfNote]]], overlap_s: float) -> list[PerfNote]:
    """Join the notes of overlapping chunks, keeping each chunk's middle (the edges are least reliable).

    `parts`: (start_s, end_s, notes with absolute times), in order. Neighbouring chunks share `overlap_s`
    seconds; the cut sits in the middle of that overlap.
    """
    out: list[PerfNote] = []
    for k, (start, end, notes) in enumerate(parts):
        lo = start + overlap_s / 2 if k > 0 else -np.inf
        hi = end - overlap_s / 2 if k < len(parts) - 1 else np.inf
        out += [n for n in notes if lo <= n.onset < hi]
    return sorted(out, key=lambda n: (n.onset, n.pitch))


def transcribe_long(audio: np.ndarray, sr: int, transcriber, chunk_s: float = 600.0, overlap_s: float = 10.0,
                    log=None) -> list[PerfNote]:
    """Transcribe a long recording in overlapping chunks of `chunk_s` seconds."""
    if chunk_s <= 2 * overlap_s:
        raise ValueError("chunk_s must be more than twice overlap_s")
    total = len(audio) / sr
    parts, start = [], 0.0
    while True:
        end = min(start + chunk_s, total)
        chunk = normalise_level(audio[int(start * sr):int(end * sr)])
        notes = transcriber.transcribe(chunk, sr).notes
        parts.append((start, end, [PerfNote(n.pitch, n.onset + start, n.offset + start, n.velocity) for n in notes]))
        if log:
            log(f"transcribed {end / 60:.1f} of {total / 60:.1f} min")
        if end >= total:
            break
        start += chunk_s - overlap_s
    return merge_chunks(parts, overlap_s)


def _save_notes(path: Path, notes: list[PerfNote], version: str, duration_s: float) -> None:
    path.write_text(json.dumps({"transcriber": version, "duration_s": round(duration_s, 2),
                                "notes": [[n.pitch, round(n.onset, 3), round(n.offset, 3), n.velocity]
                                          for n in notes]}))


def _load_notes(path: Path, version: str) -> list[PerfNote] | None:
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    return [PerfNote(*n) for n in d["notes"]] if d.get("transcriber") == version else None


def annotate_class(path: str | Path, songs: list[SongModel], transcriber, out_dir: str | Path,
                   cfg: SegmentConfig | None = None, student: str = "unknown", keep_names: bool = False,
                   cut_clips: bool = True, clip_format: str = "flac", chunk_s: float = 600.0,
                   reuse: bool = False, log=None) -> dict:
    """Write `<out>/<student>/<class>.json` (the annotation), the transcript, and the clips; return the record."""
    import soundfile as sf

    cfg = cfg or SegmentConfig()
    path, out_dir = Path(path), Path(out_dir)
    sid = student_id(student, keep_names)
    work = out_dir / sid
    work.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    t0 = time.perf_counter()
    audio = decode_audio(path, SR)
    duration = len(audio) / SR
    version = getattr(transcriber, "version", transcriber.name if hasattr(transcriber, "name") else "unknown")
    notes_path = work / f"{stem}.notes.json"
    notes = _load_notes(notes_path, version) if reuse else None
    if notes is None:
        notes = transcribe_long(audio, SR, transcriber, chunk_s=chunk_s, log=log)
        _save_notes(notes_path, notes, version, duration)
    found = find_clips(notes, songs, cfg)

    clips_dir = work / f"{stem}_clips"
    for k, clip in enumerate(found["clips"], 1):
        clip["clip_id"] = f"{sid}_{stem}_{k:02d}"
        a = max(0.0, clip["start_s"] - cfg.clip_pad_s)
        b = min(duration, clip["end_s"] + cfg.clip_pad_s)
        if b - a > cfg.max_clip_s:
            b = a + cfg.max_clip_s
            clip["review_reasons"].append("clip_cut_at_5_min")
            clip["status"] = "needs_review"
        clip.update(clip_start_s=round(a, 2), clip_end_s=round(b, 2), assess_start_s=round(clip["start_s"] - a, 2),
                    file=None)
        if cut_clips:
            clips_dir.mkdir(exist_ok=True)
            f = clips_dir / f"{clip['clip_id']}.{clip_format}"
            sf.write(f, audio[int(a * SR):int(b * SR)], SR, subtype="PCM_16")
            clip["file"] = str(f.relative_to(out_dir))

    record = {
        "class_file": f"{student}/{path.name}" if keep_names else path.name,
        "student_id": sid,
        "duration_s": round(duration, 2),
        "transcriber": version,
        "notes_file": str(notes_path.relative_to(out_dir)),
        "songs": [s.song_id for s in songs],
        "notes_transcribed": len(notes),
        "playing_regions": found["playing_regions"],
        "clips": found["clips"],
        "config": found["config"],
        "seconds_taken": round(time.perf_counter() - t0, 1),
    }
    (work / f"{stem}.json").write_text(json.dumps(record, indent=1))
    return record
