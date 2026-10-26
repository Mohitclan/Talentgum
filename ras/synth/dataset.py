"""Generate a labelled synthetic dataset: references, audio (AAC m4a like the client's), ground truth."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ras.core.score.reference import load_reference
from ras.synth.musicxml import write_musicxml
from ras.synth.performance import PerfSpec, Truth, simulate_performance
from ras.synth.render import degrade, render_notes, write_audio
from ras.synth.songs import SONGS
from ras.types import PerfNote

CONDITIONS = {
    "clean": dict(rt60=0.0, snr_db=None, lowpass_hz=None, gain_db=0.0),
    "room": dict(rt60=0.45, snr_db=35, lowpass_hz=None, gain_db=0.0),
    "phone": dict(rt60=0.3, snr_db=25, lowpass_hz=7500, gain_db=-6.0),
    "noisy_quiet": dict(rt60=0.35, snr_db=18, lowpass_hz=6500, gain_db=-14.0),
}


def truth_to_json(t: Truth) -> dict:
    return {
        "ref_status": t.ref_status, "played_pitch": t.played_pitch,
        "jitter_ms": [float(x) for x in t.jitter_ms],
        "extras": [asdict(n) for n in t.extras], "notes": [asdict(n) for n in t.notes],
        "transpose": t.transpose,
    }


def truth_from_json(d: dict) -> Truth:
    return Truth(d["ref_status"], d["played_pitch"], np.array(d["jitter_ms"]),
                 [PerfNote(**n) for n in d["extras"]], [PerfNote(**n) for n in d["notes"]],
                 d.get("transpose", 0))


def generate_dataset(out_dir: str | Path, n: int = 20, seed: int = 0, n_students: int = 8,
                     songs: list[str] | None = None, fmt: str = "m4a") -> Path:
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    song_ids = songs or list(SONGS)
    refs = {}
    for sid in song_ids:
        refs[sid] = write_musicxml(SONGS[sid], out / "refs" / f"{sid}.musicxml")
    loaded = {sid: load_reference(p) for sid, p in refs.items()}
    cond_names = list(CONDITIONS)
    manifest = []
    for i in range(n):
        sid = song_ids[i % len(song_ids)]
        student = f"student_{int(rng.integers(0, n_students)):02d}"
        cond = cond_names[(i // len(song_ids)) % len(cond_names)]
        skill = rng.choice(["strong", "average", "weak"], p=[0.3, 0.4, 0.3])
        # mistakes as a share of the score's notes: 0-4% strong, 4-12% average, 12-25% weak
        lo, hi = {"strong": (0.0, 0.04), "average": (0.04, 0.12), "weak": (0.12, 0.25)}[str(skill)]
        n_err = int(round(rng.uniform(lo, hi) * loaded[sid].n_notes))
        n_wrong = int(rng.integers(0, n_err + 1))
        spec = PerfSpec(
            tempo_factor=float(rng.uniform(0.6, 1.1)), drift=float(rng.uniform(0.0, 0.1)),
            jitter_ms=float(rng.uniform(8, 35 if skill != "strong" else 20)),
            n_wrong_pitch=n_wrong, n_wrong_octave=int(rng.integers(0, 2)) if n_err else 0,
            n_missing=n_err - n_wrong, n_extra=int(rng.integers(0, 4)) if skill != "strong" else 0,
            start_offset_s=float(rng.uniform(0.8, 2.5)), seed=int(rng.integers(0, 2**31)),
            velocity_mean=float(rng.uniform(55, 90)),
        )
        if skill == "weak" and rng.random() < 0.4:
            ref = loaded[sid]
            spec.pauses = [(int(rng.integers(5, max(6, ref.n_notes - 5))), float(rng.uniform(2, 4)))]
        truth = simulate_performance(loaded[sid], spec)
        y = render_notes(truth.notes, seed=int(rng.integers(0, 2**31)))
        y = degrade(y, seed=int(rng.integers(0, 2**31)), **CONDITIONS[cond])
        rec = f"rec_{i:03d}"
        audio_path = write_audio(y, out / "audio" / f"{rec}.{fmt}")
        truth_path = out / "truth" / f"{rec}.json"
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(json.dumps(truth_to_json(truth)))
        first = min(n_.onset for n_ in truth.notes)
        manifest.append({
            "id": rec, "song_id": sid, "student_id": student, "condition": cond, "skill": str(skill),
            "audio": str(audio_path.relative_to(out)), "truth": str(truth_path.relative_to(out)),
            "reference": str(refs[sid].relative_to(out)),
            "start_s": round(first - 0.1, 2),  # like a teacher-marked start slightly before the first note
            "duration_s": round(len(y) / 44100, 2),
            "injected": truth.counts(), "tempo_factor": round(spec.tempo_factor, 3),
        })
    path = out / "manifest.json"
    path.write_text(json.dumps({"seed": seed, "recordings": manifest}, indent=1))
    return path
