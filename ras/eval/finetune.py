"""Prepare a Transkun fine-tuning set from TalentGum recordings that have no MIDI.

The client's students play non-MIDI keyboards, so there is no paired audio + MIDI. Labels are made
from recordings whose content is known instead:
- teacher benchmark recordings, which are taken as correct performances of the score;
- reviewed student recordings, which carry the teacher's note corrections.

For each one, the score is aligned to the audio (the normal pipeline). Every note gets the timing the
model heard where it matched, and the timing map's estimate where it did not. Notes the review says
were not played are dropped. The result is MAESTRO-style audio/MIDI pairs, a metadata CSV, and an
initial checkpoint holding the released Transkun V2 weights. Transkun's own trainer
(`python -m transkun.train`) then continues from them on a GPU. `ras finetune prepare` prints the
exact command.

Fine-tune only when evaluation shows a systematic error (one keyboard model, phone microphones):
docs/DEPLOY.md, "Fine-tuning".
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

SR = 44100


def _label_notes(res: dict, ref, removed: set[int]) -> list[tuple[int, float, float, int]]:
    """(pitch, onset, offset, velocity) for every score note the performer is taken to have played."""
    out = []
    shift = int(res.get("transposition", {}).get("applied", 0))
    spb = 60.0 / (res.get("summary", {}).get("mean_bpm") or ref.bpm)  # the performer's tempo
    for n in res["notes"]:
        r = n.get("ref_index")
        if r is None or r in removed or n["status"] == "uncertain":
            continue
        on = n.get("perf_onset_s") if n["status"] == "correct" else n.get("expected_onset_s")
        if on is None:
            continue
        dur = 0.9 * float(ref.notes["duration_beat"][r]) * spb  # written length at the performed tempo
        pitch = int(ref.notes["pitch"][r]) + shift
        out.append((pitch, float(on), float(on + max(dur, 0.05)), 64))
    return out


def prepare(items: list[dict], out_dir: Path, transcriber, rubric) -> dict:
    """`items`: [{"reference", "audio", "start_s"?, "split": "train"|"validation", "removed_ref_indices"?}]."""
    import pretty_midi
    import soundfile as sf

    from ras.core.audio.decode import decode_audio
    from ras.core.score.reference import load_reference
    from ras.pipeline import assess

    out_dir = Path(out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "midi").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for k, it in enumerate(items):
        ref = load_reference(it["reference"])
        res = assess(ref, it["audio"], rubric, transcriber, start_s=it.get("start_s")).model_dump()
        if res["status"] != "completed" or not res["notes"]:
            skipped.append({"audio": str(it["audio"]), "reason": res.get("error_code") or res["status"]})
            continue
        notes = _label_notes(res, ref, set(it.get("removed_ref_indices", [])))
        name = f"rec_{k:04d}"
        audio = decode_audio(it["audio"], SR)
        sf.write(out_dir / "audio" / f"{name}.wav", audio, SR, subtype="PCM_16")
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        for p, on, off, vel in notes:
            inst.notes.append(pretty_midi.Note(vel, p, on, off))
        pm.instruments.append(inst)
        pm.write(str(out_dir / "midi" / f"{name}.mid"))
        rows.append({"canonical_composer": "talentgum", "canonical_title": Path(str(it["reference"])).stem,
                     "split": it.get("split", "train"), "year": 2026, "midi_filename": f"midi/{name}.mid",
                     "audio_filename": f"audio/{name}.wav", "duration": round(len(audio) / SR, 2)})
    with (out_dir / "metadata.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["split"])
        w.writeheader()
        w.writerows(rows)
    summary = {"recordings": len(rows), "train": sum(r["split"] == "train" for r in rows),
               "validation": sum(r["split"] == "validation" for r in rows), "skipped": skipped}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def build_pickles(out_dir: Path) -> None:
    """Transkun's dataset pickles (createDatasetMaestroCSV) from the metadata CSV."""
    import pickle

    from transkun import Data

    dataset = Data.createDatasetMaestroCSV(str(out_dir), str(out_dir / "metadata.csv"), extendSustainPedal=True)
    for split, fname in (("train", "train.pickle"), ("validation", "val.pickle")):
        with open(out_dir / fname, "wb") as f:
            pickle.dump([e for e in dataset if e["split"] == split], f, pickle.HIGHEST_PROTOCOL)


def init_checkpoint(out_path: Path, n_iter: int = 20000, max_lr: float = 2e-5) -> Path:
    """A Transkun training checkpoint whose weights are the released V2 model, so training continues from it."""
    import moduleconf
    import torch
    import transkun
    from transkun.TrainUtil import initializeCheckpoint, save_checkpoint

    pre = Path(transkun.__file__).parent / "pretrained"
    conf_mgr = moduleconf.parseFromFile(str(pre / "2.0.conf"))
    model_cls, conf = conf_mgr["Model"].module.TransKun, conf_mgr["Model"].config
    epoch, it, model, tracker, _, opt, sched = initializeCheckpoint(model_cls, device="cpu", max_lr=max_lr,
                                                                   weight_decay=1e-4, nIter=n_iter, conf=conf)
    state = torch.load(pre / "2.0.pt", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=False)
    save_checkpoint(str(out_path), epoch, it, model, tracker, model.state_dict(), opt, sched)
    return out_path


def train_command(out_dir: Path, ckpt: Path, n_iter: int, max_lr: float) -> str:
    import transkun

    conf = Path(transkun.__file__).parent / "pretrained" / "2.0.conf"
    return (f"python -m transkun.train {ckpt} --nProcess 1 --datasetPath {out_dir} "
            f"--datasetMetaFile_train {out_dir / 'train.pickle'} --datasetMetaFile_val {out_dir / 'val.pickle'} "
            f"--modelConf {conf} --max_lr {max_lr} --nIter {n_iter} --batchSize 4 --augment")


def export_weights(ckpt: Path, out_path: Path) -> Path:
    """The fine-tuned model in the released format, for RAS_TRANSKUN_WEIGHT."""
    import torch

    c = torch.load(ckpt, map_location="cpu", weights_only=False)  # our own training output, not a download
    torch.save({"state_dict": c.get("best_state_dict") or c["state_dict"]}, out_path)
    return out_path
