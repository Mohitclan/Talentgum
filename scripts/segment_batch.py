"""Run `ras segment` over many class recordings on one GPU machine, several at a time, resumable.

    python scripts/segment_batch.py /data/classes --songs data/private/songs_segment --out /data/out --workers 3

- Input: folders with one sub-folder per student (or loose files); any audio/video ffmpeg reads (.mp4, .m4a ...).
- Each worker process loads Transkun once and takes the next recording. Several workers share one GPU.
- A recording whose annotation JSON already exists is skipped, so the job can be stopped and restarted.
- At the end: <out>/clips.csv (one row per found attempt, for review) and <out>/failures.csv.

Docs: docs/GPU_ANNOTATION.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

from ras.segment import annotate_class, load_songs, student_id
from ras.segment.classfile import AUDIO_EXTS

_state: dict = {}


def _init(songs_dir: str, out: str, chunk_s: float, keep_names: bool, clips: bool) -> None:
    from ras.core.transcribe import get_transcriber

    _state.update(songs=load_songs(Path(songs_dir)), tr=get_transcriber("transkun"), out=Path(out),
                  chunk_s=chunk_s, keep_names=keep_names, clips=clips)


def _run(job: tuple[str, str]) -> dict:
    path, student = job
    t0 = time.time()
    try:
        rec = annotate_class(path, _state["songs"], _state["tr"], _state["out"], student=student,
                             keep_names=_state["keep_names"], cut_clips=_state["clips"],
                             chunk_s=_state["chunk_s"], reuse=True)
        return {"path": path, "ok": True, "clips": len(rec["clips"]),
                "review": sum(c["status"] == "needs_review" for c in rec["clips"]),
                "minutes": rec["duration_s"] / 60, "seconds": time.time() - t0}
    except Exception as exc:  # one bad file must not stop the batch
        return {"path": path, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=3), "seconds": time.time() - t0}


def find_jobs(inputs: list[Path]) -> list[tuple[str, str]]:
    jobs = []
    for c in inputs:
        if c.is_dir():
            for f in sorted(p for p in c.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS):
                rel = f.relative_to(c)
                jobs.append((str(f), rel.parts[0] if len(rel.parts) > 1 else c.name))
        elif c.is_file():
            jobs.append((str(c), c.parent.name))
    return jobs


def write_clips_csv(out: Path) -> int:
    rows = []
    for ann in sorted(out.glob("*/*.json")):
        if ann.name.endswith(".notes.json"):
            continue
        rec = json.loads(ann.read_text())
        if "clips" not in rec:
            continue
        for c in rec["clips"]:
            rows.append({"class_file": rec["class_file"], "student_id": rec["student_id"],
                         "clip_id": c.get("clip_id"), "song_id": c["song_id"], "start_s": c["start_s"],
                         "end_s": c["end_s"], "match_score": c["match_score"], "coverage": c["coverage"],
                         "coverage_pct": c["coverage_pct"], "status": c["status"],
                         "review_reasons": ";".join(c["review_reasons"]), "file": c.get("file"),
                         "annotation": str(ann.relative_to(out))})
    with open(out / "clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["class_file"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="class recordings, or folders of student folders")
    ap.add_argument("--songs", type=Path, required=True, help="folder of MXL/MIDI files; file name = song id")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2, help="recordings processed at once on the GPU")
    ap.add_argument("--chunk-s", type=float, default=600.0, help="transcribe in pieces of this many seconds")
    ap.add_argument("--keep-names", action="store_true", help="use student folder names instead of hashed ids")
    ap.add_argument("--no-clips", action="store_true", help="write annotations only, no audio clips")
    ap.add_argument("--limit", type=int, default=0, help="only the first N recordings (for a timing test)")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    jobs = find_jobs(a.inputs)
    todo = [(p, s) for p, s in jobs
            if not (a.out / student_id(s, a.keep_names) / f"{Path(p).stem}.json").is_file()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(jobs)} recordings found, {len(jobs) - len(todo)} already done, {len(todo)} to run "
          f"with {a.workers} worker(s) on {os.environ.get('RAS_DEVICE', 'auto')}", flush=True)
    if not todo:
        print(f"{write_clips_csv(a.out)} clips listed in {a.out / 'clips.csv'}")
        return

    t0, done, audio_min, failures = time.time(), 0, 0.0, []
    ctx = mp.get_context("spawn")  # CUDA cannot be shared with forked processes
    with ctx.Pool(a.workers, initializer=_init,
                  initargs=(str(a.songs), str(a.out), a.chunk_s, a.keep_names, not a.no_clips)) as pool:
        for r in pool.imap_unordered(_run, todo):
            done += 1
            el = time.time() - t0
            eta = el / done * (len(todo) - done)
            if r["ok"]:
                audio_min += r["minutes"]
                print(f"[{done}/{len(todo)}] {Path(r['path']).name}: {r['minutes']:.0f} min audio, {r['clips']} clips "
                      f"({r['review']} to check) in {r['seconds']:.0f} s | ETA {eta / 3600:.1f} h", flush=True)
            else:
                failures.append(r)
                print(f"[{done}/{len(todo)}] FAILED {r['path']}: {r['error']}", flush=True)

    el = time.time() - t0
    print(f"Done: {done - len(failures)} ok, {len(failures)} failed, {audio_min:.0f} min of audio in "
          f"{el / 60:.0f} min ({el / max(audio_min, 1e-9):.1f} s per audio minute overall)")
    if failures:
        with open(a.out / "failures.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "error", "trace"], extrasaction="ignore")
            w.writeheader()
            w.writerows(failures)
        print(f"Failures listed in {a.out / 'failures.csv'}; re-run the same command to retry them.")
    print(f"{write_clips_csv(a.out)} clips listed in {a.out / 'clips.csv'}")
    sys.exit(1 if failures and len(failures) == done else 0)


if __name__ == "__main__":
    main()
