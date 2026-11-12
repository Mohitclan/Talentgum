"""Fill a local RAS database with demo assessments for the frontend, without any AI model.

Synthetic songs and performances with known mistakes go through the real service (upload, queue, worker,
pipeline, rubric). The transcriber is replaced by the known notes, so no model download is needed and the
results are exactly what the injected mistakes should produce. Student ids are made up.

    .venv/bin/python scripts/seed_demo.py --data-dir data/demo
    .venv/bin/ras serve --data-dir data/demo --no-with-worker      # then open http://127.0.0.1:8000/app/

Uploading a new recording in this demo database needs a worker with a real transcriber (`ras serve`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from ras.core.score.reference import load_reference
from ras.core.transcribe.ground_truth import GroundTruthTranscriber
from ras.service.api import create_app
from ras.service.config import Settings
from ras.service.worker import Worker
from ras.synth.musicxml import write_musicxml
from ras.synth.performance import PerfSpec, simulate_performance
from ras.synth.render import render_notes, write_audio
from ras.synth.songs import SONGS

ROOT = Path(__file__).resolve().parents[1]
RUBRICS = [ROOT / "data" / "rubrics" / f"talentgum_{n}_v1.json" for n in ("criteria", "beginner", "advanced")]
LEVEL = {"aarav": "beginner", "rohan": "beginner", "meera": "advanced", "ishaan": "advanced"}

# (student, song, performance, teacher review or None)
TAKES = [
    ("aarav", "twinkle", dict(n_wrong_pitch=1, jitter_ms=20),
     dict(comment="Lovely steady playing, Aarav! Watch bar 3 next time.")),
    ("aarav", "mary", dict(n_wrong_pitch=2, n_missing=1, tempo_factor=0.85),
     dict(comment="Great effort. Slow and careful is the right way.")),
    ("aarav", "ode_to_joy", dict(n_missing=2, n_extra=2, jitter_ms=35), None),
    ("meera", "twinkle", dict(jitter_ms=12), dict(comment="Beautiful, almost perfect!", overall_score=94)),
    ("meera", "ode_to_joy", dict(n_wrong_pitch=3, pauses=[(8, 1.6)], jitter_ms=40), None),
    ("rohan", "mary", dict(n_wrong_octave=2, n_wrong_pitch=2, n_missing=3, tempo_factor=1.2, jitter_ms=60), None),
    ("rohan", "twinkle", dict(restarts=[(10, 0, 1.2)], n_wrong_pitch=1),
     dict(comment="Good that you started again calmly. Keep going!")),
    ("ishaan", "ode_to_joy", dict(n_wrong_pitch=1, jitter_ms=18), None),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "demo")
    ap.add_argument("--keep", action="store_true", help="add to an existing demo database instead of starting fresh")
    args = ap.parse_args()
    if args.data_dir.exists() and not args.keep:
        shutil.rmtree(args.data_dir)

    settings = Settings(data_dir=args.data_dir, transcriber="ground-truth", transcription_cache=False)
    app = create_app(settings)
    client = TestClient(app, raise_server_exceptions=True)
    tr = GroundTruthTranscriber()
    worker = Worker(settings, transcriber_factory=lambda: tr, session_factory=app.state.Session, store=app.state.store)

    def upload(path: Path, kind: str) -> str:
        r = client.post(f"/v1/files?kind={kind}", files={"file": (path.name, path.read_bytes())})
        r.raise_for_status()
        return r.json()["file_id"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        refs = {}
        for sid in sorted({t[1] for t in TAKES}):
            song = SONGS[sid]
            path = write_musicxml(song, tmp / f"{sid}.musicxml")
            refs[sid] = load_reference(path)
            client.post("/v1/songs", json={"song_id": sid, "title": song.title,
                                           "reference_file_id": upload(path, "reference")}).raise_for_status()
        for rubric in RUBRICS:
            client.post("/v1/rubrics", json=json.loads(rubric.read_text())).raise_for_status()

        for i, (student, sid, perf, review) in enumerate(TAKES):
            truth = simulate_performance(refs[sid], PerfSpec(seed=100 + i, **perf))
            audio = write_audio(render_notes(truth.notes, seed=i), tmp / f"{student}_{sid}_{i}.m4a")
            r = client.post("/v1/assessments", json={"song_id": sid, "recording_file_id": upload(audio, "audio"),
                                                     "rubric_id": f"talentgum-{LEVEL[student]}", "student_id": student})
            r.raise_for_status()
            asm = r.json()["assessment_id"]
            tr.set_notes(truth.notes)
            worker.run_once()
            res = client.get(f"/v1/assessments/{asm}").json()
            if review and res["status"] in ("completed", "review_required"):
                client.post(f"/v1/assessments/{asm}/review", json={"reviewer": "Ms. Rao", **review}).raise_for_status()
                res["status"] = "reviewed"
            print(f"{student:7} {LEVEL[student]:9} {sid:11} {res['status']:16} score={res['total_score']}")
    print(f"Demo data in {args.data_dir}")


if __name__ == "__main__":
    main()
