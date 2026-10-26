"""Run one model over a dataset in its own process (so peak RAM is attributable to that model).

    python -m ras.eval.worker --model transkun --data data/synth --out result.json
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

from ras.core.rubric import load_rubric
from ras.core.score.reference import load_reference
from ras.core.transcribe import TranscriberUnavailable, get_transcriber
from ras.core.transcribe.ground_truth import GroundTruthTranscriber
from ras.eval.metrics import detection_counts, timing_errors, transcription_metrics
from ras.pipeline import PipelineConfig, assess
from ras.synth.dataset import truth_from_json
from ras.types import PerfNote, Transcription

DEFAULT_RUBRIC = Path(__file__).resolve().parents[2] / "data" / "rubrics" / "dev_placeholder.json"


class _Recording:
    """Wraps a transcriber to keep its raw output and wall time for evaluation."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.sample_rate = getattr(inner, "sample_rate", 44100)
        self.last: Transcription | None = None
        self.wall_s = 0.0

    def transcribe(self, audio, sr):
        t0 = time.perf_counter()
        self.last = self.inner.transcribe(audio, sr)
        self.wall_s = time.perf_counter() - t0
        return self.last


def _peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**3 if platform.system() == "Darwin" else 1024**2)


def run(model: str, data_dir: Path, rubric_path: Path, limit: int | None) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text())["recordings"][:limit]
    try:
        inner = get_transcriber(model)
    except (TranscriberUnavailable, ImportError) as exc:
        return {"model": model, "available": False, "reason": str(exc)}
    rubric = load_rubric(rubric_path)
    cfg = PipelineConfig()
    out = {"model": model, "available": True, "records": [], "failures": []}
    for rec in manifest:
        ref = load_reference(data_dir / rec["reference"])
        truth = truth_from_json(json.loads((data_dir / rec["truth"]).read_text()))
        offset = max(0.0, rec["start_s"] - cfg.start_margin_s)
        if isinstance(inner, GroundTruthTranscriber):
            inner.set_notes([PerfNote(n.pitch, n.onset - offset, n.offset - offset, n.velocity)
                             for n in truth.notes])
        wrapped = _Recording(inner)
        try:
            res = assess(ref, data_dir / rec["audio"], rubric, wrapped, start_s=rec["start_s"], config=cfg)
        except Exception as exc:  # keep going: one bad file must not hide the rest
            out["failures"].append({"id": rec["id"], "error": repr(exc)})
            continue
        row = {k: rec[k] for k in ("id", "song_id", "student_id", "condition", "skill", "duration_s")}
        row.update(status=res.status, transcribe_s=wrapped.wall_s, total_s=res.timings_s.get("total"))
        if wrapped.last is not None:
            est = [PerfNote(n.pitch, n.onset + offset, n.offset + offset, n.velocity)
                   for n in wrapped.last.notes]
            row["transcription"] = transcription_metrics(truth.notes, est)
        if res.status == "completed" and res.notes:
            row["detection"] = detection_counts(res.notes, truth, ref)
            row["timing_err_ms"] = timing_errors(res.notes, truth)
            row["total_score"] = res.scoring.get("total_score")
            row["confidence"] = res.confidence.get("value")
            row["needs_review"] = res.confidence.get("needs_review")
            row["summary"] = res.summary
        out["records"].append(row)
    out["peak_rss_gb"] = round(_peak_rss_gb(), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    result = run(args.model, args.data, args.rubric, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
