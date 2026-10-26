"""Benchmark transcribers on a dataset: accuracy, error-detection quality, CPU time, peak RAM."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ras.eval.metrics import ERROR_TYPES, bootstrap_ci, pooled_prf

NOT_PACKAGED = {
    "hft-transformer": "research code (shell scripts, corpus generation); needs a wrapper before it "
                       "can be benchmarked. Slot reserved behind the Transcriber interface.",
}


def _mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def aggregate(res: dict) -> dict:
    recs = res["records"]
    agg: dict = {"model": res["model"], "n_recordings": len(recs), "failures": len(res["failures"]),
                 "peak_rss_gb": res.get("peak_rss_gb")}
    tr = [r["transcription"] for r in recs if "transcription" in r]
    for k in ("onset_f1", "onset_offset_f1", "onset_velocity_f1", "onset_precision", "onset_recall"):
        vals = [t[k] for t in tr]
        agg[k] = _mean(vals)
        if k.endswith("f1"):
            agg[k + "_ci"] = bootstrap_ci(vals, _mean)
    dur = sum(r["duration_s"] for r in recs)
    agg["transcribe_cpu_s_per_audio_min"] = (
        60 * sum(r["transcribe_s"] for r in recs) / dur if dur else float("nan"))
    agg["end_to_end_s_per_audio_min"] = (
        60 * sum(r["total_s"] or 0 for r in recs) / dur if dur else float("nan"))
    det = [r["detection"] for r in recs if "detection" in r]
    if det:
        p, r_, f = pooled_prf(det)
        agg["error_detection"] = {"precision": p, "recall": r_, "f1": f,
                                  "precision_ci": bootstrap_ci(det, lambda d: pooled_prf(d)[0]),
                                  "recall_ci": bootstrap_ci(det, lambda d: pooled_prf(d)[1]),
                                  "n_recordings": len(det)}
        agg["by_type"] = {t: dict(zip(("precision", "recall", "f1"), pooled_prf(det, t)))
                          for t in ERROR_TYPES}
    errs = [e for r in recs for e in r.get("timing_err_ms", [])]
    agg["timing_mae_ms"] = float(np.mean(errs)) if errs else float("nan")
    flags = [r["needs_review"] for r in recs if "needs_review" in r]
    agg["review_rate"] = float(np.mean(flags)) if flags else float("nan")
    by_cond: dict[str, list] = {}
    for r in recs:
        if "transcription" in r:
            by_cond.setdefault(r["condition"], []).append(r["transcription"]["onset_f1"])
    agg["onset_f1_by_condition"] = {k: _mean(v) for k, v in sorted(by_cond.items())}
    return agg


def _fmt(x, d=3):
    return "n/a" if x is None or x != x else f"{x:.{d}f}"


def _ci(ci):
    return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def report_markdown(aggs: list[dict], unavailable: list[dict], data_dir: Path, n: int) -> str:
    lines = ["# RAS benchmark\n", f"Dataset: `{data_dir}` ({n} synthetic recordings). "
             "Synthetic audio only: these numbers test the pipeline and say nothing about accuracy "
             "on TalentGum keyboards or phones.\n"]
    lines += ["## Transcription (note-level, mir_eval)\n",
              "| model | onset F1 (95% CI) | onset+offset F1 | onset+velocity F1 | CPU s per audio min "
              "| peak RAM GB | failures |", "|---|---|---|---|---|---|---|"]
    for a in aggs:
        lines.append(f"| {a['model']} | {_fmt(a['onset_f1'])} {_ci(a['onset_f1_ci'])} | "
                     f"{_fmt(a['onset_offset_f1'])} | {_fmt(a['onset_velocity_f1'])} | "
                     f"{_fmt(a['transcribe_cpu_s_per_audio_min'], 1)} | {_fmt(a['peak_rss_gb'], 2)} "
                     f"| {a['failures']} |")
    lines += ["\n## Error detection (pooled over recordings; precision matters most)\n",
              "| model | precision (CI) | recall (CI) | F1 | timing MAE ms | review rate |",
              "|---|---|---|---|---|---|"]
    for a in aggs:
        d = a.get("error_detection")
        if d:
            lines.append(f"| {a['model']} | {_fmt(d['precision'])} {_ci(d['precision_ci'])} | "
                         f"{_fmt(d['recall'])} {_ci(d['recall_ci'])} | {_fmt(d['f1'])} | "
                         f"{_fmt(a['timing_mae_ms'], 1)} | {_fmt(a['review_rate'], 2)} |")
    lines += ["\n## Per error type (precision / recall)\n",
              "| model | " + " | ".join(ERROR_TYPES) + " |", "|---|" + "---|" * len(ERROR_TYPES)]
    for a in aggs:
        if "by_type" in a:
            cells = [f"{_fmt(a['by_type'][t]['precision'], 2)} / {_fmt(a['by_type'][t]['recall'], 2)}"
                     for t in ERROR_TYPES]
            lines.append(f"| {a['model']} | " + " | ".join(cells) + " |")
    lines += ["\n## Onset F1 by recording condition (domain-shift check)\n"]
    conds = sorted({c for a in aggs for c in a["onset_f1_by_condition"]})
    lines += ["| model | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for a in aggs:
        lines.append(f"| {a['model']} | " + " | ".join(
            _fmt(a["onset_f1_by_condition"].get(c)) for c in conds) + " |")
    if unavailable:
        lines += ["\n## Not benchmarked\n"]
        lines += [f"- **{u['model']}**: {u['reason']}" for u in unavailable]
    lines.append("\n`ground-truth` feeds the known notes to the pipeline: it is the upper bound for "
                 "alignment and error detection when transcription is perfect.")
    return "\n".join(lines) + "\n"


def run_benchmark(data_dir: str | Path, models: list[str], out_dir: str | Path | None = None,
                  limit: int | None = None, log=print, cache_dir: str | Path | None = None) -> Path:
    data_dir = Path(data_dir)
    out = Path(out_dir) if out_dir else Path("data/runs") / time.strftime("bench_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    n = len(json.loads((data_dir / "manifest.json").read_text())["recordings"][:limit])
    aggs, unavailable = [], []
    for m in models:
        if m in NOT_PACKAGED:
            unavailable.append({"model": m, "reason": NOT_PACKAGED[m]})
            continue
        log(f"[bench] {m} on {n} recordings ...")
        raw = out / f"{m}.raw.json"
        cmd = [sys.executable, "-m", "ras.eval.worker", "--model", m, "--data", str(data_dir), "--out", str(raw)]
        cmd += ["--limit", str(limit)] if limit else []
        cmd += ["--cache", str(cache_dir)] if cache_dir else []
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not raw.exists():
            unavailable.append({"model": m, "reason": "worker crashed: " + proc.stderr.strip()[-300:]})
            continue
        res = json.loads(raw.read_text())
        if not res.get("available", True):
            unavailable.append({"model": m, "reason": res["reason"]})
            continue
        aggs.append(aggregate(res))
    for m in NOT_PACKAGED:
        if m not in models:
            unavailable.append({"model": m, "reason": NOT_PACKAGED[m]})
    (out / "summary.json").write_text(json.dumps(aggs, indent=1))
    (out / "report.md").write_text(report_markdown(aggs, unavailable, data_dir, n))
    return out
