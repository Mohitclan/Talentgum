"""End-to-end assessment: audio + reference + rubric -> structured result."""

from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ras import __version__
from ras.core.align import AlignmentError, estimate_transposition, match_notes
from ras.core.audio.decode import (
    DecodeError,
    QualityLimits,
    assess_quality,
    crop_from_start,
    decode_audio,
    normalise_level,
)
from ras.core.confidence import estimate_confidence
from ras.core.errors import AnalysisConfig, analyze
from ras.core.feedback import make_feedback
from ras.core.rubric import Rubric, score_analysis
from ras.core.score.reference import Reference, load_reference
from ras.core.transcribe.base import Transcriber
from ras.core.verify import OnsetEvidence, mark_uncertain_missing
from ras.types import PerfNote

ANALYSIS_VERSION = "analysis-1"


def _py(obj):
    """Recursively convert numpy scalars so results are plain JSON."""
    if isinstance(obj, dict):
        return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class AssessmentResult(BaseModel):
    model_config = {"validate_assignment": True}

    assessment_id: str
    status: str  # completed | rejected | failed
    error_code: str | None = None
    error_detail: str | None = None
    versions: dict = Field(default_factory=dict)
    quality: dict = Field(default_factory=dict)
    start_timestamp: dict = Field(default_factory=dict)
    transposition: dict = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)
    scoring: dict = Field(default_factory=dict)
    confidence: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    notes: list[dict] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    timings_s: dict = Field(default_factory=dict)

    @field_validator("*", mode="before")
    @classmethod
    def _plain(cls, v):
        return _py(v)


@dataclass
class PipelineConfig:
    sr: int = 44100
    normalize_level: bool = True
    start_margin_s: float = 1.0
    start_check_window_s: float = 2.0
    unfold_repeats: bool = True
    # Double-check 'missing' notes against the audio. OFF by default: on synthetic audio the evidence
    # did not separate dropped notes from truly absent ones; evaluate on real recordings first.
    verify_missing: bool = False
    verify_threshold_db: float = 2.5  # onset jump in excess of the neighbouring semitones
    quality: QualityLimits = field(default_factory=QualityLimits)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def assess(
    reference: str | Path | Reference,
    audio_path: str | Path,
    rubric: Rubric,
    transcriber: Transcriber,
    start_s: float | None = None,
    config: PipelineConfig | None = None,
) -> AssessmentResult:
    cfg = config or PipelineConfig()
    t: dict[str, float] = {}
    res = AssessmentResult(assessment_id=f"asm_{uuid.uuid4().hex[:12]}", status="completed")
    res.versions = {"ras": __version__, "analysis": ANALYSIS_VERSION, "commit": _git_commit(),
                    "rubric": f"{rubric.rubric_id}@{rubric.version}"}

    def mark(name: str, t0: float) -> None:
        t[name] = round(time.perf_counter() - t0, 3)

    # 1. decode and quality gate
    t0 = time.perf_counter()
    try:
        audio = decode_audio(audio_path, cfg.sr)
    except DecodeError as exc:
        res.status, res.error_code, res.error_detail = "failed", "decode_failed", str(exc)
        res.timings_s = t
        return res
    quality = assess_quality(audio, cfg.sr, cfg.quality)
    res.quality = _py(quality.to_dict())
    res.warnings += quality.warnings
    mark("decode_and_quality", t0)
    if not quality.ok:
        res.status, res.error_code = "rejected", "recording_quality_low"
        res.error_detail = quality.reject_reason
        res.timings_s = t
        return res

    # 2. crop at the start timestamp, level, transcribe
    t0 = time.perf_counter()
    audio, offset = crop_from_start(audio, cfg.sr, start_s, cfg.start_margin_s)
    if cfg.normalize_level:
        audio = normalise_level(audio)
    mark("preprocess", t0)

    t0 = time.perf_counter()
    tr = transcriber.transcribe(audio, cfg.sr)
    mark("transcribe", t0)
    res.versions["transcriber"] = tr.version
    notes = [PerfNote(n.pitch, n.onset + offset, n.offset + offset, n.velocity) for n in tr.notes]
    if not notes:
        res.warnings.append("no_notes_detected")

    start_suspect = False
    first_onset = min((n.onset for n in notes), default=None)
    if start_s is not None and first_onset is not None:
        start_suspect = abs(first_onset - start_s) > cfg.start_check_window_s
    res.start_timestamp = {"given_s": start_s, "first_onset_s": first_onset, "suspect": start_suspect}
    if start_suspect:
        res.warnings.append("start_timestamp_suspect")

    # 3. reference
    t0 = time.perf_counter()
    ref = (reference if isinstance(reference, Reference)
           else load_reference(reference, unfold_repeats=cfg.unfold_repeats))
    res.versions["reference_sha256"] = ref.sha256[:16]
    res.warnings += ref.warnings
    mark("reference", t0)

    # 4. transposition, alignment, error detection
    t0 = time.perf_counter()
    shift, evidence = estimate_transposition(ref, notes)
    allowed = rubric.options.allow_transposition
    applied = shift if allowed else 0
    res.transposition = {"detected_semitones": shift, "evidence": round(evidence, 3),
                         "applied": applied, "allowed": allowed}
    if shift:
        res.warnings.append(f"performance_transposed_{shift:+d}_semitones")
    perf = [PerfNote(n.pitch - applied, n.onset, n.offset, n.velocity) for n in notes]

    try:
        alignment = match_notes(ref, perf)
    except AlignmentError as exc:
        res.error_code, res.error_detail = "alignment_failed", str(exc)
        res.confidence = {"value": 0.0, "needs_review": True, "reasons": ["alignment_failed"]}
        res.timings_s = t
        return res
    analysis = analyze(ref, perf, alignment, AnalysisConfig(
        pause_threshold_s=rubric.options.pause_threshold_s))
    res.warnings += analysis.notes
    mark("align_and_analyse", t0)

    t0 = time.perf_counter()
    if cfg.verify_missing and analysis.n_missing:
        n_unc = mark_uncertain_missing(
            analysis, OnsetEvidence(audio, cfg.sr, offset), cfg.verify_threshold_db, performed=perf)
        if n_unc:
            res.warnings.append(f"{n_unc}_missing_notes_have_audio_evidence_marked_uncertain")
    mark("verify_missing", t0)

    # 5. scoring, confidence, feedback
    t0 = time.perf_counter()
    scoring = score_analysis(rubric, analysis)
    conf = estimate_confidence(
        analysis, quality.quality_score, rubric.options.review_confidence_threshold,
        start_suspect=start_suspect, transposed_not_allowed=bool(shift and not allowed))
    res.summary = _py(analysis.summary())
    res.scoring = _py(scoring.to_dict())
    res.confidence = {"value": conf.value, "needs_review": conf.needs_review, "reasons": conf.reasons}
    res.notes = [_py(asdict(n)) for n in analysis.results]
    res.feedback = make_feedback(analysis, ref, rubric.options.timing_tolerance_ms)
    unassessed = [k for k, v in scoring.criteria.items() if v["status"] == "not_assessed"]
    if unassessed:
        res.warnings.append("not_assessed: " + ", ".join(unassessed))
    mark("score_and_feedback", t0)
    t["total"] = round(sum(t.values()), 3)
    res.timings_s = t
    return res
