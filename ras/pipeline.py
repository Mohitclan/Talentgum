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
from ras.core import second_opinion
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
from ras.core.expression import measure as measure_expression
from ras.core.feedback import make_feedback
from ras.core.profile import compare as compare_profile
from ras.core.rubric import Rubric, score_analysis
from ras.core.rubric.engine import models_needed
from ras.core.score.reference import Reference, load_reference
from ras.core.transcribe.base import Transcriber
from ras.core.verify import OnsetEvidence, mark_uncertain_missing
from ras.models.hub import ModelHub, default_hub
from ras.types import PedalEvent, PerfNote

ANALYSIS_VERSION = "analysis-2"  # 2: expression measures, second opinions, teacher profiles


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
    scoring_inputs: dict = Field(default_factory=dict)  # lets a new rubric re-score this result later
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
    start_cut_risk_s: float = 0.3  # a note this close to the crop point means the crop cut into the playing
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


def _sounding_at_start(audio, sr: int, window_s: float = 0.25, ratio: float = 0.25) -> bool:
    """Is the music already loud at the very start of this (cropped) audio?

    With a correct start timestamp the crop keeps about a second of near-silence before the first note.
    """
    import numpy as np

    frame = int(0.05 * sr)
    n = len(audio) // frame
    if n < 10:
        return False
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    head = rms[: max(1, int(window_s / 0.05))].mean()
    return bool(head > ratio * np.percentile(rms, 90))


def _add_pedal(analysis, pedal: list[PedalEvent] | None, notes: list[PerfNote]) -> None:
    """Sustain-pedal use over the performance (`pedal` None: the source cannot detect the pedal)."""
    analysis.pedal_supported = pedal is not None
    if pedal is None or not notes:
        return
    lo, hi = min(n.onset for n in notes), max(n.offset for n in notes)
    analysis.pedal_events = len(pedal)
    down = sum(max(0.0, min(p.end, hi) - max(p.start, lo)) for p in pedal)
    analysis.pedal_down_share = round(down / (hi - lo), 3) if hi > lo else 0.0


def _intersect(a: list[PedalEvent], b: list[PedalEvent]) -> list[PedalEvent]:
    """Pedal down only where both sources say so."""
    out = []
    for x in a:
        for y in b:
            s, e = max(x.start, y.start), min(x.end, y.end)
            if e > s:
                out.append(PedalEvent(s, e))
    return out


def _shift(notes, applied: int, offset: float) -> list[PerfNote]:
    return [PerfNote(n.pitch - applied, n.onset + offset, n.offset + offset, n.velocity) for n in notes]


def _run_models(res, rubric: Rubric, hub: ModelHub, analysis, ref: Reference, tr, perf: list[PerfNote],
                audio, sr: int, offset: float, applied: int, profile: dict | None) -> None:
    """Second opinions, pedal source, audio dynamics and teacher comparison: only what the rubric needs."""
    used: dict[str, str] = {}
    so = rubric.options.second_opinion
    names = [n for n in ("hft", "kong") if hub.available(n)] if so == "auto" else list(so)
    second_results, kong_tr = [], None
    for name in names:
        model = hub.get(name)
        if model is None:
            res.warnings.append(f"model_unavailable: {name} ({hub.errors.get(name)})")
            continue
        tr2 = model.transcribe(audio, sr)
        used[name] = tr2.version
        if name == "kong":
            kong_tr = tr2
        second_results.append(second_opinion.apply(
            analysis, perf, _shift(tr2.notes, applied, offset), name,
            credit_missing=rubric.options.second_opinion_policy == "credit"))

    pedal = [PedalEvent(p.start + offset, p.end + offset, p.value) for p in tr.pedal] if tr.supports_pedal else None
    if rubric.options.pedal_source != "primary":
        if kong_tr is None and (kong := hub.get("kong")) is not None:
            kong_tr = kong.transcribe(audio, sr)
            used["kong"] = kong_tr.version
        if kong_tr is None:
            res.warnings.append("model_unavailable: kong (pedal_source falls back to the main transcriber)")
        else:
            kp = [PedalEvent(p.start + offset, p.end + offset, p.value) for p in kong_tr.pedal]
            pedal = kp if rubric.options.pedal_source == "kong" else _intersect(pedal or [], kp)
    _add_pedal(analysis, pedal, perf)

    model_levels = None
    if "dynamics" in models_needed(rubric, analysis.velocity_flat):
        est = hub.get("dynamics")
        if est is None:
            res.warnings.append(f"model_unavailable: dynamics ({hub.errors.get('dynamics')})")
        else:
            model_levels = dict(enumerate(est.levels_at(audio, sr, [n.onset for n in perf], offset)))
            used["dynamics"] = est.version

    measure_expression(ref, perf, analysis, pedal=pedal, model_levels=model_levels)
    if profile is not None:
        if profile.get("reference_sha256") not in (None, ref.sha256):
            res.warnings.append("teacher_profile_is_for_a_different_score_version_ignored")
        else:
            analysis.extras["teacher"] = compare_profile(analysis.extras, profile)
    if used:
        res.versions["models"] = used
    analysis.extras["second_opinion"] = second_results


def assess(
    reference: str | Path | Reference,
    audio_path: str | Path,
    rubric: Rubric,
    transcriber: Transcriber,
    start_s: float | None = None,
    config: PipelineConfig | None = None,
    profile: dict | None = None,
    hub: ModelHub | None = None,
) -> AssessmentResult:
    """`profile`: the song's teacher benchmark (ras.core.profile); `hub`: loaded optional models."""
    cfg = config or PipelineConfig()
    hub = hub or default_hub()
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
    full_audio = audio
    audio, offset = crop_from_start(audio, cfg.sr, start_s, cfg.start_margin_s)
    if cfg.normalize_level:
        audio = normalise_level(audio)
    mark("preprocess", t0)

    t0 = time.perf_counter()
    tr = transcriber.transcribe(audio, cfg.sr)
    start_too_late = False
    first_local = min((n.onset for n in tr.notes), default=None)
    if offset > 0 and (_sounding_at_start(audio, cfg.sr) or
                       (first_local is not None and first_local < cfg.start_cut_risk_s)):
        # playing was already under way where we cut: the timestamp is too late and the crop may
        # have removed notes, so transcribe the whole recording instead
        start_too_late = True
        audio = normalise_level(full_audio) if cfg.normalize_level else full_audio
        offset = 0.0
        tr = transcriber.transcribe(audio, cfg.sr)
        res.warnings.append("start_timestamp_too_late_reassessed_on_full_recording")
    mark("transcribe", t0)
    res.versions["transcriber"] = tr.version
    notes = [PerfNote(n.pitch, n.onset + offset, n.offset + offset, n.velocity) for n in tr.notes]
    if not notes:
        res.warnings.append("no_notes_detected")

    start_suspect = False
    first_onset = min((n.onset for n in notes), default=None)
    if start_s is not None and first_onset is not None:
        start_suspect = start_too_late or abs(first_onset - start_s) > cfg.start_check_window_s
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
    if (cfg.verify_missing or rubric.options.verify_missing_audio) and analysis.n_missing:
        n_unc = mark_uncertain_missing(
            analysis, OnsetEvidence(audio, cfg.sr, offset), cfg.verify_threshold_db, performed=perf)
        if n_unc:
            res.warnings.append(f"{n_unc}_missing_notes_have_audio_evidence_marked_uncertain")
    mark("verify_missing", t0)

    t0 = time.perf_counter()
    _run_models(res, rubric, hub, analysis, ref, tr, perf, audio, cfg.sr, offset, applied, profile)
    mark("models_and_expression", t0)

    # 5. scoring, confidence, feedback
    t0 = time.perf_counter()
    scoring = score_analysis(rubric, analysis)
    conf = estimate_confidence(
        analysis, quality.quality_score, rubric.options.review_confidence_threshold,
        start_suspect=start_suspect, transposed_not_allowed=bool(shift and not allowed))
    disagree = [s for s in analysis.extras.get("second_opinion", []) if s["note_agreement_pct"] < 70]
    if disagree:  # the transcribers heard different things: a teacher should look
        conf.value = round(conf.value * 0.85, 3)
        conf.reasons.append("transcribers_disagree")
        conf.needs_review = conf.needs_review or conf.value < rubric.options.review_confidence_threshold
    res.summary = _py(analysis.summary())
    res.summary["second_opinion"] = analysis.extras.get("second_opinion", [])
    res.scoring = _py(scoring.to_dict())
    res.scoring_inputs = _py(analysis.scoring_inputs())
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
