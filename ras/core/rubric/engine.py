"""Rubric engine: a pure function from measured metrics to criterion scores and a total.

The rubric is data (JSON). Weights and thresholds belong to the client; the bundled
`data/rubrics/dev_placeholder.json` exists only so the pipeline runs before the real rubric arrives.
A criterion whose metric cannot be measured is reported as "not assessed" and never guessed;
the total is computed over the assessed criteria with their weights renormalised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from ras.core.errors.analyze import Analysis


class RubricError(ValueError):
    pass


class MapSpec(BaseModel):
    """Map a metric value to 0-100. `linear`: from [a, b] -> [0, 100] (a > b means lower is better)."""

    type: Literal["linear", "piecewise", "step"] = "linear"
    from_: list[float] = Field(default_factory=lambda: [0.0, 100.0], alias="from")
    to: list[float] = Field(default_factory=lambda: [0.0, 100.0])
    points: list[tuple[float, float]] = Field(default_factory=list)  # piecewise / step: (x, score)

    model_config = {"populate_by_name": True}

    def apply(self, x: float) -> float:
        if self.type == "linear":
            (a, b), (lo, hi) = self.from_, self.to
            if a == b:
                raise RubricError("linear map needs distinct 'from' bounds")
            t = (x - a) / (b - a)
            return float(np.clip(lo + t * (hi - lo), min(lo, hi), max(lo, hi)))
        pts = sorted(self.points)
        if not pts:
            raise RubricError(f"{self.type} map needs points")
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if self.type == "piecewise":
            return float(np.interp(x, xs, ys))
        score = ys[0]  # step: value of the last point whose x <= value
        for px, py in pts:
            if x >= px:
                score = py
        return float(score)


class MetricChoice(BaseModel):
    metric: str
    params: dict = Field(default_factory=dict)
    map: MapSpec = Field(default_factory=MapSpec)


class Criterion(BaseModel):
    id: str
    weight: float = Field(gt=0)
    metric: str
    params: dict = Field(default_factory=dict)
    map: MapSpec = Field(default_factory=MapSpec)
    label: str | None = None
    # tried in order when `metric` cannot be measured (e.g. no teacher benchmark yet -> compare with the score)
    fallbacks: list[MetricChoice] = Field(default_factory=list)

    def choices(self) -> list[MetricChoice]:
        return [MetricChoice(metric=self.metric, params=self.params, map=self.map), *self.fallbacks]


class RubricOptions(BaseModel):
    allow_transposition: bool = True
    review_confidence_threshold: float = 0.6
    pause_threshold_s: float = 1.5
    timing_tolerance_ms: float = 120.0  # used for early/late wording in feedback
    # second transcribers that re-check missing, wrong and extra notes ("auto": those downloaded)
    second_opinion: list[str] | Literal["auto"] = Field(default_factory=list)
    # a missing note a second model heard: "uncertain" (not counted, lowers confidence) or "credit" (correct)
    second_opinion_policy: Literal["uncertain", "credit"] = "uncertain"
    # where pedal comes from: the main transcriber, Kong's model, or only where both agree
    pedal_source: Literal["primary", "kong", "agree"] = "primary"
    verify_missing_audio: bool = False  # the spectral double-check of missing notes (ras.core.verify)


class Rubric(BaseModel):
    rubric_id: str
    version: int = 1
    description: str = ""
    criteria: list[Criterion]
    score_bands: dict[str, tuple[float, float]] = Field(default_factory=dict)
    options: RubricOptions = Field(default_factory=RubricOptions)

    @model_validator(mode="after")
    def _check(self):
        total = sum(c.weight for c in self.criteria)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"criterion weights must sum to 1.0 (got {total:.4f})")
        ids = [c.id for c in self.criteria]
        if len(set(ids)) != len(ids):
            raise ValueError("criterion ids must be unique")
        for c in self.criteria:
            for m in c.choices():
                if m.metric not in METRICS:
                    raise ValueError(f"unknown metric {m.metric!r}; available: {sorted(METRICS)}")
        return self


def load_rubric(path: str | Path) -> Rubric:
    return Rubric.model_validate(json.loads(Path(path).read_text()))


# --- metrics: Analysis -> number | None (None = cannot be assessed) -----------------------------


def _pct(num: float, den: float) -> float | None:
    return None if den <= 0 else 100.0 * num / den


def _x(a: Analysis, *path, default=None):
    d = a.extras or {}
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _loud(a: Analysis, p: dict) -> dict | None:
    """The dynamics block for the requested loudness source (auto: velocity when usable, else the model)."""
    src = p.get("source", "auto")
    dyn = _x(a, "dynamics", default={}) or {}
    if src == "auto":
        src = "velocity" if "velocity" in dyn else "model"
    return dyn.get(src)


def _chord_pct(a: Analysis, key: str) -> float | None:
    c = _x(a, "chords")
    if not c or not c["tones"]:
        return None
    return _pct(sum(ch[key] for ch in c["chords"]), c["tones"])


def _per_chord(a: Analysis, count) -> float | None:
    c = _x(a, "chords")
    return None if not c or not c["total"] else _pct(sum(count(ch) for ch in c["chords"]), c["total"])


def _order_pct(a: Analysis) -> float | None:
    o = _x(a, "order")
    if not o or not o["pairs"]:
        return None
    return max(0.0, 100.0 * (1 - (o["inversions"] + o["swaps"]) / o["pairs"]))


def _tempo_dev(a: Analysis, p: dict) -> float | None:
    if not a.mean_bpm:
        return None
    target = p.get("target_bpm") or a.ref_bpm
    return abs(a.mean_bpm / target - 1.0) * 100.0


def _fluency(a: Analysis, p: dict) -> float | None:
    if a.mean_bpm is None:
        return None
    hes = len(_x(a, "rhythm", "hesitations", default=[]) or [])
    penalty = (p.get("pause", 15.0) * len(a.pauses) + p.get("restart", 20.0) * len(a.restarts)
               + p.get("hesitation", 5.0) * hes)
    return max(0.0, 100.0 - penalty)


def _artic(a: Analysis, p: dict) -> float | None:
    art = _x(a, "articulation")
    if not art:
        return None
    kinds = p.get("kinds") or list(art["by_type"])
    ok = sum(art["by_type"][k]["ok"] for k in kinds if k in art["by_type"])
    n = sum(art["by_type"][k]["checked"] for k in kinds if k in art["by_type"])
    return _pct(ok, n)


def _dyn_match(a: Analysis, p: dict) -> float | None:
    d = _loud(a, p)
    return None if not d else _pct(d["ok"], d["comparisons"])


def _pedal_match(a: Analysis, p: dict) -> float | None:
    ped = _x(a, "pedal", default={}) or {}
    return None if ped.get("agreement") is None else 100.0 * ped["agreement"]


def _teacher(a: Analysis, key: str) -> float | None:
    return _x(a, "teacher", key)


METRICS: dict[str, Callable[[Analysis, dict], float | None]] = {
    "note_accuracy_pct": lambda a, p: _pct(a.n_correct, a.n_scored),
    "notes_present_pct": lambda a, p: _pct(a.n_correct + a.n_wrong_pitch + a.n_wrong_octave, a.n_scored),
    "extra_note_rate_pct": lambda a, p: _pct(a.n_extra, a.n_scored),
    "wrong_note_rate_pct": lambda a, p: _pct(a.n_wrong_pitch + a.n_wrong_octave, a.n_scored),
    "missing_note_rate_pct": lambda a, p: _pct(a.n_missing, a.n_scored),
    "wrong_octave_rate_pct": lambda a, p: _pct(a.n_wrong_octave, a.n_scored),
    "chord_accuracy_pct": lambda a, p: _pct(a.chords_correct, a.chords_total),
    "onset_within_ms_pct": lambda a, p: (
        None if not len(a.onset_dev_ms)
        else 100.0 * float(np.mean(np.abs(a.onset_dev_ms) <= p.get("ms", 100.0)))),
    "median_abs_onset_dev_ms": lambda a, p: (
        None if not len(a.onset_dev_ms) else float(np.median(np.abs(a.onset_dev_ms)))),
    "tempo_consistency_cv_pct": lambda a, p: None if a.tempo_cv is None else a.tempo_cv * 100.0,
    "pause_count": lambda a, p: float(len(a.pauses)),
    "duration_within_pct": lambda a, p: (
        None if not len(a.duration_ratio)
        else 100.0 * float(np.mean(np.abs(np.log(np.clip(a.duration_ratio, 1e-3, None)))
                                   <= np.log(1 + p.get("tolerance", 0.5))))),
    "dynamic_variation_vel_std": lambda a, p: None if a.velocity_flat else a.velocity_std,
    "restart_count": lambda a, p: float(len(a.restarts)),
    "pedal_use_pct": lambda a, p: (
        None if not a.pedal_supported or a.pedal_down_share is None else 100.0 * a.pedal_down_share),
    # note order and chords
    "note_order_pct": lambda a, p: _order_pct(a),
    "chord_missing_tone_pct": lambda a, p: _chord_pct(a, "missing"),
    "chord_wrong_tone_pct": lambda a, p: (
        None if _chord_pct(a, "wrong") is None else _chord_pct(a, "wrong") + _chord_pct(a, "octave")),
    "chord_extra_tone_rate_pct": lambda a, p: _per_chord(a, lambda c: c["extra"]),
    "chord_voicing_ok_pct": lambda a, p: _per_chord(a, lambda c: not c["voicing_error"]),
    # tempo, rhythm, fluency
    "tempo_deviation_pct": _tempo_dev,
    "rhythm_ioi_within_pct": lambda a, p: (
        None if not _x(a, "rhythm", "ioi_checked") else _pct(a.extras["rhythm"]["ioi_within"],
                                                           a.extras["rhythm"]["ioi_checked"])),
    "hesitation_count": lambda a, p: (
        None if _x(a, "rhythm") is None else float(len(a.extras["rhythm"].get("hesitations", [])))),
    "fluency_index": _fluency,
    # articulation, dynamics, pedal against the score's marks
    "articulation_match_pct": _artic,
    "dynamics_match_pct": _dyn_match,
    "dynamic_consistency_std": lambda a, p: (_loud(a, p) or {}).get("consistency_std"),
    "dynamic_range": lambda a, p: (_loud(a, p) or {}).get("range"),
    "pedal_match_pct": _pedal_match,
    # against the teachers' benchmark recordings of the same song (ras.core.profile)
    "duration_vs_teacher_pct": lambda a, p: _teacher(a, "duration_within_pct"),
    "articulation_vs_teacher_pct": lambda a, p: _teacher(a, "articulation_within_pct"),
    "dynamics_vs_teacher_corr": lambda a, p: _teacher(a, "dynamics_corr"),
    "tempo_vs_teacher_pct": lambda a, p: _teacher(a, "tempo_deviation_pct"),
}

LOUDNESS_METRICS = {"dynamics_match_pct", "dynamic_consistency_std", "dynamic_range", "articulation_match_pct"}


def models_needed(rubric: "Rubric", velocity_flat: bool | None = None) -> set[str]:
    """Optional models this rubric's criteria call for (ras.models). `velocity_flat` None: not known yet."""
    need: set[str] = set()
    for k, m in ((k, m) for c in rubric.criteria for k, m in enumerate(c.choices())):
        if m.metric in LOUDNESS_METRICS or m.metric == "dynamics_vs_teacher_corr":
            src = m.params.get("source", "auto")
            # a model-based fallback is only reached when the velocities are unusable
            if (src == "model" and (k == 0 or velocity_flat is not False)) or (
                    src == "auto" and velocity_flat is not False):
                need.add("dynamics")
        if m.metric in ("pedal_match_pct", "pedal_use_pct") and rubric.options.pedal_source != "primary":
            need.add("kong")
    so = rubric.options.second_opinion
    if so != "auto":
        need.update(so)
    return need


@dataclass
class ScoreResult:
    criteria: dict[str, dict] = field(default_factory=dict)
    total: float | None = None
    assessed_weight: float = 0.0
    band: str | None = None

    def to_dict(self) -> dict:
        return {"criteria": self.criteria, "total_score": self.total,
                "assessed_weight": self.assessed_weight, "band": self.band}


def not_assessed_reason(metric: str, a: Analysis, params: dict) -> str:
    """Why a criterion could not be scored, in words a reviewer can act on."""
    if metric.endswith("_vs_teacher_pct") or metric == "dynamics_vs_teacher_corr":
        return "no_teacher_benchmark_for_this_song"
    if metric in LOUDNESS_METRICS and metric != "articulation_match_pct" and _loud(a, params) is None:
        return "no_loudness_source: keyboard_not_velocity_sensitive_and_dynamics_model_not_available"
    if metric == "dynamics_match_pct" and not _x(a, "dynamics", "marks_in_score"):
        return "no_dynamics_marks_in_score"
    if metric == "articulation_match_pct" and not _x(a, "articulation", "marks_in_score"):
        return "no_articulation_marks_in_score"
    if metric in ("pedal_match_pct", "pedal_use_pct"):
        if not _x(a, "pedal", "supported", default=a.pedal_supported):
            return "transcriber_does_not_detect_pedal"
        if metric == "pedal_match_pct" and not _x(a, "pedal", "marks_in_score"):
            return "no_pedal_marks_in_score"
    if metric.startswith("chord_"):
        return "no_chords_in_score"
    return "too_few_matched_notes"


def score_analysis(rubric: Rubric, analysis: Analysis) -> ScoreResult:
    out = ScoreResult()
    weighted = assessed = 0.0
    for c in rubric.criteria:
        raw, choice = None, None
        for m in c.choices():
            raw = METRICS[m.metric](analysis, m.params)
            if raw is not None:
                choice = m
                break
        if raw is None:
            out.criteria[c.id] = {"weight": c.weight, "metric": c.metric, "value": None,
                                  "score": None, "status": "not_assessed",
                                  "reason": not_assessed_reason(c.metric, analysis, c.params)}
            continue
        s = choice.map.apply(raw)
        out.criteria[c.id] = {"weight": c.weight, "metric": choice.metric, "value": round(float(raw), 3),
                              "score": round(s, 1), "status": "assessed"}
        weighted += c.weight * s
        assessed += c.weight
    out.assessed_weight = round(assessed, 4)
    if assessed > 0:
        out.total = round(weighted / assessed, 1)
        for name, (lo, hi) in rubric.score_bands.items():
            if lo <= out.total <= hi:
                out.band = name
                break
    return out
