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


class Criterion(BaseModel):
    id: str
    weight: float = Field(gt=0)
    metric: str
    params: dict = Field(default_factory=dict)
    map: MapSpec = Field(default_factory=MapSpec)
    label: str | None = None


class RubricOptions(BaseModel):
    allow_transposition: bool = True
    review_confidence_threshold: float = 0.6
    pause_threshold_s: float = 1.5
    timing_tolerance_ms: float = 120.0  # used for early/late wording in feedback


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
            if c.metric not in METRICS:
                raise ValueError(f"unknown metric {c.metric!r}; available: {sorted(METRICS)}")
        return self


def load_rubric(path: str | Path) -> Rubric:
    return Rubric.model_validate(json.loads(Path(path).read_text()))


# --- metrics: Analysis -> number | None (None = cannot be assessed) -----------------------------


def _pct(num: float, den: float) -> float | None:
    return None if den <= 0 else 100.0 * num / den


METRICS: dict[str, Callable[[Analysis, dict], float | None]] = {
    "note_accuracy_pct": lambda a, p: _pct(a.n_correct, a.n_scored),
    "notes_present_pct": lambda a, p: _pct(a.n_correct + a.n_wrong_pitch + a.n_wrong_octave, a.n_scored),
    "extra_note_rate_pct": lambda a, p: _pct(a.n_extra, a.n_scored),
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
}


@dataclass
class ScoreResult:
    criteria: dict[str, dict] = field(default_factory=dict)
    total: float | None = None
    assessed_weight: float = 0.0
    band: str | None = None

    def to_dict(self) -> dict:
        return {"criteria": self.criteria, "total_score": self.total,
                "assessed_weight": self.assessed_weight, "band": self.band}


def score_analysis(rubric: Rubric, analysis: Analysis) -> ScoreResult:
    out = ScoreResult()
    weighted = assessed = 0.0
    for c in rubric.criteria:
        raw = METRICS[c.metric](analysis, c.params)
        if raw is None:
            out.criteria[c.id] = {"weight": c.weight, "metric": c.metric, "value": None,
                                  "score": None, "status": "not_assessed"}
            continue
        s = c.map.apply(raw)
        out.criteria[c.id] = {"weight": c.weight, "metric": c.metric, "value": round(float(raw), 3),
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
