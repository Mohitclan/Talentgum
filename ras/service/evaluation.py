"""Compare system results with teacher reviews, and calibrate the review threshold.

A review stores the teacher's corrections to the system's output. A note the teacher did not touch
is taken as agreed. This is the ground truth the Research Plan asks for: error-detection
precision/recall against teachers, score agreement, and how well confidence predicts mistakes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ras.eval.metrics import bootstrap_ci, prf

from .db import Assessment, Review

ERR = ("wrong_pitch", "wrong_octave", "missing")


@dataclass
class ReviewedCase:
    assessment_id: str
    student_id: str | None
    song_id: str
    confidence: float | None
    system_score: float | None
    teacher_score: float | None
    teacher_criteria: dict
    system_criteria: dict
    counts: dict  # type -> {tp, fp, fn}


def _final_statuses(notes: list[dict], corrections: list[dict]):
    """(system status, teacher status) per scored note, and extras kept/removed."""
    fix = {c["ref_index"]: c["status"] for c in corrections if c.get("kind") == "note"}
    removed_extra = {c["perf_index"] for c in corrections if c.get("kind") == "extra" and not c.get("valid", False)}
    pairs = []
    for n in notes:
        if n["status"] == "extra":
            pairs.append(("extra", "not_played" if n["perf_index"] in removed_extra else "extra"))
        elif n["ref_index"] is not None:
            # an untouched note is agreed; "uncertain" made no error claim, so it only counts if corrected
            pairs.append((n["status"], fix.get(n["ref_index"], n["status"])))
    return pairs


def case_from(a: Assessment, review: Review) -> ReviewedCase:
    counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in (*ERR, "extra")}
    for sys_s, teach_s in _final_statuses((a.result or {}).get("notes", []), review.corrections):
        for t in (*ERR, "extra"):
            claimed, true = sys_s == t, teach_s == t
            if claimed and true:
                counts[t]["tp"] += 1
            elif claimed:
                counts[t]["fp"] += 1
            elif true:
                counts[t]["fn"] += 1
    crit = {k: v.get("score") for k, v in ((a.result or {}).get("scoring") or {}).get("criteria", {}).items()}
    return ReviewedCase(a.id, a.student_id, a.song_id, a.confidence, a.total_score, review.overall_score,
                        review.teacher_scores, crit, counts)


def load_cases(db: Session) -> list[ReviewedCase]:
    """One case per reviewed assessment (the latest review wins)."""
    cases = []
    for a in db.execute(select(Assessment).where(Assessment.status == "reviewed")).scalars():
        rv = db.execute(select(Review).where(Review.assessment_id == a.id)
                        .order_by(Review.id.desc()).limit(1)).scalar()
        if rv is not None and a.result:
            cases.append(case_from(a, rv))
    return cases


def _pooled(cases, types):
    tp = sum(c.counts[t]["tp"] for c in cases for t in types)
    fp = sum(c.counts[t]["fp"] for c in cases for t in types)
    fn = sum(c.counts[t]["fn"] for c in cases for t in types)
    return prf(tp, fp, fn)


def _corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return {"pearson": pearson, "spearman": float(np.corrcoef(rx, ry)[0, 1])}


def evaluate(cases: list[ReviewedCase], target_false_alarm: float = 0.10) -> dict:
    out: dict = {"n_reviewed": len(cases)}
    if not cases:
        return out
    types = (*ERR, "extra")
    p, r, f = _pooled(cases, types)
    out["error_detection"] = {
        "precision": p, "recall": r, "f1": f,
        "precision_ci": bootstrap_ci(cases, lambda cs: _pooled(cs, types)[0]),
        "recall_ci": bootstrap_ci(cases, lambda cs: _pooled(cs, types)[1]),
        "by_type": {t: dict(zip(("precision", "recall", "f1"), _pooled(cases, (t,)))) for t in types}}
    pairs = [(c.system_score, c.teacher_score) for c in cases
             if c.system_score is not None and c.teacher_score is not None]
    if pairs:
        sysv, teach = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        out["score_agreement"] = {"n": len(pairs), "mae": float(np.mean(np.abs(sysv - teach))),
                                  "correlation": _corr(sysv, teach)}
    per_crit = {}
    for c in cases:
        for k, tv in c.teacher_criteria.items():
            sv = c.system_criteria.get(k)
            if sv is not None:
                per_crit.setdefault(k, []).append((sv, tv))
    out["criterion_agreement"] = {k: {"n": len(v), "mae": float(np.mean([abs(a - b) for a, b in v]))}
                                  for k, v in per_crit.items()}
    out["confidence"] = calibrate_confidence(cases, target_false_alarm)
    return out


def calibrate_confidence(cases: list[ReviewedCase], target_false_alarm: float = 0.10) -> dict:
    """How well does confidence predict wrong claims?

    For each threshold t, results with confidence >= t are 'passed' (no teacher). Report how many
    pass and what share of their claimed errors the teacher rejected. Recommend the lowest t that
    keeps that false-alarm share at or below the target.
    """
    scored = [c for c in cases if c.confidence is not None]
    rows, recommended = [], None
    for t in sorted({round(c.confidence, 2) for c in scored} | {0.0, 1.0}):
        passed = [c for c in scored if c.confidence >= t]
        claimed = sum(c.counts[k]["tp"] + c.counts[k]["fp"] for c in passed for k in (*ERR, "extra"))
        wrong = sum(c.counts[k]["fp"] for c in passed for k in (*ERR, "extra"))
        rate = wrong / claimed if claimed else 0.0
        rows.append({"threshold": t, "passed": len(passed), "of": len(scored), "claimed_errors": claimed,
                     "false_alarm_rate": round(rate, 3)})
        if recommended is None and passed and rate <= target_false_alarm:
            recommended = t
    return {"target_false_alarm_rate": target_false_alarm, "recommended_threshold": recommended, "table": rows}
