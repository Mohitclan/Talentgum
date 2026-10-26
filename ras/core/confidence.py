"""Confidence and the 'needs review' flag.

HEURISTIC, not calibrated: Phase 2 replaces the constants with values fitted on
teacher-reviewed recordings (error rate above vs below the threshold).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ras.core.errors.analyze import Analysis

MANY_CLAIMS_RATE = 0.40  # share of score notes reported wrong or missing that forces a teacher review


@dataclass
class Confidence:
    value: float
    needs_review: bool
    reasons: list[str] = field(default_factory=list)


def estimate_confidence(
    analysis: Analysis,
    quality_score: float,
    threshold: float = 0.6,
    start_suspect: bool = False,
    transposed_not_allowed: bool = False,
    alignment_failed: bool = False,
) -> Confidence:
    reasons: list[str] = []
    if alignment_failed:
        return Confidence(0.0, True, ["alignment_failed"])

    conf = quality_score
    if quality_score < 0.8:
        reasons.append("audio_quality")

    # many unmatched performed notes usually means transcription noise or a restart
    if analysis.extra_ratio > 0.15:
        conf *= 1.0 - 0.7 * min(1.0, analysis.extra_ratio / 0.5)
        reasons.append("many_unexplained_notes")

    # very little of the score found: alignment is less trustworthy
    if analysis.coverage < 0.5:
        conf *= 0.5 + analysis.coverage
        reasons.append("low_score_coverage")

    # POLICY, not a tuned number: telling a child most of the piece was wrong is the claim most
    # damaging when it is mistaken, so a teacher confirms it first.
    claim_rate = (analysis.n_missing + analysis.n_wrong_pitch + analysis.n_wrong_octave) / max(analysis.n_ref, 1)
    many_claims = claim_rate > MANY_CLAIMS_RATE
    if many_claims:
        reasons.append("many_error_claims")

    unc = getattr(analysis, "n_uncertain", 0) / max(analysis.n_ref, 1)
    if unc > 0.05:
        conf *= 1.0 - min(0.6, unc * 2.0)
        reasons.append("notes_uncertain")

    if analysis.n_matched_for_timing() < 8:
        conf *= 0.8
        reasons.append("few_matched_notes")
    if start_suspect:
        conf *= 0.8
        reasons.append("start_timestamp_suspect")
    if transposed_not_allowed:
        reasons.append("performance_transposed")
    conf = max(0.0, min(1.0, conf))
    review = conf < threshold or transposed_not_allowed or many_claims
    return Confidence(round(conf, 3), review, reasons)
