"""Re-check the main transcriber's error claims with a second, independent transcriber.

Each error claim is re-checked against the second model's transcription:
- A score note reported missing that the second model clearly heard (right pitch, near where it
  belongs, and the main model has no such note there) becomes `uncertain`.
- A wrong note where the second model heard the written pitch instead becomes `uncertain`.
- An extra note the second model did not hear at all becomes `unconfirmed_extra`.

Neither of those statuses is held against the student; both lower the confidence, so many
disagreements send the result to a teacher. This never adds errors: a second model can only take
claims back.
"""

from __future__ import annotations

from ras.core.errors.analyze import Analysis
from ras.types import PerfNote


def _heard(notes: list[PerfNote], pitch: int, t: float, window: float) -> bool:
    return any(n.pitch == pitch and abs(n.onset - t) <= window for n in notes)


def apply(analysis: Analysis, primary: list[PerfNote], second: list[PerfNote], name: str,
          window_s: float = 0.12, extra_window_s: float = 0.08, credit_missing: bool = False) -> dict:
    """`primary` and `second` are in the same (transposed) pitch space and clock as the analysis.

    `credit_missing`: a missing note the second model heard counts as correct instead of uncertain
    (rubric option second_opinion_policy="credit"; turn on once teacher reviews show the second
    model's confirmations are reliable on real recordings).
    """
    retract, unconfirmed, credited = [], [], 0
    for i, r in enumerate(analysis.results):
        if r.status == "missing" and r.expected_onset_s is not None:
            hits = [n for n in second if n.pitch == r.ref_pitch and abs(n.onset - r.expected_onset_s) <= 2 * window_s]
            heard = [n for n in hits if not _heard(primary, r.ref_pitch, n.onset, window_s)]
            if heard and credit_missing:
                r.status, r.perf_onset_s, r.perf_pitch = "correct", heard[0].onset, heard[0].pitch
                analysis.n_missing -= 1
                analysis.n_correct += 1
                credited += 1
            elif heard:
                retract.append(i)
        elif r.status in ("wrong_pitch", "wrong_octave") and r.perf_onset_s is not None:
            if (_heard(second, r.ref_pitch, r.perf_onset_s, window_s)
                    and not _heard(second, r.perf_pitch, r.perf_onset_s, window_s)):
                retract.append(i)
        elif r.status == "extra" and r.perf_onset_s is not None:
            if not _heard(second, r.perf_pitch, r.perf_onset_s, extra_window_s):
                unconfirmed.append(i)
    agree = sum(1 for n in primary if _heard(second, n.pitch, n.onset, 0.05))
    analysis.mark_uncertain(retract)
    analysis.mark_unconfirmed_extra(unconfirmed)
    return {"model": name, "claims_retracted": len(retract), "missing_credited": credited,
            "extras_unconfirmed": len(unconfirmed),
            "note_agreement_pct": round(100.0 * agree / max(len(primary), 1), 1)}
