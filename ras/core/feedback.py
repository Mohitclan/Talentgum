"""Template feedback in an encouraging tone. Wording is a placeholder to be agreed with teachers."""

from __future__ import annotations

from ras.core.errors.analyze import Analysis, NoteResult
from ras.core.score.reference import Reference

_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi: int | None) -> str:
    return "?" if midi is None else f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def _where(r: NoteResult) -> str:
    if r.measure is None:
        return "somewhere in the piece"
    beat = r.beat_in_measure or 1.0
    return f"bar {r.measure}, beat {beat:g}"


def make_feedback(analysis: Analysis, ref: Reference, timing_tolerance_ms: float = 120.0,
                  max_items: int = 4) -> list[str]:
    msgs: list[str] = []
    n = max(analysis.n_scored, 1)
    right = analysis.n_correct
    if right == analysis.n_scored and not analysis.n_extra:
        msgs.append("Wonderful! Every note was right.")
    elif right / n >= 0.7:
        msgs.append(f"Great job! You played {right} of {analysis.n_scored} notes correctly.")
    elif right / n >= 0.4:
        msgs.append(f"Good effort! {right} of {analysis.n_scored} notes were right, "
                    "and every practice makes it stronger.")
    else:
        msgs.append("Nice try! Keep practising this piece slowly, one small part at a time.")

    # largest issues first: wrong notes, then missing, then timing
    issues: list[tuple[int, str]] = []
    for r in analysis.results:
        if r.status in ("wrong_pitch", "wrong_octave"):
            kind = "an octave off" if r.status == "wrong_octave" else "a different note"
            issues.append((0, f"In {_where(r)}, {note_name(r.ref_pitch)} was played as {kind} "
                              f"({note_name(r.perf_pitch)}). Try that spot again slowly."))
        elif r.status == "missing":
            issues.append((1, f"In {_where(r)}, the note {note_name(r.ref_pitch)} was not heard. "
                              "Look closely at the music there."))
    late = [r for r in analysis.results
            if r.status == "correct" and r.onset_dev_ms is not None
            and abs(r.onset_dev_ms) > 2 * timing_tolerance_ms]
    for r in sorted(late, key=lambda x: -abs(x.onset_dev_ms))[:2]:
        way = "a little late" if r.onset_dev_ms > 0 else "a little early"
        issues.append((2, f"In {_where(r)}, a note came in {way}. Tapping the beat can help."))
    for p in analysis.pauses[:1]:
        issues.append((2, f"There was a long pause around bar {p['measure']}. "
                          "Try to keep going, even slowly."))
    issues.sort(key=lambda x: x[0])
    msgs.extend(text for _, text in issues[:max_items])
    if analysis.n_extra >= 3:
        msgs.append("A few extra notes were heard. Keep your fingers on the right keys.")
    return msgs
