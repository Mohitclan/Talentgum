"""TalentGum's assessment criteria (requirements form, section 6) and how RAS measures each one.

This is the map between what the client asked for and the code: the metric in the rubric engine,
the open-source model it calls (if any), and what still limits it. `ras criteria` prints it, and
data/rubrics/talentgum_criteria_v1.json scores every one of them.

Readiness levels:
- `measured`: deterministic, works now.
- `model`: uses a downloaded open-source model.
- `needs_marks`: the MusicXML must carry the marking (dynamics, articulation, pedal).
- `needs_teacher_data`: becomes reliable once teacher recordings/grades calibrate it.
- `hardware`: depends on the student's keyboard (velocity sensitivity, a pedal).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClientCriterion:
    id: str
    group: str
    label: str
    metric: str
    fallback: str | None
    models: tuple[str, ...]
    readiness: tuple[str, ...]
    limits: str


CRITERIA: list[ClientCriterion] = [
    ClientCriterion("correct_notes", "note", "Correct notes", "note_accuracy_pct", None, ("transkun", "hft"),
                    ("measured", "model"), "Transcription errors on real student audio not yet measured."),
    ClientCriterion("incorrect_notes", "note", "Incorrect notes", "wrong_note_rate_pct", None, ("transkun", "hft"),
                    ("measured", "model"), "hFT second opinion withdraws wrong-note claims it disagrees with."),
    ClientCriterion("missed_notes", "note", "Missed notes", "missing_note_rate_pct", None, ("transkun", "hft"),
                    ("measured", "model", "needs_teacher_data"),
                    "False 'missing' were the main synthetic error; hFT now withdraws them. Confirm on real audio, "
                    "then choose second_opinion_policy (uncertain vs credit)."),
    ClientCriterion("extra_notes", "note", "Extra notes", "extra_note_rate_pct", None, ("transkun", "hft"),
                    ("measured", "model"), "Extras the second model did not hear are not counted."),
    ClientCriterion("wrong_octave", "note", "Wrong octave", "wrong_octave_rate_pct", None, ("transkun",),
                    ("measured",), ""),
    ClientCriterion("note_order", "note", "Note order", "note_order_pct", None, (), ("measured",),
                    "Counts reversed neighbouring notes and swapped pairs."),
    ClientCriterion("chord_correctness", "chord", "Chord correctness", "chord_accuracy_pct", None, (), ("measured",),
                    "Not assessed for songs without chords."),
    ClientCriterion("missing_chord_notes", "chord", "Missing chord notes", "chord_missing_tone_pct", None, (),
                    ("measured",), ""),
    ClientCriterion("extra_chord_notes", "chord", "Extra chord notes", "chord_extra_tone_rate_pct", None, (),
                    ("measured",), ""),
    ClientCriterion("chord_voicing", "chord", "Wrong chord / voicing", "chord_voicing_ok_pct", None, (),
                    ("measured",), "Voicing = octave-displaced tone or wrong bass (inversion)."),
    ClientCriterion("onset_timing", "timing", "Note onset timing", "onset_within_ms_pct", None, (), ("measured",), ""),
    ClientCriterion("note_duration", "timing", "Note duration", "duration_vs_teacher_pct", "duration_within_pct",
                    (), ("measured", "needs_teacher_data", "hardware"),
                    "Cheap keyboards cut notes short: compare with teacher recordings on the same keyboard type."),
    ClientCriterion("rhythm_accuracy", "timing", "Rhythm accuracy", "rhythm_ioi_within_pct", None, (), ("measured",),
                    "Inter-onset ratios against the score at the student's own tempo."),
    ClientCriterion("tempo", "timing", "Tempo", "tempo_deviation_pct", None, (), ("measured",),
                    "Target is the score's tempo unless the rubric sets target_bpm."),
    ClientCriterion("tempo_consistency", "timing", "Tempo consistency", "tempo_consistency_cv_pct", None, (),
                    ("measured",), ""),
    ClientCriterion("timing_deviation", "timing", "Timing deviation", "median_abs_onset_dev_ms", None, (),
                    ("measured",), ""),
    ClientCriterion("unintended_pauses", "timing", "Unintended pauses", "pause_count", None, (), ("measured",),
                    "Restarts are reported separately and not double-counted."),
    ClientCriterion("dynamics", "performance", "Dynamics", "dynamics_match_pct", "dynamics_vs_teacher_corr",
                    ("transkun", "dynamics"), ("model", "needs_marks", "hardware", "needs_teacher_data"),
                    "Needs p/f/cresc. marks in the MXL, or teacher recordings. Velocity on touch-sensitive "
                    "keyboards; otherwise the audio dynamics model (trained on concert piano, F1 0.53)."),
    ClientCriterion("dynamic_consistency", "performance", "Dynamic consistency", "dynamic_consistency_std", None,
                    ("transkun", "dynamics"), ("model", "hardware", "needs_teacher_data"),
                    "Evenness of loudness within each written dynamic."),
    ClientCriterion("articulation", "performance", "Articulation", "articulation_match_pct",
                    "articulation_vs_teacher_pct", (), ("measured", "needs_marks", "needs_teacher_data", "hardware"),
                    "Staccato/tenuto from note length, legato from gaps under slurs, accents from loudness."),
    ClientCriterion("sustain_pedal", "performance", "Sustain / pedal", "pedal_match_pct", None, ("transkun", "kong"),
                    ("model", "needs_marks", "hardware"),
                    "Needs pedal marks in the MXL and a keyboard with a pedal. Kong's model can be the pedal source."),
    ClientCriterion("fluency", "performance", "Fluency", "fluency_index", None, (), ("measured", "needs_teacher_data"),
                    "Pauses, restarts and hesitations; penalty weights to calibrate with teachers."),
]


def as_rows() -> list[dict]:
    return [c.__dict__ for c in CRITERIA]
