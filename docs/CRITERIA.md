# Assessment criteria: TalentGum's list, how each is measured, which model it calls

TalentGum's requirements form (section 6) lists 22 criteria. Every one now has a measurement and a place in
the rubric [`data/rubrics/talentgum_criteria_v1.json`](../data/rubrics/talentgum_criteria_v1.json).
`ras criteria` prints the same map from code ([`ras/core/criteria.py`](../ras/core/criteria.py)).

How it works:
- A criterion is scored only when its measurement exists for that recording. Otherwise it is
  **not assessed**, with a reason (for example `no_pedal_marks_in_score` or
  `keyboard_not_velocity_sensitive_and_dynamics_model_not_available`). The other criteria are
  re-weighted; nothing is guessed.
- A criterion can list **fallbacks**. When the first measurement is unavailable, for example when the
  song has no teacher benchmark yet, the next one is used. The result says which metric scored it.
- A model is loaded **only when a criterion in the rubric needs it** (`models_needed` in
  [`ras/core/rubric/engine.py`](../ras/core/rubric/engine.py)), once per worker process.

## The map

| Group | Criterion | Metric (fallback) | Model called | What it needs |
|---|---|---|---|---|
| Note | Correct notes | `note_accuracy_pct` | Transkun; hFT second opinion | |
| Note | Incorrect notes | `wrong_note_rate_pct` | Transkun; hFT | |
| Note | Missed notes | `missing_note_rate_pct` | Transkun; hFT | Teacher reviews to choose the second-opinion policy |
| Note | Extra notes | `extra_note_rate_pct` | Transkun; hFT | |
| Note | Wrong octave | `wrong_octave_rate_pct` | Transkun | |
| Note | Note order | `note_order_pct` | none | |
| Chord | Chord correctness | `chord_accuracy_pct` | none | Songs with chords |
| Chord | Missing chord notes | `chord_missing_tone_pct` | none | |
| Chord | Extra chord notes | `chord_extra_tone_rate_pct` | none | |
| Chord | Wrong chord / voicing | `chord_voicing_ok_pct` | none | |
| Timing | Note onset timing | `onset_within_ms_pct` | none | |
| Timing | Note duration | `duration_vs_teacher_pct` (`duration_within_pct`) | none | Teacher recordings (cheap keyboards cut notes short) |
| Timing | Rhythm accuracy | `rhythm_ioi_within_pct` | none | |
| Timing | Tempo | `tempo_deviation_pct` | none | Target tempo (score, or `target_bpm`) |
| Timing | Tempo consistency | `tempo_consistency_cv_pct` | none | |
| Timing | Timing deviation | `median_abs_onset_dev_ms` | none | |
| Timing | Unintended pauses | `pause_count` | none | |
| Performance | Dynamics | `dynamics_match_pct` (`dynamics_vs_teacher_corr`) | Transkun velocity, or the **dynamics model** | p/f/cresc. marks in the MXL, or teacher recordings |
| Performance | Dynamic consistency | `dynamic_consistency_std` (velocity, then model) | Transkun velocity, or the **dynamics model** | |
| Performance | Articulation | `articulation_match_pct` (`articulation_vs_teacher_pct`) | none (accents: velocity or dynamics model) | Staccato/slur/tenuto/accent marks, or teacher recordings |
| Performance | Sustain / pedal | `pedal_match_pct` | Transkun pedal, or **Kong** (`pedal_source`) | Pedal marks in the MXL; a keyboard with a pedal |
| Performance | Fluency | `fluency_index` | none | Penalty weights to calibrate |

### How the new measurements work ([`ras/core/expression.py`](../ras/core/expression.py))
- **Note order:** neighbouring score notes played in reversed order, plus pairs played as each other's
  pitch (swaps).
- **Chords:** for each chord, the counts of missing, wrong, octave-displaced and extra tones (extras
  within 80 ms of the chord). A voicing error is an octave-displaced tone, or the right notes with the
  wrong bass (an inversion).
- **Rhythm:** each gap between note onsets is compared with the score's note values at the student's
  local tempo, so a slow but steady student is not penalised. A gap over 1.75 beats, if it is not
  already a pause, is a **hesitation**. Fluency combines pauses, restarts and hesitations.
- **Articulation:**
  - Staccato: the performed length must be at most 60% of the written value (45% for staccatissimo).
  - Tenuto: at least 80%.
  - Slurred notes: at most 80 ms of silence before the next note in the same voice.
  - Accents: louder than the neighbouring notes.
- **Dynamics:**
  - The written dynamics are split into sections; each section's median loudness must move in the
    marked direction (p to f louder).
  - Crescendo and diminuendo hairpins must change loudness the right way.
  - Consistency is the spread of loudness inside each section.
  - Loudness comes from note velocity when the keyboard is touch-sensitive, otherwise from the audio
    dynamics model.
- **Pedal:** pedal-down time is compared with the score's pedal marks, mapped to the student's
  timeline. Results: agreement, coverage of the marked spans, and misuse outside them.

## Open-source models

`ras models list` shows each model and whether it is downloaded. `ras models download` fetches the
weights into `RAS_MODELS_DIR` (default `./models`, `/opt/models` in the image) and checks each
SHA-256. Downloaded checkpoints are never fully unpickled: [`ras/models/safe_load.py`](../ras/models/safe_load.py)
allows only tensors, arrays and our reviewed model classes.

| Model | Licence | Size | Role | Rubric switch |
|---|---|---|---|---|
| [hFT-Transformer](https://github.com/sony/hFT-Transformer) (Sony, ISMIR 2023) | MIT | 21 MB | Second transcriber. A missing or wrong note it heard correctly becomes `uncertain` (or is credited); an extra note it did not hear becomes `unconfirmed_extra`. | `options.second_opinion: ["hft"]` |
| [Kong et al.](https://github.com/qiuqiangkong/piano_transcription_inference) (ByteDance) | MIT | 164 MB | Pedal detection (pedal F1 0.92 on MAESTRO). Optional third vote. | `options.pedal_source: "kong" \| "agree"` |
| [piano-dynamic-estimation](https://github.com/zhanh-he/piano-dynamic-estimation) (He et al., ICASSP 2026) | MIT | 6 MB | pp..ff from the audio, for keyboards without velocity | criterion `params.source: "model"` or `"auto"` |

A second model can only **withdraw** error claims, never add new ones. When the transcribers agree on
fewer than 70% of notes, confidence drops and the result goes to a teacher.

## Using teacher data (no neural training needed)

1. **Teacher benchmark per song.** Assess 2-3 teacher recordings like any other, then build the profile:
   - Locally: `ras profile build song.musicxml t1.m4a t2.m4a t3.m4a --out song.profile.json`, then
     `ras assess ... --profile`.
   - In the service: `POST /v1/songs/{id}/teacher-profile` with `{"assessment_ids": [...]}`. From then on
     the worker compares every student result for that song version with the profile automatically.

   Duration, articulation, dynamics contour and tempo are then judged against what teachers actually play
   on the same kind of keyboard. The spread between teachers sets the tolerance.
2. **Calibrate the rubric.** Teachers grade 30-50 recordings per criterion (0-100 or 1-5) in a CSV, and
   `ras calibrate grades.csv --rubric data/rubrics/talentgum_criteria_v1.json --out v2.json` fits each
   criterion's thresholds (isotonic fit) and, with an overall grade, the weights (non-negative least
   squares).
3. **Fine-tune Transkun, only if evaluation shows a systematic transcription error.**
   `ras finetune prepare items.json` labels teacher and reviewed recordings by aligning them to the
   score, since there is no MIDI. It writes Transkun's dataset plus a starting checkpoint holding the
   released weights, and prints the GPU training command. `ras finetune export` turns the result into
   weights; set `RAS_TRANSKUN_WEIGHT` to use them.

## Measured so far (synthetic audio only)

With the hFT second opinion, error-detection precision went from 0.725 to 0.906 (recall 0.899 to 0.891)
on 16 synthetic recordings. Details: [benchmarks/criteria_models.md](benchmarks/criteria_models.md).

## Still not available, and what unblocks it

| Item | Why | Needs |
|---|---|---|
| Accuracy of every criterion on real recordings | only synthetic audio so far | TalentGum recordings + teacher grades |
| Dynamics on keyboards without velocity sensitivity | the audio model was trained on concert piano (F1 0.53); unproven on cheap keyboards | keyboard models list; teacher recordings to validate; else drop or show as feedback only |
| Dynamics / articulation / pedal against the score | needs those marks in the MXL | client to add p/f, staccato, slurs, pedal marks, or provide teacher recordings |
| Sustain pedal | many student keyboards have no pedal; Kong unvalidated on keyboards | client to confirm keyboards and whether songs use pedal |
| Real rubric weights and thresholds | the client has no rubric; current values are first guesses | 30-50 teacher-graded recordings per song, then `ras calibrate` |
| `second_opinion_policy` uncertain vs credit | needs evidence that hFT confirmations are right | teacher-reviewed recordings |
| Transkun fine-tuning | trainer is CUDA-only; no real data | GPU + a few hours of teacher/reviewed recordings |
| Off-the-shelf learned "musicality" scorer | none is open and commercially usable for beginners (Polytune is non-commercial; PercePiano rates virtuosos) | would be trained on TalentGum grades later |
