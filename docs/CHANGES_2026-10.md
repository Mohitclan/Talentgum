# Changes and history since commit ee6fa48

Written 30–31 Oct 2026. Read this after [HANDOFF.md](HANDOFF.md). It covers what changed in the code, what was tested on
real audio for the first time, what was decided, and what is still open.

This repository is public, so this file has no student names, class timestamps or recordings. The three real clips are
called clip A, clip B and clip C. Their audio and the detailed results stay on the original machine, outside git.

## In short

- RAS ran on real class audio for the first time: 3 clips from one beginner student.
- That exposed 4 bugs, all fixed here, plus a problem with how the wrong song was handled.
- Results improved a lot, but **no teacher has checked them, so none of this is an accuracy figure.**
- Transkun V2 stays as the main model. MIDI and MXL are the only source of truth.

## Code changes

| File | Change | Why |
|---|---|---|
| `ras/core/melody.py` (new) | `isolate_melody` picks the melody out of everything transcribed, using the score. `reference_fit` measures how well the playing fits the song | See "Bugs fixed" |
| `ras/pipeline.py` | Runs melody isolation before the transposition check. Adds the reference-fit gate, `summary.melody` and `summary.reference_fit_pct`. `ANALYSIS_VERSION` is now `analysis-3` | Wires in the above. A failed fit withholds the total score and band and replaces note feedback with a neutral message. The criteria stay in the result for the teacher |
| `ras/core/align/matcher.py` | `match_groups(any_member=True)`: a played group counts in full if it holds the score's note, whatever else sounds with it | Needed for melody isolation |
| `ras/core/rubric/engine.py` | New options `isolate_melody` (default false) and `min_reference_fit` (default 0, off) | Other rubrics keep their old behaviour |
| `data/rubrics/talentgum_criteria_v1.json` | `isolate_melody: true`, `min_reference_fit: 0.7` | Turns both on for TalentGum |
| `ras/models/__init__.py` | Downloaded model files are set to 0644 | `NamedTemporaryFile` creates them as 0600 |
| `Dockerfile` | `chmod -R a+rX /opt/models` after download | Same bug, safeguard |
| `tests/test_melody.py` (new) | 10 tests | Notes patterned on the real clips |
| `docs/HANDOFF.md` | Updated state, decisions and next steps | |

## Bugs fixed

1. **Kong could not run in the Docker image.** Weights were readable only by root, and the container runs as `ras`.
2. **Harmonics counted as extra notes.** On keyboard sound through a class call, Transkun reports notes at exact harmonics
   of the melody note (+12, +19, +24, +28, +31 semitones), often louder than the note, so loudness can't separate them.
3. **Octave shift missed when chords sit under the melody.** Chord tones matched the written octave and fooled the
   pitch histogram. Melody isolation now tries whole-octave shifts and keeps the one that finds the most score notes.
4. **A different song got a confident score.** RAS trusts the reference it is given. A clip of a song outside the five
   scored 64.1 ("good") against Mary. The score is now withheld when the fit is below the limit.

Why the filter uses the score: keeping "the lowest note" fails, because a melody E5 is the 5th harmonic of a C4 sounding
under it. Melody isolation applies only to scores with one note at a time. Songs with chords are not changed.

## Test results

Fast test suite after the changes: see "Test status" at the end. Assessments ran in the Docker image on CPU
(Intel i7-9750H), as user `ras`.

### Real clips, Transkun V2 with the hFT second opinion

| Clip | Before | After | Review |
|---|---|---|---|
| A: a song outside the five, checked against Mary | 64.1 | Withheld (fit 61.9%) | Yes |
| B: Mary | 60.2 | 76.3 | No, which is too generous for now (see open items) |
| C: Mary, chords heard under the melody in 11 of 32 places | 29.8 | 77.7 | Yes |

Clip B: extras 34 → 2, correct notes 21 → 25 of 26, tempo 115 → 155 bpm.
Clip C: shift 0 → +12, correct 11 → 24 of 26, wrong-octave 8 → 0, timing deviation 582 → 86 ms.

Model totals still disagree: clip B 53.3 to 76.3 and clip C 58.4 to 77.8 across Transkun, hFT, Kong and Basic Pitch.

### Client reference WAVs

| Model | Notes correct | False notes before | False notes after |
|---|---|---|---|
| Transkun V2 | 119/119 | 0 | 0 |
| Kong | 113/119 (114 before) | 47 | 3 |
| hFT | 111/119 | 47 | 1 |

These WAVs are clean, so this shows the engine reads clean audio. It says nothing about student audio.

### Song check

With Transkun, 13 of 13 wrong-song cases were withheld (highest fit 66.7%) and 7 of 7 right-song cases were scored (lowest
72.7%). **The 0.7 limit was set on these same 20 cases and the gap is narrow.** Retest it on new clips. It suits Transkun only:
hFT and Kong fall below it on some right-song cases.

### MIDI against MXL, all five songs

Pitches and onsets are identical. Note lengths differ on four songs (Believer is the exception); the size of the difference
hasn't been measured. Believer's MXL has no tempo, so RAS assumes 100 while its MIDI says 120.

## Criteria status (22)

**Calculated now (13), none confirmed by teachers:** correct, incorrect, missed and extra notes; wrong octave; note order;
onset timing; rhythm; tempo; tempo consistency; timing deviation; pauses; fluency.

**Partly (1):** dynamic consistency. A number comes out, but nothing can check it.

**Not yet (8):** the 4 chord criteria, dynamics, articulation and pedal need marks in the score, or client agreement that
they are N/A for phase 1. Note duration needs work: Transkun reports notes of about 0.04 s on class audio.

Weak points: note order scores 100 almost everywhere, because alignment keeps notes in order by design. The dynamics model
has F1 0.53, so use direction of change only. Kong reported pedal events on the real clips; these are unverified.

## Decisions

| Decision | Source |
|---|---|
| Transkun V2 stays the main model | Test results above |
| MIDI and MXL are the only source of truth; slides and WAVs never override them | TalentGum developer |
| Teacher recordings are not used as a reference, because teachers can make mistakes | TalentGum developer |
| Only the 5 songs are measured for now; other songs are skipped | Project team |
| Two levels: beginner and intermediate | Project team |
| Recordings are 40–60 minute classes. The song parts are annotated by hand, with assigned song and bars | Project team |
| Scores stay internal until teachers check them | Recommendation, needs team agreement |

Teachers are still needed to **mark student clips**. That is a different job from recording reference performances.

## Also tried, not in this repo

A separate CPU test pipeline used Basic Pitch with CREPE or pYIN. Lessons from it:
- pYIN and CREPE find almost no notes on class audio (0–28% confirmed), though CREPE is fine on clean audio.
- Moving notes down an octave from harmonic evidence gave wrong results. All 13 moves went down, some to implausible octaves.
- Clip-edge extension made the wrong-song problem worse, because notes outside the song diluted the match cost.
- Its readiness report describes that pipeline, not RAS, so don't use it for RAS.

## Candidates for later (untested here)

- Checker: SwiftF0 (MIT, one note at a time, CPU), Aria-AMT (Apache-2.0).
- Dynamics: direct loudness at each note first, then `zhanh-he/sfproxy-velocity-estimation`. It needs audio plus aligned MIDI,
  keeps pitch and timing, and has no stated code licence. Test on flat-velocity audio first, in case it invents dynamics.
- Skip for licence reasons: YourMT3+ and ContextAwareAMT (GPL-3.0).

## Open items

**Before the teacher check**
1. Set Believer to 120 bpm.
2. Load annotated clips with song, bars, level and hands.
3. Score only the assigned bars (not built).
4. Send every clip to review for now. Clip B got a final score with no review.
5. Marking and note-correction sheets for teachers.
6. Review screen for clips (the current screen works on notes).

**During the teacher check**
7. Fix note duration. 8. Marking per level, if the client wants it. 9. Fit weights with `ras calibrate`.
10. Retest the song check. 11. Fine-tune Transkun if errors repeat. 12. Improve or drop note order.

**Before production**
13. AWS deployment. 14. TalentGum backend integration. 15. Load, failure and security tests.
16. Monitoring and retention periods. 17. Licence check on the dynamics encoder.

**Questions for TalentGum**
- Who played the chords on clip C, and is auto-accompaniment allowed?
- Are chord, dynamics, articulation and pedal N/A for phase 1?
- How do beginner and intermediate differ: songs, hands, marking?
- Is intermediate one hand? Melody isolation depends on it.
- What agreement with teachers is good enough?
- Are keyboards touch-sensitive? This decides dynamics.
- What is the octave-button rule, and the target tempo per song?
- Do we have parents' permission for training, and what is the retention period?
- Is the Universal Keyboard Piano Rubric the one to use?
- How much teacher time is committed? This is the biggest risk.

## Reproduce on a new machine

```bash
git clone https://github.com/Mohitclan/Talentgum.git
```

```bash
docker build --target runtime --build-arg DOWNLOAD_MODELS="hft dynamics kong" -t ras:local .
```

Put private material in `data/private/`, which git ignores. Then assess a clip:

```bash
docker run --rm -v "$PWD/data/private:/work" -v "$PWD/data/rubrics:/rubrics:ro" ras:local ras assess /work/refs/song.mxl /work/clips/clip.wav --rubric /rubrics/talentgum_criteria_v1.json --out /work/out/clip.json
```

Fast tests:

```bash
docker run --rm --user root -v "$PWD:/repo" -w /repo -e PYTHONPATH=/repo --entrypoint sh ras:local -c "pip -q install pytest && python -m pytest -o addopts= -m 'not slow and not cloud' tests -q"
```

## Test status

129 passed, 1 skipped, 4 deselected (fast suite, in the Docker image, 31 Oct 2026). The slow model tests were not run on this commit.
