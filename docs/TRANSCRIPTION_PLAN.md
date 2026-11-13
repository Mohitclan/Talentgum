# Plan: finding songs in class recordings, then transcribing only those parts

Status: **plan, not built.** Written Nov 2026 after the first real class audio showed that transcribing a whole
40-60 minute class with Transkun is slow and error-prone (overtones, talking, Zoom noise suppression), and after
reviewing the note "Finding Full-Song Performances in Piano Class Recordings" (chroma + subsequence DTW).

## The idea in one paragraph

Today `ras segment` transcribes the whole class with Transkun and then searches the notes for the five songs.
The new design turns that around. **First find where each song is played by comparing sounds**, with no
transcription: chroma features and subsequence DTW (dynamic time warping) against each song. **Then run
Transkun only on those few minutes**, to confirm the song, catch partial attempts, set exact clip times and
score the notes. The first step answers *where*. Transkun answers *what exactly was played*. Transkun still
runs once per stretch, but on about 5 minutes of a class instead of 40.

## Why not one method alone

| | Sound matching (chroma + DTW) | Note matching (Transkun + `ras segment`, today) |
|---|---|---|
| Needs the whole class transcribed | No: fast, CPU only | Yes: slow, and transcription errors hurt it |
| Overtones, talking, Zoom noise | More robust | Weaker |
| Octave shifts | Ignored (chroma has no octave) | Handled |
| Another key | Not handled, unless we search all 12 keys | Handled |
| Partial or stopped attempts | Rejected (a whole-song match is required) | Found |
| Attempts with many mistakes | Often rejected | Found |
| Our five similar songs (Twinkle, Mary, Ode use almost the same notes) | Can confuse them | Checks the runner-up song |
| Which notes were wrong | Cannot tell | Yes: needed for scoring |

Used alone, sound matching would quietly drop weak students' attempts and make accuracy look better than it is.
Used first, it saves most of the transcription time and is not fooled by transcription errors.

## Stage 1: find candidates by sound (new)

1. **Decode** the class audio to mono, 22,050 Hz (FFmpeg, already in the pipeline).
2. **Features for the class:** CENS chroma (librosa `chroma_cens`), about 10 frames a second. CENS is chroma
   smoothed over time; it was designed for "find every occurrence of a piece in a recording" and is more
   robust to tone, loudness and tempo than plain chroma. Cache per recording (`<class>.cens.npy`), so new
   thresholds rerun in seconds.
3. **Features for each song:** render the song file (MXL/MIDI, through our reference loader, so repeats are
   unfolded and the level's hands are selected) to audio with the existing synthesiser (`ras.synth.render`),
   then the same CENS. Trim leading and trailing silence. Rendering through audio keeps both sides comparable.
4. **Two kinds of query per song:**
   - **Whole song:** finds full performances. Accepted with the PDF's checks: mean cost, worst quarter cost
     (rejects plays that stop halfway), and length ratio 0.5-2.5.
   - **Sections of about four bars:** shorter queries with looser limits. They find partial attempts and
     heavily mistaken plays that the whole-song query rejects. These are only candidates; stage 2 decides.
5. **All 12 keys:** rotate the song's chroma 12 ways and keep the best. The rotation found is reported as
   `key_shift`. A song played in another key then still matches.
6. **Subsequence DTW** (librosa `sequence.dtw` with `subseq=True`). The step sizes limit tempo to about
   0.5x-2x. Every low point of the matching curve at least half a song apart is a candidate, so a song played
   three times gives three candidates.
7. **Song margin:** for each candidate, also match the other four songs on the same stretch. If a rival is
   almost as good, mark the candidate `could_be_<song>`. These are the songs that confuse chroma.
8. **Candidate windows:** merge overlapping candidates, pad by 10 s each side, cap at 5 minutes. Keep
   **every** candidate, including rejected ones, with its costs, so nothing disappears unseen.

## Stage 2: transcribe only the candidates (changes to `ras segment`)

1. Run Transkun (GPU when available, `RAS_DEVICE`) on each candidate window only, cached as today.
2. Run the existing note matching (`find_clips`) inside each window. It confirms the song, splits repeated or
   abandoned attempts, marks partial coverage, suggests teacher clips (`likely_teacher`) and sets exact start
   and end times.
3. **Combine the evidence:**

   | Stage 1 (sound) | Stage 2 (notes) | Result |
   |---|---|---|
   | Same song, accepted | Same song | `auto` |
   | Same song, rejected (weak or partial) | Same song | kept, `needs_review` if few notes matched |
   | Different songs | | `needs_review`, both songs listed |
   | Candidate | No song found | kept in `playing_regions` with no clip, as today |
4. Add to each clip in the annotation JSON: `method: "hybrid"`, `chroma_cost`, `worst_quarter_cost`,
   `length_ratio`, `key_shift`, `chroma_song`. Everything else stays as documented in CLASS_RECORDINGS.md.

## Stage 3: score (unchanged)

Each confirmed clip goes through the normal assessment with the student's level rubric
(`talentgum_beginner_v1` / `talentgum_advanced_v1`). The level will come from the client's age-group folders.

## Command line

```bash
ras segment classes/ --songs songs/ --out data/classes --method hybrid   # new default once calibrated
ras segment classes/ --songs songs/ --out data/classes --method notes    # today's behaviour
ras segment classes/ --songs songs/ --out data/classes --method chroma   # stage 1 only: candidates, no Transkun
```

Thresholds (mean cost, worst quarter cost, length ratio, section limits, margin) go into `SegmentConfig`, next
to today's settings, and are written into each annotation as now.

## Calibration (before any number is trusted)

1. 5-10 class recordings where a person has written down, for every song attempt: start, end, song, teacher
   or student, full or partial. This is the same labelled set CLASS_RECORDINGS.md already asks for.
2. Run `--method chroma`, `notes` and `hybrid` on them, and score each with `ras segment-eval` (found
   attempts, invented clips per hour, song accuracy, start/end error).
3. Tune the thresholds on half of the classes; report the numbers on the other half.
4. Make `hybrid` the default only if it finds at least as many attempts as `notes` and is faster.

## Compute

- Stage 1 is CPU only: a few seconds to a minute per class for the features (cached), and seconds per song
  for the DTW.
- Stage 2 transcribes about 5-10 minutes of a 40-minute class instead of all of it: roughly 4-8x less
  Transkun time. Measured numbers come with calibration.
- 500-600 recordings become a batch job (one worker per CPU core for stage 1, the GPU for stage 2).

## Recordings: ask the client (cheapest big win)

For **future** classes, in the Zoom desktop app:
- **"Record a separate audio file for each participant"** (local recording) gives the teacher and the student
  on separate tracks. That answers "teacher or student?" and removes talking over the piano. Reported to work
  for local recordings only; cloud recording support is unclear.
- **"Original sound for musicians"** stops Zoom's noise suppression from muffling quiet piano.

## Risks

- **Similar songs:** Twinkle, Mary and Ode to Joy share most pitch classes. The margin check and stage 2
  are there for this; calibration will show how often it matters.
- **Talking over playing** raises the chroma cost. The section queries and looser stage 1 limits soften this.
- **Teacher and student playing together** may still match but at a higher cost.
- **Different arrangement** (simplified, extra repeats): reference variants can be added as extra song files.
- **Sound vs keyboard timbre:** the synthesised reference does not sound like a cheap keyboard; CENS is
  chosen to absorb that. Calibration will tell.

## Licences

librosa (ISC) and NumPy/SciPy are already dependencies; no new model is needed. If we later want sharper
boundaries from onset features, the Sync Toolbox (AudioLabs Erlangen) is an option; check its code licence
first. A speech/piano separation model (Demucs, piano stem) is not planned: its piano output is described as
experimental and the licence of its weights is disputed.

## Work, in order

1. `ras/segment/chroma.py`: CENS features with caching, song rendering, 12-key subsequence DTW, the
   whole-song and section queries, the cost checks, the song margin. Tests on synthetic classes
   (`tests/test_segment.py` already renders them), including a transposed song, a stopped attempt and two
   similar songs.
2. `ras segment --method hybrid|notes|chroma`: windowed transcription and the evidence table above. Today's
   `notes` behaviour stays the default until calibration says otherwise.
3. Annotation JSON fields and `ras segment-eval` reporting per method.
4. Calibrate on labelled classes; update CLASS_RECORDINGS.md with the measured numbers.
5. Optional, only if needed: onset features for boundaries; a piano-activity pre-filter.

**Done when** hybrid finds at least as many labelled attempts as today's method on the held-out classes, with no
more invented clips, and transcribes at least 4x less audio.

## Needed

- 5-10 labelled class recordings (see Calibration).
- The `find_full_song.py` script described in the PDF, to compare with this design.
- The age-group folder layout, so each class gets the right level.
- The Zoom settings above, for future recordings.
