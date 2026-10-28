# Class recordings: finding where the student plays

Status: **first version built** (`ras segment`, code in `ras/segment/`), tested only on synthetic classes. This is the
first stage of working with TalentGum's real data. The numbers below are starting values, to be tuned on
classes that a teacher has labelled.

## Why this stage exists

TalentGum's recordings are online classes of 40-60 minutes, not single performances. A class holds
talking, the teacher, other people, silence, and one or more attempts at a song, sometimes two songs.
There are no start timestamps and no quality labels. The assessment system takes one short recording
(at most 5 minutes, one performance), so a class must first be cut into clips: one clip per attempt,
labelled with the song.

Everything later depends on this: the first accuracy test, teacher marking, calibration and
fine-tuning all start from these clips.

## Input

- One folder per student, with that student's class recordings inside (M4A/AAC, raw, about 45 min).
- The five songs as MXL (and MIDI): Twinkle Twinkle Little Star, Mary Had a Little Lamb, Happy Birthday,
  Believer, Ode to Joy. Only Ode to Joy has been received so far.
- The student's folder name is the student id. Files that leave the team carry a short stable id
  instead of the name.

## Method

The idea: transcribe the whole class once, then search the notes for each song.

| Step | What happens | Tool |
|---|---|---|
| 1. Transcribe the class | The whole recording goes through the piano transcriber in pieces and becomes a list of notes with times. | Transkun V2 (already in the system), on the GPU |
| 2. Find the playing | Cut the notes into windows of about 10 s. A window with enough notes, in a piano range, is playing. Talking and noise give few, scattered, quiet notes. Join windows less than a few seconds apart into regions. | Plain code |
| 3. Identify the song | Compare each region's notes with each of the five songs by local alignment of the note sequence. Compare the steps between notes (up 2, down 1) instead of exact pitches, so a wrong note breaks one step and a transposed song still matches. The best song wins if enough of its notes match. | The existing local alignment (`_local_align_matches` in `ras/core/errors/analyze.py`) |
| 4. Split the attempts | Inside a region, each return to the song's first notes starts a new attempt; a long silence between separate matches also splits. Record whether the attempt is full or partial. | Plain code |
| 5. Cut and hand over | Each attempt becomes a clip (about 2 s before the first note to 2 s after the last, at most 5 min). The clip goes to the normal assessment, with its start time given. | ffmpeg and the existing `assess()` |
| 6. Teacher check | Low-confidence clips go to a teacher to confirm (below). | Review screen (needs a new view) |

Starting values, to tune: window 10 s with a 5 s step; at least about 1 note per second inside a
window; notes between MIDI 36 and 96; regions merged when gaps are under 3 s; a song accepted when at
least half of its notes are matched in the aligned stretch.

### Add-ons, only if needed

- **Piano-or-speech detector first** (a pre-trained audio tagger such as PANNs or YAMNet; check each
  licence before use). Skips the talking so only the piano parts are transcribed. Worth it only if GPU
  time turns out to be a problem.
- **Match against the song's sound** (render each MXL to audio, then match with chroma features and
  subsequence DTW). A cross-check for uncertain matches. It does not depend on transcription errors.

## Output

One record per class recording, written next to its transcript and clips as
`<out>/<student_id>/<class>.json`:

```json
{
  "class_file": "class_1.m4a",
  "student_id": "s_3f9a1c02",
  "duration_s": 2712.4,
  "transcriber": "transkun-2.0.pt-<hash>",
  "notes_file": "s_3f9a1c02/class_1.notes.json",
  "songs": ["ode_to_joy", "twinkle"],
  "playing_regions": [{"start_s": 600.1, "end_s": 912.0, "notes": 240, "clips": 3}],
  "clips": [
    {"clip_id": "s_3f9a1c02_class_1_01", "song_id": "ode_to_joy",
     "start_s": 612.4, "end_s": 668.9, "clip_start_s": 610.4, "clip_end_s": 670.9, "assess_start_s": 2.0,
     "match_score": 0.92, "matched_steps": 25, "song_steps": 29, "coverage": "full", "coverage_pct": 96.0,
     "song_bars": [1, 8], "margin": 0.49, "status": "auto", "review_reasons": [], "reviewer": null,
     "file": "s_3f9a1c02/class_1_clips/s_3f9a1c02_class_1_01.flac"}
  ]
}
```

`assess_start_s` is where the playing starts inside the clip, ready for the assessment's start timestamp.
The transcript is saved too, so thresholds can be changed and the class re-searched without transcribing
again. `status` starts as `auto` or `needs_review` (few notes matched, many wrong steps, a rival song matches
almost as well, or the clip was cut at 5 minutes); a teacher sets it to `confirmed`, `rejected` or `adjusted`. A region that matches no song stays in
`playing_regions` with no clip, so nothing is thrown away silently.

## How to run

```bash
ras segment classes/ --songs songs/ --out data/classes
```

- `classes/` holds one folder per student with that student's class recordings inside; a single file also works.
- `songs/` holds the MXL or MIDI files; the file name is the song id (`ode_to_joy.mxl`).
- Student folder names become hashed ids in every output; `--keep-names` turns that off.
- A saved transcript is reused on a second run (`--no-reuse` to redo it). `--chunk-s` sets the piece length
  (default 10 minutes, overlapping by 10 s, joined in the middle of the overlap).
- `ras segment-eval annotation.json labels.json` scores one class against a teacher's labels.

## Teacher check

A reviewer sees each clip with its audio, the song it was matched to, and the match score. They can:
- confirm the song, or pick another of the five;
- move the start and end;
- mark "teacher playing", "not a performance" or "other song";
- mark the attempt as partial.

Confirmed and adjusted clips are the labelled set that measures this stage and the starting material for
later training. The current review screen works on notes, not clips, so this view is **not built yet**. Until it is, the
reviewer can work from the JSON and the clip files.

## How we will measure it

A teacher notes the start and end of each song attempt in a few classes (about five to start), and we
compare. Four numbers:
- how many of the labelled attempts are found (a found attempt overlaps the label by at least half and
  has the right song);
- how many invented clips per hour of class;
- song identification accuracy;
- the typical error of the start and end times, in seconds.

Targets are to be agreed with TalentGum; none are promised yet.

## First trial (synthetic class)

An 8 minute class was rendered from the synthetic songs: talking-like noise about half of the time, five
attempts of four songs (one stopped and restarted, one repeated with two wrong notes), room echo and AAC
compression. Transkun transcribed it in 2 minute chunks and `ras segment` ran on the result.

| | Result |
|---|---|
| Labelled attempts found | 6 of 6, all with the right song |
| Invented clips | 0 |
| Typical start / end error | 0.02 s / 0.03 s |
| Clips marked for a teacher | 3 of 6, all close calls between songs that move by steps (Twinkle, Mary, Ode to Joy); the song was right each time |
| Time on this laptop's CPU | about 4.5 minutes for the 8 minutes |

This shows the method works when the transcription is reasonable. It says little about real classes:
the noise here is synthetic, not speech, and the piano is a simple synth. Real classes are the test.

## Compute

On this Intel laptop's CPU, Transkun took about 4.5 minutes for 8 minutes of class (wall-clock), so a
45 minute class would take roughly 25 minutes (an extrapolation). The GPU should cut this a lot, but we
have not measured it. Before planning the full run, time one real class on the GPU. This stage runs as a
batch job, outside the live service, so the 5 minute and 15 minute limits of the service do not apply.

## Risks

- **The teacher also plays.** A demonstration looks the same as a student attempt. Options: skip the
  first playing of the song in each class, or let the teacher check mark it. Not decided.
- **Online-meeting audio.** Video calls can cut out or quieten the piano when someone talks. Test on a
  few real classes before trusting any numbers.
- **Harder songs.** Believer and Happy Birthday are longer or harder to match than the simple songs.
- **Partial attempts.** The assessment currently treats a stopped attempt as missing the rest of the
  song. It needs a "score only the part played" mode before partial clips can be scored fairly.
- **Both hands.** The song's reference must say which notes to expect when a student plays one hand;
  this also affects how well a one-handed attempt matches. Waiting for the client's rule.
- **Privacy.** Classes contain children and other people. Keep them and the annotation files in
  access-controlled storage. Permission to use them for improving the system is not yet confirmed.

## What we still need

- The MXL (and MIDI) files of the other four songs.
- Three or four class recordings to prototype on, and about five with a teacher's start and end times.
- The GPU's type and how long we can use it.
- The client's decision on the hands rule and on the teacher-playing question.

## What comes after

1. First accuracy test of the assessment on the clips, against the five songs.
2. Teachers verify and correct a sample (they agreed to do this jointly).
3. Calibrate the scoring on teacher marks.
4. Only if errors repeat on their keyboards: fine-tune the transcription model on the corrected clips
   (see [DEPLOY.md](DEPLOY.md), section 4).
