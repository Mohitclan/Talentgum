# Class recordings: finding where the student plays

Status: **plan, nothing built yet** (written 2026-10-28). This is the first stage of working with TalentGum's
real data. The numbers below are starting values, to be tuned on classes that a teacher has labelled.

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

One record per class recording, kept as JSON next to the class file:

```json
{
  "class_file": "student_ab12/2026-09-14.m4a",
  "duration_s": 2712.4,
  "transcriber": "transkun-2.0.pt-<hash>",
  "playing_regions": [{"start_s": 600.1, "end_s": 912.0}],
  "clips": [
    {"clip_id": "student_ab12_2026-09-14_01", "song_id": "ode_to_joy",
     "start_s": 612.4, "end_s": 668.9, "match_score": 0.82,
     "reference_notes": 30, "notes_matched": 25, "coverage": "full",
     "status": "auto", "reviewer": null}
  ]
}
```

`status` is `auto`, `confirmed`, `rejected` or `adjusted`. A region that matches no song stays in
`playing_regions` with no clip, so nothing is thrown away silently.

## Teacher check

A reviewer sees each clip with its audio, the song it was matched to, and the match score. They can:
- confirm the song, or pick another of the five;
- move the start and end;
- mark "teacher playing", "not a performance" or "other song";
- mark the attempt as partial.

Confirmed and adjusted clips are the labelled set that measures this stage and the starting material for
later training. The current review screen works on notes, not clips, so this view must be added.

## How we will measure it

A teacher notes the start and end of each song attempt in a few classes (about five to start), and we
compare. Four numbers:
- how many of the labelled attempts are found (a found attempt overlaps the label by at least half and
  has the right song);
- how many invented clips per hour of class;
- song identification accuracy;
- the typical error of the start and end times, in seconds.

Targets are to be agreed with TalentGum; none are promised yet.

## Compute

Transcribing a 45 minute class takes about 40 CPU-minutes with Transkun on a laptop (measured:
57 CPU-seconds per audio minute). The GPU should cut this a lot, but we have not measured it. Before
planning the full run, time one class on the GPU. This stage runs as a batch job, outside the live
service, so the 5 minute and 15 minute limits of the service do not apply to it.

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
