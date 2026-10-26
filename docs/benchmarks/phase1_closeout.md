# Phase 1 close-out: what was found

## Done
* **Slow Transkun test** now run: passes.
* **Parangonar path** tested (`method="parangonar"`); the default matcher is the group-level DP, which
  was clearly better on repeated bars (precision 0.88 to 0.955 with perfect transcription).
* **Transcription cache** (`ras/core/transcribe/cache.py`): re-score without re-running the model.
* **Teacher-review evaluation and confidence calibration** (`ras evaluate`), ready for real data.
* **Review policy:** a result claiming more than 40% of the score's notes are wrong or missing always goes to
  a teacher (`many_error_claims`). This is a policy choice, not a tuned number.

## The repeated-note "false missing" problem: diagnosis
1. The missed notes were **not** specifically repeated notes. They were mostly E4/G4, mostly in recordings
   with no room or noise ("clean").
2. Making the synth more piano-like (detuned string pairs, brighter hammer, body resonance) did **not**
   help. It made Transkun worse: error-detection precision 0.70 (original synth) vs 0.34 (new synth).
3. Adding a faint noise floor, or skipping AAC, changed recall between 0.62 and 0.88 on the same notes.
4. Conclusion: Transkun is brittle on my additive synth, and its misses and ghost notes move with small
   details of the audio. **Synthetic audio cannot tell us Transkun's real accuracy.**
5. I reverted to the original synth, so the committed benchmark (`phase1_synthetic.md`) stays reproducible.

## Mitigations tried
| Idea | Result | Decision |
|---|---|---|
| Audio onset check before calling a note missing (neighbour-relative spectral jump, status `uncertain`) | On cached Transkun output, really-missing and falsely-missing notes overlap heavily (median evidence -4.5 vs +0.3 dB). Best threshold lifted precision 0.339 to 0.359 and cost recall (0.829 to 0.803). | Kept as an option, **off by default**. Re-test on real recordings. |
| Confidence from the share of error claims | Correlates 0.57 with false alarms, but even flagging 19 of 30 recordings left a 52% false-alarm share among the passed ones (66% overall). The noise is too pervasive for any score-side signal. | Only the 40% review policy above. No tuning to synthetic noise. |

## What would settle it
Benchmark on a sampled-piano SoundFont (needs FluidSynth and a free SoundFont, about 30 MB) or, better,
on the client's real recordings with teacher-corrected notes: Phase 2's review tooling collects exactly that.
