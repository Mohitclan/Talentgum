"""Check the audio itself before calling a note missing.

Transcription models sometimes drop a note that was played (re-struck repeated notes, soft notes,
unusual timbre). A score note reported missing is therefore double-checked: if the spectrum shows a
fresh onset at that pitch near the expected time, the note is marked `uncertain` rather than missing.
Uncertain notes are not held against the student and they lower the result's confidence.
"""

from __future__ import annotations

import librosa
import numpy as np


class OnsetEvidence:
    """Spectrogram of the (cropped, levelled) audio that the transcriber saw."""

    def __init__(self, audio: np.ndarray, sr: int, offset_s: float = 0.0, n_fft: int = 8192, hop: int = 512):
        self.sr, self.n_fft, self.hop, self.offset = sr, n_fft, hop, offset_s
        self.mag = np.abs(librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop))
        self.freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        self.fps = sr / hop

    def _band(self, f0: float) -> np.ndarray:
        """Energy over time at the fundamental (about +-35 cents).

        Only the fundamental is used: higher partials of neighbouring notes often land on it
        (A4's 3rd partial is E4's 4th), which would fake evidence.
        """
        half = max(f0 * (2 ** (35 / 1200) - 1), self.freqs[1])
        idx = np.where(np.abs(self.freqs - f0) <= half)[0]
        total = self.mag[idx].sum(axis=0) if len(idx) else np.zeros(self.mag.shape[1])
        return 20 * np.log10(total + 1e-6)

    def jump_db(self, pitch: int, t_expected: float, window_s: float = 0.25) -> float | None:
        """Largest rise (dB) of that pitch's energy within +-window of the expected time.

        None when the pitch is too low for this frequency resolution.
        """
        f0 = 440.0 * 2 ** ((pitch - 69) / 12)
        if pitch < 45:
            return None
        e = self._band(f0)
        c = (t_expected - self.offset) * self.fps
        lo, hi = int(max(c - window_s * self.fps, 6)), int(min(c + window_s * self.fps, len(e) - 1))
        if hi <= lo:
            return None
        pre, post = int(0.10 * self.fps), int(0.06 * self.fps)
        best = -np.inf
        for tau in range(lo, hi + 1):
            before = e[max(tau - pre, 0):tau].mean() if tau > 0 else e[0]
            after = e[tau:min(tau + post, len(e))].max()
            best = max(best, after - before)
        return float(best)

    def evidence_db(self, pitch: int, t_expected: float, window_s: float = 0.25) -> float | None:
        """Onset jump at `pitch` minus the larger jump at its two neighbouring semitones.

        Spectral leakage lifts the neighbours of a really-played note almost as much, so only the
        excess over them is evidence for that exact pitch (positive: played, <= 0: not played).
        """
        j = self.jump_db(pitch, t_expected, window_s)
        lo = self.jump_db(pitch - 1, t_expected, window_s)
        hi = self.jump_db(pitch + 1, t_expected, window_s)
        if j is None or lo is None or hi is None:
            return None
        return j - max(lo, hi)


# intervals (semitones) below a note whose partials can land on its fundamental
_HARMONIC_BELOW = (12, 19, 24, 28, 31)


def _explained_by_other_note(pitch: int, t: float, performed, window_s: float) -> bool:
    """A performed note an octave (or octave+fifth...) below, struck at the same time, rings on this pitch."""
    return any(abs(n.onset - t) <= window_s and (pitch - n.pitch) in _HARMONIC_BELOW for n in performed)


def mark_uncertain_missing(analysis, evidence: OnsetEvidence, threshold_db: float, performed=(),
                           window_s: float = 0.15) -> int:
    """Relabel `missing` notes that have audio evidence of an onset. Returns how many."""
    hits = []
    for i, r in enumerate(analysis.results):
        if r.status != "missing" or r.expected_onset_s is None or r.ref_pitch is None:
            continue
        jump = evidence.evidence_db(r.ref_pitch, r.expected_onset_s, window_s)
        r.audio_evidence_db = None if jump is None else round(jump, 1)
        if jump is not None and jump >= threshold_db and not _explained_by_other_note(
                r.ref_pitch, r.expected_onset_s, performed, 0.12):
            hits.append(i)
    analysis.mark_uncertain(hits)
    return len(hits)
