"""Smith-Waterman local alignment of two integer sequences (row-vectorised numpy, no extra dependency)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalAlignment:
    score: int
    a0: int  # aligned span of `a` is a[a0:a1]
    a1: int
    b0: int  # aligned span of `b` is b[b0:b1]
    b1: int
    matches: int
    columns: int  # aligned columns: matches, mismatches and gaps

    @property
    def precision(self) -> float:
        return self.matches / self.columns if self.columns else 0.0


EMPTY = LocalAlignment(0, 0, 0, 0, 0, 0, 0)


def local_align(a: np.ndarray, b: np.ndarray, match: int = 2, mismatch: int = -1, gap: int = -1) -> LocalAlignment:
    """Best local alignment of `a` against `b` (linear gap penalty, `gap` < 0)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return EMPTY
    H = np.zeros((n + 1, m + 1), dtype=np.int32)
    idx = np.arange(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        sub = np.where(b == a[i - 1], match, mismatch).astype(np.int32)
        v = np.zeros(m + 1, dtype=np.int32)
        v[1:] = H[i - 1, :-1] + sub  # diagonal
        v = np.maximum(v, H[i - 1] + gap)  # skip a[i-1]
        v = np.maximum(v, 0)
        # skipping b[j-1] chains along the row: H[i, j] = max over k <= j of v[k] + gap * (j - k)
        H[i] = np.maximum.accumulate(v - gap * idx) + gap * idx
    flat = int(np.argmax(H))
    best = int(H.flat[flat])
    if best <= 0:
        return EMPTY
    i, j = divmod(flat, m + 1)
    a1, b1 = i, j
    matches = columns = 0
    while i > 0 and j > 0 and H[i, j] > 0:
        sub = match if a[i - 1] == b[j - 1] else mismatch
        if H[i, j] == H[i - 1, j - 1] + sub:
            matches += int(a[i - 1] == b[j - 1])
            i, j = i - 1, j - 1
        elif H[i, j] == H[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1
        columns += 1
    return LocalAlignment(best, i, a1, j, b1, matches, columns)
