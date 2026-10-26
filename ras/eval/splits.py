"""Leakage-free splits: by student, and leave-one-song-out."""

from __future__ import annotations

import numpy as np


def grouped_kfold(groups: list[str], k: int = 5, seed: int = 0):
    """Yield (train_idx, test_idx); every group (e.g. student) lands in exactly one test fold."""
    uniq = sorted(set(groups))
    k = min(k, len(uniq))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(uniq)))
    folds = [[] for _ in range(k)]
    for rank, gi in enumerate(order):
        folds[rank % k].append(uniq[gi])
    g = np.array(groups)
    for fold in folds:
        test = np.where(np.isin(g, fold))[0]
        train = np.where(~np.isin(g, fold))[0]
        yield train, test


def leave_one_song_out(songs: list[str]):
    s = np.array(songs)
    for song in sorted(set(songs)):
        yield song, np.where(s != song)[0], np.where(s == song)[0]
