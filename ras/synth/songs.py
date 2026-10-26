"""Five public-domain beginner songs used as synthetic test material.

A hand is a list of (pitch, beats). `pitch` is a MIDI number, a tuple (chord) or None (rest).
All songs are 4/4 and laid out so notes never cross a barline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Event = tuple  # (pitch | tuple[int, ...] | None, beats)

C, D, E, F, G, A, B = 60, 62, 64, 65, 67, 69, 71
G3 = 55
C_MAJ_LH = (48, 52, 55)
G_MAJ_LH = (43, 47, 50)
F_MAJ_LH = (41, 45, 48)


@dataclass
class Song:
    song_id: str
    title: str
    bpm: float
    rh: list[Event]
    lh: list[Event] = field(default_factory=list)
    beats_per_measure: int = 4


def _seq(pitches, durs):
    return list(zip(pitches, durs))


TWINKLE = Song(
    "twinkle", "Twinkle Twinkle Little Star", 100,
    rh=_seq(
        [C, C, G, G, A, A, G, F, F, E, E, D, D, C,
         G, G, F, F, E, E, D, G, G, F, F, E, E, D,
         C, C, G, G, A, A, G, F, F, E, E, D, D, C],
        [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2] * 3,
    ),
    lh=_seq(
        [C_MAJ_LH, C_MAJ_LH, F_MAJ_LH, C_MAJ_LH, C_MAJ_LH, G_MAJ_LH, C_MAJ_LH, G_MAJ_LH,
         C_MAJ_LH, F_MAJ_LH, C_MAJ_LH, C_MAJ_LH],
        [4] * 12,
    ),
)

ODE_TO_JOY = Song(
    "ode_to_joy", "Ode to Joy", 108,
    rh=_seq(
        [E, E, F, G, G, F, E, D, C, C, D, E, E, D, D,
         E, E, F, G, G, F, E, D, C, C, D, E, D, C, C],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 0.5, 2,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 0.5, 2],
    ),
)

MARY = Song(
    "mary", "Mary Had a Little Lamb", 120,
    rh=_seq(
        [E, D, C, D, E, E, E, D, D, D, E, G, G,
         E, D, C, D, E, E, E, E, D, D, E, D, C],
        [1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
    ),
)

FRERE = Song(
    "frere_jacques", "Frere Jacques", 112,
    rh=_seq(
        [C, D, E, C, C, D, E, C, E, F, G, E, F, G,
         G, A, G, F, E, C, G, A, G, F, E, C, C, G3, C, C, G3, C],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2,
         0.5, 0.5, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 2, 1, 1, 2],
    ),
    lh=_seq([C_MAJ_LH, C_MAJ_LH, C_MAJ_LH, G_MAJ_LH, C_MAJ_LH, C_MAJ_LH, C_MAJ_LH, C_MAJ_LH],
            [4] * 8),
)

CLAIR = Song(
    "au_clair", "Au Clair de la Lune", 92,
    rh=_seq(
        [C, C, C, D, E, D, C, D, E, D, C, C, C, C, D, E, D, C, E, D, D, C],
        [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 4],
    ),
    lh=_seq(
        [C_MAJ_LH, G_MAJ_LH, C_MAJ_LH, C_MAJ_LH, C_MAJ_LH, G_MAJ_LH, C_MAJ_LH, C_MAJ_LH],
        [4] * 8,
    ),
)

SONGS: dict[str, Song] = {s.song_id: s for s in (TWINKLE, ODE_TO_JOY, MARY, FRERE, CLAIR)}
