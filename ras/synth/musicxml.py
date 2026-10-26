"""Write a Song as a minimal two-staff MusicXML file (for tests and synthetic data)."""

from __future__ import annotations

from pathlib import Path

from .songs import Song

DIVISIONS = 4  # per quarter note
_STEPS = {0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("D", 1), 4: ("E", 0), 5: ("F", 0),
          6: ("F", 1), 7: ("G", 0), 8: ("G", 1), 9: ("A", 0), 10: ("A", 1), 11: ("B", 0)}
_TYPES = {4: ("whole", 0), 3: ("half", 1), 2: ("half", 0), 1.5: ("quarter", 1),
          1: ("quarter", 0), 0.75: ("eighth", 1), 0.5: ("eighth", 0), 0.25: ("16th", 0)}


def _pitch_xml(midi: int) -> str:
    step, alter = _STEPS[midi % 12]
    octave = midi // 12 - 1
    a = f"<alter>{alter}</alter>" if alter else ""
    return f"<pitch><step>{step}</step>{a}<octave>{octave}</octave></pitch>"


def _note_xml(pitch, beats: float, staff: int, voice: int) -> str:
    dur = int(round(beats * DIVISIONS))
    typ, dots = _TYPES.get(beats, (None, 0))
    extra = (f"<type>{typ}</type>" if typ else "") + "<dot/>" * dots
    tail = f"<duration>{dur}</duration><voice>{voice}</voice>{extra}<staff>{staff}</staff>"
    if pitch is None:
        return f"<note><rest/>{tail}</note>"
    pitches = pitch if isinstance(pitch, tuple) else (pitch,)
    out = []
    for i, p in enumerate(pitches):
        chord = "<chord/>" if i else ""
        out.append(f"<note>{chord}{_pitch_xml(p)}{tail}</note>")
    return "".join(out)


def _split_measures(events, beats_per_measure: int):
    measures, cur, acc = [], [], 0.0
    for pitch, beats in events:
        if acc + beats > beats_per_measure + 1e-9:
            raise ValueError("note crosses a barline; adjust the song layout")
        cur.append((pitch, beats))
        acc += beats
        if abs(acc - beats_per_measure) < 1e-9:
            measures.append(cur)
            cur, acc = [], 0.0
    if cur:
        cur.append((None, beats_per_measure - acc))
        measures.append(cur)
    return measures


def song_to_musicxml(song: Song, repeat: bool = False) -> str:
    rh = _split_measures(song.rh, song.beats_per_measure)
    lh = _split_measures(song.lh, song.beats_per_measure) if song.lh else []
    n = max(len(rh), len(lh))
    staves = 2 if lh else 1
    measure_div = song.beats_per_measure * DIVISIONS
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<score-partwise version="3.1"><part-list><score-part id="P1">'
           "<part-name>Piano</part-name></score-part></part-list><part id=\"P1\">"]
    for m in range(n):
        out.append(f'<measure number="{m + 1}">')
        if m == 0:
            clefs = '<clef number="1"><sign>G</sign><line>2</line></clef>'
            if staves == 2:
                clefs += '<clef number="2"><sign>F</sign><line>4</line></clef>'
            out.append(
                f"<attributes><divisions>{DIVISIONS}</divisions><key><fifths>0</fifths></key>"
                f"<time><beats>{song.beats_per_measure}</beats><beat-type>4</beat-type></time>"
                f"<staves>{staves}</staves>{clefs}</attributes>"
                f'<direction placement="above"><direction-type><metronome>'
                f"<beat-unit>quarter</beat-unit><per-minute>{song.bpm:g}</per-minute>"
                f'</metronome></direction-type><sound tempo="{song.bpm:g}"/></direction>'
            )
        if repeat and m == n - 1:
            pass  # the repeat barline is appended after the notes
        for p, b in (rh[m] if m < len(rh) else [(None, song.beats_per_measure)]):
            out.append(_note_xml(p, b, 1, 1))
        if staves == 2:
            out.append(f"<backup><duration>{measure_div}</duration></backup>")
            for p, b in (lh[m] if m < len(lh) else [(None, song.beats_per_measure)]):
                out.append(_note_xml(p, b, 2, 2))
        if repeat and m == n - 1:
            out.append('<barline location="right"><bar-style>light-heavy</bar-style>'
                       '<repeat direction="backward"/></barline>')
        out.append("</measure>")
    out.append("</part></score-partwise>")
    return "\n".join(out)


def write_musicxml(song: Song, path: str | Path, repeat: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(song_to_musicxml(song, repeat=repeat), encoding="utf-8")
    return path
