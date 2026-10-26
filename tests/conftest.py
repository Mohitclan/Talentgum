import pytest

from ras.core.score.reference import load_reference
from ras.synth.musicxml import write_musicxml
from ras.synth.songs import SONGS


@pytest.fixture(scope="session")
def song_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("songs")
    for sid, song in SONGS.items():
        write_musicxml(song, d / f"{sid}.musicxml")
    write_musicxml(SONGS["mary"], d / "mary_repeat.musicxml", repeat=True)
    return d


@pytest.fixture(scope="session")
def twinkle(song_dir):
    return load_reference(song_dir / "twinkle.musicxml")


@pytest.fixture(scope="session")
def frere(song_dir):
    return load_reference(song_dir / "frere_jacques.musicxml")
