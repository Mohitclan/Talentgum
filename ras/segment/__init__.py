"""Cut long class recordings into one clip per song attempt (docs/CLASS_RECORDINGS.md)."""

from .classfile import annotate_class, load_songs, merge_chunks, student_id, transcribe_long
from .evaluate import evaluate
from .find import SegmentConfig, SongModel, find_clips, song_model

__all__ = ["SegmentConfig", "SongModel", "annotate_class", "evaluate", "find_clips", "load_songs", "merge_chunks",
           "song_model", "student_id", "transcribe_long"]
