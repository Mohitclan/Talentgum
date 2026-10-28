"""Score the splitter against a teacher's labels for the same class recording."""

from __future__ import annotations

import numpy as np


def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def evaluate(annotation: dict, labels: list[dict], min_iou: float = 0.5) -> dict:
    """`labels`: [{"song_id", "start_s", "end_s"}]. A label is found when a clip overlaps it by at least
    `min_iou` (time intersection over union) and names the same song."""
    clips = annotation["clips"]
    pairs = sorted(((_iou(c["start_s"], c["end_s"], lab["start_s"], lab["end_s"]), i, j)
                    for i, c in enumerate(clips) for j, lab in enumerate(labels)), reverse=True)
    used_c, used_l, matched = set(), set(), []
    for iou, i, j in pairs:
        if iou < min_iou:
            break
        if i not in used_c and j not in used_l:
            used_c.add(i)
            used_l.add(j)
            matched.append((i, j))
    found = [(i, j) for i, j in matched if clips[i]["song_id"] == labels[j]["song_id"]]
    wrong = [(i, j) for i, j in matched if clips[i]["song_id"] != labels[j]["song_id"]]
    hours = annotation.get("duration_s", 0) / 3600
    start_err = [abs(clips[i]["start_s"] - labels[j]["start_s"]) for i, j in found]
    end_err = [abs(clips[i]["end_s"] - labels[j]["end_s"]) for i, j in found]
    false_clips = len(clips) - len(matched)
    return {
        "labelled": len(labels),
        "found": len(found),
        "found_pct": round(100 * len(found) / len(labels), 1) if labels else None,
        "wrong_song": len(wrong),
        "missed": len(labels) - len(matched),
        "false_clips": false_clips,
        "false_clips_per_hour": round(false_clips / hours, 2) if hours else None,
        "median_start_error_s": round(float(np.median(start_err)), 2) if start_err else None,
        "median_end_error_s": round(float(np.median(end_err)), 2) if end_err else None,
    }
