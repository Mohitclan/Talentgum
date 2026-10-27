"""Bulk-load client material: songs, a rubric and recordings, queued for the worker.

Manifest (JSON), paths relative to the manifest file:
  {"rubric": "rubric.json",
   "songs": [{"song_id": "twinkle", "title": "Twinkle", "reference": "refs/twinkle.mxl", "bpm": 100}],
   "recordings": [{"song_id": "twinkle", "audio": "audio/s01.m4a", "student_id": "s01",
                   "start_s": 2.4, "external_ref": "sub-17"}]}
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from . import services as svc
from .db import DEFAULT_TENANT, Song
from .storage import ObjectStore


def ingest_manifest(db: Session, store: ObjectStore, manifest_path: str | Path,
                    tenant: str = DEFAULT_TENANT) -> dict:
    mp = Path(manifest_path)
    m = json.loads(mp.read_text())
    root = mp.parent
    out = {"songs": 0, "references_added": 0, "rubric": None, "queued": [], "errors": []}

    if m.get("rubric"):
        row = svc.create_rubric(db, tenant, json.loads((root / m["rubric"]).read_text()))
        out["rubric"] = f"{row.rubric_id}@{row.version}"
    for s in m.get("songs", []):
        try:
            ref = root / s["reference"]
            f = svc.save_file(db, store, tenant, ref.name, ref.read_bytes(), "reference")
            exists = db.query(Song).filter_by(tenant_id=tenant, song_id=s["song_id"]).first()
            if exists:
                svc.add_reference(db, store, tenant, s["song_id"], f.id, s.get("bpm"))
                out["references_added"] += 1
            else:
                svc.create_song(db, store, tenant, s["song_id"], s.get("title", s["song_id"]), f.id, s.get("bpm"))
                out["songs"] += 1
        except Exception as exc:
            out["errors"].append(f"song {s.get('song_id')}: {exc}")
    rubric_id = (m.get("rubric_id") or (out["rubric"] or "").split("@")[0]) or None
    if rubric_id is None:
        out["errors"].append("no rubric given: add \"rubric\" (a file) or \"rubric_id\" (already stored)")
        return out
    for r in m.get("recordings", []):
        try:
            audio = root / r["audio"]
            f = svc.save_file(db, store, tenant, audio.name, audio.read_bytes(), "audio")
            a, _ = svc.create_assessment(db, tenant, {
                "song_id": r["song_id"], "recording_file_id": f.id, "rubric_id": rubric_id,
                "student_id": r.get("student_id"), "start_s": r.get("start_s"),
                "external_ref": r.get("external_ref")})
            out["queued"].append(a.id)
        except Exception as exc:
            out["errors"].append(f"recording {r.get('audio')}: {exc}")
    return out
