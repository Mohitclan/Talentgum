"""End-to-end smoke test of a running deployment (docker compose locally, or AWS staging).

Uploads a reference and a recording, assesses it with the real transcriber, waits for the result.
    python scripts/smoke_stack.py --url http://127.0.0.1:8088 --data data/synth --recording rec_001
Exit code 0 only if the assessment finishes as completed or review_required.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8088")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--data", type=Path, default=Path("data/synth"), help="dataset made by `ras synth`")
    ap.add_argument("--recording", default="rec_001")
    ap.add_argument("--rubric", type=Path, default=Path("data/rubrics/dev_placeholder.json"))
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()

    rec = next(r for r in json.loads((a.data / "manifest.json").read_text())["recordings"] if r["id"] == a.recording)
    c = httpx.Client(base_url=a.url, timeout=120, headers={"X-API-Key": a.api_key} if a.api_key else {})

    def ok(r: httpx.Response) -> dict:
        if r.status_code >= 300:
            sys.exit(f"{r.request.method} {r.request.url.path} -> {r.status_code}: {r.text}")
        return r.json()

    print("health:", ok(c.get("/v1/health")))

    def upload(path: Path, kind: str) -> str:
        return ok(c.post(f"/v1/files?kind={kind}", files={"file": (path.name, path.read_bytes())}))["file_id"]

    song_id = f"smoke-{rec['song_id']}"
    ref_id = upload(a.data / rec["reference"], "reference")
    if c.get(f"/v1/songs/{song_id}").status_code == 404:
        ok(c.post("/v1/songs", json={"song_id": song_id, "title": f"Smoke {rec['song_id']}",
                                     "reference_file_id": ref_id}))
    rubric = json.loads(a.rubric.read_text())
    ok(c.post("/v1/rubrics", json=rubric))
    audio_id = upload(a.data / rec["audio"], "audio")
    t0 = time.time()
    job = ok(c.post("/v1/assessments", json={"song_id": song_id, "recording_file_id": audio_id,
                                             "rubric_id": rubric["rubric_id"], "start_s": rec["start_s"],
                                             "student_id": "smoke-student", "external_ref": rec["id"]}))
    asm = job["assessment_id"]
    print(f"queued {asm} ({rec['duration_s']:.0f} s of audio)")
    while True:
        res = ok(c.get(f"/v1/assessments/{asm}"))
        if res["status"] not in ("queued", "processing"):
            break
        if time.time() - t0 > a.timeout:
            sys.exit(f"timed out; last status {res['status']}")
        time.sleep(3)
    took = time.time() - t0
    print(f"{res['status']} in {took:.0f} s: score {res.get('total_score')}, confidence {res.get('confidence')}")
    print("summary:", res.get("summary"))
    print("injected errors:", rec["injected"])
    if res.get("error_code"):
        print("error:", res["error_code"], res.get("error_detail"))
    return 0 if res["status"] in ("completed", "review_required") else 1


if __name__ == "__main__":
    sys.exit(main())
