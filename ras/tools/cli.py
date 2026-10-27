"""`ras` command line: assess one recording, generate synthetic data, benchmark models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ras.core.rubric import load_rubric

app = typer.Typer(help="Recording Assessment System (Phase 1: local CLI)", no_args_is_help=True)
console = Console()
DEFAULT_RUBRIC = Path(__file__).resolve().parents[2] / "data" / "rubrics" / "dev_placeholder.json"


@app.command()
def assess(
    reference: Path = typer.Argument(..., exists=True, help="MusicXML/MXL or MIDI reference"),
    audio: Path = typer.Argument(..., exists=True, help="Student recording (any ffmpeg format)"),
    rubric: Path = typer.Option(DEFAULT_RUBRIC, help="Rubric JSON"),
    start_s: Optional[float] = typer.Option(None, help="Start timestamp in seconds"),
    transcriber: str = typer.Option("transkun", help="transkun | basic-pitch"),
    out: Optional[Path] = typer.Option(None, help="Write the full result JSON here"),
    midi_out: Optional[Path] = typer.Option(None, help="Write the transcribed performance as MIDI"),
):
    """Assess one recording against a reference and print a summary."""
    from ras.core.transcribe import get_transcriber
    from ras.pipeline import assess as run_assess

    tr = get_transcriber(transcriber)
    holder = {}
    if midi_out:
        inner = tr

        class _Keep:
            name, sample_rate = inner.name, getattr(inner, "sample_rate", 44100)

            def transcribe(self, a, sr):
                holder["t"] = inner.transcribe(a, sr)
                return holder["t"]

        tr = _Keep()
    with console.status("Assessing ..."):
        res = run_assess(reference, audio, load_rubric(rubric), tr, start_s=start_s)

    if res.status != "completed" and not res.summary:
        console.print(f"[red]{res.status}[/red]: {res.error_code} {res.error_detail or ''}")
    else:
        s = res.summary
        table = Table(title=f"{audio.name} vs {reference.name}")
        table.add_column("")
        table.add_column("value")
        for k in ("score_notes", "correct", "wrong_pitch", "wrong_octave", "missing", "extra",
                  "mean_bpm", "pause_count", "onset_dev_median_abs_ms"):
            v = s.get(k)
            table.add_row(k, f"{v:.1f}" if isinstance(v, float) else str(v))
        console.print(table)
        console.print(f"Total score: [bold]{res.scoring.get('total_score')}[/bold] "
                      f"({res.scoring.get('band')})   confidence {res.confidence['value']}"
                      f"{'  [yellow]NEEDS REVIEW[/yellow]' if res.confidence['needs_review'] else ''}")
        for line in res.feedback:
            console.print(f"  - {line}")
        if res.warnings:
            console.print(f"[dim]warnings: {', '.join(res.warnings)}[/dim]")
        console.print(f"[dim]time: {res.timings_s}[/dim]")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res.model_dump(mode="json"), indent=1))
        console.print(f"Result written to {out}")
    if midi_out and "t" in holder:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        for n in holder["t"].notes:
            inst.notes.append(pretty_midi.Note(n.velocity, n.pitch, n.onset, n.offset))
        pm.instruments.append(inst)
        midi_out.parent.mkdir(parents=True, exist_ok=True)
        pm.write(str(midi_out))
        console.print(f"MIDI written to {midi_out}")


@app.command()
def synth(
    out: Path = typer.Option(Path("data/synth"), help="Output directory"),
    n: int = typer.Option(20, help="Number of recordings"),
    seed: int = typer.Option(0),
    fmt: str = typer.Option("m4a", help="m4a (AAC, like the client) or wav"),
):
    """Generate synthetic songs, student performances with known mistakes, and ground truth."""
    from ras.synth.dataset import generate_dataset

    with console.status("Rendering ..."):
        path = generate_dataset(out, n=n, seed=seed, fmt=fmt)
    console.print(f"Dataset written: {path}")


@app.command()
def benchmark(
    data: Path = typer.Option(Path("data/synth"), help="Dataset made by `ras synth`"),
    models: str = typer.Option("ground-truth,transkun", help="Comma-separated model names"),
    limit: Optional[int] = typer.Option(None, help="Use only the first N recordings"),
    out: Optional[Path] = typer.Option(None, help="Output directory (default data/runs/bench_<time>)"),
    cache: Optional[Path] = typer.Option(None, help="Transcription cache directory (skips re-running models)"),
):
    """Compare transcribers: note F1, error-detection precision/recall, CPU time, peak RAM."""
    from ras.eval.benchmark import run_benchmark

    path = run_benchmark(data, [m.strip() for m in models.split(",")], out, limit, log=console.print,
                         cache_dir=cache)
    console.print((path / "report.md").read_text())
    console.print(f"Full results in {path}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    data_dir: Path = typer.Option(Path("data/service"), help="Where the database and files live"),
    transcriber: str = typer.Option("transkun", help="transkun | basic-pitch"),
    with_worker: bool = typer.Option(True, help="Run a worker thread inside the server (simplest locally)"),
):
    """Start the API and web UI (http://HOST:PORT/ui/). Add --no-with-worker and run `ras worker` separately."""
    import threading

    import uvicorn

    from ras.service.api import create_app
    from ras.service.config import Settings
    from ras.service.worker import Worker

    settings = Settings(data_dir=data_dir, transcriber=transcriber)
    app = create_app(settings)
    if with_worker:
        worker = Worker(settings, session_factory=app.state.Session, store=app.state.store)
        threading.Thread(target=worker.run_forever, daemon=True, name="ras-worker").start()
    uvicorn.run(app, host=host, port=port, log_level="info")


@app.command()
def worker(
    data_dir: Path = typer.Option(Path("data/service")),
    transcriber: str = typer.Option("transkun"),
):
    """Run a worker process that handles queued assessments."""
    import logging

    from ras.service.config import Settings
    from ras.service.worker import Worker

    logging.basicConfig(level=logging.INFO)
    Worker(Settings(data_dir=data_dir, transcriber=transcriber)).run_forever()


@app.command()
def ingest(
    manifest: Path = typer.Argument(..., exists=True, help="Client manifest JSON (see ras/service/ingest.py)"),
    data_dir: Path = typer.Option(Path("data/service")),
    tenant: str = typer.Option("local", help="Tenant that owns the data"),
):
    """Load songs, a rubric and recordings from a manifest and queue them for the worker."""
    from ras.service.config import Settings
    from ras.service.db import make_session_factory
    from ras.service.ingest import ingest_manifest
    from ras.service.storage import LocalStore

    settings = Settings(data_dir=data_dir)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Session = make_session_factory(settings.database_url)
    with Session() as db:
        out = ingest_manifest(db, LocalStore(settings.data_dir / "objects"), manifest, tenant)
    console.print(f"songs {out['songs']} (+{out['references_added']} versions), rubric {out['rubric']}, "
                  f"{len(out['queued'])} recordings queued")
    for e in out["errors"]:
        console.print(f"[red]{e}[/red]")
    console.print("Start `ras serve` (or `ras worker`) to process the queue.")


@app.command()
def evaluate(
    data_dir: Path = typer.Option(Path("data/service")),
    target_false_alarm: float = typer.Option(0.10, help="Highest acceptable share of wrong error claims"),
    out: Optional[Path] = typer.Option(None, help="Write the full JSON report here"),
    tenant: str = typer.Option("local"),
):
    """Compare results with teacher reviews: error precision/recall, score agreement, confidence threshold."""
    from ras.service.config import Settings
    from ras.service.db import make_session_factory
    from ras.service.evaluation import evaluate as run_eval
    from ras.service.evaluation import load_cases

    Session = make_session_factory(Settings(data_dir=data_dir).database_url)
    with Session() as db:
        report = run_eval(load_cases(db, tenant), target_false_alarm)
    if not report["n_reviewed"]:
        console.print("No reviewed assessments yet. Teachers review in the web UI (Review queue).")
        return
    d = report["error_detection"]
    console.print(f"{report['n_reviewed']} reviewed. Error detection vs teachers: precision "
                  f"{d['precision']:.3f} {d['precision_ci']}, recall {d['recall']:.3f} {d['recall_ci']}")
    if "score_agreement" in report:
        console.print(f"Score vs teacher score: {report['score_agreement']}")
    c = report["confidence"]
    console.print(f"Recommended confidence threshold: {c['recommended_threshold']} "
                  f"(target false-alarm rate {c['target_false_alarm_rate']})")
    if out:
        out.write_text(json.dumps(report, indent=1, default=str))


def _session(data_dir: Path):
    from ras.service.config import Settings
    from ras.service.db import make_session_factory

    return make_session_factory(Settings(data_dir=data_dir).database_url)


@app.command("tenant-create")
def tenant_create(
    tenant_id: str = typer.Argument(..., help="Short id, e.g. talentgum"),
    name: str = typer.Option("", help="Display name"),
    data_dir: Path = typer.Option(Path("data/service")),
):
    """Create a tenant (client organisation) and print its first API key. The key is shown only once."""
    from ras.service import services as svc

    with _session(data_dir)() as db:
        svc.create_tenant(db, tenant_id, name or tenant_id)
        key = svc.create_api_key(db, tenant_id, "first key")
    console.print(f"Tenant [bold]{tenant_id}[/bold] created. API key (store it now, it is not shown again):")
    console.print(key)


@app.command("key-create")
def key_create(
    tenant_id: str = typer.Argument(...),
    label: str = typer.Option(""),
    data_dir: Path = typer.Option(Path("data/service")),
):
    """Issue another API key for a tenant (for rotation). Shown only once."""
    from ras.service import services as svc

    with _session(data_dir)() as db:
        console.print(svc.create_api_key(db, tenant_id, label))


@app.command("key-revoke")
def key_revoke(
    prefix: str = typer.Argument(..., help="The first 12 characters of the key"),
    data_dir: Path = typer.Option(Path("data/service")),
):
    """Revoke an API key immediately."""
    from ras.service import services as svc

    with _session(data_dir)() as db:
        n = svc.revoke_api_key(db, prefix)
    console.print(f"{n} key(s) revoked")


@app.command("purge-audio")
def purge_audio(
    older_than_days: int = typer.Option(..., help="Delete recordings of assessments finished more than N days ago"),
    data_dir: Path = typer.Option(Path("data/service")),
    tenant: Optional[str] = typer.Option(None, help="Only this tenant (default: all)"),
):
    """Retention: delete raw recordings after the retention period. Results and reviews are kept."""
    from ras.service import services as svc
    from ras.service.storage import LocalStore

    with _session(data_dir)() as db:
        n = svc.purge_audio(db, LocalStore(data_dir / "objects"), older_than_days, tenant)
    console.print(f"{n} recording(s) deleted")


if __name__ == "__main__":
    app()
