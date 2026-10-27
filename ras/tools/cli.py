"""`ras` command line: assess one recording, generate synthetic data, benchmark models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ras.core.rubric import load_rubric

app = typer.Typer(help="Recording Assessment System", no_args_is_help=True)
console = Console()
DEFAULT_RUBRIC = Path(__file__).resolve().parents[2] / "data" / "rubrics" / "dev_placeholder.json"


@app.command()
def assess(
    reference: Path = typer.Argument(..., exists=True, help="MusicXML/MXL or MIDI reference"),
    audio: Path = typer.Argument(..., exists=True, help="Student recording (any ffmpeg format)"),
    rubric: Path = typer.Option(DEFAULT_RUBRIC, help="Rubric JSON"),
    start_s: Optional[float] = typer.Option(None, help="Start timestamp in seconds"),
    transcriber: str = typer.Option("transkun", help="transkun | hft | kong | basic-pitch"),
    profile: Optional[Path] = typer.Option(None, exists=True, help="Teacher benchmark profile (ras profile build)"),
    out: Optional[Path] = typer.Option(None, help="Write the full result JSON here"),
    midi_out: Optional[Path] = typer.Option(None, help="Write the transcribed performance as MIDI"),
):
    """Assess one recording against a reference and print a summary."""
    from ras.core.profile import load_profile
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
        res = run_assess(reference, audio, load_rubric(rubric), tr, start_s=start_s,
                         profile=load_profile(profile) if profile else None)

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
        crit = Table(title="Criteria")
        for col in ("criterion", "metric", "value", "score", "note"):
            crit.add_column(col)
        for cid, c in res.scoring.get("criteria", {}).items():
            crit.add_row(cid, c["metric"], "" if c["value"] is None else f"{c['value']:.1f}",
                         "-" if c["score"] is None else f"{c['score']:.0f}", c.get("reason", ""))
        console.print(crit)
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
    data_dir: Path = typer.Option(Path("data/service"), envvar="RAS_DATA_DIR",
                                  help="Where the database and files live"),
    transcriber: str = typer.Option("transkun", envvar="RAS_TRANSCRIBER", help="transkun | basic-pitch"),
    with_worker: bool = typer.Option(True, envvar="RAS_WITH_WORKER",
                                     help="Run a worker thread inside the server (simplest locally)"),
):
    """Start the API and web UI (http://HOST:PORT/ui/). Add --no-with-worker and run `ras worker` separately."""
    import threading

    import uvicorn

    from ras.service.api import create_app
    from ras.service.config import Settings
    from ras.service.worker import Worker

    settings = Settings(data_dir=data_dir, transcriber=transcriber)
    _setup_logging(settings)
    app = create_app(settings)
    if with_worker:
        worker = Worker(settings, session_factory=app.state.Session, store=app.state.store, signal=app.state.signal)
        threading.Thread(target=worker.run_forever, daemon=True, name="ras-worker").start()
    uvicorn.run(app, host=host, port=port, log_level="info", log_config=None if settings.log_format == "json" else
                uvicorn.config.LOGGING_CONFIG, proxy_headers=True, forwarded_allow_ips="*")


@app.command()
def worker(
    data_dir: Path = typer.Option(Path("data/service"), envvar="RAS_DATA_DIR"),
    transcriber: str = typer.Option("transkun", envvar="RAS_TRANSCRIBER"),
):
    """Run a worker process that handles queued assessments. Stops cleanly on SIGTERM."""
    from ras.service.config import Settings
    from ras.service.worker import Worker

    settings = Settings(data_dir=data_dir, transcriber=transcriber)
    _setup_logging(settings)
    Worker(settings).run_forever(install_signal_handlers=True)


def _setup_logging(settings) -> None:
    import logging

    if settings.log_format != "json":
        logging.basicConfig(level=logging.INFO)
        return

    class JsonFormatter(logging.Formatter):  # one JSON object per line, as CloudWatch Logs Insights expects
        def format(self, r):
            d = {"level": r.levelname, "logger": r.name, "message": r.getMessage(),
                 "time": self.formatTime(r, "%Y-%m-%dT%H:%M:%S")}
            if r.exc_info:
                d["exc"] = self.formatException(r.exc_info)
            return json.dumps(d)

    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[h], force=True)


@app.command()
def ingest(
    manifest: Path = typer.Argument(..., exists=True, help="Client manifest JSON (see ras/service/ingest.py)"),
    data_dir: Path = typer.Option(Path("data/service"), envvar="RAS_DATA_DIR"),
    tenant: str = typer.Option("local", help="Tenant that owns the data"),
):
    """Load songs, a rubric and recordings from a manifest and queue them for the worker."""
    from ras.service.config import Settings
    from ras.service.db import make_session_factory
    from ras.service.ingest import ingest_manifest
    from ras.service.queue import make_signal
    from ras.service.storage import make_store

    settings = Settings(data_dir=data_dir)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Session = make_session_factory(settings.database_url)
    with Session() as db:
        out = ingest_manifest(db, make_store(settings), manifest, tenant)
    signal = make_signal(settings)
    for asm_id in out["queued"]:
        signal.publish(asm_id)
    console.print(f"songs {out['songs']} (+{out['references_added']} versions), rubric {out['rubric']}, "
                  f"{len(out['queued'])} recordings queued")
    for e in out["errors"]:
        console.print(f"[red]{e}[/red]")
    console.print("Start `ras serve` (or `ras worker`) to process the queue.")


@app.command()
def evaluate(
    data_dir: Path = typer.Option(Path("data/service"), envvar="RAS_DATA_DIR"),
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
    data_dir: Path = typer.Option(Path("data/service"), envvar="RAS_DATA_DIR"),
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
    data_dir: Path = typer.Option(Path("data/service"), envvar="RAS_DATA_DIR"),
):
    """Issue another API key for a tenant (for rotation). Shown only once."""
    from ras.service import services as svc

    with _session(data_dir)() as db:
        console.print(svc.create_api_key(db, tenant_id, label))


@app.command("key-revoke")
def key_revoke(
    prefix: str = typer.Argument(..., help="The first 12 characters of the key"),
    data_dir: Path = typer.Option(Path("data/service"), envvar="RAS_DATA_DIR"),
):
    """Revoke an API key immediately."""
    from ras.service import services as svc

    with _session(data_dir)() as db:
        n = svc.revoke_api_key(db, prefix)
    console.print(f"{n} key(s) revoked")


@app.command("purge-audio")
def purge_audio(
    older_than_days: int = typer.Option(..., help="Delete recordings of assessments finished more than N days ago"),
    data_dir: Path = typer.Option(Path("data/service"), envvar="RAS_DATA_DIR"),
    tenant: Optional[str] = typer.Option(None, help="Only this tenant (default: all)"),
):
    """Retention: delete raw recordings after the retention period. Results and reviews are kept."""
    from ras.service import services as svc
    from ras.service.config import Settings
    from ras.service.storage import make_store

    with _session(data_dir)() as db:
        n = svc.purge_audio(db, make_store(Settings(data_dir=data_dir)), older_than_days, tenant)
    console.print(f"{n} recording(s) deleted")


models_app = typer.Typer(help="Optional open-source models (second transcriber, pedal, dynamics)",
                         no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list():
    """What each model does, its licence, which criteria use it, and whether it is downloaded."""
    from ras.models import models_dir, status

    t = Table(title=f"Models in {models_dir()}")
    for col in ("name", "available", "MB", "licence", "criteria", "role"):
        t.add_column(col)
    for s in status():
        t.add_row(s["name"], "yes" if s["available"] else "no", str(s["size_mb"]), s["licence"],
                  ", ".join(s["criteria"]), s["role"])
    console.print(t)


@models_app.command("download")
def models_download(
    names: list[str] = typer.Argument(None, help="hft, kong, dynamics (default: all)"),
):
    """Download, checksum-verify and convert model weights into RAS_MODELS_DIR (default ./models)."""
    from ras.models import MODELS, download

    for name in names or list(MODELS):
        if name not in MODELS:
            raise typer.BadParameter(f"unknown model {name!r}; available: {', '.join(MODELS)}")
        with console.status(f"Downloading {name} (~{MODELS[name].size_mb} MB) ..."):
            path = download(name)
        console.print(f"{name}: {path}")


@app.command()
def criteria():
    """TalentGum's criteria, the metric that scores each, the models it uses and what limits it."""
    from ras.core.criteria import CRITERIA

    t = Table(title="Client criteria -> measurement")
    for col in ("group", "criterion", "metric (fallback)", "models", "readiness", "limits"):
        t.add_column(col)
    for c in CRITERIA:
        t.add_row(c.group, c.label, c.metric + (f" ({c.fallback})" if c.fallback else ""), ", ".join(c.models),
                  ", ".join(c.readiness), c.limits)
    console.print(t)


profile_app = typer.Typer(help="Teacher benchmark profiles per song", no_args_is_help=True)
app.add_typer(profile_app, name="profile")


@profile_app.command("build")
def profile_build(
    reference: Path = typer.Argument(..., exists=True, help="The song's MusicXML/MIDI"),
    audio: list[Path] = typer.Argument(..., exists=True, help="Two or more teacher recordings of the song"),
    out: Path = typer.Option(..., help="Profile JSON to write"),
    rubric: Path = typer.Option(DEFAULT_RUBRIC.parent / "talentgum_criteria_v1.json"),
    transcriber: str = typer.Option("transkun"),
):
    """Assess the teachers' recordings and pool them into the profile students are compared with."""
    from ras.core.profile import build_profile
    from ras.core.score.reference import load_reference
    from ras.core.transcribe import get_transcriber
    from ras.pipeline import assess as run_assess

    ref, rb, tr = load_reference(reference), load_rubric(rubric), get_transcriber(transcriber)
    inputs = []
    for a in audio:
        with console.status(f"Assessing {a.name} ..."):
            res = run_assess(ref, a, rb, tr)
        if res.status != "completed":
            console.print(f"[red]{a.name}: {res.error_code}[/red] (skipped)")
            continue
        inputs.append(res.scoring_inputs)
        console.print(f"{a.name}: {res.summary.get('correct')}/{res.summary.get('score_notes')} notes matched, "
                      f"{res.summary.get('mean_bpm') or 0:.0f} bpm")
    if not inputs:
        raise typer.Exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_profile(inputs, ref.sha256), indent=1))
    console.print(f"Profile from {len(inputs)} recording(s) written to {out}")


@app.command()
def calibrate(
    grades: Path = typer.Argument(..., exists=True, help="CSV: result, <criterion ids...>, overall"),
    rubric: Path = typer.Option(..., exists=True, help="Rubric to start from"),
    out: Path = typer.Option(..., help="Calibrated rubric JSON to write"),
    scale: float = typer.Option(100.0, help="Grade scale in the CSV (100, or 5 for 1-5 grades)"),
):
    """Fit each criterion's thresholds (and, with overall grades, the weights) to teacher grades."""
    from ras.eval.calibrate import calibrate as run_cal

    new, report = run_cal(load_rubric(rubric), grades, scale)
    out.write_text(json.dumps(new, indent=1))
    for cid, info in report["criteria"].items():
        console.print(f"  {cid}: {info}")
    console.print(f"weights: {report['weights']}. Rubric v{new['version']} written to {out}")


finetune_app = typer.Typer(help="Fine-tune Transkun on TalentGum recordings (GPU)", no_args_is_help=True)
app.add_typer(finetune_app, name="finetune")


@finetune_app.command("prepare")
def finetune_prepare(
    items: Path = typer.Argument(..., exists=True,
                                 help='JSON list: [{"reference", "audio", "start_s", "split", "removed_ref_indices"}]'),
    out: Path = typer.Option(Path("data/finetune"), help="Dataset directory"),
    n_iter: int = typer.Option(20000, help="Training iterations"),
    max_lr: float = typer.Option(2e-5, help="Peak learning rate (small: this continues a trained model)"),
):
    """Label teacher/reviewed recordings via score alignment, write Transkun's dataset and a start checkpoint."""
    from ras.core.transcribe import get_transcriber
    from ras.eval import finetune as ft

    base = Path(items).parent
    entries = json.loads(Path(items).read_text())
    for e in entries:
        for k in ("reference", "audio"):
            e[k] = str((base / e[k]) if not Path(e[k]).is_absolute() else e[k])
    rb = load_rubric(DEFAULT_RUBRIC)
    with console.status("Aligning and labelling ..."):
        summary = ft.prepare(entries, out, get_transcriber("transkun"), rb)
    console.print(summary)
    if not summary["train"] or not summary["validation"]:
        console.print("[yellow]Need at least one train and one validation recording.[/yellow]")
        raise typer.Exit(1)
    ft.build_pickles(out)
    ckpt = ft.init_checkpoint(out / "checkpoint.pt", n_iter, max_lr)
    console.print("Train on a GPU machine with:")
    console.print(ft.train_command(out, ckpt, n_iter, max_lr))
    console.print(f"Then: ras finetune export {ckpt} --out transkun_talentgum.pt; set RAS_TRANSKUN_WEIGHT to it.")


@finetune_app.command("export")
def finetune_export(
    checkpoint: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(..., help="Weights file for RAS_TRANSKUN_WEIGHT"),
):
    """Turn the training checkpoint into weights the Transkun adapter loads."""
    from ras.eval.finetune import export_weights

    console.print(f"Written {export_weights(checkpoint, out)}. Benchmark it before switching: "
                  "ras benchmark --models transkun (with and without RAS_TRANSKUN_WEIGHT).")


@app.command()
def sbom(
    out_dir: Path = typer.Option(Path("docs/compliance"), help="Where to write the register"),
    name: str = typer.Option("local", help="Label for this environment, e.g. image or local"),
):
    """Licence register and CycloneDX SBOM of the installed packages and bundled models."""
    from ras.tools.sbom import write_register

    md, cdx, flagged = write_register(out_dir, name)
    console.print(f"Wrote {md} and {cdx}")
    for f in flagged:
        console.print(f"[yellow]review licence:[/yellow] {f}")


if __name__ == "__main__":
    app()
