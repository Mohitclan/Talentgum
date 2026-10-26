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
):
    """Compare transcribers: note F1, error-detection precision/recall, CPU time, peak RAM."""
    from ras.eval.benchmark import run_benchmark

    path = run_benchmark(data, [m.strip() for m in models.split(",")], out, limit, log=console.print)
    console.print((path / "report.md").read_text())
    console.print(f"Full results in {path}")


if __name__ == "__main__":
    app()
