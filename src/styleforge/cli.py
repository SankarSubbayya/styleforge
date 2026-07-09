import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import config, ingest, pipeline

app = typer.Typer(help="StyleForge — four-style video captioner (AMD Hackathon ACT II, Track 2)")
console = Console()


@app.command()
def caption(
    clip: Path = typer.Argument(..., exists=True, help="Video clip (30s-2min)"),
    mode: str = typer.Option("bon", help="baseline | bon | tuned | tuned+bon"),
    k: int = typer.Option(pipeline.DEFAULT_K, help="candidates per style for BoN modes"),
    out: Path = typer.Option(None, help="output JSON path"),
):
    """Caption CLIP in all four styles."""
    result = pipeline.caption_clip(clip, mode=mode, k=k)
    path = pipeline.save_result(result, out)

    table = Table(title=f"{clip.name} — mode={mode}")
    table.add_column("style", style="bold")
    table.add_column("caption")
    table.add_column("score")
    for style, entry in result["captions"].items():
        score = entry.get("score")
        score_txt = f"{score['overall']}" if score else "-"
        flag = " [red](fallback)[/red]" if entry.get("fallback") else ""
        table.add_row(style, escape(entry["caption"]) + flag, score_txt)
    console.print(table)
    console.print(f"[dim]{result['cost']} | saved -> {path}[/dim]")


@app.command()
def probe(clip: Path = typer.Argument(..., exists=True)):
    """Debug: show clip metadata, frame sampling, transcript."""
    info = ingest.probe(clip)
    frames = ingest.extract_frames(clip, info)
    transcript = ingest.transcribe(clip, info)
    console.print_json(
        json.dumps(
            {
                "duration_sec": info.duration,
                "has_audio": info.has_audio,
                "n_frames": len(frames),
                "frame_times": [round(t, 1) for t, _ in frames],
                "transcript": transcript or "[none]",
            }
        )
    )


@app.command()
def harness():
    """Submission entrypoint: /input/tasks.json -> /output/results.json (Participant Guide)."""
    from . import harness as h

    h.main()


@app.command("config")
def show_config():
    """Show active model configuration."""
    console.print_json(
        json.dumps(
            {
                "base_url": config.FIREWORKS_BASE_URL,
                "api_key_set": bool(config.FIREWORKS_API_KEY),
                "mock": config.MOCK,
                "perception_model": config.PERCEPTION_MODEL,
                "stylizer_model": config.STYLIZER_MODEL,
                "judge_model": config.JUDGE_MODEL,
                "whisper_model": config.WHISPER_MODEL,
                "max_frames": config.MAX_FRAMES,
                "styles": list(config.STYLES),
            }
        )
    )
