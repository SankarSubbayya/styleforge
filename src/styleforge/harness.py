"""Track 2 submission entrypoint (Participant Guide contract).

Reads /input/tasks.json:  [{"task_id", "video_url", "styles": [...]}]
Writes /output/results.json: [{"task_id", "captions": {style: caption}}]

Survival-first design: the organizer's 10-minute wall can kill this process at any
moment, so a schema-valid results.json (fallback captions) is written IMMEDIATELY on
startup and atomically upgraded as each task completes. Nothing that happens later —
malformed tasks, dead URLs, API outages, SIGKILL — can leave us with no output.
"""

import json
import os
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from . import config, pipeline

INPUT_PATH = Path(os.getenv("HARNESS_INPUT", "/input/tasks.json"))
OUTPUT_PATH = Path(os.getenv("HARNESS_OUTPUT", "/output/results.json"))
TIME_BUDGET_SEC = int(os.getenv("TIME_BUDGET_SEC", "540"))
MAX_WORKERS = int(os.getenv("HARNESS_WORKERS", "4"))
BON_K = int(os.getenv("HARNESS_K", "3"))
# Switch to single-candidate baseline when a task's share of remaining time is thin.
DEGRADE_THRESHOLD_SEC = int(os.getenv("DEGRADE_THRESHOLD_SEC", "75"))
DOWNLOAD_MAX_SEC = int(os.getenv("DOWNLOAD_MAX_SEC", "150"))
GENERIC_CAPTION = "A short video clip."


def _log(msg: str) -> None:
    print(f"[styleforge] {msg}", file=sys.stderr, flush=True)


def _requested_styles(task: dict) -> list[str]:
    raw = task.get("styles") or list(config.STYLES)
    return [str(s) for s in raw] if isinstance(raw, list) else list(config.STYLES)


def _fallback_record(task) -> dict:
    """Schema-valid record usable even for garbage tasks — every requested style
    (known or not) gets a caption; unknown styles get the generic one."""
    task_id, styles = "unknown", list(config.STYLES)
    if isinstance(task, dict):
        task_id = str(task.get("task_id", "unknown"))
        styles = _requested_styles(task)
    return {
        "task_id": task_id,
        "captions": {
            s: config.FALLBACK_CAPTIONS.get(s, GENERIC_CAPTION) for s in styles
        },
    }


def _write_results(results: list[dict]) -> None:
    """Atomic write: the file is always either the old valid JSON or the new one."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(OUTPUT_PATH)


def _download(url: str, dest: Path, deadline: float) -> Path:
    start = time.time()
    hard_stop = min(deadline, start + DOWNLOAD_MAX_SEC)
    req = urllib.request.Request(url, headers={"User-Agent": "styleforge/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            if time.time() > hard_stop:  # trickling server: per-read timeout never fires
                raise TimeoutError(f"download exceeded budget: {url}")
    return dest


def _process(task: dict, tmpdir: Path, deadline: float) -> dict:
    record = _fallback_record(task)  # never raises; baseline result
    try:
        task_id = record["task_id"]
        known = [s for s in record["captions"] if s in config.STYLES]
        if not known:
            return record
        clip = tmpdir / f"{task_id}.mp4"
        try:
            _download(task["video_url"], clip, deadline)
        except Exception as e:  # noqa: BLE001 — one retry if time remains
            if time.time() + 30 > deadline:
                raise
            _log(f"task {task_id} download retry after: {e}")
            _download(task["video_url"], clip, deadline)
        remaining = deadline - time.time()
        mode = "bon" if remaining > DEGRADE_THRESHOLD_SEC else "baseline"
        result = pipeline.caption_clip(clip, mode=mode, k=BON_K, styles=known)
        for style in known:
            entry = result["captions"].get(style)
            if entry and entry.get("caption"):
                record["captions"][style] = entry["caption"]
    except Exception as e:  # noqa: BLE001 — a task must never sink the run
        _log(f"task {record['task_id']} failed, shipping fallbacks: {e}")
    return record


def main() -> None:
    start = time.time()
    deadline = start + TIME_BUDGET_SEC

    try:
        tasks = json.loads(INPUT_PATH.read_text())
        if not isinstance(tasks, list):
            raise ValueError("tasks.json is not a list")
    except Exception as e:  # noqa: BLE001 — even unparseable input gets valid output
        _log(f"could not read tasks.json ({e}); writing empty results")
        tasks = []

    # A valid results.json exists from this moment on.
    results = [_fallback_record(t) for t in tasks]
    _write_results(results)
    _log(f"{len(tasks)} task(s), budget {TIME_BUDGET_SEC}s, fallbacks pre-written")

    if tasks:
        tmpdir = Path(tempfile.mkdtemp())
        executor = ThreadPoolExecutor(max_workers=max(1, min(MAX_WORKERS, len(tasks))))
        futures = {
            executor.submit(_process, t, tmpdir, deadline): i
            for i, t in enumerate(tasks)
        }
        pending = set(futures)
        while pending and time.time() < deadline:
            done, pending = wait(
                pending, timeout=max(1.0, deadline - time.time()),
                return_when=FIRST_COMPLETED,
            )
            for fut in done:
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:  # noqa: BLE001
                    _log(f"task index {idx} raised: {e}")
                _write_results(results)  # upgrade incrementally
        if pending:
            _log(f"{len(pending)} task(s) unfinished at deadline; fallbacks stand")
        executor.shutdown(wait=False, cancel_futures=True)

    _write_results(results)
    _log(f"final results written in {time.time() - start:.0f}s")
    # Worker threads may still be running; skip interpreter thread-joins entirely.
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
