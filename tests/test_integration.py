"""Integration tests — full pipeline and the exact harness contract the organizers
run, in mock mode (no API calls; ffmpeg + whisper are real)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "data" / "clips" / "test_synthetic.mp4"

pytestmark = pytest.mark.skipif(not CLIP.exists(), reason="run scripts/make_test_clip.sh first")


def test_caption_clip_mock_end_to_end(tmp_path, monkeypatch):
    import styleforge.config as config
    from styleforge import pipeline

    monkeypatch.setattr(config, "MOCK", True)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    result = pipeline.caption_clip(CLIP, mode="bon", k=2)
    assert set(result["captions"]) == set(config.STYLES)
    for entry in result["captions"].values():
        assert entry["caption"]
    assert result["n_frames"] > 0


def test_harness_contract_subprocess(tmp_path):
    """The organizer-eye view: tasks.json in -> results.json out, correct schema."""
    tasks = [
        {
            "task_id": "v-test",
            "video_url": CLIP.resolve().as_uri(),
            "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
        }
    ]
    inp = tmp_path / "tasks.json"
    outp = tmp_path / "results.json"
    inp.write_text(json.dumps(tasks))

    env = os.environ | {
        "STYLEFORGE_MOCK": "1",
        "HARNESS_INPUT": str(inp),
        "HARNESS_OUTPUT": str(outp),
        "STYLEFORGE_DATA": str(tmp_path / "data"),
        "TIME_BUDGET_SEC": "300",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "styleforge.harness"],
        env=env, capture_output=True, text=True, cwd=ROOT, timeout=240,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    results = json.loads(outp.read_text())
    assert isinstance(results, list) and len(results) == 1
    rec = results[0]
    assert rec["task_id"] == "v-test"
    assert set(rec["captions"]) == {
        "formal", "sarcastic", "humorous_tech", "humorous_non_tech"
    }
    assert all(isinstance(v, str) and v for v in rec["captions"].values())


def test_harness_bad_url_still_produces_all_styles(tmp_path):
    """A dead video URL must degrade to fallback captions, never a missing style."""
    tasks = [{"task_id": "v-dead", "video_url": "file:///nonexistent.mp4",
              "styles": ["formal", "sarcastic"]}]
    inp, outp = tmp_path / "tasks.json", tmp_path / "results.json"
    inp.write_text(json.dumps(tasks))
    env = os.environ | {
        "STYLEFORGE_MOCK": "1",
        "HARNESS_INPUT": str(inp),
        "HARNESS_OUTPUT": str(outp),
        "STYLEFORGE_DATA": str(tmp_path / "data"),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "styleforge.harness"],
        env=env, capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert proc.returncode == 0
    rec = json.loads(outp.read_text())[0]
    assert set(rec["captions"]) == {"formal", "sarcastic"}
    assert all(rec["captions"].values())
