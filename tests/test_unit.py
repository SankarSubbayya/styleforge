"""Unit tests — contract-critical logic, no network (mock mode)."""

import json

import pytest

import styleforge.config as config
from styleforge import judge, stylize

HARNESS_STYLES = {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"}


def test_styles_match_harness_schema_exactly():
    # Participant Guide uses underscore keys; a hyphen here scores zero on that style.
    assert set(config.STYLES) == HARNESS_STYLES


def test_fallback_captions_cover_every_style():
    assert set(config.FALLBACK_CAPTIONS) == HARNESS_STYLES
    assert all(len(v) > 10 for v in config.FALLBACK_CAPTIONS.values())


def test_stylize_rejects_unknown_style():
    with pytest.raises(ValueError):
        stylize.generate("desc", "", "poetic", k=1)


def test_stylize_mock_returns_k_candidates(monkeypatch):
    monkeypatch.setattr(config, "MOCK", True)
    cands = stylize.generate("a cat walks", "", "formal", k=3)
    assert len(cands) == 3
    assert all(isinstance(c, str) and c for c in cands)


def test_judge_score_caches_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MOCK", False)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(
        judge.fw, "chat",
        lambda *a, **k: '{"accuracy": 8, "tone": 7, "rationale": "stub"}',
    )
    s1 = judge.score("desc", "caption", "formal")
    assert {"accuracy", "tone", "overall", "rationale"} <= set(s1)
    assert s1["overall"] == (s1["accuracy"] + s1["tone"]) / 2
    cache_files = list((tmp_path / "judge").glob("*.json"))
    assert len(cache_files) == 1
    # Second call must hit the cache (poison fw.chat to prove no API path is taken)
    monkeypatch.setattr(judge.fw, "chat", lambda *a, **k: pytest.fail("cache missed"))
    assert judge.score("desc", "caption", "formal") == s1


def test_judge_mock_mode_never_touches_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MOCK", True)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    judge.score("desc", "caption", "formal")
    # Mock scores must not poison the real cache (review finding).
    assert list((tmp_path / "judge").glob("*.json")) == []


def test_ensemble_score_averages(monkeypatch):
    monkeypatch.setattr(config, "MOCK", True)
    monkeypatch.setenv("JUDGE_ENSEMBLE", "model-a,model-b")
    scores = iter(
        ['{"accuracy": 8, "tone": 6, "rationale": "a"}',
         '{"accuracy": 6, "tone": 8, "rationale": "b"}']
    )
    monkeypatch.setattr(judge.fw, "chat", lambda *a, **k: next(scores))
    monkeypatch.setattr(judge, "_cache_path", lambda key: type(
        "P", (), {"exists": lambda self: False, "write_text": lambda self, t: None}
    )())
    result = judge.ensemble_score("d", "c", "formal")
    assert result["n_judges"] == 2
    assert result["accuracy"] == 7.0
    assert result["tone"] == 7.0


def test_build_pairs_gap_threshold(tmp_path, monkeypatch):
    import training.build_pairs as bp

    rec = {
        "desc_idx": 0,
        "description": "a dog runs",
        "style": "formal",
        "candidates": [
            {"caption": "best", "accuracy": 9, "tone": 9, "overall": 9.0},
            {"caption": "mid", "accuracy": 7, "tone": 7, "overall": 7.0},
            {"caption": "worst", "accuracy": 5, "tone": 5, "overall": 5.0},
        ],
    }
    close = {**rec, "style": "sarcastic",
             "candidates": [{"caption": "a", "overall": 8.0},
                            {"caption": "b", "overall": 7.5}]}
    monkeypatch.setattr(bp, "TRAIN_DIR", tmp_path)
    (tmp_path / "candidates.jsonl").write_text(
        json.dumps(rec) + "\n" + json.dumps(close) + "\n"
    )
    bp.main()
    pairs = [json.loads(x) for x in (tmp_path / "dpo_pairs.jsonl").open()]
    sft = [json.loads(x) for x in (tmp_path / "sft_top.jsonl").open()]
    assert len(pairs) == 1  # close-gap cell skipped
    assert pairs[0]["chosen"][0]["content"] == "best"
    assert pairs[0]["rejected"][0]["content"] == "worst"
    assert len(sft) == 2  # SFT keeps top of every cell


def test_train_cfg_drops_unknown_args():
    trl = pytest.importorskip("trl")  # not installed locally; runs on the droplet
    from training.train_dpo import make_cfg

    cfg = make_cfg(trl.DPOConfig, output_dir="x", definitely_not_a_real_arg=1)
    assert not hasattr(cfg, "definitely_not_a_real_arg")
