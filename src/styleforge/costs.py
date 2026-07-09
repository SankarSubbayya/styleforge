import json
import threading
import time
from dataclasses import dataclass, field

from . import config


@dataclass
class CostTracker:
    """NFR-4: every API call's token usage is logged; running totals printable per batch."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            config.ensure_dirs()
            entry = {
                "ts": time.time(),
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            with open(config.CACHE_DIR / "costs.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")

    def summary(self) -> str:
        return (
            f"API calls: {self.calls} | prompt tok: {self.prompt_tokens:,} "
            f"| completion tok: {self.completion_tokens:,}"
        )


tracker = CostTracker()
