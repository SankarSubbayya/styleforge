"""F2 / Stage A: frames + transcript -> dense factual scene description via Fireworks VLM."""

from . import config, fw

SYSTEM = (
    "You are a meticulous video analyst. You receive frames sampled evenly across a short "
    "clip (timestamps given) plus an audio transcript. Produce a dense, strictly factual "
    "description covering: setting; people/animals/objects; the sequence of actions in order "
    "with approximate timing; any on-screen text; audio cues from the transcript; and the "
    "overall arc, including anything surprising, awkward, or funny that happens. "
    "No opinions, no jokes, no speculation beyond what is visible or audible. 150-250 words."
)

MOCK_DESCRIPTION = (
    "A test-pattern video: colored bars and a moving gradient fill the screen while a "
    "counter advances. A steady tone plays. Around the midpoint the pattern shifts and "
    "the counter continues until the clip ends. No people appear; no on-screen text "
    "other than the counter."
)


def describe(frames: list[tuple[float, str]], transcript: str) -> str:
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Audio transcript (may be empty for silent clips):\n"
                f"{transcript or '[no audio]'}\n\n"
                f"Frames ({len(frames)}), timestamps in seconds: "
                f"{', '.join(f'{t:.1f}' for t, _ in frames)}"
            ),
        }
    ]
    for _, uri in frames:
        content.append({"type": "image_url", "image_url": {"url": uri}})

    return fw.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        model=config.PERCEPTION_MODEL,
        mock_response=MOCK_DESCRIPTION,
        temperature=0.2,
        max_tokens=2500,
        # No thinking: 12-frame description is extraction, not reasoning; latency matters
        # against the harness's 10-minute cap.
        reasoning_effort="none",
    )
