"""Validate the two acceptance criteria that need a live API key.

    .venv/bin/python scripts/validate_live.py

Criterion 3: ClaudeClient.complete() round-trips a prompt with a real API key.
Criterion 4: Prompt caching is applied to the system prompt on repeated calls,
             verified via usage.cache_read_input_tokens.

Makes three billable API calls (a few cents at most).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant import config  # noqa: E402  (path setup must precede import)
from assistant.errors import AssistantError  # noqa: E402

# The model won't cache a prefix below this many tokens; a short system prompt
# silently produces zero cache reads, which looks like a failure but isn't.
# Source: shared/prompt-caching.md — 1024 for claude-sonnet-5.
_CACHE_MINIMUM_TOKENS = {
    "claude-opus-5": 512,
    "claude-fable-5": 512,
    "claude-sonnet-5": 1024,
    "claude-opus-4-8": 1024,
    "claude-sonnet-4-6": 1024,
    "claude-opus-4-7": 2048,
    "claude-opus-4-6": 4096,
    "claude-haiku-4-5": 4096,
}
_DEFAULT_CACHE_MINIMUM = 1024

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m",
    "\033[31m",
    "\033[33m",
    "\033[2m",
    "\033[0m",
)


def ok(msg: str) -> None:
    print(f"{GREEN}PASS{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"{RED}FAIL{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}WARN{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"{DIM}      {msg}{RESET}")


def preflight() -> str:
    """Confirm a key is configured before making any billable call."""
    try:
        key = config.get_api_key()
    except AssistantError as exc:
        fail(str(exc))
        print()
        print("Add your key to .env:")
        print("    ANTHROPIC_API_KEY=sk-ant-...")
        print("Get one at https://console.anthropic.com/settings/keys")
        raise SystemExit(1)

    model = config.get_model()
    info(f"key ...{key[-6:]}  model {model}")
    return model


def check_round_trip() -> bool:
    """Criterion 3: a prompt goes out and text comes back."""
    print("\nCriterion 3 — complete() round-trips a real prompt")

    from assistant.client import ClaudeClient

    client = ClaudeClient()
    try:
        reply = client.complete(
            messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
            system="You are a test harness. Follow the instruction exactly.",
            max_tokens=16,
        )
    except AssistantError as exc:
        fail(f"call failed: {exc}")
        return False

    if not reply.strip():
        fail("call succeeded but returned empty text")
        return False

    ok(f"round trip returned {len(reply)} chars")
    info(f"response: {reply.strip()[:60]!r}")
    return True


def check_prompt_caching(model: str) -> bool:
    """Criterion 4: the second identical call reads the system prompt from cache."""
    print("\nCriterion 4 — prompt caching on repeated calls")

    from assistant.client import ClaudeClient
    from assistant.ingestion.normalize import estimate_tokens
    from assistant.prompts.meeting_summary import SUMMARY_SYSTEM_PROMPT

    minimum = _CACHE_MINIMUM_TOKENS.get(model, _DEFAULT_CACHE_MINIMUM)

    # A system prompt below the model's minimum cacheable prefix never caches.
    # Pad with representative document content so the test exercises caching
    # rather than silently measuring the floor.
    system = SUMMARY_SYSTEM_PROMPT
    filler_turn = (
        "[00:0{i}:00] Speaker {i}: We reviewed the migration plan and agreed "
        "the staging soak needs a full day before the production cutover.\n"
    )
    index = 0
    while estimate_tokens(system) < minimum + 200:
        system += filler_turn.format(i=index % 10)
        index += 1

    info(f"system prompt ~{estimate_tokens(system)} tokens (minimum {minimum})")

    client = ClaudeClient()
    messages = [{"role": "user", "content": "Reply with exactly: OK"}]

    try:
        first = client.complete_raw(messages=messages, system=system, max_tokens=16)
        second = client.complete_raw(messages=messages, system=system, max_tokens=16)
    except AssistantError as exc:
        fail(f"call failed: {exc}")
        return False

    created = first.usage.cache_creation_input_tokens
    read = second.usage.cache_read_input_tokens

    info(f"call 1: cache_creation_input_tokens={created}")
    info(f"call 2: cache_read_input_tokens={read}")

    if read > 0:
        ok(f"second call read {read} tokens from cache")
        return True

    if created == 0:
        fail("nothing was written to cache — check the cache_control breakpoint")
    else:
        fail(f"wrote {created} tokens but read 0 back")
        info("a cache write with no read usually means the prefix changed between calls")
    return False


def main() -> int:
    print("Live API validation")
    print("Makes 3 billable calls.\n")

    model = preflight()

    results = [
        ("complete() round trip", check_round_trip()),
        ("prompt caching", check_prompt_caching(model)),
    ]

    print("\n" + "-" * 46)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    failed = [name for name, passed in results if not passed]
    if failed:
        print(f"\n{RED}{len(failed)} of {len(results)} criteria failed.{RESET}")
        return 1

    print(f"\n{GREEN}Both criteria verified.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
